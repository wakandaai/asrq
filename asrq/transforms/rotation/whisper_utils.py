# pyright: reportGeneralTypeIssues=false
# pyright: reportMissingImports=false
# pyright: reportPrivateImportUsage=false

from transformers import WhisperProcessor
from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration,
    BaseModelOutput, 
    BaseModelOutputWithPastAndCrossAttentions,
    create_causal_mask,
    EncoderDecoderCache,
    DynamicCache,
    logger,
)
from asrq.evaluation.english_text_normalizer import normalizer
from asrq.transforms.rotation.utils import (
    convert_model_layernorms_to_rmsnorms,
    fuse_normalization_weights_and_bias_into_adjacent_linears,
    modify_linear_with_rotation_param,
    fuse_rotation_param_into_linear,
    get_orthogonal_matrix,
)
from asrq.transforms.rotation.cayley_sgd import SGDG
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import types
from datasets import load_dataset
from typing import List, Tuple, Dict
from itertools import islice


class WhisperCalibrationDataset(torch.utils.data.Dataset):
    """LibriSpeech train-clean-100 samples for rotation training."""

    def __init__(self, processor, num_samples=128, seed=42):
        super().__init__()
        ds = load_dataset("librispeech_asr", "all", split="validation.clean") 
        ds = ds.shuffle(seed=seed)
        self.samples = list(islice(ds, num_samples)) # type: ignore
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio = sample["audio"]["array"]
        text = normalizer(sample["text"])

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.squeeze(0)

        tokens = self.processor.tokenizer.encode(text)
        tokens = tokens[2:]  # strip BOS/task tokens added by tokenizer
        tokens = [50258, 50259, 50360, 50364, *tokens]
        labels = torch.tensor(tokens, dtype=torch.long)

        return input_features, labels


def collate_fn(batch):
    input_features = torch.stack([b[0] for b in batch])
    max_len = max(b[1].size(0) for b in batch)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, : b[1].size(0)] = b[1]
    return {"input_features": input_features, "labels": labels}

def whisper_loss_fn(model, batch):
    device = next(model.parameters()).device
    return model(
        input_features=batch["input_features"].to(device),
        labels=batch["labels"].to(device),
        return_dict=True,
    ).loss


def get_whisper_norm_fusion_config(
    num_encoder_layers: int = 32,
    num_decoder_layers: int = 32,
) -> List[Tuple[str, List[str], List[str]]]:
    """Return ``(norm_name, pre_linear_names, post_linear_names)`` for Whisper.

    Only the norm **weight / bias → succeeding-linear** fusion is configured
    (``post_linear_names``).  Mean-subtraction (M) fusion into preceding
    linears is left empty because in a residual network the norm input is a
    sum of many paths, making exact fusion impossible.

    After fusion the norms become pure ``RMSNorm(x @ M)`` with weight = 1,
    bias = 0, and the downstream linears absorb the original scale/shift.
    """
    config: List[Tuple[str, List[str], List[str]]] = []

    # --- Encoder ---
    for i in range(num_encoder_layers):
        p = f"model.encoder.layers.{i}"
        # include M fusion. For first layer we fuse M by inserting it into the network, just after the conv layers and before the transformer encoder
        # for subsequent layers we fuse M into the preceding linear layers (self-attn out_proj and fc2) since the input to the norm is just the output of these linears
        # residual connections already have the mean subtraction since it was applied to the input of the encoder
        config.append((
            f"{p}.self_attn_layer_norm",
            ["first_layer/model.encoder"] if i == 0 else [f"model.encoder.layers.{i-1}.fc2"],
            [f"{p}.self_attn.q_proj", f"{p}.self_attn.k_proj", f"{p}.self_attn.v_proj"],
        ))
        config.append((
            f"{p}.final_layer_norm",
            [f"{p}.self_attn.out_proj"],
            [f"{p}.fc1"],
        ))
    # Final encoder layer_norm (output feeds cross-attn K/V — handled via
    # layer rotation, so no post-linears needed here)
    # feeds into the decoder cross attention K/V projections
    config.append((
        "model.encoder.layer_norm", 
        [f"model.encoder.layers.{num_encoder_layers-1}.fc2"],  # M-fuse the last encoder layer's fc2
        [f"model.decoder.layers.{i}.encoder_attn.k_proj" for i in range(num_decoder_layers)] +
        [f"model.decoder.layers.{i}.encoder_attn.v_proj" for i in range(num_decoder_layers)]
    ))

    # --- Decoder ---
    for i in range(num_decoder_layers):
        p = f"model.decoder.layers.{i}"
        # include the mean subtraction just before the decoder blocks
        config.append((
            f"{p}.self_attn_layer_norm",
            ["first_layer/model.decoder"] if i == 0 else [f"model.decoder.layers.{i-1}.fc2"],
            [f"{p}.self_attn.q_proj", f"{p}.self_attn.k_proj", f"{p}.self_attn.v_proj"],
        ))
        # encoder_attn_layer_norm applies to decoder hidden_states → only Q.
        # K/V come from encoder_hidden_states, NOT from this layer norm.
        config.append((
            f"{p}.encoder_attn_layer_norm",
            [f"{p}.self_attn.out_proj"],
            [f"{p}.encoder_attn.q_proj"],
        ))
        config.append((
            f"{p}.final_layer_norm",
            [f"{p}.encoder_attn.out_proj"],
            [f"{p}.fc1"],
        ))

    # Final decoder layer_norm
    # The input to this norm will already be mean-subtracted since it comes from the residual stream, so we only fuse the weight/bias into the succeeding linears (if any). 
    # In Whisper's decoder this is the final layer norm whose output goes directly into the LM head, so there are no succeeding linears to fuse into. 
    # Hence this norm is left as is, without fusion, and will be absorbed into the rotation as usual.
    config.append(("model.decoder.layer_norm", [f"model.decoder.layers.{num_decoder_layers-1}.fc2"], []))

    return config


def get_whisper_layers_to_rotate(
    num_encoder_layers: int = 32,
    num_decoder_layers: int = 32,
) -> List[Tuple[str, bool]]:
    """Return ``(layer_name, for_rotated_input)`` pairs for Whisper.

    A single rotation matrix *Q* is shared by both the encoder and decoder
    residual streams (both are ``d_model``-dimensional).  Layers whose
    **input** comes from the rotated residual get ``for_rotated_input=True``
    (their weight is right-multiplied by Q).  Layers whose **output** feeds
    back into the rotated residual get ``for_rotated_input=False`` (their
    weight is left-multiplied by Q^T).

    Cross-attention K/V projections receive **encoder** output, which is also
    in the Q-rotated basis, so they are treated as ``for_rotated_input=True``.
    """
    layers: List[Tuple[str, bool]] = []

    # --- Encoder layers ---
    for i in range(num_encoder_layers):
        p = f"model.encoder.layers.{i}"
        layers.extend([
            (f"{p}.self_attn.q_proj", True),
            (f"{p}.self_attn.k_proj", True),
            (f"{p}.self_attn.v_proj", True),
            (f"{p}.self_attn.out_proj", False),
            (f"{p}.fc1", True),
            (f"{p}.fc2", False),
        ])

    # --- Decoder layers ---
    for i in range(num_decoder_layers):
        p = f"model.decoder.layers.{i}"
        layers.extend([
            # Self-attention
            (f"{p}.self_attn.q_proj", True),
            (f"{p}.self_attn.k_proj", True),
            (f"{p}.self_attn.v_proj", True),
            (f"{p}.self_attn.out_proj", False),
            # Cross-attention (encoder output is in the same Q-rotated basis)
            (f"{p}.encoder_attn.q_proj", True),
            (f"{p}.encoder_attn.k_proj", True),
            (f"{p}.encoder_attn.v_proj", True),
            (f"{p}.encoder_attn.out_proj", False),
            # FFN
            (f"{p}.fc1", True),
            (f"{p}.fc2", False),
        ])

    return layers


# == Monkey-patching the forward methods of the Whisper encoder and decoder to include mean normalization and rotation of the residual stream input ==
def whisper_encoder_forward(
    self,
    input_features,
    attention_mask=None,
    head_mask=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
) -> BaseModelOutput:

    expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
    if input_features.shape[-1] != expected_seq_length:
        raise ValueError(
            f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
        )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    inputs_embeds = nn.functional.gelu(self.conv1(input_features))
    inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

    inputs_embeds = inputs_embeds.permute(0, 2, 1)
    all_positions = torch.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)

    hidden_states = inputs_embeds + self.embed_positions(all_positions)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    # check if head_mask has a correct number of layers specified if desired
    if head_mask is not None:
        assert head_mask.size()[0] == (len(self.layers)), (
            f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        )

    # -------- Process the input to the transformer encoder blocks with the necessary mean subtraction and rotation -----
    # Monkey Patch
    hidden_states = self.process_residual_stream_input(hidden_states)
    # --------------------------------------------------------------------------------------------------------------------
    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
        to_drop = False
        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:  # skip the layer
                to_drop = True

        if to_drop:
            layer_outputs = (None, None)
        else:
            layer_outputs = encoder_layer(
                hidden_states,
                None,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    hidden_states = self.layer_norm(hidden_states)
    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
    )


def whisper_decoder_forward(
    self,
    input_ids=None,
    attention_mask=None,
    encoder_hidden_states=None,
    head_mask=None,
    cross_attn_head_mask=None,
    past_key_values=None,
    inputs_embeds=None,
    position_ids=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
) -> BaseModelOutputWithPastAndCrossAttentions:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        if self.config.is_encoder_decoder:
            past_key_values = EncoderDecoderCache(
                DynamicCache(config=self.config), DynamicCache(config=self.config)
            )
        else:
            past_key_values = DynamicCache(config=self.config)

    past_key_values_length = 0
    if cache_position is not None:
        past_key_values_length = cache_position[0]
    elif past_key_values is not None:
        past_key_values_length = past_key_values.get_seq_length()

    if cache_position is None:
        cache_position = torch.arange(
            past_key_values_length, past_key_values_length + input_shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0).repeat(input_shape[0], 1)

    # embed positions
    if input_ids is not None:
        positions = self.embed_positions(
            input_ids, past_key_values_length=past_key_values_length, position_ids=position_ids
        )
    else:
        positions = self.embed_positions(
            inputs_embeds, past_key_values_length=past_key_values_length, position_ids=position_ids
        )

    hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
            )
            use_cache = False
    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

    # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
    for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
        if attn_mask is not None:
            assert attn_mask.size()[0] == (len(self.layers)), (
                f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                f" {head_mask.size()[0]}."
            )

    # -------- Process the input to the transformer decoder blocks with the necessary mean subtraction and rotation -----
    # Monkey Patch
    hidden_states = self.process_residual_stream_input(hidden_states)
    # --------------------------------------------------------------------------------------------------------------------
    for idx, decoder_layer in enumerate(self.layers):
        # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:
                continue

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            encoder_hidden_states=encoder_hidden_states,
            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
            past_key_values=past_key_values if use_cache else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs[2],)

    # --------- Process the output of the rmsnorm after the decoder blocks -------------
    # Applies the inverse rotation
    hidden_states = self.process_residual_stream_output_after_normalization(hidden_states)  
    # ----------------------------------------------------------------------------------

    hidden_states = self.layer_norm(hidden_states)
    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = past_key_values if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        cross_attentions=all_cross_attentions,
    )



def monkey_patch_whisper(model: nn.Module, Qe: nn.Parameter, Qd: nn.Parameter) -> None:
    """Monkey-patch the model's encoder and decoder forward methods to include the necessary mean normalization and rotation of the residual stream input."""
    # Monkey-patch the encoder and decoder forward methods

    def process_residual_stream_input_encoder(self, x):
        dtype = x.dtype
        x = x - x.mean(dim=-1, keepdim=True)
        return (x.double() @ Qe.double()).to(dtype)

    def process_residual_stream_input_decoder(self, x):
        dtype = x.dtype
        x = x - x.mean(dim=-1, keepdim=True)
        return (x.double() @ Qd.double()).to(dtype)

    def process_residual_stream_output_after_normalization_encoder(self, x):
        dtype = x.dtype
        return (x.double() @ Qe.t().double()).to(dtype)

    def process_residual_stream_output_after_normalization_decoder(self, x):
        dtype = x.dtype
        return (x.double() @ Qd.t().double()).to(dtype)

    model.model.encoder.forward = types.MethodType(whisper_encoder_forward, model.model.encoder) # type: ignore
    model.model.decoder.forward = types.MethodType(whisper_decoder_forward, model.model.decoder) # type: ignore

    # Add the process_residual_stream_input method to the model, which applies both mean normalization and rotation to the input of the transformer blocks
    model.model.encoder.process_residual_stream_input = types.MethodType(process_residual_stream_input_encoder, model.model.encoder) # type: ignore
    model.model.decoder.process_residual_stream_input = types.MethodType(process_residual_stream_input_decoder, model.model.decoder) # type: ignore

    # Add the process_residual_stream_output_after_normalization method to the model, which applies the inverse rotation to the output of the RMSNorm after the transformer blocks
    model.model.encoder.process_residual_stream_output_after_normalization = types.MethodType(process_residual_stream_output_after_normalization_encoder, model.model.encoder) # type: ignore
    model.model.decoder.process_residual_stream_output_after_normalization = types.MethodType(process_residual_stream_output_after_normalization_decoder, model.model.decoder) # type: ignore

# Note: The above monkey patching approach is used to avoid modifying the original model code and to 
# inject the necessary processing for rotation. The process_residual_stream_input method applies mean 
# normalization and rotation to the input of the transformer blocks, while the 
# process_residual_stream_output_after_normalization method applies the inverse rotation to the output
# of the RMSNorm after the transformer blocks, ensuring that the residual stream is correctly rotated 
# throughout the model.


def prepare_whisper_for_rotation(model: nn.Module) -> None:
    """Prepare the Whisper model for rotation by converting layer norms to RMSNorms and fusing normalization weights into adjacent linears."""
    # Convert LayerNorms to RMSNorms and fuse normalization weights into adjacent linears
    convert_model_layernorms_to_rmsnorms(model)

    # fuse normalization weights
    norm_fusion_cfg = get_whisper_norm_fusion_config(
        num_encoder_layers=model.config.encoder_layers, # type: ignore
        num_decoder_layers=model.config.decoder_layers, # type: ignore
    )
    fuse_normalization_weights_and_bias_into_adjacent_linears(model, norm_fusion_cfg)

def modify_whisper_layers_with_rotation_params(model: nn.Module, Qe: nn.Parameter, Qd: nn.Parameter, Q2s: Dict[str, nn.Parameter]) -> None:
    """Modify the Whisper model's linear layers to include the rotation parameters in their forward pass."""
    layers_to_rotate = get_whisper_layers_to_rotate(
                int(model.config.encoder_layers), int(model.config.decoder_layers) # type: ignore
        )
    named_modules = dict(model.named_modules())
    for layer_name, for_rotated_input in layers_to_rotate:
        layer = named_modules.get(layer_name)
        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model.")
        assert isinstance(layer, nn.Linear), f"Expected layer '{layer_name}' to be an instance of nn.Linear, but found {type(layer)}"
        # Q2 is only applied to v_proj and out_proj
        stem_name, leaf_name = layer_name.rsplit(".", 1)
        if leaf_name in ["v_proj", "out_proj"]:
            Q2 = Q2s.get(stem_name, None)
            if Q2 is None:
                raise ValueError(f"Q2 for layer '{stem_name}' not found in Q2s dictionary.")
        else:
            Q2 = None
        # encoder takes Qe, decoder takes Qd
        # Cross-attention K/V receive encoder output (Qe basis), not decoder hidden states
        if "model.encoder." in layer_name:
            Q = Qe
        elif "encoder_attn.k_proj" in layer_name or "encoder_attn.v_proj" in layer_name:
            Q = Qe  # cross-attn K/V receive encoder output in Qe basis
        else:
            Q = Qd
        modify_linear_with_rotation_param(layer, Q, Q2=Q2, for_rotated_input=for_rotated_input, quantize_row_wise=True, bit=4)


def fuse_whisper_layers_with_rotations(model, Qe: torch.Tensor, Qd: torch.Tensor, Q2s: Dict[str, torch.Tensor], device="cuda") -> None:
    named_modules = dict(model.named_modules())
    layers_to_rotate = get_whisper_layers_to_rotate(
            model.config.encoder_layers, model.config.decoder_layers
    )
    for layer_name, for_rotated_input in layers_to_rotate:
        layer = named_modules.get(layer_name)
        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model.")
        # Q2 is only applied to v_proj and out_proj
        stem_name, leaf_name = layer_name.rsplit(".", 1)
        if leaf_name in ["v_proj", "out_proj"]:
            Q2 = Q2s.get(stem_name, None)
            if Q2 is None:
                raise ValueError(f"Q2 for layer '{stem_name}' not found in Q2s dictionary.")
        else:
            Q2 = None
        # Cross-attention K/V receive encoder output (Qe basis), not decoder hidden states
        if "model.encoder." in layer_name:
            Q = Qe
        elif "encoder_attn.k_proj" in layer_name or "encoder_attn.v_proj" in layer_name:
            Q = Qe  # cross-attn K/V receive encoder output in Qe basis
        else:
            Q = Qd
        fuse_rotation_param_into_linear(layer, Q.to(device), Q2=Q2.to(device) if Q2 is not None else None, for_rotated_input=for_rotated_input)


def obtain_rotations_for_whisper(
        model: WhisperForConditionalGeneration, processor: WhisperProcessor, 
        test_audio:np.ndarray, test_audio_sr:int, calib_samples: int, 
        epochs: int, lr: float, batch_size: int, save_path: str
    ) -> None:
    device = "cuda"
    model.to(device) # type: ignore
    dtype = model.dtype
    audio = test_audio
    sr = test_audio_sr
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out_a = model.generate(inputs.input_features.to(device).to(dtype), max_new_tokens=128)
        orig_transcription = processor.batch_decode(out_a, skip_special_tokens=True)[0].strip()

    prepare_whisper_for_rotation(model)

    # create model rotations
    mode = "hadamard"
    Qe = get_orthogonal_matrix(model.config.d_model, mode=mode, device=device)
    Qd = get_orthogonal_matrix(model.config.d_model, mode=mode, device=device)
    Q2s = {}
    for i in range(model.config.encoder_layers):
        head_dim: int = model.model.encoder.layers[i].self_attn.head_dim # type: ignore
        rot = get_orthogonal_matrix(head_dim, mode=mode, device=device)
        Q2s[f"model.encoder.layers.{i}.self_attn"] = rot
    for i in range(model.config.decoder_layers):
        head_dim: int = model.model.decoder.layers[i].self_attn.head_dim # type: ignore
        rot = get_orthogonal_matrix(head_dim, mode=mode, device=device)
        Q2s[f"model.decoder.layers.{i}.self_attn"] = rot
        head_dim: int = model.model.decoder.layers[i].encoder_attn.head_dim # type: ignore
        rot = get_orthogonal_matrix(head_dim, mode=mode, device=device)
        Q2s[f"model.decoder.layers.{i}.encoder_attn"] = rot

    # Qs are parameters
    Qe = nn.Parameter(Qe.float(), requires_grad=True)
    Qd = nn.Parameter(Qd.float(), requires_grad=True)
    for k in Q2s:
        Q2s[k] = nn.Parameter(Q2s[k].float(), requires_grad=True)

    modify_whisper_layers_with_rotation_params(model, Qe, Qd, Q2s)
    monkey_patch_whisper(model, Qe, Qd)

    # ensure that model remains computational invariant despite the rotations
    with torch.no_grad():
        # transcribe an audio sample and compare with the original transcription
        out_a = model.generate(inputs.input_features.to(device).to(dtype), max_new_tokens=128)
        rot_transcription = processor.batch_decode(out_a, skip_special_tokens=True)[0].strip()
        assert orig_transcription == rot_transcription, f"Transcriptions do not match after applying rotations! \nOriginal: '{orig_transcription}', \nAfter Rotation: '{rot_transcription}'"
    # -----------------------------------------------------------------------


    calib_ds = WhisperCalibrationDataset(processor, num_samples=calib_samples)
    train_loader = torch.utils.data.DataLoader(
        calib_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    trainable_params = [Qe, Qd] + list(Q2s.values())
    optimizer = SGDG(trainable_params, lr=lr, stiefel=True)
    model.train()
    model.to(device) # type: ignore
    # This is for just one epoch
    # I want the learning rate to decay linearly to 0
    # starting with 1.5, it decays to 0
    num_steps = len(train_loader) * epochs
    lr_lambda = lambda step: max(0, (num_steps - step) / num_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = whisper_loss_fn(model, batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        

    to_save = {
        "Qe": Qe.data.detach().cpu(),
        "Qd": Qd.data.detach().cpu(),
        "Q2s": {k: v.data.detach().cpu() for k, v in Q2s.items()},
    }
    torch.save(to_save, save_path)
    

def rotate_whisper_model(
    model: WhisperForConditionalGeneration, processor: WhisperProcessor, test_audio: np.ndarray, sr:int, rotation_path:str, device="cuda"
):
    model.to(device) # type: ignore
    audio = test_audio
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out_a = model.generate(inputs.input_features.to(device).to(model.dtype), max_new_tokens=128)
        orig_transcription = processor.batch_decode(out_a, skip_special_tokens=True)[0].strip()

    prepare_whisper_for_rotation(model)  # prepare the model for rotation (convert to RMSNorm, fuse norm weights into linears)
    rotations = torch.load(rotation_path)  # load the learned rotations from file
    Qe = rotations["Qe"]
    Qd = rotations["Qd"]
    Q2s = rotations["Q2s"]
    fuse_whisper_layers_with_rotations(model, Qe, Qd, Q2s, device=device)  # fuse the rotations into the model weights
    monkey_patch_whisper(model, Qe.to(device), Qd.to(device))

    with torch.no_grad():
        out_a = model.generate(inputs.input_features.to(device).to(model.dtype), max_new_tokens=128)
        rot_transcription = processor.batch_decode(out_a, skip_special_tokens=True)[0].strip()
        assert orig_transcription == rot_transcription, f"Transcriptions do not match after fusing rotations to weights! Original: '{orig_transcription}', After Rotation: '{rot_transcription}'"
