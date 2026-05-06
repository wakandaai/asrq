# pyright: reportMissingImports=false
# pyright: reportPrivateImportUsage=false
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from typing import List, Tuple, Optional, Union, Dict
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from asrq.transforms.rotation.utils import (
    RMSNormFusedM,
    STEQuantize,
    convert_model_layernorms_to_rmsnorms,
    get_orthogonal_matrix,
    fuse_normalization_weights_and_bias_into_adjacent_linears
)
from asrq.transforms.rotation.cayley_sgd import SGDG
from transformers.models.qwen3.modeling_qwen3 import (
    BaseModelOutputWithPast,
    Cache,
    Unpack,
    TransformersKwargs,
    DynamicCache,
)
try:
    from nemo.collections.asr.modules.conformer_encoder import (
        random
    )
    from nemo.collections.speechlm2.models.salm import (
        PromptFormatter,
        replace_placeholders_and_build_targets
    )
except ImportError:
    pass
from datasets import load_dataset
from itertools import islice




class CanaryQwenCalibrationDataset(torch.utils.data.Dataset):
    """LibriSpeech train-clean-100 samples formatted for CanaryQwen rotation training."""

    def __init__(self, model, num_samples=128, seed=42):
        super().__init__()
        ds = load_dataset("librispeech_asr", "all", split="train.clean.100")
        ds = ds.shuffle(seed=seed)
        self.samples = list(islice(ds, num_samples))
        self.model = model
        self.formatter = PromptFormatter.resolve(model.cfg.prompt_format)(model.tokenizer)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio = torch.tensor(sample["audio"]["array"], dtype=torch.float32)
        audio_len = torch.tensor(audio.shape[0], dtype=torch.long)
        text = sample["text"]

        # Format prompt tokens (includes system/user/assistant template + audio_locator_tag)
        prompt = [{"role": "user", "content": f"Transcribe the following: {self.model.audio_locator_tag}"}]
        base_tokens = self.formatter.encode_dialog(turns=prompt)["input_ids"]

        # Append transcription tokens and EOS
        text_tokens = torch.tensor(self.model.tokenizer.text_to_ids(text), dtype=torch.long)
        eos = torch.tensor([self.model.tokenizer.eos], dtype=torch.long)
        tokens = torch.cat([base_tokens, text_tokens, eos])

        return audio, audio_len, tokens


def canaryqwen_collate_fn(batch, pad_id):
    """Collate function for CanaryQwenCalibrationDataset."""
    audios_list = [b[0] for b in batch]
    audio_lens = torch.stack([b[1] for b in batch])
    tokens_list = [b[2] for b in batch]

    # Right-pad audios with zeros
    max_audio_len = max(a.shape[0] for a in audios_list)
    audios = torch.zeros(len(batch), max_audio_len, dtype=torch.float32)
    for i, a in enumerate(audios_list):
        audios[i, : a.shape[0]] = a

    # Left-pad tokens (SALM convention)
    max_token_len = max(t.shape[0] for t in tokens_list)
    tokens = torch.full((len(batch), max_token_len), pad_id, dtype=torch.long)
    for i, t in enumerate(tokens_list):
        tokens[i, -t.shape[0] :] = t

    return {"audios": audios, "audio_lens": audio_lens, "tokens": tokens}


def canaryqwen_loss_fn(model, batch):
    """Compute next-token-prediction cross-entropy loss for rotation training."""
    device = next(model.parameters()).device
    audios = batch["audios"].to(device)
    audio_lens = batch["audio_lens"].to(device)
    tokens = batch["tokens"].to(device)

    # Encode audio through perception module
    audio_embeds, audio_embed_lens = model.perception(
        input_signal=audios, input_signal_length=audio_lens,
    )
    audio_embeds = [audio_embeds[i, :elen] for i, elen in enumerate(audio_embed_lens)]

    # Embed text tokens
    tokens_to_embed = tokens.where(tokens != model.audio_locator_tag_id, 0)
    token_embeds = model.embed_tokens(tokens_to_embed)

    # Replace audio placeholder positions with audio embeddings
    input_embeds, target_ids, attention_mask = replace_placeholders_and_build_targets(
        input_ids=tokens,
        embeds=token_embeds,
        padding_id=model.text_pad_id,
        placeholder_id=model.audio_locator_tag_id,
        replacements=audio_embeds,
        target_ids=tokens.where(tokens != model.text_pad_id, -100),
    )

    # Next-token prediction shift
    input_embeds = input_embeds[:, :-1]
    attention_mask = attention_mask[:, :-1]
    target_ids = target_ids[:, 1:] # type: ignore

    # Forward through model
    outputs = model(input_embeds, attention_mask=attention_mask)
    logits = outputs["logits"]

    num_frames = (target_ids != -100).long().sum()
    loss = F.cross_entropy(
        logits.flatten(0, 1),
        target_ids.flatten(0, 1),
        reduction="sum",
        ignore_index=-100,
    ) / num_frames
    return loss


def modify_linear_with_rotation_param(
        linear: Union[nn.Linear, RMSNormFusedM, nn.Conv1d],
        Q: nn.Parameter,
        Q2: Optional[nn.Parameter] = None,
        for_rotated_input: bool = True,
        for_norm_out: bool = True,
        bit: int = 4,
        include_activation_quant: bool = False,
) -> None:
    """Modify the given linear layer to include the rotation parameter Q in its forward pass."""

    def modified_forward(self, x: torch.Tensor) -> torch.Tensor:
        # quantize the input activations with STE quantization
        if include_activation_quant:
            x = STEQuantize.apply(x, bit=8) # type: ignore
        # Apply the rotation to the weight
        rotated_bias = self.bias
        rotated_weight = self.weight
        dtype = self.weight.dtype
        orig_shape = self.weight.shape

        if for_norm_out:
            # W_rotated = Q^T @ diag(weight) @ Q  (full D×D matrix when weight is 1-D)
            w = linear.weight.double()
            if w.dim() == 1:
                w = torch.diag(w)
            rotated_weight = (Q.t().double() @ w) @ Q.double()
            if rotated_bias is not None:  
                rotated_bias = (rotated_bias.unsqueeze(0).double() @ Q.double())

        elif for_rotated_input:
            if Q is not None:
                rotated_weight = self.weight.to(Q.dtype).flatten(1) @ Q
            if Q2 is not None:
                hdim = Q2.shape[0]
                w_ = rotated_weight.t()
                org_shape = w_.shape
                temp = w_.reshape(-1, org_shape[-1]//hdim, hdim)
                temp = (temp.to(Q2.dtype) @ Q2)
                rotated_weight = temp.reshape(org_shape).t()
                if self.bias is not None:
                    org_shape = self.bias.shape
                    temp = self.bias.reshape(-1, org_shape[-1]//hdim, hdim)
                    temp = (temp.to(Q2.dtype) @ Q2)
                    rotated_bias = temp.reshape(org_shape).to(self.bias.dtype)

        else:
            if Q is not None:
                rotated_weight = Q.T @ self.weight.to(Q.dtype).flatten(1)
                if self.bias is not None:
                    rotated_bias = (self.bias.data.to(Q.dtype) @ Q).to(x.dtype)
            if Q2 is not None:
                hdim = Q2.shape[0]
                org_shape = rotated_weight.shape
                temp = rotated_weight.reshape(-1, org_shape[-1]//hdim, hdim)
                temp = temp.to(Q2.dtype) @ Q2
                rotated_weight = temp.reshape(org_shape)
        
        if for_norm_out and rotated_weight.shape != orig_shape:
            w = rotated_weight
            if rotated_bias is not None: rotated_bias = rotated_bias.squeeze(0).to(x.dtype)
        else:
            w = rotated_weight.reshape(orig_shape)
            if rotated_bias is not None: rotated_bias = rotated_bias.reshape(self.bias.shape).to(x.dtype)
        # continue with the normal linear forward using the rotated weight
        if isinstance(linear, nn.Linear):
            return F.linear(x, w.to(x.dtype), rotated_bias)
        elif isinstance(linear, nn.Conv1d):
            return F.conv1d(
                x, w.to(x.dtype), rotated_bias, self.stride, self.padding, self.dilation, self.groups
            )
        elif isinstance(linear, RMSNormFusedM):
            return F.rms_norm(x, normalized_shape=self.normalized_shape, eps=self.eps) @  w.to(x.dtype) + (rotated_bias if self.bias is not None else 0.0)
        else:
            raise Exception()
        
    linear.forward = types.MethodType(modified_forward, linear)


def get_canaryqwen_norm_fusion_config(
    num_encoder_layers: int = 12,
    num_decoder_layers: int = 36,
) -> List[Tuple[str, List[str], List[str]]]:
    """Return ``(norm_name, pre_linear_names, post_linear_names)`` for the Qwen3 decoder in CanaryQwen.

    Qwen3 already uses RMSNorm (no mean subtraction needed), so ``pre_linear_names``
    is always empty.  We fuse each RMSNorm's weight into the succeeding linear layers.

    After fusion the norms become plain ``RMSNorm(x)`` with weight = 1, which is
    rotation-equivariant.

    Note: The final ``llm.model.norm`` is NOT fused into ``llm.lm_head`` so that we
    can inverse-rotate before it during training/inference.  This keeps the lm_head
    in the original basis and avoids quantising it.
    """
    config: List[Tuple[str, List[str], List[str]]] = []

    for i in range(num_encoder_layers):
        p = f"perception.encoder.layers.{i}"

        # Feed Forward 1
        config.append((
            f"{p}.norm_feed_forward1",
            ["first_layer"] if i==0 else [f"perception.encoder.layers.{i-1}.norm_out"],
            [f"{p}.feed_forward1.linear1"],
        ))

        # Self Attention
        config.append((
            f"{p}.norm_self_att",
            [f"{p}.feed_forward1.linear2"],
            [f"{p}.self_attn.linear_q", f"{p}.self_attn.linear_k", f"{p}.self_attn.linear_v"],
        ))

        # Convolution Module
        config.append((
            f"{p}.norm_conv",
            [f"{p}.self_attn.linear_out"],
            [f"{p}.conv.pointwise_conv1"],
        ))

        # Feed Forward 2
        config.append((
            f"{p}.norm_feed_forward2",
            [f"{p}.conv.pointwise_conv2"],
            [f"{p}.feed_forward2.linear1"],
        ))

        # Output Layer Norm
        config.append((
            f"{p}.norm_out",
            [f"{p}.feed_forward2.linear2"],
            [],
        ))

    for i in range(num_decoder_layers):
        p = f"llm.base_model.model.model.layers.{i}"
        # input_layernorm weight → self_attn q/k/v_proj
        config.append((
            f"{p}.input_layernorm",
            ["first_layer"] if i==0 else [f"llm.base_model.model.model.layers.{i-1}.down_proj"],
            [f"{p}.self_attn.q_proj.base_layer", f"{p}.self_attn.k_proj", f"{p}.self_attn.v_proj.base_layer", 
             f"{p}.self_attn.q_proj.lora_A.default", f"{p}.self_attn.v_proj.lora_A.default", 
             ],
        ))
        # post_attention_layernorm weight → mlp gate_proj, up_proj
        config.append((
            f"{p}.post_attention_layernorm",
            [f"{p}.self_attn.o_proj"],
            [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"],
        ))
    # Final norm: NOT fused — we inverse-rotate before this norm and the lm_head
    # so they stay in the original representation basis.
    # config.append(("llm.model.norm", [], ["llm.lm_head"]))  # intentionally omitted

    return config


def get_canaryqwen_layers_to_rotate(
    num_encoder_layers: int = 12,
    num_decoder_layers: int = 36,
) -> List[Tuple[str, bool]]:
    """Return ``(layer_name, for_rotated_input)`` pairs for the Qwen3 decoder.

    A single rotation matrix *Q* is applied to the decoder's residual stream.
    Layers whose **input** comes from the rotated residual get
    ``for_rotated_input=True`` (weight right-multiplied by Q).  Layers whose
    **output** feeds back into the rotated residual get
    ``for_rotated_input=False`` (weight left-multiplied by Q^T).
    """
    layers: List[Tuple[str, bool]] = []

    for i in range(num_encoder_layers):
        p = f"perception.encoder.layers.{i}"
        layers.extend([
            # Self-attention
            (f"{p}.self_attn.linear_q", True),
            (f"{p}.self_attn.linear_k", True),
            (f"{p}.self_attn.linear_v", True),
            (f"{p}.self_attn.linear_out", False),
            # Convolution Module
            (f"{p}.conv.pointwise_conv1", True),
            (f"{p}.conv.pointwise_conv2", False),
            # MLP 1
            (f"{p}.feed_forward1.linear1", True),
            (f"{p}.feed_forward1.linear2", False),
            # MLP 2
            (f"{p}.feed_forward2.linear1", True),
            (f"{p}.feed_forward2.linear2", False),
        ]) 
        # Output Normalization (all layers, including last)
        layers.extend([
            (f"{p}.norm_out", False),
        ])

    for i in range(num_decoder_layers):
        p = f"llm.base_model.model.model.layers.{i}"
        layers.extend([
            # Self-attention
            (f"{p}.self_attn.q_proj.base_layer", True),
            (f"{p}.self_attn.q_proj.lora_A.default", True),
            (f"{p}.self_attn.k_proj", True),
            (f"{p}.self_attn.v_proj.base_layer", True),
            (f"{p}.self_attn.v_proj.lora_A.default", True),
            (f"{p}.self_attn.v_proj.lora_B.default", True),
            (f"{p}.self_attn.o_proj", False),
            # MLP
            (f"{p}.mlp.gate_proj", True),
            (f"{p}.mlp.up_proj", True),
            (f"{p}.mlp.down_proj", False),
        ])

    return layers


# ---------------------------------------------------------------------------
# Monkey-patched Qwen3Model.forward with rotation hooks
# ---------------------------------------------------------------------------
def canaryqwen_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds

    # ====== ROTATION: Rotate input to Q-basis ======
    hidden_states = self.process_residual_stream_input(hidden_states)
    # ===============================================
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    # ====== ROTATION: Inverse-rotate before final norm ======
    hidden_states = self.process_residual_stream_output(hidden_states)
    # ========================================================
    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )


def conformer_encoder_forward(
    self,
    audio_signal,
    length,
    cache_last_channel=None,
    cache_last_time=None,
    cache_last_channel_len=None,
    bypass_pre_encode=False,
):
    """
    Forward function for the ConformerEncoder accepting an audio signal and its corresponding length.
    The `audio_signal` input supports two formats depending on the `bypass_pre_encode` boolean flag.
    This determines the required format of the input variable `audio_signal`:
    (1) bypass_pre_encode = False (default):
        `audio_signal` must be a tensor containing audio features.
        Shape: (batch, self._feat_in, n_frames)
    (2) bypass_pre_encode = True:
        `audio_signal` must be a tensor containing pre-encoded embeddings.
        Shape: (batch, n_frame, self.d_model)
    """
    if not bypass_pre_encode and audio_signal.shape[-2] != self._feat_in:
        raise ValueError(
            f"If bypass_pre_encode is False, audio_signal should have shape "
            f"(batch, {self._feat_in}, n_frame) but got last dimension {audio_signal.shape[-2]}."
        )
    if bypass_pre_encode and audio_signal.shape[-1] != self.d_model:
        raise ValueError(
            f"If bypass_pre_encode is True, audio_signal should have shape "
            f"(batch, n_frame, {self.d_model}) but got last dimension {audio_signal.shape[-1]}."
        )

    if bypass_pre_encode:
        self.update_max_seq_length(seq_length=audio_signal.size(1), device=audio_signal.device)
    else:
        self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
    
    if length is None:
        length = audio_signal.new_full(
            (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
        )

    # select a random att_context_size with the distribution specified by att_context_probs during training
    # for non-validation cases like test, validation or inference, it uses the first mode in self.att_context_size
    if self.training and len(self.att_context_size_all) > 1:
        cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
    else:
        cur_att_context_size = self.att_context_size

    if not bypass_pre_encode:
        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
            length = length.to(torch.int64)
            # `self.streaming_cfg` is set by setup_streaming_cfg(), called in the init
            if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
                audio_signal = audio_signal[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
                length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

        if self.reduction_position is not None and cache_last_channel is not None:
            raise ValueError("Caching with reduction feature is not supported yet!")

    max_audio_length = audio_signal.size(1)
    if cache_last_channel is not None:
        cache_len = self.streaming_cfg.last_channel_cache_size
        cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
        max_audio_length = max_audio_length + cache_len
        padding_length = length + cache_len
        offset = torch.neg(cache_last_channel_len) + cache_len # type: ignore
    else:
        padding_length = length
        cache_last_channel_next = None
        cache_len = 0
        offset = None

    audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

    # Create the self-attention and padding masks
    pad_mask, att_mask = self._create_masks(
        att_context_size=cur_att_context_size,
        padding_length=padding_length,
        max_audio_length=max_audio_length,
        offset=offset,
        device=audio_signal.device,
    )

    if cache_last_channel is not None:
        pad_mask = pad_mask[:, cache_len:]
        if att_mask is not None:
            att_mask = att_mask[:, cache_len:]
        # Convert caches from the tensor to list
        cache_last_time_next = []
        cache_last_channel_next = []

    audio_signal = self.process_residual_stream_input(audio_signal)
    for lth, (drop_prob, layer) in enumerate(zip(self.layer_drop_probs, self.layers)):
        original_signal = audio_signal
        if cache_last_channel is not None:
            cache_last_channel_cur = cache_last_channel[lth]
            cache_last_time_cur = cache_last_time[lth] # type: ignore
        else:
            cache_last_channel_cur = None
            cache_last_time_cur = None
        audio_signal = layer(
            x=audio_signal,
            att_mask=att_mask,
            pos_emb=pos_emb,
            pad_mask=pad_mask,
            cache_last_channel=cache_last_channel_cur,
            cache_last_time=cache_last_time_cur,
        )

        if cache_last_channel_cur is not None:
            (audio_signal, cache_last_channel_cur, cache_last_time_cur) = audio_signal
            cache_last_channel_next.append(cache_last_channel_cur) # type: ignore
            cache_last_time_next.append(cache_last_time_cur) # type: ignore

        # applying stochastic depth logic from https://arxiv.org/abs/2102.03216
        if self.training and drop_prob > 0.0:
            should_drop = torch.rand(1) < drop_prob
            # adjusting to match expectation
            if should_drop:
                # that's not efficient, but it's hard to implement distributed
                # version of dropping layers without deadlock or random seed meddling
                # so multiplying the signal by 0 to ensure all weights get gradients
                audio_signal = audio_signal * 0.0 + original_signal
            else:
                # not doing this operation if drop prob is 0 as it's identity in that case
                audio_signal = (audio_signal - original_signal) / (1.0 - drop_prob) + original_signal

        if self.reduction_position == lth:
            audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)
            max_audio_length = audio_signal.size(1)
            # Don't update the audio_signal here because then it will again scale the audio_signal
            # and cause an increase in the WER
            _, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
            pad_mask, att_mask = self._create_masks(
                att_context_size=cur_att_context_size,
                padding_length=length,
                max_audio_length=max_audio_length,
                offset=offset,
                device=audio_signal.device,
            )

        # saving tensors if required for interctc loss
        if self.is_access_enabled(getattr(self, "model_guid", None)):
            if self.interctc_capture_at_layers is None:
                self.interctc_capture_at_layers = self.access_cfg.get('interctc', {}).get('capture_layers', [])
            if lth in self.interctc_capture_at_layers:
                lth_audio_signal = audio_signal
                if self.out_proj is not None:
                    lth_audio_signal = self.out_proj(audio_signal)
                # shape is the same as the shape of audio_signal output, i.e. [B, D, T]
                self.register_accessible_tensor(
                    name=f'interctc/layer_output_{lth}', tensor=torch.transpose(lth_audio_signal, 1, 2)
                )
                self.register_accessible_tensor(name=f'interctc/layer_length_{lth}', tensor=length)

    # Inverse-rotate encoder output back to the original basis
    audio_signal = self.process_residual_stream_output(audio_signal)

    if self.out_proj is not None:
        audio_signal = self.out_proj(audio_signal)

    # Reduction
    if self.reduction_position == -1:
        audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)

    audio_signal = torch.transpose(audio_signal, 1, 2)
    length = length.to(dtype=torch.int64)

    if cache_last_channel is not None:
        cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0) # type: ignore
        cache_last_time_next = torch.stack(cache_last_time_next, dim=0) # type: ignore
        return (
            audio_signal,
            length,
            cache_last_channel_next,
            cache_last_time_next,
            torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len), # type: ignore
        )
    else:
        return audio_signal, length


def monkey_patch_canaryqwen_for_train(model: nn.Module, Qe: nn.Parameter, Qd: nn.Parameter) -> None:
    """Monkey-patch the Qwen3Model forward to apply Q rotation at residual stream boundaries."""
    qwen3_model = model.llm.base_model.model.model  # Qwen3Model # type: ignore
    conformer_encoder = model.perception.encoder  # ConformerEncoder # type: ignore

    def process_residual_stream_input_decoder(self, x):
        dtype = x.dtype
        return (x.double() @ Qd.double().to(x.device)).to(dtype)

    def process_residual_stream_input_encoder(self, x):
        x = x - x.mean(dim=-1, keepdim=True)  # zero-centering for LayerNorms in the encoder
        dtype = x.dtype
        return (x.double() @ Qe.double().to(x.device)).to(dtype)

    def process_residual_stream_output_decoder(self, x):
        dtype = x.dtype
        return (x.double() @ Qd.t().double().to(x.device)).to(dtype)

    qwen3_model.forward = types.MethodType(canaryqwen_model_forward, qwen3_model) # type: ignore
    qwen3_model.process_residual_stream_input = types.MethodType( # type: ignore
        process_residual_stream_input_decoder, qwen3_model
    )
    qwen3_model.process_residual_stream_output = types.MethodType( # type: ignore
        process_residual_stream_output_decoder, qwen3_model
    )
    def process_residual_stream_output_encoder(self, x):
        dtype = x.dtype
        return (x.double() @ Qe.t().double().to(x.device)).to(dtype)

    conformer_encoder.forward = types.MethodType(conformer_encoder_forward, conformer_encoder) # type: ignore
    conformer_encoder.process_residual_stream_input = types.MethodType( # type: ignore
        process_residual_stream_input_encoder, conformer_encoder
    )
    conformer_encoder.process_residual_stream_output = types.MethodType( # type: ignore
        process_residual_stream_output_encoder, conformer_encoder
    )
    

def prepare_canaryqwen_for_rotation(model: nn.Module) -> None:
    """Prepare the CanaryQwen model for rotation.

    2. Fuse Qwen3 RMSNorm weights (input_layernorm, post_attention_layernorm)
       into adjacent linear layers.
    3. No need to fuse q_norm/k_norm into q_proj/k_proj since no rotation will be applied to them.

    After this, all norms have weight=1 and are rotation-equivariant.
    """
    num_encoder_layers = len(model.perception.encoder.layers) # type: ignore
    num_decoder_layers = model.llm.config.num_hidden_layers # type: ignore

    # Convert LayerNorms to RMSNorms and fuse normalization weights into adjacent linears
    convert_model_layernorms_to_rmsnorms(model) 

    # Fuse RMSNorm weights into succeeding linears
    norm_fusion_cfg = get_canaryqwen_norm_fusion_config(num_encoder_layers,num_decoder_layers) # type: ignore
    fuse_normalization_weights_and_bias_into_adjacent_linears(model, norm_fusion_cfg) # type: ignore

    # Fuse q_norm/k_norm weights into q_proj/k_proj
    # No need for this
    # fuse_qkv_norms(model, num_decoder_layers)


def modify_canaryqwen_layers_with_rotation_params(
    model: nn.Module, Qe: nn.Parameter, Qd: nn.Parameter, Q2s: Dict[str, nn.Parameter], 
    include_weight_quant: bool = False,
    include_activation_quant: bool = False,
) -> None:
    """Modify linear layers to include rotation parameters Qe (encoder) and Qd (decoder) in their forward pass."""
    num_encoder_layers = len(model.perception.encoder.layers) # type: ignore
    num_decoder_layers = model.llm.config.num_hidden_layers # type: ignore
    layers_to_rotate = get_canaryqwen_layers_to_rotate(num_encoder_layers, num_decoder_layers) # type: ignore
    named_modules = dict(model.named_modules())
    for layer_name, for_rotated_input in layers_to_rotate:
        layer = named_modules.get(layer_name)
        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model.")
        stem_name, leaf_name = layer_name.rsplit(".", 1)
        Q_layer = Qe if "encoder" in layer_name else Qd  # per-layer copy; never overwrite the outer Q
        Q2_layer = None

        # For decoder
        if "v_proj" in layer_name:
            left, right = layer_name.split(".v_proj")
            if right == ".base_layer":
                Q2_layer = Q2s.get(left, None)
            elif right == ".lora_A.default":
                Q2_layer = None
            elif right == ".lora_B.default":
                Q_layer = None
                Q2_layer = Q2s.get(left, None)
            else:
                raise ValueError(f"Unexpected v_proj layer name format: '{layer_name}'")
            
        elif layer_name.endswith("o_proj"):
             Q2_layer = Q2s.get(stem_name, None)
             assert Q2_layer is not None, f"Q2 for layer '{stem_name}' not found in Q2s dictionary."
        elif "linear_v" in layer_name or "linear_out" in layer_name:
            Q2_layer = Q2s.get(stem_name, None)
            assert Q2_layer is not None, f"Q2 for layer '{stem_name}' not found in Q2s dictionary." 

        if leaf_name in ["down_proj"]:
            include_activation_quant = False

        for_norm_out = False
        # For Encoder
        if leaf_name in ["linear_v", "linear_out"]:
            Q2_layer = Q2s.get(stem_name, None)
            assert Q2_layer is not None, f"Q2 for layer '{stem_name}' not found in Q2s dictionary."
        # norm out
        elif leaf_name == "norm_out":
            # norm out has a linear layer multiplied on the left by Q^T and by Q
            # layer(XQ^T)Q
            for_norm_out = True

        modify_linear_with_rotation_param(
            layer, Q_layer, Q2=Q2_layer, for_rotated_input=for_rotated_input, # type: ignore
            for_norm_out=for_norm_out, bit=4, include_activation_quant=include_activation_quant
        )


def fuse_rotation_param_into_linear(
        linear: nn.Linear,
        Q: torch.Tensor,
        Q2: Optional[torch.Tensor] = None,
        for_rotated_input: bool = True,
) -> None:
    """Fuse the rotation parameter Q into the given linear layer's weights (and bias if for_rotated_input=False)."""
    dtype = linear.weight.data.dtype
    device = linear.weight.data.device
    if isinstance(linear, RMSNormFusedM):
        w = linear.weight.data.double()
        if w.dim() == 1:
            w = torch.diag(w)
        linear.weight.data = (Q.double().t() @ w @ Q.double()).to(linear.weight.dtype)
        if linear.bias is not None:
            linear.bias.data = (linear.bias.data.unsqueeze(0).double() @ Q.double()).to(linear.bias.dtype)
    elif for_rotated_input: 
        if Q is not None:
            Q_d = Q.double().to(device)
            linear.weight.data = (linear.weight.data.double().flatten(1) @ Q_d).to(dtype=dtype, device=device).reshape(linear.weight.shape)
        if Q2 is not None:
            hdim = Q2.shape[0]
            w_ = linear.weight.data.double().t()
            org_shape = w_.shape
            temp = w_.reshape(-1, org_shape[-1]//hdim, hdim)
            temp = (temp.double() @ Q2.double())
            linear.weight.data = temp.reshape(org_shape).t().to(dtype=dtype, device=device)
            if linear.bias is not None:
                org_shape = linear.bias.shape
                temp = linear.bias.data.double().reshape(-1, org_shape[-1]//hdim, hdim)
                temp = (temp.double() @ Q2.double())
                linear.bias.data = temp.reshape(org_shape).to(dtype=linear.bias.data.dtype, device=linear.bias.data.device)
    else:
        if Q is not None:
            Q_d = Q.double().to(device)
            linear.weight.data = (Q_d.T @ linear.weight.data.double().flatten(1)).to(dtype=dtype, device=device).reshape(linear.weight.shape)
            if linear.bias is not None:
                linear.bias.data = (linear.bias.data.double().unsqueeze(0) @ Q_d).to(dtype=dtype, device=device).reshape(linear.bias.shape)
        if Q2 is not None:
            hdim = Q2.shape[0]
            # No transpose here: Q2 rotates within heads of the INPUT dimension
            # (last dim of weight shape (out, in)), matching the on-the-fly version.
            w_ = linear.weight.data.double()
            org_shape = w_.shape
            temp = w_.reshape(-1, org_shape[-1]//hdim, hdim)
            temp = (temp.double() @ Q2.double())
            linear.weight.data = temp.reshape(org_shape).to(dtype=dtype, device=device)


def fuse_canaryqwen_layers_with_rotations(
    model: nn.Module, Qe: torch.Tensor, Qd: torch.Tensor, Q2s: Dict[str, torch.Tensor],
    device = "cuda"
) -> None:
    """Fuse rotation matrices Q (and Q2) into the model's linear layer weights."""
    named_modules = dict(model.named_modules())
    num_encoder_layers = len(model.perception.encoder.layers) # type: ignore
    num_decoder_layers = model.llm.config.num_hidden_layers # type: ignore
    layers_to_rotate = get_canaryqwen_layers_to_rotate(num_encoder_layers, num_decoder_layers) # type: ignore
    for layer_name, for_rotated_input in layers_to_rotate:
        layer = named_modules.get(layer_name)
        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model.")
        stem_name, leaf_name = layer_name.rsplit(".", 1)  # per-layer copy; never overwrite the outer Q
        Q_layer = Qe if "encoder" in layer_name else Qd
        Q2_layer = None

        if "v_proj" in layer_name:
            left, right = layer_name.split(".v_proj")
            if right == ".base_layer":
                Q2_layer = Q2s.get(left, None)
            elif right == ".lora_A.default":
                Q2_layer = None
            elif right == ".lora_B.default":
                Q_layer = None
                Q2_layer = Q2s.get(left, None)
            else:
                raise ValueError(f"Unexpected v_proj layer name format: '{layer_name}'")
        elif leaf_name == "o_proj":
             Q2_layer = Q2s.get(stem_name, None)
             assert Q2_layer is not None, f"Q2 for layer '{stem_name}' not found in Q2s dictionary."

        # For Encoder
        if leaf_name in ["linear_v", "linear_out"]:
            Q2_layer = Q2s.get(stem_name, None)
            assert Q2_layer is not None, f"Q2 for layer '{stem_name}' not found in Q2s dictionary."

        if Q_layer is not None: Q_layer = Q_layer.to(device)
        if Q2_layer is not None: Q2_layer = Q2_layer.to(device)

        fuse_rotation_param_into_linear(layer, Q_layer, Q2=Q2_layer, for_rotated_input=for_rotated_input) # type: ignore


def transcribe(model, filepath):
    answer_ids = model.generate(
        prompts=[
            [{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}", "audio": [f"{filepath}"]}]
        ],
        max_new_tokens=128,
    )
    transcript = (model.tokenizer.ids_to_text(answer_ids[0].cpu()))
    return transcript


def obtain_rotations_for_canary_qwen(model, test_audio_path:str, calib_samples:int, epochs:int, batch_size:int, lr:float, save_path:str):
    if os.path.exists(save_path):
        # exit
        sys.exit(0)

    model.to("cuda")

    # Get original transcription before any modification
    with torch.no_grad():
        orig_transcription = transcribe(model, test_audio_path)

    # Prepare model for rotation (merge LoRA, fuse norms)
    with torch.no_grad():
        prepare_canaryqwen_for_rotation(model)

    # Sanity check: Ensure transcription is unchanged after preparation steps
    with torch.no_grad():
        prep_transcription = transcribe(model, test_audio_path)
        assert orig_transcription == prep_transcription, (
            f"Transcriptions do not match after preparation steps!\n"
            f"Original: '{orig_transcription}'\n"
            f"After Preparation: '{prep_transcription}'"
        )

    # Create rotation matrices
    mode = "hadamard"
    hidden_size = model.llm.config.hidden_size
    encoder_hidden_size = model.perception.encoder.layers[0].conv.d_model
    num_decoder_layers = model.llm.config.num_hidden_layers
    num_encoder_layers = len(model.perception.encoder.layers)
    Qe = get_orthogonal_matrix(encoder_hidden_size, mode=mode, device="cuda")
    Qd = get_orthogonal_matrix(hidden_size, mode=mode, device="cuda")
    Q2s = {}
    for i in range(num_decoder_layers):
        head_dim = model.llm.base_model.model.model.layers[i].self_attn.head_dim
        rot = get_orthogonal_matrix(head_dim, mode=mode, device="cuda")
        Q2s[f"llm.base_model.model.model.layers.{i}.self_attn"] = rot
    for i in range(num_encoder_layers):
        head_dim = model.perception.encoder.layers[i].self_attn.d_k
        rot = get_orthogonal_matrix(head_dim, mode=mode, device="cuda")
        Q2s[f"perception.encoder.layers.{i}.self_attn"] = rot
    
    # Make Q, Q2s trainable parameters
    Qe = nn.Parameter(Qe.float(), requires_grad=True)
    Qd = nn.Parameter(Qd.float(), requires_grad=True)
    for k in Q2s:
        Q2s[k] = nn.Parameter(Q2s[k].double(), requires_grad=True)

    # Modify linear layers to include rotation in their forward pass
    modify_canaryqwen_layers_with_rotation_params(
        model, Qe, Qd, Q2s,
        include_weight_quant=False, include_activation_quant=False
    )
    # Monkey-patch the Qwen3Model forward to rotate residual stream
    monkey_patch_canaryqwen_for_train(model, Qe, Qd)

    # Ensure model remains computationally invariant despite the rotations
    with torch.no_grad():
        rot_transcription = transcribe(model, "outputs/rotation_test_audio.wav")
        print(f"Original Transcription: '{orig_transcription}'\n")
        print(f"Transcription after Rotation: '{rot_transcription}'")
        assert orig_transcription == rot_transcription, (
            f"Transcriptions do not match after applying rotations!\n"
            f"Original: '{orig_transcription}'\n"
            f"After Rotation: '{rot_transcription}'"
        )
    

    # Build calibration dataset and dataloader
    from functools import partial
    calib_ds = CanaryQwenCalibrationDataset(model, num_samples=calib_samples)
    collate_fn = partial(canaryqwen_collate_fn, pad_id=model.text_pad_id)
    train_loader = torch.utils.data.DataLoader(
        calib_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # Train rotation parameters
    trainable_params = [Qe, Qd] + list(Q2s.values())
    optimizer = SGDG(trainable_params, lr=lr, stiefel=True)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = canaryqwen_loss_fn(model, batch)
            loss.backward()
            optimizer.step()
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
    
def rotate_canary_qwen(model, test_audio_file:str, rotation_path:str, device="cuda"):
    device = model.device
    with torch.no_grad():
        orig_transcription = transcribe(model, test_audio_file)

    prepare_canaryqwen_for_rotation(model)  # merge LoRA, fuse norms
    rotations = torch.load(rotation_path)  # load learned rotations
    Qe = rotations["Qe"]
    Qd = rotations["Qd"]
    Q2s = rotations["Q2s"]
    fuse_canaryqwen_layers_with_rotations(model, Qe, Qd, Q2s, device=device)  # fuse rotations into weights
    monkey_patch_canaryqwen_for_train(model, Qe.to(device), Qd.to(device))

    with torch.no_grad():
        rot_transcription = transcribe(model, "outputs/rotation_test_audio.wav")
        assert orig_transcription == rot_transcription, (
            f"Transcriptions do not match after fusing rotations!\n"
            f"Original: '{orig_transcription}'\n"
            f"After Rotation: '{rot_transcription}'"
        )
