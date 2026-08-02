# pyright: reportMissingImports=false
# pyright: reportPrivateImportUsage=false

import torch
import torch.nn as nn
import torch.nn.functional as F
import types

from asrq.transforms.rotation.utils import (
    RMSNormFusedM,
    STEQuantize,
    ste_quantize_weight,
    modify_linear_with_rotation_param,
    fuse_rotation_param_into_linear,
    convert_model_layernorms_to_rmsnorms,
    fuse_hadamard_into_linear,
    fuse_normalization_weights_and_bias_into_adjacent_linears,
    get_orthogonal_matrix,
    matches_layer_suffix,
    apply_to_rotated_layers,
)
from asrq.transforms.rotation.hadamard_utils import matmul_hadU_auto
from asrq.transforms.rotation.hadamard_search import (
    HadamardSearchConfig,
    check_search_is_meaningful,
    evolutionary_sign_search,
    make_batch_evaluator,
    write_rotations_,
)
from asrq.transforms.rotation.cayley_sgd import SGDG
from functools import partial

try:
    from nemo.collections.asr.modules.conformer_encoder import (
        random
    )
except ImportError:
    random = None
from datasets import load_dataset
from itertools import islice

from typing import Any, List, Tuple, Optional, Union, Dict




class ParakeetCalibrationDataset(torch.utils.data.Dataset):
    """LibriSpeech train-clean-100 samples formatted for CanaryQwen rotation training."""

    def __init__(self, model, num_samples=128, seed=42):
        super().__init__()
        ds = load_dataset("librispeech_asr", "all", split="train.clean.100")
        ds = ds.shuffle(seed=seed)
        self.samples = list(islice(ds, num_samples))
        self.model = model

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio = torch.tensor(sample["audio"]["array"], dtype=torch.float32)
        audio_len = torch.tensor(audio.shape[0], dtype=torch.long)
        text = sample["text"]

        # Append transcription tokens and EOS
        tokens = torch.tensor(self.model.tokenizer.text_to_ids(text), dtype=torch.long)
        eos = torch.tensor([self.model.tokenizer.eos], dtype=torch.long)
        tokens = torch.cat([tokens, eos])
        tokens_len = torch.tensor(tokens.shape[0], dtype=torch.long)

        return audio, audio_len, tokens, tokens_len


def parakeet_ctc_collate_fn(batch, pad_id):
    """Collate function for CanaryQwenCalibrationDataset."""
    audios_list = [b[0] for b in batch]
    audio_lens = torch.stack([b[1] for b in batch])
    tokens_list = [b[2] for b in batch]
    token_lens = torch.stack([b[3] for b in batch])

    # Right-pad audios with zeros
    max_audio_len = max(a.shape[0] for a in audios_list)
    audios = torch.zeros(len(batch), max_audio_len, dtype=torch.float32)
    for i, a in enumerate(audios_list):
        audios[i, : a.shape[0]] = a

    # right-pad tokens (CTC convention)
    max_token_len = max(t.shape[0] for t in tokens_list)
    tokens = torch.full((len(batch), max_token_len), pad_id, dtype=torch.long)
    for i, t in enumerate(tokens_list):
        tokens[i, : t.shape[0]] = t

    return {"audios": audios, "audio_lens": audio_lens, "tokens": tokens, "token_lens": token_lens}


def parakeet_ctc_loss_fn(model, batch):
    """Compute next-token-prediction cross-entropy loss for rotation training."""
    device = next(model.parameters()).device
    audios = batch["audios"].to(device)
    audio_lens = batch["audio_lens"].to(device)
    tokens = batch["tokens"].to(device)
    token_lens = batch["token_lens"].to(device)

    log_probs, encoded_len, predictions = model(input_signal=audios, input_signal_length=audio_lens)
    loss = model.loss(
            log_probs=log_probs, targets=tokens, input_lengths=encoded_len, target_lengths=token_lens
    ) 

    return loss


# =====================================================================================================================
# =====================================================================================================================


def get_parakeet_ctc_norm_fusion_config(
    num_encoder_layers: int = 12,
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
        p = f"encoder.layers.{i}"

        # Feed Forward 1
        config.append((
            f"{p}.norm_feed_forward1",
            ["first_layer"] if i==0 else [f"encoder.layers.{i-1}.norm_out"],
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

    return config


def get_parakeet_ctc_layers_to_rotate(
    num_encoder_layers: int = 12,
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
        p = f"encoder.layers.{i}"
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

    return layers



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
        cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0] # type: ignore
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


def monkey_patch_parakeet_ctc_for_train(model: nn.Module, Qe: nn.Parameter) -> None:
    """Monkey-patch the Qwen3Model forward to apply Q rotation at residual stream boundaries."""
    conformer_encoder = model.encoder  # ConformerEncoder

    def process_residual_stream_input_encoder(self, x):
        x = x - x.mean(dim=-1, keepdim=True)  # zero-centering for LayerNorms in the encoder
        dtype = x.dtype
        return (x.double() @ Qe.double().to(x.device)).to(dtype)

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
    

def prepare_parakeet_ctc_for_rotation(model: nn.Module) -> None:
    """Prepare the CanaryQwen model for rotation.

    2. Fuse Qwen3 RMSNorm weights (input_layernorm, post_attention_layernorm)
       into adjacent linear layers.
    3. No need to fuse q_norm/k_norm into q_proj/k_proj since no rotation will be applied to them.

    After this, all norms have weight=1 and are rotation-equivariant.
    """
    num_encoder_layers = len(model.encoder.layers) # type: ignore

    # Convert LayerNorms to RMSNorms and fuse normalization weights into adjacent linears
    convert_model_layernorms_to_rmsnorms(model)

    # Fuse RMSNorm weights into succeeding linears
    norm_fusion_cfg = get_parakeet_ctc_norm_fusion_config(num_encoder_layers)
    fuse_normalization_weights_and_bias_into_adjacent_linears(model, norm_fusion_cfg) # type: ignore


# The Conformer's two feed-forward down-projections: the layers whose input is the MLP
# intermediate rather than the Q-rotated residual stream, so they are where the online
# Hadamard goes.
DOWN_PROJ_SUFFIXES: Tuple[str, ...] = (
    "feed_forward1.linear2",
    "feed_forward2.linear2",
    # Fed by the conv module's activation, so a down-projection in all but name.
    "conv.pointwise_conv2",
)


def resolve_parakeet_ctc_rotations(layer_name: str, Qe, Q2s: Dict[str, Any]):
    """Which rotations this layer needs: ``(Q, Q2, for_norm_out)``.

    The Conformer encoder has a single residual stream, so every layer takes Qe. Q2 is the
    head-wise rotation, taken only by linear_v/linear_out. norm_out is the one layer whose
    weight becomes ``Q^T diag(w) Q`` rather than a one-sided rotation.
    """
    stem_name, leaf_name = layer_name.rsplit(".", 1)
    Q2, for_norm_out = None, False
    if leaf_name in ("linear_v", "linear_out"):
        Q2 = Q2s.get(stem_name)
        assert Q2 is not None, f"Q2 for layer '{stem_name}' not found in Q2s dictionary."
    elif leaf_name == "norm_out":
        for_norm_out = True
    return Qe, Q2, for_norm_out


def _parakeet_ctc_layers(model):
    return get_parakeet_ctc_layers_to_rotate(len(model.encoder.layers))  # type: ignore


def modify_parakeet_ctc_layers_with_rotation_params(
    model: nn.Module, Qe: nn.Parameter, Q2s: Dict[str, nn.Parameter],
    activation_bits: int = 16, online_hadamard: bool = True,
    quantize_weights: bool = False, weight_bits: int = 4, weight_group_size=None,
) -> None:
    """Rotate the Conformer encoder's layers on the fly (training / search path)."""
    def resolve(layer_name):
        Q, Q2, for_norm_out = resolve_parakeet_ctc_rotations(layer_name, Qe, Q2s)
        return Q, Q2, dict(
            for_norm_out=for_norm_out, bit=weight_bits, activation_bits=activation_bits,
            quantize_weights=quantize_weights, weight_group_size=weight_group_size,
            online_hadamard=online_hadamard and matches_layer_suffix(layer_name, DOWN_PROJ_SUFFIXES),
        )

    apply_to_rotated_layers(model, _parakeet_ctc_layers(model), resolve, modify_linear_with_rotation_param)


def fuse_parakeet_ctc_layers_with_rotations(
    model: nn.Module, Qe: torch.Tensor, Q2s: Dict[str, torch.Tensor],
    device = "cuda", online_hadamard: bool = True,
    hadamard_block_size: Optional[int] = None,
) -> None:
    """Bake the Conformer encoder's rotations into the weights (inference path).

    ``online_hadamard=True`` additionally fuses a Hadamard into each down-projection's
    weight and leaves a fast Hadamard transform on its input.
    """
    def resolve(layer_name):
        # for_norm_out is not passed here: the fuse path detects RMSNormFusedM by type.
        Q, Q2, _ = resolve_parakeet_ctc_rotations(layer_name, Qe, Q2s)
        return Q.to(device), (Q2.to(device) if Q2 is not None else None), dict(
            online_hadamard=online_hadamard and matches_layer_suffix(layer_name, DOWN_PROJ_SUFFIXES),
            hadamard_block_size=hadamard_block_size,
        )

    apply_to_rotated_layers(model, _parakeet_ctc_layers(model), resolve, fuse_rotation_param_into_linear)


def transcribe(model, filepath):
    transcriptions = [
        val.text for val in
        model.transcribe([filepath], batch_size=1, verbose=False, num_workers=1)
    ]
    return transcriptions


def obtain_rotations_for_parakeet(model, text_audio_path:str, calib_samples:int, epochs:int, batch_size:int, lr:float, save_path:str, device="cuda",
                                  activation_bits: int = 16, online_hadamard: bool = True):
    
    # load parakeet_ctc model
    model.to(device)

    # Get original transcription before any modification
    with torch.no_grad():
        orig_transcription = transcribe(model, text_audio_path)

    # Prepare model for rotation (merge LoRA, fuse norms)
    with torch.no_grad():
        prepare_parakeet_ctc_for_rotation(model)

    # Sanity check: Ensure transcription is unchanged after preparation steps
    with torch.no_grad():
        prep_transcription = transcribe(model, text_audio_path)
        assert orig_transcription == prep_transcription, (
            f"Transcriptions do not match after preparation steps!\n"
            f"Original: '{orig_transcription}'\n"
            f"After Preparation: '{prep_transcription}'"
        )

    # Create rotation matrices
    mode = "hadamard"
    encoder_hidden_size = model.encoder.layers[0].conv.d_model
    num_encoder_layers = len(model.encoder.layers)
    _seed = torch.initial_seed() & 0xFFFFFFFF
    _sc = 0
    Qe = get_orthogonal_matrix(encoder_hidden_size, mode=mode, device="cuda", seed=_seed + _sc); _sc += 1
    Q2s = {}
    for i in range(num_encoder_layers):
        head_dim = model.encoder.layers[i].self_attn.d_k
        rot = get_orthogonal_matrix(head_dim, mode=mode, device="cuda", seed=_seed + _sc); _sc += 1
        Q2s[f"encoder.layers.{i}.self_attn"] = rot
    
    # Make Q, Q2s trainable parameters
    Qe = nn.Parameter(Qe.float(), requires_grad=True)
    for k in Q2s:
        Q2s[k] = nn.Parameter(Q2s[k].double(), requires_grad=True)

    # Modify linear layers to include rotation in their forward pass
    modify_parakeet_ctc_layers_with_rotation_params(
        model, Qe, Q2s, activation_bits=activation_bits, online_hadamard=online_hadamard,
        quantize_weights=quantize_weights, weight_bits=weight_bits, weight_group_size=weight_group_size,
    )
    # Monkey-patch the Qwen3Model forward to rotate residual stream
    monkey_patch_parakeet_ctc_for_train(model, Qe)

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
    calib_ds = ParakeetCalibrationDataset(model, num_samples=calib_samples, seed=_seed)
    pad_id = model.tokenizer.pad_id if hasattr(model.tokenizer, "pad_id") and model.tokenizer.pad_id > 0 else 0
    collate_fn = partial(parakeet_ctc_collate_fn, pad_id=pad_id)
    _dl_generator = torch.Generator()
    _dl_generator.manual_seed(_seed + _sc)
    train_loader = torch.utils.data.DataLoader(
        calib_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=_dl_generator,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # Train rotation parameters
    trainable_params = [Qe] + list(Q2s.values())
    optimizer = SGDG(trainable_params, lr=lr, stiefel=True)
    model.train()
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
            loss = parakeet_ctc_loss_fn(model, batch)
            # ctc_loss_backward_gpu has no deterministic kernel; disable only for this call
            torch.use_deterministic_algorithms(False)
            loss.backward()
            torch.use_deterministic_algorithms(True)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        # breakpoint()

    to_save = {
        "Qe": Qe.data.detach().cpu(),
        "Q2s": {k: v.data.detach().cpu() for k, v in Q2s.items()},
    }
    torch.save(to_save, save_path)


def search_rotations_for_parakeet(
        model, calib_samples: int, batch_size: int, save_path: str, activation_bits: int,
        search_cfg: HadamardSearchConfig = HadamardSearchConfig(),
        online_hadamard: bool = True, device: str = "cuda",
        quantize_weights: bool = False, weight_bits: int = 4, weight_group_size=None,
        rotation_block_size=None,
    ) -> None:
    """Pick Qe/Q2s by searching the sign vectors of randomized Hadamard rotations.

    Saves in the same format as :func:`obtain_rotations_for_parakeet`, so the result is
    applied by the usual :func:`rotate_parakeet` path.
    """
    check_search_is_meaningful(activation_bits, quantize_weights)
    model.to(device)
    with torch.no_grad():
        prepare_parakeet_ctc_for_rotation(model)

    num_encoder_layers = len(model.encoder.layers)
    sign_sizes = {"Qe": model.encoder.layers[0].conv.d_model}
    for i in range(num_encoder_layers):
        sign_sizes[f"encoder.layers.{i}.self_attn"] = model.encoder.layers[i].self_attn.d_k

    params = {
        name: nn.Parameter(torch.eye(size, device=device, dtype=torch.float32), requires_grad=False)
        for name, size in sign_sizes.items()
    }
    Qe = params["Qe"]
    Q2s = {name: p for name, p in params.items() if name != "Qe"}

    modify_parakeet_ctc_layers_with_rotation_params(
        model, Qe, Q2s, activation_bits=activation_bits, online_hadamard=online_hadamard
    )
    monkey_patch_parakeet_ctc_for_train(model, Qe)

    pad_id = model.tokenizer.pad_id if hasattr(model.tokenizer, "pad_id") and model.tokenizer.pad_id > 0 else 0
    calib_ds = ParakeetCalibrationDataset(model, num_samples=calib_samples)
    loader = torch.utils.data.DataLoader(
        calib_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=partial(parakeet_ctc_collate_fn, pad_id=pad_id),
    )
    batches = [b for _, b in zip(range(search_cfg.batches_per_eval), loader)]
    model.eval()

    # Only the residual-stream rotations are block diagonal; Q2 stays a full head_dim
    # rotation, since a head is already smaller than a weight quantization group.
    block_sizes = {name: (rotation_block_size if name in ("Qe", "Qd") else None) for name in sign_sizes}
    evaluate = make_batch_evaluator(model, parakeet_ctc_loss_fn, batches, params, block_sizes)
    best_signs, best_loss, history = evolutionary_sign_search(sign_sizes, evaluate, search_cfg)
    print(f"[hadamard-search] parakeet best loss {best_loss:.6f} (from {history[0]:.6f})")

    write_rotations_(params, best_signs, block_sizes)
    torch.save(
        {"Qe": Qe.data.detach().cpu(), "Q2s": {k: v.data.detach().cpu() for k, v in Q2s.items()}},
        save_path,
    )


def rotate_parakeet(model, test_audio_file:str, rotation_path:str, device="cuda", online_hadamard: bool = True):
    with torch.no_grad():
        orig_transcription = transcribe(model, test_audio_file)

    prepare_parakeet_ctc_for_rotation(model)  # merge LoRA, fuse norms
    rotations = torch.load(rotation_path)  # load learned rotations
    Qe = rotations["Qe"]
    Q2s = rotations["Q2s"]
    fuse_parakeet_ctc_layers_with_rotations(model, Qe, Q2s, device=device, online_hadamard=online_hadamard)  # fuse rotations into weights
    monkey_patch_parakeet_ctc_for_train(model, Qe.to(device))

    with torch.no_grad():
        rot_transcription = transcribe(model, test_audio_file)
        assert orig_transcription == rot_transcription, (
            f"Transcriptions do not match after fusing rotations!\n"
            f"Original: '{orig_transcription}'\n"
            f"After Rotation: '{rot_transcription}'"
        )
