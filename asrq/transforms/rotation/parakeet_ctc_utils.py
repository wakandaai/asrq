"""Parakeet-CTC-specific inputs for learn_rotations.

Provides the calibration dataloader, loss function, norm-to-neighbour mapping and rotated-layer
list for a NeMo Conformer CTC model.

A Conformer block ends with ``norm_out``, so without help the next block's first norm would have
a norm rather than a Linear feeding it, and the centering would have nothing to fold into. An
identity Linear is inserted after each block's ``norm_out`` to absorb it; see
insert_conformer_block_output_linears.
"""

from functools import partial
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from asrq.transforms.rotation.utils import (
    add_residual_stream_entry_hook,
    add_unrotate_input_hook,
    insert_linear_after_norm,
)

# Rotation types, as learn_rotations dispatches them.
ROT_INPUT_R1 = 1  # Linear(X R1.T)        -- q, k, the first layer of each feed-forward
ROT_INPUT_R1_OUTPUT_R2 = 2  # Linear(X R1.T) R2     -- v
ROT_INPUT_R2_OUTPUT_R1 = 3  # Linear(X R2.T) R1     -- attention output
ROT_OUTPUT_R1 = 4  # Linear(X) R1          -- layers reading a nonlinearity's output
ROT_INPUT_R1_OUTPUT_R1 = 5  # Linear(X R1.T) R1     -- the inserted block-output Linear


class ParakeetCalibrationDataset(torch.utils.data.Dataset):
    """Calibration utterances as CTC batches for the rotation search.

    Takes the ``(waveform, text)`` pairs asrq.calibration.data loads, whose text is the
    full-precision model's own transcript. The KL objective reads only the audio; the tokens
    are for the ``"ce"`` objective's CTC loss, which then targets what the model itself emits
    rather than reference text in another format.
    """

    def __init__(self, model, samples: Sequence[Tuple[np.ndarray, str]]):
        super().__init__()
        self.model = model
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        waveform, text = self.samples[idx]
        audio = torch.as_tensor(np.asarray(waveform, dtype=np.float32))
        tokens = torch.tensor(self.model.tokenizer.text_to_ids(text), dtype=torch.long)
        return (
            audio,
            torch.tensor(audio.shape[0], dtype=torch.long),
            tokens,
            torch.tensor(tokens.shape[0], dtype=torch.long),
        )


def parakeet_ctc_collate_fn(batch, pad_id: int):
    """Right-pad audio with zeros and tokens with pad_id; CTC takes the true lengths."""
    audio_lens = torch.stack([length for _, length, _, _ in batch])
    token_lens = torch.stack([length for _, _, _, length in batch])

    audios = torch.zeros(len(batch), max(a.shape[0] for a, _, _, _ in batch))
    for i, (audio, _, _, _) in enumerate(batch):
        audios[i, : audio.shape[0]] = audio

    tokens = torch.full(
        (len(batch), max(t.shape[0] for _, _, t, _ in batch)), pad_id, dtype=torch.long
    )
    for i, (_, _, token, _) in enumerate(batch):
        tokens[i, : token.shape[0]] = token

    return {
        "audios": audios,
        "audio_lens": audio_lens,
        "tokens": tokens,
        "token_lens": token_lens,
    }


def parakeet_ctc_loss_fn(model, batch):
    """CTC loss against the batch's tokens: the training loss for the ``"ce"`` objective."""
    device = next(model.parameters()).device
    log_probs, encoded_len, _ = model(
        input_signal=batch["audios"].to(device),
        input_signal_length=batch["audio_lens"].to(device),
    )
    return model.loss(
        log_probs=log_probs,
        targets=batch["tokens"].to(device),
        input_lengths=encoded_len,
        target_lengths=batch["token_lens"].to(device),
    )


def parakeet_ctc_logits_fn(model, batch):
    """Per-frame CTC log-probabilities, and the frames the rotation objective counts.

    The mask keeps each utterance's frames up to its encoded length, so padding frames do not
    enter the KL divergence. Log-probabilities are valid logits for it: log_softmax leaves them
    unchanged.

    Returns:
        ``(log_probs, mask)`` with log_probs ``(batch, frames, vocab)`` and mask
        ``(batch, frames)``.
    """
    device = next(model.parameters()).device
    log_probs, encoded_len, _ = model(
        input_signal=batch["audios"].to(device),
        input_signal_length=batch["audio_lens"].to(device),
    )
    frames = torch.arange(log_probs.shape[1], device=device)
    return log_probs, frames.unsqueeze(0) < encoded_len.unsqueeze(1)


def build_parakeet_dataloader(
    model, samples: Sequence[Tuple[np.ndarray, str]], batch_size: int = 4, seed: int = 42
) -> torch.utils.data.DataLoader:
    pad_id = getattr(model.tokenizer, "pad_id", 0)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ParakeetCalibrationDataset(model, samples),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=partial(parakeet_ctc_collate_fn, pad_id=pad_id if pad_id > 0 else 0),
        num_workers=0,
        pin_memory=True,
    )


def insert_conformer_block_output_linears(model) -> List[str]:
    """Insert an identity Linear after every block's ``norm_out`` but the last block's.

    A Conformer block ends with ``norm_out``, so the layer feeding the next block's first norm
    is itself a norm and cannot absorb a centering matrix. The inserted Linear gives it one.
    The last block is skipped because nothing downstream needs its output centered.

    Must run before get_parakeet_norm_layers, which builds names that depend on it. Idempotent:
    a block whose ``norm_out`` is already wrapped is left alone.

    Args:
        model: The Conformer model to modify in place.

    Returns:
        The dotted names of the inserted Linear layers.
    """
    layers = model.encoder.layers
    inserted: List[str] = []
    for i in range(len(layers) - 1):
        if isinstance(layers[i].norm_out, nn.Sequential):
            inserted.append(f"encoder.layers.{i}.norm_out.1")
            continue
        _, linear_name = insert_linear_after_norm(model, f"encoder.layers.{i}.norm_out")
        inserted.append(linear_name)
    return inserted


def _block_output_names(model, i: int) -> Tuple[str, str]:
    """``(norm_name, layer_feeding_the_next_block)`` for block i's output norm."""
    wrapped = isinstance(model.encoder.layers[i].norm_out, nn.Sequential)
    base = f"encoder.layers.{i}.norm_out"
    return (f"{base}.0", f"{base}.1") if wrapped else (base, base)


def get_parakeet_norm_layers(model) -> List[Tuple[str, List[str], List[str]]]:
    """Return ``(norm_name, previous_names, next_names)`` for every LayerNorm in the encoder.

    Call insert_conformer_block_output_linears first: without it the previous layer of each
    block's first norm would be the preceding block's ``norm_out``, which is a norm and cannot
    absorb the centering.

    The first block's first norm has an empty previous list. Its input is the subsampling
    frontend's output rather than a residual sum, and it is centered by patching the encoder's
    forward pass instead.

    The last block's ``norm_out`` has an empty next list, so it keeps its scale and shift.
    """
    layers = model.encoder.layers
    norms: List[Tuple[str, List[str], List[str]]] = []

    for i in range(len(layers)):
        p = f"encoder.layers.{i}"
        norm_out_name, block_output = _block_output_names(model, i)

        norms.append(
            (
                f"{p}.norm_feed_forward1",
                [] if i == 0 else [_block_output_names(model, i - 1)[1]],
                [f"{p}.feed_forward1.linear1"],
            )
        )
        norms.append(
            (
                f"{p}.norm_self_att",
                [f"{p}.feed_forward1.linear2"],
                [f"{p}.self_attn.linear_{proj}" for proj in ("q", "k", "v")],
            )
        )
        # the convolution module's first pointwise conv is a Linear in all but layout
        norms.append(
            (
                f"{p}.norm_conv",
                [f"{p}.self_attn.linear_out"],
                [f"{p}.conv.pointwise_conv1"],
            )
        )
        norms.append(
            (
                f"{p}.norm_feed_forward2",
                [f"{p}.conv.pointwise_conv2"],
                [f"{p}.feed_forward2.linear1"],
            )
        )
        # norm_out feeds the inserted Linear, or nothing at all in the last block
        norms.append(
            (
                norm_out_name,
                [f"{p}.feed_forward2.linear2"],
                [] if block_output == norm_out_name else [block_output],
            )
        )

    return norms


def get_parakeet_layers_to_rotate(model) -> List[Tuple[str, int, List[Tuple[str, int]]]]:
    """Return ``(attention_name, head_dim, [(layer_name, rot_type), ...])`` for the encoder.

    One R2 is learned per attention module. A Conformer block holds one, so each block
    contributes a single entry carrying its feed-forward, convolution and block-output layers
    alongside the attention projections; those use only R1 and ignore R2.

    The convolution module's depthwise conv is absent deliberately: it mixes over time with a
    wide kernel and is not a channel-wise linear map, so no rotation folds into it.
    """
    groups: List[Tuple[str, int, List[Tuple[str, int]]]] = []

    for i, layer in enumerate(model.encoder.layers):
        p = f"encoder.layers.{i}"
        entries = [
            (f"{p}.self_attn.linear_q", ROT_INPUT_R1),
            (f"{p}.self_attn.linear_k", ROT_INPUT_R1),
            (f"{p}.self_attn.linear_v", ROT_INPUT_R1_OUTPUT_R2),
            (f"{p}.self_attn.linear_out", ROT_INPUT_R2_OUTPUT_R1),
            (f"{p}.feed_forward1.linear1", ROT_INPUT_R1),
            (f"{p}.feed_forward1.linear2", ROT_OUTPUT_R1),
            (f"{p}.conv.pointwise_conv1", ROT_INPUT_R1),
            (f"{p}.conv.pointwise_conv2", ROT_OUTPUT_R1),
            (f"{p}.feed_forward2.linear1", ROT_INPUT_R1),
            (f"{p}.feed_forward2.linear2", ROT_OUTPUT_R1),
        ]
        norm_out_name, block_output = _block_output_names(model, i)
        if block_output != norm_out_name:
            # the inserted Linear reads and writes the rotated residual stream
            entries.append((block_output, ROT_INPUT_R1_OUTPUT_R1))

        groups.append((f"{p}.self_attn", layer.self_attn.d_k, entries))

    return groups


def get_parakeet_activation_roles(model, quantize_block_output_linear: bool = False) -> Dict[str, str]:
    """Return ``{layer_name: role}`` for every layer whose input is activation-quantized.

    The single definition used by both the rotation search and evaluation. The convolution
    module's pointwise convs take the feed-forward roles: pointwise_conv1 reads the residual
    stream like fc1, and pointwise_conv2 reads a nonlinearity's output like fc2, behind its own
    online Hadamard.

    The identity Linear inserted after each block's output norm (see
    insert_conformer_block_output_linears) is left out unless quantize_block_output_linear is set;
    it then gets the ``block_out`` role. It exists only once a rotation has inserted it, so a model
    without one has no such layers either way. It reads the rotated residual stream and writes it
    back, a dense d_model x d_model matmul the original model did not have (about 2% of W4A4 GPU
    time on Parakeet-CTC-1.1B).

    Args:
        model: A model whose ``.encoder`` is a ConformerEncoder.
        quantize_block_output_linear: Give the inserted block-output Linears the ``block_out`` role.
    """
    roles: Dict[str, str] = {}
    for i in range(len(model.encoder.layers)):
        p = f"encoder.layers.{i}"
        norm_out_name, block_output = _block_output_names(model, i)
        if quantize_block_output_linear and block_output != norm_out_name:
            roles[block_output] = "block_out"
        roles.update({
            f"{p}.self_attn.linear_q": "q",
            f"{p}.self_attn.linear_k": "k",
            f"{p}.self_attn.linear_v": "v",
            f"{p}.self_attn.linear_out": "attn_out",
            f"{p}.feed_forward1.linear1": "fc1",
            f"{p}.feed_forward1.linear2": "fc2",
            f"{p}.conv.pointwise_conv1": "fc1",
            f"{p}.conv.pointwise_conv2": "fc2",
            f"{p}.feed_forward2.linear1": "fc1",
            f"{p}.feed_forward2.linear2": "fc2",
        })
    return roles


def get_parakeet_online_hadamard_layers(model) -> List[Tuple[str, str]]:
    """Return ``(activation_name, fc2_name)`` for the three FC2-like layers of every block.

    Both feed-forwards read ``activation(linear1(x))`` through dropout, the identity in eval
    mode, and the convolution module's ``pointwise_conv2`` reads ``activation(batch_norm(x))``
    directly, in the (batch, channels, time) layout.

    NeMo gives every feed-forward in the encoder the same Swish instance, created once as a
    default argument, so these names resolve to one module. insert_online_hadamard_after_activation
    wraps each parent's reference separately, which is what keeps that sharing harmless.
    """
    pairs: List[Tuple[str, str]] = []
    for i in range(len(model.encoder.layers)):
        p = f"encoder.layers.{i}"
        pairs += [
            (f"{p}.feed_forward1.activation", f"{p}.feed_forward1.linear2"),
            (f"{p}.conv.activation", f"{p}.conv.pointwise_conv2"),
            (f"{p}.feed_forward2.activation", f"{p}.feed_forward2.linear2"),
        ]
    return pairs


def attach_parakeet_rotation_hooks(model, R1, persistent: bool = False) -> List:
    """Center and rotate the encoder's residual stream, and leave the rotated basis before the
    last block's output norm.

    The first block's first norm reads the subsampling frontend's output rather than a residual
    sum, so there is no layer to fold its centering into. NeMo passes the stream to each layer as
    the keyword ``x``, hence the keyword on the entry hook.

    The last block's ``norm_out`` has no next layer, so it keeps its scale and shift. RMSNorm is
    only equivariant to a rotation without those, so the stream is rotated back to the original
    basis on the way *into* that norm rather than after it. The norm's output is then already
    unrotated, which is what the CTC head expects.

    Args:
        model: A model whose ``.encoder`` is a ConformerEncoder.
        R1: The residual-stream rotation.
        persistent: Attach the transforms as ResidualStreamRotation modules that are saved with
            the model, for a final rotation; the search leaves this False.

    Returns:
        Hook handles, each with a ``.remove()``.
    """
    layers = model.encoder.layers
    return [
        add_residual_stream_entry_hook(layers[0], R1, keyword="x", persistent=persistent),
        add_unrotate_input_hook(layers[-1].norm_out, R1, persistent=persistent),
    ]


def get_parakeet_rotation_layers(model) -> Dict:
    """The structural half of the rotation inputs: which layers to rotate and where the norms go.

    Inserts the block-output Linears as a side effect, since the mappings depend on them.

    Separated from get_parakeet_rotation_inputs because applying a saved rotation needs only
    this, and building the calibration dataloader would download a dataset for nothing.
    """
    insert_conformer_block_output_linears(model)
    return {
        "layers_to_rotate": get_parakeet_layers_to_rotate(model),
        "norm_layers": get_parakeet_norm_layers(model),
        "online_hadamard_layers": get_parakeet_online_hadamard_layers(model),
        "attach_rotation_hooks": attach_parakeet_rotation_hooks,
    }


def get_parakeet_rotation_inputs(
    model, samples: Sequence[Tuple[np.ndarray, str]], quantize_block_output_linear: bool = False,
    **dataloader_kwargs,
) -> Dict:
    """Everything learn_rotations needs for a Parakeet CTC model, as keyword arguments.

    samples are the ``(waveform, transcript)`` calibration pairs, the same ones GPTQ uses.
    quantize_block_output_linear also quantizes the inputs of the inserted block-output Linears
    during the search, as evaluation will; see get_parakeet_activation_roles.
    """
    layers = get_parakeet_rotation_layers(model)
    return {
        **layers,
        "hidden_size": model.encoder.layers[0].self_attn.linear_q.in_features,
        "activation_roles": get_parakeet_activation_roles(model, quantize_block_output_linear),
        "compute_logits": parakeet_ctc_logits_fn,
        "compute_loss": parakeet_ctc_loss_fn,
        "train_loader": build_parakeet_dataloader(model, samples, **dataloader_kwargs),
    }
