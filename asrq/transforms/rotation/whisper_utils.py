"""Whisper-specific inputs for learn_rotations.

Provides the four things the generic rotation search needs for a Whisper model: a calibration
dataloader, a logits function, the norm-to-neighbour mapping, and the list of layers to rotate.

The mappings are derived from WhisperEncoder/WhisperDecoder's forward pass. Both stacks carry a
d_model-wide residual stream, and a single R1 rotates both: the encoder's output feeds the
decoder's cross-attention K/V, so those read the same rotated basis.
"""

from functools import partial
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from asrq.transforms.rotation.utils import (
    add_residual_stream_entry_hook,
    add_unrotate_input_hook,
)

# Rotation types, as learn_rotations dispatches them.
ROT_INPUT_R1 = 1  # Linear(X R1.T)        -- q, k, fc1
ROT_INPUT_R1_OUTPUT_R2 = 2  # Linear(X R1.T) R2     -- v
ROT_INPUT_R2_OUTPUT_R1 = 3  # Linear(X R2.T) R1     -- out_proj
ROT_OUTPUT_R1 = 4  # Linear(X) R1          -- fc2


def build_whisper_decoder_targets(
    tokenizer, text: str, language: str = "en", task: str = "transcribe"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forcing ``(decoder_input_ids, labels)`` for one transcript, aligned with generate().

    generate() starts the decoder from the forced prefix
    ``<|startoftranscript|> <|lang|> <|task|> <|notimestamps|>`` and predicts the text from
    there. Teacher forcing has to present the decoder the same sequence, one token ahead:

        decoder_input_ids: <|sot|>  <|en|>  <|transcribe|>  <|notimestamps|>  t1  ...  tn
        labels:            -100     -100    -100            t1                t2  ...  <|eot|>

    The prefix tokens are masked out of the loss because inference forces them: the model is
    never asked to predict them, so their loss measures nothing that is deployed and only
    dilutes the average. The end-of-text token stays a target, since predicting when to stop is
    part of transcription.

    Both tensors are returned rather than labels alone because HF's own shift, applied when
    only labels are passed, replaces every -100 with the pad token in the decoder input. The
    masked prefix would then reach the decoder as ``<|endoftext|>`` instead of the prefix.
    Leaving ``<|startoftranscript|>`` in the labels is the other way this goes wrong: the shift
    prepends it again, so every position is conditioned on a doubled start token.

    Special tokens are looked up by name, so this holds for every Whisper size; their ids differ
    between large-v3 and the earlier checkpoints.

    Args:
        tokenizer: A WhisperTokenizer.
        text: The transcript.
        language: Language code of the transcript.
        task: ``"transcribe"`` or ``"translate"``.

    Returns:
        ``(decoder_input_ids, labels)``, 1-D and of equal length.
    """
    prefix = tokenizer.convert_tokens_to_ids(
        ["<|startoftranscript|>", f"<|{language}|>", f"<|{task}|>", "<|notimestamps|>"]
    )
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    sequence = prefix + text_ids + [tokenizer.eos_token_id]
    decoder_input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
    labels = torch.tensor(sequence[1:], dtype=torch.long)
    labels[: len(prefix) - 1] = -100
    return decoder_input_ids, labels


class WhisperCalibrationDataset(torch.utils.data.Dataset):
    """Calibration utterances as teacher-forcing batches for the rotation search.

    Takes the ``(waveform, text)`` pairs asrq.calibration.data loads, whose text is the
    full-precision model's own transcript. The text is used verbatim: normalizing it would put
    the decoder in contexts the model does not produce, which is the reason the transcripts are
    used instead of the reference in the first place.
    """

    def __init__(self, processor, samples: Sequence[Tuple[np.ndarray, str]]):
        super().__init__()
        self.processor = processor
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio, text = self.samples[idx]
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.squeeze(0)
        decoder_input_ids, labels = build_whisper_decoder_targets(self.processor.tokenizer, text)
        return input_features, decoder_input_ids, labels


def whisper_collate_fn(batch, pad_token_id: int):
    """Stack features; right-pad decoder inputs with pad_token_id and labels with -100.

    The decoder is causal, so padding after a sequence cannot change any position before it,
    and the -100 labels keep the padded positions out of the loss.
    """
    input_features = torch.stack([features for features, _, _ in batch])
    max_len = max(labels.size(0) for _, _, labels in batch)
    decoder_input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (_, inputs, targets) in enumerate(batch):
        decoder_input_ids[i, : inputs.size(0)] = inputs
        labels[i, : targets.size(0)] = targets
    return {
        "input_features": input_features,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
    }


def whisper_logits_fn(model, batch):
    """Teacher-forced logits, and the positions the rotation objective counts.

    The mask is the labels' non-ignored positions: the text and end-of-text targets, without
    the forced prefix or padding. decoder_input_ids are passed explicitly so HF does not derive
    them by shifting the labels; see build_whisper_decoder_targets.

    The features are cast to the model's dtype, which the encoder's convolutions require.

    Returns:
        ``(logits, mask)`` with logits ``(batch, length, vocab)`` and mask ``(batch, length)``.
    """
    parameter = next(model.parameters())
    labels = batch["labels"].to(parameter.device)
    logits = model(
        input_features=batch["input_features"].to(device=parameter.device, dtype=parameter.dtype),
        decoder_input_ids=batch["decoder_input_ids"].to(parameter.device),
        return_dict=True,
    ).logits
    return logits, labels != -100


def whisper_loss_fn(model, batch):
    """Teacher-forced cross-entropy over the text and end-of-text targets.

    The training loss for learn_rotations' ``"ce"`` objective. The positions are the ones
    whisper_logits_fn counts, so both objectives see the same tokens.
    """
    logits, mask = whisper_logits_fn(model, batch)
    return F.cross_entropy(logits[mask].float(), batch["labels"].to(logits.device)[mask])


def build_whisper_dataloader(
    processor, samples: Sequence[Tuple[np.ndarray, str]], batch_size: int = 4, seed: int = 42
) -> torch.utils.data.DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        WhisperCalibrationDataset(processor, samples),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=partial(whisper_collate_fn, pad_token_id=processor.tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=True,
    )


def get_whisper_norm_layers(model) -> List[Tuple[str, List[str], List[str]]]:
    """Return ``(norm_name, previous_names, next_names)`` for every LayerNorm in Whisper.

    An empty previous list means the centering cannot be folded backwards. That happens at the
    first norm of each stack, whose input is ``gelu(conv2(...)) + embed_positions`` in the
    encoder and ``embed_tokens + positions`` in the decoder -- a nonlinearity in one case and
    embeddings in the other, neither of which this fold can absorb. Those two inputs are
    centered by patching the encoder's and decoder's forward passes instead, so the residual
    stream is already mean-subtracted by the time it reaches the first norm.

    An empty next list means the shift and scale stay in the norm. That happens at
    ``model.decoder.layer_norm``, whose output goes to ``proj_out``. Folding into proj_out would
    be possible, but it is weight-tied to ``model.decoder.embed_tokens`` (Whisper sets
    ``tie_word_embeddings=True``), so writing to it would silently corrupt the input embeddings.
    """
    config = model.config
    n_enc, n_dec = config.encoder_layers, config.decoder_layers
    norms: List[Tuple[str, List[str], List[str]]] = []

    for i in range(n_enc):
        p = f"model.encoder.layers.{i}"
        norms.append(
            (
                f"{p}.self_attn_layer_norm",
                [] if i == 0 else [f"model.encoder.layers.{i - 1}.fc2"],
                [f"{p}.self_attn.{proj}_proj" for proj in ("q", "k", "v")],
            )
        )
        norms.append((f"{p}.final_layer_norm", [f"{p}.self_attn.out_proj"], [f"{p}.fc1"]))

    # the encoder's output feeds every decoder layer's cross-attention K and V
    norms.append(
        (
            "model.encoder.layer_norm",
            [f"model.encoder.layers.{n_enc - 1}.fc2"],
            [
                f"model.decoder.layers.{i}.encoder_attn.{proj}_proj"
                for proj in ("k", "v")
                for i in range(n_dec)
            ],
        )
    )

    for i in range(n_dec):
        p = f"model.decoder.layers.{i}"
        norms.append(
            (
                f"{p}.self_attn_layer_norm",
                [] if i == 0 else [f"model.decoder.layers.{i - 1}.fc2"],
                [f"{p}.self_attn.{proj}_proj" for proj in ("q", "k", "v")],
            )
        )
        # cross-attention K/V come from the encoder, so only Q reads this norm
        norms.append(
            (
                f"{p}.encoder_attn_layer_norm",
                [f"{p}.self_attn.out_proj"],
                [f"{p}.encoder_attn.q_proj"],
            )
        )
        norms.append((f"{p}.final_layer_norm", [f"{p}.encoder_attn.out_proj"], [f"{p}.fc1"]))

    norms.append(("model.decoder.layer_norm", [f"model.decoder.layers.{n_dec - 1}.fc2"], []))

    return norms


def _attention_group(prefix: str, attn: str) -> List[Tuple[str, int]]:
    """The four projections of one attention module, with their rotation types."""
    p = f"{prefix}.{attn}"
    return [
        (f"{p}.q_proj", ROT_INPUT_R1),
        (f"{p}.k_proj", ROT_INPUT_R1),
        (f"{p}.v_proj", ROT_INPUT_R1_OUTPUT_R2),
        (f"{p}.out_proj", ROT_INPUT_R2_OUTPUT_R1),
    ]


def get_whisper_layers_to_rotate(model) -> List[Tuple[str, int, List[Tuple[str, int]]]]:
    """Return ``(attention_name, head_dim, [(layer_name, rot_type), ...])`` for Whisper.

    The key is an attention module because that is what owns an R2: it is a head-dimension
    rotation applied to that module's V projection and undone by its output projection. A
    decoder layer holds two attention modules, whose head rotations are independent, so it
    contributes two entries.

    The feed-forward layers use rotation types 1 and 4, which involve only R1, so they are
    listed under the attention module they share a transformer layer with.
    """
    config = model.config
    enc_head_dim = config.d_model // config.encoder_attention_heads
    dec_head_dim = config.d_model // config.decoder_attention_heads
    groups: List[Tuple[str, int, List[Tuple[str, int]]]] = []

    for i in range(config.encoder_layers):
        p = f"model.encoder.layers.{i}"
        groups.append(
            (
                f"{p}.self_attn",
                enc_head_dim,
                _attention_group(p, "self_attn")
                + [(f"{p}.fc1", ROT_INPUT_R1), (f"{p}.fc2", ROT_OUTPUT_R1)],
            )
        )

    for i in range(config.decoder_layers):
        p = f"model.decoder.layers.{i}"
        groups.append((f"{p}.self_attn", dec_head_dim, _attention_group(p, "self_attn")))
        groups.append(
            (
                f"{p}.encoder_attn",
                dec_head_dim,
                _attention_group(p, "encoder_attn")
                + [(f"{p}.fc1", ROT_INPUT_R1), (f"{p}.fc2", ROT_OUTPUT_R1)],
            )
        )

    return groups


def get_whisper_activation_roles(model) -> Dict[str, str]:
    """Return ``{layer_name: role}`` for every layer whose input is activation-quantized.

    The single definition used by both the rotation search and evaluation, so a rotation is
    searched against the same quantized layers it is evaluated with. Every attention projection
    in both stacks, the decoder's cross-attention included, and both feed-forward layers; fc2's
    input is the online Hadamard's output once a rotation is applied. Roles are described in
    asrq.quantizers.activation.
    """
    config = model.config
    roles: Dict[str, str] = {}
    attentions = (
        [(f"model.encoder.layers.{i}", ("self_attn",)) for i in range(config.encoder_layers)]
        + [(f"model.decoder.layers.{i}", ("self_attn", "encoder_attn"))
           for i in range(config.decoder_layers)]
    )
    for prefix, modules in attentions:
        for attention in modules:
            for projection, role in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"),
                                     ("out_proj", "attn_out")):
                roles[f"{prefix}.{attention}.{projection}"] = role
        roles[f"{prefix}.fc1"] = "fc1"
        roles[f"{prefix}.fc2"] = "fc2"
    return roles


def get_whisper_online_hadamard_layers(model) -> List[Tuple[str, str]]:
    """Return ``(activation_name, fc2_name)`` for every feed-forward in both stacks.

    Each FC2 reads ``activation_fn(fc1(x))`` through activation dropout, which is the identity in
    eval mode, so the online Hadamard goes after ``activation_fn``. Whisper builds a separate
    activation instance per layer, so wrapping one does not touch any other.
    """
    config = model.config
    return [
        (f"model.{stack}.layers.{i}.activation_fn", f"model.{stack}.layers.{i}.fc2")
        for stack, n_layers in (
            ("encoder", config.encoder_layers),
            ("decoder", config.decoder_layers),
        )
        for i in range(n_layers)
    ]


def attach_whisper_rotation_hooks(model, R1, persistent: bool = False) -> List:
    """Center and rotate each stack's residual stream, and leave the rotated basis before the
    decoder's final norm.

    Both stacks need centering on entry: their first norm reads the stream, which is the sum of
    the frontend's output and the positional embedding, so there is no single layer to fold the
    centering into.

    ``model.decoder.layer_norm`` has no next layer -- folding into ``proj_out`` would corrupt
    ``embed_tokens``, which it is weight-tied to -- so it keeps its scale and shift and the
    stream is rotated back on the way into it. Its output then reaches ``proj_out`` in the
    original basis.

    The encoder's final norm needs nothing: its scale folds into the decoder's cross-attention
    K/V projections, which are themselves rotated, so the encoder output stays rotated.

    Args:
        model: A WhisperForConditionalGeneration.
        R1: The residual-stream rotation.
        persistent: Attach the transforms as ResidualStreamRotation modules that are saved with
            the model, for a final rotation; the search leaves this False.

    Returns:
        Hook handles, each with a ``.remove()``.
    """
    modules = dict(model.named_modules())
    encoder_entry = modules["model.encoder.layers.0"]
    decoder_entry = modules["model.decoder.layers.0"]
    return [
        add_residual_stream_entry_hook(encoder_entry, R1, persistent=persistent),
        add_residual_stream_entry_hook(decoder_entry, R1, persistent=persistent),
        add_unrotate_input_hook(modules["model.decoder.layer_norm"], R1, persistent=persistent),
    ]


def get_whisper_rotation_layers(model) -> Dict:
    """The structural half of the rotation inputs: which layers to rotate and where the norms go.

    Separated from get_whisper_rotation_inputs because applying a saved rotation needs only
    this, and building the calibration dataloader would download a dataset for nothing.
    """
    return {
        "layers_to_rotate": get_whisper_layers_to_rotate(model),
        "norm_layers": get_whisper_norm_layers(model),
        "online_hadamard_layers": get_whisper_online_hadamard_layers(model),
        "attach_rotation_hooks": attach_whisper_rotation_hooks,
    }


def get_whisper_rotation_inputs(
    model, processor, samples: Sequence[Tuple[np.ndarray, str]], **dataloader_kwargs
) -> Dict:
    """Everything learn_rotations needs for a Whisper model, as keyword arguments.

    samples are the ``(waveform, transcript)`` calibration pairs, the same ones GPTQ uses.
    """
    return {
        **get_whisper_rotation_layers(model),
        "hidden_size": model.config.d_model,
        "activation_roles": get_whisper_activation_roles(model),
        "compute_logits": whisper_logits_fn,
        "compute_loss": whisper_loss_fn,
        "train_loader": build_whisper_dataloader(processor, samples, **dataloader_kwargs),
    }
