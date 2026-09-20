"""Canary-Qwen-specific inputs for learn_rotations.

Canary-Qwen-2.5B (NeMo SALM) is a FastConformer speech encoder feeding a Qwen3-1.7B LLM through a linear
projection. The two are separate residual streams of different widths, so each gets its own R1:

* ``encoder`` (1024): the Conformer blocks, mapped exactly as for Parakeet (see parakeet_ctc_utils)
  under the ``perception.`` prefix. Its LayerNorms are converted and centered on entry to the first
  block, and the stream is rotated back before the last block's output norm, so the projection into
  the LLM reads the unrotated encoder output.
* ``llm`` (2048): the Qwen3 decoder layers. Their RMSNorms only need their scales folded; the stream is
  rotated, without centering, on entry to the first layer and rotated back before the final norm.
  Rotating on entry rather than folding R1 into the token embedding matters because the embedding is
  tied to the output head, which reads the stream after the final norm. The audio embeddings the
  projection writes and the text embeddings both enter through that one hook.

Each Qwen layer rotates q/k (type 1), v (type 2, head-wise R2), o (type 3), gate/up (type 1) and down
(type 4). Down's input is ``act(gate(x)) * up(x)``, a product rather than one activation's output, so
its online Hadamard sits at down's own input (an activation name of None in the online Hadamard
pairs). Grouped-query attention needs nothing special: R2 acts per head on V's key-value heads, the
same matrix the output projection undoes on every query head that reads them.

The LoRA adapters on q_proj and v_proj are merged into the base weights first (merge_lora); every
mapping below refers to the merged Qwen3ForCausalLM.
"""

from functools import partial
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from asrq.calibration.data import length_sorted_batches
from nemo.collections.common.prompts.formatter import PromptFormatter
from peft import PeftModel

from asrq.transforms.rotation.parakeet_ctc_utils import (
    ROT_INPUT_R1,
    ROT_INPUT_R1_OUTPUT_R2,
    ROT_INPUT_R2_OUTPUT_R1,
    ROT_OUTPUT_R1,
    attach_parakeet_rotation_hooks,
    get_parakeet_activation_roles,
    get_parakeet_layers_to_rotate,
    get_parakeet_norm_layers,
    get_parakeet_online_hadamard_layers,
    insert_conformer_block_output_linears,
)
from asrq.transforms.rotation.utils import add_residual_stream_entry_hook, add_unrotate_input_hook

ENCODER = "encoder"
LLM = "llm"
PERCEPTION = "perception"
TRANSCRIBE_PROMPT = "Transcribe the following: {audio_locator_tag}"


def merge_lora(model) -> None:
    """Merge the LLM's LoRA adapters into its base weights, in place. A no-op once merged.

    Merged before any norm or rotation fold, which would otherwise write into the base weights while
    leaving the adapters' delta unrotated.
    """
    if isinstance(model.llm, PeftModel):
        model.llm = model.llm.merge_and_unload()


def _prefixed(prefix: str, name: str) -> str:
    return f"{prefix}.{name}"


def _llm_layer(i: int) -> str:
    return f"{LLM}.model.layers.{i}"


def get_canary_qwen_norm_layers(model) -> List[Tuple[str, List[str], List[str]]]:
    """``(norm_name, previous_names, next_names)`` for the encoder's LayerNorms and the LLM's RMSNorms.

    The LLM's final norm is absent: it keeps its scale, since the output head tied to the embedding
    cannot absorb it, and the stream is rotated back before it instead.
    """
    norms = [
        (_prefixed(PERCEPTION, norm), [_prefixed(PERCEPTION, n) for n in previous], [_prefixed(PERCEPTION, n) for n in nxt])
        for norm, previous, nxt in get_parakeet_norm_layers(model.perception)
    ]
    for i in range(len(model.llm.model.layers)):
        p = _llm_layer(i)
        norms.append((f"{p}.input_layernorm", [], [f"{p}.self_attn.{proj}_proj" for proj in ("q", "k", "v")]))
        norms.append((f"{p}.post_attention_layernorm", [], [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"]))
    return norms


def get_canary_qwen_layers_to_rotate(model) -> List[Tuple[str, int, List[Tuple[str, int]], str]]:
    """``(attention_name, head_dim, [(layer_name, rot_type), ...], stream)`` for both streams."""
    groups = [
        (_prefixed(PERCEPTION, attention), head_dim, [(_prefixed(PERCEPTION, n), t) for n, t in layers], ENCODER)
        for attention, head_dim, layers in get_parakeet_layers_to_rotate(model.perception)
    ]
    head_dim = model.llm.config.head_dim
    for i in range(len(model.llm.model.layers)):
        p = _llm_layer(i)
        groups.append((f"{p}.self_attn", head_dim, [
            (f"{p}.self_attn.q_proj", ROT_INPUT_R1),
            (f"{p}.self_attn.k_proj", ROT_INPUT_R1),
            (f"{p}.self_attn.v_proj", ROT_INPUT_R1_OUTPUT_R2),
            (f"{p}.self_attn.o_proj", ROT_INPUT_R2_OUTPUT_R1),
            (f"{p}.mlp.gate_proj", ROT_INPUT_R1),
            (f"{p}.mlp.up_proj", ROT_INPUT_R1),
            (f"{p}.mlp.down_proj", ROT_OUTPUT_R1),
        ], LLM))
    return groups


def get_canary_qwen_activation_roles(model, quantize_block_output_linear: bool = False) -> Dict[str, str]:
    """``{layer_name: role}`` for every layer whose input is activation-quantized, in both streams.

    The encoder's roles are Parakeet's. In the LLM, q/k/v take their own roles, o is attn_out, gate and
    up read the rotated residual stream like fc1, and down reads its online Hadamard's output like fc2.
    """
    roles = {
        _prefixed(PERCEPTION, name): role
        for name, role in get_parakeet_activation_roles(model.perception, quantize_block_output_linear).items()
    }
    for i in range(len(model.llm.model.layers)):
        p = _llm_layer(i)
        roles.update({
            f"{p}.self_attn.q_proj": "q",
            f"{p}.self_attn.k_proj": "k",
            f"{p}.self_attn.v_proj": "v",
            f"{p}.self_attn.o_proj": "attn_out",
            f"{p}.mlp.gate_proj": "fc1",
            f"{p}.mlp.up_proj": "fc1",
            f"{p}.mlp.down_proj": "fc2",
        })
    return roles


def get_canary_qwen_online_hadamard_layers(model) -> List[Tuple[str, str]]:
    """``(activation_name, fc2_name)`` pairs: the encoder's as for Parakeet, and ``(None, down_proj)``
    for every LLM layer, whose online Hadamard sits at down_proj's own input."""
    pairs = [
        (_prefixed(PERCEPTION, activation), _prefixed(PERCEPTION, fc2))
        for activation, fc2 in get_parakeet_online_hadamard_layers(model.perception)
    ]
    pairs += [(None, f"{_llm_layer(i)}.mlp.down_proj") for i in range(len(model.llm.model.layers))]
    return pairs


def attach_canary_qwen_rotation_hooks(model, R1, persistent: bool = False) -> List:
    """Rotate both residual streams on entry and back before the norms that keep their scales.

    Args:
        model: A merged Canary-Qwen SALM model.
        R1: ``{"encoder": R1_encoder, "llm": R1_llm}``.
        persistent: Attach the transforms as saved ResidualStreamRotation modules, for a final rotation.

    Returns:
        Hook handles, each with a ``.remove()``.
    """
    layers = model.llm.model.layers
    return attach_parakeet_rotation_hooks(model.perception, R1[ENCODER], persistent=persistent) + [
        add_residual_stream_entry_hook(layers[0], R1[LLM], persistent=persistent, center=False),
        add_unrotate_input_hook(model.llm.model.norm, R1[LLM], persistent=persistent),
    ]


def get_canary_qwen_rotation_layers(model) -> Dict:
    """The structural half of the rotation inputs. Merges the LoRA adapters and inserts the encoder's
    block-output Linears as side effects, since the mappings depend on both."""
    merge_lora(model)
    insert_conformer_block_output_linears(model.perception)
    return {
        "layers_to_rotate": get_canary_qwen_layers_to_rotate(model),
        "norm_layers": get_canary_qwen_norm_layers(model),
        "online_hadamard_layers": get_canary_qwen_online_hadamard_layers(model),
        "attach_rotation_hooks": attach_canary_qwen_rotation_hooks,
    }


def encode_transcription_turns(model, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Token ids of the transcription prompt followed by ``text`` as the assistant's answer, and the
    mask of the positions whose tokens the model predicts from the audio.

    The prompt is the one evaluation generates from. The mask covers the answer's text and its
    ``<|im_end|>``, but not the ``<|im_start|>assistant\\n`` prefix, which generation always forces.
    """
    formatter = PromptFormatter.resolve(model.cfg.prompt_format)(model.tokenizer)
    user = {"role": "user", "slots": {"message": TRANSCRIBE_PROMPT.format(audio_locator_tag=model.audio_locator_tag)}}
    encoded = formatter.encode_dialog(turns=[user, {"role": "assistant", "slots": {"message": text}}])
    input_ids = encoded["input_ids"]
    answer_start = len(formatter.encode_dialog(turns=[user])["input_ids"])
    mask = torch.zeros_like(encoded["mask"])
    end = int((input_ids == model.text_eos_id).nonzero().max()) + 1
    mask[answer_start:end] = True
    return input_ids, mask


class CanaryQwenCalibrationDataset(torch.utils.data.Dataset):
    """Calibration ``(waveform, transcript)`` pairs as teacher-forced SALM inputs."""

    def __init__(self, model, samples: Sequence[Tuple[np.ndarray, str]]):
        super().__init__()
        self.model = model
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        waveform, text = self.samples[idx]
        input_ids, loss_mask = encode_transcription_turns(self.model, text)
        return torch.as_tensor(np.asarray(waveform, dtype=np.float32)), input_ids, loss_mask


def canary_qwen_collate_fn(batch, pad_id: int):
    """Right-pad audio with zeros and tokens with pad_id; padded tokens are outside the loss mask."""
    audios = torch.zeros(len(batch), max(a.shape[0] for a, _, _ in batch))
    tokens = torch.full((len(batch), max(t.shape[0] for _, t, _ in batch)), pad_id, dtype=torch.long)
    loss_mask = torch.zeros(tokens.shape, dtype=torch.bool)
    for i, (audio, ids, mask) in enumerate(batch):
        audios[i, : audio.shape[0]] = audio
        tokens[i, : ids.shape[0]] = ids
        loss_mask[i, : mask.shape[0]] = mask
    return {
        "audios": audios,
        "audio_lens": torch.tensor([a.shape[0] for a, _, _ in batch], dtype=torch.long),
        "input_ids": tokens,
        "loss_mask": loss_mask,
    }


def _teacher_forced(model, batch):
    device = next(model.parameters()).device
    inputs = model.prepare_inputs({key: value.to(device) for key, value in batch.items()})
    logits = model(inputs["input_embeds"], attention_mask=inputs["attention_mask"])["logits"]
    return logits, inputs["target_ids"]


def canary_qwen_logits_fn(model, batch):
    """LLM logits over the teacher-forced sequence, and the positions predicting the transcript.

    Returns:
        ``(logits, mask)`` with logits ``(batch, tokens, vocab)`` and mask ``(batch, tokens)``.
    """
    logits, targets = _teacher_forced(model, batch)
    return logits, targets != -100


def canary_qwen_loss_fn(model, batch):
    """Cross-entropy on the transcript tokens: the training loss for the ``"ce"`` objective."""
    logits, targets = _teacher_forced(model, batch)
    return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=-100)


def build_canary_qwen_dataloader(
    model, samples: Sequence[Tuple[np.ndarray, str]], batch_size: int = 4, seed: int = 42, sort_by_length: bool = False,
) -> torch.utils.data.DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        CanaryQwenCalibrationDataset(model, samples),
        **(
            {"batch_sampler": length_sorted_batches(samples, batch_size, seed)} if sort_by_length
            else {"batch_size": batch_size, "shuffle": True, "generator": generator}
        ),
        collate_fn=partial(canary_qwen_collate_fn, pad_id=model.text_pad_id),
        num_workers=0,
    )


def get_canary_qwen_rotation_inputs(
    model, samples: Sequence[Tuple[np.ndarray, str]], quantize_block_output_linear: bool = False,
    **dataloader_kwargs,
) -> Dict:
    """Everything learn_rotations needs for Canary-Qwen, as keyword arguments.

    samples are the ``(waveform, transcript)`` calibration pairs, with the model's own transcripts.
    """
    layers = get_canary_qwen_rotation_layers(model)
    return {
        **layers,
        "hidden_size": {
            ENCODER: model.perception.encoder.layers[0].self_attn.linear_q.in_features,
            LLM: model.llm.config.hidden_size,
        },
        "activation_roles": get_canary_qwen_activation_roles(model, quantize_block_output_linear),
        "compute_logits": canary_qwen_logits_fn,
        "compute_loss": canary_qwen_loss_fn,
        "train_loader": build_canary_qwen_dataloader(model, samples, **dataloader_kwargs),
    }
