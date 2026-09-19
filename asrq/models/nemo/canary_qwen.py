"""Canary-Qwen 2.5B quantization support.

Provides :class:`CanaryQwenQ`, a :class:`ModelQ` subclass that implements
block-wise post-training quantization for NVIDIA's Canary-Qwen 2.5B
speech-to-text model.  The conformer-based speech encoder and the Qwen3
text decoder are quantized independently using calibration data drawn
from LibriSpeech.
"""
# pyright: reportMissingImports=false
# pyright: reportPrivateImportUsage=false
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from nemo.collections.speechlm2.models.salm import SALM
from tqdm import tqdm
from transformers import GenerationConfig

from asrq.calibration.base import CalibConfig
from asrq.core.model import ModelQ
from asrq.core.registry import ModelNames, register_model
from asrq.core.utils import cuda_empty_cache
from asrq.models.nemo.parakeet_ctc import conformer_norm_tweak_targets
from asrq.quantizers.base import QuantConfig, is_pointwise_conv1d
from asrq.transforms.rotation.canary_qwen_utils import (
    TRANSCRIBE_PROMPT,
    CanaryQwenCalibrationDataset,
    canary_qwen_collate_fn,
    canary_qwen_logits_fn,
    get_canary_qwen_activation_roles,
    get_canary_qwen_online_hadamard_layers,
    merge_lora,
)


@register_model(ModelNames.NVIDIA_CANARY_QWEN_2_5B)
class CanaryQwenQ(ModelQ):
    """Quantizer for NVIDIA Canary-Qwen 2.5B: a FastConformer speech encoder and a Qwen3-1.7B LLM.

    The LLM's LoRA adapters are merged into its base weights on load, so everything downstream --
    rotation, GPTQ, activation quantization -- sees a plain Qwen3ForCausalLM. Both the encoder's
    Conformer blocks and the LLM's decoder layers are quantized block by block with GPTQ-style
    quantizers; the calibration sequences are the model's own transcripts, teacher-forced after the
    transcription prompt evaluation uses (see asrq.transforms.rotation.canary_qwen_utils).
    """

    @classmethod
    def load_model(cls) -> Tuple[nn.Module, None]:  # type: ignore[override]
        """Load Canary-Qwen 2.5B from NeMo, in eval mode, with the LoRA adapters merged."""
        model = SALM.from_pretrained(ModelNames.NVIDIA_CANARY_QWEN_2_5B).eval()
        merge_lora(model)
        return model, None

    @classmethod
    def from_pretrained(
        cls, quant_cfg: QuantConfig, calib_cfg: CalibConfig
    ) -> "CanaryQwenQ":  # type: ignore[override]
        model, processor = cls.load_model()
        return cls(model, processor, quant_cfg, calib_cfg)

    @staticmethod
    @torch.no_grad()
    def batch_transcribe(
        audio_arrays: List[Any],
        model: nn.Module,
        device: torch.device | str,
        max_new_tokens: int = 128,
    ) -> List[str]:
        """Greedy transcripts of a batch of 16 kHz waveforms, with the prompt evaluation uses."""
        lengths = [len(a) for a in audio_arrays]
        audios = torch.zeros(len(audio_arrays), max(lengths))
        for i, audio in enumerate(audio_arrays):
            audios[i, : len(audio)] = torch.as_tensor(np.asarray(audio, dtype=np.float32))
        prompt = [{"role": "user", "slots": {"message": TRANSCRIBE_PROMPT.format(audio_locator_tag=model.audio_locator_tag)}}]
        answer_ids = model.generate(
            prompts=[prompt] * len(audio_arrays),
            audios=audios.to(device=device, dtype=model.embed_tokens.weight.dtype),
            audio_lens=torch.tensor(lengths, device=device),
            generation_config=GenerationConfig(
                max_new_tokens=max_new_tokens, bos_token_id=model.text_bos_id,
                eos_token_id=model.text_eos_id, pad_token_id=model.text_pad_id,
            ),
        )
        texts = []
        for ids in answer_ids.cpu():
            ids = ids[ids != model.text_pad_id]
            end = (ids == model.text_eos_id).nonzero()
            ids = ids[: int(end[0])] if len(end) else ids
            texts.append(model.tokenizer.ids_to_text(ids).strip())
        return texts

    def _calibration_batch(self, index: int) -> Dict[str, torch.Tensor]:
        return self.calibration_batch(*self.calibration_samples[index])

    def calibration_batch(self, audio: np.ndarray, text: str) -> Dict[str, torch.Tensor]:
        """One teacher-forced calibration batch, on the model's device: the transcription prompt with
        the audio, followed by ``text``."""
        dataset = CanaryQwenCalibrationDataset(self.model, [(audio, text)])
        batch = canary_qwen_collate_fn([dataset[0]], pad_id=self.model.text_pad_id)
        return {key: value.to(self.device) for key, value in batch.items()}

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def should_quantize_module(self, name, module) -> bool:
        """Every nn.Linear and pointwise Conv1d in the quantized blocks; see ParakeetCTCQ for the
        block-output Linears a rotation inserts into the encoder."""
        if name in self.quant_cfg.exclude_modules or name.endswith(".output_linear"):
            return False
        if name.endswith(".norm_out.1") and not self.quant_cfg.quantize_block_output_linear:
            return False
        return isinstance(module, nn.Linear) or is_pointwise_conv1d(module)

    def _capture_block_inputs(self, blocks: nn.ModuleList, run_sample) -> Tuple[List[tuple], List[dict]]:
        """Record the positional and keyword inputs of ``blocks[0]`` for every calibration sample."""
        args_list: List[tuple] = []
        kwargs_list: List[dict] = []

        class Caught(Exception):
            pass

        def record(_module, args, kwargs):
            args_list.append(args)
            kwargs_list.append(kwargs)
            raise Caught

        handle = blocks[0].register_forward_pre_hook(record, with_kwargs=True, prepend=True)
        try:
            for index in range(len(self.calibration_samples)):
                try:
                    run_sample(index)
                except Caught:
                    pass
        finally:
            handle.remove()
        return args_list, kwargs_list

    def _quantize_blocks(self, blocks: nn.ModuleList, prefix: str, args_list, kwargs_list, stream_key) -> None:
        """GPTQ each block on the captured inputs, then feed its outputs to the next block.

        The capturing pre-hook runs before any other pre-hook, so the recorded inputs are what the
        block is called with, and a rotation's entry hook on the first block still applies when the
        block runs here.
        """
        for index, block in enumerate(tqdm(blocks, desc=f"Quantizing {prefix}")):
            quantizers = {}
            hooks = []
            for name, module in block.named_modules():
                full_name = f"{prefix}.{index}.{name}"
                if not name or not self.should_quantize_module(full_name, module):
                    continue
                quantizers[full_name] = self.quant_cls(module, full_name, self.quant_cfg)

                def hook(_module, inputs, output, _name=full_name):
                    quantizers[_name].add_batch((inputs[0], output))
                hooks.append(module.register_forward_hook(hook))
            block_name = f"{prefix}.{index}"
            refit_targets = self.capture_refit_targets(block_name)
            for args, kwargs in zip(args_list, kwargs_list):
                output = block(*args, **kwargs)
                if refit_targets is not None:
                    refit_targets.append((output[0] if isinstance(output, tuple) else output).detach().cpu())
            for hook in hooks:
                hook.remove()
            tweaks = self.capture_norm_tweaks(quantizers)
            for name, quantizer in quantizers.items():
                self.qparams[name] = quantizer()
                tqdm.write(f"Quantized {name}")
            self.apply_norm_tweaks(tweaks)

            def run(i, block=block):
                return block(*args_list[i], **kwargs_list[i])

            self.refit_block_output(block_name, quantizers, run, refit_targets)
            for j, (args, kwargs) in enumerate(zip(args_list, kwargs_list)):
                output = block(*args, **kwargs)
                output = output[0] if isinstance(output, tuple) else output
                if stream_key in kwargs:
                    kwargs_list[j] = {**kwargs, stream_key: output}
                else:
                    args_list[j] = (output, *args[1:])
            cuda_empty_cache()

    def quantize_speech_encoder(self) -> None:
        """GPTQ the Conformer blocks of the speech encoder on the calibration audio."""
        encoder = self.model.perception.encoder

        def run_sample(index):
            batch = self._calibration_batch(index)
            self.model.perception(input_signal=batch["audios"], input_signal_length=batch["audio_lens"])

        args_list, kwargs_list = self._capture_block_inputs(encoder.layers, run_sample)
        self._quantize_blocks(encoder.layers, "perception.encoder.layers", args_list, kwargs_list, "x")

    def quantize_text_decoder(self) -> None:
        """GPTQ the Qwen3 decoder layers on teacher-forced transcription sequences."""
        layers = self.model.llm.model.layers

        def run_sample(index):
            canary_qwen_logits_fn(self.model, self._calibration_batch(index))

        args_list, kwargs_list = self._capture_block_inputs(layers, run_sample)
        self._quantize_blocks(layers, "llm.model.layers", args_list, kwargs_list, "hidden_states")

    def norm_tweak_targets(self) -> Dict[str, List[str]]:
        """The speech encoder's Conformer norms (see conformer_norm_tweak_targets) and each LLM layer's input
        norm, in front of q/k/v_proj, and post-attention norm, in front of gate/up_proj."""
        targets = conformer_norm_tweak_targets("perception.encoder.layers", len(self.model.perception.encoder.layers))
        for i in range(len(self.model.llm.model.layers)):
            p = f"llm.model.layers.{i}"
            targets[f"{p}.input_layernorm"] = [f"{p}.self_attn.{n}_proj" for n in ("q", "k", "v")]
            targets[f"{p}.post_attention_layernorm"] = [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"]
        return targets

    def transformer_block_widths(self) -> Dict[str, int]:
        """The Qwen3 LLM's decoder layers, which end in a residual sum."""
        width = self.model.llm.config.hidden_size
        return {f"llm.model.layers.{i}": width for i in range(len(self.model.llm.model.layers))}

    def activation_quantization_roles(self) -> Dict[str, str]:
        """Encoder attention, feed-forward and pointwise-conv layers and every LLM projection; the same
        mapping the rotation search quantizes."""
        return get_canary_qwen_activation_roles(self.model, self.quant_cfg.quantize_block_output_linear)

    def online_hadamard_layers(self) -> Dict[str, Optional[str]]:
        """Each fc2-like layer and the activation feeding its online Hadamard (None: at its own input)."""
        return {fc2: activation for activation, fc2 in get_canary_qwen_online_hadamard_layers(self.model)}
