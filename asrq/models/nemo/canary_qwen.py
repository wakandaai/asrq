"""Canary-Qwen 2.5B quantization support.

Provides :class:`CanaryQwenQ`, a :class:`ModelQ` subclass that implements
block-wise post-training quantization for NVIDIA's Canary-Qwen 2.5B
speech-to-text model.  The conformer-based speech encoder and the Qwen3
text decoder are quantized independently using calibration data drawn
from LibriSpeech.
"""
# pyright: reportMissingImports=false
from tqdm import tqdm
# pyright: reportPrivateImportUsage=false

from asrq.core.model import ModelQ
from asrq.core.registry import register_model, ModelNames
from asrq.core.types import InpArgs, InpKwargs
from asrq.quantizers.base import QuantConfig
from asrq.calibration.base import CalibConfig
from typing import Tuple, List, Dict, Any

import math

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from datasets import load_dataset
from itertools import islice
from omegaconf import OmegaConf

from asrq.evaluation.english_text_normalizer import normalizer
from asrq.core.utils import cuda_empty_cache, cuda_synchronize

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.utils import cached_file
from huggingface_hub import CONFIG_NAME

from nemo.collections.speechlm2.models.salm import (
    SALM,
    LightningModule,
    DictConfig,
    AutoTokenizer,
    maybe_install_lora,
)
from nemo.collections.speechlm2.modules.perception import (
    AudioPerceptionModule,
    NeuralModule,
)
from asrq.models.nemo.conformerq import ConformerEncoderQ


from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3PreTrainedModel,
    Qwen3Model,
    Qwen3Config,
    Qwen3RMSNorm,
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
    Qwen3Attention,
    Qwen3MLP,
    GradientCheckpointingLayer,
    ACT2FN,
    
)
from transformers import AutoConfig
from asrq.core.linear import LinearQ



class Qwen3MLPQ(Qwen3MLP):
    """Quantized replacement for the Qwen3 MLP block.

    Replaces all dense projections (gate, up, down) with :class:`LinearQ`
    layers so that the feed-forward sub-network can be served at reduced
    precision.
    """

    def __init__(self, config: Qwen3Config, bits: int = 4) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size: int = config.hidden_size
        self.intermediate_size: int = config.intermediate_size
        self.gate_proj = LinearQ(self.hidden_size, self.intermediate_size, bits, bias=False)
        self.up_proj = LinearQ(self.hidden_size, self.intermediate_size, bits, bias=False)
        self.down_proj = LinearQ(self.intermediate_size, self.hidden_size, bits, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]


class Qwen3AttentionQ(Qwen3Attention):
    """Quantized multi-headed attention for the Qwen3 decoder.

    Replaces Q/K/V/O projections with :class:`LinearQ` layers.  Supports
    grouped-query attention (GQA) with separate key-value head counts and
    optional sliding-window attention.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int, bits: int = 4) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads # type: ignore
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = LinearQ(
            config.hidden_size, config.num_attention_heads * self.head_dim, bits, bias=config.attention_bias
        )
        self.k_proj = LinearQ(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bits, bias=config.attention_bias # type: ignore
        )
        self.v_proj = LinearQ(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bits, bias=config.attention_bias # type: ignore
        )
        self.o_proj = LinearQ(
            config.num_attention_heads * self.head_dim, config.hidden_size, bits, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None # type: ignore


class Qwen3DecoderLayerQ(Qwen3DecoderLayer):
    """Quantized Qwen3 decoder layer.

    Combines :class:`Qwen3AttentionQ` and :class:`Qwen3MLPQ` with RMSNorm
    to form a single quantized transformer block.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int, bits: int = 4) -> None:
        GradientCheckpointingLayer.__init__(self)
        self.hidden_size: int = config.hidden_size

        self.self_attn = Qwen3AttentionQ(config=config, layer_idx=layer_idx, bits=bits)

        self.mlp = Qwen3MLPQ(config, bits=bits)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx] # type: ignore
 

class Qwen3ModelQ(Qwen3Model):
    """Quantized Qwen3 transformer backbone.

    Stacks :class:`Qwen3DecoderLayerQ` blocks with shared embeddings,
    RMSNorm, and rotary position embeddings.
    """

    def __init__(self, config: Qwen3Config, bits: int = 4) -> None:
        Qwen3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayerQ(config, layer_idx, bits=bits) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types # type: ignore

        # Initialize weights and apply final processing
        self.post_init()


class Qwen3ForCausalLMQ(Qwen3ForCausalLM):
    """Quantized Qwen3 causal language model.

    Wraps :class:`Qwen3ModelQ` with an unquantized ``lm_head`` projection
    for next-token prediction.
    """

    def __init__(self, config: Qwen3Config, bits: int = 4) -> None:
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3ModelQ(config, bits=bits)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()



class AudioPerceptionModuleQ(AudioPerceptionModule):
    """Quantized audio perception module.

    Initialises a :class:`ConformerEncoderQ` in place of the standard
    conformer encoder while keeping the preprocessor, spec-augmentation,
    and modality adapter unchanged.
    """

    def __init__(self, cfg: DictConfig) -> None:
        NeuralModule.__init__(self)
        # Initialize components
        self.cfg = cfg
        self.preprocessor = self.from_config_dict(cfg.preprocessor)

        temp = cfg.encoder
        temp.pop("_target_")
        self.encoder = ConformerEncoderQ(**temp)  # to check if the encoder config is valid

        if 'spec_augment' in cfg and cfg.spec_augment is not None:
            self.spec_augmentation = self.from_config_dict(cfg.spec_augment)
        else:
            self.spec_augmentation = None
        self.modality_adapter = self.from_config_dict(cfg.modality_adapter)
        if 'output_dim' not in cfg.modality_adapter and "d_model" in cfg.modality_adapter:  # e.g., conformer encoder
            self.proj = nn.Linear(cfg.modality_adapter.d_model, cfg.output_dim)
        else:
            self.proj = nn.Identity()


class SALMQ(SALM):
    """Quantized SALM (Speech-Augmented Language Model) for Canary-Qwen.

    Replaces the causal LM with :class:`Qwen3ForCausalLMQ` so that all
    decoder linear layers use :class:`LinearQ`.  The audio perception
    module is loaded from the pretrained streaming ASR checkpoint.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to SALM as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        LightningModule.__init__(self)
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)
        self.audio_locator_tag = self.cfg.audio_locator_tag

        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_locator_tag]})
        self.llm = Qwen3ForCausalLMQ(AutoConfig.from_pretrained(self.cfg.pretrained_llm)) # type: ignore

        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_tokens = self.llm.model.embed_tokens
        del self.llm.model.embed_tokens
        maybe_install_lora(self)

        # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        self.perception = AudioPerceptionModuleQ(self.cfg.perception).eval()

        self._use_fsdp = False
        self._use_tp = False



@register_model(ModelNames.NVIDIA_CANARY_QWEN_2_5B)
class CanaryQwenQ(ModelQ):
    """Quantizer for NVIDIA Canary-Qwen 2.5B.

    Implements block-wise quantization of the conformer speech encoder
    (including its pre-encode block) and the Qwen3 text decoder.
    Calibration data is drawn from LibriSpeech.
    """

    @classmethod
    def load_model(cls) -> Tuple[nn.Module, None]:  # type: ignore[override]
        """Load the Canary-Qwen 2.5B model from NeMo.

        Returns:
            Tuple[nn.Module, None]: The pretrained SALM model (eval mode)
                and ``None`` (no separate processor).
        """
        model = SALM.from_pretrained(ModelNames.NVIDIA_CANARY_QWEN_2_5B).eval()
        return model, None
    
    @classmethod
    def load_modelQ(cls) -> Tuple[nn.Module, None]:  # type: ignore[override]
        """Load the Canary-Qwen 2.5B model from NeMo with quantized submodules.

        Returns:
            Tuple[nn.Module, None]: The pretrained SALM model with quantized
                submodules (eval mode) and ``None`` (no separate processor).
        """
        config_file = cached_file(
            ModelNames.NVIDIA_CANARY_QWEN_2_5B,
            CONFIG_NAME,
            cache_dir=None,
            force_download=False,
            resume_download=False,
            proxies=None,
            local_files_only=False,
            token=None,
            revision=None,
        )
        config = OmegaConf.to_container(OmegaConf.load(config_file)) # type: ignore
        model = SALMQ(config).eval() # type: ignore
        return model, None

    @classmethod
    def from_pretrained(
        cls, quant_cfg: QuantConfig, calib_cfg: CalibConfig
    ) -> "CanaryQwenQ":  # type: ignore[override]
        """Load Canary-Qwen from HuggingFace and create a quantizer instance.

        Returns:
            A :class:`CanaryQwenQ` instance ready for quantization.
        """
        model, processor = cls.load_model()
        return cls(model, processor, quant_cfg,  calib_cfg)
    
    @staticmethod
    @torch.no_grad()
    def batch_transcribe(
        audio_arrays: List[Any],
        model: nn.Module,
        device: torch.device | str,
    ) -> List[str]:
        """Auto-regressively transcribe a batch of audio arrays.

        Args:
            audio_arrays: List of raw audio waveforms (1-D array-like).
            model: The SALM model instance.
            device: Target device for tensor computation.

        Returns:
            A list of transcribed text strings, one per audio input.
        """
        audio_lens = torch.tensor([len(s) for s in audio_arrays])

        batch_size = len(audio_arrays)

        audios = torch.zeros((batch_size, int(audio_lens.max())))

        for i in range(batch_size):
            audios[i, :audio_lens[i]] = torch.tensor(audio_arrays[i])

        # <|im_start|>user\nTranscribe the following: <audio_alocator_tag>\n<|im_start|>assistant\n
        tokens = torch.tensor(
            [[151644, 872, 198, 3167, 3114, 279, 2701, 25, 220, 151669, 151645, 198, 151644, 77091, 198]]
        )
        tokens = tokens.expand(batch_size, -1)

        batch = model.prepare_inputs( # type: ignore
            {
                "audios": audios.to(device),
                "audio_lens": audio_lens.to(device),
                "input_ids": tokens.to(device),
                "loss_mask": torch.ones_like(tokens).to(device).bool()
            }
        )

        predicted_idxs = torch.fill(torch.zeros((batch_size,1), dtype=torch.long), model.tokenizer.pad_id).to(device) # type: ignore
        is_done = torch.zeros((batch_size,1), dtype=torch.bool).to(device)
        while predicted_idxs.shape[1] < 128:
            h = model.llm.model.model( # type: ignore
                inputs_embeds=batch["input_embeds"],
                attention_mask=batch["attention_mask"],
                use_cache=False
            ).last_hidden_state
            logits = model.llm.model.lm_head(h) # type: ignore
            idx = torch.argmax(logits[:,-1,:], dim=1, keepdim=True)
            idx = torch.where(is_done, model.tokenizer.eos_id, idx) # type: ignore
            predicted_idxs = torch.cat([predicted_idxs, idx], dim=1)
            idx_embed = model.embed_tokens(idx) # type: ignore
            is_done = is_done | (idx == model.tokenizer.eos_id) # type: ignore  
            if is_done.all():
                break
            batch["input_embeds"] = torch.cat([batch["input_embeds"] , idx_embed], dim=1)
            batch["attention_mask"] = torch.cat([batch["attention_mask"], (~is_done)], dim=1)
        cuda_empty_cache()
        cuda_synchronize()
        return [ 
            model.tokenizer.ids_to_text(pred) # type: ignore
            for pred in predicted_idxs
        ]

    @staticmethod
    @torch.no_grad()
    def batch_forward(
        audio_arrays: List[Any],
        texts: List[str],
        model: nn.Module,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Run a teacher-forced forward pass over a batch of audio/text pairs.

        Args:
            audio_arrays: List of raw audio waveforms (1-D array-like).
            texts: Corresponding ground-truth transcriptions.
            model: The SALM model instance.
            device: Target device for tensor computation.

        Returns:
            Hidden-state tensor from the final decoder layer.
        """
        audio_lens = torch.tensor([len(s) for s in audio_arrays])
        batch_size = len(audio_arrays)
        audios = torch.zeros((batch_size, int(audio_lens.max())))
        for i in range(batch_size):
            audios[i, :audio_lens[i]] = torch.tensor(audio_arrays[i])

        # <|im_start|>user\nTranscribe the following: <audio_alocator_tag>\n<|im_start|>assistant\n
        init_tokens = torch.tensor(
            [151644, 872, 198, 3167, 3114, 279, 2701, 25, 220, 151669, 151645, 198, 151644, 77091, 198]
        )
        encoded_text = [torch.tensor(model.tokenizer.text_to_ids(t)) for t in texts] # type: ignore
        max_len = max([t.shape[0] for t in encoded_text]) + init_tokens.shape[0]
        tokens = torch.fill(torch.zeros((batch_size, max_len), dtype=torch.long), model.tokenizer.pad_id).to(device) # type: ignore
        for i, t in enumerate(encoded_text):
            tokens[i, -t.shape[0]:] = t
            tokens[i, -(t.shape[0]+init_tokens.shape[0]):-t.shape[0]] = init_tokens

        batch = model.prepare_inputs( # type: ignore
            {
                "audios": audios.to(device),
                "audio_lens": audio_lens.to(device),
                "input_ids": tokens.to(device),
                "loss_mask": tokens != model.tokenizer.pad_id # type: ignore
            }
        )

        h = model.llm.model.model( # type: ignore
            inputs_embeds=batch["input_embeds"],
            attention_mask=batch["attention_mask"],
            use_cache=False
        ).last_hidden_state

        cuda_empty_cache()
        cuda_synchronize()
        return h

    
    def quantize_speech_encoder(self) -> None:
        """Quantize the conformer speech encoder block-by-block.

        First quantizes the pre-encode block, then iterates through all
        conformer layers.  Calibration inputs are captured via
        :meth:`_obtain_input_into_first_conformer_block`.
        """
        print(f"---Quantizing speech encoder---")
        # obtain inputs into the conformer blocks
        inp_args, inp_kwargs = self._obtain_input_into_first_conformer_block()
        
        # Quantize the conformer blocks
        self._quantize_conformer_blocks(inp_args, inp_kwargs)

    def _obtain_input_into_first_conformer_block(
        self,
    ) -> Tuple[List[Tuple[torch.Tensor, ...]], List[Dict[str, Any]]]:
        """Capture the inputs flowing into the first conformer layer.

        A temporary ``Catcher`` wrapper is inserted around the first
        conformer block.  Each preprocessed input is fed through the
        encoder; the catcher records ``(*args, **kwargs)`` and raises
        an exception to short-circuit the graph.

        Args:
            inps: Per-sample preprocessed ``(spectrogram, length)`` pairs.

        Returns:
            A tuple ``(inp_args, inp_kwargs)`` where each list has one
            entry per calibration sample.
        """
        num_samples = len(self.calibration_samples)
        inp_args: List[Tuple[torch.Tensor, ...]] = []
        inp_kwargs: List[Dict[str, Any]] = []
        blocks: nn.ModuleList = self.model.perception.encoder.layers # type: ignore
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, *args, **kwargs):
                nonlocal inp_args, inp_kwargs
                inp_args.append(args)
                inp_kwargs.append(kwargs)
                raise Exception("Caught input")
        blocks[0] = Catcher(blocks[0])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.perception.encoder.to(device) # type: ignore
        for j in range(num_samples):
            try:
                with torch.no_grad():
                    audio, text = self.calibration_samples[j]
                    CanaryQwenQ.batch_forward([audio], [text], self.model, device)
            except Exception as e:
                # Expected to raise exception to interrupt after speech encoder
                assert str(e) == "Caught input", e
        assert isinstance(blocks[0].module, nn.Module), "Catcher was not properly inserted around the first conformer block."
        blocks[0] = blocks[0].module
        self.model.perception.encoder.cpu() # type: ignore
        cuda_empty_cache()
        cuda_synchronize()
        return inp_args, inp_kwargs
    
    def _quantize_conformer_blocks(
        self,
        inp_args: List[Tuple[torch.Tensor, ...]],
        inp_kwargs: List[Dict[str, Any]],
    ) -> None:
        """Iterate through all conformer blocks and quantize each one.

        Args:
            blocks: The encoder's conformer layer list.
            inp_args: Per-sample positional inputs captured from the
                first conformer layer.
            inp_kwargs: Per-sample keyword inputs captured from the
                first conformer layer.
        """
        blocks: nn.ModuleList = self.model.perception.encoder.layers # type: ignore
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for i, block in enumerate(tqdm(blocks, desc="Quantizing conformer blocks")):
            block.to(device)
            self._quantize_conformer_block(block, inp_args, inp_kwargs, block_idx=i)
            block.cpu()
            del block
            cuda_empty_cache()
            cuda_synchronize()

    def _quantize_conformer_block(
        self,
        block: nn.Module,
        inp_args: List[Tuple[torch.Tensor, ...]],
        inp_kwargs: List[Dict[str, Any]],
        block_idx: int = 0,
    ) -> None:
        """Quantize a single conformer block.

        Registers forward hooks on every quantisable sub-module, runs
        calibration samples, quantises the captured layers, then re-runs
        the block to update ``inp_kwargs`` for the next block.

        Args:
            block: The conformer transformer block.
            inp_args: Per-sample positional inputs (mutated in-place).
            inp_kwargs: Per-sample keyword inputs (mutated in-place).
            block_idx: Zero-based index of this block in the encoder.
        """
        num_samples = len(self.calibration_samples)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sublayers = {}
        quant_methods = {}
        hooks = []
        def get_hook(name):
            def hook(module, input, output):
                quant_methods[name].add_batch((input[0], output))
            return hook
        
        for name, module in block.named_modules():
            if not self.should_quantize_module(name, module):
                continue
            name = f"perception.encoder.layers.{block_idx}.{name}"
            sublayers[name] = module
            quant_methods[name] = self.quant_cls(module, name, self.quant_cfg)
            hooks.append(module.register_forward_hook(get_hook(name)))

        # Run calibration samples through the layer
        for j in range(num_samples):
            with torch.no_grad():
                inp_kwargs[j]["pos_emb"] = inp_kwargs[j]["pos_emb"].to(device)
                audio_signal = block(*inp_args[j], **inp_kwargs[j])

        for hook in hooks: hook.remove()

        # Quantize
        for name in sublayers.keys():
            with torch.no_grad():
                self.qparams[name] = quant_methods[name]()  
                tqdm.write(f"Quantized {name}")

        # Get outputs as inputs into next layer
        for j in range(num_samples):
            with torch.no_grad():
                audio_signal = block(*inp_args[j], **inp_kwargs[j])
                inp_kwargs[j]["x"] = audio_signal

    
    def quantize_text_decoder(self) -> None:
        """Quantize the Qwen3 text decoder block-by-block.

        Captures inputs into the first decoder layer, then quantises
        each Qwen3 transformer block sequentially.
        """
        print(f"---Quantizing text decoder---")
        use_cache = self.model.llm.config.use_cache # type: ignore
        self.model.llm.config.use_cache = False  # Disable cache for calibration # type: ignore
        
        inps, inp_kwargs = self._obtain_input_into_first_decoder_block()

        # Quantize the text decoder blocks
        self._quantize_text_decoder_blocks(inps, inp_kwargs) 
        self.model.llm.config.use_cache = use_cache  # Restore original cache setting # type: ignore

        cuda_empty_cache()
        cuda_synchronize()

    def _obtain_input_into_first_decoder_block(
        self,
    ) -> Tuple[List[Tuple[torch.Tensor, ...]], List[Dict[str, Any]]]:
        """Capture the inputs flowing into the first Qwen3 decoder layer.

        Constructs full SALM prompts (audio + text) for each calibration
        sample, wraps the first decoder layer with a ``FirstLayerCatcher``,
        and runs each sample through the model.

        Returns:
            A tuple ``(inps, inp_kwargs)`` with one entry per
            calibration sample.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_samples = len(self.calibration_samples)
        inps: List[Tuple[torch.Tensor, ...]] = []
        inp_kwargs: List[Dict[str, Any]] = []

        class FirstLayerCatcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.attention_type = module.attention_type

            def forward(self, *args, **kwargs):
                nonlocal inps, inp_kwargs
                inps.append(args)
                if "use_cache" in kwargs:
                    kwargs.pop("use_cache")
                if "past_key_value" in kwargs:
                    kwargs.pop("past_key_value")
                inp_kwargs.append(kwargs)
                raise Exception("Caught input")
        
        blocks: nn.ModuleList = self.model.llm.base_model.model.model.layers # type: ignore
        blocks[0] = FirstLayerCatcher(blocks[0])
        self.model.perception.to(device)
        self.model.embed_tokens.to(device)

        for j in range(num_samples):
            try:
                with torch.no_grad():
                    audio, text = self.calibration_samples[j]
                    CanaryQwenQ.batch_forward([audio], [text], self.model, device)
            except Exception as e:
                # Expected to raise exception to interrupt after speech encoder
                assert str(e) == "Caught input", e
        assert isinstance(blocks[0].module, nn.Module), "FirstLayerCatcher was not properly inserted around the first decoder block."
        blocks[0] = blocks[0].module
        self.model.perception.cpu()
        self.model.embed_tokens.cpu()
        cuda_empty_cache()
        cuda_synchronize()

        return inps, inp_kwargs
    
    def _quantize_text_decoder_blocks(
        self,
        inps: List[InpArgs],
        inp_kwargs: List[InpKwargs],
    ) -> None:
        """Iterate through all Qwen3 decoder blocks and quantize each one.

        Args:
            blocks: The decoder's layer list.
            td_inps: Per-sample positional inputs captured from the
                first decoder layer.
            td_inp_kwargs: Per-sample keyword inputs captured from the
                first decoder layer.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blocks: nn.ModuleList = self.model.llm.base_model.model.model.layers # type: ignore
        for i, block in enumerate(tqdm(blocks, desc="Quantizing Qwen3 decoder blocks")):
            block.to(device)
            self._quantize_text_decoder_block_GPTQ(block, inps, inp_kwargs, block_idx=i)
            del block
            cuda_empty_cache()
            cuda_synchronize()

    def _quantize_text_decoder_block_GPTQ(
        self,
        block: nn.Module,
        inps: List[InpArgs],
        inp_kwargs: List[InpKwargs],
        block_idx: int = 0,
    ) -> None:
        """Quantize a single Qwen3 decoder block.

        Mirrors conformer block quantization but with decoder-specific
        naming (``llm.base_model.model.model.layers.<idx>.*``).

        Args:
            block: The decoder transformer block.
            inps: Per-sample positional inputs (mutated in-place).
            inp_kwargs: Per-sample keyword inputs.
            block_idx: Zero-based index of this block in the decoder.
        """
        num_samples = len(self.calibration_samples)
        sublayers = {}
        quant_methods = {}
        hooks = []
        def get_hook(name):
            def hook(module, input, output):
                batch = (input[0], output)
                quant_methods[name].add_batch(batch)
            return hook
        for name, module in block.named_modules():
            name = f"llm.base_model.model.model.layers.{block_idx}.{name}"
            if not self.should_quantize_module(name, module):
                continue
            sublayers[name] = module
            quant_methods[name] = self.quant_cls(module, name, self.quant_cfg)
        
        for name in sublayers.keys():
            hooks.append(sublayers[name].register_forward_hook(get_hook(name)))
        
        # Run calibration samples through the layer
        for j in range(num_samples):
            try:
                with torch.no_grad():
                    block(*inps[j], **inp_kwargs[j])
            except Exception as e:
                # Expected to raise exception to interrupt after layer
                pass

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Quantize
        for name in sublayers.keys():
            self.qparams[name] = quant_methods[name]()  
            tqdm.write(f"Quantized {name}")

        # Get outputs as inputs into next layer. does not apply to AWQ
        for j in range(num_samples):
            with torch.no_grad():
                hidden_states = block(*inps[j], **inp_kwargs[j])
                inps[j] = (hidden_states,)


    def quantize(
        self,
    ) -> None:
        """Quantize the full Canary-Qwen model.

        Excludes LoRA layers and ``lm_head`` from quantization, then
        delegates to the base-class :meth:`ModelQ.quantize`.
        """
        self.modules_to_exclude = []
        # exclude lora layers from quantization
        for name, module in self.model.named_modules():
            if "lora_" in name or "lm_head" in name:
                self.quant_cfg.exclude_modules.append(name)

        print(f"Excluding {len(self.quant_cfg.exclude_modules)} LoRA layers from quantization.")
        return super().quantize()  # type: ignore[arg-type]


    def for_activation_quantization(self) -> List[str]:
        linears = []
        num_encoder_blocks: int = len(self.model.perception.encoder.layers) # type: ignore
        num_decoder_blocks: int = len(self.model.llm.base_model.model.model.layers) # type: ignore
        for i in range(num_encoder_blocks):
            stem = f"perception.encoder.layers.{i}"
            linears += [
                f"{stem}.feed_forward1.linear1",
                f"{stem}.self_attn.linear_q",
                f"{stem}.self_attn.linear_k",
                f"{stem}.self_attn.linear_v",
                f"{stem}.self_attn.linear_out",
                f"{stem}.feed_forward2.linear1",
            ]

        for i in range(num_decoder_blocks):
            stem = f"llm.base_model.model.model.layers.{i}"
            linears +=  [
                f"{stem}.self_attn.q_proj.base_layer",
                f"{stem}.self_attn.k_proj",
                f"{stem}.self_attn.v_proj.base_layer",
                f"{stem}.self_attn.o_proj",
                f"{stem}.mlp.gate_proj",
                f"{stem}.mlp.up_proj",
            ]

        return linears
