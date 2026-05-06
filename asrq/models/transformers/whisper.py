# pyright: reportPrivateImportUsage=false
# pyright: reportMissingImports=false

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
from asrq.core.types import InpArgs, InpKwargs, Processor

from asrq.core.registry import ModelNames, register_model
from asrq.core.model import ModelQ
from asrq.core.linear import LinearQ
import transformers
from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration, 
    WhisperAttention,
    WhisperConfig,
    WhisperEncoderLayer,
    WhisperDecoderLayer,
    WhisperDecoder,
    WhisperPositionalEmbedding,
    WhisperPreTrainedModel,
    WhisperModel,
    GradientCheckpointingLayer,
    WhisperEncoder,
    ACT2FN,
)
from asrq.core.utils import cuda_empty_cache, cuda_synchronize
from tqdm import tqdm
from asrq.transforms.base import TransformConfig
from asrq.calibration.base import CalibConfig
from asrq.quantizers.base import QuantConfig
import math




class WhisperAttentionQ(WhisperAttention):
    """Quantized multi-headed attention replacing linear projections with :class:`LinearQ`.

    Mirrors :class:`WhisperAttention` but uses quantization-aware
    ``LinearQ`` layers for the Q/K/V and output projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        layer_idx: int | None = None,
        config: WhisperConfig | None = None,
        bits: int = 4,
    ) -> None:
        """Initialise the quantized attention module.

        Args:
            embed_dim: Total dimension of the model.
            num_heads: Number of parallel attention heads.
            dropout: Dropout probability on attention weights.
            is_decoder: Whether this attention is used in the decoder.
            bias: Whether linear layers include a bias term.
            is_causal: Whether to apply a causal mask.
            layer_idx: Index of the parent transformer layer.
            config: Whisper model configuration.
            bits: Target bitwidth for quantization.
        """
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        if layer_idx is None and is_decoder:
            logger.warning_once( # type: ignore
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.layer_idx = layer_idx

        self.k_proj = LinearQ(embed_dim, embed_dim, bits=bits, bias=False)
        self.v_proj = LinearQ(embed_dim, embed_dim, bits=bits, bias=bias)
        self.q_proj = LinearQ(embed_dim, embed_dim, bits=bits, bias=bias)
        self.out_proj = LinearQ(embed_dim, embed_dim, bits=bits, bias=bias)


class WhisperEncoderLayerQ(WhisperEncoderLayer):
    """Quantized Whisper encoder layer with :class:`LinearQ` feed-forward layers."""

    def __init__(self, config: WhisperConfig, bits: int = 4) -> None:
        """Initialise the quantized encoder layer.

        Args:
            config: Whisper model configuration.
            bits: Target bitwidth for quantization.
        """
        GradientCheckpointingLayer.__init__(self)
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttentionQ(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            bits=bits,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = LinearQ(self.embed_dim, config.encoder_ffn_dim, bits=bits)
        self.fc2 = LinearQ(config.encoder_ffn_dim, self.embed_dim, bits=bits)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)


class WhisperEncoderQ(WhisperEncoder):
    """Quantized Whisper encoder using :class:`WhisperEncoderLayerQ` blocks."""

    def __init__(self, config: WhisperConfig, bits: int = 4) -> None:
        """Initialise the quantized encoder.

        Args:
            config: Whisper model configuration.
            bits: Target bitwidth for quantization.
        """
        WhisperPreTrainedModel.__init__(self, config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        self.layers = nn.ModuleList([WhisperEncoderLayerQ(config, bits=bits) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class WhisperDecoderLayerQ(WhisperDecoderLayer):
    """Quantized Whisper decoder layer with :class:`LinearQ` projections."""

    def __init__(self, config: WhisperConfig, layer_idx: int | None = None, bits: int = 4) -> None:
        """Initialise the quantized decoder layer.

        Args:
            config: Whisper model configuration.
            layer_idx: Zero-based index of this layer in the decoder stack.
            bits: Target bitwidth for quantization.
        """
        GradientCheckpointingLayer.__init__(self)
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttentionQ(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            layer_idx=layer_idx,
            config=config,
            bits=bits,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperAttentionQ(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            layer_idx=layer_idx,
            config=config,
            bits=bits,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = LinearQ(self.embed_dim, config.decoder_ffn_dim, bits=bits)
        self.fc2 = LinearQ(config.decoder_ffn_dim, self.embed_dim, bits=bits)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)


class WhisperDecoderQ(WhisperDecoder):
    """Quantized Whisper decoder using :class:`WhisperDecoderLayerQ` blocks."""

    def __init__(self, config: WhisperConfig, bits: int = 4) -> None:
        """Initialise the quantized decoder.

        Args:
            config: Whisper model configuration.
            bits: Target bitwidth for quantization.
        """
        WhisperPreTrainedModel.__init__(self, config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)

        self.layers = nn.ModuleList(
            [WhisperDecoderLayerQ(config, layer_idx, bits=bits) for layer_idx in range(config.decoder_layers)]
        )

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class WhisperModelQ(WhisperModel):
    """Quantized Whisper model combining :class:`WhisperEncoderQ` and :class:`WhisperDecoderQ`."""

    def __init__(self, config: WhisperConfig, bits: int = 4) -> None:
        """Initialise the quantized Whisper model.

        Args:
            config: Whisper model configuration.
            bits: Target bitwidth for quantization.
        """
        WhisperPreTrainedModel.__init__(self, config)
        self.encoder = WhisperEncoderQ(config, bits=bits)
        self.decoder = WhisperDecoderQ(config, bits=bits)
        # Initialize weights and apply final processing
        self.post_init()


class WhisperForConditionalGenerationQ(WhisperForConditionalGeneration):
    """Quantized Whisper conditional-generation model with :class:`WhisperModelQ` backbone."""

    def __init__(self, config: WhisperConfig, bits: int = 4) -> None:
        """Initialise the quantized conditional-generation model.

        Args:
            config: Whisper model configuration.
            bits: Target bitwidth for quantization.
        """
        WhisperPreTrainedModel.__init__(self, config)
        self.model = WhisperModelQ(config, bits=bits)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.max_target_positions = config.max_target_positions
        # Initialize weights and apply final processing
        self.post_init()



@register_model(ModelNames.OPENAI_WHISPER_LARGE_V3)
class WhisperQ(ModelQ):
    """Quantizer for OpenAI Whisper Large V3.

    Implements block-wise quantization of both the speech encoder and text
    decoder.  Calibration data is drawn from LibriSpeech when ``n_calib``
    is supplied to :meth:`from_pretrained`.
    """
    model: WhisperForConditionalGeneration
    @classmethod
    def load_model(cls) -> Tuple[nn.Module, Processor]:  # type: ignore[override]
        """Load the Whisper Large V3 model with SDPA attention.

        Returns:
            Tuple[nn.Module, Processor]: The pretrained model (eval mode)
                and its associated processor.
        """
        model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
            ModelNames.OPENAI_WHISPER_LARGE_V3,
            attn_implementation="sdpa"
        ).eval()
        processor = transformers.AutoProcessor.from_pretrained(ModelNames.OPENAI_WHISPER_LARGE_V3)
        return model, processor

    @classmethod
    def load_model_eager(cls) -> Tuple[nn.Module, Processor]:
        """Load the Whisper Large V3 model with eager attention.

        Returns:
            Tuple[nn.Module, Processor]: The pretrained model (eval mode)
                and its associated processor.
        """
        model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
            ModelNames.OPENAI_WHISPER_LARGE_V3,
            attn_implementation="eager",
        ).eval()
        processor = transformers.AutoProcessor.from_pretrained(ModelNames.OPENAI_WHISPER_LARGE_V3)
        return model, processor
    
    @classmethod
    def load_modelQ(cls) -> Tuple[nn.Module, Processor]:
        """Load the quantized Whisper Large V3 model.

        Returns:
            Tuple[nn.Module, Processor]: The quantized model (eval mode)
                and its associated processor.
        """
        config = transformers.WhisperConfig.from_pretrained(ModelNames.OPENAI_WHISPER_LARGE_V3)
        generation_config = transformers.GenerationConfig.from_pretrained(ModelNames.OPENAI_WHISPER_LARGE_V3)
        model = WhisperForConditionalGenerationQ(config).eval()
        model.generation_config = generation_config
        processor = transformers.AutoProcessor.from_pretrained(ModelNames.OPENAI_WHISPER_LARGE_V3)
        return model, processor

    @classmethod
    def from_pretrained(cls, quant_cfg: QuantConfig, calib_cfg: CalibConfig) -> "WhisperQ":  # type: ignore[override]
        # create model
        model, processor = cls.load_model()
        return cls(model, processor, quant_cfg=quant_cfg, calib_cfg=calib_cfg)
    
    @staticmethod
    def transcribe(
        audio: np.ndarray,
        model: nn.Module,
        processor: Processor,
        sr: int = 16000,
    ) -> str:
        """Autoregressively transcribe an audio waveform.

        Performs greedy decoding up to 448 tokens, stopping early on EOS.

        Args:
            audio: Raw audio waveform as a 1-D numpy array.
            model: The Whisper model (must expose ``.model.encoder``,
                ``.model.decoder``, and ``.proj_out``).
            processor: HuggingFace processor for feature extraction and
                token decoding.
            sr: Sampling rate of *audio* (default 16 000).

        Returns:
            The decoded transcription string.
        """
        model.to("cuda")
        device = model.device
        assert isinstance(processor, transformers.WhisperProcessor)
        input_features = processor(audio, return_tensors="pt", sampling_rate=sr).input_features
        gen_ids = model.generate(
            input_features=input_features.to(device).to(model.dtype), # type: ignore
            language="en",
            task="transcribe",
        )
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip() # type: ignore
        cuda_empty_cache()
        cuda_synchronize()
        return text

    @staticmethod
    def forward(
        audio: np.ndarray,
        text: str,
        model: nn.Module,
        processor: Processor,
        sr: int = 16000,
    ) -> Any:
        """Run a teacher-forced forward pass through encoder and decoder.

        The decoder receives the ground-truth *text* (tokenised and prefixed
        with Whisper special tokens) so every position attends to the correct
        history.  Useful for collecting calibration activations.

        Args:
            audio: Raw audio waveform as a 1-D numpy array.
            text: Ground-truth transcription.
            model: The Whisper model.
            processor: HuggingFace processor.
            sr: Sampling rate of *audio* (default 16 000).

        Returns:
            Decoder output (has a ``last_hidden_state`` attribute).
        """
        device = model.device
        input_features = processor(audio, return_tensors="pt", sampling_rate=sr).input_features
        enc = model.model.encoder(input_features.to(device)) # type: ignore

        prefix = [50258, 50259, 50360, 50364]
        text_ids = processor.tokenizer.encode(text, add_special_tokens=False)
        input_ids = torch.tensor([prefix + text_ids]).to(device) # type: ignore

        result = model.model.decoder( # type: ignore
            input_ids=input_ids,
            encoder_hidden_states=enc.last_hidden_state,
        )
        cuda_empty_cache()
        cuda_synchronize()
        return result

    # All Overrides
    def quantize_speech_encoder(self) -> None:
        """Quantize the speech encoder block-by-block.

        Captures the activations flowing into the first encoder layer using
        a :class:`FirstLayerCatcher`, then iterates through each encoder
        block, quantising its linear sub-modules with the configured
        quantization method.
        """
        # Obtain inputs into the first encoder block
        inp_args, inp_kwargs = self._obtain_input_into_first_encoder_block()

        # now quantize each block
        self._quantize_encoder_blocks(inp_args, inp_kwargs)

    def _obtain_input_into_first_encoder_block(
        self,
    ) -> Tuple[List[InpArgs], List[InpKwargs]]:
        """Capture the inputs flowing into the first encoder layer.

        A temporary ``FirstLayerCatcher`` wrapper is inserted around the
        first encoder block.  For each calibration sample a forward pass
        is executed; the catcher records ``(*args, **kwargs)`` and raises
        an exception to short-circuit the rest of the graph.

        Returns:
            A tuple ``(inp_args, inp_kwargs)`` where each list has one
            entry per calibration sample.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_samples = len(self.calibration_samples)
        blocks: nn.ModuleList = self.model.model.encoder.layers # type: ignore
        # get input into the first encoder layer
        inp_args = []
        inp_kwargs = []
        # Catcher to get inputs into the first layer
        class FirstLayerCatcher(nn.Module):
            def __init__(self, module:nn.Module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                nonlocal inp_args, inp_kwargs
                inp_args.append(args)
                inp_kwargs.append(kwargs)
                raise Exception("Caught input")
            
        blocks[0] = FirstLayerCatcher(blocks[0])
        self.model.model.encoder.conv1.to(device)
        self.model.model.encoder.conv2.to(device)
        self.model.model.encoder.embed_positions.to(device)
        # forward pass to get inputs
        for i in range(num_samples):
            try:
                audio, text = self.calibration_samples[i]
                with torch.no_grad():
                    WhisperQ.forward(audio, text, self.model, self.processor)
            except Exception as e:
                assert str(e) == "Caught input", str(e)
        # restore the first layer
        assert isinstance(blocks[0].module, nn.Module)
        blocks[0] = blocks[0].module
        
        # move back to cpu
        self.model.model.encoder.conv1.cpu()
        self.model.model.encoder.conv2.cpu()
        self.model.model.encoder.embed_positions.cpu()

        return inp_args, inp_kwargs

    def _quantize_encoder_blocks(
        self,
        inp_args: List[Tuple[torch.Tensor, ...]],
        inp_kwargs: List[Dict[str, Any]],
    ) -> None:
        """Iterate through all encoder blocks and quantize each one.

        Args:
            inp_args: Per-sample positional inputs captured from the
                first encoder layer.
            inp_kwargs: Per-sample keyword inputs captured from the
                first encoder layer.
        """
        blocks = self.model.model.encoder.layers
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for block_id, block in enumerate(tqdm(blocks, desc="Quantizing encoder blocks")):
            block.to(device)
            self._quantize_encoder_block(block, block_id, inp_args, inp_kwargs)
            block.cpu()
            cuda_empty_cache()
            cuda_synchronize()

    def _quantize_encoder_block(
        self,
        block: nn.Module,
        block_idx: int,
        inp_args: List[Tuple[torch.Tensor, ...]],
        inp_kwargs: List[Dict[str, Any]],
    ) -> None:
        """Quantize a single encoder block.

        Registers forward hooks on every quantizable sub-module to collect
        ``(input, output)`` pairs, runs the calibration samples through
        the block, then invokes the quantization method on each sub-module.
        Finally, re-runs the block to update ``inp_args`` for the next block.

        Args:
            block: The encoder transformer block.
            block_idx: Zero-based index of the block in the encoder stack.
            inp_args: Per-sample positional inputs (mutated in-place for
                the next block).
            inp_kwargs: Per-sample keyword inputs.
        """
        num_samples = len(self.calibration_samples)
        # register hooks to capture inputs and outputs of attention and fc layers
        quant_methods = {}
        submodules = {}
        for name, module in block.named_modules():
            name = f"model.encoder.layers.{block_idx}.{name}"
            if self.should_quantize_module(name, module):
                quant_methods[name] = self.quant_cls(module, name, self.quant_cfg)
                submodules[name] = module
        
        def get_hook(name):
            def hook(module, input, output):
                batch = (input[0], output)
                quant_methods[name].add_batch(batch)
            return hook
        hooks = []
        for name in quant_methods.keys():
            layer = submodules[name]
            hooks.append(layer.register_forward_hook(get_hook(name)))

        # forward pass through the block
        for i in range(num_samples):
            with torch.no_grad():
                block(*inp_args[i], **inp_kwargs[i])
        # remove hooks
        for h in hooks:
            h.remove()
        # quantize layers
        for name in quant_methods.keys():
            qresult = quant_methods[name]()
            self.qparams[name] = qresult
            tqdm.write(f"Quantized layer {name}")

        # get input into the next block
        for i in range(num_samples):
            with torch.no_grad():
                out = block(*inp_args[i], **inp_kwargs[i])
                inp_args[i] = out # type: ignore

    def quantize_text_decoder(self) -> None:
        """Quantize the text decoder block-by-block.

        Temporarily disables KV-cache, captures inputs into the first
        decoder layer, then quantises each decoder block sequentially.

        Args:
            save_dir: Unused — kept for interface compatibility.
        """
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        # Get inputs into the first decoder block
        inp_args, inp_kwargs = self._obtain_input_into_first_decoder_block()
        
        # now quantize each block
        self._quantize_decoder_blocks(inp_args, inp_kwargs)

        self.model.config.use_cache = use_cache

    def _obtain_input_into_first_decoder_block(
        self,
    ) -> Tuple[List[Tuple[torch.Tensor, ...]], List[Dict[str, Any]]]:
        """Capture the inputs flowing into the first decoder layer.

        Moves the full encoder and decoder embeddings to GPU, wraps the
        first decoder layer with a ``FirstLayerCatcher``, and runs each
        calibration sample through :meth:`forward`.

        Returns:
            A tuple ``(inp_args, inp_kwargs)`` with one entry per
            calibration sample.
        """
        num_samples = len(self.calibration_samples)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inp_args: List[Tuple[torch.Tensor, ...]] = []
        inp_kwargs: List[Dict[str, Any]] = []
        blocks = self.model.model.decoder.layers

         # Catcher to get inputs into the first layer
        class FirstLayerCatcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                nonlocal inp_args, inp_kwargs
                inp_args.append(args)
                inp_kwargs.append(kwargs)
                raise Exception("Caught input")

        blocks[0] = FirstLayerCatcher(blocks[0])

        self.model.model.encoder.to(device) # type: ignore
        self.model.model.decoder.embed_tokens.to(device)
        self.model.model.decoder.embed_positions.to(device)

        for i in range(num_samples):
            audio, text = self.calibration_samples[i]
            try:
                with torch.no_grad():
                    WhisperQ.forward(audio, text, self.model, self.processor)
            except Exception as e:
                assert str(e) == "Caught input"

        # restore the first layer
        assert isinstance(blocks[0].module, nn.Module)
        blocks[0] = blocks[0].module
        
        # move back to cpu
        self.model.model.encoder.cpu()
        self.model.model.decoder.embed_tokens.cpu()
        self.model.model.decoder.embed_positions.cpu()
        cuda_empty_cache()
        cuda_synchronize()

        return inp_args, inp_kwargs

    def _quantize_decoder_blocks(
        self,
        inp_args: List[Tuple[torch.Tensor, ...]],
        inp_kwargs: List[Dict[str, Any]],
    ) -> None:
        """Iterate through all decoder blocks and quantize each one.

        Args:
            inp_args: Per-sample positional inputs captured from the
                first decoder layer.
            inp_kwargs: Per-sample keyword inputs captured from the
                first decoder layer.
        """
        blocks = self.model.model.decoder.layers
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for block_id, block in enumerate(tqdm(blocks, desc="Quantizing decoder blocks")):
            block.to(device)
            self._quantize_decoder_block(block, block_id, inp_args, inp_kwargs)
            block.cpu()
            cuda_empty_cache()
            cuda_synchronize()

    def _quantize_decoder_block(
        self,
        block: nn.Module,
        block_idx: int,
        inp_args: List[Tuple[torch.Tensor, ...]],
        inp_kwargs: List[Dict[str, Any]],
    ) -> None:
        """Quantize a single decoder block.

        Mirrors :meth:`_quantize_encoder_block` but with decoder-specific
        naming (``model.decoder.layers.<idx>.*``) and updates ``inp_args``
        for the next decoder block.

        Args:
            block: The decoder transformer block.
            block_idx: Zero-based index of the block in the decoder stack.
            inp_args: Per-sample positional inputs (mutated in-place).
            inp_kwargs: Per-sample keyword inputs.
        """
        num_samples = len(self.calibration_samples)
        # register hooks to capture inputs and outputs of attention and fc layers
        quant_methods = {}
        submodules = {}
        for name, module in block.named_modules():
            name = f"model.decoder.layers.{block_idx}.{name}"
            if self.should_quantize_module(name, module):
                quant_methods[name] = self.quant_cls(module, name, self.quant_cfg)
                submodules[name] = module
        
        def get_hook(name):
            def hook(module, input, output):
                batch = (input[0], output)
                quant_methods[name].add_batch(batch)
            return hook
        hooks = []
        for name in quant_methods.keys():
            layer = submodules[name]
            hooks.append(layer.register_forward_hook(get_hook(name)))

        # forward pass through the block
        for i in range(num_samples):
            with torch.no_grad():
                block(*inp_args[i], **inp_kwargs[i])
        # remove hooks
        for h in hooks:
            h.remove()
        # quantize layers
        for name in quant_methods.keys():
            qresult = quant_methods[name]()
            self.qparams[name] = qresult
            tqdm.write(f"Quantized layer {name}")

        # get input into the next block
        for i in range(num_samples):
            with torch.no_grad():
                out = block(*inp_args[i], **inp_kwargs[i])
                inp_args[i] = out


    def for_activation_quantization(self) -> List[str]:
        linears = []
        num_encoder_blocks: int = self.model.config.encoder_layers # type: ignore
        num_decoder_blocks: int = self.model.config.decoder_layers # type: ignore
        for i in range(num_encoder_blocks):
            stem = f"model.encoder.layers.{i}"
            linears += [
                f"{stem}.self_attn.q_proj",
                f"{stem}.self_attn.k_proj",
                f"{stem}.self_attn.v_proj",
                f"{stem}.self_attn.out_proj",
                f"{stem}.fc1",
            ]

        for i in range(num_decoder_blocks):
            stem = f"model.decoder.layers.{i}"
            linears += [
                f"{stem}.self_attn.q_proj",
                f"{stem}.self_attn.k_proj",
                f"{stem}.self_attn.v_proj",
                f"{stem}.self_attn.out_proj",
                f"{stem}.encoder_attn.q_proj",
                f"{stem}.encoder_attn.k_proj",
                f"{stem}.encoder_attn.v_proj",
                f"{stem}.encoder_attn.out_proj",
                f"{stem}.fc1",
            ]

        return linears
