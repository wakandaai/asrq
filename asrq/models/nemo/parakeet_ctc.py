
# pyright: reportMissingImports=false
from tqdm import tqdm

import torch.nn as nn
from asrq.core.types import Processor
from typing import List, Tuple, Optional

from asrq.core.model import ModelQ
from asrq.core.registry import ModelNames, register_model
import torch
import torch.nn as nn

import numpy as np
import os

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ctc_bpe_models import (
    EncDecCTCModelBPE,
)

from nemo.utils import logging as nemo_logging
import logging

nemo_logging.setLevel(logging.ERROR)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


from nemo.collections.asr.models.ctc_models import (
    EncDecCTCModel,
    Trainer,
    DictConfig
)
from omegaconf import OmegaConf, ListConfig, open_dict
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.parts.submodules.ctc_decoding import (
    CTCDecoding,
    CTCDecodingConfig,
    CTCBPEDecoding,
    CTCBPEDecodingConfig,
)
from nemo.utils import model_utils
from asrq.models.nemo.conformerq import ConformerEncoderQ
from asrq.quantizers.base import QuantConfig
from asrq.calibration.base import CalibConfig
from tqdm import tqdm


class EncDecCTCModelQ(EncDecCTCModel):
    """CTC encoder-decoder model with a quantization-aware conformer encoder.

    Replaces the standard :class:`ConformerEncoder` with
    :class:`ConformerEncoderQ` so that encoder weights can be quantized
    after calibration.  All other components (decoder, loss, decoding,
    WER metric) are initialised identically to the upstream
    :class:`EncDecCTCModel`.
    """

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None) -> None:
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        ASRModel.__init__(self, cfg=cfg, trainer=trainer) # type: ignore | explicitly call ASRModel's init since we're using multiple inheritance

        self.preprocessor = EncDecCTCModel.from_config_dict(self._cfg.preprocessor)
        temp = self._cfg.encoder
        temp.pop("_target_")
        self.encoder = ConformerEncoderQ(**temp) 

        with open_dict(self._cfg):
            if "feat_in" not in self._cfg.decoder or (
                not self._cfg.decoder.feat_in and hasattr(self.encoder, '_feat_out')
            ):
                self._cfg.decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self._cfg.decoder or not self._cfg.decoder.feat_in:
                raise ValueError("param feat_in of the decoder's config is not set!")

            if self.cfg.decoder.num_classes < 1 and self.cfg.decoder.vocabulary is not None:
                logging.info(
                    "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                        self.cfg.decoder.num_classes, len(self.cfg.decoder.vocabulary)
                    )
                )
                cfg.decoder["num_classes"] = len(self.cfg.decoder.vocabulary)

        self.decoder = EncDecCTCModel.from_config_dict(self._cfg.decoder)

        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1, # type: ignore
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecCTCModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        # Setup decoding objects
        decoding_cfg = self.cfg.get('decoding', None)

        # In case decoding config not found, use default config
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.structured(CTCDecodingConfig)
            with open_dict(self.cfg):
                self.cfg.decoding = decoding_cfg

        self.decoding = CTCDecoding(self.cfg.decoding, vocabulary=OmegaConf.to_container(self.decoder.vocabulary))

        # Setup metric with decoding strategy
        self.wer = WER(
            decoding=self.decoding,
            use_cer=self._cfg.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

        # Setup optional Optimization flags
        self.setup_optimization_flags()

        # setting up interCTC loss (from InterCTCMixin)
        self.setup_interctc(decoder_name='decoder', loss_name='loss', wer_name='wer')

        # Adapter modules setup (from ASRAdapterModelMixin)
        self.setup_adapters()


class EncDecCTCModelBPEQ(EncDecCTCModelQ, EncDecCTCModelBPE):
    """CTC encoder-decoder model with BPE tokenization and a quantization-aware encoder.

    Combines :class:`EncDecCTCModelQ` (quantizable conformer encoder) with
    :class:`EncDecCTCModelBPE` (byte-pair-encoding tokenizer and BPE-aware
    CTC decoding).  The tokenizer is set up first, then the base
    :class:`EncDecCTCModelQ` ``__init__`` is called explicitly so that the
    encoder is replaced with :class:`ConformerEncoderQ`.
    """

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None) -> None:
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg, make_copy=False)

        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        # Setup the tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab() # type: ignore

        # Set the new vocabulary
        with open_dict(cfg):
            # sidestepping the potential overlapping tokens issue in aggregate tokenizers
            if self.tokenizer_type == "agg":
                cfg.decoder.vocabulary = ListConfig(vocabulary)
            else:
                cfg.decoder.vocabulary = ListConfig(list(vocabulary.keys())) # type: ignore

        # Override number of classes if placeholder provided
        num_classes = cfg.decoder["num_classes"]

        if num_classes < 1:
            logging.info(
                "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                    num_classes, len(vocabulary)
                )
            )
            cfg.decoder["num_classes"] = len(vocabulary)

        EncDecCTCModelQ.__init__(self, cfg=cfg, trainer=trainer)  # initialize the base class (EncDecCTCModelQ)
       
        # Setup decoding objects
        decoding_cfg = self.cfg.get('decoding', None)

        # In case decoding config not found, use default config
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.structured(CTCBPEDecodingConfig)
            with open_dict(self.cfg):
                self.cfg.decoding = decoding_cfg

        self.decoding = CTCBPEDecoding(self.cfg.decoding, tokenizer=self.tokenizer)

        # Setup metric with decoding strategy
        self.wer = WER(
            decoding=self.decoding,
            use_cer=self._cfg.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

    

@register_model(ModelNames.NVIDIA_PARAKEET_CTC_1_1B)
class ParakeetCTCQ(ModelQ):
    """Quantization wrapper for the NVIDIA Parakeet-CTC 1.1B model.

    Provides calibration-based quantization of the conformer speech encoder.
    The text decoder is a CTC projection layer and is left unquantized
    (``quantize_text_decoder`` is a no-op).
    """
    model: EncDecCTCModelQ

    @classmethod
    def load_model(cls) -> Tuple[nn.Module, Processor]:
        """Load the pretrained Parakeet-CTC 1.1B model in greedy-batch mode.

        Returns:
            A ``(model, processor)`` tuple.  ``processor`` is ``None`` because
            NeMo models handle their own preprocessing.
        """
        model = ASRModel.from_pretrained(ModelNames.NVIDIA_PARAKEET_CTC_1_1B).eval() # type: ignore
        model.cfg.decoding.strategy = "greedy_batch"
        model.change_decoding_strategy(model.cfg.decoding)
        return model, None
    
    @classmethod
    def load_modelQ(cls) -> Tuple[nn.Module, Processor]:
        """Load the quantization-aware Parakeet-CTC model variant.

        Returns:
            A ``(model, processor)`` tuple where ``model`` is an
            :class:`EncDecCTCModelQ` instance.
        """
        cfg = OmegaConf.load(os.path.join(__file__, "..", "config/parakeetctc_config.yaml"))
        assert isinstance(cfg, DictConfig), "Expected a DictConfig object"
        model = EncDecCTCModelQ(cfg=cfg).eval() 
        return model, None
    
    @classmethod
    def from_pretrained(cls, quant_cfg: QuantConfig,  calib_cfg: CalibConfig) -> 'ParakeetCTCQ': # type: ignore[reportIncompatibleMethodOverride]
        """Build a :class:`ParakeetCTCQ` from the pretrained checkpoint.

        Args:
            bits: Target quantization bit-width.
            method: Quantization algorithm name (e.g. ``"RTN"``, ``"GPTQ"``).
            calib_config: Dictionary containing calibration configuration options.
                Supported keys:
                    - ``n_calib``: Number of LibriSpeech calibration samples to load.
                      ``None`` means no calibration data.
                    - ``transcribe_calib``: Whether to transcribe calibration audio.
                    - ``include_short_audio``: Whether to include short audio samples.

        Returns:
            A ready-to-quantize :class:`ParakeetCTCQ` instance.
        """
        model, processor = cls.load_model()
        return cls(model, processor, quant_cfg=quant_cfg, calib_cfg=calib_cfg)


    @staticmethod
    @torch.no_grad()
    def batch_transcribe(
        audio_arrays: List[np.ndarray],
        model: nn.Module,
        device: torch.device | str,
    ) -> List[str]:
        """Transcribe a batch of audio waveforms to text.

        Args:
            audio_arrays: List of 1-D waveform tensors (or array-likes).
            model: The Parakeet-CTC model (or a quantised variant).
            device: Device to run inference on.

        Returns:
            A list of transcription strings, one per input audio.
        """
        model.eval()
        audio_lens = torch.tensor([len(s) for s in audio_arrays])
        audio = torch.zeros((len(audio_arrays), int(audio_lens.max().item())))
        for i in range(len(audio_arrays)):
            audio[i, :audio_lens[i]] = torch.tensor(audio_arrays[i])

        input_signal = audio.to(device)
        input_signal_length = audio_lens.to(device)

        log_probs, encoded_len, greedy_predictions = model(input_signal=input_signal, input_signal_length=input_signal_length)

        hypotesis = model.decoding.ctc_decoder_predictions_tensor( # type: ignore
            log_probs,
            decoder_lengths=encoded_len,
            return_hypotheses=True,
        ) 
        return [hypotesis[i].text for i in range(len(audio_arrays))]

    @staticmethod
    @torch.no_grad()
    def batch_forward(
        audio_arrays: List[np.ndarray],
        model: nn.Module,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Run a forward pass and return log-probabilities.

        Args:
            audio_arrays: List of 1-D waveform tensors (or array-likes).
            model: The Parakeet-CTC model.
            device: Device to run inference on.

        Returns:
            Log-probability tensor of shape ``(batch, time, vocab)``.
        """
        model.eval()
        audio_lens = torch.tensor([len(s) for s in audio_arrays])
        audio = torch.zeros((len(audio_arrays), int(audio_lens.max())))
        for i in range(len(audio_arrays)):
            audio[i, :audio_lens[i]] = torch.tensor(audio_arrays[i])
        input_signal = audio.to(device)
        input_signal_length = audio_lens.to(device)
        log_probs, encoded_len, greedy_predictions = \
        model(input_signal=input_signal, input_signal_length=input_signal_length)
        return log_probs

    def quantize_speech_encoder(self) -> None:
        """Quantize the conformer speech encoder using calibration data.

        Inserts a *catcher* module at the first encoder layer to capture
        intermediate activations, then iterates over every encoder block to
        collect statistics and quantize each eligible sub-layer.
        """
        self.model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = self.model.encoder
        self.model.to(device)
        
        quantizers = {}
        sublayers = {}
        
        hooks = []
        
        blocks: nn.ModuleList = encoder.layers # type: ignore
        inp_args = []
        inp_kwargs = []
        class catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, *args, **kwargs):
                inp_args.append(args)
                inp_kwargs.append(kwargs)
                raise Exception("Caught Input")
        blocks[0] = catcher(blocks[0]) # type: ignore

        # run calibration samples through the model to collect stats
        for i in range(len(self.calibration_samples)):
            audio, text = self.calibration_samples[i]
            try:
                ParakeetCTCQ.batch_forward([audio], self.model, device=device)
            except Exception as e:
                assert str(e) == "Caught Input", f"Unexpected exception {e} during quantization"
        # remove hooks
        for hook in hooks:
            hook.remove()
        
        blocks[0] = blocks[0].module # type: ignore
        
        # quantize the encoder blocks
        for idx, block in enumerate(tqdm(blocks, desc="Quantizing encoder blocks")):
            quantizers = {}
            sublayers = {}
            for name, module in block.named_modules():
                name = f"encoder.layers.{idx}.{name}"
                if self.should_quantize_module(name, module):
                    sublayers[name] = module
                    quantizers[name] = self.quant_cls(module, name, self.quant_cfg)
            # hooks
            def add_hook(name):
                def hook(module, input, output):
                    batch = (input[0], output)
                    quantizers[name].add_batch(batch)
                return hook
            hooks = []
            for name, module in sublayers.items():
                hooks.append(module.register_forward_hook(add_hook(name)))
            
            # run calibration samples through the model to collect stats
            for i in range(len(self.calibration_samples)):  # type: ignore[arg-type]
                audio_signal = block(
                    **inp_kwargs[i],
                )

            # remove hooks
            for hook in hooks:
                hook.remove()

            # quanitze
            for name, quantizer in quantizers.items():
                self.qparams[name] = quantizer()
                tqdm.write(f"Quantized {name}")

            # get the output of the block and use it as input for the next block
            for i in range(len(self.calibration_samples)):  # type: ignore[arg-type]
                audio_signal = block(
                    **inp_kwargs[i],
                )
                inp_kwargs[i]["x"] = audio_signal

            block.to("cpu")

    def quantize_text_decoder(self) -> None:
        """No-op — CTC models do not have a text decoder to quantize."""
        pass

    def for_activation_quantization(self) -> List[str]:
        linears = []
        num_encoder_blocks = len(self.model.encoder.layers)
        for i in range(num_encoder_blocks):
            stem = f"encoder.layers.{i}"
            linears += [
                f"{stem}.feed_forward1.linear1",
                f"{stem}.self_attn.linear_q",
                f"{stem}.self_attn.linear_k",
                f"{stem}.self_attn.linear_v",
                f"{stem}.self_attn.linear_out",
                f"{stem}.feed_forward2.linear1",
            ]

        return linears

