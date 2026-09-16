# pyright: reportMissingImports=false

import os

from omegaconf import DictConfig, OmegaConf

from asrq.core.registry import (
    ModelNames,
    TransformNames,
    register_transform,
    register_transform_config,
)
from asrq.transforms.base import BaseTransform, TransformConfig
from asrq.transforms.rotation.canary_qwen_utils import (
    get_canary_qwen_rotation_inputs,
    get_canary_qwen_rotation_layers,
)
from asrq.transforms.rotation.hadamard_search import EvolutionConfig
from asrq.transforms.rotation.parakeet_ctc_utils import (
    get_parakeet_rotation_inputs,
    get_parakeet_rotation_layers,
)
from asrq.transforms.rotation.utils import (
    apply_rotations,
    check_hadamard_matches_groups,
    learn_rotations,
)
from asrq.transforms.rotation.whisper_utils import (
    get_whisper_rotation_inputs,
    get_whisper_rotation_layers,
)

# Everything learn_rotations needs, per model: the calibration dataloader, the loss, the norm
# mapping and the layers to rotate.
_ROTATION_INPUTS_FNS = {
    ModelNames.OPENAI_WHISPER_LARGE_V3: (
        lambda modelQ, num_samples, batch_size, quantize_block_output_linear: get_whisper_rotation_inputs(
            modelQ.model,
            modelQ.processor,
            modelQ.calibration_samples[:num_samples],
            batch_size=batch_size,
        )
    ),
    ModelNames.NVIDIA_PARAKEET_CTC_1_1B: (
        lambda modelQ, num_samples, batch_size, quantize_block_output_linear: get_parakeet_rotation_inputs(
            modelQ.model, modelQ.calibration_samples[:num_samples],
            quantize_block_output_linear=quantize_block_output_linear, batch_size=batch_size,
        )
    ),
    ModelNames.NVIDIA_CANARY_QWEN_2_5B: (
        lambda modelQ, num_samples, batch_size, quantize_block_output_linear: get_canary_qwen_rotation_inputs(
            modelQ.model, modelQ.calibration_samples[:num_samples],
            quantize_block_output_linear=quantize_block_output_linear, batch_size=batch_size,
        )
    ),
}

# The structural half of the same mapping, for applying a rotation that is already learned.
_ROTATION_LAYERS_FNS = {
    ModelNames.OPENAI_WHISPER_LARGE_V3: get_whisper_rotation_layers,
    ModelNames.NVIDIA_PARAKEET_CTC_1_1B: get_parakeet_rotation_layers,
    ModelNames.NVIDIA_CANARY_QWEN_2_5B: get_canary_qwen_rotation_layers,
}


@register_transform_config(TransformNames.rotation)
class RotationTransformConfig(TransformConfig):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.num_samples = cfg.num_samples
        self.epochs = cfg.epochs
        self.learning_rate = cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.learn_rotation = cfg.learn_rotation
        # Block size of the online Hadamard applied to every FC2 input. Set it to the
        # activation quantization group size: the rotation only has to spread outliers across
        # channels that share a quantization scale, and mixing wider than the group measurably
        # hurts. It is saved with the rotation and checked when the rotation is applied.
        self.hadamard_block_size = cfg.hadamard_block_size
        # Activation fake quantization applied to the rotated layers' inputs during the search.
        # Without it the output is independent of the rotation and the search learns nothing,
        # so match the activation quantization used at evaluation. exp.py and rot-exp.py set
        # these four from the top-level activation_* settings, the same ones evaluation reads;
        # abits of 16 or more means no quantization.
        self.abits = cfg.abits
        if self.abits is not None and self.abits >= 16:
            self.abits = None
        # "kl" matches the full-precision model's output distribution; "ce" minimises the
        # model's training loss on the calibration labels. See learn_rotations.
        self.objective = cfg.objective
        if self.objective not in ("kl", "ce"):
            raise ValueError(f"transform.objective must be 'kl' or 'ce', got {self.objective!r}")
        self.activation_group_size = cfg.activation_group_size
        self.activation_symmetric = cfg.activation_symmetric
        # Roles quantized group-wise, the rest per token; see asrq.quantizers.activation. Set
        # from the top-level activation_groupwise_roles, which evaluation also uses.
        self.activation_groupwise_roles = list(cfg.activation_groupwise_roles)
        # 0 learns a full-width R1; otherwise R1 is block-diagonal with this block width, which
        # should equal activation_group_size so each block mixes only within its own group.
        self.r1_block_size = cfg.r1_block_size or None
        # Randomized online Hadamard (random signs before H) for every fc2. A plain Hadamard turns
        # the mostly non-negative GELU output feeding fc2 into one spike per block.
        self.hadamard_random_signs = cfg.hadamard_random_signs
        # learn_r2: False trains R1 only, with no head-wise R2 on the attention output projection.
        # fc2_online_hadamard: False drops fc2's online Hadamard. With both off, attn_out and fc2
        # inputs stay unrotated and should be quantized group-wise (activation_groupwise_roles).
        self.learn_r2 = cfg.learn_r2
        self.fc2_online_hadamard = cfg.fc2_online_hadamard
        # Conformer models: quantize the inserted block-output Linears' inputs during the search,
        # as evaluation will. Copied from the model config by exp.py and rot-exp.py.
        self.quantize_block_output_linear = bool(cfg.get("quantize_block_output_linear", False))
        self.search = cfg.get("search", "cayley")
        if self.search not in ("cayley", "evolution"):
            raise ValueError(f"transform.search must be 'cayley' or 'evolution', got {self.search!r}")
        if self.search == "evolution" and self.objective != "kl":
            raise ValueError("transform.search=evolution minimises the KL divergence; set transform.objective=kl")
        self.evolution = cfg.get("evolution", None)
        self.weight_only = bool(cfg.get("weight_only", False))
        self.wbits = cfg.get("wbits", 4)
        self.wgroup = cfg.get("wgroup", None)
        self.wsymmetric = cfg.get("wsymmetric", True)
        self.weight_only_quantizer = cfg.get("weight_only_quantizer", None) or (
            cfg.get("wmethod") if cfg.get("wmethod") in ("rtn", "gptq") else "rtn"
        )
        if self.weight_only_quantizer not in ("rtn", "gptq"):
            raise ValueError(f"transform.weight_only_quantizer must be rtn, gptq or null, got {self.weight_only_quantizer!r}")
        self.wpercdamp = cfg.get("wpercdamp", 0.01)
        self.wblock_size = cfg.get("wblock_size", 128)
        if self.weight_only and self.search != "evolution":
            raise ValueError("transform.weight_only searches R1 with transform.search=evolution")
        if self.weight_only:
            self.abits = None
        if self.abits is not None:
            check_hadamard_matches_groups(
                self.hadamard_block_size, self.activation_group_size,
                self.activation_groupwise_roles, self.fc2_online_hadamard,
            )


@register_transform(TransformNames.rotation)
class RotationTransform(BaseTransform):
    """QuaRot/SpinQuant-style rotation, learned over the Stiefel manifold or searched over Hadamard signs.

    obtain_transform searches R1 and R2 against the configured objective (``search``) and saves
    them; apply_transform folds a saved pair into the weights. Both begin by converting every
    LayerNorm to a scale-free, shift-free normalization, which is what makes the residual
    stream commute with R1 -- so a model that has had the rotation applied has also had its
    norms rewritten, and the two cannot be separated.
    """

    cfg: RotationTransformConfig

    def __init__(self, cfg: RotationTransformConfig) -> None:
        super().__init__(cfg)
        self.hook_handles = []

    def _rotation_inputs(self, modelQ) -> dict:
        if self.cfg.model_name not in _ROTATION_INPUTS_FNS:
            raise ValueError(
                f"Rotation learning not implemented for model {self.cfg.model_name}"
            )
        return _ROTATION_INPUTS_FNS[self.cfg.model_name](
            modelQ,
            num_samples=self.cfg.num_samples,
            batch_size=self.cfg.batch_size,
            quantize_block_output_linear=self.cfg.quantize_block_output_linear,
        )

    def _rotation_layers(self, modelQ) -> dict:
        if self.cfg.model_name not in _ROTATION_LAYERS_FNS:
            raise ValueError(
                f"Rotation application not implemented for model {self.cfg.model_name}"
            )
        return _ROTATION_LAYERS_FNS[self.cfg.model_name](modelQ.model)

    def obtain_transform(self, modelQ) -> None:
        if not self.cfg.learn_rotation:
            return
        learn_rotations(
            modelQ.model,
            self.cfg.path,
            epochs=self.cfg.epochs,
            learning_rate=self.cfg.learning_rate,
            hadamard_block_size=self.cfg.hadamard_block_size,
            activation_bits=self.cfg.abits,
            groupwise_roles=self.cfg.activation_groupwise_roles,
            r1_block_size=self.cfg.r1_block_size,
            hadamard_random_signs=self.cfg.hadamard_random_signs,
            learn_r2=self.cfg.learn_r2,
            fc2_online_hadamard=self.cfg.fc2_online_hadamard,
            objective=self.cfg.objective,
            activation_group_size=self.cfg.activation_group_size,
            activation_symmetric=self.cfg.activation_symmetric,
            search=self.cfg.search,
            weight_quantization=self._weight_quantization(modelQ),
            evolution=EvolutionConfig.from_mapping(
                OmegaConf.to_container(self.cfg.evolution, resolve=True) if self.cfg.evolution is not None else None
            ),
            **self._rotation_inputs(modelQ),
        )

    def _weight_quantization(self, modelQ):
        """The weight-only search's quantization: RTN or GPTQ (transform.weight_only_quantizer, by default the
        experiment's quantizer) with the quantizer's bits, group size, symmetry, damping and block size, on the
        rotated layers the quantizer quantizes."""
        if not self.cfg.weight_only:
            return None
        model = modelQ.model
        rotated = [name for group in self._rotation_layers(modelQ)["layers_to_rotate"] for name, _ in group[2]]
        return {
            "method": self.cfg.weight_only_quantizer,
            "percdamp": self.cfg.wpercdamp,
            "block_size": self.cfg.wblock_size,
            "bits": self.cfg.wbits,
            "group_size": self.cfg.wgroup,
            "symmetric": self.cfg.wsymmetric,
            "layers": [name for name in rotated if modelQ.should_quantize_module(name, model.get_submodule(name))],
        }

    def apply_transform(self, modelQ) -> None:
        """Fold the learned rotation in, and check the transcription is unchanged.

        The rotation is an exact reparameterisation, so the decoded text has to be identical:
        it is only a preconditioner for the quantizer that runs afterwards. Comparing the
        transcription rather than the logits catches the same class of bug and matches how the
        other transforms verify themselves.
        """
        if not os.path.isfile(self.cfg.path):
            raise FileNotFoundError(
                f"no learned rotation at {self.cfg.path}. Learn one with asrq/rot-exp.py (which "
                f"writes to transform.path), or point transform.path at an existing checkpoint."
            )
        original_text = self.transcribe(modelQ)
        self.hook_handles = apply_rotations(
            modelQ.model,
            self.cfg.path,
            hadamard_block_size=self.cfg.hadamard_block_size,
            **self._rotation_layers(modelQ),
        )
        text_after_rotation = self.transcribe(modelQ)
        assert original_text == text_after_rotation, (
            f"Model output changed after rotation:\n  before: {original_text!r}\n"
            f"  after:  {text_after_rotation!r}"
        )

    def transcribe(self, modelQ) -> str:
        """Transcribe the reference clip BaseTransform holds, for the invariance check."""
        if self.cfg.model_name == ModelNames.OPENAI_WHISPER_LARGE_V3:
            return modelQ.transcribe(self.audio, modelQ.model, modelQ.processor, self.sr)
        return modelQ.batch_transcribe([self.audio], modelQ.model, modelQ.model.device)[0]
