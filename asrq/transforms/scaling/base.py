# pyright: reportMissingImports=false

from omegaconf import DictConfig
from asrq.core.registry import TransformNames, register_transform, register_transform_config, ModelNames
from asrq.core.types import Processor
from asrq.transforms.base import BaseTransform, TransformConfig
from asrq.transforms.rotation.canary_qwen_utils import canary_qwen_logits_fn
from asrq.transforms.scaling.canary_qwen_utils import get_canary_qwen_layers_to_scale
from asrq.transforms.scaling.parakeet_ctc_utils import get_parakeet_ctc_layers_to_scale
from asrq.transforms.scaling.whisper_utils import get_whisper_layers_to_scale
import torch
import torch.nn as nn


def channels_last(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Put the contracted axis last: Conv1d activations are ``(B, C, T)``, linears ``(B, T, C)``."""
    return x.transpose(1, 2) if isinstance(module, nn.Conv1d) else x


def weight_2d(module: nn.Module) -> torch.Tensor:
    """``(out_features, in_features)`` view of a linear or pointwise-conv weight."""
    w = module.weight.data # type: ignore
    return w.reshape(w.shape[0], -1) if isinstance(module, nn.Conv1d) else w


class InputScale(nn.Module):
    """Divide the feature axis by a per-channel scale: ``x <- x / s``.

    The scaling counterpart of the rotation's OnlineHadamard, and placed the same way: after the
    activation that feeds a layer, as ``nn.Sequential(activation, InputScale)``. Smoothing multiplies
    the layer's weight by ``s`` over its input channels, and the reciprocal has to be taken out of the
    input,

        (x / s) @ (W * s).T + b = x @ W.T + b

    which is normally folded into the producing layer. An fc2-like layer is fed by a nonlinearity,
    which cannot absorb it, so this module applies it. Being a module in front of the layer rather than
    part of it, everything observing the layer's input -- the activation quantizer's pre-hook, the
    Hessian a GPTQ-style quantizer collects, a humming ASRQLinear replacing the layer -- sees
    ``x / s``, the tensor the scaled weight multiplies.

    Args:
        scale: ``(channels,)`` positive scale.
        channels_first: True for ``(batch, channels, time)`` inputs, as a pointwise Conv1d reads.
    """

    def __init__(self, scale: torch.Tensor, channels_first: bool = False):
        super().__init__()
        self.channels_first = channels_first
        self.register_buffer("scale", scale.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(dtype=x.dtype, device=x.device)
        return x / (scale.view(1, -1, 1) if self.channels_first else scale)

    def extra_repr(self) -> str:
        return f"channels={self.scale.numel()}, channels_first={self.channels_first}"


def insert_input_scale_after_activation(model: nn.Module, activation_name: str, layer: nn.Module, scale: torch.Tensor) -> None:
    """Wrap an activation as ``nn.Sequential(activation, InputScale(scale))``, in place.

    Only dropout, the identity in eval mode, sits between the activation and the layer it feeds in the
    supported models. The activation is wrapped in its own parent, by attribute path, because NeMo shares
    one Swish instance between all feed-forwards; wrapping one parent's reference leaves the others
    alone. An activation already ending in an InputScale has its scale multiplied instead.

    Args:
        model: The model to modify in place.
        activation_name: Dotted name of the activation feeding ``layer``.
        layer: The layer whose weight absorbed ``scale``; a Conv1d reads ``(batch, channels, time)``.
        scale: ``(channels,)`` scale to divide the activation's output by.
    """
    parent_name, _, attribute = activation_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    activation = getattr(parent, attribute)
    if isinstance(activation, nn.Sequential) and isinstance(activation[-1], InputScale):
        activation[-1].scale.mul_(scale.to(activation[-1].scale))
        return
    input_scale = InputScale(scale, channels_first=isinstance(layer, nn.Conv1d)).to(scale.device)
    setattr(parent, attribute, nn.Sequential(activation, input_scale))


@register_transform_config(TransformNames.scaling)
class ScalingTransformConfig(TransformConfig):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.obtain_scales = cfg.obtain_scales
        self.type = cfg.type
        self.wbits = cfg.wbits
        self.abits = cfg.abits
        self.wgroup = cfg.get("wgroup", None)
        self.activation_group_size = cfg.get("activation_group_size", -1)
        self.activation_symmetric = cfg.get("activation_symmetric", True)
        self.activation_groupwise_roles = list(cfg.get("activation_groupwise_roles", []) or [])
        # The migration strength, one for every entry, as SmoothQuant uses.
        self.alpha = float(cfg.get("alpha", 0.5))
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"transform.alpha must be between 0 and 1, got {self.alpha}")


ATTENTION_OUTPUTS = ("linear_out", "o_proj", "out_proj")
# The smallest scale and weight maximum obtain_scales allows, as SmoothQuant's reference code clamps them.
MIN_SCALE = 1e-5


def _is_attention_output(name: str) -> bool:
    return any(key in name for key in ATTENTION_OUTPUTS)


def _input_width_scale(scale: torch.Tensor, in_features: int) -> torch.Tensor:
    """A head-wise scale (``head_dim`` channels, shared by every head) tiled over a layer's input."""
    return scale.repeat(in_features // scale.numel())


def obtain_scales(modelQ, layers_to_scale, scale_path, forward_fn, head_dim, type="smoothquant", alpha=0.5):
    """Compute one smoothing scale per entry of ``layers_to_scale`` and save ``{entry's first layer: scale}``.

    Over every calibration sample, the per-channel maximum (smoothquant) or mean (awq) of each layer's absolute
    input is collected. An attention output projection's input is folded to ``head_dim`` channels, since its
    scale is shared by every head (the reciprocal is folded into v_proj's rows, head by head). Each entry then
    gets, with one migration strength ``alpha`` for every entry as SmoothQuant uses,

        smoothquant:  s = amax ** alpha / wmax ** (1 - alpha)
        awq:          s = amean ** alpha

    with ``wmax`` the per-channel weight maximum over all of the entry's layers, so layers sharing one input
    (q, k, v) share one scale. The scale depends on neither the weight nor the activation bits.

    Both the weight maxima and the scales are clamped below at ``MIN_SCALE``, as SmoothQuant's reference code
    does. A channel nearly silent on the calibration set would otherwise get a scale around 1e-8; evaluation
    runs in float16, where ``w * s`` then underflows to zero while ``x / s`` overflows as soon as that channel
    is active, and the model outputs NaNs even with nothing quantized.

    Args:
        modelQ: The model wrapper, whose calibration_samples are run.
        layers_to_scale: ``[(layer_names, prev), ...]``, as scale_model reads them.
        scale_path: Where the scales are saved.
        forward_fn: ``forward_fn(audio, text, modelQ)`` runs one calibration sample.
        head_dim: Attention head width.
        type: ``"smoothquant"`` or ``"awq"``.
        alpha: The migration strength, between 0 (all scale on the weights) and 1 (all on the activations).
    """
    if type not in ("smoothquant", "awq"):
        raise ValueError(f"Unsupported scaling type: {type}")
    model = modelQ.model
    moduledict = dict(model.named_modules())

    statistics = {}
    counts = {}

    def statistics_hook(name):
        def hook(module, inputs, output):
            x = channels_last(module, inputs[0].detach().float())
            x = x.reshape(-1, x.shape[-1])
            x = x[x.abs().sum(dim=1) != 0].abs()
            if _is_attention_output(name):
                x = x.reshape(-1, head_dim)
            if type == "smoothquant":
                statistics[name] = torch.maximum(statistics.get(name, torch.zeros_like(x[0])), x.amax(dim=0))
            else:
                statistics[name] = statistics.get(name, 0) + x.sum(dim=0)
                counts[name] = counts.get(name, 0) + x.shape[0]
        return hook

    def weight_max(names):
        maxima = []
        for name in names:
            w = weight_2d(moduledict[name]).abs().float().amax(dim=0)
            maxima.append(w.reshape(-1, head_dim).amax(dim=0) if _is_attention_output(name) else w)
        return torch.stack(maxima).amax(dim=0).clamp(min=MIN_SCALE)

    names = [name for layer_names, _ in layers_to_scale for name in layer_names]
    handles = [moduledict[name].register_forward_hook(statistics_hook(name)) for name in names]
    try:
        with torch.no_grad():
            for x, text in modelQ.calibration_samples:
                forward_fn(x, text, modelQ)
    finally:
        for handle in handles:
            handle.remove()
    if type == "awq":
        statistics = {name: total / counts[name] for name, total in statistics.items()}

    final_scales = {}
    for layer_names, _ in layers_to_scale:
        activation = statistics[layer_names[0]]
        if type == "smoothquant":
            scale = activation ** alpha / (weight_max(layer_names) ** (1 - alpha) + 1e-7)
        else:
            scale = activation ** alpha
        final_scales[layer_names[0]] = scale.clamp(min=MIN_SCALE).cpu()
    print(f"alpha={alpha:.2f} for all {len(final_scales)} entries; saving the scales to {scale_path}")
    torch.save(final_scales, scale_path)


def scale_model(modelQ, audio, layers_to_scale, head_dim, scale_path):
    """Move each entry's scale from its layers' input into their weights, in place.

    Each entry is ``(layer_names, prev)``: the layers' weights are multiplied by the scale over their
    input channels and the reciprocal is folded into ``prev`` -- the output rows of a linear or pointwise
    conv, a norm's weight and bias -- or, when ``prev`` is an activation with no weight to absorb it,
    applied by an InputScale inserted after that activation.
    """
    model = modelQ.model
    processor = modelQ.processor
    device = next(model.parameters()).device

    # layers_to_scale = get_canary_qwen_layers_to_scale(model)
    scales = torch.load(scale_path)
    moduledict = dict(model.named_modules())
    for layer_names, prev_name in layers_to_scale:
        layer_name = layer_names[0]
        scale = scales[layer_name].to(device)
        scale = scale.where(scale != 0, 1e-3) # avoid scaling by 0
        for layer_name in layer_names:
            module = moduledict[layer_name]
            if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
                w = module.weight.data.clone()
                w = w.reshape(-1, head_dim) * scale.unsqueeze(0)
                module.weight.data.copy_(w.reshape_as(module.weight.data))
            elif isinstance(module, torch.nn.Conv1d):
                # (out, in, k): the scale indexes input channels, i.e. dim 1.
                module.weight.data *= scale.view(1, -1, 1)
            else:
                module.weight.data *= scale.unsqueeze(0)

        # scale the prev layer
        if isinstance(prev_name, str):
            prev_names = [prev_name]
        else:
            prev_names = prev_name
        for prev_name in prev_names:
            prev_module = model.get_submodule(prev_name)
            # for linear layers
            if isinstance(prev_module, torch.nn.Conv1d):
                # A pointwise conv feeding another layer: its output channels are the
                # consumer's input channels, i.e. dim 0 of (out, in, k).
                prev_module.weight.data /= scale.view(-1, 1, 1)
                if prev_module.bias is not None:
                    prev_module.bias.data /= scale
            elif isinstance(prev_module, torch.nn.Linear):
                if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
                    # prev_module is v_proj: shape (num_heads * head_dim, embed_dim).
                    # Row i must be divided by scale[i % head_dim], so tile the scale.
                    num_repeats = prev_module.weight.shape[0] // scale.shape[0]
                    scale_expanded = scale.repeat(num_repeats)  # (num_heads * head_dim,)
                    prev_module.weight.data /= scale_expanded.unsqueeze(1)
                else:
                    prev_module.weight.data /= scale.unsqueeze(1)
                if prev_module.bias is not None:
                    if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
                        num_repeats = prev_module.bias.shape[0] // scale.shape[0]
                        scale_expanded = scale.repeat(num_repeats)
                        prev_module.bias.data /= scale_expanded
                    else:
                        prev_module.bias.data /= scale
            elif getattr(prev_module, "weight", None) is not None:
                # It has to be a normalization layer
                # We scale the weight
                prev_module.weight.data /= scale
                if hasattr(prev_module, "bias") and prev_module.bias is not None:
                    prev_module.bias.data /= scale
            else:
                insert_input_scale_after_activation(model, prev_name, moduledict[layer_names[0]], scale)


@register_transform(TransformNames.scaling)
class ScalingTransform(BaseTransform):
    cfg: ScalingTransformConfig
    def __init__(self, transform_cfg: ScalingTransformConfig) -> None:
        super().__init__(transform_cfg)

    def obtain_transform(self, modelQ) -> None:
        if self.cfg.obtain_scales is False:
            return
        layers_to_scale, _, forward_fn, head_dim  = self.prepare_for_transform(modelQ)
        obtain_scales(modelQ, layers_to_scale, self.cfg.path, forward_fn, head_dim, self.cfg.type, self.cfg.alpha)

    def prepare_for_transform(self, modelQ):
        if self.cfg.model_name == ModelNames.OPENAI_WHISPER_LARGE_V3:
            layers_to_scale = get_whisper_layers_to_scale(modelQ.model)
            transcribe_fn = lambda : modelQ.transcribe(self.audio, modelQ.model, modelQ.processor)
            forward_fn = lambda x, text, modelQ: modelQ.forward(x, text, modelQ.model, modelQ.processor, 16000)
            head_dim = modelQ.model.model.encoder.layers[0].self_attn.head_dim
        elif self.cfg.model_name == ModelNames.NVIDIA_PARAKEET_CTC_1_1B:
            layers_to_scale = get_parakeet_ctc_layers_to_scale(modelQ.model)
            transcribe_fn = lambda : modelQ.batch_transcribe([self.audio], modelQ.model, modelQ.model.device)[0]
            forward_fn = lambda x, text, modelQ: modelQ.batch_forward([x], modelQ.model, modelQ.model.device)[0]
            head_dim = modelQ.model.encoder.layers[0].self_attn.d_k
        elif self.cfg.model_name == ModelNames.NVIDIA_CANARY_QWEN_2_5B:
            layers_to_scale = get_canary_qwen_layers_to_scale(modelQ.model)
            transcribe_fn = lambda : modelQ.batch_transcribe([self.audio], modelQ.model, modelQ.device)[0]
            forward_fn = lambda x, text, modelQ: canary_qwen_logits_fn(modelQ.model, modelQ.calibration_batch(x, text))
            head_dim = modelQ.model.llm.config.head_dim
        else:
            raise ValueError(f"Unsupported model name: {self.cfg.model_name}")
        return layers_to_scale, transcribe_fn, forward_fn, head_dim 

    def apply_transform(self, modelQ) -> None:
        """Apply the scaling transformation to the given model."""
        layers_to_scale, transcribe_fn, _, head_dim = self.prepare_for_transform(modelQ)
        original_text = transcribe_fn()
        scale_model(modelQ, self.audio, layers_to_scale, head_dim, self.cfg.path)
        # scale_whisper_model(modelQ, self.audio, self.sr, self.cfg.path, alpha=0.5, device="cuda")
        text_after_scaling = transcribe_fn()
        assert original_text == text_after_scaling, "Model output changed after scaling"
        
