# pyright: reportMissingImports=false

from omegaconf import DictConfig
from asrq.core.registry import TransformNames, register_transform, register_transform_config, ModelNames
from asrq.core.types import Processor
from asrq.transforms.base import BaseTransform, TransformConfig
from asrq.transforms.scaling.canary_qwen_utils import get_canary_qwen_layers_to_scale
from asrq.transforms.scaling.parakeet_ctc_utils import get_parakeet_ctc_layers_to_scale
from asrq.transforms.scaling.whisper_utils import get_whisper_layers_to_scale
from asrq.quantizers.base import is_pointwise_conv1d
import torch
import torch.nn as nn
import numpy as np


def channels_last(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Put the contracted axis last: Conv1d activations are ``(B, C, T)``, linears ``(B, T, C)``."""
    return x.transpose(1, 2) if isinstance(module, nn.Conv1d) else x


def weight_2d(module: nn.Module) -> torch.Tensor:
    """``(out_features, in_features)`` view of a linear or pointwise-conv weight."""
    w = module.weight.data # type: ignore
    return w.reshape(w.shape[0], -1) if isinstance(module, nn.Conv1d) else w


class ScaledInputConv1d(nn.Conv1d):
    """A pointwise Conv1d that divides its input channels by a per-channel scale.

    The Conv1d counterpart of :class:`ScaledInputLinear` - needed because the Conformer's
    ``conv.pointwise_conv2`` is fed by an activation, exactly like an MLP down-projection,
    but its activations are ``(B, C, T)`` so the scale broadcasts over the channel axis.
    """

    @classmethod
    def from_conv(cls, conv: nn.Conv1d, scale: torch.Tensor) -> "ScaledInputConv1d":
        module = cls(
            conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, # type: ignore
            stride=conv.stride, padding=conv.padding, dilation=conv.dilation, # type: ignore
            groups=conv.groups, bias=conv.bias is not None, device="meta",
        )
        module.weight = conv.weight
        module.bias = conv.bias
        module.register_buffer("scale", scale.detach().clone().to(device=conv.weight.device))
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T): the scale indexes channels, so it broadcasts over time.
        return super().forward(x / self.scale.to(dtype=x.dtype, device=x.device).view(1, -1, 1))

    def extra_repr(self) -> str:
        return super().extra_repr() + f", input_scale={tuple(self.scale.shape)}"


class ScaledInputLinear(nn.Linear):
    """A linear layer that divides its input by a per-input-channel scale.

    Smoothing moves a scale ``s`` out of the activations and into the weights: ``W`` is
    multiplied by ``s`` and whatever produced those activations is divided by it. A
    down-projection has no producer that can absorb the division - it is fed by an
    activation, or in Qwen3 by ``act_fn(gate_proj(x)) * up_proj(x)`` - so the division
    lives in this layer instead::

        F.linear(x / s, W * s, b) == F.linear(x, W, b)

    Keeping it here rather than in the producer means the smoothed tensor is exactly the
    one the activation quantizer sees, and it holds regardless of how the input was
    produced - which the gated Qwen3 MLP makes awkward to do upstream.

    Subclasses ``nn.Linear``, so anything that dispatches on the layer type (scale
    collection, activation quantization, weight quantization) keeps working.
    """

    @classmethod
    def from_linear(cls, linear: nn.Linear, scale: torch.Tensor) -> "ScaledInputLinear":
        # Built on the meta device so no weights are allocated: the real Parameters are
        # adopted from `linear` rather than copied.
        module = cls(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None, device="meta",
        )
        module.weight = linear.weight
        module.bias = linear.bias
        module.register_buffer("scale", scale.detach().clone().to(device=linear.weight.device))
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x / self.scale.to(dtype=x.dtype, device=x.device))

    def extra_repr(self) -> str:
        return super().extra_repr() + f", input_scale={tuple(self.scale.shape)}"


def _replace_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Swap the submodule at the dotted path ``name`` for ``new_module``."""
    parent_name, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, leaf, new_module)


def scale_linear_input(model: nn.Module, moduledict: dict, layer_name: str, scale: torch.Tensor) -> None:
    """Make the layer at ``layer_name`` divide its own input by ``scale``.

    Handles both nn.Linear and pointwise Conv1d. Composes if the layer was already
    converted by an earlier entry, so a layer is never wrapped twice.
    """
    module = moduledict[layer_name]
    if isinstance(module, (ScaledInputLinear, ScaledInputConv1d)):
        module.scale = module.scale * scale.to(module.scale)
        return
    if isinstance(module, nn.Conv1d):
        if tuple(module.kernel_size) != (1,) or module.groups != 1:
            raise ValueError(
                f"'{layer_name}' must be a pointwise, ungrouped Conv1d to scale its input; "
                f"got kernel_size={tuple(module.kernel_size)}, groups={module.groups}"
            )
        scaled = ScaledInputConv1d.from_conv(module, scale)
    else:
        scaled = ScaledInputLinear.from_linear(module, scale)
    _replace_module(model, layer_name, scaled)
    moduledict[layer_name] = scaled


@register_transform_config(TransformNames.scaling)
class ScalingTransformConfig(TransformConfig):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.obtain_scales = cfg.obtain_scales
        self.type = cfg.type
        self.wbits = cfg.wbits
        self.abits = cfg.abits


def obtain_scales(modelQ, layers_to_scale, scale_path, forward_fn, head_dim, type="smoothquant", wbit=4, abit=8):
    # layers to scale 
    model = modelQ.model
    processor = modelQ.processor

    # Compute scales
    scales = {}
    num_samples_passed = {}
    moduledict = dict(model.named_modules())
    def create_hook(name, type=type):
        def hook(module, input, output):
            input = input[0]  # Get the input tensor from the tuple
            # Conv1d activations are (B, C, T); bring channels last before flattening,
            # or the statistics would be gathered over time instead of over channels.
            input = channels_last(module, input)
            input = input.reshape(-1, input.shape[-1])  # Flatten the input to (num_samples, num_channels)
            mask = (input.abs().sum(dim=1) != 0)
            input = input[mask]
            if "linear_out" in name or "o_proj" in name or "out_proj" in name:
                input = input.reshape(-1, head_dim)
            # mask out the tokens with all zeros (padding tokens)
            if type == "awq":
                tmp = input.shape[0]
                mean = scales.get(name, 0)
                num_samples = num_samples_passed.get(name, 0)
                mean = mean * (num_samples / (num_samples + tmp)) + input.abs().sum(dim=0) / (num_samples + tmp)
                scales[name] = mean
                num_samples_passed[name] = num_samples + tmp
            elif type == "smoothquant":
                max_val = input.abs().max(dim=0).values
                prev_max_val = scales.get(name, torch.zeros_like(max_val))
                scales[name] = torch.maximum(prev_max_val, max_val)
            else:
                raise ValueError(f"Unsupported scaling type: {type}")
        return hook

    hooks = []
    for layer_names, _ in layers_to_scale:
        # layer_name = layer_names[0]
        for layer_name in layer_names:
            module = moduledict[layer_name]
            assert module is not None, f"Module {layer_name} not found in model"
            assert isinstance(module, torch.nn.Linear) or is_pointwise_conv1d(module), \
                f"Module {layer_name} is neither a linear layer nor a pointwise Conv1d"
            hooks.append(module.register_forward_hook(create_hook(layer_name)))

    with torch.no_grad():
        for x, text in modelQ.calibration_samples:
            # modelQ.batch_forward([x], [text], model, model.device) # type: ignore
            forward_fn(x, text, modelQ)

    for hook in hooks:
        hook.remove()

    # Now we want to obtain the best scales.
    # We want the scales that lead to the best 
    # reconstruction error after quantization.
    # Here we use the output of a layer before 
    # and after quantization to compute the 
    # reconstruction error, and we want to find 
    # the scales that minimize the reconstruction error.
    # ---
    # First we obtain input into the first layer 
    selected_alpha = {}
    alphas = torch.linspace(0, 1, 10)
    losses = {}
    for alpha in alphas:
        running_losses = {}
        num_samples_passed = {}
        def creat_hook2(name):
            def hook(module, input, output):
                # Both sides go channels-last so the reconstruction error below is
                # computed in the same layout for a Conv1d as for a linear.
                input = channels_last(module, input[0])
                output = channels_last(module, output)
                input = input.reshape(-1, input.shape[-1])
                output = output.reshape(-1, output.shape[-1])
                mask = (input.abs().sum(dim=-1) != 0)
                input = input[mask]
                output = output[mask]
                W = weight_2d(module).clone()
                org_weight_shape = W.shape
                org_input_shape = input.shape
                if "linear_out" in name or "o_proj" in name or "out_proj" in name:
                    input = input.reshape(-1, head_dim)
                    W = W.reshape(-1, head_dim)
                # obtain scales for the alpha
                w_max = W.abs().max(dim=0).values
                if type=="smoothquant":
                    scale = (scales[name]**alpha) / (w_max**(1-alpha) + 1e-7)
                elif type=="awq":
                    scale = scales[name] ** alpha
                else:
                    raise ValueError(f"Unsupported scaling type: {type}")
                input_scaled = input / scale
                W_scaled = W * scale.unsqueeze(0)
                W_scaled = W_scaled.reshape(W.shape)
                input_scaled = input_scaled.reshape(input.shape)
                # quantize input
                input = input.reshape(org_input_shape)
                if abit < 16:
                    amaxq = 2 ** (abit - 1) - 1
                    qscale = input_scaled.abs().max(dim=-1).values / amaxq
                    qinput = (input_scaled / qscale.unsqueeze(-1)).round().clamp(-amaxq, amaxq)
                    dqinput = qinput * qscale.unsqueeze(-1)
                else:
                    dqinput = input_scaled
                # Quantize weights
                wmaxq = 2 ** (wbit - 1) - 1
                qscale = W_scaled.abs().max(dim=1).values / wmaxq
                qweight = (W_scaled / qscale.unsqueeze(-1)).round().clamp(-wmaxq, wmaxq)
                dqweight = qweight * qscale.unsqueeze(-1)
                # compute output with quantized weights and inputs
                dqinput = dqinput.reshape(org_input_shape)
                # The 2-D view, not module.weight.shape: a pointwise conv's weight is
                # (out, in, 1) and cannot be transposed for the matmul below.
                dqweight = dqweight.reshape(org_weight_shape)
                output_quant = torch.matmul(dqinput, dqweight.t())
                try:
                    output_quant = output_quant.reshape(output.shape)
                except:
                    breakpoint()
                if module.bias is not None:
                    output_quant += module.bias
                # compute the mse loss between output and output_quant
                mse_loss = torch.mean((output - output_quant) ** 2)
                tmp = output.numel()
                mean_loss = running_losses.get(name, 0)
                prev_num_samples = num_samples_passed.get(name, 0)
                mean_loss = mean_loss * (prev_num_samples / (prev_num_samples + tmp)) + mse_loss.item() * (tmp / (prev_num_samples + tmp))
                running_losses[name] = mean_loss
                num_samples_passed[name] = prev_num_samples + tmp

            return hook

        # calculate the mse loss for each layer
        hooks = []
        for layer_names, _ in layers_to_scale:
            # layer_name = layer_names[0]
            for layer_name in layer_names:
                module = moduledict[layer_name]
                hooks.append(module.register_forward_hook(creat_hook2(layer_name)))

        with torch.no_grad():
            for i, (x, text) in enumerate(modelQ.calibration_samples[:128]):
                # modelQ.batch_forward([x], [text], model, model.device) # type: ignore
                forward_fn(x, text, modelQ)
        for hook in hooks:
            hook.remove()
        for layer_names, _ in layers_to_scale:
            for layer_name in layer_names:
                if running_losses[layer_name] < losses.get(layer_name, float("inf")):
                    losses[layer_name] = running_losses[layer_name]
                    selected_alpha[layer_name] = alpha
        
        print(f"Processed layers for alpha {alpha:.2f}")

    # Since alpha has been determined, we can now compute the final scales using the selected alpha.
    final_scales = {}
    for layer_names, _ in layers_to_scale:
        layer_name = layer_names[0]
        if len(layer_names) > 1:
            tmp_alphas = [selected_alpha[layer_name] for layer_name in layer_names]
            alpha = np.mean(tmp_alphas).item()
            wmax = torch.cat([weight_2d(moduledict[layer_name]).abs().max(dim=0).values.unsqueeze(0) for layer_name in layer_names], dim=0).max(dim=0).values
        else:
            wmax = weight_2d(moduledict[layer_name]).abs().max(dim=0).values
            alpha = selected_alpha[layer_name]
        
        if type=="smoothquant":
            if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
                wmax = wmax.reshape(-1, head_dim).max(dim=0).values
            amax = scales[layer_name]
            final_scales[layer_name] = (amax**alpha) / (wmax**(1-alpha) + 1e-7) # type: ignore
        elif type=="awq":
            scale = scales[layer_name] ** alpha
            final_scales[layer_name] = scale.where(scale != 0, 1e-3) # type: ignore
        else:
            raise ValueError(f"Unsupported scaling type: {type}")

    print(selected_alpha)
    print(f"Final scales computed for all layers, now saving to disk at {scale_path}")

    # now save the scales to disk
    torch.save(final_scales, scale_path)


def scale_model(modelQ, audio, layers_to_scale, head_dim, scale_path):
    model = modelQ.model
    processor = modelQ.processor
    device = model.device

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

        if prev_name is None:
            # Nothing upstream can absorb the reciprocal (the producer is an activation,
            # or a gated product), so the layer divides its own input instead.
            for layer_name in layer_names:
                scale_linear_input(model, moduledict, layer_name, scale)
            continue

        # scale the prev layer
        if isinstance(prev_name, str):
            prev_names = [prev_name]
        else:
            prev_names = prev_name
        for prev_name in prev_names:
            prev_module = moduledict[prev_name]
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
                raise ValueError(
                    f"Cannot fold the reciprocal scale into '{prev_name}' "
                    f"({type(prev_module).__name__}): it has no weight. Use prev=None for "
                    f"this entry so the target layer scales its own input instead."
                )
    
            
@register_transform(TransformNames.scaling)
class ScalingTransform(BaseTransform):
    cfg: ScalingTransformConfig
    def __init__(self, transform_cfg: ScalingTransformConfig) -> None:
        super().__init__(transform_cfg)

    def obtain_transform(self, modelQ) -> None:
        if self.cfg.obtain_scales is False:
            return
        layers_to_scale, _, forward_fn, head_dim  = self.prepare_for_transform(modelQ)
        obtain_scales(modelQ, layers_to_scale, self.cfg.path, forward_fn, head_dim, self.cfg.type, self.cfg.wbits, self.cfg.abits)

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
            transcribe_fn = lambda : modelQ.batch_transcribe([self.audio], modelQ.model, modelQ.model.device)[0]
            forward_fn = lambda x, text, modelQ: modelQ.batch_forward([x], [text], modelQ.model, modelQ.model.device)[0]
            head_dim = 128
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
        
