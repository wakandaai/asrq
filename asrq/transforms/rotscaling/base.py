# pyright: reportMissingImports=false

from typing import Dict

import numpy as np
from omegaconf import DictConfig
from asrq.core.registry import TransformNames, register_transform, register_transform_config
from asrq.transforms.rotation.hadamard_utils import random_hadamard_matrix
from asrq.transforms.scaling.base import ScalingTransform, ScalingTransformConfig, scale_model
import torch


@register_transform_config(TransformNames.rotscaling)
class RotScalingTransformConfig(ScalingTransformConfig):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)


def obtain_scales(modelQ, layers_to_scale, scale_path, forward_fn, head_dim, type="mean", wbit=4, abit=8):
    # layers to scale 
    model = modelQ.model
    processor = modelQ.processor
    moduledict = dict(model.named_modules())
    device = model.device

    # We temporary modify the model with rotations and scales
    model_state_dict = model.state_dict()
    # a copy of the state dict has to be created to avoid in-place modifications
    model_state_dict = {k: v.clone() for k, v in model_state_dict.items()}

    # region Phase 1: Rotation
    # Find the best Hadamard rotation H for out projection layers.
    hadamard_rotations = {}
    hadamard_losses = {}
    for j in range(10):
        running_losses = {}
        num_samples_passed = {}
        Hs = {}

        for layer_names, _ in layers_to_scale:
            layer_name = layer_names[0]
            if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
                H = random_hadamard_matrix(head_dim, modelQ.model.device)
                Hs[layer_name] = H

        def create_hook(name):
            def hook(module, input, output):
                H = Hs.get(name, None)
                assert H is not None, f"Hadamard matrix not found for layer {name}"
                weight = module.weight.data.clone()
                input = input[0].reshape(-1, input[0].shape[-1])
                output = output.reshape(-1, output.shape[-1])
                mask = (input.abs().sum(dim=-1) != 0)
                input = input[mask]
                output = output[mask]
                input = input.reshape(-1, head_dim)
                W = weight.reshape(-1, head_dim)
                input = (input.double() @ H).to(input.dtype)
                W = (W.double() @ H).to(W.dtype)
                # reshape back
                input = input.reshape(-1, weight.shape[-1])
                W = W.reshape(weight.shape)
                # quantize input and weight
                if abit < 16:
                    amaxq = 2 ** (abit - 1) - 1
                    qscale = input.abs().max(dim=-1).values / amaxq
                    qinput = (input / qscale.unsqueeze(-1)).round().clamp(-amaxq, amaxq)
                    dqinput = qinput * qscale.unsqueeze(-1)
                else:
                    dqinput = input
                wmaxq = 2 ** (wbit - 1) - 1
                qscale = W.abs().max(dim=1).values / wmaxq
                qweight = (W / qscale.unsqueeze(-1)).round().clamp(-wmaxq, wmaxq)
                dqweight = qweight * qscale.unsqueeze(-1)
                # compute output with quantized weights and inputs
                dqinput = dqinput.reshape(input.shape)
                dqweight = dqweight.reshape(weight.shape)
                output_quant = torch.matmul(dqinput, dqweight.t())
                output_quant = output_quant.reshape(output.shape)
                if module.bias is not None:
                    output_quant += module.bias
                mse_loss = torch.mean((output - output_quant) ** 2)
                tmp = output.numel()
                mean_loss = running_losses.get(name, 0)
                prev_num_samples = num_samples_passed.get(name, 0)
                mean_loss = mean_loss * (prev_num_samples / (prev_num_samples + tmp)) + mse_loss.item() * (tmp / (prev_num_samples + tmp))
                running_losses[name] = mean_loss
                num_samples_passed[name] = prev_num_samples + tmp
            return hook

        hooks = []
        for layer_names, _ in layers_to_scale:
            layer_name = layer_names[0]
            if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
                module = moduledict[layer_name]
                hooks.append(module.register_forward_hook(create_hook(layer_name)))

        with torch.no_grad():
            for i, (x, text) in enumerate(modelQ.calibration_samples[:128]):
                forward_fn(x, text, modelQ)

        for hook in hooks:
            hook.remove()

        for layer_names, _ in layers_to_scale:
            layer_name = layer_names[0]
            if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
                if running_losses[layer_name] < hadamard_losses.get(layer_name, float("inf")):
                    hadamard_rotations[layer_name] = Hs[layer_name]
                    hadamard_losses[layer_name] = running_losses[layer_name]

        print(f"Processed hadamard matrices for iteration {j}")

    for layer_names, prev_name in layers_to_scale:
        layer_name = layer_names[0]
        module = moduledict[layer_name]
        if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
            H = hadamard_rotations[layer_name]
            w = module.weight.data.clone()
            dtype = w.dtype
            org_shape = w.shape
            temp = w.double().reshape(-1, org_shape[-1] // head_dim, head_dim)
            temp = temp @ H.double()
            w_rot = temp.reshape(org_shape).to(dtype=dtype, device=device)
            module.weight.data = w_rot
            
            if isinstance(prev_name, str):
                prev_names_list = [prev_name]
            else:
                prev_names_list = list(prev_name)
            for pn in prev_names_list:
                prev_module = moduledict[pn]
                # prev_module must be a linear layer
                assert isinstance(prev_module, torch.nn.Linear), f"Predecessor module {pn} is not a linear layer"
                w = prev_module.weight.data.clone()
                dtype = w.dtype
                w_t = w.double().t()               # (in_features, num_heads*head_dim)
                org_shape = w_t.shape
                temp = w_t.reshape(-1, org_shape[-1] // head_dim, head_dim)
                temp = temp @ H.double()
                w_rot = temp.reshape(org_shape).t().to(dtype=dtype, device=device)
                prev_module.weight.data = w_rot
                if prev_module.bias is not None:
                    b = prev_module.bias.data.clone()
                    num_heads = b.shape[0] // head_dim
                    b_rot = (b.double().reshape(num_heads, head_dim) @ H.double()).reshape(-1)
                    prev_module.bias.data = b_rot.to(dtype=b.dtype)
    # endregion

    # region Phase 2: Scales
    # Compute scales for out projections
    out_proj_scales = {}
    num_samples_passed = {}
    def create_hook2(name, type=type):
        def hook(module, input, output):
            input = input[0]  # Get the input tensor from the tuple
            input = input.reshape(-1, input.shape[-1])  # Flatten the input to (num_samples, num_channels)
            mask = (input.abs().sum(dim=1) != 0)
            input = input[mask]
            # head_dim
            input = input.reshape(-1, head_dim)
            
            if type=="max":
                max_val = input.abs().max(dim=0).values
                prev_max_val = out_proj_scales.get(name, torch.zeros_like(max_val))
                out_proj_scales[name] = torch.maximum(prev_max_val, max_val) # type: ignore
            else:
                raise ValueError(f"Unsupported scaling type: {type}")
                    
        return hook

    hooks = []
    for layer_names, _ in layers_to_scale:
        layer_name = layer_names[0]
        if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
            module = moduledict[layer_name]
            assert module is not None, f"Module {layer_name} not found in model"
            assert isinstance(module, torch.nn.Linear), f"Module {layer_name} is not a linear layer"
            hooks.append(module.register_forward_hook(create_hook2(layer_name)))

    with torch.no_grad():
        for x, text in modelQ.calibration_samples:
            forward_fn(x, text, modelQ)

    for hook in hooks:
        hook.remove()

    # alphas
    out_proj_alphas = torch.linspace(0, 1, 10) if abit < 16 else torch.tensor([0.0])
    out_proj_selected_alphas = {}
    out_proj_scale_losses = {}
    for alpha in out_proj_alphas:
        alpha = alpha.item()
        running_losses = {}
        num_samples_passed = {}
        def create_hook2_1(name):
            def hook(module, input, output):
                input = input[0]
                input = input.reshape(-1, input.shape[-1])
                output = output.reshape(-1, output.shape[-1])
                mask = (input.abs().sum(dim=-1) != 0)
                input = input[mask]
                output = output[mask]
                org_input_shape = input.shape
                input = input.reshape(-1, head_dim)
                scale = out_proj_scales[name]
                w = module.weight.data.clone()
                org_w_shape = w.shape
                wmax = w.reshape(-1, head_dim).abs().max(dim=0).values
                numerator = scale ** alpha
                denominator = wmax ** (1-alpha)
                value = numerator / denominator
                input_scaled = input / value.unsqueeze(0)
                w_scaled = w.reshape(-1, head_dim) * value.unsqueeze(0)
                input_scaled = input_scaled.reshape(org_input_shape)
                w_scaled = w_scaled.reshape(org_w_shape)
                # quantize the input
                if abit < 16:
                    amaxq = 2 ** (abit - 1) - 1
                    qscale = input_scaled.abs().max(dim=-1).values / amaxq
                    qinput = (input_scaled / qscale.unsqueeze(-1)).round().clamp(-amaxq, amaxq)
                    dqinput = qinput * qscale.unsqueeze(-1)
                else:
                    dqinput = input_scaled
                # quantize weights                wmaxq = 2 ** (wbit - 1) - 1
                wmaxq = 2 ** (wbit - 1) - 1
                qscale = w_scaled.abs().max(dim=-1).values / wmaxq
                qw = (w_scaled / qscale.unsqueeze(-1)).round().clamp(-wmaxq, wmaxq)
                dw = qw * qscale.unsqueeze(-1)
                # compute output with quantized weights and inputs=
                output_quant = torch.matmul(dqinput, dw.t())
                output_quant = output_quant.reshape(output.shape)
                if module.bias is not None:                    
                    output_quant += module.bias
                mse_loss = torch.mean((output - output_quant) ** 2)
                tmp = output.numel()
                mean_loss = running_losses.get(name, 0)
                prev_num_samples = num_samples_passed.get(name, 0)
                mean_loss = mean_loss * (prev_num_samples / (prev_num_samples + tmp)) + mse_loss.item() * (tmp / (prev_num_samples + tmp))
                running_losses[name] = mean_loss
                num_samples_passed[name] = prev_num_samples + tmp

            return hook

        hooks = []
        for layer_names, _ in layers_to_scale:
            layer_name = layer_names[0]
            if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
                module = moduledict[layer_name]
                hooks.append(module.register_forward_hook(create_hook2_1(layer_name)))

        with torch.no_grad():
            for i, (x, text) in enumerate(modelQ.calibration_samples[:128]):
                forward_fn(x, text, modelQ)

        for hook in hooks:
            hook.remove()

        for layer_names, _ in layers_to_scale:
            layer_name = layer_names[0]
            if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
                if running_losses[layer_name] < out_proj_scale_losses.get(layer_name, float("inf")):
                    out_proj_scale_losses[layer_name] = running_losses[layer_name]
                    out_proj_selected_alphas[layer_name] = alpha

        print(f"Processed output projection layers for alpha {alpha:.2f}")
    
    # Using the selected alphas, compute the final scales for the output projection layers
    for layer_names, _ in layers_to_scale:
        layer_name = layer_names[0]
        if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
            alpha = out_proj_selected_alphas[layer_name]
            scale = out_proj_scales[layer_name]
            w = moduledict[layer_name].weight.data.clone()
            org_w_shape = w.shape
            wmax = w.reshape(-1, head_dim).abs().max(dim=0).values
            numerator = scale ** alpha
            denominator = wmax ** (1-alpha)
            value = numerator / denominator
            out_proj_scales[layer_name] = value

    # Now apply the scales to the out projections
    for layer_names, prev_names in layers_to_scale:
        layer_name = layer_names[0]
        if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
            module = moduledict[layer_name]
            scale = out_proj_scales[layer_name]
            w = module.weight.data.clone()
            org_w_shape = w.shape
            w = w.reshape(-1, head_dim)
            w_scaled = w * scale.unsqueeze(0)
            w_scaled = w_scaled.reshape(org_w_shape)
            module.weight.data = w_scaled

            prev_names_list = [prev_names] if isinstance(prev_names, str) else list(prev_names)
            for pn in prev_names_list:
                prev_module = moduledict[pn]
                assert isinstance(prev_module, torch.nn.Linear), f"Predecessor module {pn} is not a linear layer"
                w = prev_module.weight.data.clone()
                dtype = w.dtype
                w_t = w.t()
                org_shape = w_t.shape
                w_t = w_t.reshape(-1, org_shape[-1] // head_dim, head_dim)
                w_t = w_t / scale.unsqueeze(0).unsqueeze(0)
                w_t = w_t.reshape(org_shape)
                w = w_t.t().to(dtype=dtype, device=device)
                prev_module.weight.data = w
                if prev_module.bias is not None:
                    b = prev_module.bias.data.clone()
                    org_shape = b.shape
                    b = b.reshape(-1, head_dim)
                    b = b / scale
                    b = b.reshape(org_shape)
                    prev_module.bias.data = b    
    # endregion

    # region Phase 3: Scales
    # Find Scale for all other layers
    # first obtain the scales
    final_scales = {}
    num_samples_passed = {}

    def create_hook3(name):
        def hook(module, input, output):
            input = input[0]
            input = input.reshape(-1, input.shape[-1])
            mask = (input.abs().sum(dim=-1) != 0)
            input = input[mask]
            prev_max = final_scales.get(name, torch.zeros(input.shape[1], device=input.device))
            max_val = input.abs().max(dim=0).values
            final_scales[name] = torch.maximum(prev_max, max_val) # type: ignore

        return hook
    hooks = []
    for layer_names, _ in layers_to_scale:
        if ("linear_out" in layer_names[0] or "o_proj" in layer_names[0] or "out_proj" in layer_names[0]):
            continue # already processed in Phase 2
        for layer_name in layer_names:
            module = moduledict[layer_name]
            hooks.append(module.register_forward_hook(create_hook3(layer_name)))
    with torch.no_grad():
        for x, text in modelQ.calibration_samples:
            forward_fn(x, text, modelQ)
    for hook in hooks:
        hook.remove()

    # Find best alpha for scales
    alphas = torch.linspace(0, 1, 10) if abit < 16 else torch.tensor([0.0])
    selected_alphas = {}
    scale_losses = {}
    for alpha in alphas:
        alpha = alpha.item()
        running_losses = {}
        num_samples_passed = {}
        def create_hook3(name):
            def hook(module, input, output):
                input = input[0]
                input = input.reshape(-1, input.shape[-1])
                output = output.reshape(-1, output.shape[-1])
                mask = (input.abs().sum(dim=-1) != 0)
                input = input[mask]
                output = output[mask]
                scale = final_scales[name]
                W = module.weight.data.clone()
                org_input_shape = input.shape
                wmax = W.abs().max(dim=0).values
                numerator = scale ** alpha
                denominator = wmax ** (1-alpha)
                value = numerator / denominator
                input_scaled = input / value.unsqueeze(0)
                W_scaled = W * value.unsqueeze(0)
                # quantize the input
                if abit < 16:
                    amaxq = 2 ** (abit - 1) - 1
                    qscale = input_scaled.abs().max(dim=-1).values / amaxq
                    qinput = (input_scaled / qscale.unsqueeze(-1)).round().clamp(-amaxq, amaxq)
                    dqinput = qinput * qscale.unsqueeze(-1)
                else:
                    dqinput = input_scaled
                # quantize weights
                wmaxq = 2 ** (wbit - 1) - 1
                qscale = W_scaled.abs().max(dim=1).values / wmaxq
                qweight = (W_scaled / qscale.unsqueeze(-1)).round().clamp(-wmaxq, wmaxq)
                dqweight = qweight * qscale.unsqueeze(-1)
                # compute output with quantized weights and inputs
                dqinput = dqinput.reshape(org_input_shape)
                dqweight = dqweight.reshape(W.shape)
                output_quant = torch.matmul(dqinput, dqweight.t())
                output_quant = output_quant.reshape(output.shape)
                if module.bias is not None:
                    output_quant += module.bias
                mse_loss = torch.mean((output - output_quant) ** 2)
                tmp = output.numel()
                mean_loss = running_losses.get(name, 0)
                prev_num_samples = num_samples_passed.get(name, 0)
                mean_loss = mean_loss * (prev_num_samples / (prev_num_samples + tmp)) + mse_loss.item() * (tmp / (prev_num_samples + tmp))
                running_losses[name] = mean_loss
                num_samples_passed[name] = prev_num_samples + tmp

            return hook

        hooks = []
        for layer_names, _ in layers_to_scale:
            for layer_name in layer_names:
                if ("linear_out" in layer_names[0] or "o_proj" in layer_names[0] or "out_proj" in layer_names[0]):
                    continue # already processed in Phase 2
                module = moduledict[layer_name]
                hooks.append(module.register_forward_hook(create_hook3(layer_name)))

        with torch.no_grad():
            for i, (x, text) in enumerate(modelQ.calibration_samples[:128]):
                forward_fn(x, text, modelQ)
        for hook in hooks:
            hook.remove()
        for layer_names, _ in layers_to_scale:
            for layer_name in layer_names:
                if ("linear_out" in layer_names[0] or "o_proj" in layer_names[0] or "out_proj" in layer_names[0]):
                    continue # already processed in Phase 2
                if running_losses[layer_name] < scale_losses.get(layer_name, float("inf")):
                    scale_losses[layer_name] = running_losses[layer_name]
                    selected_alphas[layer_name] = alpha

        print(f"Processed layers for alpha {alpha:.2f}")

    # Compute the scales using the selected alpha
    for layer_names, _ in layers_to_scale:
        layer_name = layer_names[0]
        if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
            continue # already processed in Phase 2
        amax = final_scales.get(layer_name, None)
        assert amax is not None, f"amax not found for layer {layer_name}"
        if len(layer_names) > 1:
            tmp = [selected_alphas[layer_name] for layer_name in layer_names]
            alpha = np.mean(tmp).item()
            wmax = torch.cat([moduledict[layer_name].weight.data.abs().max(dim=0).values.unsqueeze(0) for layer_name in layer_names], dim=0).max(dim=0).values
        else:
            alpha = selected_alphas[layer_name]
            wmax = moduledict[layer_name].weight.data.abs().max(dim=0).values

        final_scales[layer_name] = (amax**alpha) / (wmax**(1-alpha) + 1e-7)

    print(selected_alphas)

    print(f"Scales and rotations obtained and saved to {scale_path}")
    # endregion

    torch.save(final_scales, scale_path)
    torch.save(out_proj_scales, scale_path.replace(".pt", "_out_proj.pt"))
    torch.save(hadamard_rotations, scale_path.replace(".pt", "_hadamard.pt"))
    

    # restore original model state dict 
    model.load_state_dict(model_state_dict)



def scale_model(modelQ, audio, layers_to_scale, head_dim, scale_path):
    model = modelQ.model
    processor = modelQ.processor
    device = model.device

    scales = torch.load(scale_path)
    out_proj_scales = torch.load(scale_path.replace(".pt", "_out_proj.pt"))
    rotations = torch.load(scale_path.replace(".pt", "_hadamard.pt"))
    moduledict = dict(model.named_modules())

    # region Phase 1: Rotation
    for layer_names, prev_name in layers_to_scale:
        layer_name = layer_names[0]
        if layer_names[0] not in rotations:
            continue  # no rotation for this layer

        H = rotations[layer_name].to(device)  # (head_dim, head_dim)
        for layer_name in layer_names:
            module = moduledict[layer_name]
            w = module.weight.data
            dtype = w.dtype
            org_shape = w.shape
            temp = w.double().reshape(-1, org_shape[-1] // head_dim, head_dim)
            temp = temp @ H.double()
            module.weight.data = temp.reshape(org_shape).to(dtype=dtype, device=device)

        # Predecessor(s): W_v_new = block_diag(H^T, ..., H^T) @ W_v
        # Implemented via the transpose trick: (W_v^T @ block_diag(H))^T
        prev_names_list = [prev_name] if isinstance(prev_name, str) else list(prev_name)
        for pn in prev_names_list:
            prev_module = moduledict[pn]
            if not isinstance(prev_module, torch.nn.Linear):
                raise ValueError(f"Predecessor module {pn} is not a linear layer, cannot apply rotation")
            w = prev_module.weight.data
            dtype = w.dtype
            w_t = w.double().t()               # (in_features, num_heads*head_dim)
            org_shape = w_t.shape
            temp = w_t.reshape(-1, org_shape[-1] // head_dim, head_dim)
            temp = temp @ H.double()
            prev_module.weight.data = temp.reshape(org_shape).t().to(dtype=dtype, device=device)
            if prev_module.bias is not None:
                b = prev_module.bias.data
                num_heads = b.shape[0] // head_dim
                b_rot = (b.double().reshape(num_heads, head_dim) @ H.double()).reshape(prev_module.bias.data.shape)
                prev_module.bias.data = b_rot.to(dtype=b.dtype, device=device)
    # endregion

    # region Phase 2: Scales
    # Out projections
    for layer_names, prev_name in layers_to_scale:
        scale_name = layer_names[0]
        if "linear_out" in scale_name or "o_proj" in scale_name or "out_proj" in scale_name:
            scale = out_proj_scales[scale_name].to(device)  # 1-D, per input channel of current layer

            # Current layer(s): multiply per input channel (column)
            for layer_name in layer_names:
                module = moduledict[layer_name]
                w = module.weight.data
                dtype = w.dtype
                org_shape = w.shape
                w = w.reshape(-1, head_dim)
                w = w * scale.unsqueeze(0)
                module.weight.data = w.reshape(org_shape).to(dtype=dtype, device=device)

            # Predecessor(s): divide per output channel (row for Linear)
            prev_names_list = [prev_name] if isinstance(prev_name, str) else list(prev_name)
            for pn in prev_names_list:
                prev_module = moduledict[pn]
                assert isinstance(prev_module, torch.nn.Linear), f"Predecessor module {pn} is not a linear layer, cannot apply scaling"
                w = prev_module.weight.data
                wdtype = w.dtype
                w_t = w.t() # (in_features, num_heads*head_dim)
                org_shape = w_t.shape
                w_t = w_t.reshape(-1, head_dim) / scale.unsqueeze(0)
                w = w_t.reshape(org_shape).t().to(dtype=wdtype, device=device)
                prev_module.weight.data = w
                if prev_module.bias is not None:
                    b = prev_module.bias.data
                    bdtype = b.dtype
                    org_shape = b.shape
                    b = b.reshape(-1, head_dim) / scale.unsqueeze(0)
                    prev_module.bias.data = b.reshape(org_shape).to(dtype=bdtype, device=device)

    # endregion

    # region Phase 3: Scales
    # Other layers
    for layer_names, prev_names in layers_to_scale:
        scale_name = layer_names[0]
        if "linear_out" in scale_name or "o_proj" in scale_name or "out_proj" in scale_name:
            continue # already processed in Phase 2
        scale = scales[scale_name].to(device)  # 1-D, per input channel of current layer

        for layer_name in layer_names:
            module = moduledict[layer_name]
            w = module.weight.data
            dtype = w.dtype
            org_shape = w.shape
            w = w.reshape(-1, w.shape[-1]) * scale.unsqueeze(0)
            module.weight.data = w.reshape(org_shape).to(dtype=dtype, device=device)

        prev_names_list = [prev_names] if isinstance(prev_names, str) else list(prev_names)
        for pn in prev_names_list:
            prev_module = moduledict[pn]
            if isinstance(prev_module, torch.nn.Linear):
                w = prev_module.weight.data
                dtype = w.dtype
                w = w / scale.unsqueeze(1)
                prev_module.weight.data = w.to(dtype=dtype, device=device)
                if prev_module.bias is not None:
                    b = prev_module.bias.data
                    bdtype = b.dtype
                    b = b / scale
                    prev_module.bias.data = b.to(dtype=bdtype, device=device)
            else:
                w = prev_module.weight.data
                dtype = w.dtype
                w = w / scale
                prev_module.weight.data = w.to(dtype=dtype, device=device)
                if hasattr(prev_module, "bias") and prev_module.bias is not None:
                    bdtype = prev_module.bias.data.dtype
                    prev_module.bias.data = (prev_module.bias.data / scale).to(dtype=bdtype, device=device)
               
    # endregion 


@register_transform(TransformNames.rotscaling)
class RotScalingTransform(ScalingTransform):
    def obtain_transform(self, modelQ) -> None:
        if self.cfg.obtain_scales is False:
            return
        layers_to_scale, _, forward_fn, head_dim  = self.prepare_for_transform(modelQ)
        obtain_scales(modelQ, layers_to_scale, self.cfg.path, forward_fn, head_dim, self.cfg.type, self.cfg.wbits, self.cfg.abits)

    def apply_transform(self, modelQ) -> None:
        layers_to_scale, transcribe_fn, _, head_dim = self.prepare_for_transform(modelQ)
        original_text = transcribe_fn()
        scale_model(modelQ, None, layers_to_scale, head_dim, self.cfg.path)
        new_text = transcribe_fn()
        print(f"Original text: {original_text}")
        print(f"After        : {new_text}")
        assert original_text == new_text, "The output text has changed after rotscaling, which should not happen!"
        