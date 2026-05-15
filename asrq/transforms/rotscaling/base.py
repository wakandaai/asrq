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
    Hs = {}
    for layer_names, prev_names in layers_to_scale:
        layer_name = layer_names[0]
        if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
            H = (
                random_hadamard_matrix(head_dim, modelQ.model.device) @ \
                random_hadamard_matrix(head_dim, modelQ.model.device)
            )
            Hs[layer_name] = H
            # apply the rotations
            module = moduledict[layer_name]
            w = module.weight.data.clone()
            dtype = w.dtype
            org_shape = w.shape
            temp = w.double().reshape(-1, org_shape[-1] // head_dim, head_dim)
            temp = temp @ H.double()
            w_rot = temp.reshape(org_shape).to(dtype=dtype, device=device)
            module.weight.data = w_rot


        prev_names = [prev_names] if isinstance(prev_names, str) else prev_names
        for prev_name in prev_names:
            prev_module = moduledict[prev_name]
            if not isinstance(prev_module, torch.nn.Linear):
                raise ValueError(f"Predecessor module {prev_name} is not a linear layer, cannot apply rotation")
            w = prev_module.weight.data.clone()
            dtype = w.dtype
            w_t = w.double().t()               # (in_features, num_heads*head_dim)
            org_shape = w_t.shape
            temp = w_t.reshape(-1, org_shape[-1] // head_dim, head_dim)
            temp = temp @ H.double()
            prev_module.weight.data = temp.reshape(org_shape).t().to(dtype=dtype, device=device)
            if prev_module.bias is not None:
                b = prev_module.bias.data.clone()
                num_heads = b.shape[0] // head_dim
                b_rot = (b.double().reshape(num_heads, head_dim) @ H.double()).reshape(prev_module.bias.data.shape)
                prev_module.bias.data = b_rot.to(dtype=b.dtype, device=device)

    # endregion

    # region Phase 2:
    # smooth-quant scales
    ascales = {}
    def create_hook2(name):
        def hook(module, input, output):
            input = input[0]
            input = input.reshape(-1, input.shape[-1])
            max_val = ascales.get(name, torch.zeros(input.shape[1], device=input.device))
            max_val = torch.maximum(max_val, input.abs().max(dim=0).values)
            ascales[name] = max_val
        return hook
    hooks = []
    for layer_names, _ in layers_to_scale:
        layer_name = layer_names[0]
        if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
            continue # will be processed in the next phase
        for layer_name in layer_names:
            module = moduledict[layer_name]
            assert module is not None, f"Module {layer_name} not found in model"
            assert isinstance(module, torch.nn.Linear), f"Module {layer_name} is not a linear layer"
            hooks.append(module.register_forward_hook(create_hook2(layer_name)))
    
    with torch.no_grad():
        for x, text in modelQ.calibration_samples:
            forward_fn(x, text, modelQ)

    for hook in hooks:
        hook.remove()

    # search for best alpha
    alphas = torch.linspace(0, 1, 10) if abit < 16 else torch.tensor([0.0])
    selected_alphas = {}
    alpha_losses = {}
    for alpha in alphas:
        running_mean_loss = {}
        num_samples = {}
        def create_hook2_1(name):
            def hook(module, input, output):
                input = input[0]
                input = input.reshape(-1, input.shape[-1])
                output = output.reshape(-1, output.shape[-1])
                mask = (input.abs().sum(dim=-1) != 0)
                input = input[mask]
                output = output[mask]
                ascale = ascales[name]
                W = module.weight.data.clone()
                org_input_shape = input.shape
                wmax = W.abs().max(dim=0).values
                numerator = ascale ** alpha
                denominator = wmax ** (1-alpha)
                value = numerator / denominator.where(denominator != 0, 1)
                input_scaled = input / value.where(value != 0, 1).unsqueeze(0)
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
                mean_loss = running_mean_loss.get(name, 0)
                prev_num_samples = num_samples.get(name, 0)
                mean_loss = mean_loss * (prev_num_samples / (prev_num_samples + tmp)) + mse_loss.item() * (tmp / (prev_num_samples + tmp))
                running_mean_loss[name] = mean_loss
                num_samples[name] = prev_num_samples + tmp
            return hook
        hooks = []
        for layer_names, _ in layers_to_scale:
            layer_name = layer_names[0]
            if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
                continue # will be processed in the next phase
            for layer_name in layer_names:
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
                continue # will be processed in the next phase
            for layer_name in layer_names:
                if running_mean_loss[layer_name] < alpha_losses.get(layer_name, float("inf")):
                    alpha_losses[layer_name] = running_mean_loss[layer_name]
                    selected_alphas[layer_name] = alpha
        print(f"Processed layers for alpha {alpha:.2f}")

    # Use selected alpha
    final_scales = {}
    for layer_names, _ in layers_to_scale:
        layer_name = layer_names[0]
        if "linear_out" in layer_name or "o_proj" in layer_name or "out_proj" in layer_name:
            continue # will be processed in the next phase
        if len(layer_names) > 1:
            tmp = [selected_alphas[layer_name] for layer_name in layer_names]
            alpha = np.mean(tmp).item()
            wmax = torch.cat([moduledict[layer_name].weight.data.abs().max(dim=0).values.unsqueeze(0) for layer_name in layer_names], dim=0).max(dim=0).values
        else:
            alpha = selected_alphas[layer_name]
            wmax = moduledict[layer_name].weight.data.abs().max(dim=0).values
        ascale = ascales[layer_name]
        numerator = ascale ** alpha
        denominator = wmax ** (1-alpha)
        final_scales[layer_name] = numerator / denominator.where(denominator != 0, 1e-5)
    # endregion


    torch.save(final_scales, scale_path)
    torch.save(Hs, scale_path.replace(".pt", "_hadamard.pt"))
    

    # restore original model state dict 
    model.load_state_dict(model_state_dict)



def scale_model(modelQ, audio, layers_to_scale, head_dim, scale_path):
    model = modelQ.model
    processor = modelQ.processor
    device = model.device

    scales = torch.load(scale_path)
    rotations = torch.load(scale_path.replace(".pt", "_hadamard.pt"))
    moduledict = dict(model.named_modules())

    # region Phase 1: Rotation
    # v_o proj
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
                w = w / scale.where(scale != 0, 1).unsqueeze(1)
                prev_module.weight.data = w.to(dtype=dtype, device=device)
                if prev_module.bias is not None:
                    b = prev_module.bias.data
                    bdtype = b.dtype
                    b = b / scale.where(scale != 0, 1)
                    prev_module.bias.data = b.to(dtype=bdtype, device=device)
            else:
                w = prev_module.weight.data
                dtype = w.dtype
                w = w / scale.where(scale != 0, 1)
                prev_module.weight.data = w.to(dtype=dtype, device=device)
                if hasattr(prev_module, "bias") and prev_module.bias is not None:
                    bdtype = prev_module.bias.data.dtype
                    prev_module.bias.data = (prev_module.bias.data / scale.where(scale != 0, 1)).to(dtype=bdtype, device=device)
               
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
        