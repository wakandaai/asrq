# pyright: reportMissingImports=false

from omegaconf import DictConfig
from asrq.core.registry import TransformNames, register_transform, register_transform_config, ModelNames
from asrq.core.types import Processor
from asrq.transforms.base import BaseTransform, TransformConfig
from asrq.transforms.scaling.canary_qwen_utils import get_canary_qwen_layers_to_scale
from asrq.transforms.scaling.parakeet_ctc_utils import get_parakeet_ctc_layers_to_scale
from asrq.transforms.scaling.whisper_utils import get_whisper_layers_to_scale
import torch
import numpy as np



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
            assert isinstance(module, torch.nn.Linear), f"Module {layer_name} is not a linear layer"
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
                input = input[0]
                input = input.reshape(-1, input.shape[-1])
                output = output.reshape(-1, output.shape[-1])
                mask = (input.abs().sum(dim=-1) != 0)
                input = input[mask]
                output = output[mask]
                W = module.weight.data.clone() 
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
                dqweight = dqweight.reshape(module.weight.data.shape)
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
            wmax = torch.cat([moduledict[layer_name].weight.data.abs().max(dim=0).values.unsqueeze(0) for layer_name in layer_names], dim=0).max(dim=0).values
        else:
            wmax = moduledict[layer_name].weight.data.abs().max(dim=0).values
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
            else:
                module.weight.data *= scale.unsqueeze(0)

        # scale the prev layer
        if isinstance(prev_name, str):
            prev_names = [prev_name]
        else:
            prev_names = prev_name
        for prev_name in prev_names:
            prev_module = moduledict[prev_name]
            # for linear layers
            if isinstance(prev_module, torch.nn.Linear):
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
            else:
                # It has to be a normalization layer
                # We scale the weight
                prev_module.weight.data /= scale
                if hasattr(prev_module, "bias") and prev_module.bias is not None:
                    prev_module.bias.data /= scale
    
            
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
        
