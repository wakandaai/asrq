
import numpy as np
import torch



def get_canary_qwen_layers_to_scale(model):
    # layers to scale 
    layers_to_scale = []
    # Encoder
    num_encoder_layers = len(model.perception.encoder.layers)
    num_decoder_layers = model.llm.config.num_hidden_layers
    for i in range(num_encoder_layers):
        layers_to_scale.append(
            (
                [f"perception.encoder.layers.{i}.feed_forward1.linear1"],
                f"perception.encoder.layers.{i}.norm_feed_forward1"
            ),
        )
        
        layers_to_scale.append(
            (
                [f"perception.encoder.layers.{i}.self_attn.linear_q", f"perception.encoder.layers.{i}.self_attn.linear_k", f"perception.encoder.layers.{i}.self_attn.linear_v"],
                f"perception.encoder.layers.{i}.norm_self_att"
            ),
        )
        layers_to_scale.append(
            (
                [f"perception.encoder.layers.{i}.self_attn.linear_out"],
                f"perception.encoder.layers.{i}.self_attn.linear_v"
            ),
        )
        layers_to_scale.append(
            (
                [f"perception.encoder.layers.{i}.feed_forward2.linear1"],
                f"perception.encoder.layers.{i}.norm_feed_forward2"
            )
        )
        # The two feed-forward down-projections are fed by their activation, which has no
        # weight to absorb the reciprocal scale, so prev is None and each linear2 divides
        # its own input (see ScaledInputLinear).
        for ff in ("feed_forward1", "feed_forward2"):
            layers_to_scale.append(
                (
                    [f"perception.encoder.layers.{i}.{ff}.linear2"],
                    None
                )
            )

        # Conv module. pointwise_conv1 is fed by the norm before the block; pointwise_conv2
        # is fed by an activation (conv1 -> GLU -> depthwise -> norm -> activation ->
        # conv2), so it is a down-projection in all but name and scales its own input.
        layers_to_scale.append(
            (
                [f"perception.encoder.layers.{i}.conv.pointwise_conv1"],
                f"perception.encoder.layers.{i}.norm_conv"
            )
        )
        layers_to_scale.append(
            (
                [f"perception.encoder.layers.{i}.conv.pointwise_conv2"],
                None
            )
        )


    for i in range(num_decoder_layers):
        layers_to_scale.append(
            (
                [
                    f"llm.base_model.model.model.layers.{i}.self_attn.q_proj.base_layer",
                    f"llm.base_model.model.model.layers.{i}.self_attn.k_proj",
                    f"llm.base_model.model.model.layers.{i}.self_attn.v_proj.base_layer",
                    f"llm.base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.default",
                    f"llm.base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.default"
                ],
                f"llm.base_model.model.model.layers.{i}.input_layernorm"
            )
        )

        layers_to_scale.append(
            (
                [f"llm.base_model.model.model.layers.{i}.self_attn.o_proj"],
                [f"llm.base_model.model.model.layers.{i}.self_attn.v_proj.base_layer", f"llm.base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.default"]
            )
        )

        layers_to_scale.append(
            (
                [f"llm.base_model.model.model.layers.{i}.mlp.gate_proj", f"llm.base_model.model.model.layers.{i}.mlp.up_proj"],
                f"llm.base_model.model.model.layers.{i}.post_attention_layernorm"
            )
        )

        # Qwen3MLP is down_proj(act_fn(gate_proj(x)) * up_proj(x)), so the input to
        # down_proj is a gated product with no single producer to divide - down_proj
        # divides its own input instead (see ScaledInputLinear).
        layers_to_scale.append(
            (
                [f"llm.base_model.model.model.layers.{i}.mlp.down_proj"],
                None
            )
        )



    return layers_to_scale
    