
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
        for ff in ("feed_forward1", "feed_forward2"):
            layers_to_scale.append(
                (
                    [f"perception.encoder.layers.{i}.{ff}.linear2"],
                    f"perception.encoder.layers.{i}.{ff}.activation"
                )
            )
        layers_to_scale.append(
            (
                [f"perception.encoder.layers.{i}.conv.pointwise_conv1"],
                f"perception.encoder.layers.{i}.norm_conv"
            )
        )
        layers_to_scale.append(
            (
                [f"perception.encoder.layers.{i}.conv.pointwise_conv2"],
                f"perception.encoder.layers.{i}.conv.activation"
            )
        )

    for i in range(num_decoder_layers):
        layers_to_scale.append(
            (
                [f"llm.model.layers.{i}.self_attn.q_proj", f"llm.model.layers.{i}.self_attn.k_proj", f"llm.model.layers.{i}.self_attn.v_proj"],
                f"llm.model.layers.{i}.input_layernorm"
            )
        )

        layers_to_scale.append(
            (
                [f"llm.model.layers.{i}.self_attn.o_proj"],
                f"llm.model.layers.{i}.self_attn.v_proj"
            )
        )

        layers_to_scale.append(
            (
                [f"llm.model.layers.{i}.mlp.gate_proj", f"llm.model.layers.{i}.mlp.up_proj"],
                f"llm.model.layers.{i}.post_attention_layernorm"
            )
        )

        layers_to_scale.append(
            (
                [f"llm.model.layers.{i}.mlp.down_proj"],
                f"llm.model.layers.{i}.mlp.up_proj"
            )
        )

    return layers_to_scale
    