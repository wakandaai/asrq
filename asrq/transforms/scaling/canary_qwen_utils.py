
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
       


    return layers_to_scale
    