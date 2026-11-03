from asrq.transforms.scaling.parakeet_ctc_utils import conformer_layers_to_scale


def get_canary_qwen_layers_to_scale(model):
    layers_to_scale = conformer_layers_to_scale("perception.encoder.layers", len(model.perception.encoder.layers))
    for i in range(model.llm.config.num_hidden_layers):
        block = f"llm.model.layers.{i}"
        layers_to_scale.append((
            [f"{block}.self_attn.q_proj", f"{block}.self_attn.k_proj", f"{block}.self_attn.v_proj"],
            f"{block}.input_layernorm",
        ))
        layers_to_scale.append(([f"{block}.self_attn.o_proj"], f"{block}.self_attn.v_proj"))
        layers_to_scale.append((
            [f"{block}.mlp.gate_proj", f"{block}.mlp.up_proj"],
            f"{block}.post_attention_layernorm",
        ))
        layers_to_scale.append(([f"{block}.mlp.down_proj"], f"{block}.mlp.up_proj"))
    return layers_to_scale
