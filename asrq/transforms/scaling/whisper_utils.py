import torch



def get_whisper_layers_to_scale(model):
    # layers to scale 
    layers_to_scale = []
    # Encoder
    num_encoder_layers = model.config.encoder_layers
    for i in range(num_encoder_layers):
        layers_to_scale.append(
            (
                (f"model.encoder.layers.{i}.self_attn.q_proj", f"model.encoder.layers.{i}.self_attn.k_proj", f"model.encoder.layers.{i}.self_attn.v_proj"), 
                f"model.encoder.layers.{i}.self_attn_layer_norm"
            ),
        )
        layers_to_scale.append(
            (
                (f"model.encoder.layers.{i}.self_attn.out_proj",),
                f"model.encoder.layers.{i}.self_attn.v_proj"
            ),
        )
        layers_to_scale.append(
            (
                (f"model.encoder.layers.{i}.fc1",),
                f"model.encoder.layers.{i}.final_layer_norm"
            ),
        )
        # fc2 is fed by the activation, which has no weight to absorb the reciprocal
        # scale, so prev is None and fc2 divides its own input (see ScaledInputLinear).
        layers_to_scale.append(
            (
                (f"model.encoder.layers.{i}.fc2",),
                None
            ),
        )


    # Decoder
    num_decoder_layers = model.config.decoder_layers
    for i in range(num_decoder_layers):
        layers_to_scale.append(
            (
                (f"model.decoder.layers.{i}.self_attn.q_proj", f"model.decoder.layers.{i}.self_attn.k_proj", f"model.decoder.layers.{i}.self_attn.v_proj"), 
                f"model.decoder.layers.{i}.self_attn_layer_norm"
            ),
        )
        layers_to_scale.append(
            (
                (f"model.decoder.layers.{i}.self_attn.out_proj",),
                f"model.decoder.layers.{i}.self_attn.v_proj"
            ),
        )
        layers_to_scale.append(
            (
                (f"model.decoder.layers.{i}.encoder_attn.q_proj",), 
                f"model.decoder.layers.{i}.encoder_attn_layer_norm"
            ),
        )
        
        layers_to_scale.append(
            (
                (f"model.decoder.layers.{i}.encoder_attn.out_proj",),
                f"model.decoder.layers.{i}.encoder_attn.v_proj"
            ),
        )
        layers_to_scale.append(
            (
                (f"model.decoder.layers.{i}.fc1",),
                f"model.decoder.layers.{i}.final_layer_norm"
            ),
        )
        layers_to_scale.append(
            (
                (f"model.decoder.layers.{i}.fc2",),
                None
            ),
        )


    layers_to_scale.append(
        (
            [f"model.decoder.layers.{i}.encoder_attn.k_proj" for i in range(num_decoder_layers)] + [f"model.decoder.layers.{i}.encoder_attn.v_proj" for i in range(num_decoder_layers)],
            f"model.encoder.layer_norm"
        ),
    )
    return layers_to_scale
