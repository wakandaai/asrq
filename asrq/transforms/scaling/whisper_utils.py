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
            [f"model.decoder.layers.{i}.encoder_attn.k_proj" for i in range(num_decoder_layers)] + [f"model.decoder.layers.{i}.encoder_attn.v_proj" for i in range(num_decoder_layers)],
            f"model.encoder.layer_norm"
        ),
    )
    return layers_to_scale
