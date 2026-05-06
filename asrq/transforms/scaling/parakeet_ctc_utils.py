import torch



def get_parakeet_ctc_layers_to_scale(model):
    # layers to scale 
    layers_to_scale = []
    # Encoder
    num_encoder_layers = len(model.encoder.layers)
    for i in range(num_encoder_layers):
        layers_to_scale.append(
            (
                [f"encoder.layers.{i}.feed_forward1.linear1"],
                f"encoder.layers.{i}.norm_feed_forward1"
            ),
        )
        
        layers_to_scale.append(
            (
                [f"encoder.layers.{i}.self_attn.linear_q", f"encoder.layers.{i}.self_attn.linear_k", f"encoder.layers.{i}.self_attn.linear_v"],
                f"encoder.layers.{i}.norm_self_att"
            ),
        )
        layers_to_scale.append(
            (
                [f"encoder.layers.{i}.self_attn.linear_out"],
                f"encoder.layers.{i}.self_attn.linear_v"
            ),
        )
        layers_to_scale.append(
            (
                [f"encoder.layers.{i}.feed_forward2.linear1"],
                f"encoder.layers.{i}.norm_feed_forward2"
            )
        )
        

    return layers_to_scale
