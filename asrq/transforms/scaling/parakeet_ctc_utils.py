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
        # The two feed-forward down-projections are fed by their activation, which has no
        # weight to absorb the reciprocal scale, so prev is None and each linear2 divides
        # its own input (see ScaledInputLinear).
        for ff in ("feed_forward1", "feed_forward2"):
            layers_to_scale.append(
                (
                    [f"encoder.layers.{i}.{ff}.linear2"],
                    None
                )
            )

        # Conv module. pointwise_conv1 is fed by the norm before the block; pointwise_conv2
        # is fed by an activation (conv1 -> GLU -> depthwise -> norm -> activation ->
        # conv2), so it is a down-projection in all but name and scales its own input.
        layers_to_scale.append(
            (
                [f"encoder.layers.{i}.conv.pointwise_conv1"],
                f"encoder.layers.{i}.norm_conv"
            )
        )
        layers_to_scale.append(
            (
                [f"encoder.layers.{i}.conv.pointwise_conv2"],
                None
            )
        )


    return layers_to_scale
