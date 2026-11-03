def conformer_layers_to_scale(prefix, num_layers):
    """The ``(layer_names, prev)`` entries for a stack of Conformer blocks.

    Seven entries per block: each feed-forward's linear1 behind its norm, the attention q/k/v behind the
    self-attention norm (one entry, since they share an input), the attention output behind v_proj
    (head-wise), each feed-forward's linear2 behind its activation, the first pointwise convolution behind
    the convolution norm, and the second behind the convolution's activation.

    Both Conformer-encoder models use these; they differ only in where the block list sits.

    Args:
        prefix: The path of the block list, such as ``encoder.layers``.
        num_layers: How many blocks the list holds.
    """
    layers_to_scale = []
    for i in range(num_layers):
        block = f"{prefix}.{i}"
        layers_to_scale.append(([f"{block}.feed_forward1.linear1"], f"{block}.norm_feed_forward1"))
        layers_to_scale.append((
            [f"{block}.self_attn.linear_q", f"{block}.self_attn.linear_k", f"{block}.self_attn.linear_v"],
            f"{block}.norm_self_att",
        ))
        layers_to_scale.append(([f"{block}.self_attn.linear_out"], f"{block}.self_attn.linear_v"))
        layers_to_scale.append(([f"{block}.feed_forward2.linear1"], f"{block}.norm_feed_forward2"))
        for ff in ("feed_forward1", "feed_forward2"):
            layers_to_scale.append(([f"{block}.{ff}.linear2"], f"{block}.{ff}.activation"))
        layers_to_scale.append(([f"{block}.conv.pointwise_conv1"], f"{block}.norm_conv"))
        layers_to_scale.append(([f"{block}.conv.pointwise_conv2"], f"{block}.conv.activation"))
    return layers_to_scale


def get_parakeet_ctc_layers_to_scale(model):
    return conformer_layers_to_scale("encoder.layers", len(model.encoder.layers))
