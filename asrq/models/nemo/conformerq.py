import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import math
from torch.nn import LayerNorm

from nemo.collections.asr.modules.conformer_encoder import(
    ConformerEncoder,
    AccessMixin,
    ConformerLayer,
    MultiHeadAttention,
    NeuralModule,
    ConvSubsampling,
    StackingSubsampling,
    SubsamplingReductionModule,
    compute_stochastic_depth_drop_probs,
)
from nemo.collections.asr.parts.submodules.conformer_modules import (
    ConformerConvolution,
    ConformerFeedForward,
    Swish,
)
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
    RelPositionalEncoding,
    LocalAttRelPositionalEncoding,
    PositionalEncoding,
    avoid_float16_autocast_context,
    INF_VAL,
)

from asrq.core.linear import LinearQ


class MultiHeadAttentionQ(MultiHeadAttention):
    """Quantized multi-head attention for the conformer encoder.

    Replaces the four dense projections (Q, K, V, output) with
    :class:`LinearQ` layers while preserving the KV-cache interface.
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        max_cache_len: int = 0,
        use_bias: bool = True,
        use_pytorch_sdpa: bool = False,
        use_pytorch_sdpa_backends: Optional[List[str]] = None,
        bits: int = 4,
    ) -> None:
        nn.Module.__init__(self)
        self.use_pytorch_sdpa = use_pytorch_sdpa
        if self.use_pytorch_sdpa and use_pytorch_sdpa_backends:
            use_pytorch_sdpa_backends = list(
                map(
                    lambda backend_name: getattr(torch.nn.attention.SDPBackend, backend_name),
                    use_pytorch_sdpa_backends,
                )
            )
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends

        self.cache_drop_size = None
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = LinearQ(n_feat, n_feat, bits, bias=use_bias)
        self.linear_k = LinearQ(n_feat, n_feat, bits, bias=use_bias)
        self.linear_v = LinearQ(n_feat, n_feat, bits, bias=use_bias)
        self.linear_out = LinearQ(n_feat, n_feat, bits, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout_rate)

        self._max_cache_len = max_cache_len


class RelPositionMultiHeadAttentionQ(MultiHeadAttentionQ):
    """Quantized relative-position multi-head attention.

    Extends :class:`MultiHeadAttentionQ` with a quantized positional
    linear projection (``linear_pos``) and learnable position biases
    used in the relative-shift attention scoring.
    """

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float,
        pos_bias_u: Optional[nn.Parameter],
        pos_bias_v: Optional[nn.Parameter],
        max_cache_len: int = 0,
        use_bias: bool = True,
        use_pytorch_sdpa: bool = False,
        use_pytorch_sdpa_backends: Optional[List[str]] = None,
        bits: int = 4,
    ) -> None:
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
            dropout_rate=dropout_rate,
            max_cache_len=max_cache_len,
            use_bias=use_bias,
            use_pytorch_sdpa=use_pytorch_sdpa,
            use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
            bits=bits,
        )
        # linear transformation for positional encoding
        self.linear_pos = LinearQ(n_feat, n_feat, bits, bias=False)
        # these two learnable biases are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            # nn.init.normal_(self.pos_bias_u, 0.0, 0.02)
            # nn.init.normal_(self.pos_bias_v, 0.0, 0.02)
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Apply relative shift to the positional attention matrix.

        Args:
            x: Tensor of shape ``(batch, heads, time1, time2)``.

        Returns:
            Shifted tensor with the same shape.
        """
        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
        pos_emb: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Compute relative-position attention.

        Args:
            query: Query tensor ``(batch, time1, feat)``.
            key: Key tensor ``(batch, time2, feat)``.
            value: Value tensor ``(batch, time2, feat)``.
            mask: Optional boolean attention mask.
            pos_emb: Relative positional encoding.
            cache: Optional cached key/value for streaming.

        Returns:
            Attention output, or ``(output, cache)`` when *cache* is
            provided.
        """
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)

            n_batch_pos = pos_emb.size(0)
            n_batch = value.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # (batch, head, time1, d_k)

            # (batch, head, time1, d_k)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            # (batch, head, time1, d_k)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

            # compute attention score
            # first compute matrix a and matrix c
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            # (batch, head, time1, time2)

            # compute matrix b and matrix d
            # (batch, head, time1, time2)
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            matrix_bd = self.rel_shift(matrix_bd)

            if self.use_pytorch_sdpa:
                scale_factor = 1 / math.sqrt(q_with_bias_u.size(-1))
                matrix_bd = matrix_bd[:, :, :, : k.size(-2)] * scale_factor

                if mask is not None:
                    mask = mask.unsqueeze(1)
                    matrix_bd.masked_fill_(mask, -INF_VAL)

                dropout_rate = self.dropout_rate if self.training else 0
                if self.use_pytorch_sdpa_backends:
                    with torch.nn.attention.sdpa_kernel(self.use_pytorch_sdpa_backends): # type: ignore
                        out = torch.nn.functional.scaled_dot_product_attention(
                            q_with_bias_u, k, v, attn_mask=matrix_bd, dropout_p=dropout_rate
                        )
                else:
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q_with_bias_u, k, v, attn_mask=matrix_bd, dropout_p=dropout_rate
                    )

                # this IF block can be deleted when https://github.com/pytorch/pytorch/pull/131863 is in the stable version
                if mask is not None:
                    all_masked_rows = torch.all(mask, dim=-1)
                    all_masked_rows.unsqueeze_(-1)
                    all_masked_rows = all_masked_rows.expand(-1, out.size(1), -1, out.size(-1))
                    out = out.masked_fill(all_masked_rows, 0.0)

                out = out.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
                out = self.linear_out(out)  # (batch, time1, d_model)
            else:
                # drops extra elements in the matrix_bd to match the matrix_ac's size
                matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
                matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
                scores = (matrix_ac + matrix_bd) / self.s_d_k  # (batch, head, time1, time2)
                out = self.forward_attention(v, scores, mask)

        if cache is None:
            return out
        else:
            return out, cache


class ConformerFeedForwardQ(ConformerFeedForward):
    """Quantized conformer feed-forward module.

    Replaces both dense layers with :class:`LinearQ` while keeping the
    activation and dropout unchanged.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        activation: nn.Module = Swish(),
        use_bias: bool = True,
        bits: int = 4,
    ) -> None:
        nn.Module.__init__(self)
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = use_bias
        self.linear1 = LinearQ(d_model, d_ff, bits, bias=self.use_bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = LinearQ(d_ff, d_model, bits, bias=self.use_bias)


class ConformerLayerQ(ConformerLayer):
    """Quantized conformer layer.

    Uses :class:`ConformerFeedForwardQ` for both feed-forward modules
    while keeping the convolution and self-attention sub-modules from
    the base conformer layer.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        self_attention_model: str = 'rel_pos',
        global_tokens: int = 0,
        global_tokens_spacing: int = 1,
        global_attn_separate: bool = False,
        n_heads: int = 4,
        conv_kernel_size: int = 31,
        conv_norm_type: str = 'batch_norm',
        conv_context_size: Optional[List[int]] = None,
        dropout: float = 0.1,
        dropout_att: float = 0.1,
        pos_bias_u: Optional[nn.Parameter] = None,
        pos_bias_v: Optional[nn.Parameter] = None,
        att_context_size: List[int] = [-1, -1],
        use_bias: bool = True,
        use_pytorch_sdpa: bool = False,
        use_pytorch_sdpa_backends: Optional[List[str]] = None,
    ) -> None:
        
        torch.nn.Module.__init__(self)
        AccessMixin.__init__(self)

        self.use_pytorch_sdpa = use_pytorch_sdpa
        if use_pytorch_sdpa_backends is None:
            use_pytorch_sdpa_backends = []
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForwardQ(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
        )

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        MHA_max_cache_len = att_context_size[0]

        if self_attention_model == 'rel_pos':
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            self.self_attn = RelPositionMultiHeadAttentionLongformer(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                att_context_size=att_context_size,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                use_bias=use_bias,
            )
        elif self_attention_model == 'abs_pos':
            self.self_attn = MultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
        else:
            raise ValueError(
                f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                f"valid values can be from ['rel_pos', 'rel_pos_local_attn', 'abs_pos']"
            )

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForwardQ(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)


class ConformerEncoderQ(ConformerEncoder):
    """Quantized conformer encoder.

    Replaces each :class:`ConformerLayer` with a quantized variant that
    uses :class:`ConformerFeedForwardQ` feed-forward modules.  Subsampling,
    positional encoding, and streaming setup are inherited from the base
    :class:`ConformerEncoder`.
    """

    def __init__(
        self,
        feat_in: int,
        n_layers: int,
        d_model: int,
        feat_out: int = -1,
        causal_downsampling: bool = False,
        subsampling: str = 'striding',
        subsampling_factor: int = 4,
        subsampling_conv_chunking_factor: int = 1,
        subsampling_conv_channels: int = -1,
        reduction: Optional[str] = None,
        reduction_position: Optional[int] = None,
        reduction_factor: int = 1,
        ff_expansion_factor: int = 4,
        self_attention_model: str = 'rel_pos',
        n_heads: int = 4,
        att_context_size: Optional[List[int]] = None,
        att_context_probs: Optional[List[float]] = None,
        att_context_style: str = 'regular',
        xscaling: bool = True,
        untie_biases: bool = True,
        pos_emb_max_len: int = 5000,
        conv_kernel_size: int = 31,
        conv_norm_type: str = 'batch_norm',
        conv_context_size: Optional[List[int]] = None,
        use_bias: bool = True,
        dropout: float = 0.1,
        dropout_pre_encoder: float = 0.1,
        dropout_emb: float = 0.1,
        dropout_att: float = 0.0,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        global_tokens: int = 0,
        global_tokens_spacing: int = 1,
        global_attn_separate: bool = False,
        use_pytorch_sdpa: bool = False,
        use_pytorch_sdpa_backends: Optional[List[str]] = None,
        sync_max_audio_length: bool = True,
    ) -> None:
        NeuralModule.__init__(self)
        AccessMixin.__init__(self)

        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.att_context_style = att_context_style
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        self.self_attention_model = self_attention_model
        self.global_tokens = global_tokens
        self.global_attn_separate = global_attn_separate
        self.global_tokens_spacing = global_tokens_spacing
        self.use_pytorch_sdpa = use_pytorch_sdpa
        if use_pytorch_sdpa_backends is None:
            use_pytorch_sdpa_backends = []
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends
        self.sync_max_audio_length = sync_max_audio_length

        # Setting up the att_context_size
        (
            self.att_context_size_all,
            self.att_context_size,
            self.att_context_probs,
            self.conv_context_size,
        ) = self._calc_context_sizes(
            att_context_style=att_context_style,
            att_context_size=att_context_size,
            att_context_probs=att_context_probs,
            conv_context_size=conv_context_size,
            conv_kernel_size=conv_kernel_size,
        )

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None

        # Subsampling
        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        if subsampling and subsampling_factor > 1:
            if subsampling in ['stacking', 'stacking_norm']:
                # stacking_norm has an extra layer norm after stacking comparing to stacking
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    norm=True if subsampling == 'stacking_norm' else False,
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    conv_channels=subsampling_conv_channels,
                    subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
                    activation=nn.ReLU(True),
                    is_causal=causal_downsampling,
                )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

        # Reduction
        if reduction and reduction_factor > 1:
            assert reduction_position is not None and reduction_position >= -1 and reduction_position < n_layers
            self.reduction_subsampling = SubsamplingReductionModule(
                reduction=reduction,
                d_model=d_model,
                reduction_factor=reduction_factor,
            )
            self.reduction_position = reduction_position
        else:
            self.reduction_subsampling = None
            self.reduction_position = None

        self._feat_out = d_model

        # Biases for relative positional encoding
        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None

        # Positional encodings
        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout_pre_encoder,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            if att_context_size is None or max(att_context_size) <= 0:
                raise ValueError("When using local attention, context size must be set > 0")
            self.pos_enc = LocalAttRelPositionalEncoding(
                att_context_size=att_context_size,
                d_model=d_model,
                dropout_rate=dropout,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            pos_bias_u = None
            pos_bias_v = None
            self.pos_enc = PositionalEncoding(
                d_model=d_model, dropout_rate=dropout_pre_encoder, max_len=pos_emb_max_len, xscale=self.xscale
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                conv_context_size=self.conv_context_size,
                dropout=dropout,
                dropout_att=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                att_context_size=self.att_context_size,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
            self.layers.append(layer)

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True

        self.setup_streaming_params()
        self.export_cache_support = False

        self.layer_drop_probs = compute_stochastic_depth_drop_probs(
            len(self.layers), stochastic_depth_drop_prob, stochastic_depth_mode, stochastic_depth_start_layer
        )
        # will be set in self.forward() if defined in AccessMixin config
        self.interctc_capture_at_layers = None
