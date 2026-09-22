"""Duration head built from dense-matrix sequence mixers.

Motivation
----------
The classic convolutional duration predictor is a stack of ``Conv1d`` layers.
A convolution kernel is a 3-D parameter, and optimizers that act on the
*matrix* structure of a parameter see only the last two dimensions
(``[in_channels, kernel_size]``) of such a kernel. That has two consequences:

* a ``1x1`` kernel degenerates into a scalar rescaling of the gradient, so the
  orthogonalization such optimizers perform has no effect at all, while the
  step-size rule still scales with ``sqrt(in_channels)``;
* a larger kernel exposes only ``kernel_size`` right-singular directions.

Both effects make the effective step size and update direction differ by an
order of magnitude between layers of the very same stack, which shows up as a
module that does not train.

This head therefore keeps **every learned parameter a 2-D matrix or a 1-D
gain**: attention projections, feed-forward linears, GRU matrices and
normalization gains. Norm placement is pre-norm, so the residual stream is never
rescaled and the effective step size is not coupled to a per-block
normalization.

Structure
---------
``x -> in_proj -> N x [pre-norm local relative attention + pre-norm FFN]
   -> GRU -> out_norm`` and the caller applies the output projection.

Every operation is vectorized (no per-position Python loops) and keeps the
sequence length dynamic, so the module stays export-safe.

Group positions
---------------
When group ids are given, the head adds two learned embeddings: the position of
an item inside its group and the same position counted from the end of the
group. Both are derived from the group mask, so they need no extra input.

Padding note
------------
Padding positions are zeroed before the GRU, which makes the forward direction
independent of trailing padding: perturbing the padded tail changes the output
at real positions by float noise only. The backward direction of a bidirectional
GRU does read trailing padding states, and the resulting change at real
positions is *not* negligible (measured ~0.3 in activation units for a small
head), so ``gru_bidirectional`` defaults to ``False``. Attention already mixes
both directions across the window, so a forward-only GRU is enough for the
long-range accumulation this head needs.

Export note
-----------
Batch size stays 1 as in the other exported duration/variance graphs. Batch sizes
other than 1 combined with a dynamic sequence length are not supported by the
ONNX GRU op.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import AdamWLinear
from modules.fastspeech.grouping import group_mask, group_position_ids

__all__ = ["LocalRelativeAttention", "FFN", "Block", "DurationHeadV2"]

# Large negative score used instead of -inf so that fully masked rows stay
# finite and produce exactly zero after the output mask is applied.
_MASKED_SCORE = -1e4

# Accepted keys of the ``head_args`` configuration block.
HEAD_ARG_KEYS = (
    "num_blocks",
    "num_heads",
    "radius",
    "ffn_mult",
    "ffn_act",
    "gru_layers",
    "gru_bidirectional",
    "position_embed",
    "max_position",
)

_ACTIVATIONS = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
}


class LocalRelativeAttention(nn.Module):
    """Multi-head self-attention with a learned local relative bias.

    Attention is restricted to a window of ``2 * radius + 1`` items around each
    query. The relative bias lives in a single ``[num_heads, 2 * radius + 1]``
    table which is gathered with an integer index matrix, so the module needs no
    per-position control flow and supports a dynamic sequence length.
    """

    def __init__(self, hidden_size, num_heads=4, radius=8, dropout=0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.radius = radius
        self.scale = 1.0 / math.sqrt(self.head_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Stored flat on purpose: a 1-D parameter is not a hidden matrix, so it
        # follows the same optimizer rule as the other 1-D parameters.
        self.relative_bias = nn.Parameter(torch.zeros(num_heads * (2 * radius + 1)))

    def forward(self, x, non_pad_mask):
        """
        :param x: [B, T, C] input sequence
        :param non_pad_mask: [B, T] bool mask, True for real items
        :return: [B, T, C] attended sequence, zero at padded positions
        """
        batch, frames, channels = x.shape
        qkv = self.qkv(x).view(batch, frames, 3, self.num_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, Dh]
        query, key, value = qkv[0], qkv[1], qkv[2]

        score = torch.matmul(query, key.transpose(-1, -2)) * self.scale  # [B, H, T, T]

        index = torch.arange(frames, device=x.device)
        distance = index[:, None] - index[None, :]  # [T, T], positive if i > j
        allowed = distance.abs() <= self.radius
        bias_index = distance.clamp(min=-self.radius, max=self.radius) + self.radius
        bias_table = self.relative_bias.view(self.num_heads, 2 * self.radius + 1)
        score = score + bias_table[:, bias_index]  # broadcast over batch

        window = non_pad_mask[:, None, None, :] & allowed  # [B, 1, T, T]
        score = score.masked_fill(~window, _MASKED_SCORE)
        weight = F.softmax(score, dim=-1)
        weight = self.dropout(weight)

        context = torch.matmul(weight, value)  # [B, H, T, Dh]
        context = context.transpose(1, 2).reshape(batch, frames, channels)
        return self.out_proj(context) * non_pad_mask[:, :, None]


class FFN(nn.Module):
    """Position-wise feed-forward network made of dense matrices only."""

    def __init__(self, hidden_size, ffn_mult=4, act='gelu', dropout=0.0):
        super().__init__()
        if act not in _ACTIVATIONS:
            raise ValueError(f"unsupported activation {act!r}, expected one of {sorted(_ACTIVATIONS)}")
        self.act = act
        self.linear_1 = nn.Linear(hidden_size, ffn_mult * hidden_size)
        self.linear_2 = nn.Linear(ffn_mult * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear_2(_ACTIVATIONS[self.act](self.linear_1(x))))


class Block(nn.Module):
    """Pre-norm block: ``x = x + attn(norm(x))``, ``x = x + ffn(norm(x))``."""

    def __init__(self, hidden_size, num_heads=4, radius=8, ffn_mult=4, act='gelu', dropout=0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_size)
        self.attn = LocalRelativeAttention(hidden_size, num_heads, radius, dropout)
        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.ffn = FFN(hidden_size, ffn_mult, act, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, non_pad_mask):
        mask = non_pad_mask[:, :, None]
        x = (x + self.dropout(self.attn(self.norm_attn(x), non_pad_mask))) * mask
        x = (x + self.ffn(self.norm_ffn(x))) * mask
        return x


class DurationHeadV2(nn.Module):
    """Local relative attention blocks followed by a GRU sequence mixer.

    Params:
        in_dims: input feature width (the conditioning width of the caller)
        hidden_size: internal width, also the width of the returned features
        num_blocks: number of pre-norm attention blocks
        num_heads / radius: attention head count and local window radius
        ffn_mult / ffn_act: feed-forward expansion and activation
        dropout: dropout used by attention, feed-forward and the GRU
        gru_layers / gru_bidirectional: GRU depth and direction
        position_embed: add the forward/reverse within-group position embeddings
            (on by default; requires group ids at every call site)
        max_position: clamp for those embeddings
    """

    def __init__(self, in_dims, hidden_size, num_blocks=4, num_heads=4, radius=8,
                 ffn_mult=4, ffn_act='gelu', dropout=0.0, gru_layers=1,
                 gru_bidirectional=False, position_embed=True, max_position=8):
        super().__init__()
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
        if gru_layers < 1:
            raise ValueError(f"gru_layers must be positive, got {gru_layers}")
        if gru_bidirectional and hidden_size % 2 != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be even for a bidirectional GRU"
            )
        if position_embed and max_position < 0:
            raise ValueError(f"max_position must be non-negative, got {max_position}")
        self.in_dims = in_dims
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.gru_bidirectional = gru_bidirectional
        self.position_embed = position_embed
        self.max_position = max_position
        self.in_proj = nn.Linear(in_dims, hidden_size)
        self.blocks = nn.ModuleList(
            Block(hidden_size, num_heads, radius, ffn_mult, ffn_act, dropout)
            for _ in range(num_blocks)
        )
        self.gru = nn.GRU(
            hidden_size,
            hidden_size // 2 if gru_bidirectional else hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=gru_bidirectional,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        if position_embed:
            self.pos_embed = nn.Embedding(max_position + 1, hidden_size)
            self.reverse_pos_embed = nn.Embedding(max_position + 1, hidden_size)
        else:
            self.pos_embed = None
            self.reverse_pos_embed = None

    @classmethod
    def from_hparams(cls, in_dims, hidden_size, dropout, head_args=None):
        """Build from the ``dur_prediction_args.head_args`` configuration block."""
        args = dict(head_args or {})
        unknown = sorted(set(args) - set(HEAD_ARG_KEYS))
        if unknown:
            raise ValueError(
                f"unknown duration head argument(s): {unknown}; "
                f"supported keys are {list(HEAD_ARG_KEYS)}"
            )
        return cls(
            in_dims=in_dims,
            hidden_size=hidden_size,
            dropout=dropout,
            **{key: args[key] for key in HEAD_ARG_KEYS if key in args},
        )

    def forward(self, x, x_masks=None, group_ids=None):
        """
        :param x: [B, T, C_in] conditioning sequence
        :param x_masks: [B, T] bool mask, True for padded items
        :param group_ids: [B, T] group index per item, 1-based, 0 for padding;
            required by the position embeddings
        :return: [B, T, hidden_size] features for the output projection
        """
        if x_masks is None:
            non_pad_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            non_pad_mask = ~x_masks.bool()
        mask = non_pad_mask[:, :, None]
        hidden = self.in_proj(x)
        if self.pos_embed is not None:
            if group_ids is None:
                raise ValueError(
                    "position_embed requires group_ids to build the within-group positions"
                )
            members = group_mask(group_ids, x_masks=x_masks)
            forward, reverse = group_position_ids(members, self.max_position)
            hidden = hidden + self.pos_embed(forward) + self.reverse_pos_embed(reverse)
        hidden = hidden * mask
        for block in self.blocks:
            hidden = block(hidden, non_pad_mask)
        hidden, _ = self.gru(hidden)
        hidden = hidden * mask
        return self.out_norm(hidden)
