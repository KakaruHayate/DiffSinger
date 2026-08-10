import torch
import torch.nn as nn

from modules.commons.common_layers import (
    ATanGLU,
    AdamWConv1d,
    AdamWLinear,
    DoubleSoftSignGLU,
    SinusoidalPosEmb,
    SoftSignGLU,
    SwiGLU,
    Transpose,
)
from utils.hparams import hparams


def _shift(x, off):
    """Shift along the last (time) axis: out[..., i] = x[..., i + off], zero-padded."""
    if off == 0:
        return x
    out = torch.zeros_like(x)
    if off > 0:
        out[..., :-off] = x[..., off:]
    else:
        out[..., -off:] = x[..., :off]
    return out


def separated_depthwise_conv(x, conv, mask):
    """Frame-group separated depthwise conv (conv equivalent of Attention Separation).

    :param x: [B, C, T] input, ready for a depthwise Conv1d over time.
    :param conv: depthwise Conv1d with groups == in_channels (e.g. AdamWConv1d).
    :param mask: [B, T] frame-group indicator in {0, 1}. Frame t aggregates only
        neighbors in the same group (mask[t] == mask[neighbor]); cross-group
        contributions are removed exactly.

    By linearity of convolution: y = conv(x) - sum_k w[k] * x[t+off] * 1[mask differs].
    mask is None -> plain conv (identical numerics, zero overhead).
    """
    if mask is None:
        return conv(x)
    B, C, T = x.shape
    K = conv.kernel_size[0]
    pad = K // 2
    y = conv(x)                                    # full conv output [B, C, T]
    w = conv.weight.squeeze(1)                     # depthwise: [C, K]
    for k in range(K):
        off = k - pad
        cross = (_shift(mask, off) != mask).float().unsqueeze(1)   # [B, 1, T]
        y = y - w[:, k].view(1, C, 1) * _shift(x, off) * cross
    return y


class LYNXNet2SepBlock(nn.Module):
    def __init__(self, dim, expansion_factor, kernel_size=31, dropout=0., glu_type='swiglu',
                 separate_frames=True):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        if glu_type == 'swiglu':
            _glu = SwiGLU()
        elif glu_type == 'atanglu':
            _glu = ATanGLU()
        elif glu_type == 'softsign_glu':
            _glu = SoftSignGLU()
        elif glu_type == 'double_softsign_glu':
            _glu = DoubleSoftSignGLU()
        else:
            raise ValueError(f'{glu_type} is not a valid activation')
        if float(dropout) > 0.:
            _dropout = nn.Dropout(dropout)
        else:
            _dropout = nn.Identity()
        # Keep the exact module layout/order of LYNXNet2Block (net.0..net.9) so that
        # state_dict is interchangeable with LYNXNet2. forward() only intercepts the
        # depthwise conv when a frame-group mask is provided.
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            AdamWConv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2),
            _glu,
            nn.Linear(inner_dim, inner_dim * 2),
            _glu,
            nn.Linear(inner_dim, dim),
            _dropout
        )
        self.separate_frames = separate_frames

    def forward(self, x, mask=None):
        if self.separate_frames and mask is not None:
            h = self.net[0](x)                       # LayerNorm
            h = self.net[1](h)                       # Transpose -> [B, C, T]
            h = separated_depthwise_conv(h, self.net[2], mask)
            for layer in self.net[3:]:               # Transpose, GLU stack, dropout
                h = layer(h)
            return x + h
        return x + self.net(x)


class LYNXNet2Sep(nn.Module):
    def __init__(self, in_dims, n_feats, *, num_layers=6, num_channels=512, expansion_factor=1, kernel_size=31,
                 dropout_rate=0.0, use_conditioner_cache=False, glu_type='swiglu', separate_frames=True):
        """
        LYNXNet2Sep(Linear Gated Depthwise Separable Convolution Network Version 2, frame-separated)

        Copy of LYNXNet2 with an optional frame-group separation inside the depthwise
        conv: when a dual-timestep mask is provided during training, each output frame
        aggregates only same-group neighbors, cutting cross-group (cross-noise-level)
        frame interactions - the convolutional analogue of Attention Separation
        (arXiv:2607.02508). Inference (mask=None) is numerically identical to LYNXNet2.
        """
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.input_projection = nn.Linear(in_dims * n_feats, num_channels)
        self.use_conditioner_cache = use_conditioner_cache
        if self.use_conditioner_cache:
            # Conv1d is used for condition cache compatibility
            self.conditioner_projection = nn.Conv1d(hparams['hidden_size'], num_channels, 1)
        else:
            self.conditioner_projection = nn.Linear(hparams['hidden_size'], num_channels)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(num_channels),
            nn.Linear(num_channels, num_channels * 4),
            nn.GELU(),
            nn.Linear(num_channels * 4, num_channels),
        )
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2SepBlock(
                    dim=num_channels,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    dropout=dropout_rate,
                    glu_type=glu_type,
                    separate_frames=separate_frames,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(num_channels)
        self.output_projection = AdamWLinear(num_channels, in_dims * n_feats)
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond, diffusion_step_2=None, mask=None):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :param diffusion_step_2: [B, 1] (dual-timestep)
        :param mask: [B, T] frame-group indicator in {0, 1} (dual-timestep)
        :return:
        """

        if self.n_feats == 1:
            x = spec[:, 0]  # [B, M, T]
        else:
            x = spec.flatten(start_dim=1, end_dim=2)  # [B, F x M, T]

        x = self.input_projection(x.transpose(1, 2)) # [B, T, F x M]
        if self.use_conditioner_cache:
            x = x + self.conditioner_projection(cond).transpose(1, 2)
        else:
            x = x + self.conditioner_projection(cond.transpose(1, 2))

        if mask is not None:
            step = torch.cat((diffusion_step, diffusion_step_2), dim=0)
            step = self.diffusion_embedding(step)
            step, step_2 = torch.split(step, x.shape[0], dim=0) #[B, 1, C]
            frame_mask = mask.to(x) # [B, T]
            x = x + step + (step_2 - step) * frame_mask.unsqueeze(-1)
        else:
            frame_mask = None
            step = self.diffusion_embedding(diffusion_step)
            if step.dim() == 2:
                step = step.unsqueeze(1)
            x = x + step

        for layer in self.residual_layers:
            x = layer(x, frame_mask)

        # post-norm
        x = self.norm(x)

        # output projection
        x = self.output_projection(x).transpose(1, 2)  # [B, 128, T]

        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            # Using reshape instead of unflatten for ONNX export compatibility
            # x = x.unflatten(dim=1, sizes=(self.n_feats, self.in_dims))
            x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
        return x
