import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import SinusoidalPosEmb, SwiGLU, ATanGLU, Transpose, AdamWLinear
from utils.hparams import hparams


class LYNXNet2PlusBlock(nn.Module):
    def __init__(self, dim, expansion_factor, kernel_size=31, dropout=0., glu_type='swiglu', bias=False):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        if glu_type == 'swiglu':
            self.glu = SwiGLU()
        elif glu_type == 'atanglu':
            self.glu = ATanGLU()
        else:
            raise ValueError(f'{glu_type} is not a valid activation')
        if float(dropout) > 0.:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            Transpose((1, 2))
        )
        self.proj_v = nn.Linear(dim, inner_dim, bias=bias)
        self.proj_gate = nn.Linear(dim, inner_dim, bias=bias)
        
        self.proj_mlp_1 = nn.Linear(inner_dim, inner_dim * 2)
        self.proj_mlp_2 = nn.Linear(inner_dim, dim)

        self._reset_parameters()
        
    def _reset_parameters(self):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

        for proj in [self.proj_v, self.proj_gate, self.proj_mlp_1, self.proj_mlp_2]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.)

        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        v = self.proj_v(x)
        gate_x = self.conv(x)
        gate_logits = self.proj_gate(gate_x)
        x = v * torch.atan(gate_logits)
        x = self.proj_mlp_1(x)
        x = self.glu(x)
        x = self.proj_mlp_2(x)
        x = self.dropout(x)
        
        return residual + x


class LYNXNet2Plus(nn.Module):
    def __init__(self, in_dims, n_feats, *, num_layers=6, num_channels=512, expansion_factor=1, kernel_size=31,
                 dropout_rate=0.0, bias=False, glu_type='swiglu'):
        """
        LYNXNet2Plus(Linear Gated Depthwise Separable Convolution Network Version 2 Plus)
        """
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.input_projection = nn.Linear(in_dims * n_feats, num_channels)
        self.use_conditioner_cache = use_conditioner_cache
        self.conditioner_projection = nn.Conv1d(hparams['hidden_size'], num_channels, 1)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(num_channels),
            nn.Linear(num_channels, num_channels * 4),
            nn.GELU(),
            nn.Linear(num_channels * 4, num_channels),
        )
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2PlusBlock(
                    dim=num_channels,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    dropout=dropout_rate,
                    bias=bias,
                    glu_type=glu_type
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(num_channels)
        self.output_projection = AdamWLinear(num_channels, in_dims * n_feats)
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
        """

        if self.n_feats == 1:
            x = spec[:, 0]  # [B, M, T]
        else:
            x = spec.flatten(start_dim=1, end_dim=2)  # [B, F x M, T]

        x = self.input_projection(x.transpose(1, 2)) # [B, T, F x M]
        x = x + self.conditioner_projection(cond).transpose(1, 2)
        x = x + self.diffusion_embedding(diffusion_step).unsqueeze(1)

        for layer in self.residual_layers:
            x = layer(x)

        # post-norm
        x = self.norm(x)

        # output projection
        x = self.output_projection(x).transpose(1, 2)  # [B, 128, T]

        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            # This is the temporary solution since PyTorch 1.13
            # does not support exporting aten::unflatten to ONNX
            # x = x.unflatten(dim=1, sizes=(self.n_feats, self.in_dims))
            x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
        return x
