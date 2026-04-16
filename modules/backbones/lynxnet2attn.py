import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import SinusoidalPosEmb, SwiGLU, ATanGLU, Transpose, AdamWLinear
from utils.hparams import hparams


class GlobalMixer(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        reduction_dim = max(dim // reduction, 16)
        self.mlp = nn.Sequential(
            nn.Linear(dim, reduction_dim),
            nn.SiLU(inplace=True),
            nn.Linear(reduction_dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        # x: (Batch, Seq_Len, Dim) | mask: (Batch, Seq_Len, 1)
        if mask is not None:
            x_masked = x * mask 
            valid_len = mask.sum(dim=1, keepdim=True).clamp(min=1e-5) 
            global_mean = x_masked.sum(dim=1, keepdim=True) / valid_len
        else:
            global_mean = x.mean(dim=1, keepdim=True)
            
        global_gate = self.mlp(global_mean)
        return global_gate


class LocalMixer(nn.Module):
    def __init__(self, dim, kernel_size=31):
        super().__init__()
        self.local_mixer = nn.Sequential(
            Transpose((1, 2)), 
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            Transpose((1, 2))  
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        return self.local_mixer(x)


class SpatialGatedAttention(nn.Module):
    """
    Spatial Gated Attention (SGA) with decoupled Value and Gate projections.
    Modulates local structural features with global contextual priors.
    
    Formula:
        V = XW_v
        C_local = DWConv(X)
        C_global = Sigmoid(MLP(GAP(X)))
        G = ATan((C_local * C_global) * W_g)
        Output = (V * G)W_o
    """
    def __init__(self, dim, kernel_size=31, reduction=4, dropout=0., bias=False, use_global_mixer=False):
        super().__init__()
        self.dropout = dropout
        self.use_global_mixer = use_global_mixer
        
        if self.use_global_mixer:
            self.global_mixer = GlobalMixer(dim, reduction=reduction)
        self.local_mixer = LocalMixer(dim, kernel_size=kernel_size)
        
        self.proj_v = nn.Linear(dim, dim, bias=bias)
        self.proj_gate = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for proj in [self.proj_v, self.proj_gate, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.)

    def forward(self, x, mask=None):
        V = self.proj_v(x)
        local_mix = self.local_mixer(x)  # (B, L, D)
        
        if self.use_global_mixer:
            global_mix = self.global_mixer(x, mask=mask)  # (B, 1, D)
            fused_mix = local_mix * global_mix
        else:
            fused_mix = local_mix
        
        gate_logits = self.proj_gate(fused_mix)
        attn_weights = torch.atan(gate_logits)
        
        output = self.out_proj(attn_weights * V)
        output = F.dropout(output, self.dropout, training=self.training)
        
        return output


class MLP(nn.Module):
    def __init__(self, hidden_size, expansion_factor=2, dropout=0., glu_type='swiglu'):
        super().__init__()
        self.glu_type = glu_type
        self.dropout = dropout

        if self.glu_type == 'swiglu':
            self.glu = SwiGLU()
        elif self.glu_type == 'atanglu':
            self.glu = ATanGLU()
        else:
            raise ValueError(f'{glu_type} is not a valid activation')
            
        hidden_size_1 = hidden_size * 2
        self.ffn_1 = nn.Linear(hidden_size, hidden_size_1 * expansion_factor)
        self.ffn_2 = nn.Linear(hidden_size * expansion_factor, hidden_size)

    def forward(self, x):
        x = self.ffn_1(x)
        x = self.glu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x


class LYNXNet2AttnBlock(nn.Module):
    """
    A single block of LYNXNet2 combining Spatial Gated Attention and Channel MLP.
    """
    def __init__(self, dim, expansion_factor, kernel_size=31, reduction=4, dropout=0., 
                 attn_dropout=0., ffn_dropout=0., bias=False, use_global_mixer=False, glu_type='swiglu'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = SpatialGatedAttention(
            dim, kernel_size, reduction=reduction, dropout=attn_dropout, 
            bias=bias, use_global_mixer=use_global_mixer
        )
        self.mlp = MLP(dim, expansion_factor, dropout=ffn_dropout, glu_type=glu_type)
        self.dropout = dropout

    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)
        
        x = self.attn(x, mask=mask)
        x = self.mlp(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        return residual + x


class LYNXNet2Attn(nn.Module):
    """
    LYNXNet2Attn: Diffusion backbone utilizing Linear Gated Depthwise Separable Convolutions
    and Global-Local Context Fusion, achieving O(N) complexity for high-res generation.
    """
    def __init__(self, in_dims, n_feats, *, num_layers=6, num_channels=512, expansion_factor=4, kernel_size=31,
                 reduction=4, dropout_rate=0.0, attn_dropout=0.0, ffn_dropout=0.0, attn_bias=False, 
                 use_global_mixer=False, glu_type='swiglu'):
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.input_projection = nn.Linear(in_dims * n_feats, num_channels)
        self.conditioner_projection = nn.Conv1d(hparams['hidden_size'], num_channels, 1)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(num_channels),
            nn.Linear(num_channels, num_channels * 4),
            nn.GELU(),
            nn.Linear(num_channels * 4, num_channels),
        )
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2AttnBlock(
                    dim=num_channels,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    reduction=reduction,
                    dropout=dropout_rate,
                    attn_dropout=attn_dropout, 
                    ffn_dropout=ffn_dropout, 
                    bias=attn_bias, 
                    use_global_mixer=use_global_mixer, 
                    glu_type=glu_type
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(num_channels)
        self.output_projection = AdamWLinear(num_channels, in_dims * n_feats)
        
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond, mask=None):
        if self.n_feats == 1:
            x = spec[:, 0]  
        else:
            x = spec.flatten(start_dim=1, end_dim=2)  

        x = self.input_projection(x.transpose(1, 2)) 
        x = x + self.conditioner_projection(cond).transpose(1, 2)
        x = x + self.diffusion_embedding(diffusion_step).unsqueeze(1)

        for layer in self.residual_layers:
            x = layer(x, mask)

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
        