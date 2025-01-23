import torch
from torch.nn import functional as F
import torch.nn as nn
import math

from modules.commons.common_layers import SinusoidalPositionalEmbedding, SinusoidalPosEmb, Conv1d
from modules.commons.rotary_embedding_torch import RotaryEmbedding
from modules.commons.espnet_positional_embedding import RelPositionalEncoding
from modules.fastspeech.tts_modules import TransformerEncoderLayer
from utils.hparams import hparams


DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000


class FFT(nn.Module):
    def __init__(self, in_dims, n_feats, *, num_layers=6, num_channels=512, kernel_size=9, ffn_act='gelu',
                 dropout=0., num_heads=2, use_pos_embed=True, rel_pos=True, use_rope=False):
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        hidden_size = self.hidden_size = num_channels
        self.num_layers = num_layers
        embed_dim = self.hidden_size
        self.input_projection = Conv1d(in_dims * n_feats, hidden_size, 1)
        self.diffusion_embedding = SinusoidalPosEmb(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.Mish(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.get_decode_inp = nn.Linear(hparams['hidden_size'] + hidden_size + hidden_size, hidden_size)  # eH + dH + dH -> dH
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        if use_pos_embed and use_rope:
            rotary_embed = RotaryEmbedding(dim = embed_dim // num_heads)
        else:
            rotary_embed = None
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                self.hidden_size, self.dropout,
                kernel_size=kernel_size, act=ffn_act,
                num_heads=num_heads, rotary_embed=rotary_embed
            )
            for _ in range(self.num_layers)
        ])
        
        self.checkpoint_activations = False # 石山爆炸，用不了
        
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.padding_idx = 0
        self.rel_pos = rel_pos
        if use_rope:
            self.embed_positions = None
        elif self.rel_pos:
            self.embed_positions = RelPositionalEncoding(hidden_size, dropout_rate=0.0)
        else:
            self.embed_positions = SinusoidalPositionalEmbedding(
                hidden_size, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS,
            )

        self.get_mel_out = nn.Linear(hidden_size, in_dims * n_feats, bias=True)
        nn.init.zeros_(self.get_mel_out.weight)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            #ic(*inputs.shape)
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward_embedding(self, x, padding_mask=None):
        # embed tokens and positions
        if self.use_pos_embed and self.embed_positions is not None:
            if self.rel_pos:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(~padding_mask)
                x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, spec, diffusion_step, cond, padding_mask=None, attn_mask=None, return_hiddens=False):
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

        x = self.input_projection(x).permute([0, 2, 1])  #  [B, T, residual_channel]

        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)  # [B, dim]

        cond = cond.permute([0, 2, 1])  # [B, T, M]

        seq_len = cond.shape[1]  # [T_mel]
        time_embed = diffusion_step[:, None, :]  # [B, 1, dim]
        time_embed = time_embed.repeat([1, seq_len, 1])  # # [B, T, dim]

        decoder_inp = torch.cat([x, cond, time_embed], dim=-1)  # [B, T, M + H + H]
        decoder_inp = self.get_decode_inp(decoder_inp)  # [B, T, H]
        x = decoder_inp

        '''
        Required x: [B, T, C]
        :return: [B, T, C] or [L, B, T, C]
        '''
        padding_mask = x.abs().sum(-1).eq(0).data if padding_mask is None else padding_mask
        x = self.forward_embedding(x, padding_mask=padding_mask)  # [B, T, H]
        nonpadding_mask_TB = 1 - padding_mask.float()[:, :, None]  # [T, B, 1]
        # B x T x C -> T x B x C
        x = x * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(layer), x, padding_mask, attn_mask)
            else:
                x = layer(x, encoder_padding_mask=padding_mask, attn_mask=attn_mask)
            x = x * nonpadding_mask_TB
            if return_hiddens:
                hiddens.append(x)
        x = self.layer_norm(x) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)  # [L, T, B, C]
        x = self.get_mel_out(x).permute([0, 2, 1])  # [B, M, T]

        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            # This is the temporary solution since PyTorch 1.13
            # does not support exporting aten::unflatten to ONNX
            # x = x.unflatten(dim=1, sizes=(self.n_feats, self.in_dims))
            x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
        return x
