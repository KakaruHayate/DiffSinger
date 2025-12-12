from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.onnx.operators
from torch import nn
from torch.nn import LayerNorm, MultiheadAttention, ReLU, GELU, SiLU

import utils


class NormalInitEmbedding(torch.nn.Embedding):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int | None = None,
            *args,
            **kwargs
    ):
        super().__init__(num_embeddings, embedding_dim, *args, padding_idx=padding_idx, **kwargs)
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(self.weight[padding_idx], 0)


class XavierUniformInitLinear(torch.nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            *args,
            bias: bool = True,
            **kwargs
    ):
        super().__init__(in_features, out_features, *args, bias=bias, **kwargs)
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.constant_(self.bias, 0.)


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, x, incremental_state=None, timestep=None, positions=None):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = x.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = utils.make_positions(x, self.padding_idx) if positions is None else positions
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    @staticmethod
    def max_positions():
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class SwiGLU(nn.Module):
    # Swish-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        gate = F.silu(gate)
        if x.dtype == torch.float16:
            out_min, out_max = torch.aminmax(out.detach())
            gate_min, gate_max = torch.aminmax(gate.detach())
            max_abs_out = torch.max(-out_min, out_max).float()
            max_abs_gate = torch.max(-gate_min, gate_max).float()
            max_abs_value = max_abs_out * max_abs_gate
            if max_abs_value > 1000:
                ratio = (1000 / max_abs_value).half()
                gate *= ratio
                return (out * gate).clamp(-1000 * ratio, 1000 * ratio) / ratio
        return out * gate


class ATanGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, gate):
        atan_gate = torch.atan(gate)
        decay_out = out / gate.square().add(1.0)
        ctx.save_for_backward(decay_out, atan_gate)
        return out * atan_gate

    @staticmethod
    def backward(ctx, grad_output):
        decay_out, atan_gate = ctx.saved_tensors
        grad_out_part = grad_output * atan_gate
        grad_gate_part = grad_output * decay_out
        return grad_out_part, grad_gate_part   

       
class ATanGLU(nn.Module):
    # ArcTan-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.        
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        if self.training:
            return ATanGLUFunction.apply(out, gate)
        else:
            return out * torch.atan(gate)
        
        
class KaimingNormalConv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)
        
        
class TransformerFFNLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, kernel_size=1, dropout=0., act='gelu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        filter_size_1 = filter_size
        if self.act == 'relu':
            self.act_fn = ReLU()
        elif self.act == 'gelu':
            self.act_fn = GELU()
        elif self.act == 'swish':
            self.act_fn = SiLU()
        elif self.act == 'swiglu':
            self.act_fn = SwiGLU()
            filter_size_1 = filter_size * 2
        elif self.act == 'atanglu':
            self.act_fn = ATanGLU()
            filter_size_1 = filter_size * 2
        else:
            raise ValueError(f'{act} is not a valid activation')
        self.ffn_1 = nn.Conv1d(hidden_size, filter_size_1, kernel_size, padding=kernel_size // 2)
        self.ffn_2 = XavierUniformInitLinear(filter_size, hidden_size)

    def forward(self, x):
        # x: B x T x C
        x = self.ffn_1(x.transpose(1, 2)).transpose(1, 2)
        x = x * self.kernel_size ** -0.5

        x = self.act_fn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x


import matplotlib.pyplot as plt
import os
import datetime

class MultiheadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=False, rotary_embed=None, use_gate_attn=True, use_qk_norm=True, layer_idx=None):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear layers for Q, K, V projections
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        
        # Final linear layer after concatenation
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Rotary Embeddings
        self.rotary_embed = rotary_embed

        self.use_gate_attn = use_gate_attn
        if self.use_gate_attn:
            # 根据论文结论，最佳配置是 "SDPA Elementwise G1"
            # 这是一个输入依赖的门控：Gate = Sigmoid(X * W_gate)
            # 这里的维度是 embed_dim，实现了 "Head-Specific Elementwise" 的效果
            self.gate_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            # self.atan_sigmoid = AtanSigmoid()
            
            # 初始化门控投影层
            nn.init.xavier_uniform_(self.gate_proj.weight)
            if bias:
                nn.init.constant_(self.gate_proj.bias, 0.0)

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = LayerNorm(embed_dim // num_heads)
            self.k_norm = LayerNorm(embed_dim // num_heads)

        # Initialization parameters
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if bias:
            nn.init.constant_(self.in_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

        debug_viz_path = "./attn_viz_l6"
        self.debug_viz_path = debug_viz_path
        self.layer_idx = layer_idx if layer_idx is not None else "X" # 标记层号
        self.viz_step_count = 0

        if debug_viz_path:
            os.makedirs(debug_viz_path, exist_ok=True)

    def forward(self, x, key_padding_mask=None):
        # x: (B, L, C)
        # key_padding_mask: (B, L)
        batch_size, seq_len, embed_dim = x.size()
        
        # Project inputs to Q, K, V
        Q, K, V = torch.split(self.in_proj(x), self.embed_dim, dim=-1)
        
        # Reshape Q, K, V for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        
        if self.use_qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)
        
        # Apply RoPE
        if self.rotary_embed is not None:
            Q = self.rotary_embed.rotate_queries_or_keys(Q)
            K = self.rotary_embed.rotate_queries_or_keys(K)
            
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (B, H, L, L)

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # Expand mask to match attention scores shape
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
            scores = scores.masked_fill(mask == 1, -np.inf)  # Masked positions are set to -inf

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, L, L)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to V
        attn_output = torch.matmul(attn_weights, V)  # (B, H, L, D)
        
        # Reshape and concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # (B, L, C)
        
        gate_score = None
        if self.use_gate_attn:
            # 论文公式 (5): Y' = Y ⊙ σ(XW_θ)
            # x 是当前层的输入 (Input-Dependent)
            # attn_output 是 SDPA 的输出 (Y)
            # gate_score 是 σ(XW_θ)
            
            # gate_score = self.atan_sigmoid(self.gate_proj(x)) # (B, L, C)
            gate_score = torch.sigmoid(self.gate_proj(x)) # (B, L, C)
            attn_output = attn_output * gate_score

        visualize = False
        # visualize = True
        token_seq = "SP zh/w zh/uo zh/k zh/en zh/s zh/i0 zh/w zh/u zh/j zh/i zh/d zh/an zh/y zh/iong zh/l zh/i zh/p zh/an zh/t zh/an zh/b zh/u zh/p zh/a zh/sh zh/ir zh/c zh/uo zh/f zh/an zh/zh zh/eng zh/y zh/iu zh/h zh/ui zh/x zh/van zh/k zh/e zh/x zh/i zh/r zh/en zh/sh zh/eng zh/n zh/a zh/l zh/i zh/x zh/vn zh/g zh/e zh/d zh/u zh/d zh/ang zh/ch zh/ong zh/l zh/ai zh/j zh/ian zh/x zh/in zh/y zh/iu zh/y0 zh/v zh/er zh/y zh/ian zh/b zh/u zh/y zh/iu zh/zh zh/ong zh/m zh/ei zh/t zh/uo zh/k zh/ou zh/x zh/ian zh/b zh/ei zh/x zh/in zh/t zh/iao zh/j zh/i zh/s zh/an zh/d zh/uo zh/c zh/ong zh/r zh/ong zh/y zh/iu zh/x zh/i zh/x zh/v zh/d zh/uo zh/x zh/van zh/d zh/uan zh/a SP"
        if visualize:
            self._visualize(attn_weights, gate_score, seq_len, embed_dim, token_seq)

        # Final linear projection
        output = self.out_proj(attn_output)  # (B, L, C)
        
        return output
        

    def _visualize(self, attn_weights, gate_score, seq_len, embed_dim, token_seq):
        with torch.no_grad():
            self.viz_step_count += 1
            time_str = datetime.datetime.now().strftime("%H%M%S_%f")
            fig_name = f"step_{self.viz_step_count:04d}_L{self.layer_idx}_{time_str}"
            
            # --- 1. 处理 Token 标签 ---
            labels = None
            if token_seq is not None:
                if isinstance(token_seq, str):
                    raw_tokens = token_seq.strip().split()
                elif isinstance(token_seq, (list, tuple)):
                    raw_tokens = [str(t) for t in token_seq]
                else:
                    raw_tokens = []
                
                # 清洗前缀: "zh/w" -> "w", "en/k" -> "k", "SP" -> "SP"
                labels = [t.split('/')[-1] for t in raw_tokens]
                
                # 防御性截断或填充（防止输入长度与 Tensor 长度不一致报错）
                if len(labels) > seq_len:
                    labels = labels[:seq_len]
                elif len(labels) < seq_len:
                    labels = labels + [""] * (seq_len - len(labels))
            
            # 如果没有提供 token_seq，使用数字索引
            if labels is None:
                labels = [str(i) for i in range(seq_len)]

            # --- 2. 准备数据 ---
            attn_map_avg = attn_weights[0].mean(dim=0).cpu().numpy()
            diag_vals = np.diag(attn_map_avg)
            sink_vals = attn_map_avg[:, 0]
            max_vals = attn_map_avg.max(axis=1)
            
            if self.use_gate:
                gate_vis = gate_score[0].cpu().numpy()
                gate_avg_per_token = gate_vis.mean(axis=-1)
            
            # --- 3. 绘图设置 (放大尺寸) ---
            # 每个子图宽 8，高 8（保证正方形 Map 足够大且清晰）
            num_plots = 4 if self.use_gate else 2
            fig, ax = plt.subplots(1, num_plots, figsize=(8 * num_plots, 8))
            if num_plots == 1: ax = [ax]
            
            # 字体大小设置 (根据序列长度动态调整，防止太挤)
            tick_font_size = 10 if seq_len < 50 else 6
            
            # === 子图 1: Attention Map ===
            im0 = ax[0].imshow(attn_map_avg, aspect='equal', cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
            ax[0].set_title(f"L{self.layer_idx} Attn Map", fontsize=14)
            # 设置坐标轴
            ax[0].set_xticks(np.arange(len(labels)))
            ax[0].set_yticks(np.arange(len(labels)))
            ax[0].set_xticklabels(labels, rotation=90, fontsize=tick_font_size)
            ax[0].set_yticklabels(labels, fontsize=tick_font_size)
            ax[0].set_xlabel("Key (Source)", fontsize=12)
            ax[0].set_ylabel("Query (Target)", fontsize=12)
            plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
            
            # === 子图 2: Attention Values Stats ===
            x_axis = np.arange(seq_len)
            ax[1].plot(x_axis, max_vals, label="Max (Focus)", color='blue', alpha=0.7)
            ax[1].plot(x_axis, diag_vals, label="Self (Diag)", color='green', linestyle='--', alpha=0.7)
            ax[1].plot(x_axis, sink_vals, label="Sink (Idx0)", color='red', linestyle=':', alpha=0.7)
            
            ax[1].set_title(f"L{self.layer_idx} Weights Stats", fontsize=14)
            ax[1].set_xticks(np.arange(len(labels)))
            ax[1].set_xticklabels(labels, rotation=90, fontsize=tick_font_size)
            ax[1].set_ylim(-0.05, 1.05)
            ax[1].grid(True, alpha=0.3)
            ax[1].legend(loc='upper right')
            
            # === Gate 相关子图 ===
            if self.use_gate:
                # 子图 3: Gate Heatmap
                vis_dim = min(embed_dim, 128)
                im2 = ax[2].imshow(gate_vis[:, :vis_dim].T, aspect='auto', cmap='magma', vmin=0, vmax=1)
                ax[2].set_title(f"L{self.layer_idx} Gate Values (Top {vis_dim} Dims)", fontsize=14)
                # X轴是 Token
                ax[2].set_xticks(np.arange(len(labels)))
                ax[2].set_xticklabels(labels, rotation=90, fontsize=tick_font_size)
                ax[2].set_ylabel("Hidden Dim", fontsize=12)
                plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
                
                # 子图 4: Gate Avg Curve
                ax[3].plot(gate_avg_per_token, label="Avg Gate", color='orange', linewidth=2)
                ax[3].set_title(f"L{self.layer_idx} Avg Gate per Token", fontsize=14)
                ax[3].set_xticks(np.arange(len(labels)))
                ax[3].set_xticklabels(labels, rotation=90, fontsize=tick_font_size)
                ax[3].set_ylim(0, 1.1)
                ax[3].grid(True, alpha=0.3)
                
                # 标出均值线
                avg_val = gate_avg_per_token.mean()
                ax[3].axhline(y=avg_val, color='grey', linestyle='--', label=f"Mean: {avg_val:.2f}")
                
                # 标出被抑制得最厉害的 Token (Gate 最小的)
                min_idx = np.argmin(gate_avg_per_token)
                ax[3].scatter([min_idx], [gate_avg_per_token[min_idx]], color='red', zorder=5)
                ax[3].text(min_idx, gate_avg_per_token[min_idx]+0.02, labels[min_idx], color='red', fontsize=10, ha='center')
                
                ax[3].legend()

            plt.tight_layout()
            save_path = os.path.join(self.debug_viz_path, f"{fig_name}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"[Viz] Saved L{self.layer_idx} with tokens to {save_path}")
        

def Conv_Init(
    module: torch.nn.Module,
    w_init_gain: str
    ):
    torch.nn.init.xavier_uniform_(module.weight, gain= torch.nn.init.calculate_gain(w_init_gain))

    if not module.bias is None:
        torch.nn.init.zeros_(module.bias)

    return module

class Correct_Mixed_LayerNorm(torch.nn.LayerNorm):
    def __init__(
        self,
        channels: int,
        condition_channels: int,
        beta_distribution_concentration: float = 0.2,
        eps: float= 1e-5,
        bias: bool= True,
        device= None,
        dtype= None
        ):
        super().__init__(
            normalized_shape= channels,
            eps= eps,
            elementwise_affine= False,
            bias= bias,
            device= device,
            dtype= dtype,
            )
        
        self.beta_distribution = torch.distributions.Beta(
            beta_distribution_concentration,
            beta_distribution_concentration
            )

        self.channels = channels
        self.affine = Conv_Init(torch.nn.Linear(
            in_features= condition_channels,
            out_features= channels * 2,
            bias= True
            ), w_init_gain= 'linear')
        self.affine.bias.data[:channels] = 1
        self.affine.bias.data[channels:] = 0        

    def forward(
        self,
        x: torch.FloatTensor,
        condition: torch.FloatTensor, # -> shape [Batch, Cond_d]
        mixln_mask_embed: torch.FloatTensor = None  # -> shape [Batch, Cond_d]
        ) -> torch.FloatTensor:
        x = super().forward(x)  # [Batch, Time, X_d]
        affine_params = self.affine(condition) # .unsqueeze(1) 
        if affine_params.ndim == 2:
            affine_params = affine_params.unsqueeze(1)
        betas, gammas = torch.split(affine_params, self.channels, dim=-1)

        if not self.training or x.size(0) == 1:
            return gammas * x + betas

        shuffle_indices = torch.randperm(x.size(0), device=x.device)
        shuffled_betas = betas[shuffle_indices]
        shuffled_gammas = gammas[shuffle_indices]
        beta_samples = self.beta_distribution.sample((x.size(0), 1, 1)).to(x.device)
        mixed_betas = beta_samples * betas + (1 - beta_samples) * shuffled_betas
        mixed_gammas = beta_samples * gammas + (1 - beta_samples) * shuffled_gammas

        if mixln_mask_embed is not None:
            replacement_mask = torch.abs(mixln_mask_embed).sum(dim=1) > 0
            if replacement_mask.any():
                mask_affine_params = self.affine(mixln_mask_embed)
                if mask_affine_params.ndim == 2:
                    mask_affine_params = mask_affine_params.unsqueeze(1)
                mask_betas, mask_gammas = torch.split(mask_affine_params, self.channels, dim=-1)
                mixed_betas[replacement_mask] = mask_betas[replacement_mask].to(mixed_betas.dtype)
                mixed_gammas[replacement_mask] = mask_gammas[replacement_mask].to(mixed_betas.dtype)

        return mixed_gammas * x + mixed_betas


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class EncSALayer(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1,
                 relu_dropout=0.1, kernel_size=9, act='gelu', rotary_embed=None, layer_idx=None,
                 use_gate_attn=False, use_qk_norm=False):
        super().__init__()
        self.dropout = dropout
        if layer_idx is not None:
            self.use_mix_ln = (layer_idx in [0, 2])
        else:
            self.use_mix_ln = False
        if self.use_mix_ln:
            self.layer_norm1 = Correct_Mixed_LayerNorm(c, c)
        else:
            self.layer_norm1 = LayerNorm(c)
        if rotary_embed is None:
            self.self_attn = MultiheadAttention(
                c, num_heads, dropout=attention_dropout, bias=False, batch_first=False
            )
            self.use_rope = False
        else:
            self.self_attn = MultiheadSelfAttentionWithRoPE(
                c, num_heads, dropout=attention_dropout, bias=False, rotary_embed=rotary_embed,
                use_gate_attn=use_gate_attn, use_qk_norm=use_qk_norm
            )
            self.use_rope = True
        if self.use_mix_ln:
            self.layer_norm2 = Correct_Mixed_LayerNorm(c, c)
        else:
            self.layer_norm2 = LayerNorm(c)
        self.ffn = TransformerFFNLayer(
            c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, act=act
        )

    def forward(self, x, encoder_padding_mask=None, cond=None, mixln_mask_embed=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        residual = x
        if self.use_mix_ln:
            x = self.layer_norm1(x, cond, mixln_mask_embed)
        else:
            x = self.layer_norm1(x)
        if self.use_rope:
            x = self.self_attn(x, key_padding_mask=encoder_padding_mask)
        else:
            x = x.transpose(0, 1)
            x, _, = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask
            )
            x = x.transpose(0, 1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float())[..., None]

        residual = x
        if self.use_mix_ln:
            x = self.layer_norm2(x, cond, mixln_mask_embed)
        else:
            x = self.layer_norm2(x)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float())[..., None]
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
