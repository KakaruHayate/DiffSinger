import torch
import torch.nn as nn
import torch.nn.functional as F


class CumMax(nn.Module):
    def __init__(self, window_size=512):
        super().__init__()
        self.window_size = window_size
        self.max_pool = nn.MaxPool1d(kernel_size=window_size, stride=1, padding=0)
        self.neg_inf = -1e4

    def forward(self, x):
        # x shape: [Batch, Heads, Seq, Head_Dim]
        b, h, s, d = x.shape
        x_reshaped = x.permute(0, 1, 3, 2).reshape(b * h, d, s)
        pad_len = self.window_size - 1
        x_padded = F.pad(x_reshaped, (pad_len, 0), value=self.neg_inf)
        out = self.max_pool(x_padded)

        out = out.view(b, h, d, s).permute(0, 1, 3, 2)
        
        return out


class BiMaxStateAttention(nn.Module):
    def __init__(self, dim_size, num_heads, window_size=512):
        super().__init__()
        assert dim_size % num_heads == 0
        self.num_heads = num_heads
        
        self.combined = nn.Linear(dim_size, 4 * dim_size, bias=False)
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))
        
        self.cummax = CumMax(window_size=window_size)

    def gen_model(self, a, b, c, d, e):
        term1 = a * b
        term2 = self.alpha1 * b + self.alpha2 * d
        term3 = a * (self.alpha3 * e + d)
        term4 = b * (c + e)
        return term1 + term2 + term3 + term4 + c * e

    def forward(self, x):
        b, s, d = x.shape
        combined = self.combined(x).view(b, s, 4, self.num_heads, -1)
        a, b_in, c, d_in = combined.unbind(2)

        a = a.permute(0, 3, 1, 2)
        b_in = b_in.permute(0, 3, 1, 2)
        c = c.permute(0, 3, 1, 2)
        d_in = d_in.permute(0, 3, 1, 2)

        state_fwd = self.cummax(c)

        c_rev = torch.flip(c, dims=[2])
        state_bwd_rev = self.cummax(c_rev)
        state_bwd = torch.flip(state_bwd_rev, dims=[2])

        e = (state_fwd + state_bwd) / 2.0

        out = self.gen_model(a, b_in, c, d_in, e)
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return out


class GatedFeedForward(nn.Module):
    def __init__(self, hidden_size, expansion_factor=1):
        super().__init__()
        inner_dim = hidden_size * expansion_factor
        self.ffn1 = nn.Linear(hidden_size, inner_dim)
        self.gate = nn.Linear(hidden_size, inner_dim)
        self.ffn2 = nn.Linear(inner_dim, hidden_size)
        self.act = nn.ReLU()

    def forward(self, x):
        x1 = self.ffn1(x)
        x2 = self.act(self.gate(x))
        out = self.ffn2(x1 * x2)
        return out


class SamOutEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_expansion=1, window_size=2048):
        super().__init__()
        self.attn = BiMaxStateAttention(hidden_size, num_heads, window_size=window_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = GatedFeedForward(hidden_size, expansion_factor=ffn_expansion)
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        res = x
        x = self.attn(self.norm1(x))
        x = self.norm2(self.alpha * self.ffn(x) + (1 - self.alpha) * res)
        
        return x


if __name__ == "__main__":
    # 配置
    DIM = 256
    HEADS = 8
    SEQ_LEN = 128
    
    # 实例化 (window_size 设大一点以模拟 cummax)
    model = SamOutEncoderLayer(hidden_size=DIM, num_heads=HEADS, window_size=2048)
    model.eval() # 导出时必须是 eval 模式

    x = torch.randn(1, SEQ_LEN, DIM)

    # 导出测试
    try:
        torch.onnx.export(
            model,
            x,
            "samout_fixed.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=19, # 11 以上通常就支持 Pad+MaxPool
            dynamic_axes={"input": {1: "seq_len"}, "output": {1: "seq_len"}}
        )
        print("ONNX export successful! (Fixed using MaxPool1d)")
    except Exception as e:
        print(f"Export failed: {e}")
