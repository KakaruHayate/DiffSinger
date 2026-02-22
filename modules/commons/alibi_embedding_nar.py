import torch
import math
from torch import Tensor

def get_alibi_slopes(num_heads: int) -> list:
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    slopes = [base ** (i + 1) for i in range(closest_power_of_2)]
    
    if closest_power_of_2 < num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        extra_slopes = [extra_base ** (i + 1) for i in range(2 * closest_power_of_2)][0::2][:num_heads - closest_power_of_2]
        slopes.extend(extra_slopes)
    return slopes

class ALiBiEmbedding(torch.nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

        slopes = torch.tensor(get_alibi_slopes(num_heads), dtype=torch.float32).view(1, num_heads, 1, 1)
        self.register_buffer('slopes', slopes, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # x : [batch, heads, seq_len, seq_len] or [batch_size, seq_len, hidden]
        seq_len = x.shape[-2]
        arange = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        # For NAR, the distance must be symmetric: -|i - j|
        distances = -torch.abs(arange[:, None] - arange[None, :])
        distances = distances[None, None, :, :]
        alibi_bias = self.slopes.to(x.dtype) * distances
        
        return alibi_bias
