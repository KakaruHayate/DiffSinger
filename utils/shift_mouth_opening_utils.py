import torch
from torch import Tensor


def calculate_shifted_opec(opec_gt: Tensor, alpha: Tensor,
                           o_min: float, o_max: float) -> Tensor:
    shifted_open = opec_gt + alpha * (o_max - opec_gt)
    shifted_close = opec_gt + alpha * (opec_gt - o_min)
    shifted = torch.where(alpha >= 0, shifted_open, shifted_close)
    return torch.clamp(shifted, min=o_min, max=o_max)


def sample_truncated_normal(shape, sigma: float, lo: float, hi: float,
                            device, generator: torch.Generator = None) -> Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    total = 1
    for s in shape:
        total *= s
    out = torch.empty(total, device=device)
    filled = 0
    while filled < total:
        need = total - filled
        x = torch.randn(max(need * 2, 16), device=device, generator=generator) * sigma
        x = x[(x >= lo) & (x <= hi)]
        take = min(x.numel(), need)
        if take > 0:
            out[filled:filled + take] = x[:take]
            filled += take
    return out.view(*shape)
