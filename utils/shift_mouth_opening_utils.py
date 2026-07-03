import math

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
    """Sample from N(0, sigma^2) truncated to [lo, hi].

    Uses inverse-CDF (erfinv) for O(1) sampling — no rejection,
    deterministic number of GPU kernel launches. Mathematically
    equivalent to the previous rejection sampler but ~2x faster in
    practice and free of edge cases when the rejection rate is high.

    Args:
        shape: int or tuple; tensor shape.
        sigma: stddev of the underlying Gaussian.
        lo: lower truncation bound (inclusive).
        hi: upper truncation bound (inclusive).
        device: torch device.
        generator: optional torch.Generator for reproducibility.
    """
    if isinstance(shape, int):
        shape = (shape,)
    total = 1
    for s in shape:
        total *= s

    # CDF of N(0, sigma^2) at x: 0.5 * (1 + erf(x / (sigma * sqrt(2))))
    inv_sqrt2 = 1.0 / (sigma * math.sqrt(2.0))
    cdf_lo = 0.5 * (1.0 + math.erf(lo * inv_sqrt2))
    cdf_hi = 0.5 * (1.0 + math.erf(hi * inv_sqrt2))

    u = torch.rand(total, device=device, generator=generator) * (cdf_hi - cdf_lo) + cdf_lo
    # Inverse CDF: x = sigma * sqrt(2) * erfinv(2u - 1)
    x = sigma * math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    return x.view(*shape)

