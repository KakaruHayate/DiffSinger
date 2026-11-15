"""Bandlimited (windowed-sinc) resampling implemented with PyTorch only.

This exists so that code which just needs to convert a waveform to another sample
rate does not have to depend on `torchaudio`, which is not part of DiffSinger's
declared requirements and is often missing from torch-only installs.

The math mirrors `torchaudio.functional.resample` (`sinc_interp_hann` method) so the
output is numerically equivalent to `torchaudio.transforms.Resample` with the same
arguments, but every tensor operation is plain `torch`, which means it also works on
accelerators without a custom extension.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _get_sinc_resample_kernel(
        orig_freq: int,
        new_freq: int,
        gcd: int,
        *,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, int]:
    """Build the polyphase interpolation kernel and its half-width in input samples."""
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd
    if lowpass_filter_width <= 0:
        raise ValueError('Low pass filter width should be positive.')

    # Antialiasing: drop everything above the Nyquist frequency of the slower rate.
    base_freq = min(orig_freq, new_freq) * rolloff

    # y[j] = sum_i x[i] * sinc(pi * orig_freq * (i / orig_freq - j / new_freq)), truncated to
    # `lowpass_filter_width` zero crossings; new_freq consecutive outputs share the same set of
    # weights, which is why the convolution below can use a stride of orig_freq.
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)

    idx_dtype = dtype if dtype is not None else torch.float64
    idx = torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=device)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1, dtype=dtype, device=device)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp(-lowpass_filter_width, lowpass_filter_width)

    # Hann window evaluated at the (non-uniform) kernel positions instead of on a regular grid.
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2

    t *= math.pi
    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0, dtype=t.dtype, device=t.device), t.sin() / t)
    kernels *= window * scale
    if dtype is None:
        kernels = kernels.to(torch.float32)
    return kernels, width


def _apply_sinc_resample_kernel(
        waveform: torch.Tensor,
        orig_freq: int,
        new_freq: int,
        gcd: int,
        kernel: torch.Tensor,
        width: int,
) -> torch.Tensor:
    if not waveform.is_floating_point():
        raise TypeError(f'Expected floating point waveform, but received {waveform.dtype}.')
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])
    num_wavs, length = waveform.shape
    waveform = F.pad(waveform, (width, width + orig_freq))
    resampled = F.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = math.ceil(new_freq * length / orig_freq)
    resampled = resampled[..., :target_length]
    return resampled.view(shape[:-1] + resampled.shape[-1:])


def resample_waveform(
        waveform: torch.Tensor,
        orig_freq: int,
        new_freq: int,
        *,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
) -> torch.Tensor:
    """Resample `waveform` (..., time) from `orig_freq` to `new_freq`.

    The kernel is rebuilt on every call; for repeated calls with identical arguments use the
    `Resample` module below, which caches it per (dtype, device).
    """
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError('Original frequency and desired frequency should be positive')
    if orig_freq == new_freq:
        return waveform
    gcd = math.gcd(int(orig_freq), int(new_freq))
    kernel, width = _get_sinc_resample_kernel(
        orig_freq, new_freq, gcd,
        lowpass_filter_width=lowpass_filter_width, rolloff=rolloff,
        device=waveform.device, dtype=waveform.dtype
    )
    return _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width)


class Resample(torch.nn.Module):
    """Drop-in replacement for `torchaudio.transforms.Resample` (bandlimited sinc, Hann window).

    Like torchaudio's module, the kernel is pre-computed in `__init__` (in `float64` unless `dtype`
    is given) and registered as a buffer, so `module.to(device)` moves it and repeated calls reuse it.
    """

    def __init__(
            self,
            orig_freq: int,
            new_freq: int,
            *,
            lowpass_filter_width: int = 6,
            rolloff: float = 0.99,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.orig_freq = int(orig_freq)
        self.new_freq = int(new_freq)
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff
        self.gcd = math.gcd(self.orig_freq, self.new_freq)
        self.width = 0
        if self.orig_freq != self.new_freq:
            kernel, self.width = _get_sinc_resample_kernel(
                self.orig_freq, self.new_freq, self.gcd,
                lowpass_filter_width=self.lowpass_filter_width, rolloff=self.rolloff, dtype=dtype
            )
            self.register_buffer('kernel', kernel)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.orig_freq == self.new_freq:
            return waveform
        kernel = self.kernel
        if kernel.device != waveform.device or kernel.dtype != waveform.dtype:
            # Tolerate a module that was not moved with the input (torchaudio would raise here).
            kernel = kernel.to(device=waveform.device, dtype=waveform.dtype)
        return _apply_sinc_resample_kernel(waveform, self.orig_freq, self.new_freq, self.gcd, kernel, self.width)

    def extra_repr(self) -> str:
        return (
            f'orig_freq={self.orig_freq}, new_freq={self.new_freq}, '
            f'lowpass_filter_width={self.lowpass_filter_width}, rolloff={self.rolloff}'
        )
