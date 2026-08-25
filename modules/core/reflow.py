from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from modules.backbones import build_backbone
from utils.hparams import hparams


class RectifiedFlow(nn.Module):
    def __init__(self, out_dims, num_feats=1, t_start=0., time_scale_factor=1000,
                 backbone_type=None, backbone_args=None,
                 spec_min=None, spec_max=None):
        super().__init__()
        self.velocity_fn: nn.Module = build_backbone(out_dims, num_feats, backbone_type, backbone_args)
        self.out_dims = out_dims
        self.num_feats = num_feats
        self.use_dual_timestep = hparams.get('use_dual_timestep', False)
        self.use_shallow_diffusion = hparams.get('use_shallow_diffusion', False)
        if self.use_shallow_diffusion:
            assert 0. <= t_start <= 1., 'T_start should be in [0, 1].'
        else:
            t_start = 0.
        self.t_start = t_start
        self.time_scale_factor = time_scale_factor

        # spec: [B, T, M] or [B, F, T, M]
        # spec_min and spec_max: [1, 1, M] or [1, 1, F, M] => transpose(-3, -2) => [1, 1, M] or [1, F, 1, M]
        spec_min = torch.FloatTensor(spec_min)[None, None, :out_dims].transpose(-3, -2)
        spec_max = torch.FloatTensor(spec_max)[None, None, :out_dims].transpose(-3, -2)
        self.register_buffer('spec_min', spec_min, persistent=False)
        self.register_buffer('spec_max', spec_max, persistent=False)

    def _sample_train_timesteps(self, batch_size, num_frames, device):
        """Sample one (or a dual pair of) timesteps plus a per-frame mask.

        Mirrors the sampling distribution used by ``forward`` so that
        Explorative Modeling explores the exact same objective as the baseline:
        ``t1`` lives in ``[T_start, 1]`` (shallow diffusion start) and, when
        ``use_dual_timestep``, a second ``t2`` and a per-frame 25%-density mask
        are sampled.
        """
        t1 = self.t_start + (1.0 - self.t_start) * torch.rand((batch_size, 1), device=device)
        if self.use_dual_timestep:
            t2 = self.t_start + (1.0 - self.t_start) * torch.rand((batch_size, 1), device=device)
            mask = (torch.rand(batch_size, num_frames, device=device) < 0.25).float()
        else:
            t2 = None
            mask = None
        return t1, t2, mask

    def prepare_training_inputs(
            self, gt_spec, t1=None, t2=None, mask=None, noise=None,
            num_frames=None
    ):
        """Normalize a ground-truth target and (optionally) timestep/noise.

        :return: ``(spec, t1, t2, mask, noise)`` where ``spec`` is the
            normalized target in the shape the denoiser expects (``[B, F, M, T]``).
            Timesteps/``mask`` are supplied by the caller when known (Explorative
            Modeling samplesthem once and replays the winner) and sampled here
            otherwise using the same distribution as ``forward``. ``noise`` is the
            start-of-flow noise used for the interpolation.
        """
        spec = self.norm_spec(gt_spec).transpose(-2, -1)
        if self.num_feats == 1:
            spec = spec[:, None, :, :]
        batch_size = spec.shape[0]
        num_frames = spec.shape[-1] if num_frames is None else num_frames
        device = spec.device
        if t1 is None:
            t1, t2, mask = self._sample_train_timesteps(batch_size, num_frames, device)
        else:
            t1 = t1.to(spec)
            if t2 is not None:
                t2 = t2.to(spec)
            if mask is not None:
                mask = mask.to(spec)
        if noise is None:
            noise = torch.randn_like(spec)
        else:
            noise = noise.to(spec)
        return spec, t1, t2, mask, noise

    def p_losses(self, x_end, t1, cond, t2=None, mask=None, noise=None):
        t = t1 if mask is None else t1 + (t2 - t1) * mask
        x_start = torch.randn_like(x_end) if noise is None else noise
        # ``t`` is [B, 1] (single step) or [B, T] (dual, per-frame); the
        # ``[:, None, None, :]`` index broadcasts t over the channel/bins dims.
        x_t = x_start + t[:, None, None, :] * (x_end - x_start)
        s1 = t1 * self.time_scale_factor
        s2 = None if t2 is None else t2 * self.time_scale_factor
        v_pred = self.velocity_fn(x_t, s1, cond, s2, mask)

        return v_pred, x_end - x_start, t

    def training_forward(
            self, condition, gt_spec, t1=None, t2=None, mask=None, noise=None
    ):
        """Training entry point that keeps timestep/noise boundaries explicit.

        Explorative Modeling uses this to evaluate many candidate noises without
        gradients and then replay the winning candidate (same timesteps + noise)
        with gradients so upstream condition encoders remain trainable. When the
        timesteps/``noise`` are omitted, the standard single-flow objective is
        produced (identical distribution to ``forward``).
        """
        cond = condition.transpose(1, 2)
        spec, t1, t2, mask, noise = self.prepare_training_inputs(
            gt_spec, t1=t1, t2=t2, mask=mask, noise=noise
        )
        v_pred, v_gt, t = self.p_losses(spec, t1, cond=cond, t2=t2, mask=mask, noise=noise)
        return v_pred, v_gt, t

    def forward(self, condition, gt_spec=None, src_spec=None, infer=True):
        cond = condition.transpose(1, 2)
        b, _, n_frames = cond.shape
        device = condition.device

        if not infer:
            return self.training_forward(condition, gt_spec)
        else:
            # src_spec: [B, T, M] or [B, F, T, M]
            if src_spec is not None:
                spec = self.norm_spec(src_spec).transpose(-2, -1)
                if self.num_feats == 1:
                    spec = spec[:, None, :, :]
            else:
                spec = None
            x = self.inference(cond, b=b, x_end=spec, device=device)
            return self.denorm_spec(x)

    @torch.no_grad()
    def sample_euler(self, x, t, dt, cond):
        x += self.velocity_fn(x, self.time_scale_factor * t, cond) * dt
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk2(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, self.time_scale_factor * t, cond)
        k_2 = self.velocity_fn(x + 0.5 * k_1 * dt, self.time_scale_factor * (t + 0.5 * dt), cond)
        x += k_2 * dt
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk4(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, self.time_scale_factor * t, cond)
        k_2 = self.velocity_fn(x + 0.5 * k_1 * dt, self.time_scale_factor * (t + 0.5 * dt), cond)
        k_3 = self.velocity_fn(x + 0.5 * k_2 * dt, self.time_scale_factor * (t + 0.5 * dt), cond)
        k_4 = self.velocity_fn(x + k_3 * dt, self.time_scale_factor * (t + dt), cond)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk5(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, self.time_scale_factor * t, cond)
        k_2 = self.velocity_fn(x + 0.25 * k_1 * dt, self.time_scale_factor * (t + 0.25 * dt), cond)
        k_3 = self.velocity_fn(x + 0.125 * (k_2 + k_1) * dt, self.time_scale_factor * (t + 0.25 * dt), cond)
        k_4 = self.velocity_fn(x + 0.5 * (-k_2 + 2 * k_3) * dt, self.time_scale_factor * (t + 0.5 * dt), cond)
        k_5 = self.velocity_fn(x + 0.0625 * (3 * k_1 + 9 * k_4) * dt, self.time_scale_factor * (t + 0.75 * dt), cond)
        k_6 = self.velocity_fn(x + (-3 * k_1 + 2 * k_2 + 12 * k_3 - 12 * k_4 + 8 * k_5) * dt / 7,
                               self.time_scale_factor * (t + dt),
                               cond)
        x += (7 * k_1 + 32 * k_3 + 12 * k_4 + 32 * k_5 + 7 * k_6) * dt / 90
        t += dt
        return x, t

    @torch.no_grad()
    def inference(self, cond, b=1, x_end=None, device=None):
        noise = torch.randn(b, self.num_feats, self.out_dims, cond.shape[2], device=device)
        t_start = hparams.get('T_start_infer', self.t_start)
        if self.use_shallow_diffusion and t_start > 0:
            assert x_end is not None, 'Missing shallow diffusion source.'
            if t_start >= 1.:
                t_start = 1.
                x = x_end
            else:
                x = t_start * x_end + (1 - t_start) * noise
        else:
            t_start = 0.
            x = noise

        algorithm = hparams['sampling_algorithm']
        infer_step = hparams['sampling_steps']

        if t_start < 1:
            dt = (1.0 - t_start) / max(1, infer_step)
            algorithm_fn = {
                'euler': self.sample_euler,
                'rk2': self.sample_rk2,
                'rk4': self.sample_rk4,
                'rk5': self.sample_rk5,
            }.get(algorithm)
            if algorithm_fn is None:
                raise ValueError(f'Unsupported algorithm for Rectified Flow: {algorithm}.')
            dts = torch.tensor([dt]).to(x)
            for i in tqdm(range(infer_step), desc='sample time step', total=infer_step,
                          disable=not hparams['infer'], leave=False):
                x, _ = algorithm_fn(x, t_start + i * dts, dt, cond)
            x = x.float()
        x = x.transpose(2, 3).squeeze(1)  # [B, F, M, T] => [B, T, M] or [B, F, T, M]
        return x

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


class RepetitiveRectifiedFlow(RectifiedFlow):
    def __init__(self, vmin: float | int | list, vmax: float | int | list,
                 repeat_bins: int, time_scale_factor=1000,
                 backbone_type=None, backbone_args=None):
        assert (isinstance(vmin, (float, int)) and isinstance(vmax, (float, int))) or len(vmin) == len(vmax)
        num_feats = 1 if isinstance(vmin, (float, int)) else len(vmin)
        spec_min = [vmin] if num_feats == 1 else [[v] for v in vmin]
        spec_max = [vmax] if num_feats == 1 else [[v] for v in vmax]
        self.repeat_bins = repeat_bins
        super().__init__(
            out_dims=repeat_bins, num_feats=num_feats,
            time_scale_factor=time_scale_factor,
            backbone_type=backbone_type, backbone_args=backbone_args,
            spec_min=spec_min, spec_max=spec_max
        )

    def norm_spec(self, x):
        """

        :param x: [B, T] or [B, F, T]
        :return [B, T, R] or [B, F, T, R]
        """
        if self.num_feats == 1:
            repeats = [1, 1, self.repeat_bins]
        else:
            repeats = [1, 1, 1, self.repeat_bins]
        return super().norm_spec(x.unsqueeze(-1).repeat(repeats))

    def denorm_spec(self, x):
        """

        :param x: [B, T, R] or [B, F, T, R]
        :return [B, T] or [B, F, T]
        """
        return super().denorm_spec(x).mean(dim=-1)


class PitchRectifiedFlow(RepetitiveRectifiedFlow):
    def __init__(self, vmin: float, vmax: float,
                 cmin: float, cmax: float, repeat_bins,
                 time_scale_factor=1000,
                 backbone_type=None, backbone_args=None):
        self.vmin = vmin  # norm min
        self.vmax = vmax  # norm max
        self.cmin = cmin  # clip min
        self.cmax = cmax  # clip max
        super().__init__(
            vmin=vmin, vmax=vmax, repeat_bins=repeat_bins,
            time_scale_factor=time_scale_factor,
            backbone_type=backbone_type, backbone_args=backbone_args
        )

    def norm_spec(self, x):
        return super().norm_spec(x.clamp(min=self.cmin, max=self.cmax))

    def denorm_spec(self, x):
        return super().denorm_spec(x).clamp(min=self.cmin, max=self.cmax)


class MultiVarianceRectifiedFlow(RepetitiveRectifiedFlow):
    def __init__(
            self, ranges: List[Tuple[float, float]],
            clamps: List[Tuple[float | None, float | None] | None],
            repeat_bins, time_scale_factor=1000,
            backbone_type=None, backbone_args=None
    ):
        assert len(ranges) == len(clamps)
        self.clamps = clamps
        vmin = [r[0] for r in ranges]
        vmax = [r[1] for r in ranges]
        if len(vmin) == 1:
            vmin = vmin[0]
        if len(vmax) == 1:
            vmax = vmax[0]
        super().__init__(
            vmin=vmin, vmax=vmax, repeat_bins=repeat_bins,
            time_scale_factor=time_scale_factor,
            backbone_type=backbone_type, backbone_args=backbone_args
        )

    def clamp_spec(self, xs: list | tuple):
        clamped = []
        for x, c in zip(xs, self.clamps):
            if c is None:
                clamped.append(x)
                continue
            clamped.append(x.clamp(min=c[0], max=c[1]))
        return clamped

    def norm_spec(self, xs: list | tuple):
        """
        :param xs: sequence of [B, T]
        :return: [B, F, T] => super().norm_spec(xs) => [B, F, T, R]
        """
        assert len(xs) == self.num_feats
        clamped = self.clamp_spec(xs)
        xs = torch.stack(clamped, dim=1)  # [B, F, T]
        if self.num_feats == 1:
            xs = xs.squeeze(1)  # [B, T]
        return super().norm_spec(xs)

    def denorm_spec(self, xs):
        """
        :param xs: [B, T, R] or [B, F, T, R] => super().denorm_spec(xs) => [B, T] or [B, F, T]
        :return: sequence of [B, T]
        """
        xs = super().denorm_spec(xs)
        if self.num_feats == 1:
            xs = [xs]
        else:
            xs = xs.unbind(dim=1)
        assert len(xs) == self.num_feats
        return self.clamp_spec(xs)
