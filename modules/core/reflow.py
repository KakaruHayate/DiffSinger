from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from modules.backbones import build_backbone
from utils.hparams import hparams

# [Cite: 10] To bridge this gap, we propose k-Diff, a framework that employs a data-driven approach to learn the optimal prediction parameter k directly from data.

class RectifiedFlow(nn.Module):
    def __init__(self, out_dims, num_feats=1, t_start=0., time_scale_factor=1000,
                 backbone_type=None, backbone_args=None,
                 spec_min=None, spec_max=None):
        super().__init__()
        self.velocity_fn: nn.Module = build_backbone(out_dims, num_feats, backbone_type, backbone_args)
        self.out_dims = out_dims
        self.num_feats = num_feats
        self.use_shallow_diffusion = hparams.get('use_shallow_diffusion', False)
        
        if self.use_shallow_diffusion:
            assert 0. <= t_start <= 1., 'T_start should be in [0, 1].'
        else:
            t_start = 0.
        self.t_start = t_start
        self.time_scale_factor = time_scale_factor

        # ==================== k-Diff Experimental Setup ====================
        # [Cite: 194] We define a trainable scalar w_k such that k = sigmoid(w_k).
        # Initialize to 0.0 -> sigmoid(0.0) = 0.5 (Starts as v-prediction)
        self.k_param = nn.Parameter(torch.zeros(1))
        # ===================================================================

        # spec: [B, T, M] or [B, F, T, M]
        spec_min = torch.FloatTensor(spec_min)[None, None, :out_dims].transpose(-3, -2)
        spec_max = torch.FloatTensor(spec_max)[None, None, :out_dims].transpose(-3, -2)
        self.register_buffer('spec_min', spec_min, persistent=False)
        self.register_buffer('spec_max', spec_max, persistent=False)

    @property
    def k_value(self):
        """Returns the current actual k value (0 to 1)."""
        return torch.sigmoid(self.k_param)

    def p_losses(self, x_end, t, cond):
        """
        Calculates the loss for k-Diff.
        x_end: Clean Data (x_1)
        """
        # x_start is Noise (x_0)
        x_start = torch.randn_like(x_end)
        
        # Forward diffusion: z_t = t * x_1 + (1-t) * x_0
        # Corresponds to [Cite: 187 Algorithm 1]: z = t * x + (1-t) * e
        x_t = x_start + t[:, None, None, None] * (x_end - x_start)
        
        # Model predicts 'u' instead of 'v'
        u_pred = self.velocity_fn(x_t, t * self.time_scale_factor, cond)

        # Calculate Target u
        # [Cite: 173] u = kx - (1-k)n
        # Here: x_end is Data (x), x_start is Noise (n)
        k = self.k_value
        u_target = k * x_end - (1 - k) * x_start

        return u_pred, u_target

    def forward(self, condition, gt_spec=None, src_spec=None, noise=None, infer=True):
        cond = condition.transpose(1, 2)
        b, device = condition.shape[0], condition.device

        if not infer:
            # Training Mode
            # gt_spec: [B, T, M] or [B, F, T, M]
            spec = self.norm_spec(gt_spec).transpose(-2, -1)
            if self.num_feats == 1:
                spec = spec[:, None, :, :]
            
            t = self.t_start + (1.0 - self.t_start) * torch.rand((b,), device=device)
            u_pred, u_target = self.p_losses(spec, t, cond=cond)
            
            # Note: The caller (training loop) will calculate MSE(u_pred, u_target)
            return u_pred, u_target, t
        else:
            # Inference Mode
            if src_spec is not None:
                spec = self.norm_spec(src_spec).transpose(-2, -1)
                if self.num_feats == 1:
                    spec = spec[:, None, :, :]
            else:
                spec = None
            
            # Print current k for monitoring during inference
            # print(f"Current k-Diff value: {self.k_value.item():.4f}")
            
            x = self.inference(cond, b=b, x_end=spec, noise=noise, device=device)
            return self.denorm_spec(x)

    def get_velocity(self, x, t, cond):
        """
        Converts model prediction 'u' to velocity 'v' for the ODE solver.
        Based on [Cite: 187 Algorithm 2] and [Cite: 235] (clamping).
        """
        # Ensure t is a tensor for calculations
        if isinstance(t, (float, int)):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)

        # 1. Get model output (u_pred)
        u_pred = self.velocity_fn(x, t * self.time_scale_factor, cond)
        
        # 2. Get current k
        k = self.k_value

        # 3. Convert u -> v
        # Formula: v = ((1 - 2k) * z + u) / (k(1-t) + (1-k)t)
        # Here 'x' is the current noisy state 'z'
        numerator = (1 - 2 * k) * x + u_pred
        
        # Denominator with numerical stability clamp
        # [Cite: 235] clamp the denominator... to a minimum value of 0.05
        denominator = k * (1 - t) + (1 - k) * t
        denominator = denominator.clamp(min=0.05)

        v_pred = numerator / denominator
        return v_pred

    @torch.no_grad()
    def sample_euler(self, x, t, dt, cond):
        # Modified to use get_velocity
        v = self.get_velocity(x, t, cond)
        x += v * dt
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk2(self, x, t, dt, cond):
        # Modified to use get_velocity
        k_1 = self.get_velocity(x, t, cond)
        k_2 = self.get_velocity(x + 0.5 * k_1 * dt, t + 0.5 * dt, cond)
        x += k_2 * dt
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk4(self, x, t, dt, cond):
        # Modified to use get_velocity
        k_1 = self.get_velocity(x, t, cond)
        k_2 = self.get_velocity(x + 0.5 * k_1 * dt, t + 0.5 * dt, cond)
        k_3 = self.get_velocity(x + 0.5 * k_2 * dt, t + 0.5 * dt, cond)
        k_4 = self.get_velocity(x + k_3 * dt, t + dt, cond)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk5(self, x, t, dt, cond):
        # Modified to use get_velocity
        k_1 = self.get_velocity(x, t, cond)
        k_2 = self.get_velocity(x + 0.25 * k_1 * dt, t + 0.25 * dt, cond)
        k_3 = self.get_velocity(x + 0.125 * (k_2 + k_1) * dt, t + 0.25 * dt, cond)
        k_4 = self.get_velocity(x + 0.5 * (-k_2 + 2 * k_3) * dt, t + 0.5 * dt, cond)
        k_5 = self.get_velocity(x + 0.0625 * (3 * k_1 + 9 * k_4) * dt, t + 0.75 * dt, cond)
        k_6 = self.get_velocity(x + (-3 * k_1 + 2 * k_2 + 12 * k_3 - 12 * k_4 + 8 * k_5) * dt / 7,
                               t + dt,
                               cond)
        x += (7 * k_1 + 32 * k_3 + 12 * k_4 + 32 * k_5 + 7 * k_6) * dt / 90
        t += dt
        return x, t

    @torch.no_grad()
    def inference(self, cond, b=1, x_end=None, noise=None, device=None):
        if noise is None:
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
                # We pass the current time tensor (t_start + i * dt) to the sampler
                # This is important for k-Diff velocity conversion which depends on t
                current_time = t_start + i * dts
                x, _ = algorithm_fn(x, current_time, dt, cond)
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
        assert (isinstance(vmin, (float, int)) and isinstance(vmin, (float, int))) or len(vmin) == len(vmax)
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