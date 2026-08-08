"""
MDN (Mixture Density Network) for DiffSinger duration prediction.

Replaces the single-Gaussian regression head of the FastSpeech2 duration
predictor with a Gaussian mixture. Training minimizes NLL (negative
log-likelihood) of the ground-truth log-duration; at inference the mean of the
most probable component is returned deterministically, so the ONNX duration
interface (`dur.onnx`: {encoder_out, x_masks, ph_midi} -> {ph_dur_pred})
remains unchanged (still a single scalar per phoneme).

Numerical conventions follow
https://github.com/nnsvs/nnsvs/blob/master/nnsvs/mdn.py
(log-domain GMM, clamped log-sigma/log-pi, logsumexp NLL).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_2PI = math.log(2.0 * math.pi)


def logsumexp(x, dim=-1, keepdim=True):
    """Numerically stable log-sum-exp over `dim`.

    :param x: input tensor.
    :param dim: reduction dimension.
    :param keepdim: whether to keep the reduced dimension.
    :return: log(sum(exp(x), dim)) with the same keepdim semantics.
    """
    m, _ = x.max(dim=dim, keepdim=True)
    m = torch.clamp(m, min=-1e30)
    out = m + torch.log(torch.exp(x - m).sum(dim=dim, keepdim=True))
    return out if keepdim else out.squeeze(dim)


def log_softmax(x, dim=-1):
    """Log-softmax via logsumexp subtraction.

    :param x: input logits.
    :param dim: class dimension.
    :return: log-softmax values with the same shape as x.
    """
    return x - logsumexp(x, dim=dim, keepdim=True)


class MDNLayer(nn.Module):
    """Gaussian-mixture output head.

    x: (B, T, F)  ->  (log_pi, log_sigma, mu) each of shape (B, T, G).
    """

    def __init__(self, in_features, num_gaussians=8,
                 log_p_min=-7.0, log_sigma_min=-7.0, sigma_floor=1e-6,
                 log_scale_max=6.0, log_scale_min=-1.0):
        super().__init__()
        self.in_features = in_features
        self.num_gaussians = num_gaussians
        self.log_p_min = log_p_min
        self.log_sigma_min = log_sigma_min
        self.sigma_floor = sigma_floor
        self.log_scale_max = log_scale_max
        self.log_scale_min = log_scale_min

        self.hidden = nn.Linear(in_features, in_features)
        self.act = nn.Tanh()
        self.out = nn.Linear(in_features, 3 * num_gaussians)

        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.out.weight, std=0.02)
        with torch.no_grad():
            n = num_gaussians
            b = self.out.bias
            b[:n] = 0.0                 # pi logits: uniform baseline
            b[n:2 * n] = 0.0            # log-sigma -> sigma ~ exp(0) = 1.0
            b[2 * n:] = torch.linspace(-0.5, 0.5, n)  # mu spread

    def forward(self, x):
        """Map input features to bounded GMM parameters.

        :param x: [B, T, F] backbone features.
        :return: tuple (logit_p, log_sigma, mu), each [B, T, G], with
            logit_p clamped at log_p_min and mu clamped to [log_scale_min,
            log_scale_max] so the linear-domain duration cannot diverge.
        """
        z = self.act(self.hidden(x))
        logit_p, log_sigma, mu = self.out(z).chunk(3, dim=-1)
        logit_p = torch.clamp(logit_p, min=self.log_p_min)
        mu = torch.clamp(mu, min=self.log_scale_min, max=self.log_scale_max)
        return logit_p, log_sigma, mu

    # -- distribution utils --------------------------------------------------
    def log_pi(self, logit_p):
        """Log mixture weights from component logits.

        :param logit_p: [B, T, G] component logits.
        :return: log-softmax weights [B, T, G].
        """
        return log_softmax(logit_p, dim=-1)

    def sigma(self, log_sigma):
        """Standard deviation with configured lower clamp and floor.

        :param log_sigma: [B, T, G] log-std values.
        :return: positive sigma [B, T, G].
        """
        ls = torch.clamp(log_sigma, min=self.log_sigma_min)
        return torch.exp(ls) + self.sigma_floor

    def log_prob(self, logit_p, log_sigma, mu, target):
        """log p(target) under the mixture. target: [B, T], return [B, T]."""
        lp = self.log_pi(logit_p)                  # [B, T, G]
        s = self.sigma(log_sigma)                  # [B, T, G]
        t = target.unsqueeze(-1)                   # [B, T, 1]
        log_n = (-0.5 * (torch.log(s * s) + _LOG_2PI)
                 - (t - mu) ** 2 / (2 * s * s))    # [B, T, G]
        return logsumexp(log_n + lp, dim=-1, keepdim=False)  # [B, T]

    def point_estimate(self, logit_p, log_sigma, mu):
        """Deterministic point: mean of most-probable component. [B, T]."""
        p = F.softmax(logit_p, dim=-1)
        idx = p.argmax(dim=-1, keepdim=True)       # [B, T, 1]
        m = torch.gather(mu, dim=-1, index=idx).squeeze(-1)  # log-domain mu
        return m


def mdn_nll_loss(mdn_layer, logit_p, log_sigma, mu, target, masks=None):
    """Per-phoneme NLL (log domain), delegating density math to the layer.

    :param mdn_layer: MDNLayer instance (its configured clamps are used).
    :param logit_p, log_sigma, mu: mixture parameters from MDNLayer.forward.
    :param target: log-domain durations [B, T].
    :param masks: [B, T] bool tensor where True marks padding (excluded from loss).
    :return: per-phoneme NLL [B, T].
    """
    nll = -mdn_layer.log_prob(logit_p, log_sigma, mu, target)  # [B, T]
    if masks is not None:
        nll = nll * (~masks).float()
    return nll
