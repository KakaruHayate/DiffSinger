import torch
import torch.nn as nn
from torch import Tensor


class RectifiedFlowLoss(nn.Module):
    def __init__(self, loss_type, log_norm=True):
        super().__init__()
        self.loss_type = loss_type
        self.log_norm = log_norm
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError()

    @staticmethod
    def _mask_non_padding(v_pred, v_gt, non_padding=None):
        if non_padding is not None:
            non_padding = non_padding.transpose(1, 2).unsqueeze(1)
            return v_pred * non_padding, v_gt * non_padding
        else:
            return v_pred, v_gt

    @staticmethod
    def get_weights(t):
        eps = 1e-7
        t = t.float()
        t = torch.clip(t, 0 + eps, 1 - eps)
        weights = 0.398942 / t / (1 - t) * torch.exp(
            -0.5 * torch.log(t / (1 - t)) ** 2
        ) + eps
        return weights[:, None, None, :]

    def _forward(self, v_pred, v_gt, t=None):
        if self.log_norm:
            return self.get_weights(t) * self.loss(v_pred, v_gt)
        else:
            return self.loss(v_pred, v_gt)

    def forward(self, v_pred: Tensor, v_gt: Tensor, t: Tensor, non_padding: Tensor = None) -> Tensor:
        """
        :param v_pred: [B, F, R, T]
        :param v_gt: [B, F, R, T]
        :param t: [B, 1] or [B, T]
        :param non_padding: [B, T, M]
        """
        v_pred, v_gt = self._mask_non_padding(v_pred, v_gt, non_padding)
        return self._forward(v_pred, v_gt, t=t).mean()

    def forward_best_bin(
            self, v_pred: Tensor, v_gt: Tensor, t: Tensor,
            non_padding: Tensor = None,
    ) -> Tensor:
        """Best-of-R over the repeat-bin dimension with zero extra compute.

        RepetitiveRectifiedFlow already assigns each bin an independent noise,
        so the R bins are R natural Forward XM candidates. This selects the
        winning bin per (batch, feature) and backpropagates only through it.
        """
        v_pred, v_gt = self._mask_non_padding(v_pred, v_gt, non_padding)
        loss = self._forward(v_pred, v_gt, t=t)  # [B, F, R, T]

        per_bin = loss.sum(dim=-1)  # [B, F, R]
        if non_padding is not None:
            mask = non_padding.transpose(1, 2).unsqueeze(1).to(loss)
            mask_sum = mask.sum(dim=-1)  # [B, 1, 1]
            mask_sum = mask_sum.expand_as(per_bin).clamp_min(1.)
            per_bin = per_bin / mask_sum

        best_indices = per_bin.argmin(dim=-1, keepdim=True)  # [B, F, 1]
        best_indices = best_indices.unsqueeze(-1).expand(-1, -1, -1, v_gt.shape[-1])
        best_pred = v_pred.gather(2, best_indices).squeeze(2)
        best_gt = v_gt.gather(2, best_indices).squeeze(2)

        best_loss = self.loss(best_pred, best_gt)
        if self.log_norm:
            best_loss = self.get_weights(t).squeeze(2) * best_loss
        if non_padding is not None:
            best_loss = best_loss * non_padding.transpose(1, 2)
        return best_loss.mean()
