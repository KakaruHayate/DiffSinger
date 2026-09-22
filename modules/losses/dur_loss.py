import torch
import torch.nn as nn
from torch import Tensor

from modules.fastspeech.grouping import group_distribution


class DurationLoss(nn.Module):
    """
    Loss module as combination of phone duration loss, group allocation loss,
    word duration loss and sentence duration loss.

    The allocation term treats the phones of a group (a note or syllable) as a
    distribution over the group's frame budget and compares it with the target
    distribution with a cross entropy. It is scale invariant inside a group, so
    the word and sentence terms are the ones that keep the absolute frame scale
    meaningful; keep them non-zero unless the caller supplies the budget itself.
    """

    def __init__(self, offset, loss_type,
                 lambda_pdur=0.6, lambda_wdur=0.3, lambda_sdur=0.1, lambda_alloc=0.0):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'huber':
            self.loss = nn.HuberLoss()
        else:
            raise NotImplementedError()
        self.offset = offset

        self.lambda_pdur = lambda_pdur
        self.lambda_wdur = lambda_wdur
        self.lambda_sdur = lambda_sdur
        self.lambda_alloc = lambda_alloc

    def linear2log(self, any_dur):
        return torch.log(any_dur + self.offset)

    def forward(self, dur_pred: Tensor, dur_gt: Tensor, ph2word: Tensor) -> Tensor:
        dur_pred = dur_pred.clamp(min=0.)  # clip to avoid NaN loss

        dur_gt = dur_gt.to(dtype=dur_pred.dtype)

        # allocation loss
        alloc_loss = 0.
        if self.lambda_alloc > 0.:
            prob_pred = group_distribution(dur_pred, ph2word)
            prob_gt = group_distribution(dur_gt, ph2word)
            log_prob = prob_pred.clamp_min(1e-8).log()
            token_loss = -(prob_gt * log_prob) * (ph2word > 0)
            n_tokens = (ph2word > 0).sum().clamp_min(1)
            alloc_loss = self.lambda_alloc * token_loss.sum() / n_tokens

        # pdur_loss
        pdur_loss = self.lambda_pdur * self.loss(self.linear2log(dur_pred), self.linear2log(dur_gt))

        # wdur loss
        shape = dur_pred.shape[0], ph2word.max() + 1
        wdur_pred = dur_pred.new_zeros(*shape).scatter_add(
            1, ph2word, dur_pred
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        wdur_gt = dur_gt.new_zeros(*shape).scatter_add(
            1, ph2word, dur_gt
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        wdur_loss = self.lambda_wdur * self.loss(self.linear2log(wdur_pred), self.linear2log(wdur_gt))

        # sdur loss
        sdur_pred = dur_pred.sum(dim=1)
        sdur_gt = dur_gt.sum(dim=1)
        sdur_loss = self.lambda_sdur * self.loss(self.linear2log(sdur_pred), self.linear2log(sdur_gt))

        # combine
        dur_loss = alloc_loss + pdur_loss + wdur_loss + sdur_loss

        return dur_loss
