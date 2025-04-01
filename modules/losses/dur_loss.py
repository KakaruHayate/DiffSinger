import torch
import torch.nn as nn
from torch import Tensor


class DurationLoss(nn.Module):
    """
    Loss module as combination of phone duration loss, word duration loss and sentence duration loss.
    """

    def __init__(self, offset, loss_type,
                 lambda_pdur=0.6, lambda_wdur=0.3, lambda_sdur=0.1):
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

    def linear2log(self, any_dur):
        return torch.log(any_dur + self.offset)

    def forward(self, dur_pred: Tensor, dur_gt: Tensor, ph2word: Tensor) -> Tensor:
        dur_gt = dur_gt.to(dtype=dur_pred.dtype)

        # pdur_loss
        pdur_loss = self.lambda_pdur * self.loss(self.linear2log(dur_pred), self.linear2log(dur_gt))

        dur_pred = dur_pred.clamp(min=0.)  # clip to avoid NaN loss

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
        dur_loss = pdur_loss + wdur_loss + sdur_loss

        return dur_loss


class DurGanLoss(nn.Module):
    def __init__(self, loss_mode='D_loss'):
        super().__init__()
        self.loss_mode = loss_mode

    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        r_loss = 0
        g_loss = 0
        batch = 0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            dr = dr.float()
            dg = dg.float()
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg**2)
            r_loss += r_loss
            g_loss += g_loss
            batch += 1

        return r_loss / batch, g_loss / batch

    def generator_loss(self, disc_outputs):
        loss = 0
        batch = 0
        for dg in disc_outputs:
            dg = dg.float()
            l = torch.mean((1 - dg) ** 2)
            loss += l
            batch += 1

        return loss / batch

    def forward(self, disc_outputs, disc_generated_outputs):
        if self.loss_mode == 'D_loss':
            r_loss, g_loss = self.discriminator_loss(disc_outputs, disc_generated_outputs)
            return r_loss, g_loss
        elif self.loss_mode == 'G_loss':
            loss = self.generator_loss(disc_generated_outputs)
            return loss
