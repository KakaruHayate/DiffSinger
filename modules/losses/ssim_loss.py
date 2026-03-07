import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian(window_size, sigma):
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIMLoss(nn.Module):
    def __init__(self, loss_weight=0.01, window_size=11):
        super().__init__()
        self.loss_weight = loss_weight
        self.window_size = window_size
        self.channel = 1 
        self.register_buffer('window', create_window(window_size, self.channel))

    def _align_mask(self, non_padding, target_tensor):
        if non_padding is None:
            return None

        mask = non_padding.clone()
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, T] -> [B, 1, 1, T]
        elif mask.dim() == 3:
            if mask.shape[1] == target_tensor.shape[3]: 
                mask = mask.transpose(1, 2).unsqueeze(1) # [B, T, M] -> [B, 1, M, T]
            else:
                mask = mask.unsqueeze(1) # [B, M, T] -> [B, 1, M, T]

        return mask.expand_as(target_tensor)

    def forward(self, pred, target, non_padding=None):
        pred = torch.clamp((pred + 1.0) / 2.0, min=0.0, max=1.0)
        target = torch.clamp((target + 1.0) / 2.0, min=0.0, max=1.0)

        pad = self.window_size // 2
        
        mu1 = F.conv2d(pred, self.window, padding=pad, groups=self.channel)
        mu2 = F.conv2d(target, self.window, padding=pad, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.clamp(F.conv2d(pred * pred, self.window, padding=pad, groups=self.channel) - mu1_sq, min=0.0)
        sigma2_sq = torch.clamp(F.conv2d(target * target, self.window, padding=pad, groups=self.channel) - mu2_sq, min=0.0)
        sigma12 = F.conv2d(pred * target, self.window, padding=pad, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        mask = self._align_mask(non_padding, ssim_map)

        if mask is not None:
            ssim_map = ssim_map * mask
            valid_elements = mask.sum()
            ssim_val = ssim_map.sum() / (valid_elements + 1e-8)
        else:
            ssim_val = ssim_map.mean()

        loss = 1.0 - ssim_val

        return loss * self.loss_weight
