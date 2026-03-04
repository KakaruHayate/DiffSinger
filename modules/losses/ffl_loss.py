import torch
import torch.nn as nn

class MelFocalFrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0, spec_min=-14.0, spec_max=4.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.spec_max = spec_max
        self.spec_min = spec_min

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    @staticmethod
    def _mask_non_padding(self, pred, target, non_padding=None):
        if non_padding is not None:
            non_padding = non_padding.transpose(1, 2).unsqueeze(1)
            return pred * non_padding, target * non_padding, non_padding
        else:
            return pred, target, None

    def forward(self, pred, target, non_padding=None):
        """
        param:
            pred: [B, 1, M, T]
            target: [B, 1, M, T]
            non_padding: [B, T, M]
        """
        pred = self.denorm_spec(pred)
        target = self.denorm_spec(target)

        pred, target, non_padding = self._mask_non_padding(pred, target, non_padding)

        pred_freq = torch.fft.fft2(pred, norm='ortho')
        target_freq = torch.fft.fft2(target, norm='ortho')
        
        diff_freq = pred_freq - target_freq
        freq_distance = torch.abs(diff_freq) ** 2
        
        weight_matrix = torch.abs(diff_freq) ** self.alpha
        
        max_weight = weight_matrix.amax(dim=(-2, -1), keepdim=True)
        weight_matrix = weight_matrix / (max_weight + 1e-8)
        weight_matrix = torch.clamp(weight_matrix, min=0.0, max=1.0)
        
        weight_matrix = weight_matrix.detach()
        
        loss = weight_matrix * freq_distance
        
        if non_padding is not None:
            valid_elements = non_padding.sum()
            return (loss.sum() / (valid_elements + 1e-8)) * self.loss_weight
        else:
            return torch.mean(loss) * self.loss_weight
