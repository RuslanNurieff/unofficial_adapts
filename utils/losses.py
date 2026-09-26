import torch
import torch.nn.functional as F
from torch import nn

from .anomaly import ch_normalization


def calculate_stfpm_loss(teacher_features, student_features):
    loss = 0
    for layer_name in teacher_features:
        t_feat = ch_normalization(teacher_features[layer_name])
        s_feat = ch_normalization(student_features[layer_name])
        # sum over channels, mean over positions (STFPM / moviad stfpm_loss);
        # F.mse_loss would also divide by C and make this term ~C times weaker
        loss += torch.sum((t_feat - s_feat) ** 2, dim=1).mean()
    return loss


class SegmentationLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, masks):
        masks_resized = F.interpolate(masks, size=logits.shape[-2:], mode="nearest")

        # bce = -log(p_ij), so p_ij = exp(-bce) without taking log of a sigmoid
        bce = F.binary_cross_entropy_with_logits(logits, masks_resized, reduction="none")
        p_ij = torch.exp(-bce)
        focal_loss = torch.mean(((1 - p_ij) ** self.gamma) * bce)

        l1_loss = F.l1_loss(torch.sigmoid(logits), masks_resized)

        return focal_loss + l1_loss
