import torch
import torch.nn.functional as F
from torch import nn

from .anomaly import ch_normalization


def calculate_stfpm_loss(teacher_features, student_features):
    loss = 0
    for layer_name in teacher_features:
        t_feat = ch_normalization(teacher_features[layer_name])
        s_feat = ch_normalization(student_features[layer_name])
        loss += F.mse_loss(s_feat, t_feat)
    return loss


class SegmentationLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, preds, masks):
        masks_resized = F.interpolate(masks, size=preds.shape[-2:], mode="nearest")

        l1_loss = F.l1_loss(preds, masks_resized)

        p_ij = masks_resized * preds + (1 - masks_resized) * (1 - preds)

        focal_loss = -torch.mean(((1 - p_ij) ** self.gamma) * torch.log(p_ij + 1e-8))

        return focal_loss + l1_loss
