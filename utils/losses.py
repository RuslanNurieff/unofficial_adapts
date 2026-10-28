import torch

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
