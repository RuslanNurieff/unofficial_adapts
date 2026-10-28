import torch

from .anomaly import ch_normalization


def stfpm_layer_losses(teacher_features, student_features):
    """STFPM loss of each layer, keyed like the feature dicts."""
    losses = {}
    for layer_name in teacher_features:
        t_feat = ch_normalization(teacher_features[layer_name])
        s_feat = ch_normalization(student_features[layer_name])
        # sum over channels, mean over positions (STFPM / moviad stfpm_loss);
        # F.mse_loss would also divide by C and make this term ~C times weaker
        losses[layer_name] = torch.sum((t_feat - s_feat) ** 2, dim=1).mean()
    return losses


def calculate_stfpm_loss(teacher_features, student_features):
    return sum(stfpm_layer_losses(teacher_features, student_features).values())
