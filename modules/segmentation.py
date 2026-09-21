from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.anomaly import feature_differences


class SegmentationModule(nn.Module):
    def __init__(self, channels: Mapping[int, int]):
        super().__init__()
        self.channels = dict(channels)
        self.in_channels = int(sum(self.channels.values()))
        self.conv_layer = nn.Conv2d(
            in_channels=self.in_channels, out_channels=1, kernel_size=1
        )

    def infer_size(self, teacher: Mapping[int, torch.Tensor]):
        sizes = [teacher[s].shape[-2:] for s in self.channels]
        height = max(s[0] for s in sizes)
        width = max(s[1] for s in sizes)
        return height, width

    def forward(
        self, teacher: Mapping[int, torch.Tensor], student: Mapping[int, torch.Tensor]
    ):
        interpolation_size = tuple(self.infer_size(teacher))

        diffs = []
        for block in self.channels:
            d = feature_differences(teacher[block], student[block])
            if tuple(d.shape[-2:]) != interpolation_size:
                d = F.interpolate(
                    d, interpolation_size, mode="bilinear", align_corners=False
                )
            diffs.append(d)

        return torch.sigmoid(self.conv_layer(torch.cat(diffs, dim=1)))
