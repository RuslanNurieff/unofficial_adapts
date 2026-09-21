from __future__ import annotations

from collections.abc import Mapping

import torch.nn as nn


def build_adapter(in_channels: int, ratio: float):
    return Adapter(in_channels=in_channels, ratio=ratio)


class Adapter(nn.Module):
    def __init__(self, in_channels: int, ratio: float):
        super().__init__()
        self.latent_channels = (
            int(in_channels * ratio) if ratio >= 1 else max(1, int(in_channels * ratio))
        )

        self.conv1 = nn.Conv2d(in_channels, self.latent_channels, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.latent_channels)
        self.act = nn.ReLU(True)
        self.conv2 = nn.Conv2d(self.latent_channels, in_channels, 1, bias=True)

    def forward(self, x):
        return x + self.conv2(self.act(self.bn1(self.conv1(x))))


class AdapterSet(nn.Module):
    def __init__(self, channels: Mapping[int, int], ratio: float):
        super().__init__()
        self.adapters = nn.ModuleDict(
            {str(i): build_adapter(ic, ratio) for i, ic in channels.items()}
        )

    def __getitem__(self, idx: int):
        return self.adapters[str(idx)]

    def __contains__(self, idx: int):
        return str(idx) in self.adapters

    def as_dict(self):
        return {int(k): v for k, v in self.adapters.items()}
