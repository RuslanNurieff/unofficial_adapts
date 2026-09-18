import torch
from torch import Tensor, nn
from torchvision import models


class Backbone(nn.Module):
    def __init__(self, max_block: int = 3):
        super().__init__()

        self.OUTPUT_CHANNELS = {1: 256, 2: 512, 3: 1024, 4: 2048}

        net = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        )  # following AdapTS paper, but in future could be changed to dynamic archs

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.blocks = nn.ModuleList(
            getattr(net, f"layer{i}") for i in range(1, max_block + 1)
        )
        self.requires_grad_(False)
        self.eval()

    def forward(self, x: Tensor, adapters: dict):
        first, last = min(adapters), max(adapters)
        teacher, student = {}, {}

        with torch.no_grad():
            h = self.stem(x)
            for i in range(1, last + 1):
                h = self.blocks[i - 1](h)
                if i in adapters:
                    teacher[i] = h

        s = teacher[first]
        for i in range(first, last + 1):
            if i > first:
                s = self.blocks[i - 1](s)
            if i in adapters:
                s = adapters[i](s)
                student[i] = s

        return teacher, student
