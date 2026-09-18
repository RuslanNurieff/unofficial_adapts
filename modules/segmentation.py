import torch.nn as nn


class SegmentationModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels, out_channels=1, kernel_size=1
        )
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.sigmoid_layer(x)

        return x
