import torch
import torch.nn as nn  # noqa

from modules.adapters import AdapterSet
from modules.backbone import Backbone
from modules.segmentation import SegmentationModule


class AdapTS(nn.Module):
    def __init__(self, adapter_layers: list[int], adapter_ratio: float = 1):
        super().__init__()
        self.OUTPUT_CHANNELS = {1: 256, 2: 512, 3: 1024, 4: 2048}
        self.adapted_layers = {
            k: v for k, v in self.OUTPUT_CHANNELS.items() if k in adapter_layers
        }

        self.adapters = AdapterSet(
            self.adapted_layers,
            adapter_ratio,
        )
        self.backbone = Backbone()
        self.seg = SegmentationModule(self.adapted_layers)

    def trainable_parameters(self):
        # backbone is frozen in Backbone.__init__, only adapters + seg head train
        return [*self.adapters.parameters(), *self.seg.parameters()]

    def forward(self, x: torch.Tensor):
        teacher, student = self.backbone(x, self.adapters)
        seg_out = self.seg(teacher, student)

        return (teacher, student), seg_out
