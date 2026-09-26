import torch
import torch.nn as nn  # noqa

from modules.adapters import AdapterSet
from modules.backbone import Backbone
from modules.segmentation import SegmentationModule
from utils.anomaly import anomaly_map


class AdapTS(nn.Module):
    OUTPUT_CHANNELS = {1: 256, 2: 512, 3: 1024, 4: 2048}

    def __init__(self, adapter_layers: list[int], adapter_ratio: float = 1):
        super().__init__()
        self.config = {"adapter_layers": list(adapter_layers), "adapter_ratio": adapter_ratio}
        unknown = set(adapter_layers) - self.OUTPUT_CHANNELS.keys()
        if unknown:
            raise ValueError(f"unknown adapter layers {sorted(unknown)}")
        self.adapted_layers = {
            k: v for k, v in self.OUTPUT_CHANNELS.items() if k in adapter_layers
        }

        self.adapters = AdapterSet(
            self.adapted_layers,
            adapter_ratio,
        )
        self.backbone = Backbone(max_block=max(self.adapted_layers))
        self.seg = SegmentationModule(self.adapted_layers)

    def trainable_parameters(self):
        # backbone is frozen in Backbone.__init__, only adapters + seg head train
        return [*self.adapters.parameters(), *self.seg.parameters()]

    def save(self, path):
        # the frozen backbone comes from torchvision weights, no need to store it
        torch.save(
            {
                "config": self.config,
                "adapters": self.adapters.state_dict(),
                "seg": self.seg.state_dict(),
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        model = cls(**ckpt["config"])
        model.adapters.load_state_dict(ckpt["adapters"])
        model.seg.load_state_dict(ckpt["seg"])
        return model

    def forward(self, x: torch.Tensor):
        teacher, student = self.backbone(x, self.adapters)
        if not self.training:
            # segmentation head is discarded at inference (paper Fig. 2c)
            return (teacher, student), anomaly_map(teacher, student, x.shape[-2:])

        # no detach: the seg loss has to reach the adapters to guide separability
        seg_out = self.seg(teacher, student)
        return (teacher, student), seg_out
