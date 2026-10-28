import torch
import torch.nn as nn  # noqa
import torch.nn.functional as F

from modules.adapters import AdapterSet
from modules.backbone import Backbone


class AdapTS(nn.Module):
    def __init__(
        self,
        adapter_layers: list[int],
        adapter_ratio: float = 1,
        backbone: str = "hiera_tiny_224",
    ):
        super().__init__()
        self.config = {
            "adapter_layers": list(adapter_layers),
            "adapter_ratio": adapter_ratio,
            "backbone": backbone,
        }
        self.backbone = Backbone(backbone, max_block=max(adapter_layers))
        unknown = set(adapter_layers) - self.backbone.channels.keys()
        if unknown:
            raise ValueError(f"unknown adapter layers {sorted(unknown)}")
        self.adapted_layers = {
            k: v for k, v in self.backbone.channels.items() if k in adapter_layers
        }

        self.adapters = AdapterSet(
            self.adapted_layers,
            adapter_ratio,
        )

    def trainable_parameters(self):
        return [*self.adapters.parameters()]

    def save(self, path):
        # the frozen backbone comes from pretrained weights, no need to store it
        torch.save(
            {
                "config": self.config,
                "adapters": self.adapters.state_dict(),
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        model = cls(**ckpt["config"])
        model.adapters.load_state_dict(ckpt["adapters"])
        return model

    def forward(self, x: torch.Tensor):
        teacher, student = self.backbone(x, self.adapters)
        if not self.training:
            return self.post_process(teacher, student)

        return (teacher, student)

    def post_process(self, t_feat, s_feat, output_shape=(256, 256)) -> torch.Tensor:
        """
        This method actually produces the anomaly maps for evalution purposes

        Args:
            - t_feat: teacher features maps
            - s_feat: student features maps

        Returns:
            - anomaly maps

        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        score_maps = torch.tensor([1.0], device=device)
        for j in t_feat:
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
            sm = F.interpolate(
                sm, size=output_shape, mode="bilinear", align_corners=False
            )
            score_maps = score_maps * sm

        anomaly_scores = torch.max(score_maps.view(score_maps.size(0), -1), dim=1)[0]
        return score_maps, anomaly_scores
