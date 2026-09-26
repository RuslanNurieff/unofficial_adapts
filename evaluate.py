import argparse

import torch
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.utilities.evaluation.metrics import F1, AvgPrec, MetricLvl, ProAuc, RocAuc
from torch import nn
from torch.utils.data import DataLoader

from model import AdapTS
from utils.anomaly import gaussian_blur

METRICS = [
    RocAuc(MetricLvl.IMAGE),
    RocAuc(MetricLvl.PIXEL),
    ProAuc(MetricLvl.PIXEL),
    AvgPrec(MetricLvl.PIXEL),
    F1(MetricLvl.PIXEL),
]


class AdapTSPredictor(nn.Module):
    """Adapts AdapTS to moviad's Evaluator: forward returns (anomaly_maps, anomaly_scores)."""

    def __init__(self, model: AdapTS, sigma: float = 4.0):
        super().__init__()
        self.model = model
        self.sigma = sigma

    def forward(self, x: torch.Tensor):
        _, amap = self.model(x)
        amap = gaussian_blur(amap.float(), self.sigma)
        return amap, amap.flatten(1).amax(dim=1)


def build_test_loader(data_root, category, img_size, batch_size=8, num_workers=4):
    args = DatasetArguments(dataset_path=data_root, img_size=img_size, gt_mask_size=img_size)
    dataset = MVTecDataset(args, category, Split.TEST)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


@torch.no_grad()
def evaluate(model: AdapTS, test_loader, device, sigma: float = 4.0) -> dict:
    was_training = model.training
    report = Evaluator.evaluate(AdapTSPredictor(model, sigma), test_loader, METRICS, device)
    model.train(was_training)
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate an AdapTS checkpoint on MVTec")
    parser.add_argument("checkpoint")
    parser.add_argument("--data-root", default="data_samples")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdapTS.from_checkpoint(args.checkpoint, map_location=device).to(device)
    loader = build_test_loader(
        args.data_root, args.category, (args.img_size, args.img_size), args.bs, args.workers
    )
    for name, value in evaluate(model, loader, device).items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
