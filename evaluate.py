import argparse

import torch
from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.utilities.evaluation.metrics import F1, AvgPrec, MetricLvl, ProAuc, RocAuc

from data.dataloader import build_test_loader
from model import AdapTS

METRICS = [
    RocAuc(MetricLvl.IMAGE),
    RocAuc(MetricLvl.PIXEL),
    ProAuc(MetricLvl.PIXEL),
    AvgPrec(MetricLvl.PIXEL),
    F1(MetricLvl.PIXEL),
]

"""
Might apply Gaussian Blur to anomaly map later
from utils.anomaly import gaussian_blur
amap = gaussian_blur(amap.float(), self.sigma) then return the a_score by map.flatten(1).amax(dim=1)
"""


@torch.no_grad()
def evaluate(model: AdapTS, test_loader, device) -> dict:
    was_training = model.training
    report = Evaluator.evaluate(model, test_loader, METRICS, device)
    model.train(was_training)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an AdapTS checkpoint on MVTec"
    )
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
        args.data_root,
        args.category,
        (args.img_size, args.img_size),
        args.bs,
        args.workers,
    )
    for name, value in evaluate(model, loader, device).items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
