import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataloader import TrainGenerator
from evaluate import build_test_loader, evaluate
from train import train

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_samples")


def parse_args():
    parser = argparse.ArgumentParser(description="Train AdapTS on one MVTec category")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--anomaly-source", default=None,
                        help="DTD-style folder, defaults to <data-root>/anomaly_sources/images")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--adapter-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    img_size = (args.img_size, args.img_size)
    anomaly_source = args.anomaly_source or os.path.join(
        args.data_root, "anomaly_sources/images"
    )
    train_set = TrainGenerator(
        os.path.join(args.data_root, args.category, "train/good"), anomaly_source, img_size
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    eval_fn = None
    if not args.no_eval:
        test_loader = build_test_loader(
            args.data_root, args.category, img_size, args.bs, args.workers
        )
        eval_fn = lambda model, device: evaluate(model, test_loader, device)  # noqa: E731

    train(
        train_loader,
        adapter_layers=args.layers,
        adapter_ratio=args.adapter_ratio,
        lr=args.lr,
        epochs=args.epochs,
        eval_fn=eval_fn,
        eval_every=args.eval_every,
        out_dir=os.path.join(args.out_dir, args.category),
    )


if __name__ == "__main__":
    main()
