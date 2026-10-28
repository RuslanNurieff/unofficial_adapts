import argparse
import os
import time

import numpy as np
import torch
import wandb

from data.dataloader import build_test_loader, build_train_loader
from evaluate import evaluate
from modules.backbone import BACKBONES
from train import train

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_samples")


def parse_args():
    parser = argparse.ArgumentParser(description="Train AdapTS on MVTec categories")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--category", nargs="+", default=["bottle"])
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Hiera backbones need a multiple of 32",
    )
    parser.add_argument(
        "--backbone", default="wide_resnet50_2", choices=list(BACKBONES)
    )
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
    parser.add_argument("--wandb-project", default="adapts")
    parser.add_argument(
        "--wandb-group",
        help="shared by the categories' runs, default <backbone>-<time>",
    )
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    img_size = (args.img_size, args.img_size)
    # one run per category, grouped so wandb can average the final metrics over them
    group = args.wandb_group or f"{args.backbone}-{time.strftime('%Y%m%d-%H%M%S')}"

    categories = None
    if len(args.category) == 1 and args.category[0] == "all":
        categories = [
            "bottle",
            "cable",
            "capsule",
            "carpet",
            "grid",
            "hazelnut",
            "leather",
            "metal_nut",
            "pill",
            "screw",
            "tile",
            "toothbrush",
            "transistor",
            "wood",
            "zipper",
        ]
    else:
        categories = args.category

    for category in categories:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        train_loader = build_train_loader(
            args.data_root,
            category,
            img_size,
        )

        eval_fn = None
        if not args.no_eval:
            test_loader = build_test_loader(
                args.data_root, category, img_size, args.bs, args.workers
            )
            eval_fn = lambda model, device, loader=test_loader: evaluate(
                model, loader, device
            )

        with wandb.init(
            project=args.wandb_project,
            group=group,
            name=category,
            config={**vars(args), "category": category},
            mode="disabled" if args.no_wandb else None,
        ) as run:
            run.define_metric("epoch")
            run.define_metric("*", step_metric="epoch")
            train(
                train_loader,
                adapter_layers=args.layers,
                adapter_ratio=args.adapter_ratio,
                backbone=args.backbone,
                lr=args.lr,
                epochs=args.epochs,
                eval_fn=eval_fn,
                eval_every=args.eval_every,
                out_dir=os.path.join(args.out_dir, category),
                log_fn=run.log,
            )


if __name__ == "__main__":
    main()
