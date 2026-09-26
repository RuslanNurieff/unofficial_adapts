import os

import torch
from torch.utils.data import DataLoader

from data.dataloader import TrainGenerator
from train import train

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_samples")

train_loader = TrainGenerator(
    os.path.join(DATA_ROOT, "bottle/train/good"),
    os.path.join(DATA_ROOT, "anomaly_sources/images"),
    (256, 256),
)

train_loader = DataLoader(
    train_loader, batch_size=2, pin_memory=torch.cuda.is_available()
)


def main():
    train(train_loader)


if __name__ == "__main__":
    main()
