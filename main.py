from torch.utils.data import DataLoader

from data.dataloader import TrainGenerator
from train import train

train_loader = TrainGenerator(
    "/home/ruslannuriev/Desktop/adapts/data_samples/bottle/train/good",
    "/home/ruslannuriev/Desktop/adapts/data_samples/anomaly_sources/images",
    (256, 256),
)

train_loader = DataLoader(train_loader, batch_size=8)


def main():
    train(train_loader)


if __name__ == "__main__":
    main()
