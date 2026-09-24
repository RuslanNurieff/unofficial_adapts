import torch

from model import AdapTS
from utils.losses import SegmentationLoss, calculate_stfpm_loss


def train(
    train_loader, adapter_layers=(1, 2, 3), adapter_ratio=1.0, lr=1e-3, epochs=100
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AdapTS(list(adapter_layers), adapter_ratio).to(device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    seg_loss_fn = SegmentationLoss()

    # optional safety check: fails loudly if the backbone ever becomes trainable
    assert not any(p.requires_grad for p in model.backbone.parameters())

    for epoch in range(epochs):
        model.train()  # adapters + seg go to train mode, backbone stays in eval
        for sample in train_loader:
            images, anomaly_images, masks = (
                sample["image"],
                sample["anomaly_image"],
                sample["anomaly_mask"],
            )
            images, anomaly_images, masks = (
                images.to(device),
                anomaly_images.to(device),
                masks.to(device),
            )
            (teacher, student), seg_out = model(anomaly_images)
            loss = calculate_stfpm_loss(teacher, student) + seg_loss_fn(seg_out, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")

        # sample = {
        #     "image": img,
        #     "anomaly_image": perlin_img,
        #     "anomaly_mask": mask,
        #     "has_anomaly": label,
        # }
