import contextlib
import os

import torch

from model import AdapTS
from utils.losses import SegmentationLoss, calculate_stfpm_loss


def train(
    train_loader,
    adapter_layers=(1, 2, 3),
    adapter_ratio=1.0,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=100,
    eval_fn=None,
    eval_every=10,
    eval_key="img_roc_auc",
    out_dir="checkpoints",
):
    """eval_fn(model, device) -> dict of metrics; the best eval_key checkpoint is kept."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed input size -> faster convs

    model = AdapTS(list(adapter_layers), adapter_ratio).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    seg_loss_fn = SegmentationLoss()

    # bf16 needs no grad scaler; fall back to full precision elsewhere
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if use_amp
        else contextlib.nullcontext
    )

    # optional safety check: fails loudly if the backbone ever becomes trainable
    assert not any(p.requires_grad for p in model.backbone.parameters())

    os.makedirs(out_dir, exist_ok=True)
    best = float("-inf")

    for epoch in range(epochs):
        model.train()  # adapters + seg go to train mode, backbone stays in eval
        totals = {"loss": 0.0, "stfpm": 0.0, "seg": 0.0}
        for sample in train_loader:
            anomaly_images = sample["anomaly_image"].to(device, non_blocking=True)
            masks = sample["anomaly_mask"].to(device, non_blocking=True)

            with autocast():
                (teacher, student), seg_out = model(anomaly_images)
                stfpm_loss = calculate_stfpm_loss(teacher, student)
                seg_loss = seg_loss_fn(seg_out.float(), masks)
                loss = stfpm_loss + seg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            totals["loss"] += loss.item()
            totals["stfpm"] += stfpm_loss.item()
            totals["seg"] += seg_loss.item()
        scheduler.step()

        n = len(train_loader)
        print(
            f"Epoch {epoch}: loss {totals['loss'] / n:.4f}, "
            f"stfpm {totals['stfpm'] / n:.4f}, seg {totals['seg'] / n:.4f}, "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

        model.save(os.path.join(out_dir, "last.pt"))
        is_last = epoch == epochs - 1
        if eval_fn is not None and ((epoch + 1) % eval_every == 0 or is_last):
            report = eval_fn(model, device)
            print("  eval: " + ", ".join(f"{k} {v:.4f}" for k, v in report.items()))
            if report[eval_key] > best:
                best = report[eval_key]
                model.save(os.path.join(out_dir, "best.pt"))

    return model
