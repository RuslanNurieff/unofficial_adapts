import contextlib
import os

import torch

from model import AdapTS
from utils.losses import stfpm_layer_losses


def train(
    train_loader,
    adapter_layers=(1, 2, 3),
    adapter_ratio=1.0,
    backbone="wide_resnet50_2",
    lr=1e-3,
    weight_decay=1e-4,
    epochs=100,
    eval_fn=None,
    eval_every=10,
    eval_key="img_roc_auc",
    out_dir="checkpoints",
    log_fn=None,
):
    """eval_fn(model, device) -> dict of metrics; the best eval_key checkpoint is kept.
    log_fn(dict) receives the epoch's train losses, lr and eval metrics (e.g. wandb.log).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed input size -> faster convs

    model = AdapTS(list(adapter_layers), adapter_ratio, backbone).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # bf16 needs no grad scaler; fall back to full precision elsewhere
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if use_amp
        else contextlib.nullcontext
    )

    # not trusting myself, might somewhere activate the backbone
    assert not any(p.requires_grad for p in model.backbone.parameters())

    os.makedirs(out_dir, exist_ok=True)
    best = float("-inf")

    for epoch in range(epochs):
        model.train()  # adapters go to train mode, backbone stays in eval
        totals = {"loss": 0.0, **{f"loss_layer{k}": 0.0 for k in model.adapted_layers}}
        for sample in train_loader:
            image = sample.to(device, non_blocking=True)

            with autocast():
                teacher, student = model(image)
                layer_losses = stfpm_layer_losses(teacher, student)
                loss = sum(layer_losses.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            totals["loss"] += loss.item()
            for k, v in layer_losses.items():
                totals[f"loss_layer{k}"] += v.item()
        scheduler.step()

        n = len(train_loader)
        last_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: loss {totals['loss'] / n:.4f}, lr {last_lr:.2e}")
        # epoch counts completed epochs, so evals land on multiples of eval_every
        log = {"epoch": epoch + 1, "train/lr": last_lr}
        log.update({f"train/{k}": v / n for k, v in totals.items()})

        model.save(os.path.join(out_dir, "last.pt"))
        is_last = epoch == epochs - 1
        if eval_fn is not None and ((epoch + 1) % eval_every == 0 or is_last):
            report = eval_fn(model, device)
            print("  eval: " + ", ".join(f"{k} {v:.4f}" for k, v in report.items()))
            if report[eval_key] > best:
                best = report[eval_key]
                model.save(os.path.join(out_dir, "best.pt"))
            log.update({f"eval/{k}": v for k, v in report.items()})
            log[f"eval/best_{eval_key}"] = best

        if log_fn is not None:
            log_fn(log)

    return model
