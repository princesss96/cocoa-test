"""
Train cocoa pod disease classifier (CNN / ViT / concat / attention fusion).

Example:
  python -m src.train_cls --data_dir data_cls --variant attn --epochs 20
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .data import create_imagefolder_loaders
from .models import HybridClassifier
from .ema import ModelEMA
from .utils import set_seed, get_device, AverageMeter, save_checkpoint, softmax_to_pred, compute_classification_metrics


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    criterion,
    ema: ModelEMA | None = None,
) -> float:
    model.train()
    loss_meter = AverageMeter("loss")

    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        loss_meter.update(loss.item(), n=images.size(0))
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    return float(loss_meter.avg)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Dict:
    model.eval()
    y_true, y_pred = [], []

    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        preds = softmax_to_pred(logits)

        y_true.extend(targets.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics = compute_classification_metrics(y_true, y_pred)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to classification dataset root (train/val/test).")
    parser.add_argument("--variant", type=str, default="attn", choices=["cnn", "vit", "concat", "attn"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="runs_cls")
    parser.add_argument("--use_ema", action="store_true", help="Enable EMA (variant V8 in proposal).")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--freeze_backbones_epochs", type=int, default=0,
                        help="Optional: freeze CNN+ViT for N initial epochs, then unfreeze.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, val_loader, test_loader, info = create_imagefolder_loaders(
        args.data_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = HybridClassifier(
        num_classes=info.num_classes,
        variant=args.variant,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        pretrained=True,
    ).to(device)

    if args.freeze_backbones_epochs > 0:
        model.freeze_backbones()

    ema = ModelEMA(model, decay=args.ema_decay, device=device) if args.use_ema else None

    # --- class-weighted loss (handles imbalance) ----#
    train_ds = train_loader.dataset
    counts = torch.bincount(torch.tensor(train_ds.targets, dtype=torch.long))
    weights = counts.sum().float() / (len(counts) * counts.clamp(min=1).float())

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    print("Class counts:", counts.tolist())
    print("Class weights:", [round(w, 4) for w in weights.tolist()])
    # ---------------------------------------------

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_f1 = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbones_epochs > 0 and epoch == args.freeze_backbones_epochs + 1:
            print(f"Unfreezing backbones at epoch {epoch} ...")
            model.unfreeze_backbones()

        print(f"\nEpoch {epoch}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.2e}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion, ema=ema)
        scheduler.step()

        # Evaluate either raw model or EMA model
        eval_model = ema.ema if ema is not None else model
        metrics = evaluate(eval_model, val_loader, device)

        print(f"train_loss={train_loss:.4f} | "
              f"val_acc={metrics['accuracy']:.4f} | val_macro_f1={metrics['macro_f1']:.4f} | val_kappa={metrics['kappa']:.4f}")

        # Save last checkpoint each epoch
        save_checkpoint({
            "epoch": epoch,
            "variant": args.variant,
            "model_state": model.state_dict(),
            "ema_state": (ema.state_dict() if ema is not None else None),
            "class_to_idx": info.class_to_idx,
            "img_size": args.img_size,
            "proj_dim": args.proj_dim,
            "dropout": args.dropout,
        }, os.path.join(args.out_dir, "last.pt"))

        # Track best by macro-F1
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            save_checkpoint({
                "epoch": epoch,
                "variant": args.variant,
                "model_state": model.state_dict(),
                "ema_state": (ema.state_dict() if ema is not None else None),
                "class_to_idx": info.class_to_idx,
                "img_size": args.img_size,
                "proj_dim": args.proj_dim,
                "dropout": args.dropout,
                "best_val_metrics": metrics,
            }, best_path)
            print(f"✅ Saved best checkpoint to {best_path} (macro_f1={best_f1:.4f})")

    # Optional final test
    if test_loader is not None:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if ckpt.get("ema_state") is not None:
            # If EMA exists, prefer it for evaluation
            ema2 = ModelEMA(model, decay=args.ema_decay, device=device)
            ema2.ema.load_state_dict(ckpt["ema_state"])
            eval_model = ema2.ema
        else:
            eval_model = model

        test_metrics = evaluate(eval_model, test_loader, device)
        print("\nTest metrics:", {k: v for k, v in test_metrics.items() if k != "confusion_matrix"})
        print("Confusion matrix:\n", test_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()