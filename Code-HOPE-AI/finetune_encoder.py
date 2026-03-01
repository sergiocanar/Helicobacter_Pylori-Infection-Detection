"""
finetune_encoder.py

Fine-tune the PVT-v2-B2 encoder loaded from model.pth on the Yao H. Pylori
frame-level dataset (binary HP classification: 0=negative, 1=positive).

Weights source
--------------
Two checkpoint formats are supported (auto-detected):
  model.pth    – LSTMModel whose feature_extractor is a pvt_v2_b2 trained
                 with num_classes=2. The "feature_extractor." prefix is
                 stripped and weights are loaded strictly.
  pvt_v2_b2.pth – ImageNet pretrained weights with bare keys. The head is
                  re-initialised (shape mismatch with num_classes is skipped)
                  and all stages are trainable.


Data layout expected
--------------------
yao_images/
    video_001/frame_0000.jpg
    video_001/frame_0001.jpg
    ...
yao_labels/
    fold1_train.csv   (columns: frame_path, patient_id, HP, OLGA)
    fold1_val.csv
    fold2_train.csv   ...

Usage
-----
python finetune_encoder.py \
    --fold 1 \
    --img_size 352 \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-4 \
    --backbone_lr_scale 0.1 \
    --output_dir checkpoints/finetune
"""

import os
from os.path import join as path_join
import argparse


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from h_pylori_datasets.datasets import YaoFrameDataset
from torch.cuda.amp import GradScaler
from torch.amp import autocast

import pandas as pd
from torchvision import transforms
import json
import numpy as np
import wandb
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score,
)
from tqdm import tqdm

# Local model definition (pvt_v2_b2 lives here)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.pvtv2_lstm import pvt_v2_b2

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(img_size: int, train: bool):
    if train:
        return transforms.Compose([
            # Mild crop: retains ≥70 % of the image area so RAC / erythema /
            # nodularity patterns are not accidentally cropped away.  The
            # Yao 22-station protocol means station context matters, so we
            # keep the crop scale conservative.
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            # Horizontal flip is valid — no side-specific anatomy in frame.
            # Vertical flip is intentionally omitted: endoscope images have a
            # fixed gravitational orientation and upside-down views are not
            # realistic augmentations.
            transforms.RandomHorizontalFlip(),
            # Small geometric jitter: ±10° rotation + ≤5 % translation.
            # Camera tilt and minor position offsets occur naturally; larger
            # values would push texture patterns out of frame.
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ---------------------------------------------------------------------------
# Load encoder weights from model.pth
# ---------------------------------------------------------------------------

def load_encoder_from_checkpoint(ckpt_path: str, num_classes: int = 2) -> nn.Module:
    """Build pvt_v2_b2 and load weights from either:

    - LSTMModel checkpoint (model.pth): keys prefixed with "feature_extractor."
        feature_extractor.patch_embed1.proj.weight  →  patch_embed1.proj.weight
    - ImageNet pretrained weights (pvt_v2_b2.pth): direct keys, loaded with
        strict=False so head shape mismatches (1000 vs num_classes) are ignored.
    """
    encoder = pvt_v2_b2(num_classes=num_classes)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Support both raw dict and {'model': ...} wrappers
    full_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    prefix = "feature_extractor."
    if any(k.startswith(prefix) for k in full_sd):
        # LSTMModel checkpoint: strip prefix
        encoder_sd = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}
        source = "LSTMModel checkpoint"
        strict = True
    else:
        # ImageNet pretrained weights: keys are already bare.
        # Drop the head (1000-class classifier) so the randomly-initialised
        # head in the model is kept intact — strict=False alone is not enough
        # because PyTorch still errors on shape mismatches for shared key names.
        encoder_sd = {k: v for k, v in full_sd.items() if not k.startswith("head")}
        source = "ImageNet weights"
        strict = False

    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=strict)
    if missing:
        print(f"[WARN] Missing keys : {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    print(f"Encoder loaded from {source}: {ckpt_path}  ({len(encoder_sd)} tensors)")
    return encoder



# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(device_type=device.type):
            _, logits, _ = model(imgs)   # pvt_v2_b2 returns (feature, logits, softmax)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Returns a dict with loss, acc, auc, f1, precision, recall, ap, probs, labels."""
    model.eval()
    total_loss = 0.0
    n = 0
    all_probs  = []
    all_preds  = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type):
            _, logits, softmax = model(imgs)
            loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        n          += imgs.size(0)

        all_probs.extend(softmax[:, 1].cpu().tolist())
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    avg_loss = total_loss / n
    acc      = (y_pred == y_true).mean()

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    ap        = average_precision_score(y_true, y_prob)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)

    return {
        "loss":      avg_loss,
        "acc":       float(acc),
        "auc":       auc,
        "ap":        ap,
        "f1":        f1,
        "precision": precision,
        "recall":    recall,
        "probs":     y_prob,   # (N,)  positive-class prob
        "labels":    y_true,   # (N,)  ground-truth
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune PVT-v2-B2 encoder (H. Pylori)")


    parser.add_argument("--checkpoint",  default=str("model.pth"),
                        help="LSTMModel checkpoint to extract encoder weights from")
    parser.add_argument("--output_dir",  default=str("outputs"), help="Directory to save checkpoints and logs")

    # Data
    parser.add_argument("--fold",       type=int, default=1, choices=[1, 2, 3],
                        help="Cross-validation fold to use")
    parser.add_argument("--img_size",   type=int, default=352)
    parser.add_argument("--num_workers",type=int, default=4)

    # Training
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-4,
                        help="Learning rate for the classification head")
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--warmup_epochs",  type=int,   default=3)
    parser.add_argument("--num_classes",    type=int,   default=2)

    # W&B
    parser.add_argument("--wandb_project", default="hp-pylori-finetune",
                        help="W&B project name")
    parser.add_argument("--wandb_entity",  default=None,
                        help="W&B entity (team/user). Omit to use the default.")
    parser.add_argument("--run_name",      default=None,
                        help="Override the auto-generated W&B run name")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Paths ----
    this_dir   = os.path.dirname(os.path.abspath(__file__))
    data_dir   = path_join(this_dir, "data", "GrastroHUN_Hpylori")
    images_dir = path_join(data_dir, "yao_images")
    labels_dir = path_join(data_dir, "yao_labels")
    output_dir = path_join(this_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ---- W&B init ----
    # Use checkpoint basename to label the init type (imagenet vs hopeai)
    ckpt_tag  = "imagenet" if "pvt_v2_b2" in os.path.basename(args.checkpoint) else "hopeai"
    run_name  = args.run_name or f"{ckpt_tag}_fold{args.fold}_lr{args.lr:.0e}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=run_name,
        group=ckpt_tag,        # groups imagenet runs / hopeai runs together
        config=vars(args),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Fold: {args.fold}  |  Init: {ckpt_tag}")

    # ---- Data ----
    train_csv = os.path.join(labels_dir, f"fold{args.fold}_train.csv")
    val_csv   = os.path.join(labels_dir, f"fold{args.fold}_val.csv")

    train_ds = YaoFrameDataset(train_csv, images_dir,
                               transform=build_transforms(args.img_size, train=True))
    val_ds   = YaoFrameDataset(val_csv,   images_dir,
                               transform=build_transforms(args.img_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"Train frames: {len(train_ds)}  |  Val frames: {len(val_ds)}")

    # ---- Model ----
    model = load_encoder_from_checkpoint(args.checkpoint, num_classes=args.num_classes)
    model = model.to(device)

    # ---- Compute class weights for imbalanced data ----
    df_train = pd.read_csv(train_csv)
    n_neg = (df_train["HP"] == 0).sum()
    n_pos = (df_train["HP"] == 1).sum()
    pos_weight = n_neg / max(n_pos, 1)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    print(f"Class balance  neg={n_neg}  pos={n_pos}  weight_pos={pos_weight:.2f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---- Optimiser & Scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-7,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )
    scaler = GradScaler()

    # ---- Per-run output subfolder ----
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # ---- Save config artifact ----
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    config_artifact = wandb.Artifact(f"config-{run_name}", type="config")
    config_artifact.add_file(config_path)
    wandb.log_artifact(config_artifact)

    # ---- Training loop ----
    best_f1   = 0.0
    best_ckpt = os.path.join(run_dir, "best.pth")
    last_ckpt = os.path.join(run_dir, "last.pth")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device)
        scheduler.step()

        vm = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  train  loss={train_loss:.4f}  acc={train_acc:.3f}")
        print(f"  val    loss={vm['loss']:.4f}  acc={vm['acc']:.3f}  "
              f"AUC={vm['auc']:.4f}  AP={vm['ap']:.4f}  "
              f"F1={vm['f1']:.4f}  P={vm['precision']:.4f}  R={vm['recall']:.4f}")
        print(f"  lr={current_lr:.2e}")

        # Build PR-curve data for wandb
        # wandb.plot.pr_curve expects y_probas of shape (N, n_classes)
        y_probas_2d = np.stack([1.0 - vm["probs"], vm["probs"]], axis=1)
        pr_curve = wandb.plot.pr_curve(
            vm["labels"], y_probas_2d, labels=["negative", "positive"]
        )

        wandb.log({
            "epoch":            epoch,
            "train/loss":       train_loss,
            "train/acc":        train_acc,
            "val/loss":         vm["loss"],
            "val/acc":          vm["acc"],
            "val/auc":          vm["auc"],
            "val/ap":           vm["ap"],
            "val/f1":           vm["f1"],
            "val/precision":    vm["precision"],
            "val/recall":       vm["recall"],
            "val/pr_curve":     pr_curve,
            "lr":               current_lr,
        }, step=epoch)

        # Save last checkpoint
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_auc":   vm["auc"],
            "args":      vars(args),
        }, last_ckpt)

        # Save best checkpoint (criterion: val/f1)
        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            torch.save({
                "epoch":   epoch,
                "model":   model.state_dict(),
                "val_auc": vm["auc"],
                "val_ap":  vm["ap"],
                "val_f1":  vm["f1"],
                "args":    vars(args),
            }, best_ckpt)
            wandb.run.summary["best_f1"]    = best_f1
            wandb.run.summary["best_epoch"] = epoch
            print(f"  ** New best F1={best_f1:.4f} → {best_ckpt}")

    # ---- Final evaluation with best weights ----
    best_sd = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_sd["model"])
    vm_best = validate(model, val_loader, criterion, device)

    # PR curve using all unique predicted probabilities as thresholds (sklearn default)
    y_probas_2d = np.stack([1.0 - vm_best["probs"], vm_best["probs"]], axis=1)
    final_pr = wandb.plot.pr_curve(
        vm_best["labels"], y_probas_2d, labels=["negative", "positive"]
    )
    wandb.log({"val/final_pr_curve": final_pr})

    # ---- Save metrics JSON ----
    metrics = {
        "best_epoch":    int(best_sd["epoch"]),
        "val_f1":        float(vm_best["f1"]),
        "val_auc":       float(vm_best["auc"]),
        "val_ap":        float(vm_best["ap"]),
        "val_precision": float(vm_best["precision"]),
        "val_recall":    float(vm_best["recall"]),
        "val_acc":       float(vm_best["acc"]),
    }
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics  : {metrics}")

    # ---- Upload best model as artifact ----
    model_artifact = wandb.Artifact(f"model-{run_name}", type="model",
                                    metadata=metrics)
    model_artifact.add_file(best_ckpt)
    model_artifact.add_file(metrics_path)
    wandb.log_artifact(model_artifact)

    wandb.finish()
    print(f"\nTraining complete.  Best val F1: {best_f1:.4f}")
    print(f"Checkpoints saved in: {run_dir}")


if __name__ == "__main__":
    main()
