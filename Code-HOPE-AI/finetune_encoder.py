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
from torch.utils.data import Dataset, DataLoader
from h_pylori_datasets.datasets import YaoFrameDataset
from torch.cuda.amp import GradScaler
from torch.amp import autocast

import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score
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
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
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
        # ImageNet pretrained weights: keys are already bare
        encoder_sd = dict(full_sd)
        source = "ImageNet weights"
        strict = False   # head may have different num_classes

    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=strict)
    if missing:
        print(f"[WARN] Missing keys : {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    print(f"Encoder loaded from {source}: {ckpt_path}  ({len(encoder_sd)} tensors)")
    return encoder

def build_optimizer(model: nn.Module, lr: float, backbone_lr_scale: float,
                    weight_decay: float = 1e-4) -> optim.Optimizer:
    """Lower LR for backbone stages, full LR for the classification head.

    Stage assignment (PVT-v2):
        stage 0 : patch_embed1, block1, norm1
        stage 1 : patch_embed2, block2, norm2
        stage 2 : patch_embed3, block3, norm3
        stage 3 : patch_embed4, block4, norm4
        head    : head
    """
    head_params   = []
    stage_params  = [[] for _ in range(4)]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head"):
            head_params.append(param)
        else:
            for s in range(4):
                if (f"patch_embed{s+1}" in name or
                    f"block{s+1}" in name or
                    f"norm{s+1}" in name):
                    stage_params[s].append(param)
                    break

    # Deeper stages get a slightly higher LR (layerwise decay)
    layer_decay = 0.75   # each earlier stage is scaled by this factor
    param_groups = []
    for s in range(4):
        stage_lr = lr * backbone_lr_scale * (layer_decay ** (3 - s))
        param_groups.append({
            "params": stage_params[s],
            "lr": stage_lr,
            "name": f"stage{s+1}",
        })
    param_groups.append({"params": head_params, "lr": lr, "name": "head"})

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


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
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    all_probs  = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            _, logits, softmax = model(imgs)
            loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)

        all_probs.extend(softmax[:, 1].cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / n
    acc      = correct / n
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return avg_loss, acc, auc


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
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-4,
                        help="Learning rate for the classification head")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1,
                        help="Backbone LR = lr * this scale (before layerwise decay)")
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--warmup_epochs",  type=int,   default=3)
    parser.add_argument("--num_classes",    type=int,   default=2)

    return parser.parse_args()


def main():
    args = parse_args()
    # Paths
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = path_join(this_dir, "data", "GrastroHUN_Hpylori")
    images_dir = path_join(data_dir, "yao_images")
    labels_dir = path_join(data_dir, "yao_labels")
    output_dir = path_join(this_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Fold  : {args.fold}")

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
    class_weights = torch.tensor([1.0, pos_weight], device=device)
    print(f"Class balance  neg={n_neg}  pos={n_pos}  weight_pos={pos_weight:.2f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---- Optimiser & Scheduler ----
    optimizer = build_optimizer(model, lr=args.lr,
                                backbone_lr_scale=args.backbone_lr_scale,
                                weight_decay=args.weight_decay)

    # Cosine LR schedule with linear warmup (epoch-based)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=1e-7,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )
    scaler = GradScaler()

    # ---- Training loop ----
    best_auc   = 0.0
    best_ckpt  = os.path.join(args.output_dir, f"fold{args.fold}_best.pth")
    last_ckpt  = os.path.join(args.output_dir, f"fold{args.fold}_last.pth")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device)
        scheduler.step()

        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)

        head_lr = optimizer.param_groups[-1]["lr"]
        print(f"  train loss={train_loss:.4f}  acc={train_acc:.3f}")
        print(f"  val   loss={val_loss:.4f}  acc={val_acc:.3f}  AUC={val_auc:.4f}")
        print(f"  head lr={head_lr:.2e}")

        # Save last
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "val_auc":    val_auc,
            "args":       vars(args),
        }, last_ckpt)

        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch":   epoch,
                "model":   model.state_dict(),
                "val_auc": val_auc,
                "args":    vars(args),
            }, best_ckpt)
            print(f"  ** New best AUC={best_auc:.4f} saved to {best_ckpt}")

    print(f"\nTraining complete.  Best val AUC: {best_auc:.4f}")
    print(f"Checkpoints saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
