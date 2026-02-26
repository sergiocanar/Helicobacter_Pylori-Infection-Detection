"""
train_transformer.py

Trains HPTransformerClassifier on pre-extracted per-patient keyframe embeddings.
Primary metric: F1. Also tracks precision, recall, mAP and accuracy.

Per-run output (under <output_dir>/<wandb_run_name>/):
    best.pth               – best checkpoint (updated every time val/f1 improves)
    val_predictions.csv    – val predictions at the epoch of best val/f1
    test_predictions.csv   – test predictions from the best checkpoint
    config.json            – full hyperparameter config for the run
    metrics.json           – best-val metrics, test metrics, and per-epoch history
    train.log              – full training log

Standalone run:
    python train_transformer.py [--lr 1e-4 --hidden_dim 256 ...]

W&B sweep:
    1. Create sweep:   wandb sweep sweep_config.yaml   → prints <entity>/<project>/<SWEEP_ID>
    2. Launch agent:   wandb agent <entity>/<project>/<SWEEP_ID>
"""

import csv
import json
import os
import argparse
import logging
from os.path import join as path_join

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, accuracy_score,
)
import wandb

from h_pylori_datasets.datasets import PatientKFDataset, patient_kf_collate
from lib.hp_transformer import HPTransformerClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, criterion):
    """
    Returns:
        metrics : dict  – loss, f1, precision, recall, mAP, accuracy
        labels  : list[int]
        probs   : list[float]  – sigmoid probabilities
        pids    : list[int]    – patient IDs in loader order
    """
    model.eval()
    total_loss = 0.0
    all_labels, all_probs, all_pids = [], [], []

    with torch.no_grad():
        for embeddings, labels, pids, pad_mask in loader:
            embeddings = embeddings.to(device)
            labels     = labels.to(device).float()
            pad_mask   = pad_mask.to(device)

            logits = model(embeddings, pad_mask)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_pids.extend(pids)

    avg_loss = total_loss / len(all_labels)
    preds    = [int(p >= 0.5) for p in all_probs]
    has_both = len(set(all_labels)) > 1

    metrics = dict(
        loss      = avg_loss,
        f1        = f1_score(all_labels, preds, zero_division=0),
        precision = precision_score(all_labels, preds, zero_division=0),
        recall    = recall_score(all_labels, preds, zero_division=0),
        mAP       = average_precision_score(all_labels, all_probs) if has_both else float('nan'),
        accuracy  = accuracy_score(all_labels, preds),
    )
    return metrics, all_labels, all_probs, all_pids


def save_predictions(path, pids, labels, probs):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['patient_id', 'true_label', 'prob', 'pred'])
        for pid, lbl, prob in zip(pids, labels, probs):
            w.writerow([pid, int(lbl), f'{prob:.6f}', int(prob >= 0.5)])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict):
    """
    Core training loop.  cfg is a plain dict with all hyperparameters and paths.
    Works identically for standalone runs and W&B sweep agents.
    """
    # Skip invalid head/dim combinations gracefully (can occur during sweeps)
    if cfg['hidden_dim'] % cfg['n_heads'] != 0:
        wandb.log({'val/f1': 0.0, 'val/loss': float('inf')})
        logging.warning(
            f"Skipping: hidden_dim={cfg['hidden_dim']} not divisible by "
            f"n_heads={cfg['n_heads']}"
        )
        return

    # Each wandb run gets its own subdirectory so sweep runs never collide
    run_dir = path_join(cfg['output_dir'], wandb.run.name)
    os.makedirs(run_dir, exist_ok=True)

    # ---- Logging ----
    root_log = logging.getLogger()
    root_log.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(path_join(run_dir, 'train.log'), mode='w'),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}  |  Run dir: {run_dir}')

    # ---- Save config ----
    with open(path_join(run_dir, 'config.json'), 'w') as f:
        json.dump(dict(wandb.config), f, indent=2, default=str)

    # ---- Datasets & loaders ----
    train_ds = PatientKFDataset(cfg['train_csv'], cfg['kf_dir'])
    val_ds   = PatientKFDataset(cfg['val_csv'],   cfg['kf_dir'])
    test_ds  = PatientKFDataset(cfg['test_csv'],  cfg['kf_dir'])

    log.info(f'Patients  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}')

    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg['num_workers'], collate_fn=patient_kf_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], collate_fn=patient_kf_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], collate_fn=patient_kf_collate,
    )

    # ---- Model ----
    model = HPTransformerClassifier(
        input_dim  = 512,
        hidden_dim = cfg['hidden_dim'],
        n_heads    = cfg['n_heads'],
        n_layers   = cfg['n_layers'],
        dropout    = cfg['dropout'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Model params: {n_params:,}')
    wandb.config.update({'n_params': n_params}, allow_val_change=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    # ---- Training loop ----
    best_val_f1  = -1.0
    best_val_m   = None
    best_ckpt    = path_join(run_dir, 'best.pth')
    val_preds_path  = path_join(run_dir, 'val_predictions.csv')
    history      = []

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        train_loss = 0.0

        for embeddings, labels, _, pad_mask in train_loader:
            embeddings = embeddings.to(device)
            labels     = labels.to(device).float()
            pad_mask   = pad_mask.to(device)

            optimizer.zero_grad()
            logits = model(embeddings, pad_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(labels)

        scheduler.step()
        train_loss /= len(train_ds)

        val_m, val_labels, val_probs, val_pids = evaluate(model, val_loader, device, criterion)

        log.info(
            f'Epoch {epoch:03d}/{cfg["epochs"]}  '
            f'train_loss={train_loss:.4f}  '
            f'val_loss={val_m["loss"]:.4f}  val_f1={val_m["f1"]:.4f}  '
            f'val_prec={val_m["precision"]:.4f}  val_rec={val_m["recall"]:.4f}  '
            f'val_mAP={val_m["mAP"]:.4f}  val_acc={val_m["accuracy"]:.4f}'
        )

        wandb.log({
            'epoch'         : epoch,
            'train/loss'    : train_loss,
            'val/loss'      : val_m['loss'],
            'val/f1'        : val_m['f1'],
            'val/precision' : val_m['precision'],
            'val/recall'    : val_m['recall'],
            'val/mAP'       : val_m['mAP'],
            'val/accuracy'  : val_m['accuracy'],
        })

        history.append({
            'epoch'      : epoch,
            'train_loss' : train_loss,
            **{f'val_{k}': v for k, v in val_m.items()},
        })

        if val_m['f1'] > best_val_f1:
            best_val_f1 = val_m['f1']
            best_val_m  = {**val_m, 'epoch': epoch}

            torch.save(
                {'epoch': epoch, 'model': model.state_dict(),
                 'val_f1': val_m['f1'], 'cfg': cfg},
                best_ckpt,
            )
            save_predictions(val_preds_path, val_pids, val_labels, val_probs)

            log.info(f'  → new best saved (val_f1={val_m["f1"]:.4f})')
            wandb.run.summary['best_val_f1']    = best_val_f1
            wandb.run.summary['best_val_epoch'] = epoch

    # ---- Test evaluation on best checkpoint ----
    log.info('\n--- Test evaluation ---')
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])

    test_m, test_labels, test_probs, test_pids = evaluate(model, test_loader, device, criterion)

    log.info(
        f'test_loss={test_m["loss"]:.4f}  test_f1={test_m["f1"]:.4f}  '
        f'test_prec={test_m["precision"]:.4f}  test_rec={test_m["recall"]:.4f}  '
        f'test_mAP={test_m["mAP"]:.4f}  test_acc={test_m["accuracy"]:.4f}  '
        f'(best ckpt from epoch {ckpt["epoch"]})'
    )

    # ---- Save test predictions ----
    save_predictions(
        path_join(run_dir, 'test_predictions.csv'),
        test_pids, test_labels, test_probs,
    )

    # ---- Save metrics.json ----
    metrics_out = {
        'best_val' : best_val_m,
        'test'     : test_m,
        'history'  : history,
    }
    with open(path_join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_out, f, indent=2, default=str)

    # ---- Log to W&B ----
    wandb.log({
        'test/loss'      : test_m['loss'],
        'test/f1'        : test_m['f1'],
        'test/precision' : test_m['precision'],
        'test/recall'    : test_m['recall'],
        'test/mAP'       : test_m['mAP'],
        'test/accuracy'  : test_m['accuracy'],
    })
    wandb.run.summary.update({
        'test_f1'       : test_m['f1'],
        'test_mAP'      : test_m['mAP'],
        'test_accuracy' : test_m['accuracy'],
    })

    # Upload run artifacts to W&B
    artifact = wandb.Artifact(name=f'run-{wandb.run.name}', type='run-output')
    artifact.add_file(best_ckpt,                              name='best.pth')
    artifact.add_file(val_preds_path,                         name='val_predictions.csv')
    artifact.add_file(path_join(run_dir, 'test_predictions.csv'), name='test_predictions.csv')
    artifact.add_file(path_join(run_dir, 'config.json'),      name='config.json')
    artifact.add_file(path_join(run_dir, 'metrics.json'),     name='metrics.json')
    wandb.log_artifact(artifact)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    this_dir   = os.path.dirname(os.path.abspath(__file__))
    data_dir   = path_join(this_dir, 'data', 'GrastroHUN_Hpylori')
    labels_dir = path_join(data_dir, 'yao_labels')

    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--train_csv',      default=path_join(labels_dir, 'train.csv'))
    parser.add_argument('--val_csv',        default=path_join(labels_dir, 'val.csv'))
    parser.add_argument('--test_csv',       default=path_join(labels_dir, 'test.csv'))
    parser.add_argument('--kf_dir',         default=path_join(data_dir, 'kf_features'))
    parser.add_argument('--output_dir',     default=path_join(this_dir, 'checkpoints', 'transformer'))
    # Model (sweep-able)
    parser.add_argument('--hidden_dim',     type=int,   default=256)
    parser.add_argument('--n_heads',        type=int,   default=4)
    parser.add_argument('--n_layers',       type=int,   default=2)
    parser.add_argument('--dropout',        type=float, default=0.1)
    # Training (sweep-able)
    parser.add_argument('--epochs',         type=int,   default=50)
    parser.add_argument('--batch_size',     type=int,   default=8)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--wd',             type=float, default=1e-4)
    parser.add_argument('--num_workers',    type=int,   default=4)
    # W&B
    parser.add_argument('--wandb_project',  default='hp-transformer')
    parser.add_argument('--wandb_entity',   default=None)

    args = parser.parse_args()

    with wandb.init(
        project = args.wandb_project,
        entity  = args.wandb_entity,
        # Argparse values become wandb.config defaults.
        # When running under a sweep agent, wandb automatically overrides
        # the sweep parameters, so wandb.config is always the source of truth.
        config  = vars(args),
    ):
        cfg = {
            # Fixed paths — always from argparse
            'train_csv'  : args.train_csv,
            'val_csv'    : args.val_csv,
            'test_csv'   : args.test_csv,
            'kf_dir'     : args.kf_dir,
            'output_dir' : args.output_dir,
            'num_workers': args.num_workers,
            # Sweep-able hyperparams — wandb.config so sweep values win
            'hidden_dim' : wandb.config.hidden_dim,
            'n_heads'    : wandb.config.n_heads,
            'n_layers'   : wandb.config.n_layers,
            'dropout'    : wandb.config.dropout,
            'epochs'     : wandb.config.epochs,
            'batch_size' : wandb.config.batch_size,
            'lr'         : wandb.config.lr,
            'wd'         : wandb.config.wd,
        }
        train(cfg)
