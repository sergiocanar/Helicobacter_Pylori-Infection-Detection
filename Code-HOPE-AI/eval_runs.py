"""
eval_runs.py

Evaluate saved runs from outputs/transformer/hope_ai_finetuned (or any
directory with the same layout) and plot per-fold PR curves.

Layout expected:
    runs_dir/
        fold{N}_<run-name>/
            best.pth             # model checkpoint (used when --recompute or no CSV)
            config.json          # hyperparameters + data paths
            val_predictions.csv  # columns: patient_id, true_label, prob, pred
            metrics.json         # optional, used to annotate the plot

Usage:
    python eval_runs.py --runs_dir outputs/transformer/hope_ai_finetuned/final
    python eval_runs.py --runs_dir outputs/transformer/hope_ai_finetuned/final \\
                        --recompute --save results/pr_curves.png
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# ── project root (directory of this script) ─────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# On-the-fly inference
# ---------------------------------------------------------------------------

def _resolve(path: str) -> str:
    """Turn relative paths (from config.json) into absolute paths."""
    return path if os.path.isabs(path) else os.path.join(_THIS_DIR, path)


def run_inference(run_dir: str) -> tuple[list, list, list] | None:
    """
    Load best.pth + config.json from run_dir and run a forward pass over the
    validation set.  Returns (y_true, y_prob, y_pred) or None on failure.
    """
    ckpt_path   = os.path.join(run_dir, 'best.pth')
    config_path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(ckpt_path) or not os.path.exists(config_path):
        return None

    # Lazy imports — keep startup fast when not needed
    import torch
    from torch.utils.data import DataLoader
    from h_pylori_datasets.datasets import PatientKFDataset, patient_kf_collate
    from lib.hp_transformer import HPTransformerClassifier

    with open(config_path) as f:
        cfg = json.load(f)

    fold        = cfg['fold']
    labels_dir  = _resolve(cfg['labels_dir'])
    kf_dir      = _resolve(cfg['kf_dir'])
    hidden_dim  = cfg.get('hidden_dim', 256)
    n_heads     = cfg.get('n_heads', 4)
    n_layers    = cfg.get('n_layers', 2)
    dropout     = cfg.get('dropout', 0.1)
    mask_ratio  = cfg.get('mask_ratio', 0.0)
    batch_size  = cfg.get('batch_size', 32)
    num_workers = cfg.get('num_workers', 4)

    val_csv    = os.path.join(labels_dir, f'fold{fold}_val.csv')
    # If patient .pth files live directly in kf_dir (flat layout, base runs),
    # use it as-is. Otherwise fall into the fold-split subdir (finetuned runs).
    import glob as _glob
    if _glob.glob(os.path.join(kf_dir, 'patient_*.pth')):
        kf_val_dir = kf_dir
    else:
        kf_val_dir = os.path.join(kf_dir, f'fold{fold}_val')

    val_ds = PatientKFDataset(val_csv, kf_val_dir)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=patient_kf_collate,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HPTransformerClassifier(
        input_dim  = 512,
        hidden_dim = hidden_dim,
        n_heads    = n_heads,
        n_layers   = n_layers,
        dropout    = dropout,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    all_labels, all_probs = [], []
    with torch.no_grad():
        for embeddings, labels, _pids, pad_mask in val_loader:
            embeddings = embeddings.to(device)
            pad_mask   = pad_mask.to(device)
            logits = model(embeddings, pad_mask)
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.tolist())

    all_preds = [int(p >= 0.5) for p in all_probs]
    return all_labels, all_probs, all_preds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_run(run_dir: str, recompute: bool = False) -> dict | None:
    """Load (or compute on the fly) predictions for a run directory."""
    pred_path = os.path.join(run_dir, 'val_predictions.csv')
    name      = os.path.basename(run_dir)

    y_true = y_prob = y_pred = None

    if not recompute and os.path.exists(pred_path):
        df     = pd.read_csv(pred_path)
        y_true = df['true_label'].values.tolist()
        y_prob = df['prob'].values.tolist()
        y_pred = df['pred'].values.tolist()
    else:
        print(f'  [{name}] running inference from checkpoint …')
        result = run_inference(run_dir)
        if result is None:
            print(f'  [{name}] skipped — no best.pth / config.json')
            return None
        y_true, y_prob, y_pred = result

    metrics_path = os.path.join(run_dir, 'metrics.json')
    stored = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            stored = json.load(f)

    return dict(name=name, y_true=y_true, y_prob=y_prob, y_pred=y_pred, stored=stored)


def compute_metrics(y_true, y_prob, y_pred) -> dict:
    has_both = len(set(y_true)) > 1
    return dict(
        precision = precision_score(y_true, y_pred, zero_division=0),
        recall    = recall_score(y_true, y_pred, zero_division=0),
        f1        = f1_score(y_true, y_pred, zero_division=0),
        accuracy  = accuracy_score(y_true, y_pred),
        mAP       = average_precision_score(y_true, y_prob) if has_both else float('nan'),
    )


def discover_runs(runs_dir: str, recompute: bool = False) -> dict[str, list[dict]]:
    pattern = re.compile(r'fold(\d+)')
    groups: dict[str, list] = {}
    for name in sorted(os.listdir(runs_dir)):
        full = os.path.join(runs_dir, name)
        if not os.path.isdir(full):
            continue
        m = pattern.search(name)
        if m is None:
            continue
        fold_key = f'fold{m.group(1)}'
        run = load_run(full, recompute=recompute)
        if run is not None:
            groups.setdefault(fold_key, []).append(run)
    return dict(sorted(groups.items()))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Pastel-accessible palette (colorblind-friendly)
_PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']


def _short_name(name: str) -> str:
    """Strip the fold prefix for a cleaner legend: fold1_zesty-sweep → zesty-sweep."""
    return re.sub(r'^fold\d+_', '', name)


def plot_pr_curves(groups: dict[str, list[dict]], save_path: str | None = None):
    plt.rcParams.update({
        'font.family'     : 'DejaVu Sans',
        'font.size'       : 11,
        'axes.titlesize'  : 13,
        'axes.labelsize'  : 11,
        'legend.fontsize' : 9,
        'figure.dpi'      : 120,
    })

    n_folds = len(groups)
    fig, axes = plt.subplots(
        2, n_folds,
        figsize=(5.5 * n_folds, 10),
        squeeze=False,
    )
    fig.suptitle('Precision–Recall Curves', fontsize=15, fontweight='bold', y=1.01)

    row_labels = ['HP Positive', 'HP Negative']
    row_colors = [_PALETTE[0], '#C44E52']

    for col, (fold_key, runs) in enumerate(groups.items()):
        for row in range(2):
            ax = axes[row][col]
            color = row_colors[row]

            ax.set_title(
                f"{fold_key.replace('fold', 'Fold ')} — {row_labels[row]}",
                fontweight='semibold',
            )
            ax.set_xlabel('Recall', labelpad=6)
            ax.set_ylabel('Precision', labelpad=6)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4, color='#888888')
            ax.spines[['top', 'right']].set_visible(False)

            for run in runs:
                y_true = run['y_true']
                y_prob = run['y_prob']

                if row == 0:
                    p, r, _ = precision_recall_curve(y_true, y_prob)
                    chance = np.mean(y_true)
                else:
                    p, r, _ = precision_recall_curve(
                        [1 - v for v in y_true], [1 - v for v in y_prob]
                    )
                    chance = 1 - np.mean(y_true)

                ax.fill_between(r, p, alpha=0.08, color=color)
                ax.plot(r, p, color=color, lw=2.2, solid_capstyle='round')
                ax.axhline(chance, color='#55A868', linestyle='--', lw=1.5, label='Random')

            ax.legend(loc='lower left', framealpha=0.85, edgecolor='#cccccc')

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--runs_dir',
        default='outputs/transformer/hope_ai_finetuned/final',
        help='Directory containing fold{N}_<name>/ sub-dirs',
    )
    parser.add_argument(
        '--save',
        default=None,
        help='Optional path to save the figure (e.g. results/pr_curves.png)',
    )
    parser.add_argument(
        '--recompute',
        action='store_true',
        help='Re-run inference from checkpoints even if val_predictions.csv exists',
    )
    args = parser.parse_args()

    groups = discover_runs(args.runs_dir, recompute=args.recompute)
    if not groups:
        print(f'No runs found in {args.runs_dir}')
        return

    # ── Print table ────────────────────────────────────────────────────────
    header = f"{'Run':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'mAP':>6} {'Acc':>6}"
    print(header)
    print('-' * len(header))
    for fold_key, runs in groups.items():
        for run in runs:
            m = compute_metrics(run['y_true'], run['y_prob'], run['y_pred'])
            print(
                f"{run['name']:<45} "
                f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
                f"{m['f1']:>6.3f} {m['mAP']:>6.3f} {m['accuracy']:>6.3f}"
            )

    plot_pr_curves(groups, save_path=args.save)


if __name__ == '__main__':
    main()
