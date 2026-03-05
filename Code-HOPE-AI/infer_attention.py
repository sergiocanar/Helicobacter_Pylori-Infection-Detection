"""
infer_attention.py

Attention-based inference for H. Pylori detection.

Runs HPTransformerClassifier on keyframe embeddings and visualises which
frames drive each patient-level prediction via CLS-token attention.

Two attention methods (see lib/hp_transformer.py for details):
  rollout    (default) — Attention Rollout (Abnar & Zuidema 2020)
  last_layer           — last transformer layer, CLS row, averaged over heads

Outputs
-------
Per patient (under <output_dir>/patient_<NNN>/):
  attention_grid.jpg    — top-K most-attended frames with rank and score
  attention_bar.png     — full per-frame attention distribution (red = top-K)
  attention_scores.csv  — ranked frame list with attention weights

Global (under <output_dir>/):
  summary.csv           — one row per patient: id, GT, pred, prob, top frames

Usage
-----
  python infer_attention.py \\
    --checkpoint outputs/transformer/hope_ai_finetuned/final/fold1_zesty-sweep-89/best.pth \\
    [--fold 1] \\
    [--split val] \\
    [--top_k 5] \\
    [--attn_method rollout|last_layer] \\
    [--output_dir results/attention/fold1]
"""

import argparse
import csv
import glob
import os
from collections import Counter
from os.path import join as path_join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from lib.hp_transformer import HPTransformerClassifier

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = path_join(THIS_DIR, 'data', 'GrastroHUN_Hpylori')
META_CSV_DEF = path_join(THIS_DIR, 'data', 'GrastroHUN', 'h_pylori_metadata',
                         'yao_images_metadata.csv')
THUMB        = 224   # thumbnail size for the image grid


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _load_thumb(img_path: str, size: int = THUMB) -> Image.Image:
    """Load and resize a frame; return a grey placeholder on error."""
    try:
        return Image.open(img_path).convert('RGB').resize((size, size))
    except Exception:
        ph = Image.new('RGB', (size, size), (180, 180, 180))
        ImageDraw.Draw(ph).text((10, size // 2 - 8), 'NOT FOUND', fill=(100, 100, 100))
        return ph


def make_attention_grid(
    sorted_frame_paths: list,
    sorted_scores: np.ndarray,
    images_root: str,
    top_k: int,
    patient_id: int,
    pred: int,
    prob: float,
    true_label: int,
    output_dir: str,
    sorted_zones: list = None,   # parallel list of triple_agreement strings
) -> str:
    """Top-K attended frames laid out in a row with annotation."""
    k = min(top_k, len(sorted_frame_paths))
    thumbs = [_load_thumb(path_join(images_root, fp)) for fp in sorted_frame_paths[:k]]

    header_h = 44
    label_h  = 40   # taller to fit two lines (score + zone)
    canvas = Image.new('RGB', (k * THUMB, header_h + THUMB + label_h), (245, 245, 245))
    draw   = ImageDraw.Draw(canvas)

    gt_str   = 'HP POSITIVE' if true_label == 1 else 'HP NEGATIVE'
    pred_str = 'HP POSITIVE' if pred == 1 else 'HP NEGATIVE'
    marker   = '✓' if pred == true_label else '✗'
    draw.text(
        (8, 10),
        f'Patient {patient_id:03d}  GT:{gt_str}  Pred:{pred_str}  '
        f'P={prob:.3f}  {marker}  (top-{k} frames)',
        fill=(30, 30, 30),
    )

    for rank, (thumb, score) in enumerate(zip(thumbs, sorted_scores[:k])):
        x0   = rank * THUMB
        zone = (sorted_zones[rank] if sorted_zones and rank < len(sorted_zones) else '') or ''
        canvas.paste(thumb, (x0, header_h))
        draw.rectangle(
            [x0, header_h + THUMB, x0 + THUMB - 1, header_h + THUMB + label_h - 1],
            fill=(220, 230, 245),
        )
        draw.text((x0 + 4, header_h + THUMB + 4),  f'#{rank + 1}  {score:.4f}', fill=(0, 0, 150))
        draw.text((x0 + 4, header_h + THUMB + 22), zone,                         fill=(80, 40, 0))

    path = path_join(output_dir, 'attention_grid.jpg')
    canvas.save(path, quality=92)
    return path


def make_attention_bar(
    scores: list,
    top_k_indices: list,
    patient_id: int,
    output_dir: str,
) -> str:
    """Bar chart of per-frame attention scores; top-K bars are highlighted in red."""
    N      = len(scores)
    colors = ['#4878cf'] * N
    for i in top_k_indices:
        if i < N:
            colors[i] = '#c44e52'

    fig, ax = plt.subplots(figsize=(max(10, N * 0.18), 4))
    ax.bar(range(N), scores, color=colors, width=0.85, linewidth=0)
    ax.set_xlabel('Frame index', fontsize=11)
    ax.set_ylabel('Attention weight', fontsize=11)
    ax.set_title(
        f'Patient {patient_id:03d}: CLS-token attention  (red = top-{len(top_k_indices)})',
        fontsize=12,
    )
    ax.set_xlim(-0.5, N - 0.5)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = path_join(output_dir, 'attention_bar.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg  = ckpt.get('cfg', {})
    fold = args.fold if args.fold is not None else int(cfg.get('fold', 1))

    model = HPTransformerClassifier(
        input_dim  = 512,
        hidden_dim = cfg.get('hidden_dim', 256),
        n_heads    = cfg.get('n_heads', 4),
        n_layers   = cfg.get('n_layers', 2),
        dropout    = 0.0,   # no dropout at inference
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # ---- Resolve paths ----
    kf_dir = args.kf_dir or cfg.get('kf_dir', path_join(DATA_DIR, 'kf_features'))
    if not os.path.isabs(kf_dir):
        kf_dir = path_join(THIS_DIR, kf_dir)
    split_kf_dir = path_join(kf_dir, f'fold{fold}_{args.split}')
    if not os.path.isdir(split_kf_dir):
        split_kf_dir = kf_dir   # fall back to flat layout

    images_root = args.images_root or path_join(DATA_DIR, 'yao_images')
    labels_dir  = args.labels_dir  or path_join(DATA_DIR, 'yao_labels')

    # ---- CSV → frame-path lookup (same order as embedding rows) ----
    csv_path = path_join(labels_dir, f'fold{fold}_{args.split}.csv')
    df = pd.read_csv(csv_path)
    pid_to_frames = {
        int(pid): grp['frame_path'].tolist()
        for pid, grp in df.groupby('patient_id')
    }

    # ---- Metadata → zone lookup: frame_idx → (triple_agreement, zone) ----
    # frame_KKKK.jpg for patient P is the KKKK-th row (sorted filename order)
    # in yao_images_metadata.csv for patient P — verified by pixel comparison.
    pid_to_zones = {}
    meta_csv = args.meta_csv or META_CSV_DEF
    if meta_csv and os.path.isfile(meta_csv):
        meta_df = pd.read_csv(meta_csv)
        for pid, grp in meta_df.groupby('num_patient'):
            grp_sorted = grp.sort_values('filename').reset_index(drop=True)
            pid_to_zones[int(pid)] = list(zip(
                grp_sorted['triple_agreement'].fillna('').tolist(),
                grp_sorted['zone'].fillna('').tolist(),
            ))
    else:
        print(f'  [INFO] No metadata CSV found at {meta_csv} — zone info will be empty')

    # ---- Discover patients ----
    pth_files = sorted(glob.glob(path_join(split_kf_dir, 'patient_*.pth')))
    if not pth_files:
        raise FileNotFoundError(f'No patient_*.pth found in {split_kf_dir}')

    print(f'Checkpoint  : {args.checkpoint}')
    print(f'Fold        : {fold}  |  Split: {args.split}')
    print(f'KF dir      : {split_kf_dir}')
    print(f'Images root : {images_root}')
    print(f'Patients    : {len(pth_files)}')
    print(f'Attn method : {args.attn_method}  |  Top-K: {args.top_k}')
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = []

    for pth_file in pth_files:
        data  = torch.load(pth_file, map_location='cpu', weights_only=True)
        pid   = int(data['patient_id'])
        label = int(data['label'])
        embs  = data['embeddings']   # [N, 512]
        N     = embs.shape[0]

        frame_paths = pid_to_frames.get(pid, [])
        if len(frame_paths) != N:
            print(f'  [WARN] Patient {pid}: CSV={len(frame_paths)} frames, '
                  f'pth={N} — adjusting list')
            frame_paths = (frame_paths + ['??'] * N)[:N]

        # Zone info: (triple_agreement, zone) per frame index
        zone_list = pid_to_zones.get(pid, [])   # may be empty if no metadata

        # ---- Forward pass ----
        x = embs.unsqueeze(0).to(device)   # [1, N, 512]
        with torch.no_grad():
            logit, attn_list, pad_mask_ext = model.forward_with_attention(x)

        prob = torch.sigmoid(logit).item()
        pred = int(prob >= 0.5)
        ok   = pred == label

        # ---- Frame importance ----
        scores    = model.cls_frame_importance(attn_list, pad_mask_ext,
                                               method=args.attn_method)
        scores    = scores[0].cpu().numpy()   # [N]
        sorted_idx = np.argsort(scores)[::-1]
        top_k      = min(args.top_k, N)
        top_k_idx  = sorted_idx[:top_k].tolist()

        # ---- Per-patient output ----
        patient_dir = path_join(args.output_dir, f'patient_{pid:03d}')
        os.makedirs(patient_dir, exist_ok=True)

        fp_ranked  = [frame_paths[i] for i in sorted_idx]
        sc_ranked  = scores[sorted_idx]
        # zones in ranked order: triple_agreement string (or '' if unavailable)
        zones_ranked = [
            zone_list[i][0] if i < len(zone_list) else ''
            for i in sorted_idx
        ]

        make_attention_grid(fp_ranked, sc_ranked, images_root, top_k,
                            pid, pred, prob, label, patient_dir,
                            sorted_zones=zones_ranked)
        make_attention_bar(scores.tolist(), top_k_idx, pid, patient_dir)

        # Per-frame scores CSV — includes triple_agreement and zone
        with open(path_join(patient_dir, 'attention_scores.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['rank', 'frame_idx', 'frame_path', 'attn_score',
                        'triple_agreement', 'zone'])
            for rank, idx in enumerate(sorted_idx, 1):
                fp  = frame_paths[idx] if idx < len(frame_paths) else '??'
                ta, zn = zone_list[idx] if idx < len(zone_list) else ('', '')
                w.writerow([rank, int(idx), fp, f'{scores[idx]:.6f}', ta, zn])

        top_path = frame_paths[top_k_idx[0]] if top_k_idx else '?'
        top_zone = zones_ranked[0] if zones_ranked else ''
        # Count how often each zone appears in the top-K
        top_k_zones = zones_ranked[:top_k]
        zone_counts = Counter(z for z in top_k_zones if z)
        top_zone_summary = '|'.join(f'{z}:{c}' for z, c in zone_counts.most_common())
        summary_rows.append({
            'patient_id'      : pid,
            'true_label'      : label,
            'prob'            : f'{prob:.6f}',
            'pred'            : pred,
            'correct'         : int(ok),
            'n_frames'        : N,
            'top1_frame'      : top_path,
            'top1_score'      : f'{scores[top_k_idx[0]]:.6f}' if top_k_idx else '',
            'top1_zone'       : top_zone,
            'topK_zone_counts': top_zone_summary,
        })

        marker = '✓' if ok else '✗'
        zone_str = f'  zone={top_zone}' if top_zone else ''
        print(f'  {marker}  Patient {pid:03d}  GT={label}  pred={pred}  '
              f'P={prob:.3f}  N={N}  top={top_path}{zone_str}')

    # ---- Global summary ----
    summary_path = path_join(args.output_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Quick metrics
    labels = [r['true_label'] for r in summary_rows]
    preds  = [r['pred']       for r in summary_rows]
    probs  = [float(r['prob']) for r in summary_rows]
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score
    f1  = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    mAP = average_precision_score(labels, probs) if len(set(labels)) > 1 else float('nan')
    print()
    print(f'F1={f1:.4f}  Acc={acc:.4f}  mAP={mAP:.4f}')
    print(f'Summary → {summary_path}')
    print(f'Output  → {args.output_dir}/')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Attention-based inference for H. Pylori detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--checkpoint', required=True,
        help='Path to best.pth (HPTransformerClassifier checkpoint)',
    )
    parser.add_argument(
        '--fold', type=int, default=None, choices=[1, 2, 3],
        help='Fold number (inferred from checkpoint config if omitted)',
    )
    parser.add_argument(
        '--split', default='val', choices=['train', 'val'],
        help='Which split to run on',
    )
    parser.add_argument(
        '--kf_dir', default=None,
        help='Root dir with fold*_{train,val}/patient_*.pth '
             '(inferred from checkpoint config if omitted)',
    )
    parser.add_argument(
        '--images_root', default=None,
        help='Root dir for frame images (default: data/.../yao_images)',
    )
    parser.add_argument(
        '--labels_dir', default=None,
        help='Dir containing fold*_{train,val}.csv (default: data/.../yao_labels)',
    )
    parser.add_argument(
        '--top_k', type=int, default=5,
        help='Number of top-attended frames to visualise',
    )
    parser.add_argument(
        '--attn_method', default='rollout', choices=['rollout', 'last_layer'],
        help='Attention aggregation strategy',
    )
    parser.add_argument(
        '--output_dir', default='results/attention',
        help='Output directory',
    )
    parser.add_argument(
        '--meta_csv', default=None,
        help='Path to yao_images_metadata.csv for triple_agreement zone annotation '
             '(default: data/GrastroHUN/h_pylori_metadata/yao_images_metadata.csv)',
    )

    args = parser.parse_args()
    run_inference(args)
