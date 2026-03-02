"""
extract_features.py

Runs only the PVTv2 feature extractor (LSTMModel.feature_extractor) on every
frame listed in one or more CSV files and saves the resulting embeddings.

Output (.pth file) is a dict:
    {
        'embeddings': Tensor [N, 512],   # L2-normalised per-image features
        'frame_paths': list[str],        # relative paths (as in CSV)
        'patient_ids': list[int],
        'labels':      list[int],        # HP label (0 / 1)
    }

Modes
-----
vanilla (default)
    Extracts all frames from data/GrastroHUN_Hpylori/labels/all.csv using
    the base weights/model.pth.
    Output: data/GrastroHUN_Hpylori/features/all.pth

finetuned_kf
    For each fold k in {1, 2, 3}:
      - Loads the fold-k fine-tuned encoder from
        outputs/pvtv2/hope_ai/hopeai_fold{k}_lr*/best.pth
      - Extracts features separately for fold{k}_train.csv and fold{k}_val.csv
        using that fold's encoder (val patients were held out during fine-tuning,
        so the encoder is unbiased for them)
    Output: data/GrastroHUN_Hpylori/finetuned_kf_features/fold{k}_{train|val}.pth

Usage:
    python extract_features.py --mode vanilla
    python extract_features.py --mode finetuned_kf
"""

import glob
import os
from os.path import join as path_join

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.pvtv2_lstm import LSTMModel
from h_pylori_datasets.datasets import FrameDataset

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_extractor(ckpt_path: str, device: torch.device):
    """Load LSTMModel from checkpoint and return only the feature_extractor."""
    full_model = LSTMModel()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    state_dict = {f'feature_extractor.{k}': v for k, v in state_dict.items()}
    full_model.load_state_dict(state_dict, strict=False)
    extractor = full_model.feature_extractor   # PyramidVisionTransformerV2
    extractor.eval().to(device)
    print(f'  Encoder loaded from: {ckpt_path}')
    return extractor


def _extract_split(extractor, csv_paths: list, frames_dir: str,
                   args, device: torch.device) -> dict:
    """Extract embeddings for all frames listed in csv_paths."""
    ds = FrameDataset(csv_paths, frames_dir, img_size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f'  Frames to extract: {len(ds)}')

    all_embeddings  = torch.empty(len(ds), 512)
    all_patient_ids = [None] * len(ds)
    all_labels      = [None] * len(ds)

    with torch.no_grad():
        for imgs, pids, labels, indices in tqdm(loader, desc='  Extracting', unit='batch'):
            imgs = imgs.to(device)
            x_feature, _, _ = extractor(imgs)   # [B, 512]
            if args.normalize:
                x_feature = F.normalize(x_feature, dim=-1)
            x_feature = x_feature.cpu()
            for i, idx in enumerate(indices):
                all_embeddings[idx]  = x_feature[i]
                all_patient_ids[idx] = pids[i].item()
                all_labels[idx]      = labels[i].item()

    return {
        'embeddings':  all_embeddings,
        'frame_paths': ds.frame_paths,
        'patient_ids': all_patient_ids,
        'labels':      all_labels,
    }

# ---------------------------------------------------------------------------
# Mode: vanilla  (original behaviour — single checkpoint, all.csv)
# ---------------------------------------------------------------------------

def main_vanilla(frames_dir: str, labels_dir: str, ckpt_path: str,
                 output_dir: str, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    extractor = _load_extractor(ckpt_path, device)

    csv_path = path_join(labels_dir, 'all.csv')
    output = _extract_split(extractor, [csv_path], frames_dir, args, device)

    out_path = path_join(output_dir, 'all.pth')
    torch.save(output, out_path)
    print(f'Saved {len(output["patient_ids"])} embeddings → {out_path}')
    print(f'Embedding tensor shape: {output["embeddings"].shape}')

# ---------------------------------------------------------------------------
# Mode: finetuned_kf  (per-fold fine-tuned encoder)
# ---------------------------------------------------------------------------

def main_finetuned_kf(frames_dir: str, labels_dir: str, ckpts_root: str,
                      output_dir: str, args):
    """
    For each fold k, load the fold-specific fine-tuned encoder and extract
    features for the fold's train and val splits independently.

    Checkpoint discovery:
        <ckpts_root>/hopeai_fold{k}_lr*/best.pth   (glob, first match)

    Outputs:
        <output_dir>/fold{k}_train/patient_<pid>.pth
        <output_dir>/fold{k}_val/patient_<pid>.pth
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    os.makedirs(output_dir, exist_ok=True)

    for fold in [1, 2, 3]:
        print(f'\n{"="*60}')
        print(f'Fold {fold}')
        print(f'{"="*60}')

        # ---- Find the fine-tuned checkpoint for this fold ----
        pattern = path_join(ckpts_root, f'hopeai_fold{fold}_lr*', 'best.pth')
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(
                f'No checkpoint found for fold {fold}. Searched: {pattern}'
            )
        ckpt_path = matches[0]   # use first match (one dir per fold)

        extractor = _load_extractor(ckpt_path, device)

        for split in ['train', 'val']:
            print(f'\n  -- {split} --')
            csv_path = path_join(labels_dir, f'fold{fold}_{split}.csv')
            output = _extract_split(extractor, [csv_path], frames_dir, args, device)

            # ---- Save one file per patient ----
            split_dir = path_join(output_dir, f'fold{fold}_{split}')
            os.makedirs(split_dir, exist_ok=True)

            embeddings  = output['embeddings']    # [N, 512]
            patient_ids = output['patient_ids']
            labels      = output['labels']

            unique_pids = sorted(set(patient_ids))
            for pid in unique_pids:
                idxs = [i for i, p in enumerate(patient_ids) if p == pid]
                label = labels[idxs[0]]
                patient_embeddings = embeddings[idxs]   # [n_frames, 512]
                torch.save(
                    {'embeddings': patient_embeddings, 'label': label, 'patient_id': pid},
                    path_join(split_dir, f'patient_{pid:03d}.pth'),
                )
            print(f'  Saved {len(unique_pids)} patients → {split_dir}/')

    print('\nDone — all folds extracted.')

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract PVTv2 image embeddings')
    parser.add_argument('--mode',        choices=['vanilla', 'finetuned_kf'],
                        default='vanilla',
                        help='vanilla: single checkpoint on all.csv  |  '
                             'finetuned_kf: per-fold fine-tuned encoder')
    parser.add_argument('--img_size',    type=int, default=352)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--normalize',   action='store_true', default=True,
                        help='L2-normalise embeddings (default: True)')
    args = parser.parse_args()

    this_dir   = os.path.dirname(os.path.abspath(__file__))
    data_dir   = path_join(this_dir, 'data', 'GrastroHUN_Hpylori')
    frames_dir = path_join(data_dir, 'yao_images')
    labels_dir = path_join(data_dir, 'labels')

    if args.mode == 'vanilla':
        ckpt_path  = path_join(this_dir, 'weights', 'model.pth')
        output_dir = path_join(data_dir, 'features')
        os.makedirs(output_dir, exist_ok=True)
        main_vanilla(frames_dir, labels_dir, ckpt_path, output_dir, args)

    elif args.mode == 'finetuned_kf':
        ckpts_root = path_join(this_dir, 'outputs', 'pvtv2', 'hope_ai')
        yao_labels_dir = path_join(data_dir, 'yao_labels')
        output_dir = path_join(data_dir, 'kf_features')
        main_finetuned_kf(frames_dir, yao_labels_dir, ckpts_root, output_dir, args)
