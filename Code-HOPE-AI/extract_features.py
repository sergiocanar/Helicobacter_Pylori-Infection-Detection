"""
extract_features.py

Runs only the PVTv2 feature extractor (LSTMModel.feature_extractor) on every
frame listed in one or more CSV files and saves the resulting embeddings.

Output (.pt file) is a dict:
    {
        'embeddings': Tensor [N, 512],   # L2-normalised per-image features
        'frame_paths': list[str],        # relative paths (as in CSV)
        'patient_ids': list[int],
        'labels':      list[int],        # HP label (0 / 1)
    }

Usage:
    python extract_features.py \
        --checkpoint checkpoints/fold1_best.pth \
        --csv        data/GrastroHUN_Hpylori/labels/fold1_train.csv \
                     data/GrastroHUN_Hpylori/labels/fold1_val.csv \
        --frames_root data/GrastroHUN_Hpylori/frames \
        --output      embeddings/fold1_all.pt \
        --img_size    352 \
        --batch_size  32
"""

import os
from os.path import join as path_join

import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from lib.pvtv2_lstm import LSTMModel
from h_pylori_datasets.datasets import FrameDataset

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(frames_dir, labels_dir, ckpt_path, output_dir):
    parser = argparse.ArgumentParser(description='Extract PVTv2 image embeddings')
    parser.add_argument('--img_size',    type=int, default=352)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--normalize',   action='store_true', default=True,
                        help='L2-normalise embeddings (default: True)')
    args = parser.parse_args()

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- Load full model, keep only the feature extractor ----
    full_model = LSTMModel()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    full_model.load_state_dict(state_dict, strict=True)

    extractor = full_model.feature_extractor   # PyramidVisionTransformerV2
    extractor.eval().to(device)
    print('Feature extractor loaded from:', ckpt_path)
    csv_path = path_join(labels_dir, 'all.csv')
    # ---- Dataset ----
    valid_triple_agreement = ["A1", "L1", "P1", "G1", "A3", "L3", "G3", "P3", "A5", "L5", "G5"]
    
    ds = FrameDataset([csv_path], frames_dir, img_size=args.img_size, desired_triple_agreement=valid_triple_agreement)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f'Total frames: {len(ds)}')

    # ---- Extract ----
    all_embeddings  = torch.empty(len(ds), 512)
    all_patient_ids = [None] * len(ds)
    all_labels      = [None] * len(ds)

    with torch.no_grad():
        for imgs, pids, labels, indices in tqdm(loader, desc='Extracting', unit='batch'):
            imgs = imgs.to(device)
            x_feature, _, _ = extractor(imgs)   # [B, 512]
            if args.normalize:
                x_feature = F.normalize(x_feature, dim=-1)
            x_feature = x_feature.cpu()
            for i, idx in enumerate(indices):
                all_embeddings[idx]  = x_feature[i]
                all_patient_ids[idx] = pids[i].item()
                all_labels[idx]      = labels[i].item()

    # ---- Save ----
    output = {
        'embeddings':  all_embeddings,          # [N, 512]
        'frame_paths': ds.frame_paths,
        'patient_ids': all_patient_ids,
        'labels':      all_labels,
    }
    out_path = path_join(output_dir, 'all.pth')
    torch.save(output, out_path)
    print(f'Saved {len(ds)} embeddings → {out_path}')
    print(f'Embedding tensor shape: {all_embeddings.shape}')


if __name__ == '__main__':
    this_dir = os.path.dirname((os.path.abspath(__file__)))
    weights_dir = path_join(this_dir, 'weights')
    ckpt_path = path_join(weights_dir, 'model.pth')
    data_dir = path_join(this_dir, 'data', 'GrastroHUN_Hpylori')
    frames_dir = path_join(data_dir, 'frames')
    labels_dir = path_join(data_dir, 'labels')
    output_dir = path_join(data_dir, 'features')
    os.makedirs(output_dir, exist_ok=True)
    
    #models/Helicobacter_Pylori-Infection-Detection/Code-HOPE-AI/weights/model.pth
    #models/Helicobacter_Pylori-Infection-Detection/weights/model.pth'
    
    main(frames_dir=frames_dir, labels_dir=labels_dir, ckpt_path=ckpt_path, output_dir=output_dir)
