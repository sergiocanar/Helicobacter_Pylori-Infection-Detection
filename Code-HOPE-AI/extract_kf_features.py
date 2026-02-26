"""
extract_kf_features.py

Extracts PVTv2 features for KEYFRAMES ONLY (is_keyframe == True), saving one
.pth file per patient under <output_dir>/patient_<id>.pth.

Output per patient:
    {
        'embeddings':  Tensor [N_kf, 512],   # L2-normalised per-keyframe features
        'frame_paths': list[str],            # relative paths (as in CSV)
        'patient_id':  int,
        'label':       int,                  # HP label (0 / 1)
        'n_keyframes': int,
    }

A log file (kf_counts.log) is written to <output_dir> with one CSV line
per patient:
    patient_id,n_keyframes,label
"""

import os
import logging
from os.path import join as path_join

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.pvtv2_lstm import LSTMModel
from h_pylori_datasets.datasets import FrameDataset


def main(frames_dir, labels_dir, ckpt_path, output_dir,
         img_size=352, batch_size=32, num_workers=4, normalize=True):

    os.makedirs(output_dir, exist_ok=True)

    # ---- Logging setup ----
    log_path = path_join(output_dir, 'kf_counts.log')
    log = logging.getLogger('kf_extract')
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    log.addHandler(fh)
    log.addHandler(logging.StreamHandler())
    log.info('patient_id,n_keyframes,label')

    # ---- Device / model ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    full_model = LSTMModel()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    full_model.load_state_dict(state_dict, strict=True)

    extractor = full_model.feature_extractor
    extractor.eval().to(device)
    print('Feature extractor loaded from:', ckpt_path)

    # ---- Dataset (keyframes only, all regions) ----
    csv_path = path_join(labels_dir, 'all.csv')

    ds = FrameDataset(
        [csv_path], frames_dir,
        img_size=img_size,
        keyframe_only=False,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    print(f'Total keyframes: {len(ds)}')

    # ---- Extract features ----
    all_embeddings  = torch.empty(len(ds), 512)
    all_patient_ids = [None] * len(ds)
    all_labels      = [None] * len(ds)

    with torch.no_grad():
        for imgs, pids, labels, indices in tqdm(loader, desc='Extracting', unit='batch'):
            imgs = imgs.to(device)
            x_feature, _, _ = extractor(imgs)   # [B, 512]
            if normalize:
                x_feature = F.normalize(x_feature, dim=-1)
            x_feature = x_feature.cpu()
            for i, idx in enumerate(indices):
                all_embeddings[idx]  = x_feature[i]
                all_patient_ids[idx] = pids[i].item()
                all_labels[idx]      = labels[i].item()

    # ---- Group by patient and save one .pth each ----
    patient_ids_unique = sorted(set(all_patient_ids))
    print(len(patient_ids_unique), 'unique patients found. Saving per-patient .pth files...')
    for pid in patient_ids_unique:
        idxs  = [i for i, p in enumerate(all_patient_ids) if p == pid]
        emb   = all_embeddings[idxs]           # [N_kf, 512]
        fps   = [ds.frame_paths[i] for i in idxs]
        label = all_labels[idxs[0]]
        n_kf  = len(idxs)

        out = {
            'embeddings':  emb,
            'frame_paths': fps,
            'patient_id':  pid,
            'label':       label,
            'n_keyframes': n_kf,
        }
        out_path = path_join(output_dir, f'patient_{pid:03d}.pth')
        torch.save(out, out_path)

        log.info(f'{pid},{n_kf},{label}')

    print(f'\nSaved {len(patient_ids_unique)} patient .pth files → {output_dir}')
    print(f'Log written → {log_path}')


if __name__ == '__main__':
    this_dir   = os.path.dirname(os.path.abspath(__file__))
    ckpt_path  = path_join(this_dir, 'weights', 'model.pth')
    data_dir   = path_join(this_dir, 'data', 'GrastroHUN_Hpylori')
    frames_dir = path_join(data_dir, 'yao_images')
    labels_dir = path_join(data_dir, 'yao_labels')
    output_dir = path_join(data_dir, 'kf_features')

    main(frames_dir=frames_dir, labels_dir=labels_dir,
         ckpt_path=ckpt_path, output_dir=output_dir)
