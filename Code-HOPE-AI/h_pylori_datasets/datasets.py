import os
import random
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Flat per-image dataset (no bag grouping needed here)
# ---------------------------------------------------------------------------

class FrameDataset(Dataset):
    def __init__(self, csv_paths, frames_root, desired_triple_agreement=None, img_size=352):
        dfs = [pd.read_csv(p) for p in csv_paths]
        df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='frame_path')

        if desired_triple_agreement is not None:
            df = df[df["Triple_Agreement"].isin(desired_triple_agreement)]

        self.frames_root = frames_root
        self.frame_paths = df['frame_path'].tolist()
        self.patient_ids = df['patient_id'].astype(int).tolist()
        self.labels = df['HP'].astype(int).tolist()
        self.region = df["Triple_Agreement"].tolist()

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.frames_root, self.frame_paths[idx])
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), self.patient_ids[idx], self.labels[idx], idx


class BagDataset(Dataset):
    """
    MIL bag dataset for H. Pylori detection.

    Each item is one patient (bag) containing all their endoscopy frames.
    Label is patient-level HP status (0 = negative, 1 = positive).

    Expected CSV columns: frame_path, patient_id, HP, OLGA
    frame_path is relative to frames_root (e.g. "video_080/frame_0000.jpg").

    Args:
        max_frames: If set, randomly sample this many frames per bag on every
                    __getitem__ call. All bags then have the same shape
                    [max_frames, 3, H, W], enabling batch_size > 1 in the
                    DataLoader (no custom collate needed). Also acts as data
                    augmentation since a different subset is drawn each epoch.
                    If None (default), the full bag is returned and
                    batch_size must stay at 1 (use bag_collate).
    """

    def __init__(self, csv_path: str, frames_root: str, img_size: int = 352,
                 max_frames: int = None, desired_triple_agreement=None):
        df = pd.read_csv(csv_path)

        if desired_triple_agreement is not None:
            df = df[df["Triple_Agreement"].isin(desired_triple_agreement)]

        self.frames_root = frames_root
        self.max_frames  = max_frames
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Group frames by patient — one bag per patient
        self.bags = []
        for patient_id, group in df.groupby('patient_id'):
            self.bags.append({
                'patient_id': int(patient_id),
                'frame_paths': group['frame_path'].tolist(),
                'label': int(group['HP'].iloc[0]),
            })

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int):
        bag = self.bags[idx]
        frame_paths = bag['frame_paths']

        if self.max_frames is not None and len(frame_paths) > self.max_frames:
            frame_paths = random.sample(frame_paths, self.max_frames)

        images = []
        for fp in frame_paths:
            img_path = os.path.join(self.frames_root, fp)
            img = Image.open(img_path).convert('RGB')
            images.append(self.transform(img))

        bag_tensor = torch.stack(images)  # [N, 3, H, W]  or  [max_frames, 3, H, W]
        return bag_tensor, bag['label'], bag['patient_id']


def bag_collate(batch):
    """
    Custom collate for DataLoader with batch_size=1.
    Returns the bag tensor as-is ([N, 3, H, W]) without extra batch dim stacking,
    since bags have variable numbers of frames.
    """
    bag_tensor, label, patient_id = batch[0]
    return bag_tensor, torch.tensor(label, dtype=torch.long), patient_id


if __name__ == '__main__':

    this_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frames_root = os.path.join(this_dir, 'data', 'GrastroHUN_Hpylori', 'frames')
    labels_root = os.path.join(this_dir, 'data', 'GrastroHUN_Hpylori', 'labels')

    for fold in [1, 2, 3]:
        for split in ['train', 'val']:
            csv_path = os.path.join(labels_root, f'fold{fold}_{split}.csv')
            ds = BagDataset(csv_path, frames_root, img_size=352)

            labels      = [b['label'] for b in ds.bags]
            n_pos       = sum(labels)
            n_neg       = len(labels) - n_pos
            frame_counts = [len(b['frame_paths']) for b in ds.bags]

            print(f"fold{fold} {split:5s} | bags: {len(ds):3d} "
                  f"| pos: {n_pos:3d}  neg: {n_neg:3d} "
                  f"| frames/bag  min:{min(frame_counts):5d}  "
                  f"max:{max(frame_counts):5d}  "
                  f"avg:{sum(frame_counts)//len(frame_counts):5d}")

    # --- DataLoader round-trip: load first bag of fold1 train and check tensor ---
    print("\n--- DataLoader sanity check (fold 1 train, first bag) ---")
    csv_path = os.path.join(labels_root, 'fold1_train.csv')
    ds       = BagDataset(csv_path, frames_root, img_size=352)
    loader   = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=bag_collate)

    bag_tensor, label, pid = next(iter(loader))

    assert bag_tensor.ndim == 4,          "expected [N, 3, H, W]"
    assert bag_tensor.shape[1] == 3,      "expected 3 colour channels"
    assert bag_tensor.shape[2:] == (352, 352), "expected spatial size 352x352"
    assert label.dtype == torch.long,     "label should be long"
    assert label.shape == torch.Size([]), "label should be scalar"

    print(f"  patient_id : {pid}")
    print(f"  bag shape  : {tuple(bag_tensor.shape)}   (frames x C x H x W)")
    print(f"  label      : {label.item()}  (0=neg, 1=pos)")
    print(f"  pixel range: [{bag_tensor.min():.3f}, {bag_tensor.max():.3f}]")
    print("All checks passed.")
