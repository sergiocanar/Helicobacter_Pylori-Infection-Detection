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
    def __init__(self, csv_paths, frames_root, desired_triple_agreement=None, img_size=352,
                 keyframe_only=False):
        dfs = [pd.read_csv(p) for p in csv_paths]
        df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='frame_path')

        self.frames_root = frames_root
        self.frame_paths = df['frame_path'].tolist()
        self.patient_ids = df['patient_id'].astype(int).tolist()
        self.labels = df['HP'].astype(int).tolist()

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


# ---------------------------------------------------------------------------
# Patient keyframe dataset (pre-extracted kf embeddings, one .pth per patient)
# ---------------------------------------------------------------------------

class PatientKFDataset(Dataset):
    """
    Loads pre-extracted keyframe embeddings (one .pth per patient).

    Each item is one patient. Returns:
        embeddings  : Tensor [N_kf, 512]
        label       : int  (HP 0 / 1)
        patient_id  : int

    Args:
        csv_path        : split CSV (frame-level); used only to get the unique
                          patient-id ↔ HP-label mapping for this split.
        kf_features_dir : directory containing patient_<id>.pth files produced
                          by extract_kf_features.py.
    """

    def __init__(self, csv_path: str, kf_features_dir: str):
        df = pd.read_csv(csv_path)
        patient_labels = (
            df.groupby('patient_id')['HP']
            .first()
            .reset_index()
        )
        self.kf_features_dir = kf_features_dir
        self.patients = [
            {'patient_id': int(row['patient_id']), 'label': int(row['HP'])}
            for _, row in patient_labels.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int):
        p = self.patients[idx]
        pid = p['patient_id']
        pth_path = os.path.join(self.kf_features_dir, f'patient_{pid:03d}.pth')
        data = torch.load(pth_path, map_location='cpu', weights_only=True)
        embeddings = data['embeddings']   # [N_kf, 512]
        return embeddings, p['label'], pid


def patient_kf_collate(batch):
    """
    Collate for PatientKFDataset with batch_size > 1.
    Pads all bags to the longest sequence in the batch and returns a
    boolean padding mask (True = padded position, to match PyTorch's
    src_key_padding_mask convention).

    Returns:
        padded   : Tensor [B, max_N, 512]
        labels   : Tensor [B]  (long)
        pids     : list[int]
        pad_mask : BoolTensor [B, max_N]  — True where padded
    """
    embeddings_list, labels, pids = zip(*batch)
    max_n = max(e.shape[0] for e in embeddings_list)
    dim   = embeddings_list[0].shape[1]

    padded   = torch.zeros(len(batch), max_n, dim)
    pad_mask = torch.ones(len(batch), max_n, dtype=torch.bool)   # True = pad

    for i, e in enumerate(embeddings_list):
        n = e.shape[0]
        padded[i, :n]   = e
        pad_mask[i, :n] = False   # real tokens

    return padded, torch.tensor(labels, dtype=torch.long), list(pids), pad_mask


# ---------------------------------------------------------------------------
# Keyframe window dataset (for contextualised keyframe embeddings)
# ---------------------------------------------------------------------------

class KeyframeWindowDataset(Dataset):
    """
    For each keyframe in a fold CSV loads a fixed-size window of pre-extracted
    frame embeddings centred on that keyframe from full_seq_features.

    Each item:
        embeddings  : Tensor [window_size, 512]   (zero-padded at boundaries)
        label       : int  (HP 0 / 1)
        patient_id  : int
        kf_idx      : int  (sequential frame index of the keyframe)

    Since every item has the same shape [window_size, 512], a standard
    DataLoader with batch_size > 1 works without a custom collate.

    Args:
        csv_path        : fold CSV with columns frame_path, patient_id, HP,
                          is_keyframe (bool).
        features_dir    : directory containing patient_<id>.pth files whose
                          'embeddings' tensor is [N_frames, 512] with the same
                          sequential frame ordering as the CSV.
        window_size     : total window width (must be odd; e.g. 5 → ±2 frames).
    """

    def __init__(self, csv_path: str, features_dir: str, window_size: int = 5):
        assert window_size % 2 == 1, "window_size must be odd"
        self.half = window_size // 2
        self.window_size = window_size
        self.features_dir = features_dir

        df = pd.read_csv(csv_path)
        kf = df[df['is_keyframe'] == True].copy()
        kf['kf_idx'] = (
            kf['frame_path'].str.extract(r'frame_(\d+)\.jpg')[0].astype(int)
        )

        # Store (patient_id, kf_idx, label) for each keyframe
        self.items = [
            (int(row['patient_id']), int(row['kf_idx']), int(row['HP']))
            for _, row in kf.iterrows()
        ]

        # Cache: patient_id -> embeddings tensor [N, 512]
        self._cache: dict[int, torch.Tensor] = {}

    def _load(self, pid: int) -> torch.Tensor:
        if pid not in self._cache:
            path = os.path.join(self.features_dir, f'patient_{pid:03d}.pth')
            data = torch.load(path, map_location='cpu', weights_only=True)
            self._cache[pid] = data['embeddings']   # [N, 512]
        return self._cache[pid]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        pid, kf_idx, label = self.items[idx]
        emb = self._load(pid)          # [N, 512]
        N, D = emb.shape

        lo = kf_idx - self.half
        hi = kf_idx + self.half + 1    # exclusive

        # Clamp to valid range and record how much padding is needed
        pad_left  = max(0, -lo)
        pad_right = max(0, hi - N)
        lo_c = max(0, lo)
        hi_c = min(N, hi)

        window = emb[lo_c:hi_c]        # [valid_frames, 512]

        if pad_left > 0 or pad_right > 0:
            window = torch.cat([
                torch.zeros(pad_left,  D),
                window,
                torch.zeros(pad_right, D),
            ], dim=0)

        return window, label, pid, kf_idx


# ---------------------------------------------------------------------------
# Patient-level window dataset (end-to-end training with WindowKFContextualizer)
# ---------------------------------------------------------------------------

class PatientWindowDataset(Dataset):
    """
    Patient-level dataset for end-to-end training of WindowKFContextualizer
    + HPTransformerClassifier.

    Each item is one patient. Returns:
        windows      : Tensor [N_kf, window_size, 512]  — window of frame
                       embeddings centred on each keyframe (zero-padded at
                       patient boundaries)
        win_pad_mask : BoolTensor [N_kf, window_size]   — True = zero-padded
                       position (boundary of video)
        label        : int  (HP 0 / 1)
        patient_id   : int

    Use patient_window_collate to stack variable-length bags into batches.

    Args:
        csv_path      : fold CSV with is_keyframe column
        features_dir  : directory containing patient_<id>.pth files
                        (full_seq_features/fold{k}_{split}/)
        window_size   : odd integer, total window width per keyframe
    """

    def __init__(self, csv_path: str, features_dir: str, window_size: int = 11):
        assert window_size % 2 == 1, "window_size must be odd"
        self.half         = window_size // 2
        self.window_size  = window_size
        self.features_dir = features_dir

        df = pd.read_csv(csv_path)
        kf = df[df['is_keyframe'] == True].copy()
        kf['kf_idx'] = (
            kf['frame_path'].str.extract(r'frame_(\d+)\.jpg')[0].astype(int)
        )

        # Group by patient: list of (label, [kf_idx, ...])
        self.patients = []
        for pid, grp in kf.groupby('patient_id'):
            self.patients.append({
                'patient_id': int(pid),
                'label':      int(grp['HP'].iloc[0]),
                'kf_indices': grp['kf_idx'].tolist(),
            })

        self._cache: dict[int, torch.Tensor] = {}

    def _load(self, pid: int) -> torch.Tensor:
        if pid not in self._cache:
            path = os.path.join(self.features_dir, f'patient_{pid:03d}.pth')
            data = torch.load(path, map_location='cpu', weights_only=True)
            self._cache[pid] = data['embeddings']   # [N_frames, 512]
        return self._cache[pid]

    def _window(self, emb: torch.Tensor, kf_idx: int):
        N, D = emb.shape
        lo, hi    = kf_idx - self.half, kf_idx + self.half + 1
        pad_left  = max(0, -lo)
        pad_right = max(0, hi - N)
        lo_c, hi_c = max(0, lo), min(N, hi)
        window = emb[lo_c:hi_c]
        if pad_left > 0 or pad_right > 0:
            window = torch.cat([
                torch.zeros(pad_left,  D),
                window,
                torch.zeros(pad_right, D),
            ], dim=0)
        mask = torch.zeros(self.window_size, dtype=torch.bool)
        if pad_left  > 0: mask[:pad_left]  = True
        if pad_right > 0: mask[-pad_right:] = True
        return window, mask   # [W, 512], [W]

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int):
        p   = self.patients[idx]
        emb = self._load(p['patient_id'])

        windows, masks = [], []
        for kf_idx in p['kf_indices']:
            w, m = self._window(emb, kf_idx)
            windows.append(w)
            masks.append(m)

        return (
            torch.stack(windows),          # [N_kf, W, 512]
            torch.stack(masks),            # [N_kf, W]
            p['label'],
            p['patient_id'],
        )


def patient_window_collate(batch):
    """
    Collate for PatientWindowDataset.  Pads N_kf to the longest bag in the
    batch and returns a patient-level padding mask.

    Returns:
        windows      : [B, max_N, W, 512]
        win_pad_mask : [B, max_N, W]       True = zero-padded window frame
        pat_pad_mask : [B, max_N]          True = padded patient slot
        labels       : [B]  long
        pids         : list[int]
    """
    windows_list, wmask_list, labels, pids = zip(*batch)
    max_n = max(w.shape[0] for w in windows_list)
    W, D  = windows_list[0].shape[1], windows_list[0].shape[2]
    B     = len(batch)

    windows      = torch.zeros(B, max_n, W, D)
    win_pad_mask = torch.ones(B,  max_n, W, dtype=torch.bool)   # True = padded
    pat_pad_mask = torch.ones(B,  max_n,    dtype=torch.bool)

    for i, (w, m) in enumerate(zip(windows_list, wmask_list)):
        n = w.shape[0]
        windows[i, :n]       = w
        win_pad_mask[i, :n]  = m
        pat_pad_mask[i, :n]  = False   # real keyframes

    return (
        windows,
        win_pad_mask,
        pat_pad_mask,
        torch.tensor(labels, dtype=torch.long),
        list(pids),
    )


class YaoFrameDataset(Dataset):
    """Frame-level dataset from Yao H. Pylori CSV splits.

    Args:
        csv_path  : path to fold CSV (columns: frame_path, patient_id, HP, OLGA)
        images_root : root directory that contains video_XXX/ sub-folders
        transform   : torchvision transform
    """

    def __init__(self, csv_path: str, images_root: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_root, row["frame_path"])
        label = int(row["HP"])

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, label



if __name__ == '__main__':

    this_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    labels_root  = os.path.join(this_dir, 'data', 'GrastroHUN_Hpylori', 'labels')
    features_dir = os.path.join(this_dir, 'data', 'GrastroHUN_Hpylori',
                                'full_seq_features', 'fold1_val')

    print('--- KeyframeWindowDataset checker ---')
    ds = KeyframeWindowDataset(
        csv_path=os.path.join(labels_root, 'fold1_val.csv'),
        features_dir=features_dir,
        window_size=11,
    )
    print(f'Keyframes: {len(ds)}')

    # Sample a few windows and print stats
    for i in [0, len(ds)//2, len(ds)-1]:
        window, label, pid, kf_idx = ds[i]
        pad_left  = (window == 0).all(dim=1).cumsum(0).argmin().item()
        pad_right = (window == 0).all(dim=1).flip(0).cumsum(0).argmin().item()
        print(f'  [{i:4d}] patient={pid:3d}  kf_idx={kf_idx:5d}  '
              f'label={label}  shape={tuple(window.shape)}  '
              f'pad=({pad_left},{pad_right})')

    # Batch test
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    batch = next(iter(loader))
    print(f'Batch shape: {tuple(batch[0].shape)}')
    print('All checks passed.')