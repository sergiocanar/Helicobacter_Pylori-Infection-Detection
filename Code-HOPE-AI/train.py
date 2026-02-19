import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from os.path import join as path_join
from tqdm import tqdm

from lib.admil import CenterLoss, FocalLoss
from lib.pvtv2_lstm import LSTMModel
from h_pylori_datasets import BagDataset, bag_collate


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, center_loss, center_loss_img,
                    loader, optimizer, optimizer_center,
                    criterion, device,
                    topk=7, chunk_size=40, center_weight=0.001):
    model.train()
    center_loss.train()
    center_loss_img.train()

    total_loss = 0.0
    correct = 0
    n_bags = 0

    pbar = tqdm(loader, desc='train', leave=False, unit='bag')
    for bag_tensor, label, _ in pbar:
        # bag_tensor: [N, 3, H, W]  (N varies per patient)
        # label:      scalar long tensor
        bag_tensor = bag_tensor.to(device)
        label_tensor = label.unsqueeze(0).to(device)   # [1]
        N = bag_tensor.shape[0]
        num_chunks = (N + chunk_size - 1) // chunk_size

        optimizer.zero_grad()
        optimizer_center.zero_grad()

        # ---- Phase 1: instance loss over ALL frames + collect softmax for ranking ----
        # Each chunk's graph is freed after its own backward() call, so memory stays
        # bounded regardless of bag size. Gradients accumulate in model parameters.
        all_softmax = []
        inst_loss_total = 0.0

        for i in range(0, N, chunk_size):
            chunk = bag_tensor[i:i + chunk_size].unsqueeze(0)   # [1, c, 3, H, W]
            c = chunk.shape[1]

            _, x_logits_chunk, x_softmax_chunk = model(chunk, is_train=False)
            # x_logits_chunk: [1, c, 2],  x_softmax_chunk: [1, c, 2]

            # Instance CE/Focal loss — scale by 1/num_chunks so the total
            # instance gradient weight equals 0.5 * loss_inst (as before)
            loss_chunk = 0.5 * criterion(
                x_logits_chunk.squeeze(0),          # [c, 2]
                label_tensor.expand(c),             # [c]
            ) / num_chunks
            loss_chunk.backward()                   # accumulate grads; graph freed

            all_softmax.append(x_softmax_chunk.detach().squeeze(0))  # [c, 2]
            inst_loss_total += loss_chunk.item()

        all_softmax = torch.cat(all_softmax, dim=0)     # [N, 2]

        # ---- Top-k selection (on detached scores — no grad needed here) ----
        topk_k = min(N, topk)
        topk_indices = torch.topk(all_softmax[:, 1], topk_k)[1]
        topk_imgs = bag_tensor[topk_indices]            # [k, 3, H, W]

        # ---- Phase 2: LSTM bag-level pass on top-k (grad accumulates further) ----
        # img_feat:  [1, k, 512]  (normalized image features)
        # x_logits:  [1, k, 2]   (per-frame logits)
        # bag_feat:  [1, 256]    (normalized LSTM bag feature)
        # bag_out:   [1, 2]      (bag-level logits)
        img_feat, x_logits, bag_feat, bag_out = model(
            topk_imgs.unsqueeze(0), is_train=True
        )

        topk_labels = label_tensor.expand(topk_k)      # [k]

        loss_bag        = criterion(bag_out, label_tensor)
        loss_center_bag = center_loss(bag_feat, label_tensor)          * center_weight
        loss_center_img = center_loss_img(img_feat.squeeze(0), topk_labels) * center_weight

        loss_phase2 = loss_bag + loss_center_bag + loss_center_img
        loss_phase2.backward()

        optimizer.step()

        # Standard center-loss gradient rescaling before updating centers
        for param in center_loss.parameters():
            param.grad.data *= (1.0 / center_weight)
        for param in center_loss_img.parameters():
            param.grad.data *= (1.0 / center_weight)
        optimizer_center.step()

        total_bag_loss = inst_loss_total + loss_phase2.item()
        pred = bag_out.argmax(dim=1)
        correct    += (pred == label_tensor).sum().item()
        total_loss += total_bag_loss
        n_bags     += 1

        pbar.set_postfix(loss=f'{total_loss/n_bags:.4f}', acc=f'{correct/n_bags:.4f}')

    return total_loss / n_bags, correct / n_bags


@torch.no_grad()
def validate(model, center_loss, center_loss_img,
             loader, criterion, device,
             topk=7, chunk_size=40, center_weight=0.001):
    model.eval()
    center_loss.eval()
    center_loss_img.eval()

    total_loss = 0.0
    correct = 0
    n_bags = 0

    pbar = tqdm(loader, desc='val  ', leave=False, unit='bag')
    for bag_tensor, label, _ in pbar:
        bag_tensor   = bag_tensor.to(device)
        label_tensor = label.unsqueeze(0).to(device)
        N = bag_tensor.shape[0]

        # Phase 1: all frames → top-k
        all_softmax = []
        for i in range(0, N, chunk_size):
            chunk = bag_tensor[i:i + chunk_size].unsqueeze(0)
            _, _, x_softmax = model(chunk, is_train=False)
            all_softmax.append(x_softmax.squeeze(0))
        all_softmax = torch.cat(all_softmax, dim=0)

        topk_k = min(N, topk)
        topk_indices = torch.topk(all_softmax[:, 1], topk_k)[1]
        topk_imgs = bag_tensor[topk_indices]

        # Phase 2: LSTM bag prediction
        img_feat, x_logits, bag_feat, bag_out = model(
            topk_imgs.unsqueeze(0), is_train=True
        )

        topk_labels = label_tensor.expand(topk_k)

        loss_bag        = criterion(bag_out, label_tensor)
        loss_inst       = criterion(x_logits.squeeze(0), topk_labels)
        loss_center_bag = center_loss(bag_feat, label_tensor)     * center_weight
        loss_center_img = center_loss_img(img_feat.squeeze(0), topk_labels) * center_weight

        loss = loss_bag + 0.5 * loss_inst + loss_center_bag + loss_center_img

        pred = bag_out.argmax(dim=1)
        correct    += (pred == label_tensor).sum().item()
        total_loss += loss.item()
        n_bags     += 1

        pbar.set_postfix(loss=f'{total_loss/n_bags:.4f}', acc=f'{correct/n_bags:.4f}')

    return total_loss / n_bags, correct / n_bags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HOPE AI training")
    parser.add_argument('--fold',          type=int,   default=1,     choices=[1, 2, 3],
                        help='Cross-validation fold to train (1, 2 or 3)')
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--lr',            type=float, default=1e-5,
                        help='Learning rate for the main model')
    parser.add_argument('--lr_center',     type=float, default=0.5,
                        help='Learning rate for center loss centers (SGD)')
    parser.add_argument('--center_weight', type=float, default=0.001,
                        help='Scalar weight applied to center losses')
    parser.add_argument('--topk',         type=int,   default=7,
                        help='Number of top-scoring frames used for LSTM')
    parser.add_argument('--img_size',     type=int,   default=352)
    parser.add_argument('--chunk_size',   type=int,   default=40,
                        help='Frames processed per forward chunk (Phase 1)')
    parser.add_argument('--num_workers',  type=int,   default=4,
                        help='DataLoader workers (0 = main process)')
    parser.add_argument('--save_dir',     type=str,   default='checkpoints')
    parser.add_argument('--pretrained',  action='store_true',
                        help='Load author weights from weights/ before training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    this_dir   = os.path.dirname(os.path.abspath(__file__))
    data_root  = path_join(this_dir, 'data', 'GrastroHUN_Hpylori')
    frames_root = path_join(data_root, 'frames')
    labels_root = path_join(data_root, 'labels')

    train_csv = path_join(labels_root, f'fold{args.fold}_train.csv')
    val_csv   = path_join(labels_root, f'fold{args.fold}_val.csv')

    train_ds = BagDataset(train_csv, frames_root, img_size=args.img_size)
    val_ds   = BagDataset(val_csv,   frames_root, img_size=args.img_size)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=args.num_workers, collate_fn=bag_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=bag_collate
    )

    print(f"Fold {args.fold}: {len(train_ds)} train bags | {len(val_ds)} val bags")

    # ---- Models ----
    model               = LSTMModel().to(device)
    model_center_loss     = CenterLoss(num_classes=2, feat_dim=256).to(device)
    model_center_loss_img = CenterLoss(num_classes=2, feat_dim=512).to(device)

    if args.pretrained:
        weight_dir = path_join(this_dir, 'weights')
        model.load_state_dict(
            torch.load(path_join(weight_dir, 'model.pth'), map_location='cpu'),
            strict=False,
        )
        model_center_loss.load_state_dict(
            torch.load(path_join(weight_dir, 'center_loss.pth'), map_location='cpu'),
            strict=False,
        )
        model_center_loss_img.load_state_dict(
            torch.load(path_join(weight_dir, 'center_loss_img.pth'), map_location='cpu'),
            strict=False,
        )
        print('Loaded pretrained weights from weights/')

    # ---- Optimizers ----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    optimizer_center = torch.optim.SGD(
        list(model_center_loss.parameters()) +
        list(model_center_loss_img.parameters()),
        lr=args.lr_center
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    criterion = FocalLoss(gamma=2)

    os.makedirs(args.save_dir, exist_ok=True)

    best_val_acc = 0.0

    epoch_pbar = tqdm(range(1, args.epochs + 1), desc=f'fold{args.fold}', unit='epoch')
    for epoch in epoch_pbar:
        train_loss, train_acc = train_one_epoch(
            model, model_center_loss, model_center_loss_img,
            train_loader, optimizer, optimizer_center,
            criterion, device,
            topk=args.topk, chunk_size=args.chunk_size,
            center_weight=args.center_weight,
        )
        val_loss, val_acc = validate(
            model, model_center_loss, model_center_loss_img,
            val_loader, criterion, device,
            topk=args.topk, chunk_size=args.chunk_size,
            center_weight=args.center_weight,
        )
        scheduler.step()

        epoch_pbar.set_postfix(
            tr_loss=f'{train_loss:.4f}', tr_acc=f'{train_acc:.4f}',
            vl_loss=f'{val_loss:.4f}',  vl_acc=f'{val_acc:.4f}',
        )
        tqdm.write(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f}  acc: {val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_path = path_join(args.save_dir, f'fold{args.fold}_best.pth')
            torch.save({
                'epoch':             epoch,
                'model':             model.state_dict(),
                'center_loss':       model_center_loss.state_dict(),
                'center_loss_img':   model_center_loss_img.state_dict(),
                'optimizer':         optimizer.state_dict(),
                'val_acc':           val_acc,
            }, save_path)
            tqdm.write(f"  => Best model saved  (val_acc={val_acc:.4f})")

    tqdm.write(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
