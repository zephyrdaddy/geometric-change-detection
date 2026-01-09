# train_bidirectional.py

import os
from pathlib import Path
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.changenet_mlp import ChangeNet
# ===============================================
# Dataset
# ===============================================
class H5ChangeDataset(Dataset):
    """HDF5 dataset for bidirectional point cloud change detection."""
    def __init__(self, data_dir=None, files=None):
        if files is None:
            self.files = sorted(Path(data_dir).glob("*.h5"))
        else:
            self.files = files

        self.samples = []
        for f in self.files:
            with h5py.File(f, "r") as h5f:
                self.samples += [(f, k) for k in h5f.keys()]
    # def __init__(self, h5_files):
    #     self.samples = []
    #     for f in h5_files:
    #         with h5py.File(f, "r") as hf:
    #             for k in hf.keys():
    #                 self.samples.append((str(f), k))

    # def __init__(self, data_dir: str):
    #     self.files = sorted(Path(data_dir).glob("*.h5"))
    #     self.samples = []
    #     for f in self.files:
    #         with h5py.File(f, "r") as h5f:
    #             self.samples += [(f, i) for i in range(len(h5f.keys()))]

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        f, key = self.samples[idx]   # key is already "sample_XXX"
        with h5py.File(f, "r") as h5f:
            grp = h5f[key]

            P = np.array(grp["P"], dtype=np.float32)
            Q = np.array(grp["Q"], dtype=np.float32)
            change_p = np.array(grp["change_p"], dtype=np.float32)
            change_q = np.array(grp["change_q"], dtype=np.float32)

        return {
            "P": torch.from_numpy(P),
            "Q": torch.from_numpy(Q),
            "change_p": torch.from_numpy(change_p),
            "change_q": torch.from_numpy(change_q),
        }


    # def __getitem__(self, idx):
    #     f, si = self.samples[idx]
    #     with h5py.File(f, "r") as h5f:
    #         grp = h5f[f"sample_{si}"]
    #         P = np.array(grp["P"], dtype=np.float32)
    #         Q = np.array(grp["Q"], dtype=np.float32)
    #         change_p = np.array(grp["change_p"], dtype=np.float32)
    #         change_q = np.array(grp["change_q"], dtype=np.float32)

    #     return {
    #         "P": torch.from_numpy(P),        # [n_p, 2]
    #         "Q": torch.from_numpy(Q),        # [n_q, 2]
    #         "change_p": torch.from_numpy(change_p),  # [n_p]
    #         "change_q": torch.from_numpy(change_q),  # [n_q]
    #     }


# def collate_fn(batch):
#     """
#     batch: list of samples, each is {'P': [n_p,2], 'Q':[n_q,2], 'change_p':[n_p], 'change_q':[n_q]}
#     Returns padded tensors and masks
#     """
#     P_list = [torch.tensor(s['P'], dtype=torch.float32) for s in batch]
#     Q_list = [torch.tensor(s['Q'], dtype=torch.float32) for s in batch]
#     change_p_list = [torch.tensor(s['change_p'], dtype=torch.float32) for s in batch]
#     change_q_list = [torch.tensor(s['change_q'], dtype=torch.float32) for s in batch]

#     # Pad P
#     max_n_p = max([p.shape[0] for p in P_list])
#     P_padded = torch.zeros(len(P_list), max_n_p, 2)
#     mask_p = torch.zeros(len(P_list), max_n_p)
#     change_p_padded = torch.zeros(len(P_list), max_n_p)
#     for i, (p, c) in enumerate(zip(P_list, change_p_list)):
#         n = p.shape[0]
#         P_padded[i, :n] = p
#         mask_p[i, :n] = 1
#         change_p_padded[i, :n] = c

#     # Pad Q
#     max_n_q = max([q.shape[0] for q in Q_list])
#     Q_padded = torch.zeros(len(Q_list), max_n_q, 2)
#     mask_q = torch.zeros(len(Q_list), max_n_q)
#     change_q_padded = torch.zeros(len(Q_list), max_n_q)
#     for i, (q, c) in enumerate(zip(Q_list, change_q_list)):
#         n = q.shape[0]
#         Q_padded[i, :n] = q
#         mask_q[i, :n] = 1
#         change_q_padded[i, :n] = c

#     return {
#         'P': P_padded,
#         'Q': Q_padded,
#         'mask_p': mask_p,
#         'mask_q': mask_q,
#         'change_p': change_p_padded,
#         'change_q': change_q_padded,
#     }



def collate_fn(batch):
    """
    batch: list of samples, each is {'P': [n_p,2], 'Q':[n_q,2], 'change_p':[n_p], 'change_q':[n_q]}
    Returns:
        P: [B, max_n_p, 2]
        Q: [B, max_n_q, 2]
        mask_p: [B, max_n_p]
        mask_q: [B, max_n_q]
        change_p: [B, max_n_p]
        change_q: [B, max_n_q]
    """
    P_list = [torch.tensor(s['P'], dtype=torch.float32) for s in batch]
    Q_list = [torch.tensor(s['Q'], dtype=torch.float32) for s in batch]
    change_p_list = [torch.tensor(s['change_p'], dtype=torch.float32) for s in batch]
    change_q_list = [torch.tensor(s['change_q'], dtype=torch.float32) for s in batch]

    # Pad P
    max_n_p = max([p.shape[0] for p in P_list])
    B = len(batch)
    P_padded = torch.zeros(B, max_n_p, 2)
    mask_p = torch.zeros(B, max_n_p)
    change_p_padded = torch.zeros(B, max_n_p)
    for i, (p, c) in enumerate(zip(P_list, change_p_list)):
        n = p.shape[0]
        P_padded[i, :n] = p
        mask_p[i, :n] = 1
        change_p_padded[i, :n] = c

    # Pad Q
    max_n_q = max([q.shape[0] for q in Q_list])
    Q_padded = torch.zeros(B, max_n_q, 2)
    mask_q = torch.zeros(B, max_n_q)
    change_q_padded = torch.zeros(B, max_n_q)
    for i, (q, c) in enumerate(zip(Q_list, change_q_list)):
        n = q.shape[0]
        Q_padded[i, :n] = q
        mask_q[i, :n] = 1
        change_q_padded[i, :n] = c

    return {
        'P': P_padded,
        'Q': Q_padded,
        'mask_p': mask_p,
        'mask_q': mask_q,
        'change_p': change_p_padded,
        'change_q': change_q_padded,
    }

# def collate_fn(batch):
#     """Batch variable-length point clouds as lists."""
#     P = [b["P"] for b in batch]
#     Q = [b["Q"] for b in batch]
#     change_p = [b["change_p"] for b in batch]
#     change_q = [b["change_q"] for b in batch]
#     return {"P": P, "Q": Q, "change_p": change_p, "change_q": change_q}




# ===============================================
# Training
# ===============================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Move everything to device
        P = batch["P"].to(device)
        Q = batch["Q"].to(device)
        mask_p = batch["mask_p"].to(device)
        mask_q = batch["mask_q"].to(device)
        target_p = batch["change_p"].to(device)
        target_q = batch["change_q"].to(device)

        optimizer.zero_grad()

        # Pass masks to the model for correct global pooling
        pred_q_logits, pred_p_logits = model(P, Q, mask_p=mask_p, mask_q=mask_q)

        # Calculate loss with reduction='none' so we can mask it manually
        loss_q = F.binary_cross_entropy_with_logits(pred_q_logits, target_q, reduction='none')
        loss_p = F.binary_cross_entropy_with_logits(pred_p_logits, target_p, reduction='none')

        # Apply masks: zeros out the loss for padding points
        masked_loss_q = (loss_q * mask_q).sum() / mask_q.sum()
        masked_loss_p = (loss_p * mask_p).sum() / mask_p.sum()

        loss = masked_loss_q + masked_loss_p

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        P = batch["P"].to(device)
        Q = batch["Q"].to(device)
        target_p = batch["change_p"].to(device)
        target_q = batch["change_q"].to(device)
        mask_p = batch["mask_p"].to(device)
        mask_q = batch["mask_q"].to(device)

        pred_q, pred_p = model(P, Q)

        loss_q = F.binary_cross_entropy_with_logits(pred_q, target_q, reduction='none')
        loss_p = F.binary_cross_entropy_with_logits(pred_p, target_p, reduction='none')
        
        loss = ((loss_q * mask_q).sum() / mask_q.sum()) + ((loss_p * mask_p).sum() / mask_p.sum())
        total_loss += loss.item()

    return total_loss / len(loader)


# ===============================================
# Main
# ===============================================
def main():
    import argparse
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/generated', help='Folder with HDF5 files')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    data_dir = args.data_dir
    n_epochs = args.epochs
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_files = sorted(Path(data_dir).glob("*.h5"))
    random.shuffle(all_files)

    split_ratio = 0.8
    n_train = int(len(all_files) * split_ratio)

    train_files = all_files[:n_train]
    val_files   = all_files[n_train:]

    train_dataset = H5ChangeDataset(files=train_files)
    val_dataset = H5ChangeDataset(files=val_files)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = ChangeNet(in_dim=2, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best_val_loss = 1e10
    n_epochs = 50
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_change_model.pth")
            print(f"⭐ New best model saved at epoch {epoch+1}")
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save model
    # torch.save(model.state_dict(), "best_change_model.pth")
    # print("Model saved as best_change_model.pth")


if __name__ == "__main__":
    main()
