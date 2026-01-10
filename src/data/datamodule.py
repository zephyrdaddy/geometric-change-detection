import lightning as L
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import h5py
import torch
import numpy as np
import random

class H5ChangeDataset(Dataset):
    def __init__(self, files):
        self.files = files
        self.samples = []
        for f in self.files:
            with h5py.File(f, "r") as h5f:
                self.samples += [(f, k) for k in h5f.keys()]

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        f, key = self.samples[idx]
        with h5py.File(f, "r") as h5f:
            grp = h5f[key]
            return {
                "P": torch.from_numpy(np.array(grp["P"], dtype=np.float32)),
                "Q": torch.from_numpy(np.array(grp["Q"], dtype=np.float32)),
                "change_p": torch.from_numpy(np.array(grp["change_p"], dtype=np.float32)),
                "change_q": torch.from_numpy(np.array(grp["change_q"], dtype=np.float32)),
            }

def collate_fn(batch):
    # (Your existing padding logic stays here, but inside the module file)
    P_list = [s['P'] for s in batch]
    Q_list = [s['Q'] for s in batch]
    cp_list = [s['change_p'] for s in batch]
    cq_list = [s['change_q'] for s in batch]

    def pad_tensor(tensors, dim_feat):
        max_n = max([t.shape[0] for t in tensors])
        B = len(tensors)
        padded = torch.zeros(B, max_n, dim_feat) if dim_feat > 1 else torch.zeros(B, max_n)
        mask = torch.zeros(B, max_n)
        for i, t in enumerate(tensors):
            n = t.shape[0]
            if dim_feat > 1: padded[i, :n, :] = t
            else: padded[i, :n] = t
            mask[i, :n] = 1
        return padded, mask

    P, mask_p = pad_tensor(P_list, 2)
    Q, mask_q = pad_tensor(Q_list, 2)
    change_p, _ = pad_tensor(cp_list, 1)
    change_q, _ = pad_tensor(cq_list, 1)

    return {"P": P, "Q": Q, "mask_p": mask_p, "mask_q": mask_q, "change_p": change_p, "change_q": change_q}

class GeometricDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 8, split_ratio: float = 0.8):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        all_files = sorted(Path(self.hparams.data_dir).glob("*.h5"))
        random.seed(42) # Ensure consistent split
        random.shuffle(all_files)
        n_train = int(len(all_files) * self.hparams.split_ratio)
        
        if stage == "fit" or stage is None:
            self.train_ds = H5ChangeDataset(all_files[:n_train])
            self.val_ds = H5ChangeDataset(all_files[n_train:])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, collate_fn=collate_fn)