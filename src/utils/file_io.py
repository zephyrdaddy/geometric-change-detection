import os
import glob
import h5py
import numpy as np
from src.data.dataset import ChangeDetectionDataset # Updated import path

def generate_hdf5_dataset(
    n_samples: int,
    output_dir: str,
    batch_size: int = 1000,
    n_points: int = 512,
    overwrite: bool = False,
    noise_std: float = 0.05,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    n_files = (n_samples + batch_size - 1) // batch_size
    print(f"Generating {n_samples} samples across {n_files} HDF5 files...")

    for file_idx in range(n_files):
        start = file_idx * batch_size
        end = min(start + batch_size, n_samples)
        n_this_batch = end - start

        filename = os.path.join(output_dir, f"change_dataset_batch_{file_idx:04d}.h5")

        if os.path.exists(filename) and not overwrite:
            print(f"  Skipping (exists): {filename}")
            continue

        print(f"Generating batch {file_idx + 1}/{n_files} ({n_this_batch} samples)")

        # Dataset uses your geometric scene generation logic
        dataset = ChangeDetectionDataset(
            size=n_this_batch,
            n_points=n_points,
            noise_std=noise_std,
        )

        with h5py.File(filename, "w") as f:
            for i in range(n_this_batch):
                sample = dataset[i]
                grp = f.create_group(f"sample_{i}")

                grp.create_dataset("P", data=sample["P"].numpy())
                grp.create_dataset("Q", data=sample["Q"].numpy())
                grp.create_dataset("change_p", data=sample["change_p"].numpy())
                grp.create_dataset("change_q", data=sample["change_q"].numpy())

                grp.attrs["n_points_p"] = sample["P"].shape[0]
                grp.attrs["n_points_q"] = sample["Q"].shape[0]

def count_hdf5_samples(data_dir: str) -> int:
    total = 0
    for h5_file in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
        try:
            with h5py.File(h5_file, "r") as f:
                total += len([k for k in f.keys() if k.startswith("sample_")])
        except Exception:
            continue
    return total