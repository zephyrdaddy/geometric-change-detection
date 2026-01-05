# utils/file_io.py

import os
import argparse
import glob
import h5py
import numpy as np

from data_generators.dataset import ChangeDetectionDataset


def generate_hdf5_dataset(
    n_samples: int,
    output_dir: str,
    batch_size: int = 1000,
    n_points: int = 512,
    overwrite: bool = False,
    noise_std: float = 0.05,
) -> None:
    """
    Generate dataset and save to batched HDF5 files.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_files = (n_samples + batch_size - 1) // batch_size
    print(f"Generating {n_samples} samples across {n_files} HDF5 files...")

    for file_idx in range(n_files):
        start = file_idx * batch_size
        end = min(start + batch_size, n_samples)
        n_this_batch = end - start

        filename = os.path.join(
            output_dir, f"change_dataset_batch_{file_idx:04d}.h5"
        )

        if os.path.exists(filename) and not overwrite:
            print(f"  Skipping (exists): {filename}")
            continue

        print(f"Generating batch {file_idx + 1}/{n_files} ({n_this_batch} samples)")

        # Generate dataset batch (dataset generates samples on init)
        dataset = ChangeDetectionDataset(
            size=n_this_batch,
            n_points=n_points,
            noise_std=noise_std,
        )

        with h5py.File(filename, "w") as f:
            for i in range(n_this_batch):
                print("Generate ", i)
                sample = dataset[i]
                grp = f.create_group(f"sample_{i}")

                grp.create_dataset("P", data=sample["P"].numpy())
                grp.create_dataset("Q", data=sample["Q"].numpy())
                grp.create_dataset("change_p", data=sample["change_p"].numpy())
                grp.create_dataset("change_q", data=sample["change_q"].numpy())

                grp.attrs["n_points_p"] = sample["P"].shape[0]
                grp.attrs["n_points_q"] = sample["Q"].shape[0]
                print("")

        print(f"  Saved: {filename}")


def count_hdf5_samples(data_dir: str) -> int:
    """Count total samples across all HDF5 files."""
    total = 0
    for h5_file in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
        with h5py.File(h5_file, "r") as f:
            for key in f.keys():
                if key.startswith("sample_"):
                    total += 1
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HDF5 datasets")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--output_dir", default="data/generated")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--n_points", type=int, default=512)
    parser.add_argument("--change_prob", type=float, default=0.3)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    generate_hdf5_dataset(
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        n_points=args.n_points,
        overwrite=args.overwrite,
        noise_std=args.noise_std,
    )

    total = count_hdf5_samples(args.output_dir)
    print(f"\n✅ Generated {total} samples in {args.output_dir}")
