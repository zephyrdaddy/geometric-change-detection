# utils/file_io.py

import os
import argparse
from typing import List
import h5py
import numpy as np
import glob

from data_generators.dataset import ChangeDetectionDataset


def generate_hdf5_dataset(
    n_samples: int,
    output_dir: str,
    batch_size: int = 1000,
    n_points: int = 512,
    overwrite: bool = False,
    change_prob: float = 0.3,
) -> None:
    """
    Generate dataset and save to batched HDF5 files.
    
    Args:
        n_samples: Total samples to generate
        output_dir: Where to save .h5 files
        batch_size: Samples per HDF5 file
        n_points: Points per cloud
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = ChangeDetectionDataset(
        size=batch_size,
        n_points_per_cloud=n_points,
        change_prob=change_prob,
        mode='generate',
    )
    
    n_files = (n_samples + batch_size - 1) // batch_size
    print(f"Generating {n_samples} samples across {n_files} HDF5 files...")
    
    for file_idx in range(n_files):
        start_idx = file_idx * batch_size
        end_idx = min((file_idx + 1) * batch_size, n_samples)
        n_this_batch = end_idx - start_idx
        
        print(f"Generating batch {file_idx+1}/{n_files} ({n_this_batch} samples)...")
        
        filename = os.path.join(output_dir, f'change_dataset_batch_{file_idx:04d}.h5')
        
        if not overwrite and os.path.exists(filename):
            print(f"  Skipping (exists): {filename}")
            continue
        
        # Generate batch
        batch_samples = []
        temp_dataset = ChangeDetectionDataset(
            size=n_this_batch,
            n_points_per_cloud=n_points,
            change_prob=change_prob,
            mode='generate',
        )
        
        for idx in range(n_this_batch):
            sample = temp_dataset[idx]
            batch_samples.append({
                'P': sample['P'].numpy(),
                'Q': sample['Q'].numpy(),
                'mask_p': sample['mask_p'].numpy(),
                'mask_q': sample['mask_q'].numpy(),
                'change_p': sample['change_p'].numpy(),
                'change_q': sample['change_q'].numpy(),
                'y_global': sample['y_global'].numpy(),
            })
        
        # Save to HDF5
        with h5py.File(filename, 'w') as f:
            for sample_idx, sample_data in enumerate(batch_samples):
                grp = f.create_group(f'sample_{sample_idx}')
                grp.create_dataset('P', data=sample_data['P'])
                grp.create_dataset('Q', data=sample_data['Q'])
                grp.create_dataset('mask_p', data=sample_data['mask_p'])
                grp.create_dataset('mask_q', data=sample_data['mask_q'])
                grp.create_dataset('change_p', data=sample_data['change_p'])
                grp.create_dataset('change_q', data=sample_data['change_q'])
                grp.create_dataset('y_global', data=sample_data['y_global'])
                grp.attrs['n_points'] = sample_data['P'].shape[0]
        
        print(f"  Saved: {filename}")


def count_hdf5_samples(data_dir: str) -> int:
    """Count total samples across all HDF5 files."""
    total = 0
    for h5_file in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
        with h5py.File(h5_file, 'r') as f:
            for key in f.keys():
                if key.startswith('sample_'):
                    total += 1
    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate HDF5 datasets')
    parser.add_argument('--n_samples', type=int, default=10000, help='Total samples')
    parser.add_argument('--output_dir', default='data/generated', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1000, help='Samples per file')
    parser.add_argument('--n_points', type=int, default=512, help='Points per cloud')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    args = parser.parse_args()
    
    generate_hdf5_dataset(
        args.n_samples, args.output_dir, args.batch_size, args.n_points, args.overwrite
    )
    
    total = count_hdf5_samples(args.output_dir)
    print(f"\n✅ Generated {total} samples in {args.output_dir}")
