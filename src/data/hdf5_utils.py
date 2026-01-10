import h5py
import numpy as np
from typing import Dict, Tuple
import os

def save_sample_to_hdf5(filename: str, sample: Dict):
    """Save single sample to HDF5 file."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('P', data=sample['P'])
        f.create_dataset('Q', data=sample['Q'])
        f.create_dataset('mask_p', data=sample['mask_p'])
        f.create_dataset('mask_q', data=sample['mask_q'])
        f.create_dataset('change_p', data=sample['change_p'])
        f.create_dataset('change_q', data=sample['change_q'])
        # Metadata
        f.attrs['n_points'] = sample['P'].shape[0]

def load_sample_from_hdf5(filename: str) -> Dict:
    """Load single sample from HDF5."""
    with h5py.File(filename, 'r') as f:
        return {
            'P': np.array(f['P']),
            'Q': np.array(f['Q']),
            'mask_p': np.array(f['mask_p']),
            'mask_q': np.array(f['mask_q']),
            'change_p': np.array(f['change_p']),
            'change_q': np.array(f['change_q'])
        }

def generate_dataset_to_files(n_samples: int, output_dir: str, n_points: int = 512, 
                           batch_size: int = 1000, overwrite: bool = False):
    """Generate dataset and save to multiple HDF5 files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(0, n_samples, batch_size):
        batch = []
        for j in range(batch_size):
            if i + j >= n_samples:
                break
            dataset = ChangeDetectionDataset(size=1, n_points_per_cloud=n_points)
            sample = dataset[0]
            batch.append(sample)
        
        # Save batch
        filename = os.path.join(output_dir, f'batch_{i//batch_size:04d}.h5')
        if overwrite or not os.path.exists(filename):
            with h5py.File(filename, 'w') as f:
                for idx, sample in enumerate(batch):
                    grp = f.create_group(f'sample_{idx}')
                    grp.create_dataset('P', data=sample['P'])
                    grp.create_dataset('Q', data=sample['Q'])
                    grp.create_dataset('mask_p', data=sample['mask_p'])
                    grp.create_dataset('mask_q', data=sample['mask_q'])
                    grp.create_dataset('change_p', data=sample['change_p'])
                    grp.create_dataset('change_q', data=sample['change_q'])
            print(f"Saved {len(batch)} samples to {filename}")

