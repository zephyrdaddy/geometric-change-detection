# viz_hdf5.py - Visualize your generated HDF5 data

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_sample(h5_file: str, sample_idx: int = 0):
    with h5py.File(h5_file, 'r') as f:
        grp_name = f"sample_{sample_idx}"
        if grp_name not in f:
            print(f"[WARN] {grp_name} not found in {h5_file}")
            return None
        grp = f[grp_name]
        return {
            'P': np.array(grp['P']),
            'Q': np.array(grp['Q']),
            'mask_p': np.array(grp['mask_p']),
            'mask_q': np.array(grp['mask_q']),
            'change_p': np.array(grp['change_p']),
            'change_q': np.array(grp['change_q']),
            'y_global': np.array(grp['y_global']),
        }


def plot_sample(sample, out_path: str, title: str = ""):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    mask_p = sample['mask_p'] > 0.5
    mask_q = sample['mask_q'] > 0.5
    
    P_valid = sample['P'][mask_p]
    Q_valid = sample['Q'][mask_q]
    change_p = sample['change_p'][mask_p]
    change_q = sample['change_q'][mask_q]
    
    # P heatmap
    sc1 = ax1.scatter(P_valid[:, 0], P_valid[:, 1], c=change_p, 
                      cmap='RdYlBu_r', s=15, alpha=0.8)
    ax1.set_title(f'P ({np.mean(change_p):.0%} changed)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=ax1)
    
    # Q heatmap
    sc2 = ax2.scatter(Q_valid[:, 0], Q_valid[:, 1], c=change_q, 
                      cmap='RdYlBu_r', s=15, alpha=0.8)
    ax2.set_title(f'Q ({np.mean(change_q):.0%} changed)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=ax2)
    
    # Overlay
    ax3.scatter(P_valid[:, 0], P_valid[:, 1], c='blue', s=10, alpha=0.6, label='P')
    ax3.scatter(Q_valid[:, 0], Q_valid[:, 1], c='orange', s=10, alpha=0.6, label='Q')
    ax3.set_title(f'Global: {"CHANGED" if sample["y_global"] > 0.5 else "NO"}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    # Save instead of show (works in headless Docker)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    # Directory with your HDF5 files
    data_dir = Path('data/generated')
    h5_files = sorted(data_dir.glob('*.h5'))
    if not h5_files:
        print(f"No .h5 files found in {data_dir}")
        raise SystemExit

    # How many files and samples per file to visualize
    max_files = 3      # first 3 files
    samples_per_file = 100  # first 3 samples in each

    for fi, h5_file in enumerate(h5_files[:max_files]):
        print(f"\n=== File {fi}: {h5_file.name} ===")
        for si in range(samples_per_file):
            sample = load_sample(str(h5_file), si)
            if sample is None:
                continue
            title = f"{h5_file.name} / sample_{si}"
            out_name = f"viz_{h5_file.stem}_sample_{si}.png"
            out_path = data_dir / out_name
            plot_sample(sample, str(out_path), title)

    print("\n✅ Visualization done. Check PNG files under data/generated/")
