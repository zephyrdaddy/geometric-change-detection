import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

def load_sample(h5_file: str, sample_idx: int = 0) -> Dict | None:
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
    fig = plt.figure(figsize=(20, 5))
    
    mask_p = sample['mask_p'] > 0.5
    mask_q = sample['mask_q'] > 0.5
    
    P_valid = sample['P'][mask_p]
    Q_valid = sample['Q'][mask_q]
    change_p = sample['change_p'][mask_p]
    change_q = sample['change_q'][mask_q]
    
    # 1. P: Unchanged (gray) vs Changed (red)
    ax1 = plt.subplot(1, 4, 1)
    unchanged_p = P_valid[change_p < 0.5]
    changed_p = P_valid[change_p > 0.5]
    ax1.scatter(unchanged_p[:, 0], unchanged_p[:, 1], c='lightgray', s=20, alpha=0.7, label='Unchanged')
    if len(changed_p) > 0:
        ax1.scatter(changed_p[:, 0], changed_p[:, 1], c='red', s=25, alpha=0.9, edgecolors='darkred', linewidth=0.5, label='Changed')
    ax1.set_title(f'P\n({np.mean(change_p):.0%} changed pts)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, fontsize=8)
    ax1.set_aspect('equal')
    
    # 2. Q: Unchanged (gray) vs Changed (orange)
    ax2 = plt.subplot(1, 4, 2)
    unchanged_q = Q_valid[change_q < 0.5]
    changed_q = Q_valid[change_q > 0.5]
    ax2.scatter(unchanged_q[:, 0], unchanged_q[:, 1], c='lightgray', s=20, alpha=0.7, label='Unchanged')
    if len(changed_q) > 0:
        ax2.scatter(changed_q[:, 0], changed_q[:, 1], c='orange', s=25, alpha=0.9, edgecolors='darkorange', linewidth=0.5, label='Changed')
    ax2.set_title(f'Q\n({np.mean(change_q):.0%} changed pts)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True, fontsize=8)
    ax2.set_aspect('equal')
    
    # 3. Overlay: P(blue/gray) vs Q(red/orange)
    ax3 = plt.subplot(1, 4, 3)
    ax3.scatter(unchanged_p[:, 0], unchanged_p[:, 1], c='blue', s=12, alpha=0.6, label='P unchanged')
    ax3.scatter(changed_p[:, 0], changed_p[:, 1], c='darkblue', s=18, alpha=0.9, label='P changed', edgecolors='navy')
    ax3.scatter(unchanged_q[:, 0], unchanged_q[:, 1], c='red', s=12, alpha=0.6, label='Q unchanged')
    ax3.scatter(changed_q[:, 0], changed_q[:, 1], c='darkorange', s=18, alpha=0.9, label='Q changed', edgecolors='darkred')
    status = "CHANGED" if sample["y_global"] > 0.5 else "NO CHANGE"
    ax3.set_title(f'P vs Q Overlay\nGlobal: {status}')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # 4. Change heatmap (difference visualization)
    ax4 = plt.subplot(1, 4, 4)
    all_pts = np.vstack([P_valid, Q_valid])
    all_changes = np.concatenate([change_p, change_q])
    sc = ax4.scatter(all_pts[:, 0], all_pts[:, 1], c=all_changes, 
                     cmap='RdYlBu_r', s=15, alpha=0.8)
    ax4.set_title('Change Confidence Heatmap')
    plt.colorbar(sc, ax=ax4, label='Change Score')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"💾 Saved: {out_path} ({np.mean(change_p+change_q)/2:.1%} avg change)")




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
