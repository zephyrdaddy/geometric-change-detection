import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

def load_sample(h5_file: str, sample_idx: int = 0) -> Dict | None:
    """Load a sample from HDF5 file."""
    with h5py.File(h5_file, 'r') as f:
        grp_name = f"sample_{sample_idx}"
        if grp_name not in f:
            print(f"[WARN] {grp_name} not found in {h5_file}")
            return None
        grp = f[grp_name]
        return {
            'P': np.array(grp['P']),
            'Q': np.array(grp['Q']),
            'change_p': np.array(grp['change_p']),
            'change_q': np.array(grp['change_q']),
            # optional: if you have global label
            'y_global': np.array(grp.get('y_global', 0)),
        }

def plot_sample(sample, out_path: str, title: str = ""):
    P = sample['P']
    Q = sample['Q']
    change_p = sample['change_p']
    change_q = sample['change_q']

    fig = plt.figure(figsize=(24, 6))

    # Colors
    UNCHANGED_COLOR = '#A0A0AF'  # darker gray for visibility
    CHANGED_COLOR_P = '#D32F2F'  # bright red
    CHANGED_COLOR_Q = '#F57C00'  # bright orange
    CHANGED_EDGE = 'black'

    # --- 1. P points ---
    ax1 = plt.subplot(1, 4, 1)
    ax1.set_facecolor('white')
    ax1.scatter(P[change_p < 0.5, 0], P[change_p < 0.5, 1], c=UNCHANGED_COLOR, s=12, alpha=0.9)
    ax1.scatter(P[change_p > 0.5, 0], P[change_p > 0.5, 1], c=CHANGED_COLOR_P, s=100,
                edgecolors=CHANGED_EDGE, linewidth=1.5, alpha=1.0)
    ax1.set_title(f"P: {np.mean(change_p):.0%} CHANGED\n({len(P)} pts)", fontsize=12, fontweight='bold')
    ax1.set_aspect('equal'); ax1.grid(True, alpha=0.2)

    # --- 2. Q points ---
    ax2 = plt.subplot(1, 4, 2)
    ax2.set_facecolor('white')
    ax2.scatter(Q[change_q < 0.5, 0], Q[change_q < 0.5, 1], c=UNCHANGED_COLOR, s=12, alpha=0.9)
    ax2.scatter(Q[change_q > 0.5, 0], Q[change_q > 0.5, 1], c=CHANGED_COLOR_Q, s=100,
                edgecolors=CHANGED_EDGE, linewidth=1.5, alpha=1.0)
    ax2.set_title(f"Q: {np.mean(change_q):.0%} CHANGED\n({len(Q)} pts)", fontsize=12, fontweight='bold')
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.2)

    # --- 3. Overlay changed points only ---
    ax3 = plt.subplot(1, 4, 3)
    ax3.set_facecolor('#FAFAFA')
    ax3.scatter(P[change_p < 0.5, 0], P[change_p < 0.5, 1], c='lightblue', s=8, alpha=0.5)
    ax3.scatter(Q[change_q < 0.5, 0], Q[change_q < 0.5, 1], c='lightcoral', s=8, alpha=0.5)
    ax3.scatter(P[change_p > 0.5, 0], P[change_p > 0.5, 1], c=CHANGED_COLOR_P, s=80, alpha=1.0,
                edgecolors=CHANGED_EDGE, linewidth=1.2)
    ax3.scatter(Q[change_q > 0.5, 0], Q[change_q > 0.5, 1], c=CHANGED_COLOR_Q, s=80, alpha=1.0,
                edgecolors=CHANGED_EDGE, linewidth=1.2)
    ax3.set_aspect('equal'); ax3.grid(True, alpha=0.3)
    ax3.set_title('Changes Only', fontsize=12, fontweight='bold')

    # --- 5. Stats ---
    ax5 = plt.subplot(1, 4, 4)
    ax5.axis('off')
    stats_text = f'''
📊 CHANGE STATS
P changed: {np.mean(change_p):.1%} ({len(P)} pts)
Q changed: {np.mean(change_q):.1%} ({len(Q)} pts)
Total changed: {(change_p.sum() + change_q.sum()):.0f} pts
'''
    ax5.text(0.1, 0.95, stats_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    ax5.set_title('Summary', fontsize=12, fontweight='bold')

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"💾 {out_path} | P:{np.mean(change_p):.0%} Q:{np.mean(change_q):.0%}")

if __name__ == '__main__':
    data_dir = Path('data/generated')
    h5_files = sorted(data_dir.glob('*.h5'))
    if not h5_files:
        print(f"No .h5 files found in {data_dir}")
        raise SystemExit

    max_files = 3
    samples_per_file = 100

    for fi, h5_file in enumerate(h5_files[:max_files]):
        print(f"\n=== File {fi}: {h5_file.name} ===")
        with h5py.File(h5_file, 'r') as f:
            n_samples = len(f.keys())
        for si in range(min(samples_per_file, n_samples)):
            sample = load_sample(str(h5_file), si)
            if sample is None:
                continue
            title = f"{h5_file.name} / sample_{si}"
            out_name = f"viz_{h5_file.stem}_sample_{si}.png"
            out_path = data_dir / out_name
            plot_sample(sample, str(out_path), title)

    print("\n✅ Visualization done. Check PNG files under data/generated/")
