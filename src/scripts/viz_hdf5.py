import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

def load_sample(h5_file: str, sample_idx: int = 0) -> Dict | None:
    """Load a sample from HDF5 file, including predictions if available."""
    with h5py.File(h5_file, 'r') as f:
        grp_name = f"sample_{sample_idx}"
        if grp_name not in f:
            print(f"[WARN] {grp_name} not found in {h5_file}")
            return None
        grp = f[grp_name]
        
        data = {
            'P': np.array(grp['P']),
            'Q': np.array(grp['Q']),
            'change_p': np.array(grp['change_p']),
            'change_q': np.array(grp['change_q']),
        }
        
        # Check if predictions exist (added by inference script)
        if 'change_p_pred' in grp:
            data['change_p_pred'] = np.array(grp['change_p_pred'])
            data['change_q_pred'] = np.array(grp['change_q_pred'])
            
        return data

def plot_sample(sample, out_path: str, title: str = ""):
    P, Q = sample['P'], sample['Q']
    cp_gt, cq_gt = sample['change_p'], sample['change_q']
    
    # Check if we have predictions
    has_preds = 'change_p_pred' in sample
    if has_preds:
        cp_pred = sample['change_p_pred']
        cq_pred = sample['change_q_pred']
        # Determine number of columns based on data availability
        cols = 5 
        fig = plt.figure(figsize=(25, 5))
    else:
        cols = 4
        fig = plt.figure(figsize=(20, 5))

    # Colors & Styles
    UNCHANGED_COLOR = '#E0E0E0'
    GT_COLOR_P, GT_COLOR_Q = '#2E7D32', '#1976D2' # Green and Blue for GT
    PRED_COLOR = '#D32F2F' # Red for Predictions
    
    # --- 1. Ground Truth P ---
    ax1 = plt.subplot(1, cols, 1)
    ax1.scatter(P[cp_gt < 0.5, 0], P[cp_gt < 0.5, 1], c=UNCHANGED_COLOR, s=10, alpha=0.5)
    ax1.scatter(P[cp_gt > 0.5, 0], P[cp_gt > 0.5, 1], c=GT_COLOR_P, s=50, edgecolors='black', label='GT Change')
    ax1.set_title("P: Ground Truth")
    ax1.set_aspect('equal')

    # --- 2. Ground Truth Q ---
    ax2 = plt.subplot(1, cols, 2)
    ax2.scatter(Q[cq_gt < 0.5, 0], Q[cq_gt < 0.5, 1], c=UNCHANGED_COLOR, s=10, alpha=0.5)
    ax2.scatter(Q[cq_gt > 0.5, 0], Q[cq_gt > 0.5, 1], c=GT_COLOR_Q, s=50, edgecolors='black')
    ax2.set_title("Q: Ground Truth")
    ax2.set_aspect('equal')

    if has_preds:
        # --- 3. Predicted Changes P ---
        ax3 = plt.subplot(1, cols, 3)
        ax3.scatter(P[cp_pred < 0.5, 0], P[cp_pred < 0.5, 1], c=UNCHANGED_COLOR, s=10, alpha=0.5)
        ax3.scatter(P[cp_pred > 0.5, 0], P[cp_pred > 0.5, 1], c=PRED_COLOR, s=50, edgecolors='black', label='Pred Change')
        ax3.set_title(f"P: Predicted (Thr=0.5)")
        ax3.set_aspect('equal')

        # --- 4. Predicted Changes Q ---
        ax4 = plt.subplot(1, cols, 4)
        ax4.scatter(Q[cq_pred < 0.5, 0], Q[cq_pred < 0.5, 1], c=UNCHANGED_COLOR, s=10, alpha=0.5)
        ax4.scatter(Q[cq_pred > 0.5, 0], Q[cq_pred > 0.5, 1], c=PRED_COLOR, s=50, edgecolors='black')
        ax4.set_title(f"Q: Predicted (Thr=0.5)")
        ax4.set_aspect('equal')
    else:
        # Fallback to overlay if no preds
        ax_over = plt.subplot(1, cols, 3)
        ax_over.scatter(P[cp_gt > 0.5, 0], P[cp_gt > 0.5, 1], c=GT_COLOR_P, s=40, alpha=0.6)
        ax_over.scatter(Q[cq_gt > 0.5, 0], Q[cq_gt > 0.5, 1], c=GT_COLOR_Q, s=40, alpha=0.6)
        ax_over.set_title("Overlay GT")
        ax_over.set_aspect('equal')

    # --- Last Column: Stats ---
    ax_stat = plt.subplot(1, cols, cols)
    ax_stat.axis('off')
    
    stats_text = f"SAMPLES\nP: {len(P)} pts\nQ: {len(Q)} pts\n\n"
    stats_text += f"GT CHANGES\nP: {int(cp_gt.sum())}\nQ: {int(cq_gt.sum())}\n"
    
    if has_preds:
        p_acc = np.mean((cp_pred > 0.5) == (cp_gt > 0.5))
        q_acc = np.mean((cq_pred > 0.5) == (cq_gt > 0.5))
        stats_text += f"\nPRED CHANGES\nP: {int((cp_pred > 0.5).sum())}\nQ: {int((cq_pred > 0.5).sum())}\n"
        stats_text += f"\nACCURACY\nP: {p_acc:.1%}\nQ: {q_acc:.1%}"

    ax_stat.text(0, 0.95, stats_text, transform=ax_stat.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

if __name__ == '__main__':
    # Switch this to your results file
    h5_path = Path("results.h5") 
    
    if not h5_path.exists():
        print(f"File {h5_path} not found. Run inference first!")
    else:
        with h5py.File(h5_path, 'r') as f:
            n_samples = len(f.keys())
        
        # Visualize first 10 samples
        for si in range(min(10, n_samples)):
            sample = load_sample(str(h5_path), si)
            if sample:
                out_name = f"eval_sample_{si}.png"
                plot_sample(sample, out_name, title=f"Model Evaluation: Sample {si}")
                print(f"Saved {out_name}")