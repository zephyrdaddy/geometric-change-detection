# utils/viz.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, Tuple, Optional

def plot_sample(sample: Dict[str, torch.Tensor], 
                save_path: Optional[str] = None, 
                figsize: Tuple[int, int] = (12, 5),
                title: str = "Change Detection Sample") -> plt.Figure:
    """
    Plot P/Q clouds with ground truth predictions.
    
    Args:
        sample: Dict from dataset
        save_path: Save figure to file
        figsize: Figure size
        title: Plot title
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Extract valid points
    mask_p = sample['mask_p'].numpy()
    mask_q = sample['mask_q'].numpy()
    
    P_valid = sample['P'][mask_p.bool()].numpy()
    Q_valid = sample['Q'][mask_q.bool()].numpy()
    
    change_p_gt = sample['change_p'][mask_p.bool()].numpy()
    change_q_gt = sample['change_q'][mask_q.bool()].numpy()
    
    # Plot P cloud (blue=unchanged, red=changed)
    scatter_p = ax1.scatter(P_valid[:, 0], P_valid[:, 1], 
                           c=change_p_gt, cmap='RdYlBu_r', s=10, alpha=0.7)
    ax1.set_title('Cloud P (GT)')
    ax1.set_aspect('equal')
    plt.colorbar(scatter_p, ax=ax1, label='Change Prob')
    
    # Plot Q cloud
    scatter_q = ax2.scatter(Q_valid[:, 0], Q_valid[:, 1], 
                           c=change_q_gt, cmap='RdYlBu_r', s=10, alpha=0.7)
    ax2.set_title('Cloud Q (GT)')
    ax2.set_aspect('equal')
    plt.colorbar(scatter_q, ax=ax2, label='Change Prob')
    
    # Overlay P + Q
    ax3.scatter(P_valid[:, 0], P_valid[:, 1], c='blue', s=8, alpha=0.6, label='P')
    ax3.scatter(Q_valid[:, 0], Q_valid[:, 1], c='orange', s=8, alpha=0.6, label='Q')
    ax3.set_title('P + Q Overlay')
    ax3.legend()
    ax3.set_aspect('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    
    return fig


def plot_predictions(model_path: str, 
                     test_samples: int = 5,
                     save_dir: str = 'viz_predictions') -> None:
    """
    Load model and visualize predictions on test samples.
    """
    import torch.nn as nn
    from models.pointnet_mlp import PointNetMLP
    
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNetMLP().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dataset = ChangeDetectionDataset(size=test_samples, mode='generate')
    
    with torch.no_grad():
        for i in range(test_samples):
            sample = dataset[i]
            P = sample['P'].unsqueeze(0).to(device)
            Q = sample['Q'].unsqueeze(0).to(device)
            mask_p = sample['mask_p'].unsqueeze(0).to(device)
            mask_q = sample['mask_q'].unsqueeze(0).to(device)
            
            pred_p, pred_q, pred_global = model(P, Q, mask_p, mask_q)
            
            # Add predictions to sample for plotting
            sample_with_pred = sample.copy()
            sample_with_pred['pred_p'] = pred_p.squeeze().cpu()
            sample_with_pred['pred_q'] = pred_q.squeeze().cpu()
            sample_with_pred['pred_global'] = pred_global.squeeze().cpu()
            
            title = f"Sample {i+1}: Global Pred={pred_global.sigmoid().item():.3f}"
            plot_sample(sample_with_pred, 
                       save_path=os.path.join(save_dir, f'pred_{i+1:02d}.png'),
                       title=title)
    
    print(f"✅ Saved {test_samples} prediction plots to {save_dir}")


def quick_sample_preview(n_samples: int = 3, save_path: str = 'sample_preview.png'):
    """Quick preview of generated dataset."""
    dataset = ChangeDetectionDataset(size=n_samples, mode='generate')
    fig, axes = plt.subplots(1, n_samples, figsize=(5*n_samples, 5))
    if n_samples == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        sample = dataset[i]
        mask_p = sample['mask_p'].numpy().nonzero()[0]
        mask_q = sample['mask_q'].numpy().nonzero()[0]
        
        ax.scatter(sample['P'][mask_p, 0], sample['P'][mask_p, 1], 
                  c='blue', s=5, alpha=0.7, label='P')
        ax.scatter(sample['Q'][mask_q, 0], sample['Q'][mask_q, 1], 
                  c='orange', s=5, alpha=0.7, label='Q')
        ax.set_title(f'Sample {i+1}')
        ax.legend()
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Preview saved: {save_path}")
