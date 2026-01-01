# train.py

import os
import argparse
from typing import Dict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_generators.dataset import ChangeDetectionDataset
from models.pointnet_mlp import PointNetMLP


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: torch.device, epoch: int) -> Dict[str, float]:
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    total_acc_p = 0.0
    total_acc_q = 0.0
    n_samples = 0

    for batch_idx, batch in enumerate(loader):
        # Data to device
        P = batch['P'].to(device)
        Q = batch['Q'].to(device)
        mask_p = batch['mask_p'].to(device)
        mask_q = batch['mask_q'].to(device)
        change_p_gt = batch['change_p'].to(device)
        change_q_gt = batch['change_q'].to(device)
        y_global_gt = batch['y_global'].squeeze().to(device)

        optimizer.zero_grad()

        # Forward pass
        change_p_pred, change_q_pred, y_global_pred = model(P, Q, mask_p, mask_q)

        # Losses: per-point + global
        loss_p = criterion(change_p_pred, change_p_gt)
        loss_q = criterion(change_q_pred, change_q_gt)
        loss_global = nn.functional.binary_cross_entropy_with_logits(y_global_pred, y_global_gt)
        loss = loss_p + loss_q + 0.1 * loss_global  # weighted combination

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * P.size(0)
        total_acc_p += ((change_p_pred > 0) == (change_p_gt > 0.5)).float().mean().item() * P.size(0)
        total_acc_q += ((change_q_pred > 0) == (change_q_gt > 0.5)).float().mean().item() * P.size(0)
        n_samples += P.size(0)

        if batch_idx % 50 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(loader)}: '
                  f'Loss={loss.item():.4f}, AccP={total_acc_p/n_samples:.3f}')

    metrics = {
        'loss': total_loss / n_samples,
        'acc_p': total_acc_p / n_samples,
        'acc_q': total_acc_q / n_samples,
    }
    return metrics


@torch.no_grad()
def validate_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
                   device: torch.device) -> Dict[str, float]:
    """Validation epoch."""
    model.eval()
    total_loss = 0.0
    total_acc_p = 0.0
    total_acc_q = 0.0
    n_samples = 0

    for batch in loader:
        P = batch['P'].to(device)
        Q = batch['Q'].to(device)
        mask_p = batch['mask_p'].to(device)
        mask_q = batch['mask_q'].to(device)
        change_p_gt = batch['change_p'].to(device)
        change_q_gt = batch['change_q'].to(device)

        change_p_pred, change_q_pred, _ = model(P, Q, mask_p, mask_q)

        loss_p = criterion(change_p_pred, change_p_gt)
        loss_q = criterion(change_q_pred, change_q_gt)
        loss = (loss_p + loss_q) / 2

        total_loss += loss.item() * P.size(0)
        total_acc_p += ((change_p_pred > 0) == (change_p_gt > 0.5)).float().mean().item() * P.size(0)
        total_acc_q += ((change_q_pred > 0) == (change_q_gt > 0.5)).float().mean().item() * P.size(0)
        n_samples += P.size(0)

    metrics = {
        'loss': total_loss / n_samples,
        'acc_p': total_acc_p / n_samples,
        'acc_q': total_acc_q / n_samples,
    }
    return metrics


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data
    train_dataset = ChangeDetectionDataset(
        size=args.n_train,
        mode=args.data_mode,
        data_dir=args.data_dir,
        n_points_per_cloud=args.n_points,
    )
    val_dataset = ChangeDetectionDataset(
        size=args.n_val,
        mode='generate',  # always fresh validation
        n_points_per_cloud=args.n_points,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f'Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples')

    # Model
    model = PointNetMLP(in_dim=4, hidden_dim=128, out_dim=1).to(device)  # xyz + cloud_id

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # Logging
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tb_logs'))
    
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f'\n=== Epoch {epoch+1}/{args.epochs} ===')
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        scheduler.step()
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Log
        for k, v in train_metrics.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Train: Loss={train_metrics["loss"]:.4f}, AccP={train_metrics["acc_p"]:.3f}, '
              f'AccQ={train_metrics["acc_q"]:.3f}')
        print(f'Val:   Loss={val_metrics["loss"]:.4f}, AccP={val_metrics["acc_p"]:.3f}, '
              f'AccQ={val_metrics["acc_q"]:.3f}')
        
        # Save best model
        mean_val_acc = (val_metrics['acc_p'] + val_metrics['acc_q']) / 2
        if mean_val_acc > best_val_acc:
            best_val_acc = mean_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f'New best model saved! Val Acc: {mean_val_acc:.3f}')
            # 🔥 ADD VISUALIZATION HERE 🔥
            try:
                from utils.viz import plot_predictions
                viz_dir = os.path.join(args.output_dir, 'viz_best')
                plot_predictions(os.path.join(args.output_dir, 'best_model.pth'), 
                            test_samples=3, save_dir=viz_dir)
                print(f'  → Generated prediction visualizations in {viz_dir}')
            except ImportError:
                print('  → Visualization skipped (utils/viz.py not found)')
            except Exception as e:
                print(f'  → Visualization failed: {e}')
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    writer.close()
    print(f'\nTraining complete! Best val acc: {best_val_acc:.3f}')
    print(f'Checkpoints saved to: {args.output_dir}')
    print(f'TensorBoard: tensorboard --logdir={args.output_dir}/tb_logs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 2D Point Cloud Change Detection')
    parser.add_argument('--n_train', type=int, default=10000, help='train samples')
    parser.add_argument('--n_val', type=int, default=2000, help='val samples')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--n_points', type=int, default=512, help='points per cloud')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--data_mode', choices=['generate', 'load', 'mixed'], default='mixed',
                        help='data loading mode')
    parser.add_argument('--data_dir', type=str, default='data/generated', help='HDF5 data directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='output directory')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    args = parser.parse_args()

    main(args)
