#!/usr/bin/env python3
"""
Retrain liquid policy on merged DAgger data - simplified version.
"""
import argparse
import h5py
import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Constants
STATE_DIM = 57
ACTION_DIM = 7
PRED_HORIZON = 16


class SimpleLiquidPolicy(nn.Module):
    """Simplified liquid policy for retraining."""
    def __init__(self, state_dim=57, hidden_dim=291, action_dim=7, pred_horizon=16):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        
        # Simple MLP that processes flattened state history
        self.encoder = nn.Sequential(
            nn.Linear(state_dim * pred_horizon, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output head
        self.head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state_history):
        """
        state_history: (B, pred_horizon, state_dim)
        returns: (B, action_dim)
        """
        B = state_history.shape[0]
        # Flatten history
        x = state_history.reshape(B, -1)  # (B, pred_horizon * state_dim)
        x = self.encoder(x)
        x = self.head(x)
        return torch.tanh(x)  # Action in [-1, 1]
    
    def compute_loss(self, state_history, actions):
        """Compute MSE loss."""
        pred = self.forward(state_history)
        loss = F.mse_loss(pred, actions)
        return loss


class DAggerDataset(torch.utils.data.Dataset):
    """Load DAgger data from merged HDF5."""
    
    def __init__(self, hdf5_path, pred_horizon=16, normalize=True, stats_dict=None):
        self.hdf5_path = hdf5_path
        self.pred_horizon = pred_horizon
        self.normalize = normalize
        self.stats_dict = stats_dict
        
        # Load all episodes
        self.episodes = []
        self.total_transitions = 0
        
        with h5py.File(hdf5_path, 'r') as f:
            data_group = f.get('data', f)
            for ep_key in sorted(data_group.keys()):
                ep = data_group[ep_key]
                obs = ep['observations'][()]  # (T+1, 57)
                actions = ep['actions'][()]    # (T, 7)
                
                self.episodes.append({
                    'obs': obs.astype(np.float32),
                    'actions': actions.astype(np.float32),
                    'start_idx': self.total_transitions
                })
                self.total_transitions += len(actions)
        
        print(f"Loaded {len(self.episodes)} episodes with {self.total_transitions} transitions")
    
    def __len__(self):
        return self.total_transitions
    
    def __getitem__(self, idx):
        # Find episode
        ep_idx = 0
        for ep in self.episodes:
            if idx < ep['start_idx'] + len(ep['actions']):
                t = idx - ep['start_idx']
                break
            ep_idx += 1
        
        obs = self.episodes[ep_idx]['obs']
        actions = self.episodes[ep_idx]['actions']
        
        # Get state history and action
        if t >= self.pred_horizon - 1:
            state_hist = obs[t - self.pred_horizon + 1:t + 1]  # (pred_horizon, 57)
            action = actions[t]  # (7,)
        else:
            # Pad with first observation
            pad = np.repeat(obs[[0]], self.pred_horizon - t - 1, axis=0)
            state_hist = np.vstack([pad, obs[:t + 1]])
            action = actions[t]
        
        # Normalize
        if self.normalize and self.stats_dict:
            state_hist = (state_hist - self.stats_dict['obs_min']) / (self.stats_dict['obs_max'] - self.stats_dict['obs_min'] + 1e-8) * 2 - 1
            action = (action - self.stats_dict['act_min']) / (self.stats_dict['act_max'] - self.stats_dict['act_min'] + 1e-8) * 2 - 1
        
        return {
            'state_hist': torch.from_numpy(state_hist).float(),
            'action': torch.from_numpy(action).float()
        }


def compute_stats(hdf5_path):
    """Compute normalization statistics."""
    obs_all = []
    act_all = []
    
    with h5py.File(hdf5_path, 'r') as f:
        data_group = f.get('data', f)
        for ep_key in sorted(data_group.keys()):
            ep = data_group[ep_key]
            obs_all.append(ep['observations'][()])
            act_all.append(ep['actions'][()])
    
    obs_all = np.vstack(obs_all)
    act_all = np.vstack(act_all)
    
    stats = {
        'obs_min': obs_all.min(axis=0, keepdims=False).astype(np.float32),
        'obs_max': obs_all.max(axis=0, keepdims=False).astype(np.float32),
        'act_min': act_all.min(axis=0, keepdims=False).astype(np.float32),
        'act_max': act_all.max(axis=0, keepdims=False).astype(np.float32),
    }
    
    return stats


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5', default='artifacts/dagger/merged_training_data.hdf5')
    parser.add_argument('--hidden-dim', type=int, default=291)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', default='checkpoints')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Retraining Liquid Policy on DAgger Data")
    print("=" * 80)
    print(f"HDF5: {args.hdf5}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {device}")
    print()
    
    # Compute stats
    print("[1] Computing normalization statistics...")
    stats = compute_stats(args.hdf5)
    print(f"  Obs: min={stats['obs_min'].mean():.4f}, max={stats['obs_max'].mean():.4f}")
    print(f"  Act: min={stats['act_min'].mean():.4f}, max={stats['act_max'].mean():.4f}")
    print()
    
    # Create datasets
    print("[2] Loading datasets...")
    full_dataset = DAggerDataset(args.hdf5, normalize=True, stats_dict=stats)
    
    # 80/20 split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    print(f"  Train: {train_size} transitions ({len(train_loader)} batches)")
    print(f"  Val: {val_size} transitions ({len(val_loader)} batches)")
    print()
    
    # Create model
    print("[3] Creating model...")
    model = SimpleLiquidPolicy(
        state_dim=STATE_DIM,
        hidden_dim=args.hidden_dim,
        action_dim=ACTION_DIM,
        pred_horizon=PRED_HORIZON,
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print()
    
    # Train
    print("[4] Training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float('inf')
    best_model_path = None
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            state_hist = batch['state_hist'].to(device)
            action = batch['action'].to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(state_hist, action)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(action)
            train_count += len(action)
        
        train_loss /= train_count
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                state_hist = batch['state_hist'].to(device)
                action = batch['action'].to(device)
                loss = model.compute_loss(state_hist, action)
                val_loss += loss.item() * len(action)
                val_count += len(action)
        
        val_loss /= val_count
        
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(
                args.output_dir,
                f'dagger_liquid_retrained_best.pt'
            )
            torch.save({
                'policy': model,
                'epoch': epoch,
                'val_loss': val_loss,
                'stats': stats,
            }, best_model_path)
    
    print()
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best model: {best_model_path}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print()


if __name__ == '__main__':
    train()
