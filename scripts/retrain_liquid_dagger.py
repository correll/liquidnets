#!/usr/bin/env python3
"""
Retrain liquid policy on merged DAgger data.
"""
import argparse
import h5py
import json
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import deque, defaultdict

# Constants
OBS_KEYS = [
    "object", "robot0_eef_pos", "robot0_eef_quat", "robot0_eef_quat_site",
    "robot0_gripper_qpos", "robot0_gripper_qvel", "robot0_joint_pos",
    "robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel",
]
PRED_HORIZON = 16
ACTION_DIM = 7
STATE_DIM = 57


class CfCCell(nn.Module):
    """Liquid Time Constant (LTC) cell."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.W_i = nn.Linear(input_dim, hidden_dim, bias=False)
        # Recurrent projection
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Time constant projection
        self.W_t = nn.Linear(input_dim, hidden_dim, bias=False)
        # Bias
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, x, h):
        """x: (B, input_dim), h: (B, hidden_dim)"""
        dh = torch.tanh(self.W_i(x) + self.W_h(h) + self.bias)
        tau = torch.nn.functional.softplus(self.W_t(x))  # time constant
        h_new = (1.0 - 1.0 / (tau + 1.0)) * h + (1.0 / (tau + 1.0)) * dh
        return h_new


class MDNHead(nn.Module):
    """Mixture Density Network output head."""
    def __init__(self, input_dim, output_dim, n_gaussians=5):
        super().__init__()
        self.output_dim = output_dim
        self.n_gaussians = n_gaussians
        
        # Output projections for means, stds, and mixing coefficients
        self.fc_means = nn.Linear(input_dim, output_dim * n_gaussians)
        self.fc_stds = nn.Linear(input_dim, output_dim * n_gaussians)
        self.fc_mix = nn.Linear(input_dim, n_gaussians)
    
    def forward(self, x):
        """x: (B, input_dim) -> means, stds, mix: (B, output_dim, n_gaussians)"""
        means = self.fc_means(x).reshape(-1, self.output_dim, self.n_gaussians)
        stds = F.softplus(self.fc_stds(x)).reshape(-1, self.output_dim, self.n_gaussians)
        mix = F.softmax(self.fc_mix(x), dim=-1)  # (B, n_gaussians)
        return means, stds, mix
    
    def sample_action(self, x):
        """Sample action from mixture."""
        means, stds, mix = self.forward(x)
        # Sample component for each element in batch
        mix_idx = torch.multinomial(mix, 1).squeeze(-1)  # (B,)
        
        # Select mean and std for sampled component
        B = means.shape[0]
        selected_means = means[torch.arange(B), :, mix_idx]  # (B, output_dim)
        selected_stds = stds[torch.arange(B), :, mix_idx]    # (B, output_dim)
        
        # Sample action
        eps = torch.randn_like(selected_means)
        action = selected_means + selected_stds * eps
        return torch.clamp(action, -1, 1)


class SharedBackbone(nn.Module):
    """Transformer-based encoder for state history."""
    def __init__(self, input_dim, output_dim, n_heads=None, n_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Project each timestep to output_dim
        self.state_proj = nn.Linear(input_dim, output_dim)
        
        # Use number of heads that divides output_dim
        if n_heads is None:
            n_heads = max(1, output_dim // 64)  # Aim for 64D per head
            while output_dim % n_heads != 0 and n_heads > 1:
                n_heads -= 1
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim, nhead=n_heads, dim_feedforward=256,
            batch_first=True, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.out_proj = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        """x: (B, T, input_dim) -> (B, output_dim)"""
        # Project each timestep
        x = self.state_proj(x)  # (B, T, output_dim)
        
        # Transformer encoding
        x = self.transformer(x)  # (B, T, output_dim)
        
        # Use last timestep or mean pooling
        x = x.mean(dim=1)  # (B, output_dim)
        
        # Final projection
        x = self.out_proj(x)
        return x


class LiquidTrajectoryModel(nn.Module):
    """Liquid neural network with MDN head for action prediction."""
    def __init__(self, state_dim=57, hidden_dim=291, action_dim=7, pred_horizon=16, n_gaussians=5):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        
        # Backbone
        self.backbone = SharedBackbone(state_dim, hidden_dim)
        
        # LTC cell
        self.lstm = CfCCell(hidden_dim, hidden_dim)
        
        # MDN output head
        self.mdn_head = MDNHead(hidden_dim, action_dim, n_gaussians)
    
    def forward(self, state_history):
        """
        state_history: (B, pred_horizon, state_dim)
        returns: (B, action_dim)
        """
        # Encode state history
        x = self.backbone(state_history)  # (B, hidden_dim)
        
        # Process through LTC
        h = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        for t in range(self.pred_horizon):
            h = self.lstm(state_history[:, t], h)
        
        # Predict action
        means, stds, mix = self.mdn_head(h)
        
        # For loss computation, return means
        return means.mean(dim=-1)  # Average over components: (B, action_dim)
    
    def get_action(self, state_history):
        """Get action for rollout."""
        with torch.no_grad():
            x = self.backbone(state_history)
            h = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
            for t in range(self.pred_horizon):
                h = self.lstm(state_history[:, t], h)
            action = self.mdn_head.sample_action(h)
        return action
    
    def compute_loss(self, state_history, actions):
        """Compute MSE loss."""
        pred_action = self.forward(state_history)
        loss = F.mse_loss(pred_action, actions)
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
    
    print(f"  Train: {train_size} transitions")
    print(f"  Val: {val_size} transitions")
    print()
    
    # Create model
    print("[3] Creating model...")
    model = LiquidTrajectoryModel(
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
        
        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
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
