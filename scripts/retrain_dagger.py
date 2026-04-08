#!/usr/bin/env python3
"""
Retrain liquid/diffusion policies on aggregated DAgger data.
Uses the expanded training set that includes expert corrections at failure states.
"""

import os
import json
import argparse
import torch
from pathlib import Path


def retrain_policy_on_dagger_data(
    policy_type='liquid',
    aggregated_hdf5_path=None,
    split_names=None,
    epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    output_dir='checkpoints',
):
    """
    Retrain policy on aggregated DAgger data.
    
    policy_type: 'liquid' or 'diffusion'
    split_names: List of HDF5 split names to include (e.g., ['dagger_liquid_20260331_120000'])
    """
    print("=" * 80)
    print(f"Retraining {policy_type} on DAgger Data")
    print("=" * 80)
    print(f"Aggregated HDF5: {aggregated_hdf5_path}")
    print(f"Split names: {split_names}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()
    
    # Import after argparse to allow customization
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from datasets.robomimic_loading import build_robomimic_can_loaders
    
    print("[1] Loading data loaders...")
    
    # Create loaders with both original and DAgger data
    train_loader, val_loader, _, stats = build_robomimic_can_loaders(
        hdf5_path=aggregated_hdf5_path,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        # Can add split_names parameter if your loader supports it
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Action stats: mean={stats['action_mean']}, std={stats['action_std']}")
    
    print()
    print("[2] Creating model...")
    
    if policy_type == 'liquid':
        from models.liquid_policy import LiquidPolicy
        model = LiquidPolicy(
            input_dim=114,  # 57D * 2 for history
            hidden_dim=291,
            output_dim=7,
            action_range=(-1, 1),
        ).to(device)
    elif policy_type == 'diffusion':
        from models.diffusion_policy import DiffusionPolicy
        model = DiffusionPolicy(
            input_dim=114,
            output_dim=7,
            hidden_dim=608,
            num_steps=50,
        ).to(device)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
    
    print(f"  Model: {policy_type}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print()
    print("[3] Training...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_model_path = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            obs = batch['obs'].to(device)
            actions = batch['actions'].to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(obs, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % max(1, len(train_loader) // 5) == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}: "
                      f"loss={loss.item():.6f}")
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['obs'].to(device)
                actions = batch['actions'].to(device)
                loss = model.compute_loss(obs, actions)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(
                output_dir,
                f'dagger_{policy_type}_best_epoch{epoch+1:03d}.pt'
            )
            os.makedirs(output_dir, exist_ok=True)
            torch.save({'policy': model, 'epoch': epoch, 'val_loss': val_loss}, best_model_path)
            print(f"  ✓ Saved best model to {best_model_path}")
    
    print()
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best model: {best_model_path}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print()
    
    return best_model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy-type', default='liquid', choices=['liquid', 'diffusion'])
    parser.add_argument('--aggregated-hdf5', required=True, help='Path to aggregated DAgger HDF5')
    parser.add_argument('--split-names', nargs='+', help='HDF5 split names to include')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', default='checkpoints')
    args = parser.parse_args()
    
    retrain_policy_on_dagger_data(
        policy_type=args.policy_type,
        aggregated_hdf5_path=args.aggregated_hdf5,
        split_names=args.split_names,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
