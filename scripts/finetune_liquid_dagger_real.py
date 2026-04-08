#!/usr/bin/env python3
"""
Fine-tune the real liquid checkpoint on DAgger data (same architecture as eval script).

Key points:
- Loads the original liquid checkpoint state_dict
- Trains on merged DAgger data
- Upweights expert-intervention timesteps
- Saves state_dict-only checkpoint (compatible with existing eval tooling)
"""

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

STATE_DIM = 57
ACTION_DIM = 7
PRED_HORIZON = 16


class MockCLIPModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.image_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.image_proj = nn.Linear(8 * 8 * 3, embed_dim, bias=False)
        self.text_proj = nn.Linear(77, embed_dim, bias=False)
        for p in self.parameters():
            p.requires_grad = False

    def encode_image(self, image):
        pooled = self.image_pool(image.float()).flatten(1)
        return F.normalize(self.image_proj(pooled), dim=-1)


class SharedBackbone(nn.Module):
    def __init__(self, clip_model, state_dim=57, hidden_dim=256, num_layers=4):
        super().__init__()
        self.clip_model = clip_model
        self.hidden_dim = hidden_dim
        self.image_proj = nn.Linear(512, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.pos_emb = nn.Embedding(PRED_HORIZON, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)

    def encode_images(self, images):
        b, t, h, w, c = images.shape
        flat = images.reshape(b * t, h, w, c).permute(0, 3, 1, 2)
        with torch.no_grad():
            emb = self.clip_model.encode_image(flat)
        return emb.reshape(b, t, 512)

    def forward(self, images, state):
        t = images.shape[1]
        z = self.image_proj(self.encode_images(images)) + self.state_proj(state)
        z = z + self.pos_emb(torch.arange(t, device=images.device)).unsqueeze(0)
        latent = self.transformer(z)
        return self.context_proj(latent.mean(dim=1)), latent


class CfCCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.tau = nn.Parameter(torch.ones(hidden_size) * 0.1)
        self.W_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_cand = nn.Linear(input_size + hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h):
        combined = self.dropout(torch.cat([x, h], dim=-1))
        gate = torch.sigmoid(self.W_gate(combined))
        cand = torch.tanh(self.W_cand(combined))
        return gate * cand + (1.0 - gate) * h


class LiquidTrajectoryModel(nn.Module):
    def __init__(self, backbone, action_dim=ACTION_DIM, hidden_dim=291, seq_length=PRED_HORIZON, num_layers=2, num_mixtures=5, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_mixtures = num_mixtures
        self.context_proj = nn.Linear(backbone.hidden_dim, hidden_dim)
        self.latent_proj = nn.Linear(backbone.hidden_dim, hidden_dim)
        self.cfc_layers = nn.ModuleList([CfCCell(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)])
        self.mdn_logits = nn.Linear(hidden_dim, num_mixtures)
        self.mdn_mu = nn.Linear(hidden_dim, num_mixtures * action_dim)
        self.mdn_log_sigma = nn.Linear(hidden_dim, num_mixtures * action_dim)

    def forward(self, images, state, return_mdn=False):
        b, t = images.shape[:2]
        context, latent = self.backbone(images, state)
        context = self.context_proj(context)
        latent = self.latent_proj(latent)
        hs = [context.clone() for _ in self.cfc_layers]
        logits_all, mu_all = [], []
        for step in range(t):
            x = latent[:, step, :]
            for i, layer in enumerate(self.cfc_layers):
                hs[i] = layer(x, hs[i])
                x = hs[i]
            logits = self.mdn_logits(hs[-1])
            mu = self.mdn_mu(hs[-1]).reshape(b, self.num_mixtures, self.action_dim)
            logits_all.append(logits)
            mu_all.append(mu)
        result = {
            "logits": torch.stack(logits_all, dim=1),
            "mu": torch.stack(mu_all, dim=1),
        }
        return result if return_mdn else result["mu"]


class DAggerWindowDataset(Dataset):
    def __init__(self, hdf5_path, state_min=None, state_max=None, require_expert_actions=False, intervention_only=False):
        self.samples = []

        with h5py.File(hdf5_path, "r") as f:
            data_group = f["data"] if "data" in f else f
            for ep_key in sorted(data_group.keys()):
                ep = data_group[ep_key]
                obs = ep["observations"][()].astype(np.float32)  # (T+1, 57)
                act = ep["actions"][()].astype(np.float32)       # executed action
                expert_act = ep["expert_actions"][()].astype(np.float32) if "expert_actions" in ep else None
                inter = ep["interventions"][()].astype(np.float32) if "interventions" in ep else np.zeros((len(act),), dtype=np.float32)

                if require_expert_actions and expert_act is None:
                    continue

                T = act.shape[0]
                for t in range(T):
                    if intervention_only and inter[t] < 0.5:
                        continue
                    start = max(0, t - (PRED_HORIZON - 1))
                    hist = obs[start:t + 1]
                    if hist.shape[0] < PRED_HORIZON:
                        pad = np.repeat(hist[[0]], PRED_HORIZON - hist.shape[0], axis=0)
                        hist = np.concatenate([pad, hist], axis=0)
                    target = expert_act[t] if expert_act is not None else act[t]
                    self.samples.append((hist, target, inter[t]))

        if not self.samples:
            raise RuntimeError("No training samples found. Relax filters or collect labeled DAgger data first.")

        all_hist = np.concatenate([s[0] for s in self.samples], axis=0)
        if state_min is None or state_max is None:
            self.state_min = all_hist.min(axis=0)
            self.state_max = all_hist.max(axis=0)
        else:
            self.state_min = state_min.astype(np.float32)
            self.state_max = state_max.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist, action, intervention = self.samples[idx]
        hist = (hist - self.state_min) / (self.state_max - self.state_min + 1e-8) * 2.0 - 1.0
        return {
            "state_hist": torch.from_numpy(hist).float(),
            "action": torch.from_numpy(action).float(),
            "intervention": torch.tensor(intervention, dtype=torch.float32),
        }


def weighted_action_from_mdn(out):
    """Return expected action at last timestep from MDN outputs."""
    logits_last = out["logits"][:, -1, :]           # (B, K)
    mu_last = out["mu"][:, -1, :, :]                # (B, K, A)
    probs = torch.softmax(logits_last, dim=-1)       # (B, K)
    exp_action = (probs.unsqueeze(-1) * mu_last).sum(dim=1)  # (B, A)
    return exp_action


def run(args):
    device = torch.device(args.device)

    model = LiquidTrajectoryModel(SharedBackbone(MockCLIPModel()))
    ckpt = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt)
    model.to(device)

    dataset = DAggerWindowDataset(
        args.hdf5,
        require_expert_actions=args.require_expert_actions,
        intervention_only=args.intervention_only,
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "liquid_dagger_finetuned_best_state_dict.pt"
    stats_path = out_dir / "liquid_dagger_state_stats.json"

    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        tr_inter = 0.0

        for batch in train_loader:
            states = batch["state_hist"].to(device)
            target = batch["action"].to(device)
            inter = batch["intervention"].to(device)
            imgs = torch.zeros((states.shape[0], PRED_HORIZON, 96, 96, 3), device=device, dtype=torch.float32)

            out = model(imgs, states, return_mdn=True)
            pred = weighted_action_from_mdn(out)

            per_sample = ((pred - target) ** 2).mean(dim=-1)
            weights = 1.0 + args.intervention_weight * inter
            loss = (per_sample * weights).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += loss.item() * states.shape[0]
            tr_n += states.shape[0]
            tr_inter += inter.sum().item()

        tr_loss /= max(tr_n, 1)

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for batch in val_loader:
                states = batch["state_hist"].to(device)
                target = batch["action"].to(device)
                inter = batch["intervention"].to(device)
                imgs = torch.zeros((states.shape[0], PRED_HORIZON, 96, 96, 3), device=device, dtype=torch.float32)

                out = model(imgs, states, return_mdn=True)
                pred = weighted_action_from_mdn(out)
                per_sample = ((pred - target) ** 2).mean(dim=-1)
                weights = 1.0 + args.intervention_weight * inter
                loss = (per_sample * weights).mean()

                va_loss += loss.item() * states.shape[0]
                va_n += states.shape[0]

        va_loss /= max(va_n, 1)
        sched.step()

        print(f"epoch {epoch+1:03d}/{args.epochs} train={tr_loss:.6f} val={va_loss:.6f} inter(train)={tr_inter/tr_n:.3f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
            with open(stats_path, "w") as f:
                json.dump({
                    "state_min": dataset.state_min.tolist(),
                    "state_max": dataset.state_max.tolist(),
                }, f)

    print("=== done ===")
    print(f"best_val={best_val:.6f}")
    print(f"checkpoint={best_path}")
    print(f"state_stats={stats_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", default="artifacts/dagger/merged_training_data.hdf5")
    p.add_argument("--base-ckpt", default="checkpoints/liquid_jepa_robomimic_can_fair_halfparam_deterministic_clip_best.pt")
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--intervention-weight", type=float, default=4.0)
    p.add_argument("--require-expert-actions", action="store_true")
    p.add_argument("--intervention-only", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
