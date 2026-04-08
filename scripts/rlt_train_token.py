import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from eval_robomimic_can_closedloop_mujoco import (
    OBS_KEYS,
    PRED_HORIZON,
    build_liquid_model,
    decode_keys,
    normalize_data,
)
from rlt_models import RLTokenBottleneck


def concat_obs(obs_group):
    return np.concatenate([obs_group[k][:].astype(np.float32) for k in OBS_KEYS], axis=-1)


def get_data_stats(data):
    flat = data.reshape(-1, data.shape[-1])
    return {"min": np.min(flat, axis=0), "max": np.max(flat, axis=0)}


class StateWindowDataset(Dataset):
    def __init__(self, windows: np.ndarray):
        self.windows = windows.astype(np.float32)

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        return self.windows[idx]


def build_windows(dataset_path: str, stride: int = 4):
    with h5py.File(dataset_path, "r") as f:
        demo_keys = sorted(list(f["data"].keys()))
        states_all = [concat_obs(f["data"][k]["obs"]) for k in demo_keys]

    all_states = np.concatenate(states_all, axis=0)
    state_stats = get_data_stats(all_states)

    windows = []
    for seq in states_all:
        seq_n = normalize_data(seq, state_stats)
        if len(seq_n) < PRED_HORIZON:
            continue
        for i in range(0, len(seq_n) - PRED_HORIZON + 1, stride):
            windows.append(seq_n[i : i + PRED_HORIZON])

    return np.asarray(windows, dtype=np.float32), state_stats


def main():
    parser = argparse.ArgumentParser(description="Train RL-token bottleneck from frozen liquid backbone latents")
    parser.add_argument("--dataset", default="datasets/robomimic/v1.5/can/ph/low_dim_v15.hdf5")
    parser.add_argument("--liquid-ckpt", default="checkpoints/liquid_jepa_robomimic_can_fair_halfparam_deterministic_clip_best.pt")
    parser.add_argument("--output", default="checkpoints/rlt_token_bottleneck.pt")
    parser.add_argument("--state-stats-out", default="checkpoints/rlt_state_stats.json")
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    np.random.seed(7)
    torch.manual_seed(7)

    windows, state_stats = build_windows(args.dataset, stride=args.stride)
    ds = StateWindowDataset(windows)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    liquid = build_liquid_model(device)
    liquid.eval()
    for p in liquid.parameters():
        p.requires_grad = False

    bottleneck = RLTokenBottleneck(backbone_dim=liquid.backbone.hidden_dim, token_dim=args.token_dim, seq_length=PRED_HORIZON)
    bottleneck.to(device)

    opt = torch.optim.AdamW(bottleneck.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        losses = []
        for state_win in dl:
            state_win = state_win.to(device)
            images = torch.zeros((state_win.shape[0], PRED_HORIZON, 96, 96, 3), device=device)

            with torch.no_grad():
                _, latent = liquid.backbone(images, state_win)

            rl_token, recon = bottleneck(latent)
            loss_recon = F.mse_loss(recon, latent)
            loss_token = 1e-4 * (rl_token.pow(2).mean())
            loss = loss_recon + loss_token

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bottleneck.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        print(f"epoch {epoch:03d}/{args.epochs} recon_loss={np.mean(losses):.6f}")

    out = {
        "token_dim": args.token_dim,
        "seq_length": PRED_HORIZON,
        "state_dict": bottleneck.state_dict(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    Path(args.state_stats_out).write_text(json.dumps({"state_min": state_stats["min"].tolist(), "state_max": state_stats["max"].tolist()}, indent=2))
    print(f"saved bottleneck -> {args.output}")
    print(f"saved state stats -> {args.state_stats_out}")


if __name__ == "__main__":
    main()
