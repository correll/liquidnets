import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from eval_robomimic_can_closedloop_mujoco import (
    OBS_KEYS,
    ACTION_DIM,
    PRED_HORIZON,
    build_liquid_model,
    normalize_data,
)
from rlt_models import RLTokenBottleneck, ResidualActor


def concat_obs(obs_group):
    return np.concatenate([obs_group[k][:].astype(np.float32) for k in OBS_KEYS], axis=-1)


def load_state_stats(path):
    stats = json.loads(Path(path).read_text())
    return {
        "min": np.asarray(stats["state_min"], dtype=np.float32),
        "max": np.asarray(stats["state_max"], dtype=np.float32),
    }


def load_action_stats(dataset_path):
    with h5py.File(dataset_path, "r") as f:
        if "stats" in f and "action_min" in f["stats"]:
            action_min = f["stats"]["action_min"][()].astype(np.float32)
            action_max = f["stats"]["action_max"][()].astype(np.float32)
        else:
            action_min = np.array([-1.0] * ACTION_DIM, dtype=np.float32)
            action_max = np.array([1.0] * ACTION_DIM, dtype=np.float32)
    return {"min": action_min, "max": action_max}


@torch.no_grad()
def base_action_seq_norm(liquid_model, state_hist_norm, device):
    states_t = torch.from_numpy(state_hist_norm).unsqueeze(0).to(device=device, dtype=torch.float32)
    images_t = torch.zeros((1, PRED_HORIZON, 96, 96, 3), device=device, dtype=torch.float32)
    out = liquid_model(images_t, states_t, return_mdn=True)
    logits = out["logits"][0]
    mu = out["mu"][0]
    w = torch.softmax(logits, dim=-1).unsqueeze(-1)
    seq = (w * mu).sum(dim=1)
    return seq.reshape(-1)


@torch.no_grad()
def compute_token(liquid_model, bottleneck, state_hist_norm, device):
    states_t = torch.from_numpy(state_hist_norm).unsqueeze(0).to(device=device, dtype=torch.float32)
    images_t = torch.zeros((1, PRED_HORIZON, 96, 96, 3), device=device, dtype=torch.float32)
    _, latent = liquid_model.backbone(images_t, states_t)
    return bottleneck.encoder(latent).squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="Offline actor warm-start on all demonstrations")
    parser.add_argument("--dataset", default="datasets/robomimic/v1.5/can/ph/low_dim_v15.hdf5")
    parser.add_argument("--token-ckpt", default="checkpoints/rlt_token_bottleneck.pt")
    parser.add_argument("--state-stats", default="checkpoints/rlt_state_stats.json")
    parser.add_argument("--output", default="checkpoints/rlt_actor_warmstart.pt")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    np.random.seed(7)
    torch.manual_seed(7)
    device = torch.device(args.device)

    state_stats = load_state_stats(args.state_stats)
    action_stats = load_action_stats(args.dataset)

    liquid = build_liquid_model(device)
    liquid.eval()
    for p in liquid.parameters():
        p.requires_grad = False

    token_ckpt = torch.load(args.token_ckpt, map_location=device, weights_only=False)
    token_dim = int(token_ckpt["token_dim"])
    bottleneck = RLTokenBottleneck(backbone_dim=liquid.backbone.hidden_dim, token_dim=token_dim, seq_length=PRED_HORIZON).to(device)
    bottleneck.load_state_dict(token_ckpt["state_dict"])
    bottleneck.eval()
    for p in bottleneck.parameters():
        p.requires_grad = False

    all_tokens = []
    all_base = []
    all_target = []

    with h5py.File(args.dataset, "r") as f:
        demo_keys = sorted(list(f["data"].keys()))
        for k in demo_keys:
            grp = f["data"][k]
            states = concat_obs(grp["obs"])  # (T,57)
            actions = grp["actions"][:].astype(np.float32)  # (T,7)
            if len(states) < PRED_HORIZON or len(actions) < PRED_HORIZON:
                continue

            for i in range(0, len(actions) - PRED_HORIZON + 1, args.stride):
                state_win = states[i : i + PRED_HORIZON]
                state_win_n = normalize_data(state_win, state_stats)

                act_win = actions[i : i + PRED_HORIZON]
                act_win_n = normalize_data(act_win, action_stats).reshape(-1)

                token = compute_token(liquid, bottleneck, state_win_n, device)
                base = base_action_seq_norm(liquid, state_win_n, device)

                all_tokens.append(token.cpu())
                all_base.append(base.cpu())
                all_target.append(torch.from_numpy(act_win_n).float())

    tokens_t = torch.stack(all_tokens, dim=0)
    base_t = torch.stack(all_base, dim=0)
    target_t = torch.stack(all_target, dim=0)

    ds = TensorDataset(tokens_t, base_t, target_t)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    actor = ResidualActor(token_dim=token_dim, action_dim=ACTION_DIM, seq_length=PRED_HORIZON).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)

    print(f"warmstart pairs: {len(ds)} from all demos")
    for epoch in range(1, args.epochs + 1):
        losses = []
        actor.train()
        for tok_b, base_b, tgt_b in dl:
            tok_b = tok_b.to(device)
            base_b = base_b.to(device)
            tgt_b = tgt_b.to(device)

            pred_action, _, _ = actor(tok_b, base_b, deterministic=True)
            loss_main = F.mse_loss(pred_action, tgt_b)
            loss_anchor = 0.01 * F.mse_loss(pred_action, base_b)
            loss = loss_main + loss_anchor

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        print(f"epoch {epoch:03d}/{args.epochs} warmstart_loss={np.mean(losses):.6f}")

    out = {
        "actor": actor.state_dict(),
        "token_dim": token_dim,
        "num_pairs": len(ds),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"saved warmstart actor -> {args.output}")


if __name__ == "__main__":
    main()
