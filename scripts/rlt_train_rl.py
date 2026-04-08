import argparse
import collections
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import robosuite as suite
from robomimic.utils.file_utils import get_env_metadata_from_dataset

from eval_robomimic_can_closedloop_mujoco import (
    OBS_KEYS,
    ACTION_DIM,
    PRED_HORIZON,
    build_liquid_model,
    normalize_data,
    unnormalize_data,
)
from rlt_models import RLTokenBottleneck, ResidualActor, TwinQCritic, ReplayBuffer


def env_obs_to_state(obs):
    pieces = []
    for key in OBS_KEYS:
        env_key = "object-state" if key == "object" else key
        pieces.append(np.asarray(obs[env_key], dtype=np.float32).reshape(-1))
    return np.concatenate(pieces, axis=-1)


def concat_obs(obs_group):
    return np.concatenate([obs_group[k][:].astype(np.float32) for k in OBS_KEYS], axis=-1)


def build_env(dataset_path):
    meta = get_env_metadata_from_dataset(dataset_path)
    kwargs = dict(meta["env_kwargs"])
    kwargs["has_renderer"] = False
    kwargs["has_offscreen_renderer"] = False
    kwargs["use_camera_obs"] = False
    kwargs["reward_shaping"] = True
    kwargs["ignore_done"] = True
    return suite.make(meta["env_name"], **kwargs)


def _obs_metrics(obs):
    obj_state = np.asarray(obs.get("object-state", np.zeros(14, dtype=np.float32)), dtype=np.float32).reshape(-1)
    can_pos = np.asarray(obs.get("Can_pos", obj_state[:3] if obj_state.shape[0] >= 3 else np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(-1)
    target_pos = np.asarray(obs.get("target-object", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(-1)
    eef_pos = np.asarray(obs.get("robot0_eef_pos", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(-1)
    gripper_qpos = np.asarray(obs.get("robot0_gripper_qpos", np.zeros(2, dtype=np.float32)), dtype=np.float32).reshape(-1)
    return {
        "goal_dist": float(np.linalg.norm(can_pos[:3] - target_pos[:3])),
        "eef_obj_dist": float(np.linalg.norm(eef_pos[:3] - can_pos[:3])),
        "obj_height": float(can_pos[2]) if can_pos.shape[0] >= 3 else 0.0,
        "gripper_open": float(np.mean(gripper_qpos)) if gripper_qpos.size > 0 else 0.0,
    }


def _shaped_reward(raw_reward, prev_m, curr_m, action, success, w_goal=6.0, w_lift=2.0, w_eef=1.0, w_ctrl=0.05, success_bonus=10.0):
    progress_goal = prev_m["goal_dist"] - curr_m["goal_dist"]
    progress_lift = curr_m["obj_height"] - prev_m["obj_height"]
    progress_eef = prev_m["eef_obj_dist"] - curr_m["eef_obj_dist"]
    ctrl_pen = float(np.square(action).mean())
    shaped = float(raw_reward)
    shaped += w_goal * progress_goal
    shaped += w_lift * max(progress_lift, 0.0)
    shaped += w_eef * progress_eef
    shaped -= w_ctrl * ctrl_pen
    if success:
        shaped += success_bonus
    return shaped


def _phase_reward(
    raw_reward,
    prev_m,
    curr_m,
    action,
    success,
    contact_thresh=0.035,
    close_thresh=0.012,
    lift_thresh=0.03,
    w_approach=10.0,
    w_open=1.5,
    w_close=3.0,
    w_lift=12.0,
    w_place=15.0,
    w_ctrl=0.05,
    success_bonus=15.0,
):
    # Phase 1: move end-effector to object while keeping gripper open
    is_contact = curr_m["eef_obj_dist"] < contact_thresh
    is_closed = curr_m["gripper_open"] < close_thresh
    is_lifted = curr_m["obj_height"] > lift_thresh

    approach_progress = prev_m["eef_obj_dist"] - curr_m["eef_obj_dist"]
    close_progress = prev_m["gripper_open"] - curr_m["gripper_open"]
    lift_progress = curr_m["obj_height"] - prev_m["obj_height"]
    place_progress = prev_m["goal_dist"] - curr_m["goal_dist"]
    ctrl_pen = float(np.square(action).mean())

    r = float(raw_reward)
    if not is_contact:
        # approach with open gripper
        r += w_approach * approach_progress
        r += w_open * max(curr_m["gripper_open"], 0.0)
    elif not (is_closed and is_lifted):
        # close gripper and lift once in contact
        r += w_close * close_progress
        r += w_lift * max(lift_progress, 0.0)
    else:
        # move object into basket/target
        r += w_place * place_progress

    r -= w_ctrl * ctrl_pen
    if success:
        r += success_bonus
    return r


@torch.no_grad()
def base_action_seq_norm(liquid_model, state_hist_norm, device):
    states_t = torch.from_numpy(state_hist_norm).unsqueeze(0).to(device=device, dtype=torch.float32)
    images_t = torch.zeros((1, PRED_HORIZON, 96, 96, 3), device=device, dtype=torch.float32)
    out = liquid_model(images_t, states_t, return_mdn=True)
    logits = out["logits"][0]  # (T, K)
    mu = out["mu"][0]          # (T, K, A)
    w = torch.softmax(logits, dim=-1).unsqueeze(-1)
    seq = (w * mu).sum(dim=1)   # (T, A)
    return seq.reshape(-1)      # (T*A,)


@torch.no_grad()
def compute_token(liquid_model, bottleneck, state_hist_norm, device):
    states_t = torch.from_numpy(state_hist_norm).unsqueeze(0).to(device=device, dtype=torch.float32)
    images_t = torch.zeros((1, PRED_HORIZON, 96, 96, 3), device=device, dtype=torch.float32)
    _, latent = liquid_model.backbone(images_t, states_t)
    token = bottleneck.encoder(latent).squeeze(0)
    return token


def sac_update(actor, critic, critic_targ, actor_opt, critic_opt, replay, device, gamma=0.99, alpha=0.05, polyak=0.995, batch_size=256, bc_coef=0.02):
    if replay.size < batch_size:
        return None

    b = replay.sample(batch_size, device)
    token = b["token"]
    base_action = b["base_action"]
    action = b["action"]
    reward = b["reward"]
    next_token = b["next_token"]
    next_base_action = b["next_base_action"]
    done = b["done"]

    with torch.no_grad():
        next_action, next_logp, _ = actor(next_token, next_base_action, deterministic=False)
        q1_t, q2_t = critic_targ(next_token, next_action)
        q_t = torch.min(q1_t, q2_t) - alpha * next_logp
        y = reward + gamma * (1.0 - done) * q_t

    q1, q2 = critic(token, action)
    critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_opt.step()

    new_action, logp, _ = actor(token, base_action, deterministic=False)
    q1_pi, q2_pi = critic(token, new_action)
    actor_loss = (alpha * logp - torch.min(q1_pi, q2_pi)).mean()

    # keep actor near base policy early
    bc_reg = bc_coef * F.mse_loss(new_action, base_action)
    actor_loss = actor_loss + bc_reg

    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_opt.step()

    with torch.no_grad():
        for p, p_t in zip(critic.parameters(), critic_targ.parameters()):
            p_t.data.mul_(polyak).add_((1 - polyak) * p.data)

    return {
        "critic_loss": float(critic_loss.item()),
        "actor_loss": float(actor_loss.item()),
    }


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
def build_demo_transitions(dataset_path, state_stats, action_stats, liquid, bottleneck, device, stride=4):
    tokens = []
    base_actions = []
    demo_actions = []
    rewards = []
    next_tokens = []
    next_base_actions = []
    dones = []
    repeats = []

    with h5py.File(dataset_path, "r") as f:
        demo_keys = sorted(list(f["data"].keys()))
        for k in demo_keys:
            grp = f["data"][k]
            states = concat_obs(grp["obs"])
            actions = grp["actions"][:].astype(np.float32)
            tmax = min(len(states), len(actions))
            if tmax <= PRED_HORIZON + 1:
                continue

            for i in range(0, tmax - PRED_HORIZON - 1, stride):
                s0 = states[i : i + PRED_HORIZON]
                s1 = states[i + 1 : i + 1 + PRED_HORIZON]
                s0_n = normalize_data(s0, state_stats)
                s1_n = normalize_data(s1, state_stats)

                tok = compute_token(liquid, bottleneck, s0_n, device)
                base = base_action_seq_norm(liquid, s0_n, device)
                ntok = compute_token(liquid, bottleneck, s1_n, device)
                nbase = base_action_seq_norm(liquid, s1_n, device)

                a_demo = normalize_data(actions[i : i + PRED_HORIZON], action_stats).reshape(-1)
                a_demo_t = torch.from_numpy(a_demo).to(device=device, dtype=torch.float32)

                # Success-proxy weighting: later timesteps in a demo are more task-relevant
                frac = i / max(1, (tmax - PRED_HORIZON - 1))
                demo_bonus = 0.2 + 0.8 * frac
                base_gap = float(torch.mean((a_demo_t - base) ** 2).item())
                r = demo_bonus - 0.05 * base_gap
                d = 1.0 if (i + 1) >= (tmax - PRED_HORIZON - 1) else 0.0

                tokens.append(tok.cpu())
                base_actions.append(base.cpu())
                demo_actions.append(a_demo_t.cpu())
                rewards.append(float(r))
                next_tokens.append(ntok.cpu())
                next_base_actions.append(nbase.cpu())
                dones.append(float(d))
                repeats.append(1 + int(4 * frac))

    return {
        "token": torch.stack(tokens, dim=0),
        "base_action": torch.stack(base_actions, dim=0),
        "action": torch.stack(demo_actions, dim=0),
        "reward": torch.tensor(rewards, dtype=torch.float32),
        "next_token": torch.stack(next_tokens, dim=0),
        "next_base_action": torch.stack(next_base_actions, dim=0),
        "done": torch.tensor(dones, dtype=torch.float32),
        "repeat": torch.tensor(repeats, dtype=torch.long),
    }


def critic_pretrain_from_demos(critic, critic_targ, critic_opt, demo, device, steps=2000, batch_size=512, gamma=0.99):
    n = demo["token"].shape[0]
    if n == 0:
        return
    critic.train()
    for s in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        token = demo["token"][idx].to(device)
        action = demo["action"][idx].to(device)
        reward = demo["reward"][idx].to(device)
        next_token = demo["next_token"][idx].to(device)
        next_action = demo["action"][idx].to(device)
        done = demo["done"][idx].to(device)

        with torch.no_grad():
            q1n, q2n = critic_targ(next_token, next_action)
            y = reward + gamma * (1.0 - done) * torch.min(q1n, q2n)

        q1, q2 = critic(token, action)
        loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_opt.step()

        with torch.no_grad():
            for p, p_t in zip(critic.parameters(), critic_targ.parameters()):
                p_t.data.mul_(0.995).add_(0.005 * p.data)

        if (s + 1) % max(1, (steps // 5)) == 0:
            print(f"demo critic pretrain {s + 1}/{steps} loss={loss.item():.6f}")


def main():
    parser = argparse.ArgumentParser(description="RLT online RL (SAC residual actor) for robomimic can")
    parser.add_argument("--dataset", default="datasets/robomimic/v1.5/can/ph/low_dim_v15.hdf5")
    parser.add_argument("--liquid-ckpt", default="checkpoints/liquid_jepa_robomimic_can_fair_halfparam_deterministic_clip_best.pt")
    parser.add_argument("--token-ckpt", default="checkpoints/rlt_token_bottleneck.pt")
    parser.add_argument("--state-stats", default="checkpoints/rlt_state_stats.json")
    parser.add_argument("--output", default="checkpoints/rlt_actor_critic.pt")
    parser.add_argument("--init-actor-ckpt", default=None)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--replay-capacity", type=int, default=200000)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--reward-mode", default="phase", choices=["phase", "dense"])
    parser.add_argument("--action-chunk-k", type=int, default=4)
    parser.add_argument("--bc-coef-start", type=float, default=0.05)
    parser.add_argument("--bc-coef-end", type=float, default=0.005)
    parser.add_argument("--demo-preload", action="store_true")
    parser.add_argument("--demo-stride", type=int, default=4)
    parser.add_argument("--demo-critic-steps", type=int, default=2000)
    parser.add_argument("--demo-critic-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    np.random.seed(7)
    torch.manual_seed(7)
    device = torch.device(args.device)

    state_stats = load_state_stats(args.state_stats)
    action_stats = load_action_stats(args.dataset)

    env = build_env(args.dataset)
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

    flat_dim = ACTION_DIM * PRED_HORIZON
    actor = ResidualActor(token_dim=token_dim, action_dim=ACTION_DIM, seq_length=PRED_HORIZON).to(device)
    critic = TwinQCritic(token_dim=token_dim, action_dim=ACTION_DIM, seq_length=PRED_HORIZON).to(device)
    critic_targ = TwinQCritic(token_dim=token_dim, action_dim=ACTION_DIM, seq_length=PRED_HORIZON).to(device)
    critic_targ.load_state_dict(critic.state_dict())

    if args.init_actor_ckpt is not None and Path(args.init_actor_ckpt).exists():
        init_obj = torch.load(args.init_actor_ckpt, map_location=device, weights_only=False)
        if isinstance(init_obj, dict) and "actor" in init_obj:
            actor.load_state_dict(init_obj["actor"], strict=False)
        elif isinstance(init_obj, dict):
            actor.load_state_dict(init_obj, strict=False)
        print(f"loaded actor init from {args.init_actor_ckpt}")

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    replay = ReplayBuffer(args.replay_capacity, token_dim, flat_dim)

    if args.demo_preload:
        print("building demo transitions from all demonstrations...")
        demo = build_demo_transitions(
            args.dataset,
            state_stats,
            action_stats,
            liquid,
            bottleneck,
            device,
            stride=args.demo_stride,
        )
        print(f"demo transitions: {demo['token'].shape[0]}")

        critic_pretrain_from_demos(
            critic,
            critic_targ,
            critic_opt,
            demo,
            device,
            steps=args.demo_critic_steps,
            batch_size=args.demo_critic_batch_size,
            gamma=args.gamma,
        )

        # Success-weighted replay preload using repeat counts
        for i in range(demo["token"].shape[0]):
            rep = int(demo["repeat"][i].item())
            for _ in range(rep):
                replay.push(
                    demo["token"][i],
                    demo["base_action"][i],
                    demo["action"][i],
                    demo["reward"][i],
                    demo["next_token"][i],
                    demo["next_base_action"][i],
                    bool(demo["done"][i].item()),
                )
        print(f"replay preloaded from demos: size={replay.size}")

    summary = {"episodes": []}
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        state = env_obs_to_state(obs)
        state_hist = collections.deque([state.copy() for _ in range(PRED_HORIZON)], maxlen=PRED_HORIZON)
        prev_metrics = _obs_metrics(obs)

        ep_ret = 0.0
        ep_steps = 0
        success = False
        while ep_steps < args.max_steps:
            hist_np = np.stack(state_hist, axis=0)
            hist_n = normalize_data(hist_np, state_stats)

            token = compute_token(liquid, bottleneck, hist_n, device)
            base_flat = base_action_seq_norm(liquid, hist_n, device)

            actor.train()
            action_flat, _, _ = actor(token.unsqueeze(0), base_flat.unsqueeze(0), deterministic=False)
            action_flat = action_flat.squeeze(0).detach()

            action_seq_norm = action_flat.view(PRED_HORIZON, ACTION_DIM).cpu().numpy()
            action_seq = unnormalize_data(action_seq_norm, action_stats)
            reward_acc = 0.0
            done = False
            last_obs = obs
            for k in range(max(1, min(args.action_chunk_k, PRED_HORIZON))):
                if ep_steps >= args.max_steps:
                    break
                action = np.clip(action_seq[k], -1.0, 1.0)
                next_obs, reward, _, _ = env.step(action)
                curr_metrics = _obs_metrics(next_obs)
                done = bool(env._check_success())
                if args.reward_mode == "phase":
                    shaped_r = _phase_reward(
                        reward,
                        prev_metrics,
                        curr_metrics,
                        action,
                        done,
                    )
                else:
                    shaped_r = _shaped_reward(
                        reward,
                        prev_metrics,
                        curr_metrics,
                        action,
                        done,
                    )
                reward_acc += shaped_r
                prev_metrics = curr_metrics
                next_state = env_obs_to_state(next_obs)
                state_hist.append(next_state)
                ep_ret += float(shaped_r)
                ep_steps += 1
                last_obs = next_obs
                if done:
                    break
            obs = last_obs

            next_hist_np = np.stack(state_hist, axis=0)
            next_hist_n = normalize_data(next_hist_np, state_stats)
            next_token = compute_token(liquid, bottleneck, next_hist_n, device)
            next_base_flat = base_action_seq_norm(liquid, next_hist_n, device)
            success = success or done

            replay.push(
                token.cpu(),
                base_flat.cpu(),
                action_flat.cpu(),
                torch.tensor(reward_acc, dtype=torch.float32),
                next_token.cpu(),
                next_base_flat.cpu(),
                done,
            )

            frac = (ep - 1) / max(1, (args.episodes - 1))
            bc_coef = args.bc_coef_start + frac * (args.bc_coef_end - args.bc_coef_start)

            for _ in range(args.updates_per_step):
                sac_update(
                    actor,
                    critic,
                    critic_targ,
                    actor_opt,
                    critic_opt,
                    replay,
                    device,
                    gamma=args.gamma,
                    alpha=args.alpha,
                    batch_size=args.batch_size,
                    bc_coef=bc_coef,
                )

            if done:
                break

        print(f"ep {ep:03d}/{args.episodes} return={ep_ret:.4f} steps={ep_steps} success={success} replay={replay.size}")
        summary["episodes"].append({"episode": ep, "return": ep_ret, "steps": ep_steps, "success": bool(success)})

    out = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "token_ckpt": args.token_ckpt,
        "token_dim": token_dim,
        "summary": summary,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/rlt_rl_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved RLT actor/critic -> {args.output}")


if __name__ == "__main__":
    main()
