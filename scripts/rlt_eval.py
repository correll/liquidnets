import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from eval_robomimic_can_closedloop_mujoco import (
    OBS_KEYS,
    ACTION_DIM,
    PRED_HORIZON,
    build_env,
    build_liquid_model,
    load_eval_data,
    normalize_data,
    render_frame,
    unnormalize_data,
    write_video,
)
from rlt_models import RLTokenBottleneck, ResidualActor


def env_obs_to_state(obs):
    pieces = []
    for key in OBS_KEYS:
        env_key = "object-state" if key == "object" else key
        pieces.append(np.asarray(obs[env_key], dtype=np.float32).reshape(-1))
    return np.concatenate(pieces, axis=-1)


@torch.no_grad()
def base_flat_and_token(liquid, bottleneck, hist_norm, device):
    states_t = torch.from_numpy(hist_norm).unsqueeze(0).to(device=device, dtype=torch.float32)
    images_t = torch.zeros((1, PRED_HORIZON, 96, 96, 3), device=device, dtype=torch.float32)
    _, latent = liquid.backbone(images_t, states_t)
    token = bottleneck.encoder(latent).squeeze(0)

    out = liquid(images_t, states_t, return_mdn=True)
    logits = out["logits"][0]
    mu = out["mu"][0]
    w = torch.softmax(logits, dim=-1).unsqueeze(-1)
    base_seq = (w * mu).sum(dim=1)
    return base_seq.reshape(-1), token


def rollout(
    env,
    episodes,
    liquid,
    bottleneck,
    actor,
    state_stats,
    action_stats,
    device,
    max_episodes=20,
    record_video=False,
    video_path=None,
    video_camera="agentview",
    video_width=512,
    video_height=512,
    video_fps=20,
):
    succ = 0
    dists = []

    subset = episodes[:max_episodes]
    for i, ep in enumerate(subset, 1):
        episode_frames = []
        env.reset()
        env.sim.set_state_from_flattened(ep["sim_states"][0])
        env.sim.forward()
        obs = env._get_observations()
        if record_video and i == 1:
            episode_frames.append(render_frame(env, camera_name=video_camera, width=video_width, height=video_height))

        state = env_obs_to_state(obs)
        state_hist = collections.deque([state.copy() for _ in range(PRED_HORIZON)], maxlen=PRED_HORIZON)

        final_dist = 1e9
        success = False
        for _ in range(len(ep["expert_actions"])):
            hist = np.stack(state_hist, axis=0)
            hist_n = normalize_data(hist, state_stats)

            base_flat, token = base_flat_and_token(liquid, bottleneck, hist_n, device)
            action_flat, _, _ = actor(token.unsqueeze(0), base_flat.unsqueeze(0), deterministic=True)
            action_seq_norm = action_flat.squeeze(0).detach().view(PRED_HORIZON, ACTION_DIM).cpu().numpy()
            action_seq = unnormalize_data(action_seq_norm, action_stats)
            action = np.clip(action_seq[0], -1.0, 1.0)

            obs, _, _, _ = env.step(action)
            if record_video and i == 1:
                episode_frames.append(render_frame(env, camera_name=video_camera, width=video_width, height=video_height))
            state = env_obs_to_state(obs)
            state_hist.append(state)

            can_pos = obs.get("Can_pos", obs.get("object-state", np.zeros(14, dtype=np.float32))[:3])
            target_pos = obs.get("target-object", np.array([0.0, 0.0, 0.0], dtype=np.float32))
            final_dist = float(np.linalg.norm(np.asarray(can_pos)[:3] - np.asarray(target_pos)[:3]))
            success = bool(env._check_success())
            if success:
                break

        succ += int(success)
        dists.append(final_dist)
        print(f"ep {i:02d}/{len(subset)} success={success} dist={final_dist:.4f}")
        if record_video and i == 1 and video_path is not None:
            write_video(video_path, episode_frames, fps=video_fps)

    return {
        "episodes": len(subset),
        "successes": int(succ),
        "success_rate": float(succ / max(1, len(subset))),
        "avg_dist": float(np.mean(dists)) if dists else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLT actor on robomimic can")
    parser.add_argument("--dataset", default="datasets/robomimic/v1.5/can/ph/low_dim_v15.hdf5")
    parser.add_argument("--liquid-ckpt", default="checkpoints/liquid_jepa_robomimic_can_fair_halfparam_deterministic_clip_best.pt")
    parser.add_argument("--token-ckpt", default="checkpoints/rlt_token_bottleneck.pt")
    parser.add_argument("--actor-ckpt", default="checkpoints/rlt_actor_critic.pt")
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts/rlt_eval_robomimic_can.json")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-path", default="artifacts/videos/rlt_rollout.mp4")
    parser.add_argument("--video-camera", default="agentview")
    parser.add_argument("--video-width", type=int, default=512)
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument("--video-fps", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    np.random.seed(7)
    torch.manual_seed(7)

    state_stats, action_stats, val_episodes = load_eval_data(args.dataset)
    env = build_env(args.dataset, enable_offscreen=args.record_video)
    liquid = build_liquid_model(device)
    liquid.eval()

    token_ckpt = torch.load(args.token_ckpt, map_location=device, weights_only=False)
    bottleneck = RLTokenBottleneck(backbone_dim=liquid.backbone.hidden_dim, token_dim=token_ckpt["token_dim"], seq_length=PRED_HORIZON).to(device)
    bottleneck.load_state_dict(token_ckpt["state_dict"])
    bottleneck.eval()

    actor_ckpt = torch.load(args.actor_ckpt, map_location=device, weights_only=False)
    actor = ResidualActor(token_dim=token_ckpt["token_dim"], action_dim=ACTION_DIM, seq_length=PRED_HORIZON).to(device)
    actor.load_state_dict(actor_ckpt["actor"])
    actor.eval()

    video_path = None
    if args.record_video:
        video_path = Path(args.video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
    results = rollout(
        env,
        val_episodes,
        liquid,
        bottleneck,
        actor,
        state_stats,
        action_stats,
        device,
        max_episodes=args.max_episodes,
        record_video=args.record_video,
        video_path=video_path,
        video_camera=args.video_camera,
        video_width=args.video_width,
        video_height=args.video_height,
        video_fps=args.video_fps,
    )
    if args.record_video and video_path is not None:
        results["video"] = str(video_path)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
