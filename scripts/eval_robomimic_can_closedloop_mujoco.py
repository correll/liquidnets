import argparse
import collections
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import robosuite as suite
from robomimic.utils.file_utils import get_env_metadata_from_dataset


OBS_KEYS = [
	"object",
	"robot0_eef_pos",
	"robot0_eef_quat",
	"robot0_eef_quat_site",
	"robot0_gripper_qpos",
	"robot0_gripper_qvel",
	"robot0_joint_pos",
	"robot0_joint_pos_cos",
	"robot0_joint_pos_sin",
	"robot0_joint_vel",
]
PRED_HORIZON = 16
ACTION_DIM = 7
STATE_DIM = 57
CLIP_DIM = 512


def decode_keys(arr):
	return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]


def concat_obs(obs_group):
	return np.concatenate([obs_group[k][:].astype(np.float32) for k in OBS_KEYS], axis=-1)


def get_data_stats(data):
	flat = data.reshape(-1, data.shape[-1])
	return {"min": np.min(flat, axis=0), "max": np.max(flat, axis=0)}


def normalize_data(data, stats):
	return (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-8) * 2 - 1


def unnormalize_data(ndata, stats):
	return (ndata + 1) / 2 * (stats["max"] - stats["min"]) + stats["min"]


def env_obs_to_state(obs):
	pieces = []
	for key in OBS_KEYS:
		env_key = "object-state" if key == "object" else key
		pieces.append(np.asarray(obs[env_key], dtype=np.float32).reshape(-1))
	return np.concatenate(pieces, axis=-1)


class MockCLIPModel(nn.Module):
	def __init__(self, embed_dim=CLIP_DIM):
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
	def __init__(self, clip_model, state_dim=STATE_DIM, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.1):
		super().__init__()
		self.clip_model = clip_model
		self.hidden_dim = hidden_dim
		self.image_proj = nn.Linear(CLIP_DIM, hidden_dim)
		self.state_proj = nn.Linear(state_dim, hidden_dim)
		self.pos_emb = nn.Embedding(PRED_HORIZON, hidden_dim)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=hidden_dim,
			nhead=num_heads,
			dim_feedforward=hidden_dim * 4,
			dropout=dropout,
			batch_first=True,
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.context_proj = nn.Linear(hidden_dim, hidden_dim)

	def encode_images(self, images):
		b, t, h, w, c = images.shape
		flat = images.reshape(b * t, h, w, c).permute(0, 3, 1, 2)
		with torch.no_grad():
			emb = self.clip_model.encode_image(flat)
		return emb.reshape(b, t, CLIP_DIM)

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


class SinusoidalTimeEmbedding(nn.Module):
	def __init__(self, embed_dim):
		super().__init__()
		self.embed_dim = embed_dim

	def forward(self, t):
		half = self.embed_dim // 2
		freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
		angles = t.float().unsqueeze(1) * freq.unsqueeze(0)
		emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
		if emb.shape[-1] < self.embed_dim:
			emb = F.pad(emb, (0, self.embed_dim - emb.shape[-1]))
		return emb


class DiffusionTrajectoryModel(nn.Module):
	def __init__(self, backbone, action_dim=ACTION_DIM, hidden_dim=608, seq_length=PRED_HORIZON, num_diffusion_steps=50, num_mixtures=5, time_embed_dim=64):
		super().__init__()
		self.backbone = backbone
		self.action_dim = action_dim
		self.hidden_dim = hidden_dim
		self.seq_length = seq_length
		self.num_diffusion_steps = num_diffusion_steps

		betas = torch.linspace(0.0001, 0.02, num_diffusion_steps)
		alphas = 1.0 - betas
		acp = torch.cumprod(alphas, dim=0)
		acp_prev = torch.cat([torch.ones(1), acp[:-1]], dim=0)
		self.register_buffer("betas", betas)
		self.register_buffer("alphas", alphas)
		self.register_buffer("alphas_cumprod", acp)
		self.register_buffer("alphas_cumprod_prev", acp_prev)
		self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(acp))
		self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - acp))
		self.register_buffer("posterior_variance", betas * (1.0 - acp_prev) / (1.0 - acp))
		self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(acp_prev) / (1.0 - acp))
		self.register_buffer("posterior_mean_coef2", (1.0 - acp_prev) * torch.sqrt(alphas) / (1.0 - acp))

		self.context_proj = nn.Linear(backbone.hidden_dim, hidden_dim)
		self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
		self.time_mlp = nn.Sequential(nn.Linear(time_embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
		in_dim = action_dim * seq_length + hidden_dim + hidden_dim
		self.input_proj = nn.Linear(in_dim, hidden_dim)
		self.denoise_net = nn.Sequential(
			nn.Linear(in_dim, hidden_dim * 2),
			nn.SiLU(),
			nn.Linear(hidden_dim * 2, hidden_dim * 2),
			nn.SiLU(),
			nn.Linear(hidden_dim * 2, hidden_dim),
		)
		self.eps_head = nn.Linear(hidden_dim, action_dim * seq_length)
		self.mdn_logits = nn.Linear(hidden_dim, num_mixtures)
		self.mdn_mu = nn.Linear(hidden_dim, num_mixtures * action_dim)
		self.mdn_log_sigma = nn.Linear(hidden_dim, num_mixtures * action_dim)

	def denoise_step(self, x_t, context, t):
		t_feat = self.time_mlp(self.time_embed(t))
		inp = torch.cat([x_t, context, t_feat], dim=-1)
		hidden = F.silu(self.denoise_net(inp) + self.input_proj(inp))
		return hidden, self.eps_head(hidden)

	def forward(self, images, state, inference=False, deterministic=False):
		b = images.shape[0]
		context, _ = self.backbone(images, state)
		context = self.context_proj(context)
		x_t = torch.randn(b, self.action_dim * self.seq_length, device=images.device)
		for step in range(self.num_diffusion_steps - 1, -1, -1):
			t = torch.full((b,), step, dtype=torch.long, device=images.device)
			_, pred_eps = self.denoise_step(x_t, context, t)
			sq_ab = self.sqrt_alphas_cumprod[step]
			sq_1ab = self.sqrt_one_minus_alphas_cumprod[step]
			pred_x0 = ((x_t - sq_1ab * pred_eps) / sq_ab.clamp(min=1e-8)).clamp(-1.0, 1.0)
			mean = self.posterior_mean_coef1[step] * pred_x0 + self.posterior_mean_coef2[step] * x_t
			if step > 0:
				noise = torch.randn_like(x_t)
				x_t = mean if deterministic else (mean + torch.sqrt(self.posterior_variance[step].clamp(min=1e-8)) * noise)
			else:
				x_t = mean
		return x_t.reshape(b, self.seq_length, self.action_dim).clamp(-1.0, 1.0)


def build_liquid_model(device):
	model = LiquidTrajectoryModel(SharedBackbone(MockCLIPModel()))
	ckpt = torch.load("checkpoints/liquid_jepa_robomimic_can_fair_halfparam_deterministic_clip_best.pt", map_location=device, weights_only=False)
	model.load_state_dict(ckpt)
	model.to(device).eval()
	return model


def build_diffusion_model(device):
	model = DiffusionTrajectoryModel(SharedBackbone(MockCLIPModel()))
	ckpt = torch.load("checkpoints/diffusion_jepa_robomimic_can_fair_halfparam_deterministic_clip_best.pt", map_location=device, weights_only=False)
	model.load_state_dict(ckpt)
	model.to(device).eval()
	return model


def load_eval_data(dataset_path):
	with h5py.File(dataset_path, "r") as f:
		train_keys = decode_keys(f["mask"]["train"][:])
		rng = np.random.default_rng(7)
		perm = rng.permutation(len(train_keys))
		n_inner = max(1, int(0.85 * len(train_keys)))
		inner_train_keys = [train_keys[i] for i in perm[:n_inner]]
		val_keys = [train_keys[i] for i in perm[n_inner:]]
		if not val_keys:
			val_keys = inner_train_keys[-max(1, len(inner_train_keys) // 10):]
			inner_train_keys = inner_train_keys[:-len(val_keys)]

		tr_states, tr_actions = [], []
		for key in inner_train_keys:
			demo = f["data"][key]
			tr_states.append(concat_obs(demo["obs"]))
			tr_actions.append(demo["actions"][:].astype(np.float32))

		val_episodes = []
		for key in val_keys:
			demo = f["data"][key]
			val_episodes.append(
				{
					"demo_key": key,
					"sim_states": demo["states"][:].astype(np.float64),
					"expert_actions": demo["actions"][:].astype(np.float32),
				}
			)

	state_stats = get_data_stats(np.concatenate(tr_states, axis=0))
	action_stats = get_data_stats(np.concatenate(tr_actions, axis=0))
	return state_stats, action_stats, val_episodes


def build_env(dataset_path, enable_offscreen=False):
	meta = get_env_metadata_from_dataset(dataset_path)
	kwargs = dict(meta["env_kwargs"])
	kwargs["has_renderer"] = False
	kwargs["has_offscreen_renderer"] = bool(enable_offscreen)
	kwargs["use_camera_obs"] = False
	kwargs["reward_shaping"] = True
	kwargs["ignore_done"] = True
	return suite.make(meta["env_name"], **kwargs)


def render_frame(env, camera_name="agentview", width=512, height=512):
	frame = env.sim.render(width=width, height=height, camera_name=camera_name)
	# Mujoco often returns upside-down frames in offscreen mode
	frame = np.flipud(frame)
	return np.asarray(frame, dtype=np.uint8)


def write_video(video_path, frames, fps=20):
	if not frames:
		return
	import imageio.v2 as imageio

	video_path = Path(video_path)
	video_path.parent.mkdir(parents=True, exist_ok=True)
	with imageio.get_writer(str(video_path), fps=fps) as writer:
		for fr in frames:
			writer.append_data(fr)


@torch.no_grad()
def policy_action(model, state_hist, state_stats, action_stats, device, kind, seed, use_state_norm=True, diffusion_deterministic=False):
	states_norm = normalize_data(state_hist, state_stats) if use_state_norm else state_hist
	states_t = torch.from_numpy(states_norm).unsqueeze(0).to(device=device, dtype=torch.float32)
	images_t = torch.zeros((1, PRED_HORIZON, 96, 96, 3), device=device, dtype=torch.float32)

	if kind == "liquid":
		out = model(images_t, states_t, return_mdn=True)
		logits = out["logits"][0]
		mu = out["mu"][0]
		best_k = logits.argmax(dim=-1)
		action_seq_norm = mu[torch.arange(PRED_HORIZON, device=device), best_k].cpu().numpy()
	else:
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
		action_seq_norm = model(images_t, states_t, inference=True, deterministic=diffusion_deterministic)[0].cpu().numpy()

	action_seq = unnormalize_data(action_seq_norm, action_stats)
	return np.clip(action_seq, -1.0, 1.0)


def _select_action_from_plans(step_idx, plan_bank, use_temporal_ensembling=False, ensemble_decay=0.1):
	active = []
	for plan in plan_bank:
		rel = step_idx - plan["start"]
		if 0 <= rel < plan["seq"].shape[0]:
			active.append(plan["seq"][rel])

	if not active:
		raise RuntimeError("No active action plans available for current step")

	if not use_temporal_ensembling or len(active) == 1:
		return active[-1]

	# Recency-weighted average (newest plan gets largest weight)
	weights = np.exp(-ensemble_decay * np.arange(len(active) - 1, -1, -1, dtype=np.float32))
	weights = weights / (weights.sum() + 1e-8)
	stacked = np.stack(active, axis=0)
	return (weights[:, None] * stacked).sum(axis=0)


def rollout_policy(
	model,
	env,
	episodes,
	state_stats,
	action_stats,
	device,
	kind,
	max_episodes=None,
	use_state_norm=True,
	diffusion_deterministic=False,
	action_chunk_k=1,
	use_temporal_ensembling=False,
	ensemble_decay=0.1,
	record_video=False,
	video_path=None,
	video_camera="agentview",
	video_width=512,
	video_height=512,
	video_fps=20,
):
	successes = 0
	max_rewards = []
	final_rewards = []
	episode_stats = []
	subset = episodes if max_episodes is None else episodes[: max_episodes]
	action_chunk_k = int(max(1, min(action_chunk_k, PRED_HORIZON)))

	for ep_idx, episode in enumerate(subset):
		episode_frames = []
		env.reset()
		env.sim.set_state_from_flattened(episode["sim_states"][0])
		env.sim.forward()
		if record_video and ep_idx == 0:
			episode_frames.append(render_frame(env, camera_name=video_camera, width=video_width, height=video_height))
		obs = env._get_observations()
		state = env_obs_to_state(obs)
		state_hist = collections.deque([state.copy() for _ in range(PRED_HORIZON)], maxlen=PRED_HORIZON)

		rewards = []
		success = False
		max_steps = int(episode["expert_actions"].shape[0])
		step_idx = 0
		plan_bank = []
		while step_idx < max_steps and not success:
			if (step_idx % action_chunk_k == 0) or (len(plan_bank) == 0):
				action_seq = policy_action(
					model,
					np.stack(state_hist, axis=0),
					state_stats,
					action_stats,
					device,
					kind,
					seed=1000 + ep_idx * 100 + step_idx,
					use_state_norm=use_state_norm,
					diffusion_deterministic=diffusion_deterministic,
				)
				plan_bank.append({"start": step_idx, "seq": action_seq})

			n_exec = min(action_chunk_k, max_steps - step_idx) if not use_temporal_ensembling else 1
			for i in range(n_exec):
				action = _select_action_from_plans(
					step_idx,
					plan_bank,
					use_temporal_ensembling=use_temporal_ensembling,
					ensemble_decay=ensemble_decay,
				)
				obs, reward, _, _ = env.step(action)
				if record_video and ep_idx == 0:
					episode_frames.append(render_frame(env, camera_name=video_camera, width=video_width, height=video_height))
				state = env_obs_to_state(obs)
				state_hist.append(state)
				rewards.append(float(reward))
				step_idx += 1
				plan_bank = [p for p in plan_bank if (step_idx - p["start"]) < PRED_HORIZON]
				success = bool(env._check_success())
				if success:
					break

		successes += int(success)
		max_reward = max(rewards) if rewards else 0.0
		final_reward = rewards[-1] if rewards else 0.0
		max_rewards.append(max_reward)
		final_rewards.append(final_reward)
		episode_stats.append(
			{
				"demo_key": episode["demo_key"],
				"success": success,
				"max_reward": max_reward,
				"final_reward": final_reward,
				"num_steps": len(rewards),
			}
		)
		print(
			f"{kind:9s} | episode {ep_idx + 1:02d}/{len(subset)} | "
			f"success={success} | max_reward={max_reward:.4f} | steps={len(rewards)}"
		)
		if record_video and ep_idx == 0 and video_path is not None:
			write_video(video_path, episode_frames, fps=video_fps)

	return {
		"success_rate": successes / max(1, len(subset)),
		"avg_max_reward": float(np.mean(max_rewards)) if max_rewards else 0.0,
		"avg_final_reward": float(np.mean(final_rewards)) if final_rewards else 0.0,
		"episode_stats": episode_stats,
	}


def rollout_expert(env, episodes, max_episodes=None):
	successes = 0
	max_rewards = []
	final_rewards = []
	episode_stats = []
	subset = episodes if max_episodes is None else episodes[: max_episodes]

	for ep_idx, episode in enumerate(subset):
		env.reset()
		env.sim.set_state_from_flattened(episode["sim_states"][0])
		env.sim.forward()

		rewards = []
		success = False
		for action in episode["expert_actions"]:
			_, reward, _, _ = env.step(action)
			rewards.append(float(reward))
			success = bool(env._check_success())
			if success:
				break

		successes += int(success)
		max_reward = max(rewards) if rewards else 0.0
		final_reward = rewards[-1] if rewards else 0.0
		max_rewards.append(max_reward)
		final_rewards.append(final_reward)
		episode_stats.append(
			{
				"demo_key": episode["demo_key"],
				"success": success,
				"max_reward": max_reward,
				"final_reward": final_reward,
				"num_steps": len(rewards),
			}
		)
		print(
			f"expert    | episode {ep_idx + 1:02d}/{len(subset)} | "
			f"success={success} | max_reward={max_reward:.4f} | steps={len(rewards)}"
		)

	return {
		"success_rate": successes / max(1, len(subset)),
		"avg_max_reward": float(np.mean(max_rewards)) if max_rewards else 0.0,
		"avg_final_reward": float(np.mean(final_rewards)) if final_rewards else 0.0,
		"episode_stats": episode_stats,
	}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default="datasets/robomimic/v1.5/can/ph/low_dim_v15.hdf5")
	parser.add_argument("--output", default="artifacts/robomimic_can_closedloop_mujoco.json")
	parser.add_argument("--max-episodes", type=int, default=10)
	parser.add_argument("--action-chunk-k", type=int, default=8)
	parser.add_argument("--temporal-ensembling", action="store_true")
	parser.add_argument("--ensemble-decay", type=float, default=0.1)
	parser.add_argument("--record-video", action="store_true")
	parser.add_argument("--video-dir", default="artifacts/videos")
	parser.add_argument("--video-camera", default="agentview")
	parser.add_argument("--video-width", type=int, default=512)
	parser.add_argument("--video-height", type=int, default=512)
	parser.add_argument("--video-fps", type=int, default=20)
	parser.add_argument("--no-state-norm", action="store_true")
	parser.add_argument("--diffusion-deterministic", action="store_true")
	parser.add_argument("--run-expert-replay", action="store_true")
	args = parser.parse_args()

	device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	torch.set_grad_enabled(False)
	np.random.seed(7)
	torch.manual_seed(7)

	state_stats, action_stats, val_episodes = load_eval_data(args.dataset)
	env = build_env(args.dataset, enable_offscreen=args.record_video)
	liquid = build_liquid_model(device)
	diffusion = build_diffusion_model(device)
	video_dir = Path(args.video_dir)
	video_dir.mkdir(parents=True, exist_ok=True)

	print(f"Evaluating {min(len(val_episodes), args.max_episodes)} validation episodes with MuJoCo / robosuite")
	liquid_res = rollout_policy(
		liquid,
		env,
		val_episodes,
		state_stats,
		action_stats,
		device,
		"liquid",
		args.max_episodes,
		use_state_norm=not args.no_state_norm,
		action_chunk_k=args.action_chunk_k,
		use_temporal_ensembling=args.temporal_ensembling,
		ensemble_decay=args.ensemble_decay,
		record_video=args.record_video,
		video_path=video_dir / "robomimic_can_liquid_rollout.mp4",
		video_camera=args.video_camera,
		video_width=args.video_width,
		video_height=args.video_height,
		video_fps=args.video_fps,
	)
	diffusion_res = rollout_policy(
		diffusion,
		env,
		val_episodes,
		state_stats,
		action_stats,
		device,
		"diffusion",
		args.max_episodes,
		use_state_norm=not args.no_state_norm,
		diffusion_deterministic=args.diffusion_deterministic,
		action_chunk_k=args.action_chunk_k,
		use_temporal_ensembling=args.temporal_ensembling,
		ensemble_decay=args.ensemble_decay,
		record_video=args.record_video,
		video_path=video_dir / "robomimic_can_diffusion_rollout.mp4",
		video_camera=args.video_camera,
		video_width=args.video_width,
		video_height=args.video_height,
		video_fps=args.video_fps,
	)

	out = {
		"dataset": args.dataset,
		"num_episodes": min(len(val_episodes), args.max_episodes),
		"ablations": {
			"use_state_norm": not args.no_state_norm,
			"diffusion_deterministic": args.diffusion_deterministic,
			"action_chunk_k": args.action_chunk_k,
			"temporal_ensembling": args.temporal_ensembling,
			"ensemble_decay": args.ensemble_decay,
			"record_video": args.record_video,
			"video_camera": args.video_camera,
			"run_expert_replay": args.run_expert_replay,
		},
		"liquid": liquid_res,
		"diffusion": diffusion_res,
	}
	if args.record_video:
		out["videos"] = {
			"liquid": str(video_dir / "robomimic_can_liquid_rollout.mp4"),
			"diffusion": str(video_dir / "robomimic_can_diffusion_rollout.mp4"),
		}
	if args.run_expert_replay:
		out["expert_replay"] = rollout_expert(env, val_episodes, args.max_episodes)
	out_path = Path(args.output)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(out, indent=2))
	print(f"Saved results to {out_path}")


if __name__ == "__main__":
	main()
