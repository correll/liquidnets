#!/usr/bin/env python3
"""
DAgger Pipeline for RoboMimic Can - SIMPLIFIED DEMO VERSION

Collects expert-corrected trajectories by:
1. Running liquid policy in MuJoCo simulation
2. Detecting failures (object distance from target)
3. Logging expert labels alongside executed actions
4. Saving augmented dataset for retraining

This version stores policy actions, expert actions, and executed actions so that
fine-tuning can use true DAgger supervision instead of imitating the possibly
bad executed policy action.
"""

import os
import json
import h5py
import numpy as np
import argparse
from pathlib import Path
from collections import deque
from datetime import datetime

import robosuite as suite
from robomimic.utils.file_utils import get_env_metadata_from_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque as _deque


class ActionNormalizer:
    """Min-max normalizer for actions."""
    def __init__(self, min_val, max_val):
        self.min_val = np.array(min_val, dtype=np.float32)
        self.max_val = np.array(max_val, dtype=np.float32)
    
    def normalize(self, action):
        return (action - self.min_val) / (self.max_val - self.min_val + 1e-8) * 2 - 1
    
    def unnormalize(self, action):
        return (action + 1) / 2 * (self.max_val - self.min_val + 1e-8) + self.min_val


# ============ Full Policy Architecture ============

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
        self.state_proj = nn.Linear(state_dim, hidden_dim)  # Per-timestep projection
        self.pos_emb = nn.Embedding(16, hidden_dim)  # PRED_HORIZON=16
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
    def __init__(self, backbone, action_dim=7, hidden_dim=291, seq_length=16, num_layers=2, num_mixtures=5):
        super().__init__()
        self.backbone = backbone
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_mixtures = num_mixtures
        self.context_proj = nn.Linear(backbone.hidden_dim, hidden_dim)
        self.latent_proj = nn.Linear(backbone.hidden_dim, hidden_dim)
        self.cfc_layers = nn.ModuleList([CfCCell(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.mdn_logits = nn.Linear(hidden_dim, num_mixtures)
        self.mdn_mu = nn.Linear(hidden_dim, num_mixtures * action_dim)
        self.mdn_log_sigma = nn.Linear(hidden_dim, num_mixtures * action_dim)

    def forward(self, images, state, return_mdn=False):
        b, t = images.shape[:2]
        context, latent = self.backbone(images, state)
        context = self.context_proj(context)
        latent = self.latent_proj(latent)
        hs = [context.clone() for _ in self.cfc_layers]
        logits_all = []
        mu_all = []
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
    def __init__(self, backbone, action_dim=7, hidden_dim=608, seq_length=16, num_diffusion_steps=50):
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
        self.time_embed = SinusoidalTimeEmbedding(64)
        self.time_mlp = nn.Sequential(nn.Linear(64, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
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

    def denoise_step(self, x_t, context, t):
        t_feat = self.time_mlp(self.time_embed(t))
        inp = torch.cat([x_t, context, t_feat], dim=-1)
        hidden = F.silu(self.denoise_net(inp) + self.input_proj(inp))
        return hidden, self.eps_head(hidden)

    def forward(self, images, state, deterministic=False):
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


def load_policy_liquid(checkpoint_path, device="cpu"):
    """Load liquid policy with full architecture."""
    model = LiquidTrajectoryModel(SharedBackbone(MockCLIPModel()))
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt)
    model.to(device).eval()
    return model


def load_policy_diffusion(checkpoint_path, device="cpu"):
    """Load diffusion policy with full architecture."""
    model = DiffusionTrajectoryModel(SharedBackbone(MockCLIPModel()))
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # Filter out MDN-specific keys that don't belong in diffusion
    filtered_ckpt = {k: v for k, v in ckpt.items() if not k.startswith(('mdn_', 'logits'))}
    model.load_state_dict(filtered_ckpt, strict=False)
    model.to(device).eval()
    return model


def get_obs_vector(obs_dict):
    """Extract 57D observation vector from env obs."""
    obs_keys = [
        'object-state',
        'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_quat_site',
        'robot0_gripper_qpos', 'robot0_gripper_qvel',
        'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel'
    ]
    
    obs_parts = []
    for key in obs_keys:
        if key in obs_dict:
            val = obs_dict[key]
            if isinstance(val, (list, tuple)):
                val = np.array(val)
            obs_parts.append(val.flatten().astype(np.float32))
    
    return np.concatenate(obs_parts, axis=0)


def get_policy_action(model, state_hist_deque, action_stats, device, policy_type="liquid", action_chunk_k=1):
    """
    Get action from policy using state history.
    
    state_hist_deque: deque of past observations (PRED_HORIZON long), each 57D
    Returns: next action (normalized, 7D)
    """
    # Stack state history into (PRED_HORIZON, 57D) as expected by model
    state_hist = np.stack(list(state_hist_deque), axis=0).astype(np.float32)  # (16, 57)
    
    # Normalize state per-timestep
    state_min = action_stats.get('state_min', np.array([-1.0] * 57))
    state_max = action_stats.get('state_max', np.array([1.0] * 57))
    state_norm = (state_hist - state_min) / (state_max - state_min + 1e-8) * 2 - 1
    
    # Create dummy image tensor (all zeros, not used in this eval)
    images = torch.zeros(1, 16, 96, 96, 3, device=device, dtype=torch.float32)
    state_tensor = torch.from_numpy(state_norm).unsqueeze(0).to(device, dtype=torch.float32)  # (1, 16, 57)
    
    with torch.no_grad():
        if policy_type == "liquid":
            out = model(images, state_tensor, return_mdn=True)
            logits = out["logits"][0, -1]  # use latest timestep
            mu = out["mu"][0, -1]          # (num_mixtures, action_dim)
            probs = torch.softmax(logits, dim=-1).unsqueeze(-1)
            action = (probs * mu).sum(dim=0).cpu().numpy()
        else:  # diffusion
            action_seq = model(images, state_tensor)  # (B, T, 7)
            action = action_seq[0, 0, :].cpu().numpy()  # (7,)
    
    return action  # (7,)


def random_policy_action(action_dim=7):
    """Generate random normalized action for demo."""
    return np.random.uniform(-1, 1, action_dim).astype(np.float32)


def _phase_metrics(obs_dict, object_pos_init_xy, object_pos_init_z):
    """Compute simple phase metrics from current observation."""
    eef_pos = np.asarray(obs_dict.get('robot0_eef_pos', np.zeros(3, dtype=np.float32)), dtype=np.float32)
    obj = np.asarray(obs_dict.get('object-state', np.zeros(10, dtype=np.float32)), dtype=np.float32)
    obj_pos = obj[:3] if obj.shape[0] >= 3 else np.zeros(3, dtype=np.float32)

    eef_to_obj_xy = float(np.linalg.norm(eef_pos[:2] - obj_pos[:2]))
    eef_to_obj_3d = float(np.linalg.norm(eef_pos - obj_pos))
    obj_drift_xy = float(np.linalg.norm(obj_pos[:2] - object_pos_init_xy))
    obj_lift = float(obj_pos[2] - object_pos_init_z)

    return {
        'eef_to_obj_xy': eef_to_obj_xy,
        'eef_to_obj_3d': eef_to_obj_3d,
        'obj_drift_xy': obj_drift_xy,
        'obj_lift': obj_lift,
    }


def _should_intervene_phase_aware(
    metrics,
    eef_hist,
    lift_hist,
    failure_threshold,
    intervention_rate,
    expert_hold,
    expert_hold_counter,
    cooldown,
    cooldown_counter,
):
    """Phase-aware trigger with hysteresis + cooldown.

    Returns: (should_intervene, new_hold_counter, new_cooldown_counter)
    """
    # Keep expert for a few steps once triggered
    if expert_hold_counter > 0:
        return True, expert_hold_counter - 1, cooldown_counter

    # Cooldown to avoid rapid toggling
    if cooldown_counter > 0:
        return False, 0, cooldown_counter - 1

    eef_now = metrics['eef_to_obj_xy']
    lift_now = metrics['obj_lift']
    drift_now = metrics['obj_drift_xy']

    # Always-fail condition (legacy drift kept as safety)
    fail_drift = drift_now > failure_threshold

    # Approach / alignment failures
    poor_alignment = eef_now > 0.08
    stalled_approach = False
    if len(eef_hist) >= 12:
        # no meaningful improvement over recent window
        stalled_approach = (eef_hist[0] - eef_hist[-1]) < 0.01

    # Lift failure: close to object but not lifting
    lift_failure = False
    if len(lift_hist) >= 15:
        close_but_not_lifting = (eef_now < 0.04) and ((max(lift_hist) - min(lift_hist)) < 0.012)
        lift_failure = close_but_not_lifting

    in_failure_state = fail_drift or (poor_alignment and stalled_approach) or lift_failure
    should_intervene = in_failure_state and (np.random.rand() < intervention_rate)

    if should_intervene:
        return True, expert_hold, cooldown_counter
    if in_failure_state:
        # failure but no intervention sampled: short cooldown
        return False, 0, cooldown
    return False, 0, cooldown_counter


def run_dagger_rollouts(
    policy_model,
    expert_model,
    policy_type,
    action_normalizer,
    action_stats,
    device="cpu",
    num_episodes=10,
    max_steps=1000,
    failure_threshold=0.15,
    intervention_rate=0.5,
    intervention_mode='phase',
    expert_hold_steps=15,
    cooldown_steps=6,
):
    """
    Run rollouts with policy and expert intervention at failure states.
    
    **How probabilistic expert queries work:**
    
    Stage 1: Detect failure state
      if object_distance > failure_threshold:
          in_failure_state = True
    
    Stage 2: Probabilistically decide to query expert
      if in_failure_state and random() < intervention_rate:
          use_expert = True
      else:
          use_expert = False
    
    Example with intervention_rate=0.5:
    - 50% of timesteps in failure state → query expert
    - 50% of timesteps in failure state → policy continues (learns from mistakes)
    - This balances between having recovery demonstrations and policy learning
    """
    
    print("Setting up MuJoCo environment...")
    dataset_path = 'datasets/robomimic/v1.5/can/ph/low_dim_v15.hdf5'
    meta = get_env_metadata_from_dataset(dataset_path)
    kwargs = dict(meta["env_kwargs"])
    kwargs["has_renderer"] = False
    kwargs["has_offscreen_renderer"] = False
    kwargs["use_camera_obs"] = False
    kwargs["reward_shaping"] = True
    kwargs["ignore_done"] = True
    env = suite.make(meta["env_name"], **kwargs)
    
    trajectories = []
    stats = {
        'total_episodes': 0,
        'total_interventions': 0,
        'success_episodes': 0,
        'avg_object_distance': 0.0,
    }
    
    intervention_count = 0
    
    for ep_idx in range(num_episodes):
        obs_dict = env.reset()
        obs_vec = get_obs_vector(obs_dict)
        
        obs_seq = [obs_vec.copy()]
        action_seq = []
        policy_action_seq = []
        expert_action_seq = []
        intervention_flags = []
        # Initialize state history deque with PRED_HORIZON (16) copies
        state_history_deque = deque([obs_vec.copy() for _ in range(16)], maxlen=16)
        
        object_pos_init = obs_dict['object-state'][:3].copy()
        eef_dist_hist = _deque(maxlen=20)
        obj_lift_hist = _deque(maxlen=20)
        expert_hold_counter = 0
        cooldown_counter = 0
        
        for step in range(max_steps):
            # Get actions from policy and expert
            policy_action = get_policy_action(policy_model, state_history_deque, action_stats, device, policy_type=policy_type)
            expert_action = get_policy_action(expert_model, state_history_deque, action_stats, device, policy_type="diffusion" if policy_type == "liquid" else "liquid")
            
            metrics = _phase_metrics(obs_dict, object_pos_init[:2], object_pos_init[2])
            eef_dist_hist.append(metrics['eef_to_obj_xy'])
            obj_lift_hist.append(metrics['obj_lift'])

            # ============ PROBABILISTIC INTERVENTION LOGIC ============
            if intervention_mode == 'phase':
                should_intervene, expert_hold_counter, cooldown_counter = _should_intervene_phase_aware(
                    metrics=metrics,
                    eef_hist=eef_dist_hist,
                    lift_hist=obj_lift_hist,
                    failure_threshold=failure_threshold,
                    intervention_rate=intervention_rate,
                    expert_hold=expert_hold_steps,
                    expert_hold_counter=expert_hold_counter,
                    cooldown=cooldown_steps,
                    cooldown_counter=cooldown_counter,
                )
            else:
                # Legacy drift-only trigger
                should_intervene = (metrics['obj_drift_xy'] > failure_threshold) and (np.random.rand() < intervention_rate)
            
            if should_intervene:
                action = expert_action
                intervention_flags.append(1)
                intervention_count += 1
            else:
                action = policy_action
                intervention_flags.append(0)

            policy_action_seq.append(policy_action.copy())
            expert_action_seq.append(expert_action.copy())
            
            # Execute in environment
            action_denorm = action_normalizer.unnormalize(action)
            obs_dict, reward, terminated, info = env.step(action_denorm)
            obs_vec = get_obs_vector(obs_dict)
            
            obs_seq.append(obs_vec.copy())
            action_seq.append(action.copy())
            state_history_deque.append(obs_vec.copy())
            
            if terminated or step == max_steps - 1:
                break
        
        final_distance = np.linalg.norm(
            obs_dict['object-state'][:3][:2] - object_pos_init[:2]
        )
        success = final_distance < 0.05
        
        trajectories.append({
            'obs_seq': np.array(obs_seq, dtype=np.float32),
            'action_seq': np.array(action_seq, dtype=np.float32),
            'policy_action_seq': np.array(policy_action_seq, dtype=np.float32),
            'expert_action_seq': np.array(expert_action_seq, dtype=np.float32),
            'intervention_flags': np.array(intervention_flags, dtype=np.int32),
            'success': int(success),
            'final_distance': float(final_distance),
            'length': len(action_seq),
        })
        
        stats['total_episodes'] += 1
        if success:
            stats['success_episodes'] += 1
        stats['total_interventions'] += int(sum(intervention_flags))
        stats['avg_object_distance'] += final_distance
        
        print(f"  Ep {ep_idx+1}/{num_episodes}: success={success}, dist={final_distance:.4f}, "
              f"interventions={sum(intervention_flags)}")
    
    env.close()
    stats['avg_object_distance'] /= num_episodes
    stats['total_interventions'] = intervention_count
    
    return trajectories, stats


def save_dagger_data_to_hdf5(
    trajectories,
    output_hdf5_path,
    split_name='dagger_demo',
):
    """Save DAgger trajectories to HDF5."""
    os.makedirs(os.path.dirname(output_hdf5_path), exist_ok=True)
    
    with h5py.File(output_hdf5_path, 'w') as f:
        # Create groups
        f.create_group('data')
        f.create_group('mask')
        
        # Add trajectories
        for traj_idx, traj in enumerate(trajectories):
            demo_name = f'{split_name}_ep{traj_idx:03d}'
            demo_group = f['data'].create_group(demo_name)
            
            demo_group.create_dataset('observations', data=traj['obs_seq'])
            demo_group.create_dataset('actions', data=traj['action_seq'])
            demo_group.create_dataset('policy_actions', data=traj['policy_action_seq'])
            demo_group.create_dataset('expert_actions', data=traj['expert_action_seq'])
            demo_group.create_dataset('interventions', data=traj['intervention_flags'])
            demo_group.attrs['success'] = traj['success']
            demo_group.attrs['final_distance'] = traj['final_distance']
            demo_group.attrs['length'] = traj['length']
        
        # Create mask split
        demo_names = [f'{split_name}_ep{i:03d}' for i in range(len(trajectories))]
        f['mask'].create_dataset(split_name, data=np.array(demo_names, dtype=h5py.string_dtype()))
    
    print(f"✓ Saved {len(trajectories)} trajectories to {output_hdf5_path}")


def main():
    parser = argparse.ArgumentParser(
        description="DAgger Pipeline for RoboMimic Can with Real Policies"
    )
    parser.add_argument('--policy-type', default='liquid', choices=['liquid', 'diffusion'])
    parser.add_argument('--policy-ckpt', required=True, help='Path to policy checkpoint')
    parser.add_argument('--expert-ckpt', required=True, help='Path to expert checkpoint')
    parser.add_argument('--num-episodes', type=int, default=10, help='Number of rollouts')
    parser.add_argument('--failure-threshold', type=float, default=0.15,
                       help='Object distance threshold for failure (meters)')
    parser.add_argument('--intervention-rate', type=float, default=0.5,
                       help='Probability of expert intervention when in failure state')
    parser.add_argument('--intervention-mode', default='phase', choices=['phase', 'drift'],
                       help='Failure trigger mode: phase-aware or legacy drift-only')
    parser.add_argument('--expert-hold-steps', type=int, default=15,
                       help='When expert is triggered, keep expert control for this many steps')
    parser.add_argument('--cooldown-steps', type=int, default=6,
                       help='Cooldown steps after failure check to avoid rapid toggling')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--output-dir', default='artifacts/dagger',
                       help='Output directory for HDF5 and metadata')
    parser.add_argument('--state-stats-json', default=None,
                       help='Optional JSON with state_min/state_max for normalization')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("DAgger Pipeline - Real Policy Integration")
    print("=" * 80)
    print(f"Policy Type: {args.policy_type}")
    print(f"Policy Checkpoint: {args.policy_ckpt}")
    print(f"Expert Checkpoint: {args.expert_ckpt}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Failure Threshold: {args.failure_threshold}m")
    print(f"Intervention Rate: {args.intervention_rate}")
    print(f"Intervention Mode: {args.intervention_mode}")
    print(f"Device: {device}")
    print()
    
    # Load action normalizer
    print("[1] Loading action normalizer...")
    original_hdf5 = 'datasets/robomimic/v1.5/can/ph/low_dim_v15.hdf5'
    with h5py.File(original_hdf5, 'r') as f:
        if 'stats' in f and 'action_min' in f['stats']:
            action_min = f['stats']['action_min'][()]
            action_max = f['stats']['action_max'][()]
        else:
            action_min = np.array([-1.0] * 7)
            action_max = np.array([1.0] * 7)
    action_normalizer = ActionNormalizer(action_min, action_max)
    action_stats = {'state_min': np.array([-1.0] * 57), 'state_max': np.array([1.0] * 57)}
    if args.state_stats_json is not None and os.path.exists(args.state_stats_json):
        with open(args.state_stats_json, 'r') as f:
            state_stats = json.load(f)
        action_stats['state_min'] = np.asarray(state_stats['state_min'], dtype=np.float32)
        action_stats['state_max'] = np.asarray(state_stats['state_max'], dtype=np.float32)
        print(f"  ✓ Loaded state normalization stats from {args.state_stats_json}")
    print("  ✓ Action normalizer ready")
    print()
    
    # Load policies
    print("[2] Loading policies...")
    print(f"  Loading {args.policy_type} from {args.policy_ckpt}")
    if args.policy_type == 'liquid':
        policy = load_policy_liquid(args.policy_ckpt, device=device)
        expert = load_policy_diffusion(args.expert_ckpt, device=device)
    else:
        policy = load_policy_diffusion(args.policy_ckpt, device=device)
        expert = load_policy_liquid(args.expert_ckpt, device=device)
    print("  ✓ Policies loaded")
    print()
    
    # Run rollouts
    print("[3] Running DAgger rollouts...")
    trajectories, stats = run_dagger_rollouts(
        policy_model=policy,
        expert_model=expert,
        policy_type=args.policy_type,
        action_normalizer=action_normalizer,
        action_stats=action_stats,
        device=device,
        num_episodes=args.num_episodes,
        failure_threshold=args.failure_threshold,
        intervention_rate=args.intervention_rate,
        intervention_mode=args.intervention_mode,
        expert_hold_steps=args.expert_hold_steps,
        cooldown_steps=args.cooldown_steps,
    )
    
    print()
    print("Results:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Successful: {stats['success_episodes']}")
    print(f"  Total interventions: {stats['total_interventions']}")
    print(f"  Avg object distance: {stats['avg_object_distance']:.4f}m")
    print()
    
    # Save to HDF5
    print("[4] Saving to HDF5...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    split_name = f'dagger_{args.policy_type}_{timestamp}'
    output_hdf5 = os.path.join(args.output_dir, f'dagger_trajectories_{split_name}.hdf5')
    
    save_dagger_data_to_hdf5(trajectories, output_hdf5, split_name=split_name)
    
    # Save metadata
    metadata = {
        'policy_type': args.policy_type,
        'policy_ckpt': args.policy_ckpt,
        'expert_ckpt': args.expert_ckpt,
        'num_episodes': args.num_episodes,
        'failure_threshold': args.failure_threshold,
        'intervention_rate': args.intervention_rate,
        'device': str(device),
        'stats': stats,
        'output_hdf5': output_hdf5,
        'split_name': split_name,
        'timestamp': timestamp,
    }
    
    metadata_path = os.path.join(args.output_dir, f'dagger_metadata_{split_name}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print()
    print("=" * 80)
    print("✓ DAgger Data Collection Complete!")
    print("=" * 80)
    print(f"HDF5 Output: {output_hdf5}")
    print(f"Metadata: {metadata_path}")
    print()
    print("Probabilistic Intervention Mechanism:")
    print(f"  Failure Threshold: {args.failure_threshold}m")
    print(f"    → When object drifts > {args.failure_threshold}m from initial position")
    print(f"  Intervention Rate: {args.intervention_rate * 100:.0f}%")
    print(f"    → Of failures, {args.intervention_rate * 100:.0f}% are corrected by expert")
    print(f"    → Of failures, {(1-args.intervention_rate) * 100:.0f}% policy continues (learns)")
    print()
    print(f"Interventions Collected: {stats['total_interventions']}")
    print(f"Total Timesteps: {sum(t['length'] for t in trajectories)}")
    print()



if __name__ == '__main__':
    main()
