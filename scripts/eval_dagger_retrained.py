#!/usr/bin/env python3
"""
Evaluate retrained DAgger policy vs. original baseline policy.
"""
import argparse
import json
import torch
import numpy as np
import robosuite as suite
from collections import deque
from pathlib import Path

# Constants
OBS_KEYS = [
    "object", "robot0_eef_pos", "robot0_eef_quat", "robot0_eef_quat_site",
    "robot0_gripper_qpos", "robot0_gripper_qvel", "robot0_joint_pos",
    "robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel",
]
PRED_HORIZON = 16
ACTION_DIM = 7
STATE_DIM = 57


def get_obs_from_env(obs):
    """Extract observation vector from robosuite."""
    return np.concatenate([obs[k].astype(np.float32) for k in OBS_KEYS], axis=-1)


def get_action(policy, state_history, device='cpu', stats=None):
    """Get action from policy."""
    state_hist_tensor = torch.from_numpy(state_history).float().to(device)
    if state_hist_tensor.dim() == 2:
        state_hist_tensor = state_hist_tensor.unsqueeze(0)
    
    with torch.no_grad():
        action = policy(state_hist_tensor).squeeze(0).cpu().numpy()
    
    # Denormalize if stats provided
    if stats is not None:
        action = (action + 1) / 2 * (stats['act_max'] - stats['act_min']) + stats['act_min']
    
    return action


def evaluate_policy(policy, num_episodes=10, device='cpu', stats=None, policy_name=""):
    """Run policy on RoboMimic Can task."""
    env = suite.make(
        env_name="PickPlaceCan",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera=None,
    )
    
    successes = []
    final_distances = []
    
    for ep_idx in range(num_episodes):
        obs = env.reset()
        state_history = deque(maxlen=PRED_HORIZON)
        
        # Initialize history
        state_vec = get_obs_from_env(obs)
        for _ in range(PRED_HORIZON):
            state_history.append(state_vec)
        
        initial_obj_pos = obs['object'].copy()
        total_reward = 0
        
        for step in range(1000):
            # Get action
            state_hist_array = np.array(list(state_history))
            action = get_action(policy, state_hist_array, device=device, stats=stats)
            
            # Execute action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # Update state history
            state_vec = get_obs_from_env(obs)
            state_history.append(state_vec)
            
            if done:
                break
        
        # Get final metrics
        final_obj_pos = obs['object'].copy()
        distance = np.linalg.norm(final_obj_pos - initial_obj_pos)
        success = info.get('task_success', False)
        
        successes.append(success)
        final_distances.append(distance)
        
        print(f"  Ep {ep_idx+1}/{num_episodes}: success={success}, dist={distance:.4f}, reward={total_reward:.2f}")
    
    env.close()
    
    results = {
        'policy': policy_name,
        'num_episodes': num_episodes,
        'success_rate': np.mean(successes),
        'avg_distance': np.mean(final_distances),
        'std_distance': np.std(final_distances),
        'num_successes': int(np.sum(successes)),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-ckpt', 
                       default='checkpoints/liquid_jepa_robomimic_can_fair_halfparam_deterministic_clip_best.pt',
                       help='Original policy checkpoint')
    parser.add_argument('--retrained-ckpt',
                       default='checkpoints/dagger_liquid_retrained_best.pt',
                       help='Retrained policy checkpoint')
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print("=" * 80)
    print("Evaluating DAgger Retraining")
    print("=" * 80)
    print()
    
    # Load retrained policy
    print(f"[1] Loading retrained policy: {args.retrained_ckpt}")
    checkpoint = torch.load(args.retrained_ckpt, map_location=device)
    retrained_policy = checkpoint['policy'].to(device)
    retrained_policy.eval()
    stats = checkpoint.get('stats', None)
    print(f"  ✓ Loaded (best epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f})")
    print()
    
    # Load original policy
    print(f"[2] Loading original policy: {args.original_ckpt}")
    try:
        original_checkpoint = torch.load(args.original_ckpt, map_location=device)
        if isinstance(original_checkpoint, dict) and 'policy' in original_checkpoint:
            original_policy = original_checkpoint['policy'].to(device)
        else:
            original_policy = original_checkpoint.to(device)
        original_policy.eval()
        print("  ✓ Loaded original policy")
    except Exception as e:
        print(f"  ✗ Could not load original policy: {e}")
        print("  → Skipping original policy evaluation")
        original_policy = None
    print()
    
    # Evaluate retrained
    print(f"[3] Evaluating retrained policy ({args.num_episodes} episodes)...")
    retrained_results = evaluate_policy(
        retrained_policy, 
        num_episodes=args.num_episodes,
        device=device,
        stats=stats,
        policy_name="Retrained (DAgger)"
    )
    print()
    
    # Evaluate original if available
    if original_policy is not None:
        print(f"[4] Evaluating original policy ({args.num_episodes} episodes)...")
        original_results = evaluate_policy(
            original_policy,
            num_episodes=args.num_episodes,
            device=device,
            stats=None,
            policy_name="Original (Baseline)"
        )
        print()
    else:
        original_results = None
    
    # Summary
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"\nRetrained (DAgger):")
    print(f"  Success Rate: {retrained_results['success_rate']:.1%}")
    print(f"  Successes: {retrained_results['num_successes']}/{retrained_results['num_episodes']}")
    print(f"  Avg Distance: {retrained_results['avg_distance']:.4f}m (±{retrained_results['std_distance']:.4f})")
    
    if original_results:
        print(f"\nOriginal (Baseline):")
        print(f"  Success Rate: {original_results['success_rate']:.1%}")
        print(f"  Successes: {original_results['num_successes']}/{original_results['num_episodes']}")
        print(f"  Avg Distance: {original_results['avg_distance']:.4f}m (±{original_results['std_distance']:.4f})")
        
        improvement_sr = (retrained_results['success_rate'] - original_results['success_rate']) * 100
        improvement_dist = original_results['avg_distance'] - retrained_results['avg_distance']
        
        print(f"\nImprovement (Retrained vs Original):")
        print(f"  Success Rate: {improvement_sr:+.1f}%")
        print(f"  Distance: {improvement_dist:+.4f}m (better is lower)")
    
    print()
    
    # Save results
    results_path = 'artifacts/dagger_eval_results.json'
    Path('artifacts').mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        eval_data = {
            'retrained': retrained_results,
            'original': original_results,
            'num_episodes': args.num_episodes,
            'device': str(device),
        }
        json.dump(eval_data, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
