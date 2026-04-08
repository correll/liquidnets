#!/usr/bin/env python3
"""
Merge multiple DAgger HDF5 files into a single training dataset.
"""
import argparse
import h5py
import json
import os
from pathlib import Path


def merge_dagger_files(input_dir, output_path, min_intervention_rate=0.0, require_expert_actions=False):
    """Merge multiple DAgger HDF5 files into one."""
    
    input_dir = Path(input_dir)
    dagger_files = sorted(input_dir.glob("dagger_trajectories_*.hdf5"))
    
    if not dagger_files:
        print("❌ No DAgger trajectory files found!")
        return
    
    print(f"Found {len(dagger_files)} DAgger files to merge:")
    for f in dagger_files:
        print(f"  • {f.name}")
    print(f"Minimum intervention rate filter: {min_intervention_rate:.3f}")
    print(f"Require expert actions: {require_expert_actions}")
    
    # Create merged output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as out_file:
        ep_idx = 0
        total_transitions = 0
        total_interventions = 0
        episode_stats = []
        
        for file_path in dagger_files:
            print(f"\nProcessing {file_path.name}...")
            
            with h5py.File(file_path, 'r') as in_file:
                # Iterate over episodes in 'data' group
                data_group = in_file.get('data', in_file)
                
                for ep_key in sorted(data_group.keys()):
                    source_ep = data_group[ep_key]

                    actions = source_ep['actions'][()]
                    interventions = source_ep.get('interventions', None)
                    n_transitions = len(actions)
                    n_interventions = int(interventions[()].sum()) if interventions is not None else 0
                    intervention_rate = n_interventions / n_transitions if n_transitions > 0 else 0.0

                    if require_expert_actions and 'expert_actions' not in source_ep:
                        continue
                    if intervention_rate < min_intervention_rate:
                        continue
                    
                    # Create new episode group
                    new_ep_key = f'ep_{ep_idx:05d}'
                    ep_group = out_file.create_group(new_ep_key)
                    
                    # Copy all datasets
                    for key in source_ep.keys():
                        ep_group.copy(source_ep[key], key)
                    
                    # Track stats
                    total_transitions += n_transitions
                    total_interventions += n_interventions
                    
                    episode_stats.append({
                        'episode': ep_idx,
                        'source_file': file_path.name,
                        'transitions': n_transitions,
                        'interventions': n_interventions,
                        'intervention_rate': n_interventions / n_transitions if n_transitions > 0 else 0
                    })
                    
                    ep_idx += 1
                    
                    if ep_idx % 10 == 0:
                        print(f"  ✓ Merged {ep_idx} episodes ({total_transitions} transitions)...")
    
    print(f"\n✅ Merge complete!")
    print(f"  Total episodes: {ep_idx}")
    print(f"  Total transitions: {total_transitions}")
    print(f"  Total interventions: {total_interventions}")
    if total_transitions > 0:
        print(f"  Avg intervention rate: {100 * total_interventions / total_transitions:.1f}%")
    print(f"  Output: {output_path}")
    
    # Save merge metadata
    metadata_path = str(output_path).replace('.hdf5', '_metadata.json')
    metadata = {
        'total_episodes': ep_idx,
        'total_transitions': total_transitions,
        'total_interventions': total_interventions,
        'avg_intervention_rate': total_interventions / total_transitions if total_transitions > 0 else 0,
        'source_files': [str(f.name) for f in dagger_files],
        'episode_stats': episode_stats
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Metadata: {metadata_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='artifacts/dagger')
    parser.add_argument('--output-path', default='artifacts/dagger/merged_training_data.hdf5')
    parser.add_argument('--min-intervention-rate', type=float, default=0.0)
    parser.add_argument('--require-expert-actions', action='store_true')
    args = parser.parse_args()

    merge_dagger_files(
        args.input_dir,
        args.output_path,
        min_intervention_rate=args.min_intervention_rate,
        require_expert_actions=args.require_expert_actions,
    )
