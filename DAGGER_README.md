# DAgger Pipeline for RoboMimic Can

## Overview

This pipeline implements **Dataset Aggregation (DAgger)** to address the offline-to-online distribution shift problem in behavior cloning:

1. **Policy** (Liquid or Diffusion) runs closed-loop rollouts
2. **Failure states** detected (high object distance from target)
3. **Expert policy** queried at failure states for correct actions
4. **New trajectories** aggregated with original training data
5. **Policy retrained** on expanded dataset
6. **Process repeats** (iterate for multiple rounds)

## Key Insight

Standard behavior cloning trains only on expert demonstrations. At deployment, small errors compound, causing drift into states never seen during training. DAgger explicitly adds these out-of-distribution states with expert corrections, improving the policy's ability to recover from errors.

## Which Policy to Use?

**Start with Liquid Net** because:
- Shows better dense reward (0.0234 vs 0.0036 for diffusion baseline)
- More responsive to action corrections
- Lighter inference overhead for fast data collection

You can retrain both policies on the aggregated dataset afterward.

## Usage

### Step 1: Collect DAgger Data

```bash
python3 scripts/dagger_robomimic_can.py \
  --policy-type liquid \
  --policy-ckpt checkpoints/liquid_baseline_can.pt \
  --expert-ckpt checkpoints/robomimic_can_expert_replay.pt \
  --num-episodes 50 \
  --failure-threshold 0.15 \
  --intervention-rate 0.5 \
  --output-dir artifacts/dagger
```

**Parameters:**
- `--policy-type`: Which policy to improve ('liquid' or 'diffusion')
- `--policy-ckpt`: Checkpoint of policy to improve
- `--expert-ckpt`: Checkpoint of expert (80% success replay)
- `--num-episodes`: How many rollouts to run
- `--failure-threshold`: Object distance (m) above which to consider failure
- `--intervention-rate`: Probability of expert query when in failure state
- `--output-dir`: Where to save aggregated HDF5 and metadata

**Output:**
- `robomimic_can_aggregated_dagger_liquid_TIMESTAMP.hdf5`: New training dataset with expert corrections
- `dagger_metadata_dagger_liquid_TIMESTAMP.json`: Statistics and metadata

### Step 2: Retrain Policy

```bash
python3 scripts/retrain_dagger.py \
  --policy-type liquid \
  --aggregated-hdf5 artifacts/dagger/robomimic_can_aggregated_dagger_liquid_TIMESTAMP.hdf5 \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --output-dir checkpoints
```

**Output:**
- `checkpoints/dagger_liquid_best_epochNNN.pt`: Retrained policy

### Step 3: Evaluate Retrained Policy

```bash
python3 scripts/eval_robomimic_can_closedloop_mujoco.py \
  --checkpoint checkpoints/dagger_liquid_best_epochNNN.pt \
  --policy-type liquid \
  --num-episodes 100 \
  --action-chunk-k 8 \
  --record-video \
  --video-dir artifacts/videos
```

### Step 4: Optional - Iterate

Run DAgger again on the retrained policy to collect more failure data:

```bash
python3 scripts/dagger_robomimic_can.py \
  --policy-type liquid \
  --policy-ckpt checkpoints/dagger_liquid_best_epochNNN.pt \
  --expert-ckpt checkpoints/robomimic_can_expert_replay.pt \
  --num-episodes 50 \
  --failure-threshold 0.15 \
  --intervention-rate 0.7  # Higher intervention for more data
  --output-dir artifacts/dagger_round2
```

## Implementation Details

### Failure Detection

A state is considered a "failure" if:
- Object distance from initial position > `--failure-threshold` (default 0.15m)
- Random sampling with `--intervention-rate` probability

### Data Aggregation

- DAgger trajectories stored with per-timestep intervention flags
- Normalized using same stats as original training data
- Stored in new HDF5 split (e.g., `dagger_liquid_20260331_120000`)
- Original data preserved for comparison/ablation

### Training

- Retrains policy from scratch on aggregated dataset (original + DAgger)
- Uses same architecture and hyperparameters as initial training
- Saves best model based on validation loss

## Expected Improvements

- Round 1: Policies should learn to avoid high-distance failures
- Round 2+: Policies learn recovery behaviors (handling singularities, regrasping)

Typical progression:
- **Baseline**: 0% success (complete offline-to-online failure)
- **After DAgger R1**: 5-15% success (learns basic failure avoidance)
- **After DAgger R2+**: 30-60%+ success (learns recovery behaviors)

## Troubleshooting

### Expert policy not helping?
- Reduce `--failure-threshold` to intervene earlier
- Increase `--intervention-rate` for more expert queries
- Verify expert checkpoint path is correct (should be expert replay at 80% success)

### Training diverges?
- Reduce `--learning-rate` (try 5e-5)
- Add more DAgger episodes before retraining
- Check that aggregated HDF5 was created correctly

### Slow data collection?
- Reduce `--num-episodes` for quick iteration
- Set `--intervention-rate` lower to fewer expert calls
- Use `--device cuda` if available

## Next Steps

After reaching target success rate:
1. Retrain diffusion policy on same aggregated data
2. Compare liquid vs diffusion on closed-loop
3. Document offline-to-online gap closure in paper
4. Analyze what types of recovery behaviors were learned
