# Liquid Networks with Mixture Density Heads for Efficient Imitation Learning

This repository contains the code and notebooks for a controlled comparison of **Liquid+MDN** and **Diffusion** policy heads under a shared-backbone comparison protocol.

## Notebook Pipeline

Run notebooks in this order:

1. **01_training.ipynb**  
   Train liquid and diffusion models on Push-T, RoboMimic Can, and PointMaze; generate offline metrics.
2. **02_pointmaze_eval.ipynb**  
   Run PointMaze closed-loop evaluation with trained policies (**Python 3.12 / MuJoCo kernel**).
3. **03_analysis.ipynb**  
   Generate analysis figures and tables for the paper.
4. **04_pusht_closedloop.ipynb**  
   Run Push-T closed-loop evaluation (Table 2 results).
5. **05_libero_comparison.ipynb**  
   Optional: Extended analysis and comparisons.

---

## Environment Setup

- `.venv` (Python 3.14): Main training and analysis notebooks (`01`, `03`, `04`, `05`)
- `.venv312` (Python 3.12): PointMaze with MuJoCo notebook (`02`)

PointMaze requires `mujoco` and `gymnasium-robotics` in the Python 3.12 environment.

---

## Key Figures

### Training Progress (Push-T vs RoboMimic Can vs PointMaze)

![Training progress](training_progress.png)

### Best-of-K Performance Curve

![Best-of-K curve](best_of_k_curve.png)

### Error vs Horizon Analysis

![Error vs Horizon](error_vs_horizon.png)

### Qualitative Trajectory Samples

![Qualitative samples](qualitative_samples.png)

### Diversity vs Error Trade-off

![Diversity vs Error](diversity_vs_error.png)

## Paper Summary

This work compares **Liquid Neural Networks with Mixture Density (MDN) heads** against **Diffusion Policies** on three robotic imitation learning tasks:

- **Push-T**: Vision-based pusher task with object tracking
- **RoboMimic Can**: Canonical manipulation task from RoboMimic suite
- **PointMaze**: Navigation task with sparse geometric rewards

**Key Results:**
- Liquid policies use ~50% fewer parameters (4.3M vs 8.6M)
- Achieve 2.4× lower offline prediction error
- Run 1.8× faster at inference
- Remain more robust across low-to-medium data regimes
- Closed-loop validation confirms offline rankings on Push-T and PointMaze

---

## Artifacts & Results

- **Manuscript:** `paper/main.tex` (main research paper)
- **Figures & Data:** `artifacts/` contains all analysis outputs
  - Sample efficiency curves and learning dynamics
  - Best-of-K and diversity-error trade-offs
  - Offline and closed-loop evaluation tables

## Experimental Setup

All experiments follow a **fair shared-backbone protocol**:

1. **Backbone Encoder:** Push-T and RoboMimic Can share a vision backbone; all tasks use temporal context aggregation
2. **Policy Heads:** Compare Liquid+MDN vs Diffusion under matched:
   - Input dimensions (observation + context)
   - Training budgets (epochs, data, optimization)
   - Evaluation settings (rollout metrics, environments)
3. **Tasks & Metrics:**
   - Push-T: Success %, reward over 100 episodes
   - RoboMimic Can: Success % (task completion)
   - PointMaze: Success % (environment flag) & Distance-Success (stricter 0.20 threshold)
4. **Data Regimes:** Offline (1%–46.42% of training data) and closed-loop (100% data + adaptation)

---

## Legacy & Exploratory Code

The `transfomer/` folder contains earlier baseline implementations and exploratory notebooks.  
The numbered notebooks (01–05) represent the current canonical pipeline for paper reproducibility.
