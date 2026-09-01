# 03 — RLBench Simulator & Environment

## Overview

This module covers the paper and simulation environment **RLBench**: *The Robot Learning Benchmark & Learning Environment* (James et al., 2019). Modern robot learning requires benchmarks that are diverse, realistic, reproducible, and scalable. RLBench is designed to address these requirements by providing 100 unique, hand-designed robotic manipulation tasks using a 7-DoF Franka Panda arm, with support for automatic, infinite expert demonstration generation via motion planning.

---

## What you will learn

### 1. Key Design Properties
RLBench is designed around 6 core properties:
- **Diversity:** 100 unique, hand-designed manipulation tasks.
- **Reproducibility:** Built entirely in simulation (V-REP / CoppeliaSim + PyRep API).
- **Scale:** Automatically generates infinite demonstrations via motion planning.
- **Extensibility:** Easy to add new tasks via the Task Builder tool and PyRep API.
- **Tiered Difficulty:** Tasks range from simple reaching to complex, long-horizon multi-step tasks.
- **Realism:** Realistic robot models, physics, lighting, and domain randomization.

### 2. Hierarchy of Tasks
Understanding how tasks are structured in RLBench:
- **Episode Trajectory:** A sequence of (observation, action) pairs.
- **Variation:** A distribution over episodes where the object, target, or color changes.
- **Task:** A set of variations representing a specific semantic goal (e.g., "stack blocks").

### 3. Observation & Action Spaces
- **Observations:** Dual cameras (over-the-shoulder stereo and eye-in-hand monocular) providing RGB, Depth, and Segmentation masks, plus proprioceptive data (joint angles, velocities, torques, and gripper pose).
- **Actions:** 7 action spaces (absolute or delta joint velocities, positions, torques, or end-effector velocities/poses).

### 4. Sparse Rewards & Few-Shot Learning
- **Sparse Rewards:** Reward is $+1$ on task completion, $0$ otherwise.
- **Few-Shot Challenge:** Evaluates a robot's ability to learn a new task from a small number ($K$) of demonstrations.

---

## Notebooks

| File | Description |
|------|-------------|
| `rlbench_tutorial.ipynb` | Conceptual and visual walkthrough of the RLBench paper. Includes comparative benchmark analysis, task/variation/episode hierarchies, observation/action spaces, OMPL motion planning demonstration pipeline, and hands-on visualization code. |
| `rlbench_research_improvements.ipynb` | Conceptual tutorial and working implementations of three research improvements for RLBench: a learned task-embedding space using triplet loss metric learning, multimodal demonstration generation, and progress-based dense reward shaping via tabular Q-learning. |

---

## Setup

```bash
pip install numpy matplotlib scipy
```
