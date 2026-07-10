# Phase 05 — Robotic Lab Automation & VLA Models

**Level: Advanced / Specialist** · Completion of Phase 01 and Phase 02 (or equivalent experience with PyTorch, basic computer vision, and physical simulation concepts) is recommended.

Phase 05 moves beyond static medical scans and image-processing pipelines to focus on **robotic lab automation**. In modern digital biology laboratories, general-purpose robotic systems must execute precise, long-horizon biological protocols (e.g. pipetting, centrifuging, mixing) using visual observations. This phase introduces the challenges of robotic manipulation in labs, physical simulation pipelines, and the evaluation of Vision-Language-Action (VLA) foundation models on custom benchmarks.

---

## Contents

| # | Subfolder | Core topic | Notebooks | GPU needed? |
|---|-----------|-----------|-----------|-------------|
| 1 | [`AutoBioBenchmark`](AutoBioBenchmark/) | AutoBio simulator architecture, detent/thread physics plugins, SR/SSR metrics, VLA baselines | `AutoBio_Tutorial.ipynb`<br>`AutoBio_Threading_Physics_for_VLAs.ipynb` | No |
| 2 | [`AutoBioComputerVision`](AutoBioComputerVision/) | Multi-view reasoning, tube localization, slot symmetry, liquid-level sensing, closed-loop UI control | `autobio_cv.ipynb` | No |
| 3 | [`RLBench`](RLBench/) | RLBench simulator architecture, design principles, waypoints & motion planning, observation & action spaces, standard RL loop, and few-shot challenges | `RLBench_Tutorial.ipynb`<br>`RLBench_Research_Improvements.ipynb` | No |
| 4 | [`DiffusionPolicy`](DiffusionPolicy/) | Diffusion Policy architecture, action diffusion models, noise scheduling, and bimanual visuomotor policy training | `diffusion_policy_tutorial.ipynb` | Yes |
| 5 | [`CALVIN`](CALVIN/) | CALVIN simulator architecture, language-conditioned policy, unstructured play data, goal relabeling, and long-horizon chaining metric | `calvin_tutorial.ipynb` | No |
| 6 | [`CartPoleDQN`](CartPoleDQN/) | Deep Q-Networks from raw pixels — frame stacking, CNN Q-network, experience replay, ε-greedy exploration, and the deep Q-learning training loop | `CartPoleDQN.ipynb` | Optional (T4) |

---

## Module Summaries

### Module 01 — AutoBio Simulator & Benchmark

Robotic manipulation in biology labs presents unique challenges: sub-millimeter precision requirements, transparent materials, and long-horizon protocol sequences where errors compound. This module introduces **AutoBio** (Lan et al., 2025), a custom MuJoCo-based framework designed to digitize real-world lab instruments, simulate helical screw-caps and detent dials, and benchmark state-of-the-art VLA models (e.g., OpenVLA, RDT-1B).

You will study the three-layer pipeline (Assets → Physics → Rendering), explore the math behind screw-thread helix geometry and detent snap potentials, and analyze why task-level Success Rate (SR) and Sub-Task Success Rate (SSR) are used together to diagnose VLA model failures.

**Key skills:** Simulation pipeline design, helical thread physics, detent potential modeling, SR vs. SSR metric evaluation, bimanual VLA baseline analysis.

---

### Module 02 — AutoBio Computer Vision & Closed-Loop Control

Every physical robotic action in the lab relies on first understanding the visual state of the workspace. This module translates the biological protocols of AutoBio into core computer vision tasks, providing a self-learning guide using synthetic toy datasets.

You will implement color-based tube localization and evaluate its robustness to sensor noise, perform circular-symmetry reasoning for rotor slot alignment, estimate the fill ratio of transparent tubes by boundary detection, and construct closed-loop visual control loops to actuate digital dials and panels on thermal mixers.

**Key skills:** Multi-camera visual reasoning, color-based object localization, circular symmetry slot assignment, liquid level estimation, closed-loop panel UI reading.

---

### Module 03 — RLBench Simulator & Environment

Teaching robots general manipulation requires benchmarks that scale. This module introduces **RLBench** (James et al., 2019), a robot learning benchmark built in V-REP/CoppeliaSim using the PyRep API.

You will study the 6 design properties of RLBench, explore the hierarchical relationship between Tasks, Variations, and Episodes, analyze the 6 observation modalities (stereo + monocular wrist cameras providing RGB-D and segmentation), and examine the 7 action spaces. You will also understand how waypoints combined with OMPL motion planning enable the generation of infinite expert demonstrations to train RL/IL policies.

**Key skills:** Robot simulation design, action/observation spaces, waypoint-based motion planning, sparse reward evaluation, few-shot policy benchmark analysis.

---

### Module 04 — Diffusion Policy

This module introduces **Diffusion Policy** (Chi et al., 2023), a framework that formulates robot action generation as a conditional denoising diffusion process. You will explore how diffusion models handle multi-modal action distributions, learn about noise scheduling and conditional action prediction, and study bimanual visuomotor policy training.

**Key skills:** Denoising diffusion probabilistic models, action sequences, noise scheduling, classifier-free guidance, bimanual policy networks.

---

### Module 05 — CALVIN Benchmark

This module introduces **CALVIN** (Mees et al., 2022), a language-conditioned long-horizon robot manipulation benchmark. You will study how language instructions are grounded to actions, how policies are trained using unstructured "play" data via goal relabeling, and how the long-horizon chaining metric evaluates performance.

**Key skills:** Language-conditioned imitation learning, unstructured play data, goal relabeling (MCIL), action chunking, contrastive language alignment, hybrid action spaces.

---

### Module 06 — CartPole DQN: Deep Q-Learning from Raw Pixels

The learned policies in earlier modules rest on foundations from deep reinforcement learning. This module reconstructs the core idea of **DQN** (Mnih et al., 2013): a convolutional network trained with Q-learning plus an experience-replay memory that learns control **directly from screen pixels**. To fit the Colab free tier, the same algorithm and CNN are trained on **CartPole rendered as pixels**, which converges in minutes while still learning from raw images.

You will follow the path from Q-learning to deep Q-learning, preprocess and stack frames to recover temporal information, implement the CNN Q-network, and use experience replay and ε-greedy exploration to stabilize training. The full training loop (Algorithm 1) is logged with reward curves and rendered policy rollouts.

**Key skills:** Value-based RL, function approximation, frame preprocessing/stacking, experience replay, ε-greedy exploration, temporal-difference targets, deep Q-learning training loop.

---

## Learning Goals

After completing Phase 05 you will be able to:

- Identify the main bottlenecks in robotic lab automation (precision limits, transparent glass/liquids, compounding errors)
- Describe the design of physics plugins for helical screw-caps, detents, and orbital mixers
- Evaluate robotic policies using Success Rate (SR) and Sub-Task Success Rate (SSR)
- Build synthetic lab scenes and implement localization, slot assignment, and liquid-level estimation pipelines
- Explain the difference between open-loop and closed-loop visual feedback control for digital instrument displays
- Walk through the evaluation of foundation VLA models on multi-step experimental protocols
- Describe the key design principles of the RLBench robot learning benchmark
- Explain the hierarchy of Task, Variation, and Episode in robot manipulation
- Map the cameras (over-the-shoulder stereo and eye-in-hand monocular) and image types (RGB, Depth, Segmentation) to their use cases
- Compare the 7 joint and end-effector action modes
- Explain how OMPL and waypoints generate infinite demonstration trajectories
- Describe the Diffusion Policy architecture and how action diffusion models predict multi-modal action trajectories
- Explain goal relabeling on unstructured play data for language-conditioned robot manipulation
- Calculate the long-horizon chaining metric to measure sequential task completion and compounding errors
- Trace the path from tabular Q-learning to deep Q-learning with a neural function approximator
- Implement a DQN that learns control from raw pixels using frame stacking, experience replay, and ε-greedy exploration

---

## Setup

```bash
pip install numpy matplotlib scipy opencv-python Pillow
```
