# 01 — AutoBio Simulator & Benchmark

## Overview

This module covers the paper and simulator framework **AutoBio**: *A Simulation and Benchmark for Robotic Automation in Digital Biology Laboratory* (Lan et al., 2025). Biology laboratories require robots to handle delicate instruments with sub-millimeter tolerances, manipulate transparent glass and liquids, interact with digital UIs, and execute long sequences of steps where errors accumulate. AutoBio is a custom MuJoCo-based simulation benchmark designed specifically to train and evaluate Vision-Language-Action (VLA) models in these conditions.

---

## What you will learn

### 1. AutoBio Framework Architecture
AutoBio constructs simulated laboratory environments through a three-layer pipeline:
- **3-D Assets Pipeline:** Scanning, cleaning, and exporting real lab instruments (centrifuges, vortex mixers, pipettes) as MJCF/URDF models.
- **Physics Plugins:** Simulating mechanisms that standard rigid-body engines cannot model (thread matching, detents, orbital mixing).
- **Rendering Stack:** Using Physically-Based Rendering (PBR) to realistically display transparent materials and dynamic LCD UIs.

### 2. Custom Physics Plugins
- **Thread Mechanism (Screw Caps):** Restricts cap motion along a helix trajectory:
  $$H(t) = [r \cos t, r \sin t, p t]^T$$
  where $r$ is the thread radius and $p$ is the helical pitch.
- **Detent Dials (Click-Stops):** Uses a periodic potential to simulate discrete dials:
  $$V(\theta) = -A \cos(n\theta)$$
  where $n$ is the number of snap positions and $A$ is the snap strength.

### 3. Success Metrics
To evaluate long-horizon tasks, AutoBio combines:
- **Success Rate (SR):** The fraction of tasks fully completed (binary, all-or-nothing).
- **Sub-Task Success Rate (SSR):** The fraction of individual sub-tasks completed, providing a granular progress signal to diagnose failure steps.

---

## Notebooks

| File | Description |
|------|-------------|
| `autobio_tutorial.ipynb` | Conceptual and visual walkthrough of the AutoBio paper. Includes visualizations of helix and detent potentials, PBR rendering comparison, benchmark task hierarchies, and performance reviews of OpenVLA and RDT-1B baselines. |
| `autobio_threading_physics_for_vlas.ipynb` | Hands-on replication of threading mechanisms using MuJoCo SDF contact plugins. Verifies the physics-fidelity of SDF helix contact dynamics vs. kinematic coupling for bimanual cap-screwing. |

---

## Setup

No special hardware or simulator installation is required for this tutorial notebook.
