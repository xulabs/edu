# 05 — CALVIN: Language-Conditioned Long-Horizon Robot Manipulation

## Overview

This module covers the concepts, architectures, and research extension opportunities for **CALVIN** (Common Action Learning from Vision): *A Benchmark for Language-Conditioned Long-Horizon Robot Manipulation* (Mees et al., 2022). General-purpose robotic systems must execute long-horizon multi-step instructions specified in natural language. CALVIN provides a simulated playground with 34 different tasks (like opening a drawer, turning on a light, picking up a block) and unstructured play data (imitation learning from play).

---

## What you will learn

### 1. The Language-Conditioning Challenge
- Why language-conditioned policies scale better than one-hot/goal-image policies but present hard alignment challenges.
- How unstructured "play" data is gathered without a single specific task in mind and converted into training trajectories.

### 2. Goal Relabeling on Play Data
- Using language instructions to dynamically label sub-trajectories in unstructured play datasets.
- Training policy networks using Multi-Context Imitation Learning (MCIL).

### 3. Long-Horizon Chaining Metric
- Evaluating robot success over sequences of consecutive tasks (1 to 5 tasks in a row).
- How the chaining metric measures the compounding rate of errors.

### 4. Advanced Research Improvements
- **Action Chunking:** Reducing temporal inconsistencies and cascading errors by predicting chunks of actions.
- **Language Alignment:** Aligning vision-action representations with text instructions using contrastive alignment.
- **Hybrid Action Spaces:** Adapting continuous and discrete joint actions for complex long-horizon tasks.

---

## Notebooks

| File | Description |
|------|-------------|
| [`calvin_tutorial.ipynb`](calvin_tutorial.ipynb) | A self-learning notebook explaining the CALVIN observation and action spaces, play data goal relabeling, baseline MCIL training, and proposed research improvements with working code implementations. |
| [`calvin_tutorial_improved.ipynb`](calvin_tutorial_improved.ipynb) | An improved and optimized version of the CALVIN tutorial notebook incorporating advanced play-data goal relabeling and training enhancements. |

---

## Setup

```bash
pip install numpy matplotlib scipy torch torchvision
```
