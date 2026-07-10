# 06 — CartPole DQN: Deep Q-Learning from Raw Pixels

## Overview

This module reconstructs the core idea of **Deep Q-Networks (DQN)**: *Playing Atari with Deep Reinforcement Learning* ([Mnih et al., 2013, arXiv:1312.5602](https://arxiv.org/abs/1312.5602)) — the result that launched modern deep reinforcement learning. A single convolutional network, trained with a variant of Q-learning plus an experience-replay memory, learns a control policy **directly from screen pixels**.

To fit Google Colab's free tier (T4, 16 GB), the notebook trains the *same algorithm and the same CNN* on **CartPole rendered as pixels**, which converges in minutes while still learning control from raw images. The pipeline and learning dynamics — not the absolute scores — are the lesson.

> 🎮 For research and educational purposes only. This is a didactic reconstruction of the 2013 DQN paper, not the official benchmark.

---

## What you will learn

### 1. From Q-learning to Deep Q-learning
- Markov Decision Processes, value functions, and the Bellman optimality equation.
- Why a neural network is used as a function approximator for $Q(s, a; \theta)$.

### 2. Learning from Pixels
- Frame preprocessing: grayscale conversion, resizing, and frame stacking to recover temporal information (velocity) from static images.
- A convolutional architecture that maps stacked frames to per-action Q-values.

### 3. Stabilizing Tricks
- **Experience replay:** breaking correlations between consecutive samples with a replay memory.
- **ε-greedy exploration:** annealing exploration into exploitation over training.
- The temporal-difference target and the MSE/Huber loss used to fit it.

### 4. Training & Evaluation
- The full training loop (Algorithm 1 from the paper).
- Logging and plotting episode returns, and rendering the learned policy as a rollout.

---

## Notebooks

| File | Description |
|------|-------------|
| `CartPoleDQN.ipynb` | End-to-end DQN implementation from raw pixels: frame preprocessing and stacking, the CNN Q-network, experience replay, ε-greedy exploration, the deep Q-learning training loop, and evaluation with reward curves and policy rollouts. |

---

## Setup

Runs on Google Colab free tier (T4 GPU) or CPU-only (slower). Runtime is ~10–20 min end-to-end on a T4.

```bash
pip install torch "gymnasium[classic-control]" numpy matplotlib imageio Pillow
```
