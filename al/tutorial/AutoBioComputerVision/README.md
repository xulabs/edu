# 02 — AutoBio Computer Vision & Closed-Loop Control

## Overview

This module translates the robotic biological primitives from the AutoBio benchmark into standard, hands-on computer vision and control loop tasks. For a robot to interact with a biology laboratory, it must solve visual reasoning problems: identifying tube locations, slot grids, liquid boundaries, and digital interfaces. This module uses synthetic toy data to illustrate these concepts step-by-step.

---

## What you will learn

### 1. Object Localization in Lab Scenes
Perception converts raw pixels into structured physical states. You will implement color-based thresholding to localize a target tube in a grid and evaluate the failure modes of classical CV under varying sensor noise.

### 2. Grid & Symmetry Reasoning
Centrifuges require rotor slots to be loaded symmetrically to avoid damage. You will study how coordinate layouts and index mappings allow the robot to select the symmetrically opposite slot ($j = (i + N/2) \bmod N$) for a given occupied slot $i$.

### 3. Transparent Liquid-Level Sensing
Estimating the fill ratio of a transparent tube requires locating the liquid surface boundary. You will implement a simple vertical boundary detector inside a cropped tube region:
$$\text{fill ratio} = \frac{y_{\text{bottom}} - y_{\text{surface}}}{H}$$
and analyze why tilted cameras, shadows, or specular reflections require deep neural networks rather than simple edge thresholds.

### 4. Closed-Loop Visual Feedback Control
Digital instruments require the robot to adjust settings (RPM, temperature, time) on a dial using visual feedback. You will implement a visual control loop that compares the current displayed state with the target instruction and proposes symbolic button presses to reduce the error:
$$\mathbf{e} = \mathbf{u}_{\text{target}} - \mathbf{u}_{\text{current}}$$

---

## Notebooks

| File | Description |
|------|-------------|
| `autobio_cv.ipynb` | Hands-on computer vision tasks inspired by the AutoBio paper. Includes synthetic lab scene generation, blue tube detection, symmetry indexing for rotor loading, liquid surface boundary detection, and a simulation of closed-loop button pressing to match target settings. |

---

## Setup

```bash
pip install numpy matplotlib opencv-python Pillow
```
