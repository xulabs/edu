# Opentrons OT-2 Quality Control

## Overview

Automated liquid-handling robots like the Opentrons OT-2 are vital for high-throughput biological workflows. However, low-cost laboratory automation platforms often lack the real-time quality control (QC) features found in high-end systems. 

This module introduces a computer vision-based quality control pipeline for the Opentrons OT-2. Inspired by recent research, we implement a closed-loop system using synthetic camera feedback to detect common failure modes—such as missing pipette tips and incorrect liquid fill levels—and explore self-healing control loops.

---

## What you will learn

### 1. Liquid Handler QC Architectures
- The economic and technological trade-offs between affordable ($10k) and high-end ($100k+) liquid handlers.
- **Client-Server Split:** Offloading computationally intensive computer vision tasks from the OT-2's onboard Raspberry Pi 3+ to an external GPU server.

### 2. Pipette Tip & Volume Estimation
- Training a compact convolutional neural network (CNN) to classify pipette-tip presence and estimate liquid fill levels.
- **Geometric Localization:** A spatial coordinate algorithm for pinpointing which of the 8 multichannel pipette channels is missing a tip.
- **Volume Regression:** Using polynomial regression to convert visual fill percentage to microliter volumes, accounting for the nonlinear shape of pipette tips.

### 3. Closed-Loop Decision Logic
- Simulating a full 96-well plate protocol run with error injection.
- Implementing closed-loop control: `capture → detect → decide → go-ahead / halt`.
- **Research Extensions:** Pixel-segmentation-based volume estimation (solving low-volume detection weaknesses) and self-healing retry loops (automatically picking up missing tips).

---

## Notebooks

| File | Description |
|------|-------------|
| [`ot2_qc_tutorial.ipynb`](ot2_qc_tutorial.ipynb) | Hands-on tutorial implementing the OT-2 QC camera server, geometric missing-tip detector, fill% to volume polynomial regression, closed-loop simulation, and advanced self-healing logic. |

---

## Setup

```bash
pip install numpy matplotlib scipy torch torchvision
```
