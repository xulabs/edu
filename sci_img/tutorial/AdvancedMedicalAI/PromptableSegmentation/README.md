# 07 — Promptable Biomedical Lesion Segmentation

## Overview

Traditional segmentation models are fully automatic, outputting masks for all learned classes. In contrast, **promptable segmentation** foundation models (such as SAM and MedSAM) represent a new paradigm of human-in-the-loop AI: a user provides a prompt (e.g. bounding box or point coordinate) to select a target object, and the model segments that specific object.

This module introduces prompt-guided segmentation, showing how prompts are encoded alongside images, and implements a lightweight promptable segmentation network with MedSAM-inspired reasoning.

---

## What you will learn

### 1. The Promptable Segmentation Paradigm
- The concept of promptable foundation models (SAM & MedSAM) and how they adapt to human-in-the-loop workflows.
- Why promptable models generalize better to unseen modalities or novel objects compared to fixed-class networks.

### 2. Prompt Representation and Channel Conditioning
- Representing coordinate point prompts as Gaussian peak channels.
- Representing bounding box prompts as binary mask channels.
- Feeding prompt channels alongside raw images into convolutional backbones.

### 3. Lightweight Learned Architectures & Training
- Building a lightweight promptable segmentation network with encoder-decoder paths and prompt feature conditioning.
- Training a prompt-guided model using synthetic biomedical lesion images.
- Comparing promptable networks to classical thresholding and distance-based baselines.

### 4. Quantitative Evaluation & Stress Testing
- Measuring segmentation performance with Dice, IoU, sensitivity, and specificity.
- Visualizing error maps, confidence maps, and model uncertainty.
- Stress testing the model with noisy, misplaced, or incomplete prompts.

---

## Notebooks

| File | Description |
|------|-------------|
| [`promptable_biomedical_lesion_segmentation.ipynb`](promptable_biomedical_lesion_segmentation.ipynb) | Conceptual and practical guide for learning prompt-guided biomedical segmentation. Includes dataset generation, model architecture, training, evaluation metrics, and ablation/stress-test studies. |

---

## Setup

```bash
pip install torch torchvision numpy matplotlib scikit-image
```
