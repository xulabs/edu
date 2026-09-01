# 09 — Medical Image Segmentation

## Overview

Image segmentation involves assigning a class label to every pixel in an image, which is crucial for identifying structural boundaries of organs or abnormalities in medical imaging. This module introduces **2D Medical Image Segmentation** using **MONAI**, building upon previous image classification workflows.

---

## What you will learn

- The core components of 2D semantic image segmentation.
- Designing a MONAI-based training pipeline with pixel-level ground truth (masks).
- Implementing the standard **U-Net** architecture.
- Training with specialized segmentation loss functions, focusing on **Dice Loss** (and BCE combined).
- Evaluating predictions using the **Dice similarity coefficient** (Dice score).

---

## Notebooks

| File | Description |
|------|-------------|
| [`simple_segmentation.ipynb`](simple_segmentation.ipynb) | Hands-on tutorial on 2D image segmentation. Generates synthetic shape datasets, constructs a U-Net architecture using MONAI, trains the model with Dice loss, and evaluates performance. |
