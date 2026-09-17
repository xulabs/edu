# 08 — Medical Image Classification

## Overview

In clinical and research settings, classifying medical images (e.g., distinguishing between X-rays, CT scans, and MRIs, or identifying specific anatomical regions or pathologies) is a fundamental task. Automated classification helps triage scans, organize large clinical databases, and serve as a pre-processing step for downstream tasks like segmentation.

This module introduces **Medical Image Classification** using **MONAI** (Medical Open Network for AI), a specialized PyTorch-based framework. We train a deep convolutional neural network (**DenseNet121**) on the **MedNIST** dataset, which consists of 2D grayscale medical images across 6 distinct categories.

---

## What you will learn

### 1. Medical Image Classification Concepts
- Multi-class classification workflows for 2D medical images.
- Selecting appropriate evaluation metrics: Accuracy, Precision, Recall, F1-Score, and ROC AUC curves.

### 2. Loading and Preprocessing with MONAI
- Utilizing the `MedNISTDataset` API for standardized data retrieval.
- Designing pre-processing pipelines with channel conditioning, intensity scaling, resizing, and random data augmentations (e.g., rotations, flips) to prevent overfitting.

### 3. Model Architecture and Training
- Adapting the DenseNet121 architecture for single-channel (grayscale) 2D medical images.
- Implementing a clean training and validation loop using PyTorch and MONAI components.

### 4. Comprehensive Evaluation
- Evaluating model performance on a dedicated test set.
- Visualizing confusion matrices, ROC curves, and detailed classification reports to assess class-specific performance.

---

## Notebooks

| File | Description |
|------|-------------|
| [`medical_image_classification_mednist.ipynb`](medical_image_classification_mednist.ipynb) | Hands-on tutorial classifying 2D medical scans into categories using MONAI and DenseNet121. Covers data augmentation, training loops, test set evaluation, and visualization of classification performance metrics. |

---

## Setup

```bash
pip install torch torchvision monai[pillow] matplotlib scikit-learn
```
