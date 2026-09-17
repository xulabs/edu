# 06 — Multimodal Deep Learning for Medical Imaging

## Overview

In clinical practice, diagnostic decisions are rarely made based on a single source of data. A doctor typically integrates imaging findings (e.g. chest X-rays) with tabular or clinical features (e.g. lab values, vital signs, demographics). **Multimodal deep learning** mirrors this clinical workflow by building models that jointly ingest and fuse multiple data modalities.

This module introduces the key architectures and fusion strategies used to combine vision and clinical/tabular features, evaluates their performance, and demonstrates contrastive pretraining.

---

## What you will learn

### 1. Fusion Strategies for Image and Tabular Data
- **Early Fusion (Input-Level):** Concatenating or combining modalities at the feature extractor input.
- **Late Fusion (Decision-Level):** Processing each modality through independent encoders and fusing their representations or logits near the output.
- **Cross-Attention Fusion (Intermediate-Level):** Allowing intermediate layers of one modality's encoder to attend to features of another modality, capturing complex cross-modal relationships.

### 2. Contrastive Multimodal Pretraining
- The intuition behind CLIP-style contrastive pretraining for medical applications.
- How to implement a symmetric contrastive loss to align image and clinical text/tabular embeddings.

### 3. Missing Modalities & Robustness
- What happens to a multimodal model when a modality is noisy or missing (e.g., a lab test is not ordered).
- Training and evaluation strategies to make networks robust to missing inputs.

---

## Notebooks

| File | Description |
|------|-------------|
| [`multimodal_deep_learning_medical_imaging_tutorial.ipynb`](multimodal_deep_learning_medical_imaging_tutorial.ipynb) | Conceptual and hands-on guide to building synthetic medical multimodal datasets, implementing early/late/cross-attention fusion models, performing contrastive pretraining, and testing robustness to missing modalities. |

---

## Setup

```bash
pip install torch torchvision numpy matplotlib
```
