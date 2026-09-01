# 01 — 3-D Medical Image Segmentation with MONAI

## Overview

This module teaches you to train a full volumetric deep-segmentation pipeline from scratch
using the **MONAI** framework. The target task is spleen segmentation on abdominal CT from
the **Medical Segmentation Decathlon (MSD)** — a publicly available, standardised benchmark
that lets you reproduce results and compare against the literature.

By the end you will have trained a 3-D U-Net that segments the spleen at clinically useful
accuracy (Dice > 0.90 with 50 epochs on T4) and understand every component of the pipeline
well enough to adapt it to a new organ or dataset.

---

## What you will learn

### Data and I/O
- Why 3-D medical images are stored as **NIfTI (`.nii.gz`)** files, what the affine matrix
  encodes (voxel spacing, orientation, origin in world coordinates), and why all transforms
  must be affine-aware
- Using MONAI's `DecathlonDataset` to download, extract, and partition MSD datasets
  automatically with a reproducible 80/20 train/validation split
- Building a **MONAI transform pipeline**: `LoadImaged` → `EnsureChannelFirstd` →
  `Orientationd` → `Spacingd` → `ScaleIntensityRanged` → `CropForegroundd` →
  `RandCropByPosNegLabeld` → random augmentation

### Why patch-based training?
A full abdominal CT at 1 mm isotropic resolution is approximately 200 × 512 × 512 voxels
(~100 M floats). This does not fit in GPU memory for training. **Patch-based training**
randomly crops small 3-D sub-volumes (96 × 96 × 96) and feeds those to the network instead.
Crucially, `RandCropByPosNegLabeld` ensures that patches are sampled with equal probability
from foreground (spleen) and background regions — without this, the vast majority of patches
in an abdominal scan contain no spleen at all, and the model learns to predict "all
background" trivially.

### CacheDataset and the I/O bottleneck
Decompressing and resampling NIfTI files dominates training time on CPU, not the GPU. MONAI's
`CacheDataset` runs all deterministic transforms once during the first epoch, stores results
in RAM, and serves them instantly for all subsequent epochs.

| Configuration | RAM usage | Speedup vs plain Dataset |
|---------------|-----------|--------------------------|
| Plain `Dataset` | ~0 GB | 1× |
| `CacheDataset (rate=0.5)` | ~2–3 GB | ~4× |
| `CacheDataset (rate=1.0)` | ~4–6 GB | ~8× |

With 33 training spleen volumes on Colab (12 GB RAM), `cache_rate=1.0` is safe.

### 3-D U-Net architecture
The 3-D U-Net is the standard architecture for volumetric segmentation. It follows an
**encoder–decoder design with skip connections**:

```
Input (1, 96, 96, 96)
  │
  ├─ Enc L1: Conv3D(1→16)  ─────────────────────────────────── skip₁
  │         MaxPool3D ↓2×
  ├─ Enc L2: Conv3D(16→32) ────────────────────────────────── skip₂
  │         MaxPool3D ↓2×
  ├─ Enc L3: Conv3D(32→64) ──────────────────────────────── skip₃
  │         MaxPool3D ↓2×
  ├─ Bottleneck: Conv3D(64→128)
  │
  ├─ Dec L3: Upsample + concat(skip₃) → Conv3D(128→64)
  ├─ Dec L2: Upsample + concat(skip₂) → Conv3D(64→32)
  ├─ Dec L1: Upsample + concat(skip₁) → Conv3D(32→16)
  └─ Output: Conv3D(16→2) → Softmax
```

Skip connections pass fine spatial detail from the encoder directly to the decoder,
preventing the bottleneck from becoming a spatial information bottleneck. In 3-D this is
even more important than in 2-D because spatial context spans all three axes.

### Loss function: DiceCELoss
In spleen CT, the foreground occupies roughly 0.5–2% of all voxels. Pure cross-entropy
would achieve > 98% pixel accuracy by predicting all-background — **class imbalance** is
severe. `DiceCELoss` is a weighted combination:

$$\mathcal{L} = \lambda_{\text{Dice}} \cdot \mathcal{L}_{\text{Dice}} + \lambda_{\text{CE}} \cdot \mathcal{L}_{\text{CE}}$$

| Component | Strength | Weakness |
|-----------|----------|---------|
| Dice loss | Insensitive to class imbalance; directly optimises overlap | Noisy gradients early in training when predictions are near-zero |
| Cross-entropy | Stable per-voxel gradient signal throughout training | Ignores region-overlap quality; devastated by class imbalance alone |
| Combined | Stable early, high-quality late | Requires tuning of the λ weighting |

### Optimiser and learning rate schedule
**AdamW** uses decoupled weight decay — the regularisation is applied directly to the
weights rather than through the gradient, which is more principled and gives better
generalisation than Adam + L2 penalty. **Cosine annealing** gradually reduces the learning
rate from its initial value to near zero, which helps the model escape flat loss regions
early and fine-tune in narrow loss basins late in training.

### Mixed-precision training (AMP)
NVIDIA T4 GPUs contain **Tensor Cores** that run fp16 matrix multiplications 2–8× faster
than fp32. PyTorch AMP wraps the forward pass in `autocast()` to use fp16 where safe, then
uses a `GradScaler` to prevent fp16 gradient underflow during backprop. In practice this:
- Halves activation memory, allowing `batch_size=2` where fp32 would OOM
- Speeds up training by ~40–60% on T4
- Has negligible quality impact (gradients are upscaled back to fp32 before the optimiser step)

### Evaluation metrics
- **Dice Similarity Coefficient (DSC)** — measures volumetric overlap:
  `2·|P∩G| / (|P|+|G|)`, ranging from 0 (no overlap) to 1 (perfect). DSC > 0.90 is
  considered clinically useful for the spleen.
- **95th-percentile Hausdorff Distance (HD95)** — measures the worst-case boundary error
  at the 95th percentile (ignoring the top 5% of outlier voxels). A low HD95 means the
  boundary is well-localised even in difficult regions.

---

## Notebooks

| File | Description |
|------|-------------|
| `3d_medical_image_segmentation_monai.ipynb` | Complete pipeline: environment setup, MSD dataset download and preprocessing, CacheDataset construction, 3-D U-Net definition, DiceCELoss + AdamW + cosine LR, AMP training loop, validation with sliding-window inference, DSC/HD95 evaluation, and multi-planar result visualisation |

---

## Dataset

**MSD Task09_Spleen** — 41 portal-venous phase abdominal CT scans with binary spleen
segmentation masks. Labels: `0 = background`, `1 = spleen`. Download is ~1.6 GB and is
handled automatically by `DecathlonDataset` in the first notebook cell.

---

## Setup

```bash
pip install "monai[nibabel,tqdm]" torch torchvision matplotlib
```

Or use the install cell in the notebook (recommended on Colab).

---

## Common errors and how to fix them

| Error | Likely cause | Fix |
|-------|-------------|-----|
| CUDA out of memory | Batch size too large or fp32 used | Reduce `BATCH_SIZE` to 1, or enable AMP |
| Shape mismatch in loss | Channel dimension missing | Check your label tensor has shape `(B, 1, D, H, W)` |
| Very slow first epoch | CacheDataset is being populated | Wait — subsequent epochs will be much faster |
| DSC stuck at 0 for many epochs | Patch sampling has no foreground | Verify `pos_neg_ratio` and `num_samples` in `RandCropByPosNegLabeld` |

---

## Tips for beginners

- Print `tensor.shape` and `tensor.dtype` at every stage of the transform pipeline before
  training. Shape mismatches cause the majority of beginner errors in 3-D segmentation.
- The training loss and validation DSC should be plotted together. If loss keeps dropping
  but DSC plateaus, you are overfitting — add dropout or reduce the number of epochs.
- Start with `MAX_EPOCHS=10` to verify the pipeline runs end-to-end before committing to a
  full 50-epoch run.
- Sliding-window inference (`sliding_window_inference`) assembles the full-volume
  prediction from overlapping patches — this is the correct way to evaluate a patch-trained
  model and is different from how the model is trained.
