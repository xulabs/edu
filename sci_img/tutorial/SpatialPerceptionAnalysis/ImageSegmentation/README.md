# 03 — Image Segmentation

## Overview

Segmentation assigns a label to every pixel in an image — the most granular spatial
analysis possible. It is the foundational step for any downstream quantification: you
cannot measure cell area, track organelle movement, or compute the distance between nucleus
and membrane without first knowing which pixels belong to which object.

This module covers the full spectrum from classical geometric methods (which require no
labelled data and can be understood completely mathematically) to deep learning
(U-Net, which requires annotated training images but handles cases that classical methods
cannot). Understanding both families is essential: deep learning does not replace classical
methods; it complements them when the required training data exists.

---

## What you will learn

### Semantic vs. instance segmentation

Two distinct tasks are often both called "segmentation":

**Semantic segmentation** assigns a class label to every pixel: "this pixel is cytoplasm,
this is nucleus, this is background." Objects of the same class are indistinguishable —
two adjacent cells both labelled "cell" are merged.

**Instance segmentation** additionally distinguishes between individual objects: "this is
cell 1, this is cell 2." When cells touch, the boundary between them must be inferred.

| Task | Output | Method covered here |
|------|--------|---------------------|
| Semantic | Per-pixel class label | Thresholding, GMM, region-growing, basic U-Net |
| Instance | Per-pixel instance ID | Watershed, deep U-Net with instance heads |

### Classical method 1: intensity thresholding

The simplest segmentation: any pixel above a threshold $\tau$ is foreground; below is
background.

**Fixed threshold:** suitable only when illumination is perfectly uniform. Almost never
appropriate for real microscopy.

**Otsu's method:** finds the threshold that maximises the inter-class variance (or
equivalently minimises the weighted intra-class variance) of the two-component intensity
distribution. Works well when the image has a bimodal histogram (one peak for background,
one for foreground). Fails when the histogram is unimodal or has more than two modes.

**Adaptive thresholding:** computes a local threshold for each pixel based on its
neighbourhood mean or Gaussian-weighted mean. Essential when illumination is uneven.
Use `cv2.adaptiveThreshold` or `skimage.filters.threshold_local`.

### Classical method 2: Gaussian Mixture Models

When intensity alone does not cleanly separate classes, fit the histogram as a sum of
Gaussians and assign each pixel to the most likely component.

**GMM fitting (EM algorithm):**
1. Initialise $K$ Gaussian components with means $\mu_k$, variances $\sigma_k^2$, and
   mixing weights $\pi_k$
2. **E-step:** compute the posterior probability that each pixel belongs to each component:
   $r_{ik} \propto \pi_k \mathcal{N}(x_i; \mu_k, \sigma_k^2)$
3. **M-step:** update $\mu_k$, $\sigma_k^2$, $\pi_k$ using the weighted statistics
4. Repeat until convergence; assign each pixel to $\arg\max_k r_{ik}$

**When GMM beats Otsu:** when the image has multiple foreground types (cytoplasm vs.
organelle vs. background) or when foreground and background intensities overlap (SNR < 3).

**In scikit-learn:** `sklearn.mixture.GaussianMixture(n_components=3)` fits a GMM on the
flattened pixel intensity array. Set `n_components` by inspecting the histogram.

### Classical method 3: region-growing

Starting from a user-specified seed pixel, region-growing expands the foreground region
outward by adding neighbouring pixels that satisfy a similarity criterion:

$$|I(\text{neighbour}) - \mu_{\text{region}}| < \delta$$

where $\delta$ is a confidence interval threshold and $\mu_{\text{region}}$ is the current
region mean, updated as each pixel is added. The region grows until no more neighbours
satisfy the criterion.

**Multi-channel extension:** apply the criterion to all channels simultaneously (e.g. RGB
or multi-channel fluorescence). The joint Mahalanobis distance from the region mean
replaces the scalar intensity difference.

**Practical use:** region-growing is particularly useful for segmenting connected,
roughly-uniform structures with a known seed — a tumour region in MRI where the radiologist
marks a point inside the tumour, or a cell body where the nucleus centroid is the seed.

### Classical method 4: watershed segmentation

The **watershed algorithm** models the image as a topographic surface where pixel
intensity is elevation. Water is "flooded" upwards simultaneously from all local minima
(or from user-specified seed markers). Where two flooding regions meet, a watershed line
is drawn — this becomes the object boundary.

**Standard watershed** floods from all local minima → over-segments for noisy images.

**Marker-controlled watershed** (the recommended version):
1. Identify seed markers for each object and for the background (e.g. distance-transform
   maxima, regional minima after Gaussian smoothing, or manually placed points)
2. Flood from markers only, preventing over-segmentation
3. The watershed line between two marked regions is the segmentation boundary

**Why watershed excels at touching objects:** when two cells are adjacent, their shared
boundary produces a watershed line even without an explicit separation in the intensity
image. This is why watershed is the standard method for counting confluent cell cultures,
where cells touch extensively.

**Implementation:** `skimage.segmentation.watershed(image, markers)`. The `image` argument
should be the negative of the distance transform (so watershed "floods" from cell centres
outward) for cell segmentation.

### Deep learning: U-Net architecture

**U-Net** (Ronneberger et al., 2015) was originally designed for electron microscopy cell
segmentation and has since become the dominant architecture for biomedical image
segmentation. Its key innovation: an encoder–decoder structure with **skip connections**
that directly pass feature maps from each encoder level to the corresponding decoder level.

```
Input (1, 512, 512)
  │
  ├─ Enc1: Conv(1→64) ─────────────────────────────────────────────►  skip₁
  │        MaxPool ↓2×
  ├─ Enc2: Conv(64→128) ──────────────────────────────────────────►   skip₂
  │        MaxPool ↓2×
  ├─ Enc3: Conv(128→256) ────────────────────────────────────────►    skip₃
  │        MaxPool ↓2×
  ├─ Enc4: Conv(256→512) ──────────────────────────────────────►      skip₄
  │        MaxPool ↓2×
  ├─ Bottleneck: Conv(512→1024)
  │
  ├─ Dec4: Upsample + concat(skip₄) → Conv(1024→512)
  ├─ Dec3: Upsample + concat(skip₃) → Conv(512→256)
  ├─ Dec2: Upsample + concat(skip₂) → Conv(256→128)
  ├─ Dec1: Upsample + concat(skip₁) → Conv(128→64)
  └─ Output: Conv(64→n_classes) → softmax or sigmoid
```

**Why skip connections work:** max-pooling in the encoder progressively discards spatial
information to build semantic representations. By the bottleneck, the feature map has
collapsed to a small spatial resolution (e.g. 32×32 for a 512×512 input). Without skip
connections, the decoder must reconstruct boundary positions from this compressed
representation alone — an impossible task for thin membrane boundaries or subcellular
structures. Skip connections give the decoder direct access to the high-resolution spatial
features from each encoder stage.

**Loss functions for segmentation:**

| Loss | Formula | When to use |
|------|---------|------------|
| Binary cross-entropy | $-y\log\hat{p} - (1-y)\log(1-\hat{p})$ | Balanced classes, binary masks |
| Dice loss | $1 - 2\|P\cap G\| / (\|P\|+\|G\|)$ | Class imbalance; foreground is small |
| BCE + Dice combined | Weighted sum | Most biomedical segmentation tasks |
| Focal loss | $-(1-\hat{p})^\gamma \log\hat{p}$ | Severe class imbalance; hard example mining |

For the yeast-cell dataset in this module, the foreground (cells) is roughly 30–50% of
pixels — balanced enough that BCE alone is reasonable. For the spleen CT in Phase 03 Module
01, Dice is essential because foreground is less than 2% of all voxels.

**Pre-built U-Net implementations:**

- `segmentation_models_pytorch` (SMP): 10+ architectures (U-Net, U-Net++, FPN, DeepLab)
  with 100+ pre-trained backbone options; clean API
- MONAI: 3-D-first library with U-Net variants optimised for medical volumes (used in
  Phase 03 Module 01)
- PyTorch Hub: `torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana')` for a
  minimal 2-D U-Net

---

## Notebooks

| File | Description |
|------|-------------|
| `biomedical_cell_nuclei_segmentation.ipynb` | Self-directed study notebook on 2D cell and nuclei segmentation. Utilizes a pretrained Cellpose model for inference, covers robust intensity normalization, Otsu thresholding baselines, Dice and IoU agreement metrics, and regionprops-based morphology feature extraction. |
| `segmentation_geo_modified.ipynb` | Classical segmentation pipeline: Otsu thresholding with histogram analysis; adaptive thresholding; GMM fitting with component count selection; multi-channel region-growing; distance-transform-based marker computation; marker-controlled watershed; per-method IoU evaluation on ground-truth masks |
| `segmentation_deep.ipynb` | Deep instance segmentation: yeast-cell dataset download and DataLoader construction; U-Net from SMP with ResNet34 encoder; BCE + Dice loss training loop; validation IoU tracking; learning curve visualisation; inference on held-out images with confidence maps; comparison against watershed baseline |

---

## Dataset

**Yeast-cell microscopy dataset** — 493 brightfield microscopy images of yeast cells in
microstructure traps, with dense pixel-wise instance segmentation annotations for every
cell (arXiv:2304.07597). Each image contains 5–30 yeast cells in close contact, making
this an ideal benchmark for comparing touching-cell segmentation methods. Download is
handled automatically in the notebook.

Classical notebooks additionally use sample images from `skimage.data` (coins, cells,
microstructure) which require no download.

---

## Setup

```bash
pip install scikit-image opencv-python segmentation-models-pytorch monai \
    torch torchvision matplotlib numpy
```

---

## Hardware

| Task | GPU needed | VRAM usage |
|------|-----------|-----------|
| Classical segmentation | No | CPU only |
| U-Net training (SMP, ResNet34) | Yes | ~6 GB |
| U-Net inference | Optional | ~2 GB |

---

## Common errors and how to fix them

| Error | Likely cause | Fix |
|-------|-------------|-----|
| Otsu threshold cuts no foreground | Histogram is unimodal (no clear separation) | Try GMM; or apply background subtraction / flat-field correction first |
| Watershed over-segments | Too many local minima; seeds too close together | Apply more Gaussian smoothing to the distance transform before finding maxima |
| U-Net training loss does not decrease | Learning rate too high or too low | Try 1e-4 (start); decay with ReduceLROnPlateau if stuck |
| Validation IoU much lower than training | Overfitting; too few training images | Add augmentation (random flip, rotation, brightness jitter); reduce model depth |
| SMP model not found | Wrong architecture string | Use `smp.Unet(encoder_name='resnet34', encoder_weights='imagenet')` |

---

## Tips for beginners

- Always visualise the predicted mask **overlaid on the original image** at multiple
  examples, not just the aggregate IoU score. IoU of 0.75 can hide systematic failures on
  a specific cell morphology that only become apparent visually.
- For the watershed, the choice of markers matters more than any parameter. Good markers
  are one seed per object, reliably inside the object. Distance-transform maxima work well
  for round objects; for elongated or irregular shapes, try ridge-based seeds.
- Training a U-Net on fewer than ~100 labelled images requires heavy augmentation.
  At minimum: random horizontal/vertical flip, random 90° rotation, random brightness and
  contrast jitter. Add elastic deformation for further regularisation.
- When starting a new segmentation project, always run the classical methods first. If
  watershed achieves 0.85 IoU, training a U-Net to reach 0.87 may not be worth the
  annotation effort.
