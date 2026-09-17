# 02 — Multimodal Image Registration

## Overview

Image registration finds the geometric transformation that maps one image onto another so
that corresponding anatomical structures align. When the two images come from **different
imaging modalities** (CT ↔ MRI, T1 ↔ T2, MRI ↔ PET) this is called **multimodal
registration**, and it requires completely different similarity metrics from the single-
modality case.

This module covers the full spectrum from classical iterative optimisation to modern trained
deep-learning approaches, and gives you the mathematical and practical tools to choose the
right method for your task.

---

## What you will learn

### Why multimodal registration is hard

In single-modality registration (CT to CT, T1 to T1), the same anatomy has similar
intensities in both images, so minimising pixel-wise MSE works as a proxy for alignment.
In multimodal registration this breaks down: white matter in T1 MRI is bright; in T2 MRI
it is dark. MSE would report *maximum dissimilarity* for a perfectly aligned T1/T2 pair —
exactly backwards from what you want.

### Similarity metrics

**Mean Squared Error (MSE)** — works only for same-modality, same-scanner registration.
Avoid for any multimodal task.

**Normalised Cross-Correlation (NCC)** — tolerates linear intensity differences (one image
is a brightness/contrast-scaled version of the other). Appropriate for multi-site CT, but
not for CT↔MRI.

**Mutual Information (MI)** — the gold standard for multimodal registration:

$$\text{MI}(F, M) = \sum_{f,m} p_{FM}(f,m) \log \frac{p_{FM}(f,m)}{p_F(f)\,p_M(m)}$$

MI measures the statistical dependency between the two images' intensity distributions via
their joint histogram. When the images are perfectly aligned, the joint histogram has tight
clusters (high MI); when misaligned, the clusters smear (low MI). Crucially, MI does not
assume any particular relationship between intensity values — it is maximised regardless of
how T1 and T2 or CT and MRI contrast differs.

**Normalised Mutual Information (NMI)** is more robust when images partially overlap:

$$\text{NMI}(F, M) = \frac{H(F) + H(M)}{H(F, M)}$$

where $H$ is Shannon entropy. Use NMI when the field of view differs between modalities.

### Classical registration: SimpleITK

SimpleITK wraps the ITK registration framework and provides three key components that you
combine freely:

```
┌─────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Transform  │     │  Metric              │     │  Optimizer           │
│  ─────────  │     │  ─────────           │     │  ─────────           │
│  Rigid      │  +  │  Mattes MI           │  +  │  Gradient Descent    │
│  Affine     │     │  Joint Histogram MI  │     │  L-BFGS              │
│  B-Spline   │     │  NCC                 │     │  Evolutionary        │
└─────────────┘     └──────────────────────┘     └──────────────────────┘
```

**Multi-resolution pyramid strategy** — always register coarse-to-fine. Starting at a
downsampled resolution (e.g. ×8) provides a wide capture range and avoids local minima in
the smooth loss landscape; each refinement level reuses the previous transform estimate as
initialisation.

**Rigid registration** (6 DOF in 3-D, 3 DOF in 2-D) is appropriate when anatomical
differences are rigid — intra-subject brain MRI, or aligning two scans from the same session.

**Affine registration** (12 DOF in 3-D) adds scale and shear, accounting for scanner
calibration differences, respiratory compression along the slice axis, or linear inter-
subject shape differences.

**Deformable B-spline registration** models non-rigid deformations with a sparse grid of
control points. B-spline kernels ensure the displacement field is smooth:

$$\mathbf{u}(\mathbf{x}) = \sum_{i,j} \mathbf{c}_{ij}\, B_i(x) B_j(y)$$

Always initialise deformable registration with a rigid or affine estimate first — starting
deformable optimisation from scratch leads to degenerate solutions.

### Deformable registration: ANTs / SyN

Advanced Normalization Tools (ANTs) with Symmetric Normalization (SyN) is one of the most
widely used and best-validated deformable registration algorithms in neuroimaging. SyN
produces a **diffeomorphic** transform — smooth, invertible, and topology-preserving —
by optimising a symmetric image similarity metric simultaneously from both the fixed and
moving sides. Diffeomorphism guarantees that the Jacobian determinant is positive
everywhere (no tissue folding).

### Deep learning registration: VoxelMorph-style U-Net

Classical registration optimises a transformation *per image pair*, taking minutes to hours.
Deep learning instead trains a network **once** to learn the registration function, then
runs in milliseconds at inference.

Architecture:
```
[Fixed image | Moving image]    (2 input channels)
          │
     U-Net encoder
          │
     Bottleneck
          │
     U-Net decoder with skip connections
          │
  Dense displacement field u(x)
          │
  Spatial transformer (grid_sample) ──► Moved image M∘u
```

Loss function:
$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{sim}}(F,\, M \circ u)}_{\text{alignment}} + \lambda\, \underbrace{\|\nabla u\|^2}_{\text{smoothness}}$$

The smoothness regulariser penalises sharp discontinuities in the predicted displacement
field; without it the network learns spatially chaotic displacements that minimise the
metric by overfitting rather than by genuinely aligning anatomy.

### Evaluating registration quality

**Dice overlap on propagated segmentations** — if you have manually labelled structures in
the fixed image, warp the label map from the moving image into the fixed space using the
estimated transform and measure overlap. This is the most meaningful evaluation because it
directly captures anatomical alignment quality.

**Target Registration Error (TRE)** — if you have corresponding anatomical landmarks in
both images, measure the distance between them after registration. TRE < 2 mm is generally
considered clinically acceptable for brain MRI.

**Jacobian determinant** — for deformable transforms, compute `det(∇u + I)` everywhere.
Negative values indicate folding (physically impossible). A good registration should have
Jacobian determinant > 0 at every voxel.

---

## Notebooks

| File | Description |
|------|-------------|
| [`multimodal_image_registration.ipynb`](multimodal_image_registration.ipynb) | Synthetic brain phantom generation (T1 + T2 contrast, known ground-truth misalignment); metric comparison (MSE vs NCC vs MI); classical rigid + affine + B-spline registration with SimpleITK; SyN deformable registration with ANTs; deep-learning registration with a PyTorch U-Net + spatial transformer; Jacobian determinant analysis; comprehensive evaluation dashboard |
| [`biomedical_image_registration_and_deformation.ipynb`](biomedical_image_registration_and_deformation.ipynb) | From intensity alignment to lightweight VoxelMorph-style deformation reasoning: similarity metrics, brute-force translation baseline, tiny learned deformation-field model, lightweight training loop on synthetic data, and error visualization. |

---

## Practical context

Multimodal registration is a prerequisite for:
- **Radiotherapy planning** — aligning a diagnostic MRI (soft-tissue detail) onto a
  planning CT (electron density for dose calculation)
- **Longitudinal studies** — tracking tumour volume change across quarterly MRI scans
  from potentially different scanners
- **Atlas-based segmentation** — mapping a labelled anatomical template onto a new
  patient scan to initialise or replace manual annotation
- **Histology-to-MRI** — correlating post-mortem tissue sections with in-vivo MRI

---

## Setup

```bash
pip install SimpleITK antspyx dipy nilearn nibabel torch torchvision \
    scipy scikit-image matplotlib numpy
```

All packages are also installed in the notebook's first cell.

---

## Common pitfalls

| Pitfall | What happens | How to avoid |
|---------|-------------|-------------|
| Skipping multi-resolution pyramid | Optimizer falls into a local minimum | Always use `ShrinkFactors` and `SmoothingSigmas` in SimpleITK |
| Forgetting to normalise intensity | Metric gradients are poorly scaled | Normalise all images to [0,1] or zero-mean unit-variance before registration |
| Starting B-spline from scratch | Degenerate solution with folding | Always warm-start from a rigid/affine transform |
| Ignoring Jacobian determinant | Cannot detect physically invalid folding | Always check `det(∇u + I)` for negative values after deformable registration |

---

## Tips for beginners

- Work through the **metric comparison section** first. Plotting MI and MSE as a function
  of ground-truth misalignment angle makes the failure of MSE immediately visible.
- The SimpleITK registration object prints verbose convergence information — read it. If
  the metric is still changing rapidly at the last iteration, increase `NumberOfIterations`.
- For the deep-learning section, the smoothness weight `λ` is the key hyperparameter.
  Set it too low and the displacement field is spatially chaotic; too high and the network
  under-registers. Plot the Jacobian determinant histogram to find the sweet spot.
