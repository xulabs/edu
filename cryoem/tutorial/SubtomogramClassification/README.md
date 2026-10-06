# 13 — Subtomogram Classification with 3D Convolutional Neural Networks

## Overview

In cryo-electron tomography (cryo-ET), reconstructing a cellular tomogram is followed by identifying and classifying macromolecular complexes embedded in the cell. A key downstream task is **subtomogram classification**: given small 3D volumetric crops ("subtomograms") extracted around detected candidate coordinates, determine which macromolecular complex (e.g., spherical complexes like proteasomes, elongated complexes like filaments, or membrane-bound assemblies) each crop contains—or whether it represents pure background noise.

Subtomogram classification is a challenging 3D computer vision and deep learning problem because:
- **Extremely low signal-to-noise ratio (SNR):** Low electron doses (to preserve frozen-hydrated specimens) limit typical SNR to between 0.01 and 0.15.
- **Missing-wedge distortion:** Mechanical tilt limits (typically $\pm 60^\circ$ to $\pm 70^\circ$) create an unsampled wedge in Fourier space, causing directional blurring along the optical axis ($Z$).
- **Volumetric compute & sample complexity:** Convolutions in 3D scale cubically with volume resolution and require specialized regularizations.
- **Severe class imbalance:** In real tomograms, false-positive background crops outnumber true macromolecular complexes.

This module builds an end-to-end deep learning classification pipeline in PyTorch using synthetic 3D subtomograms. Because it generates synthetic data with realistic physical degradations (missing wedge, low SNR), the tutorial executes on any standard CPU or free-tier Google Colab GPU in minutes without requiring multi-gigabyte experimental datasets.

---

## What you will learn

- **Synthetic 3D Subtomogram Generation:** Formulating 3D geometric density phantoms (spherical complexes, rod-like filaments, and background noise volumes).
- **Frequency-Domain Missing Wedge Simulation:** Applying 3D FFT and angular Fourier masks to mimic tomographic reconstruction artifacts.
- **Cryo-ET Realistic Noise Modeling:** Corrupting 3D densities with Gaussian noise at low SNR ($\approx 0.15$) and performing per-sample volume standardization.
- **Volumetric Data Pipelines in PyTorch:** Implementing custom `Dataset` and `DataLoader` abstractions for 4D tensor volumes `(C, D, H, W)`.
- **3D Convolutional Neural Network (3D CNN) Design:** Constructing stacked `Conv3d`, `ReLU`, `MaxPool3d`, and `AdaptiveAvgPool3d` global pooling layers.
- **Evaluation on Imbalanced Scientific Data:** Moving beyond simple accuracy to multi-class confusion matrices, precision, and recall.
- **Connecting to Production Cryo-ET Tools:** Bridging toy prototypes to lab-scale tools such as `aitom`, rotation-invariant networks, and self-supervised representation learning.

---

## Notebooks

| File | Description |
|------|-------------|
| [`cryoet_ml_tutorial_PR.ipynb`](cryoet_ml_tutorial_PR.ipynb) | End-to-end PyTorch tutorial: synthetic 3D subtomogram simulation, missing-wedge frequency masking, 3D CNN classifier training, and confusion matrix / precision-recall evaluation. |

---

## Setup & Dependencies

```bash
pip install numpy torch matplotlib
```

The tutorial is designed to run efficiently on standard CPU or GPU environments (such as the Google Colab free tier).

---

## Further Reading & Related Modules

- **Related Modules in this Curriculum:**
  - [`SubtomogramAveraging`](../SubtomogramAveraging/): Subtomogram alignment, 3D template matching (NCC), and Fourier Shell Correlation (FSC).
  - [`FewShotParticleDetection`](../FewShotParticleDetection/): Particle localization using weak spherical annotations and Volume Infill augmentation.
  - [`MissingWedgeReconstruction`](../MissingWedgeReconstruction/): Physical and mathematical origins of the missing wedge in WBP and SIRT reconstructions.
- **Lab Toolkits & References:**
  - Xu, M. et al. Deep learning approaches for cryo-electron tomography and subtomogram analysis.
  - [`xulabs/aitom`](https://github.com/xulabs/aitom): Open-source AI platform for cryo-electron tomography.
