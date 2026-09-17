# Bioimage Denoising

## Overview

Biological imaging often involves a difficult trade-off: increasing exposure produces cleaner images but leads to phototoxicity and photobleaching, while reducing illumination protects the sample but results in highly noisy images. 

This module introduces **Self-Supervised Denoising**, focusing on algorithms that learn to recover underlying biological structure directly from noisy microscopy observations *without* clean ground-truth training targets. We explore the foundational concepts of **Noise2Void** and **Noise2Self**, implement classical filtering baselines, and evaluate how denoising affects the downstream preservation of biological structures.

---

## What you will learn

### The Denoising Trade-off in Microscopy
- Why high exposure causes photobleaching and phototoxicity in living cells.
- How low-illumination setups reduce sample stress but degrade image quality.
- The core challenge: training denoising models when clean reference scans cannot be acquired.

### Self-Supervised Learning from Noise
- **Supervised Denoising:** Mapping noisy inputs to clean targets ($x_{\text{noisy}} \rightarrow x_{\text{clean}}$).
- **Noise2Noise:** Training on pairs of independent noisy observations of the same scene.
- **Noise2Void & Noise2Self:** Exploiting spatial independence of noise using a "blind-spot" constraint, where the network predicts a pixel's value from its surrounding context only.

### Evaluation & Biological Preservation
- Traditional metrics like **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index).
- **Fourier Ring Correlation (FRC):** Measuring resolution in frequency space without clean references.
- Why visual smoothness is a misleading proxy, and how to verify if denoising preserves actual biological structure rather than hallucinating features.

---

## Notebooks

| File | Description |
|------|-------------|
| [`self_supervised_bioimage_denoising.ipynb`](self_supervised_bioimage_denoising.ipynb) | Full hands-on tutorial using the BBBC039 microscopy dataset: preparing low-light synthetic noise, establishing classical baselines, implementing Noise2Void/Noise2Self blind-spot masking, training a self-supervised denoiser, and evaluating structural preservation. |

---

## Setup

```bash
pip install numpy matplotlib scipy torch torchvision scikit-image
```
