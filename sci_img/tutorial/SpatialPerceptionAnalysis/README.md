# Phase 02 — Spatial Perception & Analysis

**Level: Intermediate** · Complete Phase 01 (or have equivalent experience with PyTorch tensors and basic image processing) before starting here.

Phase 01 gave you the vocabulary: tensors, filters, features. Phase 02 is about what you *do* with images — compressing them into meaningful low-dimensional spaces, finding and tracking individual objects, delineating their boundaries at pixel precision, and aligning images that were captured at different times or with different instruments. These four capabilities underpin the great majority of quantitative biology and clinical radiology workflows.

---

## Contents

| # | Subfolder | Core topic | Notebooks | GPU needed? |
|---|-----------|-----------|-----------|-------------|
| 1 | [`DimensionReductionLatentSpace`](DimensionReductionLatentSpace/) | PCA, latent-space geometry, image reconstruction | `dimension_reduction_reconstruction.ipynb` | No |
| 2 | [`ObjectDetectTracking`](ObjectDetectTracking/) | YOLOv5 detection, LAP-based 3-D cell tracking | `object_detection.ipynb`, `lap_laptrack.ipynb` | Yes (training) |
| 3 | [`ImageSegmentation`](ImageSegmentation/) | Thresholding, GMM, watershed, U-Net | `segmentation_geo_modified.ipynb`, `segmentation_deep.ipynb` | Yes (U-Net) |
| 4 | [`ImageAlignmentRegister`](ImageAlignmentRegister/) | SimpleITK registration, Pix2Pix GAN translation | `image_registration.ipynb`, `image_translation.ipynb` | Yes (Pix2Pix) |
| 5 | [`StereoVision`](StereoVision/) | Stereo geometry, DLT triangulation, block matching (SAD/SSD/NCC), rectification | `stereo.ipynb` | No |
| 6 | [`BioimageDenoising`](BioimageDenoising/) | Self-supervised denoising, Noise2Void, Noise2Self, classical baselines | `self_supervised_bioimage_denoising.ipynb` | Yes |

All GPU-using notebooks fall back to CPU automatically — training will be slower but the pipeline still runs.

---

## Module Summaries

### Module 01 — Dimensionality Reduction & Latent Space

A 512×512 fluorescence image is a point in a 262,144-dimensional space. Most of those dimensions are correlated, redundant, or dominated by noise. This module teaches you to find the small number of directions that actually matter — the principal components — and project every image onto them to obtain a compact representation that preserves the structure you care about.

The module uses the DRIVE retinal vessel dataset as a running example: watch how PC1 captures the dominant background gradient, how the cumulative explained variance curve reveals the right number of components to keep, and how images can be reconstructed from only 50 components with near-perfect visual quality. A latent-space scatter plot coloured by vessel density demonstrates whether the compression has separated the relevant biological variation — a diagnostic that reappears in Phase 03's generative model evaluation.

**Key skills:** `sklearn.decomposition.PCA`, explained variance analysis, per-component visualisation, PSNR/SSIM reconstruction quality, 2-D latent scatter plots, linear interpolation in latent space.

**Connects to:** Phase 03, Module 03 (flow-matching synthesis trains in a latent-like space); Phase 04, Module 02 (EM reconstruction can be viewed as compressed orientation estimation).

---

### Module 02 — Object Detection & Tracking

Static images tell you what a cell looks like; time-lapse images tell you what it *does*. This module builds the two capabilities needed to turn a 4-D (3-D + time) microscopy experiment into quantitative cell dynamics data.

**Detection:** You will train a YOLOv5 single-stage detector on a custom annotation dataset, learn how the CSPDarknet backbone and PANet neck extract and fuse features at three spatial scales, and evaluate predictions with precision-recall curves and mAP@0.5. Transfer learning from COCO pre-trained weights is demonstrated to show how many fewer epochs are needed compared to training from scratch.

**Tracking:** Once every frame is annotated with bounding boxes or centroids, LapTrack implements the extended Linear Assignment Problem to link detections across time into continuous trajectories. The module covers how the cost matrix encodes expected frame-to-frame displacement, how dummy rows and columns allow objects to appear and disappear naturally, and how gap-closing handles frames where the detector missed a cell.

**Key skills:** YOLOv5 annotation format (YOLO normalised coordinates), `data.yaml`, Roboflow export, anchor clustering, LAP cost matrix construction, gap-closing, napari visualisation of 3-D tracks.

**Connects to:** Phase 02, Module 03 (segmentation masks can replace bounding boxes as the detection representation); Phase 04, Module 01 (particle picking in cryoEM is template-matching detection followed by 2-D class averaging — the same detect-then-average paradigm).

---

### Module 03 — Image Segmentation

Segmentation assigns a label to every pixel — the most granular spatial analysis possible. This module covers the full spectrum from zero-data-required classical methods to data-hungry but high-performance deep learning, and gives you the tools to make the right choice for your dataset size and annotation budget.

**Classical pipeline:** Otsu thresholding (with its assumption of a bimodal histogram), adaptive thresholding for uneven illumination, Gaussian Mixture Models for multi-class intensity distributions, and multi-channel region-growing from user seeds. For instance segmentation of touching cells, the distance-transform-based marker-controlled watershed algorithm is demonstrated to reliably separate confluent cultures that threshold methods merge.

**Deep pipeline:** U-Net (via Segmentation Models PyTorch with a ResNet34 encoder) trained on a 493-image yeast-cell dataset. The module walks through DataLoader construction, the combined BCE + Dice loss that handles class imbalance, training-curve monitoring, and the critical habit of inspecting individual failure cases rather than relying on aggregate IoU.

**Key skills:** Otsu's method, `cv2.adaptiveThreshold`, `sklearn.mixture.GaussianMixture`, distance transform, `skimage.segmentation.watershed`, `segmentation_models_pytorch.Unet`, Dice loss, per-image prediction visualisation.

**Connects to:** Phase 03, Module 01 (3-D U-Net for spleen CT is a direct volumetric extension of this 2-D U-Net); Phase 02, Module 02 (segmentation masks can replace bounding boxes in the tracking pipeline).

---

### Module 04 — Image Alignment & Registration

Two images of the same structure are rarely in perfect correspondence. This module covers two complementary approaches to alignment:

**Classical iterative registration (SimpleITK):** The registration problem is formalised as optimising a similarity metric over a transformation family (rigid, affine, or deformable). The module demonstrates why Mean Squared Error fails for multimodal registration (CT ↔ MRI) and how Mutual Information solves this by measuring statistical dependency between intensity distributions rather than their values directly. A multi-resolution pyramid is shown to be mandatory for avoiding local minima — not an optional enhancement.

**Image-to-image translation (Pix2Pix GAN):** When the goal is not geometric alignment but appearance transformation — converting out-of-focus microscopy images to in-focus, or brightfield to virtual fluorescence — Pix2Pix learns the mapping from paired training examples. The U-Net generator (familiar from Module 03) is paired with a PatchGAN discriminator that classifies 70×70 overlapping patches rather than the whole image, forcing locally realistic texture generation.

**Key skills:** `sitk.ImageRegistrationMethod`, Mattes Mutual Information, multi-resolution pyramid, `ShrinkFactorsPerLevel`, Pix2Pix U-Net + PatchGAN architecture, L1 + GAN combined loss, TIFF 16-bit loading for neural training.

**Connects to:** Phase 03, Module 02 (multimodal registration with ANTs SyN and VoxelMorph-style deep networks); Phase 03, Module 03 (synthesis produces a missing modality so that registration can run monomodally with MSE).

---

### Module 05 — Stereo Vision & Depth Estimation

Reconstructing 3-D structure from 2-D images is a core task in spatial perception. This module introduces parallel stereo setups and the geometry of disparity and depth.

You will derive the mathematical inverse relationship between depth and disparity, and implement 3-D triangulation from scratch using the Direct Linear Transform (DLT) via Singular Value Decomposition. The module covers why stereo rectification is necessary to simplify matching to horizontal scanlines, and walks through the implementation of block matching cost functions (SAD, SSD, NCC) and the trade-offs of different matching window sizes. Finally, you will explore matching ambiguities such as repeated patterns and how structured light or smoothness constraints can resolve them.

**Key skills:** Stereo geometry derivation, DLT triangulation, epipolar line analysis, block matching implementation, window-size trade-off analysis, repeated pattern diagnosis.

**Connects to:** Phase 04, Module 02 (3-D reconstruction relies on similar geometric projections and gridding concepts).

---

### Module 06 — Bioimage Denoising

This module addresses the trade-off between exposure/phototoxicity and noise in biological microscopy. You will implement self-supervised learning algorithms that can denoise microscopy images directly from noisy observations without clean ground-truth targets.

You will study the theoretical foundations of **Noise2Void** and **Noise2Self**, implement classical filtering baselines (Gaussian, Median, Wiener, Bandpass), and train a self-supervised blind-spot neural network. You will also learn how to evaluate denoising results with Fourier Ring Correlation (FRC) and ensure biological structures are preserved without hallucinated features.

**Key skills:** Noise2Void blind-spot masking, Noise2Self J-invariance, classical filtering baselines, PyTorch self-supervised training loop, structural preservation validation, FRC.

**Connects to:** Phase 04, Module 01 (Noise2Void is the leading self-supervised denoiser for low-SNR cryoEM micrographs).

---

## Learning Goals

After completing Phase 02 you will be able to:

- Compress a high-dimensional image dataset with PCA, choose the right number of components from the explained-variance curve, and reconstruct images from the compact representation
- Interpret a 2-D latent scatter plot to assess whether dimensionality reduction has captured the relevant biological variation
- Prepare a custom object detection dataset in YOLO annotation format and train a YOLOv5 model using transfer learning from COCO weights
- Evaluate bounding-box predictions with precision, recall, and mAP@0.5; interpret precision-recall curves
- Build a LAP tracking pipeline for 3-D time-lapse data with LapTrack, including gap-closing and trajectory statistics
- Segment cells and subcellular structures using Otsu, adaptive, GMM, and watershed methods; choose the right method for your data's intensity characteristics
- Build and train a 2-D U-Net for pixel-wise instance segmentation, selecting the right loss for class imbalance
- Register a pair of images using gradient-descent optimisation over a Mutual Information metric, with a multi-resolution pyramid
- Train a Pix2Pix GAN to translate between imaging modalities on paired microscopy data
- Derive the relationship between depth and disparity, and triangulate 3-D points from 2-D matches using DLT
- Explain how stereo rectification simplifies correspondence searches to 1D scanlines
- Implement stereo block matching using SAD, SSD, and NCC, and analyze window size trade-offs and repeated pattern failures
- Create a controlled low-light microscopy denoising experiment and apply classical filtering baselines
- Implement Noise2Void structured masking and Noise2Self J-invariant masking in PyTorch
- Train a self-supervised blind-spot neural network and evaluate whether denoising preserves biological structures

---

## Prerequisites

- Phase 01 completed (tensors, convolution, Fourier basics, Python class definitions)
- Basic familiarity with PyTorch training loops (forward pass, loss, `optimizer.step()`)

---

## Hardware Requirements

| Module | GPU needed | VRAM usage | Notes |
|--------|-----------|-----------|-------|
| 01 — PCA | No | CPU only | Fast even on laptop |
| 02 — YOLOv5 | Yes (training) | ~4 GB | CPU fallback is very slow for training |
| 02 — LapTrack | No | CPU only | |
| 03 — Classical segmentation | No | CPU only | |
| 03 — U-Net | Yes | ~6 GB | ResNet34 encoder |
| 04 — SimpleITK | No | CPU only | |
| 04 — Pix2Pix | Yes | ~8 GB | U-Net + PatchGAN |
| 05 — Stereo Vision | No | CPU only | |

All notebooks detect GPU availability and fall back to CPU automatically.

---

## Setup

```bash
# Core dependencies — each notebook also installs its own in the first cell
pip install scikit-learn scikit-image segmentation-models-pytorch SimpleITK \
    laptrack napari torch torchvision matplotlib numpy tifffile opencv-python
```

---

## Suggested Order

The modules are designed to be taken in a specific order within this phase, but Modules 03 and 04 can be swapped once Module 01 is complete.

```
DimensionReductionLatentSpace
           ↓
ImageSegmentation         ← best before tracking (masks → centroids)
           ↓
ObjectDetectTracking
           ↓
ImageAlignmentRegister
           ↓
StereoVision
```

---

## Key Design Decisions in This Phase

**Why YOLOv5 (not a newer variant)?** YOLOv5 has extremely well-documented training scripts, a clean `data.yaml` interface, and is directly usable on Colab without custom builds. The architectural principles (backbone → neck → multi-scale head) are identical across all YOLO generations — mastering YOLOv5 gives you the mental model for evaluating its successors.

**Why classical methods before U-Net in Module 03?** Running watershed on your dataset before training a U-Net is never wasted time. If watershed achieves 0.85 IoU, the annotation effort required to close the gap to 0.90 with U-Net may not be justified. Understanding when deep learning adds value is as important as knowing how to train it.

**Why Pix2Pix alongside SimpleITK?** Registration corrects geometric misalignment; image translation corrects appearance differences. They solve different problems and are often combined in practice — synthesise a missing modality first, then register with MSE. Seeing both tools in the same phase makes the workflow clear.
