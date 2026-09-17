# Phase 01 — Computational Signal Base

**Level: Beginner** · No prior PyTorch or image-processing experience required.

This phase is the on-ramp for the entire series. By the time you finish all four modules you will be fluent in PyTorch tensors, able to load and inspect DICOM and TIFF scientific images, design and apply spatial and frequency-domain filters, detect edges and geometric structures, and classify biological images using hand-crafted texture and shape features. Every later phase assumes this vocabulary.

---

## Why start here?

Medical and biological imaging AI is built on three layers: how to represent image data in a computer, how to manipulate it mathematically, and how to extract meaningful information from it. Deep learning covers the third layer automatically once you have enough training data — but understanding the first two makes every model you build more interpretable, easier to debug, and more robust in low-data regimes.

Phase 01 builds those first two layers by hand.

---

## Contents

| # | Subfolder | Core topic | Notebooks | GPU needed? |
|---|-----------|-----------|-----------|-------------|
| 1 | [`PyTorchTensorBasic`](PyTorchTensorBasic/) | Tensors, DICOM, TIFF, volumetric I/O | `introduction_to_pytorch.ipynb`, `basic_operations.ipynb` | No |
| 2 | [`FrequencyFiltering`](FrequencyFiltering/) | Convolution, Fourier transforms, image pyramids | `image_filtering.ipynb`, `image_filtering_notebook.ipynb`, `image_pyramids_and_frequency_domain.ipynb` | No |
| 3 | [`ClassicalFeatureExtract`](ClassicalFeatureExtract/) | Edge detection (Canny, Sobel), Hough Transform | `conv_edge_detection.ipynb`, `line_curve_detection.ipynb` | No |
| 4 | [`GeometryBasedClassification`](GeometryBasedClassification/) | GLCM texture, morphological features, SVM/k-NN | `geometry_based_classification.ipynb` | No |
| 5 | [`CornerDetection`](CornerDetection/) | Harris, Shi-Tomasi, and FAST detectors; feature descriptors and matching | `corner_detection_tutorial.ipynb`, `feature_detectors_descriptors.ipynb` | No |
| 6 | [`SpatialTransformations`](SpatialTransformations/) | Coordinate systems, 2D linear/affine/projective transforms, warping | `2d_transformations.ipynb` | No |

All notebooks are CPU-only. No GPU is needed anywhere in Phase 01.

---

## Module Summaries

### Module 01 — PyTorch Tensor Basics & Image I/O

Everything in this series is eventually a tensor. This module demystifies the data structure: how to create tensors, move them between CPU and GPU, index and reshape them without loops, and use broadcasting to apply per-channel operations efficiently. The second notebook shifts from synthetic data to real scientific images — DICOM (clinical CT/MRI) with its mandatory Hounsfield rescale, and multi-page TIFF (fluorescence/electron microscopy) with its 16-bit depth. By the end you will have a pipeline for loading, normalising, and visualising any volumetric medical image that every later phase reuses.

**Key skills:** `torch.Size`, `dtype`, `device`, broadcasting, `pydicom`, `tifffile`, multi-planar reformatting.

---

### Module 02 — Frequency Filtering & Image Pyramids

The Fourier Transform reveals that an image is a sum of waves at different spatial frequencies — low frequencies carry coarse structure, high frequencies carry edges and noise. Understanding this decomposition makes filter design systematic: a Gaussian blur is a low-pass filter in the frequency domain; sharpening is amplifying high frequencies; bandpass filtering selects a meaningful frequency band and discards the rest.

This module connects frequency-domain intuition to the spatial convolution kernels that CNN layers implement, and introduces image pyramids as multi-scale representations used later in registration (Phase 02) and cryoEM preprocessing (Phase 04).

**Key skills:** `np.fft.fft2`, Butterworth filters, Gaussian / Laplacian pyramids, the Convolution Theorem, Gibbs ringing.

---

### Module 03 — Classical Feature Extraction: Edges, Lines & Curves

Before a segmentation network can learn cell boundaries, those boundaries need to be represented somehow. This module shows how gradient operators (Sobel, Prewitt, Canny) detect intensity transitions at pixel resolution, and how the Hough Transform assembles scattered edge pixels into geometrically meaningful objects — straight lines for cell division axes, circles for nucleus detection.

The deeper lesson is the explicit connection between hand-crafted convolution kernels and learned CNN filters: a 3×3 `Conv2d` layer is structurally identical to applying a Sobel filter, except the kernel values are optimised by gradient descent rather than designed by a human. That insight makes Phase 02's U-Net feel like a natural continuation rather than a black box.

**Key skills:** Canny multi-stage pipeline, non-maximum suppression, Hough accumulator, `skimage.transform.hough_circle`, kernel parameter sweep.

---

### Module 04 — Geometry-Based Classification of Bio-Images

Given a segmented cell, how do you tell a mitochondrion from a lysosome without a GPU? This module answers that using the HeLa organelle dataset: compute GLCM texture features and morphological descriptors (`regionprops`) from each object, normalise the feature matrix, and classify with an SVM or k-NN. The result is a fully interpretable pipeline — you can inspect which features drive each prediction — that still achieves useful accuracy on ten organelle classes.

This module also sets up the motivation for Phase 02's PCA module: as the number of features grows, high-dimensional classifiers degrade unless the feature space is first compressed.

**Key skills:** `mahotas.features.haralick`, `skimage.feature.graycomatrix`, `StandardScaler`, SVM with RBF kernel, confusion matrix analysis.

---

### Module 05 — Corner Detection & Feature Descriptors

Unlike edges (1-D boundaries) or flat regions, a corner is a point where intensity changes significantly in all directions. This module introduces the mathematical foundation and implementation of key classical corner detectors and how they are described for matching across different imaging conditions:
- **Harris & Shi-Tomasi Corner Detectors:** Using the second-moment structure tensor and its eigenvalues to calculate corner response and locate interest points.
- **FAST (Features from Accelerated Segment Test):** A high-speed keypoint detector comparing intensity in a circular pixel ring.
- **Feature Descriptors:** Local representation of regions (e.g. raw patches, normalized patches, SIFT-like gradient magnitude and orientation histograms) to achieve illumination and rotation invariance.
- **Descriptor Matching:** Using distance metrics (Euclidean, SSD) to find corresponding features.

**Key skills:** Structure tensor, eigenvalues, corner response function $R$, FAST thresholding, local feature descriptors, SIFT gradient histograms, Euclidean distance matching.

---

### Module 06 — Spatial Transformations & Image Warping

Before registering or aligning two different images (such as pre- and post-operative scans), we must understand how to transform coordinates. This module builds the mathematical foundations of geometric mapping:
- **Linear Transformations:** Scaling, rotation, and shearing represented by $2 \times 2$ matrices.
- **Homogeneous Coordinates:** Scaling coordinates up to represent translation as matrix multiplication (affine transforms) and perspective warping (projective transforms/homography).
- **Image Warping:** Forward mapping (which leaves holes/aliasing) vs. inverse mapping (lookup coordinate in source image).
- **Bilinear Interpolation:** Resampling pixel values at fractional coordinates to ensure smooth warps.

**Key skills:** Homogeneous coordinates, translation/affine matrices, homography projection, forward/inverse image warping, bilinear interpolation.

---

## Learning Goals

After completing Phase 01 you will be able to:

- Create, reshape, index, and move tensors between CPU and GPU in PyTorch
- Load DICOM and TIFF medical images, inspect metadata, and apply the correct normalisation for each format
- Apply smoothing, sharpening, and edge-detection filters in both the spatial domain (kernels) and the frequency domain (FFT masking)
- Explain how convolution underpins convolutional neural network layers, and interpret kernel parameters (size, padding, stride, dilation)
- Extract GLCM texture features and geometric descriptors from segmented biological objects
- Train and evaluate a classical classifier (SVM, k-NN) on a hand-crafted feature matrix

---

## Prerequisites

- Python 3.9+ with NumPy
- No prior PyTorch, signal processing, or medical imaging experience

---

## Setup

```bash
# All dependencies (each notebook also installs its own in the first cell)
pip install torch torchvision pydicom tifffile numpy scipy \
    scikit-image opencv-python matplotlib mahotas scikit-learn
```

---

## Suggested Order

```
PyTorchTensorBasic
       ↓
FrequencyFiltering
       ↓
ClassicalFeatureExtract  →  CornerDetection
       ↓                              ↓
GeometryBasedClassification  →  SpatialTransformations
```

The modules are tightly sequenced. Module 02 uses tensor operations introduced in Module 01. Module 03 uses filter concepts from Module 02. Module 04 uses the edge-detection and thresholding output from Module 03 as the assumed segmentation step. Module 05 (Corner Detection & Feature Descriptors) extends classical feature extraction to interest point matching. Module 06 (Spatial Transformations) builds the mathematical foundations of coordinate transformations and image warping, serving as a direct prerequisite for Phase 02's registration module.

---

## Common Pitfalls for Beginners

- **Shape bugs are the most common bug.** Print `tensor.shape` and `tensor.dtype` at every stage of every pipeline. This single habit prevents the majority of beginner errors.
- **DICOM pixel values are raw ADC counts, not Hounsfield Units.** Always apply `slope * pixel_array + intercept` before doing any intensity-based analysis.
- **16-bit TIFFs display as all-black** with `plt.imshow` unless you normalise to [0, 1] first.
- **GLCM must be computed on a quantised image** (8–16 grey levels). Computing it on raw 16-bit data produces a 65536×65536 matrix.
- **The Canny detector in OpenCV expects uint8** with thresholds in [0, 255]. The scikit-image version expects float in [0, 1]. They are not interchangeable without rescaling.
