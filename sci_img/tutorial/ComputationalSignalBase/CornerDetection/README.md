# 05 — Corner Detection & Feature Descriptors

## Overview

Unlike edges (1-D boundaries) or flat regions, a **corner** is a point where the image intensity changes significantly in multiple directions (2-D features). In computer vision, finding these local points of interest is crucial for tracking objects, aligning images, and matching features. Once we detect these points, we must generate a numerical representation (**feature descriptor**) that summarizes the local region and remains invariant to changes in illumination, scale, or rotation, allowing us to find matching points in other images.

This module introduces the mathematical derivation, intuition, and implementations of key classical corner/keypoint detectors, local feature descriptors, and keypoint matching:
- **Harris Corner Detector:** Uses local gradients and the structure tensor to compute corner response.
- **Shi-Tomasi (Good Features to Track):** Refines Harris by simplifying the eigenvalue criteria.
- **FAST (Features from Accelerated Segment Test):** A high-speed corner detector designed for real-time applications by comparing pixel intensity in a circular ring.
- **Feature Descriptors:** Local representations ranging from simple flattened patches and normalized patches to SIFT-like histograms of gradient orientations.
- **Descriptor Matching:** Using distance metrics (e.g. Euclidean distance, SSD) to find matching keypoints.

---

## What you will learn

### 1. Intuition: The Sliding Window Test
- **Flat region:** Shifting the window in any direction causes no intensity change.
- **Edge:** Shifting parallel to the edge causes no change; shifting perpendicular causes a large change.
- **Corner:** Shifting in *any* direction causes a large change in intensity.

### 2. The Structure Tensor ($M$)
The second-moment matrix (or structure tensor) summarizes local gradient information:

$$M = \sum \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

Corners correspond to locations where both eigenvalues ($\lambda_1, \lambda_2$) of $M$ are large.

### 3. Harris Corner Response Function
To avoid computing eigenvalues explicitly, the Harris detector uses:

$$R = \text{det}(M) - k \cdot (\text{trace}(M))^2 = (\lambda_1 \lambda_2) - k \cdot (\lambda_1 + \lambda_2)^2$$

- $R > 0$: Corner region
- $R < 0$: Edge region
- $|R|$ is small: Flat region

### 4. Shi-Tomasi (Good Features to Track)
Uses the minimum eigenvalue criterion:

$$R = \min(\lambda_1, \lambda_2)$$

If this minimum is above a threshold, the point is classified as a corner.

### 5. FAST Detector
Uses a circle of 16 pixels around a candidate pixel $p$. If a set of $N$ contiguous pixels are all brighter than $I_p + t$ or all darker than $I_p - t$, $p$ is a corner.

### 6. Local Feature Descriptors & Matching
- **Raw/Normalized Patches:** Flattening pixel values directly. Normalized patches (zero-mean unit variance) provide robustness to brightness changes.
- **SIFT-like Descriptors:** Computing gradient magnitudes and orientations, weighting by a Gaussian, and grouping them into orientation histograms to achieve robust matching.
- **Matching:** Finding correspondences between descriptor vectors using Euclidean distance metrics.

---

## Notebooks

| File | Description |
|------|-------------|
| `corner_detection_tutorial.ipynb` | Derivation of the Structure Tensor, manual implementation of Harris corner detection from scratch, and comparison with OpenCV's built-in Harris, Shi-Tomasi, and FAST detectors on synthetic and real images. |
| `feature_detectors_descriptors.ipynb` | Core concepts of local descriptors: raw patch matching, gradient computations, SIFT-like gradient histograms (magnitude & orientation), and descriptor matching under lighting changes. |

---

## Setup

```bash
pip install opencv-python numpy matplotlib scikit-image scipy scikit-learn
```

No GPU is required.
