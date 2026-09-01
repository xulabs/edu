# 06 — Spatial Transformations & Image Warping

## Overview

In computer vision and medical image analysis, we often need to transform coordinates of images. For example, aligning a pre-operative MRI scan to a post-operative MRI scan requires matching their spatial frames of reference. This mapping of pixel coordinates is called a **spatial transformation**.

This module introduces the mathematical foundation, equations, and code implementation of 2D coordinate transformations and image warping. It follows a concept-first style:
- **Filtering vs. Warping:** Filtering changes pixel values (range); warping changes pixel coordinate mapping (domain).
- **Linear Transformations:** Matrix multiplications that scale, shear, or rotate an image.
- **Homogeneous Coordinates:** A trick to combine rotation and translation (affine mapping) or perspective projections (projective mapping/homography) into single matrix multiplications.
- **Image Warping & Resampling:** Forward mapping vs. inverse mapping, and using bilinear interpolation to compute pixel intensities at non-integer coordinates.

---

## What you will learn

### 1. Images as Functions
An image is a function $f(x, y)$ that maps coordinates $(x, y) \in \mathbb{R}^2$ to an intensity value.
- **Image Filtering:** Modifies the output range: $g(x, y) = h(f(x, y))$.
- **Image Warping:** Modifies the input domain: $g(x, y) = f(x', y') = f(T(x, y))$.

### 2. 2D Linear Transformations
Points are mapped via matrix multiplication:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = A \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$

- **Scaling:** $A = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$
- **Rotation:** $A = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$
- **Shearing:** $A = \begin{bmatrix} 1 & h_x \\ h_y & 1 \end{bmatrix}$

Linear transformations always keep the origin $(0, 0)$ fixed.

### 3. Homogeneous Coordinates & Affine Transforms
To represent translation (adding an offset) as a matrix multiplication, we scale coordinates to 3D homogeneous space $[x, y, 1]^T$:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

This is an **Affine Transformation**. It preserves parallelism of lines but not necessarily lengths or angles.

### 4. Projective Transformations (Homography)
If the bottom row of the matrix is non-zero, it represents a **Homography**:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ g & h & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

This allows representing perspective warping (where parallel lines intersect at a vanishing point).

### 5. Forward vs. Inverse Warping
- **Forward Mapping:** Sends source pixel $(x, y)$ to $(x', y')$. If target coordinates land between pixels, this creates holes/aliasing in the target image.
- **Inverse Mapping (Preferred):** Loops over target pixels $(x', y')$, applies the inverse transform $T^{-1}(x', y')$ to find the corresponding source coordinate $(x, y)$, and samples the intensity there.

### 6. Bilinear Interpolation
Since the calculated source coordinate $(x, y)$ is rarely an integer, we interpolate its intensity from the four nearest neighboring pixels:

$$f(x, y) \approx (1-a)(1-b)f_{00} + a(1-b)f_{10} + (1-a)bf_{01} + abf_{11}$$

where $a$ and $b$ are the fractional parts of $x$ and $y$.

---

## Notebooks

| File | Description |
|------|-------------|
| `2d_transformations.ipynb` | A concept-first guide illustrating synthetic image generation, linear transforms, homogeneous translations, affine/projective warps, inverse mapping implementation, and bilinear interpolation from scratch. |
| `image_homographies.ipynb` | A detailed self-learning chapter covering image homographies, coordinate transformations using homogeneous coordinates, image warping, Direct Linear Transform (DLT), normalized DLT, and robust estimation using RANSAC. |

---

## Setup

```bash
pip install numpy matplotlib
```

No GPU or external libraries are required.
