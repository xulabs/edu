# 04 — Sub-tomogram Averaging & 3D Particle Picking for Cryo-ET

## Overview

Reconstructing a 3-D tomogram is only the beginning. A tomogram is a noisy, missing-wedge-degraded 3-D image containing many copies of biological macromolecules. To resolve high-resolution structure, we use **Sub-tomogram Averaging (STA)**: finding copies of the particle, aligning them in 3D, and averaging them to cancel noise.

This module walks through the mathematical and geometric principles of sub-tomogram averaging, simulated particle generation, 3-D template matching, alignment under missing-wedge distortion, and validation with Fourier Shell Correlation (FSC).

---

## What you will learn

### The Missing Wedge in 3-D
Biological specimens cannot be rotated past ±70° in the microscope column, leaving a wedge of unsampled frequencies in 3-D Fourier space. This missing wedge causes characteristic Z-elongation artifacts that complicate alignment and averaging.

### 3-D Template Matching
To identify particles in the tomogram, we implement normalized cross-correlation (NCC) in 3D Fourier space. We search over 3D rotations to locate randomly oriented molecules.

### Sub-volume Alignment & Averaging
We align the extracted 3-D sub-volumes to a common reference. The averaging process cancels noise and helps fill in missing-wedge frequencies when particles are in random orientations.

### Resolution & Conformational Heterogeneity
We compute 3-D **Fourier Shell Correlation (FSC)** to estimate resolution. We also implement multi-reference K-means classification to distinguish between different conformational states (e.g., open vs. closed gating states).

---

## Notebooks

| File | Description |
|------|-------------|
| `subtomogram_averaging_tutorial.ipynb` | Geometry and simulation of missing-wedge distortion; 3-D template matching (NCC); orientation search and sub-volume alignment; 3-D Fourier Shell Correlation (FSC) resolution assessment; multi-reference K-means classification of conformational states. |

---

## Setup

```bash
pip install numpy scipy matplotlib scikit-image scikit-learn tqdm
```

All computations use standard scientific libraries on CPU.

---

## Further Reading

- Bharat, T.A.M. & Scheres, S.H.W. (2016). "Resolving macromolecular structures from electron cryo-tomography data using subtomogram averaging in RELION." *Nature Protocols* 11, 2054–2065.
- Briggs, J.A. (2013). "Structural biology in situ--the potential of subtomogram averaging." *Current Opinion in Structural Biology* 23, 261-267.
