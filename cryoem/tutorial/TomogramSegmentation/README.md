# 05 — Interactive CryoET Tomogram Segmentation

## Overview

In cryogenic electron tomography (**CryoET**), reconstructing the three-dimensional volume of biological specimens yields highly detailed tomograms. However, segmenting these structures automatically is exceptionally challenging due to extremely low signal-to-noise ratios (SNR), boundary ambiguity, and the presence of numerous crowded macromolecular structures.

This module introduces a **human-in-the-loop prompt propagation** strategy inspired by **Segment Anything (SAM)** and **CryoSAM**. Instead of manual 3D contouring or training custom segmentation networks from scratch, a researcher provides simple 2D spatial prompts (such as positive/negative points) on a single "seed" slice. The system generates an initial 2D mask, automatically converts it into a prompt for neighboring slices, and propagates the segmentation bidirectionally to reconstruct a complete 3D mask. The user remains in the loop, inspecting slices and providing manual prompt corrections where the propagation becomes unstable.

---

## What you will learn

### 1. 3D Tomogram Slice Representation & Masking
* How a 3D tomogram $V \in \mathbb{R}^{D \times H \times W}$ is represented as a stack of 2D slices.
* The difference between a 2D slice image, a 2D binary mask, and a reconstructed 3D mask.

### 2. Prompt-Guided 2D Segmentation
* Using visual prompts (coordinates representing positive points inside a target structure) to guide a foundation segmentation model.
* Generalization advantages of promptable foundation models (like SAM) on unseen bio-imaging structures.

### 3. Bidirectional Mask Propagation (CryoSAM Paradigm)
* Converting a predicted 2D mask on slice $z$ into an automated point prompt for slice $z \pm 1$.
* Formulating propagation stopping criteria based on mask area, bounding-box centroid shift, and slice overlap to handle noise and boundary fade.

### 4. Interactive Correction & 3D Morphometry
* Implementing human-in-the-loop visual corrections by placing additional positive or negative prompts on poorly-segmented slices.
* Extracting quantitative metrics from the final 3D mask, including volume (voxel count), centroid, bounding box coordinates, and slice span.

---

## Notebooks

| File | Description |
|------|-------------|
| [`interactive_cryoet_tomogram_segmentation.ipynb`](interactive_cryoet_tomogram_segmentation.ipynb) | Hands-on tutorial implementing interactive 3D tomogram segmentation using point prompts and bidirectional mask propagation. Includes synthetic phantom verification, real CryoET ribosome segmentation, interactive error correction, and 3D volume/centroid measurements. |
| [`membrane_organelle_segmentation_tutorial.ipynb`](membrane_organelle_segmentation_tutorial.ipynb) | Hands-on tutorial training 2D/3D U-Net segmentation networks to segment membranes and organelles (mitochondria) in 3D EM volumes, addressing class imbalance and evaluating prediction accuracy. |

---

## Setup

Ensure you have the required dependencies installed:

```bash
pip install numpy scipy matplotlib torch torchvision scikit-image
```

No specialized hardware is strictly required for the introductory sections, though a GPU is recommended if performing foundation model inference locally.
