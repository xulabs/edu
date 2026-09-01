# 03 — Medical Generative Models: Cross-Modal Synthesis

## Overview

A patient who undergoes a CT scan did not necessarily receive an MRI. A patient scanned
on a 1.5 T MRI may not have a 3 T scan. Rare pathologies may appear in only one of the
available modalities. **Cross-modal synthesis** addresses this problem: given an image in
one modality, generate a plausible image in another modality showing the same anatomy.

This module implements cross-modal synthesis using **Schrödinger Bridge Conditional Flow
Matching (SB-CFM)** — a state-of-the-art generative modelling technique that is more
stable and sample-efficient than GANs, and more direct than standard diffusion models.

---

## What you will learn

### The clinical motivation

- Why acquiring every imaging modality for every patient is clinically and economically
  infeasible
- How synthesised modalities can replace or supplement missing acquisitions in downstream
  tasks such as segmentation, registration, and dose planning
- Why rare pathology augmentation matters: a model trained only on common presentations
  will be unreliable on the edge cases that matter most clinically

### Conditional Flow Matching (CFM)

Flow Matching (Lipman et al., 2022) trains a neural network to predict a **time-dependent
vector field** $v_\theta(x_t, t)$ that transports samples from a source distribution $p_0$
at time $t=0$ to a target distribution $p_1$ at $t=1$.

Rather than minimising the intractable marginal flow-matching objective, **Conditional Flow
Matching** conditions on individual paired data points $(x_0, x_1)$:

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\, (x_0, x_1),\, x_t} \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2$$

where $x_t = (1-t)\,x_0 + t\,x_1 + \sigma_{\min}\,\epsilon$ is a linear interpolation
with a small amount of noise. Crucially, **the training target is just $(x_1 - x_0)$** —
a plain vector pointing from the source to the target. No discriminator, no score
matching, no complex schedule: the loss is a simple regression.

At inference, you integrate the learned ODE from $t=0$ to $t=1$ starting from the source
image to produce the synthesised target.

### Why the Schrödinger Bridge?

Standard CFM learns a vector field using independent samples $(x_0, x_1)$ drawn from
$p_0$ and $p_1$ separately. The **Schrödinger Bridge** selects the *minimum-entropy
transport path* between the two distributions: it pairs $x_0$ and $x_1$ optimally (in the
entropy-regularised optimal transport sense), which leads to straighter trajectories and
therefore:
- Fewer ODE integration steps needed at inference (fewer NFE = faster synthesis)
- Better use of model capacity (not wasted on unnecessary detours)
- More stable training (the regression targets have lower variance)

The practical change is minimal: use the `SchrodingerBridgeConditionalFlowMatcher` from
`torchcfm` instead of the standard matcher. Training and inference code are identical.

### Architecture: time-conditional U-Net

The flow-matching network $v_\theta$ is built on MONAI's `BasicUNet` with two modifications:

1. **Sinusoidal time embedding** — $t \in [0,1]$ is encoded as a fixed Fourier feature
   vector and added to the bottleneck activations. This lets the network adapt its
   prediction based on where it currently sits along the transport path.
2. **Source image concatenation** — $x_0$ is concatenated channel-wise with $x_t$ as the
   input. Seeing both the starting point and the current interpolated state gives the
   network directional information about the transport.

### Training mechanics

Each training step:
1. Sample a batch of paired slices $(x_0, x_1)$ from the DataLoader
2. SB-CFM computes the interpolated point $x_t$ and the regression target $u_t$
3. Forward pass: $\hat{v} = v_\theta(x_t,\, t,\, x_0)$
4. Loss: $\mathcal{L} = \|\hat{v} - u_t\|^2$ (plain MSE)
5. AMP backward + AdamW step + cosine LR schedule

Hardware optimisations for T4:

| Technique | Purpose |
|-----------|---------|
| `torch.cuda.amp.autocast` | fp16 forward/backward — halves VRAM use |
| `GradScaler` | Prevents fp16 gradient underflow |
| Cosine annealing LR | Smooth convergence without manual schedule tuning |

### Inference: ODE integration

Once trained, synthesise a target image from any source by integrating:

$$x(0) = x_0^{\text{new}}, \qquad \frac{dx}{dt} = v_\theta(x_t, t, x_0^{\text{new}}), \qquad \hat{x}_1 = x(1)$$

The notebook implements and compares two solvers:

**Euler (1st-order, fast):**
$$x_{t+h} = x_t + h \cdot v_\theta(x_t, t, x_0)$$

**Midpoint / RK2 (2nd-order, higher quality):**
$$k_1 = v_\theta(x_t, t, x_0), \quad k_2 = v_\theta(x_t + \tfrac{h}{2}k_1, \tfrac{h}{2}+t, x_0)$$
$$x_{t+h} = x_t + h \cdot k_2$$

With SB-CFM's straighter trajectories, 10–20 Euler steps typically suffice for good quality.

### Evaluation metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **PSNR** (dB) | $10\log_{10}(\text{MAX}^2 / \text{MSE})$ | Higher = better; > 30 dB is considered good |
| **SSIM** | Structural Similarity Index | [0,1]; captures luminance, contrast, and structure |
| **MAE** | $\tfrac{1}{N}\sum|\hat{x}-x|$ | Lower = better; interpretable in pixel-value units |

SSIM is particularly important for medical images because it correlates better with
radiologist perception than pixel-wise PSNR — a high-PSNR image that is globally accurate
but misses structural boundaries may be less useful clinically than a slightly lower-PSNR
image with correct boundary sharpness.

### The rare-pathology augmentation demo

The notebook ends with a demonstration of the clinical use case that motivates the whole
module: synthesising T2-like images with lesions from T1 inputs that include the lesion.
By oversampling these synthesised examples during downstream classifier training, you can
improve performance on rare pathology classes without any additional acquisition cost.

---

## Notebooks

| File | Description |
|------|-------------|
| [`sb_cfm_medical_synthesis.ipynb`](sb_cfm_medical_synthesis.ipynb) | Full pipeline: synthetic IXI-style paired T1/T2 data generation, MONAI transform pipeline, SB-CFM training loop on 2-D slices, Euler and RK2 inference with step-count comparison, PSNR/SSIM/MAE evaluation, ODE trajectory visualisation, and rare-pathology augmentation demo |
| [`diffusion_for_medical_imaging.ipynb`](diffusion_for_medical_imaging.ipynb) | Diffusion Foundations for Medical Imaging: From DDPM forward noising to DDIM fast sampling, closed-form forward process, linear noise schedule, noise-prediction network training, and faster DDIM sampling. |

---

## Dataset: IXI as a T1→T2 proxy

The notebook uses the **IXI dataset** (Information eXtraction from Images) — ~600 paired
T1 and T2 brain MRI scans from three UK hospitals, freely available under CC-BY-SA.

**Why T1→T2 as a proxy for MRI→CT?** True paired MRI/CT datasets (e.g. Gold Atlas,
SynthRAD) require institutional data access. The T1→T2 transport task is mathematically
identical — we are still learning a transport map between two distinct image distributions
with the same anatomy — and all insights transfer directly to the MRI→CT setting.

**Why 2-D slices?** Training on full 3-D volumes requires much larger GPU memory and long
runtimes. The notebook extracts 2-D axial slices from the NIfTI volumes, which lets the
model train to convergence on a T4 GPU within a Colab session. The same architecture
extends to 3-D without structural changes.

---

## Setup

```bash
pip install "monai[all]" torchcfm nibabel tqdm gdown scikit-image \
    matplotlib einops "numpy<2.0.0"
```

The notebook's first cell handles all installations automatically, including pinning NumPy
below 2.0.0 (required for scipy compatibility).

---

## How this connects to other modules

- **Module 02 (registration)** is the primary downstream consumer of synthesis. If you
  only have T1 MRI and need to register to CT, synthesising a CT from the T1 lets you run
  standard monomodal CT↔CT registration with MSE instead of requiring multimodal MI.
- **Module 01 (segmentation)** benefits from synthesis when training labels exist for one
  modality but not another — synthesise the unlabelled modality and train the segmentor on
  the synthesised paired data.

---

## Tips for beginners

- The training target `u_t = x_1 - x_0` will look like a simple difference image when
  you first print it. That is correct — the loss is literally asking the network to predict
  the pixel-wise difference between source and target. As training progresses the network
  learns this direction at every $t$.
- If PSNR and SSIM look flat for the first few hundred steps, do not abort — the model is
  finding the large-scale structure first, then refining detail.
- The `einops.rearrange` calls in the notebook use string annotations like
  `'b c h w -> (b h) w c'` to make axis semantics explicit. Read these as a declaration of
  what each axis means before and after, not as opaque syntax.
- PSNR targets on the synthetic IXI-style dataset: > 28 dB after 50 epochs is reasonable.
  Real clinical datasets will yield lower numbers due to genuine inter-subject anatomical
  variability.
