# 04 — Image Alignment & Registration

## Overview

Images of the same structure are rarely in perfect correspondence. A patient's MRI from
six months ago and today were acquired at different positions in the scanner. A
histological section and the MRI slice it came from are in entirely different coordinate
spaces. Two fluorescence channels imaged sequentially on the same microscope may be offset
by a few pixels due to chromatic aberration.

**Image registration** finds the geometric transformation that maps one image into
alignment with another. This module covers two approaches: classical iterative
optimisation with SimpleITK, which requires no training data but is slow and limited to
parametric transforms; and image-to-image translation with a Pix2Pix GAN, which learns to
transform the *appearance* of one imaging modality into another, handling acquisition
artefacts that rigid registration cannot correct.

---

## What you will learn

### The registration problem

Given a **fixed image** $F$ (the reference) and a **moving image** $M$ (to be aligned),
registration finds a transform $T$ such that:

$$T^* = \arg\min_T \;\mathcal{L}_{\text{sim}}(F,\, T(M)) + \lambda \, \mathcal{R}(T)$$

where $\mathcal{L}_{\text{sim}}$ measures image dissimilarity and $\mathcal{R}(T)$ is a
regularisation term that constrains the transform to be physically reasonable.

**Three levels of transform complexity:**

| Transform | Degrees of freedom | Corrects for | Cannot correct |
|-----------|-------------------|-------------|----------------|
| Rigid | 3 (2-D) / 6 (3-D) | Translation, rotation | Scale, shear, deformation |
| Affine | 6 (2-D) / 12 (3-D) | Translation, rotation, scale, shear | Nonlinear deformation |
| Deformable | Thousands | Any smooth deformation | Topological changes |

The right transform family depends on what caused the misalignment. Misaligned CT scans
from the same session: rigid. Scans from different sessions where the patient lost weight:
affine or deformable. Aligning tissue sections to MRI: deformable.

### Similarity metrics: why MSE fails for multimodal registration

In same-modality registration (CT-to-CT, T1-to-T1), intensities are directly comparable:
aligned images have similar values at corresponding locations, so minimising MSE works as
a proxy for anatomical alignment.

In multimodal registration this breaks down. White matter appears bright in T1 MRI but
dark in T2 MRI. The same anatomy produces opposite intensities — MSE reports maximum
dissimilarity for a perfectly aligned T1/T2 pair.

**Mutual Information (MI)** solves this by measuring the statistical dependency between
the two images' intensity distributions, rather than their values directly:

$$\text{MI}(F, M) = \sum_{f,m} p_{FM}(f,m) \log \frac{p_{FM}(f,m)}{p_F(f)\,p_M(m)}$$

When images are correctly aligned, corresponding anatomy produces consistent joint intensity
pairs — the joint histogram has tight, well-defined clusters (high MI). When misaligned,
the clusters smear across the histogram (low MI). MI does not assume any particular
intensity relationship between modalities; it is maximised regardless of the specific
T1/T2 or CT/MRI contrast.

**Mattes Mutual Information** is a computationally efficient approximation that estimates
the joint histogram using Parzen windowing (kernel density estimation) rather than hard
binning. It is the default metric in SimpleITK for multimodal registration.

**Normalised Mutual Information** (NMI = $(H(F) + H(M)) / H(F,M)$) is more robust when
the field of view partially differs between modalities — use NMI instead of MI when the
two images do not fully overlap.

### Classical registration pipeline: SimpleITK

SimpleITK wraps the ITK registration framework. Every registration job configures three
independent components:

```python
R = sitk.ImageRegistrationMethod()

# 1. Transform family
R.SetInitialTransform(sitk.CentredTransformInitializer(
    fixed, moving, sitk.Euler2DTransform()))

# 2. Similarity metric
R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
R.SetMetricSamplingStrategy(R.RANDOM)
R.SetMetricSamplingPercentage(0.01)

# 3. Optimizer
R.SetOptimizerAsGradientDescent(
    learningRate=1.0, numberOfIterations=200,
    estimateLearningRate=R.EachIteration)

# Multi-resolution pyramid
R.SetShrinkFactorsPerLevel([8, 4, 2, 1])
R.SetSmoothingSigmasPerLevel([4, 2, 1, 0])
R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
```

**The multi-resolution pyramid is mandatory.** Starting registration at full resolution
means the optimiser sees only a very local view of the loss landscape and gets trapped in
local minima. Starting at ×8 downsampled resolution gives a wide capture range in a smooth
landscape; each level refines the estimate of the previous level.

**Evaluating registration:** after applying the transform to the moving image, compute MI
(should increase compared to before registration), visual overlay of structural boundaries,
and — if landmark annotations exist — Target Registration Error (TRE, the mean distance
between corresponding anatomical points after registration).

### Image-to-image translation: Pix2Pix GAN

Sometimes the goal is not geometric alignment but **appearance transformation**: converting
an out-of-focus microscopy image into an in-focus one, a low-SNR image into a denoised
version, or a brightfield image into a virtual fluorescence image. These problems have the
wrong structure for registration (there is no geometric misalignment) but are natural
image-to-image translation tasks.

**Pix2Pix** (Isola et al., 2017) learns this mapping from paired training examples using a
conditional GAN framework:

```
Generator (U-Net):
  Input image x ──► U-Net ──► Generated image Ĝ(x)

Discriminator (PatchGAN):
  {x, y}   (real pair) ──► PatchGAN ──► real
  {x, Ĝ(x)} (fake pair) ──► PatchGAN ──► fake
```

**Generator — U-Net:** the same encoder–decoder architecture from Module 03 is used as the
generator. Skip connections preserve fine spatial details from input to output — essential
for focus correction where the object locations are correct but the blur pattern must be
removed.

**Discriminator — PatchGAN:** instead of classifying the entire image as real or fake, the
PatchGAN classifies overlapping 70×70 pixel patches. The final discriminator output is an
N×N grid of real/fake scores, one per patch. This forces the generator to produce locally
realistic texture — not just globally correct mean intensity — and has a smaller
parameter count than a full-image discriminator.

**Loss function:**

$$\mathcal{L}_{\text{Pix2Pix}} = \underbrace{\mathcal{L}_{\text{GAN}}(G, D)}_{\text{adversarial}} + \lambda \underbrace{\mathbb{E}[\|y - G(x)\|_1]}_{\text{L1 reconstruction}}$$

The L1 reconstruction loss prevents the generator from ignoring the input entirely (mode
collapse). The adversarial loss prevents the L1 loss from producing blurry, averaged
outputs. The λ weighting (default: 100) controls the balance; higher λ gives more
conservative, less sharp outputs.

**Paired vs. unpaired training:** Pix2Pix requires paired training data (an in-focus and
out-of-focus version of the same field of view). When pairs are unavailable, CycleGAN
(beyond this module) learns the translation from unpaired images.

### TIFF image loading for microscopy data

Microscopy images stored as TIFF are often 16-bit, meaning pixel values range from 0 to
65535 rather than 0 to 255. Neural networks expect float32 values in [0, 1]:

```python
import tifffile
img = tifffile.imread("slide.tif").astype(np.float32)
img = (img - img.min()) / (img.max() - img.min() + 1e-8)
```

For multi-channel TIFFs (e.g. separate colour channels stored in a single file), the
channel axis may be first, last, or absent depending on the microscope software. Always
check `img.shape` and the metadata before assuming a channel layout.

---

## Notebooks

| File | Description |
|------|-------------|
| `image_registration.ipynb` | Synthetic two-modality phantom construction; MI vs MSE comparison as a function of ground-truth misalignment angle; SimpleITK rigid registration (gradient descent, Mattes MI, multi-resolution pyramid); affine registration as extension; transform parameter inspection; visual overlay evaluation; TRE computation on synthetic landmarks |
| `image_translation.ipynb` | Out-of-focus / in-focus TIFF dataset construction; 16-bit normalisation pipeline; Pix2Pix architecture (U-Net generator + PatchGAN discriminator); GAN + L1 combined loss training; training curve and generated-image visualisation at checkpoints; held-out evaluation with PSNR and SSIM |

---

## Practical context

Registration is a prerequisite for any analysis that compares two images pixel-by-pixel:

- **Longitudinal tumour tracking:** registering serial CT scans to measure volume change
  without confounding by patient positioning differences
- **Multi-stain histology:** aligning H&E and IHC sections from the same tissue block so
  that cell type (H&E) and protein expression (IHC) can be correlated
- **CryoET subtomogram alignment:** aligning extracted particle volumes to a common
  orientation before averaging — the same problem structure as 2-D particle alignment in
  Phase 04

Image-to-image translation is used for:
- **Virtual staining:** generating fluorescence images from brightfield without actual
  staining, saving reagent cost and time
- **Focus correction:** restoring depth-of-field in widefield microscopy
- **Artefact removal:** removing out-of-focus background light (haze) from confocal images

---

## Setup

```bash
pip install SimpleITK torch torchvision tifffile matplotlib numpy
```

---

## Hardware

| Task | GPU needed | VRAM usage |
|------|-----------|-----------|
| SimpleITK registration | No | CPU only |
| Pix2Pix training | Yes | ~8 GB (U-Net + PatchGAN) |
| Pix2Pix inference | Optional | ~3 GB |

---

## Common errors and how to fix them

| Error | Likely cause | Fix |
|-------|-------------|-----|
| SimpleITK registration diverges | Learning rate too high | Reduce `learningRate`; enable `estimateLearningRate=R.EachIteration` |
| MI metric barely improves | Images not overlapping at start | Apply centroid-based initialisation with `CentredTransformInitializer` |
| Pix2Pix outputs are blurry | λ (L1 weight) too high; adversarial loss not contributing | Reduce λ from 100 to 10; verify discriminator is training (its loss should not collapse to 0) |
| Discriminator loss collapses to 0 immediately | Generator too weak; training too many D steps | Use 1:1 G:D update ratio; check generator architecture has skip connections |
| TIFF loaded with wrong shape | Multi-channel TIFF with channel-first or channel-last axis | Print `img.shape` and `img.dtype` immediately after loading; transpose if needed |

---

## Tips for beginners

- For SimpleITK, always set up a `Command` callback that prints the metric value at each
  iteration. Watching MI increase over iterations makes it concrete which registration
  directions are working and how fast convergence is happening.
- The multi-resolution pyramid is the single most impactful setting for registration
  quality. Do not skip it — a single-resolution registration on a realistic medical image
  will almost always end up in a local minimum.
- For Pix2Pix, train for at least 50 epochs before judging quality. GAN training is
  unstable in the first 10–20 epochs; the loss may oscillate wildly before settling.
  Monitor the generated images visually, not just the loss values.
- If the Pix2Pix generator produces mode-collapsed outputs (all generated images look the
  same), add more training data or reduce the discriminator's effective power by lowering
  its learning rate relative to the generator's.
