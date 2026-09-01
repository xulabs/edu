# 02 — Frequency Filtering & Image Pyramids

## Overview

Every image can be decomposed into a sum of waves at different spatial frequencies — coarse
structure at low frequencies, fine detail and noise at high frequencies. This insight,
formalised by the Fourier Transform, is the theoretical foundation of virtually every
signal-processing step in medical and biological imaging.

This module makes the Fourier decomposition concrete through hands-on filtering. By the end
you will know how to suppress noise, enhance edges, and analyse images at multiple scales
— both in the familiar spatial domain (kernel convolution) and in the more powerful
frequency domain (Fourier filtering). These techniques appear unchanged in Phase 04's CTF
correction and bandpass preprocessing for cryoEM.

---

## What you will learn

### 2-D convolution: the unifying operation

**Convolution** slides a small kernel $h$ over an image $f$ and computes a weighted sum at
every position:

$$(f * h)[x, y] = \sum_{i,j} f[x+i,\, y+j]\; h[i,j]$$

Everything in this module — blur, sharpen, edge-detect, denoise — is a different choice of
kernel $h$. Understanding this single equation makes all filter types variants of the same
idea:

| Kernel | Effect |
|--------|--------|
| Constant (box kernel) | Mean blur — suppresses noise uniformly |
| Gaussian | Smooth blur — suppresses noise with edge preservation |
| Laplacian | Highlights intensity changes — sharpens or detects edges |
| Sobel X | Approximates horizontal derivative — detects vertical edges |
| Identity - Gaussian | Unsharp mask — sharpening |

**Key parameters** that also appear in convolutional neural network layers:
- **Kernel size** — larger kernels average over more pixels (more smoothing, more computation)
- **Padding** — how to handle pixels at the image border (`same` keeps output size equal to input)
- **Stride** — how many pixels to skip between applications (stride > 1 downsamples the output)

### Spatial-domain filters

**Box filter (mean blur):** replaces each pixel with the average of its neighbourhood.
Fast to compute (separable into row and column 1-D sums), but blurs edges equally with
noise — rarely the right choice for scientific images.

**Gaussian blur:** weights the neighbourhood by a 2-D Gaussian function centred on the
target pixel:

$$G_\sigma(x, y) = \frac{1}{2\pi\sigma^2} e^{-(x^2+y^2)/2\sigma^2}$$

The parameter $\sigma$ controls the blur radius. Increasing $\sigma$ removes higher
frequencies. Gaussian blur is the standard pre-processing step before edge detection
because it suppresses high-frequency noise that would otherwise generate spurious edges.

**Median filter:** replaces each pixel with the **median** (not mean) of the neighbourhood.
This is a non-linear filter — it cannot be written as a convolution kernel — which makes it
uniquely effective against impulsive noise (salt-and-pepper, detector hot pixels), where a
single bright outlier would dominate a mean but is simply sorted away by the median. Widely
used in fluorescence microscopy to remove shot-noise spikes.

**Laplacian sharpening:** the Laplacian $\nabla^2 f$ is a discrete second derivative that
is large wherever intensity changes rapidly. Adding the Laplacian back to the original
image amplifies fine detail:

$$f_{\text{sharp}} = f - \alpha \nabla^2 f$$

Unsharp masking is a related technique: subtract a blurred version of the image from itself
to isolate and amplify high-frequency content.

### Frequency-domain thinking

The **2-D Discrete Fourier Transform (DFT)** converts an image from the spatial domain
(pixel values at locations) into the frequency domain (complex amplitudes at spatial
frequencies):

$$F(u, v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x,y)\; e^{-j2\pi(ux/M + vy/N)}$$

**How to read a Fourier spectrum:**
- The **DC component** at the centre (after `fftshift`) represents the image mean
- Low-frequency components near the centre encode coarse structure (large blobs, gradients)
- High-frequency components far from the centre encode fine detail, edges, and noise
- Bright rings at a particular radius indicate strong periodic structure at that scale

**The Convolution Theorem** is why frequency-domain filtering matters in practice:

$$\mathcal{F}\{f * h\} = F \cdot H$$

Convolution in the spatial domain equals pointwise multiplication in the frequency domain.
For a large kernel or image, computing the FFT, multiplying, and inverting with IFFT is
dramatically faster than direct convolution ($O(N^2 \log N)$ vs $O(N^2 k^2)$ for a
$k \times k$ kernel).

### Frequency-domain filters

**Low-pass filter:** multiplies the Fourier transform by a mask that is 1 inside a radius
$r$ and 0 outside. This removes high-frequency noise but blurs fine structural details.

**High-pass filter:** $1 - \text{low-pass mask}$. Removes the DC component and low-frequency
gradients; isolates edges and fine texture. In microscopy, this corrects for uneven
illumination (intensity gradients across the field of view).

**Band-pass filter:** product of a high-pass and a low-pass mask, passing only the
frequency band between $r_{\min}$ and $r_{\max}$. This is the standard pre-processing
filter in cryoEM pipelines: the lower cutoff removes the ice background gradient; the upper
cutoff removes detector read-out noise.

**Butterworth filter:** a smooth frequency-domain filter without the sharp transition
(Gibbs ringing) of ideal low/high-pass masks:

$$H(r) = \frac{1}{1 + (r / r_0)^{2n}}$$

Higher order $n$ gives a steeper roll-off. Butterworth filters are widely used when ringing
artefacts from hard spectral edges are unacceptable.

### Image pyramids

An **image pyramid** is a multi-scale representation — a sequence of versions of the same
image at progressively lower resolutions.

**Gaussian pyramid** — built by alternating Gaussian blur and 2× downsampling:

```
Level 0: Original (H × W)
Level 1: Blur → downsample (H/2 × W/2)
Level 2: Blur → downsample (H/4 × W/4)
...
```

Applications: coarse-to-fine registration (Phase 03 Module 02 runs registration at
multiple pyramid levels), template matching at different scales, image blending.

**Laplacian pyramid** — each level is the difference between two adjacent Gaussian pyramid
levels, containing only the band-pass detail between those scales:

$$L_k = G_k - \text{upsample}(G_{k+1})$$

The original image can be perfectly reconstructed by summing back up the Laplacian pyramid.
This property makes Laplacian pyramids ideal for seamless image blending: blend each
frequency band separately at the splice boundary.

---

## Notebooks

| File | Description |
|------|-------------|
| `image_filtering.ipynb` | Comprehensive walkthrough: box filter, Gaussian blur, median filter, Laplacian sharpening — each shown spatially and in the Fourier power spectrum, with side-by-side visual comparisons on both a natural photo and a fluorescence microscopy image |
| `image_filtering_notebook.ipynb` | Companion exercises: parameter sweeps over kernel size and σ, noise addition and denoising benchmarks (PSNR), Butterworth filter design |
| `image_pyramids_and_frequency_domain.ipynb` | Gaussian and Laplacian pyramid construction; perfect reconstruction verification; Laplacian-pyramid image blending; frequency-domain visualisation of each pyramid level |

---

## Practical context

Frequency filtering is the invisible first step in virtually every medical imaging pipeline:

- **CryoEM** (Phase 04): bandpass filtering removes ice gradients and detector noise before
  particle picking; CTF correction is Wiener filtering in the frequency domain
- **MRI reconstruction**: k-space is the raw frequency-domain data from the scanner;
  images are the inverse Fourier transform
- **Fluorescence microscopy**: high-pass filtering corrects uneven illumination (flat-field
  correction); deconvolution (Wiener-like inverse filtering) sharpens images blurred by
  the optical PSF

---

## Setup

```bash
pip install numpy scipy scikit-image matplotlib opencv-python
```

No GPU is required. FFT computation on CPU is fast even for 2048 × 2048 images.

---

## Common errors and how to fix them

| Error | Likely cause | Fix |
|-------|-------------|-----|
| Fourier spectrum looks wrong / all noise | Image not cast to float before FFT | `img = img.astype(np.float32)` before `np.fft.fft2()` |
| Ringing artefacts around edges after ideal low-pass | Hard spectral cutoff causes Gibbs phenomenon | Switch to a Butterworth filter with gradual roll-off |
| Pyramid reconstruction does not match original | Incorrect upsampling or level ordering | Verify each `upsample → add` step against the Laplacian at that level |
| Filtered image appears too dark | Forgetting to clip values after filtering | `np.clip(result, 0, 1)` after all filtering operations |

---

## Tips for beginners

- Before writing any filter code, first visualise the **power spectrum** of your image
  with `np.fft.fftshift(np.abs(np.fft.fft2(img)) ** 2)`. Identify where the signal and
  noise live in frequency space — this makes the filter design obvious.
- Apply `np.fft.fftshift` to centre the zero-frequency component before displaying the
  spectrum. The unshifted spectrum has DC in the corner, which looks confusing.
- The Gaussian blur kernel and the Gaussian frequency-domain envelope are a Fourier pair:
  a wide Gaussian in space corresponds to a narrow Gaussian in frequency (small blur radius
  cuts only high frequencies). This reciprocal relationship reappears throughout cryoEM.
- When comparing filters, always show results on a **noisy** image, not a clean one.
  Filters that look identical on clean data behave very differently under noise.
