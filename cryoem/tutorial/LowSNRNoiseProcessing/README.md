# 01 — Low-SNR Image Processing for CryoEM

## Overview

This module is an introduction to the signal processing challenges that define cryoEM. By
the end you will have simulated a realistic cryoEM micrograph from scratch, applied and
compared every standard denoising method, picked particles automatically, computed 2-D
class averages, and measured resolution with Fourier Ring Correlation. All computation is
CPU-based NumPy/SciPy — no GPU is needed.

---

## What you will learn

### Why cryoEM is a low-SNR problem

Electron beams ionise biological molecules. To prevent radiation damage, cryoEM experiments
must keep the total electron dose extremely low — typically 20–60 e⁻/Å². This means:

- **Shot noise (Poisson)** — electron counting is discrete; at low doses the Poisson
  fluctuations are large relative to the signal
- **Read-out noise (Gaussian)** — detector electronics add additive Gaussian noise; direct
  electron detectors have dramatically lower read-out noise than earlier scintillator cameras
- **Structural noise** — the amorphous vitreous ice surrounding the molecule scatters
  electrons unpredictably, contributing a spatially correlated background
- **CTF modulation** — the imaging optics impose a frequency-dependent phase contrast
  function that smears and inverts contrast at specific spatial frequencies

The combined result: typical micrograph SNR of **0.01–0.1**. The signal is buried 10–100×
below noise. No single image of a single molecule is interpretable.

### The fundamental SNR trick: averaging

If you extract $N$ images of identical copies of the same molecule (in the same
orientation) and average them, random noise cancels while signal adds coherently:

$$\overline{I}(x) = \frac{1}{N}\sum_{i=1}^{N}(S(x) + \eta_i(x)) = S(x) + \frac{1}{N}\sum_{i=1}^{N}\eta_i(x)$$

The averaged noise term has variance $\sigma^2/N$, so:

$$\text{SNR}_{\text{avg}} = N \times \text{SNR}_{\text{single}}$$

This √N SNR improvement is the entire foundation of cryoEM. Modern datasets collect
**hundreds of thousands to millions** of particles to reach the SNR needed for
sub-ångström resolution. The notebook lets you watch this improvement happen live.

### The Contrast Transfer Function (CTF)

The electron microscope's imaging optics impose a phase contrast function on every
micrograph. The CTF oscillates as a function of spatial frequency $s$:

$$\text{CTF}(s) = -\sin\!\left(\pi \lambda \Delta z \, s^2 - \tfrac{\pi}{2}\lambda^3 C_s \, s^4\right)$$

| Parameter | Symbol | Typical value |
|-----------|--------|---------------|
| Electron wavelength | $\lambda$ | 0.0197 Å (300 kV) |
| Defocus | $\Delta z$ | −1 to −4 μm |
| Spherical aberration | $C_s$ | 2.7 mm |

Key properties of the CTF:
- Oscillates through **zero** at specific frequencies called Thon rings — information at
  those frequencies is completely destroyed in a single image
- The first zero moves to lower frequency as defocus increases (easier to see, worse for
  high-resolution information)
- Positive CTF means the phase of that frequency band is correct; negative CTF means
  phase is inverted — reconstruction without CTF correction would reinforce and cancel
  information inconsistently across particles acquired at different defoci

### CTF estimation and correction

**Phase-flip correction** — multiply the Fourier transform of each image by
$\text{sign}(\text{CTF}(s))$. This restores phase coherence (so particles at different
defoci can be averaged) but does not restore the amplitude at CTF zeros — information
there is permanently lost.

**Wiener CTF correction** — the optimal linear filter:
$$\hat{F}(s) = \frac{\text{CTF}(s)}{\text{CTF}^2(s) + \varepsilon} \cdot G(s)$$

where $\varepsilon$ is a noise regularisation constant (set to the estimated SNR reciprocal).
Wiener filtering restores amplitude at non-zero frequencies while suppressing noise
amplification near CTF zeros.

In production pipelines (RELION, cryoSPARC), CTF parameters are estimated from the power
spectrum of each micrograph by fitting the Thon ring pattern.

### Classical denoising methods

| Method | Domain | Mechanism | CryoEM relevance |
|--------|--------|-----------|-----------------|
| Gaussian smoothing | Spatial | Convolves with a Gaussian kernel | Removes high-frequency noise but blurs edges; used as a quick pre-processing step |
| Median filter | Spatial | Replaces each pixel with the local median | Non-linear, robust to impulsive (salt-and-pepper) noise; preserves edges better than Gaussian |
| Wiener filter | Frequency | Optimal MMSE filter given signal and noise PSDs | Theoretically optimal under Gaussian noise; requires noise PSD estimate |
| Bandpass filter | Fourier | Butterworth high-pass × low-pass | The standard pre-processing step in most cryoEM pipelines; removes ice gradient (low-frequency) and detector noise (high-frequency) |

The bandpass filter is the standard choice because it cleanly separates the three frequency
regimes: background gradients (< 1/200 Å⁻¹), particle signal (~1/50 to 1/5 Å⁻¹), and
detector noise (> 1/5 Å⁻¹ depending on instrument).

### Particle picking via normalised cross-correlation (NCC)

Given a reference template (a known or estimated 2-D projection of the molecule),
normalised cross-correlation locates all matching regions in the micrograph:

$$\text{NCC}(x, y) = \frac{\langle I(x+\delta x, y+\delta y),\, T(\delta x, \delta y) \rangle}{\|I\| \cdot \|T\|}$$

computed efficiently in Fourier space as:
$$\mathcal{F}^{-1}\!\left[\hat{I}(\mathbf{k}) \cdot \hat{T}^*(\mathbf{k})\right]$$

Local maxima in the NCC map above a threshold give particle coordinates. The threshold
controls the trade-off between recall (finding all particles) and precision (avoiding
false picks).

### 2-D class averaging

Particle picking produces a set of small image patches — "particle boxes". Before
reconstructing a 3-D structure, these are sorted into groups of similar orientations
(2-D classes) and averaged within each class:

1. **Extract** square boxes centred on each picked coordinate
2. **Align** each particle to a reference by maximising NCC over in-plane rotations and translations
3. **Cluster** aligned particles with K-means (or EM in production software)
4. **Average** within each cluster to produce a 2-D class average

The SNR of the class average is $\sqrt{N} \times$ the single-particle SNR. With $N = 1000$
particles per class, SNR improves by a factor of ~32.

In production pipelines (RELION, cryoSPARC), 2-D classification additionally serves as a
**data quality filter**: ice contamination, carbon-edge false picks, and aggregated material
form diffuse, featureless 2-D classes that are discarded before 3-D reconstruction.

### Fourier Ring Correlation (FRC) and resolution

FRC is the gold-standard method for measuring 2-D image resolution:

$$\text{FRC}(r) = \frac{\sum_{\mathbf{k}\in r} F_1(\mathbf{k})\, F_2^*(\mathbf{k})}{\sqrt{\sum_{\mathbf{k}\in r}|F_1(\mathbf{k})|^2 \cdot \sum_{\mathbf{k}\in r}|F_2(\mathbf{k})|^2}}$$

Split the particle stack into two halves (odd/even indices), compute the average from each
half independently, then compute their Fourier correlation as a function of spatial
frequency ring $r$. The spatial frequency at which the correlation drops below **0.143**
(the half-bit criterion) defines the resolution.

**Critical insight:** a class average that looks visually smooth and beautiful to the human
eye may fail FRC at 8 Å — worse than a grainier-looking average that passes at 4 Å. FRC
exists precisely because visual smoothness is a misleading proxy for true resolution.

### Noise2Void: self-supervised denoising

Classical denoising and Wiener filtering require an estimate of the noise power spectrum.
Deep-learning denoisers like DnCNN require clean reference images for supervised training.
Neither is available in cryoEM — we never have access to a noise-free ground truth.

**Noise2Void** sidesteps this by training a neural network to predict each pixel from its
**neighbours only** (the "blind-spot" constraint):

1. Mask 2–5% of pixels per training patch
2. Train the network to predict each masked pixel from a receptive field that excludes that pixel
3. Because noise is spatially independent (i.i.d.), neighbours carry zero information about
   the masked pixel's noise realisation
4. The network is forced to predict the underlying signal structure from context

Production cryoEM tools built on this principle: `cryoCARE` (pair-based), `Topaz-Denoise`
(semi-supervised), `noise2void` (pip installable).

---

## Notebooks

| File | Description |
|------|-------------|
| `cryoem_low_snr_tutorial.ipynb` | SNR vs averaging demo; CTF simulation (phantom → CTF → noise); denoising comparison (Gaussian, Median, Wiener, Bandpass); CTF phase-flip and Wiener correction; template-matching particle picking; K-means + NCC 2-D class averaging; FRC resolution curve; Noise2Void concept; full PSNR/SSIM benchmark |
| `self_supervised_denoising_medical.ipynb` | Hands-on tutorial on self-supervised denoising (Noise2Self J-invariant masking, Noise2Void structured masking, Probabilistic Noise2Void for uncertainty estimation), noise models in medical imaging, and FRC evaluation. |
| `lowdose_em_denoising_tutorial.ipynb` | Self-supervised blind-spot denoising (Noise2Void family) trained directly on low-dose noisy EM images using ISBI 2012 EM Segmentation Challenge data. Covers Poisson-Gaussian noise modeling, masked training, and PSNR/SSIM evaluation. |
| `particle_picking_tutorial.ipynb` | Weakly-supervised / positive-unlabeled learning for particle picking in cryo-EM micrographs (apoferritin EMPIAR-10146 dataset). Implements classical template/blob picking, a custom Patch-CNN classifier trained on bootstrap labels, and sliding-window peak detection. |


---

## Setup

```bash
pip install numpy scipy matplotlib scikit-image tqdm
```

No GPU is required. All computation uses NumPy/SciPy on CPU.

---

## Tips for beginners

- Run the **SNR vs averaging** section first. Watching SNR improve from −10 dB to +10 dB
  as $N$ increases from 1 to 1000 makes everything else in this module concrete.
- When looking at CTF plots: the oscillation frequency increases with higher defocus.
  Higher defocus makes the CTF effect more visible but pushes the first zero to lower
  spatial frequency — destroying coarser structural information first.
- The benchmark at the end reports PSNR for all methods. Do not use it in isolation —
  also study the visual comparison panels. A high-PSNR Gaussian-smoothed result looks
  blurry; a lower-PSNR bandpass result may preserve structural features better.
- The Noise2Void section is conceptual — it implements the masking idea using a small
  CNN trained in-notebook. Real cryoCARE uses a much more sophisticated architecture.
