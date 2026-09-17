# 10 — Local Resolution Estimation in Cryo-EM 3D Maps

## Overview

A single global FSC (Fourier Shell Correlation) resolution number—the standard metric reported for a cryo-EM structure—is an average over the entire map. However, real macromolecules are rarely uniformly ordered: rigid cores reconstruct sharply, while flexible loops, termini, or ligand-binding regions are smeared out and appear at much lower local resolution.

This module explores **local resolution estimation** from first principles, comparing global Gold-Standard FSC against sliding-window 3D local FSC (the algorithmic foundation of ResMap and Blocres) and validating the recovered resolution map against known ground-truth spatially varying blur.

---

## What you will learn

- **Gold-standard FSC & the 0.143 criterion:** Splitting particle stacks into independent half-sets, calculating shell-by-shell cross-correlation, and why a single scalar obscures structural variability.
- **Spatially-varying resolution:** How conformational flexibility and non-uniform angular coverage cause local resolution heterogeneity across macromolecular domains.
- **Sliding-window local FSC:** Sub-volume window extraction, soft spherical/Gaussian masking, local 3D Fourier Shell Correlation, and building 3D per-voxel resolution maps.
- **Resolution map validation:** Measuring correlation between true spatial blur fields and recovered local resolution metrics.
- **Production methods:** Algorithmic designs and trade-offs in ResMap (local structure tensor), Blocres (block-based FSC), MonoRes (monogenic signal), cryoSPARC Local Resolution, and DeepRes.

---

## Notebooks

| File | Description |
|------|-------------|
| [`local_resolution_tutorial.ipynb`](local_resolution_tutorial.ipynb) | Hands-on tutorial computing global FSC and sliding-window local FSC on 3D half-maps with ground-truth spatially varying resolution. |

---

## Key References

- Kucukelbir, A., Sigworth, F.J. & Tagare, H.D. (2014). Quantifying the local resolution of cryo-EM density maps. *Nature Methods* 11, 63–65.
- Cardone, G., Heymann, J.B. & Steven, A.C. (2013). One number does not fit all: mapping local variations in resolution in cryo-EM reconstructions. *Journal of Structural Biology* 184(2), 226–236.
- Rosenthal, P.B. & Henderson, R. (2003). Optimal determination of particle orientation, absolute hand, and contrast loss in single-particle electron cryomicroscopy. *Journal of Molecular Biology* 333(4), 721–745.
