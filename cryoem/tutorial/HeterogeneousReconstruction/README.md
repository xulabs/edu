# 11 — Heterogeneous Structure Reconstruction with Neural Networks (CryoDRGN)

## Overview

Classical cryo-EM single-particle analysis assumes a homogeneous, rigid macromolecule and averages thousands of aligned particle projections into a single consensus 3D density. However, biological machines perform their function through continuous conformational changes, flexible domain motions, and multi-state compositional equilibria. Averaging heterogeneous particles produces severe blurring precisely where conformational dynamics occur.

This module implements **heterogeneous cryo-EM reconstruction using neural networks (CryoDRGN)** from first principles. You will build a generative forward model in Fourier/Hartley space, train a Variational Autoencoder (VAE) coupled with a coordinate-based Multi-Layer Perceptron (implicit neural representation), map the continuous conformational landscape without discrete classification, and decode 3D volumes across the learned latent manifold.

---

## What you will learn

- **Limits of homogeneous reconstruction:** Why rigid averaging smears flexible domains and loses intermediate conformational states.
- **Forward imaging model in Hartley space:** Projection-slice theorem using real-valued Hartley representations, central-slice frequency sampling, and Contrast Transfer Function (CTF) modulation.
- **CryoDRGN VAE architecture:** Encoder mapping Hartley particle images to low-dimensional continuous latent space $\mathbf{z} \in \mathbb{R}^d$, paired with a coordinate-based MLP decoder evaluating density at arbitrary spatial frequencies.
- **Pose and CTF conditioning:** Training the coordinate decoder end-to-end with known particle orientations and defoci using an ELBO loss objective.
- **Latent space analysis & decoding:** Traversal of continuous conformational trajectories, comparing learned latent representations against ground truth, and generating full 3D volumes at user-specified latent coordinates.
- **Production methods:** Connections to the official cryoDRGN package, 3DFlex, RECOVAR, and DynaMight.

---

## Notebooks

| File | Description |
|------|-------------|
| [`CryoDRGN_tutorial.ipynb`](CryoDRGN_tutorial.ipynb) | Hands-on tutorial building the forward projection model, training a coordinate-based VAE, and decoding heterogeneous conformational states from noisy cryo-EM images. |

---

## Key References

- Zhong, E.D., Bepler, T., Berger, B. & Davis, J.H. (2021). CryoDRGN: reconstruction of heterogeneous cryo-EM structures using neural networks. *Nature Methods* 18, 176–185.
- Punjani, A. & Fleet, D.J. (2023). 3DFlex: determining structure and motion of flexible proteins from cryo-EM. *Nature Methods* 20, 860–870.
- Mildenhall, B. et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *ECCV*.
