# 09 — Beam-Induced Motion Correction

## Overview

In cryo-electron microscopy, high-energy electron irradiation causes physical deformation of the vitreous ice layer and mechanical drift of embedded macromolecular specimens—known as **beam-induced motion**. Because micrographs are acquired as **dose-fractionated movies** (10–40 frames per exposure), uncorrected motion severely blurs high-frequency features and diminishes reconstruction resolution.

This module implements both **whole-frame (rigid)** and **patch-based (local / per-particle)** beam-induced motion correction from first principles using real cryo-EM data (EMPIAR-10146 apoferritin).

---

## What you will learn

- **Physics of beam-induced motion:** Why electron irradiation triggers ice doming, bubbling, and localized particle movements.
- **Rigid vs. non-rigid motion:** Why global $(dx, dy)$ frame alignment fails to account for spatial variation across the field of view.
- **Whole-frame alignment:** Estimating global drift trajectories using iterative cross-correlation against a running reference frame.
- **Patch-based local correction:** Dividing the field of view into a regular spatial grid, solving local motion trajectories, and smoothly interpolating deformation fields.
- **Dose weighting & B-factor preservation:** How early frames suffer from radiation-induced initial burst motion while late frames suffer from accumulated radiation damage.
- **Production pipelines:** Connections to industry-standard tools like MotionCor2, RELION MotionCorr, and Warp.

---

## Notebooks

| File | Description |
|------|-------------|
| [`motion_correction_tutorial.ipynb`](motion_correction_tutorial.ipynb) | Hands-on tutorial implementing whole-frame cross-correlation and patch-based local motion correction on dose-fractionated cryo-EM movie frames. |
