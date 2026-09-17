# Tilt-Series Alignment

## Overview

Reconstructing a 3D tomogram from 2D projections requires precise alignment of the images collected at different tilt angles. This module covers **Fiducial-Less Patch-Tracking Tilt-Series Alignment**, simulating the alignment process and implementing cross-correlation based patch tracking.

---

## What you will learn

- Simulating a tilt series with known misalignments from a real 3D EM volume.
- The theory of fiducial-less patch-tracking alignment.
- Implementing patch tracking and cumulative cross-correlation alignment correction.
- Validating alignment accuracy using ground-truth translations.
- Reconstructing the volume using Weighted Back Projection (WBP) and assessing reconstruction quality.
- Understanding production software alignment algorithms (such as IMOD and AreTomo) and their practical limitations.

---

## Notebooks

| File | Description |
|------|-------------|
| [`tilt_series_alignment_tutorial.ipynb`](tilt_series_alignment_tutorial.ipynb) | Code-based tutorial demonstrating how to simulate misaligned projections, implement patch tracking, compute translational corrections, and reconstruct the final aligned 3D volume. |
