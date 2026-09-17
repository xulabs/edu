# Few-Shot CryoET Particle Detection

## Overview

Automatic particle picking in cryo-electron tomography (cryo-ET) is heavily bottlenecked by the scarcity of annotated data and low signal-to-noise ratios. This module introduces a deep learning pipeline for **Few-Shot Cryo-ET Particle Detection** using weak labels and Volume Infill, inspired by the SaSi framework (Adethya et al., 2025).

---

## What you will learn

- Generating spherical weak labels around sparse user-annotated particle centers.
- The concept of self-augmented **Volume Infill**: extracting particle subvolumes and pasting them into background locations to augment training samples.
- Building and training a lightweight 3D convolutional network for voxel-wise particle detection.
- Extracting 3D particle coordinates from predicted probability volumes using connected-component analysis.
- Evaluating detection performance using coordinate-based metrics (Precision, Recall, F1-score, and mean localization distance error).

---

## Notebooks

| File | Description |
|------|-------------|
| [`few_shot_cryoet_particle_detection_with_weak_labels_and_volume_infill.ipynb`](few_shot_cryoet_particle_detection_with_weak_labels_and_volume_infill.ipynb) | End-to-end tutorial demonstrating spherical weak label construction, Volume Infill data augmentation, 3D CNN training, post-processing coordinates extraction, and metric evaluation on a synthetic tomogram. |
