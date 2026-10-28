
# Medical & Biological Imaging AI — Tutorial Series

> **New to bioimage analysis, or a first-year biology major?** Start with
> [**BiologyToBioimage**](BiologyToBioimage/README.md) — a 29-module notebook course that assumes
> no prior programming or imaging background, builds a deep understanding of CMU 02-740 *Bioimage
> Informatics*, and then reaches the research frontier (foundation models, virtual staining,
> spatial omics, trustworthy AI and more). The phases below assume PyTorch/GPU fluency and are a
> natural next step after it.

A four-phase, hands-on curriculum that takes you from first principles in signal processing all the way to clinical 3-D segmentation, generative synthesis, vision-language models, and real cryo-electron tomography. Every notebook runs on Google Colab (free T4 GPU), and each phase builds deliberately on the one before it.Besides, you are encouraged to pull a issue in this repo.

---

## Who this is for

| Background | Where to start |
|------------|----------------|
| Python user curious about medical AI | Phase 01 → Phase 02 → Phase 03 |
| Biologist wanting to understand cryoEM computation | BiologyToBioimage (Module 04, Fourier analysis) → Phase 04 |
| ML practitioner new to medical imaging | Phase 02 → Phase 03 |
| Structural biology researcher / cryo-ET newcomer | Phase 04 (after BiologyToBioimage Module 04, Fourier analysis) |

---

## 🗺️ Full Roadmap

```
Phase 01 — Computational Signal Base
  ├─ 01 PyTorch Tensor Basics & Image I/O
  ├─ 02 Corner Detection & Feature Descriptors
  └─ 03 Spatial Transformations & Image Warping

Phase 02 — Spatial Perception & Analysis
  ├─ 01 Object Detection & Tracking (YOLOv5 + LAP)
  ├─ 02 Image Alignment & Registration (SimpleITK + Pix2Pix)
  ├─ 03 Stereo Vision & Depth Estimation
  ├─ 04 Self-Supervised Bioimage Denoising (Noise2Void / Noise2Self)
  └─ 05 Spatial Omics & Multiplexed Imaging (Neighborhood Graphs & Niches)

Phase 03 — Advanced Medical AI
  ├─ 01 3-D Volumetric Segmentation with MONAI
  ├─ 02 Multimodal Image Registration
  ├─ 03 Medical Generative Models (Schrödinger Bridge CFM & Diffusion Foundations)
  ├─ 04 Medical Vision-Language Models (BiomedCLIP, BLIP-2, LLaVA)
  ├─ 05 Clinical Diagnostic Assistant Case Study
  ├─ 06 Multimodal Deep Learning (Fusion & Contrastive Pretraining)
  ├─ 07 Medical Image Classification with MONAI (MedNIST)
  ├─ 08 Medical Image Segmentation with MONAI & U-Net
  └─ 09 Academic Approach to Machine Learning for Medical Imaging

Phase 04 — CryoEM & Structural Biology
  ├─ 01 Low-SNR Noise Processing & Particle Picking
  ├─ 02 3-D CryoEM Reconstruction
  ├─ 03 Tilt-Series Alignment
  ├─ 04 Build & Diagnose a Real Cryo-ET Tomogram (EMPIAR-10164)
  ├─ 05 Sub-tomogram Averaging & 3D Particle Picking (Cryo-ET)
  ├─ 06 Few-Shot CryoET Particle Detection
  ├─ 07 Tomogram Segmentation
  ├─ 08 Tomographic Reconstruction & Missing Wedge (WBP vs SIRT)
  └─ 09 Beam-Induced Motion Correction (Patch-Based & Whole-Frame)


Phase 05 — Robotic Lab Automation & VLA Models
  ├─ 01 AutoBio Simulator & Benchmark
  ├─ 02 AutoBio Computer Vision & Closed-Loop Control
  ├─ 03 RLBench Simulator & Benchmark
  ├─ 04 Diffusion Policy Visuomotor Training
  ├─ 05 CALVIN Language-Conditioned Long-Horizon Manipulation
  ├─ 06 CartPole DQN: Deep Q-Learning from Raw Pixels
  ├─ 07 AI-Driven Quality Control for Opentrons OT-2
  └─ 08 BEHAVIOR-1K Embodied AI Benchmark
```

---

## Phase Overview

### Phase 01 — Computational Signal Base
**Level: Beginner** · No prior PyTorch or image-processing experience required.

The foundation everything else rests on. You will leave Phase 01 fluent in PyTorch tensors, comfortable loading DICOM and TIFF medical images, able to detect classical corners and feature descriptors, and able to apply 2-D spatial transformations and homographies to align image pairs. (Frequency-domain filtering and hand-crafted geometric classification — formerly covered here — are now taught in [BiologyToBioimage](BiologyToBioimage/README.md) Modules 04 and 07.)

➜ [`foundation/ComputationalSignalBase`](foundation/ComputationalSignalBase/)

---

### Phase 02 — Spatial Perception & Analysis
**Level: Intermediate** · Requires Phase 01 or equivalent PyTorch familiarity.

Phase 02 is about understanding *where things are* and *what shape they have*. You will train a YOLOv5 object detector on a custom dataset and track cells across time-lapse volumes with the linear assignment problem, align image pairs using both classical iterative optimisation (SimpleITK) and a conditional GAN (Pix2Pix), estimate depth from stereo pairs, and denoise bioimages with self-supervised networks. (Dimensionality reduction with PCA and classical-to-deep segmentation — formerly covered here — are now taught in [BiologyToBioimage](BiologyToBioimage/README.md) Modules 09, 10 and 11.)

➜ [`foundation/SpatialPerceptionAnalysis`](foundation/SpatialPerceptionAnalysis/)

---

### Phase 03 — Advanced Medical AI
**Level: Advanced** · Requires Phases 01 and 02.

The phase where individual building blocks converge into complete clinical-grade systems. You will classify 2D medical images on MedNIST with MONAI, segment 2D medical shapes with U-Net, train a 3-D U-Net on the Medical Segmentation Decathlon spleen benchmark, work through the full spectrum of multimodal registration from classical SimpleITK to deep VoxelMorph-style networks, synthesise missing imaging modalities using Schrödinger Bridge Conditional Flow Matching, and probe state-of-the-art foundation models (BiomedCLIP, BLIP-2, LLaVA-1.5) with real medical images. (Promptable, prompt-guided lesion segmentation — formerly covered here — is now taught in [BiologyToBioimage](BiologyToBioimage/README.md) Module 20.)

➜ [`foundation/AdvancedMedicalAI`](foundation/AdvancedMedicalAI/)

---

### Phase 04 — CryoEM & Structural Biology
**Level: Specialist** · Requires NumPy/SciPy fluency and familiarity with Fourier analysis (now taught in [BiologyToBioimage](BiologyToBioimage/README.md) Module 04), plus Phase 01 Module 01 (PyTorch tensors).

An end-to-end computational treatment of cryo-electron microscopy. Module 01 builds the 2-D signal-processing intuition (CTF simulation, denoising, particle picking, class averaging, FRC resolution). Module 02 extends everything to 3-D: the Fourier Slice Theorem, trilinear back-projection, and expectation-maximisation pose estimation. Module 03 covers fiducial-less patch-tracking tilt-series alignment. Module 04 is a full hands-on practical using IMOD/Etomo on a real published tilt series from EMPIAR. Module 05 covers 3-D sub-tomogram averaging (STA) and particle picking for cryo-ET. Module 06 covers few-shot 3D particle detection using weak labels and Volume Infill. Module 07 covers prompt-based interactive 3-D tomogram segmentation (SAM & CryoSAM propagation). Module 08 covers tomographic reconstruction and missing-wedge effects (WBP vs SIRT). Module 09 covers beam-induced motion correction (patch-based and whole-frame drift tracking).


➜ [`cryoem/`](../../cryoem/tutorial/foundation/readme.md)

---

### Phase 05 — Robotic Lab Automation & VLA Models
**Level: Advanced / Specialist** · Requires Phase 01 and Phase 02.

An end-to-end treatment of robotic lab automation. Module 01 covers the AutoBio simulation framework (Lan et al., 2025), physics plugins for detents and screw-caps, success metrics, and baseline VLA model evaluations. Module 02 builds computer vision tasks inspired by AutoBio (tube localization, slot symmetry indexing, liquid-level estimation, closed-loop dial panel control) using synthetic toy data. Module 03 introduces the RLBench simulator, its six core design principles, action/observation spaces, and waypoint-based motion planning. Module 04 covers the Diffusion Policy framework for visuomotor control, Module 05 teaches language-conditioned, long-horizon robot manipulation using the CALVIN benchmark, Module 06 covers DQN from raw pixels, Module 07 covers computer vision-based quality control for Opentrons OT-2, and Module 08 covers the BEHAVIOR-1K embodied AI benchmark.

➜ [`../../al/tutorial/RoboticLabAutomation`](../../al/tutorial/)

---

## Prerequisites

| Tool | Phase needed | Notes |
|------|-------------|-------|
| Python 3.9+ | All | |
| NumPy, Matplotlib | All | |
| PyTorch | 01–04 | Covered from scratch in Phase 01 |
| Google Colab or local GPU | 02–04 | T4 (16 GB VRAM) sufficient for all notebooks |
| IMOD/Etomo | Phase 04 Module 03 only | Linux / macOS; install locally |

---

## How to Run

**Option A — Google Colab (recommended)**

Open any notebook directly in Colab by replacing `github.com` in the URL with
`colab.research.google.com/github`. Each notebook installs its own dependencies in the first cell.

**Option B — Local Jupyter**

```bash
git clone https://github.com/pastretender/edu.git
cd edu/Tutorials
pip install torch torchvision jupyterlab
jupyter lab
```

---

## Curriculum Map: Concepts and Their Modules

| Concept | First introduced | Used again in |
|---------|-----------------|--------------|
| PyTorch tensors | Ph01-M01 | Every subsequent module |
| Similarity metrics (MI) | Ph02-M02 | Ph03-M02 |
| CTF | Ph04-M01 | Ph04-M02, Ph04-M03 |
| Expectation-Maximisation | Ph04-M02 | Ph04-M03 (production software) |

> Convolution/kernels, the Fourier Transform, edge & feature detection, GLCM/geometric
> features, PCA/latent space, U-Net architecture, and Dice loss were formerly introduced in
> modules that have since moved to [BiologyToBioimage](BiologyToBioimage/README.md)
> (Modules 04, 06, 07, 09, 10, 11 — see that course for the canonical introduction of each).

---

## Folder Structure

```
tutorial/
├── README.md                          ← You are here
├── ComputationalSignalBase/
│   ├── README.md
│   ├── PyTorchTensorBasic/
│   ├── CornerDetection/
│   └── SpatialTransformations/
├── SpatialPerceptionAnalysis/
│   ├── README.md
│   ├── ObjectDetectTracking/
│   ├── ImageAlignmentRegister/
│   ├── StereoVision/
│   ├── BioimageDenoising/
│   └── SpatialOmics/
├── AdvancedMedicalAI/
│   ├── README.md
│   ├── Volume3DSegmentation/
│   ├── MultimodalRegistration/
│   ├── MedicalGenerativeModel/
│   ├── MedicalVisionLanguage/
│   ├── ClinicalDiagnosticAssistant/
│   ├── MultimodalDeepLearning/
│   ├── MedicalImageClassification/
│   ├── MedicalImageSegmentation/
│   └── AcademicApproach/
└── RoboticLabAutomation/
    ├── README.md
    ├── AutoBioBenchmark/
    ├── AutoBioComputerVision/
    ├── RLBench/
    ├── DiffusionPolicy/
    ├── CALVIN/
    ├── CartPoleDQN/
    ├── OpentronsOT2/
    └── BEHAVIOR-1K/
```

---

## Suggested Learning Paths

**"I want to work in medical imaging AI"**
Phase 01 (all) → Phase 02 (M02) → Phase 03 (M01, M02, M04)

**"I want to understand generative models for medical data"**
Phase 01 (M01) → Phase 02 (M02) → Phase 03 (M03)

**"I'm joining a cryoEM lab"**
Phase 01 (M01) + BiologyToBioimage (Module 04, Fourier analysis) → Phase 04 (all, in order)

**"I want the full curriculum"**
Phase 01 → Phase 02 → Phase 03 → Phase 04
