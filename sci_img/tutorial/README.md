
# Medical & Biological Imaging AI — Tutorial Series

A four-phase, hands-on curriculum that takes you from first principles in signal processing all the way to clinical 3-D segmentation, generative synthesis, vision-language models, and real cryo-electron tomography. Every notebook runs on Google Colab (free T4 GPU), and each phase builds deliberately on the one before it.Besides, you are encouraged to pull a issue in this repo.

---

## Who this is for

| Background | Where to start |
|------------|----------------|
| Python user curious about medical AI | Phase 01 → Phase 02 → Phase 03 |
| Biologist wanting to understand cryoEM computation | Phase 01 (Module 02) → Phase 04 |
| ML practitioner new to medical imaging | Phase 02 → Phase 03 |
| Structural biology researcher / cryo-ET newcomer | Phase 04 (after Phase 01 Module 02) |

---

## 🗺️ Full Roadmap

```
Phase 01 — Computational Signal Base
  ├─ 01 PyTorch Tensor Basics & Image I/O
  ├─ 02 Frequency Filtering & Image Pyramids
  ├─ 03 Classical Feature Extraction (Edges, Lines, Curves)
  ├─ 04 Geometry-Based Classification of Bio-Images
  ├─ 05 Corner Detection & Feature Descriptors
  └─ 06 Spatial Transformations & Image Warping

Phase 02 — Spatial Perception & Analysis
  ├─ 01 Dimensionality Reduction & Latent Space (PCA)
  ├─ 02 Object Detection & Tracking (YOLOv5 + LAP)
  ├─ 03 Image Segmentation (Watershed → U-Net)
  ├─ 04 Image Alignment & Registration (SimpleITK + Pix2Pix)
  ├─ 05 Stereo Vision & Depth Estimation
  └─ 06 Self-Supervised Bioimage Denoising (Noise2Void / Noise2Self)

Phase 03 — Advanced Medical AI
  ├─ 01 3-D Volumetric Segmentation with MONAI
  ├─ 02 Multimodal Image Registration
  ├─ 03 Medical Generative Models (Schrödinger Bridge CFM & Diffusion Foundations)
  ├─ 04 Medical Vision-Language Models (BiomedCLIP, BLIP-2, LLaVA)
  ├─ 05 Clinical Diagnostic Assistant Case Study
  ├─ 06 Multimodal Deep Learning (Fusion & Contrastive Pretraining)
  ├─ 07 Promptable Biomedical Lesion Segmentation (MedSAM-inspired)
  ├─ 08 Medical Image Classification with MONAI (MedNIST)
  ├─ 09 Medical Image Segmentation with MONAI & U-Net
  └─ 10 Academic Approach to Machine Learning for Medical Imaging

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

The foundation everything else rests on. You will leave Phase 01 fluent in PyTorch tensors, comfortable loading DICOM and TIFF medical images, and able to design and apply filters in both the spatial and frequency domains. The final module introduces hand-crafted geometric features and classical classifiers — still the right choice whenever labelled data is scarce.

➜ [`ComputationalSignalBase/`](ComputationalSignalBase/)

---

### Phase 02 — Spatial Perception & Analysis
**Level: Intermediate** · Requires Phase 01 or equivalent PyTorch familiarity.

Phase 02 is about understanding *where things are* and *what shape they have*. You will compress high-dimensional images with PCA, train a YOLOv5 object detector on a custom dataset, track cells across 3-D time-lapse volumes, segment structures with both classical algorithms and U-Net, and align image pairs using both iterative optimisation and a conditional GAN.

➜ [`SpatialPerceptionAnalysis/`](SpatialPerceptionAnalysis/)

---

### Phase 03 — Advanced Medical AI
**Level: Advanced** · Requires Phases 01 and 02.

The phase where individual building blocks converge into complete clinical-grade systems. You will classify 2D medical images on MedNIST with MONAI, segment 2D medical shapes with U-Net, train a 3-D U-Net on the Medical Segmentation Decathlon spleen benchmark, work through the full spectrum of multimodal registration from classical SimpleITK to deep VoxelMorph-style networks, synthesise missing imaging modalities using Schrödinger Bridge Conditional Flow Matching, and probe state-of-the-art foundation models (BiomedCLIP, BLIP-2, LLaVA-1.5) with real medical images.

➜ [`AdvancedMedicalAI/`](AdvancedMedicalAI/)

---

### Phase 04 — CryoEM & Structural Biology
**Level: Specialist** · Requires Phase 01 Module 02 (Fourier analysis) + NumPy/SciPy fluency.

An end-to-end computational treatment of cryo-electron microscopy. Module 01 builds the 2-D signal-processing intuition (CTF simulation, denoising, particle picking, class averaging, FRC resolution). Module 02 extends everything to 3-D: the Fourier Slice Theorem, trilinear back-projection, and expectation-maximisation pose estimation. Module 03 covers fiducial-less patch-tracking tilt-series alignment. Module 04 is a full hands-on practical using IMOD/Etomo on a real published tilt series from EMPIAR. Module 05 covers 3-D sub-tomogram averaging (STA) and particle picking for cryo-ET. Module 06 covers few-shot 3D particle detection using weak labels and Volume Infill. Module 07 covers prompt-based interactive 3-D tomogram segmentation (SAM & CryoSAM propagation). Module 08 covers tomographic reconstruction and missing-wedge effects (WBP vs SIRT). Module 09 covers beam-induced motion correction (patch-based and whole-frame drift tracking).


➜ [`cryoem/`](../../cryoem/readme.md)

---

### Phase 05 — Robotic Lab Automation & VLA Models
**Level: Advanced / Specialist** · Requires Phase 01 and Phase 02.

An end-to-end treatment of robotic lab automation. Module 01 covers the AutoBio simulation framework (Lan et al., 2025), physics plugins for detents and screw-caps, success metrics, and baseline VLA model evaluations. Module 02 builds computer vision tasks inspired by AutoBio (tube localization, slot symmetry indexing, liquid-level estimation, closed-loop dial panel control) using synthetic toy data. Module 03 introduces the RLBench simulator, its six core design principles, action/observation spaces, and waypoint-based motion planning. Module 04 covers the Diffusion Policy framework for visuomotor control, Module 05 teaches language-conditioned, long-horizon robot manipulation using the CALVIN benchmark, Module 06 covers DQN from raw pixels, Module 07 covers computer vision-based quality control for Opentrons OT-2, and Module 08 covers the BEHAVIOR-1K embodied AI benchmark.

➜ [`RoboticLabAutomation/`](../../al/tutorial/)

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
| Convolution / kernels | Ph01-M02 | Ph01-M03, Ph02-M03, Ph03-M01 |
| Fourier Transform | Ph01-M02 | Ph04-M01, Ph04-M02 |
| Edge & feature detection | Ph01-M03 | Ph02-M02, Ph02-M03 |
| GLCM / geometric features | Ph01-M04 | Ph02-M01 (motivation for PCA) |
| PCA / latent space | Ph02-M01 | Ph03-M03, Ph04-M02 |
| U-Net architecture | Ph02-M03 | Ph02-M04, Ph03-M01, Ph03-M02 |
| Similarity metrics (MI) | Ph02-M04 | Ph03-M02 |
| Dice loss | Ph02-M03 | Ph03-M01 |
| CTF | Ph04-M01 | Ph04-M02, Ph04-M03 |
| Expectation-Maximisation | Ph04-M02 | Ph04-M03 (production software) |

---

## Folder Structure

```
tutorial/
├── README.md                          ← You are here
├── ComputationalSignalBase/
│   ├── README.md
│   ├── PyTorchTensorBasic/
│   ├── FrequencyFiltering/
│   ├── ClassicalFeatureExtract/
│   ├── GeometryBasedClassification/
│   ├── CornerDetection/
│   └── SpatialTransformations/
├── SpatialPerceptionAnalysis/
│   ├── README.md
│   ├── DimensionReductionLatentSpace/
│   ├── ObjectDetectTracking/
│   ├── ImageSegmentation/
│   ├── ImageAlignmentRegister/
│   ├── StereoVision/
│   └── BioimageDenoising/
├── AdvancedMedicalAI/
│   ├── README.md
│   ├── Volume3DSegmentation/
│   ├── MultimodalRegistration/
│   ├── MedicalGenerativeModel/
│   ├── MedicalVisionLanguage/
│   ├── ClinicalDiagnosticAssistant/
│   ├── MultimodalDeepLearning/
│   ├── PromptableSegmentation/
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
Phase 01 (all) → Phase 02 (M01, M03, M04) → Phase 03 (M01, M02, M04)

**"I want to understand generative models for medical data"**
Phase 01 (M01, M02) → Phase 02 (M03, M04) → Phase 03 (M03)

**"I'm joining a cryoEM lab"**
Phase 01 (M01, M02) → Phase 04 (all, in order)

**"I want the full curriculum"**
Phase 01 → Phase 02 → Phase 03 → Phase 04
