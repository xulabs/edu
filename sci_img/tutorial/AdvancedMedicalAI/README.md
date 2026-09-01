# Phase 03 — Advanced Medical AI

**Level: Advanced** · Complete Phases 01 and 02 (or have equivalent experience with PyTorch training loops, U-Net segmentation, and image registration) before starting here.

Phase 03 is where the building blocks become systems. Each module targets a different axis of the same overarching problem — turning multi-modal, multi-session clinical imaging data into actionable information — and the four modules are deliberately ordered so that later ones can use the outputs of earlier ones.

```
Delineate (Module 01 — 3-D segmentation)
     ↓
Align (Module 02 — multimodal registration)
     ↓
Synthesise (Module 03 — generate a missing modality)
     ↓
Interpret (Module 04 — ask a question in natural language)
     ↓
Case Study (Module 05 — clinical diagnostic assistant)
```

---

## Contents

| # | Subfolder | Core topic | Notebook | GPU needed? |
|---|-----------|-----------|---------|-------------|
| 1 | [`Volume3DSegmentation`](Volume3DSegmentation/) | 3-D U-Net spleen segmentation with MONAI on MSD | `3d_medical_image_segmentation_monai.ipynb` | Yes |
| 2 | [`MultimodalRegistration`](MultimodalRegistration/) | Classical (SimpleITK, ANTs) + deep (VoxelMorph-style) registration | `multimodal_image_registration.ipynb`<br>`biomedical_image_registration_and_deformation.ipynb` | Yes (deep section) |
| 3 | [`MedicalGenerativeModel`](MedicalGenerativeModel/) | T1→T2 MRI synthesis via Schrödinger Bridge CFM & Diffusion Foundations | `sb_cfm_medical_synthesis.ipynb`<br>`diffusion_for_medical_imaging.ipynb` | Yes |
| 4 | [`MedicalVisionLanguage`](MedicalVisionLanguage/) | BiomedCLIP, BLIP-2, LLaVA-1.5 on real medical images + Gradio demo | `multimodal_medical_imaging_tutorial.ipynb` | Yes |
| 5 | [`ClinicalDiagnosticAssistant`](ClinicalDiagnosticAssistant/) | Simulated Transformer patch embedding and reasoning case study | `ai_driven_clinical_diagnostic_assistant.ipynb` | No |
| 6 | [`MultimodalDeepLearning`](MultimodalDeepLearning/) | Image + clinical/tabular early, late, and cross-attention fusion baselines | `multimodal_deep_learning_medical_imaging_tutorial.ipynb` | No |
| 7 | [`PromptableSegmentation`](PromptableSegmentation/) | Coordinate point/box prompt representation and MedSAM-guided lesion segmentation | `promptable_biomedical_lesion_segmentation.ipynb` | No |
| 8 | [`MedicalImageClassification`](MedicalImageClassification/) | Grayscale 2D medical image classification with MONAI on MedNIST | `medical_image_classification_mednist.ipynb` | Yes (training) |
| 9 | [`MedicalImageSegmentation`](MedicalImageSegmentation/) | Grayscale 2D medical image segmentation with MONAI U-Net on synthetic shapes | `simple_segmentation.ipynb` | No |
| 10 | [`AcademicApproach`](AcademicApproach/) | Rigorous, reproducible evaluation, dataset splitting, and baseline habits | `academic_approach_medical_imaging_ml.ipynb` | No |

---

## Module Summaries

### Module 01 — 3-D Medical Image Segmentation with MONAI

The direct volumetric extension of Phase 02's 2-D U-Net. The target task is spleen segmentation on the Medical Segmentation Decathlon (Task09_Spleen) — 41 abdominal CT scans with binary labels, a publicly available benchmark with comparable literature results.

The technical challenge that distinguishes 3-D from 2-D segmentation: a full abdominal CT at 1 mm isotropic resolution is approximately 200 × 512 × 512 voxels (~100 M floats) and does not fit in GPU memory. The module covers MONAI's solution in full: a patch-based training strategy with `RandCropByPosNegLabeld` that ensures equal sampling from spleen and background patches, `CacheDataset` to eliminate the NIfTI decompression bottleneck in subsequent epochs, mixed-precision training (AMP) to halve VRAM usage, and sliding-window inference to evaluate the full volume at test time.

The architecture is a 3-D U-Net with skip connections across all four encoder levels. Loss is `DiceCELoss` — a weighted combination of Dice (robust to the severe class imbalance: spleen is ~1% of all voxels) and cross-entropy (stable gradient signal early in training). Evaluation uses Dice Similarity Coefficient (DSC) and 95th-percentile Hausdorff Distance (HD95).

**Key skills:** NIfTI affine matrices, MONAI transform pipeline (`Orientationd`, `Spacingd`, `ScaleIntensityRanged`), `CacheDataset`, `DiceCELoss`, `AdamW` + cosine LR, `torch.cuda.amp.autocast`, `sliding_window_inference`, DSC, HD95.

**Target performance:** DSC > 0.90 after 50 epochs on a T4 GPU.

---

### Module 02 — Multimodal Image Registration

Registration finds the transform that maps one image into alignment with another. When the images come from different modalities (CT ↔ MRI, T1 ↔ T2, MRI ↔ PET) this is *multimodal registration*, and it demands metrics that capture anatomical correspondence without assuming any particular intensity relationship between modalities.

The module builds a three-level comparison using a synthetic brain phantom with known ground-truth misalignment:

**Metric comparison** — MSE, NCC, and Mutual Information are evaluated as a function of misalignment angle. This makes MI's superiority for multimodal tasks immediately visible: MSE reports maximum dissimilarity for a perfectly aligned T1/T2 pair; MI correctly identifies alignment.

**Classical registration (SimpleITK)** — rigid → affine → B-spline deformable registration, with a mandatory multi-resolution pyramid at each stage. A common mistake (starting deformable registration from scratch rather than warm-starting from an affine transform) is demonstrated and its consequences shown.

**SyN deformable registration (ANTs)** — the symmetric diffeomorphic normalisation algorithm widely used in neuroimaging, which produces a physically valid, invertible displacement field with positive Jacobian determinant everywhere.

**Deep registration (VoxelMorph-style)** — a U-Net that takes the concatenated fixed and moving images as input and outputs a dense displacement field. A spatial transformer (differentiable `grid_sample`) applies the field during training so the alignment loss back-propagates through the warp. The smoothness regularisation weight is shown to be the key hyperparameter: too small produces chaotic displacement fields; too large under-registers.

Evaluation covers propagated-segmentation Dice, Target Registration Error on anatomical landmarks, and Jacobian determinant histograms for deformable methods.

**Key skills:** Mattes MI, NMI, `sitk.ImageRegistrationMethod`, multi-resolution pyramid, ANTs SyN, `torch.nn.functional.grid_sample`, spatial transformer, Jacobian determinant.

---

### Module 03 — Medical Generative Models: Cross-Modal Synthesis

A patient who received a CT scan did not necessarily receive an MRI. A rare pathology may appear on only one of the available modalities. Cross-modal synthesis generates a plausible image in a target modality from a source, effectively filling in the missing scan without additional acquisition.

The module implements **Schrödinger Bridge Conditional Flow Matching (SB-CFM)** — a state-of-the-art generative framework that is substantially more stable and sample-efficient than GANs, and more computationally direct than standard diffusion models.

**Conditional Flow Matching (CFM)** trains a neural network to predict a time-dependent vector field that transports samples from a source distribution (T1) to a target distribution (T2). The training loss is a simple regression: `‖v_θ(x_t, t) – (x₁ – x₀)‖²` — no discriminator, no score matching. At inference, integrate the ODE from t=0 to t=1 starting from the source image.

**The Schrödinger Bridge** modification selects the minimum-entropy transport path between the two distributions, producing straighter ODE trajectories that require fewer integration steps at inference and have lower-variance training targets.

The architecture is MONAI's `BasicUNet` with a sinusoidal time embedding at the bottleneck and the source image concatenated channel-wise with the interpolated input. Two inference solvers are compared (Euler and RK2), and the module ends with a rare-pathology augmentation demo — the clinical use case that motivates the whole approach.

The dataset is IXI paired T1/T2 brain MRI, which is mathematically equivalent to MRI→CT transport and freely available without institutional access.

**Key skills:** `torchcfm.SchrodingerBridgeConditionalFlowMatcher`, sinusoidal time embeddings, Euler / RK2 ODE integration, PSNR, SSIM, MAE, `einops.rearrange`, rare-pathology oversampling.

---

### Module 04 — Medical Vision-Language Models

The previous three modules required training or fine-tuning a model on medical data. This module asks how much pre-trained foundation models can do *without* any additional training — and where they fall short.

Three open-source multimodal models are run on four real medical image types (chest X-ray, brain MRI, retinal fundus, histopathology):

**BiomedCLIP** — a CLIP model pre-trained on 15 million biomedical image-text pairs from PubMed Central. Used for zero-shot classification: embed the image with the ViT encoder, embed candidate class labels with the text encoder, and take the argmax cosine similarity. Prompt sensitivity is demonstrated explicitly: the same image can yield dramatically different confidence scores with different prompt wording.

**BLIP-2** — a frozen ViT-L (307 M params) coupled to a frozen FlanT5-XL (3 B params) via a trainable Querying Transformer (~188 M). Only the Q-Former was trained, making BLIP-2 highly parameter-efficient. Applied to captioning all four image types; the module shows where the model produces plausible clinical language and where it hallucinates anatomically confident but factually wrong descriptions.

**LLaVA-1.5-7B** — a CLIP ViT connected to Vicuna-7B via a two-layer MLP, loaded in 4-bit quantisation (~5 GB VRAM) to fit alongside the other models on a T4. Used for open-ended visual question answering with clinical prompts.

Generated text is evaluated with BERTScore (semantic similarity using BERT token embeddings), SacreBLEU (n-gram precision), and ROUGE-L (longest common subsequence recall). The module closes with a Gradio demo that wraps all three models in a single web interface.

**Key skills:** `open_clip_torch`, zero-shot classification prompt design, BLIP-2 Q-Former architecture, 4-bit quantisation with `bitsandbytes`, `BERTScore`, `sacrebleu`, CUDA memory management (`del model`, `gc.collect()`, `torch.cuda.empty_cache()`), `gradio`.

**Critical caveat:** None of these models carry regulatory clearance. They are research tools and will hallucinate. Always validate against ground-truth reports.

---

### Module 05 — Case Study: AI-Driven Clinical Diagnostic Assistant

This module is a comprehensive case study bridging traditional computer vision and deep learning sequence representation. It simulates an Automated Radiology Report Generation (ARRG) pipeline using a clinical chest X-ray:
1. **Visual Tokens & Region of Interest (ROI) Selection:** Finding complex bounding boxes in an X-ray using traditional image processing.
2. **Patch Embedding & Feature Encoding:** Simulating linear projection and feature extraction of localized patches.
3. **Spatial Awareness:** Simulating 2-D sequence and position encodings to feed into transformer layers.
4. **Diagnostic Reasoning:** Comparing extracted clinical features against known diagnostic templates using similarity scoring.

**Key skills:** Chest X-ray ROI pooling, patch extraction, linear projection, sine/cosine positional encoding, cosine similarity classification.

---

### Module 06 — Multimodal Deep Learning (Fusion & Contrastive Pretraining)

This module covers the architectures and fusion strategies used when combining visual data with tabular or clinical features (e.g. lab values, vitals, patient demographics). You will implement three core fusion strategies: early fusion (concatenation at input), late fusion (fusion of logits or features near output), and cross-attention fusion (intermediate transformer-style attention query). Additionally, you will build a toy implementation of CLIP-style contrastive pretraining to align image and clinical text embeddings and test the robustness of the fusion architectures when certain modalities are noisy or missing.

**Key skills:** Early/late/cross-attention fusion architectures, multi-head attention, contrastive loss, unimodal vs. multimodal baselines, noise robustness.

---

### Module 07 — Promptable Biomedical Lesion Segmentation (MedSAM-inspired)

This module introduces promptable foundation models for medical image segmentation (like SAM and MedSAM). Unlike traditional automatic segmentation (e.g., U-Net), promptable models accept user prompts (such as bounding boxes or point coordinates) to select and segment a specific lesion. You will learn to represent points as Gaussian peak channels and boxes as binary channels, build a lightweight prompt-conditioned segmentation network, and train it on synthetic lesion images. You will also perform ablation studies and stress-test the model against noisy or imperfect prompts.

**Key skills:** Prompt-guided segmentation, bounding box & point prompt channel encoding, MedSAM-inspired architecture, Dice/IoU/sensitivity/specificity metrics, stress-testing.

---

### Module 08 — Medical Image Classification

This module serves as a 2D beginner-friendly introduction to medical image classification using deep learning. You will use MONAI and PyTorch to classify 2D grayscale medical images (X-rays, CT scans, MRIs, hands) using the MedNIST dataset and a DenseNet121 convolutional neural network.

**Key skills:** `MedNISTDataset`, custom transforms (`Compose`, `LoadImage`, `ScaleIntensity`, `Resize`, `RandRotate90`, `RandFlip`), DenseNet121, accuracy/F1-score classification report, ROC curves, confusion matrix.

---

### Module 09 — Medical Image Segmentation

This module serves as a 2D beginner-friendly introduction to medical image segmentation. You will use MONAI and PyTorch to segment synthetic images containing simple shapes using a standard U-Net architecture. The module builds directly on image classification concepts but introduces pixel-level ground truth (masks), the U-Net architecture, Dice Loss, and the Dice score metric.

**Key skills:** Synthetic image generation (`create_test_image_2d`), U-Net architecture, pixel-level classification (masks), Dice Loss, Dice score evaluation.

---

### Module 10 — Academic Approach to Machine Learning for Medical Imaging

Building a machine learning model that works is different from building one that is academically rigorous, reproducible, and defensible as research. This notebook covers the habits and checks that separate a "working demo" from academically rigorous work — using examples from our own prior classification and segmentation notebooks where relevant.

You will learn to formulate falsifiable research questions, prevent data leakage by splitting datasets at the patient level rather than scan level, implement baseline models for robust comparison, evaluate performance using multi-metric classification reports, and adopt practices for full experiment reproducibility.

**Key skills:** Patient-level split logic (`GroupShuffleSplit`), data leakage detection, baseline comparisons, classification reports, reproducibility protocols.

---

## Learning Goals

After completing Phase 03 you will be able to:

- Set up MONAI's patch-based 3-D training pipeline and train a volumetric U-Net on a public CT benchmark to clinically useful Dice (> 0.90)
- Explain why Mutual Information is the correct metric for multimodal registration and implement it as a differentiable loss
- Apply rigid, affine, and deformable registration using SimpleITK and ANTs, and compare to a trained deep-learning registration network on the same dataset
- Derive the Conditional Flow Matching objective, understand what the Schrödinger Bridge modification adds in terms of trajectory efficiency, and train a synthesis model on paired 2-D MRI slices
- Run BiomedCLIP zero-shot classification, BLIP-2 captioning, and LLaVA VQA on medical images; evaluate generated text quantitatively; and manage multiple large models within a 16 GB VRAM budget
- Explain early, late, and cross-attention fusion strategies for multi-modal medical datasets
- Implement and train a promptable, MedSAM-inspired 2-D lesion segmentation model conditioned on box and point prompts
- Build a medical image classification model using MONAI and evaluate performance with ROC curves and confusion matrices
- Build and train a 2D U-Net segmentation model using MONAI on synthetic shape images
- Explain and implement Dice Loss and Dice similarity coefficient for segmentation evaluation
- Formulate falsifiable research questions and prevent patient-level data leakage when training medical imaging models

---

## Prerequisites

- **Phases 01 and 02 completed** — or equivalent experience with PyTorch, training loops, DataLoaders, U-Net segmentation, and image-registration concepts
- Comfort with reading and modifying Python class definitions
- Familiarity with loss functions, gradient descent, and learning rate schedulers

---

## Hardware Requirements

All notebooks target **Google Colab with a T4 GPU (16 GB VRAM)**.

| Module | Peak VRAM | Notes |
|--------|----------|-------|
| 01 — 3-D Segmentation | ~10 GB | Mixed-precision brings this within T4 limits |
| 02 — Registration | ~4 GB | Deep-learning section only; classical sections are CPU |
| 03 — Generative Model | ~6 GB | 2-D slices keep memory manageable |
| 04 — Vision-Language | ~14 GB peak | **Load and unload one model at a time** — see teardown pattern below |
| 05 — Clinical Assistant | CPU | Runs fully on CPU |
| 06 — Multimodal Fusion | CPU / GPU | Fast, runnable on CPU or Google Colab free tier |
| 07 — Promptable Seg | CPU / GPU | Fast, runnable on CPU or Google Colab free tier |
| 08 — Classification | ~4 GB | Fast, training is runnable on GPU in Colab free tier |
| 09 — Medical Seg | CPU | Fast, training is runnable on CPU or Google Colab free tier |
| 10 — Academic Approach | CPU | Runs fully on CPU |

**Module 04 teardown pattern (mandatory):**

```python
del model, processor      # Remove Python references
import gc; gc.collect()   # Python garbage collector
torch.cuda.empty_cache()  # Release CUDA memory
```

Skipping any of these three steps will cause OOM when loading the next model.

---

## Setup

```bash
# Each notebook also installs its own dependencies in the first cell
pip install "monai[nibabel,tqdm]" SimpleITK antspyx torchcfm nibabel \
    "transformers>=4.37.0" "accelerate>=0.27.0" "bitsandbytes>=0.41.0" \
    open_clip_torch bert-score sacrebleu "gradio>=4.20.0" \
    torch torchvision einops matplotlib scikit-image "numpy<2.0.0"
```

---

## Suggested Order

```
MedicalImageClassification
         ↓
MedicalImageSegmentation
         ↓
Volume3DSegmentation
         ↓
MultimodalRegistration
         ↓
MedicalGenerativeModel    ←→    MedicalVisionLanguage
         ↓                              ↓
MultimodalDeepLearning    ←→    PromptableSegmentation
                                         ↓
                                 ClinicalDiagnosticAssistant
                                              ↓
                                       AcademicApproach
```

Module 08 serves as a 2D beginner-friendly introduction to MONAI and PyTorch training loops, setting the stage. Module 09 introduces 2D image segmentation on synthetic datasets as a transition before moving to the more complex 3D segmentation task in Module 01. Modules 03 and 04 are largely independent of each other. Once you have finished 01 and 02, they can be taken in either order. Similarly, Modules 06 and 07 cover multi-modal fusion and promptable foundation concepts, which prepare you for the sequence representation logic in Module 05. Finally, Module 10 covers the academic habits and methodology rules for training and validating medical imaging models.

---

## How the Modules Connect in Practice

The modules address the same clinical challenge from different angles:

1. A 3-D scan arrives → **Module 01** delineates the structure of interest automatically
2. Serial scans from different sessions arrive → **Module 02** aligns them into a common space
3. A scan in the needed modality is missing → **Module 03** synthesises it from an available modality, enabling monomodal registration with MSE (simpler and faster than multimodal MI)
4. A clinician wants to query the processed data in natural language → **Module 04** answers zero-shot, without task-specific fine-tuning
5. Interactive or targeted segmentation is required → **Module 07** handles promptable, human-in-the-loop segmentation
6. Multiple diagnostic modalities (imaging + lab values) need to be combined → **Module 06** fuses them into unified representations
7. A clinician wants a complete assistant to explain predictions and generate reports → **Module 05** acts as the clinical reasoning sequence case study

Together they cover the paradigms of modern medical vision AI: discriminative, transformation, promptable, generative, multimodal fusion, and sequence reasoning.

---

## Further Reading

- MONAI documentation: https://docs.monai.io
- VoxelMorph (Balakrishnan et al., 2019): https://arxiv.org/abs/1809.05231
- Flow Matching (Lipman et al., 2022): https://arxiv.org/abs/2210.02747
- Schrödinger Bridge CFM (Liu et al., 2023): https://arxiv.org/abs/2303.00585
- BiomedCLIP (Zhang et al., 2023): https://arxiv.org/abs/2303.00915
- BLIP-2 (Li et al., 2023): https://arxiv.org/abs/2301.12597
- LLaVA-1.5 (Liu et al., 2023): https://arxiv.org/abs/2310.03744
- MedSAM (Ma et al., 2024): https://arxiv.org/abs/2304.12306
- Multimodal Fusion (Liang et al., 2022): https://arxiv.org/abs/2209.03430
