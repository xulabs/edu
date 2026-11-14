# 05 — Naive Bayes vs. Logistic Regression for Bio-Image Classification

**Contributor:** Rishita Prabhakar Polumatla

## Overview

You have learned how to extract hand-crafted features from images (GLCM texture, Haralick,
geometric descriptors). Now the question becomes: **which probabilistic classifier should
you use, and why?**

This module compares two foundational probabilistic classifiers — **Gaussian Naive Bayes**
and **Logistic Regression** — on the same bio-image feature vectors, so you can develop an
intuition for *when* each one wins and *why*.

Rajdeep's earlier module (Module 04) covers **SVM and k-NN**. This module is the direct
companion piece: together they give you the full classical classifier toolkit for bio-image
analysis.

---

## What you will learn

### The probabilistic classification pipeline

```
Raw fluorescence image
        │
        ▼ Feature extraction (GLCM + geometric — same as Module 04)
Feature vector per image/object
        │
        ├─► Gaussian Naive Bayes  (P(class|features) via Bayes' theorem + independence assumption)
        │
        └─► Logistic Regression   (P(class|features) via learned linear boundary + sigmoid/softmax)
        │
        ▼ Compare: accuracy, calibration, speed, interpretability
```

---

## Naive Bayes — Core Idea

Naive Bayes applies **Bayes' theorem** with the "naive" assumption that all features are
conditionally independent given the class label:

$$P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)$$

For **Gaussian Naive Bayes**, each $P(x_i \mid y)$ is modelled as a Gaussian:

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_{y,i}^2}} \exp\!\left(-\frac{(x_i - \mu_{y,i})^2}{2\sigma_{y,i}^2}\right)$$

**When it works well:**
- Very small datasets (few dozen samples) — low variance, fast fitting
- Features that are genuinely approximately independent (e.g. colour histogram bins)
- Baseline model — if Naive Bayes fails badly, something is wrong with your features

**Known limitation:**
The independence assumption almost never holds perfectly for image features. GLCM contrast
and homogeneity are correlated. This makes probability estimates poorly calibrated, even
when rank-order accuracy is fine.

---

## Logistic Regression — Core Idea

Logistic Regression learns a **linear decision boundary** and converts scores to
probabilities via the **sigmoid** (binary) or **softmax** (multi-class) function:

$$P(y=k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k \cdot \mathbf{x} + b_k}}{\sum_j e^{\mathbf{w}_j \cdot \mathbf{x} + b_j}}$$

It is trained by **minimising cross-entropy loss** with optional L2 regularisation
(controlled by hyperparameter $C = 1/\lambda$):

$$\mathcal{L} = -\sum_i \log P(y_i \mid \mathbf{x}_i) + \frac{\lambda}{2} \|\mathbf{w}\|^2$$

**When it works well:**
- Features are already well-scaled and linearly separable
- You need calibrated probability estimates (for downstream decision making)
- You need to inspect feature weights for interpretability

**Known limitation:**
Linear decision boundary can fail when classes are not linearly separable in feature
space — that is where SVM with RBF kernel (Module 04) takes over.

---

## Side-by-Side Comparison

| Property | Gaussian Naive Bayes | Logistic Regression |
|---|---|---|
| **Model type** | Generative (models $P(x\|y)$) | Discriminative (models $P(y\|x)$) |
| **Training speed** | Very fast — just compute means & variances | Slower — iterative optimisation |
| **Assumption** | Feature independence | Linear decision boundary |
| **Small data** | ✅ Excellent | ⚠️ May overfit without regularisation |
| **Large data** | ⚠️ Independence hurts | ✅ Converges to Bayes optimal |
| **Probability calibration** | ❌ Often poorly calibrated | ✅ Generally well calibrated |
| **Interpretability** | Per-class feature distributions | Feature weights directly readable |
| **Handles correlated features** | ❌ No | ✅ Yes (via joint optimisation) |

---

## The HeLa Dataset

Same dataset as Module 04 — fluorescence microscopy images of HeLa cells across
**10 organelle classes**: endosome, lysosome, mitochondria, nucleus, endoplasmic reticulum,
actin, Golgi apparatus, plasma membrane, microtubule, nucleolus.

This is a deliberately hard classification task — many classes look similar, which makes
it a perfect stress-test for understanding the limits of each classifier.

---

## Notebook

| File | Description |
|------|-------------|
| `naive_bayes_vs_logistic_regression.ipynb` | Full pipeline: HeLa dataset loading, GLCM + geometric feature extraction, StandardScaler normalisation, Gaussian Naive Bayes vs Logistic Regression training and evaluation, confusion matrices, calibration curves, feature weight visualisation, discussion questions |

---

## Prerequisites

Complete Module 04 (Geometry-Based Classification) first — this module reuses the same
feature extraction pipeline and builds directly on it.

## Setup

```bash
pip install scikit-learn scikit-image numpy matplotlib seaborn
```

No GPU required. All computation is CPU-based and runs in under 3 minutes on the full
HeLa dataset.

---

## Connection to the Tutorial Series

| Module | Classifier(s) covered |
|---|---|
| 04 — GeometryBasedClassification | SVM (linear, RBF, OvR), k-NN |
| **05 — ClassifierComparison (this module)** | **Naive Bayes, Logistic Regression** |
| Phase 2 — Deep Learning | CNN-based classifiers |

After finishing both Module 04 and 05, you will have compared five classical classifiers
on the same features — giving you a solid empirical foundation for choosing classifiers
in your own bio-image projects.
