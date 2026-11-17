# Topic 03 — Generative vs. Discriminative Classification

## Overview

In biomedical and scientific classification problems—such as distinguishing malignant from benign biopsies or classifying cell states—models predict categorical class labels $Y \in \{0, 1\}$ from continuous feature vectors $X$. Machine learning approaches this challenge via two distinct foundational paradigms:

1. **Generative Classifiers ($P(X \mid Y)$ and $P(Y)$):** Model how features are generated within each class and combine class-conditional densities with prior beliefs via Bayes' rule:
   $$P(Y=c \mid X) = \frac{P(Y=c)P(X \mid Y=c)}{\sum_{c'} P(Y=c')P(X \mid Y=c')}$$
   **Gaussian Naive Bayes** makes the modular assumption that features are conditionally independent given the class label ($P(X \mid Y) = \prod_{j=1}^p P(X_j \mid Y)$). This permits closed-form parameter estimation (per-class means and variances) in a single data pass, making it fast and robust when training samples are scarce.
2. **Discriminative Classifiers ($P(Y \mid X)$ directly):** Learn the decision boundary directly without modeling how the inputs $X$ are distributed. **Logistic Regression** models log-odds as a linear combination of features:
   $$\log\frac{P(Y=1 \mid X)}{P(Y=0 \mid X)} = w^\top X + b$$
   Parameters are optimized via convex cross-entropy minimization with gradient descent. Because it never models feature distributions, logistic regression is immune to correlation among predictors.
3. **Instance-Based Non-Parametric Learning ($k$-NN):** Classifies unseen query observations by majority vote among their $k$ closest neighbors in Euclidean feature space. By avoiding parametric assumptions, $k$-NN can trace arbitrary non-linear boundaries, but is sensitive to local noise when $k$ is small and computationally demanding as dataset size grows.
4. **The Impact of Correlated Features in Science:** Biological measurements frequently exhibit strong physical collinearity (e.g., cell nucleus radius, perimeter, and area in cytology). When features correlate, Naive Bayes double-counts evidence and produces overconfident, poorly calibrated probability estimates, whereas discriminative logistic regression adjusts weights to accommodate collinearity.
5. **Model Calibration & Evaluation:** In clinical settings, predicted probability magnitudes matter as much as binary accuracy. We evaluate classifiers using reliability diagrams, Brier score calibration, ROC curves, and precision-recall dynamics.

---

## What you will learn

- **Foundational Paradigms:** The mathematical and algorithmic difference between generative modeling ($P(X \mid Y)$) and discriminative modeling ($P(Y \mid X)$).
- **$k$-Nearest Neighbors ($k$-NN):** The geometry of instance-based classification, Euclidean metrics, and tuning hyperparameter $k$ to balance variance against bias.
- **Naive Bayes Formulation:** Deriving the conditional independence likelihood, Gaussian parameter estimation, and why independence violations distort posteriors.
- **Logistic Regression & Optimization:** Binary cross-entropy loss, sigmoid activation, and iterative gradient descent optimization without external libraries.
- **From-Scratch Implementation:** Building both `GaussianNaiveBayesScratch` and `LogisticRegressionScratch` with standard NumPy routines.
- **Real Biomedical Application:** Benchmarking both paradigms on the Wisconsin Breast Cancer Diagnostic dataset, comparing decision boundary flexibility, test accuracy, and probability calibration under real feature correlations.

---

## Notebooks

| File | Description |
|------|-------------|
| [`8. ML classification KNN & NB.ipynb`](8.%20ML%20classification%20KNN%20%26%20NB.ipynb) | Hands-on tutorial exploring instance-based $k$-NN versus generative Gaussian Naive Bayes, evaluating hyperparameter $k$, decision boundary geometry, and ROC metrics on synthetic biological measurements. |
| [`naive_bayes_vs_logreg_tutorial.ipynb`](naive_bayes_vs_logreg_tutorial.ipynb) | Rigorous from-scratch tutorial contrasting Gaussian Naive Bayes and Logistic Regression on the Wisconsin Breast Cancer dataset. Verifies both implementations against scikit-learn, compares the models over 100 random splits, measures probability quality (Brier score, log-loss, ECE with bootstrap intervals), tests *why* Naive Bayes is overconfident with feature-correlation ablations, and repairs the probabilities with Platt scaling and isotonic regression. |

---

## Key References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer, Chapter 4 (Linear Models for Classification).
- Ng, A. Y. & Jordan, M. I. (2001). On discriminative vs. generative classifiers: A comparison of logistic regression and naive Bayes. *Advances in Neural Information Processing Systems (NeurIPS)* 14, 841–848.
- Wolberg, W. H., Street, W. N. & Mangasarian, O. L. (1995). Breast Cancer Wisconsin (Diagnostic) Data Set. *UCI Machine Learning Repository*.
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer, Chapters 4 & 13.
- Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*, 625–632.
- 02-620 *Machine Learning for Scientists* course materials, Lectures 3–4 (Classification with k-NN/Naive Bayes; Logistic Regression).
