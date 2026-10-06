# Topic 01 — Fundamentals of Learning & Uncertainty

## Overview

Applying machine learning to scientific discovery requires a rigorous grasp of the foundational principles governing data generation, model formulation, and parameter uncertainty. Scientific data—unlike internet-scale consumer data—is frequently constrained by expensive measurement processes, high sensor noise, and small sample sizes. 

This module explores the core mathematical foundations of machine learning for scientific problems:
1. **Supervised vs. Unsupervised Learning Paradigms:** Distinguishing between conditional target prediction $P(Y \mid X)$ and data structure discovery $P(X)$.
2. **Probabilistic Modeling:** Formulating physical observations as realizations of underlying probability distributions corrupted by measurement noise.
3. **Parameter Estimation Frameworks:** Implementing and contrasting **Maximum Likelihood Estimation (MLE)** and **Maximum A Posteriori (MAP)** estimation.
4. **Objective Landscapes & Convergence:** Analyzing how prior beliefs regularize solutions in small-sample regimes ($N < 20$) and how MAP estimates asymptotically converge to MLE as sample evidence grows ($N \to \infty$).

---

## What you will learn

- **Learning Paradigms:** Mathematical distinction between supervised regression/classification and unsupervised structure/distribution modeling.
- **Likelihood Formulation:** Expressing data likelihood $P(D \mid \theta)$ under Gaussian observational assumptions and deriving closed-form estimators.
- **Prior Distributions as Regularizers:** Incorporating domain priors $P(\theta)$ into the estimation process to constrain parameters when empirical evidence is sparse.
- **MAP Optimization:** Deriving and visualizing the posterior landscape $P(\theta \mid D) \propto P(D \mid \theta) P(\theta)$ and tracking how the peak balances empirical observations against baseline assumptions.
- **Asymptotic Convergence:** Quantifying estimator variance and observing the empirical transition where likelihood overwhelms the prior as $N \to \infty$.

---

## Notebooks

| File | Description |
|------|-------------|
| [`fundamentals of Learning & Uncertainty.ipynb`](fundamentals%20of%20Learning%20%26%20Uncertainty.ipynb) | Hands-on tutorial covering supervised vs. unsupervised task formulation, Gaussian probabilistic modeling, MLE vs. MAP parameter estimation, and asymptotic convergence analysis. |

---

## Key References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer, Chapters 1 & 2.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press, Chapters 2 & 3.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. & Rubin, D. B. (2013). *Bayesian Data Analysis*. CRC Press.
- 02-620 *Machine Learning for Scientists* course materials, Topic 1 (Fundamentals of Learning & Uncertainty).
