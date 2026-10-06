# Machine Learning for Scientists — Tutorial Series

**Course:** Carnegie Mellon University 02-620 *Machine Learning for Scientists*  
**Prerequisites:** Python, NumPy, linear algebra (vectors/matrices), basic calculus (gradients), and probability fundamentals.

This tutorial series covers the foundational mathematical and computational frameworks of machine learning tailored for scientific and biomedical applications. Rather than treating models as black-box predictors, these modules develop learning theory from first principles, connect loss functions to probabilistic priors, and apply regularized algorithms to high-dimensional biological data.

---

## Curriculum Overview

| # | Topic / Subfolder | Core Focus | Notebooks |
|---|-------------------|------------|-----------|
| 01 | [`FundamentalsLearningUncertainty`](FundamentalsLearningUncertainty/) | Supervised vs. unsupervised paradigms, Gaussian noise modeling, MLE vs. MAP parameter estimation, objective landscapes, and asymptotic convergence | `fundamentals of Learning & Uncertainty.ipynb` |
| 02 | [`RegularizedLinearRegression`](RegularizedLinearRegression/) | High-dimensional regression ($p \gg n$), OLS rank deficiency, Ridge ($L_2$/Gaussian prior), Lasso ($L_1$/Laplace prior/coordinate descent), regularization paths, and genomic prediction | `Linear regression methods.ipynb`<br>`Ridge_and_Lasso_for_High_Dimensional_Genomics.ipynb`<br>`ridge_lasso_genomic_prediction_tutorial.ipynb` |

---

## Module Summaries

### Module 01 — Fundamentals of Learning & Uncertainty

Scientific datasets present unique statistical challenges: measurements are expensive, sensor noise can be substantial, and sample sizes are often small. This module establishes the mathematical grounding for machine learning in science:
- **Task Categorization:** Formulating supervised conditional prediction $P(Y \mid X)$ vs. unsupervised latent distribution modeling $P(X)$.
- **Parameter Estimation:** Deriving Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) estimation under Gaussian assumptions.
- **Prior Beliefs as Regularizers:** Demonstrating how prior distributions prevent severe overfitting when sample size $N$ is small, and visualizing objective landscapes.
- **Asymptotic Convergence:** Simulating convergence across sample sizes from $N=1$ to $N=150$, showing empirical evidence dominating the prior as $N \to \infty$.

### Module 02 — Regularized Linear Regression

Modern scientific studies frequently operate in the high-dimensional regime ($p \gg n$), such as whole-genome sequencing panels or transcriptomic profiles measured across limited patient cohorts. In this regime, ordinary least squares fails because $X^\top X$ is singular and non-invertible:
- **Theory & Mathematics:** Establishing the rank deficiency of OLS, the closed-form $L_2$ shrinkage of Ridge regression, and the sparsity-inducing $L_1$ geometry of Lasso regression.
- **Algorithms from Scratch:** Implementing closed-form Ridge solvers and cyclic coordinate descent with soft-thresholding without external libraries.
- **Regularization Paths:** Tracking coefficient trajectories as penalty hyperparameter $\lambda$ sweeps from zero to infinity.
- **Genomic Application:** Predicting quantitative phenotypes from molecular markers using both controlled pathway simulations and real published CIMMYT wheat genotype data.

---

## Setup & Dependencies

```bash
pip install numpy scipy matplotlib pandas scikit-learn
```
