# Topic 02 — Regularized Linear Regression

## Overview

In biological and biomedical applications, datasets routinely have far more measured features than experimental observations ($p \gg n$)—such as thousands of gene expression levels or millions of single nucleotide polymorphisms (SNPs) scored across tens or hundreds of patients. Under this high-dimensional regime, the ordinary least squares (OLS) normal equation matrix $X^\top X$ is rank-deficient and singular. OLS produces unstable, non-unique solutions that fit training noise perfectly (memorization) but fail to generalize to new individuals.

This module provides a comprehensive treatment of **regularized linear regression** methods designed to resolve the $p \gg n$ challenge:
1. **Ordinary Least Squares (OLS) Failure:** Proving singularity and rank deficiency when $p > n$ and observing severe test overfitting.
2. **Ridge Regression ($L_2$ Regularization):** Adding a quadratic penalty $(X^\top X + \lambda I)^{-1}X^\top y$ to restore full rank, smoothly shrinking coefficients toward zero, and stabilizing predictions under correlated feature structures (equivalent to MAP estimation under an isotropic Gaussian prior).
3. **Lasso Regression ($L_1$ Regularization):** Introducing an absolute value penalty to induce exact sparsity via soft-thresholding, performing automated feature selection in sparse biological architectures (equivalent to MAP estimation under an independent Laplace prior).
4. **Hyperparameter Selection:** Using $K$-fold cross-validation and validation sweeps to trace regularization paths without test set contamination.
5. **From-Scratch & Real-Data Evaluation:** Benchmarking closed-form solvers and cyclic coordinate descent on both synthetic pathway simulations and real, published genotype-phenotype data (CIMMYT international wheat breeding trial).

---

## What you will learn

- **The $p \gg n$ Breakdown:** Why sample covariance matrices lose invertibility in high dimensions and how pseudo-inverses overfit.
- **Ridge Regression Mathematics:** Derivation of the closed-form estimator, relationship to Bayesian MAP with a Gaussian prior, and variance reduction.
- **Lasso Regression & Coordinate Descent:** Why the non-differentiable $L_1$ corner forces exact zeros; deriving the soft-thresholding update $S_\lambda(z) = \operatorname{sign}(z)\max(|z| - \lambda, 0)$.
- **Regularization Paths:** Visualizing continuous weight shrinkage ($L_2$) versus progressive variable elimination ($L_1$) across penalty strengths.
- **Cross-Validation Best Practices:** Preventing data leakage by splitting prior to standardization and selecting $\lambda$ strictly on training folds.
- **Genomic Prediction:** Translating high-dimensional marker matrices into quantitative phenotype predictions, comparing against production methods (GBLUP, BayesA/B/R, `glmnet`).

---

## Notebooks

| File | Description |
|------|-------------|
| [`Linear regression methods.ipynb`](Linear%20regression%20methods.ipynb) | Introductory tutorial implementing OLS, Ridge, and Lasso on synthetic biological data, comparing test MSE and visualizing regularization paths using `scikit-learn`. |
| [`Ridge_and_Lasso_for_High_Dimensional_Genomics.ipynb`](Ridge_and_Lasso_for_High_Dimensional_Genomics.ipynb) | Detailed study of high-dimensional genomic regression ($n=120, p=200$) with correlated pathway blocks, cross-validated Ridge, and cyclic coordinate descent Lasso. |
| [`ridge_lasso_genomic_prediction_tutorial.ipynb`](ridge_lasso_genomic_prediction_tutorial.ipynb) | Complete from-scratch tutorial predicting wheat grain yield from 1,279 DArT molecular markers using the published CIMMYT dataset, tracing full regularization paths and validation $R^2$. |

---

## Key References

- Hoerl, A. E. & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics* 12(1), 55–67.
- Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. *Journal of the Royal Statistical Society: Series B* 58(1), 267–288.
- Crossa, J. et al. (2010). Prediction of genetic values of quantitative traits in plant breeding using genomic markers. *Genetics* 186(2), 713–724.
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer.
- 02-620 *Machine Learning for Scientists* course materials, Topic 2 (Regularized Linear Regression).
