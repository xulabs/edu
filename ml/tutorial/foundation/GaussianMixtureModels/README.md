# Topic 06 — Gaussian Mixture Models & Expectation-Maximization

## Overview

Biologists and data scientists frequently encounter data where latent sub-populations are real but unlabeled: cell types in single-cell transcriptomics, bacterial strains in environmental samples, patient subtypes in clinical cohorts, or species in an ecological survey. When clusters overlap, vary in volume, or exhibit non-spherical geometric orientations, hard-clustering algorithms like $k$-means fail because they assume isotropic, equal-variance clusters.

**Gaussian Mixture Models (GMMs)** resolve this by modeling the data distribution as a weighted sum of $K$ multivariate Gaussian components:
$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k), \qquad \sum_{k=1}^K \pi_k = 1, \quad \pi_k \ge 0$$
where $\pi_k$ are the mixture weights, $\mu_k$ are component mean vectors, and $\Sigma_k$ are covariance matrices.

Because the true component assignment $z_i \in \{1, \dots, K\}$ for each sample is unobserved (latent), the model parameters cannot be solved in closed form. Instead, they are fit using the **Expectation-Maximization (EM)** algorithm (Dempster et al., 1977):

1. **E-step (Expectation):** Compute the posterior probability (responsibility) that component $k$ generated observation $x_i$ using Bayes' theorem:
   $$\gamma_{ik} = P(z_i = k \mid x_i) = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$
2. **M-step (Maximization):** Update parameters via responsibility-weighted Maximum Likelihood Estimation:
   $$N_k = \sum_{i=1}^n \gamma_{ik}, \qquad \pi_k^{\text{new}} = \frac{N_k}{n}, \qquad \mu_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} x_i$$
   $$\Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} (x_i - \mu_k^{\text{new}})(x_i - \mu_k^{\text{new}})^\top + \epsilon I$$
3. **Guaranteed Monotonicity:** Each iteration of EM is guaranteed to increase (or keep constant) the incomplete log-likelihood $\ln p(X \mid \theta) = \sum_{i=1}^n \ln \sum_{k=1}^K \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)$.
4. **Covariance Flexibility:** Spherical ($\sigma_k^2 I$), Diagonal ($\text{diag}(\sigma_{k,1}^2, \dots)$), and Full ($\Sigma_k$) covariances balance geometric flexibility against degrees of freedom.
5. **Model Selection via BIC:** Choosing the number of components $K$ and covariance structure by penalizing model complexity:
   $$\text{BIC} = -2 \ln \widehat{L} + p_{\text{params}} \ln n$$

---

## What You Will Learn

- **Latent-Variable Density Modeling:** Deriving the marginal likelihood and understanding why direct gradient descent or closed-form MLE is intractable without latent indicators.
- **The EM Algorithm Mechanics:** Alternating between posterior expectation (responsibilities) and closed-form weighted maximization.
- **$k$-means as a Hard Limit:** Proving that $k$-means is mathematically equivalent to a GMM with spherical covariances $\sigma^2 I$ in the zero-variance limit $\sigma^2 \to 0$.
- **Numerical Stability in High Dimensions:** Implementing log-sum-exp stabilization, Cholesky factorizations for Mahalanobis distances, and ridge regularizers to prevent covariance singularity.
- **From-Scratch Implementation:** Building complete `KMeansScratch` and `GMMScratch` classes (supporting spherical, diagonal, and full covariances) with NumPy alone.
- **Real Ecological Discovery:** Applying GMMs to 333 wild Palmer penguins (Adelie, Chinstrap, Gentoo) with species labels hidden during fitting to evaluate whether the model re-discovers the true species partitions via the Adjusted Rand Index (ARI).
- **Model Selection & Soft Uncertainty:** Benchmarking BIC scans across component counts and using soft responsibilities to identify borderline, morphologically ambiguous specimens.

---

## Notebooks

| File | Description |
|------|-------------|
| [`gmm_penguin_species_tutorial.ipynb`](gmm_penguin_species_tutorial.ipynb) | End-to-end tutorial covering GMM theory, EM derivations from scratch, covariance geometry, BIC model selection, soft assignment diagnostics, and unsupervised species rediscovery on real Palmer Archipelago penguin measurements. |

---

## Key References

- Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1–22.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer, Chapter 9 (Mixture Models and EM).
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461–464.
- Gorman, K. B., Williams, T. D. & Fraser, W. R. (2014). Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins (genus *Pygoscelis*). *PLoS ONE*, 9(3), e90081.
- McLachlan, G. J. & Peel, D. (2000). *Finite Mixture Models*. John Wiley & Sons.
- 02-620 *Machine Learning for Scientists* course materials, Topic on Mixture Models & the Expectation-Maximization Algorithm.

