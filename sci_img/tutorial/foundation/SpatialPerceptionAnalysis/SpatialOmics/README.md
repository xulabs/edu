# Phase 02 · Spatial Perception & Analysis — Spatial Omics & Multiplexed Imaging

## Overview

Complex biological tissues are not unstructured suspensions of cells; they are spatially organized microenvironments where cell function, signaling, and disease pathology depend critically on *where* cells reside and *who sits next to whom*. Emerging spatial biology technologies—such as multiplexed immunofluorescence (MxIF), cyclic immunofluorescence (CyCIF), imaging mass cytometry (IMC), CODEX/PhenoCycler, and spatial transcriptomics—enable simultaneous measurement of dozens of molecular markers with single-cell spatial resolution.

This module introduces the mathematical, computational, and statistical principles required to extract biological insight from single-cell spatial coordinate data:

1. **Synthetic Tissue Slide Modeling:** Simulating realistic spatial tissue architectures with distinct histological compartments—such as dense tumor cores, infiltrating immune cells (T-cells) at the tumor margin, and surrounding stromal matrices.
2. **Spatial Neighborhood Graph Construction:** Moving beyond isolated cell analyses by modeling spatial tissue topology as a graph $G = (V, E)$, where nodes $V$ represent individual cells with spatial coordinates $(x_i, y_i)$, and edges $E$ connect cells within a biologically defined physical interaction radius $r$ or nearest neighbors:
   $$\mathcal{N}_r(i) = \{j \in V \mid \|(x_i, y_i) - (x_j, y_j)\|_2 \le r, \; j \ne i\}$$
3. **Microenvironment Composition Vectors:** Quantifying the local cellular niche of each individual cell by calculating its neighborhood composition vector—the proportions of neighboring cells belonging to each cell phenotype:
   $$c_{ik} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbb{I}(\text{type}_j = k)$$
4. **Spatial Interaction & Co-localization Analysis:** Quantifying cell-cell spatial interactions across the entire tissue by computing a global interaction matrix $M \in \mathbb{R}^{K \times K}$, measuring whether specific cell types preferentially contact or avoid each other.
5. **Niche Architecture & Spatial Visualization:** Mapping cell coordinates alongside interaction heatmaps to delineate structural zones and immune infiltration dynamics.

---

## What You Will Learn

- **Spatial Tissue Representation:** How to represent multiplexed single-cell imaging experiments as spatial point processes with categorical phenotypes.
- **Graph Topology & Metric Searches:** Efficient spatial graph construction using metric spatial data structures (`ball_tree` / `KDTree`) for fixed-radius and $k$-NN neighbor queries.
- **Microenvironment Feature Engineering:** Converting continuous coordinates and discrete cell labels into high-dimensional neighborhood composition profiles.
- **Cell-Cell Contact & Co-localization:** Deriving and interpreting average neighborhood interaction matrices to quantify homotypic clustering versus heterotypic infiltration.
- **Tissue Niche Visualization:** Generating coordinated spatial tissue plots and interaction heatmaps to communicate spatial biology findings.

---

## Notebooks

| File | Description |
|------|-------------|
| [`Multiplexed Imaging and Spatial Omics.ipynb`](Multiplexed%20Imaging%20and%20Spatial%20Omics.ipynb) | Hands-on tutorial simulating multiplexed tissue slides, constructing spatial neighborhood graphs, computing local microenvironment composition vectors, and quantifying cell-cell interaction niches. |

---

## Key References

- Schürch, C. M. et al. (2020). Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front. *Cell*, 182(5), 1341–1359.
- Goltsev, Y. et al. (2018). Deep profiling of mouse splenic architecture with CODEX multiplexed imaging. *Cell*, 174(4), 968–981.
- Jackson, H. W. et al. (2020). The single-cell pathology landscape of breast cancer. *Nature*, 578(7796), 615–620.
- Lin, J. R. et al. (2018). Highly multiplexed immunofluorescence imaging of human tissues and tumors using t-CyCIF. *eLife*, 7, e31657.
- Rao, A., Barkley, D., França, G. S. & Yanai, I. (2021). Exploring tissue architecture using spatial transcriptomics. *Nature*, 596(7871), 211–220.
- CMU 02-740 *Bioimage Informatics* course materials, Frontier Module on Spatial Omics and Multiplexed Imaging.

