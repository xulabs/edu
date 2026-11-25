# Frontier map: research areas → Level 2 modules

Level 2 of BiologyToML (Modules 24–43) carries the ideas of the 02-620 core (Modules 00–23) to
current computational-biology research. Use this map in both directions: when you meet a research
area or a method in a paper, find the module that teaches its foundations; when you finish a
module, find where its ideas are used and what to read and try next.

Every Level 2 module (i) explains the biology and the assay, (ii) implements the core method from
scratch on simulated data with a known truth, (iii) checks the method against that truth
(🧪 boxes), (iv) provokes its failure modes, and (v) ends with a 🧭 *From toy to real data*
section and 📄 *Research connection* boxes naming landmark papers, production tools and open
questions. Statements about the state of a field are as of 2025.

---

## 1 · How Level 2 builds on Level 1

| Level 1 idea (module) | becomes, in Level 2 … (module) |
|---|---|
| Logistic regression as a likelihood model (09); MLE/MAP (05); shrinkage (07, 21) | negative-binomial GLMs, offsets, empirical-Bayes dispersion shrinkage, FDR control (26) |
| EM with an exact E-step (13); posteriors (21) | variational inference, the ELBO, amortised inference, VAEs and a mini-scVI (27) |
| k-NN as weighted averaging of neighbours (08) | attention as a learned soft nearest-neighbour average; transformers (24) |
| Mixture densities (13); MCMC and Langevin-type samplers (21) | diffusion models, score matching and flow matching (28) |
| Message passing / belief propagation on graphs (17) | graph neural networks; equivariant networks for 3-D structures (25) |
| Ridge as a Gaussian prior (07); kernels (11) | Gaussian processes, Bayesian optimisation and active learning (29) |
| Clustering (12), mixtures (13), PCA (14) | single-cell clustering, annotation and embeddings (30); batch integration (31) |
| Markov chains (19–20); eigenvectors (14) | diffusion pseudotime, RNA velocity, absorbing Markov chains for cell fate (32) |
| Gaussian graphical models and neighbourhood dependence (18) | spatial autocorrelation, spatial domains, niches (33); Potts models and direct-coupling analysis (37) |
| PCA/SVD (14), regression (06–07) | CCA, weighted nearest neighbours, cross-modal prediction (34) |
| GWAS by regression (06), ridge/lasso (07), PCA for ancestry (14) | linear mixed models, heritability, fine-mapping, polygenic scores, Mendelian randomisation (35) |
| A CNN that finds a DNA motif (22); position-specific emissions (19) | sequence-to-function models, attribution, variant-effect prediction (36) |
| Rotations and SVD (03), eigendecomposition (14) | Kabsch superposition, multidimensional scaling, contact-guided folding (38) |
| Simulated annealing / MCMC (21) | generative and model-based design of proteins and molecules (39) |
| Random forests (10), grouped cross-validation (23) | QSAR, molecular GNNs, scaffold splits, applicability domain (40) |
| Bayesian networks, d-separation, structure learning (15–17) | causal graphs, adjustment, interventions, Perturb-seq network learning (41) |
| Cross-validation, selection bias, bias–variance (23) | leakage, distribution shift, calibration, conformal prediction (42) |
| Markov chains and n-grams (19–20), neural networks (22) | language models, retrieval-augmented generation, AI agents (43) |

## 2 · Research areas

### 🔬 Single-cell and spatial biology

| Area | Module | Landmark ideas and methods | Production tools · data to try next |
|---|---|---|---|
| Counts, differential expression, FDR | [26](notebooks/26_glm_count_data.ipynb) | negative-binomial GLMs (Anders & Huber 2010; Robinson et al. 2010; Love et al. 2014); Benjamini–Hochberg; pseudobulk vs pseudoreplication (Squair et al. 2021) | DESeq2, edgeR, limma-voom, glmGamPoi · GTEx, TCGA, recount3 |
| Probabilistic latent spaces of cells | [27](notebooks/27_variational_inference_vae.ipynb) | variational autoencoders (Kingma & Welling 2014); scVI (Lopez et al. 2018); scANVI, totalVI | scvi-tools · CELLxGENE |
| QC, normalisation, clustering, annotation | [30](notebooks/30_single_cell_rnaseq.ipynb) | best practices (Luecken & Theis 2019; Heumos et al. 2023); Pearson residuals; Louvain/Leiden; marker-based annotation | Scanpy, Seurat · 10x PBMC data sets, Human Cell Atlas, CELLxGENE |
| Batch effects, atlases, reference mapping | [31](notebooks/31_batch_integration.ipynb) | ComBat (Johnson et al. 2007); MNN (Haghverdi et al. 2018); Harmony (Korsunsky et al. 2019); scIB benchmark (Luecken et al. 2022); scArches | Harmony, scVI/scANVI, Seurat, Scanorama, Azimuth |
| Trajectories, RNA velocity, fate | [32](notebooks/32_cell_dynamics_velocity.ipynb) | diffusion pseudotime (Haghverdi et al. 2016); RNA velocity (La Manno et al. 2018; Bergen et al. 2020) and its limits (Bergen et al. 2021); Waddington-OT (Schiebinger et al. 2019); CellRank | scVelo, CellRank, moscot, Monocle, PAGA |
| Spatial transcriptomics | [33](notebooks/33_spatial_omics.ipynb) | Visium/ST, Slide-seq, MERFISH; Moran's I and spatially variable genes; spatial domains; deconvolution (RCTD, cell2location); communication analysis and its caveats | Squidpy, Giotto, cell2location, RCTD, Tangram · 10x Visium/Xenium data |
| Multimodal single cells | [34](notebooks/34_multimodal_integration.ipynb) | CITE-seq (Stoeckius et al. 2017); scATAC; weighted nearest neighbours (Hao et al. 2021); totalVI; MOFA+; enhancer–gene links | Seurat WNN, Signac, ArchR, MOFA+, scvi-tools |
| Perturbations and causality | [41](notebooks/41_perturbation_causal_inference.ipynb) | Perturb-seq (Dixit et al. 2016; Adamson et al. 2016; Replogle et al. 2022); combinatorial screens (Norman et al. 2019); response prediction and the strength of linear baselines (Ahlmann-Eltze et al. 2025) | Mixscape, GEARS, CPA, MAGeCK |

### 🧬 Genomes and genetics

| Area | Module | Landmark ideas and methods | Production tools · data to try next |
|---|---|---|---|
| Biobank GWAS, structure, heritability | [35](notebooks/35_statistical_genetics.ipynb) | PCs for stratification (Price et al. 2006); linear mixed models and SNP heritability (Yang et al. 2010; Lippert et al. 2011; Loh et al. 2015); LD-score regression | PLINK, REGENIE, BOLT-LMM, GCTA · UK Biobank, FinnGen, All of Us, GWAS Catalog |
| Fine-mapping | [35](notebooks/35_statistical_genetics.ipynb) | Bayesian credible sets (Maller et al. 2012); SuSiE (Wang et al. 2020) | SuSiE, FINEMAP |
| Polygenic scores and equity | [35](notebooks/35_statistical_genetics.ipynb), [42](notebooks/42_trustworthy_ml.ipynb) | clumping + thresholding, BLUP; portability across ancestries (Martin et al. 2019) | PRS-CS, LDpred2 |
| Mendelian randomisation | [35](notebooks/35_statistical_genetics.ipynb), [41](notebooks/41_perturbation_causal_inference.ipynb) | instrumental variables; IVW, MR-Egger (Bowden et al. 2015) | TwoSampleMR |
| Regulatory DNA, variant effects | [36](notebooks/36_regulatory_genomics_deep_learning.ipynb) | DeepBind, DeepSEA, Basset; BPNet and motif syntax; Enformer-type long-range models; attribution (in-silico mutagenesis, gradient × input); limits on personal genomes | TF-MoDISco, Kipoi · ENCODE, JASPAR |
| Genomic language models | [24](notebooks/24_attention_transformers.ipynb), [36](notebooks/36_regulatory_genomics_deep_learning.ipynb) | DNABERT, Nucleotide Transformer, Evo (self-supervised DNA models) | Hugging Face model hubs |

### 🧪 Proteins and molecules

| Area | Module | Landmark ideas and methods | Production tools · data to try next |
|---|---|---|---|
| Coevolution, contacts, variant effects | [37](notebooks/37_protein_sequences_language_models.ipynb) | direct-coupling analysis (Weigt et al. 2009; Morcos et al. 2011; Ekeberg et al. 2013); EVmutation, DeepSequence, EVE; deep mutational scanning | EVcouplings, HHblits/jackhmmer · Pfam, UniRef, ProteinGym, MaveDB |
| Protein language models | [24](notebooks/24_attention_transformers.ipynb), [37](notebooks/37_protein_sequences_language_models.ipynb) | ESM (Rives et al. 2021; Lin et al. 2023); attention tracks contacts (Rao et al. 2021); zero-shot variant effects (Meier et al. 2021) | ESM-2, ESM-1v |
| Structure representation and prediction | [38](notebooks/38_protein_structure.ipynb) | Kabsch, TM-score, lDDT; contact-guided folding; AlphaFold 2 (Jumper et al. 2021), RoseTTAFold, ESMFold, AlphaFold 3 | ColabFold, Foldseek, ChimeraX · PDB, AlphaFold DB, CASP |
| Generative models of structure | [28](notebooks/28_diffusion_flow_matching.ipynb), [39](notebooks/39_generative_design.ipynb) | DDPM, score-based models, flow matching; RFdiffusion, Chroma, AlphaFold 3's diffusion module | RFdiffusion, ProteinMPNN |
| Protein and molecule design | [39](notebooks/39_generative_design.ipynb) | Potts-model design (Russ et al. 2020); inverse folding (ProteinMPNN); model-based optimisation pitfalls; molecular generation benchmarks; responsible design | ProteinMPNN, RFdiffusion · GuacaMol, MOSES |
| Experiment design, protein engineering | [29](notebooks/29_gaussian_processes_bayesian_optimization.ipynb) | Gaussian processes and Bayesian optimisation; ML-guided directed evolution (Romero et al. 2013; Wu et al. 2019) | BoTorch, GPyTorch · FLIP, ProteinGym landscapes |
| Graphs and geometry | [25](notebooks/25_graph_neural_networks.ipynb) | GCN, MPNN, GAT, GIN; equivariant networks (EGNN); GNN-driven antibiotic discovery (Stokes et al. 2020) | PyTorch Geometric, DGL |
| Molecular property prediction | [40](notebooks/40_molecular_ml_drug_discovery.ipynb) | fingerprints (ECFP), MoleculeNet, Chemprop; scaffold splits; activity cliffs; virtual screening | RDKit, Chemprop, DeepChem · ChEMBL, PubChem, ZINC, TDC, Polaris |

### 🤖 Methods for reliable, AI-driven science

| Area | Module | Landmark ideas and methods | Production tools · data to try next |
|---|---|---|---|
| Probabilistic deep learning | [27](notebooks/27_variational_inference_vae.ipynb), [28](notebooks/28_diffusion_flow_matching.ipynb) | ELBO, reparameterisation, VAEs; diffusion and flow matching | PyTorch, Pyro |
| Transformers and foundation models | [24](notebooks/24_attention_transformers.ipynb), [43](notebooks/43_llms_ai_agents.ipynb) | attention, masked and causal language modelling, scaling; single-cell foundation models and their evaluation against simple baselines | Hugging Face |
| Trustworthy ML | [42](notebooks/42_trustworthy_ml.ipynb) | leakage and proper splits (Whalen et al. 2022; Kapoor & Narayanan 2023); DOME recommendations (Walsh et al. 2021); calibration; deep ensembles; conformal prediction | scikit-learn, MAPIE · [ConformalPrediction tutorial](../foundation/ConformalPrediction/) |
| Causal inference | [41](notebooks/41_perturbation_causal_inference.ipynb) | potential outcomes, backdoor adjustment, propensity scores, IV | DoWhy, causal-learn |
| LLMs, retrieval and agents | [43](notebooks/43_llms_ai_agents.ipynb) | retrieval-augmented generation (Lewis et al. 2020); ReAct agents (Yao et al. 2023); AI systems for chemistry and biology (Boiko et al. 2023; Bran et al. 2024; Swanson et al. 2025); responsible use and dual-use risk | LLM APIs with tool use — always with verification |
| Research practice | [44](notebooks/44_capstone_project.ipynb) | simulation-based benchmarking, power analysis, DOME, model cards, replication | — |

## 3 · Open questions you will meet (as of 2025)

- Do large foundation models of cells, DNA and proteins beat simple, well-tuned baselines on the
  tasks that matter? (24, 41, 43)
- How far does RNA velocity's steady-state logic hold, and which ground truths (lineage tracing,
  metabolic labelling) can validate trajectories? (32)
- Can integration remove batch effects without removing disease biology? (31)
- How do we predict the effects of unseen genetic perturbations and their combinations? (41)
- Do polygenic scores work equally well across ancestries, and how should they be used
  clinically? (35, 42)
- How well do sequence-to-function models predict the effects of variants in individual genomes? (36)
- How reliable are zero-shot variant-effect scores for clinical interpretation? (37)
- How do we design proteins and molecules that work in the lab, not just in silico — and govern
  the dual-use risks? (39, 40)
- How should AI agents in science be evaluated, verified and overseen? (43)
