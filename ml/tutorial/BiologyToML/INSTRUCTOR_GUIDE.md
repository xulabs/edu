# 🧑‍🏫 Instructor guide

For faculty, TAs and mentors who use **BiologyToML** alongside CMU 02-620 *Machine Learning for
Scientists*, as a bridge course for students entering computational biology, as a second course
on machine learning for biological research, or as structured self-study for undergraduate
researchers. Students should start from the [README](README.md). This guide covers formats,
learning outcomes, the map from the 02-620 lectures to the modules, how Level 2 supports a
modernised 02-620, per-module teaching notes, assessment, slide errata, and maintenance.

**Design principles.** (1) *Self-contained:* the notebooks teach the lecture content themselves —
definitions, derivations, worked examples, figures (re-created with code) and the "what you should
know" points — so students never need the slides; 📘 *Lecture notes* boxes summarise each part of
a lecture. (2) *Synthetic data with a known truth:* the repository stores no data or pictures;
every data set is simulated (see [`DATA.md`](DATA.md)), which lets every method be checked against
the truth (🧪 boxes). (3) *Implement, then use:* every algorithm is written from scratch before
production tools are named.

---

## 1 · Formats that work

| Format | Plan | Notes |
|---|---|---|
| Companion to 02-620 (Level 1) | Part A before or in weeks 1–2, then the lecture-aligned schedule in the [README](README.md#following-02-620-week-by-week) | recitations can run the module's key section live; ✏️ exercises make good warm-up problems; 📘 boxes are ready-made revision sheets |
| A modernised 02-620 (Levels 1 + 2) | the week-by-week map in §4 | Level 2 modules cover count models, VAEs, attention, single-cell, sequence and protein ML, perturbations, uncertainty, generative models and experiment design |
| Second course / graduate elective (Level 2) | Part F (24–29) in weeks 1–6, then one or two application parts (G, H or I) and the capstone tracks | assumes Level 1; each module is ~3 h of guided work plus exercises |
| Bridge for incoming students | Modules 00–14 over a summer (≈ 7 h/week for 10 weeks), Parts D–E during the semester | front-loads the mathematics most students struggle with in weeks 2–6 |
| Lab onboarding / rotations | Part A, the Level 1 part matching the student's project, then the matching Level 2 track and a capstone track | pair with the topic tutorials in [`ml/tutorial/foundation`](../foundation/README.md) |

**Before week 1.** Ask students to open Module 00 in Colab and run it top to bottom (≈ 1 minute)
to catch account and browser problems early. No downloads are needed: every data set is
generated in the notebook.

---

## 2 · Learning outcomes by part

| Part | Students can… | Evidence |
|---|---|---|
| A · Foundations (00–05) | write NumPy code on genotype and expression matrices; compute joint/conditional probabilities and apply Bayes' rule; multiply, invert and eigendecompose matrices; differentiate and run gradient descent; derive MLE and MAP estimates | lab notes; checkpoints; a hand derivation of the MLE of an allele frequency and of a Beta-prior MAP |
| B · Supervised learning (06–11) | derive least squares, ridge, lasso, naïve Bayes, logistic regression, information gain and the SVM margin/dual; implement each; evaluate with precision/recall and a held-out set | derivations on paper; a toy GWAS; an ALL/AML classifier with an honest test error |
| C · Unsupervised learning (12–14) | run and interpret k-means, hierarchical clustering, EM for mixtures and PCA; choose K; recognise degenerate EM solutions; read a PCA plot critically | class discovery of the leukaemia subtypes with an ARI; a PCA "map" with a written caveat |
| D · Graphical models (15–20) | factorise and count parameters; decide d-separation; run enumeration, variable elimination and sampling; learn CPTs, Chow–Liu trees and BIC-scored DAGs; read zeros of a precision matrix; run forward–backward, Viterbi and Baum–Welch | the 14 d-separation examples; P(B∣J,¬M) by hand; a learned signalling network with precision/recall against the true graph |
| E · Bayesian, neural, model selection (21–23) | derive conjugate posteriors; write and diagnose a Gibbs sampler; back-propagate by hand; derive the bias–variance decomposition; run nested CV without leakage | checkpoint derivations; a leakage-free CV analysis |
| F · Modern ML building blocks (24–29) | fit NB GLMs with offsets and control the FDR; derive the ELBO and train a VAE; implement attention and a small transformer; train diffusion and flow-matching models; implement message passing and an equivariant layer; run GP regression and a Bayesian-optimisation loop | ground-truth checks (FDR, latent-space accuracy, contact precision, sample quality, optimum found) |
| G · Cells and tissues (30–34) | run a single-cell workflow (QC → clusters → annotation → pseudobulk DE); integrate batches and map to a reference; infer pseudotime, velocity and fates; analyse spatial data; integrate RNA, protein and chromatin | ARI against true types; mixing vs conservation metrics; fate AUROC; deconvolution error |
| H · Genomes, proteins, molecules (35–40) | fit LMMs and estimate heritability; fine-map with credible sets; build and evaluate PRS and MR; train and interpret sequence CNNs; fit Potts models and score variants; compare and fold structures; design sequences and molecules safely; evaluate molecular models with scaffold splits | credible-set coverage; motif recovery; Spearman with DMS; TM-score; design quality by the oracle; R² under scaffold split |
| I · Causality, reliability, AI (41–43) | reason with causal graphs and interventions; analyse CRISPR screens and Perturb-seq; detect leakage and shift; calibrate and build conformal intervals; explain how LLMs, RAG and agents work and fail | estimates vs true effects; coverage of conformal sets; a guarded agent run |
| J · Capstone (44) | run a project from question to reproducible report; benchmark methods against a simulated truth; plan power; follow a reporting checklist | the capstone report, presentation and peer review |

---

## 3 · The 02-620 lectures → modules

The notebooks never point to slides, but instructors can use this map to align recitations with
lectures. Every substantive slide of each deck is taught in the listed modules (the authors
checked each deck slide by slide).

| 02-620 lecture (Spring 2025 deck) | Modules |
|---|---|
| Introduction: machine learning in science; supervised vs unsupervised; MLE vs MAP (`001_intro`) | 00, 05 (project requirements: 44) |
| Regression: GWAS, least squares, normal equations; MAP and ridge; lasso, elastic net (`002_regression`) | 06, 07 (previews in 03, 04) |
| Classification: k-NN, naïve Bayes (`003_classification-kNN-NB`) | 08 (MLE/MAP parts in 05) |
| Classification: logistic regression, generative vs discriminative (`004_classification-LR`) | 09 (gradient ascent in 04) |
| Decision trees and random forests (`005_decision-tree`) | 10 |
| Clustering: k-means, hierarchical (`006_clustering`) | 12 (distances in 03) |
| Support vector machines (`007_SVM`, extra slides) | 11 (Lagrangian duality in 04) |
| Bayesian networks: representation, independence, inference, learning (`008_BN`) | 15, 16, 17 |
| Gaussian graphical models (`009_GGM`) | 18 |
| Mixture models and EM (`010_mixture-model`) | 13 |
| PCA (`011_PCA`) | 14 (eigenvectors in 03, the variance-maximisation proof in 04) |
| HMMs and dynamic Bayesian networks (`012_HMM-DBN`) | 19, 20 |
| Bayesian learning and MCMC (`013_Bayesian-MCMC`) | 21 |
| Neural networks (`014_NN`) | 22 |
| Model selection (`015_model-selection`, extra slides) | 23 (and 07) |

---

## 4 · Level 2 and a modernised 02-620

The 2026 revision proposals for 02-620 shift time towards evaluation, count models, neural and
representation learning, single-cell, sequence and protein ML, perturbations, uncertainty and
generative modelling. The table maps a 15-week revised schedule of that kind to modules; core
material that the revision compresses stays available in Level 1 for self-study.

| Week | Topics | Modules |
|---|---|---|
| 1 | Biological questions and units of observation; probability, likelihood, priors; generalisation, leakage, reproducible splits | 00, 02, 05, 23, 42 (§2) |
| 2 | Linear regression, ridge/lasso/elastic net; count GLMs and replication | 06, 07, 26 |
| 3 | Logistic regression, k-NN, naïve Bayes; trees, forests; margins, kernels, model comparison | 08–11 |
| 4 | PCA and SVD; clustering and neighbour graphs; visualisation, stability, bias–variance | 12, 14, 23, 30 (§4–5) |
| 5 | Neural networks, optimisation, CNNs | 22, 36 (§3) |
| 6 | Mixtures and EM; graphical models and one HMM; variational inference and VAEs | 13, 15–16, 19, 27 |
| 7 | Single-cell count models and scVI; integration and reference mapping | 30, 31 |
| 8 | Attention; self-supervision and transfer; sequence-to-function and foundation models | 24, 36, 43 (§7) |
| 9 | Protein language models and fitness; structure and confidence; molecular property prediction | 37, 38, 40 |
| 10 | Multi-omics; graph learning; spatial omics | 34, 25, 33 |
| 11 | Causal questions and interventions; perturbation prediction; additive baselines | 41 (and 17–18) |
| 12 | Calibration, uncertainty, shift; interpretation; benchmark critique | 42, 21 |
| 13 | Conditional generation and diffusion; experimental design and active learning | 28, 29, 39 |
| 14 | Cell-state dynamics and trajectory assumptions; project audit | 32, 44 (Part 3) |
| 15 | Project presentations | 44 |

Electives in the revision proposals map to: statistical genetics (35), optimal transport and RNA
velocity (32), advanced graph geometry (25), Gaussian graphical models (18), full HMM learning (20).

---

## 5 · Per-module teaching notes

Each line gives the one idea to land, the most common difficulty, and a discussion prompt.

| Module | Idea to land | Typical difficulty | Discussion prompt |
|---|---|---|---|
| 00 | X, Y, f; supervised = P(Y∣X), unsupervised = P(X) | treating "AI" as one technique | For the Golub leukaemia study, what were X, Y and f — and what was *class discovery*? |
| 01 | arrays with `axis=`; vectorisation | off-by-one indexing; int8 overflow | Why does a seed make a simulation reproducible but not "true"? |
| 02 | conditional probability and conditional independence | confusing P(A∣B) with P(B∣A) | A carrier test is 99 % sensitive — why can most positives still be false? |
| 03 | Xw as a weighted sum; rank and invertibility | when N < J, why "infinitely many perfect fits" | Two SNPs in perfect LD: what can and can't the data tell us? |
| 04 | gradient = direction of steepest ascent; convexity | step-size intuition; matrix-calculus rules | Why is a concave log-likelihood good news? |
| 05 | likelihood as a function of θ; MAP = imaginary counts | "probability of the data" vs "probability of θ" | When should a prior dominate the data? |
| 06 | MLE under Gaussian noise = least squares | the matrix derivation of the normal equations | Univariate GWAS hits vs joint regression: why do they disagree under LD? |
| 07 | ridge = Gaussian prior; lasso = sparsity via soft-thresholding | why L1 gives exact zeros | Is a lasso-selected SNP causal? What would you need to claim that? |
| 08 | conditional independence turns 3ⁿ into 2n parameters | zero counts; NB probabilities are often over-confident | Why can NB classify well while its probabilities are wrong? |
| 09 | linear log-odds; gradient ascent on a concave likelihood | two sign conventions in textbooks | Generative or discriminative for 30 patients? For 30,000? |
| 10 | information gain; trees have high variance; forests average it away | conditional entropy bookkeeping | Why does a single noisy day (D15) reshape the tree? |
| 11 | margin = 2/‖w‖; support vectors; kernels as dot products | the dual and the meaning of αᵢ | "SVMs have no model selection" — what must still be chosen? |
| 12 | k-means as alternating minimisation; linkage choices | correlation vs Euclidean distance | Did clustering *discover* the leukaemia subtypes, or were the genes chosen to find them? |
| 13 | EM = alternate soft assignment and weighted MLE | why the likelihood can go to infinity | When are soft assignments worth the extra assumptions? |
| 14 | PCs are eigenvectors of the covariance; centring | `eigh` returns ascending order; reading loadings | What can a PCA "map of Europe" not tell you about ancestry? |
| 15 | joint = ∏ P(Xᵢ∣Pa); parameter counting | the chain rule as a complete DAG | What does an arrow in a gene network claim, and what not? |
| 16 | three triples; colliders open when observed | descendants of colliders | Collider bias in case-only studies: give a biological example. |
| 17 | variable elimination reuses work; Chow–Liu; BIC search | factor bookkeeping; Markov equivalence | Why did Sachs *et al.* need perturbation experiments? |
| 18 | Θᵢⱼ = 0 ⟺ conditional independence; graphical lasso | covariance vs precision; the moral graph | A partial correlation between p38 and JNK: regulation, or a shared upstream kinase? |
| 19 | forward recursion = law of total probability + Markov property | underflow; Viterbi vs posterior decoding | Why is the "tempting" path for the dice rolls not the most probable one? |
| 20 | Baum–Welch = EM with expected counts | label switching; local optima | How would you choose the number of hidden states for a genome? |
| 21 | posterior ∝ likelihood × prior; Gibbs samples conditionals | burn-in and autocorrelation | When is a full posterior worth more than a MAP estimate? |
| 22 | a neuron = logistic regression; back-propagation = chain rule | gradient shapes; overfitting with few samples | What did the CNN's filter learn, and how would you check it is biology? |
| 23 | bias–variance; CV for selection, a test set for reporting | selection bias (choosing genes before CV) | Why is "100 % CV accuracy" with pre-selected genes not a result? |
| 24 | attention = learned soft neighbours; self-supervision learns co-variation | positional information; what attention maps mean | Is a protein language model "understanding" structure? |
| 25 | message passing; symmetry (permutation, rotation) as design constraint | over-smoothing; equivariance vs invariance | When does adding a graph hurt (low homophily)? |
| 26 | counts need count models; offsets; the donor is the unit of replication | Poisson p-values for overdispersed data; FDR vs FWER | Why do cell-level tests "find" hundreds of genes that pseudobulk does not? |
| 27 | the ELBO; amortised inference; a likelihood that matches the data | KL direction; posterior collapse | Is a VAE latent axis a biological programme? How would you test it? |
| 28 | learn to denoise, then reverse; straight paths in flow matching | noise schedules; memorisation vs generalisation | What would convince you a generated protein is new and functional? |
| 29 | uncertainty drives exploration; acquisition functions | kernel choice; exploiting surrogate errors | With 96 wells left, explore or exploit? |
| 30 | QC → normalisation → graph → clusters → markers → pseudobulk | over-interpreting t-SNE/UMAP; resolution choices | Is a cluster a cell type? |
| 31 | mixing vs conservation; integration can remove biology | confounded designs; over-correction | What should an atlas do with a cell type present in only one batch? |
| 32 | snapshots + models → dynamics; velocity's assumptions | steady-state bias; unpredictable early fates | What experiment would validate an inferred trajectory? |
| 33 | location as data: autocorrelation, domains, niches, mixtures in spots | co-localisation ≠ communication; segmentation spillover | What would prove a ligand–receptor interaction in tissue? |
| 34 | some states live in one modality; weight modalities per cell | normalising ADT and ATAC; between-type confounding of links | When is measuring protein worth the cost? |
| 35 | relatedness, LD and ancestry shape every genetic result | heritability vs prediction; credible sets vs lead SNPs | Should polygenic scores be used in the clinic? For whom? |
| 36 | CNN filters ≈ motifs; attribution recovers grammar | chromosome hold-out; partly learned interactions | How far can a sequence model predict personal variant effects? |
| 37 | conservation vs co-variation; direct vs indirect couplings | APC; pseudo-likelihood; global epistasis | Would you trust a zero-shot score for a patient's variant? |
| 38 | structures compared after superposition; distances → coordinates | mirror images; RMSD vs TM-score vs lDDT | What does a high-confidence AlphaFold model not tell you? |
| 39 | design = inverse problem; surrogates fail out of distribution | trust regions, pessimism; diversity | How should dual-use risks of design tools be governed? |
| 40 | representation + split decide the claim; applicability domain | random vs scaffold split; activity cliffs | Which molecule would you synthesise next, and why? |
| 41 | intervention ≠ observation; adjustment; IV; learning from knockdowns | colliders; guide escape; additive baselines | Why are simple baselines hard to beat for perturbation prediction? |
| 42 | leakage, shift, calibration, conformal guarantees | marginal vs conditional coverage | Audit a published benchmark with the DOME checklist. |
| 43 | next-token prediction; grounding; tool use with guardrails | hallucination; evaluating agents | What must a human always verify in an AI-assisted analysis? |
| 44 | one question, an evaluation contract, honest uncertainty, reproducibility | comparing methods within noise; power | Which method would you report, and what would you tell a clinician? |

---

## 6 · Assessment

- **Checkpoints** (✅, with hidden answers) work as entry/exit tickets; 6–12 per module.
- **Lab notes** (📝 template at the end of each notebook): one per module; grade for correct
  inputs/outputs/method, a sensible failure mode and (Level 2) the ground-truth check result.
- **Derivation quizzes**: the table in [`STUDY_GUIDE.md` §3](STUDY_GUIDE.md#3-the-derivations-to-master)
  lists the derivations every student should reproduce on a blank page.
- **Hand calculations** that mirror exam questions: forward variables and the Viterbi table (19),
  information gains (10), naïve-Bayes counts (08), alarm-network queries (17), a k-means step and
  single-link merges (12), the SVM margin (11).
- **Level 2 mini-projects**: change one simulator parameter and report which methods break (e.g.
  stronger batch effect in 31, fewer sequences in 37, weaker instruments in 35, more noise in 41),
  with the ground-truth metric before and after.
- **Capstone** (Module 44): report + presentation + peer review. Suggested rubric (the notebook
  contains a student-facing version):

| Criterion | Excellent (4) | Adequate (2–3) | Insufficient (0–1) |
|---|---|---|---|
| Question and evaluation contract | biological question, unit of replication, split, metric and baseline stated before analysis | most elements present | metric or split chosen after seeing results |
| Methods | several methods, each justified, tuned within training folds with a fixed budget | tuning partly ad hoc | test data used for tuning or selection |
| Results and uncertainty | fold spread, paired comparison or bootstrap CI, calibration where relevant; checked against simulated truth where available | point estimates with some spread | single numbers only |
| Interpretation | stability of selected features, limits of biological claims | some interpretation | causal claims from predictive models |
| Reproducibility and communication | seeded, runs end to end, clear figures and report, DOME checklist | minor gaps | cannot be rerun |

---

## 7 · Slide errata and clarifications found while writing the course

This section is for the instructors who maintain the 02-620 slides. While turning every slide
into self-contained notebook content, the authors found the following slips. The notebooks teach
the correct statement directly (usually in a ⚠️ "common slip" box), without referring to the
slides; the list below lets the slides be fixed in a future revision. Slide numbers are
PowerPoint numbers of the Spring 2025 decks.

| Deck | Slide(s) | Issue | Correct statement / how the notebook handles it |
|---|---|---|---|
| `001_intro` | 18 | "ATT gene" for Huntington disease | almost certainly HTT ([00](notebooks/00_welcome_ml_for_biology.ipynb)) |
| `001_intro` | 20 (and `012_HMM-DBN` 4) | HMM gene finding on the 2001 draft genome (Haussler, Kent) called the first use of HMMs in biology | HMMs entered biology with Churchill (1989); HMM gene finders date from Krogh, Mian & Haussler (1994), Genie and GENSCAN (1997), which helped annotate the draft genome; Kent assembled the public draft and built the UCSC browser ([19](notebooks/19_hmm_inference.ipynb)) |
| `001_intro` | 28 | HapMap Phase I "1.3 million SNPs" | about 1 million SNPs in the 2005 Phase I map ([00](notebooks/00_welcome_ml_for_biology.ipynb)) |
| `001_intro` | 67 | "∼" used for "proportional to" | written ∝ ([05](notebooks/05_likelihood_mle_map.ipynb)) |
| `002_regression` | 4 vs 5 | slide 4 titled "Lung Cancer GWAS", notes and slide 5 discuss breast cancer | noted ([06](notebooks/06_linear_regression_gwas.ipynb)) |
| `002_regression` | 9 | "genotype at SNP i individual j" contradicts the matrix layout (rows = individuals) | rows = individuals, columns = SNPs |
| `002_regression` | 32 | minus sign inside the square, [−(yⁱ − xⁱᵀw)]², under an argmax | −(yⁱ − xⁱᵀw)² |
| `002_regression` | 35 vs 40/46 | "MLCE" and "MCLE" for the same estimator | maximum conditional likelihood estimate |
| `002_regression` | 45 (notes) | y = Xᵀw + ε | y = Xw + ε |
| `002_regression` | 50–51 | prior density written with σ² instead of σ₀² and (2π)^{1/2} instead of (2π)^{J/2} | final MAP objective on slide 50 is still correct ([07](notebooks/07_ridge_lasso_sparse_regression.ipynb)) |
| `002_regression` | 58 | index typos in the last column of the centred matrix | noted |
| `003_classification-kNN-NB` | 14 vs `002` 9 | AA called "minor-allele homozygous" while coded 0 (i.e. the code counts the major allele) | either coding works; 2 − G carries the same information ([01](notebooks/01_python_numpy_for_biologists.ipynb)) |
| `003_classification-kNN-NB` | 39, 50 | swapped i/j indices, Y = 1 hard-coded; two-outcome (Beta) denominator used for 3-valued genotypes | (count + a)/(N_y + 3a) for three categories ([08](notebooks/08_knn_naive_bayes.ipynb)) |
| `003_classification-kNN-NB` | 49 | stray "=" in argmax_θ = P(D∣θ)P(θ)/P(D) | noted ([05](notebooks/05_likelihood_mle_map.ipynb)) |
| `003_classification-kNN-NB` | 54 | "n features → 4n parameters" omits the class prior | 2nk + (k − 1) for k classes ([08](notebooks/08_knn_naive_bayes.ipynb)) |
| `004_classification-LR` | 8–19, 32 vs 23–31 | two sign conventions for the logistic model (1/(1+exp(w₀+Σwᵢxᵢ)) vs exp(·)/(1+exp(·))) | differ by the sign of every weight; one convention used and explained ([09](notebooks/09_logistic_regression.ipynb)) |
| `004_classification-LR` | 28–29 | step size called η, then λ | notation alert ([04](notebooks/04_calculus_and_optimization.ipynb)) |
| `004_classification-LR` | 31 | with λ = 1/(2σ²) the shrinkage term of the update should be 2ηλwᵢ | derived; scikit-learn's C = 1/(2λ) ([09](notebooks/09_logistic_regression.ipynb)) |
| `005_decision-tree` | 20–21 | information gains computed from rounded intermediate entropies (0.246, 0.151) | exact: 0.247, 0.152, 0.048, 0.029 ([10](notebooks/10_decision_trees_random_forests.ipynb)) |
| `005_decision-tree` | 35 | pruning evaluated on the "test set" | must be a separate validation set; a test set is kept untouched |
| `005_decision-tree` | 49 | C4.5 rule "if O = sunny and H ≤ 70 then PLAY" | the threshold depends on where it is placed between the observed humidities 70 and 85 (the C4.5 golf tree is usually printed with ≤ 75); the notebook avoids the exact quote ([10](notebooks/10_decision_trees_random_forests.ipynb)) |
| `005_decision-tree` | 58 | "RF-1: depth 1 trees, RF-3: depth 3 trees" | in ESL Fig. 15.3 (nested spheres) RF-k is a random forest with m = k candidate variables per split, not a tree depth ([10](notebooks/10_decision_trees_random_forests.ipynb)) |
| `005_decision-tree` | 62, 64 | "CHSM" | CHASM |
| `006_clustering` | 17 | correlation formula lacks the 1/p factor (or unnormalised σ) | fixed in the text ([12](notebooks/12_clustering_kmeans_hierarchical.ipynb)) |
| `007_SVM` | 46 (notes) | the min/max swap written as "max_x max_α" | weak and strong duality stated and checked ([04](notebooks/04_calculus_and_optimization.ipynb)) |
| `007_SVM` | 48, 51 | Lagrangian box says αᵢ ≤ 0 | αᵢ ≥ 0 ([11](notebooks/11_support_vector_machines.ipynb)) |
| `007_SVM` | 52, extra slide | C > αᵢ ≥ 0 | C ≥ αᵢ ≥ 0 (margin violators have αᵢ = C) |
| `007_SVM` | 56 | support vectors described only as points on the wrong side of the margin | points exactly on the margin (0 < αᵢ < C) are support vectors too |
| `007_SVM` | 1 | reading "Murphy Ch 4.5" | the SVM material is Murphy §14.5 |
| `008_BN_mz_021825` | 81 | Example 8's second path copied from Example 3 | W—Y—X; verdict unchanged ([16](notebooks/16_bayesian_networks_independence.ipynb)) |
| `008_BN_mz_021825` | 61–93 | evidence set written {Z}, clashing with node Z | renamed "evidence" |
| `008_BN_mz_021825` | 107 | P(B, J, ¬M) = 0.00601 from rounded terms | exact 0.0063175 ([17](notebooks/17_bayesian_networks_inference_learning.ipynb)) |
| `008_BN_mz_021825` | 123–126 | the 19 → 14 operation count holds for a fixed value of B | explained ([17](notebooks/17_bayesian_networks_inference_learning.ipynb)) |
| `009_GGM_mz_030925` | 21–22 | the 8 × 8 edge-weighted Θ is not positive definite | shown; 0.7·I added, same graph ([18](notebooks/18_gaussian_graphical_models.ipynb)) |
| `009_GGM_mz_030925` | 46–51 | S defined with 1/(2N) | 1/N, so that Θ̂ = S⁻¹ |
| `009_GGM_mz_030925` | 52 | (Θ⁻¹)ᵢᵢ = Sᵢᵢ + λ holds only if the diagonal is penalised (and in Friedman et al.'s scaling) | both versions verified in code |
| `009_GGM_mz_030925` | 40 | β₀ paired with μ₁ | true only for centred predictors |
| `010_mixture-model_mz_031725` | 60, 66 | EM "guaranteed to increase / converge"; "local minimum" | the likelihood never decreases; convergence to a stationary point (local *maximum*) if bounded; the unconstrained GMM likelihood is unbounded ([13](notebooks/13_mixture_models_em.ipynb)) |
| `010_mixture-model_mz_031725` | 69 | held-out score written with the complete-data formula | observed-data likelihood (sum over clusters) |
| `010_mixture-model_mz_031725` | 72–77 | Bernoulli data written x ∈ ℝᴶ | binary vectors |
| `011_PCA_mz_031825` | 49 | factor 2 dropped in the derivative | harmless; noted ([14](notebooks/14_pca_dimensionality_reduction.ipynb)) |
| `011_PCA_mz_031825` | 52 | strict "<" | "≤" |
| `011_PCA_mz_031825` | 55 | "361 × 261 pixels, 83,781 dimensions" | 361 × 261 = 94,221 (83,781 = 321 × 261) |
| `012_HMM-DBN_mz_032425` | 14, 20, 34 vs 107 | P(qₜ=sᵢ∣qₜ₋₁=sⱼ) vs aᵢⱼ = P(qₜ=sⱼ∣qₜ₋₁=sᵢ) | slide-107 convention used throughout ([19](notebooks/19_hmm_inference.ipynb)) |
| `012_HMM-DBN_mz_032425` | 69 | "Transition" and "Emission" labels swapped | slide 70 is correct |
| `012_HMM-DBN_mz_032425` | 90 | the displayed path A A A B B B B B B A A is not the Viterbi path of 1 2 2 1 1 2 1 1 1 2 2 | under the given parameters the most probable path is all A (1.75 × 10⁻⁸ vs 2.25 × 10⁻¹²); the notebook presents the displayed path as "a tempting answer" and verifies by brute force ([19](notebooks/19_hmm_inference.ipynb)) |
| `012_HMM-DBN_mz_032425` | 94–95 | "path defined by argmax_j δₜ(j)"; δ defined as a conditional | traceback through backpointers is required; δ is a joint probability |
| `012_HMM-DBN_mz_032425` | 108–109 | "moving window of size 2" for emissions | size 1 ([20](notebooks/20_hmm_learning.ipynb)) |
| `012_HMM-DBN_mz_032425` | 113 | "local minima" | local maxima of the likelihood |
| `012_HMM-DBN_mz_032425` | 124 | duration product index range; Poisson duration can be 0 | product over the segment t−l+1…t; 1 + Poisson(λ − 1) used |
| `013_Bayesian-MCMC_mz_032725` | 24 | p(wⱼ ∣ w₋ⱼ, data) called a "marginal" | a full conditional ([21](notebooks/21_bayesian_learning_mcmc.ipynb)) |
| `014_NN_mz_032725` | 16 | sigmoid with the opposite sign to slide 4 | flagged ([22](notebooks/22_neural_networks.ipynb)) |
| `014_NN_mz_032725` | 46–49 | δ rules assume squared error with sigmoid units (Mitchell), while slide 41 says "loss function" | both squared-error and cross-entropy rules derived |
| `015_model-selection_mz_040725` | 3 | "k-NN: number of clusters K"; "SVM: NA" | number of neighbours; SVMs need C, kernel and its parameters, class weights, preprocessing ([11](notebooks/11_support_vector_machines.ipynb), [23](notebooks/23_model_selection.ipynb)) |
| `015_model-selection_mz_040725` | 42–43 | BIC written as "min over θ" | minimise over models, each at its maximum-likelihood fit |
| `015_model-selection_mz_040725` | 55 | E[εf̂] = E[ε]E[f̂] assumes test noise independent of the training data | made explicit ([23](notebooks/23_model_selection.ipynb)) |
| `015_model-selection_mz_040725` | 61 | y-axis label P(cᵢ∣X) | p(x∣cᵢ)P(cᵢ) |

Two clarifications rather than errors are also handled throughout: more CV folds change the cost
and the bias/variance of the *estimate* (independence of samples matters more), and a learned
network edge or a lasso-selected feature is a statistical dependence, not a demonstrated mechanism.

---

---

## 8 · Technical notes and maintenance

- **Tested with** Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0, scikit-learn 1.9, Matplotlib
  3.11, NetworkX 3.6 and PyTorch 2.14 (CPU). Every notebook runs in a few seconds to about two
  minutes on a 2-core CPU; the whole course in roughly half an hour.
- **Structure check:** `python validate_notebooks.py` checks the schema, required sections and
  header keys, Colab badges, stripped outputs, relative links, that every toolkit copy matches
  `course_toolkit.py`, that no notebook refers to lecture slides, embeds or links a picture, or
  reads/downloads data, and that no data files or files over 400 kB are stored. `--execute`
  also runs every notebook.
- **Editing a notebook:** edit in Jupyter, run it top to bottom, check that the narrative still
  matches the outputs (the simulations are seeded), then clear outputs before committing
  (`jupyter nbconvert --clear-output --inplace notebooks/NN_*.ipynb`).
- **Changing a shared helper:** edit `course_toolkit.py`, then replace the definition in the
  🧰 toolkit cell of every notebook that uses it — the validator lists each stale copy. Changing a
  simulator changes the numbers in every notebook that uses it; re-check their narratives.
- **Data policy:** no data files, no pictures, no downloads (see [`DATA.md`](DATA.md)).
