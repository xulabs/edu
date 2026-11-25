# Study guide: learning the mathematics of 02-680 on your own with BiologyToMath

This guide is for the student working alone. It explains how to study a module, which worked
examples to recompute by hand, which proofs and derivations you must be able to reproduce, the
notation traps, how to prepare for 02-680 quizzes and exams, how to write mathematics in LaTeX, and
how to move on to 02-620 with the sister course. Everything you need is in the notebooks: they
teach the 02-680 content themselves.

---

## 1. The study loop (for every module)

1. **Run the notebook once, top to bottom** (Runtime → Run all on Colab). Read the 🧫 biological
   question, the 🎯 objectives and the 🗺️ contents. The goal of this pass is the big picture.
2. **Work section by section.** For each section: read the 📍 where-we-are box, read the theory and
   say every "*In words:*" sentence back in your own words, run the cells, then read 🔍 *What just
   happened?* and check it against what you see. Do every ✏️ *Your turn* — that is where the
   learning happens.
3. **Pen and paper.** Mathematics is learned with a pencil. Re-derive every displayed equation and
   redo every worked example before you look at the code that checks it; then run the "by hand vs by
   NumPy/SymPy" cell and find your mistake if the numbers differ. If a step feels like magic, the
   🧮 refresher or the earlier module named in the 📍 box has the missing piece.
4. **Consolidate with the 📘 *Lecture notes* boxes.** They summarise everything 02-680 teaches on the
   topic — definitions, key equations, the worked example with its numbers, common confusions and an
   🎓 exam tip. You should be able to explain every box without looking at the code.
5. **Self-test** with ✅ *Checkpoint*: answer on paper before opening the hidden answer. Then try one
   🧠 discussion question and one 🚀 bonus item.
6. **Write the 📝 lab note.** Five minutes; it becomes your revision sheet.

A module takes 4–6 hours this way: the 2–3 hours of estimated time in its key-information table
(reading, running and the ✏️ exercises) plus the pen-and-paper practice, self-test and lab note. It
is better to do one module properly than three quickly.
Read the 🌉 *Bridge* boxes for motivation — they show where the mathematics is used in biology and
in later courses — but do not let them distract you from the core on a first pass.

## 2. Worked examples to recompute by hand

The classic worked examples of 02-680 are in the notebooks with their numbers. Recompute each once
on paper (a calculator is fine), then check it in the notebook.

| Module | Worked example | Check your answer |
|---|---|---|
| [02](notebooks/02_logic.ipynb) | De Morgan's law ¬(p ∧ q) ≡ ¬p ∨ ¬q by the full truth table; the contrapositive is equivalent to the implication, the converse is not | identical final columns |
| [02](notebooks/02_logic.ipynb) | when is p ∧ q ⇒ ¬q true? the witnesses of ∃y ∈ ℤ: y² = \|y + y\| | exactly when at least one of p, q is false; y ∈ {−2, 0, 2} |
| [03](notebooks/03_sets_and_functions.ipynb) | C = {2+2, 2−2, 2×2, 2÷2, 2²}; B ∩ C, B ∪ C, B \ C, C \ B for B = {4, 9}, C = {0, 1, 4} | \|C\| = 3; {4}, {0,1,4,9}, {9}, {0,1} |
| [03](notebooks/03_sets_and_functions.ipynb) | \|𝒫(S)\| = 2^\|S\|; 𝒫(∅) and 𝒫({∅}) | 𝒫(∅) = {∅}, 𝒫({∅}) = {∅, {∅}} |
| [03](notebooks/03_sets_and_functions.ipynb) | the four compositions of f(x) = 2x + 1 and g(x) = x²; why 3x² + 2 is not a function into ℤ | (2x+1)², 2x² + 1, 4x + 3, x⁴; 3(1.5)² + 2 = 8.75 |
| [04](notebooks/04_tuples_sequences_strings.ipynb) | \|{0,1} × {1,2} × {0,1,2}\|; the ≤ relation on {0,1,2,3}; binary numbers of at most 8 digits without leading zeros | 12 triples; 10 pairs; 255 |
| [05](notebooks/05_asymptotics_growth_of_functions.ipynb) | 4x ∈ O(x²); 3x² + 2 ∈ O(x²); is x³ ∈ O(x²)? | witnesses c = 1, x₀ = 4 (and many others); yes; no |
| [06](notebooks/06_graphs_and_trees.ipynb) | the handshake lemma Σ deg v = 2\|E\|; leaves and internal nodes of the 13-node rooted tree | leaves {D,F,H,J,K,L,M}, internal {A,B,C,E,G,I} |
| [07](notebooks/07_vectors.ipynb) | ⟨1,2,3⟩ • ⟨4,5,6⟩; ‖[3,4]‖₂, ‖[3,2]‖₁, ‖[3,−4,2]‖∞ | 32; 5, 5, 4 |
| [08](notebooks/08_matrices.ipynb) | the 2 × 3 by 3 × 2 product with rows (1,2,3), (4,5,6) and columns (7,9,11), (8,10,12) | [[58, 64], [139, 154]] |
| [08](notebooks/08_matrices.ipynb) | det [[2,1],[1,6]]; det [[2,6,1],[3,2,5],[2,3,6]] by cofactor expansion | 11; −49 |
| [09](notebooks/09_linear_systems_inverses.ipynb) | 3z₁ + 2z₂ = −1, z₁ − 5z₂ = 3 by the 2 × 2 inverse and by Gauss–Jordan | det −17, C⁻¹ = (1/17)[[5, 2], [1, −3]], z = (1/17, −10/17) |
| [09](notebooks/09_linear_systems_inverses.ipynb) | −x₁ + 3x₂ = 5, x₁ − 3x₂ = 1 | the row 0 = 6: no solution (parallel lines) |
| [09](notebooks/09_linear_systems_inverses.ipynb) | the 3 × 5 under-determined system 2x₁ − x₂ + 4x₃ + x₄ − x₅ = 4, 3x₁ + 2x₂ − 2x₃ − 5x₄ + 2x₅ = 9, x₁ + x₂ + 6x₃ + 2x₄ + x₅ = 7 | x₁ − (12/13)x₄ = 2, x₂ − (7/13)x₄ + x₅ = 2, x₃ + (15/26)x₄ = 1/2; free x₄ = s, x₅ = t |
| [09](notebooks/09_linear_systems_inverses.ipynb) | a null-space basis by the "−1 trick" | basis vector [5, −4, 1]ᵀ; and [5, −4, 1, 0]ᵀ, [6, −5, 0, 1]ᵀ |
| [10](notebooks/10_vector_spaces.ipynb) | (1 + 2x + 3x²) + (−1 + 0x + 4x²); the non-standard addition ⟨a₁ + b₁, a₂ − b₂⟩ | 2x + 7x²; it fails commutativity and (α + β)v = αv + βv |
| [11](notebooks/11_independence_basis_rank.ipynb) | ⟨1,2⟩, ⟨2,1⟩, ⟨8,7⟩: dependent? is ⟨8,7⟩ in the span of the first two? | solutions (−2t, −3t, t); ⟨8,7⟩ = 2⟨1,2⟩ + 3⟨2,1⟩ |
| [11](notebooks/11_independence_basis_rank.ipynb) | coordinates of ⟨6,2⟩ and ⟨5,−3⟩ in the basis {⟨1/√2, 1/√2⟩, ⟨1/√2, −1/√2⟩} | ⟨4√2, 2√2⟩ and ⟨√2, 4√2⟩; lengths √40, √34 and dot product 24 preserved |
| [12](notebooks/12_eigenvalues_eigenvectors.ipynb) | A = [[3,4],[2,1]]: is (2,1) an eigenvector? all eigenvalues and eigenvectors | λ = 5; λ² − 4λ − 5 = (λ − 5)(λ + 1), eigenvectors (2,1), (1,−1) |
| [12](notebooks/12_eigenvalues_eigenvectors.ipynb) | B = [[1,1,−1],[2,3,−4],[4,1,−4]] by cofactor expansion | λ³ − 7λ + 6 = (λ − 1)(λ − 2)(λ + 3); eigenvectors (1,1,1), (1,2,1), (1,7,11) |
| [13](notebooks/13_diagonalization_svd.ipynb) | the eigendecomposition and the SVD worked "by hand" in the notebook; the shear [[1,1],[0,1]] | the shear has one eigen-direction and is not diagonalizable |
| [14](notebooks/14_calculus_derivatives_integrals.ipynb) | d(xⁿ)/dx from the binomial theorem; the optima of 3x⁴ − 10x³ − 12x² + 18x − 3 | f′ = 6(x − 3)(x + 1)(2x − 1); f″ = 36x² − 60x − 24; local minima f(−1) = −20 and f(3) = −84 (global), local maximum at x = ½ |
| [14](notebooks/14_calculus_derivatives_integrals.ipynb) | doubling time of N′ = rN; AUC of C(t) = C₀e^(−kt) | ln 2 / r; C₀/k |
| [15](notebooks/15_multivariable_calculus_dynamics.ipynb) | the Jacobian of h(x) = (3x₁², x₁x₂ − x₂²); ∇‖x‖ | [[6x₁, 0], [x₂, x₁ − 2x₂]]; x/‖x‖ |
| [16](notebooks/16_probability_spaces.ipynb) | a die: P(even ∪ at least 4); Aa × Aa | 4/6 by inclusion–exclusion; P(aa) = ¼ |
| [17](notebooks/17_conditional_probability_bayes.ipynb) | the flu/cough table 0.3, 0.2, 0.15, 0.35: P(flu ∣ cough) | 0.3/0.45 = 2/3 |
| [17](notebooks/17_conditional_probability_bayes.ipynb) | Bayes with prior 0.05, P(cough ∣ flu) = 0.8, P(cough ∣ no flu) = 0.2; then prior 0.5 | 0.04/0.23 ≈ 0.174; 0.8 |
| [17](notebooks/17_conditional_probability_bayes.ipynb) | the two-coin box (a fair and a two-headed coin): A = first toss heads, B = second toss heads | conditionally independent given the coin, but P(A ∩ B) = 5/8 ≠ P(A)P(B) = 9/16 |
| [17](notebooks/17_conditional_probability_bayes.ipynb) | a die with A = {1,2}, B = {2,4,6}, C = {1,4}; the stop codons TAA, TAG, TGA | A, B independent but not given C; P(3rd = G ∣ 2nd = A) = ½ ≠ P(3rd = G) = ⅓ |
| [17](notebooks/17_conditional_probability_bayes.ipynb) | naïve-Bayes spam filter: P(S) = 0.7, P("free" ∣ S) = 0.9, P("free" ∣ ¬S) = 0.01 | 0.63 vs 0.003 → posterior ≈ 0.995 |
| [18](notebooks/18_random_variables_distributions.ipynb) | number of heads in two tosses; the sum V of two 4-sided dice; the horse-kick data | PMF ¼, ½, ¼; P(V ≤ 4) = 3/8; mean 0.61 deaths per corps-year |
| [19](notebooks/19_multiple_random_variables.ipynb) | the BRCA1/P53/cancer table 0.24, 0.01, 0.20, 0.05, 0.15, 0.10, 0.05, 0.20 (illustrative) | P(cancer ∣ G₁, G₂) = 0.04, 0.20, 0.40, 0.80 |
| [19](notebooks/19_multiple_random_variables.ipynb) | the unfair coins table 1/10, 2/10 / 3/10, 4/10; X ~ Uniform[−1, 1], Y = X² | P(X = T) = 7/10; Cov(X, Y) = 0 although Y is a function of X |
| [20](notebooks/20_bayesian_networks_markov_chains.ipynb) | the bronchitis network: P(T ∣ S) | 0.35 · 0.8 + 0.1 · 0.2 = 0.30 |
| [20](notebooks/20_bayesian_networks_markov_chains.ipynb) | P(ACTTC) under the CpG-island and background chains (start 0.25) | 0.000831 vs 0.001044; ratio 1.26 → more likely background |
| [20](notebooks/20_bayesian_networks_markov_chains.ipynb) | the weather chain [[0.7, 0.3], [0.6, 0.4]]: sunny on Wednesday → sunny on Saturday and Sunday; the telephone-line chain three steps ahead | (P³)₁₁ · 0.7 = 0.667 · 0.7 ≈ 0.467; P(0 lines ∣ 5 lines three steps earlier) = 0.116 |
| [21](notebooks/21_expectation_variance_moments.ipynb) | E of two 4-sided dice, one 6-sided die, Uniform(−1, 3); E[X²] for Uniform(0, 1) | 5; 3.5; 1; ⅓ (≠ (E X)² = ¼) |
| [21](notebooks/21_expectation_variance_moments.ipynb) | the buggy coin: 0 or 1 with probability 0.4995 each, 100 with probability 0.001 | E = 0.5995, Var ≈ 10.14, σ ≈ 3.18 |
| [21](notebooks/21_expectation_variance_moments.ipynb) | X ~ U(0, 1), Y ∣ X = x ~ U(x, 1) | E[Y ∣ x] = (1 + x)/2, Var[Y ∣ x] = (1 − x)²/12, E[Y] = ¾ |
| [22](notebooks/22_sampling_lln_clt_confidence.ipynb) | 48 heads in 100 tosses; the standard error of a mean | x̄ = 0.48; σ/√n |
| [23](notebooks/23_maximum_likelihood.ipynb) | the MLE for 1,1,1,0,0,0,0,0,1,0, with and without logs | α̂ = 0.4 |
| [24](notebooks/24_bayesian_estimation_map.ipynb) | 0 heads in 100 tosses with pseudocounts 0, 1 and 100 | 0, 1/102, 1/3 |
| [24](notebooks/24_bayesian_estimation_map.ipynb) | the coin drawer (5 coins with p = 0.5, 3 with 0.6, 2 with 0.9): one tails; then 6 heads and 4 tails | posterior 0.641, 0.308, 0.051 (MAP: first type); 0.570, 0.418, 0.012 (MAP first type, MLE second) |
| [24](notebooks/24_bayesian_estimation_map.ipynb) | x = (0,0,1,1,0), θ ∈ {0.2, 0.5, 0.7} with prior 0.1, 0.01, 0.89 | likelihoods 0.02048, 0.03125, 0.01323; MAP θ = 0.7 (MLE θ = 0.5) |
| [26](notebooks/26_categorical_tests_multiple_testing.ipynb) | Mendel's dihybrid cross 315, 101, 108, 32 against 9:3:3:1 | expected 312.75, 104.25, 104.25, 34.75; χ² ≈ 0.47 with 3 df, p ≈ 0.925 |
| [27](notebooks/27_linear_regression_review.ipynb) | least squares for a straight line through the three points (1, 1), (2, 3), (3, 2), by the normal equations | β̂ = (XᵀX)⁻¹Xᵀy = (slope 0.5, intercept 1); RSS = 1.5, so σ̂²_MLE = RSS/n = 0.5 |

Module [25](notebooks/25_hypothesis_testing.ipynb) and the other simulated examples (the z- and
t-tests on the cholesterol trial, the insulin regression) print numbers that come from a simulator:
compute their statistics on paper from the printed summary values (means, standard deviations,
n), then compare with the notebook.

## 3. The proofs and derivations to master

These are the "can you do it on a blank page?" results of the course. Tick them off.

| Proof or derivation | Module |
|---|---|
| The contrapositive is equivalent to the implication; De Morgan's laws; mutual implication ≡ biconditional — by truth tables | [02](notebooks/02_logic.ipynb) |
| Negating a nested quantifier step by step (¬∀x ∃y P ≡ ∃x ∀y ¬P) | [02](notebooks/02_logic.ipynb) |
| \|𝒫(S)\| = 2^\|S\| via binary encoding; inclusion–exclusion for two sets; max(\|S\|, \|T\|) ≤ \|S ∪ T\| ≤ \|S\| + \|T\| | [03](notebooks/03_sets_and_functions.ipynb) |
| Set equality by two-way inclusion; C \ B = C ∩ B̄ | [03](notebooks/03_sets_and_functions.ipynb) |
| \|A × B\| = \|A\| \|B\|; \|Σᵏ\| = \|Σ\|ᵏ | [04](notebooks/04_tuples_sequences_strings.ipynb) |
| A polynomial of degree k is in O(xᵏ); sum and product rules for O; the base of a logarithm does not matter | [05](notebooks/05_asymptotics_growth_of_functions.ipynb) |
| The handshake lemma; a connected graph with \|V\| − 1 edges is a tree | [06](notebooks/06_graphs_and_trees.ipynb) |
| ‖x + y‖² = ‖x‖² + ‖y‖² for orthogonal x, y; cos θ = x · y / (‖x‖ ‖y‖) from the law of cosines | [07](notebooks/07_vectors.ipynb) |
| (AB)ᵀ = BᵀAᵀ; how row operations change the determinant; cofactor expansion | [08](notebooks/08_matrices.ipynb) |
| (AB)⁻¹ = B⁻¹A⁻¹; the 2 × 2 inverse formula; elementary operations keep the solution set | [09](notebooks/09_linear_systems_inverses.ipynb) |
| Uniqueness of the zero vector and of additive inverses; the subspace test; P₄ is a subspace; the null space is a subspace, the solution set of Ax = b ≠ 0 is not | [10](notebooks/10_vector_spaces.ipynb) |
| Independence ⇔ no vector is a combination of the others; coordinates in a basis are unique; rank–nullity | [11](notebooks/11_independence_basis_rank.ipynb) |
| A non-zero solution of (A − λI)x = 0 exists ⇔ det(A − λI) = 0; distinct eigenvalues ⇒ independent eigenvectors (two vectors); BᵀB is positive semidefinite | [12](notebooks/12_eigenvalues_eigenvectors.ipynb) |
| AP = PD column by column; Aᵏ = XΛᵏX⁻¹; the SVD from the eigendecomposition of AᵀA | [13](notebooks/13_diagonalization_svd.ipynb) |
| d(xⁿ)/dx = nxⁿ⁻¹ from the binomial theorem; the second-derivative test; ln(1 + β) < β (continuous vs discrete growth) | [14](notebooks/14_calculus_derivatives_integrals.ipynb) |
| ∇(bᵀx) = b and ∇(xᵀAx) = (A + Aᵀ)x component by component; the normal equations from ‖Ax − b‖²; the equilibrium −M⁻¹c and its stability from the eigenvalues of M | [15](notebooks/15_multivariable_calculus_dynamics.ipynb) |
| From the three axioms: P(∅) = 0, P(Aᶜ) = 1 − P(A), monotonicity, inclusion–exclusion | [16](notebooks/16_probability_spaces.ipynb) |
| Bayes' rule from the product rule; the chain rule for n events; independence ⇔ P(A ∣ B) = P(A) | [17](notebooks/17_conditional_probability_bayes.ipynb) |
| The binomial PMF from n Bernoulli trials; Bernoulli mean α and variance α(1 − α); memorylessness of the geometric distribution | [18](notebooks/18_random_variables_distributions.ipynb) |
| Marginal = sum (integral) of the joint; Cov(X, Y) = E[XY] − E[X]E[Y]; independence ⇒ Cov = 0 but not conversely | [19](notebooks/19_multiple_random_variables.ipynb) |
| The joint of a Bayesian network as ∏ p(xₖ ∣ Paₖ); P(T ∣ S) by summing out C (which independence is used where); the stationary distribution as a left eigenvector with eigenvalue 1 | [20](notebooks/20_bayesian_networks_markov_chains.ipynb) |
| Linearity of expectation (no independence needed); Var X = E[X²] − (E X)²; Var(aX + b) = a² Var X; Var X = 0 iff X is constant; why s² divides by n − 1; the law of total expectation | [21](notebooks/21_expectation_variance_moments.ipynb) |
| E[X̄] = μ and Var X̄ = σ²/n | [22](notebooks/22_sampling_lln_clt_confidence.ipynb) |
| MLEs of the Bernoulli (s/n), normal (x̄ and (1/n)Σ(xᵢ − x̄)²) and Poisson (x̄); the allele frequency (2n₂ + n₁)/(2n) | [23](notebooks/23_maximum_likelihood.ipynb) |
| The Beta–binomial posterior; the MAP (k + α − 1)/(n + α + β − 2) for k heads in n tosses; the Gamma–Poisson posterior; the normal–normal MAP as a precision-weighted average | [24](notebooks/24_bayesian_estimation_map.ipynb) |
| The z statistic under H₀; α, β and power of a one-sided z-test; the paired t statistic δ̄/(s_δ/√n) | [25](notebooks/25_hypothesis_testing.ipynb) |
| The χ² statistic Σ(Oᵢ − Eᵢ)²/Eᵢ and its degrees of freedom; the Benjamini–Hochberg procedure | [26](notebooks/26_categorical_tests_multiple_testing.ipynb) |
| Least squares = maximum likelihood under Gaussian noise; β̂ = (XᵀX)⁻¹Xᵀy; σ̂²_MLE = RSS/n | [27](notebooks/27_linear_regression_review.ipynb) |

## 4. Notation traps

Textbooks, lecturers and software use different symbols for the same thing — and the same symbol
for different things. The notebooks always say which convention they use.

| Trap | What goes wrong | Course convention |
|---|---|---|
| ⊂ vs ⊆ | some texts write ⊂ for "subset or equal", others only for a proper subset | S ⊆ T: subset (possibly equal); S ⊂ T: proper subset |
| 2 vs {2}; ∅ vs {∅} | an element is not the set containing it; {∅} has one element | 2 ∈ {2}; \|∅\| = 0, \|{∅}\| = 1 |
| { } vs ⟨ ⟩ / ( ) | sets ignore order and repetition, tuples do not | {A, T} = {T, A}, but ⟨A, T⟩ ≠ ⟨T, A⟩ |
| p ⇒ q vs q ⇒ p | the converse is not equivalent; implication is not causation | only the contrapositive ¬q ⇒ ¬p is equivalent |
| "or" | everyday "or" is often exclusive | ∨ is inclusive; ⊕ is exclusive |
| operator precedence | ∧, ∨ and ⊕ have no universally agreed order | ¬ first, ⇔ last — and use parentheses |
| g ∘ f order | some sources write f ∘ g and apply g first | (g ∘ f)(x) = g(f(x)): f first |
| path length | k nodes or k − 1 edges? | number of edges |
| rows × columns | n × m means n rows, m columns; `A.shape` is (rows, columns) | data matrices are N samples × J features |
| row vs column vectors | xᵀy vs xyᵀ; vT = α (row vectors) vs α = Tᵀv (column vectors) | vectors are columns; xᵀy is a number, xyᵀ a matrix |
| det(A − λI) vs det(λI − A) | they differ by the factor (−1)ⁿ | same roots; for 3 × 3, det(A − λI) = −(λ³ − 7λ + 6) in the classic example |
| `np.linalg.eig` vs `eigh` | `eig` returns eigenvalues in no particular order (possibly complex); `eigh` (symmetric matrices) sorts them **ascending**; eigenvectors are columns, with arbitrary sign | sort explicitly; compare eigenvectors up to sign |
| `np.linalg.svd` | returns Vᵀ (called `Vh`), not V | A = UΣVᵀ |
| P(A ∣ B) vs P(B ∣ A) | P(cough ∣ flu) ≠ P(flu ∣ cough); the base-rate fallacy | turn one into the other with Bayes' rule |
| independent vs conditionally independent vs uncorrelated | independence ⇒ Cov = 0, but not conversely; independence and conditional independence do not imply each other | check each from its definition |
| density vs probability | f(x) can exceed 1; P(X = x) = 0 for a continuous X | probability = area under f |
| σ vs σ² | N(μ, σ²) vs N(μ, σ); the CLT is X̄ ≈ N(μ, σ²/n), not N(μ, σ/√n) | the second argument of 𝒩 is the variance; SciPy's `scale` is σ |
| 1/n vs 1/(n − 1) | the MLE of σ² uses 1/n, the unbiased sample variance 1/(n − 1); `np.var` defaults to `ddof=0`, pandas `.var()` to `ddof=1` | the notebook states which one it uses |
| λ, α, β | λ: eigenvalue, Poisson rate, ridge penalty; α: significance level, Bernoulli parameter, Beta parameter, step size; β: Type II error rate, Beta parameter, regression coefficient | read the definition next to the formula |
| parameterisations in SciPy | `expon(scale=1/λ)`; `geom` counts trials (support 1, 2, …); `norm(loc=μ, scale=σ)` | check the documentation and one value by hand |
| time indexing of a Markov chain | X₀ or X₁ as the starting state changes whether you need P² or P³ | the notebook states where time starts |
| transition matrices | rows or columns as the current state | rows = current state; rows sum to 1 |
| "fail to reject" vs "accept" | not rejecting H₀ is not evidence that H₀ is true | say "fail to reject" (or "retain") |
| the p-value | not the probability that H₀ is true | P(statistic at least as extreme as observed ∣ H₀) |
| order of factors | β̂ = (XᵀX)⁻¹Xᵀy, never Xᵀy(XᵀX)⁻¹ (the sizes do not even match) | check every product's sizes |
| log | log, ln, log₂, log₁₀ | log = ln; log₂ for fold changes, log₁₀ for Phred scores and −log₁₀ p |
| indices | mathematics counts from 1, Python from 0 | x₁ is `x[0]` |

## 5. The ideas that recur everywhere

1. **Sets are the language of everything else.** Logic ↔ set operations (02 → 03); events are sets
   (16); a relation, a graph and a function are sets of tuples (04, 06, 03); a subspace is a set
   closed under linear combinations (10); a statistical model is a set of distributions (23).
2. **Linear combinations.** A bulk sample as a mixture of cell-type profiles (07, 09), a
   matrix–vector product (08), span and basis (11), a regression prediction Xβ (27), a polygenic
   score (27).
3. **Eigenvectors describe long-run behaviour.** The stable age distribution of a population (12),
   powers of a matrix (13), the stability of an equilibrium (15), the stationary distribution of a
   Markov chain (20), principal components of expression data (13).
4. **Optimisation is setting a derivative to zero.** One-variable optima (14), least squares and
   gradient descent (15), every maximum-likelihood and MAP estimate (23, 24), regression (27).
5. **Conditioning and Bayes' rule.** Diagnostic tests (17), classifiers (17, 19), Bayesian networks
   (20), conditional expectation (21), MAP estimation (24).
6. **Averages of many small things are predictable.** The law of large numbers and the central
   limit theorem (22) justify standard errors, confidence intervals and every test of Part G.
7. **Check against a known truth.** Every simulator returns the truth it was built from; every hand
   calculation is checked by code. Do the same in your own work.

## 6. Preparing for 02-680 quizzes and exams

02-680 assessments ask you to *compute by hand*, *prove short statements* and *interpret* — usually
without a computer. Check the current semester's announcements for which topics each assessment
covers; then, for those topics:

- **Redo each module's ✅ checkpoint** on paper without looking, and every worked example of §2.
- **Re-derive the rows of §3** for those topics on a blank page; explain each 🎓 exam tip to someone
  else.
- **Time yourself**: a 3 × 3 determinant, a 2 × 2 inverse, a Gauss–Jordan reduction, a Bayes
  calculation or an MLE derivation should each take a few minutes. Keep fractions exact.
- **Build a one-page summary** (if one is allowed) from your 📝 lab notes, the 📘 boxes and the
  📌 key takeaways.
- **After the exam**, check your answers with the code cells — that is what they are for.

| Topic family | On paper, you should be able to… | Practise with |
|---|---|---|
| Logic | translate English to logic and back; build truth tables; prove equivalences; negate nested quantifiers | 02 |
| Sets, functions, tuples, strings | compute unions, intersections, differences, power sets, Cartesian products and cardinalities; identify domain, codomain and range; compose functions; count strings | 03, 04 |
| Asymptotics | prove f ∈ O(g) with explicit witnesses, or disprove it; order functions by growth | 05 |
| Graphs | compute degrees and neighbourhoods; find paths, components, bipartitions, spanning trees; classify trees | 06 |
| Vectors and matrices | dot products, norms, angles; matrix products (check sizes first); transposes; traces; determinants by cofactor expansion and by row operations | 07, 08 |
| Linear systems | Gauss–Jordan to RREF; inverses by [A ∣ I]; parametric solutions; null-space bases; decide if a system has 0, 1 or ∞ solutions | 09 |
| Spaces, independence, rank | check the axioms or find the failing one; the subspace test; independence by row reduction; bases, coordinates and rank | 10, 11 |
| Eigenvalues, decompositions | characteristic polynomials; eigenvectors; diagonalisation; an SVD of a small matrix | 12, 13 |
| Calculus | derivatives with every rule; integrals; critical points and the second-derivative test; gradients, Jacobians, normal equations; equilibria and stability | 14, 15 |
| Probability | apply the axioms; conditional probability, total probability, Bayes' rule; check independence and conditional independence | 16, 17 |
| Random variables | PMF, PDF and CDF calculations; identify the right named distribution; joint, marginal and conditional tables; covariance | 18, 19 |
| Graphical models | write the factorisation of a Bayesian network; answer a query by summing out; n-step transition probabilities | 20 |
| Expectation | expectations and variances with linearity and the shortcut formula; conditional expectation | 21 |
| Estimation | write a likelihood, take logs, differentiate, solve, check the second derivative; Beta–binomial updates; MAP vs MLE | 22–24 |
| Testing and regression | state H₀ and H₁; compute z, t and χ² statistics and degrees of freedom; read p-values; Type I/II errors and power; the normal equations | 25–27 |

## 7. Writing mathematics in LaTeX

02-680 homework is typeset, and every Markdown cell of the notebooks is written in the same LaTeX
(open any cell with a double click to see its source). Use an online editor such as Overleaf or a
local TeX installation. A minimal homework file:

```latex
\documentclass[11pt]{article}
\usepackage{amsmath, amssymb, amsthm}
\newtheorem*{claim}{Claim}

\begin{document}
\section*{Problem 1}
\begin{claim}
For all finite sets $S$ and $T$, $|S \cup T| = |S| + |T| - |S \cap T|$.
\end{claim}
\begin{proof}
Write $S \cup T$ as the disjoint union $(S \setminus T) \cup (S \cap T) \cup (T \setminus S)$ \dots
\end{proof}
\end{document}
```

The constructions you will need most:

```latex
% fractions, powers, subscripts, roots
\frac{1}{17}, \qquad x_i^2, \qquad e^{-\lambda t}, \qquad \sqrt{2}, \qquad \hat{\alpha}

% sums, products, limits, derivatives, integrals
\sum_{i=1}^{n} x_i, \qquad \prod_{i=1}^{n} p(x_i \mid \theta), \qquad
\lim_{h \to 0} \frac{f(x+h) - f(x)}{h}, \qquad \frac{\partial f}{\partial x_1}, \qquad
\int_0^\infty C_0 e^{-kt}\, dt

% logic and sets
\lnot p, \; p \land q, \; p \lor q, \; p \oplus q, \; p \Rightarrow q, \; p \Leftrightarrow q
\forall x \in \mathbb{Z}\; \exists y \in \mathbb{Z} : x = -y
A \cup B, \; A \cap B, \; A \setminus B, \; \overline{A}, \; S \subseteq T, \; \mathcal{P}(S), \; \emptyset,
\{\, x \in \mathbb{Z} \mid x \text{ is even} \,\}

% vectors and matrices
\mathbf{x} \in \mathbb{R}^n, \quad A^\top, \quad A^{-1}, \quad \|\mathbf{x}\|_2, \quad
\operatorname{rank}(A), \quad \det(A - \lambda I)
A = \begin{bmatrix} 3 & 2 \\ 1 & -5 \end{bmatrix}, \qquad
\left[\begin{array}{cc|c} 3 & 2 & -1 \\ 1 & -5 & 3 \end{array}\right]

% probability and statistics
P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}, \qquad
X \sim \mathcal{N}(\mu, \sigma^2), \qquad \mathbb{E}[X], \; \operatorname{Var}(X), \qquad
\hat{\theta} = \arg\max_{\theta} \log p(\mathcal{D} \mid \theta)

% piecewise definitions
g(x) = \begin{cases} x^3 & \text{if } x \text{ is even} \\ x & \text{if } x \text{ is odd} \end{cases}

% an aligned derivation: one step per line, the reason on the right
\begin{aligned}
\ell(\alpha) &= \log\big(\alpha^{s}(1-\alpha)^{n-s}\big)          && \text{likelihood of } n \text{ tosses} \\
             &= s \log \alpha + (n-s) \log(1-\alpha)               && \text{log of a product} \\
\frac{d\ell}{d\alpha} &= \frac{s}{\alpha} - \frac{n-s}{1-\alpha} = 0 && \text{set the derivative to } 0 \\
\Longrightarrow \quad \hat{\alpha} &= \frac{s}{n}
\end{aligned}
```

Habits that make proofs readable (and easy to grade): define every symbol before using it; one
step per line with the reason on the right; use `\mid` for "given" and "such that", `\text{…}` for
words inside formulas, `\cdot` (never `*`) for multiplication, `\top` for transposes, and
`\left( … \right)` around tall expressions; end a proof by saying what was shown.

## 8. Moving on to 02-620 with BiologyToML

[BiologyToML](../../../ml/tutorial/BiologyToML/README.md), the companion to 02-620 *Machine Learning
for Scientists*, is written in the same style and picks up where this course ends. Its Part A
(Modules 00–05) teaches the mathematics it needs in compressed form; after this course it will read
as a quick review. Where each part of this course is used there:

| Mathematics from this course | Used in BiologyToML |
|---|---|
| Vectors, matrices, rank, eigenvectors, SVD (07–13) | [03 · Linear algebra for data](../../../ml/tutorial/BiologyToML/notebooks/03_linear_algebra_for_data.ipynb), [14 · PCA](../../../ml/tutorial/BiologyToML/notebooks/14_pca_dimensionality_reduction.ipynb) |
| Derivatives, gradients, least squares, the chain rule (14–15) | [04 · Calculus and optimization](../../../ml/tutorial/BiologyToML/notebooks/04_calculus_and_optimization.ipynb), [06 · Linear regression](../../../ml/tutorial/BiologyToML/notebooks/06_linear_regression_gwas.ipynb), [22 · Neural networks](../../../ml/tutorial/BiologyToML/notebooks/22_neural_networks.ipynb) |
| Probability, Bayes' rule, distributions (16–19) | [02 · Probability](../../../ml/tutorial/BiologyToML/notebooks/02_probability_for_biologists.ipynb), [08 · k-NN and naïve Bayes](../../../ml/tutorial/BiologyToML/notebooks/08_knn_naive_bayes.ipynb) |
| Bayesian networks and Markov chains (20) | [15 · Bayesian networks](../../../ml/tutorial/BiologyToML/notebooks/15_bayesian_networks_representation.ipynb), [19 · Hidden Markov models](../../../ml/tutorial/BiologyToML/notebooks/19_hmm_inference.ipynb) |
| Covariance and the multivariate normal (19, 21) | [18 · Gaussian graphical models](../../../ml/tutorial/BiologyToML/notebooks/18_gaussian_graphical_models.ipynb) |
| MLE, MAP and conjugate priors (23–24) | [05 · Likelihood, MLE and MAP](../../../ml/tutorial/BiologyToML/notebooks/05_likelihood_mle_map.ipynb), [07 · Ridge and lasso](../../../ml/tutorial/BiologyToML/notebooks/07_ridge_lasso_sparse_regression.ipynb), [21 · Bayesian learning and MCMC](../../../ml/tutorial/BiologyToML/notebooks/21_bayesian_learning_mcmc.ipynb) |
| Tests, multiple testing, regression (25–27) | [06 · Linear regression and GWAS](../../../ml/tutorial/BiologyToML/notebooks/06_linear_regression_gwas.ipynb), [23 · Model selection](../../../ml/tutorial/BiologyToML/notebooks/23_model_selection.ipynb), [26 · Count GLMs](../../../ml/tutorial/BiologyToML/notebooks/26_glm_count_data.ipynb) |
| Graphs and trees (06) | [12 · Clustering](../../../ml/tutorial/BiologyToML/notebooks/12_clustering_kmeans_hierarchical.ipynb), [25 · Graph neural networks](../../../ml/tutorial/BiologyToML/notebooks/25_graph_neural_networks.ipynb) |

Before starting 02-620, make sure you can do these on a blank page: the normal equations; the
gradient of xᵀAx; an eigendecomposition of a 2 × 2 symmetric matrix; Bayes' rule with a total-
probability denominator; the factorisation of a Bayesian network; the Bernoulli and normal MLEs; a
Beta–binomial MAP. Those are the "Part A" of machine learning.

## 9. When you are stuck

| Symptom | What to do |
|---|---|
| `NameError` for a function or variable | run the notebook from the top (setup and toolkit cells first); after a kernel restart everything must be re-run |
| Numbers slightly different from the text | library versions differ in the last digits; the conclusion should not change. If it does, re-run from the top (the seeds assume top-to-bottom order) |
| Your hand calculation disagrees with the code | redo the step the code prints in between (the "by hand" cells print intermediate values); keep fractions exact; check signs in cofactor expansions and the order of matrix products |
| A symbol you do not recognise | the notation crash course in [00](notebooks/00_welcome_math_for_biology.ipynb) and the [glossary](GLOSSARY.md) |
| A derivation step makes no sense | go to the module named in the 📍 box; the 🧮 refreshers cover every prerequisite |
| A proof feels impossible to start | write down the definitions of every term in the statement first; most proofs in this course are "unfold the definitions, then compute" |

## 10. Lab-note template

```text
Module:                         date:
Biological question:
Input (data — which simulator or typed-in table, and what the hidden truth is):
Output (what the code produced):
Method (one sentence):
Key equation (and its reading in words):
Key parameters (and what changed when I modified one):
One failure mode / assumption that can break:
Connection to 02-680 (which topic, which worked example):
One question I still have:
```
