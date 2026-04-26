# Project Plan -- Algebraic Statistics for Phylogenetics

## Goal
Build a research-quality library (`algphylo`) that exercises
**algebraic geometry + tensor decomposition + probability + numerical
linear algebra** on real DNA sequence data, progressing from the
classical Jukes-Cantor HMM to the SVD-based quartet topology test
(Chifman-Kubatko) and phylogenetic invariants for group-based models.

## Layers

### Layer 0 -- Alphabets, Markov models, Newick
- DNA / 2-state alphabets; index encoding
- Continuous-time Markov chains: Q -> M(t) = exp(Q*t)
- Classical rate-matrix models:
  - Jukes-Cantor (JC69, 1 parameter)
  - Kimura 2-parameter (K2P / K80)
  - HKY85 (4 parameters)
  - General Markov model (GMM) with arbitrary row-stochastic matrix
- Minimal Newick parser (borrow design from `phylomath`).

### Layer 1 -- Joint distribution tensor
- `joint_tensor(tree, params)`: compute P in ?^{kappa^n} via recursive
  tensor contraction (cherry-folding).
- `felsenstein(tree, params, site)`: single-site likelihood via
  pruning; O(n*kappa^2).
- Validate: Felsenstein x site counts == full tensor sum for
  enumerated site patterns.

### Layer 2 -- Edge flattenings and rank theorem
- `flatten(P, split)`: reshape tensor into matrix for a split A|B.
- `svd_score(F, rank=kappa)`: Sigma_{i>kappa} sigma_i^2, Chifman-Kubatko score.
- Validate: when split is an edge of the true tree, rank <= kappa
  numerically (small residual). When it's not, rank larger.

### Layer 3 -- SVDQuartets
- `svd_quartet(alignment, [a,b,c,d])`: compute the 3 flattenings for a
  4-taxon alignment, return scores for {ab|cd, ac|bd, ad|bc}, pick
  minimum.
- `svd_tree(alignment)`: for n >= 4 taxa, enumerate all (n choose 4)
  quartets, assemble via quartet-puzzling / median-tree reconstruction.

### Layer 4 -- Simulation
- `simulate_seq(tree, params, length)`: forward-simulate DNA at the
  leaves under a chosen model (JC, K2P, or GMM).
- Used to validate recovery on data with a *known* tree.

### Layer 5 -- Algebraic invariants
- **3x3 minors** of edge flattenings (Allman-Rhodes 2008 Theorem 4
  for 2-state; generators for 4-state).
- **Group-based invariants** in Fourier coordinates:
  - Hadamard transform on ?^{kappa^n} (the Klein-four variant).
  - Sturmfels-Sullivant binomial generators -> evaluate on real data
    and show near-vanishing for the correct topology.

### Layer 6 -- Kruskal identifiability check (theoretical aid)
- Compute k-ranks of the three factor matrices in a rank-kappa tensor
  decomposition; verify Kruskal's sum condition holds -> GMM is
  identifiable.

## Milestones

| # | Deliverable | Layer | Budget |
|---|-------------|-------|--------|
| M1 | Markov model zoo + Newick + joint tensor | 0-1 | 1 session |
| M2 | Flattenings + rank tests + golden validation | 2 | 1 session |
| M3 | SVDQuartets implementation + simulated validation | 3-4 | 1 session |
| M4 | Fourier coords + group-based binomials | 5 | 1 session |
| M5 | Primate mitochondrial real-data demo + figures | -- | 1 session |
| M6 | External: compare against SVDQuartets reference (PAUP* via `paup_py` or manual) | -- | 1 session |

## Data

Already downloaded:
* `primate_mitochondrial.fasta` -- Human, Chimpanzee, Bonobo, Gorilla,
  Orangutan (NCBI NC_012920.1 / NC_001643.1 / NC_001644.1 /
  NC_001645.1 / NC_002083.1), first 8 kb of each mitogenome.

The true topology is biologically well-established:
  (((Chimp, Bonobo), Human), Gorilla, Orangutan)
-- our algorithm must recover this from the raw sequences.

## Stack

- Python 3.10+, NumPy, SciPy (linalg.svd, linalg.expm)
- sympy (for symbolic Markov matrices + invariants, optional)
- matplotlib
- pytest
- External (optional): `ete3`, `biopython`, `dendropy` as reference
  parsers; `sagemath` for Grobner verification (via subprocess).

## Success criteria
- Felsenstein log-likelihood matches brute-force tensor sum to 1e-9
  on 4-taxon, 200-bp simulated sequences.
- On simulated 4-taxon data at 10k sites under JC69, SVDQuartets
  recovers the true topology 100% of the time.
- On the 5-primate mitochondrial data, the ranked quartet scores
  agree with the biologically established topology.
- Sturmfels-Sullivant binomial invariants for JC69 evaluate to <=
  0.01 (normalized) on estimated frequencies from the true topology,
  and >= 0.05 on flattenings of incorrect splits.
