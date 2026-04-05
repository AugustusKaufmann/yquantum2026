# Technical Notes

This document is the concise technical companion to the code.

## 1. Data Reduction

The Hartford workbook contains 50 assets and 1,200 scenarios. Rather than choosing a single “representative” asset per sector, this project aggregates all assets into 8 fixed sectors:

- Cash
- Gov Bonds
- IG Credit
- HY Credit
- Equities US
- Equities Intl
- Real Estate
- Infrastructure

Let `R` be the scenario-return matrix and `A` the asset-to-sector aggregation matrix. Sector-level scenario returns are:

```text
R_sec = R A^T
```

From these we compute:

```text
μ = mean(R_sec)
Σ = cov(R_sec)
```

This keeps the real scenario structure in the reduced 8-qubit model.

## 2. QUBO

We select exactly `B = 4` sectors with binary variables `x_i in {0,1}`.

Equal-weight portfolio:

```text
w_i = x_i / B
```

Base objective:

```text
min_x (q / B^2) x^T Σ x - (1 / B) μ^T x + λ (1^T x - B)^2
```

Insurer-aware objective:

```text
min_x (q / B^2) x^T Σ x
      - (1 / B) μ^T x
      + λ (1^T x - B)^2
      + (γ / B) c~^T x
      + (η / B) i~^T x
```

Where:

- `c~` is normalized capital charge
- `i~` is normalized illiquidity

The project evaluates four variants:

- `base`
- `capital`
- `liquidity`
- `both`

## 3. Ising Mapping

The QUBO is converted to an Ising Hamiltonian with:

```text
x_i = (1 - s_i) / 2
```

which yields:

```text
H_C = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j
```

This is the cost Hamiltonian used by the QAOA routines.

## 4. Classical Validation

Because the reduced problem has only 8 binary variables, the repo computes the exact optimum by brute force over all `256` bitstrings. This is important because it gives a clean benchmark for every quantum experiment.

The exact real-workbook optima are:

- `base`: `00011110`
- `capital`: `11110000`
- `liquidity`: `11001100`
- `both`: `11110000`

## 5. Quantum Interpretation

The best sampled quantum bitstring matches the exact classical optimum in all four model versions.

But the quantum distribution is weakly concentrated:

- `P(optimal)` is about `1.03%`
- `P(feasible)` is about `71.96%`
- `P(optimal | feasible)` is about `1.43%`

There are `C(8,4) = 70` feasible portfolios, so:

```text
1 / 70 ≈ 1.43%
```

That means the strongest noiseless QAOA runs are close to uniform over the feasible set. In plain language:

- QAOA learns the budget structure
- but it does not sharply rank the feasible portfolios

This is why the repo emphasizes technical correctness, hardware insight, and Hartford applicability rather than claiming quantum advantage on this 8-qubit instance.

## 6. Hardware-Specific Insight

The repo includes three hardware-oriented analyses:

- `atom_layout.png`: a neutral-atom layout derived from sector correlations using classical MDS
- `connectivity_tradeoff.png`: naive vs auto vs hand vs pruned schedules
- `success_vs_noise.png`: one-zone, two-zone, and depolarizing noise trends

These are the main pieces that address the interplay between problem structure and Bloqade-style execution.

## 7. Hartford-Relevant Risk Metrics

The code does not stop at mean and variance. It also exports:

- worst scenario return
- `VaR95`
- `CVaR95`
- capital charge
- liquidity

This is why the scenario-return and tradeoff figures matter so much in the final presentation.
