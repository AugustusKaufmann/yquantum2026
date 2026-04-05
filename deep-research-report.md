# Hackathon Plan for an 8‑Qubit Sector-Selection Portfolio QUBO in Bloqade

## Executive summary

This plan turns the provided 50-asset insurance dataset into an **8-variable (8-qubit) portfolio-selection QUBO** by aggregating assets into **8 sectors** and solving a **binary mean–variance selection problem** with a **cardinality (budget) constraint** implemented via a penalty term. The base model chooses **exactly** \(B=4\) sectors (equal-weighted among the selected sectors), minimizing a risk–return objective built from a **sector covariance** matrix estimated from the workbook’s **scenarios** sheet while optionally adding insurer-aware penalties for **capital charge** and **liquidity**. This matches the challenge expectations to (i) formulate an insurer-relevant portfolio optimization problem, (ii) construct a QUBO/Ising form, and (iii) run a circuit on **8 qubits in Bloqade**, then analyze effects of **noise** and **connectivity** on outcomes. fileciteturn0file0

Key design choices:
- **8 sectors ↔ 8 qubits**: aligns the reduced problem exactly with the “run on Bloqade with 8 qubits” requirement while staying faithful to insurer asset-class thinking. fileciteturn0file0  
- **Binary decision variables** \(x_s\in\{0,1\}\): consistent with standard “pick \(B\) assets out of \(n\)” quantum portfolio formulations. citeturn4view0  
- **Budget \(B=4\)**: “half-of-8” creates a nontrivial combinatorial search (70 feasible portfolios) and mirrors common tutorial baselines that set budget to \(n/2\) for demonstration scale. citeturn4view0  
- **Equal weights among selected sectors** \(w_s=x_s/B\): creates a fully binary implementable selection demo; continuous weights are treated as an extension for scaling beyond 8 qubits.

## Data preparation and sector aggregation

### Sector list and mapping

The README motivates insurer portfolios across these asset classes, which align with the **8 sectors** present in the workbook and intended by this plan: **Cash, Gov Bonds, IG Credit, HY Credit, Equities US, Equities Intl, Real Estate, Infrastructure**. fileciteturn0file0

Define a sector index set:
\[
\mathcal{S}=\{\text{Cash},\text{Gov Bonds},\text{IG Credit},\text{HY Credit},\text{Equities US},\text{Equities Intl},\text{Real Estate},\text{Infrastructure}\}.
\]

Create a mapping from each asset \(i\) to exactly one sector \(s(i)\in\mathcal{S}\) using the workbook’s `assets` sheet.

**Sector aggregation mapping table (template to fill from workbook)**

| Sector \(s\) | Assets in sector \(I_s=\{i:\,s(i)=s\}\) | Within-sector weight \(a_{s,i}\) (default) |
|---|---|---|
| Cash | list asset_ids | \(a_{s,i}=1/|I_s|\) |
| Gov Bonds | list asset_ids | \(a_{s,i}=1/|I_s|\) |
| IG Credit | list asset_ids | \(a_{s,i}=1/|I_s|\) |
| HY Credit | list asset_ids | \(a_{s,i}=1/|I_s|\) |
| Equities US | list asset_ids | \(a_{s,i}=1/|I_s|\) |
| Equities Intl | list asset_ids | \(a_{s,i}=1/|I_s|\) |
| Real Estate | list asset_ids | \(a_{s,i}=1/|I_s|\) |
| Infrastructure | list asset_ids | \(a_{s,i}=1/|I_s|\) |

**Why equal within-sector weights?**  
It keeps aggregation deterministic and transparent for stakeholders and ensures the sector return time series is directly computable from the scenarios matrix without introducing extra continuous parameters.

### Compute sector scenario returns from the scenarios sheet

Let the scenarios sheet provide a matrix of asset returns:
\[
R \in \mathbb{R}^{T\times N},\quad R_{t,i}=\text{return of asset }i\text{ in scenario }t,
\]
where each row is one scenario and each column corresponds to an asset.

Define the **sector aggregation matrix** \(A\in\mathbb{R}^{8\times N}\):
\[
A_{s,i}=\begin{cases}
\frac{1}{|I_s|}, & i\in I_s \\
0, & i\notin I_s
\end{cases}
\]

Then sector scenario returns are:
\[
R^{(sec)} = R A^\top \in \mathbb{R}^{T\times 8},\quad
R^{(sec)}_{t,s}=\sum_{i=1}^N A_{s,i}R_{t,i}
= \frac{1}{|I_s|}\sum_{i\in I_s}R_{t,i}.
\]

This step uses the scenarios exactly as provided and requires no additional assumptions about correlations; the covariance emerges empirically from scenario co-movements.

### Compute sector expected returns and the \(8\times 8\) covariance from scenarios

Treat scenarios as equally likely unless scenario probabilities are provided (none are referenced in the README). If probabilities \(p_t\) exist, replace simple averages with weighted averages; otherwise use:

**Sector expected return vector**
\[
\mu \in \mathbb{R}^{8},\quad
\mu_s = \frac{1}{T}\sum_{t=1}^T R^{(sec)}_{t,s}.
\]

**Sector covariance matrix**
\[
\Sigma \in \mathbb{R}^{8\times 8},\quad
\Sigma_{s,u}=\frac{1}{T-1}\sum_{t=1}^T\left(R^{(sec)}_{t,s}-\mu_s\right)\left(R^{(sec)}_{t,u}-\mu_u\right).
\]

This matches the Markowitz risk term structure \(w^\top \Sigma w\), where \(\Sigma\) is the covariance of returns. citeturn3search1turn3search15

**Validation check (recommended, fast, stakeholder-friendly)**  
Compute one or both of:
- Compare \(\mu_s\) from scenarios to the sector average of the `exp_return` column; large discrepancies can be explained as “scenario horizon differs from long-run expected return,” which is common in stress/scenario datasets.
- Compare \(\Sigma\) computed from scenarios to an aggregation of the asset covariance sheet using \(\Sigma^{(sec)}\approx A\,\Sigma^{(asset)}A^\top\). Agreement suggests your scenario set is consistent with the given covariance.

### Portfolio return per scenario for later stress testing

Given a binary selection \(x\in\{0,1\}^8\) and equal-weight among selected sectors \(w_s=x_s/B\), the portfolio return in scenario \(t\) is:
\[
r^{(port)}_t(x) = \sum_{s\in\mathcal{S}} \frac{x_s}{B}R^{(sec)}_{t,s}.
\]

This will drive scenario-based validation metrics later.

### Workflow overview in mermaid

```mermaid
flowchart TD
  A[Load workbook: assets, scenarios] --> B[Map each asset to 1 of 8 sectors]
  B --> C[Compute sector scenario returns R_sec = R * A^T]
  C --> D[Compute mu (mean) and Sigma (covariance) from scenarios]
  D --> E[Choose B=4, q, lambda and build base QUBO]
  E --> F[Add insurer penalties: capital_charge, liquidity]
  F --> G[Convert QUBO Q -> Ising h,J (optional) and build QAOA ansatz]
  G --> H[Run Bloqade: noiseless]
  H --> I[Run Bloqade: noise models, connectivity variants]
  I --> J[Collect bitstring distributions]
  J --> K[Evaluate: objective, return, variance, capital, liquidity, stress metrics]
  K --> L[Compare vs classical baselines and write deliverables]
```

## Base binary QUBO model

### Decision variables and budget

Let each qubit represent one sector:
\[
x_s \in \{0,1\} \quad \text{for } s\in\mathcal{S}.
\]
Interpretation: \(x_s=1\) means “include sector \(s\) in the portfolio.”

Enforce selection of exactly \(B\) sectors:
\[
\mathbf{1}^\top x = B.
\]

This is the standard “portfolio selection” (cardinality-constrained) formulation used in canonical quantum portfolio tutorials. citeturn4view0

### Why choose \(B=4\) for an 8-sector, 8-qubit demo

Choosing \(B=4\) is justified for a hackathon demo because:
- It avoids the trivial case \(B=8\) (“select everything”).  
- It enforces diversification while preserving meaningful choice: the feasible set has \(\binom{8}{4}=70\) portfolios, which is large enough to test QAOA behavior and noise sensitivity but still small enough for exact classical verification. citeturn4view0  
- It is aligned with common tutorial choices (“budget = num_assets // 2”) for didactic scale. citeturn4view0

### Objective function

Using equal weights among selected sectors \(w=x/B\), the risk–return objective becomes:
\[
\min_{x\in\{0,1\}^8}\;
\frac{q}{B^2}x^\top\Sigma x \;-\;\frac{1}{B}\mu^\top x\;+\;\lambda(\mathbf{1}^\top x - B)^2.
\]

This is exactly the Markowitz mean–variance selection structure (risk term from covariance, return term from expected return), with a penalty encoding the budget constraint to form an unconstrained binary model (QUBO). fileciteturn0file0 citeturn4view0turn3search1turn3search15

### Expand the penalty term (for Q matrix construction)

Because \(x_s^2=x_s\) for binary variables:
\[
(\mathbf{1}^\top x - B)^2
= \left(\sum_s x_s\right)^2 -2B\sum_s x_s + B^2
= \sum_s x_s + 2\sum_{s<u} x_s x_u -2B\sum_s x_s + B^2.
\]
So it contributes:
- Linear: \(\lambda(1-2B)\sum_s x_s\)
- Quadratic: \(2\lambda\sum_{s<u} x_s x_u\)
- Constant: \(\lambda B^2\) (can be dropped; constants do not affect the argmin). citeturn4view1turn0search11

## Insurer-aware extensions with capital and liquidity

The README explicitly frames this as an insurer portfolio problem with **capital/solvency** and **liquidity** considerations—not just risk/return. fileciteturn0file0  
So the strongest hackathon story is: “Start from Markowitz QUBO; add insurer-specific penalties; quantify tradeoffs.”

### Sector-level capital charge and liquidity from the assets sheet

Let each asset \(i\) have:
- `capital_charge_i` (higher is worse)
- `liquidity_i` (higher is better, ordinal)

Compute sector summaries (choose one rule and keep it consistent):
- **Default (simple, robust)**: sector mean  
  \[
  c_s=\frac{1}{|I_s|}\sum_{i\in I_s}\texttt{capital\_charge}_i,\qquad
  \ell_s=\frac{1}{|I_s|}\sum_{i\in I_s}\texttt{liquidity}_i.
  \]
- Alternative for stakeholder alignment: sector median (less sensitive to outliers).

### Normalization (required before mixing with covariance/return)

Because \(\Sigma\) and \(\mu\) are in return units while capital charge and liquidity are in different scales, normalize to comparable ranges:

**Min–max normalized capital penalty**
\[
\tilde c_s=\frac{c_s-\min_{u}c_u}{\max_{u}c_u-\min_{u}c_u}\in[0,1].
\]

**Min–max normalized liquidity score**
\[
\tilde \ell_s=\frac{\ell_s-\min_{u}\ell_u}{\max_{u}\ell_u-\min_{u}\ell_u}\in[0,1].
\]

Convert liquidity into a penalty (illiquidity):
\[
\tilde i_s = 1-\tilde \ell_s \in[0,1].
\]

### Add insurer penalties to the objective

Because you want penalties to reflect the *average per selected sector* (so results remain comparable across different \(B\)), scale by \(1/B\):

\[
\min_x\;
\frac{q}{B^2}x^\top\Sigma x
-\frac{1}{B}\mu^\top x
+\lambda(\mathbf{1}^\top x - B)^2
+\frac{\gamma}{B}\tilde c^\top x
+\frac{\eta}{B}\tilde i^\top x.
\]

Interpretation:
- \(\gamma\ge 0\) pushes toward **capital-efficient** sector choices.
- \(\eta\ge 0\) pushes toward **more liquid** sector choices.

This matches the insurer constraints and motivations described in the challenge brief. fileciteturn0file0

### Initial hyperparameters and tuning strategy for \(\gamma,\eta\)

Because \(\tilde c_s,\tilde i_s\in[0,1]\), a practical initial range is:
- \(\gamma\in[0,1]\)
- \(\eta\in[0,1]\)

Start at:
- \(\gamma_0=0.25\)
- \(\eta_0=0.25\)

Then tune using classical brute force on 8 variables (exact) before quantum runs:
- Increase \(\gamma\) until the chosen portfolio’s average capital charge hits a stakeholder target band (or improves materially without collapsing return).
- Increase \(\eta\) until average liquidity meets an internal “minimum liquidity comfort” threshold.

This exact-tune-then-run-quantum workflow keeps the narrative clean: “We select hyperparameters based on business constraints, then see how noise/connectivity affects reaching the best solution.”

## Q matrix construction and mapping to Bloqade

### Convert the objective into a QUBO matrix \(Q\)

A QUBO is commonly written as:
\[
f(x)=\sum_{i}Q_{ii}x_i + \sum_{i<j}Q_{ij}x_ix_j,
\]
equivalently \(x^\top Q x\) with \(Q\) upper-triangular. citeturn4view1

Fix an ordering of sectors to indices \(0,\dots,7\) (document it in your README; do not change it later).

Let:
- \(a=\frac{q}{B^2}\)
- \(b=\frac{1}{B}\)
- \(g=\frac{\gamma}{B}\)
- \(e=\frac{\eta}{B}\)

Then the **upper-triangular Q entries** for the insurer-extended model are:

**Diagonal terms**
\[
Q_{ss} = a\,\Sigma_{ss} \;-\; b\,\mu_s \;+\;\lambda(1-2B)\;+\;g\,\tilde c_s\;+\;e\,\tilde i_s.
\]

**Off-diagonal terms** for \(s<u\)
\[
Q_{su} = 2a\,\Sigma_{su} \;+\;2\lambda.
\]

This follows directly from expanding \(x^\top\Sigma x\) (which produces \(2\Sigma_{su}x_sx_u\) for \(s<u\)) and expanding the penalty term above. The QUBO definition and the “diagonal = linear, off-diagonal = quadratic” convention match standard references. citeturn4view1turn0search11

### Q matrix example table (structure, not numbers)

Below is the **deliverable format** your repository should include (filled with computed coefficients). Show it once, then store the numeric matrix in a file for reproducibility.

| \(Q\) (upper triangular) | Cash | Gov | IG | HY | Eq US | Eq Intl | RE | Infra |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Cash** | \(Q_{00}\) | \(Q_{01}\) | \(Q_{02}\) | \(Q_{03}\) | \(Q_{04}\) | \(Q_{05}\) | \(Q_{06}\) | \(Q_{07}\) |
| **Gov** | 0 | \(Q_{11}\) | \(Q_{12}\) | \(Q_{13}\) | \(Q_{14}\) | \(Q_{15}\) | \(Q_{16}\) | \(Q_{17}\) |
| **IG** | 0 | 0 | \(Q_{22}\) | \(Q_{23}\) | \(Q_{24}\) | \(Q_{25}\) | \(Q_{26}\) | \(Q_{27}\) |
| **HY** | 0 | 0 | 0 | \(Q_{33}\) | \(Q_{34}\) | \(Q_{35}\) | \(Q_{36}\) | \(Q_{37}\) |
| **Eq US** | 0 | 0 | 0 | 0 | \(Q_{44}\) | \(Q_{45}\) | \(Q_{46}\) | \(Q_{47}\) |
| **Eq Intl** | 0 | 0 | 0 | 0 | 0 | \(Q_{55}\) | \(Q_{56}\) | \(Q_{57}\) |
| **RE** | 0 | 0 | 0 | 0 | 0 | 0 | \(Q_{66}\) | \(Q_{67}\) |
| **Infra** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | \(Q_{77}\) |

### Scaling for quantum execution

For QAOA-style implementations, coefficients ultimately become **gate angles** (phases). Large coefficient magnitudes can lead to either:
- very small optimal \(\gamma\) values (hard for optimizers), or  
- angles that wrap around \(2\pi\) and reduce interpretability.

Because multiplying the entire objective by a positive constant does not change the optimizer \(x^\*\), apply a global scale:
\[
Q^{(scaled)} = \frac{1}{\alpha}Q,\quad \alpha = \max_{s\le u}|Q_{su}|.
\]
Then build the cost Hamiltonian from \(Q^{(scaled)}\) and keep a record of \(\alpha\) so you can report objective values in original units.

This “scale without changing argmin” principle is a standard modeling convenience in QUBO practice (constants/scales do not alter the minimizer). citeturn0search11turn4view1

### Mapping QUBO to an Ising Hamiltonian (for Bloqade QAOA)

Bloqade’s QAOA examples implement cost functions using commuting entangling phase gates (e.g., CZPhase) on graph edges. citeturn1view1turn1view0  
To use those tools, map \(x\in\{0,1\}\) to spins \(s\in\{-1,+1\}\). A standard transform is:
\[
x_i = \frac{1-s_i}{2}.
\]
Then any QUBO can be written as:
\[
f(x)=\text{const} + \sum_i h_i s_i + \sum_{i<j}J_{ij}s_is_j,
\]
matching the Ising form used in many quantum optimization implementations. citeturn4view1turn0search11

Given QUBO coefficients \(Q_{ii},Q_{ij}\) (upper-triangular), the Ising coefficients are:

- Couplings:
\[
J_{ij} = \frac{Q_{ij}}{4}\quad (i<j)
\]
- Fields:
\[
h_i = -\frac{Q_{ii}}{2} - \frac{1}{4}\sum_{j\ne i}Q_{\min(i,j),\max(i,j)}
\]
- Constant offset can be dropped for optimization.

### Construct the QAOA circuit structure in Bloqade

The README suggests QAOA/VQE/analog approaches as appropriate for the QUBO. fileciteturn0file0  
QAOA, in particular, alternates between a **cost Hamiltonian** \(H_C\) and a **mixer** \(H_M\), controlled by angles \(\gamma,\beta\), and improves with depth \(p\). citeturn3search2turn1view1

For our Ising Hamiltonian:
\[
H_C = \sum_i h_i Z_i + \sum_{i<j}J_{ij}Z_iZ_j,
\quad
H_M = \sum_i X_i.
\]

A standard \(p\)-layer QAOA ansatz is:
\[
\left|\psi(\boldsymbol{\gamma},\boldsymbol{\beta})\right\rangle
=
\prod_{k=1}^{p} e^{-i\beta_k H_M}\, e^{-i\gamma_k H_C}\, |+\rangle^{\otimes 8}
\]
where \(|+\rangle\) is prepared by Hadamards on \(|0\rangle^{\otimes 8}\). citeturn3search2turn1view1

**Bloqade implementation guidance (no code, but implementable steps):**
- Use the Bloqade recommended circuit DSL (SQUIN) to define kernels and allocate qubits, as shown in their circuits tutorial and quick start. citeturn1view2turn1view3  
- Implement \(e^{-i\gamma_k H_C}\) as:
  - single-qubit Z-rotations for \(h_i Z_i\)
  - two-qubit ZZ-phase interactions for \(J_{ij} Z_i Z_j\) (commuting), similar in spirit to the MaxCut QAOA example that applies CZPhase per graph edge. citeturn1view1turn1view0  
- Implement \(e^{-i\beta_k H_M}\) as a layer of X-rotations (one per qubit). citeturn1view1turn3search2  

### Embedding an 8-node fully connected graph and parallelizing it

Your QUBO is effectively a dense graph because:
- the covariance term generally couples many pairs, and  
- the budget penalty adds a complete-graph coupling \(2\lambda\) across all pairs.

So the logical cost graph tends toward **\(K_8\)**.

Bloqade’s “parallelism of static circuits” tutorial highlights that in QAOA-like circuits, commuting two-qubit phase gates can be parallelized using **edge coloring**, where each color is a circuit “moment.” citeturn1view0  
For \(K_8\), an optimal edge coloring uses **7** matchings, implying you can schedule all ZZ interactions in **7 parallel moments** per cost layer (instead of 28 sequential gates). This is an excellent “quantum hardware insight” talking point because it links dense QUBOs to circuit depth and scheduling.

**Plan deliverable:** show three cost-layer implementations (as in Bloqade’s tutorial approach):  
- naive sequential ordering,  
- auto-parallelized using Bloqade tooling,  
- hand-colored 7-moment schedule (document the matching sets). citeturn1view0

### Choosing \(q,\lambda,\gamma,\eta\) initial values and tuning

Start from conventional tutorial-scale values and then adapt by scaling:

1. **Pick baseline \(q\)**: start \(q=0.5\) (common in portfolio selection demos) and later sweep. citeturn4view0  
2. **Pick initial \(\lambda\)**: choose \(\lambda\) large enough that violating \(\mathbf{1}^\top x=B\) is always worse than any improvement from risk/return terms. A practical initial heuristic consistent with tutorial practice is \(\lambda\approx n\) (here \(n=8\)) after scaling. citeturn4view0  
3. **Pick insurer weights**: \(\gamma=\eta=0.25\) with normalized penalties (because those terms are in \([0,1]\)).  
4. **Scale \(Q\)** by \(\alpha\) (above).  
5. **Validate feasibility dominance**: confirm the brute-force minimum solution satisfies \(\mathbf{1}^\top x=B\). If not, increase \(\lambda\) (e.g., multiply by 2) until feasible solutions dominate.

**Recommended hyperparameter grid (deliverable)**

| Parameter | Role | Initial | Sweep values (small) | Notes |
|---|---|---:|---|---|
| \(B\) | sectors selected | 4 | {3,4,5} | keep 4 as primary story citeturn4view0 |
| \(q\) | risk aversion | 0.5 | {0.1, 0.5, 1.0, 2.0} | larger \(q\) penalizes covariance more fileciteturn0file0 |
| \(\lambda\) | budget penalty | 8 (scaled units) | {4, 8, 16, 32} | increase until feasibility is guaranteed citeturn4view0 |
| \(\gamma\) | capital penalty | 0.25 | {0, 0.25, 0.5, 1.0} | with \(\tilde c\in[0,1]\) |
| \(\eta\) | illiquidity penalty | 0.25 | {0, 0.25, 0.5, 1.0} | with \(\tilde i\in[0,1]\) |
| \(p\) | QAOA depth | 1 | {1,2,3} | deeper circuits can help but are noisier citeturn3search2turn1view1 |

## Classical baselines, evaluation, and scenario validation

### Classical baselines (required for a strong submission)

Use baselines that directly reflect the evaluation criteria and demonstrate technical correctness.

**Exact brute-force over 8 qubits**  
Enumerate all \(2^8=256\) bitstrings, filter those with \(\mathbf{1}^\top x=B\), and compute:
- objective value \(f(x)\)
- expected return \(\bar r(x)=\frac{1}{B}\mu^\top x\)
- variance \(\mathrm{Var}(x)=\frac{1}{B^2}x^\top\Sigma x\)
This provides the ground-truth optimum for every hyperparameter set and is extremely persuasive in judging. (The Qiskit tutorial explicitly frames this problem as a binary selection model with budget \(B\).) citeturn4view0turn4view1

**Greedy heuristic (fast, explainable)**  
Start with empty selection and add the sector that most improves the objective at each step until \(B\) sectors are chosen. This is a stakeholder-friendly “classical heuristic” comparator.

**Continuous Markowitz at the sector level (conceptual baseline)**  
Solve the continuous problem:
\[
\min_{w\ge 0,\ \mathbf{1}^\top w=1}\ q w^\top \Sigma w - \mu^\top w
\]
and compare its allocations to the discrete selection solution. This anchors your story in classical portfolio theory. citeturn3search1turn3search15  
(Optional) Convert it into a “top-4 sectors by weight” proxy to compare with the binary solution.

### Evaluation metrics (report these consistently)

For any selected bitstring \(x\):

- **Selected sectors:** \(B(x)=\mathbf{1}^\top x\) (should equal 4)
- **Expected return:** \(\bar r(x)=\frac{1}{B}\mu^\top x\)
- **Variance:** \(v(x)=\frac{1}{B^2}x^\top\Sigma x\)
- **Volatility:** \(\sqrt{v(x)}\)
- **Avg capital charge:** \(\bar c(x)=\frac{1}{B}c^\top x\)
- **Avg liquidity:** \(\bar \ell(x)=\frac{1}{B}\ell^\top x\)
- **Objective:** \(f(x)\) (scaled and unscaled, report both)

### Scenario-based validation (stress and resilience)

Using portfolio scenario returns \(r^{(port)}_t(x)\):

- **Worst-case (max drawdown proxy):**
\[
\text{Worst}(x) = \min_t r^{(port)}_t(x)
\]
- **VaR at level \(\alpha\)** (e.g., 95%): negative quantile of returns (or quantile of losses)
- **CVaR at level \(\alpha\)**: average of tail losses beyond VaR (optional; strong but not required)

These metrics align with the challenge’s emphasis on resilience to correlated shocks and insurer safety. fileciteturn0file0

## Bloqade experiment plan, noise/connectivity studies, and deliverables

### Bloqade execution plan (noiseless then noisy)

Bloqade provides guidance for building circuits as kernels and running them on simulation/hardware backends; it is designed as “hardware-first,” aiming for simulations that mirror hardware behavior. citeturn1view3turn2search5

**Noiseless / idealized runs**
- Run QAOA at \(p=1\), then \(p=2\), then \(p=3\).
- For each \(p\), optimize angles \(\boldsymbol{\gamma},\boldsymbol{\beta}\) using a classical optimizer (document which optimizer, stopping criteria, and random seeds).
- Collect the measured bitstring distribution; compute:
  - probability mass on feasible solutions \(\mathbf{1}^\top x=B\)
  - probability mass on the true optimum bitstring (from brute force)
  - best-feasible objective found in samples

**Noise-injected runs**
The Bloqade ecosystem includes heuristic noise models for digital hardware workflows and discusses noise modeling in tutorial form (e.g., GHZ with noise). citeturn2search0turn2search3turn2search5  
Additionally, Bloqade provides Cirq-compatible noise model components describing asymmetric depolarizing noise and other effects in a Gemini-like architecture model. citeturn2search4turn2search19

Run progressively:
- baseline hardware-like noise model
- amplified depolarizing noise (stress the algorithm)
- readout error variations

Record how success probability and best-found objective degrade.

### Connectivity and hardware-structure studies

The README explicitly encourages exploring how **dense problem graphs** interact with **hardware connectivity** and performance. fileciteturn0file0

Do two controlled connectivity experiments:

**Scheduling/parallelism experiment (same graph, different compilation)**
- Compare naive sequential scheduling vs auto-parallel vs hand-edge-colored scheduling (7-moment for \(K_8\)). This directly follows Bloqade’s parallelism tutorial emphasis. citeturn1view0

**Restricted connectivity experiment (different logical graph)**
- Prune small-magnitude couplings \(J_{ij}\) (e.g., keep top-k edges by \(|J_{ij}|\)) to create sparser graphs and measure:
  - how sparsification affects solution quality (objective vs full graph)
  - how reduced two-qubit gate count affects noise robustness

This yields a concrete “hardware insight”: sometimes a slightly approximated objective (sparser graph) can outperform a full dense objective under realistic noise due to reduced depth.

### Multiple seeds and distributional reporting

Because QAOA is variational and measurement-sampling based, report distributions, not just single outcomes:
- run each configuration across multiple initializations (seeds)
- show variance in achieved objective and in probability of the optimal bitstring  
This matches the hybrid nature of QAOA and the challenge’s call for analysis beyond a single run. citeturn3search2turn0search1

### Expected output table schema (deliverable)

Store all experiments in one tidy table (CSV/Parquet) with an explicit schema:

| Column | Type | Description |
|---|---|---|
| run_id | string | unique key |
| model_version | string | base / +capital / +liquidity / both |
| q, lambda, gamma, eta | float | QUBO hyperparameters |
| B | int | budget (expected 4) |
| p | int | QAOA depth |
| backend | string | Bloqade simulator / hardware-like / etc. |
| noise_model | string | none / gemini heuristic / depolarizing sweep / readout sweep |
| connectivity_mode | string | full / pruned / restricted + swaps (if used) |
| seed | int | random seed |
| bitstring | string | measured bitstring |
| freq | float | frequency/probability |
| feasible | bool | whether \(\mathbf{1}^\top x = B\) |
| objective_scaled | float | energy under scaled Q |
| objective_unscaled | float | energy under original Q |
| exp_return | float | \(\bar r(x)\) |
| variance | float | \(v(x)\) |
| capital_avg | float | \(\bar c(x)\) |
| liquidity_avg | float | \(\bar \ell(x)\) |
| worst_scenario_return | float | \(\min_t r_t^{(port)}(x)\) |
| var95 | float | optional |
| cvar95 | float | optional |

### Required deliverables (as requested)

**README (math + data prep + reproducibility)**
- State data sources: workbook sheets used and README challenge context. fileciteturn0file0  
- Define sector mapping, aggregation formulas, \(\mu,\Sigma\), and QUBO objective.
- Specify the fixed sector→qubit ordering.

**Q matrix artifact**
- Provide the computed \(8\times 8\) \(Q\) (scaled and unscaled) as a table and as a machine-readable file.
- Include the derived Ising \((h,J)\) and constant offset for completeness.

**Bloqade run instructions**
- Reference Bloqade’s “circuits with Bloqade” and quick start as the canonical workflow for kernels and backends. citeturn1view2turn1view3turn2search5  
- Reference Bloqade’s QAOA and parallelism tutorials to justify your ansatz and circuit-depth optimization. citeturn1view1turn1view0  
- Reference noise tutorial/docs for how noise is added and interpreted. citeturn2search0turn2search4turn2search5  

**Plots/tables to produce**
- Bitstring frequency bar chart (top 10 bitstrings) per experiment
- Objective histogram of sampled bitstrings (feasible-only vs all)
- Success probability vs noise strength
- Depth (gate count or moments) vs performance for the three scheduling variants
- Comparison table: best classical feasible vs best sampled feasible vs expected return/variance/capital/liquidity/stress metrics

### Discussion points aligned to judging criteria

**Quantum hardware insight (30%)**
- Dense \(K_8\)-like coupling from covariance + budget penalty; show how edge coloring reduces depth from 28 sequential to 7 parallel moments for ZZ layers. citeturn1view0turn0search1  
- Demonstrate performance under noise and how reducing depth (parallelism/sparsification) can improve outcomes.

**Quality of technical solution (30%)**
- Match the canonical binary selection formulation and budget constraint penalty mapping. citeturn4view0turn4view1  
- Provide explicit \(Q_{ii},Q_{ij}\) formulas and (optionally) Ising mapping.
- Verify outcomes against brute-force optimum for every hyperparameter set.

**Applicability to insurer (20%)**
- Use insurer asset classes (sectors) and incorporate solvency-like capital charge and liquidity, as highlighted in the problem statement. fileciteturn0file0  
- Report stress metrics from scenarios (worst-case, tail risk proxies).

**Analysis of approaches (10%)**
- Compare noiseless vs noisy; parallel vs sequential; dense vs pruned; base vs insurer-aware objective.
- Compare quantum results to exact classical optimum and heuristics.

**Presentation (10%)**
- Keep the narrative stakeholder-readable: “choose 4 of 8 sectors,” risk/return tradeoff, then insurer penalties.
- Provide a single summary figure/table that a non-technical sponsor can interpret.

### Notes on unspecified link content

The provided README references additional resources (Pauli propagation libraries and external QAOA tutorials). These can be cited as optional scaling/simulation enhancements, but this plan does not assume they are required for the core deliverables. fileciteturn0file0