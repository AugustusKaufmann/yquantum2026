# YQuantum Hartford Portfolio QUBO

An 8-qubit, insurer-aware portfolio optimization workflow for the Capgemini x The Hartford YQuantum challenge.

This repo takes the Hartford workbook, aggregates the 50 assets into 8 sectors, builds a QUBO and Ising model, solves the reduced problem exactly, and runs QAOA-style experiments with connectivity and noise analysis. The code is organized so the repo is both runnable in PyCharm and presentation-ready for the hackathon.

## What This Repo Does

- Loads the real workbook and aligns the `assets`, `scenarios`, and `covariance` sheets.
- Aggregates 50 assets into 8 fixed sectors:
  `Cash`, `Gov Bonds`, `IG Credit`, `HY Credit`, `Equities US`, `Equities Intl`, `Real Estate`, `Infrastructure`
- Builds four portfolio models:
  - `base`
  - `capital`
  - `liquidity`
  - `both`
- Converts the sector-selection problem into a QUBO and then an Ising Hamiltonian.
- Computes exact classical optima over all `2^8 = 256` bitstrings.
- Compares greedy, continuous, and quantum baselines.
- Studies connectivity scheduling, atom-layout intuition, and noise sensitivity.
- Exports a small, curated figure pack for the presentation.

## Why This Version Is Stronger

Compared with a simpler “pick one best asset per sector” proxy approach, this repo keeps more of the real Hartford structure:

- It uses the scenario sheet directly instead of ignoring it after load.
- It aggregates all assets into sector-level return distributions.
- It computes worst-case return, VaR, and CVaR from the 1,200 scenarios.
- It verifies the budget penalty exactly and forces the quantum comparisons to be against feasible classical optima.
- It keeps the insurer-specific capital and liquidity variants as first-class models all the way through the pipeline.

## Mathematical Model

We choose exactly `B = 4` sectors out of 8.

Binary decision variable:

```text
x_i in {0, 1}
```

Equal-weight portfolio for selected sectors:

```text
w_i = x_i / B
```

Base objective:

```text
min_x (q / B^2) x^T Σ x - (1 / B) μ^T x + λ (1^T x - B)^2
```

Insurer-aware extension:

```text
min_x (q / B^2) x^T Σ x
      - (1 / B) μ^T x
      + λ (1^T x - B)^2
      + (γ / B) c~^T x
      + (η / B) i~^T x
```

Where:

- `μ` is the sector expected return vector
- `Σ` is the sector covariance matrix
- `λ` enforces “pick exactly 4 sectors”
- `γ` penalizes capital charge
- `η` penalizes illiquidity

The scaled QUBO is converted to an Ising Hamiltonian and used by the QAOA experiments.

## Real Workbook Results

Exact best feasible portfolios on the real workbook:

| Model | Bitstring | Selected sectors |
| --- | --- | --- |
| Base | `00011110` | HY Credit, Equities US, Equities Intl, Real Estate |
| Capital | `11110000` | Cash, Gov Bonds, IG Credit, HY Credit |
| Liquidity | `11001100` | Cash, Gov Bonds, Equities US, Equities Intl |
| Both | `11110000` | Cash, Gov Bonds, IG Credit, HY Credit |

Business takeaway:

- `base` chases return and accepts more risk
- `capital` rotates into safer, capital-efficient fixed-income exposure
- `liquidity` keeps some growth exposure but improves balance-sheet flexibility
- `both` shows that, at the chosen penalty settings, capital is the stronger driver

Quantum takeaway:

- The best sampled feasible quantum bitstring matches the exact classical optimum in all four model versions.
- The optimum probability in the strongest noiseless runs is only about `1.03%`.
- Feasible probability mass is about `71.96%`.
- Conditioned on feasibility, the optimum probability is about `1.43%`, which is essentially uniform over the 70 feasible 4-of-8 portfolios.

So the honest conclusion is:

- the workflow is technically correct,
- the hardware/connectivity/noise analysis is meaningful,
- but this reduced 8-sector instance is not a quantum advantage result.

## Repo Layout

```text
.
├── artifacts/
│   ├── figures/             # Curated presentation figures only
│   ├── model_artifacts/     # Exported Q, J, and model metadata
│   └── results/             # CSV/JSON experiment outputs
├── docs/
│   ├── presentation_guide.md
│   └── technical_notes.md
├── bloqade_experiments.py
├── classical_baselines.py
├── make_figures.py
├── portfolio_model.py
├── project_config.py
├── qaoa_simulator.py
├── results_metrics.py
├── run_demo.py
└── investment_dataset_full.xlsx
```

## How To Run

Recommended Python version: `3.11` or `3.12`

Install the base dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional quantum/circuit extras:

```bash
pip install -r requirements-quantum.txt
```

Run the full real-workbook pipeline:

```bash
python run_demo.py --workbook /absolute/path/to/investment_dataset_full.xlsx
```

If the workbook is the copy stored in this repo:

```bash
python run_demo.py --workbook ./investment_dataset_full.xlsx
```

## Main Output Files

Results:

- `artifacts/results/run_summary.json`
- `artifacts/results/classical_results.csv`
- `artifacts/results/quantum_summary_results.csv`
- `artifacts/results/comparison_table.csv`
- `artifacts/results/exact_enumeration_results.csv`
- `artifacts/results/optimizer_trace.csv`

Presentation figures:

- `artifacts/figures/portfolio_tradeoff_frontier.png`
- `artifacts/figures/scenario_return_distributions.png`
- `artifacts/figures/energy_landscape_base.png`
- `artifacts/figures/qaoa_convergence_base.png`
- `artifacts/figures/qaoa_parameter_landscape_base_p1.png`
- `artifacts/figures/atom_layout.png`
- `artifacts/figures/connectivity_tradeoff.png`
- `artifacts/figures/success_vs_noise.png`
- `artifacts/figures/classical_vs_quantum.png`
- `artifacts/figures/quantum_probability_summary.png`

See `docs/presentation_guide.md` for the recommended slide order.

## Most Important Files To Understand

If you only want the shortest path through the code:

1. `run_demo.py`
2. `portfolio_model.py`
3. `classical_baselines.py`
4. `bloqade_experiments.py`
5. `make_figures.py`

## Notes

- The code runs without Bloqade installed by using a local NumPy QAOA simulator and optional Cirq-backed paths when available.
- The repo was cleaned so the tracked figure folder contains only the presentation-quality images, not every exploratory chart from earlier runs.
- The project is set up to be uploaded directly to GitHub as-is.
