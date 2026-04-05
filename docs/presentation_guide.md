# Presentation Guide

This repo already saves the final presentation figures into `artifacts/figures/`.

Use them in roughly this order:

## 1. Problem Setup

Use:

- `artifacts/figures/portfolio_tradeoff_frontier.png`

Why:

- Quickly shows how the four model variants trade off return, risk, capital, and liquidity.
- Makes the Hartford relevance obvious before talking about quantum methods.

## 2. Scenario Risk

Use:

- `artifacts/figures/scenario_return_distributions.png`

Why:

- Uses the real 1,200-scenario dataset directly.
- Shows stress-tail behavior and makes CVaR visible, which is important for insurer stakeholders.

## 3. Technical Formulation / Why QAOA Is Hard

Use:

- `artifacts/figures/energy_landscape_base.png`

Why:

- Shows the exact 256-state energy landscape.
- Helps explain why shallow QAOA struggles to strongly prefer the single best feasible portfolio.

## 4. Parameter Search

Use:

- `artifacts/figures/qaoa_parameter_landscape_base_p1.png`
- `artifacts/figures/qaoa_convergence_base.png`

Why:

- The heatmap shows the actual QAOA objective landscape in `(\gamma, \beta)`.
- The convergence figure shows how the optimizer improved the expected energy across `p=1,2,3`.

## 5. Quantum vs Classical

Use:

- `artifacts/figures/classical_vs_quantum.png`
- `artifacts/figures/quantum_probability_summary.png`

Why:

- The table answers the direct comparison question: the best sampled quantum portfolio matches the exact classical optimum.
- The probability summary makes the honest limitation clear: QAOA finds the optimum, but does not concentrate sharply on it.

## 6. Hardware Insight

Use:

- `artifacts/figures/atom_layout.png`
- `artifacts/figures/connectivity_tradeoff.png`
- `artifacts/figures/success_vs_noise.png`

Why:

- `atom_layout.png` gives the neutral-atom-specific story.
- `connectivity_tradeoff.png` shows that scheduling matters because it reduces depth without hurting quality much.
- `success_vs_noise.png` shows how performance degrades under more realistic noise assumptions.

## Recommended Short Slide Sequence

If you want a compact deck:

1. `portfolio_tradeoff_frontier.png`
2. `scenario_return_distributions.png`
3. `energy_landscape_base.png`
4. `qaoa_parameter_landscape_base_p1.png`
5. `qaoa_convergence_base.png`
6. `classical_vs_quantum.png`
7. `quantum_probability_summary.png`
8. `atom_layout.png`
9. `connectivity_tradeoff.png`
10. `success_vs_noise.png`

## If You Need To Cut Slides

Keep these five first:

1. `portfolio_tradeoff_frontier.png`
2. `energy_landscape_base.png`
3. `classical_vs_quantum.png`
4. `connectivity_tradeoff.png`
5. `success_vs_noise.png`
