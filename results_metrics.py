from __future__ import annotations

"""
Metric calculations shared by classical and quantum experiments.
"""

from typing import Any

import numpy as np

from portfolio_model import AggregatedDataset, QuboModel, qubo_energy_upper


def selection_to_bitstring(selection: np.ndarray) -> str:
    """
    Represent a binary portfolio as a readable bitstring, e.g. "10110010".
    """

    return "".join(str(int(bit)) for bit in np.asarray(selection, dtype=int))


def bitstring_to_selection(bitstring: str) -> np.ndarray:
    """
    Convert the textual bitstring back into a NumPy vector.
    """

    return np.array([int(char) for char in bitstring.strip()], dtype=int)


def portfolio_weights_for_selection(selection: np.ndarray, budget: int) -> np.ndarray:
    """
    The report defines equal weights among selected sectors as x / B.

    Note that this uses the fixed target budget B even for infeasible bitstrings.
    That choice keeps the metrics aligned with the exact QUBO definition.
    """

    selection = np.asarray(selection, dtype=float)
    return selection / float(budget)


def portfolio_scenario_returns(
    aggregated: AggregatedDataset,
    selection: np.ndarray,
    budget: int,
) -> np.ndarray:
    """
    Scenario returns for one selected portfolio.
    """

    weights = portfolio_weights_for_selection(selection, budget)
    return aggregated.sector_return_matrix @ weights


def compute_var_cvar(
    returns: np.ndarray,
    alpha: float = 0.95,
) -> tuple[float, float]:
    """
    Convert returns into positive downside metrics by looking at losses.
    """

    losses = -np.asarray(returns, dtype=float)
    var_level = float(np.quantile(losses, alpha))
    tail_losses = losses[losses >= var_level]
    cvar_level = float(np.mean(tail_losses)) if len(tail_losses) else var_level
    return var_level, cvar_level


def evaluate_selection(
    aggregated: AggregatedDataset,
    qubo_model: QuboModel,
    selection: np.ndarray,
    *,
    alpha: float = 0.95,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute all per-bitstring metrics requested by the report.
    """

    selection = np.asarray(selection, dtype=int)
    budget = qubo_model.hyperparameters.budget

    scenario_returns = portfolio_scenario_returns(aggregated, selection, budget)
    variance = float((selection @ aggregated.sigma @ selection) / (budget**2))
    volatility = float(np.sqrt(max(variance, 0.0)))
    exp_return = float(np.dot(aggregated.mu, selection) / budget)
    capital_avg = float(np.dot(aggregated.capital, selection) / budget)
    liquidity_avg = float(np.dot(aggregated.liquidity, selection) / budget)
    worst_scenario_return = float(np.min(scenario_returns))
    var95, cvar95 = compute_var_cvar(scenario_returns, alpha=alpha)

    record: dict[str, Any] = {
        "bitstring": selection_to_bitstring(selection),
        "selected_count": int(np.sum(selection)),
        "feasible": bool(np.sum(selection) == budget),
        "objective_unscaled": float(qubo_energy_upper(qubo_model.Q_unscaled, selection)),
        "objective_scaled": float(qubo_energy_upper(qubo_model.Q_scaled, selection)),
        "exp_return": exp_return,
        "variance": variance,
        "volatility": volatility,
        "capital_avg": capital_avg,
        "liquidity_avg": liquidity_avg,
        "worst_scenario_return": worst_scenario_return,
        "var95": var95,
        "cvar95": cvar95,
    }
    if metadata:
        record.update(metadata)
    return record


def summarize_probability_distribution(
    aggregated: AggregatedDataset,
    qubo_model: QuboModel,
    probabilities: np.ndarray,
    optimum_bitstring: str,
    *,
    alpha: float = 0.95,
) -> dict[str, Any]:
    """
    Summarize a full QAOA output distribution.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    budget = qubo_model.hyperparameters.budget
    n = len(aggregated.sector_names)

    feasible_mass = 0.0
    optimum_mass = 0.0
    best_feasible_objective = None
    expected_objective = 0.0

    for state_index, probability in enumerate(probabilities):
        if probability <= 0.0:
            continue
        selection = np.array([(state_index >> bit) & 1 for bit in range(n)], dtype=int)
        objective = float(qubo_energy_upper(qubo_model.Q_scaled, selection))
        expected_objective += probability * objective

        if int(np.sum(selection)) == budget:
            feasible_mass += probability
            if best_feasible_objective is None or objective < best_feasible_objective:
                best_feasible_objective = objective

        if selection_to_bitstring(selection) == optimum_bitstring:
            optimum_mass += probability

    return {
        "feasible_probability_mass": float(feasible_mass),
        "optimal_bitstring_probability": float(optimum_mass),
        "expected_scaled_objective": float(expected_objective),
        "best_feasible_scaled_objective_in_distribution": best_feasible_objective,
        "var_alpha": alpha,
    }


def sample_counts_from_probabilities(
    probabilities: np.ndarray,
    *,
    shots: int,
    seed: int,
) -> dict[str, int]:
    """
    Turn an exact probability vector into measurement-like counts.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / np.sum(probabilities)
    rng = np.random.default_rng(seed)
    outcomes = rng.choice(len(probabilities), p=probabilities, size=shots)
    counts: dict[str, int] = {}
    n = int(np.log2(len(probabilities)))
    for state_index in outcomes:
        bitstring = "".join(str((state_index >> bit) & 1) for bit in range(n))
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts

