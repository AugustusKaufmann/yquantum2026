from __future__ import annotations

"""
Classical reference methods used to validate the quantum experiments.

These baselines are important for the presentation because they let us answer:
"Did the quantum routine find something actually good?"
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from portfolio_model import AggregatedDataset, QuboModel, build_qubo_model
from results_metrics import evaluate_selection


@dataclass(slots=True)
class ExactSearchSummary:
    all_records: list[dict]
    best_overall: dict
    best_feasible: dict


def enumerate_all_bitstrings(n_bits: int) -> np.ndarray:
    """
    Enumerate the full 2^n binary search space.
    """

    return np.asarray(list(product((0, 1), repeat=n_bits)), dtype=int)


def exact_bruteforce_search(
    aggregated: AggregatedDataset,
    qubo_model: QuboModel,
    *,
    alpha: float = 0.95,
) -> ExactSearchSummary:
    """
    Exhaustively score every possible 8-bit portfolio.

    For n=8 this is only 256 states, which makes it a perfect ground-truth
    validator for the hackathon demo.
    """

    candidates = enumerate_all_bitstrings(len(aggregated.sector_names))
    all_records = [evaluate_selection(aggregated, qubo_model, x, alpha=alpha) for x in candidates]

    best_overall = min(all_records, key=lambda record: record["objective_unscaled"])
    feasible_records = [record for record in all_records if record["feasible"]]
    best_feasible = min(feasible_records, key=lambda record: record["objective_unscaled"])

    return ExactSearchSummary(
        all_records=all_records,
        best_overall=best_overall,
        best_feasible=best_feasible,
    )


def greedy_selection_baseline(
    aggregated: AggregatedDataset,
    qubo_model: QuboModel,
    *,
    alpha: float = 0.95,
) -> dict:
    """
    A simple explainable baseline: add one sector at a time while the objective
    keeps improving the most.
    """

    n = len(aggregated.sector_names)
    target_budget = qubo_model.hyperparameters.budget
    selection = np.zeros(n, dtype=int)
    chosen_indices: set[int] = set()

    for _ in range(target_budget):
        best_candidate_idx = None
        best_candidate_record = None

        for sector_index in range(n):
            if sector_index in chosen_indices:
                continue
            candidate = selection.copy()
            candidate[sector_index] = 1
            candidate_record = evaluate_selection(aggregated, qubo_model, candidate, alpha=alpha)
            if best_candidate_record is None or candidate_record["objective_unscaled"] < best_candidate_record["objective_unscaled"]:
                best_candidate_record = candidate_record
                best_candidate_idx = sector_index

        if best_candidate_idx is None or best_candidate_record is None:
            break

        selection[best_candidate_idx] = 1
        chosen_indices.add(best_candidate_idx)

    return evaluate_selection(
        aggregated,
        qubo_model,
        selection,
        alpha=alpha,
        metadata={"baseline_name": "greedy"},
    )


def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    """
    Project a real vector onto the probability simplex:
        w_i >= 0 and sum_i w_i = 1.

    This is used by the continuous Markowitz baseline.
    """

    vector = np.asarray(vector, dtype=float)
    if np.all(vector >= 0.0) and np.isclose(np.sum(vector), 1.0):
        return vector

    sorted_vector = np.sort(vector)[::-1]
    cumulative_sum = np.cumsum(sorted_vector)
    rho_candidates = np.where(sorted_vector - (cumulative_sum - 1.0) / (np.arange(len(vector)) + 1) > 0.0)[0]
    rho = rho_candidates[-1]
    theta = (cumulative_sum[rho] - 1.0) / (rho + 1)
    return np.maximum(vector - theta, 0.0)


def continuous_markowitz_baseline(
    aggregated: AggregatedDataset,
    qubo_model: QuboModel,
    *,
    steps: int = 3000,
    learning_rate: float = 0.15,
    alpha: float = 0.95,
) -> dict:
    """
    Solve the sector-level continuous mean-variance problem with projected
    gradient descent.

    We avoid a hard dependency on SciPy here so the project is easier to run in
    a fresh PyCharm environment.
    """

    q = qubo_model.hyperparameters.risk_aversion
    weights = np.full(len(aggregated.sector_names), 1.0 / len(aggregated.sector_names), dtype=float)

    for step in range(steps):
        gradient = 2.0 * q * (aggregated.sigma @ weights) - aggregated.mu
        scaled_lr = learning_rate / np.sqrt(step + 1.0)
        weights = project_to_simplex(weights - scaled_lr * gradient)

    objective = float(q * (weights @ aggregated.sigma @ weights) - aggregated.mu @ weights)
    top_indices = np.argsort(weights)[::-1][: qubo_model.hyperparameters.budget]
    discrete_proxy = np.zeros(len(weights), dtype=int)
    discrete_proxy[top_indices] = 1

    proxy_metrics = evaluate_selection(
        aggregated,
        qubo_model,
        discrete_proxy,
        alpha=alpha,
        metadata={"baseline_name": "continuous_top_budget_proxy"},
    )

    proxy_metrics["continuous_weights"] = weights.tolist()
    proxy_metrics["continuous_objective"] = objective
    proxy_metrics["continuous_weight_ranking"] = [
        (aggregated.sector_names[index], float(weights[index]))
        for index in np.argsort(weights)[::-1]
    ]
    return proxy_metrics


def ensure_feasible_penalty(
    aggregated: AggregatedDataset,
    qubo_model: QuboModel,
    *,
    alpha: float = 0.95,
    max_rounds: int = 6,
) -> tuple[QuboModel, ExactSearchSummary]:
    """
    If the best overall QUBO solution is infeasible, increase lambda until
    feasible solutions dominate.

    The report explicitly recommends validating this before running quantum
    experiments, and 8 qubits is small enough that we can do it exactly.
    """

    current_model = qubo_model
    summary = exact_bruteforce_search(aggregated, current_model, alpha=alpha)

    for _ in range(max_rounds):
        if summary.best_overall["feasible"]:
            return current_model, summary

        updated_hyperparameters = current_model.hyperparameters.__class__(
            budget=current_model.hyperparameters.budget,
            risk_aversion=current_model.hyperparameters.risk_aversion,
            penalty_lambda=current_model.hyperparameters.penalty_lambda * 2.0,
            capital_penalty=current_model.hyperparameters.capital_penalty,
            illiquidity_penalty=current_model.hyperparameters.illiquidity_penalty,
        )
        current_model = build_qubo_model(aggregated, updated_hyperparameters, version=current_model.version)
        summary = exact_bruteforce_search(aggregated, current_model, alpha=alpha)

    return current_model, summary

