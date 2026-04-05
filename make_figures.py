from __future__ import annotations

"""
Artifact export and presentation-grade plotting helpers.

The earlier version of this file produced a large number of exploratory plots.
For the final hackathon repo we keep the exported figure set intentionally
smaller and more purposeful: every saved chart should answer a question the
judges are likely to ask.
"""

import csv
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any


# Matplotlib wants a writable config directory. We set one inside the project so
# plots work smoothly in restricted environments.
_MPL_DIR = Path(__file__).resolve().parent / ".mplconfig"
_MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from portfolio_model import AggregatedDataset, QuboModel
from qaoa_simulator import apply_rx_layer, optimize_qaoa_angles, precompute_ising_energies
from results_metrics import bitstring_to_selection, portfolio_scenario_returns


MODEL_LABELS = {
    "base": "Base",
    "capital": "Capital",
    "liquidity": "Liquidity",
    "both": "Both",
}

MODEL_ORDER = ("base", "capital", "liquidity", "both")


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for record in records for key in record.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _write_matrix_csv(path: str | Path, matrix: np.ndarray, row_labels: list[str], column_labels: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + list(column_labels))
        for row_label, row_values in zip(row_labels, matrix):
            writer.writerow([row_label] + [f"{float(value):.12g}" for value in row_values])


def export_qubo_artifacts(
    aggregated: AggregatedDataset,
    qubo_model: QuboModel,
    output_dir: str | Path,
) -> None:
    """
    Save the exact model artifacts requested by the research report.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sectors = list(aggregated.sector_names)
    prefix = f"{qubo_model.version}_B{qubo_model.hyperparameters.budget}_lambda{qubo_model.hyperparameters.penalty_lambda:.3f}"

    _write_matrix_csv(output_dir / f"{prefix}_Q_unscaled.csv", qubo_model.Q_unscaled, sectors, sectors)
    _write_matrix_csv(output_dir / f"{prefix}_Q_scaled.csv", qubo_model.Q_scaled, sectors, sectors)
    _write_matrix_csv(output_dir / f"{prefix}_J.csv", qubo_model.J, sectors, sectors)

    write_json(
        output_dir / f"{prefix}_model.json",
        {
            "version": qubo_model.version,
            "sector_names": sectors,
            "sector_to_asset_ids": aggregated.sector_to_asset_ids,
            "hyperparameters": {
                "budget": qubo_model.hyperparameters.budget,
                "risk_aversion": qubo_model.hyperparameters.risk_aversion,
                "penalty_lambda": qubo_model.hyperparameters.penalty_lambda,
                "capital_penalty": qubo_model.hyperparameters.capital_penalty,
                "illiquidity_penalty": qubo_model.hyperparameters.illiquidity_penalty,
            },
            "scale_alpha": qubo_model.scale_alpha,
            "h": qubo_model.h.tolist(),
            "offset": qubo_model.offset,
            "included_capital_penalty": qubo_model.included_capital_penalty,
            "included_liquidity_penalty": qubo_model.included_liquidity_penalty,
            "mu": aggregated.mu.tolist(),
            "sigma": aggregated.sigma.tolist(),
            "capital": aggregated.capital.tolist(),
            "liquidity": aggregated.liquidity.tolist(),
            "normalized_capital": aggregated.normalized_capital.tolist(),
            "normalized_illiquidity": aggregated.normalized_illiquidity.tolist(),
            "validation_summary": aggregated.validation_summary,
        },
    )


def _best_rows_by_source(classical_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in classical_rows:
        if record.get("source") == "exact_best_feasible":
            output[record["model_version"]] = record
    return output


def _best_noiseless_runs(summary_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in summary_records:
        if record.get("noise_model") != "none":
            continue
        version = record["model_version"]
        current = best.get(version)
        if current is None or float(record["optimal_bitstring_probability"]) > float(current["optimal_bitstring_probability"]):
            best[version] = record
    return best


def _classical_mds(distance_matrix: np.ndarray, dimensions: int = 2) -> np.ndarray:
    """
    Tiny NumPy-only classical MDS implementation.

    This lets us keep the neutral-atom layout figure without adding a heavy
    dependency just for one visualization.
    """

    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = distance_matrix.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distance_matrix**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order][:dimensions], 0.0)
    eigenvectors = eigenvectors[:, order[:dimensions]]
    return eigenvectors * np.sqrt(eigenvalues)


def plot_portfolio_tradeoff_frontier(classical_rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """
    Stakeholder-friendly summary of the four exact portfolio optima.
    """

    exact_rows = _best_rows_by_source(classical_rows)
    if not exact_rows:
        return

    plt.figure(figsize=(9, 6))
    x_values = []
    y_values = []
    sizes = []
    colors = []
    labels = []
    for version in MODEL_ORDER:
        row = exact_rows.get(version)
        if row is None:
            continue
        x_values.append(100.0 * float(row["volatility"]))
        y_values.append(100.0 * float(row["exp_return"]))
        sizes.append(280.0 + 90.0 * float(row["liquidity_avg"]))
        colors.append(float(row["capital_avg"]))
        labels.append(MODEL_LABELS[version])

    scatter = plt.scatter(
        x_values,
        y_values,
        s=sizes,
        c=colors,
        cmap="viridis_r",
        alpha=0.9,
        edgecolors="black",
        linewidths=1.0,
    )
    label_offsets = {
        "Base": (8, 8),
        "Capital": (-18, 10),
        "Liquidity": (8, 8),
        "Both": (8, -18),
    }
    for x_value, y_value, label in zip(x_values, y_values, labels):
        plt.annotate(label, (x_value, y_value), xytext=label_offsets.get(label, (8, 8)), textcoords="offset points")

    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Average capital charge")
    plt.xlabel("Volatility (%)")
    plt.ylabel("Expected return (%)")
    plt.title("Exact Optimal Portfolios Across Model Variants")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_scenario_return_distributions(
    aggregated: AggregatedDataset,
    classical_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Show how the exact optimal portfolios behave under all 1,200 scenarios.
    """

    exact_rows = _best_rows_by_source(classical_rows)
    if not exact_rows:
        return

    plt.figure(figsize=(10, 6))
    for version in MODEL_ORDER:
        row = exact_rows.get(version)
        if row is None:
            continue
        selection = bitstring_to_selection(row["bitstring"])
        scenario_returns = portfolio_scenario_returns(
            aggregated,
            selection,
            budget=int(row["selected_count"]),
        )
        hist, bins = np.histogram(100.0 * scenario_returns, bins=50, density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        plt.plot(
            centers,
            hist,
            linewidth=2.0,
            label=f"{MODEL_LABELS[version]} (CVaR95={100.0 * float(row['cvar95']):.2f}%)",
        )

    plt.xlabel("Scenario portfolio return (%)")
    plt.ylabel("Density")
    plt.title("Scenario Return Distributions for Exact Optimal Portfolios")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_energy_landscape(
    exact_rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    model_version: str = "base",
) -> None:
    """
    Explain why QAOA has a hard time by showing the exact classical landscape.
    """

    rows = [row for row in exact_rows if row["model_version"] == model_version]
    if not rows:
        return

    rows = sorted(rows, key=lambda row: float(row["objective_unscaled"]))
    ranks = np.arange(1, len(rows) + 1)
    values = np.array([float(row["objective_unscaled"]) for row in rows], dtype=float)
    feasible = np.array([bool(row["feasible"]) for row in rows], dtype=bool)

    plt.figure(figsize=(10, 5.5))
    plt.scatter(ranks[~feasible], values[~feasible], s=14, color="#b0b0b0", alpha=0.7, label="Infeasible")
    plt.scatter(ranks[feasible], values[feasible], s=18, color="#1f77b4", alpha=0.9, label="Feasible")
    plt.plot(ranks, values, color="#1f77b4", alpha=0.25, linewidth=1.0)

    top_gap = float(values[1] - values[0]) if len(values) > 1 else 0.0
    plt.annotate(
        f"Gap 1→2 = {top_gap:.6f}",
        xy=(1, values[0]),
        xytext=(20, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    top_label_offsets = {
        1: (8, -16),
        2: (16, -28),
        3: (24, -40),
    }
    for rank, row in enumerate(rows[:3], start=1):
        plt.annotate(
            row["bitstring"],
            xy=(rank, float(row["objective_unscaled"])),
            xytext=top_label_offsets.get(rank, (8, -16)),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("Exact rank across all 256 bitstrings")
    plt.ylabel("Objective (unscaled)")
    plt.title(f"Exact Energy Landscape: {MODEL_LABELS.get(model_version, model_version.title())} Model")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_optimizer_convergence(
    optimizer_records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    model_version: str = "base",
) -> None:
    """
    Loss-curve plot requested by judges and mentors.
    """

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    best_value_by_group: dict[tuple[int, int], float] = {}
    for record in optimizer_records:
        if record["model_version"] != model_version:
            continue
        key = (int(record["depth"]), int(record["seed"]))
        grouped.setdefault(key, []).append(record)
        best_value_by_group[key] = float(record["expected_energy"])

    if not grouped:
        return

    selected_groups: list[tuple[int, int]] = []
    for depth in sorted({depth for depth, _seed in grouped.keys()}):
        depth_keys = [key for key in grouped if key[0] == depth]
        selected_groups.append(min(depth_keys, key=lambda key: best_value_by_group[key]))

    fig, axes = plt.subplots(1, len(selected_groups), figsize=(5 * len(selected_groups), 4), squeeze=False)
    for axis, (depth, seed) in zip(axes[0], selected_groups):
        rows = grouped[(depth, seed)]
        trace_values = [float(row["value"]) for row in rows]
        axis.plot(range(1, len(trace_values) + 1), trace_values, color="#1f77b4", linewidth=1.8)
        axis.axhline(min(trace_values), color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8)
        axis.set_title(f"{MODEL_LABELS.get(model_version, model_version.title())}: p={depth}, seed={seed}")
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Expected energy")
        axis.grid(alpha=0.2)

    fig.suptitle("QAOA Optimization Convergence", fontsize=13)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_parameter_landscape(
    qubo_model: QuboModel,
    output_path: str | Path,
    *,
    grid_points: int = 45,
    optimizer_seed: int = 7,
) -> None:
    """
    p=1 parameter heatmap showing how expected energy varies with gamma and beta.
    """

    energies = precompute_ising_energies(qubo_model.h, qubo_model.J)
    num_qubits = len(qubo_model.h)
    plus_state = np.ones(2**num_qubits, dtype=complex) / np.sqrt(2**num_qubits)
    gamma_values = np.linspace(-np.pi, np.pi, grid_points)
    beta_values = np.linspace(0.0, 0.5 * np.pi, grid_points)
    surface = np.zeros((len(beta_values), len(gamma_values)), dtype=float)

    for gamma_index, gamma in enumerate(gamma_values):
        phased_state = plus_state * np.exp(-1j * gamma * energies)
        for beta_index, beta in enumerate(beta_values):
            mixed_state = apply_rx_layer(phased_state, beta, num_qubits)
            surface[beta_index, gamma_index] = float(np.dot(np.abs(mixed_state) ** 2, energies))

    qaoa_result = optimize_qaoa_angles(
        qubo_model.h,
        qubo_model.J,
        depth=1,
        seed=optimizer_seed,
    )
    gamma_star = float(qaoa_result.gammas[0])
    beta_star = float(qaoa_result.betas[0])

    plt.figure(figsize=(8.5, 6))
    image = plt.imshow(
        surface,
        origin="lower",
        aspect="auto",
        extent=[gamma_values[0], gamma_values[-1], beta_values[0], beta_values[-1]],
        cmap="magma_r",
    )
    plt.scatter([gamma_star], [beta_star], color="cyan", edgecolors="black", s=90, label="Optimized point")
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$\beta$")
    plt.title(f"QAOA Parameter Landscape (p=1, {MODEL_LABELS.get(qubo_model.version, qubo_model.version.title())})")
    plt.colorbar(image, label="Expected Ising energy")
    plt.legend(loc="upper right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_atom_layout(aggregated: AggregatedDataset, output_path: str | Path) -> None:
    """
    Neutral-atom-inspired layout using correlation distances and classical MDS.
    """

    sigma = np.asarray(aggregated.sigma, dtype=float)
    vol = np.sqrt(np.maximum(np.diag(sigma), 1e-15))
    corr = sigma / np.outer(vol, vol)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    distance = 1.0 - np.abs(corr)
    np.fill_diagonal(distance, 0.0)
    coords = _classical_mds(distance)

    plt.figure(figsize=(7, 6))
    for i in range(len(aggregated.sector_names)):
        for j in range(i + 1, len(aggregated.sector_names)):
            strength = abs(corr[i, j])
            plt.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color="#4c72b0",
                alpha=0.15 + 0.55 * strength,
                linewidth=0.8 + 2.2 * strength,
            )

    plt.scatter(coords[:, 0], coords[:, 1], s=260, color="#dd8452", edgecolors="black", linewidths=1.0, zorder=3)
    for (x_coord, y_coord), label in zip(coords, aggregated.sector_names):
        plt.annotate(label, (x_coord, y_coord), xytext=(8, 8), textcoords="offset points")

    plt.title("Neutral-Atom Layout from Sector Correlations")
    plt.xlabel("MDS axis 1")
    plt.ylabel("MDS axis 2")
    plt.grid(alpha=0.15)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_success_vs_noise(summary_records: list[dict[str, Any]], output_path: str | Path) -> None:
    """
    Clean noise chart: one line per noise model, averaged across the full sweep.
    """

    ideal_rows = [record for record in summary_records if record.get("noise_model") == "none"]
    noisy_rows = [record for record in summary_records if record.get("noise_model") not in (None, "none")]
    if not noisy_rows:
        return

    plt.figure(figsize=(9, 5.5))
    if ideal_rows:
        plt.axhline(
            mean(float(row["optimal_bitstring_probability"]) for row in ideal_rows),
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Noiseless mean",
        )

    grouped: dict[str, dict[float, list[float]]] = {}
    for row in noisy_rows:
        model = row["noise_model"]
        grouped.setdefault(model, {})
        grouped[model].setdefault(float(row["noise_strength"]), []).append(float(row["optimal_bitstring_probability"]))

    for noise_model, strengths in sorted(grouped.items()):
        x_values = sorted(strengths.keys())
        y_values = [mean(strengths[x]) for x in x_values]
        plt.plot(x_values, y_values, marker="o", linewidth=2.0, label=noise_model.replace("_", " ").title())

    plt.xlabel("Noise strength")
    plt.ylabel("Optimal bitstring probability")
    plt.title("Quantum Success Probability Under Noise")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_depth_vs_performance(summary_records: list[dict[str, Any]], output_path: str | Path) -> None:
    """
    Connectivity comparison with one point per strategy rather than a cluttered scatter.
    """

    rows = [record for record in summary_records if record.get("noise_model") == "none"]
    if not rows:
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["connectivity_mode"], []).append(row)

    plt.figure(figsize=(8, 5.5))
    for connectivity_mode in ("naive", "auto", "hand", "pruned"):
        mode_rows = grouped.get(connectivity_mode, [])
        if not mode_rows:
            continue
        mean_depth = mean(float(row["circuit_depth"]) for row in mode_rows)
        mean_prob = mean(float(row["optimal_bitstring_probability"]) for row in mode_rows)
        plt.scatter([mean_depth], [mean_prob], s=180, edgecolors="black", linewidths=0.8, label=connectivity_mode)
        annotation_offsets = {
            "naive": (8, 8),
            "auto": (8, 8),
            "hand": (8, -16),
            "pruned": (8, 8),
        }
        plt.annotate(
            connectivity_mode,
            (mean_depth, mean_prob),
            xytext=annotation_offsets.get(connectivity_mode, (8, 8)),
            textcoords="offset points",
        )

    plt.xlabel("Average circuit depth (noiseless runs)")
    plt.ylabel("Average optimal bitstring probability")
    plt.title("Connectivity Tradeoff: Depth vs Solution Quality")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def build_comparison_rows(
    classical_rows: list[dict[str, Any]],
    summary_records: list[dict[str, Any]],
    sample_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build a stakeholder-friendly comparison table with the best classical and
    best sampled quantum results.
    """

    comparison_rows = list(classical_rows)

    best_quantum_by_version: dict[str, dict[str, Any]] = {}
    for record in summary_records:
        if record.get("noise_model") != "none":
            continue
        version = record["model_version"]
        current = best_quantum_by_version.get(version)
        current_probability = float(current["optimal_bitstring_probability"]) if current is not None else -1.0
        candidate_probability = float(record["optimal_bitstring_probability"])
        if current is None or candidate_probability > current_probability:
            best_quantum_by_version[version] = record

    sample_lookup: dict[tuple[str, str], dict[str, Any]] = {
        (record["run_id"], record["bitstring"]): record for record in sample_records
    }

    for version, summary in best_quantum_by_version.items():
        bitstring = summary.get("best_feasible_sampled_bitstring")
        if bitstring is None:
            continue
        sample_row = sample_lookup.get((summary["run_id"], bitstring))
        if sample_row is None:
            continue
        comparison_rows.append(
            {
                "source": "quantum_best_sampled",
                "model_version": version,
                "bitstring": sample_row["bitstring"],
                "objective_unscaled": sample_row["objective_unscaled"],
                "exp_return": sample_row["exp_return"],
                "variance": sample_row["variance"],
                "capital_avg": sample_row["capital_avg"],
                "liquidity_avg": sample_row["liquidity_avg"],
                "worst_scenario_return": sample_row["worst_scenario_return"],
                "p": summary["p"],
                "connectivity_mode": summary["connectivity_mode"],
                "noise_model": summary["noise_model"],
                "noise_strength": summary["noise_strength"],
                "optimal_bitstring_probability": summary["optimal_bitstring_probability"],
                "feasible_probability_mass": summary["feasible_probability_mass"],
            }
        )

    return comparison_rows


def plot_quantum_probability_summary(summary_records: list[dict[str, Any]], output_path: str | Path) -> None:
    """
    Honest probability summary that makes the QAOA strength and weakness explicit.
    """

    best_runs = _best_noiseless_runs(summary_records)
    if not best_runs:
        return

    labels = []
    absolute_opt = []
    feasible_mass = []
    conditional_opt = []
    uniform_baseline = []
    for version in MODEL_ORDER:
        record = best_runs.get(version)
        if record is None:
            continue
        labels.append(MODEL_LABELS[version])
        opt_prob = float(record["optimal_bitstring_probability"])
        feas_prob = float(record["feasible_probability_mass"])
        absolute_opt.append(opt_prob)
        feasible_mass.append(feas_prob)
        conditional_opt.append(opt_prob / feas_prob if feas_prob > 0 else 0.0)
        feasible_count = math.comb(8, int(record["B"]))
        uniform_baseline.append(1.0 / feasible_count)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(labels))
    width = 0.34

    axes[0].bar(x - width / 2.0, absolute_opt, width=width, label="P(optimal)")
    axes[0].bar(x + width / 2.0, feasible_mass, width=width, label="P(feasible)")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Absolute Quantum Success Metrics")
    axes[0].legend()

    axes[1].bar(x, conditional_opt, width=0.52, label="P(optimal | feasible)", color="#4c72b0")
    axes[1].plot(x, uniform_baseline, color="black", linestyle="--", marker="o", label="Uniform over feasible set")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Conditional probability")
    axes[1].set_title("How Sharply QAOA Ranks Feasible Portfolios")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_classical_vs_quantum_objectives(comparison_rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """
    Replace the old unreadable bar chart with a compact summary table figure.
    """

    exact_rows = {row["model_version"]: row for row in comparison_rows if row.get("source") == "exact_best_feasible"}
    quantum_rows = {row["model_version"]: row for row in comparison_rows if row.get("source") == "quantum_best_sampled"}
    greedy_rows = {
        row["model_version"]: row
        for row in comparison_rows
        if row.get("source") == "greedy" or row.get("baseline_name") == "greedy"
    }
    continuous_rows = {
        row["model_version"]: row
        for row in comparison_rows
        if row.get("source") == "continuous_top_budget_proxy" or row.get("baseline_name") == "continuous_top_budget_proxy"
    }
    if not exact_rows:
        return

    columns = [
        "Model",
        "Exact bitstring",
        "Greedy match",
        "Continuous gap",
        "Quantum bitstring",
        "Quantum match",
        "P(opt)",
        "P(feasible)",
    ]
    table_rows: list[list[str]] = []

    for version in MODEL_ORDER:
        exact = exact_rows.get(version)
        if exact is None:
            continue
        quantum = quantum_rows.get(version, {})
        greedy = greedy_rows.get(version, {})
        continuous = continuous_rows.get(version, {})

        exact_objective = float(exact["objective_unscaled"])
        continuous_objective = float(continuous["objective_unscaled"]) if continuous else float("nan")
        table_rows.append(
            [
                MODEL_LABELS[version],
                str(exact["bitstring"]),
                "Yes" if greedy and greedy.get("bitstring") == exact["bitstring"] else "No",
                f"{continuous_objective - exact_objective:+.4f}" if continuous else "n/a",
                str(quantum.get("bitstring", "n/a")),
                "Yes" if quantum and quantum.get("bitstring") == exact["bitstring"] else "No",
                f"{100.0 * float(quantum.get('optimal_bitstring_probability', 0.0)):.2f}%",
                f"{100.0 * float(quantum.get('feasible_probability_mass', 0.0)):.2f}%",
            ]
        )

    fig, axis = plt.subplots(figsize=(14, 3.6 + 0.42 * len(table_rows)))
    axis.axis("off")
    table = axis.table(cellText=table_rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.55)

    for (row_index, column_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#1f77b4")
            cell.set_text_props(color="white", weight="bold")
        elif row_index % 2 == 1:
            cell.set_facecolor("#eef5fb")

    axis.set_title("Classical vs Quantum Summary", fontsize=14, pad=18)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
