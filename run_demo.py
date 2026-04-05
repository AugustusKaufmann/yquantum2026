from __future__ import annotations

"""
Legacy Cirq-first entrypoint for the full research pipeline.

For the hackathon-required Bloqade execution path, use:
    python run_bloqade_demo.py

Typical usage in PyCharm:
    python run_demo.py

With the real workbook:
    python run_demo.py --workbook /path/to/insurance_dataset.xlsx
"""

import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

from bloqade_experiments import run_quantum_experiments
from classical_baselines import (
    continuous_markowitz_baseline,
    ensure_feasible_penalty,
    greedy_selection_baseline,
)
from make_figures import (
    build_comparison_rows,
    export_qubo_artifacts,
    plot_atom_layout,
    plot_classical_vs_quantum_objectives,
    plot_depth_vs_performance,
    plot_energy_landscape,
    plot_optimizer_convergence,
    plot_parameter_landscape,
    plot_portfolio_tradeoff_frontier,
    plot_quantum_probability_summary,
    plot_scenario_return_distributions,
    plot_success_vs_noise,
    write_csv,
    write_json,
)
from portfolio_model import (
    aggregate_to_sectors,
    build_qubo_model,
    generate_synthetic_dataset,
    load_workbook_dataset,
)
from project_config import ExecutionConfig, ModelHyperparameters, RunConfig, SweepConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 8-qubit portfolio QUBO demo pipeline.")
    parser.add_argument("--workbook", type=str, default=None, help="Path to the hackathon Excel workbook.")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Folder for results and figures.")
    parser.add_argument("--assets-sheet", type=str, default=None, help="Explicit assets sheet name.")
    parser.add_argument("--scenarios-sheet", type=str, default=None, help="Explicit scenarios sheet name.")
    parser.add_argument("--covariance-sheet", type=str, default=None, help="Explicit covariance sheet name.")
    parser.add_argument("--budget", type=int, default=4, help="Number of sectors to select.")
    parser.add_argument("--risk-aversion", type=float, default=0.5, help="Markowitz risk aversion q.")
    parser.add_argument("--lambda-penalty", type=float, default=8.0, help="Budget-constraint penalty lambda.")
    parser.add_argument("--capital-penalty", type=float, default=0.25, help="Capital penalty gamma.")
    parser.add_argument("--illiquidity-penalty", type=float, default=0.25, help="Illiquidity penalty eta.")
    parser.add_argument("--shots", type=int, default=2048, help="Measurement shots per experiment.")
    parser.add_argument("--synthetic-seed", type=int, default=123, help="Seed for synthetic demo data.")
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Run the full hyperparameter grid from the report. This can take a while.",
    )
    return parser.parse_args()


def build_run_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        workbook_path=Path(args.workbook) if args.workbook else None,
        output_dir=Path(args.output_dir),
        assets_sheet_name=args.assets_sheet,
        scenarios_sheet_name=args.scenarios_sheet,
        covariance_sheet_name=args.covariance_sheet,
        hyperparameters=ModelHyperparameters(
            budget=args.budget,
            risk_aversion=args.risk_aversion,
            penalty_lambda=args.lambda_penalty,
            capital_penalty=args.capital_penalty,
            illiquidity_penalty=args.illiquidity_penalty,
        ),
        sweep=SweepConfig(),
        execution=ExecutionConfig(
            shots=args.shots,
            synthetic_seed=args.synthetic_seed,
        ),
    )


def load_dataset(run_config: RunConfig):
    """
    Load the user workbook if available, otherwise generate a synthetic dataset
    so the project is runnable out of the box.
    """

    workbook_path = run_config.workbook_path
    if workbook_path is not None and workbook_path.exists():
        print(f"Loading workbook: {workbook_path}")
        return load_workbook_dataset(
            workbook_path,
            assets_sheet_name=run_config.assets_sheet_name,
            scenarios_sheet_name=run_config.scenarios_sheet_name,
            covariance_sheet_name=run_config.covariance_sheet_name,
        )

    if not run_config.execution.use_synthetic_data_if_missing:
        raise FileNotFoundError("Workbook was not found and synthetic fallback is disabled.")

    print("Workbook not supplied or not found. Using synthetic development dataset instead.")
    return generate_synthetic_dataset(
        n_assets=run_config.execution.synthetic_assets,
        n_scenarios=run_config.execution.synthetic_scenarios,
        seed=run_config.execution.synthetic_seed,
    )


def build_hyperparameter_list(run_config: RunConfig, full_grid: bool) -> list[ModelHyperparameters]:
    """
    Either run a single presentation-friendly configuration or build the report
    sweep grid.
    """

    if not full_grid:
        return [run_config.hyperparameters]

    hyperparameters: list[ModelHyperparameters] = []
    for budget, risk_aversion, lambda_penalty, capital_penalty, illiquidity_penalty in product(
        run_config.sweep.budgets,
        run_config.sweep.risk_aversion_values,
        run_config.sweep.lambda_values,
        run_config.sweep.capital_penalty_values,
        run_config.sweep.illiquidity_penalty_values,
    ):
        hyperparameters.append(
            ModelHyperparameters(
                budget=budget,
                risk_aversion=risk_aversion,
                penalty_lambda=lambda_penalty,
                capital_penalty=capital_penalty,
                illiquidity_penalty=illiquidity_penalty,
            )
        )
    return hyperparameters


def choose_representative_run_ids(summary_records: list[dict[str, Any]]) -> list[str]:
    """
    Pick a small number of interesting runs for plotting.
    """

    representative: list[str] = []
    for model_version in sorted({record["model_version"] for record in summary_records}):
        noiseless_runs = [
            record
            for record in summary_records
            if record["model_version"] == model_version and record["noise_model"] == "none"
        ]
        if not noiseless_runs:
            continue
        best = max(noiseless_runs, key=lambda record: record["optimal_bitstring_probability"])
        representative.append(best["run_id"])
    return representative


def main() -> None:
    args = parse_args()
    run_config = build_run_config(args)
    folders = run_config.ensure_output_dirs()

    raw_dataset = load_dataset(run_config)
    aggregated = aggregate_to_sectors(raw_dataset)

    all_classical_rows: list[dict[str, Any]] = []
    all_exact_enumeration_rows: list[dict[str, Any]] = []
    all_sample_records: list[dict[str, Any]] = []
    all_summary_records: list[dict[str, Any]] = []
    all_optimizer_records: list[dict[str, Any]] = []
    model_run_summaries: list[dict[str, Any]] = []
    representative_models: dict[str, Any] = {}

    for hyperparameters in build_hyperparameter_list(run_config, args.full_grid):
        for version in run_config.execution.include_model_versions:
            print(
                f"Building model version={version} "
                f"(B={hyperparameters.budget}, q={hyperparameters.risk_aversion}, "
                f"lambda={hyperparameters.penalty_lambda}, gamma={hyperparameters.capital_penalty}, "
                f"eta={hyperparameters.illiquidity_penalty})"
            )

            qubo_model = build_qubo_model(aggregated, hyperparameters, version=version)
            qubo_model, exact_summary = ensure_feasible_penalty(
                aggregated,
                qubo_model,
                alpha=run_config.execution.var_alpha,
            )
            representative_models.setdefault(version, qubo_model)
            all_exact_enumeration_rows.extend(
                [
                    {
                        "source": "exact_enumeration",
                        "model_version": version,
                        "q": qubo_model.hyperparameters.risk_aversion,
                        "lambda": qubo_model.hyperparameters.penalty_lambda,
                        "gamma": qubo_model.hyperparameters.capital_penalty,
                        "eta": qubo_model.hyperparameters.illiquidity_penalty,
                        "B": qubo_model.hyperparameters.budget,
                        **record,
                    }
                    for record in exact_summary.all_records
                ]
            )

            export_qubo_artifacts(aggregated, qubo_model, folders["artifacts"])

            greedy_result = greedy_selection_baseline(
                aggregated,
                qubo_model,
                alpha=run_config.execution.var_alpha,
            )
            continuous_result = continuous_markowitz_baseline(
                aggregated,
                qubo_model,
                alpha=run_config.execution.var_alpha,
            )

            classical_rows = [
                {
                    "source": "exact_best_overall",
                    "model_version": version,
                    **exact_summary.best_overall,
                },
                {
                    "source": "exact_best_feasible",
                    "model_version": version,
                    **exact_summary.best_feasible,
                },
                {
                    "source": greedy_result.get("baseline_name", "greedy"),
                    "model_version": version,
                    **greedy_result,
                },
                {
                    "source": continuous_result.get("baseline_name", "continuous_top_budget_proxy"),
                    "model_version": version,
                    **continuous_result,
                },
            ]
            all_classical_rows.extend(classical_rows)

            sample_records, summary_records, optimizer_records = run_quantum_experiments(
                aggregated,
                qubo_model,
                optimum_bitstring=exact_summary.best_feasible["bitstring"],
                depths=run_config.sweep.qaoa_depths,
                seeds=run_config.sweep.seeds,
                shots=run_config.execution.shots,
                connectivity_modes=run_config.execution.include_connectivity_modes,
                noise_models=run_config.execution.include_noise_models,
                noise_strengths=run_config.sweep.noise_strengths,
                readout_error_values=run_config.sweep.readout_error_values,
                qaoa_optimizer_restarts=run_config.execution.qaoa_optimizer_restarts,
                qaoa_coordinate_descent_rounds=run_config.execution.qaoa_coordinate_descent_rounds,
                qaoa_initial_step=run_config.execution.qaoa_initial_step,
                var_alpha=run_config.execution.var_alpha,
                kernel_output_dir=folders["kernels"],
            )
            all_sample_records.extend(sample_records)
            all_summary_records.extend(summary_records)
            all_optimizer_records.extend(optimizer_records)

            model_run_summaries.append(
                {
                    "model_version": version,
                    "hyperparameters": asdict(qubo_model.hyperparameters),
                    "best_exact_feasible_bitstring": exact_summary.best_feasible["bitstring"],
                    "best_exact_feasible_objective": exact_summary.best_feasible["objective_unscaled"],
                    "workbook_summary": raw_dataset.workbook_summary,
                }
            )

    write_json(folders["results"] / "run_summary.json", model_run_summaries)
    write_json(folders["results"] / "aggregated_validation.json", aggregated.validation_summary)
    write_csv(folders["results"] / "classical_results.csv", all_classical_rows)
    write_csv(folders["results"] / "exact_enumeration_results.csv", all_exact_enumeration_rows)
    write_csv(folders["results"] / "quantum_sample_results.csv", all_sample_records)
    write_csv(folders["results"] / "quantum_summary_results.csv", all_summary_records)
    write_csv(folders["results"] / "optimizer_trace.csv", all_optimizer_records)

    comparison_rows = build_comparison_rows(
        all_classical_rows,
        all_summary_records,
        all_sample_records,
    )
    write_csv(folders["results"] / "comparison_table.csv", comparison_rows)

    for old_figure in folders["figures"].glob("*.png"):
        old_figure.unlink()

    plot_portfolio_tradeoff_frontier(
        all_classical_rows,
        folders["figures"] / "portfolio_tradeoff_frontier.png",
    )
    plot_scenario_return_distributions(
        aggregated,
        all_classical_rows,
        folders["figures"] / "scenario_return_distributions.png",
    )
    plot_energy_landscape(
        all_exact_enumeration_rows,
        folders["figures"] / "energy_landscape_base.png",
        model_version="base",
    )
    plot_atom_layout(
        aggregated,
        folders["figures"] / "atom_layout.png",
    )
    plot_success_vs_noise(all_summary_records, folders["figures"] / "success_vs_noise.png")
    plot_depth_vs_performance(all_summary_records, folders["figures"] / "connectivity_tradeoff.png")
    plot_quantum_probability_summary(
        all_summary_records,
        folders["figures"] / "quantum_probability_summary.png",
    )
    plot_classical_vs_quantum_objectives(comparison_rows, folders["figures"] / "classical_vs_quantum.png")

    if not args.full_grid and "base" in representative_models:
        plot_optimizer_convergence(
            all_optimizer_records,
            folders["figures"] / "qaoa_convergence_base.png",
            model_version="base",
        )
        plot_parameter_landscape(
            representative_models["base"],
            folders["figures"] / "qaoa_parameter_landscape_base_p1.png",
        )

    print(f"Done. Results written to: {folders['root'].resolve()}")


if __name__ == "__main__":
    main()
