from __future__ import annotations

"""
Quantum experiment helpers.

This file serves two purposes:
1. Provide a reliable local experiment path even if Bloqade/Cirq are missing.
2. Automatically upgrade to real Cirq/Bloqade circuit export, parallelization,
   and noise transforms when those packages are installed.
"""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from portfolio_model import AggregatedDataset, QuboModel
from qaoa_simulator import QAOAOptimizationResult, optimize_qaoa_angles
from results_metrics import (
    bitstring_to_selection,
    evaluate_selection,
    sample_counts_from_probabilities,
    summarize_probability_distribution,
)


@dataclass(slots=True)
class CircuitBuildResult:
    circuit: Any | None
    connectivity_mode: str
    circuit_depth: int
    two_qubit_gate_count: int
    edge_count: int
    edges: list[tuple[int, int, float]]
    notes: list[str]


def _try_import(module_name: str):
    try:
        return import_module(module_name)
    except Exception:
        return None


def _get_cirq():
    return _try_import("cirq")


def _get_bloqade_parallelize():
    module = _try_import("bloqade.cirq_utils.parallelize")
    if module is None:
        return None
    return getattr(module, "parallelize", None)


def _get_bloqade_noise_module():
    return _try_import("bloqade.cirq_utils.noise")


def _get_bloqade_load_circuit():
    module = _try_import("bloqade.squin.cirq")
    if module is None:
        return None
    return getattr(module, "load_circuit", None)


def interaction_edges_from_ising(
    J: np.ndarray,
    *,
    top_k: int | None = None,
    min_abs_weight: float = 0.0,
) -> list[tuple[int, int, float]]:
    """
    Convert the symmetric Ising coupling matrix into an explicit weighted edge
    list. This is what we use for scheduling and connectivity experiments.
    """

    edges: list[tuple[int, int, float]] = []
    n = J.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            weight = float(J[i, j])
            if abs(weight) > min_abs_weight:
                edges.append((i, j, weight))

    edges.sort(key=lambda edge: abs(edge[2]), reverse=True)
    if top_k is not None:
        edges = edges[:top_k]
    return edges


def greedy_edge_coloring(edges: list[tuple[int, int, float]]) -> list[list[tuple[int, int, float]]]:
    """
    Partition edges into disjoint matchings.

    For K8 we expect 7 color classes, which matches the presentation idea in
    the report.
    """

    colors: list[list[tuple[int, int, float]]] = []

    for edge in edges:
        u, v, _ = edge
        placed = False
        for color_class in colors:
            used_vertices = {vertex for colored_edge in color_class for vertex in colored_edge[:2]}
            if u not in used_vertices and v not in used_vertices:
                color_class.append(edge)
                placed = True
                break
        if not placed:
            colors.append([edge])

    return colors


def _append_disjoint_zz_group(cirq, circuit, qubits, weighted_edges, gamma: float) -> None:
    """
    Append one parallelizable ZZ group using a CNOT/RZ/CNOT decomposition.

    We want to implement:
        exp(-i * gamma * weight * Z_i Z_j)

    In Cirq, `rz(phi)` applies exp(-i * phi * Z / 2), so the standard
    decomposition becomes:
        CNOT(i, j) -> Rz(2 * gamma * weight) on j -> CNOT(i, j)

    The earlier implementation used H/CZ/RZ/CZ/H, which does *not* realize the
    desired ZZ phase and was the root cause of the mismatch between the NumPy
    simulator and the Cirq ideal-circuit path.
    """

    if not weighted_edges:
        return

    circuit.append(cirq.Moment(cirq.CNOT(qubits[i], qubits[j]) for i, j, _ in weighted_edges))
    circuit.append(
        cirq.Moment(
            cirq.rz(2.0 * gamma * weight)(qubits[max(i, j)])
            for i, j, weight in weighted_edges
        )
    )
    circuit.append(cirq.Moment(cirq.CNOT(qubits[i], qubits[j]) for i, j, _ in weighted_edges))


def build_cirq_qaoa_circuit(
    qubo_model: QuboModel,
    qaoa_result: QAOAOptimizationResult,
    *,
    connectivity_mode: str,
    pruned_top_k: int | None = None,
) -> CircuitBuildResult:
    """
    Build a Cirq circuit for the optimized QAOA angles if Cirq is available.
    """

    cirq = _get_cirq()
    if cirq is None:
        edges = interaction_edges_from_ising(
            qubo_model.J,
            top_k=pruned_top_k if connectivity_mode == "pruned" else None,
        )
        color_classes = greedy_edge_coloring(edges)
        zz_moments = len(edges) * 3 if connectivity_mode == "naive" else len(color_classes) * 3
        circuit_depth = 1 + qaoa_result.depth * (1 + zz_moments + 1)
        return CircuitBuildResult(
            circuit=None,
            connectivity_mode=connectivity_mode,
            circuit_depth=circuit_depth,
            two_qubit_gate_count=2 * len(edges) * qaoa_result.depth,
            edge_count=len(edges),
            edges=edges,
            notes=["Cirq not installed; using analytic depth estimate only."],
        )

    qubits = cirq.LineQubit.range(len(qubo_model.h))
    edges = interaction_edges_from_ising(
        qubo_model.J,
        top_k=pruned_top_k if connectivity_mode == "pruned" else None,
    )
    color_classes = greedy_edge_coloring(edges)

    circuit = cirq.Circuit()
    notes: list[str] = []

    # QAOA starts in |+>^n, prepared by Hadamards on all qubits.
    circuit.append(cirq.Moment(cirq.H(qubit) for qubit in qubits))

    for gamma, beta in zip(qaoa_result.gammas, qaoa_result.betas):
        # Single-qubit field terms in the Ising Hamiltonian.
        circuit.append(cirq.Moment(cirq.rz(2.0 * gamma * field)(qubit) for qubit, field in zip(qubits, qubo_model.h)))

        if connectivity_mode == "naive":
            zz_groups = [[edge] for edge in edges]
        elif connectivity_mode == "pruned":
            zz_groups = color_classes
        else:
            zz_groups = color_classes

        for group in zz_groups:
            _append_disjoint_zz_group(cirq, circuit, qubits, group, gamma)

        # Standard X-mixer layer.
        circuit.append(cirq.Moment(cirq.rx(2.0 * beta)(qubit) for qubit in qubits))

    if connectivity_mode == "auto":
        parallelize = _get_bloqade_parallelize()
        if parallelize is not None:
            circuit = parallelize(circuit)
            notes.append("Applied Bloqade auto-parallelization.")
        else:
            notes.append("Bloqade auto-parallelizer not installed; using hand-colored schedule as fallback.")

    return CircuitBuildResult(
        circuit=circuit,
        connectivity_mode=connectivity_mode,
        circuit_depth=len(circuit),
        two_qubit_gate_count=2 * len(edges) * qaoa_result.depth,
        edge_count=len(edges),
        edges=edges,
        notes=notes,
    )


def export_bloqade_kernel(circuit, output_path: str | Path) -> bool:
    """
    Try to convert the Cirq circuit into a Bloqade SQUIN kernel and save a text
    representation for inspection.
    """

    load_circuit = _get_bloqade_load_circuit()
    if load_circuit is None or circuit is None:
        return False

    try:
        kernel = load_circuit(circuit)
    except Exception:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(repr(kernel), encoding="utf-8")
    return True


def _probabilities_from_cirq_circuit(circuit, *, noisy: bool) -> np.ndarray:
    """
    Simulate either a unitary or noisy Cirq circuit.
    """

    cirq = _get_cirq()
    if cirq is None:
        raise RuntimeError("Cirq is required for direct circuit simulation.")

    if noisy:
        simulator = cirq.DensityMatrixSimulator()
        result = simulator.simulate(circuit)
        density_matrix = result.final_density_matrix
        probabilities = np.real(np.diag(density_matrix))
    else:
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        statevector = result.final_state_vector
        probabilities = np.abs(statevector) ** 2

    probabilities = np.asarray(probabilities, dtype=float)
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise RuntimeError("Circuit simulation produced zero total probability.")
    probabilities = probabilities / total

    # Cirq orders amplitudes in big-endian qubit order, while the rest of this
    # project uses little-endian integer-to-bitstring mapping:
    #   bit_i = (state_index >> i) & 1
    #
    # Reordering here keeps every downstream metric and bitstring label aligned
    # with the NumPy simulator, the brute-force enumeration, and the exported
    # results tables.
    num_qubits = int(np.log2(len(probabilities)))
    reordered = np.zeros_like(probabilities)
    for little_endian_index in range(len(probabilities)):
        big_endian_index = 0
        for bit in range(num_qubits):
            big_endian_index = (big_endian_index << 1) | ((little_endian_index >> bit) & 1)
        reordered[little_endian_index] = probabilities[big_endian_index]

    return reordered / np.sum(reordered)


def _apply_uniform_mixing_noise(probabilities: np.ndarray, strength: float) -> np.ndarray:
    """
    Mix the distribution with the uniform distribution as a lightweight
    depolarizing-style approximation.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    strength = float(np.clip(strength, 0.0, 0.999))
    uniform = np.full_like(probabilities, 1.0 / len(probabilities))
    mixed = (1.0 - strength) * probabilities + strength * uniform
    return mixed / np.sum(mixed)


def _apply_readout_error(probabilities: np.ndarray, error_rate: float) -> np.ndarray:
    """
    Apply independent bit-flip measurement noise to a full probability vector.
    """

    error_rate = float(np.clip(error_rate, 0.0, 0.5))
    if error_rate <= 0.0:
        return np.asarray(probabilities, dtype=float)

    probabilities = np.asarray(probabilities, dtype=float)
    n = int(np.log2(len(probabilities)))
    corrected = np.zeros_like(probabilities, dtype=float)

    for source in range(len(probabilities)):
        source_probability = probabilities[source]
        if source_probability <= 0.0:
            continue
        for target in range(len(probabilities)):
            hamming_distance = bin(source ^ target).count("1")
            transition_probability = ((1.0 - error_rate) ** (n - hamming_distance)) * (error_rate**hamming_distance)
            corrected[target] += source_probability * transition_probability

    return corrected / np.sum(corrected)


def approximate_noisy_probabilities(
    ideal_probabilities: np.ndarray,
    *,
    circuit_depth: int,
    two_qubit_gate_count: int,
    noise_model: str,
    noise_strength: float,
    readout_error: float,
) -> np.ndarray:
    """
    Fallback noise model used when Bloqade's native noise stack is unavailable.

    The idea is not to pretend this is hardware-accurate. It is simply a
    practical way to stress the algorithm and demonstrate the type of analysis
    the report asks for.
    """

    base_scale = {
        "none": 0.0,
        "one_zone": 1.0,
        "two_zone": 0.75,
        "depolarizing": 1.25,
    }.get(noise_model, 1.0)

    depth_component = 0.0015 * circuit_depth
    entangling_component = 0.0020 * two_qubit_gate_count
    effective_strength = min(0.95, base_scale * noise_strength * (depth_component + entangling_component))

    noisy = _apply_uniform_mixing_noise(ideal_probabilities, effective_strength)
    noisy = _apply_readout_error(noisy, readout_error)
    return noisy / np.sum(noisy)


def simulate_with_noise_if_available(
    circuit_result: CircuitBuildResult,
    ideal_probabilities: np.ndarray,
    *,
    ideal_backend: str,
    noise_model: str,
    noise_strength: float,
    readout_error: float,
) -> tuple[np.ndarray, str, list[str]]:
    """
    Use the best simulation path available for a given noise setting.
    """

    notes = list(circuit_result.notes)

    if noise_model == "none":
        return ideal_probabilities, ideal_backend, notes

    # Try Bloqade's hardware-style noise transform when we can.
    noise_module = _get_bloqade_noise_module()
    cirq = _get_cirq()
    if circuit_result.circuit is not None and cirq is not None and noise_module is not None and noise_strength == 1.0:
        try:
            if noise_model == "one_zone":
                model = noise_module.GeminiOneZoneNoiseModel()
            elif noise_model == "two_zone":
                model = noise_module.GeminiTwoZoneNoiseModel()
            else:
                model = None

            if model is not None:
                noisy_circuit = noise_module.transform_circuit(circuit_result.circuit, model=model)
                if readout_error > 0.0:
                    noisy_circuit = noisy_circuit.copy()
                    noisy_circuit.append(
                        cirq.Moment(cirq.bit_flip(readout_error).on(q) for q in sorted(noisy_circuit.all_qubits()))
                    )
                probabilities = _probabilities_from_cirq_circuit(noisy_circuit, noisy=True)
                notes.append("Used Bloqade/Cirq noise transform.")
                return probabilities, f"bloqade_{noise_model}", notes
        except Exception as exc:
            notes.append(f"Bloqade noise simulation failed; using approximate noise instead: {exc}")

    approximate = approximate_noisy_probabilities(
        ideal_probabilities,
        circuit_depth=circuit_result.circuit_depth,
        two_qubit_gate_count=circuit_result.two_qubit_gate_count,
        noise_model=noise_model,
        noise_strength=noise_strength,
        readout_error=readout_error,
    )
    notes.append("Used approximate noise fallback.")
    return approximate, "approximate_noise", notes


def run_quantum_experiments(
    aggregated: AggregatedDataset,
    qubo_model: QuboModel,
    *,
    optimum_bitstring: str,
    depths: tuple[int, ...],
    seeds: tuple[int, ...],
    shots: int,
    connectivity_modes: tuple[str, ...],
    noise_models: tuple[str, ...],
    noise_strengths: tuple[float, ...],
    readout_error_values: tuple[float, ...],
    qaoa_optimizer_restarts: int,
    qaoa_coordinate_descent_rounds: int,
    qaoa_initial_step: float,
    var_alpha: float,
    kernel_output_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Run ideal and noisy QAOA studies for one QUBO model.

    Returns three datasets:
    - sample_records: one row per sampled bitstring per run
    - summary_records: one row per experiment configuration
    - optimizer_records: one row per optimization trace point
    """

    sample_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    optimizer_records: list[dict[str, Any]] = []
    model_version_index = {name: index for index, name in enumerate(("base", "capital", "liquidity", "both"))}
    connectivity_index = {name: index for index, name in enumerate(("naive", "auto", "hand", "pruned"))}

    for depth in depths:
        for seed in seeds:
            qaoa_result = optimize_qaoa_angles(
                qubo_model.h,
                qubo_model.J,
                depth,
                seed=seed,
                restarts=qaoa_optimizer_restarts,
                coordinate_descent_rounds=qaoa_coordinate_descent_rounds,
                initial_step=qaoa_initial_step,
            )

            for trace_row in qaoa_result.optimization_trace:
                optimizer_records.append(
                    {
                        "model_version": qubo_model.version,
                        "depth": depth,
                        "seed": seed,
                        "expected_energy": qaoa_result.expected_energy,
                        **trace_row,
                    }
                )

            for connectivity_mode in connectivity_modes:
                pruned_top_k = 12 if connectivity_mode == "pruned" else None
                circuit_result = build_cirq_qaoa_circuit(
                    qubo_model,
                    qaoa_result,
                    connectivity_mode=connectivity_mode,
                    pruned_top_k=pruned_top_k,
                )

                if kernel_output_dir is not None and circuit_result.circuit is not None:
                    kernel_filename = (
                        f"{qubo_model.version}_p{depth}_seed{seed}_{connectivity_mode}.txt"
                    )
                    export_bloqade_kernel(
                        circuit_result.circuit,
                        Path(kernel_output_dir) / kernel_filename,
                    )

                connectivity_ideal_probabilities = qaoa_result.probabilities
                connectivity_ideal_backend = "numpy_exact"
                if circuit_result.circuit is not None and _get_cirq() is not None:
                    try:
                        connectivity_ideal_probabilities = _probabilities_from_cirq_circuit(
                            circuit_result.circuit,
                            noisy=False,
                        )
                        connectivity_ideal_backend = "cirq_ideal"
                    except Exception as exc:
                        circuit_result.notes.append(
                            f"Failed to simulate ideal circuit for connectivity mode '{connectivity_mode}': {exc}"
                        )

                for noise_model in noise_models:
                    applicable_strengths = (0.0,) if noise_model == "none" else noise_strengths
                    applicable_readout = (0.0,) if noise_model == "none" else readout_error_values

                    for noise_strength in applicable_strengths:
                        for readout_error in applicable_readout:
                            probabilities, backend, notes = simulate_with_noise_if_available(
                                circuit_result,
                                connectivity_ideal_probabilities,
                                ideal_backend=connectivity_ideal_backend,
                                noise_model=noise_model,
                                noise_strength=noise_strength,
                                readout_error=readout_error,
                            )

                            # We intentionally keep the sampling seed independent
                            # of the noise-model label. That way, if two
                            # configurations produce the exact same probability
                            # distribution (for example `noise_model=none` and a
                            # zero-strength noise model), their finite-shot
                            # samples also match, which makes side-by-side
                            # comparisons much easier to interpret.
                            sampling_seed = (
                                seed * 1_000_003
                                + depth * 10_007
                                + model_version_index.get(qubo_model.version, 0) * 101
                                + connectivity_index.get(connectivity_mode, 0) * 17
                            ) % (2**32)
                            counts = sample_counts_from_probabilities(
                                probabilities,
                                shots=shots,
                                seed=sampling_seed,
                            )

                            run_id = (
                                f"{qubo_model.version}-p{depth}-seed{seed}-"
                                f"{connectivity_mode}-{noise_model}-ns{noise_strength:.2f}-ro{readout_error:.3f}"
                            )

                            run_records: list[dict[str, Any]] = []
                            for bitstring, count in sorted(counts.items()):
                                selection = bitstring_to_selection(bitstring)
                                record = evaluate_selection(
                                    aggregated,
                                    qubo_model,
                                    selection,
                                    alpha=var_alpha,
                                    metadata={
                                        "run_id": run_id,
                                        "model_version": qubo_model.version,
                                        "q": qubo_model.hyperparameters.risk_aversion,
                                        "lambda": qubo_model.hyperparameters.penalty_lambda,
                                        "gamma": qubo_model.hyperparameters.capital_penalty,
                                        "eta": qubo_model.hyperparameters.illiquidity_penalty,
                                        "B": qubo_model.hyperparameters.budget,
                                        "p": depth,
                                        "backend": backend,
                                        "noise_model": noise_model,
                                        "noise_strength": noise_strength,
                                        "readout_error": readout_error,
                                        "connectivity_mode": connectivity_mode,
                                        "seed": seed,
                                        "count": count,
                                        "freq": count / shots,
                                        "circuit_depth": circuit_result.circuit_depth,
                                        "two_qubit_gate_count": circuit_result.two_qubit_gate_count,
                                        "edge_count": circuit_result.edge_count,
                                    },
                                )
                                run_records.append(record)

                            sample_records.extend(run_records)

                            best_feasible_sample = None
                            feasible_run_records = [record for record in run_records if record["feasible"]]
                            if feasible_run_records:
                                best_feasible_sample = min(
                                    feasible_run_records,
                                    key=lambda record: record["objective_unscaled"],
                                )

                            summary = summarize_probability_distribution(
                                aggregated,
                                qubo_model,
                                probabilities,
                                optimum_bitstring=optimum_bitstring,
                                alpha=var_alpha,
                            )
                            summary.update(
                                {
                                    "run_id": run_id,
                                    "model_version": qubo_model.version,
                                    "q": qubo_model.hyperparameters.risk_aversion,
                                    "lambda": qubo_model.hyperparameters.penalty_lambda,
                                    "gamma": qubo_model.hyperparameters.capital_penalty,
                                    "eta": qubo_model.hyperparameters.illiquidity_penalty,
                                    "B": qubo_model.hyperparameters.budget,
                                    "p": depth,
                                    "backend": backend,
                                    "noise_model": noise_model,
                                    "noise_strength": noise_strength,
                                    "readout_error": readout_error,
                                    "connectivity_mode": connectivity_mode,
                                    "seed": seed,
                                    "qaoa_expected_energy": qaoa_result.expected_energy,
                                    "qaoa_best_bitstring": qaoa_result.best_bitstring,
                                    "qaoa_best_probability": qaoa_result.best_probability,
                                    "circuit_depth": circuit_result.circuit_depth,
                                    "two_qubit_gate_count": circuit_result.two_qubit_gate_count,
                                    "edge_count": circuit_result.edge_count,
                                    "notes": " | ".join(notes),
                                    "best_feasible_sampled_bitstring": best_feasible_sample["bitstring"] if best_feasible_sample else None,
                                    "best_feasible_sampled_objective_unscaled": (
                                        best_feasible_sample["objective_unscaled"] if best_feasible_sample else None
                                    ),
                                }
                            )
                            summary_records.append(summary)

    return sample_records, summary_records, optimizer_records
