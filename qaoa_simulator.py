from __future__ import annotations

"""
Exact 8-qubit QAOA simulator implemented with NumPy only.

This gives us a dependable local path for development and presentation prep
even when Bloqade/Cirq are not installed yet.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(slots=True)
class QAOAOptimizationResult:
    depth: int
    gammas: np.ndarray
    betas: np.ndarray
    statevector: np.ndarray
    probabilities: np.ndarray
    expected_energy: float
    best_bitstring: str
    best_probability: float
    optimization_trace: list[dict]


@lru_cache(maxsize=None)
def basis_bit_table(num_qubits: int) -> np.ndarray:
    """
    Precompute all computational basis states as a matrix of 0/1 bits.
    """

    states = np.zeros((2**num_qubits, num_qubits), dtype=int)
    for state_index in range(2**num_qubits):
        for bit in range(num_qubits):
            states[state_index, bit] = (state_index >> bit) & 1
    return states


def precompute_ising_energies(h: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    Since the cost Hamiltonian is diagonal in the computational basis, we can
    precompute the cost of every basis state once and reuse it for all angle
    evaluations.
    """

    num_qubits = len(h)
    bits = basis_bit_table(num_qubits)
    spins = 1 - 2 * bits

    energies = np.zeros(2**num_qubits, dtype=float)
    for state_index, spin_state in enumerate(spins):
        pair_energy = 0.0
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                pair_energy += J[i, j] * spin_state[i] * spin_state[j]
        energies[state_index] = np.dot(h, spin_state) + pair_energy
    return energies


def apply_rx_layer(statevector: np.ndarray, beta: float, num_qubits: int) -> np.ndarray:
    """
    Apply exp(-i beta X) to every qubit in place.

    The implementation below works directly on the statevector without building
    a huge dense matrix, which keeps the code both fast and easy to follow for
    an 8-qubit system.
    """

    state = np.asarray(statevector, dtype=complex).copy()
    cos_beta = np.cos(beta)
    minus_i_sin_beta = -1j * np.sin(beta)

    for qubit in range(num_qubits):
        stride = 1 << qubit
        period = stride << 1
        for block_start in range(0, len(state), period):
            for offset in range(stride):
                index0 = block_start + offset
                index1 = index0 + stride
                amp0 = state[index0]
                amp1 = state[index1]
                state[index0] = cos_beta * amp0 + minus_i_sin_beta * amp1
                state[index1] = minus_i_sin_beta * amp0 + cos_beta * amp1
    return state


def simulate_qaoa_state(h: np.ndarray, J: np.ndarray, gammas: np.ndarray, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a p-layer QAOA state exactly.
    """

    num_qubits = len(h)
    energies = precompute_ising_energies(h, J)
    state = np.ones(2**num_qubits, dtype=complex) / np.sqrt(2**num_qubits)

    for gamma, beta in zip(gammas, betas):
        # Cost layer: diagonal in the computational basis.
        state *= np.exp(-1j * gamma * energies)
        # Mixer layer: product of single-qubit RX rotations.
        state = apply_rx_layer(state, beta, num_qubits)

    return state, energies


def expected_energy(statevector: np.ndarray, energies: np.ndarray) -> float:
    """
    Compute <psi|H|psi> for a diagonal Hamiltonian from basis probabilities.
    """

    probabilities = np.abs(statevector) ** 2
    return float(np.dot(probabilities, energies))


def _angles_to_vector(gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(gammas, dtype=float), np.asarray(betas, dtype=float)])


def _vector_to_angles(vector: np.ndarray, depth: int) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(vector[:depth], dtype=float), np.asarray(vector[depth:], dtype=float)


def _random_initial_angles(depth: int, rng: np.random.Generator) -> np.ndarray:
    """
    Random but sensible QAOA initialization ranges.
    """

    gammas = rng.uniform(-np.pi, np.pi, size=depth)
    betas = rng.uniform(0.0, np.pi / 2.0, size=depth)
    return _angles_to_vector(gammas, betas)


def optimize_qaoa_angles(
    h: np.ndarray,
    J: np.ndarray,
    depth: int,
    *,
    seed: int = 7,
    restarts: int = 12,
    coordinate_descent_rounds: int = 5,
    initial_step: float = 0.6,
) -> QAOAOptimizationResult:
    """
    Optimize QAOA angles with a lightweight multi-start coordinate descent.

    This is intentionally simple and dependency-light. For 8 qubits and shallow
    depths it works well enough for demos and benchmark comparisons.
    """

    rng = np.random.default_rng(seed)
    num_parameters = 2 * depth
    trace: list[dict] = []

    def objective(parameter_vector: np.ndarray) -> tuple[float, np.ndarray]:
        gammas, betas = _vector_to_angles(parameter_vector, depth)
        statevector, energies = simulate_qaoa_state(h, J, gammas, betas)
        return expected_energy(statevector, energies), statevector

    best_parameters = None
    best_statevector = None
    best_value = None

    # Global exploration: random initializations.
    for restart in range(restarts):
        parameters = _random_initial_angles(depth, rng)
        value, statevector = objective(parameters)
        trace.append({"stage": "restart", "restart": restart, "value": float(value)})
        if best_value is None or value < best_value:
            best_value = value
            best_parameters = parameters.copy()
            best_statevector = statevector.copy()

    assert best_parameters is not None
    assert best_statevector is not None

    # Local refinement: coordinate descent around the best random start.
    current = best_parameters.copy()
    current_value = float(best_value)
    step = float(initial_step)

    for round_index in range(coordinate_descent_rounds):
        improved = False
        for parameter_index in range(num_parameters):
            for direction in (-1.0, 1.0):
                candidate = current.copy()
                candidate[parameter_index] += direction * step
                candidate_value, candidate_statevector = objective(candidate)
                trace.append(
                    {
                        "stage": "coordinate_descent",
                        "round": round_index,
                        "parameter_index": parameter_index,
                        "step": step,
                        "value": float(candidate_value),
                    }
                )
                if candidate_value < current_value:
                    current = candidate
                    current_value = float(candidate_value)
                    best_statevector = candidate_statevector.copy()
                    improved = True
        step *= 0.5
        if not improved:
            break

    gammas, betas = _vector_to_angles(current, depth)
    probabilities = np.abs(best_statevector) ** 2
    best_state_index = int(np.argmax(probabilities))
    best_bitstring = "".join(str((best_state_index >> bit) & 1) for bit in range(len(h)))

    return QAOAOptimizationResult(
        depth=depth,
        gammas=gammas,
        betas=betas,
        statevector=best_statevector,
        probabilities=probabilities,
        expected_energy=current_value,
        best_bitstring=best_bitstring,
        best_probability=float(np.max(probabilities)),
        optimization_trace=trace,
    )

