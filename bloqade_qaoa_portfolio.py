# # QAOA Portfolio Optimization on Bloqade
# 
# This notebook implements the 8-qubit insurer-aware portfolio QAOA circuit using
# QuEra's Bloqade SDK. We build the circuit as a `@squin.kernel`, apply
# Bloqade's auto-parallelization and hardware noise models, and compare the
# results against our classical brute-force optima.
# 
# **Pipeline:**
# 1. Load the Hartford workbook and aggregate 50 assets into 8 sectors
# 2. Build the QUBO and convert to Ising Hamiltonian
# 3. Optimize QAOA angles with the NumPy simulator
# 4. Construct the QAOA circuit as a Bloqade squin kernel
# 5. Compare naive, auto-parallelized, and hand-tuned circuits
# 6. Apply Gemini noise models and measure fidelity degradation
# 7. Sample bitstrings and evaluate portfolio quality

# ## 1. Setup and Imports

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import cirq
from bloqade import squin, cirq_utils
import bloqade.cirq_utils as utils

# Our project modules
from portfolio_model import (
    load_workbook_dataset,
    generate_synthetic_dataset,
    aggregate_to_sectors,
    build_qubo_model,
)
from classical_baselines import ensure_feasible_penalty
from qaoa_simulator import optimize_qaoa_angles
from project_config import ModelHyperparameters
from results_metrics import (
    evaluate_selection,
    bitstring_to_selection,
    sample_counts_from_probabilities,
    summarize_probability_distribution,
)

print("All imports successful.")

# ## 2. Load Data and Build QUBO

# Load the real Hartford workbook (fall back to synthetic if not available)
from pathlib import Path

workbook_path = Path("investment_dataset_full.xlsx")
if workbook_path.exists():
    print(f"Loading real workbook: {workbook_path}")
    raw_dataset = load_workbook_dataset(workbook_path)
else:
    print("Workbook not found, using synthetic data.")
    raw_dataset = generate_synthetic_dataset(n_assets=50, n_scenarios=250, seed=123)

aggregated = aggregate_to_sectors(raw_dataset)
print(f"Sectors: {aggregated.sector_names}")
print(f"Expected returns (mu): {np.round(aggregated.mu, 6)}")
print(f"Covariance matrix shape: {aggregated.sigma.shape}")

# Build all four model variants
hyperparams = ModelHyperparameters(
    budget=4,
    risk_aversion=0.5,
    penalty_lambda=8.0,
    capital_penalty=0.25,
    illiquidity_penalty=0.25,
)

models = {}
exact_optima = {}

for version in ["base", "capital", "liquidity", "both"]:
    qubo_model = build_qubo_model(aggregated, hyperparams, version=version)
    qubo_model, exact_summary = ensure_feasible_penalty(aggregated, qubo_model)
    models[version] = qubo_model
    exact_optima[version] = exact_summary.best_feasible
    selected_sectors = [
        aggregated.sector_names[i]
        for i, bit in enumerate(exact_summary.best_feasible["bitstring"])
        if bit == "1"
    ]
    print(f"{version:10s} | optimum: {exact_summary.best_feasible['bitstring']} | {selected_sectors}")

print("\nAll four QUBO models built and validated.")

# ## 3. Optimize QAOA Angles
# 
# We use our NumPy exact simulator to find the best QAOA parameters.
# These optimized angles are then used to construct the Bloqade circuits.

# Optimize for the base model at depths p=1,2,3
qubo = models["base"]
optimum_bitstring = exact_optima["base"]["bitstring"]

qaoa_results = {}
for depth in [1, 2, 3]:
    result = optimize_qaoa_angles(
        qubo.h, qubo.J, depth,
        seed=7, restarts=12,
        coordinate_descent_rounds=5,
        initial_step=0.6,
    )
    qaoa_results[depth] = result
    print(f"p={depth}: expected energy = {result.expected_energy:.6f}, "
          f"best bitstring = {result.best_bitstring}, "
          f"P(best) = {result.best_probability:.4f}")

print(f"\nClassical optimum: {optimum_bitstring}")

# ## 4. Build QAOA Circuit as Bloqade Squin Kernel (Naive)
# 
# Following the Bloqade tutorial pattern, we implement the QAOA cost layer
# using the native neutral-atom decomposition:
# 
# $$e^{-i \gamma J_{ij} Z_i Z_j} = H_j \cdot CZ(i,j) \cdot R_x(2\gamma J_{ij})_j \cdot CZ(i,j) \cdot H_j$$
# 
# Single-qubit field terms use $R_z(2\gamma h_i)$.
# 
# The mixer layer applies $R_x(2\beta)$ on all qubits.

def build_portfolio_qaoa_naive(
    h: np.ndarray,
    J: np.ndarray,
    gammas: np.ndarray,
    betas: np.ndarray,
) -> cirq.Circuit:
    """
    Build a naive (sequential) QAOA circuit for the portfolio Ising Hamiltonian
    using Bloqade's squin kernel.

    H_C = sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j
    """
    n = len(h)
    p = len(gammas)

    # Extract edges with non-zero coupling
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                edges.append((i, j, float(J[i, j])))

    # Store in lists for squin kernel access
    edge_i = [e[0] for e in edges]
    edge_j = [e[1] for e in edges]
    edge_w = [e[2] for e in edges]
    n_edges = len(edges)
    h_list = [float(x) for x in h]
    gamma_list = [float(g) for g in gammas]
    beta_list = [float(b) for b in betas]

    @squin.kernel
    def qaoa_portfolio_kernel():
        q = squin.qalloc(n)

        # Initial state: |+>^n
        for i in range(n):
            squin.h(q[i])

        # QAOA layers
        for layer in range(p):
            gamma = gamma_list[layer]
            beta = beta_list[layer]

            # Cost layer: single-qubit field terms Rz(2*gamma*h_i)
            for i in range(n):
                squin.rz(2.0 * gamma * h_list[i], q[i])

            # Cost layer: two-qubit ZZ terms
            # exp(-i*gamma*J*Z_i*Z_j) = H_j -> CZ(i,j) -> Rx(2*gamma*J) -> CZ(i,j) -> H_j
            for k in range(n_edges):
                u = edge_i[k]
                v = edge_j[k]
                w = edge_w[k]
                squin.h(q[v])
                squin.cz(q[u], q[v])
                squin.rx(2.0 * gamma * w, q[v])
                squin.cz(q[u], q[v])
                squin.h(q[v])

            # Mixer layer: Rx(2*beta) on all qubits
            for i in range(n):
                squin.rx(2.0 * beta, q[i])

    qubits = cirq.LineQubit.range(n)
    circuit = cirq_utils.emit_circuit(qaoa_portfolio_kernel, circuit_qubits=qubits)
    return circuit


# Build naive circuit for p=1
res = qaoa_results[1]
circuit_naive = build_portfolio_qaoa_naive(qubo.h, qubo.J, res.gammas, res.betas)
print(f"Naive circuit depth: {len(circuit_naive)} moments")
print(f"Total operations: {sum(len(m) for m in circuit_naive)}")

# ## 5. Hand-Tuned Parallel Circuit via Edge Coloring
# 
# For a complete graph $K_8$ (28 edges), greedy edge coloring gives 7 color
# classes. Edges in the same color class act on disjoint qubits and can be
# executed in parallel, reducing circuit depth from 28 sequential ZZ groups
# to 7 parallel groups.
# 
# We follow the Bloqade tutorial's approach: use `networkx` line-graph coloring
# and a minimum vertex cover to minimize Hadamard gate overhead.

def build_portfolio_qaoa_hand_tuned(
    h: np.ndarray,
    J: np.ndarray,
    gammas: np.ndarray,
    betas: np.ndarray,
) -> cirq.Circuit:
    """
    Build a hand-tuned parallel QAOA circuit using edge coloring
    to minimize circuit depth, following the Bloqade tutorial pattern.
    """
    n = len(h)
    p = len(gammas)

    # Build the interaction graph from Ising couplings
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    edge_weights = {}
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                graph.add_edge(i, j)
                edge_weights[(i, j)] = float(J[i, j])

    # Edge coloring via line graph greedy coloring
    # Try multiple strategies and pick the one with fewest colors
    linegraph = nx.line_graph(graph)
    best_num_colors = 1e99
    best_coloring = None
    for strategy in [
        "largest_first",
        "random_sequential",
        "smallest_last",
        "independent_set",
        "connected_sequential_bfs",
        "connected_sequential_dfs",
        "saturation_largest_first",
    ]:
        coloring = nx.coloring.greedy_color(linegraph, strategy=strategy)
        num_colors = len(set(coloring.values()))
        if num_colors < best_num_colors:
            best_num_colors = num_colors
            best_coloring = coloring

    color_groups = [
        [edge for edge, c in best_coloring.items() if c == color_idx]
        for color_idx in sorted(set(best_coloring.values()))
    ]

    # Minimum vertex cover to minimize Hadamard overhead
    mis = nx.algorithms.approximation.maximum_independent_set(graph)
    hadamard_qubits = set(graph.nodes) - set(mis)

    # Determine which qubit gets the Hadamard for each edge
    # Normalize edge keys to (min, max) so lookup never fails due to ordering
    h_qubit_map = {}
    for edge in graph.edges:
        u, v = edge
        key = (min(u, v), max(u, v))
        if u in hadamard_qubits:
            h_qubit_map[key] = u
        else:
            h_qubit_map[key] = v

    # Flatten for squin kernel
    h_list = [float(x) for x in h]
    gamma_list = [float(g) for g in gammas]
    beta_list = [float(b) for b in betas]

    # Precompute group data as flat lists for squin
    group_data = []
    for group in color_groups:
        group_edges = []
        for edge in group:
            u, v = edge
            key = (min(u, v), max(u, v))
            w = edge_weights.get(key, 0.0)
            hq = h_qubit_map[key]
            group_edges.append((u, v, w, hq))
        group_data.append(group_edges)

    @squin.kernel
    def qaoa_parallel_kernel():
        q = squin.qalloc(n)

        # Initial state: |+>^n
        for i in range(n):
            squin.h(q[i])

        for layer in range(p):
            gamma = gamma_list[layer]
            beta = beta_list[layer]

            # Cost layer: single-qubit fields
            for i in range(n):
                squin.rz(2.0 * gamma * h_list[i], q[i])

            # Cost layer: parallel ZZ groups
            for group in group_data:
                # Hadamard layer
                for u, v, w, hq in group:
                    squin.h(q[hq])
                # First CZ layer
                for u, v, w, hq in group:
                    squin.cz(q[u], q[v])
                # Rotation layer
                for u, v, w, hq in group:
                    squin.rx(2.0 * gamma * w, q[hq])
                # Second CZ layer
                for u, v, w, hq in group:
                    squin.cz(q[u], q[v])
                # Second Hadamard layer
                for u, v, w, hq in group:
                    squin.h(q[hq])

            # Mixer layer
            for i in range(n):
                squin.rx(2.0 * beta, q[i])

    qubits = cirq.LineQubit.range(n)
    circuit = cirq_utils.emit_circuit(qaoa_parallel_kernel, circuit_qubits=qubits)

    # Optimize: merge redundant single-qubit gates and convert to native CZ gateset
    circuit = cirq.merge_single_qubit_moments_to_phxz(circuit)
    circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())
    return circuit


circuit_hand = build_portfolio_qaoa_hand_tuned(qubo.h, qubo.J, res.gammas, res.betas)
print(f"Hand-tuned circuit depth: {len(circuit_hand)} moments")


# ## 6. Auto-Parallelized Circuit
# 
# Bloqade's `utils.parallelize()` automatically compresses the naive circuit
# by building a DAG of gate dependencies and solving an ILP to minimize depth.

# Auto-parallelize the naive circuit
circuit_auto = utils.parallelize(circuit=circuit_naive)
circuit_auto = utils.remove_tags(circuit_auto)

print(f"Naive circuit depth:       {len(circuit_naive)}")
print(f"Auto-parallel depth:       {len(circuit_auto)}")
print(f"Hand-tuned parallel depth: {len(circuit_hand)}")

# ## 7. Circuit Depth Comparison Across QAOA Depths

depth_data = {"p": [], "Naive": [], "Auto": [], "Hand-tuned": []}

for p_depth in [1, 2, 3]:
    res_p = qaoa_results[p_depth]
    c_naive = build_portfolio_qaoa_naive(qubo.h, qubo.J, res_p.gammas, res_p.betas)
    c_hand = build_portfolio_qaoa_hand_tuned(qubo.h, qubo.J, res_p.gammas, res_p.betas)
    c_auto = utils.parallelize(circuit=c_naive)
    c_auto = utils.remove_tags(c_auto)

    depth_data["p"].append(p_depth)
    depth_data["Naive"].append(len(c_naive))
    depth_data["Auto"].append(len(c_auto))
    depth_data["Hand-tuned"].append(len(c_hand))

    print(f"p={p_depth}: Naive={len(c_naive):4d}  Auto={len(c_auto):4d}  Hand={len(c_hand):4d}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(depth_data["p"]))
width = 0.25
ax.bar(x - width, depth_data["Naive"], width, label="Naive", edgecolor="black")
ax.bar(x, depth_data["Auto"], width, label="Auto-parallel", edgecolor="black")
ax.bar(x + width, depth_data["Hand-tuned"], width, label="Hand-tuned", edgecolor="black")
ax.set_xlabel("QAOA Depth p")
ax.set_ylabel("Circuit Depth (moments)")
ax.set_title("Circuit Depth: Naive vs Auto vs Hand-Tuned")
ax.set_xticks(x)
ax.set_xticklabels([f"p={p}" for p in depth_data["p"]])
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("artifacts/figures/bloqade_depth_comparison.png", dpi=220)
plt.show()

# ## 8. Noise Analysis with Gemini Noise Models
# 
# Bloqade provides hardware-calibrated noise models for QuEra's neutral-atom
# architecture. We use:
# 
# - **GeminiOneZoneNoiseModel**: Single atom-loss zone (dominant error source)
# - **GeminiTwoZoneNoiseModel**: Two independent loss zones
# 
# The workflow follows the tutorial:
# 1. Build the ideal circuit
# 2. Apply `noise.transform_circuit()` to inject realistic noise channels
# 3. Simulate with `DensityMatrixSimulator` to get the noisy density matrix
# 4. Compare against the ideal state to compute fidelity

# Initialize noise models and simulator
noise_one_zone = utils.noise.GeminiOneZoneNoiseModel()
noise_two_zone = utils.noise.GeminiTwoZoneNoiseModel()
simulator = cirq.DensityMatrixSimulator()

print("Noise models initialized.")

def compute_fidelity(circuit, noise_model, sim):
    """Compute fidelity between ideal and noisy circuit execution."""
    noisy_circuit = utils.noise.transform_circuit(circuit, model=noise_model)
    rho_ideal = sim.simulate(circuit).final_density_matrix
    rho_noisy = sim.simulate(noisy_circuit).final_density_matrix
    return float(np.trace(rho_ideal @ rho_noisy).real)


def get_probabilities(circuit, sim, noisy=False):
    """Extract probability vector from circuit simulation."""
    if noisy:
        result = sim.simulate(circuit)
        probs = np.real(np.diag(result.final_density_matrix))
    else:
        result = sim.simulate(circuit)
        probs = np.real(np.diag(result.final_density_matrix))

    probs = np.asarray(probs, dtype=float)
    probs = probs / np.sum(probs)

    # Cirq uses big-endian; our project uses little-endian
    n_qubits = int(np.log2(len(probs)))
    reordered = np.zeros_like(probs)
    for le_idx in range(len(probs)):
        be_idx = 0
        for bit in range(n_qubits):
            be_idx = (be_idx << 1) | ((le_idx >> bit) & 1)
        reordered[le_idx] = probs[be_idx]
    return reordered / np.sum(reordered)


# Compare fidelity across circuit variants and noise models
circuits = {
    "Naive": circuit_naive,
    "Auto-parallel": circuit_auto,
    "Hand-tuned": circuit_hand,
}

noise_models = {
    "One Zone": noise_one_zone,
    "Two Zone": noise_two_zone,
}

print(f"{'Circuit':<16} {'Depth':<8} {'One Zone Fid.':<16} {'Two Zone Fid.':<16}")
print("-" * 56)

fidelity_results = {}
for circ_name, circ in circuits.items():
    fid_one = compute_fidelity(circ, noise_one_zone, simulator)
    fid_two = compute_fidelity(circ, noise_two_zone, simulator)
    fidelity_results[circ_name] = {"One Zone": fid_one, "Two Zone": fid_two}
    print(f"{circ_name:<16} {len(circ):<8} {fid_one:<16.4f} {fid_two:<16.4f}")

# Plot fidelity comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

labels = list(circuits.keys())
depths = [len(c) for c in circuits.values()]
fid_one_vals = [fidelity_results[name]["One Zone"] for name in labels]
fid_two_vals = [fidelity_results[name]["Two Zone"] for name in labels]

x = np.arange(len(labels))
width = 0.35

# Fidelity plot
bars1 = ax1.bar(x - width/2, fid_one_vals, width, label="One Zone", edgecolor="black")
bars2 = ax1.bar(x + width/2, fid_two_vals, width, label="Two Zone", edgecolor="black")
ax1.set_ylabel("Fidelity")
ax1.set_title("Fidelity Under Gemini Noise Models")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()
ax1.grid(axis="y", alpha=0.3)
for bar, v in zip(bars1, fid_one_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{v:.3f}",
             ha="center", fontsize=9, fontweight="bold")
for bar, v in zip(bars2, fid_two_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{v:.3f}",
             ha="center", fontsize=9, fontweight="bold")

# Depth plot
colors = ["#555555", "#888888", "#bbbbbb"]
bars = ax2.bar(labels, depths, color=colors, edgecolor="black")
ax2.set_ylabel("Circuit Depth")
ax2.set_title("Circuit Depth Comparison")
ax2.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, depths):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(v),
             ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("artifacts/figures/bloqade_fidelity_comparison.png", dpi=220)
plt.show()

# ## 9. Sample Portfolios from Noisy Circuits
# 
# We sample bitstrings from the noisy circuit output and evaluate the
# resulting portfolio quality against the classical optimum.

# Use the hand-tuned circuit (best fidelity)
best_circuit = circuit_hand

# Get ideal probabilities
probs_ideal = get_probabilities(best_circuit, simulator)

# Get noisy probabilities (one zone)
noisy_circuit_one = utils.noise.transform_circuit(best_circuit, model=noise_one_zone)
probs_noisy = get_probabilities(noisy_circuit_one, simulator, noisy=True)

# Summarize for both
for label, probs in [("Ideal", probs_ideal), ("Noisy (One Zone)", probs_noisy)]:
    summary = summarize_probability_distribution(
        aggregated, qubo, probs,
        optimum_bitstring=optimum_bitstring,
    )
    print(f"\n--- {label} ---")
    print(f"  P(optimal):              {summary['optimal_bitstring_probability']:.4f}")
    print(f"  P(feasible):             {summary['feasible_probability_mass']:.4f}")
    cond = summary['optimal_bitstring_probability'] / max(summary['feasible_probability_mass'], 1e-12)
    print(f"  P(optimal | feasible):   {cond:.4f}")
    print(f"  Uniform baseline (1/70): {1/70:.4f}")

# Sample bitstrings and find the best feasible portfolio
shots = 2048

for label, probs in [("Ideal", probs_ideal), ("Noisy (One Zone)", probs_noisy)]:
    counts = sample_counts_from_probabilities(probs, shots=shots, seed=42)

    # Evaluate all sampled bitstrings
    records = []
    for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
        selection = bitstring_to_selection(bitstring)
        record = evaluate_selection(aggregated, qubo, selection)
        record["count"] = count
        records.append(record)

    # Best feasible
    feasible = [r for r in records if r["feasible"]]
    if feasible:
        best = min(feasible, key=lambda r: r["objective_unscaled"])
        match = "YES" if best["bitstring"] == optimum_bitstring else "NO"
        selected = [
            aggregated.sector_names[i]
            for i, b in enumerate(best["bitstring"])
            if b == "1"
        ]
        print(f"\n--- {label} ({shots} shots) ---")
        print(f"  Best feasible: {best['bitstring']} -> {selected}")
        print(f"  Matches exact optimum ({optimum_bitstring}): {match}")
        print(f"  Expected return: {best['exp_return']:.6f}")
        print(f"  Volatility:     {best['volatility']:.6f}")
        print(f"  VaR95:          {best['var95']:.6f}")
        print(f"  CVaR95:         {best['cvar95']:.6f}")
    else:
        print(f"\n--- {label}: No feasible bitstrings sampled ---")

# ## 10. All Four Model Variants on Bloqade
# 
# Run the full QAOA pipeline for all four insurance model variants
# (base, capital, liquidity, both) and verify quantum matches classical.

print(f"{'Model':<12} {'Exact Opt.':<12} {'Quantum Best':<14} {'Match':<8} {'Fidelity':<10}")
print("-" * 56)

for version in ["base", "capital", "liquidity", "both"]:
    q_model = models[version]
    opt_bs = exact_optima[version]["bitstring"]

    # Optimize QAOA angles for this model
    qaoa_res = optimize_qaoa_angles(
        q_model.h, q_model.J, depth=1,
        seed=7, restarts=12,
    )

    # Build hand-tuned circuit
    circ = build_portfolio_qaoa_hand_tuned(
        q_model.h, q_model.J,
        qaoa_res.gammas, qaoa_res.betas,
    )

    # Compute fidelity under noise
    fid = compute_fidelity(circ, noise_one_zone, simulator)

    # Get ideal probabilities and sample
    probs = get_probabilities(circ, simulator)
    counts = sample_counts_from_probabilities(probs, shots=2048, seed=42)

    # Find best feasible
    best_bs = None
    best_obj = float("inf")
    for bs, cnt in counts.items():
        sel = bitstring_to_selection(bs)
        rec = evaluate_selection(aggregated, q_model, sel)
        if rec["feasible"] and rec["objective_unscaled"] < best_obj:
            best_obj = rec["objective_unscaled"]
            best_bs = bs

    match = "YES" if best_bs == opt_bs else "NO"
    print(f"{version:<12} {opt_bs:<12} {best_bs or 'N/A':<14} {match:<8} {fid:<10.4f}")

# ## 11. Interaction Graph Visualization
# 
# Visualize the Ising coupling structure as an edge-colored graph,
# showing how the 28 edges of $K_8$ are partitioned into parallel groups.

# Build the coupling graph
graph = nx.Graph()
n = len(qubo.h)
graph.add_nodes_from(range(n))
for i in range(n):
    for j in range(i + 1, n):
        if abs(qubo.J[i, j]) > 1e-12:
            graph.add_edge(i, j, weight=abs(qubo.J[i, j]))

# Edge coloring
linegraph = nx.line_graph(graph)
coloring = min(
    [nx.coloring.greedy_color(linegraph, strategy=s)
     for s in ["largest_first", "smallest_last", "saturation_largest_first"]],
    key=lambda c: len(set(c.values())),
)
color_groups = [
    [e for e, c in coloring.items() if c == idx]
    for idx in sorted(set(coloring.values()))
]

# Plot
pos = nx.kamada_kawai_layout(graph)
fig, ax = plt.subplots(figsize=(8, 7))

edge_cmap = plt.cm.tab10
for color_idx, group in enumerate(color_groups):
    edge_color = edge_cmap(color_idx / max(len(color_groups) - 1, 1))
    for u, v in group:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=2.5, zorder=1)

nx.draw_networkx_nodes(graph, pos, node_size=600, node_color="white",
                       edgecolors="black", linewidths=2, ax=ax)
sector_labels = {i: aggregated.sector_names[i].replace(" ", "\n")
                 for i in range(n)}
nx.draw_networkx_labels(graph, pos, labels=sector_labels, font_size=7,
                        font_weight="bold", ax=ax)

legend_elements = [
    plt.Line2D([0], [0], color=edge_cmap(i / max(len(color_groups) - 1, 1)),
               lw=4, label=f"Moment {i+1} ({len(color_groups[i])} edges)")
    for i in range(len(color_groups))
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
ax.set_title(f"Ising Coupling Graph: {len(color_groups)} Parallel Moments "
             f"(from {len(graph.edges)} edges)", fontsize=13)
ax.axis("off")
plt.tight_layout()
plt.savefig("artifacts/figures/bloqade_edge_coloring.png", dpi=220)
plt.show()

print(f"Total edges: {len(graph.edges)}")
print(f"Color classes (parallel moments): {len(color_groups)}")
print(f"Chromatic index of K8 = 7 (Vizing's theorem)")

# ## 12. Fidelity Scaling with QAOA Depth
# 
# How does noise affect deeper circuits? We compare fidelity at p=1, 2, 3
# for each parallelization strategy.

print(f"{'p':<4} {'Strategy':<16} {'Depth':<8} {'Fidelity (1-zone)':<20}")
print("-" * 48)

fid_by_depth = {"Naive": [], "Auto": [], "Hand-tuned": []}
p_values = [1, 2, 3]

for p_depth in p_values:
    res_p = qaoa_results[p_depth]

    c_naive = build_portfolio_qaoa_naive(qubo.h, qubo.J, res_p.gammas, res_p.betas)
    c_hand = build_portfolio_qaoa_hand_tuned(qubo.h, qubo.J, res_p.gammas, res_p.betas)
    c_auto = utils.parallelize(circuit=c_naive)
    c_auto = utils.remove_tags(c_auto)

    for name, circ in [("Naive", c_naive), ("Auto", c_auto), ("Hand-tuned", c_hand)]:
        fid = compute_fidelity(circ, noise_one_zone, simulator)
        fid_by_depth[name].append(fid)
        print(f"{p_depth:<4} {name:<16} {len(circ):<8} {fid:<20.4f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
for name, fids in fid_by_depth.items():
    ax.plot(p_values, fids, "o-", linewidth=2, markersize=8, label=name)

ax.set_xlabel("QAOA Depth p", fontsize=12)
ax.set_ylabel("Fidelity (Gemini One Zone)", fontsize=12)
ax.set_title("Fidelity vs QAOA Depth by Parallelization Strategy", fontsize=13)
ax.set_xticks(p_values)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("artifacts/figures/bloqade_fidelity_vs_depth.png", dpi=220)
plt.show()

# ## 13. Summary
# 
# | Requirement | Status |
# |---|---|
# | Formulate portfolio as optimization problem | QUBO with 4 insurer-aware variants |
# | Construct QUBO / Ising representation | Both constructed and exported |
# | Run circuit on Bloqade with 8 qubits | `@squin.kernel` circuits built and simulated |
# | Investigate connectivity and noise | Naive/auto/hand-tuned + Gemini noise models |
# | Demonstrate hardware assumptions affect outcomes | Fidelity degrades with depth; parallelization helps |
# | Stakeholder-applicable solution | Best quantum portfolio matches classical optimum in all 4 models |
