from __future__ import annotations

"""
Data loading, sector aggregation, QUBO construction, and Ising conversion.

This file intentionally contains a lot of comments because it is the part of
the project where most of the domain assumptions are encoded.
"""

from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from project_config import CANONICAL_SECTORS, COLUMN_ALIASES, ModelHyperparameters, SECTOR_ALIASES


@dataclass(slots=True)
class AssetRecord:
    asset_id: str
    asset_name: str
    sector: str
    expected_return: float | None
    capital_charge: float | None
    liquidity: float | None


@dataclass(slots=True)
class RawDataset:
    """
    A clean, workbook-independent representation of the input data.

    We convert spreadsheets into this object first so the rest of the code does
    not have to care whether the source was Excel or synthetic demo data.
    """

    asset_records: list[AssetRecord]
    scenario_labels: list[str]
    asset_ids_in_scenarios: list[str]
    scenario_returns: np.ndarray
    scenario_probabilities: np.ndarray | None = None
    asset_covariance: np.ndarray | None = None
    workbook_summary: dict[str, Any] | None = None


@dataclass(slots=True)
class AggregatedDataset:
    """
    Sector-level data used by every downstream stage.
    """

    sector_names: tuple[str, ...]
    sector_to_asset_ids: dict[str, list[str]]
    asset_ids: list[str]
    aggregation_matrix: np.ndarray
    asset_return_matrix: np.ndarray
    sector_return_matrix: np.ndarray
    scenario_labels: list[str]
    scenario_probabilities: np.ndarray | None
    mu: np.ndarray
    sigma: np.ndarray
    capital: np.ndarray
    liquidity: np.ndarray
    normalized_capital: np.ndarray
    normalized_illiquidity: np.ndarray
    asset_expected_returns: np.ndarray | None
    validation_summary: dict[str, float | None]


@dataclass(slots=True)
class QuboModel:
    """
    The model artifact we export for reproducibility.
    """

    version: str
    hyperparameters: ModelHyperparameters
    Q_unscaled: np.ndarray
    Q_scaled: np.ndarray
    scale_alpha: float
    h: np.ndarray
    J: np.ndarray
    offset: float
    included_capital_penalty: bool
    included_liquidity_penalty: bool


def _normalize_text(value: Any) -> str:
    """
    Normalization helper used for fuzzy matching of sheet names, headers, and
    sector labels.
    """

    text = str(value or "").strip().lower()
    for char in ("_", "-", "/", "\\", "(", ")", ".", ","):
        text = text.replace(char, " ")
    text = " ".join(text.split())
    return text


def _safe_float(value: Any) -> float | None:
    """
    Convert spreadsheet values to floats when possible.
    """

    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(numeric):
        return None
    return numeric


def _require_openpyxl():
    """
    Import `openpyxl` lazily so the project can still be imported even when the
    optional Excel dependency is not installed yet.
    """

    try:
        return import_module("openpyxl")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading Excel workbooks requires `openpyxl`. Install it in PyCharm "
            "with `pip install openpyxl`."
        ) from exc


def _guess_sheet_name(sheet_names: list[str], preferred: str | None, keywords: tuple[str, ...]) -> str:
    """
    Pick the most likely worksheet name if the user did not specify one.
    """

    if preferred:
        if preferred not in sheet_names:
            raise ValueError(f"Worksheet '{preferred}' was not found. Available sheets: {sheet_names}")
        return preferred

    normalized_to_original = {_normalize_text(name): name for name in sheet_names}

    for keyword in keywords:
        for normalized_name, original_name in normalized_to_original.items():
            if keyword in normalized_name:
                return original_name

    raise ValueError(
        f"Could not infer the correct worksheet from names {sheet_names}. "
        f"Please pass the sheet name explicitly."
    )


def _worksheet_to_rows(worksheet) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Convert an Excel sheet into headers plus row dictionaries.

    We keep the code simple and explicit so it is easy to debug when the user's
    workbook schema differs slightly from expectations.
    """

    rows = list(worksheet.iter_rows(values_only=True))
    if not rows:
        raise ValueError(f"Worksheet '{worksheet.title}' is empty.")

    raw_headers = rows[0]
    headers = [str(cell).strip() if cell is not None else f"column_{i}" for i, cell in enumerate(raw_headers)]

    dict_rows: list[dict[str, Any]] = []
    for raw_row in rows[1:]:
        row_dict = {headers[i]: raw_row[i] if i < len(raw_row) else None for i in range(len(headers))}
        # Skip rows that are completely blank.
        if any(value not in (None, "") for value in row_dict.values()):
            dict_rows.append(row_dict)
    return headers, dict_rows


def _find_column(headers: list[str], canonical_key: str) -> str | None:
    """
    Match a workbook header to one of our canonical semantic column names.
    """

    normalized_headers = {_normalize_text(header): header for header in headers}
    for alias in COLUMN_ALIASES[canonical_key]:
        if alias in normalized_headers:
            return normalized_headers[alias]
    return None


def canonicalize_sector_name(raw_value: Any) -> str:
    """
    Convert noisy sector labels into the exact 8-sector vocabulary used in the
    report and in the qubit ordering.
    """

    normalized = _normalize_text(raw_value)
    if normalized in SECTOR_ALIASES:
        return SECTOR_ALIASES[normalized]

    # If the value already exactly matches one of the canonical sectors after
    # normalization, use it directly.
    for sector in CANONICAL_SECTORS:
        if normalized == _normalize_text(sector):
            return sector

    raise ValueError(
        f"Could not map sector label '{raw_value}' to one of the expected 8 sectors: "
        f"{CANONICAL_SECTORS}"
    )


def load_workbook_dataset(
    workbook_path: str | Path,
    assets_sheet_name: str | None = None,
    scenarios_sheet_name: str | None = None,
    covariance_sheet_name: str | None = None,
) -> RawDataset:
    """
    Load the hackathon workbook into a clean in-memory representation.
    """

    workbook_path = Path(workbook_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")

    openpyxl = _require_openpyxl()
    workbook = openpyxl.load_workbook(workbook_path, data_only=True, read_only=True)
    sheet_names = workbook.sheetnames

    assets_sheet = workbook[_guess_sheet_name(sheet_names, assets_sheet_name, ("asset", "assets"))]
    scenarios_sheet = workbook[_guess_sheet_name(sheet_names, scenarios_sheet_name, ("scenario", "scenarios"))]

    covariance_sheet = None
    if covariance_sheet_name is not None:
        covariance_sheet = workbook[_guess_sheet_name(sheet_names, covariance_sheet_name, ("cov",))]
    else:
        for sheet_name in sheet_names:
            if "cov" in _normalize_text(sheet_name):
                covariance_sheet = workbook[sheet_name]
                break

    asset_headers, asset_rows = _worksheet_to_rows(assets_sheet)
    scenario_headers, scenario_rows = _worksheet_to_rows(scenarios_sheet)

    asset_id_col = _find_column(asset_headers, "asset_id")
    asset_name_col = _find_column(asset_headers, "asset_name")
    sector_col = _find_column(asset_headers, "sector")
    exp_return_col = _find_column(asset_headers, "expected_return")
    capital_col = _find_column(asset_headers, "capital_charge")
    liquidity_col = _find_column(asset_headers, "liquidity")

    if sector_col is None:
        raise ValueError(
            f"Could not find a sector-like column in sheet '{assets_sheet.title}'. "
            f"Headers were: {asset_headers}"
        )

    asset_records: list[AssetRecord] = []
    for index, row in enumerate(asset_rows):
        asset_id = str(row.get(asset_id_col) or row.get(asset_name_col) or f"asset_{index}").strip()
        asset_name = str(row.get(asset_name_col) or asset_id).strip()
        sector = canonicalize_sector_name(row.get(sector_col))
        asset_records.append(
            AssetRecord(
                asset_id=asset_id,
                asset_name=asset_name,
                sector=sector,
                expected_return=_safe_float(row.get(exp_return_col)) if exp_return_col else None,
                capital_charge=_safe_float(row.get(capital_col)) if capital_col else None,
                liquidity=_safe_float(row.get(liquidity_col)) if liquidity_col else None,
            )
        )

    probability_col = _find_column(scenario_headers, "scenario_probability")
    scenario_label_col = _find_column(scenario_headers, "scenario_label")

    # We match scenario columns to asset ids first, then to asset names as a
    # fallback. The normalized lookup allows minor workbook formatting changes.
    header_lookup = {_normalize_text(header): header for header in scenario_headers}

    aligned_asset_ids: list[str] = []
    aligned_returns: list[list[float]] = []

    for asset in asset_records:
        matched_header = None
        for candidate in (asset.asset_id, asset.asset_name):
            normalized_candidate = _normalize_text(candidate)
            if normalized_candidate in header_lookup:
                matched_header = header_lookup[normalized_candidate]
                break
        if matched_header is None:
            raise ValueError(
                f"Could not find a scenario-return column for asset '{asset.asset_id}' / '{asset.asset_name}'. "
                f"Scenario headers were: {scenario_headers}"
            )

        aligned_asset_ids.append(asset.asset_id)
        aligned_returns.append([_safe_float(row.get(matched_header)) or 0.0 for row in scenario_rows])

    scenario_returns = np.asarray(aligned_returns, dtype=float).T
    scenario_labels = [
        str(row.get(scenario_label_col) or f"scenario_{i}")
        for i, row in enumerate(scenario_rows)
    ]

    probabilities = None
    if probability_col:
        weights = np.asarray([_safe_float(row.get(probability_col)) or 0.0 for row in scenario_rows], dtype=float)
        if np.sum(weights) > 0.0:
            probabilities = weights / np.sum(weights)

    covariance_matrix = None
    if covariance_sheet is not None:
        cov_headers, cov_rows = _worksheet_to_rows(covariance_sheet)
        # We accept a square matrix where both rows and columns use asset ids or names.
        row_label_header = cov_headers[0]
        numeric_headers = cov_headers[1:]
        covariance_matrix = np.zeros((len(asset_records), len(asset_records)), dtype=float)
        asset_lookup = {
            _normalize_text(asset.asset_id): idx
            for idx, asset in enumerate(asset_records)
        }
        asset_lookup.update(
            {
                _normalize_text(asset.asset_name): idx
                for idx, asset in enumerate(asset_records)
            }
        )

        # First build a dense lookup from the workbook.
        workbook_cov: dict[tuple[int, int], float] = {}
        for row in cov_rows:
            row_label = row.get(row_label_header)
            row_idx = asset_lookup.get(_normalize_text(row_label))
            if row_idx is None:
                continue
            for header in numeric_headers:
                col_idx = asset_lookup.get(_normalize_text(header))
                if col_idx is None:
                    continue
                workbook_cov[(row_idx, col_idx)] = _safe_float(row.get(header)) or 0.0

        for i in range(len(asset_records)):
            for j in range(len(asset_records)):
                covariance_matrix[i, j] = workbook_cov.get((i, j), 0.0)

    return RawDataset(
        asset_records=asset_records,
        scenario_labels=scenario_labels,
        asset_ids_in_scenarios=aligned_asset_ids,
        scenario_returns=scenario_returns,
        scenario_probabilities=probabilities,
        asset_covariance=covariance_matrix,
        workbook_summary={
            "workbook_path": str(workbook_path),
            "assets_sheet": assets_sheet.title,
            "scenarios_sheet": scenarios_sheet.title,
            "covariance_sheet": covariance_sheet.title if covariance_sheet is not None else None,
        },
    )


def generate_synthetic_dataset(
    n_assets: int = 50,
    n_scenarios: int = 250,
    seed: int = 123,
) -> RawDataset:
    """
    Create a synthetic portfolio dataset so the project can run immediately even
    when the real hackathon workbook is not in the repository.

    This is only a convenience dataset for local development. The real
    submission should still be run on the provided workbook.
    """

    rng = np.random.default_rng(seed)
    sector_names = list(CANONICAL_SECTORS)

    # We distribute assets as evenly as possible across the 8 sectors.
    base_count = n_assets // len(sector_names)
    extras = n_assets % len(sector_names)
    counts = [base_count + (1 if i < extras else 0) for i in range(len(sector_names))]

    sector_mean_returns = np.array([0.01, 0.018, 0.03, 0.045, 0.055, 0.05, 0.035, 0.04], dtype=float)
    sector_vols = np.array([0.005, 0.012, 0.018, 0.028, 0.035, 0.038, 0.022, 0.025], dtype=float)
    sector_capital = np.array([0.05, 0.15, 0.22, 0.4, 0.55, 0.58, 0.45, 0.32], dtype=float)
    sector_liquidity = np.array([0.98, 0.9, 0.78, 0.62, 0.8, 0.74, 0.5, 0.58], dtype=float)

    # Correlation structure chosen to produce a dense but realistic covariance.
    correlation = np.array(
        [
            [1.00, 0.30, 0.18, 0.10, 0.05, 0.05, 0.02, 0.04],
            [0.30, 1.00, 0.45, 0.20, 0.12, 0.10, 0.08, 0.10],
            [0.18, 0.45, 1.00, 0.55, 0.20, 0.18, 0.15, 0.16],
            [0.10, 0.20, 0.55, 1.00, 0.30, 0.28, 0.18, 0.20],
            [0.05, 0.12, 0.20, 0.30, 1.00, 0.72, 0.28, 0.22],
            [0.05, 0.10, 0.18, 0.28, 0.72, 1.00, 0.25, 0.20],
            [0.02, 0.08, 0.15, 0.18, 0.28, 0.25, 1.00, 0.35],
            [0.04, 0.10, 0.16, 0.20, 0.22, 0.20, 0.35, 1.00],
        ],
        dtype=float,
    )
    sector_covariance = np.outer(sector_vols, sector_vols) * correlation
    sector_factors = rng.multivariate_normal(sector_mean_returns, sector_covariance, size=n_scenarios)

    asset_records: list[AssetRecord] = []
    asset_return_columns: list[np.ndarray] = []

    counter = 0
    for sector_index, sector in enumerate(sector_names):
        for local_index in range(counts[sector_index]):
            counter += 1
            asset_id = f"A{counter:03d}"
            asset_name = f"{sector} Asset {local_index + 1}"

            # Assets inherit the sector factor plus a small idiosyncratic term.
            idiosyncratic_scale = sector_vols[sector_index] * 0.35
            idiosyncratic_noise = rng.normal(0.0, idiosyncratic_scale, size=n_scenarios)
            asset_return_series = sector_factors[:, sector_index] + idiosyncratic_noise

            asset_records.append(
                AssetRecord(
                    asset_id=asset_id,
                    asset_name=asset_name,
                    sector=sector,
                    expected_return=float(np.mean(asset_return_series)),
                    capital_charge=float(np.clip(sector_capital[sector_index] + rng.normal(0.0, 0.03), 0.01, 1.0)),
                    liquidity=float(np.clip(sector_liquidity[sector_index] + rng.normal(0.0, 0.05), 0.01, 1.0)),
                )
            )
            asset_return_columns.append(asset_return_series)

    asset_return_matrix = np.asarray(asset_return_columns, dtype=float).T
    asset_covariance = np.cov(asset_return_matrix, rowvar=False, ddof=1)

    return RawDataset(
        asset_records=asset_records,
        scenario_labels=[f"scenario_{i:03d}" for i in range(n_scenarios)],
        asset_ids_in_scenarios=[asset.asset_id for asset in asset_records],
        scenario_returns=asset_return_matrix,
        scenario_probabilities=None,
        asset_covariance=asset_covariance,
        workbook_summary={"source": "synthetic"},
    )


def _weighted_mean_and_cov(matrix: np.ndarray, probabilities: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance using either equal-weight or user-supplied
    scenario probabilities.
    """

    if probabilities is None:
        mean = np.mean(matrix, axis=0)
        cov = np.cov(matrix, rowvar=False, ddof=1)
        return mean, np.asarray(cov, dtype=float)

    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / np.sum(probabilities)
    mean = probabilities @ matrix
    centered = matrix - mean

    # This is the probability-weighted analogue of the sample covariance. The
    # denominator adjustment keeps the estimate from becoming too optimistic.
    denom = 1.0 - np.sum(probabilities**2)
    if denom <= 0.0:
        denom = 1.0
    cov = (centered * probabilities[:, None]).T @ centered / denom
    return mean, cov


def _min_max_normalize(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a metric to [0, 1]. If all values are equal, return zeros so the
    penalty stays well-defined instead of dividing by zero.
    """

    minimum = float(np.min(vector))
    maximum = float(np.max(vector))
    spread = maximum - minimum
    if spread <= 1e-12:
        return np.zeros_like(vector, dtype=float)
    return (vector - minimum) / spread


def aggregate_to_sectors(dataset: RawDataset) -> AggregatedDataset:
    """
    Convert asset-level data into the exact 8-sector representation described in
    the research report.
    """

    asset_ids = [asset.asset_id for asset in dataset.asset_records]
    if list(dataset.asset_ids_in_scenarios) != asset_ids:
        raise ValueError(
            "The scenario-return matrix must already be aligned with the asset "
            "records. If you are loading from Excel, the loader should do that "
            "alignment for you."
        )

    sector_to_indices: dict[str, list[int]] = {sector: [] for sector in CANONICAL_SECTORS}
    for asset_index, asset in enumerate(dataset.asset_records):
        sector_to_indices[asset.sector].append(asset_index)

    missing = [sector for sector, indices in sector_to_indices.items() if not indices]
    if missing:
        raise ValueError(
            f"Every one of the 8 sectors needs at least one asset. Missing: {missing}"
        )

    aggregation_matrix = np.zeros((len(CANONICAL_SECTORS), len(dataset.asset_records)), dtype=float)
    for sector_index, sector in enumerate(CANONICAL_SECTORS):
        indices = sector_to_indices[sector]
        aggregation_matrix[sector_index, indices] = 1.0 / len(indices)

    sector_return_matrix = dataset.scenario_returns @ aggregation_matrix.T
    mu, sigma = _weighted_mean_and_cov(sector_return_matrix, dataset.scenario_probabilities)

    capital = np.array(
        [
            np.mean([dataset.asset_records[idx].capital_charge or 0.0 for idx in sector_to_indices[sector]])
            for sector in CANONICAL_SECTORS
        ],
        dtype=float,
    )
    liquidity = np.array(
        [
            np.mean([dataset.asset_records[idx].liquidity or 0.0 for idx in sector_to_indices[sector]])
            for sector in CANONICAL_SECTORS
        ],
        dtype=float,
    )

    normalized_capital = _min_max_normalize(capital)
    normalized_liquidity = _min_max_normalize(liquidity)
    normalized_illiquidity = 1.0 - normalized_liquidity

    asset_expected_returns = None
    if any(asset.expected_return is not None for asset in dataset.asset_records):
        asset_expected_returns = np.array(
            [asset.expected_return or 0.0 for asset in dataset.asset_records],
            dtype=float,
        )

    validation_summary: dict[str, float | None] = {
        "scenario_vs_asset_return_rmse": None,
        "scenario_vs_covariance_frobenius": None,
    }

    if asset_expected_returns is not None:
        sector_expected_from_assets = aggregation_matrix @ asset_expected_returns
        validation_summary["scenario_vs_asset_return_rmse"] = float(
            np.sqrt(np.mean((mu - sector_expected_from_assets) ** 2))
        )

    if dataset.asset_covariance is not None:
        aggregated_covariance_from_asset_sheet = aggregation_matrix @ dataset.asset_covariance @ aggregation_matrix.T
        validation_summary["scenario_vs_covariance_frobenius"] = float(
            np.linalg.norm(sigma - aggregated_covariance_from_asset_sheet, ord="fro")
        )

    sector_to_asset_ids = {
        sector: [dataset.asset_records[index].asset_id for index in sector_to_indices[sector]]
        for sector in CANONICAL_SECTORS
    }

    return AggregatedDataset(
        sector_names=CANONICAL_SECTORS,
        sector_to_asset_ids=sector_to_asset_ids,
        asset_ids=asset_ids,
        aggregation_matrix=aggregation_matrix,
        asset_return_matrix=dataset.scenario_returns,
        sector_return_matrix=sector_return_matrix,
        scenario_labels=dataset.scenario_labels,
        scenario_probabilities=dataset.scenario_probabilities,
        mu=np.asarray(mu, dtype=float),
        sigma=np.asarray(sigma, dtype=float),
        capital=capital,
        liquidity=liquidity,
        normalized_capital=normalized_capital,
        normalized_illiquidity=normalized_illiquidity,
        asset_expected_returns=asset_expected_returns,
        validation_summary=validation_summary,
    )


def build_qubo_model(
    aggregated: AggregatedDataset,
    hyperparameters: ModelHyperparameters,
    version: str = "both",
) -> QuboModel:
    """
    Build one upper-triangular QUBO matrix and its Ising equivalent.

    `version` lets us compare the base Markowitz-only model against the
    insurer-aware extensions requested in the report.
    """

    include_capital = version in {"capital", "both"}
    include_liquidity = version in {"liquidity", "both"}

    budget = hyperparameters.budget
    a = hyperparameters.risk_aversion / (budget**2)
    b = 1.0 / budget
    g = (hyperparameters.capital_penalty / budget) if include_capital else 0.0
    e = (hyperparameters.illiquidity_penalty / budget) if include_liquidity else 0.0
    lam = hyperparameters.penalty_lambda

    n = len(aggregated.sector_names)
    Q = np.zeros((n, n), dtype=float)

    for i in range(n):
        Q[i, i] = (
            a * aggregated.sigma[i, i]
            - b * aggregated.mu[i]
            + lam * (1.0 - 2.0 * budget)
            + g * aggregated.normalized_capital[i]
            + e * aggregated.normalized_illiquidity[i]
        )

    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = 2.0 * a * aggregated.sigma[i, j] + 2.0 * lam

    scale_alpha = float(np.max(np.abs(Q))) if np.max(np.abs(Q)) > 0 else 1.0
    Q_scaled = Q / scale_alpha
    h, J, offset = qubo_to_ising(Q_scaled)

    return QuboModel(
        version=version,
        hyperparameters=hyperparameters,
        Q_unscaled=Q,
        Q_scaled=Q_scaled,
        scale_alpha=scale_alpha,
        h=h,
        J=J,
        offset=offset,
        included_capital_penalty=include_capital,
        included_liquidity_penalty=include_liquidity,
    )


def qubo_to_ising(Q_upper: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convert an upper-triangular QUBO matrix into Ising fields and couplings.
    """

    n = Q_upper.shape[0]
    h = np.zeros(n, dtype=float)
    J = np.zeros((n, n), dtype=float)
    offset = 0.0

    for i in range(n):
        qii = Q_upper[i, i]
        offset += qii / 2.0
        h[i] += -qii / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            qij = Q_upper[i, j]
            offset += qij / 4.0
            h[i] += -qij / 4.0
            h[j] += -qij / 4.0
            J[i, j] += qij / 4.0
            J[j, i] += qij / 4.0

    return h, J, float(offset)


def qubo_energy_upper(Q_upper: np.ndarray, x: np.ndarray) -> float:
    """
    Evaluate an upper-triangular QUBO directly without needing to densify the
    lower triangle.
    """

    x = np.asarray(x, dtype=float)
    diagonal = float(np.sum(np.diag(Q_upper) * x))
    quadratic = 0.0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            quadratic += Q_upper[i, j] * x[i] * x[j]
    return diagonal + quadratic


def spins_from_bits(bits: np.ndarray) -> np.ndarray:
    """
    Convert computational basis bits {0,1} to Ising spins {+1,-1}.
    """

    bits = np.asarray(bits, dtype=int)
    return 1 - 2 * bits


def ising_energy(h: np.ndarray, J: np.ndarray, spins: np.ndarray, offset: float = 0.0) -> float:
    """
    Evaluate the Ising form. Helpful for sanity checks.
    """

    spins = np.asarray(spins, dtype=float)
    pair_energy = 0.0
    for i in range(len(spins)):
        for j in range(i + 1, len(spins)):
            pair_energy += J[i, j] * spins[i] * spins[j]
    return float(offset + np.dot(h, spins) + pair_energy)

