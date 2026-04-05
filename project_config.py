from __future__ import annotations

"""
Central configuration objects used across the project.

The goal of this file is to keep all "knobs" in one place so the rest of the
code can stay focused on the actual math and experiment logic.
"""

from dataclasses import dataclass, field
from pathlib import Path


# We keep the sector order fixed everywhere because the report explicitly asks
# for a documented sector-to-qubit ordering. Once we choose this order we
# should never silently change it later, otherwise the Q matrix, plots, and
# bitstring interpretations all become inconsistent.
CANONICAL_SECTORS: tuple[str, ...] = (
    "Cash",
    "Gov Bonds",
    "IG Credit",
    "HY Credit",
    "Equities US",
    "Equities Intl",
    "Real Estate",
    "Infrastructure",
)


# These aliases make the loader more forgiving when reading a user workbook.
# Real spreadsheets often use slightly different labels, so we normalize them.
SECTOR_ALIASES: dict[str, str] = {
    "cash": "Cash",
    "cash instruments": "Cash",
    "government bonds": "Gov Bonds",
    "gov bonds": "Gov Bonds",
    "gov bond": "Gov Bonds",
    "sovereign bonds": "Gov Bonds",
    "ig credit": "IG Credit",
    "investment grade": "IG Credit",
    "investment grade credit": "IG Credit",
    "high yield": "HY Credit",
    "hy credit": "HY Credit",
    "high yield credit": "HY Credit",
    "equities us": "Equities US",
    "us equities": "Equities US",
    "equities usa": "Equities US",
    "equities intl": "Equities Intl",
    "international equities": "Equities Intl",
    "intl equities": "Equities Intl",
    "equities international": "Equities Intl",
    "real estate": "Real Estate",
    "property": "Real Estate",
    "infrastructure": "Infrastructure",
    "infra": "Infrastructure",
}


# Column detection aliases. The workbook is not present in the repository, so
# the loader has to be defensive and accept common naming variants.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "asset_id": (
        "asset_id",
        "asset id",
        "id",
        "ticker",
        "symbol",
        "asset_code",
        "asset code",
    ),
    "asset_name": (
        "asset_name",
        "asset name",
        "name",
        "description",
        "asset",
    ),
    "sector": (
        "sector",
        "asset_class",
        "asset class",
        "class",
        "category",
        "group",
    ),
    "expected_return": (
        "exp_return",
        "expected_return",
        "expected return",
        "mean_return",
        "return",
        "mu",
    ),
    "capital_charge": (
        "capital_charge",
        "capital charge",
        "capital",
        "solvency_charge",
        "scr",
        "capital_requirement",
    ),
    "liquidity": (
        "liquidity",
        "liquidity_score",
        "liquidity score",
        "liquidity_bucket",
        "liquid_score",
    ),
    "scenario_label": (
        "scenario",
        "scenario_id",
        "scenario id",
        "name",
        "label",
    ),
    "scenario_probability": (
        "probability",
        "prob",
        "weight",
        "scenario_probability",
        "scenario probability",
    ),
}


@dataclass(slots=True)
class ModelHyperparameters:
    """
    Parameters that directly define one QUBO model instance.
    """

    budget: int = 4
    risk_aversion: float = 0.5
    penalty_lambda: float = 8.0
    capital_penalty: float = 0.25
    illiquidity_penalty: float = 0.25


@dataclass(slots=True)
class SweepConfig:
    """
    Small defaults for interactive use, with the same value ranges proposed in
    the research report. `full_grid=True` in the main script turns these into a
    larger cartesian product.
    """

    budgets: tuple[int, ...] = (3, 4, 5)
    risk_aversion_values: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0)
    lambda_values: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0)
    capital_penalty_values: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    illiquidity_penalty_values: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    qaoa_depths: tuple[int, ...] = (1, 2, 3)
    noise_strengths: tuple[float, ...] = (0.0, 1.0, 2.0)
    readout_error_values: tuple[float, ...] = (0.0, 0.01, 0.03)
    seeds: tuple[int, ...] = (7, 11, 19)


@dataclass(slots=True)
class ExecutionConfig:
    """
    Execution options that do not change the mathematical model itself.
    """

    shots: int = 2048
    qaoa_optimizer_restarts: int = 12
    qaoa_coordinate_descent_rounds: int = 5
    qaoa_initial_step: float = 0.6
    use_synthetic_data_if_missing: bool = True
    synthetic_seed: int = 123
    synthetic_assets: int = 50
    synthetic_scenarios: int = 250
    var_alpha: float = 0.95
    include_model_versions: tuple[str, ...] = ("base", "capital", "liquidity", "both")
    include_connectivity_modes: tuple[str, ...] = ("naive", "auto", "hand", "pruned")
    include_noise_models: tuple[str, ...] = ("none", "one_zone", "two_zone", "depolarizing")


@dataclass(slots=True)
class RunConfig:
    """
    High-level project configuration used by the CLI entrypoint.
    """

    workbook_path: Path | None = None
    output_dir: Path = field(default_factory=lambda: Path("artifacts"))
    assets_sheet_name: str | None = None
    scenarios_sheet_name: str | None = None
    covariance_sheet_name: str | None = None
    hyperparameters: ModelHyperparameters = field(default_factory=ModelHyperparameters)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def ensure_output_dirs(self) -> dict[str, Path]:
        """
        Create a simple output layout that is easy to browse in PyCharm.
        """

        folders = {
            "root": self.output_dir,
            "results": self.output_dir / "results",
            "figures": self.output_dir / "figures",
            "artifacts": self.output_dir / "model_artifacts",
            "kernels": self.output_dir / "bloqade_kernels",
        }
        for folder in folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        return folders
