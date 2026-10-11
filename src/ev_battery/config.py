"""Explicit project paths and proposed research settings; no import-time I/O.

Capacity/EOL defaults are planning choices, not facts inferred from unseen files.
Check the selected batteries' documentation before constructing any targets.
"""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

OFFICIAL_REPOSITORY_URL = (
    "https://www.nasa.gov/intelligent-systems-division/"
    "discovery-and-systems-health/pcoe/pcoe-data-set-repository/"
)
OFFICIAL_ARCHIVE_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"


def default_project_root() -> Path:
    """Locate the source checkout; installed wheels use the current directory."""
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").is_file() else Path.cwd()


@dataclass(frozen=True)
class ProjectPaths:
    """All paths derive from one explicitly overridable project root."""

    root: Path = field(default_factory=default_project_root)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def results(self) -> Path:
        return self.reports / "results"

    def to_dict(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in (
            "root", "raw", "processed", "models", "reports", "figures", "results"
        )}


@dataclass(frozen=True)
class ModelParameters:
    """Untrained starting parameters, to be tuned using training folds only."""

    random_forest_estimators: int = 300
    random_forest_min_samples_leaf: int = 2
    svr_c: float = 10.0
    svr_epsilon: float = 0.1
    xgboost_estimators: int = 300
    xgboost_max_depth: int = 3
    xgboost_learning_rate: float = 0.05
    lstm_hidden_size: int = 32
    lstm_sequence_length: int = 20
    lstm_batch_size: int = 32
    lstm_epochs: int = 100
    lstm_learning_rate: float = 0.001


@dataclass(frozen=True)
class ProjectConfig:
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    random_seed: int = 42
    nominal_capacity_ah: float = 2.0
    eol_soh_fraction: float = 0.70
    model_parameters: ModelParameters = field(default_factory=ModelParameters)

    def __post_init__(self) -> None:
        if not math.isfinite(self.nominal_capacity_ah) or self.nominal_capacity_ah <= 0:
            raise ValueError("nominal_capacity_ah must be finite and greater than zero.")
        if not math.isfinite(self.eol_soh_fraction) or not 0 < self.eol_soh_fraction < 1:
            raise ValueError("eol_soh_fraction must be a fraction strictly between 0 and 1.")
        if type(self.random_seed) is not int or not 0 <= self.random_seed < 2**32:
            raise ValueError("random_seed must be an integer in [0, 2**32).")

    @property
    def eol_capacity_ah(self) -> float:
        return self.nominal_capacity_ah * self.eol_soh_fraction

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["paths"] = self.paths.to_dict()
        result["eol_capacity_ah"] = self.eol_capacity_ah
        result["settings_status"] = "proposed; not validated against selected batteries"
        return result
