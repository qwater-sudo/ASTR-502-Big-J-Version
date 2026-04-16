from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(*parts: str) -> Path:
    """Resolve a path relative to the repository root."""
    return repo_root().joinpath(*parts)


DATA_RAW_DIR = resolve_repo_path("data", "raw")
CATALOGS_DIR = DATA_RAW_DIR / "catalogs"
SPOTS_ISOS_DIR = DATA_RAW_DIR / "isochrones" / "SPOTS" / "isos"

OUTPUTS_DIR = resolve_repo_path("outputs")
OUTPUT_LOGS_DIR = OUTPUTS_DIR / "logs"
OUTPUT_RESULTS_DIR = OUTPUTS_DIR / "results"
OUTPUT_FIGS_DIR = OUTPUTS_DIR / "figs"

DEFAULT_MEGA_CSV_PATH = CATALOGS_DIR / "ASTR502_Mega_Target_List.csv"
DEFAULT_PHOT_CSV_PATH = CATALOGS_DIR / "ASTR502_Master_Photometry_List.csv"

K2_AGES = CATALOGS_DIR / "k2_star_ages.csv"
KEPLER_AGES = CATALOGS_DIR / "kepler_star_ages.csv"
