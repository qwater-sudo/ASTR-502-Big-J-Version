from __future__ import annotations

import glob
import logging
from pathlib import Path
import re
import time

import pandas as pd

logger = logging.getLogger(__name__)

REQUESTED_BANDS = ("G", "BP", "RP", "J", "H", "K", "W1", "W2", "W3", "W4", "g", "r", "i", "z")

BAND_COLUMN_CANDIDATES = {
    "G": ("G_mag", "Gaia_G_EDR3", "G"),
    "BP": ("BP_mag", "Gaia_BP_EDR3", "BP"),
    "RP": ("RP_mag", "Gaia_RP_EDR3", "RP"),
    "J": ("J_mag", "2MASS_J", "J"),
    "H": ("H_mag", "2MASS_H", "H"),
    "K": ("K_mag", "2MASS_Ks", "K"),
    "W1": ("W1_mag", "WISE_W1", "W1"),
    "W2": ("W2_mag", "WISE_W2", "W2"),
    "W3": ("W3_mag", "WISE_W3", "W3"),
    "W4": ("W4_mag", "WISE_W4", "W4"),
    "g": ("g_mag", "SDSS_g", "g"),
    "r": ("r_mag", "SDSS_r", "r"),
    "i": ("i_mag", "SDSS_i", "i"),
    "z": ("z_mag", "SDSS_z", "z"),
}


class LoggingUtils:
    @staticmethod
    def configure_debug_logging(log_dir: str | Path = "outputs/logs") -> Path:
        logs_path = Path(log_dir)
        logs_path.mkdir(parents=True, exist_ok=True)
        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = logs_path / f"interpolate_{run_stamp}.log"

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="a", encoding="utf-8"),
                logging.StreamHandler(),
            ],
            force=True,
        )
        logger.debug("Debug logging initialized at %s", log_file)
        return log_file


class IsochroneUtils:
    @staticmethod
    def discover_spot_files(pattern: str = "data/raw/isochrones/SPOTS/isos/*.isoc") -> list[str]:
        return sorted(glob.glob(pattern))

    @staticmethod
    def extract_metallicity_from_path(file_path: str | Path) -> float:
        stem = Path(file_path).stem.lower()
        match = re.search(r"f([pm]?)(\d+)", stem)
        if not match:
            return float("nan")

        sign = match.group(1)
        digits = match.group(2)
        value = float(digits) / 100.0
        if sign == "m":
            value *= -1.0
        return value

    @staticmethod
    def select_rows(section: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        selected = section.copy()
        if "phase" in selected.columns:
            selected = selected[selected["phase"].isin([-1, 0, 2, 3])].copy()

        mass_col = "Mass" if "Mass" in selected.columns else ("mass" if "mass" in selected.columns else None)
        if mass_col is None:
            raise ValueError("SPOT section is missing a mass column (Mass or mass)")

        selected = selected.sort_values(mass_col).reset_index(drop=True)
        return selected, mass_col

    @staticmethod
    def find_band_column(section: pd.DataFrame, band: str) -> str | None:
        for candidate in BAND_COLUMN_CANDIDATES.get(band, ()):  # exact matches first
            if candidate in section.columns:
                return candidate

        lower = {c.lower(): c for c in section.columns}
        for candidate in BAND_COLUMN_CANDIDATES.get(band, ()):  # pragma: no branch
            for low_name, original in lower.items():
                if candidate.lower() in low_name:
                    return original
        return None
