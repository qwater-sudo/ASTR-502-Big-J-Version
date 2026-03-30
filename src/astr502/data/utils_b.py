from __future__ import annotations

import glob
import logging
from pathlib import Path
import re
import time

import pandas as pd

logger = logging.getLogger(__name__)

REQUESTED_BANDS = ("G", "BP", "RP", "J", "H", "K", "W1", "W2", "W3", "W4", "g", "r", "i", "z")

# Each band maps to a priority-ordered list of candidate column names.
# SPOT candidates come first; PARSEC CMD 3.x names follow.
BAND_COLUMN_CANDIDATES = {
    "G":  ("G_mag",   "Gaia_G_EDR3",  "G",    "Gmag"),
    "BP": ("BP_mag",  "Gaia_BP_EDR3", "BP",   "G_BPmag",  "GBPmag",  "BP_mag"),
    "RP": ("RP_mag",  "Gaia_RP_EDR3", "RP",   "G_RPmag",  "GRPmag",  "RP_mag"),
    "J":  ("J_mag",   "2MASS_J",      "J",    "Jmag"),
    "H":  ("H_mag",   "2MASS_H",      "H",    "Hmag"),
    "K":  ("K_mag",   "2MASS_Ks",     "K",    "Ksmag",    "Kmag"),
    "W1": ("W1_mag",  "WISE_W1",      "W1",   "W1mag"),
    "W2": ("W2_mag",  "WISE_W2",      "W2",   "W2mag"),
    "W3": ("W3_mag",  "WISE_W3",      "W3",   "W3mag"),
    "W4": ("W4_mag",  "WISE_W4",      "W4",   "W4mag"),
    "g":  ("g_mag",   "SDSS_g",       "g",    "gmag",     "g_sdss"),
    "r":  ("r_mag",   "SDSS_r",       "r",    "rmag",     "r_sdss"),
    "i":  ("i_mag",   "SDSS_i",       "i",    "imag",     "i_sdss"),
    "z":  ("z_mag",   "SDSS_z",       "z",    "zmag",     "z_sdss"),
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
    # ------------------------------------------------------------------ #
    # File discovery                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def discover_spot_files(
        pattern: str = "/Users/archon/classes/ASTR_502/workstation/data/raw/isochrones/SPOTS/isos/*.isoc",
    ) -> list[str]:
        """Return sorted list of SPOT .isoc files matching *pattern*."""
        return sorted(glob.glob(pattern))

    @staticmethod
    def discover_parsec_files(
        pattern: str = "/Users/archon/classes/ASTR_502/workstation/data/raw/isochrones/PARSEC/*.dat",
    ) -> list[str]:
        """Return sorted list of PARSEC .dat files matching *pattern*."""
        return sorted(glob.glob(pattern))

    # ------------------------------------------------------------------ #
    # Metallicity extraction (SPOT filename convention only)              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def extract_metallicity_from_path(file_path: str | Path) -> float:
        """
        Parse [Fe/H] from a SPOT filename such as ``fm050.isoc`` → -0.50,
        ``fp025.isoc`` → +0.25, ``f000.isoc`` → 0.00.

        Not used for PARSEC (metallicity is read from the MH column directly).
        """
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

    # ------------------------------------------------------------------ #
    # Row selection (shared by both model types)                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def select_rows(section: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """
        Filter to main-sequence / relevant evolutionary phases and locate the
        mass column.  Works for both SPOT and PARSEC DataFrames.

        - SPOT uses a ``phase`` column with integer codes (-1, 0, 2, 3).
        - PARSEC uses a ``label`` column: 0=PMS, 1=MS, 2=SGB, 3=RGB, 4=CHeB.
          We keep labels 0–4 by default.

        Returns
        -------
        (filtered_df, mass_col_name)
        """
        selected = section.copy()

        # Phase / label filtering
        if "phase" in selected.columns:
            selected = selected[selected["phase"].isin([-1, 0, 2, 3])].copy()
        elif "label" in selected.columns:
            selected = selected[selected["label"].isin([0, 1, 2, 3, 4])].copy()

        # Mass column — prefer initial mass (Mini) for PARSEC, current Mass for SPOT
        for candidate in ("Mass", "Mini", "mass"):
            if candidate in selected.columns:
                mass_col = candidate
                break
        else:
            raise ValueError(
                "Section is missing a mass column. "
                f"Columns present: {list(selected.columns)}"
            )

        selected = selected.sort_values(mass_col).reset_index(drop=True)
        return selected, mass_col

    # ------------------------------------------------------------------ #
    # Band column lookup (shared by both model types)                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def find_band_column(section: pd.DataFrame, band: str) -> str | None:
        """
        Return the first matching column name for *band* in *section*, or
        ``None`` if the band is not present.

        Matching proceeds in priority order (exact name first, then
        case-insensitive substring).
        """
        candidates = BAND_COLUMN_CANDIDATES.get(band, ())

        # 1. Exact match
        for candidate in candidates:
            if candidate in section.columns:
                return candidate

        # 2. Case-insensitive substring fallback
        lower_map = {c.lower(): c for c in section.columns}
        for candidate in candidates:
            for low_name, original in lower_map.items():
                if candidate.lower() in low_name:
                    return original

        return None
