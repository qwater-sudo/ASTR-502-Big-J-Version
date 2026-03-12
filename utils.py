from __future__ import annotations

import glob
import logging
from pathlib import Path
import re
import time
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUESTED_BANDS = ("G", "BP", "RP", "J", "H", "K", "W1", "W2", "W3", "W4", "g", "r", "i", "z")
DEFAULT_MEGA_CSV = "ASTR502_Mega_Target_List.csv"
DEFAULT_PHOT_CSV = "ASTR502_Master_Photometry_List.csv"

OBS_MAP = {
    "G": "gaia_Gmag",
    "BP": "gaia_BPmag",
    "RP": "gaia_RPmag",
    "J": "Jmag",
    "H": "Hmag",
    "K": "Kmag",
    "W1": "w1mag",
    "W2": "w2mag",
    "W3": "w3mag",
    "W4": "w4mag",
    "g": "gmag",
    "r": "rmag",
    "i": "imag",
    "z": "zmag",
}

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
    def configure_debug_logging(log_dir: str | Path = "logs") -> Path:
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


class CatalogUtils:
    @staticmethod
    def apparent_to_absolute(m_app: float, distance_pc: float) -> float:
        return float(m_app - 5.0 * np.log10(distance_pc / 10.0))

    @staticmethod
    def get_star_rows(hostname: str, mega_df: pd.DataFrame, phot_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        m = mega_df[mega_df["hostname"] == hostname]
        p = phot_df[phot_df["hostname"] == hostname]
        if len(m) == 0:
            raise KeyError(f"{hostname} not found in Mega_Target_List")
        if len(p) == 0:
            raise KeyError(f"{hostname} not found in Master_Photometry_List")
        return m.iloc[0], p.iloc[0]

    @staticmethod
    def get_star_obs_abs(
        hostname: str,
        mega_df: pd.DataFrame,
        phot_df: pd.DataFrame,
    ) -> tuple[dict[str, float], float]:
        mrow, prow = CatalogUtils.get_star_rows(hostname, mega_df=mega_df, phot_df=phot_df)

        distance_pc = float(mrow["bj_dist_pc"])
        if not np.isfinite(distance_pc) or distance_pc <= 0:
            raise ValueError(f"{hostname}: invalid bj_dist_pc={distance_pc}")

        obs_abs: dict[str, float] = {}
        for band, col in OBS_MAP.items():
            if col in prow.index and np.isfinite(prow[col]):
                obs_abs[band] = CatalogUtils.apparent_to_absolute(float(prow[col]), distance_pc)

        if len(obs_abs) < 3:
            raise ValueError(f"{hostname}: only {len(obs_abs)} usable bands; need >= 3 for a stable fit")

        return obs_abs, distance_pc

    @staticmethod
    def get_param_prior(hostname: str, mega_df: pd.DataFrame, phot_df: pd.DataFrame, fallback_sigma: float = 0.25) -> dict[str, float]:
        mrow, _ = CatalogUtils.get_star_rows(hostname, mega_df=mega_df, phot_df=phot_df)

        mass0 = float(mrow["st_mass"]) if np.isfinite(mrow["st_mass"]) else np.nan
        age0 = float(mrow["st_age"]) if np.isfinite(mrow["st_age"]) else np.nan
        feh0 = float(mrow["st_met"]) if np.isfinite(mrow["st_met"]) else np.nan

        if (
            "st_ageerr1" in mrow.index
            and "st_ageerr2" in mrow.index
            and np.isfinite(mrow["st_ageerr1"])
            and np.isfinite(mrow["st_ageerr2"])
        ):
            sig_age_hi = float(mrow["st_ageerr1"])
            sig_age_lo = abs(float(mrow["st_ageerr2"]))
            if sig_age_hi <= 0:
                sig_age_hi = fallback_sigma
            if sig_age_lo <= 0:
                sig_age_lo = fallback_sigma
        else:
            sig_age_hi = fallback_sigma
            sig_age_lo = fallback_sigma

        return {
            "m0": mass0,
            "a0_gyr": age0,
            "feh0": feh0,
            "sig_m": fallback_sigma,
            "sig_feh": fallback_sigma,
            "sig_age_hi": sig_age_hi,
            "sig_age_lo": sig_age_lo,
        }


class IsochroneUtils:
    @staticmethod
    def discover_spot_files(pattern: str = "isochrones/SPOTS/isos/*.isoc") -> list[str]:
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
        # permissive fallback: substring match
        lower = {c.lower(): c for c in section.columns}
        for candidate in BAND_COLUMN_CANDIDATES.get(band, ()):  # pragma: no branch
            for low_name, original in lower.items():
                if candidate.lower() in low_name:
                    return original
        return None


class CatalogStore:
    def __init__(self) -> None:
        self.mega_df: pd.DataFrame | None = None
        self.phot_df: pd.DataFrame | None = None

    def load_catalogs(self, mega_csv_path: str | Path = DEFAULT_MEGA_CSV, phot_csv_path: str | Path = DEFAULT_PHOT_CSV) -> None:
        self.mega_df = pd.read_csv(mega_csv_path)
        self.phot_df = pd.read_csv(phot_csv_path)

    def ensure_loaded(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.mega_df is None or self.phot_df is None:
            raise RuntimeError(
                "Catalogs not loaded. Call load_catalogs(mega_csv_path='ASTR502_Mega_Target_List.csv', "
                "phot_csv_path='ASTR502_Master_Photometry_List.csv') first."
            )
        return self.mega_df, self.phot_df
