from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MEGA_CSV = "/Users/archon/classes/ASTR_502/workstation/data/raw/catalogs/ASTR502_Mega_Target_List.csv"
DEFAULT_PHOT_CSV = "/Users/archon/classes/ASTR_502/workstation/data/raw/catalogs/ASTR502_Master_Photometry_List.csv"

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
                "Catalogs not loaded. Call load_catalogs(...) first. "
                f"Defaults are mega='{DEFAULT_MEGA_CSV}' phot='{DEFAULT_PHOT_CSV}'."
            )
        return self.mega_df, self.phot_df
