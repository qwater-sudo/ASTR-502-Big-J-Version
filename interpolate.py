"""Fit Gaia absolute magnitudes to ezpadova isochrones.

This module provides a production-ready workflow to estimate stellar ages and
metallicities from Gaia absolute magnitudes using ezpadova's built-in
phase-aware interpolation (`QuickInterpolator`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from find_mag import PhotometryMerger

try:
    import ezpadova
    from ezpadova.interpolate import QuickInterpolator
except ImportError as exc:  # pragma: no cover - handled at runtime in environments without ezpadova
    ezpadova = None
    QuickInterpolator = None
    _EZPADOVA_IMPORT_ERROR = exc
else:
    _EZPADOVA_IMPORT_ERROR = None

LOGGER = logging.getLogger(__name__)

REQUIRED_MAG_COLS = ("G_abs", "BP_abs", "RP_abs")
TARGET_ID_COL = "target_id"


@dataclass(frozen=True)
class IsochroneGridSpec:
    """Configuration for ezpadova isochrone grid generation."""

    logage_min: float = 6.0
    logage_max: float = 10.2
    logage_step: float = 0.05
    mh_min: float = -2.0
    mh_max: float = 0.6
    mh_step: float = 0.05
    photsys: str = "gaiaEDR3"

    def logage_grid(self) -> np.ndarray:
        return np.arange(self.logage_min, self.logage_max + 0.5 * self.logage_step, self.logage_step)

    def mh_grid(self) -> np.ndarray:
        return np.arange(self.mh_min, self.mh_max + 0.5 * self.mh_step, self.mh_step)


def _require_ezpadova() -> None:
    if ezpadova is None or QuickInterpolator is None:
        raise ImportError(
            "ezpadova is required for interpolation/fitting but is not available in this environment"
        ) from _EZPADOVA_IMPORT_ERROR


def _validate_target_table(targets: pd.DataFrame) -> None:
    required = {TARGET_ID_COL, *REQUIRED_MAG_COLS}
    missing = required.difference(targets.columns)
    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise ValueError(f"targets is missing required column(s): {missing_sorted}")


def build_isochrone_interpolator(
    grid_spec: IsochroneGridSpec,
    *,
    extra_columns: Optional[Iterable[str]] = None,
) -> tuple[QuickInterpolator, np.ndarray, np.ndarray, list[str]]:
    """Create an ezpadova isochrone grid and `QuickInterpolator`.

    Parameters
    ----------
    grid_spec
        Grid definition for (logAge, MH) sampling and photometric system.
    extra_columns
        Optional additional columns to request from interpolated isochrones.

    Returns
    -------
    interpolator, logage_grid, mh_grid, columns
        Interpolator object, coordinate grids and model columns requested.
    """

    _require_ezpadova()

    logages = grid_spec.logage_grid()
    mhs = grid_spec.mh_grid()

    raw_grid = ezpadova.get_isochrones(
        logage=(float(logages.min()), float(logages.max()), float(grid_spec.logage_step)),
        MH=(float(mhs.min()), float(mhs.max()), float(grid_spec.mh_step)),
        photsys=grid_spec.photsys,
    )

    if isinstance(raw_grid, list):
        iso_df = pd.concat(raw_grid, ignore_index=True)
    elif isinstance(raw_grid, pd.DataFrame):
        iso_df = raw_grid.copy()
    else:
        iso_df = pd.DataFrame(raw_grid)

    interpolator = QuickInterpolator(iso_df)

    model_cols = ["Gmag", "G_BPmag", "G_RPmag", "evol"]
    if extra_columns:
        for col in extra_columns:
            if col not in model_cols:
                model_cols.append(col)

    return interpolator, logages, mhs, model_cols


def prepare_targets_from_csv(
    phot_csv: str,
    dist_csv: str,
    *,
    target_id_col: str = "target_id",
    join_on: Optional[str] = None,
    merge_how: str = "inner",
) -> pd.DataFrame:
    """Build target table with absolute magnitudes using `find_mag.PhotometryMerger`."""

    merger = PhotometryMerger()
    merged = merger.join_photometry_and_distances(phot_csv, dist_csv, on=join_on, how=merge_how)

    if target_id_col not in merged.columns:
        # Fall back to a generated identifier if caller did not provide one.
        merged[target_id_col] = merged.index.astype(str)

    targets = merged.rename(columns={target_id_col: TARGET_ID_COL}).copy()
    return targets


def _to_numeric_with_default(series: pd.Series, default: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.fillna(float(default))


def _coerce_interp_result(model: Any, expected_columns: list[str]) -> Optional[pd.DataFrame]:
    if isinstance(model, pd.DataFrame):
        model_df = model.copy()
    elif isinstance(model, dict):
        model_df = pd.DataFrame(model)
    else:
        model_df = pd.DataFrame(model)

    if model_df.empty:
        return None

    for col in expected_columns:
        if col not in model_df.columns:
            return None

    return model_df


def _build_model_cache(
    interpolator: QuickInterpolator,
    logage_grid: np.ndarray,
    mh_grid: np.ndarray,
    model_cols: list[str],
) -> dict[tuple[float, float], Optional[dict[str, np.ndarray]]]:
    cache: dict[tuple[float, float], Optional[dict[str, np.ndarray]]] = {}

    for logage in logage_grid:
        for mh in mh_grid:
            key = (float(logage), float(mh))
            try:
                raw_model = interpolator(float(logage), float(mh), what=model_cols)
            except Exception:
                cache[key] = None
                continue

            model_df = _coerce_interp_result(raw_model, model_cols)
            if model_df is None:
                cache[key] = None
                continue

            for mag_col in ("Gmag", "G_BPmag", "G_RPmag"):
                model_df[mag_col] = pd.to_numeric(model_df[mag_col], errors="coerce")

            finite_rows = np.isfinite(model_df[["Gmag", "G_BPmag", "G_RPmag"]]).all(axis=1)
            if finite_rows.sum() == 0:
                cache[key] = None
                continue

            model_df = model_df.loc[finite_rows].reset_index(drop=True)

            cache[key] = {
                "Gmag": model_df["Gmag"].to_numpy(dtype=float),
                "G_BPmag": model_df["G_BPmag"].to_numpy(dtype=float),
                "G_RPmag": model_df["G_RPmag"].to_numpy(dtype=float),
                "phase": np.arange(len(model_df), dtype=int),
            }

    return cache


def _fit_single_star(
    star_row: pd.Series,
    model_cache: dict[tuple[float, float], Optional[dict[str, np.ndarray]]],
    logage_grid: np.ndarray,
    mh_grid: np.ndarray,
    *,
    delta_chi2_1sig_joint: float,
    store_surface: bool,
) -> tuple[dict[str, Any], Optional[np.ndarray]]:
    target_id = star_row[TARGET_ID_COL]

    obs_g = float(star_row["G_abs"])
    obs_bp = float(star_row["BP_abs"])
    obs_rp = float(star_row["RP_abs"])

    e_g = float(star_row["e_G_abs"])
    e_bp = float(star_row["e_BP_abs"])
    e_rp = float(star_row["e_RP_abs"])

    chi2_surface = np.full((len(logage_grid), len(mh_grid)), np.nan, dtype=float)
    phase_surface = np.full((len(logage_grid), len(mh_grid)), -1, dtype=int)

    best_chi2 = np.inf
    best_phase = np.nan
    best_logage = np.nan
    best_mh = np.nan

    n_grid_evaluated = 0

    for i, logage in enumerate(logage_grid):
        for j, mh in enumerate(mh_grid):
            model = model_cache.get((float(logage), float(mh)))
            if model is None:
                continue

            chi2_points = (
                ((obs_g - model["Gmag"]) / e_g) ** 2
                + ((obs_bp - model["G_BPmag"]) / e_bp) ** 2
                + ((obs_rp - model["G_RPmag"]) / e_rp) ** 2
            )

            if chi2_points.size == 0 or not np.isfinite(chi2_points).any():
                continue

            k = int(np.nanargmin(chi2_points))
            chi2_min = float(chi2_points[k])
            if not np.isfinite(chi2_min):
                continue

            chi2_surface[i, j] = chi2_min
            phase_surface[i, j] = int(model["phase"][k])
            n_grid_evaluated += 1

            if chi2_min < best_chi2:
                best_chi2 = chi2_min
                best_logage = float(logage)
                best_mh = float(mh)
                best_phase = int(model["phase"][k])

    if not np.isfinite(best_chi2):
        row = {
            "target_id": target_id,
            "best_logAge": np.nan,
            "best_age_yr": np.nan,
            "best_MH": np.nan,
            "best_chi2": np.nan,
            "best_phase_index": np.nan,
            "n_grid_evaluated": int(n_grid_evaluated),
            "logAge_lo_1sig": np.nan,
            "logAge_hi_1sig": np.nan,
            "MH_lo_1sig": np.nan,
            "MH_hi_1sig": np.nan,
            "fit_status": "no_valid_iso",
        }
        return row, (chi2_surface if store_surface else None)

    in_1sig = np.isfinite(chi2_surface) & (chi2_surface <= best_chi2 + float(delta_chi2_1sig_joint))
    if np.any(in_1sig):
        age_idx, mh_idx = np.where(in_1sig)
        logage_lo = float(logage_grid[np.min(age_idx)])
        logage_hi = float(logage_grid[np.max(age_idx)])
        mh_lo = float(mh_grid[np.min(mh_idx)])
        mh_hi = float(mh_grid[np.max(mh_idx)])
    else:
        logage_lo = np.nan
        logage_hi = np.nan
        mh_lo = np.nan
        mh_hi = np.nan

    row = {
        "target_id": target_id,
        "best_logAge": best_logage,
        "best_age_yr": float(10.0 ** best_logage),
        "best_MH": best_mh,
        "best_chi2": best_chi2,
        "best_phase_index": best_phase,
        "n_grid_evaluated": int(n_grid_evaluated),
        "logAge_lo_1sig": logage_lo,
        "logAge_hi_1sig": logage_hi,
        "MH_lo_1sig": mh_lo,
        "MH_hi_1sig": mh_hi,
        "fit_status": "ok",
    }
    return row, (chi2_surface if store_surface else None)


def fit_targets_to_isochrones(
    targets: pd.DataFrame,
    *,
    logage_min: float = 6.0,
    logage_max: float = 10.2,
    logage_step: float = 0.05,
    mh_min: float = -2.0,
    mh_max: float = 0.6,
    mh_step: float = 0.05,
    photsys: str = "gaiaEDR3",
    default_e_g: float = 0.05,
    default_e_bp: float = 0.08,
    default_e_rp: float = 0.08,
    delta_chi2_1sig_joint: float = 2.30,
    store_surfaces: bool = False,
    log_level: int = logging.INFO,
) -> tuple[pd.DataFrame, Optional[dict[str, Any]]]:
    """Fit each target to ezpadova isochrones using QuickInterpolator.

    Parameters
    ----------
    targets
        Input table with at least: `target_id`, `G_abs`, `BP_abs`, `RP_abs`.
        Optional errors: `e_G_abs`, `e_BP_abs`, `e_RP_abs`.

    Returns
    -------
    result_df, diagnostics
        Result table with one row per target and optional diagnostic surfaces.
    """

    _require_ezpadova()
    _validate_target_table(targets)

    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")

    working = targets.copy()
    for col in REQUIRED_MAG_COLS:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    if "e_G_abs" not in working.columns:
        working["e_G_abs"] = float(default_e_g)
    if "e_BP_abs" not in working.columns:
        working["e_BP_abs"] = float(default_e_bp)
    if "e_RP_abs" not in working.columns:
        working["e_RP_abs"] = float(default_e_rp)

    working["e_G_abs"] = _to_numeric_with_default(working["e_G_abs"], default_e_g)
    working["e_BP_abs"] = _to_numeric_with_default(working["e_BP_abs"], default_e_bp)
    working["e_RP_abs"] = _to_numeric_with_default(working["e_RP_abs"], default_e_rp)

    for err_col, default_val in (("e_G_abs", default_e_g), ("e_BP_abs", default_e_bp), ("e_RP_abs", default_e_rp)):
        non_positive = working[err_col] <= 0
        if non_positive.any():
            working.loc[non_positive, err_col] = float(default_val)

    valid_mask = working[list(REQUIRED_MAG_COLS)].notna().all(axis=1)
    valid_targets = working.loc[valid_mask].copy()
    invalid_targets = working.loc[~valid_mask].copy()

    grid_spec = IsochroneGridSpec(
        logage_min=logage_min,
        logage_max=logage_max,
        logage_step=logage_step,
        mh_min=mh_min,
        mh_max=mh_max,
        mh_step=mh_step,
        photsys=photsys,
    )

    interpolator, logage_grid, mh_grid, model_cols = build_isochrone_interpolator(grid_spec)
    model_cache = _build_model_cache(interpolator, logage_grid, mh_grid, model_cols)

    results: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "logAge_grid": logage_grid.copy(),
        "MH_grid": mh_grid.copy(),
        "surfaces": {},
    }

    for _, row in invalid_targets.iterrows():
        results.append(
            {
                "target_id": row[TARGET_ID_COL],
                "best_logAge": np.nan,
                "best_age_yr": np.nan,
                "best_MH": np.nan,
                "best_chi2": np.nan,
                "best_phase_index": np.nan,
                "n_grid_evaluated": 0,
                "logAge_lo_1sig": np.nan,
                "logAge_hi_1sig": np.nan,
                "MH_lo_1sig": np.nan,
                "MH_hi_1sig": np.nan,
                "fit_status": "insufficient_data",
            }
        )

    for _, row in valid_targets.iterrows():
        fitted, surface = _fit_single_star(
            row,
            model_cache,
            logage_grid,
            mh_grid,
            delta_chi2_1sig_joint=delta_chi2_1sig_joint,
            store_surface=store_surfaces,
        )
        results.append(fitted)
        if store_surfaces and surface is not None:
            diagnostics["surfaces"][str(row[TARGET_ID_COL])] = surface

    result_df = pd.DataFrame(results)

    status_counts = result_df["fit_status"].value_counts(dropna=False).to_dict() if not result_df.empty else {}
    LOGGER.info(
        "Finished fitting %d targets. Status counts: %s",
        len(result_df),
        status_counts,
    )

    if not store_surfaces:
        diagnostics = None

    return result_df, diagnostics


def _demo_synthetic_targets() -> pd.DataFrame:
    """Create a tiny synthetic absolute-magnitude table (no external files)."""

    return pd.DataFrame(
        {
            "target_id": ["star_1", "star_2", "star_3", "star_4"],
            "G_abs": [4.8, 2.1, 6.3, np.nan],
            "BP_abs": [5.2, 2.6, 6.9, 7.4],
            "RP_abs": [4.2, 1.6, 5.8, 6.7],
            "e_G_abs": [0.03, 0.05, 0.07, 0.06],
            "e_BP_abs": [0.04, 0.07, 0.08, 0.09],
            "e_RP_abs": [0.04, 0.07, 0.08, 0.09],
        }
    )


def _run_demo() -> None:
    """Minimal runnable example requested in project requirements."""

    demo_targets = _demo_synthetic_targets()
    result_df, _ = fit_targets_to_isochrones(
        demo_targets,
        logage_min=8.0,
        logage_max=10.0,
        logage_step=0.2,
        mh_min=-1.0,
        mh_max=0.3,
        mh_step=0.2,
        store_surfaces=False,
    )

    display_cols = [
        "target_id",
        "best_logAge",
        "best_age_yr",
        "best_MH",
        "best_chi2",
        "best_phase_index",
        "fit_status",
    ]
    print(result_df[display_cols])


if __name__ == "__main__":
    _run_demo()


__all__ = [
    "IsochroneGridSpec",
    "build_isochrone_interpolator",
    "prepare_targets_from_csv",
    "fit_targets_to_isochrones",
]
