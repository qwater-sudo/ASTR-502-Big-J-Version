"""Fit Gaia absolute magnitudes to ezpadova isochrones.

This module provides a production-ready workflow to estimate stellar ages and
metallicities from Gaia absolute magnitudes using ezpadova's built-in
phase-aware interpolation (`QuickInterpolator`).
"""

from __future__ import annotations

import logging
import time
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
        LOGGER.error(
            "Target table validation failed: missing required columns=%s; available columns=%s",
            sorted(missing),
            list(targets.columns),
        )
        raise ValueError(f"targets is missing required column(s): {missing_sorted}")


def _sample_list(values: Iterable[Any], n: int = 10) -> list[Any]:
    values_list = list(values)
    return values_list[:n]


def _missing_reason(row: pd.Series, required_cols: Iterable[str]) -> str:
    missing = [col for col in required_cols if pd.isna(row.get(col))]
    return f"missing: {','.join(missing)}" if missing else "missing: none"


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
    total_grid_points = len(logages) * len(mhs)
    LOGGER.info(
        "Building isochrone interpolator: n_logAge=%d, n_MH=%d, total_grid_points=%d, photsys=%s",
        len(logages),
        len(mhs),
        total_grid_points,
        grid_spec.photsys,
    )

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

    if LOGGER.isEnabledFor(logging.DEBUG):
        raw_grid_shape = getattr(raw_grid, "shape", None)
        raw_grid_len = len(raw_grid) if hasattr(raw_grid, "__len__") else None
        LOGGER.debug(
            "Isochrone raw grid normalized: raw_type=%s, raw_shape=%s, raw_len=%s, iso_df_shape=%s",
            type(raw_grid).__name__,
            raw_grid_shape,
            raw_grid_len,
            iso_df.shape,
        )

    interpolator = QuickInterpolator(iso_df)

    model_cols = ["Gmag", "G_BPmag", "G_RPmag", "evol"]
    if extra_columns:
        for col in extra_columns:
            if col not in model_cols:
                model_cols.append(col)

    LOGGER.debug("Interpolator model columns requested: %s", model_cols)

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
    attempted_points = 0
    exception_count = 0
    missing_columns_count = 0
    no_finite_rows_count = 0
    accepted_count = 0
    debug_failed_examples: list[str] = []
    debug_accepted_examples: list[str] = []
    debug_cap = 10

    for logage in logage_grid:
        for mh in mh_grid:
            attempted_points += 1
            key = (float(logage), float(mh))
            try:
                raw_model = interpolator(float(logage), float(mh), what=model_cols)
            except Exception as exc:
                exception_count += 1
                if LOGGER.isEnabledFor(logging.DEBUG) and len(debug_failed_examples) < debug_cap:
                    debug_failed_examples.append(f"key={key};reason=exception;message={exc}")
                cache[key] = None
                continue

            model_df = _coerce_interp_result(raw_model, model_cols)
            if model_df is None:
                missing_columns_count += 1
                if LOGGER.isEnabledFor(logging.DEBUG) and len(debug_failed_examples) < debug_cap:
                    debug_failed_examples.append(f"key={key};reason=missing_expected_columns")
                cache[key] = None
                continue

            for mag_col in ("Gmag", "G_BPmag", "G_RPmag"):
                model_df[mag_col] = pd.to_numeric(model_df[mag_col], errors="coerce")

            finite_rows = np.isfinite(model_df[["Gmag", "G_BPmag", "G_RPmag"]]).all(axis=1)
            if finite_rows.sum() == 0:
                no_finite_rows_count += 1
                if LOGGER.isEnabledFor(logging.DEBUG) and len(debug_failed_examples) < debug_cap:
                    debug_failed_examples.append(f"key={key};reason=zero_finite_photometric_rows")
                cache[key] = None
                continue

            model_df = model_df.loc[finite_rows].reset_index(drop=True)

            cache[key] = {
                "Gmag": model_df["Gmag"].to_numpy(dtype=float),
                "G_BPmag": model_df["G_BPmag"].to_numpy(dtype=float),
                "G_RPmag": model_df["G_RPmag"].to_numpy(dtype=float),
                "phase": np.arange(len(model_df), dtype=int),
            }
            accepted_count += 1
            if LOGGER.isEnabledFor(logging.DEBUG) and len(debug_accepted_examples) < debug_cap:
                debug_accepted_examples.append(f"key={key};n_phase_points={len(model_df)}")

    LOGGER.info(
        "Model cache build summary: attempted=%d, exceptions=%d, missing_expected_columns=%d, zero_finite_photometric_rows=%d, accepted=%d",
        attempted_points,
        exception_count,
        missing_columns_count,
        no_finite_rows_count,
        accepted_count,
    )
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Model cache failed key samples (up to %d): %s", debug_cap, debug_failed_examples)
        LOGGER.debug("Model cache accepted key samples (up to %d): %s", debug_cap, debug_accepted_examples)

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
    n_grid_visited = 0
    n_missing_cached_model = 0
    n_nonfinite_chi2_arrays = 0

    LOGGER.debug(
        "Fitting target_id=%s with obs_mags=(G_abs=%.6g,BP_abs=%.6g,RP_abs=%.6g) and obs_errs=(e_G_abs=%.6g,e_BP_abs=%.6g,e_RP_abs=%.6g)",
        target_id,
        obs_g,
        obs_bp,
        obs_rp,
        e_g,
        e_bp,
        e_rp,
    )

    for i, logage in enumerate(logage_grid):
        for j, mh in enumerate(mh_grid):
            n_grid_visited += 1
            model = model_cache.get((float(logage), float(mh)))
            if model is None:
                n_missing_cached_model += 1
                continue

            chi2_points = (
                ((obs_g - model["Gmag"]) / e_g) ** 2
                + ((obs_bp - model["G_BPmag"]) / e_bp) ** 2
                + ((obs_rp - model["G_RPmag"]) / e_rp) ** 2
            )

            if chi2_points.size == 0 or not np.isfinite(chi2_points).any():
                n_nonfinite_chi2_arrays += 1
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
        LOGGER.warning(
            "No valid isochrone for target_id=%s: grid_visited=%d, missing_cached_model=%d, nonfinite_chi2_arrays=%d, grid_evaluated=%d",
            target_id,
            n_grid_visited,
            n_missing_cached_model,
            n_nonfinite_chi2_arrays,
            n_grid_evaluated,
        )
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
    LOGGER.debug(
        "Fit success target_id=%s: best=(logAge=%.6g,MH=%.6g), best_chi2=%.6g, best_phase_index=%d, n_grid_evaluated=%d, bounds_1sig=(logAge_lo=%.6g,logAge_hi=%.6g,MH_lo=%.6g,MH_hi=%.6g)",
        target_id,
        best_logage,
        best_mh,
        best_chi2,
        best_phase,
        n_grid_evaluated,
        logage_lo,
        logage_hi,
        mh_lo,
        mh_hi,
    )
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

    if LOGGER.isEnabledFor(logging.DEBUG):
        required_debug_cols = [TARGET_ID_COL, *REQUIRED_MAG_COLS]
        required_debug_cols.extend([col for col in ("e_G_abs", "e_BP_abs", "e_RP_abs") if col in targets.columns])
        debug_dtypes = {col: str(targets[col].dtype) for col in required_debug_cols if col in targets.columns}
        LOGGER.debug("fit_targets_to_isochrones input columns: %s", list(targets.columns))
        LOGGER.debug("fit_targets_to_isochrones required field dtypes: %s", debug_dtypes)

    LOGGER.info(
        "Starting fit_targets_to_isochrones: n_rows=%d, logAge=(min=%.6g,max=%.6g,step=%.6g), MH=(min=%.6g,max=%.6g,step=%.6g), photsys=%s, store_surfaces=%s, default_errors=(e_G_abs=%.6g,e_BP_abs=%.6g,e_RP_abs=%.6g)",
        len(targets),
        logage_min,
        logage_max,
        logage_step,
        mh_min,
        mh_max,
        mh_step,
        photsys,
        store_surfaces,
        default_e_g,
        default_e_bp,
        default_e_rp,
    )

    t_start = time.perf_counter()
    t_preprocess_start = t_start

    working = targets.copy()
    for col in REQUIRED_MAG_COLS:
        if LOGGER.isEnabledFor(logging.DEBUG):
            before_nan = int(working[col].isna().sum())
        working[col] = pd.to_numeric(working[col], errors="coerce")
        if LOGGER.isEnabledFor(logging.DEBUG):
            after_nan = int(working[col].isna().sum())
            LOGGER.debug(
                "Magnitude coercion for %s: nan_before=%d, nan_after=%d, nan_introduced=%d",
                col,
                before_nan,
                after_nan,
                after_nan - before_nan,
            )

    if "e_G_abs" not in working.columns:
        working["e_G_abs"] = float(default_e_g)
    if "e_BP_abs" not in working.columns:
        working["e_BP_abs"] = float(default_e_bp)
    if "e_RP_abs" not in working.columns:
        working["e_RP_abs"] = float(default_e_rp)

    for err_col, default_val in (("e_G_abs", default_e_g), ("e_BP_abs", default_e_bp), ("e_RP_abs", default_e_rp)):
        numeric = pd.to_numeric(working[err_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing_or_nonfinite = int(numeric.isna().sum())
        working[err_col] = numeric.fillna(float(default_val))
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "Error coercion for %s: missing_or_nonfinite_replaced=%d, default=%.6g",
                err_col,
                missing_or_nonfinite,
                default_val,
            )

    for err_col, default_val in (("e_G_abs", default_e_g), ("e_BP_abs", default_e_bp), ("e_RP_abs", default_e_rp)):
        non_positive = working[err_col] <= 0
        n_non_positive = int(non_positive.sum())
        if non_positive.any():
            working.loc[non_positive, err_col] = float(default_val)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "Error validation for %s: non_positive_reset=%d, default=%.6g",
                err_col,
                n_non_positive,
                default_val,
            )

    valid_mask = working[list(REQUIRED_MAG_COLS)].notna().all(axis=1)
    valid_targets = working.loc[valid_mask].copy()
    invalid_targets = working.loc[~valid_mask].copy()
    LOGGER.info(
        "Target validity split: valid=%d, invalid=%d",
        len(valid_targets),
        len(invalid_targets),
    )
    if LOGGER.isEnabledFor(logging.DEBUG) and not invalid_targets.empty:
        missing_counts = {col: int(invalid_targets[col].isna().sum()) for col in REQUIRED_MAG_COLS}
        invalid_examples = [
            {
                "target_id": row[TARGET_ID_COL],
                "reason": _missing_reason(row, REQUIRED_MAG_COLS),
            }
            for _, row in invalid_targets.iterrows()
        ]
        LOGGER.debug("Invalid target missing counts by magnitude: %s", missing_counts)
        LOGGER.debug("Invalid target samples (up to 10): %s", _sample_list(invalid_examples, n=10))

    t_preprocess_end = time.perf_counter()

    grid_spec = IsochroneGridSpec(
        logage_min=logage_min,
        logage_max=logage_max,
        logage_step=logage_step,
        mh_min=mh_min,
        mh_max=mh_max,
        mh_step=mh_step,
        photsys=photsys,
    )

    t_model_start = time.perf_counter()
    interpolator, logage_grid, mh_grid, model_cols = build_isochrone_interpolator(grid_spec)
    model_cache = _build_model_cache(interpolator, logage_grid, mh_grid, model_cols)
    t_model_end = time.perf_counter()

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

    t_fit_start = time.perf_counter()
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
    t_fit_end = time.perf_counter()

    result_df = pd.DataFrame(results)

    status_counts = result_df["fit_status"].value_counts(dropna=False).to_dict() if not result_df.empty else {}
    LOGGER.info(
        "Finished fitting %d targets. Status counts: %s",
        len(result_df),
        status_counts,
    )
    LOGGER.info(
        "Runtime summary (seconds): preprocess=%.3f, model_setup=%.3f, fitting=%.3f, total=%.3f",
        t_preprocess_end - t_preprocess_start,
        t_model_end - t_model_start,
        t_fit_end - t_fit_start,
        t_fit_end - t_start,
    )
    if LOGGER.isEnabledFor(logging.DEBUG) and not result_df.empty:
        sample_df = result_df[["target_id", "fit_status", "n_grid_evaluated"]].head(10)
        LOGGER.debug("Result sample (up to 10 rows): %s", sample_df.to_dict(orient="records"))

    if not store_surfaces:
        diagnostics = None

    return result_df, diagnostics




if __name__ == "__main__":

    phot_csv = '/Users/archon/classes/ASTR_502/Astro502_Sp26/ASTR502_Master_Photometry_List.csv'
    dist_csv = '/Users/archon/classes/ASTR_502/Astro502_Sp26/ASTR502_Mega_Target_List.csv'

    targets = prepare_targets_from_csv(phot_csv, dist_csv)

    results, _ = fit_targets_to_isochrones(
        targets.head(5),
        logage_min=8.0,
        logage_max=10.1,
        logage_step=0.1,
        mh_min=-2.0,
        mh_max=1.0,
        mh_step=0.4,
        photsys="gaiaEDR3",
        store_surfaces=False
    )

    print(results[[
        "target_id",
        "best_logAge",
        "best_age_yr",
        "best_MH",
        "best_chi2",
        "best_phase_index",
        "fit_status",
    ]])

__all__ = [
    "IsochroneGridSpec",
    "build_isochrone_interpolator",
    "prepare_targets_from_csv",
    "fit_targets_to_isochrones",
]
