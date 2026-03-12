from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from find_mag import PhotometryMerger
from read_spot_models import SPOT
from stats import LikelihoodSummary, dataframe_log_likelihood
from utils import (
    DataFrameUtils,
    IsochroneUtils,
    LoggingUtils,
    ResultsManager,
    TargetDataLoader,
)


logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class IsochroneFitResult:
    """Best-fit statistics for one (age, metallicity) isochrone section."""

    age_log10_yr: float
    metallicity_dex: float
    log_likelihood: float
    n_used: int
    predicted_mass_mean: float
    predicted_mass_median: float


def fit_isochrone_section_to_targets(
    isochrone_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    age_log10_yr: float,
    metallicity_dex: float,
    sigma_mag: float = 0.05,
) -> tuple[IsochroneFitResult, LikelihoodSummary, pd.DataFrame]:
    """Fit one isochrone section to the target CMD and score with log-likelihood."""
    logger.info(
        "Starting fit for isochrone section: logAge=%.3f, [M/H]=%.3f, sigma_mag=%.3f",
        age_log10_yr,
        metallicity_dex,
        sigma_mag,
    )
    targets = DataFrameUtils.as_numeric(targets_df)
    host_col = DataFrameUtils.find_col(targets_df, ["hostname"])
    if host_col and host_col in targets_df.columns:
        targets["hostname"] = targets_df[host_col]

    # Target photometric columns created by PhotometryMerger.join_photometry_and_distances.
    target_color_col = "BP_RP_abs"
    target_mag_col = "G_abs"

    try:
        work, mass_col = IsochroneUtils.prepare_isochrone_track(isochrone_df)
    except ValueError:
        logger.error(
            "Could not identify BP/RP/G columns for logAge=%.3f [M/H]=%.3f. Columns=%s",
            age_log10_yr,
            metallicity_dex,
            list(isochrone_df.columns),
        )
        raise

    interpolator = IsochroneUtils.build_color_mag_interpolator(work, "iso_color", "iso_mag")
    if interpolator is None:
        logger.warning(
            "Skipping isochrone section (insufficient points for interpolation): "
            "logAge=%.3f, [M/H]=%.3f",
            age_log10_yr,
            metallicity_dex,
        )
        result = IsochroneFitResult(
            age_log10_yr=float(age_log10_yr),
            metallicity_dex=float(metallicity_dex),
            log_likelihood=float("-inf"),
            n_used=0,
            predicted_mass_mean=float("nan"),
            predicted_mass_median=float("nan"),
        )
        empty_columns = [target_color_col, target_mag_col, "iso_mag_pred", "mass_pred"]
        if host_col is not None:
            empty_columns.insert(0, "hostname")
        empty = pd.DataFrame(columns=empty_columns)
        summary = LikelihoodSummary(0, float("-inf"), pd.Series(dtype=float))
        return result, summary, empty

    mask = targets[target_color_col].notna() & targets[target_mag_col].notna()
    eval_columns = [target_color_col, target_mag_col]
    if host_col is not None and "hostname" in targets.columns:
        eval_columns.insert(0, "hostname")
    eval_df = targets.loc[mask, eval_columns].copy()
    eval_df["iso_mag_pred"] = interpolator(eval_df[target_color_col].to_numpy())

    ll_summary = dataframe_log_likelihood(
        observed=eval_df[target_mag_col],
        predicted=eval_df["iso_mag_pred"],
        sigma=sigma_mag,
    )
    logger.debug(
        "Likelihood computed for logAge=%.3f [M/H]=%.3f: n_used=%d, logL=%.5f",
        age_log10_yr,
        metallicity_dex,
        ll_summary.n_used,
        ll_summary.log_likelihood,
    )

    # Estimate masses by nearest color location on the isochrone if mass exists.
    if mass_col:
        ref = work[["iso_color", mass_col]].dropna().sort_values("iso_color")
        if len(ref) >= 2 and ref["iso_color"].nunique() >= 2:
            mass_interp = interp1d(
                ref["iso_color"].to_numpy(),
                ref[mass_col].to_numpy(),
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
            eval_df["mass_pred"] = mass_interp(eval_df[target_color_col].to_numpy())
        else:
            eval_df["mass_pred"] = np.nan
    else:
        eval_df["mass_pred"] = np.nan

    result = IsochroneFitResult(
        age_log10_yr=float(age_log10_yr),
        metallicity_dex=float(metallicity_dex),
        log_likelihood=ll_summary.log_likelihood,
        n_used=ll_summary.n_used,
        predicted_mass_mean=float(pd.to_numeric(eval_df["mass_pred"], errors="coerce").mean()),
        predicted_mass_median=float(pd.to_numeric(eval_df["mass_pred"], errors="coerce").median()),
    )

    logger.info(
        "Fit complete for logAge=%.3f [M/H]=%.3f: success=%s, n_used=%d, logL=%.5f, "
        "mass_mean=%.4f, mass_median=%.4f",
        age_log10_yr,
        metallicity_dex,
        np.isfinite(result.log_likelihood),
        result.n_used,
        result.log_likelihood,
        result.predicted_mass_mean,
        result.predicted_mass_median,
    )

    return result, ll_summary, eval_df


def fit_spot_grid_to_targets(
    targets_df: pd.DataFrame,
    spot_iso_files: str | Path | Iterable[str | Path],
    sigma_mag: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit every SPOT age section from every metallicity file to target data."""
    results: list[IsochroneFitResult] = []
    best_eval: pd.DataFrame | None = None
    best_iso_track: pd.DataFrame | None = None
    best_ll = float("-inf")

    if isinstance(spot_iso_files, (str, Path)):
        iso_files = [spot_iso_files]
    else:
        iso_files = list(spot_iso_files)

    logger.info("Preparing to fit SPOT grid for %d isochrone file(s)", len(iso_files))
    if not iso_files:
        logger.warning("No SPOT isochrone files were supplied")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for iso_file in iso_files:
        metallicity = IsochroneUtils.extract_metallicity_from_path(iso_file)
        if np.isnan(metallicity):
            logger.debug(
                "Could not infer metallicity from file name '%s'; proceeding with NaN",
                iso_file,
            )
        logger.info(
            "Loading SPOT isochrone file: %s (detected [M/H]=%.3f)",
            iso_file,
            metallicity,
        )
        try:
            sections = SPOT(str(iso_file)).read_iso_file()
        except Exception:
            logger.exception("Failed to parse SPOT isochrone file: %s", iso_file)
            continue

        if not sections:
            logger.warning("Isochrone file %s did not produce any age sections", iso_file)
            continue

        logger.info("File %s contains %d age sections", iso_file, len(sections))

        for age, section_df in sections.items():
            logger.info(
                "Matching targets against isochrone age section: logAge=%s from file %s",
                age,
                iso_file,
            )
            try:
                fit, _, eval_df = fit_isochrone_section_to_targets(
                    isochrone_df=section_df,
                    targets_df=targets_df,
                    age_log10_yr=float(age),
                    metallicity_dex=metallicity,
                    sigma_mag=sigma_mag,
                )
            except ValueError:
                logger.warning(
                    "Failed fit for logAge=%s in %s due to invalid/missing columns",
                    age,
                    iso_file,
                )
                continue

            results.append(fit)
            if not np.isfinite(fit.log_likelihood):
                logger.debug(
                    "Discarding non-finite fit score for logAge=%.3f [M/H]=%.3f (n_used=%d)",
                    fit.age_log10_yr,
                    fit.metallicity_dex,
                    fit.n_used,
                )

            if fit.log_likelihood > best_ll:
                best_ll = fit.log_likelihood
                best_eval = eval_df
                try:
                    best_iso_track, _ = IsochroneUtils.prepare_isochrone_track(section_df)
                except ValueError:
                    best_iso_track = None
                logger.info(
                    "New best fit detected: logAge=%.3f [M/H]=%.3f logL=%.5f",
                    fit.age_log10_yr,
                    fit.metallicity_dex,
                    fit.log_likelihood,
                )

    if not results:
        logger.warning("No valid SPOT fits were produced across all input files")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    finite_count = int(np.isfinite([r.log_likelihood for r in results]).sum())
    logger.info(
        "Completed SPOT grid fit: %d candidate fit(s), %d with finite log-likelihood",
        len(results),
        finite_count,
    )

    results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values(
        "log_likelihood", ascending=False
    )
    return (
        results_df.reset_index(drop=True),
        (best_eval if best_eval is not None else pd.DataFrame()),
        (best_iso_track if best_iso_track is not None else pd.DataFrame()),
    )


def save_best_fit_candidates(
    best_eval_df: pd.DataFrame,
    best_age_log10_yr: float,
    output_dir: str | Path = "results",
) -> Path:
    """Write candidate observed/fitted magnitudes for the best-fit isochrone."""
    _ = best_age_log10_yr
    return ResultsManager.save_best_fit_candidates(best_eval_df=best_eval_df, output_dir=output_dir)


def load_targets(phot_csv: str | Path, dist_csv: str | Path) -> pd.DataFrame:
    """Load merged target list from photometry + distance catalogues."""
    return TargetDataLoader.load_targets(phot_csv=phot_csv, dist_csv=dist_csv)

def plot_fitted_model_against_targets(
    targets_df: pd.DataFrame,
    fitted_eval_df: pd.DataFrame,
    title: str = "Best-fit SPOT isochrone vs target data",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot observed targets and best-fit predicted magnitudes on CMD axes.

    Parameters
    ----------
    targets_df
        Master target table containing at least ``BP_RP_abs`` and ``G_abs``.
    fitted_eval_df
        Best-fit per-target evaluation table. Must contain ``BP_RP_abs`` and
        ``iso_mag_pred`` produced from the best ``logAge`` isochrone.
    title
        Plot title.
    save_path
        Optional output path for saving the figure.
    """
    target_color_col = "BP_RP_abs"
    target_mag_col = "G_abs"
    pred_mag_col = "iso_mag_pred"

    required_target_cols = {target_color_col, target_mag_col}
    required_fit_cols = {target_color_col, pred_mag_col}

    if not required_target_cols.issubset(set(targets_df.columns)):
        raise ValueError(
            f"targets_df must include columns {sorted(required_target_cols)}"
        )
    if not required_fit_cols.issubset(set(fitted_eval_df.columns)):
        raise ValueError(
            f"fitted_eval_df must include columns {sorted(required_fit_cols)}"
        )

    plot_targets = targets_df[[target_color_col, target_mag_col]].copy()
    plot_targets = plot_targets.dropna()

    plot_fit = fitted_eval_df[[target_color_col, pred_mag_col]].copy()
    plot_fit = plot_fit.dropna().sort_values(target_color_col)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.scatter(
        plot_targets[target_color_col],
        plot_targets[target_mag_col],
        s=8,
        alpha=0.45,
        color="black",
        label="Targets",
    )
    ax.scatter(
        plot_fit[target_color_col],
        plot_fit[pred_mag_col],
        s=8,
        alpha=1,
        color="tab:red",
        label="Best logAge fit (per target)",
    )


    ax.set_xlabel("BP-RP")
    ax.set_ylabel("G")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def test_fit_and_plot(
    phot_csv: str | Path,
    dist_csv: str | Path,
    spot_iso_files: str | Path | Iterable[str | Path],
    sigma_mag: float = 0.05,
    save_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[plt.Figure, plt.Axes], Path]:
    """Test helper: run grid fitting and plot the best fitted model.

    Returns
    -------
    results_df, best_eval_df, best_isochrone_df, (fig, ax), save_csv
        Ranked fit table, evaluated best-fit model table, best-fit isochrone track,
        matplotlib handles, and saved candidate fit CSV path.
    """
    targets_df = load_targets(phot_csv=phot_csv, dist_csv=dist_csv)
    results_df, best_eval_df, best_isochrone_df = fit_spot_grid_to_targets(
        targets_df=targets_df,
        spot_iso_files=spot_iso_files,
        sigma_mag=sigma_mag,
    )

    if results_df.empty or best_eval_df.empty:
        logger.error(
            "No valid SPOT fits were produced. results_df.empty=%s, best_eval_df.empty=%s",
            results_df.empty,
            best_eval_df.empty,
        )
        raise RuntimeError("No valid SPOT fits were produced; cannot generate test plot")

    best = results_df.iloc[0]
    save_csv = save_best_fit_candidates(
        best_eval_df=best_eval_df,
        best_age_log10_yr=float(best["age_log10_yr"]),
        output_dir="results",
    )
    resolved_save_path = (
        Path(save_path) if save_path is not None else ResultsManager.default_plot_save_path("figs")
    )
    fig_ax = plot_fitted_model_against_targets(
        targets_df=targets_df,
        fitted_eval_df=best_eval_df,
        title=(
            "Best-fit SPOT model "
            f"(logAge={best['age_log10_yr']:.3f}, [M/H]={best['metallicity_dex']:.3f})"
        ),
        save_path=resolved_save_path,
    )
    logger.info("Wrote best-fit CMD figure to %s", resolved_save_path)
    return results_df, best_eval_df, best_isochrone_df, fig_ax, save_csv


def configure_debug_logging(log_dir: str | Path = "logs") -> Path:
    """Configure root logger to write debug output to ``log_dir``.

    A new file is created on each run using the current timestamp.

    Returns
    -------
    Path
        Full path to the log file.
    """
    return LoggingUtils.configure_debug_logging(log_dir=log_dir)


if __name__ == "__main__":
    configure_debug_logging("logs")

    test_fit_and_plot(
        phot_csv='/Users/archon/classes/ASTR_502/Astro502_Sp26/ASTR502_Master_Photometry_List.csv',
        dist_csv='/Users/archon/classes/ASTR_502/Astro502_Sp26/ASTR502_Mega_Target_List.csv',
        spot_iso_files=glob.glob('/Users/archon/classes/ASTR_502/workstation/isochrones/SPOTS/isos/*.isoc')
    )
    plt.show()
