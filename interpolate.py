from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Iterable
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from find_mag import PhotometryMerger
from read_spot_models import SPOT
from stats import LikelihoodSummary, dataframe_log_likelihood


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


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for cand in candidates:
        for col in df.columns:
            if cand.lower() in str(col).lower():
                return str(col)
    return None


def _as_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _extract_metallicity_from_path(file_path: str | Path) -> float:
    """Infer metallicity from SPOT filename style like f000.isoc or fm05.isoc."""
    stem = Path(file_path).stem.lower()

    # f000 -> 0.00, fp05 -> +0.5, fm05 -> -0.5
    m = re.search(r"f([pm]?)(\d+)", stem)
    if not m:
        return float("nan")

    sign = m.group(1)
    digits = m.group(2)
    value = float(digits) / 100.0
    if sign == "m":
        value *= -1.0
    return value


def _build_color_mag_interpolator(
    isochrone_df: pd.DataFrame,
    color_col: str,
    mag_col: str,
):
    """Build scipy.interpolate interp1d(color -> magnitude) for one isochrone."""
    track = isochrone_df[[color_col, mag_col]].dropna().sort_values(color_col)
    if len(track) < 2:
        return None

    # Remove repeated x values to keep interp1d stable.
    track = track.loc[~track[color_col].duplicated(keep="first")]
    if len(track) < 2:
        return None

    return interp1d(
        track[color_col].to_numpy(),
        track[mag_col].to_numpy(),
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )


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
    iso = _as_numeric(isochrone_df)
    targets = _as_numeric(targets_df)

    # Target photometric columns created by PhotometryMerger.join_photometry_and_distances.
    target_color_col = "BP_RP_abs"
    target_mag_col = "G_abs"

    iso_bp_col = _find_col(iso, ["g_bp", "bp"])
    iso_rp_col = _find_col(iso, ["g_rp", "rp"])
    iso_g_col = _find_col(iso, ["g", "gmag", "gaia_g"])
    mass_col = _find_col(iso, ["m/m", "mass", "mini", "m_ini", "mact"])

    if not (iso_bp_col and iso_rp_col and iso_g_col):
        logger.error(
            "Could not identify BP/RP/G columns for logAge=%.3f [M/H]=%.3f. Columns=%s",
            age_log10_yr,
            metallicity_dex,
            list(iso.columns),
        )
        raise ValueError(
            "Could not identify BP/RP/G columns in SPOT isochrone section; "
            f"columns were: {list(iso.columns)}"
        )

    work = iso[[iso_bp_col, iso_rp_col, iso_g_col] + ([mass_col] if mass_col else [])].copy()
    work["iso_color"] = work[iso_bp_col] - work[iso_rp_col]
    work["iso_mag"] = work[iso_g_col]

    interpolator = _build_color_mag_interpolator(work, "iso_color", "iso_mag")
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
        empty = pd.DataFrame(columns=[target_color_col, target_mag_col, "iso_mag_pred", "mass_pred"])
        summary = LikelihoodSummary(0, float("-inf"), pd.Series(dtype=float))
        return result, summary, empty

    mask = targets[target_color_col].notna() & targets[target_mag_col].notna()
    eval_df = targets.loc[mask, [target_color_col, target_mag_col]].copy()
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every SPOT age section from every metallicity file to target data."""
    results: list[IsochroneFitResult] = []
    best_eval: pd.DataFrame | None = None
    best_ll = float("-inf")

    if isinstance(spot_iso_files, (str, Path)):
        iso_files = [spot_iso_files]
    else:
        iso_files = list(spot_iso_files)

    logger.info("Preparing to fit SPOT grid for %d isochrone file(s)", len(iso_files))
    if not iso_files:
        logger.warning("No SPOT isochrone files were supplied")
        return pd.DataFrame(), pd.DataFrame()

    for iso_file in iso_files:
        metallicity = _extract_metallicity_from_path(iso_file)
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
                logger.info(
                    "New best fit detected: logAge=%.3f [M/H]=%.3f logL=%.5f",
                    fit.age_log10_yr,
                    fit.metallicity_dex,
                    fit.log_likelihood,
                )

    if not results:
        logger.warning("No valid SPOT fits were produced across all input files")
        return pd.DataFrame(), pd.DataFrame()

    finite_count = int(np.isfinite([r.log_likelihood for r in results]).sum())
    logger.info(
        "Completed SPOT grid fit: %d candidate fit(s), %d with finite log-likelihood",
        len(results),
        finite_count,
    )

    results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values(
        "log_likelihood", ascending=False
    )
    return results_df.reset_index(drop=True), (best_eval if best_eval is not None else pd.DataFrame())


def load_targets(phot_csv: str | Path, dist_csv: str | Path) -> pd.DataFrame:
    """Load merged target list from photometry + distance catalogues."""
    logger.info("Loading targets from photometry=%s and distances=%s", phot_csv, dist_csv)
    merger = PhotometryMerger()
    merged = merger.join_photometry_and_distances(phot_csv=phot_csv, dist_csv=dist_csv)
    logger.info("Loaded merged target table with %d rows and %d columns", *merged.shape)
    required = {"BP_RP_abs", "G_abs"}
    missing = required.difference(merged.columns)
    if missing:
        logger.error("Merged target table is missing required columns: %s", sorted(missing))
    else:
        valid = merged[list(required)].dropna()
        logger.debug(
            "Targets with finite BP_RP_abs and G_abs: %d / %d",
            len(valid),
            len(merged),
        )
    return merged

def plot_fitted_model_against_targets(
    targets_df: pd.DataFrame,
    fitted_eval_df: pd.DataFrame,
    title: str = "Best-fit SPOT isochrone vs target data",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot observed target CMD and fitted model CMD (mag vs BP-RP).

    Parameters
    ----------
    targets_df
        Master target table containing at least ``BP_RP_abs`` and ``G_abs``.
    fitted_eval_df
        DataFrame returned by ``fit_isochrone_section_to_targets`` or
        ``fit_spot_grid_to_targets`` best model output. Must contain
        ``BP_RP_abs`` and ``iso_mag_pred``.
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
    ax.plot(
        plot_fit[target_color_col],
        plot_fit[pred_mag_col],
        color="tab:red",
        linewidth=2,
        label="Fitted isochrone",
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
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[plt.Figure, plt.Axes]]:
    """Test helper: run grid fitting and plot the best fitted model.

    Returns
    -------
    results_df, best_eval_df, (fig, ax)
        Ranked fit table, evaluated best-fit model table, and matplotlib handles.
    """
    targets_df = load_targets(phot_csv=phot_csv, dist_csv=dist_csv)
    results_df, best_eval_df = fit_spot_grid_to_targets(
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
    fig_ax = plot_fitted_model_against_targets(
        targets_df=targets_df,
        fitted_eval_df=best_eval_df,
        title=(
            "Best-fit SPOT model "
            f"(logAge={best['age_log10_yr']:.3f}, [M/H]={best['metallicity_dex']:.3f})"
        ),
        save_path=save_path,
    )
    return results_df, best_eval_df, fig_ax


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    test_fit_and_plot(
        phot_csv='/Users/archon/classes/ASTR_502/Astro502_Sp26/ASTR502_Master_Photometry_List.csv',
        dist_csv='/Users/archon/classes/ASTR_502/Astro502_Sp26/ASTR502_Mega_Target_List.csv',
        spot_iso_files=glob.glob('/Users/archon/classes/ASTR_502/workstation/isochrones/SPOTS/isos/*.isoc')
    )
