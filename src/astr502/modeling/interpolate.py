from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from astr502.data.catalogs import CatalogStore, CatalogUtils, DEFAULT_MEGA_CSV, DEFAULT_PHOT_CSV
from astr502.data.readers.read_spot_models import SPOT
from astr502.data.utils import IsochroneUtils, REQUESTED_BANDS
from astr502.domain.schemas import FitResultSchema
from astr502.domain.stats import summarize_chi_square
from astr502.modeling.extinction import get_band_extinction

_CATALOG_STORE = CatalogStore()
_INTERPOLATORS: dict[str, RegularGridInterpolator] | None = None
_GRIDS: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
_ACTIVE_BANDS: list[str] | None = None


def load_catalogs(
    mega_csv_path: str = DEFAULT_MEGA_CSV,
    phot_csv_path: str = DEFAULT_PHOT_CSV,
) -> None:
    _CATALOG_STORE.load_catalogs(mega_csv_path=mega_csv_path, phot_csv_path=phot_csv_path)


def _build_interpolators(
    spot_iso_files: list[str] | None = None,
    mass_points: int = 300,
) -> tuple[dict[str, RegularGridInterpolator], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    global _ACTIVE_BANDS

    if spot_iso_files is None:
        spot_iso_files = IsochroneUtils.discover_spot_files()
    if not spot_iso_files:
        raise FileNotFoundError("No SPOT isochrone files found at data/raw/isochrones/SPOTS/isos/*.isoc")

    all_sections: list[tuple[float, float, pd.DataFrame]] = []
    age_values: set[float] = set()
    feh_values: set[float] = set()

    for iso_file in spot_iso_files:
        feh = IsochroneUtils.extract_metallicity_from_path(iso_file)
        sections = SPOT(str(iso_file), verbose=False).read_iso_file()
        for age_log10, section in sections.items():
            age_yr = 10.0 ** float(age_log10)
            all_sections.append((age_yr, float(feh), section))
            age_values.add(age_yr)
            feh_values.add(float(feh))

    age_grid = np.array(sorted(age_values), dtype=float)
    feh_grid = np.array(sorted(feh_values), dtype=float)
    mass_grid = np.linspace(0.1, 3.0, mass_points)

    first_section = all_sections[0][2]
    _ACTIVE_BANDS = [b for b in REQUESTED_BANDS if IsochroneUtils.find_band_column(first_section, b) is not None]

    magnitude_grids = {
        band: np.full((mass_grid.size, age_grid.size, feh_grid.size), np.nan)
        for band in _ACTIVE_BANDS
    }

    age_index = {age: idx for idx, age in enumerate(age_grid)}
    feh_index = {feh: idx for idx, feh in enumerate(feh_grid)}

    for age_yr, feh, section in all_sections:
        try:
            selected, mass_col = IsochroneUtils.select_rows(section)
        except ValueError:
            continue

        masses = selected[mass_col].to_numpy(dtype=float)
        if len(masses) < 2:
            continue

        ai = age_index[age_yr]
        fi = feh_index[feh]

        for band in _ACTIVE_BANDS:
            band_col = IsochroneUtils.find_band_column(selected, band)
            if band_col is None:
                continue
            values = selected[band_col].to_numpy(dtype=float)

            in_range = (mass_grid >= masses[0]) & (mass_grid <= masses[-1])
            magnitude_grids[band][in_range, ai, fi] = np.interp(mass_grid[in_range], masses, values)
            magnitude_grids[band][mass_grid > masses[-1], ai, fi] = values[-1]
            magnitude_grids[band][mass_grid < masses[0], ai, fi] = values[0]

    for band in _ACTIVE_BANDS:
        grid = magnitude_grids[band]
        if np.all(np.isnan(grid)):
            continue
        mask = np.isnan(grid)
        if np.any(mask):
            from scipy.ndimage import distance_transform_edt

            idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
            magnitude_grids[band] = grid[tuple(idx)]

    interpolators = {
        band: RegularGridInterpolator(
            (mass_grid, age_grid, feh_grid),
            magnitude_grids[band],
            bounds_error=False,
            fill_value=None,
        )
        for band in _ACTIVE_BANDS
    }

    return interpolators, (mass_grid, age_grid, feh_grid)


def _get_interpolators() -> tuple[dict[str, RegularGridInterpolator], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    global _INTERPOLATORS, _GRIDS
    if _INTERPOLATORS is None or _GRIDS is None:
        _INTERPOLATORS, _GRIDS = _build_interpolators()
    return _INTERPOLATORS, _GRIDS


def get_model_mag(mass: float, age: float, feh: float, av: float = 0.0) -> dict[str, float]:
    interpolators, _ = _get_interpolators()
    points = np.array([[float(mass), float(age), float(feh)]])
    bands = list(interpolators)
    extinction = get_band_extinction(bands=bands, av=av)

    output: dict[str, float] = {}
    for band, interpolator in interpolators.items():
        output[band] = float(interpolator(points)[0] + extinction.get(band, 0.0))
    return output


def fit_best_params(
    hostname: str,
    sigma_phot: float = 0.5,
    fallback_sigma_param: float = 0.25,
    av_bounds: tuple[float, float] = (0.0, 3.0),
    bounds: list[tuple[float, float]] | None = None,
    verbose: bool = True,
) -> tuple[FitResultSchema, object]:
    mega_df, phot_df = _CATALOG_STORE.ensure_loaded()
    obs_abs, distance_pc = CatalogUtils.get_star_obs_abs(hostname, mega_df=mega_df, phot_df=phot_df)
    prior = CatalogUtils.get_param_prior(
        hostname,
        mega_df=mega_df,
        phot_df=phot_df,
        fallback_sigma=fallback_sigma_param,
    )

    m0 = prior["m0"] if np.isfinite(prior["m0"]) else 1.0
    a0 = prior["a0_gyr"] if np.isfinite(prior["a0_gyr"]) else 5.0
    feh0 = prior["feh0"] if np.isfinite(prior["feh0"]) else 0.0
    x0 = np.array([m0, np.log10(a0 * 1e9), feh0, 0.0], dtype=float)

    if bounds is None:
        bounds = [(0.1, 3.0), (6.0, np.log10(13.8e9)), (-1.0, 0.5), av_bounds]

    def objective(x: np.ndarray) -> float:
        mass, log10_age, feh, av = x
        if mass <= 0 or av < 0:
            return 1e30

        model = get_model_mag(mass=mass, age=10.0 ** log10_age, feh=feh, av=av)
        return summarize_chi_square(
            model_mags=model,
            observed_abs_mags=obs_abs,
            sigma_phot=sigma_phot,
            mass=mass,
            log10_age=log10_age,
            feh=feh,
            prior=prior,
        ).chi2_total

    result = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
    mass_b, log10_age_b, feh_b, av_b = result.x
    age_yr_b = 10.0 ** log10_age_b
    model_best = get_model_mag(mass=mass_b, age=age_yr_b, feh=feh_b, av=av_b)
    chi2 = summarize_chi_square(
        model_mags=model_best,
        observed_abs_mags=obs_abs,
        sigma_phot=sigma_phot,
        mass=mass_b,
        log10_age=log10_age_b,
        feh=feh_b,
        prior=prior,
    )

    fit = FitResultSchema(
        hostname=hostname,
        mass=float(mass_b),
        age_yr=float(age_yr_b),
        feh=float(feh_b),
        av=float(av_b),
        chi2_phot=float(chi2.chi2_phot),
        chi2_prior=float(chi2.chi2_prior),
        chi2_total=float(chi2.chi2_total),
        distance_pc=float(distance_pc),
        model_magnitudes=model_best,
    )

    if verbose:
        print(f"[{hostname}] Best-fit parameters (chi2_phot + chi2_prior)")
        print(f"  mass = {fit.mass:.4f} Msun")
        print(f"  age  = {fit.age_yr:.3e} yr")
        print(f"  feh  = {fit.feh:.4f} dex")
        print(f"  Av   = {fit.av:.4f} mag")
        print(f"  chi2_total = {fit.chi2_total:.3f}")
        print(f"  d_pc used = {fit.distance_pc:.3f}")
        print(f"  success = {result.success} | {result.message}")

    return fit, result


def get_bestfit_model_mag_for_star(hostname: str, **fit_kwargs) -> tuple[FitResultSchema, dict[str, float]]:
    fit, _ = fit_best_params(hostname=hostname, **fit_kwargs)
    return fit, dict(fit.model_magnitudes)


def save_fit_results_to_csv(results: list[FitResultSchema], output_csv: str = "outputs/results/interpolate_best_fit_results.csv") -> str:
    final_path = Path(output_csv)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.to_record() for r in results]).to_csv(final_path, index=False)
    return str(final_path)


__all__ = [
    "load_catalogs",
    "get_model_mag",
    "fit_best_params",
    "get_bestfit_model_mag_for_star",
    "save_fit_results_to_csv",
]
