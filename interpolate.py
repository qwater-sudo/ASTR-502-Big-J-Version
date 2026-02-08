from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from typing import Dict, Iterable, Optional, Tuple

from fetch_iso import IsochroneFetcher
from find_mag import PhotometryMerger


def _available_bands(obs: Dict[str, float],
                     model_df: pd.DataFrame,
                     mag_cols: Optional[Iterable[str]] = None):
    obs_keys = set(obs.keys())
    if mag_cols is None:
        cand = set(model_df.columns)
    else:
        cand = set(mag_cols)
    bands = list(obs_keys.intersection(cand))
    return bands

def _row_log_likelihood(row: pd.Series,
                        obs: Dict[str, float],
                        errs: Dict[str, float],
                        bands: Iterable[str]) -> float:
    ll = 0.0
    for b in bands:
        m_obs = obs.get(b)
        s = errs.get(b)
        m_mod = row.get(b)
        if m_obs is None or s is None or m_mod is None:
            continue
        if pd.isna(m_obs) or pd.isna(s) or pd.isna(m_mod) or s <=0:
            continue
        resid = m_obs - m_mod
        ll += -0.5 * (resid * resid) / (s * s) + np.log(2.0 * np.pi * s * s)
    return ll

def brute_force(
        observed_mags: Dict[str, float],
        observed_errs: Dict[str, float],
        model_df: pd.DataFrame,
        mag_cols: Optional[Iterable[str]] = None,
        mass_col: str = "mass",
        age_col: str = "age",
        feh_col: str = "mh"
) -> Tuple[pd.Series, pd.DataFrame]:
    bands = _available_bands(observed_mags, model_df, mag_cols)
    if len(bands) == 0:
        raise ValueError("No bands available")

    loglikes = model_df.apply(lambda r: _row_log_likelihood(r, observed_mags, observed_errs,bands), axis=1).to_numpy()
    max_ll = np.nanmax(loglikes)

    if not np.isfinite(max_ll):
        raise RuntimeError("All likelihoods are non-finite")

    shift = loglikes - max_ll
    probs = np.exp(shift)

    probs[~np.isfinite(shift)] = 0.0
    s = probs.sum()
    if s <=0:
        probs[:] = 0.0
    else:
        probs /= s

    results = model_df.copy()
    results['loglike'] = loglikes
    results['prob'] = probs

    best_idx = int(np.nanargmax(loglikes))
    best_row = results.iloc[best_idx]
    return best_row, results


class IsochroneInterpolator:
    """
    Build interpolators from model magnitudes -> (mass, age, feh).
    Usage:
      interp = IsochroneInterpolator(model_df, mag_cols=['G_abs','BP_RP_abs'])
      result = interp.interpolate({'G_abs': 5.12, 'BP_RP_abs': 0.65})
      result -> dict with keys 'mass','age','feh' (floats or np.nan if not available)
    """

    def __init__(self,
                 model_df: pd.DataFrame,
                 mag_cols: Optional[Iterable[str]] = None,
                 mass_col: str = "mass",
                 age_col: str = "age",
                 feh_col: str = "mh"):
        self.model_df = model_df.copy()
        # determine mag columns to use
        if mag_cols is None:
            # pick commonly used mag columns if present
            candidates = ['G_abs', 'BP_abs', 'RP_abs', 'BP_RP_abs']
            self.bands = [c for c in candidates if c in self.model_df.columns]
        else:
            self.bands = [c for c in mag_cols if c in self.model_df.columns]

        if len(self.bands) == 0:
            raise ValueError("No magnitude columns found in model dataframe for interpolation")

        self.mass_col = mass_col
        self.age_col = age_col
        self.feh_col = feh_col
        if self.feh_col not in self.model_df.columns:
            for alt in ("feh", "mh"):
                if alt in self.model_df.columns:
                    self.feh_col = alt
                    break

        # prepare training points: rows where all band mags are finite and outputs finite
        valid = np.ones(len(self.model_df), dtype=bool)
        for b in self.bands:
            valid &= np.isfinite(pd.to_numeric(self.model_df[b], errors='coerce'))
        valid &= np.isfinite(pd.to_numeric(self.model_df[self.mass_col], errors='coerce'))
        valid &= np.isfinite(pd.to_numeric(self.model_df[self.age_col], errors='coerce'))
        # feh optional
        has_feh = self.feh_col in self.model_df.columns
        if has_feh:
            valid &= np.isfinite(pd.to_numeric(self.model_df[self.feh_col], errors='coerce'))

        pts = self.model_df.loc[valid, self.bands].values
        if pts.shape[0] == 0:
            raise ValueError("No valid model points for interpolation")

        # outputs
        mass_y = self.model_df.loc[valid, self.mass_col].values
        age_y = self.model_df.loc[valid, self.age_col].values
        feh_y = (self.model_df.loc[valid, self.feh_col].values if has_feh else None)

        # try linear interpolator (works inside convex hull), fallback to nearest
        try:
            self.mass_lin = LinearNDInterpolator(pts, mass_y)
            self.age_lin = LinearNDInterpolator(pts, age_y)
            self.feh_lin = (LinearNDInterpolator(pts, feh_y) if has_feh else None)
            self.mass_near = NearestNDInterpolator(pts, mass_y)
            self.age_near = NearestNDInterpolator(pts, age_y)
            self.feh_near = (NearestNDInterpolator(pts, feh_y) if has_feh else None)
        except Exception:
            # if LinearNDInterpolator fails (e.g. degenerate points), build only nearest
            self.mass_lin = None
            self.age_lin = None
            self.feh_lin = None
            self.mass_near = NearestNDInterpolator(pts, mass_y)
            self.age_near = NearestNDInterpolator(pts, age_y)
            self.feh_near = (NearestNDInterpolator(pts, feh_y) if has_feh else None)

    def interpolate(self, observed: Dict[str, float], prefer: str = 'linear') -> Dict[str, float]:
        """
        Interpolate to predict mass, age, feh for a single observed magnitude dict.
        prefer: 'linear' (default) or 'nearest' to force nearest-neighbor.
        Returns dict with keys 'mass','age','feh' (feh may be None if not in model).
        If linear interpolation returns NaN (outside convex hull), falls back to nearest.
        """
        x = []
        for b in self.bands:
            v = observed.get(b, None)
            if v is None or not np.isfinite(v):
                raise ValueError(f"Observed magnitude for band `{b}` is missing or non-finite")
            x.append(float(v))
        x = np.asarray(x)
        # evaluate
        res = {}

        # helper to evaluate with fallback
        def _eval(lin_interp, near_interp):
            if prefer == 'nearest' or lin_interp is None:
                return float(near_interp(x))
            val = float(lin_interp(x))
            if not np.isfinite(val):
                return float(near_interp(x))
            return val

        res['mass'] = _eval(self.mass_lin, self.mass_near)
        res['age'] = _eval(self.age_lin, self.age_near)
        if self.feh_lin is not None or self.feh_near is not None:
            res['feh'] = _eval(self.feh_lin, self.feh_near)
        else:
            res['feh'] = None
        return res


def _find_iso_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for cand in candidates:
        for col in df.columns:
            if cand.lower() in col.lower():
                return col
    return None


def _standardize_isochrones(df: pd.DataFrame,
                            fetcher: IsochroneFetcher,
                            logage: float,
                            mh: float) -> pd.DataFrame:
    color, mag, (bp_col, rp_col, g_col) = fetcher.photometry(df)
    mass_col = _find_iso_col(df, ['mini', 'm_ini', 'minit', 'mass', 'mact'])
    age_col = _find_iso_col(df, ['logage', 'log_age', 'age'])
    mh_col = _find_iso_col(df, ['mh', 'm/h', 'feh', 'metallicity'])

    if mass_col is None:
        raise RuntimeError("Could not locate a mass column in the isochrone data")
    if age_col is None:
        raise RuntimeError("Could not locate an age column in the isochrone data")

    standardized = pd.DataFrame({
        'G_abs': pd.to_numeric(df[g_col], errors='coerce'),
        'BP_abs': pd.to_numeric(df[bp_col], errors='coerce'),
        'RP_abs': pd.to_numeric(df[rp_col], errors='coerce'),
        'BP_RP_abs': pd.to_numeric(color, errors='coerce'),
        'mass': pd.to_numeric(df[mass_col], errors='coerce'),
    })

    age_vals = pd.to_numeric(df[age_col], errors='coerce')
    if 'log' in age_col.lower():
        standardized['age'] = age_vals
    else:
        standardized['age'] = np.log10(age_vals)

    if mh_col is not None:
        standardized['mh'] = pd.to_numeric(df[mh_col], errors='coerce')
    else:
        standardized['mh'] = mh

    standardized['logage_request'] = logage
    standardized['mh_request'] = mh
    return standardized


def build_isochrone_grid(fetcher: IsochroneFetcher,
                         logages: Iterable[float],
                         mhs: Iterable[float]) -> pd.DataFrame:
    frames = []
    logages = list(logages)
    mhs = list(mhs)
    if hasattr(fetcher, "fetch_grid"):
        raw_frames = iter(fetcher.fetch_grid(logages, mhs))
        for logage in logages:
            for mh in mhs:
                raw = next(raw_frames)
                frames.append(_standardize_isochrones(raw, fetcher, logage, mh))
    else:
        for logage in logages:
            for mh in mhs:
                raw = fetcher.fetch(logage, mh)
                frames.append(_standardize_isochrones(raw, fetcher, logage, mh))
    if not frames:
        raise ValueError("No isochrones fetched; check logage/mh inputs")
    return pd.concat(frames, ignore_index=True)


def interpolate_catalog(merged_df: pd.DataFrame,
                        model_df: pd.DataFrame,
                        mag_cols: Optional[Iterable[str]] = None,
                        prefer: str = "linear",
                        prefix: str = "iso_") -> pd.DataFrame:
    interpolator = IsochroneInterpolator(model_df, mag_cols=mag_cols)
    mags = interpolator.bands
    results = {f"{prefix}mass": [], f"{prefix}age": [], f"{prefix}feh": []}

    for _, row in merged_df.iterrows():
        try:
            observed = {b: row.get(b) for b in mags}
            interp = interpolator.interpolate(observed, prefer=prefer)
            results[f"{prefix}mass"].append(interp['mass'])
            results[f"{prefix}age"].append(interp['age'])
            results[f"{prefix}feh"].append(interp['feh'])
        except Exception:
            results[f"{prefix}mass"].append(np.nan)
            results[f"{prefix}age"].append(np.nan)
            results[f"{prefix}feh"].append(np.nan)

    enriched = merged_df.copy()
    for col, values in results.items():
        enriched[col] = values
    return enriched


def interpolate_targets(phot_csv: str,
                        dist_csv: str,
                        logages: Iterable[float],
                        mhs: Iterable[float],
                        *,
                        fetcher: Optional[IsochroneFetcher] = None,
                        mag_cols: Optional[Iterable[str]] = None,
                        prefer: str = "linear",
                        prefix: str = "iso_",
                        join_key: Optional[str] = None,
                        how: str = "inner") -> pd.DataFrame:
    merger = PhotometryMerger()
    merged = merger.join_photometry_and_distances(phot_csv, dist_csv, on=join_key, how=how)
    if mag_cols is None:
        mag_cols = merger.interpolation_bands(merged)
    if fetcher is None:
        fetcher = IsochroneFetcher()
    model_df = build_isochrone_grid(fetcher, logages, mhs)
    return interpolate_catalog(merged, model_df, mag_cols=mag_cols, prefer=prefer, prefix=prefix)


__all__ = [
    "IsochroneInterpolator",
    "build_isochrone_grid",
    "interpolate_catalog",
    "interpolate_targets",
    "brute_force",
]
