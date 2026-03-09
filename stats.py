from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LikelihoodSummary:
    """Container for per-star and aggregate likelihood metrics."""

    n_used: int
    log_likelihood: float
    per_star_log_likelihood: pd.Series



def gaussian_log_likelihood(
    residuals: np.ndarray,
    sigma: np.ndarray | float,
) -> np.ndarray:
    """Return Gaussian log-likelihood values for each residual.

    Parameters
    ----------
    residuals
        Difference between observed and model-predicted quantities.
    sigma
        1-sigma uncertainty. Can be scalar or vector. Values must be > 0.
    """
    r = np.asarray(residuals, dtype=float)
    s = np.asarray(sigma, dtype=float)

    if np.any(~np.isfinite(r)):
        raise ValueError("residuals must be finite")
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("sigma must contain strictly positive finite values")

    return -0.5 * ((r / s) ** 2 + np.log(2.0 * np.pi * s**2))



def dataframe_log_likelihood(
    observed: pd.Series,
    predicted: pd.Series,
    sigma: float | pd.Series = 0.05,
) -> LikelihoodSummary:
    """Compute Gaussian log-likelihood from observed and predicted series."""
    obs = pd.to_numeric(observed, errors="coerce")
    pred = pd.to_numeric(predicted, errors="coerce")

    if isinstance(sigma, pd.Series):
        sig = pd.to_numeric(sigma, errors="coerce")
    else:
        sig = pd.Series(float(sigma), index=obs.index)

    mask = obs.notna() & pred.notna() & sig.notna()
    if mask.sum() == 0:
        return LikelihoodSummary(
            n_used=0,
            log_likelihood=float("-inf"),
            per_star_log_likelihood=pd.Series(dtype=float),
        )

    ll = gaussian_log_likelihood(
        residuals=(obs[mask] - pred[mask]).to_numpy(),
        sigma=sig[mask].to_numpy(),
    )

    ll_series = pd.Series(ll, index=obs[mask].index, name="log_likelihood")

    return LikelihoodSummary(
        n_used=int(mask.sum()),
        log_likelihood=float(np.sum(ll)),
        per_star_log_likelihood=ll_series,
    )



def combined_log_likelihood(
    summaries: Iterable[LikelihoodSummary],
) -> LikelihoodSummary:
    """Combine multiple likelihood summaries by summing log-likelihood terms."""
    summaries = list(summaries)
    if not summaries:
        return LikelihoodSummary(
            n_used=0,
            log_likelihood=float("-inf"),
            per_star_log_likelihood=pd.Series(dtype=float),
        )

    combined = pd.concat([s.per_star_log_likelihood for s in summaries], axis=1)
    per_star = combined.sum(axis=1, min_count=1)

    return LikelihoodSummary(
        n_used=int(sum(s.n_used for s in summaries)),
        log_likelihood=float(sum(s.log_likelihood for s in summaries)),
        per_star_log_likelihood=per_star,
    )
