"""ASTR-502 interpolation and fitting toolkit."""

from astr502.domain.schemas import FitResultSchema
from astr502.modeling.interpolate import fit_best_params, get_model_mag, load_catalogs
from astr502.services.fit_runtime import fit_single_star_runtime, fit_target_list_runtime

__all__ = [
    "FitResultSchema",
    "load_catalogs",
    "get_model_mag",
    "fit_best_params",
    "fit_single_star_runtime",
    "fit_target_list_runtime",
]
