from __future__ import annotations

from src.astr502.data.catalogs import DEFAULT_MEGA_CSV, DEFAULT_PHOT_CSV
from src.astr502.domain.schemas import FitResultSchema
from src.astr502.modeling.interpolate import fit_best_params, load_catalogs, save_fit_results_to_csv


def fit_single_star_runtime(
    hostname: str,
    *,
    mega_csv_path: str = DEFAULT_MEGA_CSV,
    phot_csv_path: str = DEFAULT_PHOT_CSV,
    output_csv: str | None = None,
    **fit_kwargs,
) -> FitResultSchema:
    """Runtime helper to fit one star from the catalog inputs."""
    load_catalogs(mega_csv_path=mega_csv_path, phot_csv_path=phot_csv_path)
    fit, _ = fit_best_params(hostname=hostname, **fit_kwargs)
    if output_csv is not None:
        save_fit_results_to_csv([fit], output_csv=output_csv)
    return fit


def fit_target_list_runtime(
    *,
    mega_csv_path: str = DEFAULT_MEGA_CSV,
    phot_csv_path: str = DEFAULT_PHOT_CSV,
    hostnames: list[str] | None = None,
    output_csv: str = "outputs/results/interpolate_best_fit_results.csv",
    continue_on_error: bool = True,
    verbose: bool = True,
    **fit_kwargs,
) -> tuple[list[FitResultSchema], list[tuple[str, str]]]:
    """Runtime helper to fit a user-provided or catalog-derived host list."""
    load_catalogs(mega_csv_path=mega_csv_path, phot_csv_path=phot_csv_path)

    if hostnames is None:
        import pandas as pd

        mega_df = pd.read_csv(mega_csv_path)
        hostnames = [str(h) for h in mega_df["hostname"].dropna().unique().tolist()]

    fits: list[FitResultSchema] = []
    failures: list[tuple[str, str]] = []

    for hostname in hostnames:
        try:
            fit, _ = fit_best_params(hostname=hostname, verbose=verbose, **fit_kwargs)
            fits.append(fit)
        except Exception as exc:
            if not continue_on_error:
                raise
            failures.append((hostname, str(exc)))
            if verbose:
                print(f"[{hostname}] fit failed: {exc}")

    if fits:
        save_fit_results_to_csv(fits, output_csv=output_csv)

    if verbose:
        print(f"Completed fits: {len(fits)} success, {len(failures)} failed")
        if fits:
            print(f"Saved successful fits to: {output_csv}")

    return fits, failures
