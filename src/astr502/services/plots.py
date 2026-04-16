from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

from astr502.data.paths import DEFAULT_MEGA_CSV_PATH, OUTPUT_FIGS_DIR, OUTPUT_RESULTS_DIR
from astr502.data.utils import LoggingUtils

logger = logging.getLogger(__name__)


def _save_age_scatter(
    output_path: Path,
    x_values: np.ndarray,
    y_values: np.ndarray,
    reduced_chi2: np.ndarray,
    *,
    title: str,
    chi2_close_tolerance: float,
    colorbar_limits: tuple[float, float],
) -> None:
    """Render and save one age scatter plot with consistent styling."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    vmin, vmax = colorbar_limits
    reduced_for_color = np.ma.masked_invalid(reduced_chi2)

    chi2_cmap = LinearSegmentedColormap.from_list(
        "chi2_blue_green_red",
        [(0.0, "#1f77b4"), (0.5, "#2ca02c"), (1.0, "#d62728")],
    )
    chi2_cmap.set_bad(color="0.65")
    chi2_cmap.set_under("#1f77b4")
    chi2_cmap.set_over("#d62728")

    finite_reduced = reduced_chi2[np.isfinite(reduced_chi2)]
    n_blue = int(np.sum(finite_reduced < 1.0 - chi2_close_tolerance))
    n_green = int(
        np.sum(
            (finite_reduced >= 1.0 - chi2_close_tolerance)
            & (finite_reduced <= 1.0 + chi2_close_tolerance)
        )
    )
    n_red = int(np.sum(finite_reduced > 1.0 + chi2_close_tolerance))

    scatter = ax.scatter(
        x_values,
        y_values,
        c=reduced_for_color,
        cmap=chi2_cmap,
        norm=TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax),
        s=24,
        alpha=0.85,
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(r"Reduced $\chi^2$")
    cbar.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0])

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1f77b4",
            label=fr"Blue: $\chi^2_{{red}}$ < {1.0 - chi2_close_tolerance:.1f} (n={n_blue})",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#2ca02c",
            label=(
                fr"Green: |$\chi^2_{{red}}$-1| ≤ {chi2_close_tolerance:.1f} "
                f"(n={n_green})"
            ),
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d62728",
            label=fr"Red: $\chi^2_{{red}}$ > {1.0 + chi2_close_tolerance:.1f} (n={n_red})",
            markersize=7,
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=9)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r"$(Age_{obs} - Age_{table}) / Age_{table}$")
    ax.set_ylabel("Age_table (Gyr)")
    ax.set_title(title)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info("Saved age scatter plot to %s", output_path)


def plot_observed_vs_table_age_scatter(
    catalog_csv: str | Path = DEFAULT_MEGA_CSV_PATH,
    observed_csv: str | Path = OUTPUT_RESULTS_DIR / "interpolate_best_fit_results.csv",
    output_path: str | Path | None = None,
) -> Path:
    """Plot fractional age residuals vs. table age for targets with both age values.

    The x-axis is (Age_obs - Age_table) / Age_table, where:
      - Age_obs is read from `age_yr` in the observed results file and converted to Gyr.
      - Age_table is read from `st_age` in the mega target list (already in Gyr).

    The y-axis is Age_table.
    """

    catalog_csv = Path(catalog_csv)
    observed_csv = Path(observed_csv)
    output_path = (
        Path(output_path)
        if output_path is not None
        else LoggingUtils.timestamped_output_path(
            output_dir=OUTPUT_FIGS_DIR,
            suffix="age_obs_vs_table_scatter.png",
        )
    )

    table_age_by_host: dict[str, float] = {}
    with catalog_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            hostname = (row.get("hostname") or "").strip()
            st_age_raw = (row.get("st_age") or "").strip()
            if not hostname or not st_age_raw:
                continue
            try:
                st_age = float(st_age_raw)
            except ValueError:
                continue
            if st_age <= 0:
                continue
            table_age_by_host[hostname] = st_age

    x_values: list[float] = []
    y_values: list[float] = []
    reduced_chi2_values: list[float] = []

    with observed_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            hostname = (row.get("hostname") or "").strip()
            age_obs_raw = (row.get("age_yr") or "").strip()
            chi2_raw = (row.get("chi2_reduced") or "").strip()

            if not hostname or not age_obs_raw or hostname not in table_age_by_host:
                continue

            try:
                age_obs_yr = float(age_obs_raw)
                chi2 = float(chi2_raw)
            except ValueError:
                continue
            if age_obs_yr <= 0:
                continue

            age_table_gyr = table_age_by_host[hostname]
            age_obs_gyr = age_obs_yr / 1e9
            fractional_residual = (age_obs_gyr - age_table_gyr) / age_table_gyr

            reduced_chi2_values.append(chi2)


            x_values.append(fractional_residual)
            y_values.append(age_table_gyr)

    if not x_values:
        raise ValueError(
            "No overlapping valid targets found between catalog 'st_age' and results 'age_yr' columns."
        )

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    reduced = np.asarray(reduced_chi2_values, dtype=float)

    # Keep color scale anchored near reduced-chi2=1 without clipping/removing data.
    # Out-of-range values map to the end colors of the colormap.
    vmin, vmax = 0.0, 2.0
    chi2_close_tolerance = 0.5
    blue_green_mask = reduced <= 1.0 + chi2_close_tolerance
    red_mask = reduced > 1.0 + chi2_close_tolerance

    # 1) Full scatter (existing behavior)
    _save_age_scatter(
        output_path=output_path,
        x_values=x,
        y_values=y,
        reduced_chi2=reduced,
        title="Observed vs Table Age (point color = reduced $\\chi^2$)",
        chi2_close_tolerance=chi2_close_tolerance,
        colorbar_limits=(vmin, vmax),
    )

    # 2) Additional scatter with blue+green points only
    blue_green_path = output_path.with_name(f"{output_path.stem}_blue_green{output_path.suffix}")
    _save_age_scatter(
        output_path=blue_green_path,
        x_values=x[blue_green_mask],
        y_values=y[blue_green_mask],
        reduced_chi2=reduced[blue_green_mask],
        title="Observed vs Table Age (blue+green reduced $\\chi^2$ only)",
        chi2_close_tolerance=chi2_close_tolerance,
        colorbar_limits=(vmin, vmax),
    )

    # 3) Additional scatter with red points only
    red_path = output_path.with_name(f"{output_path.stem}_red{output_path.suffix}")
    _save_age_scatter(
        output_path=red_path,
        x_values=x[red_mask],
        y_values=y[red_mask],
        reduced_chi2=reduced[red_mask],
        title="Observed vs Table Age (red reduced $\\chi^2$ only)",
        chi2_close_tolerance=chi2_close_tolerance,
        colorbar_limits=(vmin, vmax),
    )

    return output_path


if __name__ == "__main__":
    saved_path = plot_observed_vs_table_age_scatter()
    print(f"Saved plot to: {saved_path}")
