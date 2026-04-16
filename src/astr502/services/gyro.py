from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from statistics import median

from astr502.data.paths import KEPLER_AGES, OUTPUT_RESULTS_DIR


def _extract_first_float(raw: str | None) -> float | None:
    """Extract the first float from a value that may look like '[4.07 4.07]'."""
    if raw is None:
        return None

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(raw))
    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def _latest_candidate_results_file(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("interpolate_*_candidate_fits.csv"))
    if not candidates:
        raise FileNotFoundError(f"No candidate fits CSV found in {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_kepler_ages(kepler_catalog_csv: Path = KEPLER_AGES) -> dict[str, float]:
    tic_to_age_gyr: dict[str, float] = {}
    with kepler_catalog_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tic_id = re.sub(r"^TIC\s*", "", (row.get("tic_ids") or "").strip(), flags=re.IGNORECASE)
            age_gyr = _extract_first_float(row.get("st_age"))
            if not tic_id or age_gyr is None or not math.isfinite(age_gyr):
                continue

            tic_to_age_gyr[tic_id] = age_gyr

    return tic_to_age_gyr


def compare_gyro_ages(results_csv: Path | None = None) -> None:
    """Compare fitted ages in outputs/results against Kepler catalog ages."""
    results_path = results_csv or _latest_candidate_results_file(OUTPUT_RESULTS_DIR)
    kepler_age_by_tic = _load_kepler_ages()

    matched_rows: list[tuple[str, str, float, float, float]] = []
    unmatched_stars: list[str] = []

    with results_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            hostname = (row.get("hostname") or "").strip()
            age_yr = _extract_first_float(row.get("age_yr"))
            tic_id = (row.get("tic_id") or row.get("tic_ids") or "").strip()
            if not hostname or age_yr is None or not math.isfinite(age_yr):
                continue

            fit_age_gyr = age_yr / 1e9
            kepler_age_gyr = kepler_age_by_tic.get(tic_id or "")

            if not tic_id or kepler_age_gyr is None:
                unmatched_stars.append(hostname)
                continue

            delta_gyr = fit_age_gyr - kepler_age_gyr
            matched_rows.append((hostname, tic_id, fit_age_gyr, kepler_age_gyr, delta_gyr))

    print(f"Results file: {results_path}")
    print(f"Matched stars: {len(matched_rows)}")
    print(f"Unmatched stars: {len(unmatched_stars)}")

    if not matched_rows:
        print("No overlapping stars were found between results and kepler_star_ages.")
        return

    abs_errors = [abs(delta) for *_, delta in matched_rows]
    mean_abs_err = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(delta **2 for *_, delta in matched_rows) / len(matched_rows))
    med_abs_err = median(abs_errors)

    print("\nAlignment summary (fit age - Kepler st_age, in Gyr):")
    print(f"  Mean absolute error:   {mean_abs_err:.3f}")
    print(f"  Median absolute error: {med_abs_err:.3f}")
    print(f"  RMSE:                  {rmse:.3f}")

    print("\nSample comparisons (up to first 15 rows):")
    for hostname, tic_id, fit_age, kepler_age, delta in matched_rows[:15]:
        print(
            f"  {hostname:25s} {tic_id:12s} "
            f"fit={fit_age:6.3f} Gyr  kepler={kepler_age:6.3f} Gyr  "
            f"delta={delta:+6.3f} Gyr"
        )


if __name__ == "__main__":
    compare_gyro_ages()
