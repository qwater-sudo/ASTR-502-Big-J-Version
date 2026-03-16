#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.astr502.services.fit_runtime import fit_target_list_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit all or a subset of target stars.")
    parser.add_argument("--mega-csv", default=None, help="Path to Mega target list CSV")
    parser.add_argument("--phot-csv", default=None, help="Path to photometry CSV")
    parser.add_argument("--hostnames", nargs="*", default=None, help="Optional list of hostnames to fit")
    parser.add_argument("--output-csv", default="outputs/results/interpolate_best_fit_results.csv")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first failed target")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runtime_kwargs = {
        "hostnames": args.hostnames,
        "output_csv": args.output_csv,
        "continue_on_error": not args.stop_on_error,
        "verbose": not args.quiet,
    }
    if args.mega_csv:
        runtime_kwargs["mega_csv_path"] = args.mega_csv
    if args.phot_csv:
        runtime_kwargs["phot_csv_path"] = args.phot_csv

    fits, failures = fit_target_list_runtime(**runtime_kwargs)
    print(f"success={len(fits)} failures={len(failures)} output={args.output_csv}")


if __name__ == "__main__":
    main()
