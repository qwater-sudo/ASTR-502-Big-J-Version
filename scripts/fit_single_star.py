#!/usr/bin/env python3
from __future__ import annotations

import argparse

from astr502.services.fit_runtime import fit_single_star_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a single target star.")
    parser.add_argument("hostname", help="Host name from the catalog")
    parser.add_argument("--mega-csv", default=None, help="Path to Mega target list CSV")
    parser.add_argument("--phot-csv", default=None, help="Path to photometry CSV")
    parser.add_argument("--output-csv", default=None, help="Optional one-row output CSV")
    parser.add_argument("--model-type", default="parsec", choices=["spot", "parsec"])
    parser.add_argument("--sigma-phot", type=float, default=0.5)
    parser.add_argument("--fallback-sigma-param", type=float, default=0.25)
    parser.add_argument("--av-min", type=float, default=0.0)
    parser.add_argument("--av-max", type=float, default=3.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {
        "sigma_phot": args.sigma_phot,
        "fallback_sigma_param": args.fallback_sigma_param,
        "av_bounds": (args.av_min, args.av_max),
        "model_type": args.model_type,
        "verbose": not args.quiet,
    }
    runtime_kwargs = {
        "hostname": args.hostname,  # ← use the CLI argument directly,
        "output_csv": args.output_csv,
        **kwargs,
    }
    if args.mega_csv:
        runtime_kwargs["mega_csv_path"] = args.mega_csv
    if args.phot_csv:
        runtime_kwargs["phot_csv_path"] = args.phot_csv

    fit = fit_single_star_runtime(**runtime_kwargs)
    if args.quiet:
        print(f"{fit.hostname},{fit.mass:.5f},{fit.age_yr:.5e},{fit.feh:.4f},{fit.av:.4f},{fit.chi2_total:.3f}")


if __name__ == "__main__":
    main()
