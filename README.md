# ASTR-502

Repository for ASTR 502 stellar isochrone interpolation and target fitting workflows.

## Current structure

```text
src/astr502/
  domain/
    schemas.py            # fit result dataclasses
    stats.py              # chi-square terms
  data/
    catalogs.py           # catalog loading + priors + observed magnitudes
    utils.py              # SPOT file discovery + helpers
    readers/
      read_spot_models.py
      read_mist_models.py
  modeling/
    extinction.py         # extinction per photometric band
    interpolate.py        # interpolation + optimizer core
  services/
    fit_runtime.py        # single-target and batch runtime wrappers

scripts/
  fit_single_star.py      # CLI for one hostname
  fit_target_list.py      # CLI for full/subset runs
  fetch_iso.py            # ezpadova helper + plotting utility
  find_mag.py             # photometry/distance merge utility

data/raw/isochrones/
  MIST/
  SPOTS/
  PARSEC/
```

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a single-star fit:

```bash
python scripts/fit_single_star.py <hostname> --mega-csv <mega.csv> --phot-csv <phot.csv>
python scripts/fit_single_star.py <hostname> --mega-csv data\raw\catalogs\ASTR502_Mega_Target_List.csv --phot-csv data\raw\catalogs\ASTR502_Master_Photometry_List.csv
'''

Run a target-list fit:

```bash
python scripts/fit_target_list.py --mega-csv <mega.csv> --phot-csv <phot.csv>
python scripts/fit_target_list.py --mega-csv data\raw\catalogs\ASTR502_Mega_Target_List.csv --phot-csv data\raw\catalogs\ASTR502_Master_Photometry_List.csv
```

> Note: the default CSV paths resolve to `data/raw/catalogs/...` and can also be overridden via
> `ASTR502_MEGA_CSV` and `ASTR502_PHOT_CSV` environment variables.

Instructor: Dr. Andrew Mann

Repository owner: James Atkisson, Caiden Ray

Email: atk@unc.edu