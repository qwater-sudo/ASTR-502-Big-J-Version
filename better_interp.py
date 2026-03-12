import numpy as np
from scipy.interpolate import RegularGridInterpolator
from isochrones import get_ichrone
import astropy.units as u

try:
    from synphot.reddening import ReddeningLaw
except ImportError:
    ReddeningLaw = None

_REQUESTED_BANDS = ("G", "BP", "RP", "J", "H", "K", "W1", "W2", "W3", "W4", "g", "r", "i", "z")

_BAND_COLUMNS = {
    "G": "G_mag", "BP": "BP_mag", "RP": "RP_mag",
    "J": "J_mag", "H": "H_mag", "K": "K_mag",
    "W1": "W1_mag", "W2": "W2_mag", "W3": "W3_mag", "W4": "W4_mag",
    "g": "g_mag", "r": "r_mag", "i": "i_mag", "z": "z_mag",
}

_MIST = None
_INTERPOLATORS = None
_GRIDS = None
_ACTIVE_BANDS = None

_BAND_EFFECTIVE_WAVELENGTH_ANGSTROM = {
    "G": 6730.0,
    "BP": 5320.0,
    "RP": 7970.0,
    "J": 12350.0,
    "H": 16620.0,
    "K": 21590.0,
    "W1": 33526.0,
    "W2": 46028.0,
    "W3": 115608.0,
    "W4": 220883.0,
    "g": 4770.0,
    "r": 6231.0,
    "i": 7625.0,
    "z": 9134.0,
}


def _get_mist():
    global _MIST
    if _MIST is None:
        _MIST = get_ichrone("mist", bands=list(_REQUESTED_BANDS))
        _MIST.initialize()
    return _MIST


def _select_rows(iso):
    """
    Keep pre-MS, MS, subgiant, and RGB phases.
    Using a wider range of phases ensures stars like WASP-96
    don't 'fall off' the grid at old ages.
    """
    # Phase 0=MS, 2=Subgiant, 3=RGB. Including 3 is critical for older stars.
    selected = iso[iso["phase"].isin([-1, 0, 2, 3])].copy()

    # Use 'initial_mass' if available, otherwise 'mass'
    m_col = "initial_mass" if "initial_mass" in selected.columns else "mass"
    selected = selected.sort_values(m_col).reset_index(drop=True)

    return selected


def _build_interpolators(age_grid=None, feh_grid=None, mass_points=300):
    global _ACTIVE_BANDS
    mist = _get_mist()

    if age_grid is None:
        age_grid = np.logspace(np.log10(1e6), np.log10(13.8e9), 60)
    if feh_grid is None:
        feh_grid = np.linspace(-1.0, 0.5, 31)

    mass_grid = np.linspace(0.1, 3.0, mass_points)

    iso0 = mist.isochrone(age=9.0, feh=0.0)
    available_cols = set(iso0.columns)
    _ACTIVE_BANDS = [b for b in _REQUESTED_BANDS if _BAND_COLUMNS.get(b) in available_cols]

    magnitude_grids = {
        band: np.full((mass_grid.size, age_grid.size, feh_grid.size), np.nan)
        for band in _ACTIVE_BANDS
    }

    for age_index, age in enumerate(age_grid):
        for feh_index, feh in enumerate(feh_grid):
            try:
                iso = mist.isochrone(age=np.log10(age), feh=feh)
            except Exception:
                continue

            selected = _select_rows(iso)
            if len(selected) < 2:
                continue

            m_col = "initial_mass" if "initial_mass" in selected.columns else "mass"
            masses = selected[m_col].to_numpy()

            for band in _ACTIVE_BANDS:
                col = _BAND_COLUMNS[band]
                values = selected[col].to_numpy()

                # Interpolate valid range
                in_range = (mass_grid >= masses[0]) & (mass_grid <= masses[-1])
                magnitude_grids[band][in_range, age_index, feh_index] = np.interp(
                    mass_grid[in_range], masses, values
                )

                # CRITICAL: Fill "Dead Star" points.
                # If a star is too massive to exist at this age, we assign it
                # the magnitude of the last valid point (the tip of the RGB).
                too_massive = (mass_grid > masses[-1])
                magnitude_grids[band][too_massive, age_index, feh_index] = values[-1]

                # Fill low-mass points (below 0.1 Msun) with the smallest available star
                too_light = (mass_grid < masses[0])
                magnitude_grids[band][too_light, age_index, feh_index] = values[0]

    # Final pass to fill any remaining NaN holes in the 3D cube
    for band in _ACTIVE_BANDS:
        grid = magnitude_grids[band]
        mask = np.isnan(grid)
        if np.any(mask):
            # This handles cases where entire age/feh slices failed
            # We fill them using the nearest valid age/feh slice
            from scipy.ndimage import distance_transform_edt
            idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
            magnitude_grids[band] = grid[tuple(idx)]

    interpolators = {
        band: RegularGridInterpolator(
            (mass_grid, age_grid, feh_grid),
            magnitude_grids[band],
            bounds_error=False,
            fill_value=None,  # Allow extrapolation
        )
        for band in _ACTIVE_BANDS
    }

    return interpolators, (mass_grid, age_grid, feh_grid)


def _get_interpolators():
    global _INTERPOLATORS, _GRIDS
    if _INTERPOLATORS is None:
        _INTERPOLATORS, _GRIDS = _build_interpolators()
    return _INTERPOLATORS, _GRIDS


def _get_band_extinction(av, rv=3.1, extinction_model="mwavg"):
    """
    Compute A_lambda (mag) for each supported band using synphot.
    """
    if av <= 0:
        return {b: 0.0 for b in _ACTIVE_BANDS}

    if ReddeningLaw is None:
        raise ImportError(
            "synphot is required for extinction support. Install it with `pip install synphot`."
        )

    law = ReddeningLaw.from_extinction_model(extinction_model)
    ebv = av / rv
    curve = law.extinction_curve(ebv)

    ext = {}
    for band in _ACTIVE_BANDS:
        lam = _BAND_EFFECTIVE_WAVELENGTH_ANGSTROM.get(band)
        if lam is None:
            ext[band] = 0.0
            continue

        trans = curve(lam * u.AA)
        trans_val = float(np.atleast_1d(trans.value)[0])
        trans_val = np.clip(trans_val, 1e-12, 1.0)
        ext[band] = -2.5 * np.log10(trans_val)

    return ext


def get_model_mag(mass, age, feh, av=0.0, rv=3.1, extinction_model="mwavg"):
    """
    Parameters
    ----------
    mass : float or array
        Stellar mass in solar masses.
    age : float or array
        Stellar age in years (e.g. 10e6 for 10 Myr, 3e9 for 3 Gyr).
    feh : float or array
        Metallicity [Fe/H] in dex.
    av : float
        V-band extinction in magnitudes.

    Returns
    -------
    dict mapping band name -> absolute magnitude (float or array)
    """

    interpolators, (mass_grid, age_grid, feh_grid) = _get_interpolators()

    mass_arr = np.asarray(mass, dtype=float)
    age_arr  = np.asarray(age,  dtype=float)
    feh_arr  = np.asarray(feh,  dtype=float)
    mass_arr, age_arr, feh_arr = np.broadcast_arrays(mass_arr, age_arr, feh_arr)

    if np.any(mass_arr < mass_grid[0]) or np.any(mass_arr > mass_grid[-1]):
        print(f"Warning: mass {mass} outside grid [{mass_grid[0]:.2f}, {mass_grid[-1]:.2f}] Msun")
    if np.any(age_arr < age_grid[0]) or np.any(age_arr > age_grid[-1]):
        print(f"Warning: age {age:.3e} outside grid [{age_grid[0]:.3e}, {age_grid[-1]:.3e}] yr")
    if np.any(feh_arr < feh_grid[0]) or np.any(feh_arr > feh_grid[-1]):
        print(f"Warning: feh {feh} outside grid [{feh_grid[0]:.2f}, {feh_grid[-1]:.2f}]")

    points = np.column_stack([mass_arr.ravel(), age_arr.ravel(), feh_arr.ravel()])

    outputs = {}
    extinction_by_band = _get_band_extinction(av, rv=rv, extinction_model=extinction_model)
    for band, interp in interpolators.items():
        outputs[band] = interp(points).reshape(mass_arr.shape) + extinction_by_band.get(band, 0.0)

    if mass_arr.shape == ():
        return {b: float(outputs[b]) for b in outputs}

    return outputs

import pandas as pd
from scipy.optimize import minimize

_MEGA = None
_PHOT = None

# Map observed photometry columns (Master_Photometry_List) -> your band names
_OBS_MAP = {
    "G":  "gaia_Gmag",
    "BP": "gaia_BPmag",
    "RP": "gaia_RPmag",
    "J":  "Jmag",
    "H":  "Hmag",
    "K":  "Kmag",
    "W1": "w1mag",
    "W2": "w2mag",
    "W3": "w3mag",
    "W4": "w4mag",
    "g":  "gmag",
    "r":  "rmag",
    "i":  "imag",
    "z":  "zmag",
}

def load_catalogs(mega_csv_path, phot_csv_path):
    """
    Call once in your notebook/script before fitting stars.
    """
    global _MEGA, _PHOT
    _MEGA = pd.read_csv(mega_csv_path)
    _PHOT = pd.read_csv(phot_csv_path)

def _apparent_to_absolute(m_app, d_pc):
    return m_app - 5.0 * np.log10(d_pc / 10.0)

def _get_star_rows(hostname):
    if _MEGA is None or _PHOT is None:
        raise RuntimeError("Catalogs not loaded. Call load_catalogs(mega_csv_path, phot_csv_path) first.")

    m = _MEGA[_MEGA["hostname"] == hostname]
    p = _PHOT[_PHOT["hostname"] == hostname]
    if len(m) == 0:
        raise KeyError(f"{hostname} not found in Mega_Target_List")
    if len(p) == 0:
        raise KeyError(f"{hostname} not found in Master_Photometry_List")
    return m.iloc[0], p.iloc[0]

def _get_star_obs_abs(hostname):
    """
    Returns:
      obs_abs: dict band -> absolute magnitude (float)
      d_pc: distance used
    """
    mrow, prow = _get_star_rows(hostname)

    d_pc = float(mrow["bj_dist_pc"])
    if not np.isfinite(d_pc) or d_pc <= 0:
        raise ValueError(f"{hostname}: invalid bj_dist_pc={d_pc}")

    obs_abs = {}
    for band, col in _OBS_MAP.items():
        if col in prow.index:
            val = prow[col]
            if np.isfinite(val):
                obs_abs[band] = float(_apparent_to_absolute(val, d_pc))

    if len(obs_abs) < 3:
        raise ValueError(f"{hostname}: only {len(obs_abs)} usable bands; need >= 3 for a stable fit.")

    return obs_abs, d_pc

def _get_param_prior(hostname, fallback_sigma=0.25):
    """
    Mega list provides st_age (Gyr) and age errors st_ageerr1 (upper), st_ageerr2 (lower, negative).
    Mega list in your file does NOT include mass/met errors, so those fall back to ±fallback_sigma.
    """
    mrow, _ = _get_star_rows(hostname)

    m0   = float(mrow["st_mass"]) if np.isfinite(mrow["st_mass"]) else np.nan
    a0   = float(mrow["st_age"])  if np.isfinite(mrow["st_age"])  else np.nan  # Gyr
    feh0 = float(mrow["st_met"])  if np.isfinite(mrow["st_met"])  else np.nan

    # fallback (symmetric)
    sig_m   = fallback_sigma          # Msun
    sig_feh = fallback_sigma          # dex

    # age: asymmetric if present, else fallback (Gyr)
    if ("st_ageerr1" in mrow.index) and ("st_ageerr2" in mrow.index) and np.isfinite(mrow["st_ageerr1"]) and np.isfinite(mrow["st_ageerr2"]):
        sig_age_hi = float(mrow["st_ageerr1"])            # +Gyr
        sig_age_lo = abs(float(mrow["st_ageerr2"]))       # make +Gyr
        if not (sig_age_hi > 0): sig_age_hi = fallback_sigma
        if not (sig_age_lo > 0): sig_age_lo = fallback_sigma
    else:
        sig_age_hi = fallback_sigma
        sig_age_lo = fallback_sigma

    return {
        "m0": m0, "a0_gyr": a0, "feh0": feh0,
        "sig_m": sig_m, "sig_feh": sig_feh,
        "sig_age_hi": sig_age_hi, "sig_age_lo": sig_age_lo
    }

def _chi2_phot(mass, log10_age, feh, av, obs_abs, sigma_phot=0.5):
    """
    Fixed per-band magnitude tolerance sigma_phot (mag).
    """
    age_yr = 10.0 ** log10_age
    model = get_model_mag(mass=mass, age=age_yr, feh=feh, av=av)

    chi2 = 0.0
    n = 0
    for band, Mobs in obs_abs.items():
        Mmod = model.get(band, np.nan)
        if not np.isfinite(Mmod):
            continue
        chi2 += ((Mobs - Mmod) / sigma_phot) ** 2
        n += 1

    if n == 0:
        return 1e30
    return chi2

def _chi2_prior(mass, log10_age, feh, prior):
    """
    Gaussian priors on mass, feh, and asymmetric Gaussian prior on age (in Gyr).
    This is what "uses Mega errors for mass/age/feh, fallback if missing" means in practice.
    """
    chi2 = 0.0

    # mass prior
    if np.isfinite(prior["m0"]):
        chi2 += ((mass - prior["m0"]) / prior["sig_m"]) ** 2

    # feh prior
    if np.isfinite(prior["feh0"]):
        chi2 += ((feh - prior["feh0"]) / prior["sig_feh"]) ** 2

    # age prior (asymmetric)
    if np.isfinite(prior["a0_gyr"]):
        age_gyr = (10.0 ** log10_age) / 1e9
        if age_gyr >= prior["a0_gyr"]:
            sig = prior["sig_age_hi"]
        else:
            sig = prior["sig_age_lo"]
        chi2 += ((age_gyr - prior["a0_gyr"]) / sig) ** 2

    return chi2

def fit_best_params(hostname,
                    sigma_phot=0.5,
                    fallback_sigma_param=0.25,
                    av_bounds=(0.0, 3.0),
                    bounds=None,
                    verbose=True):
    """
    Returns best-fit (mass, age_yr, feh, av) using:
      chi2_total = chi2_phot + chi2_prior

    sigma_phot: fixed per-band photometric uncertainty (mag)
    fallback_sigma_param: ± error used for mass & feh (and age if missing) when Mega doesn't provide errors
    """
    obs_abs, d_pc = _get_star_obs_abs(hostname)
    prior = _get_param_prior(hostname, fallback_sigma=fallback_sigma_param)

    # Initial guess from Mega (if missing, default)
    m0 = prior["m0"] if np.isfinite(prior["m0"]) else 1.0
    a0 = prior["a0_gyr"] if np.isfinite(prior["a0_gyr"]) else 5.0
    feh0 = prior["feh0"] if np.isfinite(prior["feh0"]) else 0.0
    x0 = np.array([m0, np.log10(a0 * 1e9), feh0, 0.0], dtype=float)

    # Bounds: (mass, log10_age_yr, feh, av)
    if bounds is None:
        bounds = [
            (0.1, 3.0),
            (6.0, np.log10(13.8e9)),
            (-1.0, 0.5),
            av_bounds,
        ]

    def obj(x):
        mass, log10_age, feh, av = x
        # quick physical guards
        if mass <= 0 or av < 0:
            return 1e30
        return (_chi2_phot(mass, log10_age, feh, av, obs_abs, sigma_phot=sigma_phot)
                + _chi2_prior(mass, log10_age, feh, prior))

    res = minimize(obj, x0=x0, bounds=bounds, method="L-BFGS-B")

    mass_b, log10_age_b, feh_b, av_b = res.x
    age_yr_b = 10.0 ** log10_age_b
    age_gyr_b = age_yr_b / 1e9

    if verbose:
        print(f"\n[{hostname}] Best-fit parameters (chi2_phot + chi2_prior)")
        print(f"  mass = {mass_b:.4f} Msun")
        print(f"  age  = {age_yr_b:.3e} yr  ({age_gyr_b:.4f} Gyr)")
        print(f"  feh  = {feh_b:.4f} dex")
        print(f"  Av   = {av_b:.4f} mag")
        print(f"  success = {res.success} | {res.message}")
        print(f"  chi2_total = {res.fun:.2f} | N_obs_bands = {len(obs_abs)}")
        print(f"  d_pc used = {d_pc:.3f}")

    return mass_b, age_yr_b, feh_b, av_b, res

def get_bestfit_model_mag_for_star(hostname,
                                  sigma_phot=0.5,
                                  fallback_sigma_param=0.25,
                                  av_bounds=(0.0, 3.0),
                                  bounds=None,
                                  verbose=True):
    """
    One-stop call:
      - finds best-fit mass/age/feh/Av for hostname
      - prints best-fit params
      - returns (best_params, model_mags_at_bestfit)
    """
    m, a_yr, feh, av, res = fit_best_params(
        hostname,
        sigma_phot=sigma_phot,
        fallback_sigma_param=fallback_sigma_param,
        av_bounds=av_bounds,
        bounds=bounds,
        verbose=verbose
    )
    mags = get_model_mag(mass=m, age=a_yr, feh=feh, av=av)
    if verbose:
        print(f"  get_bestfit_model_mag_for_star -> best-fit Av = {av:.4f} mag")
    return (m, a_yr, feh, av), mags

# # Example usage
# print("Results from interpolator and get_model_mag:")
# mags = get_model_mag(1.0, 4.6e9, 0.0122)
# for k in sorted(mags.keys()):
#     print(f"{k}: {mags[k]}")
#
# print("--------------------------------")
# print("Results from brute-force likelihood calculator:")

def _validate_observations(observed_mags, observed_errs):
    if observed_mags is None or observed_errs is None:
        raise ValueError("Both observed_mags and observed_errs are required.")

    bands = sorted(set(observed_mags) & set(observed_errs))
    if not bands:
        raise ValueError("No overlapping bands between observed_mags and observed_errs.")

    ERR_FLOOR = 0.02  # 0.02 mag systematic/model floor
    for band in bands:
        err = observed_errs[band]
        if err is None or err <= 0:
            raise ValueError(f"Non-positive error for band '{band}'.")
        # Apply floor in quadrature
        observed_errs[band] = np.hypot(err, ERR_FLOOR)
    return bands


def brute_force_likelihood(observed_mags, observed_errs):
    observed_errs = dict(observed_errs)  # copy
    bands = _validate_observations(observed_mags, observed_errs)
    interpolators, grids = _get_interpolators()
    mass_grid, age_grid, feh_grid = grids

    bands = _validate_observations(observed_mags, observed_errs)
    bands = [b for b in bands if b in interpolators]
    if not bands:
        raise ValueError("None of the observed bands are available in the interpolator grid.")

    m_grid, a_grid, f_grid = np.meshgrid(mass_grid, age_grid, feh_grid, indexing="ij")
    points = np.column_stack([m_grid.ravel(), a_grid.ravel(), f_grid.ravel()])

    # after you build points, and inside the loop you currently do chi2 += ...
    # instead, compute model mags for all bands first

    model_stack = []
    obs_stack = []
    sig_stack = []

    for band in bands:
        model_mag = interpolators[band](points)
        model_stack.append(model_mag)
        obs_stack.append(np.full_like(model_mag, observed_mags[band], dtype=float))
        sig_stack.append(np.full_like(model_mag, observed_errs[band], dtype=float))

    model_stack = np.vstack(model_stack)  # (nband, npts)
    obs_stack = np.vstack(obs_stack)
    sig_stack = np.vstack(sig_stack)

    valid = np.all(np.isfinite(model_stack), axis=0) & np.all(sig_stack > 0, axis=0)

    w = 1.0 / (sig_stack ** 2)

    return best_mass, best_age, best_feh

# # Example likelihood usage
# observed_mags = {"G": 4.67, "BP": 5.00, "RP": 4.21}
# observed_errs = {"G": 0.02, "BP": 0.02, "RP": 0.02}
# mass, age, feh= brute_force_likelihood(observed_mags, observed_errs)
# print("mass:", mass)
# print("age:", age)
# print("feh:", feh)





#-------------------------
#Code for grabbing a single isochrone rather than building a grid
#-------------------------

# import numpy as np
# from isochrones import get_ichrone
#
# _REQUESTED_BANDS = ("G", "BP", "RP", "J", "H", "K", "W1", "W2", "W3", "W4", "g", "r", "i", "z")
#
# _BAND_COLUMNS = {
#     "G": "G_mag", "BP": "BP_mag", "RP": "RP_mag",
#     "J": "J_mag", "H": "H_mag", "K": "K_mag",
#     "W1": "W1_mag", "W2": "W2_mag", "W3": "W3_mag", "W4": "W4_mag",
#     "g": "g_mag", "r": "r_mag", "i": "i_mag", "z": "z_mag",
# }
#
# _MIST = None
#
#
# def _get_mist():
#     global _MIST
#     if _MIST is None:
#         _MIST = get_ichrone("mist", bands=list(_REQUESTED_BANDS))
#         _MIST.initialize()
#     return _MIST
#
#
# def _select_rows(iso):
#     """
#     Keep pre-MS (phase=-1), MS (phase=0), and early subgiant (phase=2)
#     rows, truncated at the first point where mass turns over.
#     """
#     selected = iso[iso["phase"].isin([-1, 0, 2])].copy()
#     selected = selected.sort_values("eep").reset_index(drop=True)
#     masses = selected["mass"].to_numpy()
#     cutoff = len(masses)
#     for i in range(1, len(masses)):
#         if masses[i] < masses[i - 1]:
#             cutoff = i
#             break
#     return selected.iloc[:cutoff]
#
#
# def get_model_mag(mass, age, feh):
#     """
#     Parameters
#     ----------
#     mass : float
#         Stellar mass in solar masses.
#     age : float
#         Stellar age in years (e.g. 10e6 for 10 Myr, 3e9 for 3 Gyr).
#     feh : float
#         Metallicity [Fe/H] in dex.
#
#     Returns
#     -------
#     dict mapping band name -> absolute magnitude (float)
#     """
#     mist = _get_mist()
#
#     # Fetch the single isochrone at this age and metallicity
#     iso = mist.isochrone(age=np.log10(age), feh=feh)
#     selected = _select_rows(iso)
#
#     if len(selected) < 2:
#         return {b: np.nan for b in _BAND_COLUMNS}
#
#     masses = selected["mass"].to_numpy()
#     sort_idx = np.argsort(masses)
#     masses_sorted = masses[sort_idx]
#
#     # Deduplicate
#     _, unique_idx = np.unique(masses_sorted, return_index=True)
#     masses_sorted = masses_sorted[unique_idx]
#
#     if mass < masses_sorted[0] or mass > masses_sorted[-1] + 0.1:
#         print(f"Warning: mass {mass} outside isochrone range "
#               f"[{masses_sorted[0]:.3f}, {masses_sorted[-1]:.3f}] Msun for "
#               f"age={age:.3e}, feh={feh:.2f}")
#
#     results = {}
#     for band, col in _BAND_COLUMNS.items():
#         if col not in selected.columns:
#             results[band] = np.nan
#             continue
#
#         values = selected[col].to_numpy()[sort_idx][unique_idx]
#
#         if mass <= masses_sorted[-1]:
#             # Standard interpolation within the isochrone mass range
#             results[band] = float(np.interp(mass, masses_sorted, values,
#                                             left=np.nan, right=np.nan))
#         elif mass <= masses_sorted[-1] + 0.1:
#             # Linear extrapolation up to 0.1 Msun beyond the upper boundary
#             dm = masses_sorted[-1] - masses_sorted[-2]
#             dv = values[-1] - values[-2]
#             slope = dv / dm if dm != 0 else 0.0
#             results[band] = float(values[-1] + slope * (mass - masses_sorted[-1]))
#         else:
#             results[band] = np.nan
#
#     return results
