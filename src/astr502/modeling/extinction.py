from __future__ import annotations

import numpy as np

try:
    from synphot.reddening import ReddeningLaw
    import astropy.units as u
except ImportError:  # optional dependency
    ReddeningLaw = None
    u = None

BAND_EFFECTIVE_WAVELENGTH_ANGSTROM = {
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


def get_band_extinction(
    bands: list[str],
    av: float,
    rv: float = 3.1,
    extinction_model: str = "mwavg",
) -> dict[str, float]:
    if av <= 0:
        return {b: 0.0 for b in bands}

    if ReddeningLaw is None or u is None:
        raise ImportError("synphot and astropy are required for extinction support")

    law = ReddeningLaw.from_extinction_model(extinction_model)
    ebv = av / rv
    curve = law.extinction_curve(ebv)

    ext = {}
    for band in bands:
        lam = BAND_EFFECTIVE_WAVELENGTH_ANGSTROM.get(band)
        if lam is None:
            ext[band] = 0.0
            continue
        trans = curve(lam * u.AA)
        trans_val = float(np.atleast_1d(trans.value)[0])
        trans_val = np.clip(trans_val, 1e-12, 1.0)
        ext[band] = -2.5 * np.log10(trans_val)
    return ext
