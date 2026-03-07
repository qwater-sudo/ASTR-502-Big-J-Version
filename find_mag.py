from fetch_iso import IsochroneFetcher, IsochronePlotter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Optional, Iterable


class PhotometryMerger:
    GAIA_G_CANDS = ['gaiaGmag']
    GAIA_BP_CANDS = ['gaiaBPmag']
    GAIA_RP_CANDS = ['gaiaRPmag']
    PARALLAX_CANDS = ['parallax', 'plx', 'parallax_mas']
    DIST_CANDS = ['dist_pc']

    def __init__(self,
                 extra_join_candidates: Optional[Iterable[str]] = None):
        self.extra_join_candidates = list(extra_join_candidates or [])

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
        for cand in candidates:
            for c in df.columns:
                if cand.lower() in c.lower():
                    return c
        return None


    def _join_key(self,
                  df1: pd.DataFrame,
                  df2: pd.DataFrame) -> Optional[str]:
        common = set(df1.columns).intersection(df2.columns)
        id_phot = 'gaia_dr3_id_phot'
        id_dist = 'gaia_dr3_id_dist'

        if common:
            for pref in ('source_id', 'id', 'star_id', 'obj_id', 'object_id'):
                if pref in common:
                    return pref
            return sorted(common)[0]
        candidates = (list(self.extra_join_candidates) +
                      ['source_id','id','star_id','name','objid','gaia_source_id','sourceid'])
        for cand in candidates:
            if cand in df1.columns and cand in df2.columns:
                return cand
        return None

    @staticmethod

    def _abs_mag_series(apparent: pd.Series, distance_pc: pd.Series) -> pd.Series:
        # apparent may be non-numeric -> coerce; distance must be positive
        a = pd.to_numeric(apparent, errors='coerce')
        d = pd.to_numeric(distance_pc, errors='coerce')
        valid = d > 0
        abs_mag = pd.Series(np.nan, index=a.index, dtype=float)
        abs_mag[valid] = a[valid] - 5.0 * np.log10(d[valid]) + 5.0
        return abs_mag


    def join_photometry_and_distances(self,
                                      phot_csv: Path | str,
                                      dist_csv: Path | str,
                                      on: Optional[str] = None,
                                      how: str = 'inner') -> pd.DataFrame:
        phot_df = pd.read_csv(phot_csv)
        dist_df = pd.read_csv(dist_csv)

        if on is None:
            on = self._join_key(phot_df, dist_df)
            if on is None:
                raise ValueError('Could not find a join key. Provide a func')

        joined = phot_df.merge(dist_df, on=on, how=how, suffixes=('_phot', '_dist'))

        dist_col = self._find_col(dist_df, self.DIST_CANDS)
        if dist_col is None:
            raise ValueError('Could not find a distance column in the distance CSV')


        g_col = self._find_col(joined, self.GAIA_G_CANDS)
        bp_col = self._find_col(joined, self.GAIA_BP_CANDS)
        rp_col = self._find_col(joined, self.GAIA_RP_CANDS)

        # compute absolute mags where possible
        if g_col is not None:
            joined['G_abs'] = self._abs_mag_series(joined[g_col], joined[dist_col])
        else:
            joined['G_abs'] = np.nan

        if bp_col is not None:
            joined['BP_abs'] = self._abs_mag_series(joined[bp_col], joined[dist_col])
        else:
            joined['BP_abs'] = np.nan

        if rp_col is not None:
            joined['RP_abs'] = self._abs_mag_series(joined[rp_col], joined[dist_col])
        else:
            joined['RP_abs'] = np.nan

        # convenience color column
        if 'BP_abs' in joined.columns and 'RP_abs' in joined.columns:
            joined['BP_RP_abs'] = joined['BP_abs'] - joined['RP_abs']
        return joined

    @staticmethod
    def interpolation_bands(df: pd.DataFrame) -> list[str]:
        candidates = ['G_abs', 'BP_abs', 'RP_abs', 'BP_RP_abs']
        return [c for c in candidates if c in df.columns]

# merge = PhotometryMerger()
# phot_csv_file = '/Users/archon/classes/ASTR_502/Astro502_Sp26/ASTR502_Master_Photometry_List.csv'
# dist_csv_file = '/Users/archon/classes/ASTR_502/Astro502_Sp26/ASTR502_Mega_Target_List.csv'
# df = merge.join_photometry_and_distances(phot_csv_file, dist_csv_file)
# print(df)
# def plot(pd: pd.DataFrame):
#     fig, ax = plt.subplots(figsize=(8,10))
#     ax.scatter(pd['BP_RP_abs'], pd['G_abs'], s=1, color='black', label='Target List')
#
# fetcher = IsochroneFetcher(photsys='gaiaEDR3', step_age=0.1, step_mh=0.1)
# plotter= IsochronePlotter(fetcher)
# plot(df)
#
# fig = plotter.plot(np.log10(1e9), 0.0, label='Isochrone')
# plt.legend()
# plt.show()
