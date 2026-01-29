import ezpadova
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

class IsochroneFetcher:
    def __init__(self,
                 photsys='gaiaEDR3',
                 step_age:float=0.01,
                 step_mh: float=0.01,):
        self.photsys = photsys
        self.step_age = float(step_age)
        self.step_mh = float(step_mh)
        self._cache = {}

        self.bp_candidates = ['bp', 'g_bp', 'gbp', 'phot_bp_mean_mag']
        self.rp_candidates = ['rp', 'g_rp', 'grp', 'phot_rp_mean_mag']
        self.g_candidates  = ['g ', ' g', 'gmag', 'phot_g_mean_mag', 'g_band', 'g']

    def _ensure_df(self, obj):
        #ezpadova returns a list of df
        if isinstance(obj, list):
            print("Concatenated list into DataFrame")
            return pd.concat(obj, ignore_index=True)
        return obj

    def _find_col(self, df: pd.DataFrame, candidates):
        for cand in candidates:
            for c in df.columns:
                if cand.lower() in c.lower():
                    return c
        return None

    def _norm_triplet(self,
                      val: Optional[float],
                      step: float) -> Optional[Tuple[float, float, float]]:
        if val is None:
            return None
        if isinstance(val, (list, tuple, np.ndarray)):
            return tuple(val)
        return float(val), float(val), float(step)

    def _cache_key(self, logage_triplet, mh_triplet):
        return (tuple(logage_triplet) if logage_triplet is not None else None,
                tuple(mh_triplet) if mh_triplet is not None else None,
                self.photsys)

    def fetch(self, logage: float, MH: float) -> pd.DataFrame:
        logage_t = self._norm_triplet(logage, self.step_age)
        mh_t = self._norm_triplet(MH, self.step_mh)
        key = self._cache_key(logage_t, mh_t)
        if key in self._cache:
            return self._cache[key]

        raw = ezpadova.get_isochrones(logage=logage_t, MH=mh_t, photsys_file=self.photsys)
        df = self._ensure_df(raw)

        self._cache[key] = df
        return df

    def photometry(self, df: pd.DataFrame):
        bp_col = self._find_col(df, self.bp_candidates)
        rp_col = self._find_col(df, self.rp_candidates)
        g_col  = self._find_col(df, self.g_candidates)

        if not (bp_col and rp_col and g_col):
            raise RuntimeError('Could not locate Gaia BP/RP/G columns in isochrone data')

        color = df[bp_col] - df[rp_col]
        mag = df[g_col]
        return color, mag, (bp_col, rp_col, g_col)

class IsochronePlotter:
    def __init__(self, fetcher: IsochroneFetcher):
        self.fetcher = fetcher

    def plot(self, logage1, MH1, label='A'):


        df1 = self.fetcher.fetch(logage1, MH1)
        #df2 = self.fetcher.fetch(logage2, MH2)
        c1, m1, _ = self.fetcher.photometry(df1)
        #c2, m2, _ = self.fetcher.photometry(df2)

        plt.plot(c1, m1, '.', ms=4, color='C0', label=label)
        #plt.plot(c2, m2, '.', ms=4, color='C1', label=f'{labels[1]}')
        plt.gca().invert_yaxis()
        plt.xlabel('BP-RP')
        plt.ylabel('G (mag)')
        plt.grid()
        return plt.gcf()




# #initialize fetcher and plotter
# fetcher = IsochroneFetcher(photsys='gaiaEDR3', step_age=0.1, step_mh=0.1)
# plotter= IsochronePlotter(fetcher)
#
# #1 ) The 1 Gyr vs. 5 Gyr Isochrones (log10 ages)
# age1_log = np.log10(1e9) #9.0
# age5_log = np.log10(5e9) #9.7ish
# fig = plotter.plot(age1_log, 0.0, age5_log, 0.0, labels=('1 Gyr', '5 Gyr'))
# fig.suptitle('Isochrones at 1 Gyr and 5 Gyr, [M/H]=0.0')
# plt.show()
#
# #2 )  0.0 isochrone vs. a [Fe/H] = -0.5 isochrone
#
# age_log = age5_log #just picking a fixed age
# fig2 = plotter.plot(age_log, 0.0, age_log, -0.5, labels=('[Fe/H]=0.0', '[Fe/H]=-0.5'))
# fig2.suptitle('Metallicity effect on 5 Gyr Isochrone')
# plt.show()