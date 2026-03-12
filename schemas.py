from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class FitResultSchema:
    """Serializable schema for one star's best-fit interpolation result."""

    hostname: str
    mass: float
    age_yr: float
    feh: float
    av: float
    chi2_phot: float
    chi2_prior: float
    chi2_total: float
    distance_pc: float
    model_magnitudes: Mapping[str, float] = field(default_factory=dict)

    def to_record(self) -> dict[str, float | str]:
        record: dict[str, float | str] = {
            "hostname": self.hostname,
            "mass": self.mass,
            "age_yr": self.age_yr,
            "feh": self.feh,
            "av": self.av,
            "chi2_phot": self.chi2_phot,
            "chi2_prior": self.chi2_prior,
            "chi2": self.chi2_total,
            "distance_pc": self.distance_pc,
        }
        for band, mag in self.model_magnitudes.items():
            record[f"model_{band}"] = mag
        return record
