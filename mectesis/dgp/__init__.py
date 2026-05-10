"""Data Generating Processes (DGP) module."""

from .base import BaseDGP
from .ar import AR1
from .rw import RandomWalk
from .ar_trend import AR1WithTrend
from .seasonal import SeasonalDGP
from .ar_break import AR1WithBreak
from .garch import AR1ARCH, AR1GARCH, PureGARCH, AR1GJRGARCH, AR1EGARCH
from .markov_switching import MarkovSwitchingAR
from .ets_dgps import (
    LocalLevelDGP, LocalTrendDGP, DampedTrendDGP,
    DeterministicSeasonalDGP, SeasonalRandomWalkDGP, LocalLevelSeasonalDGP,
)
from .var_dgp import VARDGP, VARGARCHDiagonalDGP
from .vecm_dgp import VECMBivariateDGP

__all__ = [
    "BaseDGP",
    "AR1", "RandomWalk", "AR1WithTrend", "SeasonalDGP", "AR1WithBreak",
    "AR1ARCH", "AR1GARCH", "PureGARCH", "AR1GJRGARCH", "AR1EGARCH",
    "MarkovSwitchingAR",
    "LocalLevelDGP", "LocalTrendDGP", "DampedTrendDGP",
    "DeterministicSeasonalDGP", "SeasonalRandomWalkDGP", "LocalLevelSeasonalDGP",
    "VARDGP", "VARGARCHDiagonalDGP", "VECMBivariateDGP",
]
