"""
Auto-selection wrappers over Nixtla's statsforecast (numba/C port of
Hyndman's R `forecast` package). Adapt AutoARIMA, AutoETS, AutoTheta
to the repo's BaseModel interface (fit, forecast, forecast_intervals,
compute_crps).

Reference: Garza, F., Mergenthaler-Canseco, M., Olivares, K. G., &
Challu, C. (2022). statsforecast: Lightning fast forecasting with
statistical and econometric models. Nixtla.
"""

import numpy as np
from scipy.stats import norm

from ..models.base import BaseModel


_DEFAULT_LEVELS = [80, 95]


class _StatsForecastWrapper(BaseModel):
    """Common scaffolding for statsforecast Auto* models."""

    _sf_levels = _DEFAULT_LEVELS

    def __init__(self, season_length: int = 1):
        self._season_length = season_length
        self._sf_model = None
        self._cache: dict = {}

    def _build(self):
        raise NotImplementedError

    def fit(self, y_train: np.ndarray, **kwargs):
        self._cache = {}
        self._sf_model = self._build()
        self._sf_model.fit(y=np.asarray(y_train, dtype=float))

    def _predict(self, horizon: int) -> dict:
        if horizon not in self._cache:
            self._cache[horizon] = self._sf_model.predict(
                h=horizon, level=self._sf_levels
            )
        return self._cache[horizon]

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        return np.asarray(self._predict(horizon)["mean"])

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95):
        pct = int(round(level * 100))
        if pct not in self._sf_levels:
            raise ValueError(
                f"{self.name} only supports levels {self._sf_levels}, got {pct}"
            )
        out = self._predict(horizon)
        return np.asarray(out[f"lo-{pct}"]), np.asarray(out[f"hi-{pct}"])

    @property
    def supports_crps(self) -> bool:
        return True

    def compute_crps(self, y_true: np.ndarray, horizon: int) -> np.ndarray:
        from properscoring import crps_gaussian
        out = self._predict(horizon)
        mu = np.asarray(out["mean"])
        lo95 = np.asarray(out["lo-95"])
        hi95 = np.asarray(out["hi-95"])
        sigma = np.maximum((hi95 - lo95) / (2.0 * norm.ppf(0.975)), 1e-8)
        return crps_gaussian(np.asarray(y_true[:horizon]), mu, sigma)


class AutoARIMAModel(_StatsForecastWrapper):
    """statsforecast AutoARIMA (Hyndman-Khandakar) wrapper."""

    def _build(self):
        from statsforecast.models import AutoARIMA
        return AutoARIMA(season_length=self._season_length)

    @property
    def name(self) -> str:
        return "AutoARIMA"


class AutoETSModel(_StatsForecastWrapper):
    """statsforecast AutoETS wrapper (AICc selection over ETS family)."""

    def _build(self):
        from statsforecast.models import AutoETS
        return AutoETS(season_length=self._season_length)

    @property
    def name(self) -> str:
        return "AutoETS"


class AutoThetaModel(_StatsForecastWrapper):
    """statsforecast AutoTheta wrapper (Standard/Optimized/Dyn variants)."""

    def _build(self):
        from statsforecast.models import AutoTheta
        return AutoTheta(season_length=self._season_length)

    @property
    def name(self) -> str:
        return "AutoTheta"
