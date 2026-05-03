"""
ARIMA model implementation.
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from .base import BaseModel


class ARIMAModel(BaseModel):
    """ARIMA(p, d, q) model using statsmodels."""

    def __init__(self, order: tuple = (1, 0, 0)):
        self.order = order
        self.fitted_model = None
        self._y_train = None

    def fit(self, y_train: np.ndarray, **kwargs):
        self._y_train = y_train
        model = ARIMA(y_train, order=self.order)
        self.fitted_model = model.fit(**kwargs)

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("Call fit() before forecast().")
        fcst = self.fitted_model.get_forecast(steps=horizon)
        return np.array(fcst.predicted_mean)

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95):
        if self.fitted_model is None:
            raise ValueError("Call fit() before forecast_intervals().")
        alpha = 1.0 - level
        fcst = self.fitted_model.get_forecast(steps=horizon)
        ci = fcst.conf_int(alpha=alpha)
        ci = np.asarray(ci)
        return ci[:, 0], ci[:, 1]

    @property
    def name(self) -> str:
        return f"ARIMA{self.order}"
