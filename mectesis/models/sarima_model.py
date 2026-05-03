"""
SARIMA model implementation using statsmodels SARIMAX.
"""

import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import BaseModel


class SARIMAModel(BaseModel):
    """
    SARIMA(p,d,q)(P,D,Q)_s model using statsmodels SARIMAX.

    Exp 1.6: SARIMAModel(order=(1,0,0), seasonal_order=(1,0,0,4))
    Exp 1.7: SARIMAModel(order=(0,1,0), seasonal_order=(0,1,0,12))
    """

    def __init__(self, order: tuple = (1, 0, 0),
                 seasonal_order: tuple = (1, 0, 0, 4),
                 trend: str = None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.fitted_model = None

    def fit(self, y_train: np.ndarray, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.fitted_model = model.fit(disp=False, **kwargs)

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("Call fit() before forecast().")
        return np.array(self.fitted_model.forecast(steps=horizon))

    @property
    def supports_intervals(self) -> bool:
        return True

    def forecast_intervals(self, horizon: int, level: float = 0.95):
        if self.fitted_model is None:
            raise ValueError("Call fit() before forecast_intervals().")
        alpha = 1.0 - level
        fcst = self.fitted_model.get_forecast(steps=horizon)
        ci = np.asarray(fcst.conf_int(alpha=alpha))
        return ci[:, 0], ci[:, 1]

    @property
    def name(self) -> str:
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        return f"SARIMA({p},{d},{q})({P},{D},{Q})_{s}"
