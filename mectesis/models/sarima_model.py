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

    Covers Exp 1.6: SARIMAModel(order=(1,0,0), seasonal_order=(1,0,0,4))
    Covers Exp 1.7: SARIMAModel(order=(0,1,0), seasonal_order=(0,1,0,12))
    """

    def __init__(self, order: tuple = (1, 0, 0),
                 seasonal_order: tuple = (1, 0, 0, 4),
                 trend: str = None):
        """
        Parameters
        ----------
        order : tuple
            ARIMA order (p, d, q).
        seasonal_order : tuple
            Seasonal order (P, D, Q, s).
        trend : str, optional
            Trend specification passed to SARIMAX. Default is None.
        """
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
            raise ValueError("Model must be fitted before forecasting.")
        fcst = self.fitted_model.forecast(steps=horizon)
        return np.array(fcst)

    @property
    def name(self) -> str:
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        return f"SARIMA({p},{d},{q})({P},{D},{Q})_{s}"
