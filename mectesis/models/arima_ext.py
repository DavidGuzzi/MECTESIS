"""
Extended ARIMA models: with deterministic trend and structural break.
"""

import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import BaseModel


class ARIMAWithTrendModel(BaseModel):
    """
    ARIMA with deterministic trend via statsmodels ARIMA trend parameter.

    Fits: Y_t = alpha + beta*t + phi*Y_{t-1} + eps_t  (with trend='ct')

    Covers Exp 1.5 with order=(1,0,0) and trend='ct'.
    """

    def __init__(self, order: tuple = (1, 0, 0), trend: str = 'ct'):
        """
        Parameters
        ----------
        order : tuple
            ARIMA order (p, d, q).
        trend : str
            Trend specification: 'c' (constant), 't' (linear), 'ct' (both).
        """
        self.order = order
        self.trend = trend
        self.fitted_model = None

    def fit(self, y_train: np.ndarray, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(y_train, order=self.order, trend=self.trend)
            self.fitted_model = model.fit(**kwargs)

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting.")
        fcst = self.fitted_model.get_forecast(steps=horizon)
        return np.array(fcst.predicted_mean)

    @property
    def name(self) -> str:
        return f"ARIMA{self.order}+trend"


class ARIMAWithBreakModel(BaseModel):
    """
    ARIMA with structural break dummy as exogenous variable (via SARIMAX).

    Constructs a 0/1 break indicator based on T_total and break_fraction.
    In the forecast period (always post-break), the dummy is set to 1.

    Covers Exp 1.8. Must be instantiated with T_total matching the full series length T.
    """

    def __init__(self, order: tuple = (1, 0, 0), T_total: int = 200,
                 break_fraction: float = 0.5):
        """
        Parameters
        ----------
        order : tuple
            ARIMA order (p, d, q).
        T_total : int
            Total series length (T), used to determine the break index.
        break_fraction : float
            Break position as a fraction of T_total. Default is 0.5.
        """
        self.order = order
        self.T_total = T_total
        self.break_fraction = break_fraction
        self.break_idx = int(T_total * break_fraction)
        self.fitted_model = None

    def fit(self, y_train: np.ndarray, **kwargs):
        n_train = len(y_train)
        break_exog = (np.arange(n_train) >= self.break_idx).astype(float).reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_train,
                order=self.order,
                exog=break_exog,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.fitted_model = model.fit(disp=False, **kwargs)

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting.")
        # Entire forecast period is after the break
        exog_future = np.ones((horizon, 1))
        fcst = self.fitted_model.forecast(steps=horizon, exog=exog_future)
        return np.array(fcst)

    @property
    def name(self) -> str:
        return f"ARIMA{self.order}+break"
