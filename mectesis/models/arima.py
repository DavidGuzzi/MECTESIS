"""
ARIMA model implementation.
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from .base import BaseModel


class ARIMAModel(BaseModel):
    """
    ARIMA(p, d, q) model using statsmodels.

    This class wraps the statsmodels ARIMA implementation to conform
    to the BaseModel interface.
    """

    def __init__(self, order: tuple = (1, 0, 0)):
        """
        Initialize ARIMA model.

        Parameters
        ----------
        order : tuple of int, optional
            ARIMA order (p, d, q). Default is (1, 0, 0) for AR(1).
        """
        self.order = order
        self.fitted_model = None
        self._y_train = None

    def fit(self, y_train: np.ndarray, **kwargs):
        """
        Fit ARIMA model to training data.

        Parameters
        ----------
        y_train : np.ndarray
            Training time series.
        **kwargs : dict
            Additional arguments passed to ARIMA.fit().
        """
        self._y_train = y_train
        model = ARIMA(y_train, order=self.order)
        self.fitted_model = model.fit(**kwargs)

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        """
        Generate multi-step ahead forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        **kwargs : dict
            Additional arguments (not used for ARIMA).

        Returns
        -------
        np.ndarray
            Point forecasts.

        Raises
        ------
        ValueError
            If model has not been fitted.
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting. Call fit() first.")

        fcst = self.fitted_model.get_forecast(steps=horizon)
        return np.array(fcst.predicted_mean)

    @property
    def name(self) -> str:
        """Return model name."""
        return f"ARIMA{self.order}"
