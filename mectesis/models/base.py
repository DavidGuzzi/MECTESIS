"""
Base class for forecasting models.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for forecasting models.

    All forecasting model implementations must inherit from this class
    and implement the fit(), forecast(), and name property.
    """

    @abstractmethod
    def fit(self, y_train: np.ndarray, **kwargs):
        """
        Fit the model to training data.

        Parameters
        ----------
        y_train : np.ndarray
            Training time series of shape (T_train,).
        **kwargs : dict
            Model-specific fitting parameters.
        """
        pass

    @abstractmethod
    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        """
        Generate multi-step ahead forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        **kwargs : dict
            Model-specific forecasting parameters.

        Returns
        -------
        np.ndarray
            Point forecasts of shape (horizon,).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return a descriptive name for the model.

        Returns
        -------
        str
            Model name for reporting and logging.
        """
        pass
