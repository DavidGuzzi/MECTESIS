"""
Base class for forecasting models.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for forecasting models.

    Subclasses must implement fit(), forecast(), and name.
    Subclasses that support prediction intervals should override
    supports_intervals (return True) and forecast_intervals().
    """

    @abstractmethod
    def fit(self, y_train: np.ndarray, **kwargs):
        """Fit the model to training data of shape (T_train,)."""
        pass

    @abstractmethod
    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        """Return point forecasts of shape (horizon,)."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Descriptive model name for reporting."""
        pass

    @property
    def supports_intervals(self) -> bool:
        """True if this model implements forecast_intervals()."""
        return False

    def forecast_intervals(
        self, horizon: int, level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (lower, upper) prediction interval arrays of shape (horizon,).

        Parameters
        ----------
        horizon : int
        level : float
            Coverage probability, e.g. 0.80 or 0.95.

        Raises
        ------
        NotImplementedError
            If supports_intervals is False.
        """
        raise NotImplementedError(
            f"{self.name} does not support prediction intervals. "
            "Override forecast_intervals() and set supports_intervals = True."
        )
