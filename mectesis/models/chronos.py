"""
Chronos Time Series Foundation Model implementation.
"""

import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline
from .base import BaseModel


class ChronosModel(BaseModel):
    """
    Amazon Chronos-Bolt Time Series Foundation Model.

    This class wraps the Chronos2Pipeline to conform to the BaseModel interface.
    """

    def __init__(self, model_size: str = "tiny", device: str = "cpu"):
        """
        Initialize Chronos model.

        Parameters
        ----------
        model_size : str, optional
            Model size variant: "tiny", "mini", "small", "base", "large".
            Default is "tiny" for fast execution.
        device : str, optional
            Device to run the model on: "cpu" or "cuda". Default is "cpu".
        """
        self.model_size = model_size
        self.device = device
        self.pipeline = None
        self.y_train = None
        self._load_pipeline()

    def _load_pipeline(self):
        """Load the Chronos pipeline from pretrained weights."""
        model_name = f"amazon/chronos-bolt-{self.model_size}"
        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=self.device
        )

    def fit(self, y_train: np.ndarray, **kwargs):
        """
        Store training data for Chronos.

        Note: Chronos is a zero-shot forecasting model and doesn't require
        traditional fitting. This method simply stores the training data
        for context when forecasting.

        Parameters
        ----------
        y_train : np.ndarray
            Training time series.
        **kwargs : dict
            Not used for Chronos.
        """
        self.y_train = y_train

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        """
        Generate multi-step ahead forecasts using Chronos.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        **kwargs : dict
            Additional arguments (not used for Chronos).

        Returns
        -------
        np.ndarray
            Point forecasts (median quantile).

        Raises
        ------
        ValueError
            If training data has not been provided via fit().
        """
        if self.y_train is None:
            raise ValueError("Training data required. Call fit() first.")

        # Chronos requires data in tabular format with timestamps
        df = pd.DataFrame({
            "item_id": ["series_1"] * len(self.y_train),
            "timestamp": pd.date_range("2000-01-01", periods=len(self.y_train), freq="D"),
            "target": self.y_train,
        })

        # Generate forecast using median quantile as point forecast
        fcst = self.pipeline.predict(
            df,
            prediction_length=horizon,
            quantile_levels=[0.5]  # Use median as point forecast
        )

        # Filter for median quantile and sort by timestamp
        fcst_median = fcst[fcst["quantile"] == 0.5].sort_values("timestamp")

        return fcst_median["mean"].to_numpy()

    @property
    def name(self) -> str:
        """Return model name."""
        return f"Chronos-{self.model_size.capitalize()}"
