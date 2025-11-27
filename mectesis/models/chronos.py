"""
Chronos Time Series Foundation Model implementation.
"""

import numpy as np
import torch
from chronos import ChronosPipeline
from .base import BaseModel


class ChronosModel(BaseModel):
    """
    Amazon Chronos-T5 Time Series Foundation Model.

    This class wraps the ChronosPipeline (Chronos-T5 variant) to conform
    to the BaseModel interface. Chronos-T5 is the original, well-documented
    variant that uses T5 architecture for time series forecasting.
    """

    def __init__(self, model_size: str = "tiny", device: str = "cpu"):
        """
        Initialize Chronos-T5 model.

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
        """Load the Chronos-T5 pipeline from pretrained weights."""
        model_name = f"amazon/chronos-t5-{self.model_size}"
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=torch.float32
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
        Generate multi-step ahead forecasts using Chronos-T5.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        **kwargs : dict
            Additional arguments (not used for Chronos).

        Returns
        -------
        np.ndarray
            Point forecasts (median across samples).

        Raises
        ------
        ValueError
            If training data has not been provided via fit().
        """
        if self.y_train is None:
            raise ValueError("Training data required. Call fit() first.")

        # Convert training data to torch tensor
        context = torch.tensor(self.y_train, dtype=torch.float32)

        # Generate forecast (returns tensor of shape [num_samples, horizon])
        forecast_samples = self.pipeline.predict(
            context=context,
            prediction_length=horizon
        )

        # Convert to numpy and compute median across samples as point forecast
        # Shape: (num_samples, horizon) -> (horizon,)
        forecast_median = np.median(forecast_samples.numpy(), axis=0)

        return forecast_median

    @property
    def name(self) -> str:
        """Return model name."""
        return f"Chronos-T5-{self.model_size.capitalize()}"
