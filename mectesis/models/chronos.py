"""
Chronos Time Series Foundation Model implementation.
"""

import numpy as np
import torch
from chronos import Chronos2Pipeline
from .base import BaseModel


class ChronosModel(BaseModel):
    """
    Amazon Chronos-2 Time Series Foundation Model.

    Wraps Chronos2Pipeline using the dict-based input API. Chronos-2
    supports covariates and has a context window of 8192 tokens.
    The model is loaded once and reused across Monte Carlo replicas.
    """

    def __init__(self, device: str = "cpu"):
        """
        Parameters
        ----------
        device : str
            "cpu" or "cuda". Default "cpu".
        """
        self.device = device
        self.pipeline = None
        self.y_train = None
        self._load_pipeline()

    def _load_pipeline(self):
        self.pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map=self.device,
            dtype=torch.bfloat16,
        )

    def fit(self, y_train: np.ndarray, **kwargs):
        """Store training context (Chronos-2 is zero-shot)."""
        self.y_train = y_train

    def forecast(self, horizon: int, **kwargs) -> np.ndarray:
        """
        Return point forecast (median) for the next `horizon` steps.

        Returns
        -------
        np.ndarray of shape (horizon,)
        """
        if self.y_train is None:
            raise ValueError("Training data required. Call fit() first.")

        inputs = [{"target": self.y_train}]
        quantiles, _ = self.pipeline.predict_quantiles(
            inputs=inputs,
            prediction_length=horizon,
            quantile_levels=[0.5],
            batch_size=1,
        )
        # quantiles[0] shape: [n_variates, horizon, n_quantiles]
        return quantiles[0].squeeze().numpy()

    @property
    def name(self) -> str:
        return "Chronos-2"
