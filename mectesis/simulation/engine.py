"""
Monte Carlo simulation engine for forecast comparison.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from mectesis.dgp.base import BaseDGP
from mectesis.models.base import BaseModel
from mectesis.metrics.decomposition import BiasVarianceMSE


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for comparing forecast models.

    This class orchestrates the simulation process:
    1. Generate data from a DGP
    2. Fit multiple models
    3. Generate forecasts
    4. Compute and aggregate forecast errors
    """

    def __init__(self, dgp: BaseDGP, models: List[BaseModel], seed: int = None):
        """
        Initialize Monte Carlo engine.

        Parameters
        ----------
        dgp : BaseDGP
            Data generating process to simulate from.
        models : list of BaseModel
            List of forecasting models to compare.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.dgp = dgp
        self.models = models
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def run_single_simulation(self, T: int, horizon: int,
                              dgp_params: dict) -> Dict[str, np.ndarray]:
        """
        Run a single Monte Carlo simulation.

        Parameters
        ----------
        T : int
            Total length of the simulated series.
        horizon : int
            Forecast horizon.
        dgp_params : dict
            Parameters for the DGP simulation (e.g., phi, mu, sigma).

        Returns
        -------
        dict
            Dictionary with model names as keys and forecast errors as values.
            Each error array has shape (horizon,).

        Notes
        -----
        The series is split into:
        - Training: first (T - horizon) observations
        - Test: last horizon observations
        """
        # Simulate series from DGP
        y = self.dgp.simulate(T=T, **dgp_params)

        # Split train/test
        T_train = T - horizon
        y_train = y[:T_train]
        y_test = y[T_train:]
        assert len(y_test) == horizon, f"Expected {horizon} test obs, got {len(y_test)}"

        # Generate forecasts and compute errors for each model
        errors = {}
        for model in self.models:
            # Fit model
            model.fit(y_train)

            # Forecast
            y_hat = model.forecast(horizon)

            # Compute errors (actual - predicted)
            errors[model.name] = y_test - y_hat

        return errors

    def run_monte_carlo(self, n_sim: int, T: int, horizon: int,
                        dgp_params: dict, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run Monte Carlo simulations and aggregate results.

        Parameters
        ----------
        n_sim : int
            Number of Monte Carlo replications.
        T : int
            Length of each simulated series.
        horizon : int
            Forecast horizon.
        dgp_params : dict
            Parameters for the DGP.
        verbose : bool, optional
            If True, print progress messages. Default is True.

        Returns
        -------
        dict
            Dictionary with model names as keys and DataFrames as values.
            Each DataFrame contains bias, variance, MSE, RMSE metrics
            per horizon, plus an aggregated row.

        Examples
        --------
        >>> from mectesis.dgp import AR1
        >>> from mectesis.models import ARIMAModel
        >>> dgp = AR1(seed=123)
        >>> models = [ARIMAModel(order=(1,0,0))]
        >>> engine = MonteCarloEngine(dgp, models, seed=123)
        >>> results = engine.run_monte_carlo(
        ...     n_sim=100, T=200, horizon=12,
        ...     dgp_params={"phi": 0.7, "mu": 0.0, "sigma": 1.0}
        ... )
        """
        if verbose:
            print(f"Running {n_sim} Monte Carlo simulations...")
            print(f"  DGP: {self.dgp.__class__.__name__}")
            print(f"  Models: {[m.name for m in self.models]}")
            print(f"  T={T}, horizon={horizon}")
            print(f"  DGP params: {dgp_params}")

        # Initialize error matrices: {model_name: np.ndarray of shape (n_sim, horizon)}
        error_matrices = {model.name: np.empty((n_sim, horizon)) for model in self.models}

        # Run simulations
        for s in range(n_sim):
            if verbose and (s + 1) % max(1, n_sim // 10) == 0:
                print(f"  Progress: {s+1}/{n_sim} simulations completed")

            # Run single simulation
            errors = self.run_single_simulation(T, horizon, dgp_params)

            # Store errors
            for model_name, error_vec in errors.items():
                error_matrices[model_name][s] = error_vec

        # Compute metrics for each model
        results = {}
        for model_name, errors_matrix in error_matrices.items():
            results[model_name] = BiasVarianceMSE.compute_summary_table(errors_matrix)

        if verbose:
            print(f"✓ Simulations completed!")

        return results
