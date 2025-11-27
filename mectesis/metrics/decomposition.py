"""
Bias-Variance-MSE decomposition for forecast evaluation.
"""

import numpy as np
import pandas as pd


class BiasVarianceMSE:
    """
    Compute bias-variance decomposition of forecast errors.

    This class provides methods to analyze forecast errors from Monte Carlo
    simulations, decomposing the Mean Squared Error (MSE) into bias and variance
    components.
    """

    @staticmethod
    def compute_from_errors(errors_matrix: np.ndarray) -> dict:
        """
        Compute bias, variance, MSE, and RMSE from forecast errors.

        Parameters
        ----------
        errors_matrix : np.ndarray
            Matrix of forecast errors with shape (n_sim, horizon),
            where errors = y_true - y_pred.

        Returns
        -------
        dict
            Dictionary with keys:
            - "bias": np.ndarray of shape (horizon,), mean error at each horizon
            - "variance": np.ndarray of shape (horizon,), variance of errors
            - "mse": np.ndarray of shape (horizon,), mean squared error
            - "rmse": np.ndarray of shape (horizon,), root mean squared error

        Notes
        -----
        The bias-variance decomposition states that:
            MSE = Bias^2 + Variance
        However, for empirical errors this may not hold exactly due to sampling.
        """
        bias = errors_matrix.mean(axis=0)
        variance = errors_matrix.var(axis=0, ddof=1)
        mse = (errors_matrix ** 2).mean(axis=0)
        rmse = np.sqrt(mse)

        return {
            "bias": bias,
            "variance": variance,
            "mse": mse,
            "rmse": rmse
        }

    @staticmethod
    def compute_summary_table(errors_matrix: np.ndarray) -> pd.DataFrame:
        """
        Create a summary table of metrics per forecast horizon.

        Parameters
        ----------
        errors_matrix : np.ndarray
            Matrix of forecast errors with shape (n_sim, horizon).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: horizon, bias, variance, mse, rmse.
            Includes a final row with averages across all horizons.

        Examples
        --------
        >>> errors = np.random.randn(1000, 12)  # 1000 sims, 12-step horizon
        >>> table = BiasVarianceMSE.compute_summary_table(errors)
        >>> print(table)
        """
        horizon = errors_matrix.shape[1]
        metrics = BiasVarianceMSE.compute_from_errors(errors_matrix)

        # Create per-horizon table
        df = pd.DataFrame({
            "horizon": np.arange(1, horizon + 1),
            "bias": metrics["bias"],
            "variance": metrics["variance"],
            "mse": metrics["mse"],
            "rmse": metrics["rmse"]
        })

        # Add aggregated row (average across all horizons)
        df_agg = pd.DataFrame({
            "horizon": ["avg_all"],
            "bias": [metrics["bias"].mean()],
            "variance": [metrics["variance"].mean()],
            "mse": [metrics["mse"].mean()],
            "rmse": [metrics["rmse"].mean()]
        })

        return pd.concat([df, df_agg], ignore_index=True)
