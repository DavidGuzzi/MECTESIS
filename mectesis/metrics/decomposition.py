"""
Bias-Variance-MSE-MAE decomposition and interval metrics for forecast evaluation.
"""

import numpy as np
import pandas as pd


class BiasVarianceMSE:
    """
    Compute forecast error metrics from Monte Carlo error matrices.

    Supports point metrics (bias, variance, MSE, RMSE, MAE) and
    prediction interval metrics (coverage and mean width at multiple levels).
    """

    @staticmethod
    def compute_from_errors(errors_matrix: np.ndarray) -> dict:
        """
        Compute point metrics from forecast errors.

        Parameters
        ----------
        errors_matrix : np.ndarray, shape (n_sim, horizon)
            errors = y_true - y_pred

        Returns
        -------
        dict with keys: bias, variance, mse, rmse, mae (each shape (horizon,))
        """
        bias     = errors_matrix.mean(axis=0)
        variance = errors_matrix.var(axis=0, ddof=1)
        mse      = (errors_matrix ** 2).mean(axis=0)
        rmse     = np.sqrt(mse)
        mae      = np.abs(errors_matrix).mean(axis=0)

        return {"bias": bias, "variance": variance, "mse": mse, "rmse": rmse, "mae": mae}

    @staticmethod
    def compute_summary_table(
        errors_matrix: np.ndarray,
        coverage_data: dict = None,
        width_data: dict = None,
    ) -> pd.DataFrame:
        """
        Build per-horizon summary table with point and interval metrics.

        Parameters
        ----------
        errors_matrix : np.ndarray, shape (n_sim, horizon)
        coverage_data : dict, optional
            {level_int: np.ndarray(n_sim, horizon)} e.g. {80: ..., 95: ...}
            Each entry is 1 if y_test fell inside the interval, 0 otherwise.
        width_data : dict, optional
            {level_int: np.ndarray(n_sim, horizon)}
            Width of the prediction interval at each simulation and horizon.

        Returns
        -------
        pd.DataFrame with columns:
            horizon, bias, variance, mse, rmse, mae
            [, cov_80, width_80, cov_95, width_95]  if interval data provided
        """
        horizon = errors_matrix.shape[1]
        m = BiasVarianceMSE.compute_from_errors(errors_matrix)

        row_dict = {
            "horizon":  np.arange(1, horizon + 1),
            "bias":     m["bias"],
            "variance": m["variance"],
            "mse":      m["mse"],
            "rmse":     m["rmse"],
            "mae":      m["mae"],
        }

        agg_dict = {
            "horizon":  "avg_all",
            "bias":     m["bias"].mean(),
            "variance": m["variance"].mean(),
            "mse":      m["mse"].mean(),
            "rmse":     m["rmse"].mean(),
            "mae":      m["mae"].mean(),
        }

        # Interval columns
        for prefix, data in [("cov", coverage_data), ("width", width_data)]:
            if data:
                for level_key, mat in sorted(data.items()):
                    col = f"{prefix}_{level_key}"
                    row_dict[col] = mat.mean(axis=0)
                    agg_dict[col] = mat.mean()

        df = pd.DataFrame(row_dict)
        df_agg = pd.DataFrame([agg_dict])

        return pd.concat([df, df_agg], ignore_index=True)
