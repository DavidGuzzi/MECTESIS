"""
Rolling-origin (a.k.a. walk-forward) backtest for forecasting models on
a single empirical time series. Mirrors the metric set produced by
`MonteCarloEngine` (RMSE, MAE, CRPS, coverage 80/95, Winkler 80/95,
mean width 80/95) but indexed by **origin × horizon** instead of by
Monte Carlo replicate × horizon.

Supports three modes:
  - univariate:   y is pd.Series
  - covariates:   y is pd.Series, X is pd.DataFrame (exog passed at
                  fit and forecast)
  - multivariate: y is pd.DataFrame (one column per variable). Models
                  must accept and return arrays of shape (T, k) /
                  (horizon, k).
"""

from typing import Callable

import numpy as np
import pandas as pd

from ..models.base import BaseModel


class RollingOriginBacktest:
    """
    Refit-at-every-origin backtest. A fresh model is instantiated via
    `model_factory()` at each origin to avoid carry-over state.

    Parameters
    ----------
    model_factory : callable() -> BaseModel
    y : pd.Series | pd.DataFrame
        Target series (Series → univariate / DataFrame → multivariate).
    horizons : list[int]
    initial_window : int
        Number of observations used to fit the first origin.
    scheme : {'expanding', 'sliding'}
    step : int
        Stride between consecutive origins.
    X : pd.DataFrame | None
        Exogenous regressors aligned with `y` (covariate mode).
    levels : tuple[float, ...]
        Prediction interval coverage levels to evaluate.
    """

    def __init__(
        self,
        model_factory: Callable[[], BaseModel],
        y: pd.Series | pd.DataFrame,
        horizons: list[int],
        initial_window: int = 72,
        scheme: str = "expanding",
        step: int = 1,
        X: pd.DataFrame | None = None,
        levels: tuple = (0.80, 0.95),
    ):
        if scheme not in ("expanding", "sliding"):
            raise ValueError("scheme must be 'expanding' or 'sliding'")
        self.model_factory = model_factory
        self.y = y
        self.horizons = sorted(horizons)
        self.initial_window = initial_window
        self.scheme = scheme
        self.step = step
        self.X = X
        self.levels = levels
        self._multivariate = isinstance(y, pd.DataFrame)

    def _origins(self) -> list[int]:
        max_h = max(self.horizons)
        T = self.y.shape[0]
        last = T - max_h
        return list(range(self.initial_window, last + 1, self.step))

    def _slice_train(self, t: int):
        if self.scheme == "expanding":
            start = 0
        else:
            start = max(0, t - self.initial_window)
        return start, t

    @staticmethod
    def _winkler(y_true, lo, hi, level):
        alpha = 1.0 - level
        penalty = 2.0 / alpha
        return ((hi - lo)
                + penalty * np.maximum(lo - y_true, 0.0)
                + penalty * np.maximum(y_true - hi, 0.0))

    def run(self, verbose: bool = False) -> dict[int, pd.DataFrame]:
        origins = self._origins()
        n_orig = len(origins)
        if n_orig == 0:
            raise ValueError(
                f"No valid origins for T={self.y.shape[0]}, "
                f"initial_window={self.initial_window}, max_h={max(self.horizons)}"
            )

        max_h = max(self.horizons)
        y_arr = self.y.to_numpy()
        X_arr = self.X.to_numpy() if self.X is not None else None
        k = y_arr.shape[1] if self._multivariate else 1

        shape = (n_orig, max_h, k) if self._multivariate else (n_orig, max_h)
        errors = np.full(shape, np.nan)
        cov = {lv: np.full(shape, np.nan) for lv in self.levels}
        wid = {lv: np.full(shape, np.nan) for lv in self.levels}
        wnk = {lv: np.full(shape, np.nan) for lv in self.levels}
        crps = np.full(shape, np.nan)
        has_iv = has_crps = False

        for i, t in enumerate(origins):
            start, end = self._slice_train(t)
            y_train = y_arr[start:end]
            y_test = y_arr[end:end + max_h]

            model = self.model_factory()
            fit_kwargs = {}
            forecast_kwargs = {}
            if self.X is not None:
                fit_kwargs["X_train"] = X_arr[start:end]
                forecast_kwargs["X_future"] = X_arr[end:end + max_h]

            model.fit(y_train, **fit_kwargs)
            y_hat = model.forecast(max_h, **forecast_kwargs)
            errors[i] = y_test - y_hat

            if model.supports_intervals:
                has_iv = True
                for lv in self.levels:
                    lo, hi = model.forecast_intervals(max_h, level=lv, **forecast_kwargs) \
                        if model.supports_covariates else \
                        model.forecast_intervals(max_h, level=lv)
                    inside = (y_test >= lo) & (y_test <= hi)
                    cov[lv][i] = inside.astype(float)
                    wid[lv][i] = hi - lo
                    wnk[lv][i] = self._winkler(y_test, lo, hi, lv)

            if model.supports_crps:
                has_crps = True
                crps[i] = model.compute_crps(y_test, max_h, **forecast_kwargs) \
                    if model.supports_covariates else \
                    model.compute_crps(y_test, max_h)

            if verbose:
                print(f"  origin {i+1}/{n_orig} (t={t})")

        return self._summarise(errors, cov, wid, wnk, crps, has_iv, has_crps, k)

    def _summarise(self, errors, cov, wid, wnk, crps, has_iv, has_crps, k):
        out: dict[int, pd.DataFrame] = {}
        for h in self.horizons:
            idx = h - 1
            if self._multivariate:
                rows = []
                for j, name in enumerate(self.y.columns):
                    err = errors[:, idx, j]
                    row = {
                        "variable": name,
                        "n_origins": int(np.sum(~np.isnan(err))),
                        "rmse": float(np.sqrt(np.nanmean(err ** 2))),
                        "mae": float(np.nanmean(np.abs(err))),
                        "bias": float(np.nanmean(err)),
                    }
                    if has_iv:
                        for lv in self.levels:
                            tag = int(round(lv * 100))
                            row[f"cov{tag}"] = float(np.nanmean(cov[lv][:, idx, j]))
                            row[f"width{tag}"] = float(np.nanmean(wid[lv][:, idx, j]))
                            row[f"winkler{tag}"] = float(np.nanmean(wnk[lv][:, idx, j]))
                    if has_crps:
                        row["crps"] = float(np.nanmean(crps[:, idx, j]))
                    rows.append(row)
                df = pd.DataFrame(rows)
                trace = float(np.nanmean(np.nansum(errors[:, idx, :] ** 2, axis=1)))
                df.attrs["trace_msfe"] = trace
            else:
                err = errors[:, idx]
                row = {
                    "n_origins": int(np.sum(~np.isnan(err))),
                    "rmse": float(np.sqrt(np.nanmean(err ** 2))),
                    "mae": float(np.nanmean(np.abs(err))),
                    "bias": float(np.nanmean(err)),
                }
                if has_iv:
                    for lv in self.levels:
                        tag = int(round(lv * 100))
                        row[f"cov{tag}"] = float(np.nanmean(cov[lv][:, idx]))
                        row[f"width{tag}"] = float(np.nanmean(wid[lv][:, idx]))
                        row[f"winkler{tag}"] = float(np.nanmean(wnk[lv][:, idx]))
                if has_crps:
                    row["crps"] = float(np.nanmean(crps[:, idx]))
                df = pd.DataFrame([row])
            out[h] = df
        return out


def compare_models(
    factories: dict[str, Callable[[], BaseModel]],
    y: pd.Series | pd.DataFrame,
    horizons: list[int],
    initial_window: int = 72,
    scheme: str = "expanding",
    X: pd.DataFrame | None = None,
    levels: tuple = (0.80, 0.95),
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Convenience helper: run RollingOriginBacktest for each factory and
    return a long-format DataFrame with (model, horizon) as identifiers.
    """
    rows = []
    for name, factory in factories.items():
        if verbose:
            print(f"[backtest] {name}")
        bt = RollingOriginBacktest(
            factory, y, horizons,
            initial_window=initial_window, scheme=scheme,
            X=X, levels=levels,
        )
        res = bt.run(verbose=False)
        for h, df in res.items():
            tmp = df.copy()
            tmp.insert(0, "horizon", h)
            tmp.insert(0, "model", name)
            rows.append(tmp)
    return pd.concat(rows, ignore_index=True)
