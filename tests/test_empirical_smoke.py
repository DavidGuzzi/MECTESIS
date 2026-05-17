"""
Smoke test for the `mectesis.empirical` package using synthetic data.

Runs end-to-end:
  - transforms (log-diff, panel alignment)
  - diagnostics (unit roots, ARCH, Johansen, Granger, cross-corr)
  - backtest (univariate, with covariates, multivariate)
  - autoselect wrappers (AutoARIMA, AutoETS, AutoTheta)

Does NOT depend on data/raw/* CSVs being present.
"""

import numpy as np
import pandas as pd

from mectesis.empirical import diagnostics as d
from mectesis.empirical.autoselect import AutoARIMAModel, AutoETSModel, AutoThetaModel
from mectesis.empirical.backtest import RollingOriginBacktest, compare_models
from mectesis.empirical.transforms import log_diff, align_panel, to_monthly_inflation
from mectesis.models.arima import ARIMAModel
from mectesis.models.sarimax_model import SARIMAXModel
from mectesis.models.var_model import VARModel


def _synthetic_panel(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-01-01", periods=n, freq="MS")
    ipc = pd.Series(100.0 * np.cumprod(1.0 + 0.02 + 0.01 * rng.standard_normal(n)),
                     index=idx, name="ipc")
    tcn = pd.Series(50.0 * np.cumprod(1.0 + 0.025 + 0.015 * rng.standard_normal(n)),
                     index=idx, name="tcn")
    m2 = pd.Series(1e6 * np.cumprod(1.0 + 0.018 + 0.01 * rng.standard_normal(n)),
                    index=idx, name="m2")
    rem_12m = pd.Series(2.5 + 0.3 * rng.standard_normal(n), index=idx, name="rem_12m").cumsum() / 30
    badlar = pd.Series(40.0 + 5.0 * rng.standard_normal(n), index=idx, name="badlar")
    return pd.concat([ipc, tcn, m2, rem_12m, badlar], axis=1)


def test_transforms():
    panel = _synthetic_panel()
    pi = to_monthly_inflation(panel["ipc"])
    assert pi.shape[0] == panel.shape[0] - 1
    aligned = align_panel(pd.concat([panel["tcn"], pi], axis=1))
    assert not aligned.isna().any().any()


def test_diagnostics_univariate():
    panel = _synthetic_panel()
    pi = to_monthly_inflation(panel["ipc"])
    assert d.unit_root_battery(pi).shape[0] >= 4
    assert d.serial_correlation(pi).shape[0] == 2
    assert d.heteroscedasticity(pi).shape[0] == 1
    assert d.normality(pi).shape[0] == 1
    assert d.structural_breaks(pi).shape[0] == 1


def test_diagnostics_multivariate():
    panel = _synthetic_panel()
    diffs = pd.concat({
        "pi": to_monthly_inflation(panel["ipc"]),
        "dlog_tcn": log_diff(panel["tcn"], 100),
        "dlog_m2": log_diff(panel["m2"], 100),
    }, axis=1).dropna()
    gm = d.granger_matrix(diffs, maxlag=3)
    assert gm.shape == (3, 3)
    jh = d.johansen_test(pd.DataFrame({
        "log_ipc": np.log(panel["ipc"]),
        "log_tcn": np.log(panel["tcn"]),
    }).dropna())
    assert "stat" in jh.columns


def test_backtest_univariate():
    panel = _synthetic_panel(n=100)
    pi = to_monthly_inflation(panel["ipc"])
    bt = RollingOriginBacktest(
        model_factory=lambda: AutoARIMAModel(season_length=12),
        y=pi,
        horizons=[1, 3],
        initial_window=60,
        step=4,
    )
    res = bt.run()
    assert set(res.keys()) == {1, 3}
    for h, df in res.items():
        assert {"rmse", "mae", "bias", "cov80", "cov95", "crps"}.issubset(df.columns)
        assert df["n_origins"].iloc[0] > 0


def test_backtest_covariates():
    panel = _synthetic_panel(n=100)
    pi = to_monthly_inflation(panel["ipc"])
    X = pd.concat({
        "dlog_tcn": log_diff(panel["tcn"], 100),
        "badlar": panel["badlar"].diff(),
    }, axis=1).reindex(pi.index).dropna()
    pi = pi.loc[X.index]
    bt = RollingOriginBacktest(
        model_factory=lambda: SARIMAXModel(order=(1, 0, 0)),
        y=pi,
        horizons=[1, 3],
        initial_window=60,
        step=4,
        X=X,
    )
    res = bt.run()
    assert all(df["n_origins"].iloc[0] > 0 for df in res.values())


def test_backtest_multivariate():
    panel = _synthetic_panel(n=100)
    Y = pd.concat({
        "pi": to_monthly_inflation(panel["ipc"]),
        "dlog_tcn": log_diff(panel["tcn"], 100),
    }, axis=1).dropna()
    bt = RollingOriginBacktest(
        model_factory=lambda: VARModel(lags=1),
        y=Y,
        horizons=[1, 3],
        initial_window=60,
        step=4,
    )
    res = bt.run()
    for h, df in res.items():
        assert set(df["variable"]) == {"pi", "dlog_tcn"}
        assert "trace_msfe" in df.attrs


def test_compare_models():
    panel = _synthetic_panel(n=100)
    pi = to_monthly_inflation(panel["ipc"])
    long = compare_models(
        factories={
            "AutoARIMA": lambda: AutoARIMAModel(season_length=12),
            "AutoETS":   lambda: AutoETSModel(season_length=12),
            "AutoTheta": lambda: AutoThetaModel(season_length=12),
        },
        y=pi,
        horizons=[1, 3],
        initial_window=60,
    )
    assert set(long["model"].unique()) == {"AutoARIMA", "AutoETS", "AutoTheta"}
    assert set(long["horizon"].unique()) == {1, 3}


if __name__ == "__main__":
    test_transforms(); print("transforms OK")
    test_diagnostics_univariate(); print("diag univ OK")
    test_diagnostics_multivariate(); print("diag mult OK")
    test_backtest_univariate(); print("bt univ OK")
    test_backtest_covariates(); print("bt cov  OK")
    test_backtest_multivariate(); print("bt mult OK")
    test_compare_models(); print("compare_models OK")
    print("\nALL EMPIRICAL SMOKE TESTS PASSED")
