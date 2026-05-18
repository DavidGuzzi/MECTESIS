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
from mectesis.empirical.autoselect import (
    AutoARIMAModel, AutoETSModel, AutoThetaModel, AutoSARIMAXModel,
)
from mectesis.empirical.backtest import (
    RollingOriginBacktest, compare_models, to_wide_table, predict_all_at_last_origin,
    _mase_scale,
)
from mectesis.empirical.transforms import (
    log_diff, align_panel, to_monthly_inflation,
    select_optimal_lags, apply_lags,
)
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
        assert {"rmse", "mae", "bias", "crps", "mase"}.issubset(df.columns)
        assert df["n_origins"].iloc[0] > 0
        assert np.isfinite(df["mase"].iloc[0])


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


def test_apply_lags_and_select():
    # x leads y by 2 periods: x_{t-2} should be the best predictor of y_t.
    rng = np.random.default_rng(0)
    n = 60
    idx = pd.date_range("2017-01-01", periods=n, freq="MS")
    base = rng.standard_normal(n)
    y = pd.Series(base, index=idx, name="y")
    x_lead = pd.Series(
        np.concatenate([rng.standard_normal(2), base[:-2]])
        + 0.1 * rng.standard_normal(n),
        index=idx, name="x",
    )
    X = pd.DataFrame({"x": x_lead})

    lag_map = select_optimal_lags(y, X, max_lag=6, min_lag=1)
    assert lag_map["x"] == 2

    X_lag = apply_lags(X, lag_map)
    assert X_lag.index.equals(X.index)
    assert "x__lag2" in X_lag.columns
    assert X_lag["x__lag2"].iloc[:2].isna().all()
    assert X_lag["x__lag2"].iloc[2:].notna().all()

    # apply_lags must reject lag=0 to prevent look-ahead.
    try:
        apply_lags(X, {"x": 0})
    except ValueError:
        pass
    else:
        raise AssertionError("apply_lags should reject lag=0")


def test_backtest_with_lagged_covariates():
    panel = _synthetic_panel(n=100)
    pi = to_monthly_inflation(panel["ipc"])
    X_raw = pd.concat({
        "dlog_tcn": log_diff(panel["tcn"], 100),
        "badlar":   panel["badlar"].diff(),
    }, axis=1).reindex(pi.index).dropna()
    pi = pi.loc[X_raw.index]

    lag_map = select_optimal_lags(pi, X_raw, max_lag=6, min_lag=1)
    assert all(L >= 1 for L in lag_map.values())
    X_lag = apply_lags(X_raw, lag_map).dropna()
    y = pi.loc[X_lag.index]

    bt = RollingOriginBacktest(
        model_factory=lambda: SARIMAXModel(order=(1, 0, 0)),
        y=y, X=X_lag, horizons=[1, 3], initial_window=60, step=4,
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
        assert {"rmse", "mae", "crps", "mase"}.issubset(df.columns)


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


def test_mase_scale():
    rng = np.random.default_rng(0)
    y_uni = rng.standard_normal(120)
    scale_uni = _mase_scale(y_uni, season=12)
    assert scale_uni.shape == (1,) and np.isfinite(scale_uni[0]) and scale_uni[0] > 0
    y_mul = rng.standard_normal((120, 3))
    scale_mul = _mase_scale(y_mul, season=12)
    assert scale_mul.shape == (3,) and np.all(np.isfinite(scale_mul))


def test_auto_sarimax():
    panel = _synthetic_panel(n=100)
    pi = to_monthly_inflation(panel["ipc"])
    X = pd.concat({
        "dlog_tcn": log_diff(panel["tcn"], 100),
        "badlar":   panel["badlar"].diff(),
    }, axis=1).reindex(pi.index).dropna()
    pi = pi.loc[X.index]
    bt = RollingOriginBacktest(
        model_factory=lambda: AutoSARIMAXModel(season_length=12),
        y=pi, X=X, horizons=[1, 3], initial_window=60, step=8,
    )
    res = bt.run()
    for h, df in res.items():
        assert {"rmse", "mae", "crps", "mase"}.issubset(df.columns)
        assert df["n_origins"].iloc[0] > 0


def test_to_wide_table():
    long = compare_models(
        factories={
            "AutoARIMA": lambda: AutoARIMAModel(season_length=12),
            "AutoETS":   lambda: AutoETSModel(season_length=12),
        },
        y=to_monthly_inflation(_synthetic_panel(n=100)["ipc"]),
        horizons=[1, 3, 6],
        initial_window=60,
    )
    wide = to_wide_table(long, metrics=("rmse", "mae", "crps", "mase"))
    assert wide.shape[0] == 2
    assert ("rmse", 1) in wide.columns and ("mase", 6) in wide.columns
    assert set(wide.index) == {"AutoARIMA", "AutoETS"}


def test_predict_at_last_origin():
    panel = _synthetic_panel(n=100)
    pi = to_monthly_inflation(panel["ipc"])
    forecasts = predict_all_at_last_origin(
        factories={
            "AutoARIMA": lambda: AutoARIMAModel(season_length=12),
            "AutoETS":   lambda: AutoETSModel(season_length=12),
        },
        y=pi, horizons=[1, 3, 6], initial_window=60,
    )
    for name, f in forecasts.items():
        assert {"mean", "lo80", "hi80", "lo95", "hi95"}.issubset(f.keys())
        assert f["mean"].shape == (6,)


if __name__ == "__main__":
    test_transforms(); print("transforms OK")
    test_diagnostics_univariate(); print("diag univ OK")
    test_diagnostics_multivariate(); print("diag mult OK")
    test_backtest_univariate(); print("bt univ OK")
    test_backtest_covariates(); print("bt cov  OK")
    test_apply_lags_and_select(); print("lags helpers OK")
    test_backtest_with_lagged_covariates(); print("bt cov lagged OK")
    test_backtest_multivariate(); print("bt mult OK")
    test_compare_models(); print("compare_models OK")
    test_mase_scale(); print("mase_scale OK")
    test_auto_sarimax(); print("auto_sarimax OK")
    test_to_wide_table(); print("to_wide_table OK")
    test_predict_at_last_origin(); print("predict_at_last_origin OK")
    print("\nALL EMPIRICAL SMOKE TESTS PASSED")
