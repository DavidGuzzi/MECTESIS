"""
Series description: summary statistics and standardised plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL


def summary_stats(y: pd.Series) -> pd.DataFrame:
    s = y.dropna()
    return pd.DataFrame([{
        "n": int(s.shape[0]),
        "start": s.index.min(),
        "end": s.index.max(),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
        "skew": float(s.skew()),
        "kurtosis": float(s.kurtosis()),
    }])


def plot_series(y: pd.Series, title: str = "", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    y.plot(ax=ax, color="#1f4068")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_acf_pacf(y: pd.Series, lags: int = 36, title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    plot_acf(y.dropna(), lags=lags, ax=axes[0])
    plot_pacf(y.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title(f"ACF — {title}")
    axes[1].set_title(f"PACF — {title}")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_decomposition(y: pd.Series, period: int = 12, title: str = ""):
    res = STL(y.dropna(), period=period, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, comp, name in zip(
        axes,
        [y.dropna(), res.trend, res.seasonal, res.resid],
        ["Observed", "Trend", "Seasonal", "Residual"],
    ):
        comp.plot(ax=ax, color="#1f4068")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0].set_title(f"STL — {title}")
    fig.tight_layout()
    return fig


def plot_panel(df: pd.DataFrame, title: str = "Panel de series"):
    n = df.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.8 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, df.columns):
        df[col].plot(ax=ax, color="#1f4068")
        ax.set_ylabel(col)
        ax.grid(alpha=0.3)
    axes[0].set_title(title)
    fig.tight_layout()
    return fig
