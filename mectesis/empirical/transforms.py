"""
Transformations for empirical series: log-diff, alignment, etc.
"""

import numpy as np
import pandas as pd


def log_diff(x: pd.Series, scale: float = 1.0) -> pd.Series:
    return scale * np.log(x).diff().dropna()


def to_monthly_inflation(ipc: pd.Series) -> pd.Series:
    return log_diff(ipc, scale=100.0).rename("pi_mensual")


def to_yoy_inflation(ipc: pd.Series) -> pd.Series:
    return (100.0 * (np.log(ipc) - np.log(ipc.shift(12)))).dropna().rename("pi_yoy")


def align_panel(df: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
    return df.dropna(how="any" if how == "inner" else "all")
