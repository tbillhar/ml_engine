"""Feature engineering for FX data."""

from __future__ import annotations

import pandas as pd


def compute_returns(df_prices: pd.DataFrame, pairs: list[str], date_col: str = "Date") -> pd.DataFrame:
    """Compute daily percentage returns for each pair from price columns."""
    out = df_prices[[date_col]].copy()
    for pair in pairs:
        out[f"ret_{pair}"] = df_prices[pair].pct_change()
    return out


def compute_rolling_vol(df_ret: pd.DataFrame, pairs: list[str], window: int = 30) -> pd.DataFrame:
    """Compute rolling volatility features from return columns."""
    out = df_ret.copy()
    for pair in pairs:
        col = f"ret_{pair}"
        out[f"vol{window}_{pair}"] = out[col].rolling(window=window, min_periods=window).std()
    return out


def compute_momentum_features(df_ret: pd.DataFrame, pairs: list[str]) -> pd.DataFrame:
    """Compute 5-day and 10-day momentum features from return columns."""
    out = df_ret.copy()
    for pair in pairs:
        col = f"ret_{pair}"
        out[f"mom5_{pair}"] = out[col].rolling(window=5, min_periods=5).sum()
        out[f"mom10_{pair}"] = out[col].rolling(window=10, min_periods=10).sum()
    return out


def compute_future_returns(df: pd.DataFrame, horizon: int, ret_prefix: str = "ret_") -> pd.DataFrame:
    """Attach next-day and compounded HORIZON-day forward returns."""
    ret_cols = [c for c in df.columns if c.startswith(ret_prefix)]
    ret_mat = df[ret_cols].copy()
    next_day_mat = ret_mat.shift(-1)

    future_mat = (
        (1.0 + ret_mat)
        .rolling(window=horizon, min_periods=horizon)
        .apply(lambda x: x.prod(), raw=True)
        .shift(-horizon)
        - 1.0
    )

    valid_mask = ~future_mat.isna().any(axis=1) & ~next_day_mat.isna().any(axis=1)
    out = df.loc[valid_mask].reset_index(drop=True)
    next_day_mat = next_day_mat.loc[valid_mask].reset_index(drop=True)
    future_mat = future_mat.loc[valid_mask].reset_index(drop=True)

    for col in ret_cols:
        pair = col.replace(ret_prefix, "")
        out[f"next_ret_{pair}"] = next_day_mat[col]
        out[f"future_ret_{pair}"] = future_mat[col]

    return out
