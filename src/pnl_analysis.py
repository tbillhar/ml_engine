"""PnL computation and summary statistics for FX ranking outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_topK_pnl(df: pd.DataFrame, K: int) -> pd.DataFrame:
    """Daily PnL from trading Top-K pairs by model prediction."""
    rows = []
    for d, grp in df.groupby("Date"):
        chosen = grp.sort_values("pred", ascending=False).head(K)
        pnl = chosen["future_ret"].mean()
        rows.append({"Date": d, "pnl": pnl})
    return pd.DataFrame(rows).sort_values("Date")


def compute_equal_weight_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight daily benchmark PnL using all pairs."""
    rows = []
    for d, grp in df.groupby("Date"):
        rows.append({"Date": d, "pnl": grp["future_ret"].mean()})
    return pd.DataFrame(rows).sort_values("Date")


def cumulative_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Convert daily pnl into cumulative return curve."""
    out = df.copy()
    out["cum"] = (1.0 + out["pnl"]).cumprod()
    return out


def perf_stats(df: pd.DataFrame, horizon_days: int) -> dict[str, float]:
    """Compute annualized and cumulative performance summary."""
    pnl = df["pnl"].values
    daily = pnl / horizon_days
    ann_ret = daily.mean() * 252
    ann_vol = daily.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + pnl).prod() - 1
    return {
        "Annualized Return": ann_ret,
        "Annualized Vol": ann_vol,
        "Sharpe": sharpe,
        "Cumulative Return": cum,
    }
