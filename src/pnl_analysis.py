"""PnL computation and summary statistics for FX ranking outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_topK_pnl(
    df: pd.DataFrame,
    K: int,
    transaction_loss_pct: float = 0.0,
) -> pd.DataFrame:
    """Daily PnL from Top-K predictions using next-day realized returns."""
    rows = []
    transaction_loss = transaction_loss_pct / 100.0
    previous_pairs: set[str] | None = None
    for d, grp in df.groupby("Date", sort=True):
        chosen = grp.sort_values("pred", ascending=False).head(K)
        current_pairs = set(chosen["pair"])
        if previous_pairs is None:
            turnover = 1.0
        else:
            turnover = len(current_pairs - previous_pairs) / K
        pnl = chosen["next_ret"].mean() - (transaction_loss * turnover)
        rows.append({"Date": d, "pnl": pnl})
        previous_pairs = current_pairs
    return pd.DataFrame(rows).sort_values("Date")


def compute_equal_weight_pnl(
    df: pd.DataFrame,
    transaction_loss_pct: float = 0.0,
) -> pd.DataFrame:
    """Equal-weight daily benchmark PnL using next-day realized returns."""
    rows = []
    transaction_loss = transaction_loss_pct / 100.0
    first_row = True
    for d, grp in df.groupby("Date", sort=True):
        # The full-universe equal-weight basket has no constituent turnover after entry.
        pnl = grp["next_ret"].mean() - (transaction_loss if first_row else 0.0)
        rows.append({"Date": d, "pnl": pnl})
        first_row = False
    return pd.DataFrame(rows).sort_values("Date")


def cumulative_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Convert daily pnl into cumulative return curve."""
    out = df.copy()
    out["cum"] = (1.0 + out["pnl"]).cumprod()
    return out


def cumulative_annualized_curve(
    df: pd.DataFrame,
    trading_days_per_year: int,
) -> pd.DataFrame:
    """Convert strategy pnl into an annualized return-to-date curve."""
    out = df.copy()
    out["wealth"] = (1.0 + out["pnl"]).cumprod()
    elapsed_periods = np.arange(1, len(out) + 1)
    out["cum_ann"] = np.power(out["wealth"], trading_days_per_year / elapsed_periods) - 1.0
    return out.drop(columns=["wealth"])


def perf_stats(df: pd.DataFrame, trading_days_per_year: int) -> dict[str, float]:
    """Compute annualized and cumulative performance summary."""
    pnl = df["pnl"].values
    cum = (1 + pnl).prod() - 1
    n_periods = len(pnl)
    ann_ret = (1 + cum) ** (trading_days_per_year / n_periods) - 1 if n_periods else np.nan
    ann_vol = pnl.std() * np.sqrt(trading_days_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return {
        "Annualized Return": ann_ret,
        "Annualized Vol": ann_vol,
        "Sharpe": sharpe,
        "Cumulative Return": cum,
    }
