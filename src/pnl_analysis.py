"""PnL computation and summary statistics for probability-threshold strategy outputs."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _portfolio_turnover(
    previous_weights: dict[str, float],
    current_weights: dict[str, float],
) -> float:
    """Return one-way turnover between two equal-weight baskets."""
    all_pairs = set(previous_weights) | set(current_weights)
    return 0.5 * sum(abs(current_weights.get(pair, 0.0) - previous_weights.get(pair, 0.0)) for pair in all_pairs)


def compute_threshold_pnl(
    df: pd.DataFrame,
    pred_col: str,
    p_win_threshold: float,
    transaction_loss_pct: float = 0.0,
    max_positions: int | None = None,
) -> pd.DataFrame:
    """Daily PnL from probability-threshold predictions using next-day returns."""
    rows = []
    transaction_loss = transaction_loss_pct / 100.0
    previous_weights: dict[str, float] = {}
    for d, grp in df.groupby("Date", sort=True):
        eligible = grp[grp[pred_col] > p_win_threshold].sort_values(pred_col, ascending=False)
        if max_positions is not None:
            eligible = eligible.head(max_positions)

        if eligible.empty:
            current_weights: dict[str, float] = {}
            gross_return = 0.0
            trade_count = 0
            avg_predicted_prob = np.nan
        else:
            weight = 1.0 / len(eligible)
            current_weights = {pair: weight for pair in eligible["pair"]}
            gross_return = float(eligible["next_ret"].mean())
            trade_count = int(len(eligible))
            avg_predicted_prob = float(eligible[pred_col].mean())

        turnover = _portfolio_turnover(previous_weights, current_weights)
        pnl = gross_return - (transaction_loss * turnover)
        rows.append(
            {
                "Date": d,
                "pnl": pnl,
                "gross_return": gross_return,
                "turnover": turnover,
                "trade_count": trade_count,
                "avg_predicted_prob": avg_predicted_prob,
            }
        )
        previous_weights = current_weights
    return pd.DataFrame(rows).sort_values("Date")


def compute_top_k_pnl(
    df: pd.DataFrame,
    pred_col: str,
    transaction_loss_pct: float = 0.0,
    k: int = 1,
) -> pd.DataFrame:
    """Daily PnL from selecting the top-k predictions each day."""
    rows = []
    transaction_loss = transaction_loss_pct / 100.0
    previous_weights: dict[str, float] = {}
    for d, grp in df.groupby("Date", sort=True):
        chosen = grp.sort_values(pred_col, ascending=False).head(k)
        weight = 1.0 / len(chosen)
        current_weights = {pair: weight for pair in chosen["pair"]}
        gross_return = float(chosen["next_ret"].mean())
        turnover = _portfolio_turnover(previous_weights, current_weights)
        pnl = gross_return - (transaction_loss * turnover)
        rows.append(
            {
                "Date": d,
                "pnl": pnl,
                "gross_return": gross_return,
                "turnover": turnover,
                "trade_count": int(len(chosen)),
                "avg_predicted_prob": float(chosen[pred_col].mean()),
            }
        )
        previous_weights = current_weights
    return pd.DataFrame(rows).sort_values("Date")


def compute_top_quantile_pnl(
    df: pd.DataFrame,
    pred_col: str,
    transaction_loss_pct: float = 0.0,
    top_quantile: float = 0.10,
) -> pd.DataFrame:
    """Daily PnL from selecting the top prediction quantile each day."""
    rows = []
    transaction_loss = transaction_loss_pct / 100.0
    previous_weights: dict[str, float] = {}
    for d, grp in df.groupby("Date", sort=True):
        position_count = max(1, int(math.ceil(len(grp) * top_quantile)))
        chosen = grp.sort_values(pred_col, ascending=False).head(position_count)
        weight = 1.0 / len(chosen)
        current_weights = {pair: weight for pair in chosen["pair"]}
        gross_return = float(chosen["next_ret"].mean())
        turnover = _portfolio_turnover(previous_weights, current_weights)
        pnl = gross_return - (transaction_loss * turnover)
        rows.append(
            {
                "Date": d,
                "pnl": pnl,
                "gross_return": gross_return,
                "turnover": turnover,
                "trade_count": int(len(chosen)),
                "avg_predicted_prob": float(chosen[pred_col].mean()),
            }
        )
        previous_weights = current_weights
    return pd.DataFrame(rows).sort_values("Date")


def compute_equal_weight_pnl(
    df: pd.DataFrame,
    transaction_loss_pct: float = 0.0,
) -> pd.DataFrame:
    """Equal-weight daily benchmark PnL using next-day realized returns."""
    rows = []
    transaction_loss = transaction_loss_pct / 100.0
    previous_weights: dict[str, float] = {}
    for d, grp in df.groupby("Date", sort=True):
        weight = 1.0 / len(grp)
        current_weights = {pair: weight for pair in grp["pair"]}
        turnover = _portfolio_turnover(previous_weights, current_weights)
        pnl = float(grp["next_ret"].mean()) - (transaction_loss * turnover)
        rows.append(
            {
                "Date": d,
                "pnl": pnl,
                "gross_return": float(grp["next_ret"].mean()),
                "turnover": turnover,
                "trade_count": int(len(grp)),
                "avg_predicted_prob": np.nan,
            }
        )
        previous_weights = current_weights
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
    avg_turnover = float(df["turnover"].mean()) if "turnover" in df.columns and not df.empty else np.nan
    avg_trade_count = float(df["trade_count"].mean()) if "trade_count" in df.columns and not df.empty else np.nan
    trade_days = int((df["trade_count"] > 0).sum()) if "trade_count" in df.columns else 0
    trade_rate = trade_days / n_periods if n_periods else np.nan
    avg_predicted_prob = (
        float(df["avg_predicted_prob"].dropna().mean())
        if "avg_predicted_prob" in df.columns and not df["avg_predicted_prob"].dropna().empty
        else np.nan
    )
    return {
        "Annualized Return": ann_ret,
        "Annualized Vol": ann_vol,
        "Sharpe": sharpe,
        "Cumulative Return": cum,
        "Avg Turnover": avg_turnover,
        "Avg Trades/Day": avg_trade_count,
        "Trade Rate": trade_rate,
        "Avg Predicted P(Win)": avg_predicted_prob,
    }
