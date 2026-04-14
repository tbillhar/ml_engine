"""PnL computation and summary statistics for score-based strategy outputs."""

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
            avg_selected_score = np.nan
        else:
            weight = 1.0 / len(eligible)
            current_weights = {pair: weight for pair in eligible["pair"]}
            gross_return = float(eligible["next_ret"].mean())
            trade_count = int(len(eligible))
            avg_selected_score = float(eligible[pred_col].mean())

        turnover = _portfolio_turnover(previous_weights, current_weights)
        pnl = gross_return - (transaction_loss * turnover)
        rows.append(
            {
                "Date": d,
                "pnl": pnl,
                "gross_return": gross_return,
                "turnover": turnover,
                "trade_count": trade_count,
                "avg_selected_score": avg_selected_score,
            }
        )
        previous_weights = current_weights
    return pd.DataFrame(rows).sort_values("Date")


def compute_top_k_pnl(
    df: pd.DataFrame,
    pred_col: str,
    transaction_loss_pct: float = 0.0,
    k: int = 1,
    rebalance_days: int = 1,
    ascending: bool = False,
) -> pd.DataFrame:
    """Daily PnL from selecting the top/bottom-k predictions on a rebalance schedule."""
    rows = []
    transaction_loss = transaction_loss_pct / 100.0
    previous_weights: dict[str, float] = {}
    current_weights: dict[str, float] = {}
    held_pairs: list[str] = []
    held_avg_score = np.nan
    for idx, (d, grp) in enumerate(df.groupby("Date", sort=True)):
        should_rebalance = (idx % rebalance_days == 0) or not current_weights
        if should_rebalance:
            chosen = grp.sort_values(pred_col, ascending=ascending).head(k)
            weight = 1.0 / len(chosen)
            current_weights = {pair: weight for pair in chosen["pair"]}
            held_pairs = list(chosen["pair"])
            held_avg_score = float(chosen[pred_col].mean())
        held_rows = grp.set_index("pair").reindex(held_pairs).dropna(subset=["next_ret"]).reset_index()
        gross_return = float(
            sum(current_weights.get(pair, 0.0) * next_ret for pair, next_ret in zip(held_rows["pair"], held_rows["next_ret"]))
        )
        turnover = _portfolio_turnover(previous_weights, current_weights) if should_rebalance else 0.0
        pnl = gross_return - (transaction_loss * turnover)
        rows.append(
            {
                "Date": d,
                "pnl": pnl,
                "gross_return": gross_return,
                "turnover": turnover,
                "trade_count": int(len(current_weights)),
                "avg_selected_score": held_avg_score,
            }
        )
        previous_weights = current_weights.copy()
    return pd.DataFrame(rows).sort_values("Date")


def compute_bottom_k_pnl(
    df: pd.DataFrame,
    pred_col: str,
    transaction_loss_pct: float = 0.0,
    k: int = 1,
    rebalance_days: int = 1,
) -> pd.DataFrame:
    """Daily PnL from selecting the bottom-k predictions on a rebalance schedule."""
    return compute_top_k_pnl(
        df,
        pred_col=pred_col,
        transaction_loss_pct=transaction_loss_pct,
        k=k,
        rebalance_days=rebalance_days,
        ascending=True,
    )


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
                "avg_selected_score": float(chosen[pred_col].mean()),
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
                "avg_selected_score": np.nan,
            }
        )
        previous_weights = current_weights
    return pd.DataFrame(rows).sort_values("Date")


def compute_top_k_excess_vs_equal_weight_pnl(
    df: pd.DataFrame,
    pred_col: str,
    transaction_loss_pct: float = 0.0,
    k: int = 1,
    rebalance_days: int = 1,
) -> pd.DataFrame:
    """Daily excess PnL of Top-k versus equal-weight benchmark."""
    top_df = compute_top_k_pnl(
        df,
        pred_col=pred_col,
        transaction_loss_pct=transaction_loss_pct,
        k=k,
        rebalance_days=rebalance_days,
    )
    eq_df = compute_equal_weight_pnl(df, transaction_loss_pct=transaction_loss_pct)
    merged = top_df.merge(
        eq_df[["Date", "pnl", "gross_return"]],
        on="Date",
        suffixes=("", "_eq"),
    )
    merged["pnl"] = merged["pnl"] - merged["pnl_eq"]
    merged["gross_return"] = merged["gross_return"] - merged["gross_return_eq"]
    return merged[["Date", "pnl", "gross_return", "turnover", "trade_count", "avg_selected_score"]]


def compute_bottom_k_excess_vs_equal_weight_pnl(
    df: pd.DataFrame,
    pred_col: str,
    transaction_loss_pct: float = 0.0,
    k: int = 1,
    rebalance_days: int = 1,
) -> pd.DataFrame:
    """Daily excess PnL of Bottom-k versus equal-weight benchmark."""
    bottom_df = compute_bottom_k_pnl(
        df,
        pred_col=pred_col,
        transaction_loss_pct=transaction_loss_pct,
        k=k,
        rebalance_days=rebalance_days,
    )
    eq_df = compute_equal_weight_pnl(df, transaction_loss_pct=transaction_loss_pct)
    merged = bottom_df.merge(
        eq_df[["Date", "pnl", "gross_return"]],
        on="Date",
        suffixes=("", "_eq"),
    )
    merged["pnl"] = merged["pnl"] - merged["pnl_eq"]
    merged["gross_return"] = merged["gross_return"] - merged["gross_return_eq"]
    return merged[["Date", "pnl", "gross_return", "turnover", "trade_count", "avg_selected_score"]]


def compute_top_bottom_spread_pnl(
    df: pd.DataFrame,
    pred_col: str,
    transaction_loss_pct: float = 0.0,
    k: int = 1,
    rebalance_days: int = 1,
) -> pd.DataFrame:
    """Daily long-top-k minus short-bottom-k PnL."""
    rows = []
    transaction_loss = transaction_loss_pct / 100.0
    previous_weights: dict[str, float] = {}
    current_weights: dict[str, float] = {}
    held_long_pairs: list[str] = []
    held_short_pairs: list[str] = []
    held_avg_score = np.nan

    for idx, (d, grp) in enumerate(df.groupby("Date", sort=True)):
        should_rebalance = (idx % rebalance_days == 0) or not current_weights
        if should_rebalance:
            ranked = grp.sort_values(pred_col, ascending=False)
            long_side = ranked.head(k)
            short_side = ranked.tail(k)
            long_weight = 1.0 / len(long_side)
            short_weight = -1.0 / len(short_side)
            current_weights = {pair: long_weight for pair in long_side["pair"]}
            current_weights.update({pair: short_weight for pair in short_side["pair"]})
            held_long_pairs = list(long_side["pair"])
            held_short_pairs = list(short_side["pair"])
            held_avg_score = float(long_side[pred_col].mean() - short_side[pred_col].mean())

        grp_indexed = grp.set_index("pair")
        long_rows = grp_indexed.reindex(held_long_pairs).dropna(subset=["next_ret"]).reset_index()
        short_rows = grp_indexed.reindex(held_short_pairs).dropna(subset=["next_ret"]).reset_index()
        long_return = float(sum(current_weights.get(pair, 0.0) * next_ret for pair, next_ret in zip(long_rows["pair"], long_rows["next_ret"])))
        short_return = float(sum(current_weights.get(pair, 0.0) * next_ret for pair, next_ret in zip(short_rows["pair"], short_rows["next_ret"])))
        gross_return = long_return + short_return
        turnover = _portfolio_turnover(previous_weights, current_weights) if should_rebalance else 0.0
        pnl = gross_return - (transaction_loss * turnover)
        rows.append(
            {
                "Date": d,
                "pnl": pnl,
                "gross_return": gross_return,
                "turnover": turnover,
                "trade_count": int(len(current_weights)),
                "avg_selected_score": held_avg_score,
            }
        )
        previous_weights = current_weights.copy()

    return pd.DataFrame(rows).sort_values("Date")


def compute_bottom_top_spread_pnl(
    df: pd.DataFrame,
    pred_col: str,
    transaction_loss_pct: float = 0.0,
    k: int = 1,
    rebalance_days: int = 1,
) -> pd.DataFrame:
    """Daily long-bottom-k minus short-top-k PnL."""
    spread_df = compute_top_bottom_spread_pnl(
        df,
        pred_col=pred_col,
        transaction_loss_pct=transaction_loss_pct,
        k=k,
        rebalance_days=rebalance_days,
    ).copy()
    spread_df["pnl"] = -spread_df["pnl"]
    spread_df["gross_return"] = -spread_df["gross_return"]
    spread_df["avg_selected_score"] = -spread_df["avg_selected_score"]
    return spread_df


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
    gross = df["gross_return"].values if "gross_return" in df.columns else pnl
    gross_cum = (1 + gross).prod() - 1
    gross_ann_ret = (1 + gross_cum) ** (trading_days_per_year / n_periods) - 1 if n_periods else np.nan
    ann_vol = pnl.std() * np.sqrt(trading_days_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    avg_turnover = float(df["turnover"].mean()) if "turnover" in df.columns and not df.empty else np.nan
    avg_trade_count = float(df["trade_count"].mean()) if "trade_count" in df.columns and not df.empty else np.nan
    trade_days = int((df["trade_count"] > 0).sum()) if "trade_count" in df.columns else 0
    trade_rate = trade_days / n_periods if n_periods else np.nan
    avg_selected_score = (
        float(df["avg_selected_score"].dropna().mean())
        if "avg_selected_score" in df.columns and not df["avg_selected_score"].dropna().empty
        else np.nan
    )
    return {
        "Gross Annualized Return": gross_ann_ret,
        "Gross Cumulative Return": gross_cum,
        "Annualized Return": ann_ret,
        "Annualized Vol": ann_vol,
        "Sharpe": sharpe,
        "Cumulative Return": cum,
        "Avg Turnover": avg_turnover,
        "Avg Trades/Day": avg_trade_count,
        "Trade Rate": trade_rate,
        "Avg Selected Score": avg_selected_score,
    }
