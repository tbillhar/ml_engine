"""Matplotlib plotting utilities for FX strategy curves and regimes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pnl_curves(cum1: pd.DataFrame, cum3: pd.DataFrame, cum5: pd.DataFrame, cum_eq: pd.DataFrame) -> None:
    """Plot cumulative PnL curves for strategies and equal-weight benchmark."""
    plt.figure(figsize=(12, 6))
    plt.plot(cum1["Date"], cum1["cum"], label="Top-1 Strategy", linewidth=2.5)
    plt.plot(cum3["Date"], cum3["cum"], label="Top-3 Strategy", linewidth=2.2)
    plt.plot(cum5["Date"], cum5["cum"], label="Top-5 Strategy", linewidth=2.0)
    plt.plot(cum_eq["Date"], cum_eq["cum"], label="Equal-Weight Benchmark", linewidth=2.0, linestyle="--")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Sliding-Window Walk-Forward PnL")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_with_regimes(
    df_wide: pd.DataFrame,
    pnl_top1: pd.DataFrame,
    pnl_top3: pd.DataFrame,
    pnl_top5: pd.DataFrame,
    pnl_eq: pd.DataFrame,
) -> None:
    """Plot cumulative PnL curves with EWMA volatility-regime shading."""
    ret_cols = [c for c in df_wide.columns if c.startswith("ret_")]

    daily_ret = df_wide[["Date"] + ret_cols].copy()
    daily_ret["eq_ret"] = daily_ret[ret_cols].mean(axis=1)

    start_date = max(
        daily_ret["Date"].min(),
        min(pnl_top1["Date"].min(), pnl_top3["Date"].min(), pnl_top5["Date"].min(), pnl_eq["Date"].min()),
    )
    end_date = min(
        daily_ret["Date"].max(),
        max(pnl_top1["Date"].max(), pnl_top3["Date"].max(), pnl_top5["Date"].max(), pnl_eq["Date"].max()),
    )

    mask = (daily_ret["Date"] >= start_date) & (daily_ret["Date"] <= end_date)
    daily_ret = daily_ret.loc[mask].reset_index(drop=True)

    lam = 0.94
    ewma_var = []
    prev_var = float(daily_ret["eq_ret"].iloc[0] ** 2)
    for r in daily_ret["eq_ret"]:
        var_t = lam * prev_var + (1.0 - lam) * (r ** 2)
        ewma_var.append(var_t)
        prev_var = var_t

    daily_ret["ewma_vol"] = np.sqrt(ewma_var)

    warmup = max(int(0.5 * len(daily_ret)), 30)
    base = daily_ret["ewma_vol"].iloc[:warmup]

    low_thr = base.quantile(0.33)
    high_thr = base.quantile(0.66)

    def classify_regime(v, low=low_thr, high=high_thr):
        if v <= low:
            return 0
        if v >= high:
            return 2
        return 1

    daily_ret["regime"] = daily_ret["ewma_vol"].apply(classify_regime)

    segments = []
    cur_reg = None
    seg_start = None

    for d, r in zip(daily_ret["Date"], daily_ret["regime"]):
        if cur_reg is None:
            cur_reg = r
            seg_start = d
        elif r != cur_reg:
            segments.append((seg_start, d, cur_reg))
            cur_reg = r
            seg_start = d

    segments.append((seg_start, daily_ret["Date"].iloc[-1], cur_reg))

    colors = {
        0: (0.8, 0.9, 1.0, 0.25),
        1: (0.8, 1.0, 0.8, 0.20),
        2: (1.0, 0.8, 0.8, 0.20),
    }
    labels = {
        0: "Calm vol regime",
        1: "Normal vol regime",
        2: "Stressed vol regime",
    }

    def _cumulative(df_pnl: pd.DataFrame) -> pd.DataFrame:
        df2 = df_pnl.copy()
        df2 = df2[(df2["Date"] >= start_date) & (df2["Date"] <= end_date)]
        df2 = df2.sort_values("Date")
        df2["cum"] = (1.0 + df2["pnl"]).cumprod()
        return df2

    cum1 = _cumulative(pnl_top1)
    cum3 = _cumulative(pnl_top3)
    cum5 = _cumulative(pnl_top5)
    cum_eq = _cumulative(pnl_eq)

    plt.figure(figsize=(14, 7))

    for (start, end, reg) in segments:
        plt.axvspan(start, end, color=colors[reg], zorder=0)

    plt.plot(cum1["Date"], cum1["cum"], label="Top-1 Strategy", linewidth=2.2)
    plt.plot(cum3["Date"], cum3["cum"], label="Top-3 Strategy", linewidth=2.0)
    plt.plot(cum5["Date"], cum5["cum"], label="Top-5 Strategy", linewidth=1.8)
    plt.plot(cum_eq["Date"], cum_eq["cum"], label="Equal-Weight Benchmark", linewidth=1.8, linestyle="--")

    handles, leg_labels = plt.gca().get_legend_handles_labels()
    for reg in sorted(set(daily_ret["regime"])):
        handles.append(plt.Line2D([0], [0], color=colors[reg], lw=10, alpha=colors[reg][3]))
        leg_labels.append(labels[reg])

    plt.legend(handles, leg_labels, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.title("Cumulative PnL with Predictive Volatility Regimes (EWMA-based)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
