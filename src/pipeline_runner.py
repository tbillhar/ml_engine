"""Shared FX pipeline execution flow used by batch and GUI entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from src.data_pipeline import load_csv, parse_and_sort_dates
from src.feature_engineering import compute_future_returns
from src.long_format import build_long
from src.pnl_analysis import (
    compute_equal_weight_pnl,
    compute_topK_pnl,
    cumulative_annualized_curve,
    perf_stats,
)
from src.walkforward_model import run_walkforward_model

LogFn = Callable[[str], None] | None
ProgressFn = Callable[[int], None] | None


def run_pipeline(
    csv_path: str,
    train_days: int,
    test_days: int,
    step_days: int,
    horizon: int,
    transaction_loss_pct: float,
    trading_days_per_year: int,
    output_dir: Path,
    log_fn: LogFn = None,
    progress_fn: ProgressFn = None,
) -> tuple[pd.DataFrame, Path]:
    """Run the existing FX pipeline and save artifacts to output_dir."""

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    def set_progress(value: int) -> None:
        if progress_fn:
            progress_fn(value)

    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading input CSV: {csv_path}")
    set_progress(10)
    df = load_csv(csv_path)
    df = parse_and_sort_dates(df)

    ret_cols = [c for c in df.columns if c.startswith("ret_")]
    pairs = [c.replace("ret_", "") for c in ret_cols]
    log(f"Detected {len(pairs)} pairs")

    log(f"Computing {horizon}-day future returns")
    set_progress(25)
    df = compute_future_returns(df, horizon=horizon)

    log("Building long-format ranking table")
    set_progress(40)
    long_df = build_long(df, pairs)

    log(
        "Running walk-forward model "
        f"(train={train_days}, test={test_days}, step={step_days})"
    )
    set_progress(60)
    pred_df = run_walkforward_model(
        long_df,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )

    log("Computing daily mark-to-market PnL series")
    set_progress(75)
    pnl_top1 = compute_topK_pnl(pred_df, 1, transaction_loss_pct=transaction_loss_pct)
    pnl_top3 = compute_topK_pnl(pred_df, 3, transaction_loss_pct=transaction_loss_pct)
    pnl_top5 = compute_topK_pnl(pred_df, 5, transaction_loss_pct=transaction_loss_pct)
    pnl_eq = compute_equal_weight_pnl(pred_df, transaction_loss_pct=transaction_loss_pct)

    cum1 = cumulative_annualized_curve(pnl_top1, trading_days_per_year=trading_days_per_year)
    cum3 = cumulative_annualized_curve(pnl_top3, trading_days_per_year=trading_days_per_year)
    cum5 = cumulative_annualized_curve(pnl_top5, trading_days_per_year=trading_days_per_year)
    cum_eq = cumulative_annualized_curve(pnl_eq, trading_days_per_year=trading_days_per_year)

    top1_stats = perf_stats(pnl_top1, trading_days_per_year=trading_days_per_year)
    top3_stats = perf_stats(pnl_top3, trading_days_per_year=trading_days_per_year)
    top5_stats = perf_stats(pnl_top5, trading_days_per_year=trading_days_per_year)
    eq_stats = perf_stats(pnl_eq, trading_days_per_year=trading_days_per_year)

    log("Saving CSV outputs")
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    pnl_top1.to_csv(output_dir / "pnl_top1.csv", index=False)
    pnl_top3.to_csv(output_dir / "pnl_top3.csv", index=False)
    pnl_top5.to_csv(output_dir / "pnl_top5.csv", index=False)
    pnl_eq.to_csv(output_dir / "pnl_equal_weight.csv", index=False)

    stats_df = pd.DataFrame(
        [
            {"strategy": "Top-1", **top1_stats},
            {"strategy": "Top-3", **top3_stats},
            {"strategy": "Top-5", **top5_stats},
            {"strategy": "Equal-weight", **eq_stats},
        ]
    )
    stats_df.to_csv(output_dir / "performance_summary.csv", index=False)

    log("Saving PnL plot")
    set_progress(90)
    plot_path = output_dir / "pnl_curves.png"
    plt.figure(figsize=(12, 6))
    plt.plot(cum1["Date"], cum1["cum_ann"], label="Top-1", linewidth=2.5)
    plt.plot(cum3["Date"], cum3["cum_ann"], label="Top-3", linewidth=2.2)
    plt.plot(cum5["Date"], cum5["cum_ann"], label="Top-5", linewidth=2.0)
    plt.plot(
        cum_eq["Date"],
        cum_eq["cum_ann"],
        label="Equal-Weight",
        linestyle="--",
        linewidth=1.8,
    )
    plt.title("Cumulative Annualized Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Annualized Return %")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    set_progress(100)
    log(f"Done. Outputs saved to: {output_dir.resolve()}")
    return stats_df, plot_path
