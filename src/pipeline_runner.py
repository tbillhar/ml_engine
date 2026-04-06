"""Shared FX pipeline execution flow used by batch and GUI entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import brier_score_loss

from src.data_pipeline import load_csv, parse_and_sort_dates
from src.feature_engineering import compute_future_returns
from src.long_format import build_long
from src.pnl_analysis import (
    compute_equal_weight_pnl,
    compute_threshold_pnl,
    cumulative_annualized_curve,
    perf_stats,
)
from src.walkforward_model import run_walkforward_model

LogFn = Callable[[str], None] | None
ProgressFn = Callable[[int], None] | None
DEFAULT_THRESHOLD_SWEEP = [0.50, 0.52, 0.55, 0.58, 0.60]


def build_probability_bucket_diagnostics(
    pred_df: pd.DataFrame,
    model_cols: list[str],
) -> pd.DataFrame:
    """Summarize calibration and realized outcomes by model probability bucket."""
    rows: list[dict[str, float | int | str]] = []
    for model_col in model_cols:
        bucket_df = pred_df[[model_col, "profit_target", "next_ret"]].copy()
        if bucket_df[model_col].nunique() < 2:
            rows.append(
                {
                    "model": model_col,
                    "bucket": "all",
                    "count": int(len(bucket_df)),
                    "mean_probability": float(bucket_df[model_col].mean()),
                    "realized_win_rate": float(bucket_df["profit_target"].mean()),
                    "mean_next_ret": float(bucket_df["next_ret"].mean()),
                }
            )
            continue

        bucket_df["bucket_id"] = pd.qcut(
            bucket_df[model_col].rank(method="first"),
            q=min(10, len(bucket_df)),
            labels=False,
            duplicates="drop",
        )
        grouped = (
            bucket_df.groupby("bucket_id", dropna=False)
            .agg(
                count=(model_col, "size"),
                mean_probability=(model_col, "mean"),
                realized_win_rate=("profit_target", "mean"),
                mean_next_ret=("next_ret", "mean"),
            )
            .reset_index()
            .sort_values("bucket_id")
        )
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "model": model_col,
                    "bucket": f"q{int(row['bucket_id']) + 1}",
                    "count": int(row["count"]),
                    "mean_probability": float(row["mean_probability"]),
                    "realized_win_rate": float(row["realized_win_rate"]),
                    "mean_next_ret": float(row["mean_next_ret"]),
                }
            )
    return pd.DataFrame(rows)


def build_threshold_sweep(
    pred_df: pd.DataFrame,
    model_cols: list[str],
    thresholds: list[float],
    transaction_loss_pct: float,
    trading_days_per_year: int,
) -> pd.DataFrame:
    """Evaluate thresholded trading behavior across a small grid of thresholds."""
    rows: list[dict[str, float | str]] = []
    for model_col in model_cols:
        for threshold in thresholds:
            pnl_df = compute_threshold_pnl(
                pred_df,
                pred_col=model_col,
                p_win_threshold=threshold,
                transaction_loss_pct=transaction_loss_pct,
            )
            stats = perf_stats(pnl_df, trading_days_per_year=trading_days_per_year)
            rows.append(
                {
                    "model": model_col,
                    "threshold": threshold,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def run_pipeline(
    csv_path: str,
    fit_days: int,
    calibration_days: int,
    test_days: int,
    step_days: int,
    horizon: int,
    transaction_loss_pct: float,
    trading_days_per_year: int,
    p_win_threshold: float,
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
    long_df["ev_target"] = long_df["next_ret"] - (transaction_loss_pct / 100.0)
    long_df["profit_target"] = (long_df["ev_target"] > 0).astype(int)

    log(
        "Running walk-forward ensemble "
        f"(fit={fit_days}, calibration={calibration_days}, test={test_days}, "
        f"step={step_days}, p_win_threshold={p_win_threshold})"
    )
    set_progress(60)
    pred_df, window_diag_df = run_walkforward_model(
        long_df,
        fit_days=fit_days,
        calibration_days=calibration_days,
        test_days=test_days,
        step_days=step_days,
        transaction_loss_pct=transaction_loss_pct,
        log_fn=log,
    )

    log("Computing daily mark-to-market probability-threshold PnL series")
    set_progress(75)
    pnl_threshold = compute_threshold_pnl(
        pred_df,
        pred_col="pred_ensemble",
        p_win_threshold=p_win_threshold,
        transaction_loss_pct=transaction_loss_pct,
    )
    pnl_top1 = compute_threshold_pnl(
        pred_df,
        pred_col="pred_ensemble",
        p_win_threshold=p_win_threshold,
        transaction_loss_pct=transaction_loss_pct,
        max_positions=1,
    )
    pnl_top3 = compute_threshold_pnl(
        pred_df,
        pred_col="pred_ensemble",
        p_win_threshold=p_win_threshold,
        transaction_loss_pct=transaction_loss_pct,
        max_positions=3,
    )
    pnl_top5 = compute_threshold_pnl(
        pred_df,
        pred_col="pred_ensemble",
        p_win_threshold=p_win_threshold,
        transaction_loss_pct=transaction_loss_pct,
        max_positions=5,
    )
    pnl_eq = compute_equal_weight_pnl(pred_df, transaction_loss_pct=transaction_loss_pct)

    cum_threshold = cumulative_annualized_curve(pnl_threshold, trading_days_per_year=trading_days_per_year)
    cum1 = cumulative_annualized_curve(pnl_top1, trading_days_per_year=trading_days_per_year)
    cum3 = cumulative_annualized_curve(pnl_top3, trading_days_per_year=trading_days_per_year)
    cum5 = cumulative_annualized_curve(pnl_top5, trading_days_per_year=trading_days_per_year)
    cum_eq = cumulative_annualized_curve(pnl_eq, trading_days_per_year=trading_days_per_year)

    threshold_stats = perf_stats(pnl_threshold, trading_days_per_year=trading_days_per_year)
    top1_stats = perf_stats(pnl_top1, trading_days_per_year=trading_days_per_year)
    top3_stats = perf_stats(pnl_top3, trading_days_per_year=trading_days_per_year)
    top5_stats = perf_stats(pnl_top5, trading_days_per_year=trading_days_per_year)
    eq_stats = perf_stats(pnl_eq, trading_days_per_year=trading_days_per_year)

    log("Computing model correlation and ensemble benefit diagnostics")
    model_cols = [
        "pred_lgbm_shallow",
        "pred_lgbm_base",
        "pred_lgbm_deep",
        "pred_rf",
        "pred_logreg",
        "pred_ensemble",
        "pred_ensemble_brier",
    ]
    correlation_df = pred_df[model_cols].corr()
    spearman_correlation_df = pred_df[model_cols].corr(method="spearman")
    threshold_decisions = pred_df[model_cols].gt(p_win_threshold).astype(int)
    threshold_agreement_df = threshold_decisions.corr()

    diagnostic_rows = []
    for model_col in model_cols:
        stats = perf_stats(
            compute_threshold_pnl(
                pred_df,
                pred_col=model_col,
                p_win_threshold=p_win_threshold,
                transaction_loss_pct=transaction_loss_pct,
            ),
            trading_days_per_year=trading_days_per_year,
        )
        diagnostic_rows.append(
            {
                "model": model_col,
                "brier_score": brier_score_loss(pred_df["profit_target"], pred_df[model_col]),
                "mean_probability": float(pred_df[model_col].mean()),
                "std_probability": float(pred_df[model_col].std()),
                "trade_fraction_above_threshold": float((pred_df[model_col] > p_win_threshold).mean()),
                "Annualized Return": stats["Annualized Return"],
                "Sharpe": stats["Sharpe"],
                "Trade Rate": stats["Trade Rate"],
            }
        )
    model_diagnostics_df = pd.DataFrame(diagnostic_rows).sort_values("brier_score")
    single_model_df = model_diagnostics_df[
        ~model_diagnostics_df["model"].isin(["pred_ensemble", "pred_ensemble_brier"])
    ].copy()
    best_single_brier = single_model_df.sort_values("brier_score").iloc[0]
    best_single_return = single_model_df.sort_values("Annualized Return", ascending=False).iloc[0]
    best_single_sharpe = single_model_df.sort_values("Sharpe", ascending=False).iloc[0]
    ensemble_row = model_diagnostics_df[model_diagnostics_df["model"] == "pred_ensemble"].iloc[0]
    ensemble_brier_row = model_diagnostics_df[model_diagnostics_df["model"] == "pred_ensemble_brier"].iloc[0]
    ensemble_benefit_df = pd.DataFrame(
        [
            {
                "metric": "Primary Ensemble Brier Improvement vs Best Single Brier",
                "value": float(best_single_brier["brier_score"] - ensemble_row["brier_score"]),
            },
            {
                "metric": "Primary Ensemble Annualized Return Improvement vs Best Single Return",
                "value": float(ensemble_row["Annualized Return"] - best_single_return["Annualized Return"]),
            },
            {
                "metric": "Primary Ensemble Sharpe Improvement vs Best Single Sharpe",
                "value": float(ensemble_row["Sharpe"] - best_single_sharpe["Sharpe"]),
            },
            {
                "metric": "Inverse-Brier Ensemble Brier Improvement vs Best Single Brier",
                "value": float(best_single_brier["brier_score"] - ensemble_brier_row["brier_score"]),
            },
            {
                "metric": "Inverse-Brier Ensemble Annualized Return Improvement vs Best Single Return",
                "value": float(
                    ensemble_brier_row["Annualized Return"] - best_single_return["Annualized Return"]
                ),
            },
            {
                "metric": "Inverse-Brier Ensemble Sharpe Improvement vs Best Single Sharpe",
                "value": float(ensemble_brier_row["Sharpe"] - best_single_sharpe["Sharpe"]),
            },
        ]
    )
    bucket_diagnostics_df = build_probability_bucket_diagnostics(pred_df, model_cols)
    threshold_sweep_models = ["pred_ensemble", "pred_ensemble_brier", "pred_lgbm_deep", "pred_logreg", "pred_rf"]
    threshold_values = sorted({p_win_threshold, *DEFAULT_THRESHOLD_SWEEP})
    threshold_sweep_df = build_threshold_sweep(
        pred_df,
        model_cols=threshold_sweep_models,
        thresholds=threshold_values,
        transaction_loss_pct=transaction_loss_pct,
        trading_days_per_year=trading_days_per_year,
    )
    run_parameters_df = pd.DataFrame(
        [
            {
                "csv_path": csv_path,
                "fit_days": fit_days,
                "calibration_days": calibration_days,
                "test_days": test_days,
                "step_days": step_days,
                "horizon": horizon,
                "transaction_loss_pct": transaction_loss_pct,
                "trading_days_per_year": trading_days_per_year,
                "p_win_threshold": p_win_threshold,
            }
        ]
    )

    log("Saving CSV outputs")
    run_parameters_df.to_csv(output_dir / "run_parameters.csv", index=False)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    window_diag_df.to_csv(output_dir / "window_model_diagnostics.csv", index=False)
    pnl_threshold.to_csv(output_dir / "pnl_threshold.csv", index=False)
    pnl_top1.to_csv(output_dir / "pnl_top1.csv", index=False)
    pnl_top3.to_csv(output_dir / "pnl_top3.csv", index=False)
    pnl_top5.to_csv(output_dir / "pnl_top5.csv", index=False)
    pnl_eq.to_csv(output_dir / "pnl_equal_weight.csv", index=False)
    correlation_df.to_csv(output_dir / "model_prediction_correlation.csv")
    spearman_correlation_df.to_csv(output_dir / "model_prediction_rank_correlation.csv")
    threshold_agreement_df.to_csv(output_dir / "model_threshold_agreement.csv")
    model_diagnostics_df.to_csv(output_dir / "model_diagnostics.csv", index=False)
    ensemble_benefit_df.to_csv(output_dir / "ensemble_benefit.csv", index=False)
    bucket_diagnostics_df.to_csv(output_dir / "probability_bucket_diagnostics.csv", index=False)
    threshold_sweep_df.to_csv(output_dir / "threshold_sweep.csv", index=False)

    stats_df = pd.DataFrame(
        [
            {"strategy": "P(Win) Threshold", **threshold_stats},
            {"strategy": "P(Win) Threshold Top-1", **top1_stats},
            {"strategy": "P(Win) Threshold Top-3", **top3_stats},
            {"strategy": "P(Win) Threshold Top-5", **top5_stats},
            {"strategy": "Equal-weight", **eq_stats},
        ]
    )
    stats_df.to_csv(output_dir / "performance_summary.csv", index=False)
    log(
        "Best single-model diagnostics: "
        f"brier={best_single_brier['model']}:{best_single_brier['brier_score']:.4f}, "
        f"return={best_single_return['model']}:{best_single_return['Annualized Return']:.4%}, "
        f"sharpe={best_single_sharpe['model']}:{best_single_sharpe['Sharpe']:.4f}; "
        f"primary_ensemble_brier={ensemble_row['brier_score']:.4f}, "
        f"inverse_brier_ensemble_brier={ensemble_brier_row['brier_score']:.4f}"
    )
    log(
        "Average pairwise prediction correlation: "
        f"{correlation_df.where(~np.eye(len(correlation_df), dtype=bool)).stack().mean():.4f}"
    )

    log("Saving PnL plot")
    set_progress(90)
    plot_path = output_dir / "pnl_curves.png"
    plt.figure(figsize=(12, 6))
    plt.plot(cum_threshold["Date"], cum_threshold["cum_ann"], label="P(Win) Threshold", linewidth=2.6)
    plt.plot(cum1["Date"], cum1["cum_ann"], label="P(Win) Top-1", linewidth=2.4)
    plt.plot(cum3["Date"], cum3["cum_ann"], label="P(Win) Top-3", linewidth=2.2)
    plt.plot(cum5["Date"], cum5["cum_ann"], label="P(Win) Top-5", linewidth=2.0)
    plt.plot(
        cum_eq["Date"],
        cum_eq["cum_ann"],
        label="Equal-Weight",
        linestyle="--",
        linewidth=1.8,
    )
    plt.title("Cumulative Annualized Return From Calibrated P(Win) Decisions")
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
