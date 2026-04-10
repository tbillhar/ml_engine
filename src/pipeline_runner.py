"""Shared FX pipeline execution flow used by batch and GUI entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from src.data_pipeline import load_csv, parse_and_sort_dates
from src.feature_engineering import compute_future_returns
from src.long_format import build_long
from src.pnl_analysis import (
    compute_equal_weight_pnl,
    compute_top_k_pnl,
    cumulative_annualized_curve,
    perf_stats,
)
from src.walkforward_model import run_walkforward_model

LogFn = Callable[[str], None] | None
ProgressFn = Callable[[int], None] | None
LIVE_MODEL_COLUMNS = {
    "ensemble": "pred_ensemble",
    "specialist_ensemble": "pred_specialist_ensemble",
    "lgbm_deep": "pred_lgbm_deep",
    "lgbm_deep_returns_momentum": "pred_lgbm_deep_returns_momentum",
    "lgbm_deep_corr_regime": "pred_lgbm_deep_corr_regime",
    "lgbm_deep_volatility": "pred_lgbm_deep_volatility",
    "rf": "pred_rf",
    "rf_returns_momentum": "pred_rf_returns_momentum",
    "rf_corr_regime": "pred_rf_corr_regime",
    "logreg": "pred_logreg",
    "logreg_returns_momentum": "pred_logreg_returns_momentum",
    "logreg_corr_regime": "pred_logreg_corr_regime",
    "logreg_volatility": "pred_logreg_volatility",
}


def numeric_prediction_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric prediction columns only, excluding metadata fields."""
    return [
        col
        for col in df.columns
        if col.startswith("pred_") and pd.api.types.is_numeric_dtype(df[col])
    ]


def build_score_bucket_diagnostics(pred_df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Summarize score buckets and realized outcomes by model."""
    rows: list[dict[str, float | int | str]] = []
    for model_col in model_cols:
        bucket_df = pred_df[[model_col, "profit_target", "next_ret"]].copy()
        if bucket_df[model_col].nunique() < 2:
            rows.append(
                {
                    "model": model_col,
                    "bucket": "all",
                    "count": int(len(bucket_df)),
                    "mean_score": float(bucket_df[model_col].mean()),
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
                mean_score=(model_col, "mean"),
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
                    "mean_score": float(row["mean_score"]),
                    "realized_win_rate": float(row["realized_win_rate"]),
                    "mean_next_ret": float(row["mean_next_ret"]),
                }
            )
    return pd.DataFrame(rows)


def build_ensemble_benefit(
    top_selection_df: pd.DataFrame,
    ensemble_models: list[str],
) -> pd.DataFrame:
    """Compare Top-1 ensembles against the best non-ensemble single model."""
    rows: list[dict[str, float | str]] = []
    non_ensemble_df = top_selection_df[~top_selection_df["model"].isin(ensemble_models)].copy()
    selection_non_ensemble = non_ensemble_df[non_ensemble_df["selection"] == "top1"]
    if selection_non_ensemble.empty:
        return pd.DataFrame(rows)
    best_single_return = selection_non_ensemble.sort_values(
        ["Annualized Return", "Sharpe"], ascending=[False, False]
    ).iloc[0]
    best_single_sharpe = selection_non_ensemble.sort_values(
        ["Sharpe", "Annualized Return"], ascending=[False, False]
    ).iloc[0]
    for ensemble_model in ensemble_models:
        ensemble_row = top_selection_df[
            (top_selection_df["model"] == ensemble_model) & (top_selection_df["selection"] == "top1")
        ]
        if ensemble_row.empty:
            continue
        ensemble_row = ensemble_row.iloc[0]
        rows.append(
            {
                "model": ensemble_model,
                "selection": "top1",
                "metric": "Annualized Return Improvement vs Best Single",
                "value": float(ensemble_row["Annualized Return"] - best_single_return["Annualized Return"]),
            }
        )
        rows.append(
            {
                "model": ensemble_model,
                "selection": "top1",
                "metric": "Sharpe Improvement vs Best Single",
                "value": float(ensemble_row["Sharpe"] - best_single_sharpe["Sharpe"]),
            }
        )
    return pd.DataFrame(rows)


def diagnostics_guide_text(
    specialist_ensemble_models: list[str] | None = None,
    specialist_weight_lookback_days: int | None = None,
    rebalance_days: int | None = None,
    specialist_weighting_mode: str | None = None,
    specialist_min_model_hold_days: int | None = None,
    specialist_switch_margin_min_avg_ev: float | None = None,
    specialist_switch_require_positive_ev: bool | None = None,
    retrain_deterioration_lookback_days: int | None = None,
    retrain_deterioration_min_win_rate: float | None = None,
    retrain_deterioration_max_avg_ev: float | None = None,
) -> str:
    """Return a concise guide to the main diagnostics and ensemble construction."""
    specialist_text = (
        ", ".join(specialist_ensemble_models)
        if specialist_ensemble_models
        else "configured specialist members"
    )
    lookback_text = (
        str(specialist_weight_lookback_days)
        if specialist_weight_lookback_days is not None
        else "configured lookback"
    )
    rebalance_text = str(rebalance_days) if rebalance_days is not None else "configured rebalance"
    weighting_mode_text = specialist_weighting_mode or "configured weighting mode"
    min_hold_text = str(specialist_min_model_hold_days) if specialist_min_model_hold_days is not None else "configured min hold"
    switch_margin_text = (
        f"{specialist_switch_margin_min_avg_ev:.6f}"
        if specialist_switch_margin_min_avg_ev is not None
        else "configured switch margin"
    )
    positive_ev_text = (
        str(specialist_switch_require_positive_ev)
        if specialist_switch_require_positive_ev is not None
        else "configured positive-EV rule"
    )
    deterioration_lookback_text = (
        str(retrain_deterioration_lookback_days)
        if retrain_deterioration_lookback_days is not None
        else "configured deterioration lookback"
    )
    deterioration_win_rate_text = (
        f"{retrain_deterioration_min_win_rate:.2f}"
        if retrain_deterioration_min_win_rate is not None
        else "configured deterioration win-rate"
    )
    deterioration_avg_ev_text = (
        f"{retrain_deterioration_max_avg_ev:.6f}"
        if retrain_deterioration_max_avg_ev is not None
        else "configured deterioration avg-ev"
    )
    return "\n".join(
        [
            "Pipeline behavior",
            "- Trains all models on each walk-forward fit window.",
            "- Generates predictions for all models on every test/holdout date.",
            "- Evaluates all models diagnostically on the holdout slice.",
            "- Shows all model Top-1 holdout results in the main leaderboard and plots the top performers graphically.",
            "",
            "Key files",
            "- performance_summary.csv: holdout Top-1 leaderboard for all models plus Equal-weight.",
            "- model_diagnostics.csv: score dispersion diagnostics plus Top-1 economics for all models.",
            "- model_prediction_correlation.csv: holdout prediction correlation matrix.",
            "- top1_pick_overlap.csv: fraction of holdout dates where each pair of models picked the same Top-1 pair.",
            "- model_switch_log.csv: daily active specialist model and switch/stay reason for the specialist router.",
            "- ensemble_benefit.csv: Top-1 ensemble improvements versus the best single non-ensemble model.",
            "- diagnostics_summary.txt: concise holdout observations shown in the GUI.",
            "",
            "Ensembles",
            "- pred_ensemble: fixed-weight broad blend of lgbm_deep, logreg, and rf scores.",
            (
                f"- pred_specialist_ensemble: daily rank-percentile blend of {specialist_text}, "
                f"weighted by trailing realized Top-1 success over the prior {lookback_text} out-of-sample dates "
                f"using '{weighting_mode_text}' mode. Sticky settings: min hold {min_hold_text}, "
                f"switch margin {switch_margin_text}, require positive EV {positive_ev_text}."
            ),
            f"- Rebalancing: Top-1 strategies rebalance every {rebalance_text} day(s).",
            (
                "- Retraining: models retrain on cadence or earlier if current-window selected-live-model Top-1 "
                f"performance over the last {deterioration_lookback_text} out-of-sample days has "
                f"win rate <= {deterioration_win_rate_text} and avg ev_target <= {deterioration_avg_ev_text}."
            ),
            "",
            "Interpretation",
            "- Scores are ranking signals, not calibrated probabilities.",
            "- For specialist_ensemble, the score is a rank-based selection score.",
        ]
    )


def selected_trade_rows(pred_df: pd.DataFrame, pred_col: str, rebalance_days: int) -> pd.DataFrame:
    """Return the Top-1 rows selected on the configured rebalance schedule."""
    selected_rows = []
    for idx, (_, grp) in enumerate(pred_df.groupby("Date", sort=True)):
        if idx % rebalance_days != 0:
            continue
        selected_rows.append(grp.sort_values(pred_col, ascending=False).head(1))
    return pd.concat(selected_rows, ignore_index=True) if selected_rows else pred_df.iloc[0:0].copy()


def build_model_top1_curves(
    pred_df: pd.DataFrame,
    model_cols: list[str],
    transaction_loss_pct: float,
    trading_days_per_year: int,
    rebalance_days: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build Top-1 leaderboard rows and cumulative curves for all models."""
    rows: list[dict[str, float | str]] = []
    curves: dict[str, pd.DataFrame] = {}
    for model_col in model_cols:
        pnl_df = compute_top_k_pnl(
            pred_df,
            pred_col=model_col,
            transaction_loss_pct=transaction_loss_pct,
            k=1,
            rebalance_days=rebalance_days,
        )
        stats = perf_stats(pnl_df, trading_days_per_year=trading_days_per_year)
        rows.append({"model": model_col, **stats})
        curves[model_col] = cumulative_annualized_curve(pnl_df, trading_days_per_year=trading_days_per_year)

    eq_df = compute_equal_weight_pnl(pred_df, transaction_loss_pct=transaction_loss_pct)
    eq_stats = perf_stats(eq_df, trading_days_per_year=trading_days_per_year)
    rows.append({"model": "Equal-weight", **eq_stats})
    curves["Equal-weight"] = cumulative_annualized_curve(eq_df, trading_days_per_year=trading_days_per_year)
    leaderboard_df = pd.DataFrame(rows).sort_values(
        ["Sharpe", "Annualized Return"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return leaderboard_df, curves


def build_top_selection_diagnostics(
    pred_df: pd.DataFrame,
    model_cols: list[str],
    transaction_loss_pct: float,
    trading_days_per_year: int,
    rebalance_days: int,
) -> pd.DataFrame:
    """Compare models as Top-1 selectors on the evaluation slice."""
    rows: list[dict[str, float | str]] = []
    for model_col in model_cols:
        pnl_df = compute_top_k_pnl(
            pred_df,
            pred_col=model_col,
            transaction_loss_pct=transaction_loss_pct,
            k=1,
            rebalance_days=rebalance_days,
        )
        stats = perf_stats(pnl_df, trading_days_per_year=trading_days_per_year)
        selected_df = selected_trade_rows(pred_df, pred_col=model_col, rebalance_days=rebalance_days)
        realized_win_rate = float(selected_df["profit_target"].mean()) if not selected_df.empty else np.nan
        avg_selected_next_ret = float(selected_df["next_ret"].mean()) if not selected_df.empty else np.nan
        rows.append(
            {
                "model": model_col,
                "selection": "top1",
                "selected_rows": int(len(selected_df)),
                "realized_win_rate": realized_win_rate,
                "avg_selected_next_ret": avg_selected_next_ret,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def build_top1_pick_overlap(pred_df: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Compute pairwise same-pick rates for daily Top-1 selections."""
    top1_picks: dict[str, pd.Series] = {}
    for model_col in model_cols:
        picks = (
            pred_df.sort_values(["Date", model_col], ascending=[True, False])
            .groupby("Date", sort=True)
            .head(1)[["Date", "pair"]]
            .set_index("Date")["pair"]
        )
        top1_picks[model_col] = picks

    overlap_df = pd.DataFrame(index=model_cols, columns=model_cols, dtype=float)
    for left_model in model_cols:
        for right_model in model_cols:
            aligned = pd.concat(
                [top1_picks[left_model], top1_picks[right_model]],
                axis=1,
                keys=["left", "right"],
            ).dropna()
            overlap = float((aligned["left"] == aligned["right"]).mean()) if not aligned.empty else np.nan
            overlap_df.loc[left_model, right_model] = overlap
    return overlap_df


def build_correlation_heatmap(
    correlation_df: pd.DataFrame,
    output_path: Path,
    ranked_models: list[str],
    live_pred_col: str,
) -> None:
    """Save a readable heatmap for the leading model correlations."""
    top_models = [model for model in ranked_models if model in correlation_df.index][:10]
    if live_pred_col in correlation_df.index and live_pred_col not in top_models:
        top_models = [live_pred_col, *top_models]
    top_models = list(dict.fromkeys(top_models))
    heatmap_df = correlation_df.loc[top_models, top_models]

    plt.figure(figsize=(10, 8))
    im = plt.imshow(heatmap_df.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(top_models)), top_models, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(top_models)), top_models, fontsize=8)
    plt.title("Holdout Prediction Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def resolve_live_prediction_column(live_model: str) -> str:
    """Map a user-facing live model name to the prediction column."""
    if live_model not in LIVE_MODEL_COLUMNS:
        raise ValueError(f"Unsupported LIVE_MODEL '{live_model}'. Expected one of: {sorted(LIVE_MODEL_COLUMNS)}")
    return LIVE_MODEL_COLUMNS[live_model]


def split_holdout_period(pred_df: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split predictions into research and final holdout periods using unique dates."""
    if holdout_days <= 0:
        return pred_df.copy(), pred_df.copy()
    unique_dates = sorted(pred_df["Date"].unique())
    if holdout_days >= len(unique_dates):
        raise ValueError("HOLDOUT_DAYS must be smaller than the number of out-of-sample test dates")
    holdout_dates = set(unique_dates[-holdout_days:])
    research_df = pred_df[~pred_df["Date"].isin(holdout_dates)].copy()
    holdout_df = pred_df[pred_df["Date"].isin(holdout_dates)].copy()
    if research_df.empty or holdout_df.empty:
        raise ValueError("Holdout split produced an empty research or holdout dataset")
    return research_df, holdout_df


def strategy_frame(
    pred_df: pd.DataFrame,
    pred_col: str,
    transaction_loss_pct: float,
    trading_days_per_year: int,
    live_model_label: str,
    rebalance_days: int,
) -> tuple[list[tuple[str, pd.DataFrame]], pd.DataFrame]:
    """Build Top-1 strategy and benchmark PnL frames for a chosen live prediction column."""
    strategy_series = [
        (
            f"{live_model_label} Top-1",
            compute_top_k_pnl(
                pred_df,
                pred_col=pred_col,
                transaction_loss_pct=transaction_loss_pct,
                k=1,
                rebalance_days=rebalance_days,
            ),
        ),
        (
            "Equal-weight",
            compute_equal_weight_pnl(pred_df, transaction_loss_pct=transaction_loss_pct),
        ),
    ]
    stats_df = pd.DataFrame(
        [
            {"strategy": name, **perf_stats(series_df, trading_days_per_year=trading_days_per_year)}
            for name, series_df in strategy_series
        ]
    )
    return strategy_series, stats_df


def resolve_live_strategy_name(live_model_label: str) -> str:
    """Return the single live strategy row name."""
    return f"{live_model_label} Top-1"


def build_diagnostics_summary(
    selected_strategy_name: str,
    leaderboard_df: pd.DataFrame,
    top_selection_df: pd.DataFrame,
    model_diagnostics_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    live_pred_col: str,
) -> str:
    """Create a concise human-readable diagnostics summary for the GUI."""
    selected_row = leaderboard_df[leaderboard_df["model"] == live_pred_col].iloc[0]
    top1_df = top_selection_df[top_selection_df["selection"] == "top1"].sort_values(
        ["Sharpe", "Annualized Return"],
        ascending=[False, False],
    )
    best_top1 = top1_df.iloc[0]
    best_top1_model_row = model_diagnostics_df.sort_values(
        ["Top-1 Sharpe", "Top-1 Annualized Return"],
        ascending=[False, False],
    ).iloc[0]
    avg_corr = correlation_df.where(~np.eye(len(correlation_df), dtype=bool)).stack().mean()
    return "\n".join(
        [
            (
                f"Selected live strategy: {selected_strategy_name} | "
                f"Holdout AnnRet {selected_row['Annualized Return'] * 100:.1f}% | "
                f"Sharpe {selected_row['Sharpe']:.2f}"
            ),
            (
                f"Best holdout Top-1 selector: {best_top1['model']} | "
                f"AnnRet {best_top1['Annualized Return'] * 100:.1f}% | "
                f"Sharpe {best_top1['Sharpe']:.2f} | "
                f"Win rate {best_top1['realized_win_rate'] * 100:.1f}%"
            ),
            (
                f"Highest-scoring Top-1 model diagnostics row: {best_top1_model_row['model']} | "
                f"AnnRet {best_top1_model_row['Top-1 Annualized Return'] * 100:.1f}% | "
                f"Sharpe {best_top1_model_row['Top-1 Sharpe']:.2f}"
            ),
            f"Average pairwise prediction correlation: {avg_corr:.3f}",
        ]
    )


def run_pipeline(
    csv_path: str,
    fit_days: int,
    step_days: int,
    rebalance_days: int,
    horizon: int,
    transaction_loss_pct: float,
    trading_days_per_year: int,
    holdout_days: int,
    live_model: str,
    specialist_weighting_mode: str,
    specialist_ensemble_models: list[str],
    specialist_weight_lookback_days: int,
    specialist_min_model_hold_days: int,
    specialist_switch_margin_min_avg_ev: float,
    specialist_switch_require_positive_ev: bool,
    retrain_deterioration_lookback_days: int,
    retrain_deterioration_min_win_rate: float,
    retrain_deterioration_max_avg_ev: float,
    output_dir: Path,
    log_fn: LogFn = None,
    progress_fn: ProgressFn = None,
) -> tuple[pd.DataFrame, Path, Path, str]:
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
        "Running walk-forward models "
        f"(fit={fit_days}, "
        f"step={step_days}, rebalance_days={rebalance_days}, holdout_days={holdout_days}, "
        f"live_model={live_model}, live_decision_mode=top1, "
        f"specialist_weighting_mode={specialist_weighting_mode}, "
        f"specialist_ensemble_members={','.join(specialist_ensemble_models)}, "
        f"specialist_weight_lookback_days={specialist_weight_lookback_days}, "
        f"retrain_det_lookback_days={retrain_deterioration_lookback_days}, "
        f"retrain_det_min_win_rate={retrain_deterioration_min_win_rate}, "
        f"retrain_det_max_avg_ev={retrain_deterioration_max_avg_ev})"
    )
    set_progress(60)
    pred_df, window_diag_df = run_walkforward_model(
        long_df,
        fit_days=fit_days,
        step_days=step_days,
        holdout_days=holdout_days,
        live_model=live_model,
        specialist_weighting_mode=specialist_weighting_mode,
        specialist_ensemble_models=specialist_ensemble_models,
        specialist_weight_lookback_days=specialist_weight_lookback_days,
        specialist_min_model_hold_days=specialist_min_model_hold_days,
        specialist_switch_margin_min_avg_ev=specialist_switch_margin_min_avg_ev,
        specialist_switch_require_positive_ev=specialist_switch_require_positive_ev,
        retrain_deterioration_lookback_days=retrain_deterioration_lookback_days,
        retrain_deterioration_min_win_rate=retrain_deterioration_min_win_rate,
        retrain_deterioration_max_avg_ev=retrain_deterioration_max_avg_ev,
        log_fn=log,
    )

    live_pred_col = resolve_live_prediction_column(live_model)
    research_pred_df, holdout_pred_df = split_holdout_period(pred_df, holdout_days=holdout_days)
    log(
        f"Using final {holdout_days} out-of-sample dates as holdout "
        f"({holdout_pred_df['Date'].min()} to {holdout_pred_df['Date'].max()})"
    )

    log("Computing daily mark-to-market strategy series")
    set_progress(75)
    strategy_frame(
        holdout_pred_df,
        pred_col=live_pred_col,
        transaction_loss_pct=transaction_loss_pct,
        trading_days_per_year=trading_days_per_year,
        live_model_label=live_model,
        rebalance_days=rebalance_days,
    )
    selected_strategy_name = resolve_live_strategy_name(live_model)

    log("Computing model correlation and ensemble benefit diagnostics")
    eval_pred_df = holdout_pred_df
    model_cols = numeric_prediction_columns(eval_pred_df)
    correlation_df = eval_pred_df[model_cols].corr()
    spearman_correlation_df = eval_pred_df[model_cols].corr(method="spearman")
    top1_overlap_df = build_top1_pick_overlap(eval_pred_df, model_cols=model_cols)
    leaderboard_df, holdout_curves = build_model_top1_curves(
        eval_pred_df,
        model_cols=model_cols,
        transaction_loss_pct=transaction_loss_pct,
        trading_days_per_year=trading_days_per_year,
        rebalance_days=rebalance_days,
    )
    diagnostic_rows = []
    for model_col in model_cols:
        top1_stats = leaderboard_df[leaderboard_df["model"] == model_col].iloc[0]
        diagnostic_rows.append(
            {
                "model": model_col,
                "mean_score": float(eval_pred_df[model_col].mean()),
                "std_score": float(eval_pred_df[model_col].std()),
                "min_score": float(eval_pred_df[model_col].min()),
                "max_score": float(eval_pred_df[model_col].max()),
                "Top-1 Annualized Return": top1_stats["Annualized Return"],
                "Top-1 Sharpe": top1_stats["Sharpe"],
                "Top-1 Avg Selected Score": top1_stats["Avg Selected Score"],
            }
        )
    model_diagnostics_df = pd.DataFrame(diagnostic_rows).sort_values(
        ["Top-1 Sharpe", "Top-1 Annualized Return"],
        ascending=[False, False],
    )
    _ = build_score_bucket_diagnostics(eval_pred_df, model_cols)
    top_selection_df = build_top_selection_diagnostics(
        eval_pred_df,
        model_cols=model_cols,
        transaction_loss_pct=transaction_loss_pct,
        trading_days_per_year=trading_days_per_year,
        rebalance_days=rebalance_days,
    )
    ensemble_benefit_df = build_ensemble_benefit(
        top_selection_df,
        ensemble_models=["pred_ensemble", "pred_specialist_ensemble"],
    )
    run_parameters_df = pd.DataFrame(
        [
            {
                "csv_path": csv_path,
                "fit_days": fit_days,
                "step_days": step_days,
                "rebalance_days": rebalance_days,
                "horizon": horizon,
                "transaction_loss_pct": transaction_loss_pct,
                "trading_days_per_year": trading_days_per_year,
                "holdout_days": holdout_days,
                "live_model": live_model,
                "live_decision_mode": "top1",
                "live_prediction_column": live_pred_col,
                "specialist_weighting_mode": specialist_weighting_mode,
                "specialist_ensemble_members": "|".join(specialist_ensemble_models),
                "specialist_weight_lookback_days": specialist_weight_lookback_days,
                "specialist_min_model_hold_days": specialist_min_model_hold_days,
                "specialist_switch_margin_min_avg_ev": specialist_switch_margin_min_avg_ev,
                "specialist_switch_require_positive_ev": specialist_switch_require_positive_ev,
                "retrain_deterioration_lookback_days": retrain_deterioration_lookback_days,
                "retrain_deterioration_min_win_rate": retrain_deterioration_min_win_rate,
                "retrain_deterioration_max_avg_ev": retrain_deterioration_max_avg_ev,
                "research_test_dates": int(research_pred_df["Date"].nunique()),
                "holdout_test_dates": int(holdout_pred_df["Date"].nunique()),
            }
        ]
    )

    log("Saving CSV outputs")
    model_switch_log_df = (
        holdout_pred_df[
            [
                "Date",
                "specialist_ensemble_active_model",
                "specialist_ensemble_switch_reason",
                "specialist_ensemble_weight_info",
            ]
        ]
        .drop_duplicates(subset=["Date"])
        .rename(
            columns={
                "specialist_ensemble_active_model": "active_model",
                "specialist_ensemble_switch_reason": "switch_reason",
                "specialist_ensemble_weight_info": "weight_info",
            }
        )
    )
    run_parameters_df.to_csv(output_dir / "run_parameters.csv", index=False)
    correlation_df.to_csv(output_dir / "model_prediction_correlation.csv")
    spearman_correlation_df.to_csv(output_dir / "model_prediction_rank_correlation.csv")
    top1_overlap_df.to_csv(output_dir / "top1_pick_overlap.csv")
    model_switch_log_df.to_csv(output_dir / "model_switch_log.csv", index=False)
    model_diagnostics_df.to_csv(output_dir / "model_diagnostics.csv", index=False)
    ensemble_benefit_df.to_csv(output_dir / "ensemble_benefit.csv", index=False)
    leaderboard_df.to_csv(output_dir / "performance_summary.csv", index=False)
    (output_dir / "diagnostics_guide.txt").write_text(
        diagnostics_guide_text(
            specialist_ensemble_models,
            specialist_weight_lookback_days,
            rebalance_days,
            specialist_weighting_mode,
            specialist_min_model_hold_days,
            specialist_switch_margin_min_avg_ev,
            specialist_switch_require_positive_ev,
            retrain_deterioration_lookback_days,
            retrain_deterioration_min_win_rate,
            retrain_deterioration_max_avg_ev,
        ),
        encoding="utf-8",
    )
    diagnostics_summary = build_diagnostics_summary(
        selected_strategy_name=selected_strategy_name,
        leaderboard_df=leaderboard_df,
        top_selection_df=top_selection_df,
        model_diagnostics_df=model_diagnostics_df,
        correlation_df=correlation_df,
        live_pred_col=live_pred_col,
    )
    (output_dir / "diagnostics_summary.txt").write_text(diagnostics_summary, encoding="utf-8")
    log(
        "Best holdout Top-1 selector: "
        f"{top_selection_df.sort_values(['Sharpe','Annualized Return'], ascending=[False, False]).iloc[0]['model']}"
    )
    log(
        "Average pairwise prediction correlation: "
        f"{correlation_df.where(~np.eye(len(correlation_df), dtype=bool)).stack().mean():.4f}"
    )
    log(
        "Average Top-1 pick overlap: "
        f"{top1_overlap_df.where(~np.eye(len(top1_overlap_df), dtype=bool)).stack().mean():.4f}"
    )

    log("Saving PnL plot")
    set_progress(90)
    plot_path = output_dir / "pnl_curves.png"
    ranked_models = leaderboard_df[leaderboard_df["model"] != "Equal-weight"]["model"].tolist()
    plotted_models = ranked_models[:6]
    if live_pred_col not in plotted_models:
        plotted_models = [live_pred_col, *plotted_models[:5]]
    plotted_models = list(dict.fromkeys(plotted_models))
    plt.figure(figsize=(12, 6))
    for model_name in [*plotted_models, "Equal-weight"]:
        curve_df = holdout_curves[model_name]
        is_selected = model_name == live_pred_col
        plt.plot(
            curve_df["Date"],
            curve_df["cum_ann"],
            label=model_name,
            linewidth=2.8 if is_selected else (1.8 if model_name == "Equal-weight" else 2.0),
            linestyle="--" if model_name == "Equal-weight" else "-",
            alpha=1.0 if is_selected else 0.8,
        )
    plt.title("Holdout Top-1 Cumulative Annualized Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Annualized Return %")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plt.ylim(-0.5, 0.5)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    heatmap_path = output_dir / "model_correlation_heatmap.png"
    build_correlation_heatmap(
        correlation_df=correlation_df,
        output_path=heatmap_path,
        ranked_models=ranked_models,
        live_pred_col=live_pred_col,
    )

    set_progress(100)
    log(f"Done. Outputs saved to: {output_dir.resolve()}")
    return leaderboard_df, plot_path, heatmap_path, diagnostics_summary
