"""Run parameter sweeps for the FX pipeline and compare results."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from src.config import (
    FEATURE_DATA_FILENAME,
    FIT_DAYS,
    HORIZON,
    HOLDOUT_DAYS,
    LIVE_MODEL,
    MODEL_FIT_WINDOWS,
    MODEL_ROUTER_CANDIDATES,
    REBALANCE_DAYS,
    RETRAIN_DETERIORATION_LOOKBACK_DAYS,
    RETRAIN_DETERIORATION_MAX_AVG_EV,
    RETRAIN_DETERIORATION_MIN_WIN_RATE,
    ROUTER_FALLBACK_MODEL,
    ROUTER_MIN_EDGE_OVER_NEXT,
    ROUTER_SELECTION_MODE,
    SPECIALIST_ENSEMBLE_MEMBERS,
    SPECIALIST_MIN_MODEL_HOLD_DAYS,
    SPECIALIST_SWITCH_MARGIN_MIN_AVG_EV,
    SPECIALIST_SWITCH_REQUIRE_POSITIVE_EV,
    SPECIALIST_WEIGHTING_MODE,
    SPECIALIST_WEIGHT_LOOKBACK_DAYS,
    STEP_DAYS,
    TRADING_DAYS_PER_YEAR,
    TRANSACTION_LOSS_PCT,
)
from src.pipeline_runner import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FX pipeline over a grid of parameter combinations.",
    )
    parser.add_argument(
        "--csv-path",
        default=str(Path("data") / FEATURE_DATA_FILENAME),
        help="Input feature CSV to use for all experiments.",
    )
    parser.add_argument(
        "--results-dir",
        default="experiment_results",
        help="Directory where per-run outputs and comparison CSVs are written.",
    )
    parser.add_argument(
        "--fit-days",
        nargs="+",
        type=int,
        default=[FIT_DAYS],
        help="One or more primary FIT_DAYS values. The primary value keeps backward-compatible base prediction names.",
    )
    parser.add_argument(
        "--model-fit-windows",
        nargs="+",
        type=int,
        default=MODEL_FIT_WINDOWS,
        help="Fit-window variants to train for every model family.",
    )
    parser.add_argument(
        "--step-days",
        nargs="+",
        type=int,
        default=[STEP_DAYS],
        help="One or more STEP_DAYS values.",
    )
    parser.add_argument(
        "--rebalance-days",
        nargs="+",
        type=int,
        default=[REBALANCE_DAYS],
        help="One or more REBALANCE_DAYS values.",
    )
    parser.add_argument(
        "--horizon",
        nargs="+",
        type=int,
        default=[HORIZON],
        help="One or more HORIZON values.",
    )
    parser.add_argument(
        "--transaction-loss-pct",
        nargs="+",
        type=float,
        default=[TRANSACTION_LOSS_PCT],
        help="One or more transaction loss percentages, entered like 0.025 for 0.025%%.",
    )
    parser.add_argument(
        "--trading-days-per-year",
        nargs="+",
        type=int,
        default=[TRADING_DAYS_PER_YEAR],
        help="One or more TRADING_DAYS_PER_YEAR values.",
    )
    parser.add_argument(
        "--holdout-days",
        nargs="+",
        type=int,
        default=[HOLDOUT_DAYS],
        help="One or more final holdout window sizes, in test dates.",
    )
    parser.add_argument(
        "--live-model",
        nargs="+",
        default=[LIVE_MODEL],
        choices=[
            "ensemble",
            "specialist_ensemble",
            "lgbm_deep",
            "lgbm_deep_returns_momentum",
            "lgbm_deep_corr_regime",
            "lgbm_deep_volatility",
            "rf",
            "rf_returns_momentum",
            "rf_corr_regime",
            "logreg",
            "logreg_returns_momentum",
            "logreg_corr_regime",
            "logreg_volatility",
        ],
        help="One or more live decision models.",
    )
    parser.add_argument(
        "--live-decision-mode",
        nargs="+",
        default=["top1"],
        choices=["top1"],
        help="Top-1 only live decision mode.",
    )
    parser.add_argument(
        "--specialist-ensemble-members",
        nargs="+",
        default=SPECIALIST_ENSEMBLE_MEMBERS,
        help="Model names to combine in the rank-based specialist ensemble.",
    )
    parser.add_argument(
        "--model-router-candidates",
        nargs="+",
        default=MODEL_ROUTER_CANDIDATES,
        help="Model names eligible for sticky specialist routing.",
    )
    parser.add_argument(
        "--specialist-weighting-mode",
        default=SPECIALIST_WEIGHTING_MODE,
        choices=["equal", "soft_dynamic", "winner_take_all", "winner_take_most", "sticky_winner"],
        help="Weighting mode for the specialist ensemble.",
    )
    parser.add_argument(
        "--router-selection-mode",
        default=ROUTER_SELECTION_MODE,
        choices=["single", "top3_blend", "top5_blend"],
        help="How the sticky router converts the trailing-ranked candidates into live exposure.",
    )
    parser.add_argument(
        "--router-min-edge-over-next",
        type=float,
        default=ROUTER_MIN_EDGE_OVER_NEXT,
        help="Minimum trailing edge required for the rank-1 candidate over the next-ranked candidate before using the router pick.",
    )
    parser.add_argument(
        "--router-fallback-model",
        default=ROUTER_FALLBACK_MODEL,
        help="Fallback model to hold when the router conviction gate is not met.",
    )
    parser.add_argument(
        "--specialist-weight-lookback-days",
        type=int,
        default=SPECIALIST_WEIGHT_LOOKBACK_DAYS,
        help="Trailing out-of-sample days used for dynamic specialist weights.",
    )
    parser.add_argument(
        "--specialist-min-model-hold-days",
        type=int,
        default=SPECIALIST_MIN_MODEL_HOLD_DAYS,
        help="Minimum days to keep the current sticky specialist before switching.",
    )
    parser.add_argument(
        "--specialist-switch-margin-min-avg-ev",
        type=float,
        default=SPECIALIST_SWITCH_MARGIN_MIN_AVG_EV,
        help="Minimum trailing avg ev_target margin required for sticky specialist switching.",
    )
    parser.add_argument(
        "--specialist-switch-require-positive-ev",
        type=str,
        default=str(SPECIALIST_SWITCH_REQUIRE_POSITIVE_EV).lower(),
        choices=["true", "false"],
        help="Require the challenger sticky specialist to have positive trailing avg ev_target before switching.",
    )
    parser.add_argument(
        "--retrain-deterioration-lookback-days",
        type=int,
        default=RETRAIN_DETERIORATION_LOOKBACK_DAYS,
        help="Current-window out-of-sample lookback used for early retrain checks.",
    )
    parser.add_argument(
        "--retrain-deterioration-min-win-rate",
        type=float,
        default=RETRAIN_DETERIORATION_MIN_WIN_RATE,
        help="Early retrain trigger when trailing Top-1 win rate is at or below this value.",
    )
    parser.add_argument(
        "--retrain-deterioration-max-avg-ev",
        type=float,
        default=RETRAIN_DETERIORATION_MAX_AVG_EV,
        help="Early retrain trigger when trailing Top-1 average ev_target is at or below this value.",
    )
    return parser.parse_args()


def format_run_name(params: dict[str, int | float]) -> str:
    return (
        f"fit{params['fit_days']}_"
        f"step{params['step_days']}_"
        f"reb{params['rebalance_days']}_"
        f"h{params['horizon']}_"
        f"cost{params['transaction_loss_pct']}_"
        f"tdpy{params['trading_days_per_year']}_"
        f"holdout{params['holdout_days']}_"
        f"live{params['live_model']}_"
        f"mode{params['live_decision_mode']}"
    )


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    combos = list(
        itertools.product(
            args.fit_days,
            args.step_days,
            args.rebalance_days,
            args.horizon,
            args.transaction_loss_pct,
            args.trading_days_per_year,
            args.holdout_days,
            args.live_model,
            args.live_decision_mode,
        )
    )
    print(f"Running {len(combos)} experiment(s) using {csv_path.resolve()}")

    summary_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for idx, combo in enumerate(combos, start=1):
        params = {
            "fit_days": combo[0],
            "step_days": combo[1],
            "rebalance_days": combo[2],
            "horizon": combo[3],
            "transaction_loss_pct": combo[4],
            "trading_days_per_year": combo[5],
            "holdout_days": combo[6],
            "live_model": combo[7],
            "live_decision_mode": combo[8],
        }
        run_name = format_run_name(params)
        run_dir = results_dir / run_name
        print(f"[{idx}/{len(combos)}] {run_name}")

        def log(message: str) -> None:
            print(f"  {message}")

        try:
            stats_df, plot_path, _, _ = run_pipeline(
                csv_path=str(csv_path),
                fit_days=params["fit_days"],
                model_fit_windows=args.model_fit_windows,
                step_days=params["step_days"],
                rebalance_days=params["rebalance_days"],
                horizon=params["horizon"],
                transaction_loss_pct=params["transaction_loss_pct"],
                trading_days_per_year=params["trading_days_per_year"],
                holdout_days=params["holdout_days"],
                live_model=params["live_model"],
                specialist_weighting_mode=args.specialist_weighting_mode,
                specialist_ensemble_models=args.specialist_ensemble_members,
                model_router_candidates=args.model_router_candidates,
                router_selection_mode=args.router_selection_mode,
                router_min_edge_over_next=args.router_min_edge_over_next,
                router_fallback_model=args.router_fallback_model,
                specialist_weight_lookback_days=args.specialist_weight_lookback_days,
                specialist_min_model_hold_days=args.specialist_min_model_hold_days,
                specialist_switch_margin_min_avg_ev=args.specialist_switch_margin_min_avg_ev,
                specialist_switch_require_positive_ev=(args.specialist_switch_require_positive_ev == "true"),
                retrain_deterioration_lookback_days=args.retrain_deterioration_lookback_days,
                retrain_deterioration_min_win_rate=args.retrain_deterioration_min_win_rate,
                retrain_deterioration_max_avg_ev=args.retrain_deterioration_max_avg_ev,
                output_dir=run_dir,
                log_fn=log,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"run_name": run_name, **params, "error": str(exc)})
            print(f"  FAILED: {exc}")
            continue

        for _, row in stats_df.iterrows():
            summary_rows.append(
                {
                    "run_name": run_name,
                    **params,
                    "model": row["model"],
                    "Annualized Return": row["Annualized Return"],
                    "Annualized Vol": row["Annualized Vol"],
                    "Sharpe": row["Sharpe"],
                    "Cumulative Return": row["Cumulative Return"],
                    "plot_path": str(plot_path),
                    "output_dir": str(run_dir.resolve()),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    failures_df = pd.DataFrame(failures)

    summary_path = results_dir / "experiment_summary.csv"
    failures_path = results_dir / "experiment_failures.csv"
    best_path = results_dir / "best_by_strategy.csv"

    summary_df.to_csv(summary_path, index=False)
    failures_df.to_csv(failures_path, index=False)

    if not summary_df.empty:
        best_df = (
            summary_df.sort_values(["model", "Sharpe"], ascending=[True, False])
            .groupby("model", as_index=False)
            .first()
        )
        best_df.to_csv(best_path, index=False)
        print(f"Saved summary to: {summary_path.resolve()}")
        print(f"Saved best-by-strategy table to: {best_path.resolve()}")
    else:
        print("No successful experiments were completed.")

    if not failures_df.empty:
        print(f"Saved failure log to: {failures_path.resolve()}")


if __name__ == "__main__":
    main()
