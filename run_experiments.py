"""Run parameter sweeps for the FX pipeline and compare results."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from src.config import (
    CALIBRATION_DAYS,
    FEATURE_DATA_FILENAME,
    FIT_DAYS,
    HORIZON,
    LIVE_DECISION_MODE,
    HOLDOUT_DAYS,
    LIVE_MODEL,
    P_WIN_THRESHOLD,
    STEP_DAYS,
    TEST_DAYS,
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
        help="One or more FIT_DAYS values.",
    )
    parser.add_argument(
        "--calibration-days",
        nargs="+",
        type=int,
        default=[CALIBRATION_DAYS],
        help="One or more CALIBRATION_DAYS values.",
    )
    parser.add_argument(
        "--test-days",
        nargs="+",
        type=int,
        default=[TEST_DAYS],
        help="One or more TEST_DAYS values.",
    )
    parser.add_argument(
        "--step-days",
        nargs="+",
        type=int,
        default=[STEP_DAYS],
        help="One or more STEP_DAYS values.",
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
        "--p-win-threshold",
        nargs="+",
        type=float,
        default=[P_WIN_THRESHOLD],
        help="One or more calibrated win-probability thresholds between 0 and 1.",
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
            "ensemble_brier",
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
        default=[LIVE_DECISION_MODE],
        choices=["threshold", "threshold_top1", "threshold_top3", "top1", "top3", "top_decile"],
        help="One or more live decision modes.",
    )
    return parser.parse_args()


def format_run_name(params: dict[str, int | float]) -> str:
    return (
        f"fit{params['fit_days']}_"
        f"cal{params['calibration_days']}_"
        f"test{params['test_days']}_"
        f"step{params['step_days']}_"
        f"h{params['horizon']}_"
        f"cost{params['transaction_loss_pct']}_"
        f"tdpy{params['trading_days_per_year']}_"
        f"pwin{params['p_win_threshold']}_"
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
            args.calibration_days,
            args.test_days,
            args.step_days,
            args.horizon,
            args.transaction_loss_pct,
            args.trading_days_per_year,
            args.p_win_threshold,
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
            "calibration_days": combo[1],
            "test_days": combo[2],
            "step_days": combo[3],
            "horizon": combo[4],
            "transaction_loss_pct": combo[5],
            "trading_days_per_year": combo[6],
            "p_win_threshold": combo[7],
            "holdout_days": combo[8],
            "live_model": combo[9],
            "live_decision_mode": combo[10],
        }
        run_name = format_run_name(params)
        run_dir = results_dir / run_name
        print(f"[{idx}/{len(combos)}] {run_name}")

        def log(message: str) -> None:
            print(f"  {message}")

        try:
            stats_df, plot_path, _ = run_pipeline(
                csv_path=str(csv_path),
                fit_days=params["fit_days"],
                calibration_days=params["calibration_days"],
                test_days=params["test_days"],
                step_days=params["step_days"],
                horizon=params["horizon"],
                transaction_loss_pct=params["transaction_loss_pct"],
                trading_days_per_year=params["trading_days_per_year"],
                p_win_threshold=params["p_win_threshold"],
                holdout_days=params["holdout_days"],
                live_model=params["live_model"],
                live_decision_mode=params["live_decision_mode"],
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
                    "strategy": row["strategy"],
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
            summary_df.sort_values(["strategy", "Sharpe"], ascending=[True, False])
            .groupby("strategy", as_index=False)
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
