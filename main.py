"""Run the FX pipeline from the command line using the shared modules."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import HORIZON, STEP_DAYS, TEST_DAYS, TRAIN_DAYS
from src.data_pipeline import load_csv, parse_and_sort_dates
from src.feature_engineering import compute_future_returns
from src.long_format import build_long
from src.pnl_analysis import (
    compute_equal_weight_pnl,
    compute_topK_pnl,
    cumulative_curve,
    perf_stats,
)
from src.walkforward_model import run_walkforward_model


def main() -> None:
    csv_path = "data/fx_features_wide.csv"
    output_dir = Path("outputs")
    # Keep batch-mode outputs in a dedicated folder for later inspection.
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading input CSV: {csv_path}")
    df = load_csv(csv_path)
    df = parse_and_sort_dates(df)

    ret_cols = [c for c in df.columns if c.startswith("ret_")]
    pairs = [c.replace("ret_", "") for c in ret_cols]
    print(f"Detected {len(pairs)} pairs")

    print(f"Computing {HORIZON}-day future returns")
    df = compute_future_returns(df, horizon=HORIZON)

    print("Building long-format ranking table")
    long_df = build_long(df, pairs)

    print(
        f"Running walk-forward model "
        f"(train={TRAIN_DAYS}, test={TEST_DAYS}, step={STEP_DAYS})"
    )
    pred_df = run_walkforward_model(
        long_df,
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        step_days=STEP_DAYS,
    )

    print("Computing PnL series")
    pnl_top1 = compute_topK_pnl(pred_df, 1)
    pnl_top3 = compute_topK_pnl(pred_df, 3)
    pnl_top5 = compute_topK_pnl(pred_df, 5)
    pnl_eq = compute_equal_weight_pnl(pred_df)

    cum1 = cumulative_curve(pnl_top1)
    cum3 = cumulative_curve(pnl_top3)
    cum5 = cumulative_curve(pnl_top5)
    cum_eq = cumulative_curve(pnl_eq)

    top1_stats = perf_stats(pnl_top1, horizon_days=HORIZON)
    top3_stats = perf_stats(pnl_top3, horizon_days=HORIZON)
    top5_stats = perf_stats(pnl_top5, horizon_days=HORIZON)
    eq_stats = perf_stats(pnl_eq, horizon_days=HORIZON)

    print("Top-1 stats:", top1_stats)
    print("Top-3 stats:", top3_stats)
    print("Top-5 stats:", top5_stats)
    print("Equal-weight stats:", eq_stats)

    print("Saving CSV outputs")
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

    print("Saving PnL plot")
    plt.figure(figsize=(12, 6))
    plt.plot(cum1["Date"], cum1["cum"], label="Top-1", linewidth=2.5)
    plt.plot(cum3["Date"], cum3["cum"], label="Top-3", linewidth=2.2)
    plt.plot(cum5["Date"], cum5["cum"], label="Top-5", linewidth=2.0)
    plt.plot(
        cum_eq["Date"],
        cum_eq["cum"],
        label="Equal-Weight",
        linestyle="--",
        linewidth=1.8,
    )

    plt.title("Cumulative PnL")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pnl_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("=====================================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Outputs saved to: {output_dir.resolve()}")
    print("Files created:")
    for path in sorted(output_dir.iterdir()):
        print(f" - {path.name}")
    print("=====================================")


if __name__ == "__main__":
    main()
