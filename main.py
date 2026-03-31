"""Example script running the FX notebook pipeline via reusable modules."""

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

    df = load_csv(csv_path)
    df = parse_and_sort_dates(df)

    ret_cols = [c for c in df.columns if c.startswith("ret_")]
    pairs = [c.replace("ret_", "") for c in ret_cols]

    df = compute_future_returns(df, horizon=HORIZON)
    long_df = build_long(df, pairs)

    pred_df = run_walkforward_model(
        long_df,
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        step_days=STEP_DAYS,
    )

    pnl_top1 = compute_topK_pnl(pred_df, 1)
    pnl_top3 = compute_topK_pnl(pred_df, 3)
    pnl_top5 = compute_topK_pnl(pred_df, 5)
    pnl_eq = compute_equal_weight_pnl(pred_df)

    _ = cumulative_curve(pnl_top1)
    _ = cumulative_curve(pnl_top3)
    _ = cumulative_curve(pnl_top5)
    _ = cumulative_curve(pnl_eq)

    print("Top-1 stats:", perf_stats(pnl_top1, horizon_days=HORIZON))
    print("Top-3 stats:", perf_stats(pnl_top3, horizon_days=HORIZON))
    print("Top-5 stats:", perf_stats(pnl_top5, horizon_days=HORIZON))
    print("Equal-weight stats:", perf_stats(pnl_eq, horizon_days=HORIZON))


if __name__ == "__main__":
    main()
