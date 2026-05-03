"""Run the FX pipeline from the command line using shared pipeline modules."""

from pathlib import Path

from src.config import (
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


def main() -> None:
    csv_path = "data/fx_features_wide.csv"
    output_dir = Path("outputs")
    stats_df, _, _, diagnostics_summary = run_pipeline(
        csv_path=csv_path,
        fit_days=FIT_DAYS,
        model_fit_windows=MODEL_FIT_WINDOWS,
        step_days=STEP_DAYS,
        rebalance_days=REBALANCE_DAYS,
        horizon=HORIZON,
        transaction_loss_pct=TRANSACTION_LOSS_PCT,
        trading_days_per_year=TRADING_DAYS_PER_YEAR,
        holdout_days=HOLDOUT_DAYS,
        live_model=LIVE_MODEL,
        specialist_weighting_mode=SPECIALIST_WEIGHTING_MODE,
        router_selection_mode=ROUTER_SELECTION_MODE,
        router_min_edge_over_next=ROUTER_MIN_EDGE_OVER_NEXT,
        router_fallback_model=ROUTER_FALLBACK_MODEL,
        specialist_ensemble_models=SPECIALIST_ENSEMBLE_MEMBERS,
        model_router_candidates=MODEL_ROUTER_CANDIDATES,
        specialist_weight_lookback_days=SPECIALIST_WEIGHT_LOOKBACK_DAYS,
        specialist_min_model_hold_days=SPECIALIST_MIN_MODEL_HOLD_DAYS,
        specialist_switch_margin_min_avg_ev=SPECIALIST_SWITCH_MARGIN_MIN_AVG_EV,
        specialist_switch_require_positive_ev=SPECIALIST_SWITCH_REQUIRE_POSITIVE_EV,
        retrain_deterioration_lookback_days=RETRAIN_DETERIORATION_LOOKBACK_DAYS,
        retrain_deterioration_min_win_rate=RETRAIN_DETERIORATION_MIN_WIN_RATE,
        retrain_deterioration_max_avg_ev=RETRAIN_DETERIORATION_MAX_AVG_EV,
        output_dir=output_dir,
        log_fn=print,
    )

    for _, row in stats_df.iterrows():
        model = row["model"]
        stats = row.drop(labels=["model"]).to_dict()
        print(f"{model} stats:", stats)

    print("=====================================")
    print(diagnostics_summary)
    print("=====================================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Outputs saved to: {output_dir.resolve()}")
    print("Files created:")
    for path in sorted(output_dir.iterdir()):
        print(f" - {path.name}")
    print("=====================================")


if __name__ == "__main__":
    main()
