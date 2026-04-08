"""Run the FX pipeline from the command line using shared pipeline modules."""

from pathlib import Path

from src.config import (
    FIT_DAYS,
    HORIZON,
    HOLDOUT_DAYS,
    LIVE_MODEL,
    SPECIALIST_ENSEMBLE_MEMBERS,
    STEP_DAYS,
    TEST_DAYS,
    TRADING_DAYS_PER_YEAR,
    TRANSACTION_LOSS_PCT,
)
from src.pipeline_runner import run_pipeline


def main() -> None:
    csv_path = "data/fx_features_wide.csv"
    output_dir = Path("outputs")
    stats_df, _, diagnostics_summary = run_pipeline(
        csv_path=csv_path,
        fit_days=FIT_DAYS,
        test_days=TEST_DAYS,
        step_days=STEP_DAYS,
        horizon=HORIZON,
        transaction_loss_pct=TRANSACTION_LOSS_PCT,
        trading_days_per_year=TRADING_DAYS_PER_YEAR,
        holdout_days=HOLDOUT_DAYS,
        live_model=LIVE_MODEL,
        specialist_ensemble_models=SPECIALIST_ENSEMBLE_MEMBERS,
        output_dir=output_dir,
        log_fn=print,
    )

    for _, row in stats_df.iterrows():
        strategy = row["strategy"]
        stats = row.drop(labels=["strategy"]).to_dict()
        print(f"{strategy} stats:", stats)

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
