"""Data-loading and validation helpers for FX pipeline."""

from __future__ import annotations

import pandas as pd


REQUIRED_DATE_COLUMN = "Date"


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "Date"})
    return df


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate that all required columns exist."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def parse_and_sort_dates(df: pd.DataFrame, date_col: str = REQUIRED_DATE_COLUMN) -> pd.DataFrame:
    """Parse a date column as UTC and sort ascending."""
    validate_columns(df, [date_col])
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], utc=True)
    out = out.sort_values(date_col).reset_index(drop=True)
    return out
