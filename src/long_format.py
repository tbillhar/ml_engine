"""Wide-to-long conversion helpers for FX ranking."""

from __future__ import annotations

import pandas as pd

from src.config import PAIR_PREFIXES


def is_pair_feature(col: str, pair_prefixes: list[str] | None = None) -> bool:
    """Check whether a column is treated as a per-pair feature."""
    prefixes = pair_prefixes or PAIR_PREFIXES
    return any(col.startswith(p) for p in prefixes)


def build_long(df: pd.DataFrame, pairs: list[str]) -> pd.DataFrame:
    """Convert wide matrix to long format and build per-date relevance labels."""
    global_features = [
        c for c in df.columns
        if c != "Date"
        and not is_pair_feature(c)
        and not c.startswith("next_ret_")
        and not c.startswith("future_ret_")
    ]

    rows = []
    for _, row in df.iterrows():
        d0 = row["Date"]
        for pair in pairs:
            fut_col = f"future_ret_{pair}"
            next_col = f"next_ret_{pair}"
            fut = row[fut_col]
            next_ret = row[next_col]

            entry = {
                "Date": d0,
                "pair": pair,
                "next_ret": next_ret,
                "future_ret": fut,
            }

            for pref in PAIR_PREFIXES:
                col = f"{pref}{pair}"
                if col in df.columns:
                    entry[col] = row[col]

            for col in global_features:
                entry[col] = row[col]

            rows.append(entry)

    long = pd.DataFrame(rows)
    long = long.sort_values(["Date", "pair"]).reset_index(drop=True)

    long["rel"] = (
        long.groupby("Date")["future_ret"]
        .rank(method="dense", ascending=True)
        .astype(int) - 1
    )
    return long
