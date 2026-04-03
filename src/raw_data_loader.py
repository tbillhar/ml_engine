"""Yahoo Finance FX data download helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import yfinance as yf

from src.config import FX_PAIRS, YAHOO_DOWNLOAD_INTERVAL, YAHOO_DOWNLOAD_PERIOD

LogFn = Callable[[str], None] | None


def download_raw_fx_data(
    output_csv: Path,
    pairs: Sequence[str] = FX_PAIRS,
    period: str = YAHOO_DOWNLOAD_PERIOD,
    interval: str = YAHOO_DOWNLOAD_INTERVAL,
    log_fn: LogFn = None,
) -> Path:
    """Download raw FX OHLC data from Yahoo Finance and save tidy CSV output."""

    def log(message: str) -> None:
        if log_fn:
            log_fn(message)

    log(f"Downloading {len(pairs)} FX pairs from Yahoo Finance")
    raw = yf.download(
        tickers=list(pairs),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError("Yahoo Finance returned no data")

    rows: list[pd.DataFrame] = []
    for pair in pairs:
        if pair not in raw:
            raise ValueError(f"Missing downloaded data for pair: {pair}")
        pair_df = raw[pair].reset_index().dropna()
        if pair_df.empty:
            raise ValueError(f"Downloaded data is empty for pair: {pair}")
        pair_df["pair"] = pair
        rows.append(pair_df)

    tidy = pd.concat(rows, ignore_index=True)
    tidy = tidy.rename(columns={"index": "Date"})
    if "Date" not in tidy.columns:
        first_column = tidy.columns[0]
        tidy = tidy.rename(columns={first_column: "Date"})
    tidy["Date"] = pd.to_datetime(tidy["Date"], utc=True)
    tidy = tidy[["Date", "pair", "Open", "High", "Low", "Close"]]
    tidy = tidy.sort_values(["pair", "Date"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(output_csv, index=False)
    log(f"Saved raw FX data to: {output_csv.resolve()}")
    return output_csv
