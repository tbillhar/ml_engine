"""Raw FX data download helpers for multiple data sources."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import yfinance as yf

from src.config import (
    ALPHAVANTAGE_API_KEY,
    FX_PAIRS,
    RAW_DATA_SOURCE,
    YAHOO_DOWNLOAD_INTERVAL,
    YAHOO_DOWNLOAD_PERIOD,
)

LogFn = Callable[[str], None] | None

SUPPORTED_DATA_SOURCES = ("yfinance", "alphavantage", "ecb")

DATA_SOURCE_INFO: dict[str, dict[str, str]] = {
    "yfinance": {
        "type": "Market OHLC",
        "expected_history_depth": "Current app default: about 1000 calendar days",
        "warning": "Default and easiest option. History depth is controlled by YAHOO_DOWNLOAD_PERIOD.",
    },
    "alphavantage": {
        "type": "Market OHLC",
        "expected_history_depth": "Many majors can go back to 2007, but some crosses only to around 2023",
        "warning": "Requires Alpha Vantage API key and is subject to free-tier rate limits.",
    },
    "ecb": {
        "type": "Reference-rate daily close",
        "expected_history_depth": "About January 1999 onward for your currency set",
        "warning": "ECB is reference-rate only, not true OHLC. Open/High/Low/Close are all set to the same daily reference value.",
    },
}


def _pair_to_symbols(pair: str) -> tuple[str, str]:
    symbol = pair.replace("=X", "")
    if len(symbol) != 6:
        raise ValueError(f"Unsupported FX ticker format: {pair}")
    return symbol[:3], symbol[3:]


def _align_to_common_date_history(tidy: pd.DataFrame) -> pd.DataFrame:
    if tidy.empty:
        return tidy

    pair_dates = tidy.groupby("pair")["Date"].apply(lambda s: set(pd.to_datetime(s, utc=True)))
    common_dates = set.intersection(*pair_dates.tolist())
    if not common_dates:
        raise ValueError("No overlapping dates across all FX pairs for the selected data source")

    aligned = tidy[tidy["Date"].isin(common_dates)].copy()
    aligned = aligned.sort_values(["Date", "pair"]).reset_index(drop=True)

    overlap_start = aligned["Date"].min()
    overlap_end = aligned["Date"].max()
    return aligned[(aligned["Date"] >= overlap_start) & (aligned["Date"] <= overlap_end)].copy()


def _save_tidy_output(tidy: pd.DataFrame, output_csv: Path, log_fn: LogFn = None) -> Path:
    def log(message: str) -> None:
        if log_fn:
            log_fn(message)

    tidy["Date"] = pd.to_datetime(tidy["Date"], utc=True)
    tidy = tidy[["Date", "pair", "Open", "High", "Low", "Close"]]
    tidy = tidy.dropna().sort_values(["pair", "Date"]).reset_index(drop=True)
    tidy = _align_to_common_date_history(tidy)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(output_csv, index=False)

    overlap_start = tidy["Date"].min()
    overlap_end = tidy["Date"].max()
    log(
        "Saved raw FX data to: "
        f"{output_csv.resolve()} | common overlap: {overlap_start.date()} to {overlap_end.date()} "
        f"({tidy['Date'].nunique()} dates)"
    )
    return output_csv


def _download_yfinance(
    pairs: Sequence[str],
    period: str,
    interval: str,
    log_fn: LogFn = None,
) -> pd.DataFrame:
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
        tidy = tidy.rename(columns={tidy.columns[0]: "Date"})
    return tidy


def _read_json_url(url: str) -> dict:
    try:
        with urlopen(url) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:  # noqa: PERF203
        raise ValueError(f"HTTP error downloading data: {exc}") from exc
    except URLError as exc:
        raise ValueError(f"Network error downloading data: {exc}") from exc


def _download_alphavantage(
    pairs: Sequence[str],
    api_key: str,
    log_fn: LogFn = None,
) -> pd.DataFrame:
    def log(message: str) -> None:
        if log_fn:
            log_fn(message)

    if not api_key:
        raise ValueError("Alpha Vantage requires an API key")

    rows: list[pd.DataFrame] = []
    for idx, pair in enumerate(pairs, start=1):
        base, quote = _pair_to_symbols(pair)
        params = {
            "function": "FX_DAILY",
            "from_symbol": base,
            "to_symbol": quote,
            "outputsize": "full",
            "apikey": api_key,
        }
        url = f"https://www.alphavantage.co/query?{urlencode(params)}"
        log(f"Downloading {idx}/{len(pairs)} from Alpha Vantage: {pair}")
        payload = _read_json_url(url)

        if "Error Message" in payload:
            raise ValueError(f"Alpha Vantage error for {pair}: {payload['Error Message']}")
        if "Note" in payload:
            raise ValueError(f"Alpha Vantage limit hit for {pair}: {payload['Note']}")

        series = payload.get("Time Series FX (Daily)")
        if not series:
            raise ValueError(f"Alpha Vantage returned no daily FX series for {pair}")

        pair_df = (
            pd.DataFrame.from_dict(series, orient="index")
            .rename(
                columns={
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "4. close": "Close",
                }
            )
            .reset_index()
            .rename(columns={"index": "Date"})
        )
        for col in ("Open", "High", "Low", "Close"):
            pair_df[col] = pd.to_numeric(pair_df[col], errors="coerce")
        pair_df["Date"] = pd.to_datetime(pair_df["Date"], utc=True)
        pair_df["pair"] = pair
        rows.append(pair_df[["Date", "pair", "Open", "High", "Low", "Close"]].dropna())

    return pd.concat(rows, ignore_index=True)


def _download_ecb(
    pairs: Sequence[str],
    log_fn: LogFn = None,
) -> pd.DataFrame:
    def log(message: str) -> None:
        if log_fn:
            log_fn(message)

    url = "https://data-api.ecb.europa.eu/service/data/EXR/D..EUR.SP00.A"
    params = {"format": "csvdata"}
    log("Downloading ECB EUR reference-rate history")
    try:
        csv_bytes = urlopen(f"{url}?{urlencode(params)}").read()  # noqa: S310
    except HTTPError as exc:
        raise ValueError(f"HTTP error downloading ECB data: {exc}") from exc
    except URLError as exc:
        raise ValueError(f"Network error downloading ECB data: {exc}") from exc
    exr = pd.read_csv(BytesIO(csv_bytes))

    exr = exr.rename(
        columns={
            "TIME_PERIOD": "Date",
            "OBS_VALUE": "value",
            "CURRENCY": "currency",
        }
    )
    if not {"Date", "value", "currency"}.issubset(exr.columns):
        raise ValueError("ECB response schema changed; expected Date/value/currency columns")

    exr["Date"] = pd.to_datetime(exr["Date"], utc=True)
    exr["value"] = pd.to_numeric(exr["value"], errors="coerce")
    eur_rates = (
        exr[["Date", "currency", "value"]]
        .dropna()
        .pivot(index="Date", columns="currency", values="value")
        .sort_index()
    )
    eur_rates["EUR"] = 1.0

    rows: list[pd.DataFrame] = []
    for pair in pairs:
        base, quote = _pair_to_symbols(pair)
        if base not in eur_rates.columns or quote not in eur_rates.columns:
            raise ValueError(f"ECB does not provide enough currencies to derive pair: {pair}")
        close = eur_rates[quote] / eur_rates[base]
        pair_df = pd.DataFrame(
            {
                "Date": close.index,
                "pair": pair,
                "Open": close.values,
                "High": close.values,
                "Low": close.values,
                "Close": close.values,
            }
        ).dropna()
        rows.append(pair_df)

    log(
        "ECB provides reference rates rather than true OHLC; "
        "Open/High/Low/Close are all set to the same daily reference value"
    )
    return pd.concat(rows, ignore_index=True)


def download_raw_fx_data(
    output_csv: Path,
    pairs: Sequence[str] = FX_PAIRS,
    source: str = RAW_DATA_SOURCE,
    period: str = YAHOO_DOWNLOAD_PERIOD,
    interval: str = YAHOO_DOWNLOAD_INTERVAL,
    alphavantage_api_key: str = ALPHAVANTAGE_API_KEY,
    log_fn: LogFn = None,
) -> Path:
    """Download raw FX data from the selected source and save tidy CSV output."""

    source = source.strip().lower()
    if source not in SUPPORTED_DATA_SOURCES:
        raise ValueError(
            f"Unsupported data source '{source}'. Expected one of: {', '.join(SUPPORTED_DATA_SOURCES)}"
        )

    if source == "yfinance":
        tidy = _download_yfinance(pairs=pairs, period=period, interval=interval, log_fn=log_fn)
    elif source == "alphavantage":
        tidy = _download_alphavantage(
            pairs=pairs,
            api_key=alphavantage_api_key,
            log_fn=log_fn,
        )
    else:
        tidy = _download_ecb(pairs=pairs, log_fn=log_fn)

    return _save_tidy_output(tidy, output_csv=output_csv, log_fn=log_fn)
