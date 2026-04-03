"""Feature dataset builder from raw Yahoo Finance OHLC data."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

LogFn = Callable[[str], None] | None


def _rolling_avg_corr(
    wide_ret: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    features: list[list[float]] = []
    for i in range(len(wide_ret)):
        if i < window:
            features.append([np.nan] * len(wide_ret.columns))
            continue
        corr_mat = wide_ret.iloc[i - window:i].corr()
        np.fill_diagonal(corr_mat.values, np.nan)
        features.append(corr_mat.mean(axis=1).values.tolist())
    return pd.DataFrame(
        features,
        index=wide_ret.index,
        columns=[f"corr{window}_{c.replace('ret_', '')}" for c in wide_ret.columns],
    )


def _rolling_linear_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = np.square(x_centered).sum()

    def slope(values: np.ndarray) -> float:
        centered = values - values.mean()
        return float(np.dot(x_centered, centered) / denom)

    return series.rolling(window=window, min_periods=window).apply(slope, raw=True)


def _build_rolling_pca_features(
    wide_ret: pd.DataFrame,
    window: int = 120,
    n_components: int = 3,
) -> pd.DataFrame:
    pc_values = np.full((len(wide_ret), n_components), np.nan, dtype=float)
    explained_values = np.full((len(wide_ret), n_components), np.nan, dtype=float)

    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    for i in range(window, len(wide_ret)):
        hist = wide_ret.iloc[i - window:i]
        if hist.isna().any().any():
            continue
        scaled = scaler.fit_transform(hist.values)
        pca.fit(scaled)
        current_scaled = scaler.transform(wide_ret.iloc[[i]].values)
        transformed = pca.transform(current_scaled)[0]
        pc_values[i, :] = transformed
        explained_values[i, :] = pca.explained_variance_ratio_

    data = {}
    for idx in range(n_components):
        data[f"PC{idx + 1}"] = pc_values[:, idx]
        data[f"PC{idx + 1}_evr"] = explained_values[:, idx]
    return pd.DataFrame(data, index=wide_ret.index)


def build_feature_dataset(
    raw_csv_path: str | Path,
    output_csv_path: str | Path,
    log_fn: LogFn = None,
) -> Path:
    """Build the full feature matrix from raw OHLC input and save it."""

    def log(message: str) -> None:
        if log_fn:
            log_fn(message)

    raw_csv_path = Path(raw_csv_path)
    output_csv_path = Path(output_csv_path)

    log(f"Loading raw FX data from: {raw_csv_path}")
    tidy = pd.read_csv(raw_csv_path)
    tidy["Date"] = pd.to_datetime(tidy["Date"], utc=True)
    tidy = tidy.sort_values(["pair", "Date"]).reset_index(drop=True)

    log("Computing base return, volatility, and momentum features")
    df = tidy.copy()
    df["log_close"] = np.log(df["Close"])
    df["log_ret"] = df.groupby("pair")["log_close"].diff()
    df["vol30"] = (
        df.groupby("pair")["log_ret"]
        .rolling(30)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["mom5"] = df.groupby("pair")["log_close"].diff(5)
    df["mom10"] = df.groupby("pair")["log_close"].diff(10)

    wide_ret = df.pivot(index="Date", columns="pair", values="log_ret").add_prefix("ret_")
    wide_vol30 = df.pivot(index="Date", columns="pair", values="vol30").add_prefix("vol30_")
    wide_mom5 = df.pivot(index="Date", columns="pair", values="mom5").add_prefix("mom5_")
    wide_mom10 = df.pivot(index="Date", columns="pair", values="mom10").add_prefix("mom10_")

    wide = pd.concat([wide_ret, wide_vol30, wide_mom5, wide_mom10], axis=1)
    wide = wide.sort_index()

    log("Adding multi-horizon rolling correlation features")
    corr_frames = [_rolling_avg_corr(wide_ret, window) for window in (20, 40, 60, 120)]

    log("Adding relative-strength and cross-sectional rank features")
    ret1_mean = wide_ret.mean(axis=1)
    ret5 = wide_ret.rolling(5, min_periods=5).sum()
    ret10 = wide_ret.rolling(10, min_periods=10).sum()
    ret20 = wide_ret.rolling(20, min_periods=20).sum()
    rs_frames = [
        wide_ret.sub(ret1_mean, axis=0).rename(columns=lambda c: c.replace("ret_", "rs1_")),
        ret5.sub(ret5.mean(axis=1), axis=0).rename(columns=lambda c: c.replace("ret_", "rs5_")),
        ret10.sub(ret10.mean(axis=1), axis=0).rename(columns=lambda c: c.replace("ret_", "rs10_")),
        ret20.sub(ret20.mean(axis=1), axis=0).rename(columns=lambda c: c.replace("ret_", "rs20_")),
    ]
    rank_frames = [
        wide_ret.rank(axis=1, pct=True).rename(columns=lambda c: c.replace("ret_", "rank_ret_")),
        wide_mom5.rank(axis=1, pct=True).rename(columns=lambda c: c.replace("mom5_", "rank_mom5_")),
        wide_mom10.rank(axis=1, pct=True).rename(columns=lambda c: c.replace("mom10_", "rank_mom10_")),
        wide_vol30.rank(axis=1, pct=True).rename(columns=lambda c: c.replace("vol30_", "rank_vol30_")),
    ]

    log("Adding regime, calendar, and latent state features")
    avg_corr20 = _rolling_avg_corr(wide_ret, 20).mean(axis=1).rename("avg_corr20")
    avg_corr60 = _rolling_avg_corr(wide_ret, 60).mean(axis=1).rename("avg_corr60")
    regime_df = pd.DataFrame(
        {
            "market_vol20": wide_ret.abs().mean(axis=1).rolling(20, min_periods=20).mean(),
            "dispersion20": wide_ret.std(axis=1).rolling(20, min_periods=20).mean(),
            "avg_corr20": avg_corr20,
            "avg_corr60": avg_corr60,
            "day_of_week": wide_ret.index.dayofweek.astype(float),
            "is_month_end": wide_ret.index.is_month_end.astype(float),
            "is_quarter_end": wide_ret.index.is_quarter_end.astype(float),
        },
        index=wide_ret.index,
    )
    pca_df = _build_rolling_pca_features(wide_ret, window=120, n_components=3)

    log("Adding trend, mean-reversion, normalized, and turnover-aware features")
    accel_df = (wide_mom5 - wide_mom10).rename(columns=lambda c: c.replace("mom5_", "accel_"))
    slope_df = pd.DataFrame(index=wide_ret.index)
    for col in wide_ret.columns:
        slope_df[col.replace("ret_", "slope20_")] = _rolling_linear_slope(wide_ret[col], 20)
    dvol_df = wide_vol30.diff().rename(columns=lambda c: c.replace("vol30_", "dvol30_"))

    ret_mean20 = wide_ret.rolling(20, min_periods=20).mean()
    ret_std20 = wide_ret.rolling(20, min_periods=20).std()
    zret_df = ((wide_ret - ret_mean20) / ret_std20).rename(
        columns=lambda c: c.replace("ret_", "zret20_")
    )
    mom5_mean20 = wide_mom5.rolling(20, min_periods=20).mean()
    mom5_std20 = wide_mom5.rolling(20, min_periods=20).std()
    zmom5_df = ((wide_mom5 - mom5_mean20) / mom5_std20).rename(
        columns=lambda c: c.replace("mom5_", "zmom5_20_")
    )
    mom10_mean20 = wide_mom10.rolling(20, min_periods=20).mean()
    mom10_std20 = wide_mom10.rolling(20, min_periods=20).std()
    zmom10_df = ((wide_mom10 - mom10_mean20) / mom10_std20).rename(
        columns=lambda c: c.replace("mom10_", "zmom10_20_")
    )
    norm_mom5_df = wide_mom5.div(wide_vol30).rename(
        columns=lambda c: c.replace("mom5_", "norm_mom5_")
    )
    norm_mom10_df = wide_mom10.div(wide_vol30).rename(
        columns=lambda c: c.replace("mom10_", "norm_mom10_")
    )

    rank_ret = wide_ret.rank(axis=1, pct=True)
    prev_top1_df = rank_ret.eq(rank_ret.max(axis=1), axis=0).shift(1).astype(float).rename(
        columns=lambda c: c.replace("ret_", "prev_top1_")
    )
    rank_persist_df = rank_ret.rolling(5, min_periods=5).mean().rename(
        columns=lambda c: c.replace("ret_", "rank_persist5_")
    )
    signal_stability_df = (
        1.0 / (1.0 + rank_ret.diff().abs().rolling(5, min_periods=5).mean())
    ).rename(columns=lambda c: c.replace("ret_", "signal_stability5_"))

    feature_frames = [
        wide,
        *corr_frames,
        *rs_frames,
        *rank_frames,
        regime_df,
        pca_df,
        accel_df,
        slope_df,
        dvol_df,
        zret_df,
        zmom5_df,
        zmom10_df,
        norm_mom5_df,
        norm_mom10_df,
        prev_top1_df,
        rank_persist_df,
        signal_stability_df,
    ]

    feature_wide = pd.concat(feature_frames, axis=1)
    feature_wide = feature_wide.replace([np.inf, -np.inf], np.nan)
    feature_wide = feature_wide.sort_index().dropna(how="any")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    feature_wide.reset_index().to_csv(output_csv_path, index=False)
    log(
        "Saved feature matrix "
        f"with shape {feature_wide.shape} to: {output_csv_path.resolve()}"
    )
    return output_csv_path
