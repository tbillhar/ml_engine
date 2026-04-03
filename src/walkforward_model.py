"""Sliding-window walk-forward ranking model utilities."""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMRanker


def sliding_windows(dates, train_len: int, test_len: int, step: int):
    """Yield (train_dates, test_dates) for rolling/sliding windows."""
    n = len(dates)
    for start in range(0, n - train_len - test_len + 1, step):
        train = dates[start:start + train_len]
        test = dates[start + train_len:start + train_len + test_len]
        yield train, test


def make_xy(subdf: pd.DataFrame, date_list, feature_cols: list[str]):
    """Build sorted subset, design matrix, labels, and group sizes."""
    s = subdf[subdf["Date"].isin(date_list)].sort_values(["Date", "pair"])
    X = s[feature_cols].values
    y = s["rel"].values
    g = s.groupby("Date").size().values
    return s, X, y, g


def run_walkforward_model(
    long_df: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
) -> pd.DataFrame:
    """Run sliding-window LightGBM ranker and return concatenated predictions."""
    feature_cols = [
        c for c in long_df.columns
        if c not in ["Date", "pair", "next_ret", "future_ret", "rel"]
    ]

    all_dates = sorted(long_df["Date"].unique())

    ranker = LGBMRanker(
        objective="lambdarank",
        boosting_type="gbdt",
        num_leaves=63,
        learning_rate=0.05,
        n_estimators=600,
        subsample=0.8,
        colsample_bytree=0.8,
        metric="ndcg",
        random_state=42,
    )

    all_test_chunks = []

    for train_dates, test_dates in sliding_windows(all_dates, train_days, test_days, step_days):
        train_sub, X_train, y_train, g_train = make_xy(long_df, train_dates, feature_cols)
        test_sub, X_test, y_test, g_test = make_xy(long_df, test_dates, feature_cols)

        ranker.fit(
            X_train,
            y_train,
            group=g_train,
            eval_set=[(X_test, y_test)],
            eval_group=[g_test],
            eval_at=[1, 3, 5],
            callbacks=[lgb.log_evaluation(50)],
        )

        preds = ranker.predict(X_test)
        test_sub = test_sub.copy()
        test_sub["pred"] = preds
        all_test_chunks.append(
            test_sub[["Date", "pair", "next_ret", "future_ret", "rel", "pred"]]
        )

    pred_df = pd.concat(all_test_chunks, ignore_index=True)
    pred_df = pred_df.sort_values(["Date", "pair"]).reset_index(drop=True)
    return pred_df
