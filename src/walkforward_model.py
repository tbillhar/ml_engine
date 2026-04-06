"""Sliding-window walk-forward ensemble regression utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
    X = s[feature_cols]
    y = s["ev_target"].values
    g = s.groupby("Date").size().values
    return s, X, y, g


def build_models() -> dict[str, object]:
    """Construct the base regressors used in the ensemble."""
    return {
        "lgbm": LGBMRegressor(
            objective="regression",
            boosting_type="gbdt",
            num_leaves=63,
            learning_rate=0.05,
            n_estimators=400,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        ),
        "rf": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=250,
                max_depth=None,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
    }


def run_walkforward_model(
    long_df: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
    transaction_loss_pct: float,
) -> pd.DataFrame:
    """Run walk-forward ensemble regression and return concatenated predictions."""
    feature_cols = [
        c for c in long_df.columns
        if c not in ["Date", "pair", "next_ret", "future_ret", "rel", "ev_target"]
    ]

    all_dates = sorted(long_df["Date"].unique())

    all_test_chunks = []
    transaction_loss = transaction_loss_pct / 100.0

    for train_dates, test_dates in sliding_windows(all_dates, train_days, test_days, step_days):
        train_sub, X_train, y_train, g_train = make_xy(long_df, train_dates, feature_cols)
        test_sub, X_test, y_test, g_test = make_xy(long_df, test_dates, feature_cols)
        _ = (train_sub, g_train, y_test, g_test)

        test_sub = test_sub.copy()
        model_preds = []
        for model_name, model in build_models().items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            test_sub[f"pred_{model_name}"] = preds
            model_preds.append(preds)

        ensemble_pred = np.mean(np.column_stack(model_preds), axis=1)
        test_sub["pred_ensemble"] = ensemble_pred
        test_sub["pred_ensemble_gross"] = ensemble_pred + transaction_loss
        all_test_chunks.append(
            test_sub[
                [
                    "Date",
                    "pair",
                    "next_ret",
                    "future_ret",
                    "ev_target",
                    "rel",
                    "pred_lgbm",
                    "pred_rf",
                    "pred_ridge",
                    "pred_ensemble",
                    "pred_ensemble_gross",
                ]
            ]
        )

    pred_df = pd.concat(all_test_chunks, ignore_index=True)
    pred_df = pred_df.sort_values(["Date", "pair"]).reset_index(drop=True)
    return pred_df
