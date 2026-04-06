"""Sliding-window walk-forward calibrated classifier ensemble utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
    y = s["profit_target"].values
    g = s.groupby("Date").size().values
    return s, X, y, g


def build_models() -> dict[str, object]:
    """Construct the base classifiers used in the ensemble."""
    return {
        "lgbm": LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            num_leaves=63,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        ),
        "rf": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=250,
                max_depth=None,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "logreg": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
            ),
        ),
    }


def split_train_and_calibration_dates(train_dates, calibration_fraction: float = 0.2):
    """Split ordered training dates into subtrain and calibration slices."""
    calibration_len = max(20, int(len(train_dates) * calibration_fraction))
    calibration_len = min(calibration_len, len(train_dates) - 20)
    if calibration_len <= 0:
        raise ValueError("Not enough training dates to create a calibration slice")
    return train_dates[:-calibration_len], train_dates[-calibration_len:]


def fit_platt_calibrator(raw_scores: np.ndarray, y_true: np.ndarray):
    """Fit a logistic calibration layer over one-dimensional model scores."""
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    calibrator.fit(raw_scores.reshape(-1, 1), y_true)
    return calibrator


def apply_calibrator(calibrator, raw_scores: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator or a constant fallback probability."""
    if isinstance(calibrator, float):
        return np.full(raw_scores.shape[0], calibrator, dtype=float)
    return calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]


def predict_positive_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Return the model score for the positive class."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def run_walkforward_model(
    long_df: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
    transaction_loss_pct: float,
) -> pd.DataFrame:
    """Run walk-forward calibrated classifier ensemble and return predictions."""
    feature_cols = [
        c for c in long_df.columns
        if c not in ["Date", "pair", "next_ret", "future_ret", "rel", "profit_target", "ev_target"]
    ]

    all_dates = sorted(long_df["Date"].unique())

    all_test_chunks = []

    for train_dates, test_dates in sliding_windows(all_dates, train_days, test_days, step_days):
        subtrain_dates, calibration_dates = split_train_and_calibration_dates(train_dates)
        train_sub, X_train, y_train, g_train = make_xy(long_df, subtrain_dates, feature_cols)
        calibration_sub, X_calibration, y_calibration, g_calibration = make_xy(
            long_df,
            calibration_dates,
            feature_cols,
        )
        test_sub, X_test, y_test, g_test = make_xy(long_df, test_dates, feature_cols)
        _ = (train_sub, g_train, calibration_sub, g_calibration, y_test, g_test, transaction_loss_pct)

        test_sub = test_sub.copy()
        model_preds: list[np.ndarray] = []
        for model_name, base_model in build_models().items():
            model = clone(base_model)
            model.fit(X_train, y_train)
            calibration_scores = predict_positive_scores(model, X_calibration)
            calibrator = fit_platt_calibrator(calibration_scores, y_calibration)
            test_scores = predict_positive_scores(model, X_test)
            calibrated_probs = apply_calibrator(calibrator, test_scores)
            test_sub[f"pred_{model_name}"] = calibrated_probs
            model_preds.append(calibrated_probs)

        ensemble_pred = np.mean(np.column_stack(model_preds), axis=1)
        test_sub["pred_ensemble"] = ensemble_pred
        all_test_chunks.append(
            test_sub[
                [
                    "Date",
                    "pair",
                    "next_ret",
                    "future_ret",
                    "profit_target",
                    "ev_target",
                    "rel",
                    "pred_lgbm",
                    "pred_rf",
                    "pred_logreg",
                    "pred_ensemble",
                ]
            ]
        )

    pred_df = pd.concat(all_test_chunks, ignore_index=True)
    pred_df = pred_df.sort_values(["Date", "pair"]).reset_index(drop=True)
    return pred_df
