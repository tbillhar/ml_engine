"""Sliding-window walk-forward calibrated classifier ensemble utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sliding_windows(dates, fit_len: int, calibration_len: int, test_len: int, step: int):
    """Yield (fit_dates, calibration_dates, test_dates) for rolling windows."""
    n = len(dates)
    total_train_len = fit_len + calibration_len
    for start in range(0, n - total_train_len - test_len + 1, step):
        fit_dates = dates[start:start + fit_len]
        calibration_dates = dates[start + fit_len:start + total_train_len]
        test = dates[start + total_train_len:start + total_train_len + test_len]
        yield fit_dates, calibration_dates, test


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
        "lgbm_shallow": LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            num_leaves=15,
            learning_rate=0.05,
            n_estimators=200,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=41,
            verbosity=-1,
        ),
        "lgbm_base": LGBMClassifier(
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
        "lgbm_deep": LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            num_leaves=127,
            learning_rate=0.03,
            n_estimators=450,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=43,
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


def inverse_brier_weights(
    calibrated_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> dict[str, float]:
    """Return normalized inverse-Brier weights for calibrated model predictions."""
    epsilon = 1e-6
    inv_scores = {}
    for model_name, preds in calibrated_predictions.items():
        brier = brier_score_loss(y_true, preds)
        inv_scores[model_name] = 1.0 / max(brier, epsilon)
    total = sum(inv_scores.values())
    return {model_name: score / total for model_name, score in inv_scores.items()}


def run_walkforward_model(
    long_df: pd.DataFrame,
    fit_days: int,
    calibration_days: int,
    test_days: int,
    step_days: int,
    transaction_loss_pct: float,
    log_fn=None,
) -> pd.DataFrame:
    """Run walk-forward calibrated classifier ensemble and return predictions."""
    def log(message: str) -> None:
        if log_fn:
            log_fn(message)

    feature_cols = [
        c for c in long_df.columns
        if c not in ["Date", "pair", "next_ret", "future_ret", "rel", "profit_target", "ev_target"]
    ]

    all_dates = sorted(long_df["Date"].unique())

    all_test_chunks = []
    window_splits = list(sliding_windows(all_dates, fit_days, calibration_days, test_days, step_days))

    for window_idx, (fit_dates, calibration_dates, test_dates) in enumerate(window_splits, start=1):
        train_sub, X_train, y_train, g_train = make_xy(long_df, fit_dates, feature_cols)
        calibration_sub, X_calibration, y_calibration, g_calibration = make_xy(
            long_df,
            calibration_dates,
            feature_cols,
        )
        test_sub, X_test, y_test, g_test = make_xy(long_df, test_dates, feature_cols)
        _ = (train_sub, g_train, calibration_sub, g_calibration, y_test, g_test, transaction_loss_pct)
        log(
            f"Training ensemble window {window_idx}/{len(window_splits)} "
            f"(fit dates: {len(fit_dates)}, calibration dates: {len(calibration_dates)}, test dates: {len(test_dates)})"
        )

        test_sub = test_sub.copy()
        model_preds: list[np.ndarray] = []
        calibration_preds: dict[str, np.ndarray] = {}
        for model_name, base_model in build_models().items():
            model = clone(base_model)
            model.fit(X_train, y_train)
            calibration_scores = predict_positive_scores(model, X_calibration)
            calibrator = fit_platt_calibrator(calibration_scores, y_calibration)
            calibration_probs = apply_calibrator(calibrator, calibration_scores)
            calibration_preds[model_name] = calibration_probs
            test_scores = predict_positive_scores(model, X_test)
            calibrated_probs = apply_calibrator(calibrator, test_scores)
            test_sub[f"pred_{model_name}"] = calibrated_probs
            model_preds.append(calibrated_probs)

        weights = inverse_brier_weights(calibration_preds, y_calibration)
        log(
            "Window weights: "
            + ", ".join(f"{model_name}={weight:.3f}" for model_name, weight in weights.items())
        )
        ensemble_pred = sum(test_sub[f"pred_{model_name}"] * weights[model_name] for model_name in weights)
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
                    "pred_lgbm_shallow",
                    "pred_lgbm_base",
                    "pred_lgbm_deep",
                    "pred_rf",
                    "pred_logreg",
                    "pred_ensemble",
                ]
            ]
        )

    pred_df = pd.concat(all_test_chunks, ignore_index=True)
    pred_df = pred_df.sort_values(["Date", "pair"]).reset_index(drop=True)
    return pred_df
