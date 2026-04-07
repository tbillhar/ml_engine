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

PRIMARY_ENSEMBLE_WEIGHTS = {
    "lgbm_deep": 0.55,
    "logreg": 0.30,
    "rf": 0.15,
}

META_COLUMNS = ["Date", "pair", "next_ret", "future_ret", "rel", "profit_target", "ev_target"]


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


def feature_subset_columns(feature_cols: list[str]) -> dict[str, list[str]]:
    """Group raw features into reusable specialist subsets."""
    returns_momentum_prefixes = (
        "ret_",
        "mom5_",
        "mom10_",
        "rs1_",
        "rs5_",
        "rs10_",
        "rs20_",
        "rank_ret_",
        "rank_mom5_",
        "rank_mom10_",
        "accel_",
        "slope20_",
        "zret20_",
        "zmom5_20_",
        "zmom10_20_",
        "norm_mom5_",
        "norm_mom10_",
    )
    corr_regime_prefixes = (
        "corr20_",
        "corr40_",
        "corr60_",
        "corr120_",
        "PC",
        "PC1_evr",
        "PC2_evr",
        "PC3_evr",
        "avg_corr",
        "market_vol",
        "dispersion",
        "day_of_week",
        "is_month_end",
        "is_quarter_end",
        "prev_top1_",
        "rank_persist5_",
        "signal_stability5_",
    )
    volatility_prefixes = (
        "vol30_",
        "rank_vol30_",
        "dvol30_",
    )

    def matching(prefixes: tuple[str, ...]) -> list[str]:
        return [col for col in feature_cols if col.startswith(prefixes)]

    return {
        "full": feature_cols,
        "returns_momentum": matching(returns_momentum_prefixes),
        "corr_regime": matching(corr_regime_prefixes),
        "volatility": matching(volatility_prefixes),
    }


def build_models() -> dict[str, object]:
    """Construct the base classifiers used in the ensemble."""
    return {
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
        "lgbm_deep_returns_momentum": LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            num_leaves=63,
            learning_rate=0.04,
            n_estimators=300,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=44,
            verbosity=-1,
        ),
        "lgbm_deep_corr_regime": LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            num_leaves=63,
            learning_rate=0.04,
            n_estimators=300,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=45,
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
        "logreg_returns_momentum": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=0.75,
                max_iter=1000,
                random_state=43,
            ),
        ),
        "logreg_corr_regime": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=0.75,
                max_iter=1000,
                random_state=44,
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


def primary_ensemble_weights(model_names: list[str]) -> dict[str, float]:
    """Return normalized fixed ensemble weights for the primary live ensemble."""
    active = {name: PRIMARY_ENSEMBLE_WEIGHTS[name] for name in model_names if name in PRIMARY_ENSEMBLE_WEIGHTS}
    total = sum(active.values())
    if total <= 0:
        raise ValueError("Primary ensemble weights do not match any active model names")
    return {name: weight / total for name, weight in active.items()}


def model_feature_subset_map(model_names: list[str]) -> dict[str, str]:
    """Map each model to the feature subset it should consume."""
    mapping = {name: "full" for name in model_names}
    for name in model_names:
        if "returns_momentum" in name:
            mapping[name] = "returns_momentum"
        elif "corr_regime" in name:
            mapping[name] = "corr_regime"
        elif "volatility" in name:
            mapping[name] = "volatility"
    return mapping


def run_walkforward_model(
    long_df: pd.DataFrame,
    fit_days: int,
    calibration_days: int,
    test_days: int,
    step_days: int,
    transaction_loss_pct: float,
    log_fn=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward calibrated classifier ensemble and return predictions and window diagnostics."""
    def log(message: str) -> None:
        if log_fn:
            log_fn(message)

    feature_cols = [c for c in long_df.columns if c not in META_COLUMNS]
    subset_cols = feature_subset_columns(feature_cols)
    models = build_models()
    model_subsets = model_feature_subset_map(list(models))

    all_dates = sorted(long_df["Date"].unique())

    all_test_chunks = []
    window_diagnostics = []
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
        calibration_preds: dict[str, np.ndarray] = {}
        calibration_briers: dict[str, float] = {}
        for model_name, base_model in models.items():
            subset_name = model_subsets[model_name]
            model_feature_cols = subset_cols[subset_name]
            if not model_feature_cols:
                raise ValueError(f"Feature subset '{subset_name}' for model '{model_name}' is empty")
            X_train = train_sub[model_feature_cols]
            X_calibration = calibration_sub[model_feature_cols]
            X_test = test_sub[model_feature_cols]
            model = clone(base_model)
            model.fit(X_train, y_train)
            calibration_scores = predict_positive_scores(model, X_calibration)
            calibrator = fit_platt_calibrator(calibration_scores, y_calibration)
            calibration_probs = apply_calibrator(calibrator, calibration_scores)
            calibration_preds[model_name] = calibration_probs
            calibration_briers[model_name] = brier_score_loss(y_calibration, calibration_probs)
            test_scores = predict_positive_scores(model, X_test)
            calibrated_probs = apply_calibrator(calibrator, test_scores)
            test_sub[f"pred_{model_name}"] = calibrated_probs

        inverse_weights = inverse_brier_weights(calibration_preds, y_calibration)
        primary_weights = primary_ensemble_weights(list(calibration_preds))
        ensemble_calibration_pred = sum(
            calibration_preds[model_name] * primary_weights[model_name]
            for model_name in primary_weights
        )
        ensemble_calibration_brier = brier_score_loss(y_calibration, ensemble_calibration_pred)
        brier_ensemble_calibration_pred = sum(
            calibration_preds[model_name] * inverse_weights[model_name]
            for model_name in inverse_weights
        )
        brier_ensemble_calibration_brier = brier_score_loss(y_calibration, brier_ensemble_calibration_pred)
        log(
            "Primary ensemble weights: "
            + ", ".join(
                f"{model_name}={primary_weights[model_name]:.6f}"
                for model_name in primary_weights
            )
            + f"; primary_ensemble_brier={ensemble_calibration_brier:.6f}"
        )
        log(
            "Inverse-Brier diagnostic weights: "
            + ", ".join(
                f"{model_name}={inverse_weights[model_name]:.6f} (brier={calibration_briers[model_name]:.6f})"
                for model_name in inverse_weights
            )
            + f"; inverse_brier_ensemble_brier={brier_ensemble_calibration_brier:.6f}"
        )
        ensemble_pred = sum(
            test_sub[f"pred_{model_name}"] * primary_weights[model_name]
            for model_name in primary_weights
        )
        brier_ensemble_pred = sum(
            test_sub[f"pred_{model_name}"] * inverse_weights[model_name]
            for model_name in inverse_weights
        )
        test_sub["pred_ensemble"] = ensemble_pred
        test_sub["pred_ensemble_brier"] = brier_ensemble_pred
        test_sub["window_id"] = window_idx

        for model_name in calibration_preds:
            window_diagnostics.append(
                {
                    "window_id": window_idx,
                    "model": model_name,
                    "fit_dates": len(fit_dates),
                    "calibration_dates": len(calibration_dates),
                    "test_dates": len(test_dates),
                    "feature_subset": model_subsets[model_name],
                    "primary_weight": float(primary_weights.get(model_name, 0.0)),
                    "inverse_brier_weight": float(inverse_weights[model_name]),
                    "calibration_brier": float(calibration_briers[model_name]),
                }
            )
        window_diagnostics.append(
            {
                "window_id": window_idx,
                "model": "pred_ensemble",
                "fit_dates": len(fit_dates),
                "calibration_dates": len(calibration_dates),
                "test_dates": len(test_dates),
                "feature_subset": "primary_blend",
                "primary_weight": 1.0,
                "inverse_brier_weight": np.nan,
                "calibration_brier": float(ensemble_calibration_brier),
            }
        )
        window_diagnostics.append(
            {
                "window_id": window_idx,
                "model": "pred_ensemble_brier",
                "fit_dates": len(fit_dates),
                "calibration_dates": len(calibration_dates),
                "test_dates": len(test_dates),
                "feature_subset": "inverse_brier_blend",
                "primary_weight": np.nan,
                "inverse_brier_weight": 1.0,
                "calibration_brier": float(brier_ensemble_calibration_brier),
            }
        )
        all_test_chunks.append(
            test_sub[
                [
                    "window_id",
                    "Date",
                    "pair",
                    "next_ret",
                    "future_ret",
                    "profit_target",
                    "ev_target",
                    "rel",
                    "pred_lgbm_deep",
                    "pred_lgbm_deep_returns_momentum",
                    "pred_lgbm_deep_corr_regime",
                    "pred_rf",
                    "pred_logreg",
                    "pred_logreg_returns_momentum",
                    "pred_logreg_corr_regime",
                    "pred_ensemble",
                    "pred_ensemble_brier",
                ]
            ]
        )

    pred_df = pd.concat(all_test_chunks, ignore_index=True)
    pred_df = pred_df.sort_values(["Date", "pair"]).reset_index(drop=True)
    window_diag_df = pd.DataFrame(window_diagnostics)
    return pred_df, window_diag_df
