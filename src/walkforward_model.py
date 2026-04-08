"""Sliding-window walk-forward classifier utilities."""

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

PRIMARY_ENSEMBLE_WEIGHTS = {
    "lgbm_deep": 0.55,
    "logreg": 0.30,
    "rf": 0.15,
}

META_COLUMNS = ["Date", "pair", "next_ret", "future_ret", "rel", "profit_target", "ev_target"]


def sliding_windows(dates, fit_len: int, test_len: int, step: int):
    """Yield (fit_dates, test_dates) for rolling windows."""
    n = len(dates)
    for start in range(0, n - fit_len - test_len + 1, step):
        fit_dates = dates[start:start + fit_len]
        test_dates = dates[start + fit_len:start + fit_len + test_len]
        yield fit_dates, test_dates


def make_xy(subdf: pd.DataFrame, date_list, feature_cols: list[str]):
    """Build sorted subset, design matrix, labels, and group sizes."""
    s = subdf[subdf["Date"].isin(date_list)].sort_values(["Date", "pair"])
    X = s[feature_cols]
    y = s["profit_target"].values
    g = s.groupby("Date").size().values
    return s, X, y, g


def feature_subset_columns(feature_cols: list[str]) -> dict[str, list[str]]:
    """Group raw features into reusable specialist subsets."""
    returns_only_prefixes = ("ret_", "rs1_")
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
        "norm_mom5_",
        "norm_mom10_",
    )

    def matching(prefixes: tuple[str, ...]) -> list[str]:
        return [col for col in feature_cols if col.startswith(prefixes)]

    return {
        "full": feature_cols,
        "returns_only": matching(returns_only_prefixes),
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
        "lgbm_deep_volatility": LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=250,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=46,
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
        "rf_returns_momentum": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=250,
                max_depth=None,
                min_samples_leaf=2,
                random_state=43,
                n_jobs=-1,
            ),
        ),
        "rf_corr_regime": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=250,
                max_depth=None,
                min_samples_leaf=2,
                random_state=44,
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
        "logreg_volatility": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=0.75,
                max_iter=1000,
                random_state=45,
            ),
        ),
    }


def predict_positive_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Return the model score for the positive class."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def primary_ensemble_weights(model_names: list[str]) -> dict[str, float]:
    """Return normalized fixed ensemble weights for the broad diagnostic ensemble."""
    active = {name: PRIMARY_ENSEMBLE_WEIGHTS[name] for name in model_names if name in PRIMARY_ENSEMBLE_WEIGHTS}
    total = sum(active.values())
    if total <= 0:
        raise ValueError("Primary ensemble weights do not match any active model names")
    return {name: weight / total for name, weight in active.items()}


def model_feature_subset_map(model_names: list[str]) -> dict[str, str]:
    """Map each model to the feature subset it should consume."""
    mapping = {name: "full" for name in model_names}
    for name in model_names:
        if "returns_only" in name:
            mapping[name] = "returns_only"
        elif "returns_momentum" in name:
            mapping[name] = "returns_momentum"
        elif "corr_regime" in name:
            mapping[name] = "corr_regime"
        elif "volatility" in name:
            mapping[name] = "volatility"
    return mapping


def add_specialist_ensemble_scores(
    test_sub: pd.DataFrame,
    specialist_ensemble_models: list[str],
) -> pd.DataFrame:
    """Build a Top-1 specialist ensemble from daily normalized specialist scores."""
    if len(specialist_ensemble_models) < 2:
        raise ValueError("SPECIALIST_ENSEMBLE_MEMBERS must contain at least two model names")
    ensemble_models = [f"pred_{name}" for name in specialist_ensemble_models]
    missing = [col for col in ensemble_models if col not in test_sub.columns]
    if missing:
        raise ValueError(f"Missing specialist ensemble prediction columns: {missing}")

    test_sub = test_sub.copy()
    ranked_cols = []
    for model_col in ensemble_models:
        ranked_col = f"{model_col}__rankpct"
        ranked_cols.append(ranked_col)
        test_sub[ranked_col] = test_sub.groupby("Date")[model_col].rank(method="average", pct=True)
    test_sub["pred_specialist_ensemble"] = test_sub[ranked_cols].mean(axis=1)
    test_sub.drop(columns=ranked_cols, inplace=True)
    return test_sub


def run_walkforward_model(
    long_df: pd.DataFrame,
    fit_days: int,
    test_days: int,
    step_days: int,
    specialist_ensemble_models: list[str],
    log_fn=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward classifiers and return predictions and window diagnostics."""

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
    window_splits = list(sliding_windows(all_dates, fit_days, test_days, step_days))
    prediction_cols = [f"pred_{name}" for name in models]
    prediction_cols.extend(["pred_ensemble", "pred_specialist_ensemble"])

    for window_idx, (fit_dates, test_dates) in enumerate(window_splits, start=1):
        train_sub, _, y_train, _ = make_xy(long_df, fit_dates, feature_cols)
        test_sub, _, _, _ = make_xy(long_df, test_dates, feature_cols)
        log(
            f"Training window {window_idx}/{len(window_splits)} "
            f"(fit dates: {len(fit_dates)}, test dates: {len(test_dates)})"
        )

        test_sub = test_sub.copy()
        for model_name, base_model in models.items():
            subset_name = model_subsets[model_name]
            model_feature_cols = subset_cols[subset_name]
            if not model_feature_cols:
                raise ValueError(f"Feature subset '{subset_name}' for model '{model_name}' is empty")
            model = clone(base_model)
            model.fit(train_sub[model_feature_cols], y_train)
            test_sub[f"pred_{model_name}"] = predict_positive_scores(model, test_sub[model_feature_cols])
            window_diagnostics.append(
                {
                    "window_id": window_idx,
                    "model": model_name,
                    "fit_dates": len(fit_dates),
                    "test_dates": len(test_dates),
                    "feature_subset": subset_name,
                }
            )

        primary_weights = primary_ensemble_weights(list(models))
        test_sub["pred_ensemble"] = sum(
            test_sub[f"pred_{model_name}"] * primary_weights[model_name]
            for model_name in primary_weights
        )
        test_sub = add_specialist_ensemble_scores(test_sub, specialist_ensemble_models)
        log(
            "Primary diagnostic ensemble weights: "
            + ", ".join(f"{model_name}={primary_weights[model_name]:.3f}" for model_name in primary_weights)
        )
        log(
            "Specialist Top-1 ensemble: "
            + ", ".join(specialist_ensemble_models)
            + " via daily rank averaging"
        )

        window_diagnostics.append(
            {
                "window_id": window_idx,
                "model": "pred_ensemble",
                "fit_dates": len(fit_dates),
                "test_dates": len(test_dates),
                "feature_subset": "broad_fixed_blend",
            }
        )
        window_diagnostics.append(
            {
                "window_id": window_idx,
                "model": "pred_specialist_ensemble",
                "fit_dates": len(fit_dates),
                "test_dates": len(test_dates),
                "feature_subset": "rank_blend_" + "_".join(specialist_ensemble_models),
            }
        )

        test_sub["window_id"] = window_idx
        all_test_chunks.append(test_sub[["window_id", *META_COLUMNS, *prediction_cols]])

    pred_df = pd.concat(all_test_chunks, ignore_index=True)
    pred_df = pred_df.sort_values(["Date", "pair"]).reset_index(drop=True)
    window_diag_df = pd.DataFrame(window_diagnostics)
    return pred_df, window_diag_df
