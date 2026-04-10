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

LIVE_MODEL_COLUMNS = {
    "ensemble": "pred_ensemble",
    "specialist_ensemble": "pred_specialist_ensemble",
    "lgbm_deep": "pred_lgbm_deep",
    "lgbm_deep_returns_momentum": "pred_lgbm_deep_returns_momentum",
    "lgbm_deep_corr_regime": "pred_lgbm_deep_corr_regime",
    "lgbm_deep_volatility": "pred_lgbm_deep_volatility",
    "rf": "pred_rf",
    "rf_returns_momentum": "pred_rf_returns_momentum",
    "rf_corr_regime": "pred_rf_corr_regime",
    "logreg": "pred_logreg",
    "logreg_returns_momentum": "pred_logreg_returns_momentum",
    "logreg_corr_regime": "pred_logreg_corr_regime",
    "logreg_volatility": "pred_logreg_volatility",
}

META_COLUMNS = ["Date", "pair", "next_ret", "future_ret", "rel", "profit_target", "ev_target"]
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


def resolve_live_prediction_column(live_model: str) -> str:
    """Map a user-facing live model name to its prediction column."""
    if live_model not in LIVE_MODEL_COLUMNS:
        raise ValueError(f"Unsupported LIVE_MODEL '{live_model}'. Expected one of: {sorted(LIVE_MODEL_COLUMNS)}")
    return LIVE_MODEL_COLUMNS[live_model]


def specialist_weights_from_history(
    member_top1_returns: dict[str, list[float]],
    ensemble_models: list[str],
    specialist_weight_lookback_days: int,
    specialist_weighting_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return current specialist ensemble weights and raw trailing scores."""
    equal_weight = 1.0 / len(ensemble_models)

    def softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exps = np.exp(shifted)
        return exps / exps.sum()

    def softened_weights(raw_scores: np.ndarray) -> np.ndarray:
        if np.all(np.isnan(raw_scores)):
            return np.full(len(ensemble_models), equal_weight)
        filled = np.nan_to_num(raw_scores, nan=0.0)
        if specialist_weighting_mode == "sticky_winner":
            return np.full(len(ensemble_models), equal_weight)
        if specialist_weighting_mode == "equal":
            return np.full(len(ensemble_models), equal_weight)
        if specialist_weighting_mode == "winner_take_all":
            winner_idx = int(np.argmax(filled))
            weights = np.zeros(len(ensemble_models), dtype=float)
            weights[winner_idx] = 1.0
            return weights
        if specialist_weighting_mode == "winner_take_most":
            winner_idx = int(np.argmax(filled))
            winner_weight = 0.70
            remainder = 1.0 - winner_weight
            weights = np.full(
                len(ensemble_models),
                remainder / max(len(ensemble_models) - 1, 1),
                dtype=float,
            )
            weights[winner_idx] = winner_weight
            if len(ensemble_models) == 1:
                weights[winner_idx] = 1.0
            return weights / weights.sum()
        if specialist_weighting_mode != "soft_dynamic":
            raise ValueError(
                "Unsupported SPECIALIST_WEIGHTING_MODE "
                f"'{specialist_weighting_mode}'. Expected one of: "
                "equal, soft_dynamic, winner_take_all, winner_take_most, sticky_winner"
            )
        centered = filled - filled.mean()
        scale = max(float(np.std(centered)), 1e-4)
        dynamic = softmax(centered / scale)
        blended = 0.5 * dynamic + 0.5 * np.full(len(ensemble_models), equal_weight)
        max_weight = min(0.70, 2.0 * equal_weight)
        clipped = np.clip(blended, 1e-9, max_weight)
        return clipped / clipped.sum()

    raw_scores = []
    for model_col in ensemble_models:
        history_slice = member_top1_returns[model_col][-specialist_weight_lookback_days:]
        raw_scores.append(float(np.mean(history_slice)) if history_slice else np.nan)
    raw_scores_arr = np.array(raw_scores, dtype=float)
    return softened_weights(raw_scores_arr), raw_scores_arr


def sticky_specialist_router_step(
    raw_scores: np.ndarray,
    ensemble_models: list[str],
    previous_active_idx: int | None,
    previous_hold_days: int,
    specialist_min_model_hold_days: int,
    specialist_switch_margin_min_avg_ev: float,
    specialist_switch_require_positive_ev: bool,
) -> tuple[np.ndarray, int | None, int, str]:
    """Route to one active specialist with persistence and switching guards."""
    equal_weight = 1.0 / len(ensemble_models)
    if np.all(np.isnan(raw_scores)):
        return np.full(len(ensemble_models), equal_weight), previous_active_idx, previous_hold_days, "no_history_equal"

    filled = np.nan_to_num(raw_scores, nan=-np.inf)
    leader_idx = int(np.argmax(filled))
    leader_score = filled[leader_idx]

    if previous_active_idx is None or previous_active_idx >= len(ensemble_models):
        if specialist_switch_require_positive_ev and leader_score <= 0:
            return np.full(len(ensemble_models), equal_weight), None, 0, "no_positive_leader_equal"
        weights = np.zeros(len(ensemble_models), dtype=float)
        weights[leader_idx] = 1.0
        return weights, leader_idx, 1, "init_leader"

    current_idx = previous_active_idx
    current_score = filled[current_idx]
    new_hold_days = previous_hold_days + 1

    if leader_idx == current_idx:
        weights = np.zeros(len(ensemble_models), dtype=float)
        weights[current_idx] = 1.0
        return weights, current_idx, new_hold_days, "stay_current_best"

    if previous_hold_days < specialist_min_model_hold_days:
        weights = np.zeros(len(ensemble_models), dtype=float)
        weights[current_idx] = 1.0
        return weights, current_idx, new_hold_days, "hold_lock"

    if specialist_switch_require_positive_ev and leader_score <= 0:
        weights = np.zeros(len(ensemble_models), dtype=float)
        weights[current_idx] = 1.0
        return weights, current_idx, new_hold_days, "no_positive_challenger"

    if not np.isfinite(current_score):
        current_score = -np.inf
    if leader_score <= current_score + specialist_switch_margin_min_avg_ev:
        weights = np.zeros(len(ensemble_models), dtype=float)
        weights[current_idx] = 1.0
        return weights, current_idx, new_hold_days, "margin_not_met"

    weights = np.zeros(len(ensemble_models), dtype=float)
    weights[leader_idx] = 1.0
    return weights, leader_idx, 1, "switch_margin"


def add_specialist_ensemble_scores(
    test_sub: pd.DataFrame,
    specialist_ensemble_models: list[str],
    specialist_weight_lookback_days: int,
    specialist_weighting_mode: str,
    specialist_min_model_hold_days: int,
    specialist_switch_margin_min_avg_ev: float,
    specialist_switch_require_positive_ev: bool,
) -> pd.DataFrame:
    """Build a rank-based Top-1 specialist ensemble under a chosen weighting mode."""
    if len(specialist_ensemble_models) < 2:
        raise ValueError("SPECIALIST_ENSEMBLE_MEMBERS must contain at least two model names")
    ensemble_models = [f"pred_{name}" for name in specialist_ensemble_models]
    missing = [col for col in ensemble_models if col not in test_sub.columns]
    if missing:
        raise ValueError(f"Missing specialist ensemble prediction columns: {missing}")

    test_sub = test_sub.copy()
    ranked_cols: dict[str, str] = {}
    for model_col in ensemble_models:
        ranked_col = f"{model_col}__rankpct"
        ranked_cols[model_col] = ranked_col
        test_sub[ranked_col] = test_sub.groupby("Date")[model_col].rank(method="average", pct=True)

    date_order = sorted(test_sub["Date"].unique())
    member_top1_returns: dict[str, list[float]] = {model_col: [] for model_col in ensemble_models}
    router_active_idx: int | None = None
    router_hold_days = 0
    test_sub["specialist_ensemble_active_model"] = ""
    test_sub["specialist_ensemble_switch_reason"] = ""
    test_sub["specialist_ensemble_weight_info"] = ""

    for current_idx, current_date in enumerate(date_order):
        normalized_weights, raw_scores = specialist_weights_from_history(
            member_top1_returns=member_top1_returns,
            ensemble_models=ensemble_models,
            specialist_weight_lookback_days=specialist_weight_lookback_days,
            specialist_weighting_mode=specialist_weighting_mode,
        )
        switch_reason = "stateless_mode"
        if specialist_weighting_mode == "sticky_winner":
            (
                normalized_weights,
                router_active_idx,
                router_hold_days,
                switch_reason,
            ) = sticky_specialist_router_step(
                raw_scores=raw_scores,
                ensemble_models=ensemble_models,
                previous_active_idx=router_active_idx,
                previous_hold_days=router_hold_days,
                specialist_min_model_hold_days=specialist_min_model_hold_days,
                specialist_switch_margin_min_avg_ev=specialist_switch_margin_min_avg_ev,
                specialist_switch_require_positive_ev=specialist_switch_require_positive_ev,
            )

        date_mask = test_sub["Date"] == current_date
        weighted_score = np.zeros(int(date_mask.sum()), dtype=float)
        for idx, model_col in enumerate(ensemble_models):
            weighted_score += normalized_weights[idx] * test_sub.loc[date_mask, ranked_cols[model_col]].to_numpy()
        test_sub.loc[date_mask, "pred_specialist_ensemble"] = weighted_score

        date_slice = test_sub.loc[date_mask]
        for model_col in ensemble_models:
            chosen_row = date_slice.sort_values(model_col, ascending=False).iloc[0]
            member_top1_returns[model_col].append(float(chosen_row["ev_target"]))
        active_model = ensemble_models[router_active_idx] if router_active_idx is not None else ""
        test_sub.loc[date_mask, "specialist_ensemble_active_model"] = active_model
        test_sub.loc[date_mask, "specialist_ensemble_switch_reason"] = switch_reason
        weight_parts = []
        for idx, model_col in enumerate(ensemble_models):
            weight_parts.append(f"{model_col}={normalized_weights[idx]:.3f}")
        test_sub.loc[date_mask, "specialist_ensemble_weight_info"] = "|".join(weight_parts)

    test_sub.drop(columns=list(ranked_cols.values()), inplace=True)
    return test_sub


def run_walkforward_model(
    long_df: pd.DataFrame,
    fit_days: int,
    step_days: int,
    live_model: str,
    specialist_weighting_mode: str,
    specialist_ensemble_models: list[str],
    specialist_weight_lookback_days: int,
    specialist_min_model_hold_days: int,
    specialist_switch_margin_min_avg_ev: float,
    specialist_switch_require_positive_ev: bool,
    retrain_deterioration_lookback_days: int,
    retrain_deterioration_min_win_rate: float,
    retrain_deterioration_max_avg_ev: float,
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
    live_pred_col = resolve_live_prediction_column(live_model)
    specialist_prediction_cols = [f"pred_{name}" for name in specialist_ensemble_models]

    all_dates = sorted(long_df["Date"].unique())
    all_test_chunks = []
    window_diagnostics = []
    prediction_cols = [f"pred_{name}" for name in models]
    prediction_cols.append("pred_ensemble")
    primary_weights = primary_ensemble_weights(list(models))
    global_specialist_top1_returns: dict[str, list[float]] = {
        model_col: [] for model_col in specialist_prediction_cols
    }
    router_active_idx: int | None = None
    router_hold_days = 0
    fit_start = 0
    window_idx = 0
    while fit_start + fit_days < len(all_dates):
        fit_dates = all_dates[fit_start:fit_start + fit_days]
        train_sub, _, y_train, _ = make_xy(long_df, fit_dates, feature_cols)
        window_idx += 1
        log(
            f"Training window {window_idx} "
            f"(fit dates: {len(fit_dates)}, cadence step: {step_days}, "
            f"live model: {live_model})"
        )
        fitted_models: dict[str, object] = {}
        for model_name, base_model in models.items():
            subset_name = model_subsets[model_name]
            model_feature_cols = subset_cols[subset_name]
            if not model_feature_cols:
                raise ValueError(f"Feature subset '{subset_name}' for model '{model_name}' is empty")
            model = clone(base_model)
            model.fit(train_sub[model_feature_cols], y_train)
            fitted_models[model_name] = model

        current_test_chunks = []
        trigger_returns: list[float] = []
        days_into_test = 0
        trigger_avg_ev = np.nan
        trigger_win_rate = np.nan
        retrain_reason = "end_of_data"
        next_date_idx = fit_start + fit_days

        while next_date_idx < len(all_dates):
            current_date = all_dates[next_date_idx]
            day_sub = long_df[long_df["Date"] == current_date].sort_values(["Date", "pair"]).copy()
            for model_name, model in fitted_models.items():
                subset_name = model_subsets[model_name]
                model_feature_cols = subset_cols[subset_name]
                day_sub[f"pred_{model_name}"] = predict_positive_scores(model, day_sub[model_feature_cols])
            day_sub["pred_ensemble"] = sum(
                day_sub[f"pred_{model_name}"] * primary_weights[model_name]
                for model_name in primary_weights
            )
            specialist_weights, raw_scores = specialist_weights_from_history(
                member_top1_returns=global_specialist_top1_returns,
                ensemble_models=specialist_prediction_cols,
                specialist_weight_lookback_days=specialist_weight_lookback_days,
                specialist_weighting_mode=specialist_weighting_mode,
            )
            switch_reason = "stateless_mode"
            if specialist_weighting_mode == "sticky_winner":
                (
                    specialist_weights,
                    router_active_idx,
                    router_hold_days,
                    switch_reason,
                ) = sticky_specialist_router_step(
                    raw_scores=raw_scores,
                    ensemble_models=specialist_prediction_cols,
                    previous_active_idx=router_active_idx,
                    previous_hold_days=router_hold_days,
                    specialist_min_model_hold_days=specialist_min_model_hold_days,
                    specialist_switch_margin_min_avg_ev=specialist_switch_margin_min_avg_ev,
                    specialist_switch_require_positive_ev=specialist_switch_require_positive_ev,
                )
            specialist_rank_sum = np.zeros(len(day_sub), dtype=float)
            for idx, model_col in enumerate(specialist_prediction_cols):
                rank_pct = day_sub[model_col].rank(method="average", pct=True).to_numpy()
                specialist_rank_sum += specialist_weights[idx] * rank_pct
            day_sub["pred_specialist_ensemble"] = specialist_rank_sum
            day_sub["specialist_ensemble_active_model"] = (
                specialist_prediction_cols[router_active_idx] if router_active_idx is not None else ""
            )
            day_sub["specialist_ensemble_switch_reason"] = switch_reason
            day_sub["specialist_ensemble_weight_info"] = "|".join(
                f"{model_col}={specialist_weights[idx]:.3f}"
                for idx, model_col in enumerate(specialist_prediction_cols)
            )
            day_sub["window_id"] = window_idx
            current_test_chunks.append(
                day_sub[
                    [
                        "window_id",
                        *META_COLUMNS,
                        *prediction_cols,
                        "specialist_ensemble_active_model",
                        "specialist_ensemble_switch_reason",
                        "specialist_ensemble_weight_info",
                    ]
                ]
            )

            trigger_row = day_sub.sort_values(live_pred_col, ascending=False).iloc[0]
            trigger_returns.append(float(trigger_row["ev_target"]))
            for model_col in specialist_prediction_cols:
                chosen_row = day_sub.sort_values(model_col, ascending=False).iloc[0]
                global_specialist_top1_returns[model_col].append(float(chosen_row["ev_target"]))
            days_into_test += 1
            next_date_idx += 1

            if days_into_test >= step_days:
                retrain_reason = "cadence_step_days"
                break
            if days_into_test >= retrain_deterioration_lookback_days:
                trailing = np.array(trigger_returns[-retrain_deterioration_lookback_days:], dtype=float)
                trigger_avg_ev = float(np.mean(trailing))
                trigger_win_rate = float(np.mean(trailing > 0))
                if (
                    trigger_avg_ev <= retrain_deterioration_max_avg_ev
                    and trigger_win_rate <= retrain_deterioration_min_win_rate
                ):
                    retrain_reason = "deterioration_trigger"
                    break

        if not current_test_chunks:
            break

        test_sub = pd.concat(current_test_chunks, ignore_index=True)
        log(
            "Primary diagnostic ensemble weights: "
            + ", ".join(f"{model_name}={primary_weights[model_name]:.3f}" for model_name in primary_weights)
        )
        log(
            "Specialist Top-1 ensemble: "
            + ", ".join(specialist_ensemble_models)
            + (
                f" via daily rank averaging with weighting_mode={specialist_weighting_mode} "
                f"and trailing lookback={specialist_weight_lookback_days}; "
                f"min_hold_days={specialist_min_model_hold_days}; "
                f"switch_margin_min_avg_ev={specialist_switch_margin_min_avg_ev}; "
                f"require_positive_ev={specialist_switch_require_positive_ev}"
            )
        )
        log(
            f"Completed window {window_idx} with {days_into_test} test dates; "
            f"retrain_reason={retrain_reason}; "
            f"trigger_avg_ev={trigger_avg_ev if not np.isnan(trigger_avg_ev) else float('nan'):.6f}; "
            f"trigger_win_rate={trigger_win_rate if not np.isnan(trigger_win_rate) else float('nan'):.3f}"
        )

        for model_name in models:
            window_diagnostics.append(
                {
                    "window_id": window_idx,
                    "model": model_name,
                    "fit_dates": len(fit_dates),
                    "test_dates": days_into_test,
                    "feature_subset": model_subsets[model_name],
                    "retrain_reason": retrain_reason,
                    "trigger_avg_ev": trigger_avg_ev,
                    "trigger_win_rate": trigger_win_rate,
                }
            )
        window_diagnostics.append(
            {
                "window_id": window_idx,
                "model": "pred_ensemble",
                "fit_dates": len(fit_dates),
                "test_dates": days_into_test,
                "feature_subset": "broad_fixed_blend",
                "retrain_reason": retrain_reason,
                "trigger_avg_ev": trigger_avg_ev,
                "trigger_win_rate": trigger_win_rate,
            }
        )
        window_diagnostics.append(
            {
                "window_id": window_idx,
                "model": "pred_specialist_ensemble",
                "fit_dates": len(fit_dates),
                "test_dates": days_into_test,
                "feature_subset": "rank_blend_" + "_".join(specialist_ensemble_models),
                "retrain_reason": retrain_reason,
                "trigger_avg_ev": trigger_avg_ev,
                "trigger_win_rate": trigger_win_rate,
            }
        )
        all_test_chunks.append(test_sub)
        fit_start += days_into_test

    pred_df = pd.concat(all_test_chunks, ignore_index=True)
    pred_df = pred_df.sort_values(["Date", "pair"]).reset_index(drop=True)
    pred_df = add_specialist_ensemble_scores(
        pred_df,
        specialist_ensemble_models,
        specialist_weight_lookback_days,
        specialist_weighting_mode,
        specialist_min_model_hold_days,
        specialist_switch_margin_min_avg_ev,
        specialist_switch_require_positive_ev,
    )
    window_diag_df = pd.DataFrame(window_diagnostics)
    return pred_df, window_diag_df
