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
ROUTER_CANDIDATE_COLUMNS = [
    "pred_ensemble",
    "pred_lgbm_deep",
    "pred_lgbm_deep_returns_momentum",
    "pred_lgbm_deep_corr_regime",
    "pred_lgbm_deep_volatility",
    "pred_rf",
    "pred_rf_returns_momentum",
    "pred_rf_corr_regime",
    "pred_logreg",
    "pred_logreg_returns_momentum",
    "pred_logreg_corr_regime",
    "pred_logreg_volatility",
]

META_COLUMNS = ["Date", "pair", "next_ret", "future_ret", "rel", "profit_target", "ev_target"]


def normalize_fit_windows(fit_days: int, model_fit_windows: list[int] | None) -> list[int]:
    """Return sorted unique fit windows, always including the primary FIT_DAYS value."""
    windows = [int(window) for window in (model_fit_windows or []) if int(window) > 0]
    windows.append(int(fit_days))
    return sorted(set(windows))


def prediction_column_name(model_name: str, fit_window: int, primary_fit_days: int) -> str:
    """Return the prediction column name for a model/window variant."""
    base_col = f"pred_{model_name}"
    return base_col if fit_window == primary_fit_days else f"{base_col}_fit{fit_window}"


def model_name_from_prediction_column(pred_col: str, primary_fit_days: int) -> tuple[str, int]:
    """Return base model name and fit window from a prediction column."""
    if not pred_col.startswith("pred_"):
        raise ValueError(f"Prediction column must start with 'pred_': {pred_col}")
    name = pred_col.removeprefix("pred_")
    if "_fit" in name:
        base_name, fit_window_text = name.rsplit("_fit", 1)
        return base_name, int(fit_window_text)
    return name, primary_fit_days


def router_history_lookback_observations(lookback_days: int, rebalance_days: int) -> int:
    """Convert a day-based lookback into rebalance-observation count."""
    return max(1, int(np.ceil(float(lookback_days) / float(max(rebalance_days, 1)))))


def realized_rebalance_outcome(chosen_row: pd.Series, transaction_loss_pct: float) -> float:
    """Net realized holding-period return for a rebalance-date top pick."""
    return float(chosen_row["future_ret"]) - (transaction_loss_pct / 100.0)


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
        "ret3_",
        "ret5_",
        "ret7_",
        "ret10_",
        "ret15_",
        "ret20_",
        "mom5_",
        "mom10_",
        "rs1_",
        "rs3_",
        "rs5_",
        "rs7_",
        "rs10_",
        "rs15_",
        "rs20_",
        "rank_ret_",
        "rank_ret3_",
        "rank_ret5_",
        "rank_ret7_",
        "rank_ret10_",
        "rank_ret15_",
        "rank_ret20_",
        "rank_mom5_",
        "rank_mom10_",
        "accel_",
        "slope20_",
        "zret20_",
        "zmom5_20_",
        "zmom10_20_",
        "norm_mom5_",
        "norm_mom10_",
        "norm_ret3_",
        "norm_ret5_",
        "norm_ret7_",
        "norm_ret10_",
        "norm_ret15_",
        "norm_ret20_",
        "dist_high",
        "dist_low",
        "range_pos",
        "usd_resid_",
        "rank_usd_resid_",
        "ret_x_market_vol20_",
        "ret20_x_avg_corr20_",
        "ret20_x_usd_mom20_",
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
        "usd_factor_",
        "usd_resid_",
        "rank_usd_resid_",
        "ret_x_market_vol20_",
        "ret20_x_avg_corr20_",
        "ret20_x_usd_mom20_",
    )
    volatility_prefixes = (
        "vol30_",
        "rank_vol30_",
        "dvol30_",
        "norm_mom5_",
        "norm_mom10_",
        "norm_ret3_",
        "norm_ret5_",
        "norm_ret7_",
        "norm_ret10_",
        "norm_ret15_",
        "norm_ret20_",
        "dist_high",
        "dist_low",
        "range_pos",
        "market_vol",
        "dispersion",
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


def resolve_router_candidate_columns(
    model_names: list[str],
    available_prediction_cols: list[str] | None = None,
) -> list[str]:
    """Map router candidate model names to concrete prediction columns."""
    if any(model_name.strip().lower() == "all" for model_name in model_names):
        if available_prediction_cols is None:
            return ROUTER_CANDIDATE_COLUMNS.copy()
        return [
            col
            for col in available_prediction_cols
            if col.startswith("pred_") and col != "pred_specialist_ensemble"
        ]

    columns: list[str] = []
    for model_name in model_names:
        if model_name == "specialist_ensemble":
            continue
        if model_name.startswith("pred_"):
            pred_col = model_name
        elif model_name not in LIVE_MODEL_COLUMNS:
            raise ValueError(f"Unsupported MODEL_ROUTER_CANDIDATE '{model_name}'. Expected one of: {sorted(LIVE_MODEL_COLUMNS)}")
        else:
            pred_col = LIVE_MODEL_COLUMNS[model_name]
        if pred_col not in columns:
            columns.append(pred_col)
    if not columns:
        raise ValueError("MODEL_ROUTER_CANDIDATES must resolve to at least one prediction column")
    return columns


def specialist_weights_from_history(
    member_top1_returns: dict[str, list[float]],
    ensemble_models: list[str],
    specialist_weight_lookback_observations: int,
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
        history_slice = member_top1_returns[model_col][-specialist_weight_lookback_observations:]
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
) -> tuple[np.ndarray, int | None, int, str, bool]:
    """Route to one active specialist with persistence and switching guards."""
    equal_weight = 1.0 / len(ensemble_models)
    if np.all(np.isnan(raw_scores)):
        return np.full(len(ensemble_models), equal_weight), previous_active_idx, previous_hold_days, "no_history_equal", False

    filled = np.nan_to_num(raw_scores, nan=-np.inf)
    leader_idx = int(np.argmax(filled))
    leader_score = filled[leader_idx]

    if previous_active_idx is None or previous_active_idx >= len(ensemble_models):
        if specialist_switch_require_positive_ev and leader_score <= 0:
            return np.zeros(len(ensemble_models), dtype=float), None, 0, "cash_no_positive_model", True
        weights = np.zeros(len(ensemble_models), dtype=float)
        weights[leader_idx] = 1.0
        return weights, leader_idx, 1, "init_leader", False

    current_idx = previous_active_idx
    current_score = filled[current_idx]
    new_hold_days = previous_hold_days + 1

    if leader_idx == current_idx:
        if specialist_switch_require_positive_ev and leader_score <= 0:
            return np.zeros(len(ensemble_models), dtype=float), None, 0, "cash_no_positive_model", True
        weights = np.zeros(len(ensemble_models), dtype=float)
        weights[current_idx] = 1.0
        return weights, current_idx, new_hold_days, "stay_current_best", False

    if previous_hold_days < specialist_min_model_hold_days:
        if specialist_switch_require_positive_ev and np.nanmax(filled) <= 0:
            return np.zeros(len(ensemble_models), dtype=float), None, 0, "cash_no_positive_model", True
        weights = np.zeros(len(ensemble_models), dtype=float)
        weights[current_idx] = 1.0
        return weights, current_idx, new_hold_days, "hold_lock", False

    if specialist_switch_require_positive_ev and leader_score <= 0:
        return np.zeros(len(ensemble_models), dtype=float), None, 0, "cash_no_positive_model", True

    if not np.isfinite(current_score):
        current_score = -np.inf
    if leader_score <= current_score + specialist_switch_margin_min_avg_ev:
        weights = np.zeros(len(ensemble_models), dtype=float)
        weights[current_idx] = 1.0
        return weights, current_idx, new_hold_days, "margin_not_met", False

    weights = np.zeros(len(ensemble_models), dtype=float)
    weights[leader_idx] = 1.0
    return weights, leader_idx, 1, "switch_margin", False


def add_specialist_ensemble_scores(
    test_sub: pd.DataFrame,
    specialist_ensemble_models: list[str],
    model_router_candidates: list[str],
    specialist_weight_lookback_days: int,
    rebalance_days: int,
    transaction_loss_pct: float,
    specialist_weighting_mode: str,
    specialist_min_model_hold_days: int,
    specialist_switch_margin_min_avg_ev: float,
    specialist_switch_require_positive_ev: bool,
) -> pd.DataFrame:
    """Build a rank-based Top-1 specialist ensemble under a chosen weighting mode."""
    if len(specialist_ensemble_models) < 2:
        raise ValueError("SPECIALIST_ENSEMBLE_MEMBERS must contain at least two model names")
    ensemble_models = [f"pred_{name}" for name in specialist_ensemble_models]
    router_models = resolve_router_candidate_columns(
        model_router_candidates,
        available_prediction_cols=[
            col
            for col in test_sub.columns
            if col.startswith("pred_") and pd.api.types.is_numeric_dtype(test_sub[col])
        ],
    )
    missing = [col for col in ensemble_models if col not in test_sub.columns]
    if missing:
        raise ValueError(f"Missing specialist ensemble prediction columns: {missing}")
    missing_router = [col for col in router_models if col not in test_sub.columns]
    if missing_router:
        raise ValueError(f"Missing router candidate prediction columns: {missing_router}")

    test_sub = test_sub.copy()
    ranked_cols: dict[str, str] = {}
    for model_col in sorted(set(ensemble_models + router_models)):
        ranked_col = f"{model_col}__rankpct"
        ranked_cols[model_col] = ranked_col
        test_sub[ranked_col] = test_sub.groupby("Date")[model_col].rank(method="average", pct=True)

    date_order = sorted(test_sub["Date"].unique())
    member_top1_returns: dict[str, list[float]] = {model_col: [] for model_col in router_models}
    router_active_idx: int | None = None
    router_hold_days = 0
    test_sub["specialist_ensemble_active_model"] = ""
    test_sub["specialist_ensemble_switch_reason"] = ""
    test_sub["specialist_ensemble_weight_info"] = ""
    test_sub["specialist_ensemble_cash_gate"] = 0
    lookback_observations = router_history_lookback_observations(
        specialist_weight_lookback_days,
        rebalance_days,
    )
    current_weights = np.full(len(router_models), 1.0 / len(router_models), dtype=float)
    current_switch_reason = "no_history_equal"
    current_cash_gate = False

    for current_idx, current_date in enumerate(date_order):
        is_rebalance_date = current_idx % rebalance_days == 0
        if is_rebalance_date:
            normalized_weights, raw_scores = specialist_weights_from_history(
                member_top1_returns=member_top1_returns,
                ensemble_models=router_models,
                specialist_weight_lookback_observations=lookback_observations,
                specialist_weighting_mode=specialist_weighting_mode,
            )
            switch_reason = "stateless_mode"
            cash_gate = False
            if specialist_weighting_mode == "sticky_winner":
                (
                    normalized_weights,
                    router_active_idx,
                    router_hold_days,
                    switch_reason,
                    cash_gate,
                ) = sticky_specialist_router_step(
                    raw_scores=raw_scores,
                    ensemble_models=router_models,
                    previous_active_idx=router_active_idx,
                    previous_hold_days=router_hold_days,
                    specialist_min_model_hold_days=specialist_min_model_hold_days,
                    specialist_switch_margin_min_avg_ev=specialist_switch_margin_min_avg_ev,
                    specialist_switch_require_positive_ev=specialist_switch_require_positive_ev,
                )
            current_weights = normalized_weights
            current_switch_reason = switch_reason
            current_cash_gate = cash_gate
        else:
            normalized_weights = current_weights
            switch_reason = "between_rebalances"
            cash_gate = current_cash_gate

        date_mask = test_sub["Date"] == current_date
        weighted_score = np.zeros(int(date_mask.sum()), dtype=float)
        weighted_models = router_models if specialist_weighting_mode == "sticky_winner" else ensemble_models
        for idx, model_col in enumerate(weighted_models):
            weighted_score += normalized_weights[idx] * test_sub.loc[date_mask, ranked_cols[model_col]].to_numpy()
        test_sub.loc[date_mask, "pred_specialist_ensemble"] = weighted_score

        active_model = router_models[router_active_idx] if router_active_idx is not None else ""
        test_sub.loc[date_mask, "specialist_ensemble_active_model"] = active_model
        test_sub.loc[date_mask, "specialist_ensemble_switch_reason"] = (
            current_switch_reason if is_rebalance_date else switch_reason
        )
        test_sub.loc[date_mask, "specialist_ensemble_cash_gate"] = int(cash_gate)
        weight_parts = []
        for idx, model_col in enumerate(weighted_models):
            weight_parts.append(f"{model_col}={normalized_weights[idx]:.3f}")
        test_sub.loc[date_mask, "specialist_ensemble_weight_info"] = "|".join(weight_parts)
        if is_rebalance_date:
            date_slice = test_sub.loc[date_mask]
            for model_col in router_models:
                chosen_row = date_slice.sort_values(model_col, ascending=False).iloc[0]
                member_top1_returns[model_col].append(
                    realized_rebalance_outcome(chosen_row, transaction_loss_pct)
                )

    test_sub.drop(columns=list(ranked_cols.values()), inplace=True)
    return test_sub


def run_walkforward_model(
    long_df: pd.DataFrame,
    fit_days: int,
    model_fit_windows: list[int],
    step_days: int,
    rebalance_days: int,
    holdout_days: int,
    transaction_loss_pct: float,
    live_model: str,
    specialist_weighting_mode: str,
    specialist_ensemble_models: list[str],
    model_router_candidates: list[str],
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
    fit_windows = normalize_fit_windows(fit_days, model_fit_windows)
    max_fit_days = max(fit_windows)
    model_subsets = model_feature_subset_map(list(models))
    live_pred_col = resolve_live_prediction_column(live_model)
    specialist_prediction_cols = [f"pred_{name}" for name in specialist_ensemble_models]
    model_prediction_cols = [
        prediction_column_name(model_name, fit_window, fit_days)
        for fit_window in fit_windows
        for model_name in models
    ]
    ensemble_prediction_cols = [
        prediction_column_name("ensemble", fit_window, fit_days)
        for fit_window in fit_windows
    ]
    prediction_cols = [*model_prediction_cols, *ensemble_prediction_cols]
    router_prediction_cols = resolve_router_candidate_columns(
        model_router_candidates,
        available_prediction_cols=prediction_cols,
    )

    all_dates = sorted(long_df["Date"].unique())
    total_oos_days = max(len(all_dates) - max_fit_days, 0)
    holdout_start_oos_idx = max(total_oos_days - holdout_days, 0)
    all_test_chunks = []
    window_diagnostics = []
    primary_weights = primary_ensemble_weights(list(models))
    global_specialist_top1_returns: dict[str, list[float]] = {model_col: [] for model_col in router_prediction_cols}
    router_lookback_observations = router_history_lookback_observations(
        specialist_weight_lookback_days,
        rebalance_days,
    )
    current_specialist_weights = np.full(
        len(router_prediction_cols if specialist_weighting_mode == "sticky_winner" else specialist_prediction_cols),
        1.0 / len(router_prediction_cols if specialist_weighting_mode == "sticky_winner" else specialist_prediction_cols),
        dtype=float,
    )
    current_cash_gate = False
    current_switch_reason = "no_history_equal"
    router_active_idx: int | None = None
    router_hold_days = 0
    fit_start = 0
    window_idx = 0
    processed_oos_days = 0
    holdout_started = False
    while fit_start + max_fit_days < len(all_dates):
        train_end_idx = fit_start + max_fit_days
        window_idx += 1
        log(
            f"Training window {window_idx} "
            f"(fit windows: {','.join(str(window) for window in fit_windows)}, "
            f"cadence step: {step_days}, live model: {live_model})"
        )
        fitted_models: dict[tuple[str, int], object] = {}
        for fit_window in fit_windows:
            fit_dates = all_dates[train_end_idx - fit_window:train_end_idx]
            train_sub, _, y_train, _ = make_xy(long_df, fit_dates, feature_cols)
            for model_name, base_model in models.items():
                subset_name = model_subsets[model_name]
                model_feature_cols = subset_cols[subset_name]
                if not model_feature_cols:
                    raise ValueError(f"Feature subset '{subset_name}' for model '{model_name}' is empty")
                model = clone(base_model)
                model.fit(train_sub[model_feature_cols], y_train)
                fitted_models[(model_name, fit_window)] = model

        current_test_chunks = []
        trigger_returns: list[float] = []
        days_into_test = 0
        trigger_avg_ev = np.nan
        trigger_win_rate = np.nan
        retrain_reason = "end_of_data"
        next_date_idx = train_end_idx

        while next_date_idx < len(all_dates):
            current_date = all_dates[next_date_idx]
            is_rebalance_date = processed_oos_days % rebalance_days == 0
            day_sub = long_df[long_df["Date"] == current_date].sort_values(["Date", "pair"]).copy()
            for (model_name, fit_window), model in fitted_models.items():
                subset_name = model_subsets[model_name]
                model_feature_cols = subset_cols[subset_name]
                pred_col = prediction_column_name(model_name, fit_window, fit_days)
                day_sub[pred_col] = predict_positive_scores(model, day_sub[model_feature_cols])
            for fit_window in fit_windows:
                ensemble_col = prediction_column_name("ensemble", fit_window, fit_days)
                day_sub[ensemble_col] = sum(
                    day_sub[prediction_column_name(model_name, fit_window, fit_days)] * primary_weights[model_name]
                    for model_name in primary_weights
                )
            if is_rebalance_date:
                specialist_weights, raw_scores = specialist_weights_from_history(
                    member_top1_returns=global_specialist_top1_returns,
                    ensemble_models=router_prediction_cols if specialist_weighting_mode == "sticky_winner" else specialist_prediction_cols,
                    specialist_weight_lookback_observations=router_lookback_observations,
                    specialist_weighting_mode=specialist_weighting_mode,
                )
                switch_reason = "stateless_mode"
                cash_gate = False
                if specialist_weighting_mode == "sticky_winner":
                    (
                        specialist_weights,
                        router_active_idx,
                        router_hold_days,
                        switch_reason,
                        cash_gate,
                    ) = sticky_specialist_router_step(
                        raw_scores=raw_scores,
                        ensemble_models=router_prediction_cols,
                        previous_active_idx=router_active_idx,
                        previous_hold_days=router_hold_days,
                        specialist_min_model_hold_days=specialist_min_model_hold_days,
                        specialist_switch_margin_min_avg_ev=specialist_switch_margin_min_avg_ev,
                        specialist_switch_require_positive_ev=specialist_switch_require_positive_ev,
                    )
                current_specialist_weights = specialist_weights
                current_cash_gate = cash_gate
                current_switch_reason = switch_reason
            else:
                specialist_weights = current_specialist_weights
                switch_reason = "between_rebalances"
                cash_gate = current_cash_gate
            specialist_rank_sum = np.zeros(len(day_sub), dtype=float)
            weighted_models = router_prediction_cols if specialist_weighting_mode == "sticky_winner" else specialist_prediction_cols
            for idx, model_col in enumerate(weighted_models):
                rank_pct = day_sub[model_col].rank(method="average", pct=True).to_numpy()
                specialist_rank_sum += specialist_weights[idx] * rank_pct
            day_sub["pred_specialist_ensemble"] = specialist_rank_sum
            day_sub["specialist_ensemble_active_model"] = (
                router_prediction_cols[router_active_idx] if router_active_idx is not None else ""
            )
            day_sub["specialist_ensemble_switch_reason"] = (
                current_switch_reason if is_rebalance_date else switch_reason
            )
            day_sub["specialist_ensemble_cash_gate"] = int(cash_gate)
            day_sub["specialist_ensemble_weight_info"] = "|".join(
                f"{model_col}={specialist_weights[idx]:.3f}" for idx, model_col in enumerate(weighted_models)
            )
            day_sub["window_id"] = window_idx
            day_sub["oos_day_index"] = processed_oos_days
            day_sub["is_rebalance_date"] = int(is_rebalance_date)
            current_test_chunks.append(
                day_sub[
                    [
                        "window_id",
                        "oos_day_index",
                        "is_rebalance_date",
                        *META_COLUMNS,
                        *prediction_cols,
                        "specialist_ensemble_active_model",
                        "specialist_ensemble_switch_reason",
                        "specialist_ensemble_cash_gate",
                        "specialist_ensemble_weight_info",
                    ]
                ]
            )

            trigger_row = day_sub.sort_values(live_pred_col, ascending=False).iloc[0]
            trigger_returns.append(float(trigger_row["ev_target"]))
            if is_rebalance_date:
                for model_col in router_prediction_cols:
                    chosen_row = day_sub.sort_values(model_col, ascending=False).iloc[0]
                    global_specialist_top1_returns[model_col].append(
                        realized_rebalance_outcome(chosen_row, transaction_loss_pct)
                    )
            days_into_test += 1
            next_date_idx += 1
            processed_oos_days += 1

            if not holdout_started and processed_oos_days > holdout_start_oos_idx:
                holdout_started = True
                holdout_day_num = processed_oos_days - holdout_start_oos_idx
                log(
                    f"Entering holdout period: day {holdout_day_num}/{holdout_days} "
                    f"({current_date})"
                )
            elif holdout_started:
                holdout_day_num = processed_oos_days - holdout_start_oos_idx
                if holdout_day_num == holdout_days or holdout_day_num % 5 == 0:
                    log(
                        f"Holdout progress: {holdout_day_num}/{holdout_days} days processed "
                        f"({current_date})"
                    )

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
                f"router_candidates={','.join(model_router_candidates)}; "
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

        for fit_window in fit_windows:
            for model_name in models:
                window_diagnostics.append(
                    {
                        "window_id": window_idx,
                        "model": prediction_column_name(model_name, fit_window, fit_days),
                        "fit_dates": fit_window,
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
                    "model": prediction_column_name("ensemble", fit_window, fit_days),
                    "fit_dates": fit_window,
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
                "fit_dates": max_fit_days,
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
        model_router_candidates,
        specialist_weight_lookback_days,
        rebalance_days,
        transaction_loss_pct,
        specialist_weighting_mode,
        specialist_min_model_hold_days,
        specialist_switch_margin_min_avg_ev,
        specialist_switch_require_positive_ev,
    )
    window_diag_df = pd.DataFrame(window_diagnostics)
    return pred_df, window_diag_df
