"""Central configuration for FX walk-forward pipeline."""

FIT_DAYS = 180
MODEL_FIT_WINDOWS = [30, 60, 90, 120, 150, 180]
STEP_DAYS = 30
REBALANCE_DAYS = 7
HORIZON = 7
TRANSACTION_LOSS_PCT = 0.025
TRADING_DAYS_PER_YEAR = 260
HOLDOUT_DAYS = 5000
LIVE_MODEL = "specialist_ensemble"
SPECIALIST_WEIGHTING_MODE = "sticky_winner"
SPECIALIST_ENSEMBLE_MEMBERS = [
    "lgbm_deep_returns_momentum",
    "lgbm_deep_volatility",
    "lgbm_deep_corr_regime",
    "rf_returns_momentum",
    "logreg_returns_momentum",
]
MODEL_ROUTER_CANDIDATES = ["all"]
SPECIALIST_WEIGHT_LOOKBACK_DAYS = 21
SPECIALIST_MIN_MODEL_HOLD_DAYS = 4
SPECIALIST_SWITCH_MARGIN_MIN_AVG_EV = 0.0
SPECIALIST_SWITCH_REQUIRE_POSITIVE_EV = True
ROUTER_SELECTION_MODE = "top3_blend"
ROUTER_MIN_EDGE_OVER_NEXT = 0.002
ROUTER_FALLBACK_MODEL = "pred_logreg_returns_momentum_fit150"
RETRAIN_DETERIORATION_LOOKBACK_DAYS = 30
RETRAIN_DETERIORATION_MIN_WIN_RATE = 0.45
RETRAIN_DETERIORATION_MAX_AVG_EV = 0.0
RAW_DATA_FILENAME = "fx_raw_ohlc.csv"
FEATURE_DATA_FILENAME = "fx_features_full.csv"
RAW_DATA_SOURCE = "yfinance"
ALPHAVANTAGE_API_KEY = ""
YAHOO_DOWNLOAD_PERIOD = "1000d"
YAHOO_DOWNLOAD_INTERVAL = "1d"

FX_PAIRS = [
    "EURUSD=X",
    "USDJPY=X",
    "GBPUSD=X",
    "AUDUSD=X",
    "NZDUSD=X",
    "USDCHF=X",
    "USDCAD=X",
    "EURJPY=X",
    "GBPJPY=X",
    "AUDJPY=X",
    "EURGBP=X",
    "EURNZD=X",
    "EURCHF=X",
    "CHFJPY=X",
    "CADJPY=X",
    "GBPCHF=X",
    "GBPCAD=X",
    "AUDNZD=X",
    "AUDCAD=X",
    "AUDCHF=X",
    "NZDJPY=X",
    "NZDCAD=X",
    "NZDCHF=X",
    "CADCHF=X",
    "EURCAD=X",
]

PAIR_PREFIXES = [
    "ret_",
    "ret3_",
    "ret5_",
    "ret7_",
    "ret10_",
    "ret15_",
    "ret20_",
    "vol30_",
    "mom5_",
    "mom10_",
    "corr20_",
    "corr40_",
    "corr60_",
    "corr120_",
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
    "rank_vol30_",
    "accel_",
    "slope20_",
    "dvol30_",
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
    "dist_high5_",
    "dist_high10_",
    "dist_high20_",
    "dist_high60_",
    "dist_low5_",
    "dist_low10_",
    "dist_low20_",
    "dist_low60_",
    "range_pos5_",
    "range_pos10_",
    "range_pos20_",
    "range_pos60_",
    "usd_resid_",
    "rank_usd_resid_",
    "ret_x_market_vol20_",
    "ret20_x_avg_corr20_",
    "ret20_x_usd_mom20_",
    "prev_top1_",
    "rank_persist5_",
    "signal_stability5_",
]
