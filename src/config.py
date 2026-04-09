"""Central configuration for FX walk-forward pipeline."""

FIT_DAYS = 480
STEP_DAYS = 30
REBALANCE_DAYS = 1
HORIZON = 5
TRANSACTION_LOSS_PCT = 0.025
TRADING_DAYS_PER_YEAR = 260
HOLDOUT_DAYS = 60
LIVE_MODEL = "specialist_ensemble"
SPECIALIST_WEIGHTING_MODE = "soft_dynamic"
SPECIALIST_ENSEMBLE_MEMBERS = [
    "lgbm_deep_returns_momentum",
    "logreg_returns_momentum",
]
SPECIALIST_WEIGHT_LOOKBACK_DAYS = 60
RETRAIN_DETERIORATION_LOOKBACK_DAYS = 20
RETRAIN_DETERIORATION_MIN_WIN_RATE = 0.45
RETRAIN_DETERIORATION_MAX_AVG_EV = 0.0
RAW_DATA_FILENAME = "fx_raw_ohlc.csv"
FEATURE_DATA_FILENAME = "fx_features_full.csv"
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
    "vol30_",
    "mom5_",
    "mom10_",
    "corr20_",
    "corr40_",
    "corr60_",
    "corr120_",
    "rs1_",
    "rs5_",
    "rs10_",
    "rs20_",
    "rank_ret_",
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
    "prev_top1_",
    "rank_persist5_",
    "signal_stability5_",
]
