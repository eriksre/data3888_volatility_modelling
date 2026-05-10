# Shared paths, constants and feature definitions.
# Update DATA_DIR to point to your local copy of the parquet files.

import os

DATA_DIR    = r"C:\Users\Ayush\Downloads\DATA3888\data3888 finc6\Optiver_extracted\individual_book_train"
CLUSTER_CSV = r"C:\Users\Ayush\Downloads\data3888_volatility_modelling\plots\recluster\stock_cluster_assignments_FINAL.csv"
PLOTS_DIR   = r"C:\Users\Ayush\Downloads\data3888_volatility_modelling\plots"

FEATURE_COLS = [
    # volatility
    "rv_first", "std_log_return", "skew_log_return", "kurt_log_return",
    "max_log_return", "min_log_return",
    "wap_range", "wap_std", "n_price_changes",
    # spread / liquidity
    "mean_spread", "std_spread", "spread_range",
    "mean_rel_spread", "std_rel_spread",
    # order book
    "mean_depth", "std_depth",
    "mean_imbalance", "std_imbalance",
    "mean_bid_size1", "mean_ask_size1",
    # activity & momentum
    "n_rows", "arrival_rate", "wap_momentum",
    # lags
    "rv_first_lag1", "rv_first_lag2",
    "log_rv_lag1", "log_rv_lag2", "log_rv_lag3", "log_rv_rolling5",
    # interactions
    "vol_per_depth", "vol_x_spread", "pressure_x_vol",
]

COLOURS = {
    "RF":       "#2196F3",
    "XGBoost":  "#FF5722",
    "LightGBM": "#4CAF50",
    "HAR":      "#FF9800",
    "GARCH":    "#9C27B0",
    "C1":       "#2196F3",
    "C2":       "#FF5722",
}
