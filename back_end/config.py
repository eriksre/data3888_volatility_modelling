DATA_DIR = (
    "/Users/vaniakumar/Desktop/data3888/"
    "data3888_volatility_modelling/"
    "individual_book_train_parquet"
)

FEATURE_COLS = [

    "rv_first",

    "std_log_return",
    "skew_log_return",
    "kurt_log_return",

    "max_log_return",
    "min_log_return",

    "realized_quarticity",
    "abs_return_mean",
    "max_abs_return",

    "wap_range",
    "wap_std",

    "n_price_changes",

    "mean_spread",
    "std_spread",
    "spread_range",

    "mean_rel_spread",
    "std_rel_spread",

    "mean_depth",
    "std_depth",

    "mean_imbalance",
    "std_imbalance",

    "mean_bid_size1",
    "mean_ask_size1",

    "arrival_rate",
    "wap_momentum",

    "acf1",
    "acf5",

    "pacf1",
    "pacf5",

    "rv_first_lag1",
    "rv_first_lag2",

    "log_rv_lag1",
    "log_rv_lag2",
    "log_rv_lag3",

    "log_rv_rolling5",

    "vol_per_depth",
    "vol_x_spread",
    "pressure_x_vol",
]