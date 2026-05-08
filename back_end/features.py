import numpy as np
import pandas as pd
import os
import gc
from config import DATA_DIR, FEATURE_COLS
from statsmodels.tsa.stattools import acf, pacf

def make_equally_spaced(df):

    all_time_ids = df["time_id"].unique()

    out = []

    for tid in all_time_ids:

        sub = (
            df[df["time_id"] == tid]
            .sort_values("seconds_in_bucket")
            .copy()
        )

        full_index = pd.DataFrame({
            "seconds_in_bucket": np.arange(600)
        })

        sub = full_index.merge(
            sub,
            on="seconds_in_bucket",
            how="left"
        )

        sub["time_id"] = tid

        sub = sub.ffill()

        out.append(sub)

    return pd.concat(out, ignore_index=True)

def compute_wap(df):

    return (
        (
            df["bid_price1"] * df["ask_size1"]
            + df["ask_price1"] * df["bid_size1"]
            + df["bid_price2"] * df["ask_size2"]
            + df["ask_price2"] * df["bid_size2"]
        )
        /
        (
            df["bid_size1"]
            + df["ask_size1"]
            + df["bid_size2"]
            + df["ask_size2"]
        )
    )

def safe_acf(x, lag=1):

    x = x.dropna()

    if len(x) <= lag or np.std(x) == 0:
        return np.nan

    return acf(x, nlags=lag)[lag]


def safe_pacf(x, lag=1):

    x = x.dropna()

    if len(x) <= lag or np.std(x) == 0:
        return np.nan

    return pacf(x, nlags=lag)[lag]

def extract_features(first_half):
    df = first_half.copy()
    df["wap"]        = compute_wap(df)
    df["mid_price"]  = (df["bid_price1"] + df["ask_price1"]) / 2
    df["spread"]     = df["ask_price1"] - df["bid_price1"]
    df["rel_spread"] = df["spread"] / df["mid_price"]
    df["depth"]      = df["bid_size1"] + df["ask_size1"]
    df["imbalance"]  = (df["bid_size1"] - df["ask_size1"]) / df["depth"]
    df["log_return"] = df.groupby(["stock_id", "time_id"])["wap"].transform(
        lambda x: np.log(x).diff()
    )

    feats = df.groupby(["stock_id", "time_id"]).agg(
    rv_first        = ("log_return", lambda x: np.sqrt((x.dropna() ** 2).sum())),

    std_log_return  = ("log_return", "std"),
    skew_log_return = ("log_return", lambda x: x.dropna().skew()),
    kurt_log_return = ("log_return", lambda x: x.dropna().kurt()),

    max_log_return  = ("log_return", lambda x: x.dropna().max()),
    min_log_return  = ("log_return", lambda x: x.dropna().min()),

    realized_quarticity = ("log_return", lambda x: np.sum((x.dropna() ** 4))),

    abs_return_mean = ("log_return", lambda x: np.mean(np.abs(x.dropna()))),

    max_abs_return  = ("log_return", lambda x: np.max(np.abs(x.dropna()))),

    wap_range       = ("wap", lambda x: x.max() - x.min()),
    wap_std         = ("wap", "std"),

    n_price_changes = ("wap", lambda x: (x.diff().dropna() != 0).sum()),

    mean_spread     = ("spread", "mean"),
    std_spread      = ("spread", "std"),
    spread_range    = ("spread", lambda x: x.max() - x.min()),

    mean_rel_spread = ("rel_spread", "mean"),
    std_rel_spread  = ("rel_spread", "std"),

    mean_depth      = ("depth", "mean"),
    std_depth       = ("depth", "std"),

    mean_imbalance  = ("imbalance", "mean"),
    std_imbalance   = ("imbalance", "std"),

    mean_bid_size1  = ("bid_size1", "mean"),
    mean_ask_size1  = ("ask_size1", "mean"),

    n_rows          = ("wap", "count"),
    wap_first       = ("wap", "first"),
    wap_last        = ("wap", "last"),

    acf1 = ("log_return", lambda x: safe_acf(x ** 2, lag=1)),
    acf5 = ("log_return", lambda x: safe_acf(x ** 2, lag=5)),

    pacf1 = ("log_return", lambda x: safe_pacf(x ** 2, lag=1)),
    pacf5 = ("log_return", lambda x: safe_pacf(x ** 2, lag=5)),
).reset_index()

    feats["arrival_rate"] = feats["n_rows"] / 300
    feats["wap_momentum"] = feats["wap_last"] - feats["wap_first"]
    feats = feats.drop(columns=["wap_first", "wap_last"])
    return feats


def compute_target(second_half):
    df = second_half.copy()
    df["wap"] = compute_wap(df)

    def _rv(g):
        return np.sqrt((np.log(g["wap"]).diff().dropna() ** 2).sum())

    tgt = df.groupby(["stock_id", "time_id"]).apply(_rv, include_groups=False).reset_index()
    tgt.columns = ["stock_id", "time_id", "rv_second"]
    tgt["log_rv_second"] = np.log(tgt["rv_second"] + 1e-8)
    return tgt[["stock_id", "time_id", "log_rv_second"]]


def add_lag_features(data):
    data = data.sort_values(["stock_id", "time_id"]).copy()
    grp  = data.groupby("stock_id")["log_rv_second"]

    data["rv_first_lag1"]   = data.groupby("stock_id")["rv_first"].shift(1)
    data["rv_first_lag2"]   = data.groupby("stock_id")["rv_first"].shift(2)
    data["log_rv_lag1"]     = grp.shift(1)
    data["log_rv_lag2"]     = grp.shift(2)
    data["log_rv_lag3"]     = grp.shift(3)
    data["log_rv_rolling5"] = grp.shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    return data


def add_interaction_features(data):
    eps = 1e-10
    data["vol_per_depth"]  = data["rv_first"] / (data["mean_depth"] + eps)
    data["vol_x_spread"]   = data["rv_first"] *  data["mean_rel_spread"]
    data["pressure_x_vol"] = data["mean_imbalance"].abs() * data["rv_first"]
    return data


def load_cluster_data(stock_ids):
    all_feats, all_tgts = [], []
    for sid in stock_ids:
        df = pd.read_parquet(os.path.join(DATA_DIR, f"stock_{sid}.parquet"))
        all_feats.append(extract_features(df[df["seconds_in_bucket"] <  300]))
        all_tgts.append(compute_target(df[df["seconds_in_bucket"]    >= 300]))
        del df
        gc.collect()

    feats = pd.concat(all_feats, ignore_index=True)
    tgts  = pd.concat(all_tgts,  ignore_index=True)
    del all_feats, all_tgts
    gc.collect()

    data = feats.merge(tgts, on=["stock_id", "time_id"])
    del feats, tgts
    gc.collect()

    data = add_lag_features(data)
    data = add_interaction_features(data)
    data = data.dropna(subset=FEATURE_COLS + ["log_rv_second"])
    return data


def load_single_stock(stock_id):
    df    = pd.read_parquet(os.path.join(DATA_DIR, f"stock_{stock_id}.parquet"))
    feats = extract_features(df[df["seconds_in_bucket"] <  300])
    tgt   = compute_target(df[df["seconds_in_bucket"]   >= 300])
    data  = feats.merge(tgt, on=["stock_id", "time_id"])
    data  = add_lag_features(data)
    data  = add_interaction_features(data)
    data  = data.dropna(subset=FEATURE_COLS + ["log_rv_second"])
    return data