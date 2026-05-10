"""
timeseries_models.py
HAR-RV and GARCH(1,1) on representative stocks from each cluster.
Benchmarks these classical time-series models against the ML cluster models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from arch import arch_model

from config import PLOTS_DIR, COLOURS
from utils import qlike

PLOT_DIR = os.path.join(PLOTS_DIR, "timeseries_models")
os.makedirs(PLOT_DIR, exist_ok=True)

DATA_DIR = r"C:\Users\Ayush\Downloads\DATA3888\data3888 finc6\Optiver_extracted\individual_book_train"

# 5 representative stocks per cluster
CLUSTER_STOCKS = {
    1: [18, 37, 80, 5, 97],
    2: [14, 43, 29, 99, 125],
}

# ML cluster benchmarks from cluster_models.py
ML_BENCHMARK = {
    1: {"RMSE": 0.2783, "QLIKE": 0.1633},
    2: {"RMSE": 0.2282, "QLIKE": 0.1117},
}


# ── Data loading ──────────────────────────────────────────────────────────────

def compute_wap(df):
    return (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )


def load_stock(stock_id):
    """Returns one row per time_id with RV and last WAP."""
    df = pd.read_parquet(os.path.join(DATA_DIR, f"stock_{stock_id}.parquet"))
    df["wap"] = compute_wap(df)
    df["log_return"] = df.groupby("time_id")["wap"].transform(lambda x: np.log(x).diff())

    def bucket_rv(g):
        return np.sqrt((g["log_return"].dropna() ** 2).sum())

    rv1 = df[df["seconds_in_bucket"] <  300].groupby("time_id").apply(bucket_rv)
    rv2 = df[df["seconds_in_bucket"] >= 300].groupby("time_id").apply(bucket_rv)
    wap_last = df.groupby("time_id")["wap"].last()

    out = pd.DataFrame({"rv_first": rv1, "rv_second": rv2, "last_wap": wap_last})
    out = out.dropna().sort_index().reset_index()
    out["log_rv_first"]  = np.log(out["rv_first"]  + 1e-8)
    out["log_rv_second"] = np.log(out["rv_second"] + 1e-8)
    return out


# ── HAR-RV ────────────────────────────────────────────────────────────────────

def fit_har(data):
    """
    Log-HAR with non-overlapping lag windows:
      Daily   = log(RV_{t-1})
      Weekly  = mean(log(RV_{t-2..t-5}))
      Monthly = mean(log(RV_{t-6..t-22}))
    Uses lags of the target variable (log_rv_second) — standard HAR-RV formulation.
    """
    df = data.copy()
    df["lag_d"] = df["log_rv_second"].shift(1)
    df["lag_w"] = df["log_rv_second"].shift(2).rolling(4).mean()
    df["lag_m"] = df["log_rv_second"].shift(6).rolling(17).mean()
    df = df.dropna(subset=["lag_d", "lag_w", "lag_m", "log_rv_second"])

    split   = int(len(df) * 0.8)
    X_cols  = ["lag_d", "lag_w", "lag_m"]
    X_tr, y_tr = df[X_cols].values[:split], df["log_rv_second"].values[:split]
    X_te, y_te = df[X_cols].values[split:], df["log_rv_second"].values[split:]

    model  = LinearRegression().fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    return {
        "y_test": y_te,
        "y_pred": y_pred,
        "RMSE":   np.sqrt(mean_squared_error(y_te, y_pred)),
        "QLIKE":  qlike(y_te, y_pred),
        "coefs":  dict(zip(["intercept", "daily", "weekly", "monthly"],
                           [round(model.intercept_, 4)] +
                           [round(c, 4) for c in model.coef_])),
    }


# ── GARCH(1,1) ────────────────────────────────────────────────────────────────

def fit_garch(data):
    """
    GARCH(1,1) on bucket-to-bucket WAP log returns (scaled to %).
    Conditional variance is linearly calibrated to log(RV) on the training set.
    """
    df = data.copy()
    df["ret"] = np.log(df["last_wap"]).diff() * 100
    df = df.dropna(subset=["ret", "log_rv_second"])

    split  = int(len(df) * 0.8)
    ret_tr = df["ret"].values[:split]
    ret_all = df["ret"].values

    res_tr = arch_model(ret_tr, mean="Constant", vol="Garch", p=1, q=1,
                        dist="normal", rescale=False).fit(disp="off", show_warning=False)

    res_full = arch_model(ret_all, mean="Constant", vol="Garch", p=1, q=1,
                          dist="normal", rescale=False).fit(
        disp="off", show_warning=False,
        starting_values=res_tr.params.values
    )

    cond_var = np.asarray(res_full.conditional_volatility) ** 2

    # Calibrate: log(RV) ~ a + b * 0.5*log(sigma^2 / 10000) on training set
    log_vol_tr = 0.5 * np.log(np.maximum(cond_var[:split] / 10000, 1e-20))
    log_rv_tr  = df["log_rv_second"].values[:split]
    X_cal = np.column_stack([np.ones(split), log_vol_tr])
    cal   = np.linalg.lstsq(X_cal, log_rv_tr, rcond=None)[0]

    log_vol_te = 0.5 * np.log(np.maximum(cond_var[split:] / 10000, 1e-20))
    y_pred = np.column_stack([np.ones(len(log_vol_te)), log_vol_te]) @ cal
    y_te   = df["log_rv_second"].values[split:]

    min_len = min(len(y_te), len(y_pred))
    y_te, y_pred = y_te[:min_len], y_pred[:min_len]
    mask = np.isfinite(y_pred) & np.isfinite(y_te)
    y_te, y_pred = y_te[mask], y_pred[mask]

    if len(y_pred) == 0:
        return None

    p = res_tr.params
    return {
        "y_test":      y_te,
        "y_pred":      y_pred,
        "RMSE":        np.sqrt(mean_squared_error(y_te, y_pred)),
        "QLIKE":       qlike(y_te, y_pred),
        "persistence": float(p.get("alpha[1]", 0)) + float(p.get("beta[1]", 0)),
        "params": {
            "omega": float(p.get("omega", np.nan)),
            "alpha": float(p.get("alpha[1]", np.nan)),
            "beta":  float(p.get("beta[1]",  np.nan)),
        },
    }


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = {}

    for cid, stock_ids in CLUSTER_STOCKS.items():
        print(f"\nCluster {cid}")
        print("-" * 50)
        all_results[cid] = {}

        for sid in stock_ids:
            data  = load_stock(sid)
            har   = fit_har(data)
            garch = fit_garch(data)

            if garch is None:
                print(f"  Stock {sid}: GARCH failed, skipping")
                continue

            all_results[cid][sid] = {"HAR": har, "GARCH": garch}
            print(f"  Stock {sid:3d}  "
                  f"HAR  RMSE={har['RMSE']:.4f}  QLIKE={har['QLIKE']:.4f}  |  "
                  f"GARCH RMSE={garch['RMSE']:.4f}  QLIKE={garch['QLIKE']:.4f}  "
                  f"persist={garch['persistence']:.3f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    for cid, stocks in all_results.items():
        for sid, res in stocks.items():
            rows.append({
                "Cluster":         cid,
                "Stock":           sid,
                "HAR RMSE":        round(res["HAR"]["RMSE"],   4),
                "HAR QLIKE":       round(res["HAR"]["QLIKE"],  4),
                "GARCH RMSE":      round(res["GARCH"]["RMSE"], 4),
                "GARCH QLIKE":     round(res["GARCH"]["QLIKE"],4),
                "GARCH persist":   round(res["GARCH"]["persistence"], 3),
                "ML Cluster RMSE": ML_BENCHMARK[cid]["RMSE"],
            })

    df = pd.DataFrame(rows)
    print(f"\n{'='*75}")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(PLOT_DIR, "timeseries_results.csv"), index=False)

    # ── RMSE comparison plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("HAR-RV vs GARCH vs ML — RMSE by Stock", fontweight="bold")

    for ax, cid in zip(axes, [1, 2]):
        stocks     = list(all_results[cid].keys())
        har_rmse   = [all_results[cid][s]["HAR"]["RMSE"]   for s in stocks]
        garch_rmse = [all_results[cid][s]["GARCH"]["RMSE"] for s in stocks]
        x, w = np.arange(len(stocks)), 0.3

        ax.bar(x - w/2, har_rmse,   w, label="HAR-RV",     color=COLOURS["HAR"],   alpha=0.85)
        ax.bar(x + w/2, garch_rmse, w, label="GARCH(1,1)", color=COLOURS["GARCH"], alpha=0.85)
        ax.axhline(ML_BENCHMARK[cid]["RMSE"], color=COLOURS["LightGBM"],
                   linestyle="--", linewidth=2,
                   label=f"ML Cluster (RMSE={ML_BENCHMARK[cid]['RMSE']})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Stock {s}" for s in stocks], rotation=15)
        ax.set_title(f"Cluster {cid}", fontweight="bold")
        ax.set_ylabel("RMSE")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rmse_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── HAR coefficient plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("HAR-RV Coefficients by Stock", fontweight="bold")

    for ax, cid in zip(axes, [1, 2]):
        stocks = list(all_results[cid].keys())
        bd = [all_results[cid][s]["HAR"]["coefs"]["daily"]   for s in stocks]
        bw = [all_results[cid][s]["HAR"]["coefs"]["weekly"]  for s in stocks]
        bm = [all_results[cid][s]["HAR"]["coefs"]["monthly"] for s in stocks]
        x, w = np.arange(len(stocks)), 0.25
        ax.bar(x - w, bd, w, label="Daily (lag-1)",    color="#FF9800", alpha=0.85)
        ax.bar(x,     bw, w, label="Weekly (lags 2-5)",color="#4CAF50", alpha=0.85)
        ax.bar(x + w, bm, w, label="Monthly (6-22)",   color="#2196F3", alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s}" for s in stocks])
        ax.set_title(f"Cluster {cid}")
        ax.set_ylabel("Coefficient")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "har_coefficients.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nPlots saved to: {PLOT_DIR}")
