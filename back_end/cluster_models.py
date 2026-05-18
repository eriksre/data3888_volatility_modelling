import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc

from sklearn.metrics import mean_squared_error

from config import FEATURE_COLS, CLUSTER_CSV, PLOTS_DIR
from utils import load_cluster_data, temporal_split, qlike, make_models

PLOT_DIR = os.path.join(PLOTS_DIR, "cluster_models")
os.makedirs(PLOT_DIR, exist_ok=True)


def train_cluster(cluster_id, stock_ids):
    print(f"\nCluster {cluster_id}  ({len(stock_ids)} stocks)")

    data = load_cluster_data(stock_ids)
    print(f"Samples: {len(data):,}")

    train, test, cutoff = temporal_split(data)
    X_tr = train[FEATURE_COLS].values.astype(np.float32)
    y_tr = train["log_rv_second"].values.astype(np.float32)
    X_te = test[FEATURE_COLS].values.astype(np.float32)
    y_te = test["log_rv_second"].values.astype(np.float32)
    del data, train, test
    gc.collect()

    print(f"Train: {len(X_tr):,}  |  Test: {len(X_te):,}  (cutoff={cutoff})")

    results = {"n_stocks": len(stock_ids)}
    rf_importances = None

    for name, model in make_models().items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
        ql     = qlike(y_te, y_pred)
        print(f"  {name:<12} RMSE={rmse:.4f}  QLIKE={ql:.4f}")

        if name == "RF":
            rf_importances = model.feature_importances_.copy()

        results[name] = {"RMSE": rmse, "QLIKE": ql, "y_test": y_te, "y_pred": y_pred}
        del model
        gc.collect()

    results["rf_importances"] = rf_importances
    del X_tr, y_tr, X_te
    gc.collect()
    return results


def save_summary(all_results):
    rows = []
    for cid, res in all_results.items():
        row = {"Cluster": cid, "N Stocks": res["n_stocks"]}
        for m in make_models():
            row[f"{m} RMSE"]  = round(res[m]["RMSE"],  4)
            row[f"{m} QLIKE"] = round(res[m]["QLIKE"], 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PLOT_DIR, "cluster_results.csv"), index=False)
    print(f"\n{df.to_string(index=False)}")
    return df


def plot_comparison(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Cluster Models — RF vs XGBoost vs LightGBM (Temporal Split)",
                 fontweight="bold")

    cluster_labels = [f"Cluster {cid}" for cid in sorted(all_results)]
    x, w = np.arange(len(cluster_labels)), 0.25

    for ax, metric in zip(axes, ["RMSE", "QLIKE"]):
        for i, m in enumerate(make_models()):
            vals = [all_results[cid][m][metric] for cid in sorted(all_results)]
            bars = ax.bar(x + (i - 1) * w, vals, w, label=m, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_labels)
        ax.set_ylabel(metric)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(all_results):
    fig, axes = plt.subplots(1, len(all_results), figsize=(8 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]
    fig.suptitle("RF Feature Importances by Cluster", fontweight="bold")

    for ax, cid in zip(axes, sorted(all_results)):
        imp = pd.Series(all_results[cid]["rf_importances"],
                        index=FEATURE_COLS).sort_values()
        imp.plot(kind="barh", ax=ax, alpha=0.8)
        ax.set_title(f"Cluster {cid}")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rf_feature_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    cluster_df = pd.read_csv(CLUSTER_CSV, names=["stock_id", "cluster"], header=0)

    all_results = {}
    for cid in sorted(cluster_df["cluster"].unique()):
        stock_ids = cluster_df[cluster_df["cluster"] == cid]["stock_id"].tolist()
        all_results[cid] = train_cluster(cid, stock_ids)

    save_summary(all_results)
    plot_comparison(all_results)
    plot_feature_importance(all_results)
    print(f"\nPlots saved to: {PLOT_DIR}")
