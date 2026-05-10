import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_squared_error

from config import FEATURE_COLS, CLUSTER_CSV, PLOTS_DIR, COLOURS
from utils import load_single_stock, temporal_split, qlike, make_models

PLOT_DIR   = os.path.join(PLOTS_DIR, "individual_models")
BENCH_PATH = os.path.join(PLOTS_DIR, "cluster_models", "cluster_results.csv")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_benchmarks():
    if not os.path.exists(BENCH_PATH):
        print("WARNING: run cluster_models.py first to generate benchmarks.")
        return {}
    df = pd.read_csv(BENCH_PATH)
    bench = {}
    for _, row in df.iterrows():
        cid = int(row["Cluster"])
        bench[cid] = {m: float(row[f"{m} RMSE"]) for m in make_models()}
    print("Cluster benchmarks:", bench)
    return bench


def train_stock(stock_id):
    data = load_single_stock(stock_id)
    if len(data) < 100:
        return None

    train, test, _ = temporal_split(data)
    X_tr, y_tr = train[FEATURE_COLS].values, train["log_rv_second"].values
    X_te, y_te = test[FEATURE_COLS].values,  test["log_rv_second"].values

    row = {"stock_id": stock_id, "n_train": len(X_tr), "n_test": len(X_te)}
    for name, model in make_models().items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        row[f"{name}_RMSE"]  = round(np.sqrt(mean_squared_error(y_te, y_pred)), 4)
        row[f"{name}_QLIKE"] = round(qlike(y_te, y_pred), 4)
    return row


def plot_rmse_scatter(results_df, benchmarks):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Per-Stock RMSE  —  Blue = Cluster 1  |  Red = Cluster 2",
                 fontweight="bold")
    cmap = {1: COLOURS["C1"], 2: COLOURS["C2"]}

    for ax, m in zip(axes, make_models()):
        ax.scatter(results_df["stock_id"], results_df[f"{m}_RMSE"],
                   c=results_df["cluster"].map(cmap),
                   s=40, alpha=0.8, edgecolors="white", linewidths=0.4)
        for cid, color in cmap.items():
            if benchmarks.get(cid, {}).get(m):
                ax.axhline(benchmarks[cid][m], color=color, linestyle="--",
                           linewidth=1.5, label=f"C{cid} cluster model")
        ax.set_xlabel("Stock ID"); ax.set_ylabel("RMSE")
        ax.set_title(m, fontweight="bold")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rmse_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_win_rate(results_df, benchmarks):
    if not benchmarks:
        return

    win_rates = {}
    for m in make_models():
        wins = sum(
            results_df.apply(
                lambda r: r[f"{m}_RMSE"] < benchmarks.get(int(r["cluster"]), {}).get(m, np.inf),
                axis=1
            )
        )
        win_rates[m] = wins / len(results_df) * 100
        print(f"  {m:<12} wins {wins}/{len(results_df)} ({win_rates[m]:.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(win_rates.keys(), win_rates.values(),
                  color=[COLOURS[m] for m in win_rates], alpha=0.85)
    for bar, v in zip(bars, win_rates.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(50, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("% of stocks where individual beats cluster")
    ax.set_title("Individual vs Cluster — Win Rate", fontweight="bold")
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "win_rate.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_rmse_boxplot(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RMSE Distribution Across Individual Stocks by Cluster", fontweight="bold")

    for ax, metric in zip(axes, ["RMSE", "QLIKE"]):
        plot_data, labels, colors = [], [], []
        for cid in sorted(results_df["cluster"].unique()):
            sub = results_df[results_df["cluster"] == cid]
            for m in make_models():
                plot_data.append(sub[f"{m}_{metric}"].dropna().values)
                labels.append(f"C{cid}\n{m}")
                colors.append(COLOURS[m])

        bp = ax.boxplot(plot_data, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(metric)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rmse_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    cluster_df = pd.read_csv(CLUSTER_CSV, names=["stock_id", "cluster"], header=0)
    benchmarks = load_benchmarks()
    stock_ids  = sorted(cluster_df["stock_id"].tolist())
    all_rows   = []

    for i, sid in enumerate(stock_ids):
        cid = int(cluster_df[cluster_df["stock_id"] == sid]["cluster"].iloc[0])
        try:
            row = train_stock(sid)
            if row is None:
                print(f"  [{i+1}/{len(stock_ids)}] Stock {sid}: skipped")
                continue
            row["cluster"] = cid
            all_rows.append(row)
            print(f"  [{i+1}/{len(stock_ids)}] Stock {sid:3d} (C{cid})  "
                  f"RF={row['RF_RMSE']:.4f}  "
                  f"XGB={row['XGBoost_RMSE']:.4f}  "
                  f"LGBM={row['LightGBM_RMSE']:.4f}")
        except Exception as e:
            print(f"  [{i+1}/{len(stock_ids)}] Stock {sid}: {e}")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(os.path.join(PLOT_DIR, "individual_results.csv"), index=False)
    print(f"\nSaved {len(results_df)} stocks.")

    print("\nCluster summary:")
    for cid in sorted(results_df["cluster"].unique()):
        sub = results_df[results_df["cluster"] == cid]
        print(f"\nCluster {cid} ({len(sub)} stocks):")
        for m in make_models():
            print(f"  {m:<12} avg={sub[f'{m}_RMSE'].mean():.4f}  "
                  f"min={sub[f'{m}_RMSE'].min():.4f}  "
                  f"max={sub[f'{m}_RMSE'].max():.4f}")

    print("\nIndividual vs Cluster win rate:")
    plot_win_rate(results_df, benchmarks)
    plot_rmse_scatter(results_df, benchmarks)
    plot_rmse_boxplot(results_df)
    print(f"\nPlots saved to: {PLOT_DIR}")
