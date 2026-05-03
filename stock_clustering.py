import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_DIR = r"C:\Users\Ayush\Downloads\DATA3888\data3888 finc6\Optiver_extracted\individual_book_train"
PLOT_DIR = r"C:\Users\Ayush\Downloads\data3888_volatility_modelling\plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def compute_wap(df):
    return (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )


def compute_stock_summary(stock_id):
    """Compute summary features for a single stock across all time_ids."""
    path = os.path.join(DATA_DIR, f"stock_{stock_id}.parquet")
    df = pd.read_parquet(path)

    df["wap"] = compute_wap(df)
    df["spread"] = df["ask_price1"] - df["bid_price1"]
    df["log_return"] = df.groupby("time_id")["wap"].transform(lambda x: np.log(x).diff())

    # Per time_id: realised volatility
    rv_per_bucket = df.groupby("time_id").apply(
        lambda g: np.sqrt((g["log_return"].dropna() ** 2).sum())
    )

    summary = {
        "stock_id": stock_id,
        "mean_rv":        rv_per_bucket.mean(),
        "std_rv":         rv_per_bucket.std(),       # vol-of-vol
        "skew_rv":        rv_per_bucket.skew(),
        "mean_spread":    df["spread"].mean(),
        "std_spread":     df["spread"].std(),
        "mean_bid_size1": df["bid_size1"].mean(),
        "mean_ask_size1": df["ask_size1"].mean(),
        "n_buckets":      rv_per_bucket.count(),
    }
    return summary


# ── Step 1: Compute features for all stocks ──────────────────────────────────
print("Computing summary features for all stocks...")
records = []
parquet_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
stock_ids = sorted([int(f.replace("stock_", "").replace(".parquet", "")) for f in parquet_files])

for i, sid in enumerate(stock_ids):
    summary = compute_stock_summary(sid)
    records.append(summary)
    print(f"  [{i+1}/{len(stock_ids)}] stock_{sid} done — mean_rv={summary['mean_rv']:.6f}")

summary_df = pd.DataFrame(records).set_index("stock_id")
summary_df.to_csv(os.path.join(PLOT_DIR, "stock_summary_features.csv"))
print(f"\nSummary features saved. Shape: {summary_df.shape}")
print(summary_df.describe())


# ── Step 2: Scale features ────────────────────────────────────────────────────
feature_cols = ["mean_rv", "std_rv", "skew_rv", "mean_spread", "std_spread",
                "mean_bid_size1", "mean_ask_size1"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(summary_df[feature_cols])


# ── Step 3: Hierarchical Clustering + Dendrogram ─────────────────────────────
print("\nRunning hierarchical clustering...")
Z = linkage(X_scaled, method="ward")

fig, ax = plt.subplots(figsize=(20, 6))
dendrogram(Z, labels=summary_df.index.astype(str).tolist(), leaf_rotation=90, leaf_font_size=7, ax=ax)
ax.set_title("Hierarchical Clustering of Stocks (Ward linkage)", fontsize=13, fontweight="bold")
ax.set_xlabel("Stock ID")
ax.set_ylabel("Distance")
ax.axhline(y=8, color="red", linestyle="--", linewidth=1, label="Cut line (n=5 clusters)")
ax.legend()
plt.tight_layout()
path = os.path.join(PLOT_DIR, "dendrogram.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Dendrogram saved: {path}")


# ── Step 4: Assign cluster labels (cut at 5 clusters) ─────────────────────────
N_CLUSTERS = 5
cluster_labels = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")
summary_df["cluster"] = cluster_labels
print(f"\nCluster distribution (n={N_CLUSTERS}):")
print(summary_df["cluster"].value_counts().sort_index())


# ── Step 5: Visualise clusters (mean_rv vs mean_spread) ──────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.tab10.colors
for c in sorted(summary_df["cluster"].unique()):
    mask = summary_df["cluster"] == c
    ax.scatter(
        summary_df.loc[mask, "mean_spread"],
        summary_df.loc[mask, "mean_rv"],
        label=f"Cluster {c} (n={mask.sum()})",
        s=60, alpha=0.8, color=colors[c - 1]
    )
    # annotate stock ids
    for sid in summary_df[mask].index:
        ax.annotate(str(sid), (summary_df.loc[sid, "mean_spread"],
                               summary_df.loc[sid, "mean_rv"]),
                    fontsize=6, alpha=0.6)

ax.set_xlabel("Mean Bid-Ask Spread")
ax.set_ylabel("Mean Realised Volatility")
ax.set_title("Stock Clusters: Mean RV vs Mean Spread", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
path2 = os.path.join(PLOT_DIR, "cluster_scatter.png")
plt.savefig(path2, dpi=150)
plt.close()
print(f"Cluster scatter saved: {path2}")


# ── Step 6: Save cluster assignments ─────────────────────────────────────────
cluster_path = os.path.join(PLOT_DIR, "stock_cluster_assignments.csv")
summary_df[["cluster"]].to_csv(cluster_path)
print(f"Cluster assignments saved: {cluster_path}")
print("\nDone.")
