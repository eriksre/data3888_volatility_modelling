"""
cluster_validation.py
Validates the k=2 Ward clustering through 4 checks:
  1. Average RV time series per cluster
  2. PCA scatter coloured by cluster
  3. Silhouette score vs alternative k values
  4. Between-cluster vs within-cluster variance ratio
Run after recluster.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from config import DATA_DIR, PLOTS_DIR, CLUSTER_CSV

PLOT_DIR = os.path.join(PLOTS_DIR, "cluster_validation")
os.makedirs(PLOT_DIR, exist_ok=True)

FEAT_CSV = os.path.join(PLOTS_DIR, "recluster", "stock_behavioural_features.csv")

CLUSTER_FEATURES = [
    "mean_rv", "std_rv", "skew_rv", "kurt_rv", "rv_autocorr",
    "p90_rv", "mean_rel_spread", "std_rel_spread", "mean_imbalance",
]


# ── Load data ─────────────────────────────────────────────────────────────────

cluster_df = pd.read_csv(CLUSTER_CSV)
cluster_df.columns = ["stock_id", "cluster"]

feat_df = pd.read_csv(FEAT_CSV, index_col="stock_id")
feat_df = feat_df[CLUSTER_FEATURES].dropna()
feat_df = feat_df.loc[feat_df.index.isin(cluster_df["stock_id"])]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(feat_df.values)
labels   = cluster_df.set_index("stock_id").loc[feat_df.index, "cluster"].values

COLORS   = {1: "#2196F3", 2: "#FF5722"}
C_LABELS = {1: "Cluster 1 (High Vol)", 2: "Cluster 2 (Low Vol)"}


def compute_wap(df):
    return (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )


# ── Validation 1: Average RV per cluster over time ───────────────────────────

print("Validation 1: Average RV per cluster...")

rv_series = {}
for sid in cluster_df["stock_id"]:
    path = os.path.join(DATA_DIR, f"stock_{sid}.csv")
    df   = pd.read_csv(path)
    df["wap"] = compute_wap(df)
    df["log_return"] = df.groupby("time_id")["wap"].transform(lambda x: np.log(x).diff())
    rv = df.groupby("time_id")["log_return"].apply(
        lambda x: np.sqrt((x.dropna() ** 2).sum())
    )
    rv_series[sid] = rv
    print(f"  Loaded stock_{sid}")

common_tids = sorted(set.intersection(*[set(s.index) for s in rv_series.values()]))
rv_matrix   = pd.DataFrame({sid: rv_series[sid].reindex(common_tids)
                             for sid in rv_series}).dropna(axis=1)

fig, ax = plt.subplots(figsize=(14, 4))
for cid in sorted(cluster_df["cluster"].unique()):
    stocks = cluster_df[cluster_df["cluster"] == cid]["stock_id"].tolist()
    stocks = [s for s in stocks if s in rv_matrix.columns]
    avg    = rv_matrix[stocks].mean(axis=1)
    ax.plot(range(len(avg)), avg, color=COLORS[cid],
            linewidth=1.2, alpha=0.9, label=C_LABELS[cid])
ax.set_title("Average RV per Cluster over Time", fontweight="bold")
ax.set_xlabel("Time bucket index"); ax.set_ylabel("Mean RV")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "rv_timeseries_per_cluster.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved.\n")


# ── Validation 2: PCA scatter ─────────────────────────────────────────────────

print("Validation 2: PCA scatter...")
pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
ev    = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(9, 6))
for cid in sorted(np.unique(labels)):
    mask = labels == cid
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               label=C_LABELS[cid], color=COLORS[cid],
               s=60, alpha=0.85, edgecolors="white", linewidths=0.4)
    for j, sid in enumerate(feat_df.index[mask]):
        ax.annotate(str(sid), (X_pca[mask][j, 0], X_pca[mask][j, 1]),
                    fontsize=6, alpha=0.5)
ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
ax.set_title("PCA of Behavioural Features — Cluster Assignments",
             fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "pca_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved.\n")


# ── Validation 3: Silhouette scores k=2..6 ───────────────────────────────────

print("Validation 3: Silhouette scores for k=2..6...")
Z = linkage(X_scaled, method="ward")

k_range, sil_scores = range(2, 7), []
for k in k_range:
    lbl = fcluster(Z, t=k, criterion="maxclust")
    sil = silhouette_score(X_scaled, lbl)
    sil_scores.append(sil)
    marker = " <-- current" if k == 2 else ""
    print(f"  k={k}: silhouette={sil:.4f}{marker}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(k_range), sil_scores, "o-", color="#2196F3", linewidth=2)
ax.axvline(2, color="red", linestyle="--", linewidth=1.5, label="Current k=2")
ax.axhline(0.25, color="orange", linestyle=":", linewidth=1, label="Reasonable (0.25)")
ax.set_xlabel("k"); ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Score vs Number of Clusters", fontweight="bold")
ax.set_xticks(list(k_range))
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "silhouette_vs_k.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved.\n")


# ── Validation 4: Between vs within cluster variance ─────────────────────────

print("Validation 4: Variance ratio...")
grand_mean = X_scaled.mean(axis=0)
cluster_ids = np.unique(labels)

between_var = sum(
    (labels == c).sum() * np.sum((X_scaled[labels == c].mean(axis=0) - grand_mean) ** 2)
    for c in cluster_ids
)
within_var = sum(
    np.sum((X_scaled[labels == c] - X_scaled[labels == c].mean(axis=0)) ** 2)
    for c in cluster_ids
)
ratio = between_var / within_var
print(f"  Between-cluster variance : {between_var:.2f}")
print(f"  Within-cluster variance  : {within_var:.2f}")
print(f"  Ratio (higher = better)  : {ratio:.2f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Between", "Within"], [between_var, within_var],
       color=["#2196F3", "#FF5722"], alpha=0.85)
ax.set_title(f"Variance Analysis  (ratio = {ratio:.2f})", fontweight="bold")
ax.set_ylabel("Total variance")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "variance_ratio.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved.\n")


# ── Summary ───────────────────────────────────────────────────────────────────

print("=" * 50)
print("CLUSTER VALIDATION SUMMARY")
print("=" * 50)
print(f"Stocks validated : {len(feat_df)}")
print(f"Silhouette (k=2) : {sil_scores[0]:.4f}")
print(f"Variance ratio   : {ratio:.2f}")
print(f"\nCluster sizes:")
for cid in cluster_ids:
    n = (labels == cid).sum()
    mv = feat_df.loc[feat_df.index[labels == cid], "mean_rv"].mean()
    print(f"  Cluster {cid}: {n} stocks  avg mean_rv={mv:.6f}")
print(f"\nPlots saved to: {PLOT_DIR}")
print("Done.")
