"""
recluster_v2.py
===============
Two improved clustering strategies beyond the current Ward hierarchical k=2:

Strategy A — Correlation-based clustering
  Build a pairwise RV correlation matrix across all stocks.
  Distance = 1 - |correlation|.
  Apply hierarchical clustering on this distance matrix.
  RATIONALE: stocks whose volatility moves together are more likely to
  benefit from a shared predictive model.

Strategy B — Gaussian Mixture Model (GMM) on behavioural features
  GMM is a soft/probabilistic version of k-means — it fits elliptical
  Gaussian components and allows different cluster shapes and sizes.
  Use BIC to pick the optimal k (avoids over-fitting to k).

Both are compared against the existing Ward k=2 via silhouette score.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from config import DATA_DIR, PLOTS_DIR, CLUSTER_CSV

PLOT_DIR = os.path.join(PLOTS_DIR, "recluster_v2")
os.makedirs(PLOT_DIR, exist_ok=True)


# ── Step 1: Compute per-stock bucket RV series ────────────────────────────────
def compute_wap(df):
    return (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )


print("Loading RV series for all stocks...")
old_clusters = pd.read_csv(CLUSTER_CSV)
old_clusters.columns = ["stock_id", "cluster"]
stock_ids = sorted(old_clusters["stock_id"].tolist())

rv_dict = {}       # stock_id -> pd.Series of RV values indexed by time_id
feat_dict = {}     # stock_id -> behavioural feature vector (for GMM)

for sid in stock_ids:
    path = os.path.join(DATA_DIR, f"stock_{sid}.parquet")
    try:
        df   = pd.read_parquet(path)
        df["wap"] = compute_wap(df)
        df["log_return"] = df.groupby("time_id")["wap"].transform(
            lambda x: np.log(x).diff()
        )
        # Full-bucket RV (all 10 mins) for correlation clustering
        rv_full = df.groupby("time_id")["log_return"].apply(
            lambda x: np.sqrt((x.dropna()**2).sum())
        )
        rv_dict[sid] = rv_full

        # Behavioural features for GMM (same 9 as original recluster.py)
        spread     = df["ask_price1"] - df["bid_price1"]
        mid        = (df["ask_price1"] + df["bid_price1"]) / 2
        rel_spread = spread / mid
        imbalance  = (df["bid_size1"] - df["ask_size1"]) / (df["bid_size1"] + df["ask_size1"])

        rv_s = rv_full[rv_full > 0]
        feat_dict[sid] = {
            "mean_rv":         rv_s.mean(),
            "std_rv":          rv_s.std(),
            "skew_rv":         rv_s.skew(),
            "kurt_rv":         rv_s.kurt(),
            "rv_autocorr":     rv_s.autocorr(lag=1) if len(rv_s) > 10 else 0,
            "p90_rv":          rv_s.quantile(0.90),
            "mean_rel_spread": rel_spread.mean(),
            "std_rel_spread":  rel_spread.std(),
            "mean_imbalance":  imbalance.abs().mean(),
        }
        print(f"  Loaded stock_{sid}")
    except FileNotFoundError:
        print(f"  stock_{sid}: not found, skipping")

print(f"\nLoaded {len(rv_dict)} stocks.\n")


# ── Step 2: Build pairwise correlation matrix ─────────────────────────────────
# Align all RV series to common time_ids
all_time_ids = sorted(set.intersection(*[set(s.index) for s in rv_dict.values()]))
print(f"Common time_ids across all stocks: {len(all_time_ids)}")

rv_matrix = pd.DataFrame(
    {sid: rv_dict[sid].reindex(all_time_ids) for sid in stock_ids}
).dropna(axis=1, how="any")

valid_stocks = list(rv_matrix.columns)
print(f"Stocks with complete RV series: {len(valid_stocks)}")

corr_matrix = rv_matrix.corr()
dist_matrix = 1 - corr_matrix.abs()
np.fill_diagonal(dist_matrix.values, 0)
dist_matrix = np.maximum(dist_matrix, 0)  # numerical safety


# ── Strategy A: Correlation-based hierarchical clustering ────────────────────
print("\n" + "="*60)
print("STRATEGY A: Correlation-based Hierarchical Clustering")
print("="*60)

condensed = squareform(dist_matrix.values, checks=False)
Z_corr    = linkage(condensed, method="ward")

# Test k=2,3,4,5
corr_results = {}
for k in range(2, 6):
    labels = fcluster(Z_corr, k, criterion="maxclust")
    sil    = silhouette_score(dist_matrix.values, labels, metric="precomputed")
    corr_results[k] = {"labels": labels, "silhouette": sil}
    counts = pd.Series(labels).value_counts().sort_index().to_dict()
    print(f"  k={k}: silhouette={sil:.4f}  sizes={counts}")

best_k_corr = max(corr_results, key=lambda k: corr_results[k]["silhouette"])
print(f"\nBest k (correlation): {best_k_corr}  "
      f"(silhouette={corr_results[best_k_corr]['silhouette']:.4f})")

# Save correlation-based assignments
corr_labels = pd.DataFrame({
    "stock_id": valid_stocks,
    "cluster_corr": corr_results[best_k_corr]["labels"]
})


# ── Strategy B: GMM on behavioural features ───────────────────────────────────
print("\n" + "="*60)
print("STRATEGY B: Gaussian Mixture Model (BIC selection)")
print("="*60)

feat_df = pd.DataFrame(feat_dict).T
feat_df.index = feat_df.index.astype(int)
feat_df = feat_df.loc[feat_df.index.isin(valid_stocks)].dropna()

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(feat_df.values)

gmm_results = {}
for k in range(2, 7):
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          random_state=42, n_init=10)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    bic    = gmm.bic(X_scaled)
    sil    = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
    gmm_results[k] = {"labels": labels + 1, "bic": bic, "silhouette": sil, "model": gmm}
    counts = pd.Series(labels + 1).value_counts().sort_index().to_dict()
    print(f"  k={k}: BIC={bic:.1f}  silhouette={sil:.4f}  sizes={counts}")

best_k_gmm = min(gmm_results, key=lambda k: gmm_results[k]["bic"])  # lower BIC = better
print(f"\nBest k (GMM, BIC): {best_k_gmm}  "
      f"(BIC={gmm_results[best_k_gmm]['bic']:.1f}  "
      f"silhouette={gmm_results[best_k_gmm]['silhouette']:.4f})")

gmm_labels = pd.DataFrame({
    "stock_id": feat_df.index.tolist(),
    "cluster_gmm": gmm_results[best_k_gmm]["labels"]
})


# ── Compare all three approaches ───────────────────────────────────────────────
print("\n" + "="*60)
print("COMPARISON: Ward k=2 vs Correlation vs GMM")
print("="*60)

# Original Ward silhouette
feat_all = pd.DataFrame(feat_dict).T.dropna()
feat_all.index = feat_all.index.astype(int)
old_sub  = old_clusters[old_clusters["stock_id"].isin(feat_all.index)]
feat_aligned = feat_all.loc[old_sub["stock_id"].values]
X_all    = StandardScaler().fit_transform(feat_aligned.values)
ward_sil = silhouette_score(X_all, old_sub["cluster"].values)

print(f"\n  Ward k=2        : silhouette={ward_sil:.4f}")
print(f"  Correlation k={best_k_corr}  : silhouette={corr_results[best_k_corr]['silhouette']:.4f}")
print(f"  GMM k={best_k_gmm}          : silhouette={gmm_results[best_k_gmm]['silhouette']:.4f}")


# ── Save new cluster assignments ───────────────────────────────────────────────
combined = old_clusters.copy()
combined = combined.merge(
    corr_labels.rename(columns={"stock_id": "stock_id"}),
    on="stock_id", how="left"
)
combined = combined.merge(gmm_labels, on="stock_id", how="left")
combined.to_csv(os.path.join(PLOT_DIR, "cluster_comparison_all_methods.csv"), index=False)
print(f"\nCluster comparison saved.")

# Save best new assignment (whichever wins silhouette)
sil_scores = {
    "ward_k2":        ward_sil,
    f"corr_k{best_k_corr}": corr_results[best_k_corr]["silhouette"],
    f"gmm_k{best_k_gmm}":   gmm_results[best_k_gmm]["silhouette"],
}
winner = max(sil_scores, key=sil_scores.get)
print(f"\nWINNER: {winner}  (silhouette={sil_scores[winner]:.4f})")


# ── Plots ─────────────────────────────────────────────────────────────────────

# Plot 1: Dendrogram for correlation-based clustering
fig, ax = plt.subplots(figsize=(16, 6))
dendrogram(Z_corr, labels=[str(s) for s in valid_stocks],
           leaf_rotation=90, leaf_font_size=6, ax=ax,
           color_threshold=Z_corr[-(best_k_corr-1), 2])
ax.set_title(f"Correlation-based Dendrogram (Ward linkage on RV correlation distance)\n"
             f"Best k={best_k_corr}", fontsize=11, fontweight="bold")
ax.set_xlabel("Stock ID"); ax.set_ylabel("Distance (1 - |corr|)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "corr_dendrogram.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Dendrogram saved.")


# Plot 2: BIC curve for GMM
fig, ax = plt.subplots(figsize=(8, 4))
ks   = list(gmm_results.keys())
bics = [gmm_results[k]["bic"] for k in ks]
sils = [gmm_results[k]["silhouette"] for k in ks]
ax.plot(ks, bics, "o-", color="#2196F3", linewidth=2, label="BIC (lower=better)")
ax2 = ax.twinx()
ax2.plot(ks, sils, "s--", color="#FF5722", linewidth=2, label="Silhouette (higher=better)")
ax.axvline(best_k_gmm, color="green", linestyle=":", linewidth=2, label=f"Best k={best_k_gmm}")
ax.set_xlabel("Number of clusters k")
ax.set_ylabel("BIC", color="#2196F3")
ax2.set_ylabel("Silhouette", color="#FF5722")
ax.set_title("GMM Model Selection (BIC + Silhouette)", fontsize=11, fontweight="bold")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "gmm_bic_curve.png"), dpi=150, bbox_inches="tight")
plt.close()
print("BIC curve saved.")


# Plot 3: PCA scatter — compare Ward vs GMM vs Correlation labels
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("PCA of Behavioural Features — Clustering Method Comparison",
             fontsize=12, fontweight="bold")

pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_all)
ev    = pca.explained_variance_ratio_

label_sets = [
    ("Ward k=2 (current)",         old_sub["cluster"].values,              ward_sil),
    (f"Correlation k={best_k_corr}", corr_results[best_k_corr]["labels"],   corr_results[best_k_corr]["silhouette"]),
    (f"GMM k={best_k_gmm}",         gmm_results[best_k_gmm]["labels"],      gmm_results[best_k_gmm]["silhouette"]),
]

# For correlation labels, need to align to same stocks as feat_all
corr_label_series = corr_labels.set_index("stock_id")["cluster_corr"]
gmm_label_series  = gmm_labels.set_index("stock_id")["cluster_gmm"]

stock_list = old_sub["stock_id"].values

corr_aligned = np.array([
    corr_label_series.get(s, 0) for s in stock_list
])
gmm_aligned = np.array([
    gmm_label_series.get(s, 0) for s in stock_list
])

label_sets = [
    ("Ward k=2 (current)",          old_sub["cluster"].values, ward_sil),
    (f"Correlation k={best_k_corr}", corr_aligned,              corr_results[best_k_corr]["silhouette"]),
    (f"GMM k={best_k_gmm}",          gmm_aligned,               gmm_results[best_k_gmm]["silhouette"]),
]

for ax, (title, labels, sil) in zip(axes, label_sets):
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10",
                         s=60, alpha=0.8, edgecolors="white", linewidths=0.4)
    ax.set_title(f"{title}\nSilhouette={sil:.4f}", fontsize=10)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    plt.colorbar(scatter, ax=ax, label="Cluster")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "clustering_pca_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("PCA comparison plot saved.")


# Plot 4: Silhouette comparison bar chart
fig, ax = plt.subplots(figsize=(8, 4))
methods = list(sil_scores.keys())
scores  = list(sil_scores.values())
colors  = ["#2196F3" if m == winner else "#90CAF9" for m in methods]
bars    = ax.bar(methods, scores, color=colors, alpha=0.85)
for bar, v in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Silhouette Score")
ax.set_title("Clustering Method Comparison — Silhouette Scores\n(Higher is better)",
             fontsize=11, fontweight="bold")
ax.set_ylim(0, max(scores) * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "silhouette_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Silhouette comparison saved.")

print(f"\nAll outputs saved to: {PLOT_DIR}")
print("\nDone.")
