"""
recluster.py
Cluster all stocks by behavioural features using Ward hierarchical clustering.
Tests k=2–8, selects k=2 (best silhouette), and saves final assignments.
Must be run before any model training script.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from config import DATA_DIR, PLOTS_DIR

PLOT_DIR = os.path.join(PLOTS_DIR, "recluster")
os.makedirs(PLOT_DIR, exist_ok=True)

CLUSTER_FEATURES = [
    "mean_rv", "std_rv", "skew_rv", "kurt_rv", "rv_autocorr",
    "p90_rv", "mean_rel_spread", "std_rel_spread", "mean_imbalance",
]

FINAL_K = 2  # validated as best by silhouette — change here if re-evaluating


# ── Feature extraction ────────────────────────────────────────────────────────

def compute_wap(df):
    return (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )


def stock_behavioural_features(stock_id):
    path = os.path.join(DATA_DIR, f"stock_{stock_id}.parquet")
    df   = pd.read_parquet(path)

    df["wap"]        = compute_wap(df)
    df["spread"]     = df["ask_price1"] - df["bid_price1"]
    df["mid_price"]  = (df["bid_price1"] + df["ask_price1"]) / 2
    df["rel_spread"] = df["spread"] / df["mid_price"]
    df["depth"]      = df["bid_size1"] + df["ask_size1"]
    df["imbalance"]  = (df["bid_size1"] - df["ask_size1"]) / df["depth"]
    df["log_return"] = df.groupby("time_id")["wap"].transform(lambda x: np.log(x).diff())

    rv = df.groupby("time_id")["log_return"].apply(
        lambda x: np.sqrt((x.dropna() ** 2).sum())
    )
    rv = rv[rv > 0]

    return {
        "stock_id":        stock_id,
        "mean_rv":         rv.mean(),
        "std_rv":          rv.std(),
        "skew_rv":         rv.skew(),
        "kurt_rv":         rv.kurt(),
        "rv_autocorr":     rv.autocorr(lag=1) if len(rv) > 10 else 0,
        "p90_rv":          rv.quantile(0.90),
        "mean_rel_spread": df["rel_spread"].mean(),
        "std_rel_spread":  df["rel_spread"].std(),
        "mean_imbalance":  df["imbalance"].abs().mean(),
    }


# ── Load all stocks ───────────────────────────────────────────────────────────

print("Computing behavioural features...")
stock_ids = sorted([
    int(f.replace("stock_", "").replace(".parquet", ""))
    for f in os.listdir(DATA_DIR) if f.endswith(".parquet")
])

records = []
for i, sid in enumerate(stock_ids):
    feat = stock_behavioural_features(sid)
    records.append(feat)
    print(f"  [{i+1}/{len(stock_ids)}] stock_{sid}  "
          f"mean_rv={feat['mean_rv']:.6f}  "
          f"rel_spread={feat['mean_rel_spread']:.6f}")

feat_df = pd.DataFrame(records).set_index("stock_id")
feat_df.to_csv(os.path.join(PLOT_DIR, "stock_behavioural_features.csv"))

feat_clean = feat_df[CLUSTER_FEATURES].dropna()
print(f"\nStocks: {len(feat_clean)}")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(feat_clean.values)


# ── Find optimal k ────────────────────────────────────────────────────────────

print("\nEvaluating k=2..8:")
k_range, inertias, sil_scores = range(2, 9), [], []

for k in k_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=20)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, lbl)
    sil_scores.append(sil)
    print(f"  k={k}: inertia={km.inertia_:.1f}  silhouette={sil:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Optimal Number of Clusters", fontweight="bold")
axes[0].plot(list(k_range), inertias, "bo-", linewidth=2)
axes[0].set_title("Elbow (KMeans Inertia)")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
axes[0].set_xticks(list(k_range))
axes[1].plot(list(k_range), sil_scores, "rs-", linewidth=2)
axes[1].axhline(0.5,  color="green",  linestyle="--", linewidth=1, label="Strong (0.5)")
axes[1].axhline(0.25, color="orange", linestyle="--", linewidth=1, label="Reasonable (0.25)")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
axes[1].set_xticks(list(k_range))
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "optimal_k.png"), dpi=150, bbox_inches="tight")
plt.close()


# ── Final clustering (FINAL_K) ────────────────────────────────────────────────

print(f"\nFinal clustering: k={FINAL_K}")
Z      = linkage(X_scaled, method="ward")
labels = fcluster(Z, t=FINAL_K, criterion="maxclust")

feat_clean = feat_clean.copy()
feat_clean["cluster"] = labels

print(feat_clean["cluster"].value_counts().sort_index().to_string())
print("\nCluster means:")
print(feat_clean.groupby("cluster")[["mean_rv", "mean_rel_spread", "rv_autocorr"]].mean().round(6))

final_sil = silhouette_score(X_scaled, labels)
print(f"\nSilhouette (k={FINAL_K}): {final_sil:.4f}")


# ── Dendrogram ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(18, 5))
dendrogram(Z, labels=feat_clean.index.astype(str).tolist(),
           leaf_rotation=90, leaf_font_size=7, ax=ax)
ax.set_title(f"Hierarchical Clustering Dendrogram  (k={FINAL_K}, Ward linkage)",
             fontweight="bold")
ax.set_xlabel("Stock ID"); ax.set_ylabel("Distance")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"dendrogram_k{FINAL_K}.png"), dpi=150, bbox_inches="tight")
plt.close()


# ── Scatter: mean RV vs relative spread ──────────────────────────────────────

colors = plt.cm.tab10.colors
fig, ax = plt.subplots(figsize=(10, 6))
for c in sorted(feat_clean["cluster"].unique()):
    mask = feat_clean["cluster"] == c
    ax.scatter(feat_clean.loc[mask, "mean_rel_spread"],
               feat_clean.loc[mask, "mean_rv"],
               label=f"Cluster {c} (n={mask.sum()})",
               s=65, alpha=0.85, color=colors[c - 1])
    for sid in feat_clean[mask].index:
        ax.annotate(str(sid),
                    (feat_clean.loc[sid, "mean_rel_spread"],
                     feat_clean.loc[sid, "mean_rv"]),
                    fontsize=6, alpha=0.6)
ax.set_xlabel("Mean Relative Spread")
ax.set_ylabel("Mean Realised Volatility")
ax.set_title(f"Stock Clusters (k={FINAL_K}): Mean RV vs Relative Spread",
             fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"scatter_k{FINAL_K}.png"), dpi=150, bbox_inches="tight")
plt.close()


# ── Save final assignments ────────────────────────────────────────────────────

out = feat_clean[["cluster"]].reset_index()
out.columns = ["stock_id", "cluster"]
out.to_csv(os.path.join(PLOT_DIR, "stock_cluster_assignments_FINAL.csv"), index=False)
print(f"\nAssignments saved: {PLOT_DIR}/stock_cluster_assignments_FINAL.csv")
print("Done.")
