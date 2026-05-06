"""
DBSCAN: density-based anomaly detection + visualization.
Run: python -m models.anomaly.dbscan_eval (or wherever you drop it)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from models.data_loader import load_training_data

df, feature_cols = load_training_data()
X = StandardScaler().fit_transform(df[feature_cols])

# ============================================================
# 1. Run DBSCAN
# ============================================================
db = DBSCAN(eps=0.8, min_samples=10).fit(X)
df["cluster"] = db.labels_
df["prediction"] = (db.labels_ == -1).astype(int)  # -1 = noise = anomaly

# ============================================================
# 2. Evaluation (same format as IF for comparison)
# ============================================================
print(classification_report(df["label"], df["prediction"]))

bots = df[df["agent_type"] == "bot"]
fraud = df[df["agent_type"] == "fraud"]
print(f"Bots caught: {(bots['prediction']==1).sum()} / {len(bots)}")
print(f"Fraud caught: {(fraud['prediction']==1).sum()} / {len(fraud)}")

n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_noise = (db.labels_ == -1).sum()
print(f"\nClusters found: {n_clusters}")
print(f"Points labeled noise (anomaly): {n_noise}")

# ============================================================
# 3. Cluster purity — what's in each cluster?
# ============================================================
# This is the interview-gold table: for each cluster DBSCAN found,
# show what agent types ended up in it.
print("\n=== Cluster composition (rows=cluster, cols=agent_type) ===")
purity = pd.crosstab(df["cluster"], df["agent_type"])
print(purity)

# ============================================================
# 4. Project to PCA for visualization
# ============================================================
pca = PCA(n_components=2)
pcs = pca.fit_transform(X)
df["pc1"] = pcs[:, 0]
df["pc2"] = pcs[:, 1]

# ============================================================
# 5. Side-by-side plots: ground truth vs DBSCAN clusters
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# LEFT: ground truth (what we want to find)
truth_colors = {"normal": "steelblue", "churning": "gray",
                "bot": "orange", "fraud": "red"}
for agent, color in truth_colors.items():
    sub = df[df["agent_type"] == agent]
    axes[0].scatter(sub["pc1"], sub["pc2"], alpha=0.5, s=15,
                    label=f"{agent} (n={len(sub)})", color=color)
axes[0].set_title("GROUND TRUTH — agent_type")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
axes[0].legend()

# RIGHT: DBSCAN's predicted clusters (what it found)
unique_clusters = sorted(df["cluster"].unique())
cmap = plt.cm.tab10
for i, c in enumerate(unique_clusters):
    sub = df[df["cluster"] == c]
    if c == -1:
        # Noise = predicted anomaly
        axes[1].scatter(sub["pc1"], sub["pc2"], alpha=0.7, s=20,
                        label=f"noise/anomaly (n={len(sub)})",
                        color="black", marker="x")
    else:
        axes[1].scatter(sub["pc1"], sub["pc2"], alpha=0.5, s=15,
                        label=f"cluster {c} (n={len(sub)})",
                        color=cmap(i % 10))
axes[1].set_title(f"DBSCAN PREDICTIONS — eps=0.5, min_samples=10")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("dbscan_vs_truth.png", dpi=100)
print("\nSaved dbscan_vs_truth.png")

# ============================================================
# 6. Confusion view: where did DBSCAN succeed/fail?
# ============================================================
# Color each point by whether DBSCAN got it right.
fig, ax = plt.subplots(figsize=(10, 7))
df["correct"] = (df["prediction"] == df["label"])

# True positives (caught anomalies) — green
tp = df[(df["prediction"] == 1) & (df["label"] == 1)]
ax.scatter(tp["pc1"], tp["pc2"], alpha=0.6, s=20,
           label=f"True Positive (n={len(tp)})", color="green")

# False negatives (missed anomalies) — red, the ones to worry about
fn = df[(df["prediction"] == 0) & (df["label"] == 1)]
ax.scatter(fn["pc1"], fn["pc2"], alpha=0.8, s=30,
           label=f"False Negative (n={len(fn)}) ← MISSED", color="red", marker="x")

# False positives (flagged normals) — orange
fp = df[(df["prediction"] == 1) & (df["label"] == 0)]
ax.scatter(fp["pc1"], fp["pc2"], alpha=0.5, s=15,
           label=f"False Positive (n={len(fp)})", color="orange")

# True negatives (correct normals) — light gray, in background
tn = df[(df["prediction"] == 0) & (df["label"] == 0)]
ax.scatter(tn["pc1"], tn["pc2"], alpha=0.2, s=10,
           label=f"True Negative (n={len(tn)})", color="lightgray")

ax.set_title("DBSCAN errors in PCA space")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.legend()
plt.tight_layout()
plt.savefig("dbscan_errors.png", dpi=100)
print("Saved dbscan_errors.png")
print("\nLook at the red X's — those are the anomalies DBSCAN missed.")
print("If they cluster in one PCA region, you have a story for why.")