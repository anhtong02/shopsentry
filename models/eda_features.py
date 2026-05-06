import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from models.data_loader import load_training_data

sns.set_style("whitegrid")
df, feature_cols = load_training_data()
print(f"Total sessions: {len(df)}")
print(f"\nClass counts:\n{df['agent_type'].value_counts()}\n")

# ============================================================
# 1. DISTRIBUTION OVERLAP — per-feature histograms by class
# ============================================================
# This tells you: for any single feature, can you draw a vertical line
# that separates the classes? If histograms overlap heavily, IF can't help.


fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, feat in zip(axes.flat, feature_cols):
    for agent in df["agent_type"].unique():
        vals = df[df["agent_type"] == agent][feat]
        # log scale for skewed features (events_per_minute etc)
        if vals.max() / max(vals.min() + 0.001, 0.001) > 100:
            vals = np.log1p(vals)
            ax.set_xlabel(f"log1p({feat})")
        else:
            ax.set_xlabel(feat)
        ax.hist(vals, bins=30, alpha=0.4, label=agent, density=True)
    ax.legend(fontsize=7)
    ax.set_ylabel("density")
plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=100)
print("Saved eda_distributions.png")

# ============================================================
# 2. CORRELATION HEATMAP — which features are redundant?
# ============================================================
# If two features are 0.95 correlated, you really only have one.
# IF and autoencoder both suffer when features are redundant.

plt.figure(figsize=(10, 8))
corr = df[feature_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, cbar_kws={"shrink": 0.8})
plt.title("Feature correlation")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=100)
print("Saved eda_correlation.png")

# ============================================================
# 3. PAIR PLOT — 2D separability for top features
# ============================================================
# Pick 4 features and look at all 2D combinations.
# You're asking: "is there ANY 2D plane where fraud separates from normal?"
top_feats = ["events_per_minute", "session_duration_seconds",
             "cart_to_purchase_ratio", "event_type_diversity"]
sample = df.sample(min(2000, len(df)), random_state=1)  # sample for speed
g = sns.pairplot(sample, vars=top_feats, hue="agent_type",
                 plot_kws={"alpha": 0.5, "s": 15}, height=2.5)
g.fig.suptitle("Pairwise feature view by agent type", y=1.01)
plt.savefig("eda_pairplot.png", dpi=100)
print("Saved eda_pairplot.png")
 
# ============================================================
# 4. PCA — collapse 9D → 2D, see if classes cluster
# ============================================================
# This is the BIG one. PCA finds the 2 directions of maximum variance.
# If fraud forms a tight cluster separate from normal in PC space,
# IF should catch it. If they overlap, no model using only these
# features will do well — you need new features.
X_scaled = StandardScaler().fit_transform(df[feature_cols])
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
df["pc1"] = pcs[:, 0]
df["pc2"] = pcs[:, 1]
 
plt.figure(figsize=(10, 7))
for agent, color in zip(["normal", "churning", "bot", "fraud"],
                         ["steelblue", "gray", "orange", "red"]):
    sub = df[df["agent_type"] == agent]
    plt.scatter(sub["pc1"], sub["pc2"], alpha=0.5, s=20,
                label=f"{agent} (n={len(sub)})", color=color)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.title("PCA — can we see the classes separate?")
plt.legend()
plt.tight_layout()
plt.savefig("eda_pca.png", dpi=100)
print("Saved eda_pca.png")
print(f"PC1 + PC2 explain {sum(pca.explained_variance_ratio_):.1%} of variance")
 
# Which features drive each PC? (this tells you what each axis means)
loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"],
                        index=feature_cols)
print("\nPCA loadings (which features drive each axis):")
print(loadings.round(2))
 
# ============================================================
# 5. t-SNE — non-linear projection (often clearer than PCA)
# ============================================================
# PCA is linear; t-SNE preserves local neighborhoods. If fraud forms
# a non-linear cluster, t-SNE shows it where PCA wouldn't.
print("\nRunning t-SNE (this takes ~30s)...")
sample_idx = df.sample(min(2000, len(df)), random_state=1).index
tsne = TSNE(n_components=2, random_state=1, perplexity=30)
tsne_pts = tsne.fit_transform(X_scaled[sample_idx])
 
plt.figure(figsize=(10, 7))
sample_df = df.loc[sample_idx]
for agent, color in zip(["normal", "churning", "bot", "fraud"],
                         ["steelblue", "gray", "orange", "red"]):
    mask = sample_df["agent_type"].values == agent
    plt.scatter(tsne_pts[mask, 0], tsne_pts[mask, 1], alpha=0.6, s=20,
                label=f"{agent}", color=color)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE — non-linear class separation")
plt.legend()
plt.tight_layout()
plt.savefig("eda_tsne.png", dpi=100)
print("Saved eda_tsne.png")
 
# ============================================================
# 6. CLASS-CONDITIONAL STATS — quantify what the eye sees
# ============================================================
print("\n=== Mean (std) per feature by class ===")
agg = df.groupby("agent_type")[feature_cols].agg(["mean", "std"]).round(2)
print(agg.T)
 
print("\nDone. Open the 5 PNGs in order: distributions, correlation, pairplot, pca, tsne")
 