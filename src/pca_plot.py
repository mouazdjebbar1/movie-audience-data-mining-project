"""PCA visualisation of the clusters. Owner: Abdallah."""

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _detect_column(df: pd.DataFrame, candidates: Iterable[str], role: str) -> str:
    """Find the first matching column name in a DataFrame."""
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Could not find {role} column among {list(candidates)}. Available: {list(df.columns)}")


def plot_pca(
    profiles_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    max_plot_users: int = 20_000,
    random_state: int = 42,
    output_dir: str | Path = "../figures",
    show: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """2D PCA plot of users coloured by cluster, with explained variance and loadings."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Match user ID and cluster label columns automatically
    prof_user_col = _detect_column(profiles_df, ("userId", "user_id", "user"), "user ID")
    lbl_user_col = _detect_column(labels_df, ("userId", "user_id", "user"), "user ID")
    cluster_col = _detect_column(labels_df, ("cluster", "cluster_label", "label"), "cluster label")

    labels_clean = labels_df[[lbl_user_col, cluster_col]].drop_duplicates(subset=[lbl_user_col]).copy()
    if lbl_user_col != prof_user_col:
        labels_clean = labels_clean.rename(columns={lbl_user_col: prof_user_col})

    merged = profiles_df.merge(labels_clean, on=prof_user_col, how="inner")
    if merged.empty:
        raise ValueError("No overlapping users between profiles_df and labels_df.")

    # 2. Grab numeric feature columns (already scaled by Mouaz—do NOT re-scale)
    non_feature_cols = {prof_user_col, cluster_col}
    feature_cols = [
        c for c in merged.columns
        if c not in non_feature_cols and pd.api.types.is_numeric_dtype(merged[c])
    ]
    X = merged[feature_cols].to_numpy(dtype=float)

    # 3. Fit 2D PCA on all matched users
    pca_2d = PCA(n_components=2, random_state=random_state)
    coords_2d = pca_2d.fit_transform(X)

    pc1_var = pca_2d.explained_variance_ratio_[0] * 100.0
    pc2_var = pca_2d.explained_variance_ratio_[1] * 100.0
    total_2d_var = pc1_var + pc2_var

    print(f"PC1 explained variance: {pc1_var:.2f}%")
    print(f"PC2 explained variance: {pc2_var:.2f}%")
    print(f"Combined PC1 + PC2 explained variance: {total_2d_var:.2f}%")

    # 4. Sample up to 20,000 users for a clean scatter plot
    plot_df = pd.DataFrame(
        {
            "PC1": coords_2d[:, 0],
            "PC2": coords_2d[:, 1],
            "cluster": merged[cluster_col].to_numpy(),
        }
    )
    if len(plot_df) > max_plot_users:
        plot_df = plot_df.sample(n=max_plot_users, random_state=random_state)

    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster_val in sorted(plot_df["cluster"].unique(), key=str):
        sub = plot_df[plot_df["cluster"] == cluster_val]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            label=f"Cluster {cluster_val}",
            alpha=0.55,
            s=14,
            edgecolors="none",
        )

    ax.set_title(f"2D PCA of User Profiles (n={len(plot_df):,} plotted)")
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
    ax.legend(title="Cluster", markerscale=2, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path / "pca_clusters.png", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)

    # 5. Build table of top 5 positive and top 5 negative feature weights for PC1 and PC2
    loadings_rows = []
    for idx, pc_name in enumerate(("PC1", "PC2")):
        weights = pd.Series(pca_2d.components_[idx], index=feature_cols)
        for rank, (feat, val) in enumerate(weights.sort_values(ascending=False).head(5).items(), start=1):
            loadings_rows.append(
                {"component": pc_name, "direction": "top_5_positive", "rank": rank, "feature": feat, "loading": float(val)}
            )
        for rank, (feat, val) in enumerate(weights.sort_values(ascending=True).head(5).items(), start=1):
            loadings_rows.append(
                {"component": pc_name, "direction": "top_5_negative", "rank": rank, "feature": feat, "loading": float(val)}
            )
    loadings_df = pd.DataFrame(loadings_rows)

    # 6. Plot cumulative explained variance across all components
    pca_full = PCA(random_state=random_state)
    pca_full.fit(X)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100.0
    components_idx = np.arange(1, len(cum_var) + 1)

    fig_cum, ax_cum = plt.subplots(figsize=(9, 5))
    ax_cum.plot(components_idx, cum_var, marker="o", markersize=4, linewidth=2)
    ax_cum.axhline(total_2d_var, linestyle="--", color="gray", label=f"2D PCA ({total_2d_var:.1f}%)")
    ax_cum.set_title("Cumulative Explained Variance Across PCA Components")
    ax_cum.set_xlabel("Number of Principal Components")
    ax_cum.set_ylabel("Cumulative Explained Variance (%)")
    ax_cum.set_xticks(components_idx if len(components_idx) <= 25 else np.linspace(1, len(components_idx), 10, dtype=int))
    ax_cum.set_ylim(0, 103)
    ax_cum.legend()
    ax_cum.grid(True, linestyle="--", alpha=0.3)
    fig_cum.tight_layout()
    fig_cum.savefig(out_path / "pca_cumulative_variance.png", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig_cum)

    return loadings_df, plot_df
