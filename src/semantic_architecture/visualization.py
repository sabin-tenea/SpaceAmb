"""
visualization.py — Plotting utilities for the SpaceAmb prototype.

All functions return matplotlib Figure objects and optionally save to disk.
They are intentionally kept simple — clarity over polish.

Available plots
---------------
plot_heatmap          — similarity matrix as a seaborn heatmap
plot_top_items        — horizontal bar chart of top-ranked atoms/descriptors
plot_pca_projection   — 2-D PCA scatter of embeddings coloured by family
plot_umap_projection  — 2-D UMAP scatter (optional; falls back gracefully)
plot_family_comparison — grouped bar chart comparing families across queries
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


# Consistent colour palette keyed by family name (12-family taxonomy).
# Colours chosen for visual distinctness across the full set.
FAMILY_COLORS: Dict[str, str] = {
    # Physical/object families
    "architecture": "#4e79a7",   # steel blue
    "furniture":    "#9c755f",   # warm brown
    "fixture":      "#bab0ac",   # warm grey
    "decoration":   "#d4a6c8",   # mauve
    "technology":   "#17becf",   # cyan
    # Descriptive families
    "material":     "#f28e2b",   # orange
    "color":        "#e15759",   # red
    "quality":      "#76b7b2",   # teal
    "lighting":     "#59a14f",   # green
    # Spatial/relational/behavioural families
    "spatial":      "#edc948",   # yellow
    "relation":     "#b07aa1",   # purple
    "behavioral":   "#ff9da7",   # pink
}

DEFAULT_FIGSIZE = (12, 7)


def _save_fig(fig: plt.Figure, save_path: Optional[str | Path]) -> None:
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"[viz] Saved figure → {p}")


# ------------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------------

def plot_heatmap(
    df: pd.DataFrame,
    title: str = "Similarity Matrix",
    figsize: tuple = (14, 8),
    save_path: Optional[str | Path] = None,
    cmap: str = "RdYlGn",
    vmin: float = -0.1,
    vmax: float = 0.8,
    annot: bool = True,
) -> plt.Figure:
    """
    Render a labelled similarity matrix as a heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Index = row labels (atoms/descriptors), columns = query labels.
    title : str
    figsize : tuple
    save_path : str or Path or None
    cmap : str
    vmin, vmax : float
        Colour scale limits.  Cosine similarities typically in [-1, 1]
        but the useful range for this data is roughly [0, 0.8].
    annot : bool
        Whether to annotate cells with numeric values.  Set False for
        large matrices.
    """
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    fig, ax = plt.subplots(figsize=figsize)

    if sns is not None:
        sns.heatmap(
            df,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=annot,
            fmt=".2f" if annot else "",
            linewidths=0.3,
            linecolor="white",
        )
    else:
        im = ax.imshow(df.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)
        plt.colorbar(im, ax=ax)

    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ------------------------------------------------------------------
# Top-items bar chart
# ------------------------------------------------------------------

def plot_top_items(
    ranking_df: pd.DataFrame,
    query_label: str,
    top_k: int = 20,
    score_col: str = "weighted_score",
    color_by_family: bool = True,
    family_col: str = "family",
    figsize: tuple = DEFAULT_FIGSIZE,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of top-ranked atoms or descriptors.

    Parameters
    ----------
    ranking_df : pd.DataFrame
        Must contain columns: text, score_col, and optionally family_col.
    query_label : str
        Used in the chart title.
    top_k : int
        Maximum rows to show.
    score_col : str
        Column used for bar lengths and sorting.
    color_by_family : bool
        If True and family_col is present, bars are coloured by family.
    family_col : str
    figsize : tuple
    save_path : str or Path or None
    """
    df = ranking_df.head(top_k).copy()
    # Reverse for horizontal bars (highest at top)
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=figsize)

    if color_by_family and family_col in df.columns:
        colors = [
            FAMILY_COLORS.get(f, "#aaaaaa") for f in df[family_col]
        ]
    else:
        colors = "#4e79a7"

    bars = ax.barh(df["text"], df[score_col], color=colors)

    ax.set_xlabel(score_col.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Top {top_k} items — {query_label}", fontsize=13)
    ax.set_xlim(0, max(df[score_col].max() * 1.15, 0.1))

    # Add value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}", va="center", fontsize=8,
        )

    # Legend for families if applicable
    if color_by_family and family_col in df.columns:
        families = df[family_col].unique()
        handles = [
            plt.Rectangle(
                (0, 0), 1, 1,
                color=FAMILY_COLORS.get(f, "#aaaaaa"),
                label=f,
            )
            for f in sorted(families)
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=8, title="family")

    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ------------------------------------------------------------------
# PCA projection
# ------------------------------------------------------------------

def plot_pca_projection(
    embeddings: np.ndarray,
    labels: List[str],
    families: Optional[List[str]] = None,
    title: str = "PCA Projection",
    figsize: tuple = (10, 8),
    save_path: Optional[str | Path] = None,
    annotate: bool = True,
) -> plt.Figure:
    """
    2-D PCA scatter of embeddings, coloured by family.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n, d)
    labels : list of str
        Text label for each point.
    families : list of str or None
        Family tags for colouring.  If None, all points are blue.
    title : str
    figsize : tuple
    save_path : str or Path or None
    annotate : bool
        If True, add text labels next to each point.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    var_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=figsize)

    if families is not None:
        unique_fams = sorted(set(families))
        cmap_fam = {f: FAMILY_COLORS.get(f, "#aaaaaa") for f in unique_fams}
        for fam in unique_fams:
            idx = [i for i, f in enumerate(families) if f == fam]
            ax.scatter(
                coords[idx, 0], coords[idx, 1],
                c=cmap_fam[fam], label=fam, alpha=0.85, s=60,
            )
        ax.legend(fontsize=8, title="family", loc="best")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=60)

    if annotate:
        for i, txt in enumerate(labels):
            ax.annotate(
                txt, (coords[i, 0], coords[i, 1]),
                fontsize=7, alpha=0.75,
                xytext=(3, 3), textcoords="offset points",
            )

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} var)", fontsize=10)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ------------------------------------------------------------------
# UMAP projection (optional)
# ------------------------------------------------------------------

def plot_umap_projection(
    embeddings: np.ndarray,
    labels: List[str],
    families: Optional[List[str]] = None,
    title: str = "UMAP Projection",
    figsize: tuple = (10, 8),
    save_path: Optional[str | Path] = None,
    annotate: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> Optional[plt.Figure]:
    """
    2-D UMAP scatter.  Returns None if umap-learn is not installed.

    Parameters
    ----------
    Same as plot_pca_projection plus n_neighbors and min_dist.
    """
    try:
        import umap
    except ImportError:
        print("[viz] umap-learn not installed; skipping UMAP plot.")
        return None

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)

    if families is not None:
        unique_fams = sorted(set(families))
        cmap_fam = {f: FAMILY_COLORS.get(f, "#aaaaaa") for f in unique_fams}
        for fam in unique_fams:
            idx = [i for i, f in enumerate(families) if f == fam]
            ax.scatter(
                coords[idx, 0], coords[idx, 1],
                c=cmap_fam[fam], label=fam, alpha=0.85, s=60,
            )
        ax.legend(fontsize=8, title="family", loc="best")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=60)

    if annotate:
        for i, txt in enumerate(labels):
            ax.annotate(
                txt, (coords[i, 0], coords[i, 1]),
                fontsize=7, alpha=0.75,
                xytext=(3, 3), textcoords="offset points",
            )

    ax.set_xlabel("UMAP-1", fontsize=10)
    ax.set_ylabel("UMAP-2", fontsize=10)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ------------------------------------------------------------------
# Family comparison across queries
# ------------------------------------------------------------------

def plot_family_comparison(
    scores_df: pd.DataFrame,
    query_ids: List[str],
    families: Optional[List[str]] = None,
    score_col: str = "weighted_score",
    figsize: tuple = DEFAULT_FIGSIZE,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Grouped bar chart: for each query, show the mean score per family.

    Good for comparing "which atom families are most relevant for each
    space-ambiance pair?"

    Parameters
    ----------
    scores_df : pd.DataFrame
    query_ids : list of str
    families : list of str or None
        If None, all families in the data are used.
    score_col : str
    figsize : tuple
    save_path : str or Path or None
    """
    subset = scores_df[scores_df["query_id"].isin(query_ids)]
    if families is not None:
        subset = subset[subset["family"].isin(families)]

    pivot = (
        subset.groupby(["query_id", "family"])[score_col]
        .mean()
        .unstack(fill_value=0)
    )

    # Replace query_id with combined_text for readable labels
    id_to_label = (
        scores_df[["query_id", "combined_text"]]
        .drop_duplicates()
        .set_index("query_id")["combined_text"]
        .to_dict()
    )
    pivot.index = [id_to_label.get(i, i) for i in pivot.index]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(pivot))
    fam_names = list(pivot.columns)
    n_fams = len(fam_names)
    bar_w = 0.8 / n_fams

    for j, fam in enumerate(fam_names):
        offset = (j - n_fams / 2) * bar_w + bar_w / 2
        ax.bar(
            x + offset,
            pivot[fam],
            width=bar_w,
            color=FAMILY_COLORS.get(fam, "#aaaaaa"),
            label=fam,
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"Mean {score_col.replace('_', ' ')}", fontsize=10)
    ax.set_title("Mean atom score by family across queries", fontsize=13)
    ax.legend(title="family", fontsize=8, loc="upper right")
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig
