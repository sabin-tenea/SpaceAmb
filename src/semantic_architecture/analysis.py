"""
analysis.py — High-level ranking and comparison workflows.

This module sits above the scoring layer and provides the research-facing
queries you actually want to run:

  * Top atoms for a given (space, ambiance) query
  * Top atoms broken down by family
  * Top descriptors for a given query
  * Cross-query comparison for a single atom or descriptor
  * Full similarity matrix (items × queries)

All functions return plain DataFrames so results can be inspected
interactively or saved with io_utils.save_csv.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .scoring import (
    ScoringWeights,
    cosine_similarity_matrix,
    cosine_similarity_vec,
    rank_items,
    score_items_against_queries,
    similarity_matrix_df,
)


# ------------------------------------------------------------------
# Per-query atom ranking
# ------------------------------------------------------------------

def top_atoms_for_query(
    query_id: str,
    scores_df: pd.DataFrame,
    k: int = 20,
    score_col: str = "weighted_score",
) -> pd.DataFrame:
    """
    Return the top-k atoms for a single combined query, sorted by score.

    Parameters
    ----------
    query_id : str
        Must match a value in scores_df["query_id"].
    scores_df : pd.DataFrame
        Output of scoring.score_items_against_queries.
    k : int
    score_col : str
        Column to sort on.  One of: weighted_score, sim_space,
        sim_ambiance, sim_combined.

    Returns
    -------
    pd.DataFrame
        Columns: text, family, sim_space, sim_ambiance, sim_combined,
                 weighted_score, rank
    """
    subset = scores_df[scores_df["query_id"] == query_id].copy()
    subset = subset.sort_values(score_col, ascending=False).head(k)
    subset["rank"] = range(1, len(subset) + 1)
    base_cols = ["rank", "text", "family", "sim_space", "sim_ambiance",
                 "sim_combined", "weighted_score"]
    # Always include the active ranking column so the user can see what drove the order
    extra_cols = ["discriminative_score", "zscore_score"]
    display_cols = base_cols + [c for c in extra_cols if c in subset.columns and c not in base_cols]
    return subset[display_cols].reset_index(drop=True)


def top_atoms_by_family(
    query_id: str,
    scores_df: pd.DataFrame,
    k: int = 10,
    score_col: str = "weighted_score",
) -> Dict[str, pd.DataFrame]:
    """
    Return a dict mapping family name → top-k DataFrame for that family.

    Parameters
    ----------
    query_id : str
    scores_df : pd.DataFrame
    k : int
        Top-k per family.
    score_col : str

    Returns
    -------
    dict of str → pd.DataFrame
    """
    subset = scores_df[scores_df["query_id"] == query_id]
    result: Dict[str, pd.DataFrame] = {}
    for fam, group in subset.groupby("family"):
        top = group.sort_values(score_col, ascending=False).head(k).copy()
        top["rank"] = range(1, len(top) + 1)
        base = ["rank", "text", "sim_space", "sim_ambiance", "sim_combined", "weighted_score"]
        extra = ["discriminative_score", "zscore_score"]
        cols = base + [c for c in extra if c in top.columns]
        result[fam] = top[cols].reset_index(drop=True)
    return result


def top_descriptors_for_query(
    query_id: str,
    desc_scores_df: pd.DataFrame,
    k: int = 20,
    score_col: str = "weighted_score",
) -> pd.DataFrame:
    """
    Return the top-k descriptors for a single combined query.

    Parameters
    ----------
    query_id : str
    desc_scores_df : pd.DataFrame
        Output of score_items_against_queries run on descriptors.
    k : int
    score_col : str

    Returns
    -------
    pd.DataFrame
    """
    subset = desc_scores_df[desc_scores_df["query_id"] == query_id].copy()
    subset = subset.sort_values(score_col, ascending=False).head(k)
    subset["rank"] = range(1, len(subset) + 1)
    cols = ["rank", "text", "family", "sim_space", "sim_ambiance",
            "sim_combined", "weighted_score", "discriminative_score", "zscore_score"]
    available = [c for c in cols if c in subset.columns]
    return subset[available].reset_index(drop=True)


# ------------------------------------------------------------------
# Cross-query comparison
# ------------------------------------------------------------------

def compare_item_across_queries(
    item_text: str,
    scores_df: pd.DataFrame,
    query_ids: Optional[List[str]] = None,
    score_col: str = "weighted_score",
) -> pd.DataFrame:
    """
    Show how a single atom or descriptor scores across multiple queries.

    Useful for questions like: "how does 'sofa' behave across different
    ambiances within the same space?"

    Parameters
    ----------
    item_text : str
        The exact text label of the atom or descriptor.
    scores_df : pd.DataFrame
    query_ids : list of str or None
        If None, all queries in the dataframe are used.
    score_col : str

    Returns
    -------
    pd.DataFrame
        Columns: query_id, combined_text, space_text, ambiance_text,
                 sim_space, sim_ambiance, sim_combined, weighted_score
        Sorted by score_col descending.
    """
    subset = scores_df[scores_df["text"] == item_text].copy()
    if query_ids is not None:
        subset = subset[subset["query_id"].isin(query_ids)]
    subset = subset.sort_values(score_col, ascending=False)
    cols = ["query_id", "combined_text", "space_text", "ambiance_text",
            "sim_space", "sim_ambiance", "sim_combined", "weighted_score",
            "discriminative_score", "zscore_score"]
    available = [c for c in cols if c in subset.columns]
    return subset[available].reset_index(drop=True)


def compare_items_for_query(
    query_id: str,
    item_texts: List[str],
    scores_df: pd.DataFrame,
    score_col: str = "weighted_score",
) -> pd.DataFrame:
    """
    Compare a specific list of items (atoms or descriptors) for one query.

    Parameters
    ----------
    query_id : str
    item_texts : list of str
    scores_df : pd.DataFrame
    score_col : str

    Returns
    -------
    pd.DataFrame sorted by score descending.
    """
    subset = scores_df[
        (scores_df["query_id"] == query_id)
        & (scores_df["text"].isin(item_texts))
    ].copy()
    return subset.sort_values(score_col, ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------
# Similarity matrices
# ------------------------------------------------------------------

def build_atom_query_matrix(
    atom_texts: List[str],
    query_texts: List[str],
    atom_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> pd.DataFrame:
    """
    Build a labelled atoms × queries cosine similarity matrix.

    Returns
    -------
    pd.DataFrame, shape (n_atoms, n_queries)
    """
    return similarity_matrix_df(
        atom_texts, query_texts, atom_embeddings, query_embeddings
    )


def build_descriptor_query_matrix(
    descriptor_texts: List[str],
    query_texts: List[str],
    descriptor_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> pd.DataFrame:
    """Same as build_atom_query_matrix but for descriptors."""
    return similarity_matrix_df(
        descriptor_texts, query_texts, descriptor_embeddings, query_embeddings
    )


def top_scenes_for_query(
    query_id: str,
    scene_scores_df: pd.DataFrame,
    k: int = 20,
    score_col: str = "weighted_score",
) -> pd.DataFrame:
    """
    Return the top-k scene descriptions for a single combined query.

    Parameters
    ----------
    query_id : str
    scene_scores_df : pd.DataFrame
        Output of score_items_against_queries run on scenes.
    k : int
    score_col : str

    Returns
    -------
    pd.DataFrame
        Columns: rank, text (truncated), family (=space), sim_combined,
                 weighted_score, discriminative_score (if present)
    """
    subset = scene_scores_df[scene_scores_df["query_id"] == query_id].copy()
    subset = subset.sort_values(score_col, ascending=False).head(k)
    subset["rank"] = range(1, len(subset) + 1)
    # Truncate long scene texts for display
    subset = subset.copy()
    subset["text"] = subset["text"].str[:80] + "…"
    cols = ["rank", "text", "family", "sim_combined", "weighted_score",
            "discriminative_score", "zscore_score"]
    available = [c for c in cols if c in subset.columns]
    return subset[available].reset_index(drop=True)


def top_queries_for_item(
    item_text: str,
    scores_df: pd.DataFrame,
    k: int = 20,
    score_col: str = "weighted_score",
) -> pd.DataFrame:
    """
    For a given atom or descriptor, return the queries it scores highest on.

    This is the inverse of top_atoms_for_query — useful for "which
    space-ambiance pairs does this descriptor best fit?"
    """
    subset = scores_df[scores_df["text"] == item_text].copy()
    subset = subset.sort_values(score_col, ascending=False).head(k)
    subset["rank"] = range(1, len(subset) + 1)
    cols = ["rank", "combined_text", "space_text", "ambiance_text",
            "sim_space", "sim_ambiance", "sim_combined", "weighted_score"]
    available = [c for c in cols if c in subset.columns]
    return subset[available].reset_index(drop=True)
