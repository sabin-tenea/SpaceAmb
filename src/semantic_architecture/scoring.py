"""
scoring.py — Cosine similarity, ranking, and weighted aggregate scoring.

All embeddings in this system are L2-normalised, so cosine similarity
reduces to a simple dot product.  This module makes that explicit and
builds analysis-friendly DataFrame outputs on top of it.

Weighted scoring formula
------------------------
For an item x and a query defined by (space s, ambiance a):

    score(x | s, a) = w_space * sim(x, s)
                    + w_ambiance * sim(x, a)
                    + w_combined * sim(x, "{a} {s}")

Default weights: w_space=0.25, w_ambiance=0.25, w_combined=0.50.
The combined phrasing carries the most weight because it captures the
joint ambiance-program meaning rather than either component alone.

Weights are configurable so you can experiment, e.g. setting w_combined=1.0
to use only the combined phrase, or equalising all three.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Low-level similarity
# ------------------------------------------------------------------

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarities between rows of A and rows of B.

    Both arrays must be L2-normalised (unit vectors), in which case this
    is simply the dot product.  If they are not normalised the result is
    the true cosine similarity regardless.

    Parameters
    ----------
    A : np.ndarray, shape (n, d)
    B : np.ndarray, shape (m, d)

    Returns
    -------
    np.ndarray, shape (n, m)
        S[i, j] = cosine_similarity(A[i], B[j])
    """
    # Normalise defensively (no-op if already unit vectors)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_norm @ B_norm.T


def cosine_similarity_vec(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between a single vector *v* and each row of *M*.

    Parameters
    ----------
    v : np.ndarray, shape (d,)
    M : np.ndarray, shape (n, d)

    Returns
    -------
    np.ndarray, shape (n,)
    """
    v_norm = v / (np.linalg.norm(v) + 1e-10)
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-10)
    return M_norm @ v_norm


# ------------------------------------------------------------------
# Ranking
# ------------------------------------------------------------------

def rank_items(
    texts: List[str],
    families: List[str],
    item_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Rank items by cosine similarity to a single query embedding.

    Parameters
    ----------
    texts : list of str
        Human-readable labels for each item.
    families : list of str
        Family tag for each item (same length as *texts*).
    item_embeddings : np.ndarray, shape (n, d)
    query_embedding : np.ndarray, shape (d,)
    top_k : int

    Returns
    -------
    pd.DataFrame
        Columns: text, family, sim_score, rank
        Sorted by sim_score descending, limited to top_k rows.
    """
    sims = cosine_similarity_vec(query_embedding, item_embeddings)
    idx = np.argsort(sims)[::-1][:top_k]
    return pd.DataFrame(
        {
            "text": [texts[i] for i in idx],
            "family": [families[i] for i in idx],
            "sim_score": sims[idx].round(4),
            "rank": range(1, len(idx) + 1),
        }
    )


# ------------------------------------------------------------------
# Weights dataclass
# ------------------------------------------------------------------

@dataclass
class ScoringWeights:
    """
    Configurable weights for the three-component scoring formula.

    They need not sum to 1 — they are treated as relative importances.
    The weighted score is a linear combination, not a probability.
    """
    space: float = 0.25
    ambiance: float = 0.25
    combined: float = 0.50

    @classmethod
    def from_config(cls, cfg: dict) -> "ScoringWeights":
        w = cfg.get("weights", {})
        return cls(
            space=w.get("space", 0.25),
            ambiance=w.get("ambiance", 0.25),
            combined=w.get("combined", 0.50),
        )

    def weighted(
        self,
        sim_space: float,
        sim_ambiance: float,
        sim_combined: float,
    ) -> float:
        """Return the weighted aggregate score."""
        return (
            self.space * sim_space
            + self.ambiance * sim_ambiance
            + self.combined * sim_combined
        )


# ------------------------------------------------------------------
# Full scoring table
# ------------------------------------------------------------------

def score_items_against_queries(
    item_texts: List[str],
    item_families: List[str],
    item_ids: List[str],
    item_embeddings: np.ndarray,
    space_queries: List,      # List[Query]
    ambiance_queries: List,   # List[Query]
    combined_queries: List,   # List[Query]
    space_embeddings: np.ndarray,
    ambiance_embeddings: np.ndarray,
    combined_embeddings: np.ndarray,
    weights: ScoringWeights,
) -> pd.DataFrame:
    """
    Compute weighted scores for every item × every combined query.

    For each combined query (ambiance + space pair), the function looks up
    the corresponding space-only and ambiance-only similarity scores and
    computes the weighted aggregate.

    Parameters
    ----------
    item_texts, item_families, item_ids : parallel lists of length n
    item_embeddings : (n, d)
    space_queries, ambiance_queries, combined_queries : lists of Query
    space_embeddings : (n_spaces, d)
    ambiance_embeddings : (n_ambiances, d)
    combined_embeddings : (n_combined, d)
    weights : ScoringWeights

    Returns
    -------
    pd.DataFrame
        One row per (item, combined_query).
        Columns: item_id, text, family, query_id, space_id, ambiance_id,
                 space_text, ambiance_text, combined_text,
                 sim_space, sim_ambiance, sim_combined, weighted_score
    """
    # Build id → index maps for space and ambiance queries
    space_idx = {q.space_id: i for i, q in enumerate(space_queries)}
    ambiance_idx = {q.ambiance_id: i for i, q in enumerate(ambiance_queries)}

    # Similarity matrices: items vs each query set
    # Shape: (n_items, n_spaces / n_ambiances / n_combined)
    sim_space_mat = cosine_similarity_matrix(item_embeddings, space_embeddings)
    sim_amb_mat = cosine_similarity_matrix(item_embeddings, ambiance_embeddings)
    sim_comb_mat = cosine_similarity_matrix(item_embeddings, combined_embeddings)

    rows = []
    for cq_i, cq in enumerate(combined_queries):
        s_i = space_idx[cq.space_id]
        a_i = ambiance_idx[cq.ambiance_id]
        for item_i in range(len(item_texts)):
            ss = float(sim_space_mat[item_i, s_i])
            sa = float(sim_amb_mat[item_i, a_i])
            sc = float(sim_comb_mat[item_i, cq_i])
            rows.append(
                {
                    "item_id": item_ids[item_i],
                    "text": item_texts[item_i],
                    "family": item_families[item_i],
                    "query_id": cq.id,
                    "space_id": cq.space_id,
                    "ambiance_id": cq.ambiance_id,
                    "space_text": cq.space_text,
                    "ambiance_text": cq.ambiance_text,
                    "combined_text": cq.combined_text,
                    "sim_space": round(ss, 4),
                    "sim_ambiance": round(sa, 4),
                    "sim_combined": round(sc, 4),
                    "weighted_score": round(
                        weights.weighted(ss, sa, sc), 4
                    ),
                }
            )

    return pd.DataFrame(rows)


def similarity_matrix_df(
    item_texts: List[str],
    query_texts: List[str],
    item_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> pd.DataFrame:
    """
    Build a labelled similarity matrix DataFrame.

    Parameters
    ----------
    item_texts : list of n strings (row labels)
    query_texts : list of m strings (column labels)
    item_embeddings : (n, d)
    query_embeddings : (m, d)

    Returns
    -------
    pd.DataFrame, shape (n, m)
        Rows = items, columns = queries, values = cosine similarity.
    """
    mat = cosine_similarity_matrix(item_embeddings, query_embeddings)
    return pd.DataFrame(mat.round(4), index=item_texts, columns=query_texts)


def enrich_with_discriminative_scores(
    scores_df: pd.DataFrame,
    score_col: str = "weighted_score",
) -> pd.DataFrame:
    """
    Add discriminative_score and zscore_score columns to a scores DataFrame.

    Motivation
    ----------
    Absolute cosine similarity rewards atoms that are broadly similar to
    all room concepts (e.g. "hospital bed" is semantically close to every
    room type because "bed" overlaps with domestic space embeddings).
    Discriminative scoring corrects for this by subtracting each atom's
    mean score across all queries — analogous to TF-IDF for embeddings.

    Columns added
    -------------
    mean_score           : item's mean weighted_score across all queries
    std_score            : item's std across all queries
    discriminative_score : score − mean_score
                           ≈ 0 → generically relevant everywhere
                           > 0 → specifically relevant to this query
                           < 0 → less relevant here than average
    zscore_score         : (score − mean_score) / std_score
                           Normalises for cross-atom comparison

    Parameters
    ----------
    scores_df : pd.DataFrame
        Output of score_items_against_queries.  Must have columns
        ``item_id`` and *score_col*.
    score_col : str
        Column to derive discriminative scores from.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with four new columns appended.
    """
    grp = scores_df.groupby("item_id")[score_col]
    mean_s = grp.transform("mean")
    std_s = grp.transform("std").clip(lower=1e-6)

    df = scores_df.copy()
    df["mean_score"] = mean_s.round(4)
    df["std_score"] = std_s.round(4)
    df["discriminative_score"] = (df[score_col] - mean_s).round(4)
    df["zscore_score"] = ((df[score_col] - mean_s) / std_s).round(4)
    return df
