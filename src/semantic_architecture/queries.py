"""
queries.py — Query data model for spaces, ambiances, and combined phrases.

A Query is a target condition against which atoms and descriptors are
scored.  Three query modes are supported:

  space-only      — e.g. "living room"
  ambiance-only   — e.g. "relaxing"
  combined        — e.g. "relaxing living room"

The combined phrasing follows the pattern ``"{ambiance} {space}"``, which
tends to read naturally in English.  The combined text is what gets
displayed; ``embedding_text`` is what gets embedded (may be richer when
disambiguation descriptions are present in the data files).

Combined queries are generated as the Cartesian product of all spaces and
all ambiances, giving N × M pairs from the datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Query:
    """
    A scoring target.

    Attributes
    ----------
    id : str
        Unique identifier, e.g. ``"q_relaxing_living_room"``.
    combined_text : str
        Short readable label used for display and as a dict key.
        E.g. ``"relaxing living room"``.
    embedding_text : str
        The string actually embedded.  Defaults to ``combined_text``.
        When disambiguation descriptions are present in the data files
        this is a richer phrase, e.g.
        ``"relaxing: effortless calm ... living room: domestic social ..."``.
    space_text : str or None
        Raw program/space label, e.g. ``"living room"``.
    ambiance_text : str or None
        Raw ambiance label, e.g. ``"relaxing"``.
    space_id : str or None
        Source program id, e.g. ``"prog_living_room"``.
    ambiance_id : str or None
        Source ambiance id, e.g. ``"amb_relaxing"``.
    mode : str
        One of ``"space"``, ``"ambiance"``, or ``"combined"``.
    """

    id: str
    combined_text: str
    embedding_text: str = field(default="")
    space_text: Optional[str] = None
    ambiance_text: Optional[str] = None
    space_id: Optional[str] = None
    ambiance_id: Optional[str] = None
    mode: str = "combined"

    def __post_init__(self) -> None:
        # Default embedding_text to combined_text when not set explicitly
        if not self.embedding_text:
            self.embedding_text = self.combined_text

    def __str__(self) -> str:
        return f"Query({self.combined_text!r})"


def load_programs(path: str | Path) -> List[dict]:
    """
    Load program/space definitions from JSON.

    Returns a list of dicts with keys: id, text, and optionally description.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data["programs"]


def load_ambiances(path: str | Path) -> List[dict]:
    """
    Load ambiance definitions from JSON.

    Returns a list of dicts with keys: id, text, and optionally description.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data["ambiances"]


def generate_space_queries(programs: List[dict]) -> List[Query]:
    """Create one space-only Query per program."""
    queries = []
    for prog in programs:
        emb_text = prog.get("description", prog["text"])
        queries.append(
            Query(
                id=f"q_space_{prog['id']}",
                combined_text=prog["text"],
                embedding_text=emb_text,
                space_text=prog["text"],
                space_id=prog["id"],
                mode="space",
            )
        )
    return queries


def generate_ambiance_queries(ambiances: List[dict]) -> List[Query]:
    """Create one ambiance-only Query per ambiance."""
    queries = []
    for amb in ambiances:
        emb_text = amb.get("description", amb["text"])
        queries.append(
            Query(
                id=f"q_amb_{amb['id']}",
                combined_text=amb["text"],
                embedding_text=emb_text,
                ambiance_text=amb["text"],
                ambiance_id=amb["id"],
                mode="ambiance",
            )
        )
    return queries


def generate_combined_queries(
    programs: List[dict], ambiances: List[dict]
) -> List[Query]:
    """
    Generate all ambiance × space combined queries.

    ``combined_text`` is the short readable label ``"{ambiance} {space}"``.
    ``embedding_text`` uses disambiguation descriptions when present,
    giving a richer phrase that separates near-synonym ambiances and
    programs in the embedding space.
    """
    queries = []
    for amb in ambiances:
        for prog in programs:
            combined = f"{amb['text']} {prog['text']}"
            qid = f"q_{amb['id']}_{prog['id']}"

            amb_emb = amb.get("description", amb["text"])
            prog_emb = prog.get("description", prog["text"])
            emb_text = f"{amb_emb} {prog_emb}"

            queries.append(
                Query(
                    id=qid,
                    combined_text=combined,
                    embedding_text=emb_text,
                    space_text=prog["text"],
                    ambiance_text=amb["text"],
                    space_id=prog["id"],
                    ambiance_id=amb["id"],
                    mode="combined",
                )
            )
    return queries


def generate_all_queries(
    programs: List[dict], ambiances: List[dict]
) -> dict[str, List[Query]]:
    """
    Convenience: build all three query sets and return as a labelled dict.

    Keys: ``"space"``, ``"ambiance"``, ``"combined"``.
    """
    return {
        "space": generate_space_queries(programs),
        "ambiance": generate_ambiance_queries(ambiances),
        "combined": generate_combined_queries(programs, ambiances),
    }


def queries_by_id(queries: List[Query]) -> dict[str, Query]:
    """Return a mapping from query id → Query."""
    return {q.id: q for q in queries}
