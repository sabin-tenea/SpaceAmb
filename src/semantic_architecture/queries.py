"""
queries.py — Query data model for spaces, ambiances, and combined phrases.

A Query is a target condition against which atoms and descriptors are
scored.  Three query modes are supported:

  space-only      — e.g. "living room"
  ambiance-only   — e.g. "relaxing"
  combined        — e.g. "relaxing living room"

The combined phrasing follows the pattern ``"{ambiance} {space}"``, which
tends to read naturally in English.  The combined text is what gets
embedded for composite scoring.

Combined queries are generated as the Cartesian product of all spaces and
all ambiances, giving 14 × 14 = 196 pairs from the default datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
        The string that is embedded for combined scoring.
        For space-only queries this equals ``space_text``;
        for ambiance-only it equals ``ambiance_text``.
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
    space_text: Optional[str] = None
    ambiance_text: Optional[str] = None
    space_id: Optional[str] = None
    ambiance_id: Optional[str] = None
    mode: str = "combined"

    def __str__(self) -> str:
        return f"Query({self.combined_text!r})"


def load_programs(path: str | Path) -> List[dict]:
    """
    Load program/space definitions from JSON.

    Returns a list of dicts with keys: id, text, notes.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data["programs"]


def load_ambiances(path: str | Path) -> List[dict]:
    """
    Load ambiance definitions from JSON.

    Returns a list of dicts with keys: id, text, notes.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data["ambiances"]


def generate_space_queries(programs: List[dict]) -> List[Query]:
    """Create one space-only Query per program."""
    queries = []
    for prog in programs:
        queries.append(
            Query(
                id=f"q_space_{prog['id']}",
                combined_text=prog["text"],
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
        queries.append(
            Query(
                id=f"q_amb_{amb['id']}",
                combined_text=amb["text"],
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

    The phrase template is ``"{ambiance} {space}"``, e.g.
    ``"relaxing living room"``.

    Returns 14 × 14 = 196 queries for the default datasets.
    """
    queries = []
    for amb in ambiances:
        for prog in programs:
            combined = f"{amb['text']} {prog['text']}"
            qid = f"q_{amb['id']}_{prog['id']}"
            queries.append(
                Query(
                    id=qid,
                    combined_text=combined,
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
