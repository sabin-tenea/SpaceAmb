"""
atoms.py — Atom data model and loading utilities.

An Atom is the minimal typed semantic unit in the SpaceAmb system.
Atoms are categorised by family (architecture, furniture, fixture,
decoration, technology, material, color, quality, lighting, spatial,
relation, behavioral) and an optional subtype.

The 'quality' family is special: each quality atom carries a 'spectrums'
dict that locates it on one or more perceptual axes (temperature,
luminosity, texture, …). All other families have spectrums=None.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Canonical family names — used for validation and filtering.
VALID_FAMILIES: tuple[str, ...] = (
    "architecture",
    "furniture",
    "fixture",
    "decoration",
    "technology",
    "material",
    "color",
    "quality",
    "lighting",
    "spatial",
    "relation",
    "behavioral",
)


@dataclass
class Atom:
    """
    A typed minimal semantic unit.

    Attributes
    ----------
    id : str
        Unique snake_case identifier, e.g. ``"mat_wood"``.
    text : str
        Human-readable label that is embedded, e.g. ``"wood"``.
    family : str
        One of VALID_FAMILIES.
    subtype : str or None
        Finer-grained category within the family, e.g. ``"textile"``.
    spectrums : dict or None
        Only populated for quality-family atoms.  Maps spectrum name
        (e.g. ``"temperature"``) to a pole label (e.g. ``"high"``).
    notes : str or None
        Optional human-readable annotation.
    """

    id: str
    text: str
    family: str
    subtype: Optional[str] = None
    spectrums: Optional[Dict[str, str]] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if self.family not in VALID_FAMILIES:
            raise ValueError(
                f"Unknown family '{self.family}' for atom '{self.id}'. "
                f"Valid families: {VALID_FAMILIES}"
            )

    def __str__(self) -> str:
        return f"Atom({self.text!r}, family={self.family!r})"

    def __repr__(self) -> str:
        return (
            f"Atom(id={self.id!r}, text={self.text!r}, "
            f"family={self.family!r}, subtype={self.subtype!r})"
        )


def load_atoms(path: str | Path) -> List[Atom]:
    """
    Load atoms from a JSON file.

    Expected structure::

        {
          "atoms": [
            {"id": "...", "text": "...", "family": "...", ...},
            ...
          ]
        }

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    list of Atom
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [
        Atom(
            id=entry["id"],
            text=entry["text"],
            family=entry["family"],
            subtype=entry.get("subtype"),
            spectrums=entry.get("spectrums"),
            notes=entry.get("notes"),
        )
        for entry in data["atoms"]
    ]


def filter_by_family(atoms: List[Atom], family: str) -> List[Atom]:
    """Return only atoms that belong to *family*."""
    return [a for a in atoms if a.family == family]


def filter_by_subtype(atoms: List[Atom], subtype: str) -> List[Atom]:
    """Return atoms matching *subtype* (exact match)."""
    return [a for a in atoms if a.subtype == subtype]


def atoms_to_dict(atoms: List[Atom]) -> Dict[str, Atom]:
    """Return a mapping from atom id → Atom for fast lookup."""
    return {a.id: a for a in atoms}


def atoms_summary(atoms: List[Atom]) -> str:
    """Return a human-readable summary of family counts."""
    from collections import Counter
    counts = Counter(a.family for a in atoms)
    lines = [f"  {fam}: {counts.get(fam, 0)}" for fam in VALID_FAMILIES]
    return "Atom summary:\n" + "\n".join(lines) + f"\n  TOTAL: {len(atoms)}"
