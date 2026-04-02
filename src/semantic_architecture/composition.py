"""
composition.py — Similarity-weighted descriptor generation.

Design philosophy
-----------------
Descriptors are not hand-authored.  They emerge from the embedding space.

Given a pool of atoms and their pre-computed embeddings, we build
descriptors of length 2 or 3 by *weighted random sampling*:

  1. Pick a seed atom at random.
  2. For each subsequent slot, compute the cosine similarity between the
     seed embedding and every remaining atom's embedding.
  3. Convert similarities to sampling probabilities via softmax scaled by
     a *temperature* parameter.
  4. Sample the next atom from that distribution.
  5. Join the selected atoms' texts with spaces.

Temperature controls exploration:
  - low  (e.g. 0.3): highly likely to pick the nearest neighbour → tight,
    semantically coherent descriptors (e.g. "warm indirect", "velvet sofa")
  - high (e.g. 3.0): nearly uniform distribution → more surprising
    cross-family combinations
  - 1.0 (default): balanced

No explicit allowed/banned rules are enforced.  The embedding space is the
judge.  Unusual combinations (e.g. "shadowed concrete") can and do appear
— this is intentional for research purposes.

Descriptor provenance
---------------------
Every Descriptor stores the IDs, texts, and families of its source atoms,
plus the pairwise cosine similarities between them.  This makes it easy
to answer "why did this descriptor get generated?"

Deduplication
-------------
Descriptors with the same frozenset of atom IDs (regardless of order)
are deduplicated.  Text order follows the sampling order (seed first).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .atoms import Atom
from .scoring import cosine_similarity_vec


# ------------------------------------------------------------------
# Architectural Grammar
# ------------------------------------------------------------------

# Defines which families are allowed to be combined with each other.
# A seed atom of family X can only be paired with an atom of family Y
# if Y is in ALLOWED_COMBINATIONS[X].
ALLOWED_COMBINATIONS: Dict[str, List[str]] = {
    "architecture": ["material", "color", "quality", "lighting", "spatial", "relation"],
    "furniture": ["material", "color", "quality", "spatial", "behavioral"],
    "fixture": ["material", "color", "quality", "lighting"],
    "decoration": ["material", "color", "quality"],
    "technology": ["architecture", "furniture", "spatial"],
    "material": ["architecture", "furniture", "fixture", "decoration", "color", "quality"],
    "color": ["material", "architecture", "furniture", "fixture", "decoration", "quality"],
    "quality": [
        "architecture",
        "furniture",
        "fixture",
        "decoration",
        "material",
        "color",
        "spatial",
        "lighting",
    ],
    "lighting": ["architecture", "spatial", "material", "quality"],
    "spatial": ["architecture", "furniture", "behavioral", "relation"],
    "relation": ["architecture", "spatial", "quality"],
    "behavioral": ["spatial", "architecture", "furniture"],
}


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

@dataclass
class Descriptor:
    """
    A composed architectural phrase built from 2–3 atoms.

    Attributes
    ----------
    id : str
        Deterministic hash of the sorted atom ids.
    text : str
        Space-joined atom texts in sampling order, e.g. "warm indirect".
    source_atom_ids : list of str
        Ordered list of atom ids used to build this descriptor.
    source_atom_texts : list of str
        Ordered list of atom texts (parallel to source_atom_ids).
    source_atom_families : list of str
        Ordered list of atom families (parallel to source_atom_ids).
    pairwise_sims : dict
        Pairwise cosine similarities between source atoms.
        Key format: "{id_a}|{id_b}", value: float.
    """

    id: str
    text: str
    source_atom_ids: List[str]
    source_atom_texts: List[str]
    source_atom_families: List[str]
    pairwise_sims: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Descriptor({self.text!r})"

    def __repr__(self) -> str:
        return (
            f"Descriptor(id={self.id!r}, text={self.text!r}, "
            f"families={self.source_atom_families!r})"
        )

    def provenance_str(self) -> str:
        """Human-readable provenance summary."""
        parts = [
            f"  [{i+1}] '{t}' ({f})"
            for i, (t, f) in enumerate(
                zip(self.source_atom_texts, self.source_atom_families)
            )
        ]
        sim_parts = [
            f"  sim({k}) = {v:.3f}"
            for k, v in self.pairwise_sims.items()
        ]
        return (
            f"Descriptor: '{self.text}'\n"
            + "\n".join(parts)
            + ("\n" + "\n".join(sim_parts) if sim_parts else "")
        )


def _descriptor_id(atom_ids: List[str]) -> str:
    """Deterministic ID based on the *sorted* set of atom ids."""
    key = "|".join(sorted(atom_ids))
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    """
    Softmax with temperature scaling.

    Higher temperature → flatter distribution (more exploration).
    Lower temperature → peaks at the highest value (more exploitation).
    """
    x_scaled = x / max(temperature, 1e-6)
    x_shifted = x_scaled - x_scaled.max()  # numerical stability
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum()


# ------------------------------------------------------------------
# Core composition function
# ------------------------------------------------------------------

def compose_one(
    seed_atom: Atom,
    seed_embedding: np.ndarray,
    all_atoms: List[Atom],
    all_embeddings: np.ndarray,
    length: int,
    temperature: float,
    rng: random.Random,
) -> Optional[Descriptor]:
    """
    Build a single descriptor starting from *seed_atom*.

    Parameters
    ----------
    seed_atom : Atom
        The first atom (already chosen externally so callers can control
        seed selection, e.g. stratified by family).
    seed_embedding : np.ndarray, shape (d,)
        Pre-computed embedding for seed_atom.
    all_atoms : list of Atom
        Full atom pool including seed_atom.
    all_embeddings : np.ndarray, shape (n, d)
        Embeddings parallel to all_atoms.
    length : int
        Number of atoms in the descriptor (2 or 3).
    temperature : float
        Sampling temperature.
    rng : random.Random
        Seeded RNG for reproducibility.

    Returns
    -------
    Descriptor or None
        None if the same atom would be selected twice (degenerate case).
    """
    seed_idx = next(
        (i for i, a in enumerate(all_atoms) if a.id == seed_atom.id), None
    )
    if seed_idx is None:
        return None

    selected_atoms: List[Atom] = [seed_atom]
    selected_embeddings: List[np.ndarray] = [seed_embedding]
    excluded_ids = {seed_atom.id}

    # Build pool excluding seed and restricted by grammar
    allowed_families = ALLOWED_COMBINATIONS.get(seed_atom.family, [])
    pool_indices = [
        i
        for i, a in enumerate(all_atoms)
        if a.id not in excluded_ids and a.family in allowed_families
    ]

    for _ in range(length - 1):
        if not pool_indices:
            break

        pool_embs = all_embeddings[pool_indices]
        # Similarity of each pool atom to the current seed embedding
        sims = cosine_similarity_vec(seed_embedding, pool_embs)

        # Shift to [0, 1] range so softmax stays positive even at low temp
        sims_shifted = (sims + 1.0) / 2.0

        probs = _softmax(sims_shifted, temperature)
        # rng.choices needs a plain Python list
        chosen_local_idx = rng.choices(range(len(pool_indices)), weights=probs.tolist(), k=1)[0]
        chosen_global_idx = pool_indices[chosen_local_idx]

        chosen_atom = all_atoms[chosen_global_idx]
        selected_atoms.append(chosen_atom)
        selected_embeddings.append(all_embeddings[chosen_global_idx])
        excluded_ids.add(chosen_atom.id)
        pool_indices = [i for i in pool_indices if all_atoms[i].id not in excluded_ids]

    if len(selected_atoms) < 2:
        return None

    # Compute pairwise similarities for provenance
    pairwise: Dict[str, float] = {}
    for i in range(len(selected_atoms)):
        for j in range(i + 1, len(selected_atoms)):
            a, b = selected_atoms[i], selected_atoms[j]
            sim = float(
                np.dot(selected_embeddings[i], selected_embeddings[j])
            )
            pairwise[f"{a.id}|{b.id}"] = round(sim, 4)

    return Descriptor(
        id=_descriptor_id([a.id for a in selected_atoms]),
        text=" ".join(a.text for a in selected_atoms),
        source_atom_ids=[a.id for a in selected_atoms],
        source_atom_texts=[a.text for a in selected_atoms],
        source_atom_families=[a.family for a in selected_atoms],
        pairwise_sims=pairwise,
    )


# ------------------------------------------------------------------
# Batch generation
# ------------------------------------------------------------------

def generate_descriptors(
    atoms: List[Atom],
    atom_embeddings: np.ndarray,
    n_descriptors: int = 300,
    descriptor_lengths: List[int] = None,
    temperature: float = 1.0,
    seed: int = 42,
) -> List[Descriptor]:
    """
    Generate a diverse set of descriptors by similarity-weighted sampling.

    Descriptors are split evenly across the requested lengths.
    Duplicates (same frozenset of atom ids) are discarded, and extra
    samples are drawn to compensate until n_descriptors unique ones are
    collected or the attempt budget is exhausted.

    Parameters
    ----------
    atoms : list of Atom (length n)
    atom_embeddings : np.ndarray, shape (n, d)
    n_descriptors : int
        Target number of unique descriptors.
    descriptor_lengths : list of int
        E.g. [2, 3].  Descriptors are split equally across lengths.
    temperature : float
    seed : int

    Returns
    -------
    list of Descriptor
    """
    if descriptor_lengths is None:
        descriptor_lengths = [2, 3]

    rng = random.Random(seed)
    seen_ids: set[str] = set()
    results: List[Descriptor] = []

    n_per_length = n_descriptors // len(descriptor_lengths)
    remainder = n_descriptors % len(descriptor_lengths)

    for length_i, length in enumerate(descriptor_lengths):
        target = n_per_length + (1 if length_i < remainder else 0)
        attempts = 0
        max_attempts = target * 20  # safety ceiling

        while len(results) < sum(
            n_per_length + (1 if li < remainder else 0)
            for li in range(length_i + 1)
        ) and attempts < max_attempts:
            attempts += 1
            seed_atom = rng.choice(atoms)
            seed_idx = next(
                i for i, a in enumerate(atoms) if a.id == seed_atom.id
            )
            desc = compose_one(
                seed_atom=seed_atom,
                seed_embedding=atom_embeddings[seed_idx],
                all_atoms=atoms,
                all_embeddings=atom_embeddings,
                length=length,
                temperature=temperature,
                rng=rng,
            )
            if desc is None:
                continue
            if desc.id in seen_ids:
                continue
            seen_ids.add(desc.id)
            results.append(desc)

    return results


# ------------------------------------------------------------------
# Serialisation helpers
# ------------------------------------------------------------------

def descriptors_to_records(descriptors: List[Descriptor]) -> List[dict]:
    """Convert descriptors to a list of plain dicts (JSON-serialisable)."""
    return [
        {
            "id": d.id,
            "text": d.text,
            "source_atom_ids": d.source_atom_ids,
            "source_atom_texts": d.source_atom_texts,
            "source_atom_families": d.source_atom_families,
            "pairwise_sims": d.pairwise_sims,
            "n_atoms": len(d.source_atom_ids),
            "family_pattern": "+".join(d.source_atom_families),
        }
        for d in descriptors
    ]


def descriptors_to_df(descriptors: List[Descriptor]) -> "pd.DataFrame":
    """Convert to a flat DataFrame for quick inspection."""
    import pandas as pd
    return pd.DataFrame(descriptors_to_records(descriptors))
