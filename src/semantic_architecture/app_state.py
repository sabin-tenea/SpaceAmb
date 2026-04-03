"""
app_state.py — Shared loaded state for the SpaceAmb pipeline.

AppState is a single container that holds every loaded artefact: atoms,
queries, embeddings, descriptors, and score tables.  It is the single
entry point for loading everything so that:

  * notebook cells don't need to orchestrate loading order
  * a future Streamlit/Gradio UI can call load_all() once at startup
  * CLI commands receive a populated state without duplication

Usage
-----
    from semantic_architecture.app_state import AppState

    state = AppState.load(config_path="config/config.yaml")

    # All artefacts are then available:
    state.atoms           # List[Atom]
    state.atom_scores     # pd.DataFrame
    state.descriptors     # List[Descriptor]
    state.descriptor_scores  # pd.DataFrame
    # etc.

Design notes
------------
load() is intentionally eager: it runs the full pipeline on first call.
For large datasets or slow machines, a lazy/incremental variant could be
added later without changing the public interface.

The class is not a singleton — you can create multiple states with
different configs for comparative experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from .atoms import Atom, load_atoms
from .composition import Descriptor, descriptors_to_df, generate_descriptors
from .embeddings import EmbeddingModel
from .io_utils import resolve_path, save_csv, save_json, ensure_dir
from .queries import (
    Query,
    generate_all_queries,
    load_ambiances,
    load_programs,
    queries_by_id,
)
from .scenes import Scene, load_scenes, validate_scenes
from .scoring import (
    ScoringWeights,
    enrich_with_discriminative_scores,
    score_items_against_queries,
    similarity_matrix_df,
)


@dataclass
class AppState:
    """
    Fully loaded pipeline state.

    All list attributes are parallel-indexed (i.e. atoms[i] corresponds
    to atom_embeddings[i]).
    """

    # Raw data
    atoms: List[Atom]
    programs: List[dict]
    ambiances: List[dict]

    # Query sets
    space_queries: List[Query]
    ambiance_queries: List[Query]
    combined_queries: List[Query]

    # Embeddings (L2-normalised float32)
    atom_embeddings: np.ndarray           # (n_atoms, d)
    space_embeddings: np.ndarray          # (n_spaces, d)
    ambiance_embeddings: np.ndarray       # (n_ambiances, d)
    combined_embeddings: np.ndarray       # (n_combined, d)

    # Composed descriptors
    descriptors: List[Descriptor]
    descriptor_embeddings: np.ndarray     # (n_descriptors, d)

    # Phase 3: Scene descriptions (optional — None if no scenes file configured)
    scenes: Optional[List[Scene]]
    scene_embeddings: Optional[np.ndarray]       # (n_scenes, d) or None
    scene_scores: Optional[pd.DataFrame]         # (n_scenes × n_combined) or None

    # Score tables
    atom_scores: pd.DataFrame             # (n_atoms × n_combined, 13 cols)
    descriptor_scores: pd.DataFrame       # (n_descriptors × n_combined, 13 cols)

    # Config snapshot
    config: dict

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def atom_texts(self) -> List[str]:
        return [a.text for a in self.atoms]

    @property
    def atom_families(self) -> List[str]:
        return [a.family for a in self.atoms]

    @property
    def atom_ids(self) -> List[str]:
        return [a.id for a in self.atoms]

    @property
    def descriptor_texts(self) -> List[str]:
        return [d.text for d in self.descriptors]

    @property
    def combined_query_texts(self) -> List[str]:
        return [q.combined_text for q in self.combined_queries]

    def get_atom(self, text: str) -> Optional[Atom]:
        """Return the first atom whose text matches *text* (case-insensitive)."""
        text_l = text.lower()
        return next((a for a in self.atoms if a.text.lower() == text_l), None)

    def get_query(self, combined_text: str) -> Optional[Query]:
        """Return the combined query whose combined_text matches."""
        ct_l = combined_text.lower()
        return next(
            (q for q in self.combined_queries if q.combined_text.lower() == ct_l),
            None,
        )

    def get_descriptor(self, text: str) -> Optional[Descriptor]:
        """Return the first descriptor whose text matches."""
        text_l = text.lower()
        return next(
            (d for d in self.descriptors if d.text.lower() == text_l), None
        )

    @property
    def scene_texts(self) -> List[str]:
        return [s.text for s in self.scenes] if self.scenes else []

    @property
    def scene_ids(self) -> List[str]:
        return [s.id for s in self.scenes] if self.scenes else []

    def get_scene(self, scene_id: str) -> Optional[Scene]:
        """Return the scene whose id matches."""
        if not self.scenes:
            return None
        return next((s for s in self.scenes if s.id == scene_id), None)

    # ------------------------------------------------------------------
    # Factory / loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, config_path: str | Path = "config/config.yaml") -> "AppState":
        """
        Run the full pipeline and return a populated AppState.

        Steps
        -----
        1. Load config
        2. Load atoms, programs, ambiances
        3. Generate query sets
        4. Embed everything (with caching)
        5. Generate descriptors
        6. Embed descriptors
        7. Score atoms against all combined queries
        8. Score descriptors against all combined queries

        Parameters
        ----------
        config_path : str or Path
            Path to config.yaml, relative to project root or absolute.
        """
        cfg_path = resolve_path(config_path)
        with cfg_path.open(encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        root = resolve_path(".")

        # ── 1. Load raw data ───────────────────────────────────────────
        atoms = load_atoms(root / cfg["data"]["atoms_path"])
        programs = load_programs(root / cfg["data"]["programs_path"])
        ambiances = load_ambiances(root / cfg["data"]["ambiances_path"])

        print(f"[state] Loaded {len(atoms)} atoms, "
              f"{len(programs)} programs, {len(ambiances)} ambiances.")

        # ── 2. Queries ─────────────────────────────────────────────────
        query_sets = generate_all_queries(programs, ambiances)
        space_qs = query_sets["space"]
        ambiance_qs = query_sets["ambiance"]
        combined_qs = query_sets["combined"]

        print(f"[state] Queries: {len(space_qs)} space, "
              f"{len(ambiance_qs)} ambiance, {len(combined_qs)} combined.")

        # ── 3. Embedding model ─────────────────────────────────────────
        emb_cfg = cfg.get("embedding", {})
        cache_dir = root / emb_cfg.get("cache_dir", "data/processed/embeddings")
        model = EmbeddingModel(
            model_name=emb_cfg.get("model_name", "all-MiniLM-L6-v2"),
            cache_dir=cache_dir,
            batch_size=emb_cfg.get("batch_size", 64),
        )

        atom_texts = [a.text for a in atoms]
        # Use embedding_text (description-enriched) for the actual embed calls;
        # combined_text (short label) is kept for display and as table keys.
        space_texts = [q.embedding_text for q in space_qs]
        ambiance_texts = [q.embedding_text for q in ambiance_qs]
        combined_texts = [q.embedding_text for q in combined_qs]

        atom_embs = model.load_or_compute(atom_texts, "atoms")
        space_embs = model.load_or_compute(space_texts, "queries_space")
        ambiance_embs = model.load_or_compute(ambiance_texts, "queries_ambiance")
        combined_embs = model.load_or_compute(combined_texts, "queries_combined")

        # ── 4. Descriptors ─────────────────────────────────────────────
        comp_cfg = cfg.get("composition", {})
        descriptors = generate_descriptors(
            atoms=atoms,
            atom_embeddings=atom_embs,
            n_descriptors=comp_cfg.get("n_descriptors", 300),
            descriptor_lengths=comp_cfg.get("descriptor_lengths", [2, 3]),
            temperature=comp_cfg.get("temperature", 1.0),
            seed=comp_cfg.get("seed", 42),
        )
        print(f"[state] Generated {len(descriptors)} descriptors.")

        desc_texts = [d.text for d in descriptors]
        desc_embs = model.load_or_compute(desc_texts, "descriptors")

        # ── 5. Scoring ─────────────────────────────────────────────────
        weights = ScoringWeights.from_config(cfg.get("scoring", {}))

        atom_ids = [a.id for a in atoms]
        atom_families = [a.family for a in atoms]

        print("[state] Scoring atoms …")
        atom_scores = score_items_against_queries(
            item_texts=atom_texts,
            item_families=atom_families,
            item_ids=atom_ids,
            item_embeddings=atom_embs,
            space_queries=space_qs,
            ambiance_queries=ambiance_qs,
            combined_queries=combined_qs,
            space_embeddings=space_embs,
            ambiance_embeddings=ambiance_embs,
            combined_embeddings=combined_embs,
            weights=weights,
        )

        # Descriptors don't have a meaningful "family" — use family_pattern
        desc_ids = [d.id for d in descriptors]
        desc_families = [d.source_atom_families[0] if d.source_atom_families else "unknown"
                         for d in descriptors]

        print("[state] Scoring descriptors …")
        descriptor_scores = score_items_against_queries(
            item_texts=desc_texts,
            item_families=desc_families,
            item_ids=desc_ids,
            item_embeddings=desc_embs,
            space_queries=space_qs,
            ambiance_queries=ambiance_qs,
            combined_queries=combined_qs,
            space_embeddings=space_embs,
            ambiance_embeddings=ambiance_embs,
            combined_embeddings=combined_embs,
            weights=weights,
        )

        # Enrich both score tables with discriminative_score and zscore_score.
        # These subtract each item's mean score across all queries so that
        # generically relevant atoms (high baseline) are not artificially
        # promoted over specifically relevant ones.
        atom_scores = enrich_with_discriminative_scores(atom_scores)
        descriptor_scores = enrich_with_discriminative_scores(descriptor_scores)

        # ── 6. Scenes (Phase 3 — optional) ────────────────────────────
        scenes: Optional[List[Scene]] = None
        scene_embs: Optional[np.ndarray] = None
        scene_scores: Optional[pd.DataFrame] = None

        scenes_path_cfg = cfg.get("data", {}).get("scenes_path")
        if scenes_path_cfg:
            scenes_full_path = root / scenes_path_cfg
            if scenes_full_path.exists():
                scenes = load_scenes(scenes_full_path)
                print(f"[state] Loaded {len(scenes)} scenes.")

                # Validate against known programs/ambiances
                valid_spaces = [p["text"] for p in programs]
                valid_ambs = [a["text"] for a in ambiances]
                warnings = validate_scenes(scenes, valid_spaces, valid_ambs)
                for w in warnings:
                    print(f"[state] WARNING: {w}")

                scene_texts = [s.text for s in scenes]
                scene_embs = model.load_or_compute(scene_texts, "scenes")

                scene_ids = [s.id for s in scenes]
                # Use space field as a proxy family label for display
                scene_families = [s.space for s in scenes]

                print("[state] Scoring scenes …")
                scene_scores = score_items_against_queries(
                    item_texts=scene_texts,
                    item_families=scene_families,
                    item_ids=scene_ids,
                    item_embeddings=scene_embs,
                    space_queries=space_qs,
                    ambiance_queries=ambiance_qs,
                    combined_queries=combined_qs,
                    space_embeddings=space_embs,
                    ambiance_embeddings=ambiance_embs,
                    combined_embeddings=combined_embs,
                    weights=weights,
                )
                scene_scores = enrich_with_discriminative_scores(scene_scores)
            else:
                print(f"[state] scenes_path configured but file not found: {scenes_full_path}")

        print("[state] Pipeline complete.")

        return cls(
            atoms=atoms,
            programs=programs,
            ambiances=ambiances,
            space_queries=space_qs,
            ambiance_queries=ambiance_qs,
            combined_queries=combined_qs,
            atom_embeddings=atom_embs,
            space_embeddings=space_embs,
            ambiance_embeddings=ambiance_embs,
            combined_embeddings=combined_embs,
            descriptors=descriptors,
            descriptor_embeddings=desc_embs,
            scenes=scenes,
            scene_embeddings=scene_embs,
            scene_scores=scene_scores,
            atom_scores=atom_scores,
            descriptor_scores=descriptor_scores,
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_all(self, output_dir: Optional[str | Path] = None) -> Path:
        """
        Save all score tables, matrices, and descriptors to disk.

        Parameters
        ----------
        output_dir : str or Path or None
            Defaults to config["output"]["dir"].

        Returns
        -------
        Path
            The resolved output directory.
        """
        if output_dir is None:
            output_dir = resolve_path(
                self.config.get("output", {}).get("dir", "data/processed")
            )
        out = Path(output_dir)
        ensure_dir(out)

        save_csv(self.atom_scores, out / "atom_scores.csv", "atom scores")
        save_csv(
            self.descriptor_scores,
            out / "descriptor_scores.csv",
            "descriptor scores",
        )

        # Phase 3: export scene scores if available
        if self.scene_scores is not None:
            save_csv(self.scene_scores, out / "scene_scores.csv", "scene scores")
            if self.scenes:
                save_json(
                    [
                        {
                            "id": s.id,
                            "text": s.text,
                            "space": s.space,
                            "ambiance": s.ambiance,
                            "intended_query": s.intended_query,
                            "notes": s.notes,
                        }
                        for s in self.scenes
                    ],
                    out / "scenes.json",
                )

        save_json(
            [
                {
                    "id": d.id,
                    "text": d.text,
                    "source_atom_ids": d.source_atom_ids,
                    "source_atom_texts": d.source_atom_texts,
                    "source_atom_families": d.source_atom_families,
                    "pairwise_sims": d.pairwise_sims,
                    "family_pattern": "+".join(d.source_atom_families),
                }
                for d in self.descriptors
            ],
            out / "descriptors.json",
        )

        # Save similarity matrices for the combined queries
        cfg_out = self.config.get("output", {})
        if cfg_out.get("save_matrices", True):
            # Atoms × combined queries
            mat_df = similarity_matrix_df(
                self.atom_texts,
                self.combined_query_texts,
                self.atom_embeddings,
                self.combined_embeddings,
            )
            save_csv(mat_df, out / "matrix_atoms_combined.csv", "atom-query matrix")

        print(f"[state] All outputs saved to {out}")
        return out
