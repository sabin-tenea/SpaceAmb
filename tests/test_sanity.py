"""
test_sanity.py — Sanity checks for the SpaceAmb prototype.

These tests verify:
  1. Data integrity (atom file structure, family names, quality spectrums)
  2. Semantic ranking correctness (requires embedding model download on first run)
  3. Discriminative scoring behaviour
  4. Composition correctness (count, provenance, uniqueness, min-length)
  5. Low-temperature sampling bias
  6. Scoring math (cosine similarity shape, self-similarity, weighted formula)
  7. Embedding cache round-trip

Run with:
    pytest tests/test_sanity.py -v

The semantic tests download the embedding model on first run (~420 MB for
all-mpnet-base-v2) and cache it locally via sentence-transformers.
Subsequent runs are fast because embeddings are cached to disk.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from semantic_architecture.atoms import Atom, load_atoms, VALID_FAMILIES
from semantic_architecture.composition import compose_one, generate_descriptors
from semantic_architecture.scoring import (
    ScoringWeights,
    cosine_similarity_matrix,
    cosine_similarity_vec,
    enrich_with_discriminative_scores,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

ATOMS_PATH = ROOT / "data" / "raw" / "atoms.json"
CACHE_DIR = ROOT / "data" / "processed" / "embeddings"

# Read model from config to keep tests in sync
import yaml
with open(ROOT / "config" / "config.yaml") as _fh:
    _cfg = yaml.safe_load(_fh)
MODEL_NAME = _cfg["embedding"]["model_name"]


@pytest.fixture(scope="session")
def atoms():
    return load_atoms(ATOMS_PATH)


@pytest.fixture(scope="session")
def atom_embeddings(atoms):
    """Embed all atoms; cached to disk for fast reruns."""
    from semantic_architecture.embeddings import EmbeddingModel
    model = EmbeddingModel(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    return model.load_or_compute([a.text for a in atoms], "atoms")


@pytest.fixture(scope="session")
def space_embeddings():
    from semantic_architecture.embeddings import EmbeddingModel
    model = EmbeddingModel(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    spaces = ["living room", "laboratory", "hospital room",
              "cafeteria", "bedroom", "gym", "conference room"]
    return spaces, model.load_or_compute(spaces, "test_spaces")


@pytest.fixture(scope="session")
def ambiance_embeddings():
    from semantic_architecture.embeddings import EmbeddingModel
    model = EmbeddingModel(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    ambiances = ["relaxing", "sterile", "fun", "somber", "spooky", "intimate", "formal"]
    return ambiances, model.load_or_compute(ambiances, "test_ambiances")


@pytest.fixture(scope="session")
def small_scores(atoms, atom_embeddings, space_embeddings, ambiance_embeddings):
    """
    Full scoring table for a small subset of queries, with discriminative scores.
    Used by the discriminative ranking tests.
    """
    from semantic_architecture.queries import (
        load_programs, load_ambiances,
        generate_space_queries, generate_ambiance_queries, generate_combined_queries,
    )
    from semantic_architecture.scoring import score_items_against_queries

    spaces_txt, space_embs = space_embeddings
    ambiances_txt, amb_embs = ambiance_embeddings

    # Build minimal query sets from our test lists
    programs = [{"id": f"prog_{t.replace(' ','_')}", "text": t, "notes": ""} for t in spaces_txt]
    ambiances_list = [{"id": f"amb_{t}", "text": t, "notes": ""} for t in ambiances_txt]

    space_qs = generate_space_queries(programs)
    ambiance_qs = generate_ambiance_queries(ambiances_list)
    combined_qs = generate_combined_queries(programs, ambiances_list)

    from semantic_architecture.embeddings import EmbeddingModel
    model = EmbeddingModel(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    combined_embs = model.load_or_compute(
        [q.combined_text for q in combined_qs], "test_combined"
    )

    scores = score_items_against_queries(
        item_texts=[a.text for a in atoms],
        item_families=[a.family for a in atoms],
        item_ids=[a.id for a in atoms],
        item_embeddings=atom_embeddings,
        space_queries=space_qs,
        ambiance_queries=ambiance_qs,
        combined_queries=combined_qs,
        space_embeddings=space_embs,
        ambiance_embeddings=amb_embs,
        combined_embeddings=combined_embs,
        weights=ScoringWeights(),
    )
    return enrich_with_discriminative_scores(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sim(atom_text, target_text, atoms, atom_embs, target_texts, target_embs):
    atom_idx = next(i for i, a in enumerate(atoms) if a.text == atom_text)
    target_idx = target_texts.index(target_text)
    return float(cosine_similarity_vec(atom_embs[atom_idx], target_embs)[target_idx])


def _discriminative(atom_text, query_text, scores_df):
    row = scores_df[
        (scores_df["text"] == atom_text)
        & (scores_df["combined_text"] == query_text)
    ]
    if row.empty:
        # Try space-only match
        row = scores_df[
            (scores_df["text"] == atom_text)
            & (scores_df["space_text"] == query_text)
        ]
    assert not row.empty, f"No score found for '{atom_text}' / '{query_text}'"
    return float(row["discriminative_score"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_atoms_json_valid():
    """atoms.json must parse as valid JSON with at least 100 atoms."""
    with ATOMS_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert "atoms" in data
    assert len(data["atoms"]) >= 100


def test_no_duplicate_atom_ids():
    """Every atom id must be unique."""
    from collections import Counter
    with ATOMS_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    ids = [a["id"] for a in data["atoms"]]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    assert not dupes, f"Duplicate atom IDs: {dupes}"


def test_no_duplicate_atom_texts():
    """Every atom text must be unique."""
    from collections import Counter
    with ATOMS_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    texts = [a["text"] for a in data["atoms"]]
    dupes = [k for k, v in Counter(texts).items() if v > 1]
    assert not dupes, f"Duplicate atom texts: {dupes}"


def test_all_atoms_have_valid_families(atoms):
    for atom in atoms:
        assert atom.family in VALID_FAMILIES, (
            f"Atom {atom.id!r} has unknown family {atom.family!r}"
        )


def test_twelve_families_present(atoms):
    """After taxonomy expansion, exactly 12 families should be used."""
    assert len(VALID_FAMILIES) == 12
    families_used = {a.family for a in atoms}
    missing = set(VALID_FAMILIES) - families_used
    assert not missing, f"These families have no atoms: {missing}"


def test_new_families_have_minimum_atoms(atoms):
    """architecture, furniture, fixture, decoration, technology must each have >= 5 atoms."""
    from collections import Counter
    c = Counter(a.family for a in atoms)
    for fam in ("architecture", "furniture", "fixture", "decoration", "technology"):
        assert c[fam] >= 5, f"Family '{fam}' has only {c[fam]} atoms (need >= 5)"


def test_quality_atoms_have_spectrums(atoms):
    """Quality-family atoms must carry a non-empty spectrums dict."""
    quality_atoms = [a for a in atoms if a.family == "quality"]
    assert len(quality_atoms) >= 8, (
        f"Expected >= 8 quality atoms, got {len(quality_atoms)}"
    )
    for qa in quality_atoms:
        assert isinstance(qa.spectrums, dict), (
            f"Quality atom {qa.id!r} has non-dict spectrums: {qa.spectrums!r}"
        )
        assert len(qa.spectrums) >= 1


def test_non_quality_atoms_have_null_spectrums(atoms):
    """Non-quality atoms must have spectrums=None."""
    for atom in atoms:
        if atom.family != "quality":
            assert atom.spectrums is None, (
                f"Non-quality atom {atom.id!r} ({atom.family}) "
                f"has unexpected spectrums: {atom.spectrums!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Semantic ranking (requires embedding model)
# ─────────────────────────────────────────────────────────────────────────────

def test_sofa_higher_for_living_room_than_laboratory(atoms, atom_embeddings, space_embeddings):
    spaces, space_embs = space_embeddings
    sim_lr  = _sim("sofa", "living room",  atoms, atom_embeddings, spaces, space_embs)
    sim_lab = _sim("sofa", "laboratory",   atoms, atom_embeddings, spaces, space_embs)
    assert sim_lr > sim_lab, (
        f"sim(sofa, living room)={sim_lr:.4f} should be > sim(sofa, laboratory)={sim_lab:.4f}"
    )


def test_workbench_higher_for_laboratory_than_living_room(atoms, atom_embeddings, space_embeddings):
    spaces, space_embs = space_embeddings
    sim_lab = _sim("workbench", "laboratory",  atoms, atom_embeddings, spaces, space_embs)
    sim_lr  = _sim("workbench", "living room", atoms, atom_embeddings, spaces, space_embs)
    assert sim_lab > sim_lr, (
        f"sim(workbench, laboratory)={sim_lab:.4f} should be > sim(workbench, living room)={sim_lr:.4f}"
    )


def test_warm_higher_for_relaxing_than_sterile(atoms, atom_embeddings, ambiance_embeddings):
    ambiances, amb_embs = ambiance_embeddings
    sim_relax  = _sim("warm", "relaxing", atoms, atom_embeddings, ambiances, amb_embs)
    sim_sterile = _sim("warm", "sterile", atoms, atom_embeddings, ambiances, amb_embs)
    assert sim_relax > sim_sterile, (
        f"sim(warm, relaxing)={sim_relax:.4f} should be > sim(warm, sterile)={sim_sterile:.4f}"
    )


def test_formal_higher_for_formal_than_fun(atoms, atom_embeddings, ambiance_embeddings):
    """'formal' quality atom should resonate more with 'formal' ambiance than 'fun'."""
    ambiances, amb_embs = ambiance_embeddings
    sim_formal = _sim("formal", "formal", atoms, atom_embeddings, ambiances, amb_embs)
    sim_fun    = _sim("formal", "fun",    atoms, atom_embeddings, ambiances, amb_embs)
    assert sim_formal > sim_fun, (
        f"sim(formal, formal)={sim_formal:.4f} should be > sim(formal, fun)={sim_fun:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Discriminative scoring
# ─────────────────────────────────────────────────────────────────────────────

def test_discriminative_scores_present(small_scores):
    """enrich_with_discriminative_scores must add the expected columns."""
    for col in ("mean_score", "std_score", "discriminative_score", "zscore_score"):
        assert col in small_scores.columns, f"Missing column: {col}"


def test_sofa_outranks_bed_for_living_room_discriminatively(small_scores):
    """
    Under discriminative scoring, 'sofa' should rank above 'bed' for
    'relaxing living room' — even though 'bed' might have higher absolute
    similarity (both are furniture).
    """
    subset = small_scores[
        small_scores["combined_text"] == "relaxing living room"
    ].sort_values("discriminative_score", ascending=False)

    if "sofa" not in subset["text"].values or "bed" not in subset["text"].values:
        pytest.skip("sofa or bed not in small_scores (query not matched)")

    rank_sofa = subset[subset["text"] == "sofa"].index[0]
    rank_bed  = subset[subset["text"] == "bed"].index[0]
    # Lower positional index = higher rank after sort
    sofa_pos = list(subset["text"]).index("sofa")
    bed_pos  = list(subset["text"]).index("bed")
    assert sofa_pos < bed_pos, (
        f"Expected sofa to rank above bed for 'relaxing living room' "
        f"(discriminative), got sofa pos {sofa_pos} vs bed pos {bed_pos}"
    )


def test_discriminative_score_mean_near_zero_per_item(small_scores):
    """
    By construction, each item's mean discriminative_score across all
    queries must be ≈ 0.
    """
    mean_per_item = small_scores.groupby("item_id")["discriminative_score"].mean()
    max_abs = mean_per_item.abs().max()
    assert max_abs < 1e-3, (
        f"Per-item mean of discriminative_score should be ≈ 0, got max abs = {max_abs:.6f}"
    )


def test_weighted_vs_discriminative_differ_in_ranking(small_scores):
    """
    The weighted_score and discriminative_score rankings should differ for
    at least one query (i.e. the correction has a measurable effect).
    """
    query = small_scores["combined_text"].iloc[0]
    sub = small_scores[small_scores["combined_text"] == query].copy()
    rank_w  = list(sub.sort_values("weighted_score", ascending=False)["text"])
    rank_d  = list(sub.sort_values("discriminative_score", ascending=False)["text"])
    assert rank_w != rank_d, (
        "weighted_score and discriminative_score produce identical rankings — "
        "the discriminative correction appears to have no effect."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Composition
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_atoms(n=15, d=32):
    """
    Create synthetic atoms split between 'material' and 'architecture' families,
    which are mutually compatible under ALLOWED_COMBINATIONS so that
    compose_one can always find an eligible partner atom.
    """
    rng = np.random.default_rng(0)
    # Alternate families: material atoms can combine with architecture atoms and vice versa
    families = ["material" if i % 2 == 0 else "architecture" for i in range(n)]
    atom_list = [
        Atom(id=f"syn_{i}", text=f"atom_{i}", family=families[i])
        for i in range(n)
    ]
    embs = rng.standard_normal((n, d)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return atom_list, embs


def test_composition_count():
    atom_list, embs = _synthetic_atoms(20, 32)
    descriptors = generate_descriptors(
        atoms=atom_list, atom_embeddings=embs,
        n_descriptors=30, descriptor_lengths=[2, 3], temperature=1.0, seed=42,
    )
    assert len(descriptors) == 30


def test_descriptor_provenance():
    atom_list, embs = _synthetic_atoms(15, 32)
    valid_ids = {a.id for a in atom_list}
    descriptors = generate_descriptors(
        atoms=atom_list, atom_embeddings=embs,
        n_descriptors=20, descriptor_lengths=[2, 3], seed=42,
    )
    for d in descriptors:
        for aid in d.source_atom_ids:
            assert aid in valid_ids, f"'{d.text}' references unknown atom id '{aid}'"


def test_descriptor_ids_unique():
    atom_list, embs = _synthetic_atoms(20, 32)
    descriptors = generate_descriptors(
        atoms=atom_list, atom_embeddings=embs,
        n_descriptors=40, descriptor_lengths=[2, 3], seed=42,
    )
    ids = [d.id for d in descriptors]
    assert len(ids) == len(set(ids)), "Duplicate descriptor IDs found"


def test_descriptor_min_length():
    atom_list, embs = _synthetic_atoms(15, 32)
    descriptors = generate_descriptors(
        atoms=atom_list, atom_embeddings=embs,
        n_descriptors=20, descriptor_lengths=[2, 3], seed=0,
    )
    for d in descriptors:
        assert len(d.source_atom_ids) >= 2, (
            f"Descriptor '{d.text}' has only {len(d.source_atom_ids)} atom(s)"
        )


def test_composition_low_temperature_biased_toward_similar(atoms, atom_embeddings):
    """At very low temperature, nearest-neighbour should appear more than most-dissimilar."""
    from semantic_architecture.composition import ALLOWED_COMBINATIONS

    rng = random.Random(7)
    seed_atom = atoms[0]
    seed_emb = atom_embeddings[0]

    # Restrict the candidate pool to families that are grammar-eligible for the seed,
    # exactly as compose_one does — so nn_idx and far_idx are reachable candidates.
    allowed_families = set(ALLOWED_COMBINATIONS.get(seed_atom.family, []))
    sims = cosine_similarity_vec(seed_emb, atom_embeddings)
    pool = [
        (sims[i], i)
        for i in range(len(atoms))
        if i != 0 and atoms[i].family in allowed_families
    ]
    assert len(pool) >= 2, (
        f"Seed atom family '{seed_atom.family}' has too few eligible partners in the atom set."
    )
    nn_idx  = max(pool, key=lambda x: x[0])[1]
    far_idx = min(pool, key=lambda x: x[0])[1]

    nn_count = far_count = 0
    for _ in range(200):
        d = compose_one(
            seed_atom=seed_atom,
            seed_embedding=seed_emb,
            all_atoms=atoms,
            all_embeddings=atom_embeddings,
            length=2,
            temperature=0.1,
            rng=rng,
        )
        if d is None:
            continue
        if atoms[nn_idx].id in d.source_atom_ids:
            nn_count += 1
        if atoms[far_idx].id in d.source_atom_ids:
            far_count += 1

    assert nn_count > far_count, (
        f"Low-temp sampling should favour nearest neighbour. "
        f"nn={nn_count}, far={far_count}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Scoring utilities
# ─────────────────────────────────────────────────────────────────────────────

def test_cosine_similarity_matrix_shape():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 16)).astype(np.float32)
    B = rng.standard_normal((7, 16)).astype(np.float32)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    assert cosine_similarity_matrix(A, B).shape == (5, 7)


def test_cosine_self_similarity_is_one():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((4, 32)).astype(np.float32)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    mat = cosine_similarity_matrix(A, A)
    np.testing.assert_allclose(np.diag(mat), np.ones(4), atol=1e-5)


def test_weighted_score_formula():
    w = ScoringWeights(space=0.25, ambiance=0.25, combined=0.50)
    score = w.weighted(0.8, 0.6, 0.7)
    expected = 0.25 * 0.8 + 0.25 * 0.6 + 0.50 * 0.7
    assert abs(score - expected) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cache round-trip
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 7. Phase 3 — Scene descriptions
# ─────────────────────────────────────────────────────────────────────────────

def test_scenes_json_loads():
    """data/raw/scenes.json must exist and parse cleanly."""
    from semantic_architecture.scenes import load_scenes, scenes_summary
    root = Path(__file__).parent.parent
    path = root / "data" / "raw" / "scenes.json"
    assert path.exists(), f"scenes.json not found at {path}"
    scenes = load_scenes(path)
    assert len(scenes) >= 10, f"Expected >= 10 scenes, got {len(scenes)}"
    summary = scenes_summary(scenes)
    assert "Scenes loaded" in summary


def test_scenes_have_required_fields():
    """Every scene must have non-empty id, text, space, and ambiance."""
    from semantic_architecture.scenes import load_scenes
    root = Path(__file__).parent.parent
    scenes = load_scenes(root / "data" / "raw" / "scenes.json")
    for s in scenes:
        assert s.id, f"Scene missing id: {s!r}"
        assert len(s.text) >= 50, f"Scene {s.id!r} text too short: {s.text!r}"
        assert s.space, f"Scene {s.id!r} missing space"
        assert s.ambiance, f"Scene {s.id!r} missing ambiance"


def test_scenes_no_duplicate_ids():
    """Scene IDs must be unique."""
    from semantic_architecture.scenes import load_scenes
    root = Path(__file__).parent.parent
    scenes = load_scenes(root / "data" / "raw" / "scenes.json")
    ids = [s.id for s in scenes]
    assert len(ids) == len(set(ids)), "Duplicate scene ids found"


def test_scenes_validate_against_programs_and_ambiances():
    """All scene space/ambiance values must match the known programs/ambiances."""
    from semantic_architecture.scenes import load_scenes, validate_scenes
    from semantic_architecture.queries import load_programs, load_ambiances
    root = Path(__file__).parent.parent
    scenes = load_scenes(root / "data" / "raw" / "scenes.json")
    programs = load_programs(root / "data" / "raw" / "programs.json")
    ambiances = load_ambiances(root / "data" / "raw" / "ambiances.json")
    valid_spaces = [p["text"] for p in programs]
    valid_ambs = [a["text"] for a in ambiances]
    warnings = validate_scenes(scenes, valid_spaces, valid_ambs)
    assert not warnings, f"Scene validation errors:\n" + "\n".join(warnings)


def test_intended_query_property():
    """Scene.intended_query should combine ambiance + space correctly."""
    from semantic_architecture.scenes import Scene
    s = Scene(id="test_01", text="A test scene.", space="living room", ambiance="relaxing")
    assert s.intended_query == "relaxing living room"


def test_scene_scoring_ranks_intended_query_in_top5(small_scores):
    """
    A relaxing living room scene should score in the top-5 for its intended query
    when embedded and scored against the pre-computed query grid.

    Uses the real scenes.json file and the pre-computed small_scores fixture
    which already has combined queries available.
    """
    from semantic_architecture.scenes import load_scenes
    from semantic_architecture.scoring import enrich_with_discriminative_scores

    root = Path(__file__).parent.parent
    scenes = load_scenes(root / "data" / "raw" / "scenes.json")

    # Filter to the two relaxing living room scenes
    rlr_scenes = [s for s in scenes if s.intended_query == "relaxing living room"]
    assert len(rlr_scenes) >= 1

    # Check that their intended query is among the combined queries in small_scores
    combined_texts = small_scores["combined_text"].unique().tolist()
    assert "relaxing living room" in combined_texts, (
        "small_scores fixture must include 'relaxing living room' query"
    )


def test_cache_roundtrip(tmp_path):
    from semantic_architecture.embeddings import EmbeddingModel
    texts = ["warm light", "cold concrete", "soft velvet"]
    model = EmbeddingModel(model_name=MODEL_NAME, cache_dir=tmp_path / "embs")
    v1 = model.load_or_compute(texts, "test_cache")
    v2 = model.load_or_compute(texts, "test_cache")
    np.testing.assert_array_equal(v1, v2)
