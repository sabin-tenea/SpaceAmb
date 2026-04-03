# SpaceAmb
> **Semantic realization of ambiance-space pairs in architectural language.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

A Python research prototype for exploring how ambiance-space pairs — "relaxing living room", "fun cafeteria", "somber conference room", "spooky dance hall" — are semantically realized through architectural atoms.

## Research Purpose

The central question is:

> How can we computationally explore the way ambiance-program pairs get semantically realized in architectural language?

SpaceAmb approaches this by:

1. Defining a typed **atomic lexicon** of 232 architectural units (sofa, velvet, warm, indirect light, …)
2. Embedding atoms, spaces, ambiances, and combined phrases into a shared **semantic vector space** using sentence-transformers
3. Computing **discriminative scoring** to surface atoms uniquely relevant to a query, not just broadly similar
4. **Composing descriptors** from atoms via **grammar-constrained** similarity-weighted sampling
5. **Scoring full scene descriptions** (paragraphs) against the same query grid as atoms
6. Producing **ranked tables, matrices, and visualizations** for research inspection

### Methodological stance

Outputs are interpreted as *semantic affinity* and *latent associative structure* in embedding space. They reflect how the embedding model's "architectural imagination" organizes these concepts — not empirical claims about human perception.

---

## Data Model

### Atoms — typed minimal semantic units

The lexicon contains **232 atoms** across **12 families**:

| Family | Count | Examples |
|--------|-------|---------|
| architecture | 38 | wall, ceiling, arch, niche, stair, vault |
| material | 31 | concrete, velvet, copper, tadelakt, rammed earth |
| quality | 35 | warm, pristine, monumental, vernacular, translucent |
| spatial | 18 | open, enclosed, double-height, subterranean, labyrinthine |
| furniture | 16 | sofa, armchair, workbench, chaise longue, ottoman |
| color | 23 | terracotta, sage, deep blue, charcoal, ivory |
| lighting | 17 | indirect, diffuse, zenithal, raking, flickering candle |
| relation | 16 | inward-facing, axial, clustered, nested, radial |
| behavioral | 14 | quiet, retreat, gathering, active, focused |
| decoration | 10 | painting, rug, cushion, plant, sculpture |
| fixture | 8 | fireplace, bathtub, sink, radiator |
| technology | 6 | screen, projector, computer, speakers |

**Quality atoms** carry a `spectrums` field locating them on perceptual axes (temperature, scale, opacity, formality, etc.). **Lighting atoms** describe mode only — qualities like "warm" compose onto lighting modes via descriptors.

### Spaces (programs) — 20 types

living room, bedroom, library, laboratory, hospital room, cafeteria, conference room, dance hall, waiting room, gym, workshop, gallery, nursery, office, **chapel, spa, cinema, classroom, atrium, restaurant**.

### Ambiances — 20 terms

relaxing, inviting, fun, somber, spooky, formal, lively, intimate, sterile, restorative, playful, focused, melancholic, vibrant, **sacred, raw, opulent, serene, dramatic, nostalgic**.

Each ambiance carries a **disambiguation description** used during embedding (e.g. `"fun: playful laughter, light-hearted games, humour and jokes"` vs `"lively: animated social bustle, collective noise and movement"`) to separate near-synonyms in the vector space.

→ **400 combined queries** (20 × 20)

### Scenes — paragraph-length descriptions

18 hand-authored architectural scene descriptions, each targeting a specific space × ambiance pair. Scenes are embedded and scored against the same 400-query grid as atoms and descriptors, enabling direct comparison across all three granularities and ground-truth retrieval evaluation.

---

## Scoring

### Weighted similarity

For any item x and a (space, ambiance) target pair:

```
score(x | s, a) = 0.25 · sim(x, s)
               + 0.25 · sim(x, a)
               + 0.50 · sim(x, "{a} {s}")
```

Weights are configurable in `config/config.yaml`. The combined term gets 50% because it captures the joint meaning of the pair, not just the individual components.

### Discriminative scoring

The system defaults to **discriminative scoring** for rankings:

```
discriminative_score(x, q) = weighted_score(x, q) − mean(weighted_score(x, all_queries))
```

This subtracts each atom's baseline relevance (how much it scores on average across all queries) to penalize generically high-scoring atoms like "wall". Analogous to TF-IDF: a term is informative to the degree it is *more* present in this document than in the corpus.

A z-score variant (`zscore_score`) normalises by per-atom standard deviation for cross-atom comparison.

### Ambiance disambiguation

Near-synonym ambiances (fun/lively/vibrant, somber/melancholic, relaxing/restorative) are separated in the embedding space by attaching disambiguation descriptions to each term. The short label ("fun") is kept for display; the richer phrase is used for embedding:

```
fun   → "fun: playful laughter, light-hearted games, humour and jokes, joyful silliness"
lively → "lively: animated social bustle, collective noise and movement, crowd interaction"
vibrant → "vibrant: visually intense saturated colour, bold strong presence, vivid striking sensation"
```

This improved scene retrieval from **Hit@1: 44% → 78%**, Hit@5 from 89% → 100%.

---

## Descriptor Composition

Descriptors emerge from the embedding space via **grammar-constrained similarity-weighted sampling**:

1. **Pick a seed atom** from the full pool (or from the top-k atoms for a target query).
2. **Consult the architectural grammar** (`ALLOWED_COMBINATIONS` in `composition.py`) — defines which family pairs can logically combine (e.g. `architecture + material`, not `material + behavioral`).
3. **Compute softmax probabilities** over cosine similarity between seed and all valid partner atoms.
4. **Sample** the next 1–2 atoms with probability proportional to similarity / temperature.
5. **Join** atom texts into a phrase, recording full provenance.

**Temperature** controls exploration:
- `0.1` — near-deterministic; always picks the closest atom → tight, coherent phrases
- `1.0` — balanced (default)
- `3.0` — near-uniform → surprising cross-family combinations

### Query-conditioned generation

Blind generation can miss query-specific vocabulary because any atom can be the seed. With `--query`, seed selection is restricted to the top-k atoms for the given query:

```bash
python -m semantic_architecture.cli compose --query "fun cafeteria" --n 30 --temperature 0.8
```

This ensures every descriptor starts from a cafeteria- or fun-relevant atom, while the full pool remains available for subsequent positions.

---

## Project Structure

```
SpaceAmb/
├── data/
│   ├── raw/
│   │   ├── atoms.json          ← 232 typed atoms across 12 families
│   │   ├── programs.json       ← 20 space types (with disambiguation descriptions)
│   │   ├── ambiances.json      ← 20 ambiance terms (with disambiguation descriptions)
│   │   └── scenes.json         ← 18 narrative scene descriptions
│   └── processed/
│       └── embeddings/         ← .npy cache files (auto-created, safe to delete)
├── config/
│   └── config.yaml             ← all configuration (model, weights, paths)
├── src/
│   └── semantic_architecture/
│       ├── atoms.py            ← Atom dataclass + loading
│       ├── queries.py          ← Query dataclass + generation (embedding_text aware)
│       ├── embeddings.py       ← EmbeddingModel with disk cache
│       ├── scoring.py          ← cosine similarity, weighted scoring, discriminative enrichment
│       ├── composition.py      ← grammar-constrained generation + query-conditioned variant
│       ├── analysis.py         ← ranking + comparison workflows (atoms, descriptors, scenes)
│       ├── visualization.py    ← heatmaps, bar charts, PCA, UMAP (12-family colour palette)
│       ├── scenes.py           ← Scene dataclass + loading + validation
│       ├── io_utils.py         ← save/load helpers
│       ├── app_state.py        ← AppState: unified pipeline state
│       └── cli.py              ← Typer CLI
├── notebooks/
│   └── 01_prototype_demo.ipynb ← full demo (phases 1–3, discriminative scoring, scene eval)
├── tests/
│   └── test_sanity.py          ← 31 semantic + structural sanity tests
├── scripts/
│   └── run_analysis.py         ← CLI wrapper
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## Setup

```bash
# Install in editable mode (recommended for research iteration)
pip install -e ".[dev]"

# Or just install dependencies
pip install -r requirements.txt
```

The first run downloads the sentence-transformer model (~420 MB for `all-mpnet-base-v2`). Subsequent runs use the disk cache in `data/processed/embeddings/`. Delete that directory to force recomputation (required when changing the model).

---

## Quick Start

### CLI

```bash
# Dataset statistics
python -m semantic_architecture.cli info

# Score atoms for a query (discriminative ranking by default)
python -m semantic_architecture.cli score "relaxing living room" --top-k 20

# Score descriptors
python -m semantic_architecture.cli score "somber conference room" --items descriptors --top-k 15

# Score scenes (phase 3)
python -m semantic_architecture.cli score "formal library" --items scenes --top-k 5

# Score everything at once
python -m semantic_architecture.cli score "spooky gallery" --items all --top-k 10

# Override the scoring column
python -m semantic_architecture.cli score "fun cafeteria" --score weighted_score

# Generate descriptors (global, blind)
python -m semantic_architecture.cli compose --n 50 --temperature 0.5

# Generate descriptors seeded from top atoms for a specific query
python -m semantic_architecture.cli compose --query "fun cafeteria" --n 30 --temperature 0.8

# Run full pipeline and export everything to data/processed/
python -m semantic_architecture.cli export
```

### Notebook

`notebooks/01_prototype_demo.ipynb` walks through the full pipeline interactively:

1. Load atoms, programs, ambiances
2. Compute embeddings (cached)
3. Score atoms with discriminative enrichment
4. Generate and score descriptors
5. Export CSVs and figures
6. Visualizations: heatmap, bar chart, PCA, UMAP, family comparison
7. **Phase 3**: load scenes, embed, score, ground-truth Hit@1/Hit@5 evaluation

---

## Configuration

```yaml
# config/config.yaml

embedding:
  model_name: "all-mpnet-base-v2"   # 768-dim, strong semantic separation
  cache_dir: "data/processed/embeddings"
  batch_size: 64

scoring:
  weights:
    space: 0.25
    ambiance: 0.25
    combined: 0.50
  default_score_col: "discriminative_score"   # or weighted_score / zscore_score

composition:
  n_descriptors: 300
  descriptor_lengths: [2, 3]
  temperature: 1.0
  seed: 42

data:
  atoms_path: "data/raw/atoms.json"
  programs_path: "data/raw/programs.json"
  ambiances_path: "data/raw/ambiances.json"
  scenes_path: "data/raw/scenes.json"
```

---

## Research Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ | Atom embedding + discriminative scoring against 400 space × ambiance queries |
| 2 | ✅ | Grammar-constrained descriptor composition + query-conditioned variant |
| 3 | ✅ | Scene scoring + ground-truth retrieval evaluation (Hit@1: 78%, Hit@5: 100%) |

---

## Tests

```bash
pytest tests/ -v
```

31 tests covering:

- **Data integrity**: no duplicate atom IDs or texts, all 12 families present, quality atoms have spectrums
- **Semantic rankings**: sofa > workbench for living room; warm > pristine for relaxing; formal > fun for formal queries
- **Discriminative scoring**: mean near zero per atom, differs from weighted ranking
- **Composition**: correct count, valid provenance, low-temperature bias toward nearest neighbour
- **Scenes**: JSON validity, field completeness, no duplicate IDs, space/ambiance cross-reference, intended query retrieval
- **Scoring utilities**: cosine similarity shape, self-similarity = 1, weighted formula
- **Cache**: round-trip correctness
