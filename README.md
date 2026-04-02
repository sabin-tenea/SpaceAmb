# SpaceAmb
> **Semantic realization of ambiance-space pairs in architectural language.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

A Python research prototype for exploring how ambiance-space pairs — "relaxing living room", "fun cafeteria", "somber conference room", "spooky dance hall" — are semantically realized through architectural atoms.

## Research Purpose

The central question is:

> How can we computationally explore the way ambiance-program pairs — "relaxing living room", "fun cafeteria", "somber conference room", "spooky dance hall" — get semantically realized in architectural language?

SpaceAmb approaches this by:

1. Defining a typed **atomic lexicon** of architectural units (sofa, velvet, warm, inward-facing, …)
2. Embedding atoms, spaces, ambiances, and combined phrases into a shared **semantic vector space** using sentence-transformers
3. Computing **discriminative scoring** (raw similarity minus mean) to find atoms that uniquely characterize a query
4. **Composing descriptors** from atoms using **grammar-constrained** similarity-weighted random sampling
5. Producing **ranked tables, matrices, and visualizations** for research inspection

### Methodological stance

Outputs are interpreted as *semantic affinity* and *latent associative structure* in embedding space.  They reflect how the embedding model's "architectural imagination" organizes these concepts — not empirical claims about human perception.

---

## Data Model

### Atoms — typed minimal semantic units

The lexicon contains **230+ atoms** across 12 families:

| Family | Examples |
|--------|---------|
| architecture | wall, ceiling, arch, portal, niche, atrium |
| furniture | sofa, armchair, desk, workbench, bookshelf |
| fixture | fireplace, sink, radiator, built-in shelf |
| decoration | painting, plant, rug, sculpture, curtain |
| technology | screen, computer, projector, speakers |
| material | wood, concrete, velvet, steel, marble, copper |
| color | deep blue, beige, terracotta, warm white |
| quality | warm, dim, rough, soft, muted, aged (multi-dimensional) |
| lighting | indirect, direct, diffuse, ambient (mode only) |
| spatial | open, enclosed, tall, compact, expansive, mezzanine |
| relation | inward-facing, clustered, dispersed, axial |
| behavioral | quiet, active, gathering, retreat, focused |

**Quality atoms** are special: they carry a `spectrums` field that locates them on perceptual axes (temperature, luminosity, texture, weight, formality, saturation, patina). This lets you filter or cluster by perceptual dimension.

**Lighting atoms** describe *mode only* (how light travels/distributes). Qualities like "warm" or "dim" attach to lighting modes through composition, allowing emergent phrases like "warm indirect" or "dim diffuse".

### Spaces (programs)

14 types: living room, bedroom, library, laboratory, hospital room, cafeteria, conference room, dance hall, waiting room, gym, workshop, gallery, nursery, office.

### Ambiances

14 terms: relaxing, inviting, fun, somber, spooky, formal, lively, intimate, sterile, restorative, playful, focused, melancholic, vibrant.

---

## Scoring Formula

For any item x and a (space, ambiance) target pair:

```
score(x | s, a) = w_space * sim(x, s)
               + w_ambiance * sim(x, a)
               + w_combined * sim(x, "{a} {s}")
```

Default weights: `space=0.25, ambiance=0.25, combined=0.50`.
Configurable in `config/config.yaml`.

The system defaults to **discriminative scoring** for rankings: `score(x, query) - mean(score(x, all_queries))`. This highlights atoms that are uniquely similar to the specific query, rather than broadly high-scoring (e.g. "wall").

---

## Descriptor Composition

Descriptors emerge from the embedding space via **Grammar-Constrained Similarity-Weighted Sampling**:

1. **Pick a seed atom** at random (optionally stratified by family).
2. **Consult the Architectural Grammar** (`ALLOWED_COMBINATIONS` in `composition.py`) to see which families can logically pair with the seed (e.g., `material` + `architecture`, but not `material` + `behavioral`).
3. **Compute probabilities**: Cosine similarity of the seed to all *valid* remaining atoms, converted via `softmax(sim / temperature)`.
4. **Sample**: Pick the next 1–2 atoms from that distribution.
5. **Join**: Selected atom texts are joined into a phrase.

**Temperature** controls exploration:
- `0.1` — near-deterministic; always picks the most similar atom
- `1.0` — balanced (default)
- `3.0` — near-uniform; any combination is possible

---

## Project Structure

```
SpaceAmb/
├── data/
│   ├── raw/
│   │   ├── atoms.json          ← 230+ typed atoms across 12 families
│   │   ├── programs.json       ← 14 space/program types
│   │   ├── ambiances.json      ← 14 ambiance terms
│   │   └── scenes.json         ← 18 narrative scene descriptions (Phase 3)
│   └── processed/
│       └── embeddings/         ← .npy cache files (auto-created)
├── config/
│   └── config.yaml             ← all configuration (model, weights, paths)
├── src/
│   └── semantic_architecture/
│       ├── atoms.py            ← Atom dataclass + loading
│       ├── queries.py          ← Query dataclass + generation
│       ├── embeddings.py       ← EmbeddingModel with disk cache
│       ├── scoring.py          ← cosine similarity + weighted scoring
│       ├── composition.py      ← grammar-constrained generation
│       ├── analysis.py         ← ranking + comparison workflows
│       ├── visualization.py    ← heatmaps, bar charts, PCA, UMAP
│       ├── io_utils.py         ← save/load helpers
│       ├── app_state.py        ← AppState: unified pipeline state
│       └── cli.py              ← Typer CLI
├── notebooks/
│   └── 01_prototype_demo.ipynb ← full demo walkthrough
├── tests/
│   └── test_sanity.py          ← semantic + structural sanity checks
├── scripts/
│   ├── run_analysis.py         ← CLI wrapper
│   └── make_notebook.py        ← notebook generator
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## Setup

### Requirements

- Python 3.10+
- The packages in `requirements.txt`

```bash
# Install in editable mode (recommended for research iteration)
pip install -e ".[dev]"

# Or just install dependencies
pip install -r requirements.txt
```

The first run downloads the sentence-transformer model (~420 MB for `all-mpnet-base-v2`). Subsequent runs use the disk cache.

---

## Quick Start

### CLI

```bash
# Score atoms against a query
python -m semantic_architecture.cli score "relaxing living room" --top-k 20

# Score using a specific metric (weighted, discriminative, zscore)
python -m semantic_architecture.cli score "fun cafeteria" --score-col weighted_score

# Generate descriptors and print examples
python -m semantic_architecture.cli compose --n 50 --temperature 0.5

# Run full pipeline + export everything to data/processed/
python -m semantic_architecture.cli export

# Print dataset statistics (atom counts, scenes, etc.)
python -m semantic_architecture.cli info
```

---

## Configuration

All settings live in `config/config.yaml`:

```yaml
embedding:
  model_name: "all-mpnet-base-v2"   # higher quality semantic separation
  cache_dir: "data/processed/embeddings"

scoring:
  default_score_col: "discriminative_score" 
  weights:
    space: 0.25
    ambiance: 0.25
    combined: 0.50

composition:
  n_descriptors: 300
  descriptor_lengths: [2, 3]
  temperature: 1.0
  seed: 42
```

---

## Research Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Implemented | Atom embedding + similarity scoring against spaces/ambiances/combined |
| 2 | ✅ Implemented | Descriptor composition (Grammar-constrained) + scoring |
| 3 | ✅ Implemented | Scene analysis (narrative descriptions embedded and scored) |

---

## Tests

```bash
pytest tests/ -v
```

The sanity tests verify:
- Semantic rankings make intuitive sense ("sofa" > "workbench" for "living room")
- Quality atoms have `spectrums` dicts; non-quality atoms have `None`
- Composition respects the grammar and produces the right count of unique descriptors
- Provenance is fully traceable
- Embedding cache round-trips correctly
