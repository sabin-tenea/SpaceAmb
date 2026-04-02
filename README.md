# SpaceAmb

A Python research prototype for exploring the semantic realization of ambiance-space pairs in architectural language.

---

## Research Purpose

The central question is:

> How can we computationally explore the way ambiance-program pairs — "relaxing living room", "fun cafeteria", "somber conference room", "spooky dance hall" — get semantically realized in architectural language?

SpaceAmb approaches this by:

1. Defining a typed **atomic lexicon** of architectural units (sofa, velvet, warm, inward-facing, …)
2. Embedding atoms, spaces, ambiances, and combined phrases into a shared **semantic vector space** using sentence-transformers
3. Computing **cosine similarity** between atoms/descriptors and target queries
4. **Composing descriptors** from atoms using similarity-weighted random sampling
5. Producing **ranked tables, matrices, and visualizations** for research inspection

### Methodological stance

Outputs are interpreted as *semantic affinity* and *latent associative structure* in embedding space.  They reflect how the embedding model's "architectural imagination" organizes these concepts — not empirical claims about human perception.

---

## Data Model

### Atoms — typed minimal semantic units

| Family | Examples |
|--------|---------|
| object | sofa, workbench, fireplace, hospital bed |
| material | wood, concrete, velvet, steel, marble |
| color | deep blue, beige, terracotta, warm white |
| quality | warm, dim, rough, soft, muted, aged (multi-dimensional) |
| lighting | indirect, direct, diffuse, ambient (mode only) |
| spatial | open, enclosed, tall, compact, expansive |
| relation | inward-facing, clustered, dispersed |
| behavioral | quiet, active, gathering, retreat |

**Quality atoms** are special: they carry a `spectrums` field that locates them on perceptual axes (temperature, luminosity, texture, weight, formality, saturation, patina). This lets you filter or cluster by perceptual dimension.

**Lighting atoms** describe *mode only* (how light travels/distributes). Qualities like "warm" or "dim" attach to lighting modes through composition, allowing emergent phrases like "warm indirect" or "dim diffuse".

### Spaces (programs)

14 types: living room, bedroom, library, laboratory, hospital room, cafeteria, conference room, dance hall, waiting room, gym, workshop, gallery, nursery, office.

### Ambiances

14 terms: relaxing, inviting, fun, somber, spooky, formal, lively, intimate, sterile, restorative, playful, focused, melancholic, vibrant.

### Combined queries

Programmatically generated as `"{ambiance} {space}"` — 14 × 14 = 196 pairs.

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

---

## Descriptor Composition

Descriptors are not hand-authored.  They emerge from the embedding space via **similarity-weighted random sampling**:

1. Pick a seed atom at random
2. For each additional slot, compute cosine similarity of the seed to all remaining atoms
3. Convert similarities to sampling probabilities via `softmax(sim / temperature)`
4. Sample the next atom from that distribution
5. Join selected atom texts with spaces

**Temperature** controls exploration:
- `0.1` — near-deterministic; always picks the most similar atom
- `1.0` — balanced (default)
- `3.0` — near-uniform; any combination is possible

Every descriptor stores full **provenance**: source atom IDs, families, and pairwise similarities.

---

## Project Structure

```
SpaceAmb/
├── data/
│   ├── raw/
│   │   ├── atoms.json          ← 70+ typed atoms across 8 families
│   │   ├── programs.json       ← 14 space/program types
│   │   └── ambiances.json      ← 14 ambiance terms
│   └── processed/
│       └── embeddings/         ← .npy cache files (auto-created)
├── config/
│   └── config.yaml             ← all configuration
├── src/
│   └── semantic_architecture/
│       ├── atoms.py            ← Atom dataclass + loading
│       ├── queries.py          ← Query dataclass + generation
│       ├── embeddings.py       ← EmbeddingModel with disk cache
│       ├── scoring.py          ← cosine similarity + weighted scoring
│       ├── composition.py      ← similarity-weighted descriptor generation
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

The first run downloads the sentence-transformer model (~90 MB for `all-MiniLM-L6-v2`).  Subsequent runs use the disk cache.

---

## Quick Start

### CLI

```bash
# Score atoms against a query
python -m semantic_architecture.cli score "relaxing living room" --top-k 20

# Score descriptors instead
python -m semantic_architecture.cli score "somber conference room" --items descriptors

# Score both atoms and descriptors
python -m semantic_architecture.cli score "fun cafeteria" --items both

# Generate descriptors and print examples
python -m semantic_architecture.cli compose --n 50 --temperature 0.5

# Run full pipeline + export everything to data/processed/
python -m semantic_architecture.cli export

# Print dataset statistics
python -m semantic_architecture.cli info
```

### Notebook

Open `notebooks/01_prototype_demo.ipynb` in Jupyter and run all cells.  The notebook covers the complete pipeline with plots.

### Python API

```python
from semantic_architecture.app_state import AppState

# Load everything (runs the pipeline; uses cache where available)
state = AppState.load("config/config.yaml")

# Top 20 atoms for a combined query
from semantic_architecture.analysis import top_atoms_for_query
q = state.get_query("relaxing living room")
top20 = top_atoms_for_query(q.id, state.atom_scores, k=20)

# How does 'sofa' score across all queries?
from semantic_architecture.analysis import compare_item_across_queries
sofa_cross = compare_item_across_queries("sofa", state.atom_scores)

# Export all outputs
state.export_all()
```

---

## Configuration

All settings live in `config/config.yaml`:

```yaml
embedding:
  model_name: "all-MiniLM-L6-v2"   # swap to all-mpnet-base-v2 for quality
  cache_dir: "data/processed/embeddings"

scoring:
  weights:
    space: 0.25
    ambiance: 0.25
    combined: 0.50          # combined phrase carries the most weight

composition:
  n_descriptors: 300
  descriptor_lengths: [2, 3]
  temperature: 1.0          # lower = tighter semantic clusters
  seed: 42
```

---

## Extending the System

### Add atoms

Edit `data/raw/atoms.json`.  Add entries with any `family` in:
`object | material | color | quality | lighting | spatial | relation | behavioral`.

Quality atoms should include a `spectrums` field:
```json
{ "id": "qua_crisp", "text": "crisp", "family": "quality",
  "subtype": "texture", "spectrums": {"texture": "sharp"}, "notes": null }
```

Delete the embedding cache files (`data/processed/embeddings/atoms.*`) to force recomputation.

### Add spaces or ambiances

Edit `data/raw/programs.json` or `data/raw/ambiances.json`.  Combined queries are regenerated automatically.

### Change embedding model

Set `embedding.model_name` in `config.yaml` to any [sentence-transformers model](https://www.sbert.net/docs/pretrained_models.html).  Delete the cache directory to recompute.

### Adjust scoring weights

Change `scoring.weights` in `config.yaml`.  Setting `combined: 1.0` and others to `0.0` gives pure combined-phrase similarity.

### Adjust composition temperature

Lower temperature (e.g. `0.3`) produces tighter, more expected pairs.  Higher (e.g. `2.0`) produces more surprising cross-family combinations.  Both are interesting for research.

### Add a new analysis

Import from `semantic_architecture.analysis` or extend it.  All functions operate on DataFrames and return DataFrames, so they compose naturally.

### Build an interactive interface

`AppState.load()` returns a fully populated state object designed for a future Streamlit or Gradio app.  A minimal starting point:

```python
# app.py (future Streamlit interface)
import streamlit as st
from semantic_architecture.app_state import AppState

@st.cache_resource
def load():
    return AppState.load()

state = load()
query = st.selectbox("Query", state.combined_query_texts)
# ... display rankings, heatmaps, etc.
```

---

## Tests

```bash
pytest tests/ -v
```

The sanity tests verify:
- Semantic rankings make intuitive sense ("sofa" > "workbench" for "living room")
- Quality atoms have `spectrums` dicts; non-quality atoms have `None`
- Composition produces the right count of unique descriptors
- Provenance is fully traceable
- Embedding cache round-trips correctly

---

## Research Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Implemented | Atom embedding + similarity scoring against spaces/ambiances/combined |
| 2 | ✅ Implemented | Descriptor composition + scoring |
| 3 | 🔲 Scaffold ready | Scene descriptions (same pipeline, longer texts) |

Phase 3 requires only: (a) a list of scene description strings, (b) embedding them with `model.load_or_compute(descriptions, "scenes")`, and (c) running `score_items_against_queries` as for atoms and descriptors.
