"""Generate the demo notebook programmatically."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "01_prototype_demo.ipynb"

cells = []


def md(src):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": src if isinstance(src, list) else [src],
    })


def code(src):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src if isinstance(src, list) else [src],
    })


# ── Title ──────────────────────────────────────────────────────────────────
md(
    "# SpaceAmb — Prototype Demo\n\n"
    "Architectural-semantic research prototype.  "
    "This notebook demonstrates the full pipeline:\n\n"
    "1. Load atoms, programs, ambiances\n"
    "2. Generate embeddings (with disk cache)\n"
    "3. Score atoms against example queries\n"
    "4. Generate descriptors via grammar-constrained similarity-weighted composition\n"
    "5. Score descriptors\n"
    "6. Export all outputs\n"
    "7. Plots: heatmap · bar chart · PCA · (optional UMAP)\n\n"
    "> **Methodological note:** Similarity scores reflect *semantic affinity* in\n"
    "> embedding space — they are a research instrument, not empirical perceptual ground truth."
)

# ── 0. Setup ───────────────────────────────────────────────────────────────
md("## 0. Setup")
code(
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "# When running from notebooks/, project root is one level up\n"
    "ROOT = Path('.').resolve().parent\n"
    "src_path = ROOT / 'src'\n"
    "if str(src_path) not in sys.path:\n"
    "    sys.path.insert(0, str(src_path))\n"
    "\n"
    "import matplotlib\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "matplotlib.rcParams['figure.dpi'] = 120\n"
    "%matplotlib inline"
)

# ── 1. Load Config and Datasets ────────────────────────────────────────────
md("## 1. Load Config and Datasets")
code(
    "import yaml\n"
    "from semantic_architecture.atoms import load_atoms, atoms_summary\n"
    "from semantic_architecture.queries import (\n"
    "    load_programs, load_ambiances, generate_all_queries\n"
    ")\n"
    "\n"
    "CONFIG_PATH = ROOT / 'config' / 'config.yaml'\n"
    "with open(CONFIG_PATH) as fh:\n"
    "    cfg = yaml.safe_load(fh)\n"
    "\n"
    "atoms     = load_atoms(ROOT / cfg['data']['atoms_path'])\n"
    "programs  = load_programs(ROOT / cfg['data']['programs_path'])\n"
    "ambiances = load_ambiances(ROOT / cfg['data']['ambiances_path'])\n"
    "\n"
    "print(atoms_summary(atoms))\n"
    "print(f\"\\nPrograms : {[p['text'] for p in programs]}\")\n"
    "print(f\"Ambiances: {[a['text'] for a in ambiances]}\")"
)

# ── 2. Embeddings ──────────────────────────────────────────────────────────
md("## 2. Generate Embeddings (Cached)")
code(
    "from semantic_architecture.embeddings import EmbeddingModel\n"
    "\n"
    "emb_cfg = cfg['embedding']\n"
    "model = EmbeddingModel(\n"
    "    model_name=emb_cfg['model_name'],\n"
    "    cache_dir=ROOT / emb_cfg['cache_dir'],\n"
    "    batch_size=emb_cfg['batch_size'],\n"
    ")\n"
    "\n"
    "query_sets    = generate_all_queries(programs, ambiances)\n"
    "space_qs      = query_sets['space']\n"
    "ambiance_qs   = query_sets['ambiance']\n"
    "combined_qs   = query_sets['combined']\n"
    "\n"
    "atom_texts     = [a.text for a in atoms]\n"
    "atom_families  = [a.family for a in atoms]\n"
    "atom_ids       = [a.id for a in atoms]\n"
    "\n"
    "# Each collection is embedded and cached independently\n"
    "atom_embs      = model.load_or_compute(atom_texts,                               'atoms')\n"
    "space_embs     = model.load_or_compute([q.combined_text for q in space_qs],      'queries_space')\n"
    "ambiance_embs  = model.load_or_compute([q.combined_text for q in ambiance_qs],   'queries_ambiance')\n"
    "combined_embs  = model.load_or_compute([q.combined_text for q in combined_qs],   'queries_combined')\n"
    "\n"
    "print(f'Atom embedding shape    : {atom_embs.shape}')\n"
    "print(f'Combined embedding shape: {combined_embs.shape}')"
)

# ── 3. Score Atoms ─────────────────────────────────────────────────────────
md("## 3. Score Atoms Against Example Queries")
code(
    "from semantic_architecture.scoring import (\n"
    "    ScoringWeights, score_items_against_queries\n"
    ")\n"
    "from semantic_architecture.analysis import (\n"
    "    top_atoms_for_query, top_atoms_by_family, compare_item_across_queries\n"
    ")\n"
    "\n"
    "weights = ScoringWeights.from_config(cfg.get('scoring', {}))\n"
    "\n"
    "atom_scores = score_items_against_queries(\n"
    "    item_texts=atom_texts, item_families=atom_families, item_ids=atom_ids,\n"
    "    item_embeddings=atom_embs,\n"
    "    space_queries=space_qs, ambiance_queries=ambiance_qs,\n"
    "    combined_queries=combined_qs,\n"
    "    space_embeddings=space_embs, ambiance_embeddings=ambiance_embs,\n"
    "    combined_embeddings=combined_embs,\n"
    "    weights=weights,\n"
    ")\n"
    "print(f'Score table shape: {atom_scores.shape}')\n"
    "atom_scores.head(3)"
)

code(
    "# Identify the four example query IDs\n"
    "EXAMPLE_QUERIES = [\n"
    "    'relaxing living room',\n"
    "    'fun cafeteria',\n"
    "    'somber conference room',\n"
    "    'spooky dance hall',\n"
    "]\n"
    "\n"
    "ex_ids = {\n"
    "    q.combined_text: q.id\n"
    "    for q in combined_qs\n"
    "    if q.combined_text in EXAMPLE_QUERIES\n"
    "}\n"
    "print('Matched query IDs:')\n"
    "for k, v in ex_ids.items():\n"
    "    print(f'  {v}: {k}')"
)

code(
    "# Top 20 atoms for 'relaxing living room'\n"
    "q_text = 'relaxing living room'\n"
    "top20 = top_atoms_for_query(ex_ids[q_text], atom_scores, k=20)\n"
    "print(f'Top 20 atoms for \"{q_text}\":')\n"
    "top20"
)

code(
    "# Top atoms by family for 'somber conference room'\n"
    "q_text = 'somber conference room'\n"
    "by_fam = top_atoms_by_family(ex_ids[q_text], atom_scores, k=5)\n"
    "for fam, df in sorted(by_fam.items()):\n"
    "    print(f'\\n--- {fam} ---')\n"
    "    print(df.to_string(index=False))"
)

code(
    "# Cross-query: how does 'sofa' score across all 196 queries?\n"
    "sofa_cross = compare_item_across_queries('sofa', atom_scores)\n"
    "print(\"Top-10 queries for 'sofa':\")\n"
    "print(sofa_cross.head(10)[['combined_text', 'sim_space', 'sim_ambiance',\n"
    "                            'sim_combined', 'weighted_score']].to_string(index=False))\n"
    "print('\\nBottom-5 queries:')\n"
    "print(sofa_cross.tail(5)[['combined_text', 'weighted_score']].to_string(index=False))"
)

# ── 4. Descriptors ─────────────────────────────────────────────────────────
md("## 4. Generate Descriptors via Grammar-Constrained Similarity-Weighted Composition")
code(
    "from semantic_architecture.composition import generate_descriptors, descriptors_to_df\n"
    "\n"
    "comp_cfg = cfg.get('composition', {})\n"
    "descriptors = generate_descriptors(\n"
    "    atoms=atoms,\n"
    "    atom_embeddings=atom_embs,\n"
    "    n_descriptors=comp_cfg.get('n_descriptors', 300),\n"
    "    descriptor_lengths=comp_cfg.get('descriptor_lengths', [2, 3]),\n"
    "    temperature=comp_cfg.get('temperature', 1.0),\n"
    "    seed=comp_cfg.get('seed', 42),\n"
    ")\n"
    "\n"
    "desc_df = descriptors_to_df(descriptors)\n"
    "print(f'Generated {len(descriptors)} unique descriptors')\n"
    "print(f'\\nFamily patterns (top 15):')\n"
    "print(desc_df['family_pattern'].value_counts().head(15).to_string())\n"
    "desc_df[['text', 'family_pattern', 'n_atoms']].head(25)"
)

code(
    "# Show full provenance for first 5 descriptors\n"
    "for d in descriptors[:5]:\n"
    "    print(d.provenance_str())\n"
    "    print()"
)

# ── 5. Score Descriptors ───────────────────────────────────────────────────
md("## 5. Score Descriptors")
code(
    "desc_texts    = [d.text for d in descriptors]\n"
    "desc_embs     = model.load_or_compute(desc_texts, 'descriptors')\n"
    "desc_families = [d.source_atom_families[0] for d in descriptors]\n"
    "desc_ids      = [d.id for d in descriptors]\n"
    "\n"
    "descriptor_scores = score_items_against_queries(\n"
    "    item_texts=desc_texts, item_families=desc_families, item_ids=desc_ids,\n"
    "    item_embeddings=desc_embs,\n"
    "    space_queries=space_qs, ambiance_queries=ambiance_qs,\n"
    "    combined_queries=combined_qs,\n"
    "    space_embeddings=space_embs, ambiance_embeddings=ambiance_embs,\n"
    "    combined_embeddings=combined_embs,\n"
    "    weights=weights,\n"
    ")\n"
    "print(f'Descriptor score table shape: {descriptor_scores.shape}')"
)

code(
    "from semantic_architecture.analysis import top_descriptors_for_query\n"
    "\n"
    "for q_text in EXAMPLE_QUERIES:\n"
    "    if q_text not in ex_ids:\n"
    "        continue\n"
    "    top_d = top_descriptors_for_query(ex_ids[q_text], descriptor_scores, k=15)\n"
    "    print(f'\\nTop 15 descriptors for \"{q_text}\":')\n"
    "    print(top_d[['rank', 'text', 'family', 'weighted_score']].to_string(index=False))"
)

# ── 6. Export ──────────────────────────────────────────────────────────────
md("## 6. Export All Results")
code(
    "from semantic_architecture.io_utils import save_csv, save_json\n"
    "\n"
    "out_dir = ROOT / cfg['output']['dir']\n"
    "out_dir.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "save_csv(atom_scores,       out_dir / 'atom_scores.csv',       'atom scores')\n"
    "save_csv(descriptor_scores, out_dir / 'descriptor_scores.csv', 'descriptor scores')\n"
    "save_csv(desc_df,           out_dir / 'descriptors.csv',       'descriptor table')\n"
    "\n"
    "# Per-query rankings\n"
    "for q_text, qid in ex_ids.items():\n"
    "    safe = q_text.replace(' ', '_')\n"
    "    save_csv(\n"
    "        top_atoms_for_query(qid, atom_scores, k=30),\n"
    "        out_dir / f'ranking_atoms_{safe}.csv'\n"
    "    )\n"
    "    save_csv(\n"
    "        top_descriptors_for_query(qid, descriptor_scores, k=30),\n"
    "        out_dir / f'ranking_descriptors_{safe}.csv'\n"
    "    )\n"
    "\n"
    "print('Export complete.')"
)

# ── 7. Visualizations ──────────────────────────────────────────────────────
md("## 7. Visualizations")

code(
    "from semantic_architecture.visualization import (\n"
    "    plot_heatmap, plot_top_items, plot_pca_projection,\n"
    "    plot_umap_projection, plot_family_comparison,\n"
    ")"
)

code(
    "# --- 7a. Heatmap: top-25 atoms × 4 example queries ---\n"
    "from semantic_architecture.scoring import similarity_matrix_df\n"
    "\n"
    "# Gather embeddings for the 4 example combined queries\n"
    "ex_combined_embs = []\n"
    "ex_labels = []\n"
    "for qt in EXAMPLE_QUERIES:\n"
    "    idx = next(i for i, q in enumerate(combined_qs) if q.combined_text == qt)\n"
    "    ex_combined_embs.append(combined_embs[idx])\n"
    "    ex_labels.append(qt)\n"
    "ex_combined_embs = np.stack(ex_combined_embs)\n"
    "\n"
    "# Select top-25 atoms by mean weighted score across these 4 queries\n"
    "ex_qids = list(ex_ids.values())\n"
    "mean_scores = (\n"
    "    atom_scores[atom_scores['query_id'].isin(ex_qids)]\n"
    "    .groupby('text')['weighted_score'].mean()\n"
    "    .sort_values(ascending=False)\n"
    "    .head(25)\n"
    ")\n"
    "top25_texts = list(mean_scores.index)\n"
    "top25_idx   = [atom_texts.index(t) for t in top25_texts]\n"
    "top25_embs  = atom_embs[top25_idx]\n"
    "\n"
    "mat = similarity_matrix_df(top25_texts, ex_labels, top25_embs, ex_combined_embs)\n"
    "fig = plot_heatmap(mat, title='Top-25 atoms vs 4 example queries',\n"
    "                   figsize=(10, 10), annot=True)\n"
    "fig.savefig(out_dir / 'heatmap_atoms_4queries.png', bbox_inches='tight', dpi=150)\n"
    "plt.show()"
)

code(
    "# --- 7b. Bar chart: top 20 atoms for 'relaxing living room' ---\n"
    "q_text = 'relaxing living room'\n"
    "top20  = top_atoms_for_query(ex_ids[q_text], atom_scores, k=20)\n"
    "fig = plot_top_items(\n"
    "    top20, query_label=q_text, top_k=20,\n"
    "    score_col='weighted_score', color_by_family=True,\n"
    ")\n"
    "fig.savefig(out_dir / 'bar_atoms_relaxing_living_room.png', bbox_inches='tight', dpi=150)\n"
    "plt.show()"
)

code(
    "# --- 7c. PCA of all atoms ---\n"
    "fig = plot_pca_projection(\n"
    "    atom_embs, atom_texts, families=atom_families,\n"
    "    title='PCA of all atoms (coloured by family)',\n"
    "    annotate=True, figsize=(12, 9),\n"
    ")\n"
    "fig.savefig(out_dir / 'pca_atoms.png', bbox_inches='tight', dpi=150)\n"
    "plt.show()"
)

code(
    "# --- 7d. UMAP (skipped if umap-learn not installed) ---\n"
    "fig = plot_umap_projection(\n"
    "    atom_embs, atom_texts, families=atom_families,\n"
    "    title='UMAP of all atoms (coloured by family)',\n"
    "    annotate=True, figsize=(12, 9),\n"
    ")\n"
    "if fig:\n"
    "    fig.savefig(out_dir / 'umap_atoms.png', bbox_inches='tight', dpi=150)\n"
    "    plt.show()"
)

code(
    "# --- 7e. Family comparison across 4 queries ---\n"
    "fig = plot_family_comparison(\n"
    "    atom_scores, query_ids=ex_qids,\n"
    "    score_col='weighted_score', figsize=(14, 6),\n"
    ")\n"
    "fig.savefig(out_dir / 'family_comparison_4queries.png', bbox_inches='tight', dpi=150)\n"
    "plt.show()"
)

md(
    "## Summary\n\n"
    "All outputs have been saved to `data/processed/`:\n\n"
    "| File | Contents |\n"
    "|------|----------|\n"
    "| `atom_scores.csv` | Full scoring table (atoms × all 196 queries) |\n"
    "| `descriptor_scores.csv` | Full scoring table (descriptors × all 196 queries) |\n"
    "| `descriptors.csv` | Descriptor list with provenance |\n"
    "| `ranking_atoms_*.csv` | Per-query atom rankings |\n"
    "| `ranking_descriptors_*.csv` | Per-query descriptor rankings |\n"
    "| `heatmap_atoms_4queries.png` | Similarity heatmap |\n"
    "| `bar_atoms_relaxing_living_room.png` | Bar chart |\n"
    "| `pca_atoms.png` | PCA projection |\n\n"
    "> **Next steps:** Adjust `config/config.yaml` to change the embedding model,\n"
    "> composition temperature, or scoring weights, then re-run this notebook."
)

# ── Build notebook JSON ────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with NB_PATH.open("w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)

print(f"Notebook written to {NB_PATH}")
