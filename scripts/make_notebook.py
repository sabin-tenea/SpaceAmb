"""Generate the SpaceAmb demo notebook programmatically.

Run from the project root:
    python scripts/make_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "01_prototype_demo.ipynb"

cells = []
_cell_id_counter = [0]


def _next_id():
    _cell_id_counter[0] += 1
    return f"{_cell_id_counter[0]:08x}"


def md(src):
    cells.append({
        "cell_type": "markdown",
        "id": _next_id(),
        "metadata": {},
        "source": src if isinstance(src, list) else [src],
    })


def code(src):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": _next_id(),
        "metadata": {},
        "outputs": [],
        "source": src if isinstance(src, list) else [src],
    })


# ── Title ──────────────────────────────────────────────────────────────────
md(
    "# SpaceAmb — Prototype Demo\n\n"
    "Architectural-semantic research prototype.  Full pipeline:\n\n"
    "1. Load atoms, programs, ambiances, scenes\n"
    "2. Generate embeddings (disk-cached)\n"
    "3. Score atoms, descriptors, and scenes against all space × ambiance queries\n"
    "4. Explore: change settings in **Section 0b** to point at any query or scene\n"
    "5. Evaluate: Hit@1 / Hit@5 ground-truth metrics for scene retrieval\n"
    "6. Export and visualise\n\n"
    "> **Methodological note:** Similarity scores reflect *semantic affinity* in\n"
    "> embedding space — a research instrument, not perceptual ground truth."
)

# ── 0. Setup ───────────────────────────────────────────────────────────────
md("## 0. Setup")
code(
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
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

# ── 0b. Settings ───────────────────────────────────────────────────────────
md(
    "## 0b. Settings  <-- **edit here to explore different queries & scenes**\n\n"
    "Change `EXPLORE_SPACE` and `EXPLORE_AMBIANCE` to drill into any combination.\n"
    "`EXAMPLE_QUERIES` drives the comparison charts and exports throughout the notebook.\n\n"
    "Valid values are any `text` entries in `data/raw/programs.json` / `data/raw/ambiances.json`."
)
code(
    "# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "#  PRIMARY EXPLORATION TARGET\n"
    "# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "# Space (program) — must match a text value in data/raw/programs.json\n"
    "EXPLORE_SPACE    = 'living room'\n"
    "# Ambiance — must match a text value in data/raw/ambiances.json\n"
    "EXPLORE_AMBIANCE = 'relaxing'\n"
    "\n"
    "# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "#  COMPARISON QUERIES (used in charts, exports, heatmap)\n"
    "# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "EXAMPLE_QUERIES = [\n"
    "    'relaxing living room',\n"
    "    'somber conference room',\n"
    "    'vibrant gym',\n"
    "    'spooky gallery',\n"
    "]\n"
    "\n"
    "# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "#  ATOM TO TRACK ACROSS ALL QUERIES\n"
    "# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "CROSS_QUERY_ATOM = 'sofa'\n"
    "\n"
    "# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "#  SCENE TO DEEP-DIVE  (must match an id in data/raw/scenes.json)\n"
    "# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "EXPLORE_SCENE_ID = 'scene_relaxing_living_room_01'\n"
    "\n"
    "# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "#  DISPLAY DEPTH\n"
    "# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "TOP_K        = 20   # atoms / descriptors per query\n"
    "TOP_K_SCENES = 5    # scenes per query\n"
    "\n"
    "# Derived — do not edit\n"
    "EXPLORE_QUERY = f'{EXPLORE_AMBIANCE} {EXPLORE_SPACE}'\n"
    "print(f'Explore query  : {EXPLORE_QUERY!r}')\n"
    "print(f'Example queries: {EXAMPLE_QUERIES}')"
)

# ── 1. Load Config and Datasets ────────────────────────────────────────────
md("## 1. Load Config and Datasets")
code(
    "import yaml\n"
    "from semantic_architecture.atoms import load_atoms, atoms_summary\n"
    "from semantic_architecture.queries import load_programs, load_ambiances\n"
    "from semantic_architecture.scenes import load_scenes, scenes_summary, validate_scenes\n"
    "\n"
    "CONFIG_PATH = ROOT / 'config' / 'config.yaml'\n"
    "with open(CONFIG_PATH) as fh:\n"
    "    cfg = yaml.safe_load(fh)\n"
    "\n"
    "atoms     = load_atoms(ROOT / cfg['data']['atoms_path'])\n"
    "programs  = load_programs(ROOT / cfg['data']['programs_path'])\n"
    "ambiances = load_ambiances(ROOT / cfg['data']['ambiances_path'])\n"
    "scenes    = load_scenes([ROOT / cfg['data']['scenes_path'],\n"
    "                         ROOT / 'data' / 'raw' / 'generated_scenes.json'])\n"
    "\n"
    "print(atoms_summary(atoms))\n"
    "print(f\"\\nPrograms  ({len(programs)}): {[p['text'] for p in programs]}\")\n"
    "print(f\"Ambiances ({len(ambiances)}): {[a['text'] for a in ambiances]}\")\n"
    "print()\n"
    "print(scenes_summary(scenes))\n"
    "\n"
    "# Validate scenes\n"
    "valid_spaces = [p['text'] for p in programs]\n"
    "valid_ambs   = [a['text'] for a in ambiances]\n"
    "for w in validate_scenes(scenes, valid_spaces, valid_ambs):\n"
    "    print(f'WARNING: {w}')"
)

# ── 2. Embeddings ──────────────────────────────────────────────────────────
md("## 2. Generate Embeddings (Cached)")
code(
    "from semantic_architecture.embeddings import EmbeddingModel\n"
    "from semantic_architecture.queries import generate_all_queries\n"
    "\n"
    "emb_cfg = cfg['embedding']\n"
    "model = EmbeddingModel(\n"
    "    model_name=emb_cfg['model_name'],\n"
    "    cache_dir=ROOT / emb_cfg['cache_dir'],\n"
    "    batch_size=emb_cfg['batch_size'],\n"
    ")\n"
    "\n"
    "query_sets   = generate_all_queries(programs, ambiances)\n"
    "space_qs     = query_sets['space']\n"
    "ambiance_qs  = query_sets['ambiance']\n"
    "combined_qs  = query_sets['combined']\n"
    "\n"
    "atom_texts    = [a.text for a in atoms]\n"
    "atom_families = [a.family for a in atoms]\n"
    "atom_ids      = [a.id for a in atoms]\n"
    "\n"
    "# Use embedding_text (disambiguation-aware) for all query embeddings\n"
    "atom_embs     = model.load_or_compute(atom_texts,                                   'atoms')\n"
    "space_embs    = model.load_or_compute([q.embedding_text for q in space_qs],         'queries_space')\n"
    "ambiance_embs = model.load_or_compute([q.embedding_text for q in ambiance_qs],      'queries_ambiance')\n"
    "combined_embs = model.load_or_compute([q.embedding_text for q in combined_qs],      'queries_combined')\n"
    "\n"
    "scene_texts    = [s.text for s in scenes]\n"
    "scene_families = [s.space for s in scenes]\n"
    "scene_ids      = [s.id for s in scenes]\n"
    "scene_embs     = model.load_or_compute(scene_texts, 'scenes')\n"
    "\n"
    "print(f'Atoms      : {atom_embs.shape}')\n"
    "print(f'Queries    : {len(space_qs)} space, {len(ambiance_qs)} ambiance, {len(combined_qs)} combined')\n"
    "print(f'Scenes     : {scene_embs.shape}')"
)

code(
    "# Build query-text -> query-id lookup (used throughout the notebook)\n"
    "q_lookup = {q.combined_text: q.id for q in combined_qs}\n"
    "\n"
    "# Resolve EXAMPLE_QUERIES -> IDs\n"
    "ex_ids = {}\n"
    "for qt in EXAMPLE_QUERIES:\n"
    "    if qt not in q_lookup:\n"
    "        print(f\"WARNING: '{qt}' not found - check programs/ambiances data\")\n"
    "    else:\n"
    "        ex_ids[qt] = q_lookup[qt]\n"
    "\n"
    "# Resolve EXPLORE_QUERY -> ID\n"
    "if EXPLORE_QUERY not in q_lookup:\n"
    "    print(f\"WARNING: EXPLORE_QUERY '{EXPLORE_QUERY}' not found - check EXPLORE_SPACE / EXPLORE_AMBIANCE settings\")\n"
    "    explore_qid = None\n"
    "else:\n"
    "    explore_qid = q_lookup[EXPLORE_QUERY]\n"
    "\n"
    "print(f'Explore query ID : {explore_qid}')\n"
    "print(f'Example query IDs: {list(ex_ids.keys())}')"
)

# ── 3. Score Atoms ─────────────────────────────────────────────────────────
md("## 3. Score Atoms")
code(
    "from semantic_architecture.scoring import (\n"
    "    ScoringWeights, score_items_against_queries, enrich_with_discriminative_scores\n"
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
    "    space_queries=space_qs, ambiance_queries=ambiance_qs, combined_queries=combined_qs,\n"
    "    space_embeddings=space_embs, ambiance_embeddings=ambiance_embs,\n"
    "    combined_embeddings=combined_embs,\n"
    "    weights=weights,\n"
    ")\n"
    "atom_scores = enrich_with_discriminative_scores(atom_scores)\n"
    "print(f'Atom score table: {atom_scores.shape}')\n"
    "print('Columns:', list(atom_scores.columns))"
)

code(
    "# Top atoms for EXPLORE_QUERY (discriminative ranking)\n"
    "# Change EXPLORE_SPACE / EXPLORE_AMBIANCE in the Settings cell to target a different query.\n"
    "top_atoms = top_atoms_for_query(explore_qid, atom_scores, k=TOP_K,\n"
    "                                 score_col='discriminative_score')\n"
    "print(f'Top {TOP_K} atoms for {EXPLORE_QUERY!r} (discriminative):')\n"
    "top_atoms"
)

code(
    "# Top atoms grouped by family for EXPLORE_QUERY\n"
    "by_fam = top_atoms_by_family(explore_qid, atom_scores, k=5,\n"
    "                               score_col='discriminative_score')\n"
    "for fam, df in sorted(by_fam.items()):\n"
    "    print(f'\\n--- {fam} ---')\n"
    "    # Only show columns that are available in the returned dataframe\n"
    "    show_cols = [c for c in ['text', 'discriminative_score'] if c in df.columns]\n"
    "    print(df[show_cols].to_string(index=False))"
)

code(
    "# Cross-query: how does CROSS_QUERY_ATOM score across all 400 queries?\n"
    "# Change CROSS_QUERY_ATOM in the Settings cell to track a different atom.\n"
    "atom_cross = compare_item_across_queries(CROSS_QUERY_ATOM, atom_scores)\n"
    "print(f\"Top-10 queries for '{CROSS_QUERY_ATOM}':\")\n"
    "print(atom_cross.head(10)[['combined_text', 'sim_space', 'sim_ambiance',\n"
    "                             'sim_combined', 'weighted_score', 'discriminative_score']\n"
    "                          ].to_string(index=False))\n"
    "print(f\"\\nBottom-5 queries for '{CROSS_QUERY_ATOM}':\")\n"
    "print(atom_cross.tail(5)[['combined_text', 'discriminative_score']].to_string(index=False))"
)

# ── 4. Generate Descriptors ─────────────────────────────────────────────────
md("## 4. Generate Descriptors")
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
    "desc_df[['text', 'family_pattern', 'n_atoms']].head(20)"
)

code(
    "# Show full provenance for the first 5 descriptors\n"
    "for d in descriptors[:5]:\n"
    "    print(d.provenance_str())\n"
    "    print()"
)

# ── 5. Score Descriptors ───────────────────────────────────────────────────
md("## 5. Score Descriptors")
code(
    "from semantic_architecture.analysis import top_descriptors_for_query\n"
    "\n"
    "desc_texts    = [d.text for d in descriptors]\n"
    "desc_embs     = model.load_or_compute(desc_texts, 'descriptors')\n"
    "desc_families = [d.source_atom_families[0] for d in descriptors]\n"
    "desc_ids      = [d.id for d in descriptors]\n"
    "\n"
    "descriptor_scores = score_items_against_queries(\n"
    "    item_texts=desc_texts, item_families=desc_families, item_ids=desc_ids,\n"
    "    item_embeddings=desc_embs,\n"
    "    space_queries=space_qs, ambiance_queries=ambiance_qs, combined_queries=combined_qs,\n"
    "    space_embeddings=space_embs, ambiance_embeddings=ambiance_embs,\n"
    "    combined_embeddings=combined_embs,\n"
    "    weights=weights,\n"
    ")\n"
    "descriptor_scores = enrich_with_discriminative_scores(descriptor_scores)\n"
    "print(f'Descriptor score table: {descriptor_scores.shape}')"
)

code(
    "# Top descriptors for EXPLORE_QUERY\n"
    "top_d = top_descriptors_for_query(explore_qid, descriptor_scores, k=TOP_K,\n"
    "                                   score_col='discriminative_score')\n"
    "print(f'Top {TOP_K} descriptors for {EXPLORE_QUERY!r}:')\n"
    "cols = ['rank', 'text', 'family', 'weighted_score', 'discriminative_score']\n"
    "top_d[[c for c in cols if c in top_d.columns]]"
)

code(
    "# Descriptors for all EXAMPLE_QUERIES side-by-side\n"
    "for q_text in EXAMPLE_QUERIES:\n"
    "    if q_text not in ex_ids:\n"
    "        print(f\"Skip: '{q_text}' not in ex_ids\")\n"
    "        continue\n"
    "    top_d = top_descriptors_for_query(ex_ids[q_text], descriptor_scores, k=10,\n"
    "                                       score_col='discriminative_score')\n"
    "    print(f'\\n-- Top 10 descriptors for {q_text!r} --')\n"
    "    cols = ['rank', 'text', 'family', 'discriminative_score']\n"
    "    print(top_d[[c for c in cols if c in top_d.columns]].to_string(index=False))"
)

# ── 6. Score Scenes ────────────────────────────────────────────────────────
md(
    "## 6. Score Scenes\n\n"
    "Each scene has an **intended query** (its `space` x `ambiance` pair).\n"
    "Scoring scenes against all 400 combined queries lets us check whether\n"
    "the embedding space can retrieve the right context for a given description."
)
code(
    "from semantic_architecture.analysis import top_scenes_for_query\n"
    "\n"
    "scene_scores = score_items_against_queries(\n"
    "    item_texts=scene_texts, item_families=scene_families, item_ids=scene_ids,\n"
    "    item_embeddings=scene_embs,\n"
    "    space_queries=space_qs, ambiance_queries=ambiance_qs, combined_queries=combined_qs,\n"
    "    space_embeddings=space_embs, ambiance_embeddings=ambiance_embs,\n"
    "    combined_embeddings=combined_embs,\n"
    "    weights=weights,\n"
    ")\n"
    "scene_scores = enrich_with_discriminative_scores(scene_scores)\n"
    "print(f'Scene score table: {scene_scores.shape}')"
)

# ── 7. Scene Explorer ──────────────────────────────────────────────────────
md(
    "## 7. Scene Explorer\n\n"
    "Use **`EXPLORE_SPACE` / `EXPLORE_AMBIANCE`** and **`EXPLORE_SCENE_ID`**\n"
    "from the Settings cell to focus on any space x ambiance pair or scene."
)

code(
    "# -- 7a. Scene browser: all scenes with their intended vs predicted query --\n"
    "rows = []\n"
    "for s in scenes:\n"
    "    row = scene_scores[scene_scores['item_id'] == s.id].copy()\n"
    "    row = row.sort_values('discriminative_score', ascending=False)\n"
    "    top_query = row.iloc[0]['combined_text'] if len(row) > 0 else '-'\n"
    "    rows.append({\n"
    "        'id': s.id,\n"
    "        'intended': s.intended_query,\n"
    "        'top_predicted': top_query,\n"
    "        'hit@1': top_query == s.intended_query,\n"
    "        'preview': s.text[:90] + '...',\n"
    "    })\n"
    "browser_df = pd.DataFrame(rows)\n"
    "print(f'{len(scenes)} scenes loaded')\n"
    "browser_df"
)

code(
    "# -- 7b. Top scenes for EXPLORE_QUERY --\n"
    "# Change EXPLORE_SPACE / EXPLORE_AMBIANCE in the Settings cell.\n"
    "if explore_qid:\n"
    "    top_s = top_scenes_for_query(explore_qid, scene_scores, k=TOP_K_SCENES,\n"
    "                                  score_col='discriminative_score')\n"
    "    print(f'Top {TOP_K_SCENES} scenes for {EXPLORE_QUERY!r}:')\n"
    "    show_cols = [c for c in ['rank', 'text', 'family', 'discriminative_score'] if c in top_s.columns]\n"
    "    print(top_s[show_cols].to_string(index=False))\n"
    "else:\n"
    "    print('EXPLORE_QUERY not resolved - check settings.')"
)

code(
    "# -- 7c. Top scenes for each EXAMPLE_QUERY --\n"
    "for q_text in EXAMPLE_QUERIES:\n"
    "    if q_text not in ex_ids:\n"
    "        continue\n"
    "    top_s = top_scenes_for_query(ex_ids[q_text], scene_scores, k=TOP_K_SCENES,\n"
    "                                  score_col='discriminative_score')\n"
    "    print(f'\\n-- Top {TOP_K_SCENES} scenes for {q_text!r} --')\n"
    "    show_cols = [c for c in ['rank', 'text', 'family', 'discriminative_score'] if c in top_s.columns]\n"
    "    print(top_s[show_cols].to_string(index=False))"
)

code(
    "# -- 7d. Scene deep-dive: full text + top queries for EXPLORE_SCENE_ID --\n"
    "# Change EXPLORE_SCENE_ID in the Settings cell to inspect a different scene.\n"
    "scene_match = [s for s in scenes if s.id == EXPLORE_SCENE_ID]\n"
    "if not scene_match:\n"
    "    print(f\"Scene '{EXPLORE_SCENE_ID}' not found - check EXPLORE_SCENE_ID in Settings\")\n"
    "else:\n"
    "    scene = scene_match[0]\n"
    "    print(f'Scene  : {scene.id}')\n"
    "    print(f'Intended query: {scene.intended_query!r}')\n"
    "    if scene.notes:\n"
    "        print(f'Notes  : {scene.notes}')\n"
    "    print()\n"
    "    print('-- Full text --')\n"
    "    print(scene.text)\n"
    "    print()\n"
    "    s_row = scene_scores[scene_scores['item_id'] == scene.id].copy()\n"
    "    s_row = s_row.sort_values('discriminative_score', ascending=False).head(10)\n"
    "    print('-- Top 10 matching queries (discriminative) --')\n"
    "    print(s_row[['combined_text', 'discriminative_score', 'weighted_score']].to_string(index=False))"
)

code(
    "# -- 7e. All scenes ranked for EXPLORE_QUERY (full ranking table) --\n"
    "if explore_qid:\n"
    "    all_scene_rows = scene_scores[scene_scores['query_id'] == explore_qid].copy()\n"
    "    all_scene_rows = all_scene_rows.sort_values('discriminative_score', ascending=False)\n"
    "    display_cols = [c for c in ['item_id', 'text', 'sim_space', 'sim_ambiance',\n"
    "                                 'sim_combined', 'weighted_score',\n"
    "                                 'discriminative_score'] if c in all_scene_rows.columns]\n"
    "    all_scene_rows = all_scene_rows[display_cols].reset_index(drop=True)\n"
    "    all_scene_rows.index += 1\n"
    "    print(f'All {len(scenes)} scenes ranked for {EXPLORE_QUERY!r}:')\n"
    "    all_scene_rows"
)

# ── 8. Ground-Truth Evaluation ─────────────────────────────────────────────
md(
    "## 8. Ground-Truth Evaluation\n\n"
    "For each scene, does the pipeline rank its intended `space x ambiance` query first?"
)
code(
    "results = []\n"
    "for scene in scenes:\n"
    "    intended = scene.intended_query\n"
    "    s_row = scene_scores[scene_scores['item_id'] == scene.id].copy()\n"
    "    s_row = s_row.sort_values('discriminative_score', ascending=False).reset_index(drop=True)\n"
    "\n"
    "    top1_text = s_row.iloc[0]['combined_text']\n"
    "    intended_mask = s_row['combined_text'] == intended\n"
    "    rank_of_intended = int(s_row.index[intended_mask][0]) + 1 if intended_mask.any() else None\n"
    "\n"
    "    results.append({\n"
    "        'scene_id': scene.id,\n"
    "        'intended': intended,\n"
    "        'top1_query': top1_text,\n"
    "        'hit@1': top1_text == intended,\n"
    "        'rank_of_intended': rank_of_intended,\n"
    "    })\n"
    "\n"
    "eval_df = pd.DataFrame(results)\n"
    "hit1 = eval_df['hit@1'].mean()\n"
    "hit5 = (eval_df['rank_of_intended'] <= 5).mean()\n"
    "print(f'Hit@1  : {hit1:.0%}  ({eval_df[\"hit@1\"].sum()}/{len(eval_df)})')\n"
    "print(f'Hit@5  : {hit5:.0%}  ({(eval_df[\"rank_of_intended\"] <= 5).sum()}/{len(eval_df)})')\n"
    "print()\n"
    "eval_df[['scene_id', 'intended', 'top1_query', 'hit@1', 'rank_of_intended']]"
)

# ── 9. Export All Results ──────────────────────────────────────────────────
md("## 9. Export All Results")
code(
    "from semantic_architecture.io_utils import save_csv, save_json\n"
    "\n"
    "out_dir = ROOT / cfg['output']['dir']\n"
    "out_dir.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "save_csv(atom_scores,       out_dir / 'atom_scores.csv',       'atom scores')\n"
    "save_csv(descriptor_scores, out_dir / 'descriptor_scores.csv', 'descriptor scores')\n"
    "save_csv(desc_df,           out_dir / 'descriptors.csv',       'descriptor table')\n"
    "save_csv(scene_scores,      out_dir / 'scene_scores.csv',      'scene scores')\n"
    "save_csv(eval_df,           out_dir / 'scene_eval.csv',        'scene evaluation')\n"
    "\n"
    "# Per-query rankings for EXAMPLE_QUERIES\n"
    "for q_text, qid in ex_ids.items():\n"
    "    safe = q_text.replace(' ', '_')\n"
    "    save_csv(top_atoms_for_query(qid, atom_scores, k=30),\n"
    "             out_dir / f'ranking_atoms_{safe}.csv')\n"
    "    save_csv(top_descriptors_for_query(qid, descriptor_scores, k=30),\n"
    "             out_dir / f'ranking_descriptors_{safe}.csv')\n"
    "    save_csv(top_scenes_for_query(qid, scene_scores, k=len(scenes)),\n"
    "             out_dir / f'ranking_scenes_{safe}.csv')\n"
    "\n"
    "print('Export complete.')"
)

# ── 10. Visualizations ─────────────────────────────────────────────────────
md("## 10. Visualizations")
code(
    "from semantic_architecture.visualization import (\n"
    "    plot_top_items, plot_heatmap, plot_pca_projection,\n"
    "    plot_umap_projection, plot_family_comparison,\n"
    ")\n"
    "from semantic_architecture.scoring import similarity_matrix_df"
)

code(
    "# -- 10a. Bar chart: top atoms for EXPLORE_QUERY --\n"
    "fig = plot_top_items(\n"
    "    top_atoms_for_query(explore_qid, atom_scores, k=TOP_K,\n"
    "                        score_col='discriminative_score'),\n"
    "    query_label=EXPLORE_QUERY, top_k=TOP_K,\n"
    "    score_col='discriminative_score', color_by_family=True,\n"
    ")\n"
    "safe_q = EXPLORE_QUERY.replace(' ', '_')\n"
    "fig.savefig(out_dir / f'bar_atoms_{safe_q}.png', bbox_inches='tight', dpi=150)\n"
    "plt.show()"
)

code(
    "# -- 10b. Heatmap: top-25 atoms x EXAMPLE_QUERIES --\n"
    "ex_combined_embs, ex_labels = [], []\n"
    "for qt in EXAMPLE_QUERIES:\n"
    "    idx = next((i for i, q in enumerate(combined_qs) if q.combined_text == qt), None)\n"
    "    if idx is not None:\n"
    "        ex_combined_embs.append(combined_embs[idx])\n"
    "        ex_labels.append(qt)\n"
    "ex_combined_embs = np.stack(ex_combined_embs)\n"
    "\n"
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
    "fig = plot_heatmap(mat, title=f'Top-25 atoms vs {len(ex_labels)} example queries',\n"
    "                   figsize=(10, 10), annot=True)\n"
    "fig.savefig(out_dir / 'heatmap_atoms_example_queries.png', bbox_inches='tight', dpi=150)\n"
    "plt.show()"
)

code(
    "# -- 10c. PCA of all atoms --\n"
    "fig = plot_pca_projection(\n"
    "    atom_embs, atom_texts, families=atom_families,\n"
    "    title='PCA of all atoms (coloured by family)',\n"
    "    annotate=True, figsize=(12, 9),\n"
    ")\n"
    "fig.savefig(out_dir / 'pca_atoms.png', bbox_inches='tight', dpi=150)\n"
    "plt.show()"
)

code(
    "# -- 10d. UMAP (skipped if umap-learn not installed) --\n"
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
    "# -- 10e. Family comparison across EXAMPLE_QUERIES --\n"
    "fig = plot_family_comparison(\n"
    "    atom_scores, query_ids=ex_qids,\n"
    "    score_col='weighted_score', figsize=(14, 6),\n"
    ")\n"
    "fig.savefig(out_dir / 'family_comparison_example_queries.png', bbox_inches='tight', dpi=150)\n"
    "plt.show()"
)

code(
    "# -- 10f. PCA atoms + scenes overlaid --\n"
    "all_embs     = np.vstack([atom_embs, scene_embs])\n"
    "all_texts    = atom_texts + [f'[{s.id}]' for s in scenes]\n"
    "all_families = atom_families + [f'scene:{s.space}' for s in scenes]\n"
    "\n"
    "fig = plot_pca_projection(\n"
    "    all_embs, all_texts, families=all_families,\n"
    "    title='PCA - atoms (dots) + scenes (triangles) overlaid',\n"
    "    annotate=False, figsize=(13, 10),\n"
    ")\n"
    "fig.savefig(out_dir / 'pca_atoms_scenes.png', bbox_inches='tight', dpi=150)\n"
    "plt.show()"
)

md(
    "## Summary\n\n"
    "Outputs saved to `data/processed/`:\n\n"
    "| File | Contents |\n"
    "|------|----------|\n"
    "| `atom_scores.csv` | Atoms x 400 queries |\n"
    "| `descriptor_scores.csv` | Descriptors x 400 queries |\n"
    "| `scene_scores.csv` | Scenes x 400 queries |\n"
    "| `scene_eval.csv` | Hit@1 / Hit@5 ground-truth evaluation |\n"
    "| `descriptors.csv` | Descriptor list with provenance |\n"
    "| `ranking_atoms_*.csv` | Per-query atom rankings |\n"
    "| `ranking_descriptors_*.csv` | Per-query descriptor rankings |\n"
    "| `ranking_scenes_*.csv` | Per-query scene rankings |\n"
    "| `bar_atoms_*.png` | Bar chart for explore query |\n"
    "| `heatmap_atoms_example_queries.png` | Heatmap |\n"
    "| `pca_atoms.png` | PCA projection (atoms only) |\n"
    "| `pca_atoms_scenes.png` | PCA projection (atoms + scenes) |\n"
    "| `umap_atoms.png` | UMAP projection |\n"
    "| `family_comparison_example_queries.png` | Family comparison chart |\n\n"
    "> **To explore a different query:** change `EXPLORE_SPACE` and `EXPLORE_AMBIANCE`\n"
    "> in the Settings cell (section 0b), then re-run sections 3-7.\n"
    ">\n"
    "> **To explore a different scene:** change `EXPLORE_SCENE_ID` in the Settings cell.\n"
    ">\n"
    "> **To add comparison queries:** add entries to `EXAMPLE_QUERIES` in the Settings cell."
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
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.11",
        },
    },
    "cells": cells,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with NB_PATH.open("w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)

print(f"Notebook written to {NB_PATH}")
