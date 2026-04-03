"""
cli.py — Lightweight command-line interface for the SpaceAmb prototype.

Commands
--------
  score    Score atoms and/or descriptors against a query string.
  compose  Generate descriptors and print examples.
  export   Run the full pipeline and export all artefacts.
  info     Print dataset statistics.

Usage examples
--------------
  python -m semantic_architecture.cli score "relaxing living room" --top-k 20
  python -m semantic_architecture.cli score "relaxing living room" --items descriptors
  python -m semantic_architecture.cli compose --n 50 --temperature 0.5
  python -m semantic_architecture.cli export
  python -m semantic_architecture.cli info
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="spaceamb",
    help="SpaceAmb — architectural-semantic research prototype.",
    add_completion=False,
)

# Default config path (relative to project root)
DEFAULT_CONFIG = "config/config.yaml"


def _load_state(config: str) -> "AppState":
    """Import and load AppState (delayed import to keep CLI startup fast)."""
    from .app_state import AppState
    typer.echo("[cli] Loading pipeline state …")
    return AppState.load(config_path=config)


# ------------------------------------------------------------------
# score
# ------------------------------------------------------------------

@app.command()
def score(
    query: str = typer.Argument(
        ..., help="Query string, e.g. 'relaxing living room'."
    ),
    items: str = typer.Option(
        "atoms",
        "--items", "-i",
        help="What to score: 'atoms', 'descriptors', 'scenes', 'both', or 'all'.",
    ),
    top_k: int = typer.Option(20, "--top-k", "-k", help="Number of results to show."),
    score_col: Optional[str] = typer.Option(
        None,
        "--score", "-s",
        help=(
            "Column to rank by: discriminative_score (default) | weighted_score | "
            "zscore_score | sim_combined | sim_space | sim_ambiance. "
            "Defaults to scoring.default_score_col in config.yaml."
        ),
    ),
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    """
    Score atoms and/or descriptors against QUERY, print ranked table.
    """
    import yaml as _yaml
    from .analysis import top_atoms_for_query, top_descriptors_for_query

    # Resolve default score column from config if not explicitly passed
    if score_col is None:
        with Path(config).open() as _fh:
            _cfg = _yaml.safe_load(_fh)
        score_col = _cfg.get("scoring", {}).get("default_score_col", "weighted_score")

    state = _load_state(config)

    # Find the query id that matches the input string
    matched = state.get_query(query)
    if matched is None:
        # Fall back to embedding-based lookup for ad-hoc queries
        typer.echo(
            f"[cli] Query '{query}' not in pre-computed set. "
            "Embedding on-the-fly …"
        )
        from .embeddings import EmbeddingModel
        import yaml, numpy as np
        from .scoring import cosine_similarity_vec

        with Path(config).open() as fh:
            cfg = yaml.safe_load(fh)
        model = EmbeddingModel(
            model_name=cfg["embedding"]["model_name"],
            cache_dir=cfg["embedding"]["cache_dir"],
        )
        q_emb = model.embed_one(query)

        from .scoring import rank_items as _rank
        if items in ("atoms", "both", "all"):
            df = _rank(
                state.atom_texts, state.atom_families,
                state.atom_embeddings, q_emb, top_k=top_k
            )
            typer.echo(f"\nTop {top_k} atoms for '{query}':")
            typer.echo(df.to_string(index=False))

        if items in ("descriptors", "both", "all"):
            desc_families = [d.source_atom_families[0] for d in state.descriptors]
            df = _rank(
                state.descriptor_texts, desc_families,
                state.descriptor_embeddings, q_emb, top_k=top_k
            )
            typer.echo(f"\nTop {top_k} descriptors for '{query}':")
            typer.echo(df.to_string(index=False))

        if items in ("scenes", "all"):
            if state.scenes and state.scene_embeddings is not None:
                scene_families = [s.space for s in state.scenes]
                df = _rank(
                    state.scene_texts, scene_families,
                    state.scene_embeddings, q_emb, top_k=top_k
                )
                # Truncate long scene texts for display
                df["text"] = df["text"].str[:80] + "…"
                typer.echo(f"\nTop {top_k} scenes for '{query}':")
                typer.echo(df.to_string(index=False))
            else:
                typer.echo("\n[cli] No scenes loaded.")
        return

    qid = matched.id

    if items in ("atoms", "both", "all"):
        df = top_atoms_for_query(qid, state.atom_scores, k=top_k, score_col=score_col)
        typer.echo(f"\nTop {top_k} atoms for '{query}':")
        typer.echo(df.to_string(index=False))

    if items in ("descriptors", "both", "all"):
        df = top_descriptors_for_query(
            qid, state.descriptor_scores, k=top_k, score_col=score_col
        )
        typer.echo(f"\nTop {top_k} descriptors for '{query}':")
        typer.echo(df.to_string(index=False))

    if items in ("scenes", "all"):
        if state.scene_scores is None:
            typer.echo(
                "\n[cli] No scene scores available. "
                "Add a scenes_path to config.yaml and ensure data/raw/scenes.json exists."
            )
        else:
            from .analysis import top_scenes_for_query
            df = top_scenes_for_query(
                qid, state.scene_scores, k=top_k, score_col=score_col
            )
            typer.echo(f"\nTop {top_k} scenes for '{query}':")
            typer.echo(df.to_string(index=False))


# ------------------------------------------------------------------
# compose
# ------------------------------------------------------------------

@app.command()
def compose(
    n: int = typer.Option(50, "--n", help="Number of descriptors to generate."),
    temperature: float = typer.Option(
        1.0, "--temperature", "-t",
        help="Sampling temperature (lower = more semantically tight).",
    ),
    lengths: str = typer.Option(
        "2,3", "--lengths", "-l",
        help="Comma-separated descriptor lengths, e.g. '2,3'.",
    ),
    seed: int = typer.Option(42, "--seed"),
    query: Optional[str] = typer.Option(
        None, "--query", "-q",
        help=(
            "If given, use query-conditioned generation: seeds are drawn from "
            "the top atoms for this query rather than the full pool. "
            "E.g. 'relaxing living room'."
        ),
    ),
    seed_top_k: int = typer.Option(
        30, "--seed-top-k",
        help="Number of top atoms used as seed pool for query-conditioned generation.",
    ),
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    """
    Generate descriptors using similarity-weighted sampling and print examples.

    With --query, seeds are restricted to the top atoms for that query
    (query-conditioned generation).  The full atom pool is still used for
    subsequent positions within each descriptor.
    """
    from .atoms import load_atoms
    from .embeddings import EmbeddingModel
    from .composition import generate_descriptors, generate_descriptors_for_query
    import yaml as _yaml

    with Path(config).open() as fh:
        cfg = _yaml.safe_load(fh)

    from .io_utils import resolve_path
    root = resolve_path(".")
    atoms = load_atoms(root / cfg["data"]["atoms_path"])

    emb_cfg = cfg.get("embedding", {})
    model = EmbeddingModel(
        model_name=emb_cfg.get("model_name", "all-MiniLM-L6-v2"),
        cache_dir=root / emb_cfg.get("cache_dir", "data/processed/embeddings"),
        batch_size=emb_cfg.get("batch_size", 64),
    )
    atom_texts = [a.text for a in atoms]
    atom_embs = model.load_or_compute(atom_texts, "atoms")
    length_list = [int(x.strip()) for x in lengths.split(",")]

    if query:
        # Query-conditioned: load the pre-computed atom scores
        typer.echo(f"[cli] Query-conditioned generation for '{query}' …")
        state = _load_state(config)
        matched = next(
            (q for q in state.combined_queries if q.combined_text == query), None
        )
        if matched is None:
            typer.echo(
                f"[cli] Query '{query}' not found in pre-computed set. "
                "Run with a query that matches a combined query label exactly."
            )
            raise typer.Exit(1)

        score_col = cfg.get("scoring", {}).get("default_score_col", "discriminative_score")
        descriptors = generate_descriptors_for_query(
            query_id=matched.id,
            scores_df=state.atom_scores,
            atoms=atoms,
            atom_embeddings=atom_embs,
            n_descriptors=n,
            seed_top_k=seed_top_k,
            score_col=score_col,
            descriptor_lengths=length_list,
            temperature=temperature,
            seed=seed,
        )
        typer.echo(
            f"\nGenerated {len(descriptors)} descriptors for '{query}' "
            f"(temp={temperature}, seed_top_k={seed_top_k}):\n"
        )
    else:
        descriptors = generate_descriptors(
            atoms=atoms,
            atom_embeddings=atom_embs,
            n_descriptors=n,
            descriptor_lengths=length_list,
            temperature=temperature,
            seed=seed,
        )
        typer.echo(f"\nGenerated {len(descriptors)} descriptors (temp={temperature}):\n")

    for i, d in enumerate(descriptors[:n], 1):
        families = "+".join(d.source_atom_families)
        typer.echo(f"  {i:3d}. {d.text:<40}  [{families}]")


# ------------------------------------------------------------------
# export
# ------------------------------------------------------------------

@app.command()
def export(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o",
        help="Override output directory from config.",
    ),
) -> None:
    """
    Run the full pipeline and export all artefacts to disk.
    """
    state = _load_state(config)
    out = state.export_all(output_dir)
    typer.echo(f"\n[cli] Export complete → {out}")


# ------------------------------------------------------------------
# info
# ------------------------------------------------------------------

@app.command()
def info(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    """
    Print dataset and embedding model information without running scoring.
    """
    from .atoms import load_atoms, atoms_summary
    from .queries import load_programs, load_ambiances, generate_combined_queries
    from .scenes import load_scenes, scenes_summary
    import yaml

    with Path(config).open() as fh:
        cfg = yaml.safe_load(fh)

    from .io_utils import resolve_path
    root = resolve_path(".")
    atoms = load_atoms(root / cfg["data"]["atoms_path"])
    programs = load_programs(root / cfg["data"]["programs_path"])
    ambiances = load_ambiances(root / cfg["data"]["ambiances_path"])
    combined_qs = generate_combined_queries(programs, ambiances)

    typer.echo(atoms_summary(atoms))
    typer.echo(f"\nPrograms  : {len(programs)}")
    typer.echo(f"Ambiances : {len(ambiances)}")
    typer.echo(f"Combined queries : {len(combined_qs)}")
    typer.echo(f"\nEmbedding model  : {cfg['embedding']['model_name']}")
    typer.echo(f"Composition temp : {cfg['composition'].get('temperature', 1.0)}")
    typer.echo(f"Target descriptors: {cfg['composition'].get('n_descriptors', 300)}")

    scenes_path_cfg = cfg.get("data", {}).get("scenes_path")
    if scenes_path_cfg:
        scenes_full = root / scenes_path_cfg
        if scenes_full.exists():
            scenes = load_scenes(scenes_full)
            typer.echo(f"\n--- Phase 3: Scenes ---")
            typer.echo(scenes_summary(scenes))
        else:
            typer.echo(f"\nScenes    : (file not found: {scenes_path_cfg})")


if __name__ == "__main__":
    app()
