"""
Generate architectural scene descriptions using OpenAI's API.
Grounded in the project's atomic lexicon (atoms and descriptors).
"""
import os
import json
import yaml
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load project modules
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from semantic_architecture.atoms import load_atoms
from semantic_architecture.queries import load_programs, load_ambiances, generate_all_queries
from semantic_architecture.scoring import ScoringWeights, score_items_against_queries, enrich_with_discriminative_scores
from semantic_architecture.embeddings import EmbeddingModel

# Configuration
load_dotenv(ROOT / ".env")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use GPT-4o-mini for balance of quality and cost
MODEL = "gpt-4o-mini"
MAX_TOKENS = 150
TEMPERATURE = 0.8

# Path to output file
OUTPUT_PATH = ROOT / "data" / "raw" / "generated_scenes.json"

async def generate_scene_text(space: str, ambiance: str, top_atoms: list[str]) -> str:
    """Generate a single scene description using OpenAI."""
    prompt = (
        f"You are an architectural writer and atmosphere designer. "
        f"Write an evocative, paragraph-length description (approx. 60-90 words) for a architecture project "
        f"focused on a '{space}' with a '{ambiance}' ambiance. "
        f"Incorporate the following architectural 'atoms' (elements, materials, or qualities) as physical anchors of the space:\n"
        f"Atoms: {', '.join(top_atoms)}.\n\n"
        f"Focus on the sensory experience, material qualities, and lighting. Do not use generic buzzwords. "
        f"The tone should be professional yet atmospheric."
    )
    
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating scene for {ambiance} {space}: {e}")
        return ""

async def main():
    # 1. Load Data
    with open(ROOT / "config" / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    
    atoms = load_atoms(ROOT / cfg["data"]["atoms_path"])
    programs = load_programs(ROOT / cfg["data"]["programs_path"])
    ambiances = load_ambiances(ROOT / cfg["data"]["ambiances_path"])
    
    # 2. Setup Embeddings & Scoring
    emb_cfg = cfg["embedding"]
    model = EmbeddingModel(
        model_name=emb_cfg["model_name"],
        cache_dir=ROOT / emb_cfg["cache_dir"],
        batch_size=emb_cfg["batch_size"],
    )
    
    query_sets = generate_all_queries(programs, ambiances)
    combined_qs = query_sets["combined"]
    
    atom_texts = [a.text for a in atoms]
    atom_families = [a.family for a in atoms]
    atom_ids = [a.id for a in atoms]
    atom_embs = model.load_or_compute(atom_texts, "atoms")
    
    space_qs = query_sets["space"]
    ambiance_qs = query_sets["ambiance"]
    space_embs = model.load_or_compute([q.embedding_text for q in space_qs], "queries_space")
    ambiance_embs = model.load_or_compute([q.embedding_text for q in ambiance_qs], "queries_ambiance")
    combined_embs = model.load_or_compute([q.embedding_text for q in combined_qs], "queries_combined")
    
    weights = ScoringWeights.from_config(cfg.get("scoring", {}))
    
    # 3. Perform Scoring to get Top Atoms
    print("Scoring atoms to find grounding elements...")
    atom_scores = score_items_against_queries(
        item_texts=atom_texts, item_families=atom_families, item_ids=atom_ids,
        item_embeddings=atom_embs,
        space_queries=space_qs, ambiance_queries=ambiance_qs, combined_queries=combined_qs,
        space_embeddings=space_embs, ambiance_embeddings=ambiance_embs, combined_embeddings=combined_embs,
        weights=weights,
    )
    atom_scores = enrich_with_discriminative_scores(atom_scores)
    
    # 4. Generate Scenes
    print(f"Generating scenes for {len(combined_qs)} queries...")
    
    # Optional: Load existing generated scenes to avoid duplicates/costs
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            existing_data = json.load(f)
            generated_scenes = existing_data.get("scenes", [])
    else:
        generated_scenes = []
    
    existing_ids = {s["id"] for s in generated_scenes}
    
    # Parallel generation with limited semaphore to avoid rate limits
    semaphore = asyncio.Semaphore(5)
    
    async def process_query(q):
        async with semaphore:
            sid = f"gen_{q.ambiance_text.replace(' ', '_')}_{q.space_text.replace(' ', '_')}"
            if sid in existing_ids:
                return None
            
            # Get top-10 atoms for this query
            q_scores = atom_scores[atom_scores["query_id"] == q.id].copy()
            q_scores = q_scores.sort_values("discriminative_score", ascending=False).head(10)
            top_atoms = q_scores["text"].tolist()
            
            print(f"Generating: {q.combined_text}...")
            text = await generate_scene_text(q.space_text, q.ambiance_text, top_atoms)
            
            if text:
                return {
                    "id": sid,
                    "text": text,
                    "space": q.space_text,
                    "ambiance": q.ambiance_text,
                    "notes": f"Generated grounded in: {', '.join(top_atoms[:3])}..."
                }
            return None

    tasks = [process_query(q) for q in combined_qs]
    results = await asyncio.gather(*tasks)
    
    new_scenes = [r for r in results if r]
    generated_scenes.extend(new_scenes)
    
    # 5. Save Output
    print(f"Finished. Generated {len(new_scenes)} new scenes. Total generated: {len(generated_scenes)}")
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"scenes": generated_scenes}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())
