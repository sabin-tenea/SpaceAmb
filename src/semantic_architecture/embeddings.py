"""
embeddings.py — Sentence-transformer embedding layer with disk caching.

Design goals
------------
* Wrap SentenceTransformer so the model name is configurable.
* Return L2-normalised vectors (cosine similarity = dot product).
* Cache embeddings to .npy files keyed by a collection name so that
  re-running the pipeline does not re-embed unchanged data.
* Support incremental updates: if the index shows a mismatch in texts,
  recompute and overwrite.

Cache format
------------
  data/processed/embeddings/{collection}.npy     — float32 array, shape (n, d)
  data/processed/embeddings/{collection}_index.json — list of text strings
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer with caching.

    Parameters
    ----------
    model_name : str
        Any model accepted by sentence-transformers, e.g.
        ``"all-MiniLM-L6-v2"`` (fast) or ``"all-mpnet-base-v2"`` (quality).
    cache_dir : str or Path
        Directory where .npy files and index JSONs are stored.
    batch_size : int
        Passed to SentenceTransformer.encode.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str | Path = "data/processed/embeddings",
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self._model = None  # lazy-loaded on first encode call

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the SentenceTransformer model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            print(f"[embed] Loading model '{self.model_name}' …")
            self._model = SentenceTransformer(self.model_name)
            print("[embed] Model loaded.")
        return self._model

    def _npy_path(self, collection: str) -> Path:
        return self.cache_dir / f"{collection}.npy"

    def _index_path(self, collection: str) -> Path:
        return self.cache_dir / f"{collection}_index.json"

    def _cache_valid(self, collection: str, texts: List[str]) -> bool:
        """Return True if cached embeddings exist and match *texts* exactly."""
        npy = self._npy_path(collection)
        idx = self._index_path(collection)
        if not npy.exists() or not idx.exists():
            return False
        with idx.open(encoding="utf-8") as fh:
            cached_texts = json.load(fh)
        return cached_texts == texts

    def _save_cache(
        self, collection: str, texts: List[str], vectors: np.ndarray
    ) -> None:
        np.save(self._npy_path(collection), vectors)
        with self._index_path(collection).open("w", encoding="utf-8") as fh:
            json.dump(texts, fh, ensure_ascii=False)

    def _load_cache(self, collection: str) -> np.ndarray:
        return np.load(self._npy_path(collection))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_or_compute(
        self, texts: List[str], collection: str
    ) -> np.ndarray:
        """
        Return L2-normalised embeddings for *texts*, using cache when valid.

        Parameters
        ----------
        texts : list of str
            The strings to embed.  Order matters — the returned array rows
            correspond 1-to-1 with *texts*.
        collection : str
            A unique name for this set (e.g. ``"atoms"``, ``"queries_combined"``).
            Used as the file stem for the .npy and _index.json files.

        Returns
        -------
        np.ndarray, shape (n, d), float32
            L2-normalised embedding matrix.
        """
        if self._cache_valid(collection, texts):
            print(f"[embed] Cache hit for '{collection}' ({len(texts)} items).")
            return self._load_cache(collection)

        print(
            f"[embed] Computing embeddings for '{collection}' "
            f"({len(texts)} items) …"
        )
        model = self._load_model()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 20,
            normalize_embeddings=True,  # L2-normalise at source
            convert_to_numpy=True,
        ).astype(np.float32)

        self._save_cache(collection, texts, vectors)
        print(f"[embed] Done. Shape: {vectors.shape}. Saved to cache.")
        return vectors

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed *texts* without caching.  Useful for ad-hoc queries.

        Returns L2-normalised float32 array of shape (n, d).
        """
        model = self._load_model()
        return model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string; return shape (d,) array."""
        return self.embed([text])[0]

    @property
    def dim(self) -> Optional[int]:
        """Embedding dimension, or None if the model hasn't been loaded."""
        if self._model is None:
            return None
        return self._model.get_sentence_embedding_dimension()
