"""
io_utils.py — Save/load helpers for CSV, JSON, and directory management.

All paths passed to these functions should be absolute or relative to the
project root.  The helpers create parent directories automatically so
callers never have to call os.makedirs manually.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """
    Create *path* (and all parents) if it does not exist.

    Parameters
    ----------
    path : str or Path
        A file or directory path.  If the path looks like a file
        (has a suffix), the parent directory is created.

    Returns
    -------
    Path
        The resolved Path object.
    """
    p = Path(path)
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: str | Path, desc: str = "") -> Path:
    """
    Save a DataFrame to CSV, creating parent directories as needed.

    Parameters
    ----------
    df : pd.DataFrame
    path : str or Path
    desc : str
        Optional description printed to stdout on save.

    Returns
    -------
    Path
        The resolved path where the file was written.
    """
    p = ensure_dir(path)
    df.to_csv(p, index=True, encoding="utf-8")
    if desc:
        print(f"[io] Saved {desc} → {p} ({len(df)} rows)")
    else:
        print(f"[io] Saved CSV → {p} ({len(df)} rows)")
    return p


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV into a DataFrame."""
    return pd.read_csv(Path(path), encoding="utf-8", index_col=0)


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """
    Serialise *data* to a JSON file.

    Parameters
    ----------
    data : any JSON-serialisable object
    path : str or Path
    indent : int
        JSON indentation level.

    Returns
    -------
    Path
    """
    p = ensure_dir(path)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)
    print(f"[io] Saved JSON → {p}")
    return p


def load_json(path: str | Path) -> Any:
    """Load and return the contents of a JSON file."""
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def project_root() -> Path:
    """
    Return the project root directory.

    Assumes this file lives at ``<root>/src/semantic_architecture/io_utils.py``.
    """
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(rel_path: str | Path) -> Path:
    """
    Resolve a path relative to the project root.

    If *rel_path* is already absolute it is returned unchanged.
    """
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return project_root() / p
