#!/usr/bin/env python
"""
run_analysis.py — Thin script wrapper around the SpaceAmb CLI.

Runs the full pipeline for four example queries and exports all outputs.
Call directly:

    python scripts/run_analysis.py

Or pass a query:

    python scripts/run_analysis.py --query "spooky dance hall"
"""

import sys
from pathlib import Path

# Ensure the src directory is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from semantic_architecture.cli import app

if __name__ == "__main__":
    # If no args given, run the export command which covers the full pipeline
    if len(sys.argv) == 1:
        sys.argv = ["run_analysis.py", "export"]
    app()
