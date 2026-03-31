"""Canonical path derivation for the 3-pass pipeline.

All commands that read or write pass-2 identified JSON must use
``identified_suffix`` so the naming convention stays consistent across
``identify``, ``eval-id``, and any future command that consumes identified
output.
"""

from __future__ import annotations

from pathlib import Path


def identified_suffix(*, no_llm: bool, embeddings: bool) -> str:
    """Return the ``_identified*.json`` filename suffix for a strategy combination.

    Strategy   | no_llm | embeddings | suffix
    -----------|--------|------------|----------------------------
    Gemini LLM | False  | False      | _identified.json
    Heuristic  | True   | False      | _identified_heuristic.json
    Embeddings | True   | True       | _identified_embeddings.json
    """
    if embeddings:
        return "_identified_embeddings.json"
    if no_llm:
        return "_identified_heuristic.json"
    return "_identified.json"


def identified_path(video: Path, *, no_llm: bool, embeddings: bool) -> Path:
    """Return the default output/input path for an identified JSON next to *video*."""
    return video.with_name(video.stem + identified_suffix(no_llm=no_llm, embeddings=embeddings))
