"""beach stitch-rallies — Combine per-rally *_fixed.json files into a single
action JSON whose timestamps are relative to the original full video.

For each rally the script:
  1. Reads the rally's start offset from ``<video_stem>_rallies.json``
     (produced by ``beach detect-rallies``).
  2. Finds the most recently modified ``*_fixed.json`` in the rally directory.
  3. Adds ``rally_start_sec`` to every ``timestamp_sec`` to restore absolute
     video time.
  4. Appends a ``rally_id`` field to each event.

Events from all rallies are merged in chronological order and written to a
single output file.

Usage
-----
    beach stitch-rallies videos/GH021569_court.mp4

    # Filter to a specific model when multiple fixed JSONs exist per rally:
    beach stitch-rallies videos/GH021569_court.mp4 --model gemini-2.5-pro

    # Override output path:
    beach stitch-rallies videos/GH021569_court.mp4 --output results/actions.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _find_fixed_json(rally_dir: Path, model_filter: Optional[str]) -> Optional[Path]:
    """Return the best *_fixed.json in *rally_dir*, or None if none found.

    When *model_filter* is set only files containing that string in their name
    are considered (e.g. ``'gemini-2.5-pro'``).
    Picks the most recently modified among the candidates.
    """
    candidates = list(rally_dir.glob("*_fixed.json"))
    if model_filter:
        candidates = [p for p in candidates if model_filter in p.name]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_stitch_rallies(
    video_path: Path,
    rallies_json: Optional[Path],
    output_path: Optional[Path],
    model_filter: Optional[str],
) -> Path:
    """Stitch per-rally *_fixed.json files into one absolute-timestamp JSON.

    Parameters
    ----------
    video_path:   The original full video (used to derive sibling paths).
    rallies_json: Override for the ``*_rallies.json`` file.
    output_path:  Override for the output file.
    model_filter: Only use fixed JSONs whose filename contains this string.

    Returns
    -------
    Path to the written output JSON.
    """
    video_dir  = video_path.parent
    video_stem = video_path.stem

    # Locate the rallies index JSON.
    rallies_path = rallies_json or (video_dir / f"{video_stem}_rallies.json")
    if not rallies_path.exists():
        raise click.ClickException(
            f"Rallies JSON not found: {rallies_path}\n"
            "Run 'beach detect-rallies' first."
        )

    rallies: list[dict] = json.loads(rallies_path.read_text())
    rallies_dir = video_dir / f"{video_stem}_rallies"

    if not rallies_dir.is_dir():
        raise click.ClickException(
            f"Rallies directory not found: {rallies_dir}\n"
            "Run 'beach split-rallies' first."
        )

    out_path = output_path or (video_dir / f"{video_stem}_actions.json")

    print(f"Video       : {video_path}")
    print(f"Rallies JSON: {rallies_path}  ({len(rallies)} rallies)")
    print(f"Rallies dir : {rallies_dir}")
    if model_filter:
        print(f"Model filter: {model_filter}")
    print()

    all_events: list[dict] = []
    skipped: list[str] = []

    col = f"  {'rally':<10}  {'offset':>8}  {'events':>6}  file"
    print(col)
    print("  " + "─" * (len(col) - 2))

    for rally in rallies:
        rid      = rally["rally_id"]
        rid_str  = f"rally_{rid:02d}"
        offset   = rally["start_sec"]
        rally_dir = rallies_dir / rid_str

        fixed_path = _find_fixed_json(rally_dir, model_filter)
        if fixed_path is None:
            label = f"(no {'model-matching ' if model_filter else ''}*_fixed.json)"
            print(f"  {rid_str:<10}  {offset:>7.2f}s  {'—':>6}  {label}")
            skipped.append(rid_str)
            continue

        events: list[dict] = json.loads(fixed_path.read_text())

        for ev in events:
            stitched = dict(ev)
            stitched["timestamp_sec"] = round(ev["timestamp_sec"] + offset, 3)
            stitched["rally_id"]      = rid
            stitched["rally_start_sec"] = offset
            all_events.append(stitched)

        print(f"  {rid_str:<10}  {offset:>7.2f}s  {len(events):>6}  {fixed_path.name}")

    print()

    # Sort by absolute timestamp (rallies should already be ordered, but be safe).
    all_events.sort(key=lambda e: e["timestamp_sec"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_events, indent=2))

    print(f"Total events : {len(all_events)}")
    if skipped:
        print(f"Skipped      : {', '.join(skipped)}")
    print(f"Written      → {out_path}")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("stitch-rallies")
@click.argument(
    "video",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--rallies-json",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Override the *_rallies.json file (default: <video_stem>_rallies.json sibling).",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output file (default: <video_stem>_actions.json sibling).",
)
@click.option(
    "--model",
    default=None,
    help=(
        "Only use fixed JSONs whose filename contains this string "
        "(e.g. 'gemini-2.5-pro').  Useful when multiple models were run."
    ),
)
def stitch_rallies_cmd(
    video: Path,
    rallies_json: Optional[Path],
    output: Optional[Path],
    model: Optional[str],
) -> None:
    """Stitch per-rally *_fixed.json files into one full-video action JSON.

    Reads rally start offsets from ``<video_stem>_rallies.json``, adds each
    rally's offset to the event timestamps, and writes a single sorted JSON.

    Each event gains two extra fields:

    \b
      rally_id        — which rally the event came from
      rally_start_sec — the rally's start time in the full video
    """
    print()
    run_stitch_rallies(video, rallies_json, output, model)
    print()
