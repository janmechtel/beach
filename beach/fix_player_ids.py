"""beach fix-player-ids — Replace LLM-assigned player IDs in an action JSON
with ground-truth players from the corresponding rally's *_merged.json.

Given a video file the script automatically discovers:
  • The rally's *_merged.json (touches ground truth) — same directory as the
    video.
  • The LLM-produced action JSON — any file in that directory whose name starts
    with ``<video_stem>_`` but is not a ``_merged.json`` or ``_fixed.json``.
    When several candidates exist the most recently modified one is used; pass
    ``--llm-json`` to override.

For each touch event the script:
  1. Matches the LLM event to the nearest entry in the rally's ``touches`` list
     (embedded in *_merged.json by ``beach analytics``) by timestamp.
  2. Takes the player ID from that touch entry as ground truth.
  3. Replaces player_id with the touch player.
  4. Adds diagnostic fields: original_player_id, corrected, matched_touch_sec,
     matched_frame, delta_sec, touch_angle_deg.

Output is written next to the LLM JSON with ``_fixed`` appended before the
``.json`` extension.

Usage
-----
    beach fix-player-ids videos/GH021569_court_rallies/rally_00/rally_00.mp4

    # Explicit LLM JSON (skip auto-discovery):
    beach fix-player-ids rally_00.mp4 --llm-json path/to/rally_00_gemini-2.5-pro_20260403_123456.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click


# ── core helpers ──────────────────────────────────────────────────────────────


def _find_merged_json(video_dir: Path) -> Path:
    """Return the *_merged.json in *video_dir*."""
    candidates = list(video_dir.glob("*_merged.json"))
    if not candidates:
        raise click.ClickException(
            f"No *_merged.json found in {video_dir}\n"
            "Run 'beach split-rallies' then 'beach analytics' first."
        )
    if len(candidates) > 1:
        # Prefer the one whose name matches the directory name (most specific).
        dir_match = [c for c in candidates if c.stem.rstrip("_merged") == video_dir.name]
        if dir_match:
            return dir_match[0]
    return candidates[0]


def _find_touches(video_dir: Path) -> list[dict]:
    """Return touch events from the rally's *_merged.json in *video_dir*."""
    merged_path = _find_merged_json(video_dir)
    data = json.loads(merged_path.read_text())
    touches = data.get("touches")
    if not touches:
        raise click.ClickException(
            f"No 'touches' key in {merged_path}\n"
            "Run 'beach analytics' (includes detect-touches) first."
        )
    return touches


def _find_llm_json(video_path: Path) -> Path:
    """Auto-discover the LLM action JSON for *video_path*.

    Looks in the same directory for files whose name starts with
    ``<video_stem>_`` and excludes ``*_merged.json`` and ``*_fixed.json``.
    When multiple candidates exist, the most recently modified file wins.

    The analyze.py output pattern is::

        {video_stem}_{model_tag}[_seeded][_run{N}]_{YYYYMMDD_HHMMSS}.json

    so any extra suffix after the video stem is valid (model name, run tag,
    timestamp, etc.).
    """
    video_dir = video_path.parent
    stem = video_path.stem

    candidates = [
        p for p in video_dir.glob(f"{stem}_*.json")
        if not p.name.endswith("_merged.json")
        and not p.name.endswith("_fixed.json")
        and p.name != ".gemini_file_cache.json"
    ]

    if not candidates:
        raise click.ClickException(
            f"No LLM action JSON found in {video_dir} for video stem '{stem}'.\n"
            "Run 'beach analyze' first, or pass --llm-json explicitly."
        )

    if len(candidates) == 1:
        return candidates[0]

    # Multiple matches — pick the most recently modified.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(
        f"Multiple LLM JSONs found — using most recent: {candidates[0].name}\n"
        f"  (others: {', '.join(p.name for p in candidates[1:])})\n"
        f"  Pass --llm-json to override."
    )
    return candidates[0]


def _match_touch(touches: list[dict], target_sec: float, exclude_player: Optional[str] = None) -> dict:
    """Return the touches entry whose timestamp is closest to target_sec.

    When *exclude_player* is set, only touches from a different player are
    considered (used to prevent assigning the same player twice in a row).
    Falls back to unrestricted matching when no alternative touch exists.
    """
    pool = [t for t in touches if t["player"] != exclude_player] if exclude_player else touches
    if not pool:
        pool = touches  # no alternative available — fall back
    return min(pool, key=lambda t: abs(t["timestamp_sec"] - target_sec))


def _apply_touch(ev: dict, touch: dict) -> None:
    """Mutate *ev* in-place to reflect *touch* as the ground-truth match."""
    delta = touch["timestamp_sec"] - ev["timestamp_sec"]
    ev["player_id"]         = touch["player"]
    ev["corrected"]         = ev["original_player_id"] != touch["player"]
    ev["matched_touch_sec"] = touch["timestamp_sec"]
    ev["matched_frame"]     = touch["frame"]
    ev["delta_sec"]         = round(delta, 3)
    ev["touch_angle_deg"]   = touch["angle_change_deg"]


def _resolve_consecutive_duplicates(fixed: list[dict], touches: list[dict]) -> None:
    """Mutate *fixed* in-place so no two adjacent events share the same player.

    When a consecutive repeat is found the second event is re-matched to the
    nearest touch whose player differs from the previous event's player.
    Iterates until stable so that a repair cannot introduce a new duplicate
    further along the list.
    """
    changed = True
    while changed:
        changed = False
        for i in range(1, len(fixed)):
            if fixed[i]["player_id"] == fixed[i - 1]["player_id"]:
                touch = _match_touch(touches, fixed[i]["timestamp_sec"],
                                     exclude_player=fixed[i - 1]["player_id"])
                _apply_touch(fixed[i], touch)
                changed = True


def fix_events(events: list[dict], touches: list[dict]) -> list[dict]:
    """Patch player_id in every event using the nearest touch entry.

    After the initial nearest-touch assignment a consecutive-duplicate pass
    ensures no two adjacent events are credited to the same player (which is
    physically impossible in beach volleyball).
    """
    fixed = []
    for ev in events:
        touch = _match_touch(touches, ev["timestamp_sec"])
        new_ev = dict(ev)
        new_ev["original_player_id"] = ev["player_id"]
        # Populate all touch-derived fields via shared helper.
        _apply_touch(new_ev, touch)
        fixed.append(new_ev)

    _resolve_consecutive_duplicates(fixed, touches)
    return fixed


def _print_summary(fixed_events: list[dict]) -> None:
    corrections = sum(1 for e in fixed_events if e["corrected"])
    print(f"Corrections  : {corrections} / {len(fixed_events)}\n")
    print(
        f"  {'':1}  {'LLM t':>6}  {'action':<18}  "
        f"{'LLM':>4}  {'truth':>5}  {'touch t':>7}  {'delta':>6}  {'frame':>5}  {'angle':>6}"
    )
    sep = "  " + "─" * 74
    print(sep)
    for e in fixed_events:
        marker = "✗" if e["corrected"] else "✓"
        print(
            f"  {marker}  {e['timestamp_sec']:>6.2f}s  {e['action']:<18}  "
            f"{str(e['original_player_id']):>4}  {str(e['player_id']):>5}  "
            f"{e['matched_touch_sec']:>6.2f}s  "
            f"{e['delta_sec']:>+6.2f}s  "
            f"{e['matched_frame']:>5}  "
            f"{e['touch_angle_deg']:>5.1f}°"
        )
    print(sep)


# ── core runner (usable from other modules) ───────────────────────────────────


def run_fix_player_ids(
    video_path: Path,
    llm_json: Optional[Path] = None,
) -> Path:
    """Fix player IDs for the given *video_path*.

    Parameters
    ----------
    video_path: Path to the rally video (used to locate sibling files).
    llm_json:   Explicit LLM action JSON.  Auto-discovered when *None*.

    Returns
    -------
    Path to the written *_fixed.json.
    """
    video_dir = video_path.parent

    llm_path = llm_json or _find_llm_json(video_path)
    touches  = _find_touches(video_dir)

    print(f"Video   : {video_path}")
    print(f"LLM file: {llm_path}")

    events: list[dict] = json.loads(llm_path.read_text())

    print(f"LLM events  : {len(events)}")
    print(f"Touch events: {len(touches)}")
    print()

    fixed_events = fix_events(events, touches)
    _print_summary(fixed_events)

    out_path = llm_path.with_stem(llm_path.stem + "_fixed")
    out_path.write_text(json.dumps(fixed_events, indent=2))
    print(f"\nWritten → {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────


@click.command("fix-player-ids")
@click.argument(
    "video",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--llm-json",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Explicit path to the LLM action JSON.  When omitted the most recently "
        "modified '<video_stem>_*.json' file in the video's directory is used."
    ),
)
def fix_player_ids_cmd(
    video: Path,
    llm_json: Optional[Path],
) -> None:
    """Fix player IDs in an action JSON using ground-truth touch data.

    VIDEO is the rally video file.  The script automatically finds:

    \b
      • *_merged.json  — touch ground truth (same directory as VIDEO)
      • LLM action JSON — most recent <video_stem>_*.json in that directory
        (pass --llm-json to pick a specific file)

    Matches each LLM-assigned event to the nearest touch by timestamp and
    replaces player_id with the tracker-derived ground truth.  Adds diagnostic
    fields (original_player_id, corrected, delta_sec, …).

    Output: <llm_stem>_fixed.json written next to the LLM JSON.
    """
    print()
    run_fix_player_ids(video, llm_json)
    print()
