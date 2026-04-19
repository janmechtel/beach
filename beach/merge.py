"""beach merge — Merge player identification + ball tracking + rallies into *_merged.json.

Joins:
  - *_identified_heuristic.json  (P1-P4 player bboxes per frame, from beach identify)
  - *_ball.csv                   (VballNet ball positions, from beach ball-track)
  - *_rallies.json               (rally windows, from beach detect-rallies)
    Falls back to ball field in the identified JSON when no CSV is available,
    but that is usually empty for this dataset.

Outputs *_merged.json:
{
  "fps": 50.0,
  "total_frames": 14200,
  "rallies": [...],             // from *_rallies.json; [] when not provided
  "frames": [
    {
      "frame": 42,
      "timestamp_sec": 0.84,
      "ball": {"x": 620, "y": 310, "visible": true},
      "closest_player_id": "P1",   // null when ball not visible or no player close
      "players": [
        {
          "player_id": "P1",
          "cx": 400.0, "cy": 600.0,
          "x1": 350.0, "y1": 320.0, "x2": 450.0, "y2": 680.0
        },
        ...
      ]
    },
    ...
  ]
}

Closest player is determined by Euclidean distance from ball centre to each
player's foot point (cx, y2 = bottom centre of bounding box).  Only assigned
when distance < PROXIMITY_THRESHOLD_PX; null otherwise.

Usage
-----
    beach merge --video videos/GH021569_court.mp4
    beach merge --video videos/GH021569_court.mp4 --ball videos/GH021569_court_ball.csv
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from beach.detect_touches import (
    compute_velocities,
    detect_touches as _detect_touches,
    deduplicate_events,
    format_touches,
    print_summary,
)
from typing import Optional

import click

PROXIMITY_THRESHOLD_PX = 400  # max ball-to-foot distance to assign closest player


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def build_merged(
    identified_path: Path,
    ball_csv_path: Optional[Path],
    rallies_path: Optional[Path],
    output_path: Path,
) -> None:
    """Merge player + ball + rally data into a single analytics JSON."""

    # --- Load identified JSON ---
    ident = json.loads(identified_path.read_text())
    id_frames: list[dict] = ident["frames"]

    # fps from timestamps
    fps = 50.0
    if len(id_frames) >= 2:
        last = id_frames[-1]
        if last["timestamp_sec"] > 0:
            fps = last["frame"] / last["timestamp_sec"]
    fps = round(fps, 4)

    # --- Load ball CSV (if provided) ---
    ball_by_frame: dict[int, dict] = {}
    if ball_csv_path and ball_csv_path.exists():
        import csv
        with open(ball_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_idx = int(row["Frame"])
                vis = int(row["Visibility"])
                x = float(row["X"])
                y = float(row["Y"])
                ball_by_frame[frame_idx] = {
                    "x": x if vis else -1.0,
                    "y": y if vis else -1.0,
                    "visible": bool(vis),
                }
        print(f"  Ball CSV loaded: {ball_csv_path.name}  "
              f"({sum(1 for b in ball_by_frame.values() if b['visible'])} visible / "
              f"{len(ball_by_frame)} frames)")
    else:
        # Fall back to ball field in identified JSON (often null)
        for f in id_frames:
            b = f.get("ball")
            if b:
                ball_by_frame[f["frame"]] = {
                    "x": float(b["cx"]),
                    "y": float(b["cy"]),
                    "visible": True,
                }
        if ball_csv_path:
            print(f"  WARNING: ball CSV not found ({ball_csv_path}) — "
                  f"using ball field from identified JSON "
                  f"({len(ball_by_frame)} frames with ball).")
        else:
            print(f"  No ball CSV provided — "
                  f"using ball field from identified JSON "
                  f"({len(ball_by_frame)} frames with ball).")

    # --- Load rallies (if provided) ---
    rallies: list[dict] = []
    if rallies_path and rallies_path.exists():
        rallies = json.loads(rallies_path.read_text())
        print(f"  Rallies loaded: {rallies_path.name}  ({len(rallies)} rallies)")

    # --- Build merged frames ---
    merged_frames: list[dict] = []
    total = len(id_frames)

    for fd in id_frames:
        frame_idx = fd["frame"]

        # Ball
        ball_info = ball_by_frame.get(frame_idx, {"x": -1.0, "y": -1.0, "visible": False})

        # Players — keep only fields needed downstream; drop embedding/colour noise
        players = []
        for p in fd.get("persons", []):
            pid = p.get("player_id")
            players.append({
                "player_id": pid,
                "cx": p["cx"],
                "cy": p["cy"],
                "x1": p["x1"],
                "y1": p["y1"],
                "x2": p["x2"],
                "y2": p["y2"],
            })

        # Closest player to ball
        closest_pid: Optional[str] = None
        if ball_info["visible"] and players:
            bx, by = ball_info["x"], ball_info["y"]
            best_dist = float("inf")
            for p in players:
                if p["player_id"] is None:
                    continue
                # foot point = bottom-centre of bbox
                foot_x = p["cx"]
                foot_y = p["y2"]
                dist = math.hypot(bx - foot_x, by - foot_y)
                if dist < best_dist:
                    best_dist = dist
                    closest_pid = p["player_id"]
            if best_dist > PROXIMITY_THRESHOLD_PX:
                closest_pid = None  # ball too far from all players

        merged_frames.append({
            "frame": frame_idx,
            "timestamp_sec": fd["timestamp_sec"],
            "ball": ball_info,
            "closest_player_id": closest_pid,
            "players": players,
        })

    # --- Detect touches ---
    vels = compute_velocities(merged_frames)
    raw_events = _detect_touches(merged_frames, vels)
    events = deduplicate_events(raw_events)
    touches = format_touches(events)
    print_summary(events, output_path.stem)
    print(f"  {len(touches)} touch event(s) detected")

    # --- Write output ---
    output = {
        "fps": fps,
        "total_frames": total,
        "proximity_threshold_px": PROXIMITY_THRESHOLD_PX,
        "rallies": rallies,
        "touches": touches,
        "frames": merged_frames,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    frames_with_ball = sum(1 for f in merged_frames if f["ball"]["visible"])
    frames_with_closest = sum(1 for f in merged_frames if f["closest_player_id"])
    print(f"  Merged JSON written: {output_path.name}  "
          f"({total} frames, {frames_with_ball} with ball, "
          f"{frames_with_closest} with closest player)")


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------
@click.command("merge")
@click.option(
    "--video", "-v",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Source video (used to derive sibling file paths).",
)
@click.option(
    "--identified", "-i",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Identified JSON (default: <video_stem>_identified_heuristic.json).",
)
@click.option(
    "--ball", "-b",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Ball CSV from ball-track (default: <video_stem>_ball.csv).",
)
@click.option(
    "--rallies", "-r",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Rallies JSON (default: <video_stem>_rallies.json).",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output path (default: <video_stem>_merged.json).",
)
def merge_cmd(
    video: Path,
    identified: Optional[Path],
    ball: Optional[Path],
    rallies: Optional[Path],
    output: Optional[Path],
) -> None:
    """Merge player bboxes + ball tracking + rallies into a single analytics JSON."""
    identified_path = identified or video.with_name(video.stem + "_identified_heuristic.json")
    ball_csv = ball or video.with_name(video.stem + "_ball.csv")
    rallies_path = rallies or video.with_name(video.stem + "_rallies.json")
    output_path = output or video.with_name(video.stem + "_merged.json")

    if not identified_path.exists():
        raise click.ClickException(
            f"Identified JSON not found: {identified_path}\n"
            "Run 'beach run' or 'beach identify --no-llm' first."
        )

    build_merged(
        identified_path=identified_path,
        ball_csv_path=ball_csv if ball_csv.exists() else None,
        rallies_path=rallies_path if rallies_path.exists() else None,
        output_path=output_path,
    )
