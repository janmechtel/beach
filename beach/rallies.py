"""beach detect-rallies — Detect rally start/end times from ball visibility.

Reads *_ball.csv (from beach ball-track) and groups contiguous ball-visible
stretches into rallies, separated by pauses longer than --max-pause seconds.

Output *_rallies.json:
[
  {
    "rally_id": 0,
    "start_frame": 312,
    "end_frame":   784,
    "start_sec":   6.24,
    "end_sec":    15.68
  },
  ...
]

Start/end are extended by --extension seconds before/after the raw ball
visibility window so clips include the moment before the serve and the
moment after the point.

Usage
-----
    beach detect-rallies --video videos/GH021569_court.mp4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def detect_rallies(
    ball_csv_path: Path,
    output_path: Path,
    fps: float = 50.0,
    total_frames: Optional[int] = None,
    max_pause_sec: float = 2.0,
    min_rally_sec: float = 3.0,
    extension_sec: float = 1.0,
) -> list[dict]:
    """Detect rallies from ball visibility in *_ball.csv.

    Returns the list of rally dicts (also written to *output_path*).
    """
    import csv

    ball_visible: dict[int, bool] = {}
    with open(ball_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ball_visible[int(row["Frame"])] = bool(int(row["Visibility"]))

    if not ball_visible:
        raise ValueError(f"No data in ball CSV: {ball_csv_path}")

    if total_frames is None:
        total_frames = max(ball_visible.keys()) + 1

    max_pause_frames = int(max_pause_sec * fps)
    min_rally_frames = int(min_rally_sec * fps)
    extension_frames = int(extension_sec * fps)

    # --- Find raw rally segments (contiguous ball-visible runs) ---
    # Group frames by whether ball is visible; merge runs separated by short gaps.
    raw_rallies: list[tuple[int, int]] = []  # (start_frame, end_frame)
    current_start: Optional[int] = None
    last_visible_frame: Optional[int] = None

    for fidx in range(total_frames):
        visible = ball_visible.get(fidx, False)

        if visible:
            if current_start is None:
                current_start = fidx
            last_visible_frame = fidx
        else:
            # Check if this gap is long enough to end the current rally
            if current_start is not None and last_visible_frame is not None:
                gap = fidx - last_visible_frame
                if gap > max_pause_frames:
                    raw_rallies.append((current_start, last_visible_frame))
                    current_start = None
                    last_visible_frame = None

    # Close any open rally at end of video
    if current_start is not None and last_visible_frame is not None:  # noqa: SIM102
        raw_rallies.append((current_start, last_visible_frame))

    # --- Filter short rallies + apply extension ---
    rallies: list[dict] = []
    rally_id = 0
    for raw_start, raw_end in raw_rallies:
        duration_frames = raw_end - raw_start
        if duration_frames < min_rally_frames:
            continue  # too short — likely a stray detection

        ext_start = max(0, raw_start - extension_frames)
        ext_end = min(total_frames - 1, raw_end + extension_frames)

        rallies.append({
            "rally_id": rally_id,
            "start_frame": ext_start,
            "end_frame": ext_end,
            "start_sec": round(ext_start / fps, 3),
            "end_sec": round(ext_end / fps, 3),
            "raw_start_frame": raw_start,
            "raw_end_frame": raw_end,
        })
        rally_id += 1

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rallies, indent=2))

    total_rally_sec = sum(r["end_sec"] - r["start_sec"] for r in rallies)
    print(f"  Rallies detected: {len(rallies)}  "
          f"(total rally time: {total_rally_sec:.1f}s / "
          f"{total_frames/fps:.1f}s video)")
    for r in rallies:
        print(f"    Rally {r['rally_id']}: "
              f"{r['start_sec']:.1f}s – {r['end_sec']:.1f}s  "
              f"({r['end_sec']-r['start_sec']:.1f}s)")

    output_path.write_text(json.dumps(rallies, indent=2))
    return rallies


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------
@click.command("detect-rallies")
@click.option(
    "--video", "-v",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Source video (used to derive sibling file paths and read fps).",
)
@click.option(
    "--ball", "-b",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Ball CSV (default: <video_stem>_ball.csv).",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output path (default: <video_stem>_rallies.json).",
)
@click.option("--max-pause", default=2.0, show_default=True, type=float,
              help="Pause longer than this (seconds) splits rallies.")
@click.option("--min-rally", default=3.0, show_default=True, type=float,
              help="Rallies shorter than this (seconds) are discarded.")
@click.option("--extension", default=1.0, show_default=True, type=float,
              help="Extend each rally start/end by this many seconds.")
def detect_rallies_cmd(
    video: Path,
    ball: Optional[Path],
    output: Optional[Path],
    max_pause: float,
    min_rally: float,
    extension: float,
) -> None:
    """Detect rally timings from ball tracking CSV."""
    import cv2

    ball_csv = ball or video.with_name(video.stem + "_ball.csv")
    output_path = output or video.with_name(video.stem + "_rallies.json")

    if not ball_csv.exists():
        raise click.ClickException(
            f"Ball CSV not found: {ball_csv}\n"
            "Run 'beach ball-track' first."
        )

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()

    detect_rallies(
        ball_csv_path=ball_csv,
        output_path=output_path,
        fps=fps,
        total_frames=total_frames,
        max_pause_sec=max_pause,
        min_rally_sec=min_rally,
        extension_sec=extension,
    )
