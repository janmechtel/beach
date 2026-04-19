"""beach analytics — Full analytics pipeline: ball-track → detect-rallies → merge → render.

Requires 'beach run' (or 'beach identify --no-llm') to have been run first so
that *_identified_heuristic.json already exists.

ball-track and detect-rallies are independent of the player pipeline and run
first; merge is the convergence point that joins everything and runs touch detection.

Steps
-----
1. ball-track      — VballNet inference → *_ball.csv
2. detect-rallies  — ball visibility → *_rallies.json  (reads *_ball.csv only)
3. merge           — players + ball + rallies + touches → *_merged.json
4. render          — overlay everything onto source video → *_analytics.mp4

Use --skip-* flags to reuse existing intermediate files.

Usage
-----
    beach analytics --video videos/GH021569_court.mp4
    beach analytics --video videos/GH021569_court.mp4 --skip-ball-track
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from beach.ball_track import run_ball_tracking, _DEFAULT_MODEL
from beach.merge import build_merged
from beach.rallies import detect_rallies
from beach.analytics_render import render_analytics


@click.command("analytics")
@click.option(
    "--video", "-v",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Source video file.",
)
@click.option(
    "--render",
    is_flag=True, default=False,
    help="Render an output video with analytics overlay (default: off).",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output rendered video (default: <video_stem>_analytics.mp4). Implies --render.",
)
@click.option(
    "--skip-ball-track",
    is_flag=True, default=False,
    help="Reuse existing *_ball.csv and skip VballNet inference.",
)
@click.option(
    "--skip-rallies",
    is_flag=True, default=False,
    help="Reuse existing *_rallies.json and skip rally detection.",
)
@click.option(
    "--skip-merge",
    is_flag=True, default=False,
    help="Reuse existing *_merged.json and skip merge step.",
)
@click.option(
    "--skip-render",
    is_flag=True, default=False,
    help="Deprecated: render is now off by default. Kept for backwards compatibility.",
)
@click.option(
    "--model",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help=f"VballNet ONNX model (default: {_DEFAULT_MODEL.name}).",
)
def analytics_cmd(
    video: Path,
    render: bool,
    output: Optional[Path],
    skip_ball_track: bool,
    skip_rallies: bool,
    skip_merge: bool,
    skip_render: bool,
    model: Optional[Path],
) -> None:
    """Full analytics pipeline: ball-track → detect-rallies → merge → render."""

    ball_csv     = video.with_name(video.stem + "_ball.csv")
    rallies_path = video.with_name(video.stem + "_rallies.json")
    merged_path  = video.with_name(video.stem + "_merged.json")
    render_path  = output or video.with_name(video.stem + "_analytics.mp4")
    identified   = video.with_name(video.stem + "_identified_heuristic.json")
    do_render    = render or (output is not None)

    if not identified.exists():
        raise click.ClickException(
            f"Identified JSON not found: {identified}\n"
            "Run 'beach run' first to generate player tracking data."
        )

    # ------------------------------------------------------------------
    # Step 1: Ball tracking
    # ------------------------------------------------------------------
    if skip_ball_track and ball_csv.exists():
        print(f"[1/5] ball-track     — skipped (reusing {ball_csv.name})")
    else:
        print(f"\n[1/5] ball-track     — VballNet inference → {ball_csv.name}")
        run_ball_tracking(
            video_path=video,
            output_csv=ball_csv,
            model_path=model or _DEFAULT_MODEL,
            skip_existing=False,
        )

    # ------------------------------------------------------------------
    # Step 2: Rally detection  (reads ball.csv only — no player data needed)
    # ------------------------------------------------------------------
    import cv2 as _cv2
    cap = _cv2.VideoCapture(str(video))
    _fps = cap.get(_cv2.CAP_PROP_FPS) or 50.0
    _total_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()

    if skip_rallies and rallies_path.exists():
        print(f"[2/5] detect-rallies — skipped (reusing {rallies_path.name})")
    else:
        print(f"\n[2/5] detect-rallies — ball visibility → {rallies_path.name}")
        detect_rallies(
            ball_csv_path=ball_csv,
            output_path=rallies_path,
            fps=_fps,
            total_frames=_total_frames,
        )

    # ------------------------------------------------------------------
    # Step 3: Merge  (players + ball + rallies + touch detection)
    # ------------------------------------------------------------------
    if skip_merge and merged_path.exists():
        print(f"[3/4] merge          — skipped (reusing {merged_path.name})")
    else:
        print(f"\n[3/4] merge          — players + ball + rallies + touches → {merged_path.name}")
        build_merged(
            identified_path=identified,
            ball_csv_path=ball_csv if ball_csv.exists() else None,
            rallies_path=rallies_path if rallies_path.exists() else None,
            output_path=merged_path,
        )

    # ------------------------------------------------------------------
    # Step 4: Render
    # ------------------------------------------------------------------
    if not do_render or skip_render:
        print(f"[4/4] render         — skipped (use --render to enable)")
    else:
        print(f"\n[4/4] render         — analytics overlay → {render_path.name}")
        render_analytics(
            video_path=video,
            merged_path=merged_path,
            output_path=render_path,
        )

    print(f"\nDone.")
    print(f"  Ball CSV    : {ball_csv}")
    print(f"  Rallies     : {rallies_path}")
    print(f"  Merged      : {merged_path}  (includes touches)")
    if do_render and not skip_render:
        print(f"  Rendered    : {render_path}")
