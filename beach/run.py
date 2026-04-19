"""beach run — end-to-end pipeline: track → annotate first frame → identify → render.

Steps
-----
1. track    — YOLO person + ball detection → *_detections.json
2. annotate — browser UI shows the first clean 4-player frame; assign P1–P4,
              confirm with Enter, then Ctrl-C to close → *_gt.json
3. identify — no-LLM rolling tracker seeded from the confirmed GT frame
              → *_identified_heuristic.json
4. render   — overlay coloured boxes + player labels onto source video
              → *_rendered.mp4

Use --skip-track / --skip-annotate to reuse existing outputs and iterate on
the later steps without re-running the slow parts.

Example
-------
    beach run --video videos/GH021569_court_001.mp4
    # iterate on identify/render without re-tracking or re-annotating:
    beach run --video videos/GH021569_court_001.mp4 --skip-track --skip-annotate
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from typing import Optional

import click
import cv2

from beach.annotate_gt import (
    DEFAULT_PLAYERS_PATH,
    _build_new_gt,
    _load_json,
    _load_or_init_players,
    _build_players_config,
    _merge_existing_gt,
    _save_gt,
    extract_annotated_frame,
    preseed_keyframes,
    run_annotation_server,
    select_first_frame,
)
from beach.identify import _render_identified, identify_players
from beach.track import run_detection


# ---------------------------------------------------------------------------
# Internal helper: run the annotation server for a single seed frame
# ---------------------------------------------------------------------------

def _annotate_first_frame(
    video: Path,
    detections_path: Path,
    output_path: Path,
    players_path: Path,
    port: int,
) -> None:
    """Open the browser annotation UI for the first clean 4-player frame.

    Blocks until the user confirms the frame and presses Ctrl-C (or the
    server is shut down via /api/exit).  Raises ClickException when no
    frame is confirmed on exit.
    """
    det_json = _load_json(detections_path)
    det_frames = det_json.get("frames", [])

    existing_gt: dict | None = None
    if output_path.exists():
        existing_payload = _load_json(output_path)
        if isinstance(existing_payload, dict):
            existing_gt = existing_payload

    players_data = _load_or_init_players(players_path)
    players_ui, player_colors_rgb = _build_players_config(players_data)

    selected = select_first_frame(det_frames)
    if not selected:
        raise click.ClickException(
            "No frames with person detections found in the detections JSON. "
            "Check that 'beach track' ran successfully."
        )

    keyframes = preseed_keyframes(selected, [], player_colors_rgb)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise click.ClickException(f"Cannot open video: {video}")
    try:
        for kf in keyframes:
            kf["image_b64"] = extract_annotated_frame(cap, kf["frame_idx"])
    finally:
        cap.release()

    gt = _build_new_gt(video, detections_path, keyframes)
    if existing_gt is not None:
        gt = _merge_existing_gt(gt, existing_gt)

    state: dict = {"keyframes": keyframes, "gt": gt, "players_ui": players_ui}

    url = f"http://localhost:{port}"
    print(f"  Opening {url} — assign P1–P4 to each detected player.")
    print(f"  Press Enter (or click Confirm frame) when done, then Ctrl-C here to continue.")
    webbrowser.open(url)

    try:
        run_annotation_server(state, output_path, port)
    except KeyboardInterrupt:
        pass
    finally:
        _save_gt(state["gt"], output_path)

    confirmed = [a for a in state["gt"].get("annotations", []) if a.get("confirmed")]
    if not confirmed:
        raise click.ClickException(
            "No frames were confirmed. Assign P1–P4 to all detected persons and "
            "press Enter before stopping the server."
        )
    print(f"  GT saved: {output_path}  ({len(confirmed)} confirmed frame(s))")


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@click.command("run")
@click.option(
    "--video", "-v",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input video file.",
)
@click.option(
    "--render",
    is_flag=True,
    default=False,
    help="Render an output video with overlaid boxes (default: off).",
)
@click.option(
    "--render-output", "-r",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output rendered video path (default: <stem>_rendered.mp4 next to video). Implies --render.",
)
@click.option(
    "--skip-track",
    is_flag=True,
    default=False,
    help="Reuse existing *_detections.json and skip YOLO pass.",
)
@click.option(
    "--skip-annotate",
    is_flag=True,
    default=False,
    help="Reuse existing *_gt.json and skip first-frame annotation.",
)
@click.option(
    "--players",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_PLAYERS_PATH,
    show_default=True,
    help="players.json for player names/colors (created with defaults if missing).",
)
@click.option(
    "--port",
    type=int,
    default=7780,
    show_default=True,
    help="Port for the annotation browser UI.",
)
def run_cmd(
    video: Path,
    render: bool,
    render_output: Optional[Path],
    skip_track: bool,
    skip_annotate: bool,
    players: Path,
    port: int,
) -> None:
    """Full pipeline: track → annotate first frame → identify → render."""

    detections_path  = video.with_name(video.stem + "_detections.json")
    gt_path          = video.with_name(video.stem + "_gt.json")
    identified_path  = video.with_name(video.stem + "_identified_heuristic.json")
    render_path      = render_output or video.with_name(video.stem + "_rendered.mp4")
    do_render        = render or (render_output is not None)

    # ------------------------------------------------------------------
    # Step 1: Track
    # ------------------------------------------------------------------
    if skip_track and detections_path.exists():
        print(f"[1/4] track     — skipped (reusing {detections_path.name})")
    else:
        print(f"\n[1/4] track     — YOLO + ByteTrack → {detections_path.name}")
        run_detection(video_path=video, json_path=detections_path)

    # ------------------------------------------------------------------
    # Step 2: Annotate first frame
    # ------------------------------------------------------------------
    if skip_annotate and gt_path.exists():
        print(f"[2/4] annotate  — skipped (reusing {gt_path.name})")
    else:
        print(f"\n[2/4] annotate  — first-frame seed UI → {gt_path.name}")
        _annotate_first_frame(video, detections_path, gt_path, players, port)

    # ------------------------------------------------------------------
    # Step 3: Identify
    # ------------------------------------------------------------------
    print(f"\n[3/4] identify  — no-LLM rolling tracker + GT seed → {identified_path.name}")
    identify_players(
        video_path=video,
        detections_path=detections_path,
        output_path=identified_path,
        render_path=None,
        sample_window_frac=0.30,
        api_key="",
        use_llm=False,
        use_embeddings=False,
        seed_gt_path=gt_path,
    )

    # ------------------------------------------------------------------
    # Step 4: Render
    # ------------------------------------------------------------------
    if do_render:
        print(f"\n[4/4] render    — overlay boxes → {render_path.name}")
        identified_data = json.loads(identified_path.read_text())
        _render_identified(video, identified_data["frames"], render_path)
    else:
        print(f"[4/4] render    — skipped (use --render to enable)")

    print(f"\nDone.\n  Detections : {detections_path}\n  GT seed    : {gt_path}\n  Identified : {identified_path}")
    if do_render:
        print(f"  Rendered   : {render_path}")
