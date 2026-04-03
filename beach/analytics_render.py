"""beach analytics-render — Render merged analytics overlay onto the source video.

Draws on every frame:
  • Player bounding boxes — coloured by P1-P4, label "P1 Name"
  • Closest player to ball — gold/yellow thicker box + crown indicator
  • Ball position — bright yellow filled circle
  • Rally state banner — top-centre strip: "● RALLY N" during rally,
    flashes "START" / "END" at transitions for a brief window

Usage
-----
    beach analytics-render --video videos/GH021569_court.mp4
    beach analytics-render --video v.mp4 --merged v_merged.json \\
                            --rallies v_rallies.json --output v_analytics.mp4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD       = cv2.FONT_HERSHEY_DUPLEX
BOX_THICKNESS   = 2
CLOSEST_THICKNESS = 4
BALL_RADIUS     = 12
LABEL_PAD       = 4
FONT_SCALE      = 0.60
FONT_THICKNESS  = 2

# Per-player colours (BGR)
PLAYER_COLORS: dict[str, tuple[int, int, int]] = {
    "P1": (200, 200, 200),   # white-ish  (Denny)
    "P2": ( 50, 200, 255),   # orange     (O-Love)
    "P3": ( 50, 200,  50),   # green      (Ibu 800)
    "P4": ( 50, 255, 200),   # yellow-green (Bjirk)
}
UNKNOWN_COLOR   = (120, 120, 120)
CLOSEST_COLOR   = (0, 215, 255)   # gold — closest player highlight
BALL_COLOR      = (0, 220, 255)   # bright yellow

# Rally banner
BANNER_H            = 40          # pixels
RALLY_COLOR         = (0, 180, 0) # green text during rally
TRANSITION_FRAMES   = 60          # how long START/END flash lasts (~1.2s at 50fps)
TRANSITION_COLOR    = (0, 255, 255)  # yellow for transition text

# Player names (fallback when not in JSON)
_DEFAULT_NAMES: dict[str, str] = {
    "P1": "P1", "P2": "P2", "P3": "P3", "P4": "P4",
}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _draw_player(
    frame: np.ndarray,
    p: dict,
    player_names: dict[str, str],
    is_closest: bool,
) -> None:
    pid   = p.get("player_id")
    color = PLAYER_COLORS.get(pid, UNKNOWN_COLOR)
    x1, y1 = int(p["x1"]), int(p["y1"])
    x2, y2 = int(p["x2"]), int(p["y2"])
    name  = player_names.get(pid, pid or "?")
    label = f"{pid} {name}" if pid else "?"

    thickness = CLOSEST_THICKNESS if is_closest else BOX_THICKNESS
    border_color = CLOSEST_COLOR if is_closest else color

    # Black shadow box, then coloured box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness + 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, thickness)

    # Label background + text
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    tag_y0 = max(y1 - th - LABEL_PAD * 2, 0)
    cv2.rectangle(frame, (x1, tag_y0), (x1 + tw + LABEL_PAD * 2, y1), border_color, cv2.FILLED)
    cv2.putText(
        frame, label,
        (x1 + LABEL_PAD, y1 - LABEL_PAD),
        FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA,
    )

    # Gold crown dot above box for closest player
    if is_closest:
        crown_y = max(y1 - th - LABEL_PAD * 2 - 8, 4)
        crown_x = (x1 + x2) // 2
        cv2.circle(frame, (crown_x, crown_y), 6, CLOSEST_COLOR, -1)
        cv2.circle(frame, (crown_x, crown_y), 6, (0, 0, 0), 1)


def _draw_ball(frame: np.ndarray, bx: int, by: int) -> None:
    cv2.circle(frame, (bx, by), BALL_RADIUS + 2, (0, 0, 0), -1)
    cv2.circle(frame, (bx, by), BALL_RADIUS, BALL_COLOR, -1)
    # Small cross to make it easier to see exact position
    cv2.line(frame, (bx - BALL_RADIUS - 4, by), (bx + BALL_RADIUS + 4, by), (0, 0, 0), 1)
    cv2.line(frame, (bx, by - BALL_RADIUS - 4), (bx, by + BALL_RADIUS + 4), (0, 0, 0), 1)


def _draw_rally_banner(
    frame: np.ndarray,
    rally_id: Optional[int],
    transition_label: Optional[str],
    transition_alpha: float,
    frame_width: int,
) -> None:
    """Draw semi-transparent rally state banner across the top of the frame."""
    if rally_id is None and transition_label is None:
        return

    overlay = frame.copy()

    # Background strip
    bg_color = (20, 80, 20) if rally_id is not None else (20, 20, 80)
    cv2.rectangle(overlay, (0, 0), (frame_width, BANNER_H), bg_color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Main rally label
    if rally_id is not None:
        main_text = f"RALLY  {rally_id + 1}"
        (tw, th), _ = cv2.getTextSize(main_text, FONT_BOLD, 0.8, 2)
        tx = frame_width // 2 - tw // 2
        ty = BANNER_H // 2 + th // 2
        cv2.putText(frame, main_text, (tx, ty),
                    FONT_BOLD, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, main_text, (tx, ty),
                    FONT_BOLD, 0.8, RALLY_COLOR, 2, cv2.LINE_AA)

    # Transition flash (START / END)
    if transition_label and transition_alpha > 0:
        alpha_int = max(0, min(255, int(transition_alpha * 255)))
        (tw2, th2), _ = cv2.getTextSize(transition_label, FONT_BOLD, 1.1, 3)
        # Centre-right to not overlap main label
        tx2 = frame_width * 3 // 4 - tw2 // 2
        ty2 = BANNER_H // 2 + th2 // 2
        # Draw with alpha fade using a temporary overlay
        tmp = frame.copy()
        cv2.putText(tmp, transition_label, (tx2, ty2),
                    FONT_BOLD, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(tmp, transition_label, (tx2, ty2),
                    FONT_BOLD, 1.1, TRANSITION_COLOR, 3, cv2.LINE_AA)
        cv2.addWeighted(tmp, transition_alpha, frame, 1 - transition_alpha, 0, frame)


# ---------------------------------------------------------------------------
# Build rally lookup
# ---------------------------------------------------------------------------
def _build_rally_lookup(
    rallies: list[dict],
    total_frames: int,
) -> tuple[list[Optional[int]], list[Optional[str]], list[float]]:
    """Return per-frame arrays:
        rally_id_per_frame  — rally index or None
        transition_per_frame — "START" / "END" / None
        transition_alpha    — 0.0..1.0 fade value
    """
    rally_id_arr: list[Optional[int]] = [None] * total_frames
    transition_arr: list[Optional[str]] = [None] * total_frames
    alpha_arr: list[float] = [0.0] * total_frames

    for r in rallies:
        rid    = r["rally_id"]
        sf     = r["start_frame"]
        ef     = min(r["end_frame"], total_frames - 1)
        raw_sf = r.get("raw_start_frame", sf)
        raw_ef = r.get("raw_end_frame", ef)

        for fi in range(sf, ef + 1):
            if fi < total_frames:
                rally_id_arr[fi] = rid

        # START flash: first TRANSITION_FRAMES of the rally
        for i in range(TRANSITION_FRAMES):
            fi = sf + i
            if fi >= total_frames:
                break
            alpha = 1.0 - (i / TRANSITION_FRAMES)
            if transition_arr[fi] is None or alpha > alpha_arr[fi]:
                transition_arr[fi] = "START"
                alpha_arr[fi] = alpha

        # END flash: last TRANSITION_FRAMES of the rally
        for i in range(TRANSITION_FRAMES):
            fi = ef - i
            if fi < 0:
                break
            alpha = 1.0 - (i / TRANSITION_FRAMES)
            if transition_arr[fi] is None or alpha > alpha_arr[fi]:
                transition_arr[fi] = "END"
                alpha_arr[fi] = alpha

    return rally_id_arr, transition_arr, alpha_arr


# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------
def render_analytics(
    video_path: Path,
    merged_path: Path,
    output_path: Path,
) -> None:
    # --- Load data ---
    merged = json.loads(merged_path.read_text())
    fps_data: float = merged.get("fps", 50.0)
    merged_frames: list[dict] = merged["frames"]
    total_frames: int = merged.get("total_frames", len(merged_frames))

    # Player names from merged JSON (may not be present — fall back to IDs)
    player_names = dict(_DEFAULT_NAMES)

    rallies: list[dict] = merged.get("rallies", [])

    # Build per-frame lookup for O(1) access during render loop
    frame_lookup: dict[int, dict] = {f["frame"]: f for f in merged_frames}

    # Rally state arrays
    rally_id_arr, transition_arr, alpha_arr = _build_rally_lookup(rallies, total_frames)

    # --- Open video ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or fps_data

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {output_path}")

    print(f"  Rendering analytics to {output_path.name} …")
    print(f"  {w}×{h} @ {fps:.1f} fps  {total_frames} frames  "
          f"{len(rallies)} rallies")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fd = frame_lookup.get(idx)
        if fd:
            closest_pid = fd.get("closest_player_id")

            # Draw players (non-closest first so closest renders on top)
            non_closest = [p for p in fd["players"] if p.get("player_id") != closest_pid]
            closest_p   = [p for p in fd["players"] if p.get("player_id") == closest_pid]
            for p in non_closest:
                _draw_player(frame, p, player_names, is_closest=False)
            for p in closest_p:
                _draw_player(frame, p, player_names, is_closest=True)

            # Draw ball
            ball = fd.get("ball", {})
            if ball.get("visible"):
                _draw_ball(frame, int(ball["x"]), int(ball["y"]))

        # Rally banner (even on frames without player data)
        rid        = rally_id_arr[idx] if idx < total_frames else None
        trans      = transition_arr[idx] if idx < total_frames else None
        trans_alpha = alpha_arr[idx] if idx < total_frames else 0.0
        _draw_rally_banner(frame, rid, trans, trans_alpha, w)

        writer.write(frame)
        idx += 1
        if idx % 500 == 0:
            print(f"  … frame {idx}/{total_frames}")

    cap.release()
    writer.release()
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Done. {output_path.name}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------
@click.command("analytics-render")
@click.option(
    "--video", "-v",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Source video file.",
)
@click.option(
    "--merged", "-m",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Merged JSON (default: <video_stem>_merged.json).",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output video (default: <video_stem>_analytics.mp4).",
)
def analytics_render_cmd(
    video: Path,
    merged: Optional[Path],
    output: Optional[Path],
) -> None:
    """Render analytics overlay (players + ball + rally markers) onto the video."""
    merged_path  = merged or video.with_name(video.stem + "_merged.json")
    output_path  = output or video.with_name(video.stem + "_analytics.mp4")

    if not merged_path.exists():
        raise click.ClickException(
            f"Merged JSON not found: {merged_path}\n"
            "Run 'beach merge' first."
        )

    render_analytics(
        video_path=video,
        merged_path=merged_path,
        output_path=output_path,
    )
