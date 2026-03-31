"""
Pass 1: Detect persons and ball; assign anonymous temporal track IDs (H1..Hn).

Pipeline
--------
1. Run YOLO yolo11n on the source video to detect persons (class 0) using
   ByteTrack (built into ultralytics) for robust cross-frame identity.
2. Run a volleyball-specific YOLO model for ball detection (optional).
3. ByteTrack assigns stable anonymous IDs H1, H2, … using Kalman-filter
   motion prediction + a two-stage re-association that recovers tracks after
   occlusions.  These IDs are NOT player IDs — they are anonymous appearance
   anchors for pass 2.
4. Write per-frame detections to --output-json (always).
5. Optionally render an annotated video to --render-video (only when flag given).

ID stability notes
------------------
- ByteTrack (tracker="bytetrack.yaml", persist=True) handles re-association
  internally.  It keeps a "lost" pool and tries to recover tracks before
  spawning new IDs, which avoids the ID-switch problem of pure IoU matching.
- When the number of active tracks exceeds the expected player count (4) the
  tracker will still assign IDs, but pass 2 should treat those detections as
  uncertain.

Output JSON schema
------------------
{
  "frames": [
    {
      "frame": 123,
      "timestamp_sec": 2.46,
      "persons": [
        {
          "cx": 512.3, "cy": 401.7,
          "x1": 470.0, "y1": 220.1, "x2": 556.2, "y2": 690.4,
          "conf": 0.91,
          "human_track_id": "H2"   # null when no stable track could be assigned
        }
      ],
      "ball": {"cx": 620.2, "cy": 188.0}   # null when not detected
    }
  ]
}

Usage
-----
    # JSON only (default — fast):
    uv run beach track --video chunks/GH021569_001.mp4 \
                             --output-json chunks/GH021569_001_detections.json

    # JSON + rendered video:
    uv run beach track --video chunks/GH021569_001.mp4 \
                             --output-json chunks/GH021569_001_detections.json \
                             --render-video chunks/GH021569_001_annotated.mp4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COCO_PERSON = 0

PERSON_CONF = 0.35
BALL_CONF   = 0.20


# Render constants (only used when --render-video is given)
BOX_THICKNESS  = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.60
FONT_THICKNESS = 2
LABEL_PAD      = 5
BALL_RADIUS    = 14
BALL_COLOR_BGR = (0, 220, 255)   # bright yellow in BGR
SWATCH_W       = 14              # colour swatch square appended to label tag
LABEL_GAP      = 4               # gap between text and swatch

# One BGR colour per H-track index so tracks are visually distinct when rendered.
_TRACK_COLORS = [
    (255, 255, 255),  # H1 — white
    (0,   200, 255),  # H2 — orange
    (100, 255, 100),  # H3 — green
    (255,  80, 200),  # H4 — magenta
    (255,  80,  80),  # H5 — blue (overflow)
    (200, 200,   0),  # H6 — cyan (overflow)
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bbox_centre(box_xyxy: np.ndarray) -> tuple[float, float]:
    return float((box_xyxy[0] + box_xyxy[2]) / 2), float((box_xyxy[1] + box_xyxy[3]) / 2)


def _track_color(h_id: Optional[str]) -> tuple[int, int, int]:
    if h_id is None:
        return (200, 200, 200)
    idx = int(h_id[1:]) - 1  # "H1" -> 0
    return _TRACK_COLORS[idx % len(_TRACK_COLORS)]


# Sand HSV bounds — pixels matching all three ranges are treated as background.
# Tuned for indoor beach volleyball courts: warm low-saturation beige/tan.
# Hue 8-28 covers the yellow-brown range; low S filters out vivid non-sand
# colours; V floor rejects shadows.
_SAND_H = (8,  28)
_SAND_S = (8,  110)
_SAND_V = (110, 255)


def _torso_color_hsv(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """Clothing colour descriptor for the torso region (rows 20%–65% of bbox).

    Two-stage filtering:
    1. Sand mask: removes background pixels that match the court colour.
    2. Mid-brightness gate (V 30–160): dark pixels carry no hue information
       (black shorts/hair collapse S to zero and noise H); very bright pixels
       are reflections or sand that slipped the mask.  Mean H and S are computed
       from what remains so that e.g. a black shirt next to a coloured one still
       shows a distinct tint difference.

    V is reported as the mean of ALL clothing pixels (not just mid-V) so that a
    predominantly dark outfit (low V) is distinguishable from a bright one.

    Returns [H, S, V] in OpenCV ranges (H: 0-180, S: 0-255, V: 0-255).
    Returns [0.0, 0.0, 0.0] when the crop is degenerate.
    """
    H_frame, W_frame = frame.shape[:2]
    bx1 = max(0, int(x1))
    bx2 = min(W_frame, int(x2))
    h_box = y2 - y1
    ty1 = max(0, int(y1 + 0.20 * h_box))
    ty2 = min(H_frame, int(y1 + 0.65 * h_box))
    if ty2 <= ty1 or bx2 <= bx1:
        return [0.0, 0.0, 0.0]
    crop = frame[ty1:ty2, bx1:bx2]
    if crop.size == 0:
        return [0.0, 0.0, 0.0]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    sand = (
        (H >= _SAND_H[0]) & (H <= _SAND_H[1]) &
        (S >= _SAND_S[0]) & (S <= _SAND_S[1]) &
        (V >= _SAND_V[0]) & (V <= _SAND_V[1])
    )
    clothing     = ~sand
    # Mid-brightness subset for H+S: excludes very dark pixels (no hue info)
    # and very bright ones (reflections / sand leakage).
    mid_v_clothing = clothing & (V >= 30) & (V <= 160)

    if clothing.sum() < 20:
        # Fully occluded or degenerate — fall back to raw mean.
        raw = cv2.mean(hsv)[:3]
        return [round(float(raw[0]), 1), round(float(raw[1]), 1), round(float(raw[2]), 1)]

    # H and S from mid-brightness clothing pixels; V from all clothing pixels.
    if mid_v_clothing.sum() >= 10:
        mean_h = float(H[mid_v_clothing].mean())
        mean_s = float(S[mid_v_clothing].mean())
    else:
        # Outfit is predominantly very dark (e.g. all-black) or very bright.
        mean_h = float(H[clothing].mean())
        mean_s = float(S[clothing].mean())
    mean_v = float(V[clothing].mean())

    return [round(mean_h, 1), round(mean_s, 1), round(mean_v, 1)]


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def run_detection(
    video_path: Path,
    json_path: Path,
    render_path: Optional[Path] = None,
    person_model_path: Path = Path("yolo11n.pt"),
    ball_model_path: Path = Path("volleyball_yolo11n.pt"),
) -> None:
    """Run YOLO person + ball detection on *video_path*.

    Writes JSON detections to *json_path*.  If *render_path* is given, also
    renders an annotated video (significantly slower due to re-encoding).
    """
    model = YOLO(str(person_model_path))

    ball_model = YOLO(str(ball_model_path)) if ball_model_path.exists() else None
    if ball_model is None:
        print("volleyball_yolo11n.pt not found — ball detection disabled.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Source : {video_path}  {w}×{h} @ {fps:.1f} fps  {total} frames")
    print(f"JSON   : {json_path}")
    if render_path:
        print(f"Render : {render_path}")
    else:
        print("Render : disabled (pass --render-video to enable)")

    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up optional writer
    writer = None
    if render_path:
        render_path.parent.mkdir(parents=True, exist_ok=True)
        # avc1 (H.264) triggers h264_v4l2m2m probe on Linux which always fails on
        # x86 — go straight to mp4v (MPEG-4 Part 2), universally available via FFmpeg.
        writer = cv2.VideoWriter(
            str(render_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        if not writer.isOpened():
            raise RuntimeError("Cannot open VideoWriter for render path")

    json_frames: list[dict] = []
    frames_with_ball = 0
    frame_idx = 0

    for result in model.track(
        source=str(video_path),
        classes=[COCO_PERSON],
        conf=PERSON_CONF,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
    ):
        frame = result.orig_img  # read-only reference; copy only if rendering
        if render_path:
            frame = frame.copy()

        # --- Persons ---
        persons_json: list[dict] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()      # (N, 4)
            conf = result.boxes.conf.cpu().numpy()       # (N,)

            # ByteTrack assigns integer IDs; format as H1/H2/… to preserve
            # the downstream contract.  id tensor is None if tracking lost all.
            raw_ids = result.boxes.id
            id_array = raw_ids.cpu().numpy().astype(int) if raw_ids is not None else None
            track_ids = [f"H{id_array[i]}" if id_array is not None else None
                         for i in range(len(xyxy))]

            for box, c, h_id in zip(xyxy, conf, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = _bbox_centre(box)
                color     = _track_color(h_id)
                color_hsv = _torso_color_hsv(
                    result.orig_img,
                    float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                )

                if render_path:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), BOX_THICKNESS + 1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
                    label = f"{h_id or '?'} {c:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
                    tag_w = tw + LABEL_PAD * 2 + LABEL_GAP + SWATCH_W
                    ly1   = max(y1 - th - LABEL_PAD * 2, 0)
                    cv2.rectangle(frame, (x1, ly1), (x1 + tag_w, y1), color, cv2.FILLED)
                    cv2.putText(frame, label, (x1 + LABEL_PAD, y1 - LABEL_PAD),
                                FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)
                    # Detected-colour swatch: non-sand torso colour from pass 1
                    if color_hsv and color_hsv != [0.0, 0.0, 0.0]:
                        h_val = min(int(round(color_hsv[0])), 179)
                        s_val = min(int(round(color_hsv[1])), 255)
                        # Boost V: floor at 50 so dark swatches are visible but still look dark.
                        v_val = max(int(round(color_hsv[2])), 50)
                        bgr = cv2.cvtColor(
                            np.array([[[h_val, s_val, v_val]]], dtype=np.uint8),
                            cv2.COLOR_HSV2BGR,
                        )[0][0].tolist()
                        sx1 = x1 + tw + LABEL_PAD * 2 + LABEL_GAP
                        sx2 = sx1 + SWATCH_W
                        sy1 = ly1 + 2
                        sy2 = y1 - 2
                        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 0), 1)
                        cv2.rectangle(
                            frame, (sx1 + 1, sy1 + 1), (sx2 - 1, sy2 - 1),
                            (int(bgr[0]), int(bgr[1]), int(bgr[2])), cv2.FILLED,
                        )

                persons_json.append({
                    "cx": round(cx, 1),
                    "cy": round(cy, 1),
                    "x1": round(float(box[0]), 1),
                    "y1": round(float(box[1]), 1),
                    "x2": round(float(box[2]), 1),
                    "y2": round(float(box[3]), 1),
                    "conf": round(float(c), 3),
                    "color_hsv": color_hsv,
                    "human_track_id": h_id,
                })

        # --- Ball ---
        ball_json = None
        if ball_model is not None:
            ball_result = ball_model.predict(
                source=result.orig_img, conf=BALL_CONF, verbose=False
            )[0]
            if ball_result.boxes is not None and len(ball_result.boxes) > 0:
                bboxes = ball_result.boxes.xyxy.cpu().numpy()
                areas  = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                best   = int(np.argmax(areas))
                bx, by = _bbox_centre(bboxes[best])
                ball_json = {"cx": round(bx, 1), "cy": round(by, 1)}
                frames_with_ball += 1

                if render_path:
                    cv2.circle(frame, (int(bx), int(by)), BALL_RADIUS + 2, (0, 0, 0), -1)
                    cv2.circle(frame, (int(bx), int(by)), BALL_RADIUS, BALL_COLOR_BGR, -1)
                    lbl = "BALL"
                    (btw, _), _ = cv2.getTextSize(lbl, FONT, 0.55, 2)
                    cv2.putText(frame, lbl, (int(bx) - btw // 2, int(by) - BALL_RADIUS - 8),
                                FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, lbl, (int(bx) - btw // 2, int(by) - BALL_RADIUS - 8),
                                FONT, 0.55, BALL_COLOR_BGR, 2, cv2.LINE_AA)

        json_frames.append({
            "frame": frame_idx,
            "timestamp_sec": round(frame_idx / fps, 4),
            "persons": persons_json,
            "ball": ball_json,
        })

        if writer is not None:
            writer.write(frame)

        frame_idx += 1
        if frame_idx % 150 == 0:
            print(f"  frame {frame_idx}/{total}")

    if writer is not None:
        writer.release()

    json_path.write_text(json.dumps({"frames": json_frames}, indent=2))

    print(f"\nDone. {frame_idx} frames processed.")
    print(f"Ball detected : {frames_with_ball}/{frame_idx} frames "
          f"({100 * frames_with_ball / max(frame_idx, 1):.1f}%)")
    if render_path and render_path.exists():
        print(f"Video written : {render_path}  ({render_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"JSON written  : {json_path}  ({json_path.stat().st_size / 1024:.1f} KB)")


@click.command("track")
@click.option("--video", "-v", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Input video file.")
@click.option("--output-json", "-j", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Output JSON path (default: <video_stem>_detections.json next to video).")
@click.option("--render-video", "-r", type=click.Path(dir_okay=False, path_type=Path), default=None, help="If given, render annotated video to this path (slow).")
@click.option("--person-model", type=click.Path(dir_okay=False, path_type=Path), default=Path("yolo11n.pt"), show_default=True, help="YOLO model for person detection.")
@click.option("--ball-model", type=click.Path(dir_okay=False, path_type=Path), default=Path("volleyball_yolo11n.pt"), show_default=True, help="YOLO model for ball detection.")
def track_cmd(video, output_json, render_video, person_model, ball_model):
    """Pass 1: YOLO person + ball detection with ByteTrack IDs."""
    json_path = output_json or video.with_name(video.stem + "_detections.json")
    run_detection(
        video_path=video,
        json_path=json_path,
        render_path=render_video,
        person_model_path=person_model,
        ball_model_path=ball_model,
    )
