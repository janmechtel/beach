# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ultralytics>=8.3",
#   "opencv-python-headless>=4.9",
#   "scipy>=1.11",
# ]
# ///
"""
Pass 1: Detect persons and ball; assign anonymous temporal track IDs (H1..Hn).

Pipeline
--------
1. Run YOLO yolo11n on the source video to detect persons (class 0).
2. Run a volleyball-specific YOLO model for ball detection (optional).
3. Track persons across frames with a greedy IoU tracker, assigning stable
   anonymous IDs H1, H2, … for the duration they remain visible.  These IDs
   are NOT player IDs — they are anonymous appearance anchors for pass 2.
4. Write per-frame detections to --output-json (always).
5. Optionally render an annotated video to --render-video (only when flag given).

ID stability notes
------------------
- IoU-based greedy matching.  A track is kept alive for MAX_MISSING_FRAMES
  frames after its last detection so brief occlusions don't break continuity.
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
    uv run annotate_video.py --video chunks/GH021569_001.mp4 \
                             --output-json chunks/GH021569_001_detections.json

    # JSON + rendered video:
    uv run annotate_video.py --video chunks/GH021569_001.mp4 \
                             --output-json chunks/GH021569_001_detections.json \
                             --render-video chunks/GH021569_001_annotated.mp4
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COCO_PERSON = 0

PERSON_CONF = 0.35
BALL_CONF   = 0.20

# Tracker: a track is kept alive this many frames after last detection.
# Beach volleyball is ~30 fps; 0.5 s = 15 frames covers a brief occlusion.
MAX_MISSING_FRAMES = 15
# Minimum IoU to consider a detection a match for an existing track.
IOU_MATCH_THRESHOLD = 0.25

# Render constants (only used when --render-video is given)
BOX_THICKNESS  = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.60
FONT_THICKNESS = 2
LABEL_PAD      = 5
BALL_RADIUS    = 14
BALL_COLOR_BGR = (0, 220, 255)   # bright yellow in BGR

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
# IoU tracker
# ---------------------------------------------------------------------------
@dataclass
class _Track:
    track_id: str          # "H1", "H2", …
    box: np.ndarray        # [x1,y1,x2,y2] float32
    frames_since_seen: int = 0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class IoUTracker:
    """Greedy frame-to-frame IoU tracker.

    Assigns stable H-IDs to detections across frames.  No Kalman filter —
    beach volleyball players move fast but the camera is usually static, so
    plain box overlap is sufficient for continuity over short occlusions.
    """

    def __init__(self) -> None:
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: np.ndarray) -> list[Optional[str]]:
        """Match detections to existing tracks; return per-detection H-IDs.

        Parameters
        ----------
        detections:
            Shape (N, 4) float32 array of [x1,y1,x2,y2] boxes.

        Returns
        -------
        List of N strings ("H1", "H2", …) or None for unmatched detections
        that did not reach IOU_MATCH_THRESHOLD against any active track.
        In practice None should be rare once tracks are established.
        """
        n_det = len(detections)
        assigned_track_ids: list[Optional[str]] = [None] * n_det

        # Age all active tracks
        for t in self._tracks:
            t.frames_since_seen += 1

        if n_det == 0:
            # Prune dead tracks
            self._tracks = [t for t in self._tracks if t.frames_since_seen <= MAX_MISSING_FRAMES]
            return assigned_track_ids

        # Build IoU matrix: tracks × detections
        active = [t for t in self._tracks if t.frames_since_seen <= MAX_MISSING_FRAMES]
        if active:
            iou_matrix = np.zeros((len(active), n_det), dtype=np.float32)
            for ti, track in enumerate(active):
                for di, det in enumerate(detections):
                    iou_matrix[ti, di] = _iou(track.box, det)

            # Greedy matching: repeatedly pick the highest IoU pair
            matched_tracks: set[int] = set()
            matched_dets:   set[int] = set()
            while True:
                if iou_matrix.max() < IOU_MATCH_THRESHOLD:
                    break
                ti, di = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                ti, di = int(ti), int(di)
                if ti in matched_tracks or di in matched_dets:
                    iou_matrix[ti, di] = 0.0
                    continue
                # Accept match
                active[ti].box = detections[di].astype(np.float32)
                active[ti].frames_since_seen = 0
                assigned_track_ids[di] = active[ti].track_id
                matched_tracks.add(ti)
                matched_dets.add(di)
                iou_matrix[ti, :] = 0.0
                iou_matrix[:, di] = 0.0

        # Spawn new tracks for unmatched detections
        for di in range(n_det):
            if assigned_track_ids[di] is None:
                new_id = f"H{self._next_id}"
                self._next_id += 1
                new_track = _Track(
                    track_id=new_id,
                    box=detections[di].astype(np.float32),
                    frames_since_seen=0,
                )
                self._tracks.append(new_track)
                assigned_track_ids[di] = new_id

        # Prune dead tracks
        self._tracks = [t for t in self._tracks if t.frames_since_seen <= MAX_MISSING_FRAMES]

        return assigned_track_ids


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


def _torso_color_hsv(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """Mean HSV of the torso region: rows 20%–65% of bbox height, full width.

    Skips the head (top 20%) and legs/sand (bottom 35%) to focus on shirt color.
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
    means = cv2.mean(hsv)[:3]
    return [round(float(means[0]), 1), round(float(means[1]), 1), round(float(means[2]), 1)]


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def run_detection(
    video_path: Path,
    json_path: Path,
    render_path: Optional[Path] = None,
) -> None:
    """Run YOLO person + ball detection on *video_path*.

    Writes JSON detections to *json_path*.  If *render_path* is given, also
    renders an annotated video (significantly slower due to re-encoding).
    """
    model = YOLO("yolo11n.pt")

    ball_model_path = Path("volleyball_yolo11n.pt")
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

    tracker = IoUTracker()
    json_frames: list[dict] = []
    frames_with_ball = 0
    frame_idx = 0

    for result in model.predict(
        source=str(video_path),
        classes=[COCO_PERSON],
        conf=PERSON_CONF,
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

            track_ids = tracker.update(xyxy)

            for box, c, h_id in zip(xyxy, conf, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = _bbox_centre(box)
                color = _track_color(h_id)

                if render_path:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), BOX_THICKNESS + 1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
                    label = f"{h_id or '?'} {c:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
                    ly1 = max(y1 - th - LABEL_PAD * 2, 0)
                    cv2.rectangle(frame, (x1, ly1), (x1 + tw + LABEL_PAD * 2, y1), color, cv2.FILLED)
                    cv2.putText(frame, label, (x1 + LABEL_PAD, y1 - LABEL_PAD),
                                FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)

                persons_json.append({
                    "cx": round(cx, 1),
                    "cy": round(cy, 1),
                    "x1": round(float(box[0]), 1),
                    "y1": round(float(box[1]), 1),
                    "x2": round(float(box[2]), 1),
                    "y2": round(float(box[3]), 1),
                    "conf": round(float(c), 3),
                    "color_hsv": _torso_color_hsv(
                        result.orig_img,
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ),
                    "human_track_id": h_id,
                })
        else:
            # No detections — still advance the tracker (ages all tracks)
            tracker.update(np.empty((0, 4), dtype=np.float32))

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pass 1: YOLO person + ball detection with anonymous track IDs. "
            "Writes a JSON detections file; renders video only when --render-video is given."
        )
    )
    parser.add_argument(
        "--video", "-v",
        type=Path,
        required=True,
        help="Input video file.",
    )
    parser.add_argument(
        "--output-json", "-j",
        type=Path,
        default=None,
        help="Output JSON path (default: <video_stem>_detections.json next to the video).",
    )
    parser.add_argument(
        "--render-video", "-r",
        type=Path,
        default=None,
        help="If given, render an annotated video to this path (slow; disabled by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    video_path: Path = args.video

    json_path: Path = args.output_json or video_path.with_name(
        video_path.stem + "_detections.json"
    )
    render_path: Optional[Path] = args.render_video

    run_detection(
        video_path=video_path,
        json_path=json_path,
        render_path=render_path,
    )


if __name__ == "__main__":
    main()
