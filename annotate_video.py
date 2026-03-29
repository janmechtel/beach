# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ultralytics>=8.3",
#   "opencv-python-headless>=4.9",
# ]
# ///
"""
Annotate a beach volleyball video with YOLO person detections and ball markers.

Pipeline
--------
1. Run YOLO yolo11n on the source video to detect persons (class 0) and
   optionally the ball via a volleyball-specific model.
2. Draw a white bounding box + confidence score for every detected person.
3. Draw a yellow circle for the ball when detected.
4. Encode the result as data/first30_annotated_<timestamp>.mp4.
5. Write a JSON of per-frame detections alongside the video.

No player ID assignment, no tracking, no ByteTrack — pure per-frame inference.

Usage
-----
    uv run annotate_video.py [--video data/first30.mp4]
                             [--output data/first30_annotated.mp4]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COCO_PERSON = 0

PERSON_CONF = 0.35
BALL_CONF   = 0.20

BOX_THICKNESS  = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.60
FONT_THICKNESS = 2
LABEL_PAD      = 5

BALL_RADIUS    = 14
BALL_COLOR_BGR = (0, 220, 255)   # bright yellow in BGR
PERSON_COLOR   = (255, 255, 255) # white


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def bbox_centre(box_xyxy: np.ndarray) -> tuple[float, float]:
    return float((box_xyxy[0] + box_xyxy[2]) / 2), float((box_xyxy[1] + box_xyxy[3]) / 2)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def annotate_video(video_path: Path, output_path: Path) -> None:
    model = YOLO("yolo11n.pt")

    ball_model_path = Path("volleyball_yolo11n.pt")
    ball_model = YOLO(str(ball_model_path)) if ball_model_path.exists() else None
    if ball_model is None:
        print("volleyball_yolo11n.pt not found — ball detection disabled.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps        = cap.get(cv2.CAP_PROP_FPS)
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Source : {video_path}  {w}×{h} @ {fps:.1f} fps  {total} frames")
    print(f"Output : {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h)
    )
    if not writer.isOpened():
        print("avc1 unavailable — falling back to mp4v")
        writer.release()
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
    if not writer.isOpened():
        raise RuntimeError("Cannot open VideoWriter")

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
        frame = result.orig_img.copy()

        # --- Persons ---
        persons_json: list[dict] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()

            for box, c in zip(xyxy, conf):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = bbox_centre(box)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), BOX_THICKNESS + 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), PERSON_COLOR, BOX_THICKNESS)

                label = f"person {c:.2f}"
                (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
                ly1 = max(y1 - th - LABEL_PAD * 2, 0)
                cv2.rectangle(frame, (x1, ly1), (x1 + tw + LABEL_PAD * 2, y1), PERSON_COLOR, cv2.FILLED)
                cv2.putText(frame, label, (x1 + LABEL_PAD, y1 - LABEL_PAD),
                            FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)

                persons_json.append({
                    "cx": round(cx, 1), "cy": round(cy, 1),
                    "x1": round(float(box[0]), 1), "y1": round(float(box[1]), 1),
                    "x2": round(float(box[2]), 1), "y2": round(float(box[3]), 1),
                    "conf": round(float(c), 3),
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
                bx, by = bbox_centre(bboxes[best])
                ball_json = {"cx": round(bx, 1), "cy": round(by, 1)}
                frames_with_ball += 1

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

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 150 == 0:
            print(f"  frame {frame_idx}/{total}")

    writer.release()

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps({"frames": json_frames}, indent=2))

    print(f"\nDone. {frame_idx} frames processed.")
    print(f"Ball detected : {frames_with_ball}/{frame_idx} frames ({100*frames_with_ball/max(frame_idx,1):.1f}%)")
    print(f"Video written : {output_path}  ({output_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"JSON written  : {json_path}  ({json_path.stat().st_size/1024:.1f} KB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Annotate volleyball video with YOLO person detection."
    )
    parser.add_argument("--video",  "-v", type=Path, default=Path("data/first30.mp4"))
    parser.add_argument("--output", "-o", type=Path,
                        default=Path(f"data/first30_annotated_{ts}.mp4"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    annotate_video(video_path=args.video, output_path=args.output)


if __name__ == "__main__":
    main()
