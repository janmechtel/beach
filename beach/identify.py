from __future__ import annotations

"""
Pass 2: Identify which anonymous track (H1..Hn) corresponds to which named
player (P1..P4) using Gemini vision, then propagate that mapping across all
frames in the pass-1 JSON.

Algorithm
---------
1. Load the pass-1 detections JSON and open the source video.
2. Sample "calibration frames": evenly-spaced frames near the start of the
   clip where exactly 4 persons are detected and no track is missing.
   Default window: first 30 % of the video (configurable with --sample-window).
3. For each calibration frame:
   a. Extract the full frame as a JPEG.
   b. Crop each person's bounding box into a JPEG thumbnail.
   c. Call Gemini with the full frame + numbered crops + a player-description
      prompt.  Response is a JSON array mapping detection index → player ID.
4. Consensus across calibration frames: for each (human_track_id, player_id)
   pair, count votes across all Gemini responses.  Accept a mapping when one
   player_id holds a strict majority (> 50 %) of votes for that track.
5. Propagate: for every frame, assign player_id via the accepted mapping.
   When a detection's human_track_id is not in the mapping (e.g. a spurious
   extra detection), try to assign by nearest-centroid distance to the
   expected court positions of each player; fall back to null.
6. Write the enriched JSON to --output.
   Optionally render an identified video with coloured named boxes (--render-identified).

Outputs
-------
Enriched JSON schema:
{
  "players": {
    "P1": {"name": "Denny"},
    "P2": {"name": "O-Love"},
    "P3": {"name": "Ibu 800"},
    "P4": {"name": "Bjirk"}
  },
  "track_map": {
    "H2": "P1",
    "H1": "P3",
    ...
  },
  "calibration": {
    "frames_sampled": 6,
    "frames_used": 5,
    "consensus_threshold": 0.5
  },
  "frames": [
    {
      "frame": 0,
      "timestamp_sec": 0.0,
      "persons": [
        {
          "cx": 512.3, "cy": 401.7,
          "x1": 470.0, "y1": 220.1, "x2": 556.2, "y2": 690.4,
          "conf": 0.91,
          "human_track_id": "H2",
          "player_id": "P1"   # null when unresolvable
        }
      ],
      "ball": null
    }
  ]
}
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
from google import genai
from google.genai import types
from scipy.optimize import linear_sum_assignment

from beach.paths import identified_path
# ---------------------------------------------------------------------------
# Players — fixed roster
# ---------------------------------------------------------------------------
PLAYERS: dict[str, dict] = {
    "P1": {"name": "Denny",    "description": "black tshirt with sleeves",          "team": "A"},
    "P2": {"name": "O-Love",   "description": "grey shirt, bald",                    "team": "A"},
    "P3": {"name": "Ibu 800",  "description": "blue shirt",                          "team": "B"},
    "P4": {"name": "Bjirk",    "description": "black tank-top, green/yellow shorts", "team": "B"},
}
PLAYER_IDS = ["P1", "P2", "P3", "P4"]

# ---------------------------------------------------------------------------
# Gemini config
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"   # fast + vision-capable; adequate for crops
GEMINI_TEMPERATURE = 0.1            # near-deterministic

# Gemini response schema: list of {detection_index, player_id}
_ID_SCHEMA = types.Schema(
    type="ARRAY",
    items=types.Schema(
        type="OBJECT",
        properties={
            "detection_index": types.Schema(type="INTEGER"),
            "player_id":       types.Schema(type="STRING", enum=PLAYER_IDS),
        },
        required=["detection_index", "player_id"],
    ),
)

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------
# Sample up to this many calibration frames.
MAX_CALIB_FRAMES = 8
# Only consider frames where ALL 4 expected players are detected with a unique track each.
EXACT_PLAYER_COUNT = 4
# Minimum fraction of calibration votes for a track→player assignment to be accepted.
CONSENSUS_THRESHOLD = 0.5
# Sample from this fraction of the video (early portion has stable court presence).
SAMPLE_WINDOW_FRAC = 0.30
# Render constants
BOX_THICKNESS  = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.65
FONT_THICKNESS = 2
LABEL_PAD      = 5
BALL_RADIUS    = 14
BALL_COLOR_BGR = (0, 220, 255)
# No-LLM seed: number of frames to observe after seed frame for colour refs.
COLOR_SEED_FRAMES = 30

# One distinct BGR colour per player ID
_PLAYER_COLORS: dict[str, tuple[int, int, int]] = {
    "P1": (200, 200, 200),   # near-white (Denny)
    "P2": ( 50, 200, 255),   # orange     (O-Love)
    "P3": ( 50, 200,  50),   # green      (Ibu 800)
    "P4": ( 50, 255, 200),   # yellow-green (Bjirk)
}
_UNKNOWN_COLOR = (120, 120, 120)


# ---------------------------------------------------------------------------
# Frame extraction helpers
# ---------------------------------------------------------------------------
def _extract_jpeg(cap: cv2.VideoCapture, frame_idx: int, quality: int = 90) -> bytes:
    """Seek to *frame_idx* and return the frame as raw JPEG bytes."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx}")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for frame {frame_idx}")
    return buf.tobytes()


def _crop_jpeg(
    cap: cv2.VideoCapture,
    frame_idx: int,
    x1: float, y1: float, x2: float, y2: float,
    pad: int = 20,
    quality: int = 85,
) -> bytes:
    """Return a padded crop of the bounding box as JPEG bytes."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx}")
    H, W = frame.shape[:2]
    ix1 = max(0, int(x1) - pad)
    iy1 = max(0, int(y1) - pad)
    ix2 = min(W, int(x2) + pad)
    iy2 = min(H, int(y2) + pad)
    crop = frame[iy1:iy2, ix1:ix2]
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode (crop) failed")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Calibration frame selection
# ---------------------------------------------------------------------------
def _min_pairwise_dist(persons: list[dict]) -> float:
    """Smallest Euclidean distance between any two person centroids."""
    pts = [(p["cx"], p["cy"]) for p in persons]
    best = float("inf")
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = ((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2) ** 0.5
            if d < best:
                best = d
    return best


def _color_distance(c1: list[float], c2: list[float]) -> float:
    """Normalized distance between two [H, S, V] color vectors (0.0–1.0).

    Hue distance is circular (OpenCV H: 0–180) and weighted by the lower of
    the two saturation values — unsaturated colours (grey, black) are close
    to all hues, so hue alone is not a useful discriminator for them.
    """
    h1, s1, v1 = c1
    h2, s2, v2 = c2
    hue_diff = min(abs(h1 - h2), 180.0 - abs(h1 - h2)) / 90.0  # 0-1
    sat_weight = min(s1, s2) / 255.0
    hue_cost = hue_diff * sat_weight
    sat_cost = abs(s1 - s2) / 255.0
    val_cost = abs(v1 - v2) / 255.0
    return 0.5 * hue_cost + 0.3 * sat_cost + 0.2 * val_cost


def _select_calib_frames(
    frames: list[dict],
    total_video_frames: int,
    sample_window_frac: float,
    n: int,
) -> list[dict]:
    """Pick up to *n* calibration frames from the early window.

    Selection rules:
    - Candidates must have exactly EXACT_PLAYER_COUNT persons, each with a unique
      non-null human_track_id.
    - Slot 0 is ALWAYS the candidate with the highest minimum pairwise player
      distance (best-spread frame).  This guarantees the rolling tracker is seeded
      from a frame where all 4 players are clearly separated, preventing the
      degenerate case where two players start at the same pixel.
    - Remaining slots are filled evenly from the rest of the candidates.

    Returns a list of frame dicts (subset of *frames*).
    """
    window_end_frame = int(total_video_frames * sample_window_frac)

    candidates = [
        f for f in frames
        if f["frame"] <= window_end_frame
        and len(f["persons"]) == EXACT_PLAYER_COUNT
        and all(p.get("human_track_id") is not None for p in f["persons"])
        and len({p["human_track_id"] for p in f["persons"]}) == EXACT_PLAYER_COUNT
    ]

    if not candidates:
        print(
            f"  WARNING: no frames with exactly {EXACT_PLAYER_COUNT} uniquely-tracked "
            f"persons found in the first {sample_window_frac*100:.0f}% of the video. "
            "Calibration will sample any frames with 4 persons, regardless of window."
        )
        candidates = [
            f for f in frames
            if len(f["persons"]) == EXACT_PLAYER_COUNT
            and all(p.get("human_track_id") is not None for p in f["persons"])
            and len({p["human_track_id"] for p in f["persons"]}) == EXACT_PLAYER_COUNT
        ]

    if not candidates:
        return []

    # Slot 0: best-spread frame (max of min pairwise distance).
    best_frame = max(candidates, key=lambda f: _min_pairwise_dist(f["persons"]))
    spread = _min_pairwise_dist(best_frame["persons"])
    print(f"  Best-spread seed frame: {best_frame['frame']} ({best_frame['timestamp_sec']:.1f}s), "
          f"min pairwise dist = {spread:.0f} px")

    if len(candidates) <= 1:
        return [best_frame]

    # Remaining slots: evenly-spaced from candidates excluding the best-spread frame.
    rest = [f for f in candidates if f is not best_frame]
    remaining_slots = n - 1
    if len(rest) <= remaining_slots:
        return [best_frame] + rest
    step = len(rest) / remaining_slots
    return [best_frame] + [rest[int(i * step)] for i in range(remaining_slots)]

# ---------------------------------------------------------------------------
# Gemini calibration call
# ---------------------------------------------------------------------------
def _build_prompt(detections: list[dict]) -> str:
    roster = "\n".join(
        f"  {pid}: {PLAYERS[pid]['name']} — {PLAYERS[pid]['description']}"
        for pid in PLAYER_IDS
    )
    det_list = "\n".join(
        f"  Detection {i}: the cropped image labelled 'Detection {i}'"
        for i in range(len(detections))
    )
    return f"""You are identifying beach volleyball players from video frames.

Roster (use ONLY these IDs):
{roster}

You are shown:
  1. The full video frame (first image).
  2. Then {len(detections)} cropped images of individual detected persons,
     in order: Detection 0, Detection 1, …, Detection {len(detections)-1}.

For EACH detection, identify which player it shows.
Each player must appear exactly once. If you cannot determine a player with
reasonable confidence, still assign the best match — do not leave any
detection unassigned.

Return a JSON array with one object per detection:
  [{{ "detection_index": <int>, "player_id": "<P1|P2|P3|P4>" }}, ...]

Detections to classify:
{det_list}
"""


def _call_gemini(
    client: genai.Client,
    full_frame_bytes: bytes,
    crops: list[bytes],
    detections: list[dict],
) -> Optional[list[dict]]:
    """Call Gemini with one full frame + N crops.

    Returns a list of {detection_index, player_id} dicts, or None on failure.
    """
    parts: list[types.Part] = []
    parts.append(types.Part.from_bytes(data=full_frame_bytes, mime_type="image/jpeg"))
    for crop in crops:
        parts.append(types.Part.from_bytes(data=crop, mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=_build_prompt(detections)))

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_ID_SCHEMA,
                temperature=GEMINI_TEMPERATURE,
            ),
        )
        raw = response.text
        assignments = json.loads(raw)
        # Validate: must have one entry per detection, valid player IDs
        if not isinstance(assignments, list) or len(assignments) != len(detections):
            print(f"    Gemini returned {len(assignments) if isinstance(assignments, list) else '?'} "
                  f"assignments for {len(detections)} detections — skipping frame")
            return None
        player_ids_used = {a["player_id"] for a in assignments}
        if len(player_ids_used) != len(detections):
            # Duplicate player IDs — Gemini hallucinated; still usable as partial evidence
            print(f"    Gemini returned duplicate player IDs: {player_ids_used} — using anyway")
        return assignments
    except Exception as exc:
        print(f"    Gemini call failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Consensus building
# ---------------------------------------------------------------------------
def _build_consensus(
    calib_results: list[list[dict]],
    calib_frame_persons: list[list[dict]],
    threshold: float,
) -> dict[str, str]:
    """Derive track_id → player_id mapping from calibration votes.

    For each (track_id, player_id) pair, accumulate how many calibration
    frames nominated that player for that track.  Accept when the top vote
    exceeds *threshold* fraction of votes for that track.

    Returns a partial dict; missing tracks must be handled by the caller.
    """
    # votes[track_id][player_id] = count
    votes: dict[str, Counter] = defaultdict(Counter)

    for frame_assignments, frame_persons in zip(calib_results, calib_frame_persons):
        for assignment in frame_assignments:
            di = assignment["detection_index"]
            pid = assignment["player_id"]
            if di < len(frame_persons):
                hid = frame_persons[di].get("human_track_id")
                if hid:
                    votes[hid][pid] += 1

    track_map: dict[str, str] = {}
    for hid, counter in votes.items():
        total = sum(counter.values())
        top_pid, top_count = counter.most_common(1)[0]
        if total > 0 and top_count / total > threshold:
            track_map[hid] = top_pid
        else:
            print(f"  Track {hid}: inconclusive votes {dict(counter)} — excluded from map")

    return track_map


def _resolve_conflicts(track_map: dict[str, str]) -> dict[str, str]:
    """If two H-tracks map to the same player, keep the one with more votes
    by running Hungarian assignment on the vote matrix.

    This is a safety net; in normal operation (4 players, 4 tracks) it should
    be a no-op.
    """
    # Check for duplicate player assignments
    assigned_players = list(track_map.values())
    if len(assigned_players) == len(set(assigned_players)):
        return track_map  # clean — no conflicts

    print("  WARNING: multiple tracks map to the same player. Running conflict resolution.")
    tracks = list(track_map.keys())
    n_tracks = len(tracks)
    n_players = len(PLAYER_IDS)

    # Build a cost matrix: lower cost = better assignment
    # We just try to keep as many existing mappings as possible
    cost = np.ones((n_tracks, n_players), dtype=float)
    for ti, hid in enumerate(tracks):
        pid = track_map[hid]
        pi = PLAYER_IDS.index(pid)
        cost[ti, pi] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    resolved: dict[str, str] = {}
    for ti, pi in zip(row_ind, col_ind):
        resolved[tracks[ti]] = PLAYER_IDS[pi]
    return resolved



# ---------------------------------------------------------------------------
# No-LLM bootstrap
# ---------------------------------------------------------------------------
# GT-derived HSV colour templates for no-LLM seeding.
# These are the cluster centres from the manually annotated clip.
# Used by _seed_from_detections to assign players at boot time instead of
# the catastrophically wrong left-to-right order.
_SEED_COLOR_TEMPLATES: dict[str, list[float]] = {
    "P1": [85.0,  34.0,  95.0],  # Denny    — dark shirt, warm low-sat
    "P2": [78.0,  28.0, 105.0],  # O-Love   — grey, lowest sat, brightest
    "P3": [100.0, 114.0, 118.0], # Ibu 800  — blue, high sat, clearly distinct
    "P4": [68.0,  50.0,  88.0],  # Bjirk    — dark tank, moderate sat
}


def _nms_iou(a: dict, b: dict) -> float:
    """IoU between two person detection dicts (each has x1,y1,x2,y2 fields)."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _seed_from_detections(
    frames: list[dict],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[float]]]:
    """Seed player slots from the earliest full-court frame; no LLM required.

    Uses Hungarian assignment on HSV colour-template distance to assign
    P1..P4 to the detections in the seed frame.  This replaces the old
    left-to-right ordering which was catastrophically wrong whenever a player
    was off-screen or the court was rotated relative to expectation.

    Falls back to left-to-right if no detection has colour data (e.g. legacy
    detections JSON without color_hsv fields).

    Then runs a lightweight rolling tracker for COLOR_SEED_FRAMES frames to
    build per-player colour references from the torso HSV stored in the
    detections JSON.

    Returns:
        player_pos       -- initial rolling positions {pid: np.array([cx, cy])}
        calib_anchor     -- copy of initial positions (stale-track reset target)
        player_color_ref -- mean HSV per player {pid: [H, S, V]}; empty if none
    """
    player_pos: dict[str, np.ndarray] = {}
    seed_fi = -1

    # IoU threshold for suppressing duplicate detections in the seed frame.
    # ByteTrack occasionally fires two IDs on the same person (observed IoU≈0.55).
    # A seed frame with such duplicates assigns a real player slot to a ghost,
    # poisoning the rolling tracker for the entire clip.
    SEED_NMS_IOU = 0.3

    for frame in frames:
        persons = frame["persons"]
        if len(persons) < EXACT_PLAYER_COUNT:
            continue
        candidates = persons[:EXACT_PLAYER_COUNT]
        # Prefer the top-4 by confidence if more than 4 detected.
        if len(persons) > EXACT_PLAYER_COUNT:
            candidates = sorted(persons, key=lambda p: p.get("conf", 0.0), reverse=True)[:EXACT_PLAYER_COUNT]

        # Skip frames where any two candidates overlap significantly — indicates
        # a duplicate detection (same person detected twice).  Such frames
        # produce ghost players that steal a player slot from the real roster.
        has_overlap = any(
            _nms_iou(candidates[i], candidates[j]) > SEED_NMS_IOU
            for i in range(len(candidates))
            for j in range(i + 1, len(candidates))
        )
        if has_overlap:
            continue

        # Skip frames where any candidate is partially off-screen (entering/exiting).
        # cx < 150 or cx > 1770 on a 1920-wide frame means the player is cropped
        # at the boundary — colour/position are unreliable for that detection.
        if any(p["cx"] < 150 or p["cx"] > 1770 for p in candidates):
            continue

        # Use Hungarian colour-template assignment when HSV data is present.
        has_color = any(p.get("color_hsv") for p in candidates)
        if has_color:
            n = len(candidates)
            cost = np.full((n, len(PLAYER_IDS)), 1e9, dtype=float)
            for di, p in enumerate(candidates):
                hsv = p.get("color_hsv")
                if hsv:
                    for pi, pid in enumerate(PLAYER_IDS):
                        cost[di, pi] = _color_distance(hsv, _SEED_COLOR_TEMPLATES[pid])
            row_ind, col_ind = linear_sum_assignment(cost)
            assignment: dict[int, str] = {}
            for ri, ci in zip(row_ind, col_ind):
                if cost[ri, ci] < 1e8:
                    assignment[ri] = PLAYER_IDS[ci]
            player_pos = {
                pid: np.array([candidates[di]["cx"], candidates[di]["cy"]], dtype=float)
                for di, pid in assignment.items()
            }
        else:
            # Legacy fallback: sort by cx left-to-right.
            sorted_p = sorted(candidates, key=lambda p: p["cx"])
            player_pos = {
                PLAYER_IDS[i]: np.array([p["cx"], p["cy"]], dtype=float)
                for i, p in enumerate(sorted_p)
            }

        seed_fi = frame["frame"]
        print(
            f"  Seed frame {seed_fi} ({frame['timestamp_sec']:.1f}s): "
            + ", ".join(
                f"{pid}<-cx{player_pos[pid][0]:.0f}"
                for pid in PLAYER_IDS if pid in player_pos
            )
        )
        break

    if not player_pos:
        # Partial seed: fewer than EXACT_PLAYER_COUNT players visible simultaneously.
        for frame in frames:
            if frame["persons"]:
                persons = frame["persons"]
                has_color = any(p.get("color_hsv") for p in persons)
                if has_color:
                    n = min(len(persons), len(PLAYER_IDS))
                    candidates = persons[:n]
                    cost = np.full((n, n), 1e9, dtype=float)
                    pids = PLAYER_IDS[:n]
                    for di, p in enumerate(candidates):
                        hsv = p.get("color_hsv")
                        if hsv:
                            for pi, pid in enumerate(pids):
                                cost[di, pi] = _color_distance(hsv, _SEED_COLOR_TEMPLATES[pid])
                    row_ind, col_ind = linear_sum_assignment(cost)
                    player_pos = {}
                    for ri, ci in zip(row_ind, col_ind):
                        if cost[ri, ci] < 1e8:
                            player_pos[pids[ci]] = np.array(
                                [candidates[ri]["cx"], candidates[ri]["cy"]], dtype=float
                            )
                else:
                    sorted_p = sorted(persons, key=lambda p: p["cx"])
                    player_pos = {
                        PLAYER_IDS[i]: np.array([p["cx"], p["cy"]], dtype=float)
                        for i, p in enumerate(sorted_p[: len(PLAYER_IDS)])
                    }
                seed_fi = frame["frame"]
                print(
                    f"  WARNING: no frame with {EXACT_PLAYER_COUNT} persons. "
                    f"Partial seed from frame {seed_fi} ({len(player_pos)} slots filled)."
                )
                break

    if not player_pos:
        return {}, {}, {}

    calib_anchor: dict[str, np.ndarray] = {pid: pos.copy() for pid, pos in player_pos.items()}

    # Build colour refs from the next COLOR_SEED_FRAMES frames using the same
    # cost-matrix approach as the main tracker but with a loose distance gate.
    # Loose gate is intentional: we want enough samples even if players move fast
    # early on; quality is enforced by averaging over many observations.
    SEED_MOVE_GATE = 20.0 * 5   # px -- 5x per-frame budget covers dives/fast motion
    POS_SCALE = 750.0
    pid_colors: dict[str, list[list[float]]] = defaultdict(list)
    running_pos = {pid: pos.copy() for pid, pos in player_pos.items()}
    seed_frames_seen = 0

    for frame in frames:
        if frame["frame"] < seed_fi:
            continue
        if seed_frames_seen >= COLOR_SEED_FRAMES:
            break
        persons = frame["persons"]
        if not persons:
            continue

        n_det = len(persons)
        cost = np.full((n_det, len(PLAYER_IDS)), 1e9, dtype=float)
        for di, p in enumerate(persons):
            det = np.array([p["cx"], p["cy"]], dtype=float)
            for pi, pid in enumerate(PLAYER_IDS):
                ref = running_pos.get(pid)
                if ref is None:
                    continue
                dist = float(np.linalg.norm(det - ref))
                if dist <= SEED_MOVE_GATE:
                    cost[di, pi] = min(dist / POS_SCALE, 1.0)

        row_ind, col_ind = linear_sum_assignment(cost)
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] < 1e8:
                pid = PLAYER_IDS[ci]
                p = persons[ri]
                det = np.array([p["cx"], p["cy"]], dtype=float)
                running_pos[pid] = 0.5 * det + 0.5 * running_pos[pid]
                if p.get("color_hsv"):
                    pid_colors[pid].append(p["color_hsv"])

        seed_frames_seen += 1

    player_color_ref: dict[str, list[float]] = {
        pid: list(np.mean(pid_colors[pid], axis=0))
        for pid in PLAYER_IDS
        if pid_colors[pid]
    }

    return player_pos, calib_anchor, player_color_ref


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def identify_players(
    video_path: Path,
    detections_path: Path,
    output_path: Path,
    render_path: Optional[Path],
    sample_window_frac: float,
    api_key: str,
    use_llm: bool = True,
    use_embeddings: bool = False,
    seed_gt_path: Optional[Path] = None,
) -> None:
    data = json.loads(detections_path.read_text())
    frames: list[dict] = data["frames"]
    total_video_frames = frames[-1]["frame"] + 1 if frames else 0

    print(f"Loaded {len(frames)} frames from {detections_path}")

    if use_llm:
        # ------------------------------------------------------------------
        # Gemini calibration path
        # ------------------------------------------------------------------
        calib_frames = _select_calib_frames(
            frames,
            total_video_frames,
            sample_window_frac,
            MAX_CALIB_FRAMES,
        )
        print(f"Calibration: {len(calib_frames)} frames selected "
              f"(window: first {sample_window_frac*100:.0f}%)")

        if not calib_frames:
            print("ERROR: No suitable calibration frames found. "
                  "The video may not have 4 simultaneously-visible, uniquely-tracked persons.")
            sys.exit(1)

        client = genai.Client(api_key=api_key)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        calib_results: list[list[dict]] = []
        calib_persons_list: list[list[dict]] = []

        for i, cf in enumerate(calib_frames):
            fi = cf["frame"]
            persons = cf["persons"]
            print(f"  Calibration frame {i+1}/{len(calib_frames)}: "
                  f"frame {fi} ({cf['timestamp_sec']:.1f}s), "
                  f"{len(persons)} persons "
                  f"[{', '.join(p['human_track_id'] for p in persons)}]")

            try:
                full_jpeg = _extract_jpeg(cap, fi)
                crops = [
                    _crop_jpeg(cap, fi, p["x1"], p["y1"], p["x2"], p["y2"])
                    for p in persons
                ]
            except RuntimeError as exc:
                print(f"    Frame extraction failed: {exc} — skipping")
                continue

            assignments = _call_gemini(client, full_jpeg, crops, persons)
            if assignments is None:
                continue

            print(f"    Assignments: "
                  + ", ".join(
                      f"{persons[a['detection_index']]['human_track_id']}\u2192{a['player_id']}"
                      for a in sorted(assignments, key=lambda x: x["detection_index"])
                  ))
            calib_results.append(assignments)
            calib_persons_list.append(persons)

        cap.release()

        if not calib_results:
            print("ERROR: All Gemini calibration calls failed.")
            sys.exit(1)

        # Consensus
        print(f"\nBuilding consensus from {len(calib_results)} successful calibration frames \u2026")
        track_map = _build_consensus(calib_results, calib_persons_list, CONSENSUS_THRESHOLD)
        track_map = _resolve_conflicts(track_map)

        print("Track map:")
        for hid, pid in sorted(track_map.items()):
            print(f"  {hid} \u2192 {pid} ({PLAYERS[pid]['name']})")
        unmapped_pids = set(PLAYER_IDS) - set(track_map.values())
        if unmapped_pids:
            print(f"  WARNING: players without a confident track: {unmapped_pids}")

        # Anchor centroids (average per-player position across all calib frames).
        pid_cx: dict[str, list[float]] = defaultdict(list)
        pid_cy: dict[str, list[float]] = defaultdict(list)
        for cf_persons, cf_assignments in zip(calib_persons_list, calib_results):
            for a in cf_assignments:
                di = a["detection_index"]
                if di < len(cf_persons):
                    p = cf_persons[di]
                    hid = p.get("human_track_id")
                    if hid and hid in track_map:
                        pid = track_map[hid]
                        pid_cx[pid].append(p["cx"])
                        pid_cy[pid].append(p["cy"])

        print("Anchor centroids (cx, cy):")
        for pid in PLAYER_IDS:
            if pid_cx[pid]:
                ax = np.mean(pid_cx[pid])
                ay = np.mean(pid_cy[pid])
                print(f"  {pid} ({PLAYERS[pid]['name']}): ({ax:.0f}, {ay:.0f})")

        # Initial player positions: seed from the best-spread calib frame.
        first_persons = calib_persons_list[0]
        first_assignments = calib_results[0]
        player_pos: dict[str, np.ndarray] = {}
        for a in first_assignments:
            di = a["detection_index"]
            if di < len(first_persons):
                p = first_persons[di]
                player_pos[a["player_id"]] = np.array([p["cx"], p["cy"]], dtype=float)
        # Fallback via average for any player absent from the seed frame.
        pid_cx2: dict[str, list[float]] = defaultdict(list)
        pid_cy2: dict[str, list[float]] = defaultdict(list)
        for cf_persons, cf_assignments in zip(calib_persons_list, calib_results):
            for a in cf_assignments:
                di = a["detection_index"]
                if di < len(cf_persons):
                    p = cf_persons[di]
                    pid_cx2[a["player_id"]].append(p["cx"])
                    pid_cy2[a["player_id"]].append(p["cy"])
        for pid in PLAYER_IDS:
            if pid not in player_pos and pid_cx2[pid]:
                player_pos[pid] = np.array([np.mean(pid_cx2[pid]), np.mean(pid_cy2[pid])], dtype=float)

        calib_anchor: dict[str, np.ndarray] = {k: v.copy() for k, v in player_pos.items()}

        # Per-player colour references from calibration detections.
        pid_colors: dict[str, list[list[float]]] = defaultdict(list)
        for cf_persons, cf_assignments in zip(calib_persons_list, calib_results):
            for a in cf_assignments:
                di = a["detection_index"]
                if di < len(cf_persons):
                    p = cf_persons[di]
                    if p.get("color_hsv"):
                        pid_colors[a["player_id"]].append(p["color_hsv"])
        player_color_ref: dict[str, list[float]] = {
            pid: list(np.mean(pid_colors[pid], axis=0))
            for pid in PLAYER_IDS
            if pid_colors[pid]
        }
        if player_color_ref:
            print("Player colour references (mean H, S, V):")
            for pid, c in player_color_ref.items():
                print(f"  {pid} ({PLAYERS[pid]['name']}): H={c[0]:.0f} S={c[1]:.0f} V={c[2]:.0f}")
        else:
            print("  No color_hsv in detections — colour blending disabled")

    else:
        # ------------------------------------------------------------------
        # No-LLM path: seed from first full-court frame, build colour refs
        # from the following COLOR_SEED_FRAMES frames.
        # ------------------------------------------------------------------
        if seed_gt_path is not None:
            # --seed-gt: load the first confirmed GT annotation and use it to
            # seed player positions directly.  This isolates the tracking quality
            # from seeding quality — tells us the ceiling of the tracker.
            print(f"No-LLM mode (seed-gt): seeding from GT annotation {seed_gt_path} …")
            gt_data = json.loads(seed_gt_path.read_text())
            confirmed = [
                a for a in gt_data.get("annotations", [])
                if a.get("confirmed", True)
            ]
            if not confirmed:
                print("ERROR: no confirmed annotations in GT file.")
                sys.exit(1)
            confirmed.sort(key=lambda a: int(a.get("frame", -1)))

            # Use the first annotation that has all 4 players confirmed.
            # Annotations with fewer players (e.g. one is off-screen) leave slots
            # unseeded, which the heuristic then fills incorrectly.
            four_player_anns = [
                a for a in confirmed
                if {assign.get("player_id") for assign in a.get("assignments", [])} >= set(PLAYER_IDS)
            ]
            if not four_player_anns:
                print("WARNING: no annotation with all 4 players found; using first confirmed annotation.")
                four_player_anns = confirmed
            first_ann = four_player_anns[0]
            seed_frame_no = first_ann["frame"]
            n_assigned = len(first_ann.get("assignments", []))
            print(f"  GT seed frame: {seed_frame_no} ({first_ann.get('timestamp_sec', 0):.1f}s)  ({n_assigned} players)")

            # Resolve detections for the seed frame.
            det_by_frame: dict[int, dict] = {f["frame"]: f for f in frames}
            seed_det_frame = det_by_frame.get(seed_frame_no)
            persons_in_seed: list[dict] = seed_det_frame.get("persons", []) if seed_det_frame else []

            player_pos = {}
            for assign in first_ann.get("assignments", []):
                pid = assign.get("player_id")
                di = assign.get("detection_index")
                if pid in PLAYER_IDS and isinstance(di, int) and di < len(persons_in_seed):
                    p = persons_in_seed[di]
                    player_pos[pid] = np.array([p["cx"], p["cy"]], dtype=float)

            # Fall back to manual_positions for any player absent from assignments
            # (e.g. P4 is off-screen in the seed frame and was annotated by hand).
            for pid, mp in first_ann.get("manual_positions", {}).items():
                if pid in PLAYER_IDS and pid not in player_pos:
                    player_pos[pid] = np.array([float(mp["x"]), float(mp["y"])], dtype=float)

            if not player_pos:
                print("ERROR: Could not resolve any player positions from GT seed annotation.")
                sys.exit(1)

            calib_anchor = {pid: pos.copy() for pid, pos in player_pos.items()}

            # Bootstrap colour refs: use the same rolling seed over COLOR_SEED_FRAMES.
            _, _, player_color_ref = _seed_from_detections(frames)

            print("  GT-seeded positions:")
            for pid, pos in player_pos.items():
                print(f"    {pid}: ({pos[0]:.0f}, {pos[1]:.0f})")
        else:
            print("No-LLM mode: seeding from first full-court frame …")
            player_pos, calib_anchor, player_color_ref = _seed_from_detections(frames)
        if not player_pos:
            print("ERROR: No detections found in the video.")
            sys.exit(1)
        if player_color_ref:
            print("Player colour references (mean H, S, V):")
            for pid, c in player_color_ref.items():
                print(f"  {pid}: H={c[0]:.0f} S={c[1]:.0f} V={c[2]:.0f}")
        else:
            print("  No color_hsv in detections — colour tracking disabled")
        track_map: dict[str, str] = {}  # not meaningful without LLM
        calib_frames: list[dict] = []
        calib_results: list[list[dict]] = []

    print("Initial player positions:")
    for pid, pos in player_pos.items():
        name_str = f" ({PLAYERS[pid]['name']})" if use_llm else ""
        print(f"  {pid}{name_str}: ({pos[0]:.0f}, {pos[1]:.0f})")

    # Strategy B: DINOv2 embedding gallery.
    # Initialized here; enrollment happens in two places:
    #   1. Calibration frames (LLM path) — trusted crops for known players.
    #   2. Per-frame assignment loop — after each confident Phase B resolution.
    gallery = None
    embed_cap = None
    if use_embeddings:
        from beach.embeddings import EmbeddingGallery
        gallery = EmbeddingGallery(PLAYER_IDS)
        embed_cap = cv2.VideoCapture(str(video_path))
        if not embed_cap.isOpened():
            print("  WARNING: Could not open video for embedding enrollment — disabling embeddings.")
            gallery = None
            embed_cap = None
        else:
            # Enroll from calibration frames (LLM path) or the seed frame (no-LLM path).
            enroll_frames = calib_frames if use_llm else []
            if not enroll_frames and player_pos:
                # No-LLM: find the seed frame in the detections and enroll it.
                for f in frames:
                    if len(f["persons"]) >= EXACT_PLAYER_COUNT:
                        enroll_frames = [f]
                        break
            enrolled = 0
            for ef in enroll_frames:
                fi = ef["frame"]
                persons_ef = ef["persons"]
                # Build a detection_index -> player_id map for this enroll frame.
                if use_llm:
                    # Use the calibration result for this frame if available.
                    # calib_results aligns with calib_frames by index.
                    try:
                        ci = calib_frames.index(ef)
                        if ci < len(calib_results):
                            det_to_pid = {
                                a["detection_index"]: a["player_id"]
                                for a in calib_results[ci]
                            }
                        else:
                            det_to_pid = {}
                    except ValueError:
                        det_to_pid = {}
                else:
                    # No-LLM seed frame: use left-to-right P1..P4 order.
                    sorted_p = sorted(persons_ef, key=lambda p: p["cx"])[:EXACT_PLAYER_COUNT]
                    det_to_pid = {persons_ef.index(sp): PLAYER_IDS[i] for i, sp in enumerate(sorted_p)}
                embed_cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, enroll_frame = embed_cap.read()
                if not ret:
                    continue
                H_ef, W_ef = enroll_frame.shape[:2]
                for di, p in enumerate(persons_ef):
                    pid = det_to_pid.get(di)
                    if pid is None:
                        continue
                    x1c = max(0, int(p["x1"]))
                    y1c = max(0, int(p["y1"]))
                    x2c = min(W_ef, int(p["x2"]))
                    y2c = min(H_ef, int(p["y2"]))
                    if x2c > x1c and y2c > y1c:
                        crop = enroll_frame[y1c:y2c, x1c:x2c]
                        gallery.enroll(pid, crop)
                        enrolled += 1
            print(f"  Embedding gallery: enrolled {enrolled} crops from {len(enroll_frames)} calibration frame(s).")
            for pid in PLAYER_IDS:
                status = "enrolled" if gallery.has_enrollment(pid) else "MISSING"
                print(f"    {pid} ({PLAYERS[pid]['name']}): {status}")
    # Rolling tracker parameters.
    # At 50 fps, a volleyball player at full sprint (~5 m/s) over a court that spans
    # ~1200 px moves at most ~6 px/frame.  We allow MOVE_PX_PER_FRAME * (1 + absent)
    # so the cap is tight for frame-to-frame movement but grows linearly with the
    # number of frames the player has been absent, enabling re-acquisition after
    # occlusions without ever allowing a jump across the full court in one step.
    MOVE_PX_PER_FRAME = 35.0   # generous per-frame budget; wider gate reduces missed
                               # re-acquisitions after ByteTrack ID churn or occlusions
    EMA_ALPHA         = 0.5    # position smoothing: 0=frozen anchor, 1=raw detection
    MAX_MISSING_FR    = 60     # 1.2 s at 50 fps — hold last position before marking lost

    # frames_missing[pid] counts consecutive frames a player was not assigned.
    frames_missing: dict[str, int] = {pid: 0 for pid in PLAYER_IDS}

    # Strategy A: running H-ID -> player map.  Seeded from calibration track_map.
    # Extended incrementally as new H-IDs are confidently assigned by the cost matrix.
    # This is the authoritative map — when a known H-ID reappears, its player assignment
    # is inherited directly without running the cost matrix, preventing drift on crossings.
    running_hid_map: dict[str, str] = dict(track_map)   # copy so we can extend it

    # --- Propagate across all frames using rolling position tracker ---
    enriched_frames: list[dict] = []
    unresolved_count = 0

    for frame_data in frames:
        persons = frame_data["persons"]
        n_det   = len(persons)

        per_frame_assigned: dict[int, str] = {}  # detection_index -> player_id

        if n_det > 0:
            # Phase A: H-ID continuity — inherit assignment for known tracks.
            # This is a hard constraint: if we have already confidently mapped this
            # H-ID to a player, we trust it.  This eliminates most drift from the old
            # approach where every frame re-ran the full cost matrix.
            unknown_det_indices: list[int] = []   # detections needing cost-matrix resolution
            for di, p in enumerate(persons):
                hid = p.get("human_track_id")
                if hid and hid in running_hid_map:
                    pid = running_hid_map[hid]
                    # Guard: reject if another detection in this frame already claimed pid.
                    if pid not in per_frame_assigned.values():
                        per_frame_assigned[di] = pid
                    else:
                        # Collision — two detections map to the same player via the HID map.
                        # This can happen when ByteTrack briefly assigns the same ID to two
                        # tracks (rare).  Treat as unknown and let the cost matrix resolve.
                        unknown_det_indices.append(di)
                else:
                    unknown_det_indices.append(di)

            # Per-frame NMS: suppress unknown detections that heavily overlap an
            # already-assigned detection.  ByteTrack occasionally fires two IDs on
            # the same person (observed IoU>0.6); the second detection must not
            # consume a player slot — it would force an incorrect player into the
            # only remaining free slot and lock that mistake into running_hid_map.
            FRAME_NMS_IOU = 0.3
            if unknown_det_indices and per_frame_assigned:
                suppressed: set[int] = set()
                for di in unknown_det_indices:
                    for assigned_di in per_frame_assigned:
                        if _nms_iou(persons[di], persons[assigned_di]) > FRAME_NMS_IOU:
                            suppressed.add(di)
                            break
                if suppressed:
                    unknown_det_indices = [di for di in unknown_det_indices if di not in suppressed]

            # Phase B: cost-matrix assignment for detections with unknown H-IDs.
            if unknown_det_indices:
                # Players already claimed by Phase A cannot be assigned again.
                claimed_pids = set(per_frame_assigned.values())
                free_player_indices = [
                    pi for pi, pid in enumerate(PLAYER_IDS)
                    if pid not in claimed_pids
                ]

                # COLOR_WEIGHT blends appearance into the cost; 0 = position only.
                # In no-LLM mode bump colour weight: without a Gemini-confirmed name
                # mapping, colour is the primary disambiguation signal when two players
                # come close together after an occlusion.
                COLOR_WEIGHT = 0.25 if use_llm else 0.40
                # EMBED_WEIGHT: when the gallery is available, embedding similarity
                # is a richer signal than 3-float HSV.  Blend it on top.
                # Combined cost = (1 - EMBED_WEIGHT) * pos_color_blend + EMBED_WEIGHT * embed_cost
                EMBED_WEIGHT = 0.40 if (gallery is not None) else 0.0
                # Scale position distance to 0-1 (half frame width ≈ 750 px).
                POS_SCALE = 750.0

                # Pre-compute embedding similarities for unknown detections if gallery active.
                # embed_sims[uki][pid] = cosine similarity (0-1); higher = more similar.
                embed_sims: list[dict[str, float]] = []
                if gallery is not None and embed_cap is not None:
                    # Read this frame once for all unknown detections.
                    embed_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_data["frame"])
                    ret_e, embed_frame = embed_cap.read()
                    H_e, W_e = (embed_frame.shape[:2] if ret_e else (0, 0))
                    for di in unknown_det_indices:
                        p_e = persons[di]
                        if ret_e:
                            x1e = max(0, int(p_e["x1"]))
                            y1e = max(0, int(p_e["y1"]))
                            x2e = min(W_e, int(p_e["x2"]))
                            y2e = min(H_e, int(p_e["y2"]))
                            if x2e > x1e and y2e > y1e:
                                crop_e = embed_frame[y1e:y2e, x1e:x2e]
                                sims = {pid: gallery.similarity(crop_e, pid) for pid in PLAYER_IDS}
                            else:
                                sims = {pid: 0.0 for pid in PLAYER_IDS}
                        else:
                            sims = {pid: 0.0 for pid in PLAYER_IDS}
                        embed_sims.append(sims)
                else:
                    # No gallery: fill with zeros so the cost formula degrades cleanly.
                    embed_sims = [{pid: 0.0 for pid in PLAYER_IDS} for _ in unknown_det_indices]

                n_unk = len(unknown_det_indices)
                n_free = len(free_player_indices)
                if n_unk > 0 and n_free > 0:
                    cost = np.full((n_unk, n_free), 1e9, dtype=float)
                    for uki, di in enumerate(unknown_det_indices):
                        p = persons[di]
                        det = np.array([p["cx"], p["cy"]], dtype=float)
                        p_color = p.get("color_hsv")
                        uki_sims = embed_sims[uki]  # pid -> similarity (0-1)
                        for fki, pi in enumerate(free_player_indices):
                            pid = PLAYER_IDS[pi]
                            ref = player_pos.get(pid)
                            if ref is None:
                                continue
                            dist = float(np.linalg.norm(det - ref))
                            absent = frames_missing.get(pid, 0)
                            max_dist = MOVE_PX_PER_FRAME * max(1, absent + 1)
                            if dist > max_dist:
                                continue  # forbidden — leave as 1e9
                            pos_cost = min(dist / POS_SCALE, 1.0)
                            # Base appearance blend: position + colour.
                            if p_color and pid in player_color_ref:
                                c_cost = _color_distance(p_color, player_color_ref[pid])
                                app_cost = (1.0 - COLOR_WEIGHT) * pos_cost + COLOR_WEIGHT * c_cost
                            else:
                                app_cost = pos_cost
                            # Blend in DINOv2 embedding similarity (1 - sim = cost).
                            if EMBED_WEIGHT > 0.0 and gallery is not None and gallery.has_enrollment(pid):
                                embed_cost = 1.0 - uki_sims.get(pid, 0.0)
                                cost[uki, fki] = (1.0 - EMBED_WEIGHT) * app_cost + EMBED_WEIGHT * embed_cost
                            else:
                                cost[uki, fki] = app_cost

                    row_ind, col_ind = linear_sum_assignment(cost)
                    for ri, ci in zip(row_ind, col_ind):
                        if cost[ri, ci] < 1e8:
                            di = unknown_det_indices[ri]
                            pid = PLAYER_IDS[free_player_indices[ci]]
                            per_frame_assigned[di] = pid
                            # Extend the running H-ID map with this newly resolved track.
                            hid = persons[di].get("human_track_id")
                            if hid:
                                running_hid_map[hid] = pid
                            # Strategy B: enroll this crop into the gallery so that future
                            # appearances of this player are matched by visual similarity.
                            if gallery is not None and embed_cap is not None:
                                # embed_frame was captured above for this frame; reuse it.
                                p_enr = persons[di]
                                if ret_e:
                                    x1r = max(0, int(p_enr["x1"]))
                                    y1r = max(0, int(p_enr["y1"]))
                                    x2r = min(W_e, int(p_enr["x2"]))
                                    y2r = min(H_e, int(p_enr["y2"]))
                                    if x2r > x1r and y2r > y1r:
                                        gallery.enroll(pid, embed_frame[y1r:y2r, x1r:x2r])

        # Update rolling positions for assigned players (EMA); age unassigned ones.
        assigned_pids = set(per_frame_assigned.values())
        for di, pid in per_frame_assigned.items():
            det = np.array([persons[di]["cx"], persons[di]["cy"]], dtype=float)
            if pid in player_pos:
                player_pos[pid] = EMA_ALPHA * det + (1 - EMA_ALPHA) * player_pos[pid]
            else:
                player_pos[pid] = det
            frames_missing[pid] = 0

        for pid in PLAYER_IDS:
            if pid not in assigned_pids:
                frames_missing[pid] += 1
                # Strategy A: do NOT snap to calibration anchor when a player is
                # missing too long.  Snapping to a stale anchor causes ghost assignments
                # when a player re-enters far from that anchor.  Instead, keep the
                # last-known position and let the growing max_dist gate handle
                # re-acquisition naturally.

        # Count genuinely unresolvable detections (all 4 slots occupied = extra persons)
        for di in range(n_det):
            if di not in per_frame_assigned:
                unresolved_count += 1

        # Build enriched persons list
        enriched_persons = [
            {**p, "player_id": per_frame_assigned.get(di)}
            for di, p in enumerate(persons)
        ]

        enriched_frames.append({
            **frame_data,
            "persons": enriched_persons,
        })
    # (cap was opened and released inside the use_llm block above)
    if embed_cap is not None:
        embed_cap.release()

    if unresolved_count:
        print(f"\n  {unresolved_count} detections could not be assigned a player_id "
              "(no track in map and centroid fallback failed).")

    # --- Write output JSON ---
    output: dict = {
        "players": {pid: {"name": PLAYERS[pid]["name"]} for pid in PLAYER_IDS},
        "frames": enriched_frames,
    }
    if use_llm:
        output["track_map"] = track_map
        output["track_map_extended"] = running_hid_map  # full map after propagation
        output["calibration"] = {
            "frames_sampled": len(calib_frames),
            "frames_used": len(calib_results),
            "consensus_threshold": CONSENSUS_THRESHOLD,
        }
    else:
        output["mode"] = "heuristic"
        output["track_map_extended"] = running_hid_map
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nIdentified JSON written: {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")

    # --- Optional render ---
    if render_path:
        _render_identified(video_path, enriched_frames, render_path)


# ---------------------------------------------------------------------------
# Optional render
# ---------------------------------------------------------------------------
def _render_identified(
    video_path: Path,
    enriched_frames: list[dict],
    render_path: Path,
) -> None:
    """Render video with per-person bounding boxes, H-ID, P-ID, and colour swatch.

    Each person gets:
      - A bounding box in the assigned player's colour.
      - A filled label tag: "{h_id} -> {pid} {name}".
      - A small colour swatch showing the detected torso HSV from pass 1.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for render: {video_path}")
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_dim = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    render_path.parent.mkdir(parents=True, exist_ok=True)
    # mp4v avoids the h264_v4l2m2m probe failure on x86 Linux.
    writer = cv2.VideoWriter(
        str(render_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h_dim)
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot open VideoWriter for render path")

    print(f"Rendering identified video to {render_path} …")
    frame_lookup = {f["frame"]: f for f in enriched_frames}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    SWATCH_W  = 14   # width/height of the detected-colour square
    LABEL_GAP = 4    # gap between text and swatch inside the tag
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fd = frame_lookup.get(idx)
        if fd:
            for p in fd["persons"]:
                pid  = p.get("player_id")
                h_id = p.get("human_track_id") or "?"
                color = _PLAYER_COLORS.get(pid, _UNKNOWN_COLOR)
                name  = PLAYERS[pid]["name"] if pid else "?"

                x1, y1 = int(p["x1"]), int(p["y1"])
                x2, y2 = int(p["x2"]), int(p["y2"])

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), BOX_THICKNESS + 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

                # Build label: "{h_id} -> {pid} {name}" or "{h_id} -> ?"
                label = f"{h_id} -> {pid} {name}" if pid else f"{h_id} -> ?"
                (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

                # Tag background: label width + gap + swatch square
                tag_w = tw + LABEL_PAD * 2 + LABEL_GAP + SWATCH_W
                tag_y0 = max(y1 - th - LABEL_PAD * 2, 0)
                cv2.rectangle(frame, (x1, tag_y0), (x1 + tag_w, y1), color, cv2.FILLED)

                # Label text in black on the coloured background
                cv2.putText(
                    frame, label,
                    (x1 + LABEL_PAD, y1 - LABEL_PAD),
                    FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA,
                )

                # Detected-colour swatch (torso HSV from pass 1)
                if p.get("color_hsv"):
                    h_val = min(int(round(p["color_hsv"][0])), 179)
                    s_val = min(int(round(p["color_hsv"][1])), 255)
                    # Floor at 50 so dark swatches are visible but still look dark.
                    v_val = max(int(round(p["color_hsv"][2])), 50)
                    bgr = cv2.cvtColor(
                        np.array([[[h_val, s_val, v_val]]], dtype=np.uint8),
                        cv2.COLOR_HSV2BGR,
                    )[0][0].tolist()
                    sx1 = x1 + tw + LABEL_PAD * 2 + LABEL_GAP
                    sx2 = sx1 + SWATCH_W
                    sy1 = tag_y0 + 2
                    sy2 = y1 - 2
                    # thin black border then filled swatch
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 0), 1)
                    cv2.rectangle(
                        frame, (sx1 + 1, sy1 + 1), (sx2 - 1, sy2 - 1),
                        (int(bgr[0]), int(bgr[1]), int(bgr[2])), cv2.FILLED,
                    )

            ball = fd.get("ball")
            if ball:
                bx, by = int(ball["cx"]), int(ball["cy"])
                cv2.circle(frame, (bx, by), BALL_RADIUS + 2, (0, 0, 0), -1)
                cv2.circle(frame, (bx, by), BALL_RADIUS, BALL_COLOR_BGR, -1)

        writer.write(frame)
        idx += 1
        if idx % 150 == 0:
            print(f"  rendered frame {idx}/{total}")
    writer.release()
    cap.release()
    print(f"Render written: {render_path}  ({render_path.stat().st_size / 1024 / 1024:.1f} MB)")


@click.command("identify")
@click.option("--video", "-v", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Source video file (default: derived from --detections stem).")
@click.option("--detections", "-d", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Pass-1 detections JSON (default: <video_stem>_detections.json).")
@click.option("--output", "-o", default=None, type=click.Path(dir_okay=False, path_type=Path), help="Output path for identified JSON (default: <video_stem>_identified.json).")
@click.option("--render-identified", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Optional: render annotated video to this path.")
@click.option("--sample-window", type=float, default=0.30, show_default=True, help="(LLM only) Fraction of video to sample calibration frames from.")
@click.option("--no-llm", is_flag=True, default=False, help="Skip Gemini; identify via proximity + colour only (no API key required).")
@click.option("--embeddings", is_flag=True, default=False, help="Use DINOv2 visual embeddings (Strategy B) to improve re-identification accuracy.")
@click.option("--seed-gt", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="(--no-llm only) Seed player positions from the first confirmed annotation in this GT JSON instead of auto-detecting from colour templates.  Use to isolate tracker ceiling from seeding quality.")
def identify_cmd(video, detections, output, render_identified, sample_window, no_llm, embeddings, seed_gt):
    """Pass 2: Assign P1-P4 to detections via Gemini (default) or heuristic (--no-llm)."""
    import os

    # Resolve mutual defaults: each of video/detections can anchor the other.
    if video is None and detections is None:
        raise click.UsageError(
            "Provide at least --video or --detections so the other path can be inferred."
        )

    if video is None:
        # Strip '_detections' suffix if present, look for matching video.
        stem = detections.stem
        if stem.endswith("_detections"):
            stem = stem[: -len("_detections")]
        # Try common video extensions in order.
        for ext in (".mp4", ".MP4", ".mov", ".MOV", ".avi"):
            candidate = detections.with_name(stem + ext)
            if candidate.exists():
                video = candidate
                break
        if video is None:
            raise click.ClickException(
                f"Could not find a video file for stem '{stem}' next to {detections}.\n"
                "Supply --video explicitly."
            )

    detections_path = detections or video.with_name(video.stem + "_detections.json")
    output_path = output or identified_path(video, no_llm=no_llm, embeddings=embeddings)

    if not detections_path.exists():
        raise click.ClickException(
            f"Detections file not found: {detections_path}\n"
            "Run 'beach track' first, or supply --detections explicitly."
        )

    use_llm = not no_llm
    api_key = ""
    if use_llm:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise click.ClickException(
                "GOOGLE_API_KEY environment variable not set. "
                "Pass --no-llm to run without Gemini."
            )
    identify_players(
        video_path=video,
        detections_path=detections_path,
        output_path=output_path,
        render_path=render_identified,
        sample_window_frac=sample_window,
        api_key=api_key,
        use_llm=use_llm,
        use_embeddings=embeddings,
        seed_gt_path=seed_gt,
    )