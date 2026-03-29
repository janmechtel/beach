# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "google-genai>=1.69.0",
#   "opencv-python-headless>=4.9",
#   "scipy>=1.11",
# ]
# ///
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

Usage
-----
    uv run identify_players.py \
        --video  chunks/GH021569_court.mp4 \
        --detections chunks/GH021569_court_detections.json \
        --output chunks/GH021569_court_identified.json

    # Optional rendered video:
    uv run identify_players.py ... --render-identified chunks/GH021569_court_identified.mp4
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from google import genai
from google.genai import types
from scipy.optimize import linear_sum_assignment

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
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.65
FONT_THICKNESS = 2
LABEL_PAD      = 5
BALL_RADIUS    = 14
BALL_COLOR_BGR = (0, 220, 255)

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
# Core pipeline
# ---------------------------------------------------------------------------
def identify_players(
    video_path: Path,
    detections_path: Path,
    output_path: Path,
    render_path: Optional[Path],
    sample_window_frac: float,
    api_key: str,
) -> None:
    data = json.loads(detections_path.read_text())
    frames: list[dict] = data["frames"]
    total_video_frames = frames[-1]["frame"] + 1 if frames else 0

    print(f"Loaded {len(frames)} frames from {detections_path}")

    # --- Select calibration frames ---
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

    # --- Gemini calibration ---
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
                  f"{persons[a['detection_index']]['human_track_id']}→{a['player_id']}"
                  for a in sorted(assignments, key=lambda x: x["detection_index"])
              ))
        calib_results.append(assignments)
        calib_persons_list.append(persons)

    if not calib_results:
        print("ERROR: All Gemini calibration calls failed.")
        sys.exit(1)

    # --- Consensus ---
    print(f"\nBuilding consensus from {len(calib_results)} successful calibration frames …")
    track_map = _build_consensus(calib_results, calib_persons_list, CONSENSUS_THRESHOLD)
    track_map = _resolve_conflicts(track_map)

    print("Track map:")
    for hid, pid in sorted(track_map.items()):
        print(f"  {hid} → {pid} ({PLAYERS[pid]['name']})")
    unmapped_pids = set(PLAYER_IDS) - set(track_map.values())
    if unmapped_pids:
        print(f"  WARNING: players without a confident track: {unmapped_pids}")

    # Build anchor centroids from the calibration frames for the centroid fallback.
    # Average centroid per player across all calibration frames where that track appeared.
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

    anchor_centroids: dict[str, tuple[float, float]] = {
        pid: (np.mean(pid_cx[pid]), np.mean(pid_cy[pid]))
        for pid in PLAYER_IDS
        if pid_cx[pid]
    }
    print("Anchor centroids (cx, cy):")
    for pid, (ax, ay) in anchor_centroids.items():
        print(f"  {pid} ({PLAYERS[pid]['name']}): ({ax:.0f}, {ay:.0f})")

    # Build initial player positions from the earliest calibration frame where all 4
    # Gemini-confirmed players appear.  These seed the rolling tracker.
    pid_cx: dict[str, list[float]] = defaultdict(list)
    pid_cy: dict[str, list[float]] = defaultdict(list)
    for cf_persons, cf_assignments in zip(calib_persons_list, calib_results):
        for a in cf_assignments:
            di = a["detection_index"]
            if di < len(cf_persons):
                p = cf_persons[di]
                pid_cx[a["player_id"]].append(p["cx"])
                pid_cy[a["player_id"]].append(p["cy"])

    # player_pos[pid] = current rolling position estimate [cx, cy].
    # calib_frames[0] is guaranteed by _select_calib_frames to be the best-spread frame
    # (max min-pairwise-distance), so seeding from it gives the rolling tracker the
    # clearest initial separation between players.
    first_persons    = calib_persons_list[0]
    first_assignments = calib_results[0]
    player_pos: dict[str, np.ndarray] = {}
    for a in first_assignments:
        di = a["detection_index"]
        if di < len(first_persons):
            p = first_persons[di]
            player_pos[a["player_id"]] = np.array([p["cx"], p["cy"]], dtype=float)
    # Fallback for any player absent from the seed frame (shouldn't happen).
    pid_cx: dict[str, list[float]] = defaultdict(list)
    pid_cy: dict[str, list[float]] = defaultdict(list)
    for cf_persons, cf_assignments in zip(calib_persons_list, calib_results):
        for a in cf_assignments:
            di = a["detection_index"]
            if di < len(cf_persons):
                p = cf_persons[di]
                pid_cx[a["player_id"]].append(p["cx"])
                pid_cy[a["player_id"]].append(p["cy"])
    for pid in PLAYER_IDS:
        if pid not in player_pos and pid_cx[pid]:
            player_pos[pid] = np.array([np.mean(pid_cx[pid]), np.mean(pid_cy[pid])], dtype=float)

    # Fixed anchor used when a player's rolling position goes stale.
    # Reset to these coordinates after MAX_MISSING_FR frames without a detection.
    calib_anchor: dict[str, np.ndarray] = {k: v.copy() for k, v in player_pos.items()}

    # Build per-player color references from calibration detections.
    # Average color_hsv across all calibration appearances of each player.
    # Falls back to None if detections have no color_hsv (pre-color pass-1 data).
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
        print("Player color references (mean H, S, V):")
        for pid, c in player_color_ref.items():
            print(f"  {pid} ({PLAYERS[pid]['name']}): H={c[0]:.0f} S={c[1]:.0f} V={c[2]:.0f}")
    else:
        print("  No color_hsv in detections — color blending disabled")


    print("Initial player positions (from calibration):")
    for pid, pos in player_pos.items():
        print(f"  {pid} ({PLAYERS[pid]['name']}): ({pos[0]:.0f}, {pos[1]:.0f})")

    # Rolling tracker parameters.
    # At 50 fps, a volleyball player at full sprint (~5 m/s) over a court that spans
    # ~1200 px moves at most ~6 px/frame.  We allow MOVE_PX_PER_FRAME * (1 + absent)
    # so the cap is tight for frame-to-frame movement but grows linearly with the
    # number of frames the player has been absent, enabling re-acquisition after
    # occlusions without ever allowing a jump across the full court in one step.
    MOVE_PX_PER_FRAME = 20.0   # generous per-frame budget (covers camera shake/dives)
    EMA_ALPHA         = 0.5    # position smoothing: 0=frozen anchor, 1=raw detection
    MAX_MISSING_FR    = 60     # 1.2 s at 50 fps — hold last position before resetting

    # frames_missing[pid] counts consecutive frames a player was not assigned
    frames_missing: dict[str, int] = {pid: 0 for pid in PLAYER_IDS}

    # --- Propagate across all frames using rolling position tracker ---
    enriched_frames: list[dict] = []
    unresolved_count = 0

    for frame_data in frames:
        persons = frame_data["persons"]
        n_det   = len(persons)

        per_frame_assigned: dict[int, str] = {}  # detection_index -> player_id

        if n_det > 0:
            # Cost matrix: rows = detections, cols = players.
            # The allowed search radius grows linearly with how many frames a player has
            # been absent.  When last seen 0 frames ago: radius = MOVE_PX_PER_FRAME.
            # When absent for 30 frames: radius = 620 px (effectively the whole court),
            # so the player can be re-acquired wherever they re-enter the frame.
            # COLOR_WEIGHT blends appearance into the cost; 0 = position only.
            # Position is still the HARD gate (velocity cap) — color only breaks
            # ties among candidates that pass the distance threshold.
            COLOR_WEIGHT = 0.25
            # Scale position distance to 0-1 (half frame width ≈ 750 px).
            POS_SCALE = 750.0

            cost = np.full((n_det, len(PLAYER_IDS)), 1e9, dtype=float)
            for di, p in enumerate(persons):
                det = np.array([p["cx"], p["cy"]], dtype=float)
                p_color = p.get("color_hsv")  # may be None for old detections
                for pi, pid in enumerate(PLAYER_IDS):
                    ref = player_pos.get(pid, calib_anchor.get(pid))
                    if ref is None:
                        continue
                    dist = float(np.linalg.norm(det - ref))
                    absent = frames_missing.get(pid, 0)
                    max_dist = MOVE_PX_PER_FRAME * max(1, absent + 1)
                    if dist > max_dist:
                        continue  # forbidden — leave as 1e9
                    pos_cost = min(dist / POS_SCALE, 1.0)
                    if p_color and pid in player_color_ref:
                        c_cost = _color_distance(p_color, player_color_ref[pid])
                        cost[di, pi] = (1.0 - COLOR_WEIGHT) * pos_cost + COLOR_WEIGHT * c_cost
                    else:
                        cost[di, pi] = pos_cost


            row_ind, col_ind = linear_sum_assignment(cost)
            for ri, ci in zip(row_ind, col_ind):
                if cost[ri, ci] < 1e8:  # accepted (not forbidden)
                    per_frame_assigned[int(ri)] = PLAYER_IDS[int(ci)]

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
                # If stale for too long, reset to calibration anchor so the player
                # can be re-acquired when they re-enter the frame.
                if frames_missing[pid] > MAX_MISSING_FR and pid in calib_anchor:
                    player_pos[pid] = calib_anchor[pid].copy()

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
    cap.release()

    if unresolved_count:
        print(f"\n  {unresolved_count} detections could not be assigned a player_id "
              "(no track in map and centroid fallback failed).")

    # --- Write output JSON ---
    output = {
        "players": {pid: {"name": PLAYERS[pid]["name"]} for pid in PLAYER_IDS},
        "track_map": track_map,
        "calibration": {
            "frames_sampled": len(calib_frames),
            "frames_used": len(calib_results),
            "consensus_threshold": CONSENSUS_THRESHOLD,
        },
        "frames": enriched_frames,
    }
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
    """Render the source video with coloured bounding boxes and player names."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for render: {video_path}")
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_dim = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    render_path.parent.mkdir(parents=True, exist_ok=True)
    # avc1 (H.264) triggers h264_v4l2m2m probe on Linux which always fails on
    # x86 — go straight to mp4v (MPEG-4 Part 2), universally available via FFmpeg.
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
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fd = frame_lookup.get(idx)
        if fd:
            for p in fd["persons"]:
                pid = p.get("player_id")
                color = _PLAYER_COLORS.get(pid, _UNKNOWN_COLOR)
                name = PLAYERS[pid]["name"] if pid else (p.get("human_track_id") or "?")
                label = f"{pid} {name}" if pid else name
                # Draw label at top-centre of the bounding box: black shadow then coloured text
                lx = int(p["cx"]) - 2
                ly = max(int(p["y1"]) - 6, 12)
                cv2.putText(frame, label, (lx + 1, ly + 1),
                            FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS + 1, cv2.LINE_AA)
                cv2.putText(frame, label, (lx, ly),
                            FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pass 2: Identify players (P1..P4) from anonymous track IDs "
            "using Gemini vision calibration, then propagate across all frames."
        )
    )
    parser.add_argument("--video",        "-v", type=Path, required=True,
                        help="Source video (same file used for pass 1).")
    parser.add_argument("--detections",   "-d", type=Path, required=True,
                        help="Pass-1 detections JSON (contains human_track_id).")
    parser.add_argument("--output",       "-o", type=Path, required=True,
                        help="Output enriched JSON path.")
    parser.add_argument("--render-identified", type=Path, default=None,
                        help="If given, render a named+coloured video to this path.")
    parser.add_argument("--sample-window", type=float, default=SAMPLE_WINDOW_FRAC,
                        help=f"Fraction of video to sample for calibration (default: {SAMPLE_WINDOW_FRAC}).")
    return parser.parse_args()


def main() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        # Fall back to reading .env manually (no python-dotenv dependency)
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("GOOGLE_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set in environment or .env file.")
        sys.exit(1)

    args = _parse_args()
    identify_players(
        video_path=args.video,
        detections_path=args.detections,
        output_path=args.output,
        render_path=args.render_identified,
        sample_window_frac=args.sample_window,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()
