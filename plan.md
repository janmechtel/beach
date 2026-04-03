# Beach Volleyball Analytics — Enhancement Plan

## Current State

### `beach` repo (`/home/jan/kontext/beach`)
- **`beach track`** — YOLO person detection (ByteTrack) + volleyball-YOLO ball detection → `*_detections.json`
  - Output per frame: `{frame, timestamp_sec, persons: [{cx,cy,x1,y1,x2,y2,conf,color_hsv,human_track_id}], ball: {cx,cy}}`
  - Ball detection via `volleyball_yolo11n.pt` is basic (single best box per frame, no tracking continuity)
- **`beach annotate`** / **`beach run`** — seed-frame annotation UI + rolling heuristic player identity tracker → `*_identified_heuristic.json`
- **`beach render`** — overlay coloured boxes + labels → `*_rendered.mp4`
- **`models.py`** — `Action` Pydantic model: `{timestamp_sec, player_id, action, player_description}`
- **`viewer/`** — React viewer that loads `*_identified*.json` + actions

### `fast-volleyball-tracking-inference` repo (`/home/jan/kontext/fast-volleyball-tracking-inference`)
- **`inference_onnx_seq_gray_v2.py`** — Runs VballNet (ONNX, sequence-based, grayscale) on a video → `ball.csv` (`Frame,Visibility,X,Y`) at **model input resolution** (288×512 default, or 432×768 for Grid models)
- **`rally_detecter.py`** — Reads `ball.csv`, groups frames into rallies by pause gaps, filters short flights, interpolates → `DataFrame` with `Rally_ID, Frame, X_interp, Y_interp`
- **`make_reels.py`** — Given rally track JSONs, crops and exports rally video clips centered on ball X position
- **`ball_tracker.py`** — `BallTracker` class: multi-track ball tracking with velocity prediction, outputs `Track` objects with `{positions: [([x,y], frame)], start_frame, last_frame}`
- **`player_tracker.py`** — `PlayerTracker`: YOLO player detection + perspective transform to court coords + closest-player-to-ball
- **`pose_detector.py`** — `PoseDetector`: MediaPipe per-player pose in ROI, `find_closest_player` by bbox center distance

---

## What We Want to Build

Given a raw beach volleyball video, produce:
1. **Ball positions** per frame (robust, sequence-model quality)
2. **Rally start/end times** (frame numbers + seconds)
3. **Player bounding boxes** per frame with stable identities (P1–P4)
4. **Per-touch events**: `{frame, timestamp_sec, player_id, action_type}`
5. **Rally reels** — cropped clips one per rally

---

## Key Alignment Problem

The two pipelines both read the same source video with `cv2.VideoCapture` and use 0-based frame indices — so frame numbers are already aligned by definition.

The one thing that does need fixing before merging: **ball CSV X/Y are in model-input resolution** (e.g. 512×288), while player bbox coordinates from `beach track` are in original video resolution. X/Y must be rescaled to original pixels in Step 1 before the merge in Step 4.

**Rule**: every artifact uses `frame_idx` (0-based, original video) as its canonical key. Seconds-based fields are always derived as `frame_idx / fps`.

---

## Steps

### Step 1 — Ball Tracking (fast-volleyball-tracking-inference → CSV)

**Goal**: Produce `<video_stem>_ball.csv` with columns `Frame, Visibility, X_orig, Y_orig` where X/Y are in **original video pixel coordinates**.

**What exists**: `inference_onnx_seq_gray_v2.py` already outputs `Frame,Visibility,X,Y` using the same source video and the same 0-based frame index as `beach track`. The frame numbers are already aligned — no timestamp translation needed.

The only issue is that X/Y are in model-input space (e.g. 512×288). Since the model internally resizes each frame, the output coordinates need to be rescaled back to original resolution before merging with the player bbox data (which is in original pixels).

**Work**:
- Add coordinate rescaling to original resolution: `X_orig = X * frame_width / input_width`, same for Y
- Add `timestamp_sec` column: `Frame / fps`
- Sanity check: assert `len(csv) ≈ total_frames` of source video (within ±batch_size tolerance due to sequence buffering at start)

**Inputs**: raw `.mp4`  
**Outputs**: `<stem>_ball.csv` — `Frame, Visibility, X_orig, Y_orig, timestamp_sec`  
**Script**: extend `inference_onnx_seq_gray_v2.py` or wrap in a new `run_ball_tracking.py`

---

### Step 2 — Rally Detection

**Goal**: Produce `<video_stem>_rallies.json` — list of `{rally_id, start_frame, end_frame, start_sec, end_sec}`.

**What exists**: `rally_detecter.py::process_volleyball_data()` already does this from the CSV, but returns a DataFrame. Needs a JSON-serialising wrapper.

**Work**:
- Write `export_rallies_json(df, fps, output_path)` that groups by `Rally_ID`, reads min/max `Frame` per group, computes `start_sec = start_frame/fps`, `end_sec = end_frame/fps`
- Input CSV must use original-video frame numbers (after Step 1 rescaling)
- Configurable params: `max_pause` (default 2 s), `min_flight_duration` (default 1 s), `time_extension` (default 1 s)

**Inputs**: `<stem>_ball.csv`  
**Outputs**: `<stem>_rallies.json`
```json
[
  {"rally_id": 0, "start_frame": 312, "end_frame": 784, "start_sec": 10.4, "end_sec": 26.1},
  ...
]
```
**Script**: new `export_rallies.py` (thin wrapper around existing `rally_detecter.py`)

---

### Step 3 — Player Bounding Boxes (beach repo, already works)

**Goal**: `<video_stem>_detections.json` — per-frame player bboxes with ByteTrack IDs.

**What exists**: `beach track` already does this. Output has `persons[{cx,cy,x1,y1,x2,y2,human_track_id}]` and `ball:{cx,cy}` per frame in original-video pixel space.

**Work**:
- The ball detection in `beach track` is a simple single-frame YOLO prediction (not sequence-based) — it can stay as a fallback but **Step 1's CSV is the authoritative ball source**
- The player identity (P1–P4) from `beach run` / `beach annotate` gives us `<stem>_identified_heuristic.json`

**Inputs**: raw `.mp4`  
**Outputs**: `<stem>_detections.json`, `<stem>_identified_heuristic.json`  
**Script**: existing `beach run` — no changes needed for this step

---

### Step 4 — Timestamp Alignment & Merging

**Goal**: Produce `<video_stem>_merged.json` — per-frame record combining ball position + player bboxes with stable P1–P4 identity.

**Why this is its own step**: The two pipelines (ball CSV from fast-inference, player JSON from beach) are run independently and need to be joined on `frame_idx`.

**Work**:
- Write `merge_tracks.py`:
  - Load `*_ball.csv` (Frame → ball X/Y/Visibility)
  - Load `*_identified_heuristic.json` (frame → persons with `player_id`)
  - Assert same FPS and compatible frame ranges (warn if lengths differ by > 5 frames)
  - Inner-join on `frame_idx`; fill missing ball frames with `Visibility=0`
  - Output one record per frame

**Output schema** (`<stem>_merged.json`):
```json
{
  "fps": 30.0,
  "total_frames": 9000,
  "frames": [
    {
      "frame": 42,
      "timestamp_sec": 1.4,
      "ball": {"x": 620, "y": 310, "visible": true},
      "players": [
        {"player_id": "P1", "cx": 400, "cy": 600, "x1": 350, "y1": 320, "x2": 450, "y2": 680},
        ...
      ]
    }
  ]
}
```
**Script**: new `merge_tracks.py`

---

### Step 5 — Closest Player to Ball

**Goal**: For each frame where the ball is visible, annotate which player is closest.

**Source of player data**: `*_identified_heuristic.json` from `beach run` — this gives P1–P4 labelled bboxes `{x1,y1,x2,y2}` per frame. This is the only player tracking we use; the `player_tracker.py` from `fast-volleyball-tracking-inference` is **not used**.

**Work**:
- Add `closest_player_id` field to each frame in `*_merged.json` where `ball.visible == true`
- Use foot point `(cx, y2)` — center-bottom of bbox — as each player's ground position
- Euclidean distance from ball `(X_orig, Y_orig)` to each player foot point
- Distance threshold: only assign if `distance < threshold_px` (configurable, default 300 px) — if ball is far from all players, `closest_player_id = null`

**Inputs**: `<stem>_merged.json`  
**Outputs**: adds `closest_player_id` field in-place (or new `<stem>_merged_proximity.json`)  
**Script**: extend `merge_tracks.py` or new `annotate_proximity.py`

---

### Step 6 — Action Detection

Two complementary approaches, to be evaluated independently:

#### 6a — Pose-Based Action Detection

**Goal**: For each frame where ball is near the closest player, classify the action from body pose.

**Note on pose tooling**: `pose_detector.py` from `fast-volleyball-tracking-inference` uses MediaPipe, which is fine as a library dependency, but the rest of that repo's player tracking is **not used**. We implement pose detection directly in the `beach` repo using MediaPipe.

**Work**:
- For each frame where `closest_player_id` is set and `distance(ball, player) < touch_threshold`:
  - Extract player ROI from `(x1,y1,x2,y2)` + padding (sourced from `*_identified_heuristic.json`)
  - Run MediaPipe pose → 33 landmarks
  - Use landmark geometry to classify action:
    - Arms above head + ball near hands → **Set** or **Attack** or **Serve**
    - Arms extended low → **Reception** or **Dig**
    - Arms raised at net level → **Block**
    - Rule-based heuristics on wrist/elbow/shoulder angles first; later a small classifier
- Output candidate touch frames: `{frame, player_id, pose_landmarks, action_candidate}`
- Dedup: merge frames within ±5 frames into a single touch event (keep frame with highest pose confidence)

**Inputs**: `<stem>_merged.json`, raw video  
**Outputs**: `<stem>_touch_candidates_pose.json`  
**Script**: new `beach/detect_actions_pose.py` (MediaPipe only, no fast-inference player tracker)

#### 6b — LLM-Based Action Detection (existing capability)

**Goal**: Use Gemini to classify actions from short video segments around each touch candidate.

**What exists**: `beach/identify.py` already calls Gemini with video frames. `models.py` defines `Action` schema.

**Work**:
- For each rally (from `*_rallies.json`), extract a ±2 s clip around each touch candidate frame
- Feed clip to Gemini with a structured prompt: given player P1–P4 labels, return `[{timestamp_sec, player_id, action}]`
- Validate against `Action` schema (Pydantic)
- Use pose candidates (Step 6a) as *hints* in the prompt to anchor Gemini's attention

**Inputs**: `<stem>_touch_candidates_pose.json`, `<stem>_rallies.json`, raw video  
**Outputs**: `<stem>_actions.json` — list of `Action` objects  
**Script**: extend existing `beach/identify.py` or new `detect_actions_llm.py`

**Note**: LLM approach is slower and costs money; pose-based is free and runs offline. Target: use pose for real-time first pass, LLM for quality review.

---

### Step 7 — Rally Clips (optional, for LLM upload)

**Goal**: For each rally, produce a cropped `.mp4` clip centered on the ball.

**Primary use**: The viewer works off the full-length video + timestamps from `*_rallies.json` — it does **not** need these clips. The clips are kept solely as a convenience for uploading short segments to an LLM (Step 6b) without sending the entire video.

**What exists**: `make_reels.py::crop_and_save_track_payload()` already does this from a track JSON with `{start_frame, last_frame, positions: [([x,y], frame)]}`.

**Work**:
- Write `export_reels.py` that:
  - Reads `<stem>_rallies.json` + `<stem>_ball.csv`
  - For each rally, builds a track payload from the CSV rows in that rally's frame range
  - Calls `crop_and_save_track_payloads()` with `smoothing="moving_avg"`, `padding="mirror"`
  - Output: `reels/<stem>_rally_<N>.mp4`
- Overlay P1–P4 labels using player bbox data from `<stem>_merged.json` so the LLM sees player identities
- **Can be skipped** when running the viewer pipeline — only needed before Step 6b LLM calls

**Inputs**: `<stem>_rallies.json`, `<stem>_ball.csv`, `<stem>_merged.json`, raw video  
**Outputs**: `reels/<stem>_rally_<N>.mp4`  
**Script**: new `export_reels.py`

---

### Step 8 — Unified Pipeline Script

**Goal**: Single entry point `beach analytics --video <file>` that runs all steps.

**Work**:
- New `beach/analytics.py` CLI command (add to `cli.py`)
- Orchestrate steps 1–7 with `--skip-*` flags for each step (same pattern as `beach run`)
- All intermediate files live next to the video with predictable names:
  ```
  videos/
    GH021569_court.mp4               ← source
    GH021569_court_detections.json   ← Step 3 (beach track, existing)
    GH021569_court_identified.json   ← Step 3 (beach run, existing)
    GH021569_court_ball.csv          ← Step 1
    GH021569_court_rallies.json      ← Step 2
    GH021569_court_merged.json       ← Steps 4+5
    GH021569_court_actions.json      ← Step 6
    reels/
      GH021569_court_rally_0.mp4     ← Step 7
      GH021569_court_rally_1.mp4
      ...
  ```
- Progress reporting per step (existing pattern: `[N/M] step_name — ...`)

---

## Dependency Graph

```
raw video
    │
    ├──► [Step 1] inference_onnx_seq_gray_v2.py ──► *_ball.csv
    │         │
    │         └──► [Step 2] export_rallies.py ──────► *_rallies.json
    │
    ├──► [Step 3] beach run ──────────────────────── *_detections.json
    │         (YOLO + ByteTrack, beach repo only)     *_identified_heuristic.json
    │
    ├──(Step 1 + Step 3)──► [Step 4+5] merge_tracks.py ──► *_merged.json
    │                                                        (+ closest_player_id)
    │
    ├──(*_merged.json + raw video)──► [Step 6a] detect_actions_pose.py ──► *_touch_candidates.json
    │
    ├──(*_touch_candidates + *_rallies + raw video)──► [Step 6b] detect_actions_llm.py ──► *_actions.json
    │
    └──(*_rallies + *_ball.csv + *_merged + raw video)──► [Step 7] export_reels.py ──► reels/*.mp4
```

Steps 1–3 are independent and can run in parallel.  
Steps 4–7 each depend on outputs from earlier steps.

**Player tracking source**: exclusively `beach` repo (`beach run` / `beach track`). `player_tracker.py` from `fast-volleyball-tracking-inference` is not part of the pipeline.

---

## Open Questions / Risks

1. **Ball CSV frame alignment**: The inference script processes batches of `seq=9` frames, so output frame indices may be offset by up to 8 frames at the start. Need to verify and document the exact frame-to-output-index mapping.

2. **Model input resolution vs. original resolution**: The CSV X/Y are in 512×288 space. The rescaling is straightforward but must be applied before any merge with player data (which is in original resolution).

3. **Player ID stability across rallies**: ByteTrack IDs can reset across long gaps. The `beach run` heuristic tracker handles this with a GT seed frame, but needs testing on longer videos with multiple rallies.

4. **Pose accuracy near edges**: MediaPipe struggles when a player is partially out of frame (e.g. near the net). May need fallback to geometric heuristics only in those cases.

5. **LLM cost and latency**: Gemini calls are expensive for full matches. Rate-limit to 1 call per rally, not per touch candidate.

6. **Action label granularity**: The `Action` schema has 8 types. Pose-based detection is realistic for: Serve, Attack, Block, Set (arm position). Reception/Dig are harder to distinguish from pose alone — ball trajectory direction (rising vs falling) is a better signal.

---

## Implementation Order

| Priority | Step | Effort | Value |
|----------|------|--------|-------|
| 1 | Step 1 — Ball CSV with rescaled coords | Low | High — unblocks everything |
| 2 | Step 2 — Rally JSON export | Low | High — enables reels |
| 3 | Step 4+5 — Merge + proximity | Medium | High — core data join |
| 4 | Step 7 — Rally reels | Low | High — immediate visual output |
| 5 | Step 6a — Pose action detection | Medium | Medium — offline, free |
| 6 | Step 8 — Unified CLI | Medium | High — usability |
| 7 | Step 6b — LLM action detection | High | High — accuracy |
| 8 | Step 3 improvements | Low | Low — already works |
