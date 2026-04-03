# Beach Volleyball Analysis — Viewer & Pipeline

This repo analyses beach volleyball video footage: detecting players, identifying them by name, detecting ball contacts and actions, and presenting results in a React viewer with a built-in ground-truth editor.

---

## Quick start

```bash
# 1. Install Python deps
uv sync

# 2. Build the viewer
cd viewer && npm install && npm run build && cd ..

# 3a. Single command — track, annotate one frame, identify, render
uv run beach run --video videos/GH021569_court_001.mp4
# → opens browser to assign P1–P4, then produces *_rendered.mp4

# 3b. Or run each step individually (see pipeline below)
uv run beach track   --video videos/GH021569_court_001.mp4
uv run beach annotate-gt --video videos/GH021569_court_001.mp4 --first-frame
uv run beach identify --video videos/GH021569_court_001.mp4 --no-llm --seed-gt videos/GH021569_court_001_gt.json
uv run beach render  --video videos/GH021569_court_001.mp4 -o videos/GH021569_court_001_identified_heuristic.json

# 4. Open the action viewer
uv run beach serve
# → http://localhost:8080
```

---

## Pipeline — file flow

### Fast path — `beach run` (single command)

```bash
beach run --video clip.mp4
# produces: clip_detections.json  clip_gt.json  clip_identified_heuristic.json  clip_rendered.mp4

# skip already-done steps when iterating:
beach run --video clip.mp4 --skip-track --skip-annotate
```

The `--skip-track` and `--skip-annotate` flags let you re-run only identify + render after tweaking things, without waiting for YOLO or opening the browser again.

### Full pipeline (individual steps)

A single clip passes through these stages in order:

```
raw video
    │
    ▼
[1] beach track          → *_detections.json   (+ optional *_annotated.mp4)
    │
    ├────────────────────────────────────────────────────────────────┐
    ▼                                                                │
[2] beach identify       → *_identified.json                        │
    │   (LLM: Gemini calibration)                                    │
    │   (--no-llm: colour templates → auto-seed)                     │
    │   (--no-llm --seed-gt: needs *_gt.json ◄──────────────────┐   │
    │                                                            │   │
    │                                         beach annotate-gt ┘   │
    │                                         (browser UI, uses ─────┘
    │                                          detections + identified)
    │                                         → *_gt.json
    │                                         beach eval-id
    │                                         (scores identified vs gt)
    │
    ├──► beach render     → *_rendered.mp4  (re-render overlay without re-running)
    │
    ▼
[3] beach analyze        → *_<model>_<ts>.json  (action events)
    │
    ├──► beach compare   (scores action JSON vs reference)
    │
    ▼
[4] beach serve          (viewer + editor served at localhost:8080)
```

### Pass 1 — `beach track`

**What it does:**
Runs YOLO11n for person detection and a volleyball-specific YOLO model for ball detection on every frame. ByteTrack (Kalman-filter + two-stage IoU re-association) assigns stable anonymous IDs `H1`, `H2`, … across frames. These are *not* player names — they are just temporal anchors.

For each detected person the torso region (rows 20–65 % of the bounding box) is sampled and sand-masked to produce a `[H, S, V]` colour descriptor stored as `color_hsv`. This is used downstream for player identification.

**Inputs:** raw video (`.mp4`)  
**Outputs:** `<stem>_detections.json`

```jsonc
// detections JSON schema
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
          "color_hsv": [85.0, 34.0, 95.0],
          "human_track_id": "H2"   // null when tracking lost
        }
      ],
      "ball": { "cx": 620.2, "cy": 188.0 }  // null when not detected
    }
  ]
}
```

```bash
beach track --video videos/GH021569_court_001.mp4
# → videos/GH021569_court_001_detections.json

beach track --video videos/GH021569_court_001.mp4 --render-video videos/output/debug.mp4
# renders coloured H-ID boxes + ball circle (slow, for debugging)
```

---

### Pass 2 — `beach identify`

**What it does:**
Maps anonymous track IDs (`H1`…`Hn`) → named player IDs (`P1`…`P4`). Two modes:

#### LLM mode (default)
1. Select up to 8 *calibration frames* from the first 30 % of the clip where exactly 4 uniquely-tracked persons are visible and well-separated.
2. For each calibration frame: extract the full JPEG + 4 person crops → send to Gemini Flash with a player-description prompt → receive `[{detection_index, player_id}]`.
3. Vote across calibration frames; accept the assignment when one player holds > 50 % of votes for that track.
4. Build a `track_map: {H2: P1, H1: P3, …}` and propagate it across all frames with a rolling position-tracker (see below).

#### Heuristic mode (`--no-llm`)
Skips Gemini entirely. Seeds initial player positions using Hungarian assignment on HSV colour-template distance against hardcoded cluster centres (`_SEED_COLOR_TEMPLATES` in `identify.py`, derived from the manually annotated clip). Then runs the same rolling tracker.

#### GT-seeded heuristic mode (`--no-llm --seed-gt <gt.json>`)
Same rolling tracker but skips colour-template auto-seeding. Instead, reads the first confirmed annotation from a `*_gt.json` (produced by `beach annotate-gt`) and uses those known player positions as the seed. This isolates tracker quality from seeding quality — it shows the ceiling the tracker can reach when given a perfect starting point.

```bash
beach track       --video clip.mp4
beach annotate-gt --video clip.mp4          # confirm a few early frames
beach identify    --video clip.mp4 --no-llm --seed-gt clip_gt.json
beach eval-id     --video clip.mp4 --no-llm
```

#### Embeddings mode (`--no-llm --embeddings`)
Adds a DINOv2 visual-embedding gallery (Strategy B) on top of either heuristic mode. Each confidently-identified crop is enrolled; future detections are matched by cosine similarity blended with position + colour cost. Can be combined with `--seed-gt`:
```bash
beach identify --video clip.mp4 --no-llm --embeddings --seed-gt clip_gt.json
```

#### Rolling tracker (used by all modes)
After calibration, all frames are processed with a Hungarian cost-matrix assignment:
- **Phase A (H-ID continuity):** if the incoming `human_track_id` already appears in `running_hid_map`, inherit that assignment directly — no cost matrix needed.
- **Phase B (cost matrix):** for unknown tracks, build a `(n_unknown_detections × n_free_players)` cost matrix blending:
  - position distance (EMA-smoothed running position, gated by `MOVE_PX_PER_FRAME × (1 + frames_missing)`)
  - HSV colour distance (weight 0.25 LLM / 0.40 heuristic)
  - DINOv2 cosine distance (weight 0.40, only with `--embeddings`)
- Hungarian assignment selects the minimum-cost assignment; new `H-ID → P-ID` bindings are added to `running_hid_map`.

**Inputs:** video + `*_detections.json`  
**Outputs:** `*_identified.json` (LLM), `*_identified_heuristic.json` (heuristic), `*_identified_embeddings.json` (embeddings)

```bash
beach identify --video videos/GH021569_court_001.mp4
beach identify --video videos/GH021569_court_001.mp4 --no-llm
beach identify --video videos/GH021569_court_001.mp4 --no-llm --embeddings
beach identify --video videos/GH021569_court_001.mp4 --render-identified videos/output/debug_identified.mp4
```

---

### Ground truth — `beach annotate-gt` + `beach eval-id`

Before measuring identification quality you need labelled frames.

```bash
# Open browser annotation UI at http://localhost:7780
beach annotate-gt --video videos/GH021569_court_001.mp4

# Score the identified output against the GT
beach eval-id --video videos/GH021569_court_001.mp4
beach eval-id --video videos/GH021569_court_001.mp4 --no-llm
```

`annotate-gt` samples one keyframe every ~2 s, pre-seeds them from the identified JSON, and serves a browser UI where you confirm or correct P1–P4 assignments per frame. Saves to `*_gt.json`.

`eval-id` reports: overall accuracy, per-player precision/recall, identity swap rate, and a confusion matrix.

---

### Render — `beach render`

Re-render an identified JSON onto the source video without re-running identification (e.g. after editing the JSON):

```bash
beach render -v videos/GH021569_court_001.mp4 -o videos/output/GH021569_court_001_identified_heuristic.json
# → *_rendered.mp4
```

---

### Pass 3 — `beach analyze`

Uploads the video to Gemini and extracts timestamped player actions. Can be run on the raw video or on an annotated/rendered video with P1–P4 bounding-box labels already drawn (use `--annotated` to add an additional hint in the prompt).

Supports multi-run convergence analysis (`--runs N`), timestamp seeding from a prior run (`--input`), and auto-seeding (`--auto-seed`).

**Outputs:** `data/<stem>/<stem>_<model>_[seeded_]run<n>_<ts>.json`

```jsonc
// action JSON schema
[
  { "timestamp_sec": 3.5, "player_id": "P1", "action": "Serve", "player_description": "Denny (black tshirt)" },
  { "timestamp_sec": 5.1, "player_id": "P3", "action": "Reception" },
  ...
]
```

Action types: `Serve`, `Reception`, `Set`, `Attack`, `Dig`, `Block`, `Free Ball Sent`, `Free Ball Received`.

```bash
beach analyze --video data/first30/first30.mp4 --runs 5 --auto-seed
beach analyze --video data/first30/first30.mp4 --annotated
```

---

### Compare — `beach compare`

Score a candidate action JSON against a reference (ground truth) using ±2 s timestamp matching and per-pair action + player scoring:

```bash
beach compare data/first30/run1.json --ref data/first30/first30_Manual.json
```

---

### Viewer — `beach serve`

Serves the built React viewer and data directory. API:

| Endpoint | Description |
|---|---|
| `GET /api/videos` | List `data/<stem>/` subdirectory names |
| `GET /api/videos/<stem>/actions` | List all files in `data/<stem>/` |
| `GET /api/videos/<stem>/actions/<file>` | Fetch a JSON file |
| `POST /api/videos/<stem>/actions/<file>` | Save edited JSON back to disk |
| `GET /data/<stem>/<file>` | Serve any file with HTTP Range support (for video seeking) |

The viewer lists videos from the left selector, shows the video + timeline of action events, and allows editing actions directly in the browser.

```bash
beach serve                          # port 8080
beach serve --port 9000 --reload     # custom port + hot-reload Python on save
```

---

## Data layout

Currently data lives in two places and this causes friction. Proposed clean layout:

```
videos/
  GH021569.MP4                  ← original raw recordings (large, not committed)
  first30.mp4                   ← short test clip

data/
  <video-stem>/                 ← one folder per clip
    <stem>.mp4                  ← symlink or copy of the clip (for the viewer)
    <stem>_detections.json      ← pass 1 output
    <stem>_identified.json      ← pass 2 LLM output
    <stem>_identified_heuristic.json
    <stem>_identified_embeddings.json
    <stem>_gt.json              ← ground truth annotations
    <stem>_<model>_run1_<ts>.json   ← pass 3 action outputs
    <stem>_Manual.json          ← hand-labelled actions (ground truth)
    players.json                ← player roster for this clip
    .gemini_file_cache.json     ← cached Gemini upload URIs (hidden)
```

> **Problem today:** raw videos live in `videos/`, cropped sub-clips in `videos/output/`, pass-1/2 outputs scattered next to their source video, and pass-3 action JSONs in `data/<stem>/`. The viewer's `--data-dir` (default `data/`) only sees action JSONs — not detection/identified JSONs. Consider either consolidating everything into `data/<stem>/` (with symlinks for raw video) or adding a second data dir to the serve command.

---

## Player detection — current approach vs. alternatives

### What the code does

**Pass 1 (YOLO + ByteTrack):** Standard YOLO person detection, ByteTrack for temporal consistency. Good but produces anonymous `H1..Hn` IDs that can reset after occlusions or long absences.

**Pass 2 identification — LLM path:**
- Samples calibration frames and calls Gemini Flash with full frame + per-person crops.
- Majority vote across frames builds `H-ID → P-ID` map.
- Rolling Hungarian tracker propagates the map frame-by-frame using position (dominant) + HSV colour (secondary) + optionally DINOv2 embeddings.

**Pass 2 identification — heuristic path:**
- Bootstraps from hardcoded HSV colour templates (`_SEED_COLOR_TEMPLATES` in `identify.py`) derived from one specific match.
- Works well when player outfits match the templates; breaks on different clothing.

### Known weaknesses

| Problem | Where it bites |
|---|---|
| Hardcoded colour templates | Heuristic mode only works for this specific player roster and outfit |
| ByteTrack H-ID resets after occlusion | Phase A inherits stale mappings; Phase B has to re-match |
| Seed frame sensitivity | `_seed_from_detections` can pick a frame with a ghost detection, poisoning the rolling tracker for the whole clip |
| Duplicate detections (IoU>0.3) | NMS guard mitigates but can suppress real players when they overlap |
| Fixed MOVE_PX_PER_FRAME gate | Too tight during dives/sprints, too loose for long absences |
| colour_hsv is a 3-float summary | Loses spatial information; jerseys with two colours (e.g. Bjirk) collapse |

### Ideas for making identification more robust

1. **Re-ID model as drop-in.** Replace the HSV descriptor with OSNet or a lightweight person re-ID backbone. The gallery/enrollment structure in `embeddings.py` already supports this — swap the DINOv2 backbone for a re-ID-trained model.

2. **Court-position prior.** In beach volleyball, teams stay on their half. Encoding a soft left/right-half prior into the Phase B cost matrix would break ties and reduce cross-net misassignments.

3. **Tracklet-level rather than frame-level identification.** Instead of deciding P-ID per frame, group frames into ByteTrack tracklets and run identification once per tracklet. Reduces Gemini API cost and makes the decision more robust.

4. **Automatic colour template extraction.** Rather than hardcoded HSV clusters, run k-means on the `color_hsv` values from the first N frames (or from a GT-confirmed frame) to derive per-run templates automatically.

5. **Confidence-gated propagation.** When the Phase B cost-matrix minimum exceeds a threshold (ambiguous assignment), emit `player_id = null` rather than a low-confidence guess, so downstream `eval-id` and the viewer can flag uncertain frames.

6. **Better seeding with `--seed-gt`.** The `--seed-gt` flag (already implemented) bypasses auto-seeding entirely and uses the first confirmed GT annotation. Use this to measure the ceiling of the tracker independent of seeding quality.

---

## Viewer — current state and intended direction

The viewer is a **dual-purpose tool**:

- **Editing mode (now):** review action JSONs produced by `beach analyze`, correct player IDs / action types, save back to disk.
- **Viewing mode (future):** browse games, scrub the timeline, filter by player or action, compare multiple model runs side by side.

### Data loading

The viewer calls `GET /api/videos` to list stems, then `GET /api/videos/<stem>/actions` to list all files in `data/<stem>/`. The frontend filters for `.json` and `.mp4` files. Select a video stem → the viewer loads the video and all action JSONs found in the same directory.

### Planned: ground-truth integration

The viewer should be able to:
- Load `*_gt.json` (from `beach annotate-gt`) alongside action run files.
- Visually diff a run against the GT (colour-coded OK / PARTIAL / WRONG per event).
- Mark events as confirmed, corrected, or deleted, and POST the result back.

This closes the loop: `annotate-gt` creates the frame-level player GT; the viewer creates and edits the action GT; `compare` and `eval-id` score everything.

---

## CLI reference

```
beach run           Full pipeline: track → annotate first frame → identify → render
beach track         Pass 1: YOLO person + ball detection, ByteTrack IDs → *_detections.json
beach identify      Pass 2: H-ID → P-ID via Gemini or heuristic → *_identified*.json
beach annotate-gt   Browser UI for frame-level GT annotation → *_gt.json
                      --first-frame   one seed frame only (fast, for beach run)
beach eval-id       Score identified vs GT (accuracy, swaps, confusion matrix)
beach eval-frame    Score single-frame identification strategies vs GT
beach render        Re-render identified JSON onto source video → *_rendered.mp4
beach analyze       Pass 3: action extraction via Gemini → action JSON
beach compare       Score action JSON vs reference (timestamp matching)
beach serve         Dev server for the viewer (localhost:8080)
```
