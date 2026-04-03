# Beach Volleyball Analysis — Viewer & Pipeline

This repo analyses beach volleyball video footage: detecting players, identifying them by name, tracking the ball, detecting rallies, and presenting results in a React viewer with a built-in ground-truth editor.

---

## Quick start

```bash
# 1. Install Python deps
uv sync

# 2. Build the viewer
cd viewer && npm install && npm run build && cd ..

# 3a. Player tracking — track, annotate one seed frame, identify
uv run beach run --video videos/GH021569_court.mp4
# → opens browser to assign P1–P4, then produces *_identified_heuristic.json

# 3b. Analytics pipeline — ball tracking, rally detection, combined render
uv run beach analytics --video videos/GH021569_court.mp4
# → *_ball.csv  *_merged.json  *_rallies.json  *_analytics.mp4

# 4. Open the action viewer
uv run beach serve
# → http://localhost:8080
```

---

## Pipeline — file flow

Three pipelines produce data for the viewer. The **player pipeline** must run before **analytics** (analytics needs the identified player data). **Actions** can run independently.

### Player pipeline

```
beach track       →  *_detections.json      persons detected & tracked (anonymous H1…Hn)
    │
    └─ beach identify  →  *_identified*.json    players identified (H-IDs → P1–P4)
            │
            └─ beach render  →  *_rendered.mp4  players rendered onto video
```

> **Shortcut — `beach run`** = track + identify + render, opening a browser seed UI before identify
>
> ```bash
> beach run --video clip.mp4
> beach run --video clip.mp4 --skip-track --skip-annotate  # re-run only identify + render
> ```

### Ball & Rallies pipeline

Requires `*_identified_heuristic.json` from the player pipeline.

```
beach ball-track   →  *_ball.csv          ball tracked (VballNet)
    │
    ├─ beach detect-rallies  →  *_rallies.json     rallies detected
    │           │
    │           └──────────────────────────────────┐
    │                                              ├─ beach merge  →  *_merged.json   everything merged
    └──────────────────────────────────────────────┤                        │
                                                   │           beach analytics-render  →  *_analytics.mp4
beach identify  →  *_identified*.json  ────────────┘
```

`beach merge` is the convergence point — it joins identified players, ball positions, and rally windows into a single file. `beach analytics-render` then only needs that one file.

> **Shortcut — `beach analytics`** = ball-track + detect-rallies + merge + analytics-render
>
> ```bash
> beach analytics --video clip.mp4
> beach analytics --video clip.mp4 --skip-ball-track --skip-rallies  # reuse existing files
> ```

### Actions pipeline

Independent — can run at any time on the raw or rendered video.

```
beach analyze  →  *_actions.json    actions analyzed via Gemini
```

### Ground truth

```
beach annotate-gt  →  *_gt.json    players annotated frame-by-frame (browser UI)
    │
    └─ beach eval-id               identification accuracy scored vs GT
```

---

```
beach serve  →  viewer at localhost:8080
```

### Individual steps

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
beach track --video videos/GH021569_court.mp4
# → videos/GH021569_court_detections.json

beach track --video videos/GH021569_court.mp4 --render-video videos/output/debug.mp4
# renders coloured H-ID boxes (slow, for debugging)
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
beach identify --video videos/GH021569_court.mp4
beach identify --video videos/GH021569_court.mp4 --no-llm
beach identify --video videos/GH021569_court.mp4 --no-llm --embeddings
beach identify --video videos/GH021569_court.mp4 --render-identified videos/output/debug_identified.mp4
```

---

### Analytics pipeline — `beach ball-track` → `beach merge` → `beach detect-rallies` → `beach analytics-render`

#### `beach ball-track`

Runs the VballNet sequence model (from `fast-volleyball-tracking-inference`) on the video and writes a ball position CSV. Delegates to that repo's own venv via subprocess — no extra Python deps needed in the `beach` venv.

**Model:** `VballNetV2_seq9_grayscale_320_h288_w512.onnx` (default). Override with `--model`.

**Outputs:** `*_ball.csv`

```
Frame, Visibility, X, Y
0,     0,          -1, -1
1,     1,          1068, 310
...
```

Coordinates are in original video pixel space. `Visibility=0` means no detection that frame.

```bash
beach ball-track --video videos/GH021569_court.mp4
# → videos/GH021569_court_ball.csv

beach ball-track --video videos/GH021569_court.mp4 --skip-existing
# skip if CSV already present
```

#### `beach merge`

Joins `*_identified_heuristic.json` (player bboxes with P1–P4 labels) and `*_ball.csv` (ball positions) into a single per-frame JSON. Also computes `closest_player_id` — the P1–P4 player whose foot point (bottom-centre of bbox) is nearest to the ball, within 400 px.

**Outputs:** `*_merged.json`

```jsonc
{
  "fps": 50.0,
  "total_frames": 14200,
  "frames": [
    {
      "frame": 42,
      "timestamp_sec": 0.84,
      "ball": { "x": 620.0, "y": 310.0, "visible": true },
      "closest_player_id": "P1",
      "players": [
        { "player_id": "P1", "cx": 400.0, "cy": 600.0,
          "x1": 350.0, "y1": 320.0, "x2": 450.0, "y2": 680.0 },
        ...
      ]
    }
  ]
}
```

```bash
beach merge --video videos/GH021569_court.mp4
```

#### `beach detect-rallies`

Groups ball-visible frames from `*_merged.json` into rally windows. Pauses longer than `--max-pause` seconds split rallies; windows shorter than `--min-rally` seconds are discarded. Each window is extended by `--extension` seconds on both ends.

**Outputs:** `*_rallies.json`

```jsonc
[
  { "rally_id": 0, "start_frame": 0,   "end_frame": 494,  "start_sec": 0.0,  "end_sec": 9.9  },
  { "rally_id": 1, "start_frame": 524, "end_frame": 784,  "start_sec": 10.5, "end_sec": 15.7 },
  ...
]
```

```bash
beach detect-rallies --video videos/GH021569_court.mp4
beach detect-rallies --video videos/GH021569_court.mp4 --max-pause 3.0 --min-rally 2.0
```

#### `beach analytics-render`

Renders `*_merged.json` + `*_rallies.json` as an overlay on the source video. Every frame shows:

- **Player bounding boxes** — coloured by P1–P4 (`P1`=white, `P2`=orange, `P3`=green, `P4`=yellow-green)
- **Closest player** — gold/thicker box + gold dot above the label
- **Ball** — bright yellow circle + crosshair
- **Rally banner** — green `RALLY N` strip across the top during a rally; yellow `START` / `END` flash at transitions (~1.2 s fade)

```bash
beach analytics-render --video videos/GH021569_court.mp4
# → videos/GH021569_court_analytics.mp4
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

### Primary steps — player pipeline
```
beach track            Persons detected & tracked (YOLO + ByteTrack) → *_detections.json
beach identify         Players identified (H-IDs → P1–P4, Gemini or heuristic) → *_identified*.json
beach render           Players rendered onto video → *_rendered.mp4
```

### Primary steps — analytics pipeline
```
beach ball-track       Ball tracked (VballNet) → *_ball.csv
beach detect-rallies   Rallies detected (from *_ball.csv) → *_rallies.json
beach merge            Players + ball + rallies merged → *_merged.json
beach analytics-render Analytics rendered onto video → *_analytics.mp4
                         (players + ball + rally markers overlay)
```

### Primary steps — actions pipeline
```
beach analyze          Actions analyzed via Gemini → *_actions.json
beach compare          Action JSON scored vs reference (timestamp matching)
```

### Primary steps — ground truth
```
beach annotate-gt      Players annotated frame-by-frame (browser UI) → *_gt.json
                         --first-frame   one seed frame only (used by beach run)
beach eval-id          Identification accuracy scored vs GT (swaps, confusion matrix)
beach eval-frame       Single-frame identification strategies scored vs GT
```

### Shortcuts
```
beach run              Player pipeline: track + identify + render (+ browser seed UI)
                         --skip-track      reuse existing *_detections.json
                         --skip-annotate   skip browser seed UI
beach analytics        Analytics pipeline: ball-track + detect-rallies + merge + analytics-render
                         --skip-ball-track  reuse existing *_ball.csv
                         --skip-rallies     reuse existing *_rallies.json
                         --skip-merge       reuse existing *_merged.json
                         --skip-render      data files only, no video output
```

### Viewer
```
beach serve            Viewer at localhost:8080
                         --port N      custom port
                         --reload      hot-reload Python on save
```
