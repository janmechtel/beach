# Periodic Gemini Re-Identification Design

## Problem
`identify_players()` currently calls Gemini once during calibration, then relies on a static `track_map` from `human_track_id -> player_id`.

With ByteTrack-style H-ID fragmentation (ID switches after occlusion, overlap, or detector instability), new H-IDs appear within seconds. Once that happens, the original map becomes stale and assignments degrade.

## Proposed Solution
Add periodic and event-driven Gemini re-identification:

- Re-run Gemini every `REID_INTERVAL_SEC` seconds (configurable, default `10`).
- Re-run Gemini immediately when a new H-ID appears that is not in current `track_map`.

This keeps identity mapping fresh while bounding cost.

## Trigger Conditions
1. **Periodic trigger**: every `REID_INTERVAL_SEC` seconds.
2. **New-ID trigger**: any observed `human_track_id` not present in current `track_map`.

Use periodic trigger only when there are enough visible candidates for reliable re-identification.

## Implementation Sketch
Integration point: inside the per-frame loop in `identify_players()` (`beach/identify.py`).

State:
- `last_reid_frame: int` (initialize to calibration frame or `0`)
- `gemini_calls: int`
- `MAX_REID_CALLS: int = 20` (configurable)
- `REID_INTERVAL_SEC: float = 10.0` (configurable)

Per frame:
1. Compute `interval_frames = int(fps * REID_INTERVAL_SEC)`.
2. Compute `periodic_due = (current_frame - last_reid_frame) > interval_frames`.
3. Compute `has_new_hid = any(hid not in track_map for hid in current_frame_hids)`.
4. If `(periodic_due or has_new_hid)` and `len(persons) >= 3` and `gemini_calls < MAX_REID_CALLS`:
   - Extract full-frame JPEG plus person crops.
   - Call Gemini for identity mapping.
   - Merge/update `track_map` (prefer latest high-confidence map entries).
   - Set `last_reid_frame = current_frame`.
   - Increment `gemini_calls`.

Budget behavior:
- When budget is exhausted, skip Gemini calls and continue with non-LLM fallback logic.

## Tradeoffs
- **Cost**: More Gemini calls increase API usage; budget cap prevents runaway spend.
- **Latency**: Synchronous calls can stall frame processing; async queueing can reduce blocking but increases complexity.
- **Consistency**: Sequential Gemini calls may contradict earlier mappings; requires deterministic merge policy (e.g., latest wins only above confidence threshold).

## Notes
- Keep re-identification decoupled from initial calibration so stale mappings are corrected over time.
- Keep trigger logic simple and explicit so behavior is predictable and tunable.
