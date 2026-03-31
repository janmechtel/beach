# Hybrid Multi-Signal Player Identification Strategy

## Goal
Combine complementary signals to improve robustness across ID switches, occlusions, and appearance changes:
- H-ID continuity
- DINOv2 embedding similarity
- Periodic Gemini re-identification
- Position gate (Hungarian assignment on position+color cost)

## Signal Hierarchy (Priority Order)
When signals conflict, evaluate in this order:

1. **H-ID continuity**
   - If current `human_track_id` matches prior frame and exists in `track_map`, reuse mapped `player_id`.
2. **DINOv2 embedding similarity**
   - Compare crop embedding against gallery prototypes.
   - If similarity `> 0.85`, use best match.
3. **Gemini re-identification**
   - If periodic/new-ID trigger fires, call Gemini and update `track_map`.
4. **Position gate fallback**
   - Use Hungarian assignment over position+color cost when other signals are unavailable/weak.

## Per-Signal Confidence Model
Each signal emits `(player_id, confidence)`:

- H-ID continuity: high confidence when stable recent history exists.
- DINOv2: confidence derived from cosine similarity margin.
- Gemini: confidence from model response quality/consistency checks.
- Position gate: confidence inverse to normalized assignment cost.

## Arbitration Rule
1. Collect all candidate `(player_id, confidence, signal_type)` proposals.
2. Pick the highest-confidence proposal.
3. If top two confidences differ by `<= 0.05`, prefer the higher-priority signal from the hierarchy above.

This avoids instability when scores are nearly tied.

## Gallery Management
Maintain an embedding gallery that is quality-controlled:

- **Enroll** crops that are Gemini-confirmed (trusted labels).
- **Reject** low-quality enrollments when DINOv2 similarity is below threshold.
- Keep multiple prototypes per player to cover viewpoint/lighting variation.
- Optionally decay or prune stale prototypes to limit drift.

## When to Apply
Adopt this hybrid arbitration after Strategy A and Strategy B are benchmarked individually, so gains are attributable and thresholds can be tuned with clean baselines.

## Expected Gains
Each signal covers another signal's failure mode:
- H-ID continuity handles short-term temporal consistency.
- DINOv2 handles appearance-based recovery after track fragmentation.
- Gemini resolves ambiguous identity cases and re-seeds mapping.
- Hungarian position gate provides deterministic fallback when semantic signals are weak.

Result: fewer persistent identity swaps and faster recovery from ByteTrack H-ID churn.
