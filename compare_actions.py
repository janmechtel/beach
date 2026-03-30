#!/usr/bin/env python3
"""
compare_actions.py — compare a candidate action JSON against the manual ground truth.

Usage:
    python compare_actions.py <candidate.json> [--ref output/first30_Manual.json] [--tol 2.0]

Scoring:
    Matching is performed by optimal assignment: each reference event is paired with
    the closest-in-time candidate event within a configurable tolerance window. Unpaired
    events on either side are counted as misses / hallucinations.

    For every matched pair a per-field score is computed:
        action    — 1.0 if identical (case-insensitive), else 0.0
        player_id — 1.0 if identical, else 0.0

    Pair score = (action_match + player_id_match) / 2

    Similarity = mean(pair_scores) over ALL reference events
                 (unmatched reference events contribute 0 to the mean)

    Comparing the reference against itself always yields 100 %.

Exit codes: 0 success, 1 file/parse error.
"""

import json
import sys
import argparse
from pathlib import Path


DEFAULT_TOL = 2.0   # seconds — small deviations are acceptable


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _load(path: Path) -> list[dict]:
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array at the top level")
    return data


def _match(ref: list[dict], cand: list[dict], tol: float) -> list[tuple]:
    """
    Greedy nearest-neighbour matching within `tol` seconds.

    Each reference event is matched to the unmatched candidate event whose
    timestamp is closest, provided the distance is within the tolerance.
    Returns a list of (ref_event, cand_event | None) pairs covering every
    reference event exactly once.
    """
    available = list(enumerate(cand))   # (original_index, event)
    pairs: list[tuple] = []

    for ref_ev in ref:
        rt = ref_ev["timestamp_sec"]
        best_i, best_ev, best_dist = None, None, float("inf")

        for pos, (_, ev) in enumerate(available):
            dist = abs(ev["timestamp_sec"] - rt)
            if dist < best_dist:
                best_dist = dist
                best_i = pos
                best_ev = ev

        if best_i is not None and best_dist <= tol:
            pairs.append((ref_ev, best_ev))
            available.pop(best_i)
        else:
            pairs.append((ref_ev, None))

    return pairs


def _pair_score(ref_ev: dict, cand_ev: dict) -> float:
    """Return [0, 1] for a matched pair based on action + player_id."""
    action_ok = int(ref_ev["action"].lower() == cand_ev["action"].lower())
    player_ok = int(ref_ev["player_id"] == cand_ev["player_id"])
    return (action_ok + player_ok) / 2


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_event(ev: dict) -> str:
    return f"{ev['timestamp_sec']:5.1f}s  {ev['player_id']}  {ev['action']}"


def _symbol(ref_ev: dict, cand_ev: dict | None) -> str:
    if cand_ev is None:
        return "MISS  "
    action_ok = ref_ev["action"].lower() == cand_ev["action"].lower()
    player_ok = ref_ev["player_id"] == cand_ev["player_id"]
    if action_ok and player_ok:
        return "OK    "
    if action_ok or player_ok:
        return "PART  "
    return "WRONG "


def compare(ref_path: Path, cand_path: Path, tol: float) -> None:
    ref  = _load(ref_path)
    cand = _load(cand_path)

    pairs = _match(ref, cand, tol)

    # Candidate events not matched to any reference event
    matched_cands = {id(c) for _, c in pairs if c is not None}
    extra = [ev for ev in cand if id(ev) not in matched_cands]

    # Score
    scores = [_pair_score(r, c) if c is not None else 0.0 for r, c in pairs]
    similarity = (sum(scores) / len(scores) * 100) if scores else 0.0

    # Delta: positive → candidate has more events
    delta = len(cand) - len(ref)

    # -----------------------------------------------------------------------
    # Detailed output
    # -----------------------------------------------------------------------
    col_w = max(len(str(cand_path)), 50)
    print(f"\n{'─' * 72}")
    print(f"  Reference : {ref_path}")
    print(f"  Candidate : {cand_path}")
    print(f"  Tolerance : ±{tol}s")
    print(f"{'─' * 72}")

    print(f"\n{'  REF event':<32}  {'STATUS':<6}  {'CANDIDATE event'}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*30}")

    for ref_ev, cand_ev in pairs:
        sym = _symbol(ref_ev, cand_ev)
        cand_str = _fmt_event(cand_ev) if cand_ev is not None else "(no match)"
        ref_str  = _fmt_event(ref_ev)
        # highlight partial mismatches
        extras = ""
        if cand_ev is not None and sym not in ("OK    ",):
            parts = []
            if ref_ev["action"].lower() != cand_ev["action"].lower():
                parts.append(f"action: {cand_ev['action']!r}")
            if ref_ev["player_id"] != cand_ev["player_id"]:
                parts.append(f"player: {cand_ev['player_id']!r}")
            if parts:
                extras = f"  ← {', '.join(parts)}"
        print(f"  {ref_str:<30}  {sym}  {cand_str}{extras}")

    if extra:
        print(f"\n  EXTRA (candidate-only, {len(extra)} event(s)):")
        for ev in extra:
            print(f"    {_fmt_event(ev)}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    n_ok   = sum(1 for r, c in pairs if c is not None and _pair_score(r, c) == 1.0)
    n_part = sum(1 for r, c in pairs if c is not None and 0 < _pair_score(r, c) < 1.0)
    n_miss = sum(1 for _, c in pairs if c is None)
    n_wrong = sum(1 for r, c in pairs if c is not None and _pair_score(r, c) == 0.0)

    print(f"\n{'─' * 72}")
    print(f"  Reference events  : {len(ref):>4}")
    print(f"  Candidate events  : {len(cand):>4}")
    print(f"  Total delta       : {delta:>+4}  ({'more' if delta > 0 else 'fewer' if delta < 0 else 'same'} in candidate)")
    print()
    print(f"  Matched — correct : {n_ok:>4}  (action + player both right)")
    print(f"  Matched — partial : {n_part:>4}  (one of action/player wrong)")
    print(f"  Matched — wrong   : {n_wrong:>4}  (action + player both wrong)")
    print(f"  Missed            : {n_miss:>4}  (no candidate within ±{tol}s)")
    print(f"  Extra (halluc.)   : {len(extra):>4}  (candidate events with no ref match)")
    print()
    print(f"  ┌─────────────────────────────────┐")
    print(f"  │  Similarity score : {similarity:6.1f} %      │")
    print(f"  └─────────────────────────────────┘")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a candidate volleyball action JSON against the manual ground truth.",
    )
    parser.add_argument("candidate", type=Path, help="Path to the candidate JSON file")
    parser.add_argument(
        "--ref",
        type=Path,
        required=True,
        help="Path to the reference JSON file.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        metavar="SECS",
        help=f"Timestamp tolerance in seconds for matching (default: {DEFAULT_TOL})",
    )
    args = parser.parse_args()

    try:
        compare(args.ref, args.candidate, args.tol)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
