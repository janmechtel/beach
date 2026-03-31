"""Frame-level identification strategy evaluator.

Tests pluggable identification strategies on individual GT-annotated frames
without any video I/O or multi-frame tracking.  This makes strategy iteration
cheap: a full evaluation run takes seconds.

Strategies
----------
A  — team-side + colour (unsupervised)
     Splits detections by court side (left = Team A, right = Team B), then
     uses saturation/value within each team to distinguish teammates.

B  — colour-template matching (supervised upper bound)
     Hungarian assignment on distance to GT-derived HSV centroids.  Uses
     ground-truth-derived profiles; represents the ceiling of what colour alone
     can achieve.

C  — court-zone + colour blend
     Hungarian assignment on a weighted blend of positional zone distance and
     colour-template distance.  Combines spatial priors with colour.

Usage
-----
    beach eval-frame -v videos/output/GH021569_court_001.mp4
    beach eval-frame -v ... --strategy A
    beach eval-frame -v ... --verbose
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import click
import numpy as np
from scipy.optimize import linear_sum_assignment

from beach.eval_id import (
    PLAYER_IDS,
    ALL_LABELS,
    _iou,
    _bbox_from_person,
    _bbox_center,
    _norm_player_id,
    _match_detection_index,
)
from beach.identify import _color_distance

# ---------------------------------------------------------------------------
# Strategy type
# ---------------------------------------------------------------------------

@dataclass
class StrategyContext:
    """Immutable context passed to every strategy call."""
    # Player roster with team membership
    players: dict[str, dict]          # PLAYER_IDS → {"name", "team", ...}
    teams: dict[str, list[str]]       # team_id → [player_ids]
    # GT-derived HSV centroids; used by supervised strategies
    color_templates: dict[str, list[float]]  # pid → [H, S, V]
    # GT-derived x-zone centroids (mean cx per player across GT frames)
    zone_templates: dict[str, float]         # pid → mean cx
    # Court midpoint estimate (x-coordinate)
    court_midpoint: float


# Strategy function: (persons_in_frame, context) → {detection_index: player_id | None}
StrategyFn = Callable[[list[dict[str, Any]], StrategyContext], dict[int, str | None]]


# ---------------------------------------------------------------------------
# Strategy A — team-side + colour (unsupervised)
# ---------------------------------------------------------------------------

# Team assignments
_TEAM_A = ["P1", "P2"]  # left side
_TEAM_B = ["P3", "P4"]  # right side

# Within-team discrimination thresholds derived from GT profiles:
#   Team B: P3 (S≈114, blue) vs P4 (S≈50, dark tank). Saturation cuts them.
#   Team A: P2 (S≈28, V≈105, grey) vs P1 (S≈34, V≈95, dark shirt).
#           Saturation is very close; value is the better signal (P2 brighter).
_B_SAT_THRESHOLD = 80.0   # S >= threshold → P3 (blue); below → P4
_A_VAL_THRESHOLD = 100.0  # V >= threshold → P2 (lighter grey); below → P1


def strategy_a(
    persons: list[dict[str, Any]],
    ctx: StrategyContext,
) -> dict[int, str | None]:
    """Assign players using court-side split + within-team colour heuristic.

    No colour templates required; uses only intrinsic HSV values.
    Handles <4 detections by assigning what it can and leaving the rest None.
    """
    mid = ctx.court_midpoint
    left: list[int] = []
    right: list[int] = []

    for di, p in enumerate(persons):
        if p.get("cx", mid) < mid:
            left.append(di)
        else:
            right.append(di)

    result: dict[int, str | None] = {di: None for di in range(len(persons))}

    # --- Team B (right side) ---
    # P3 = high saturation (blue shirt), P4 = lower saturation (dark tank)
    if len(right) == 2:
        di_a, di_b = right
        s_a = persons[di_a].get("color_hsv", [0, 0, 0])[1]
        s_b = persons[di_b].get("color_hsv", [0, 0, 0])[1]
        # Higher saturation → P3
        if s_a >= s_b:
            result[di_a] = "P3"
            result[di_b] = "P4"
        else:
            result[di_b] = "P3"
            result[di_a] = "P4"
    elif len(right) == 1:
        di = right[0]
        s = persons[di].get("color_hsv", [0, 0, 0])[1]
        result[di] = "P3" if s >= _B_SAT_THRESHOLD else "P4"
    elif len(right) == 0 and len(left) >= 4:
        # All players on left — fall back to sorted-by-x + team split won't work;
        # we'll do our best with saturation alone for team B slots.
        sorted_left = sorted(left, key=lambda di: persons[di].get("cx", 0))
        # Rightmost two on-screen are Team B
        for di in sorted_left[-2:]:
            right.append(di)
            left.remove(di)
        if len(right) == 2:
            di_a, di_b = right
            s_a = persons[di_a].get("color_hsv", [0, 0, 0])[1]
            s_b = persons[di_b].get("color_hsv", [0, 0, 0])[1]
            if s_a >= s_b:
                result[di_a] = "P3"
                result[di_b] = "P4"
            else:
                result[di_b] = "P3"
                result[di_a] = "P4"

    # --- Team A (left side) ---
    # P2 = higher value (brighter grey), P1 = darker shirt
    if len(left) == 2:
        di_a, di_b = left
        v_a = persons[di_a].get("color_hsv", [0, 0, 0])[2]
        v_b = persons[di_b].get("color_hsv", [0, 0, 0])[2]
        # Higher value → P2
        if v_a >= v_b:
            result[di_a] = "P2"
            result[di_b] = "P1"
        else:
            result[di_b] = "P2"
            result[di_a] = "P1"
    elif len(left) == 1:
        di = left[0]
        v = persons[di].get("color_hsv", [0, 0, 0])[2]
        result[di] = "P2" if v >= _A_VAL_THRESHOLD else "P1"

    return result


# ---------------------------------------------------------------------------
# Strategy B — colour-template matching (supervised upper bound)
# ---------------------------------------------------------------------------

def strategy_b(
    persons: list[dict[str, Any]],
    ctx: StrategyContext,
) -> dict[int, str | None]:
    """Assign via Hungarian matching on HSV distance to GT-derived colour templates.

    This is a supervised upper bound: it uses ground-truth-derived profiles,
    so it answers "what is the ceiling of colour-only identification?"

    Players with no color_hsv in detections are left unassigned (None).
    """
    if not persons:
        return {}

    n_det = len(persons)
    n_pid = len(PLAYER_IDS)
    cost = np.full((n_det, n_pid), 1e9, dtype=float)

    for di, p in enumerate(persons):
        hsv = p.get("color_hsv")
        if not hsv:
            continue
        for pi, pid in enumerate(PLAYER_IDS):
            tmpl = ctx.color_templates.get(pid)
            if tmpl:
                cost[di, pi] = _color_distance(hsv, tmpl)

    row_ind, col_ind = linear_sum_assignment(cost)
    result: dict[int, str | None] = {di: None for di in range(n_det)}
    for ri, ci in zip(row_ind, col_ind):
        if cost[ri, ci] < 1e8:
            result[ri] = PLAYER_IDS[ci]
    return result


# ---------------------------------------------------------------------------
# Strategy C — court-zone + colour blend
# ---------------------------------------------------------------------------

# Weight for colour component in the blended cost.  Position is primary signal
# since zones are well-separated; colour breaks ties when players overlap.
_ZONE_COLOR_WEIGHT = 0.40   # 40% colour, 60% zone distance


def strategy_c(
    persons: list[dict[str, Any]],
    ctx: StrategyContext,
) -> dict[int, str | None]:
    """Hungarian matching on blended zone-distance + colour-template cost.

    Position zones from GT (mean cx): P2≈350, P1≈500, P4≈950, P3≈1050.
    Cost = (1 - w) * zone_cost + w * colour_cost.

    Zone cost is normalised by half-frame-width (750 px) → [0,1].
    """
    if not persons:
        return {}

    n_det = len(persons)
    n_pid = len(PLAYER_IDS)
    ZONE_SCALE = 750.0  # half frame width

    cost = np.full((n_det, n_pid), 1e9, dtype=float)

    for di, p in enumerate(persons):
        cx = p.get("cx", 0.0)
        hsv = p.get("color_hsv")
        for pi, pid in enumerate(PLAYER_IDS):
            zone_cx = ctx.zone_templates.get(pid)
            if zone_cx is None:
                continue
            zone_cost = min(abs(cx - zone_cx) / ZONE_SCALE, 1.0)

            if hsv and pid in ctx.color_templates:
                c_cost = _color_distance(hsv, ctx.color_templates[pid])
                cost[di, pi] = (1.0 - _ZONE_COLOR_WEIGHT) * zone_cost + _ZONE_COLOR_WEIGHT * c_cost
            else:
                cost[di, pi] = zone_cost

    row_ind, col_ind = linear_sum_assignment(cost)
    result: dict[int, str | None] = {di: None for di in range(n_det)}
    for ri, ci in zip(row_ind, col_ind):
        if cost[ri, ci] < 1e8:
            result[ri] = PLAYER_IDS[ci]
    return result


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, tuple[StrategyFn, str]] = {
    "A": (strategy_a, "Team-side + colour (unsupervised)"),
    "B": (strategy_b, "Colour-template matching (supervised upper bound)"),
    "C": (strategy_c, "Court-zone + colour blend"),
}


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(
    gt_annotations: list[dict[str, Any]],
    det_by_frame: dict[int, dict[str, Any]],
) -> StrategyContext:
    """Derive StrategyContext from confirmed GT annotations + their detections.

    Colour templates: mean HSV per player across all GT-confirmed assignments
    where the referenced detection has a color_hsv field.

    Zone templates: mean cx per player across the same set.

    Court midpoint: mean of all player-detection cx values (or 750 if no data).
    """
    from beach.identify import PLAYERS

    pid_hsv: dict[str, list[list[float]]] = {pid: [] for pid in PLAYER_IDS}
    pid_cx: dict[str, list[float]] = {pid: [] for pid in PLAYER_IDS}
    all_cx: list[float] = []

    confirmed = [a for a in gt_annotations if a.get("confirmed", True)]

    for ann in confirmed:
        frame_no = ann.get("frame")
        det_frame = det_by_frame.get(frame_no)
        persons: list[dict] | None = None
        if det_frame:
            persons = det_frame.get("persons")

        for assign in ann.get("assignments", []):
            pid = _norm_player_id(assign.get("player_id"))
            if pid not in PLAYER_IDS:
                continue
            di = assign.get("detection_index")
            if not isinstance(di, int) or persons is None:
                continue
            if di < 0 or di >= len(persons):
                continue
            p = persons[di]
            hsv = p.get("color_hsv")
            cx = p.get("cx")
            if hsv:
                pid_hsv[pid].append(hsv)
            if cx is not None:
                pid_cx[pid].append(float(cx))
                all_cx.append(float(cx))

    color_templates: dict[str, list[float]] = {}
    for pid in PLAYER_IDS:
        if pid_hsv[pid]:
            color_templates[pid] = list(np.mean(pid_hsv[pid], axis=0))
        else:
            # GT-derived fallback from plan if not in annotations
            _FALLBACK = {
                "P1": [85.0, 34.0, 95.0],
                "P2": [78.0, 28.0, 105.0],
                "P3": [100.0, 114.0, 118.0],
                "P4": [68.0, 50.0, 88.0],
            }
            color_templates[pid] = _FALLBACK[pid]

    zone_templates: dict[str, float] = {}
    for pid in PLAYER_IDS:
        if pid_cx[pid]:
            zone_templates[pid] = float(np.mean(pid_cx[pid]))
        else:
            _FALLBACK_ZONE = {"P1": 500.0, "P2": 350.0, "P3": 1050.0, "P4": 950.0}
            zone_templates[pid] = _FALLBACK_ZONE[pid]

    court_midpoint = float(np.mean(all_cx)) if all_cx else 750.0

    teams = {"A": list(_TEAM_A), "B": list(_TEAM_B)}

    return StrategyContext(
        players=PLAYERS,
        teams=teams,
        color_templates=color_templates,
        zone_templates=zone_templates,
        court_midpoint=court_midpoint,
    )


# ---------------------------------------------------------------------------
# Per-frame evaluation
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    frame_no: int
    n_gt: int          # number of GT assignments in this frame
    n_correct: int     # assignments the strategy got right
    assignments: list[tuple[str | None, str | None]]  # (gt_pid, pred_pid)


def _evaluate_frame(
    ann: dict[str, Any],
    det_frame: dict[str, Any] | None,
    strategy_fn: StrategyFn,
    ctx: StrategyContext,
) -> FrameResult:
    """Run strategy on one GT-annotated frame and score against GT."""
    frame_no = ann["frame"]
    gt_assignments = [a for a in ann.get("assignments", []) if isinstance(a, dict)]

    # Get detection persons for this frame
    persons: list[dict] = []
    if det_frame:
        persons = det_frame.get("persons") or []

    # Run strategy
    pred_map = strategy_fn(persons, ctx)  # detection_index → player_id | None

    # Build a list of identified persons with player_id for matching
    id_persons = [
        {**p, "player_id": pred_map.get(di)}
        for di, p in enumerate(persons)
    ]

    # Match GT assignments to identified detections via IoU / centroid distance
    used_pred: set[int] = set()
    pair_list: list[tuple[str | None, str | None]] = []

    for assign in sorted(gt_assignments, key=lambda a: a.get("detection_index", -1)):
        gt_pid = _norm_player_id(assign.get("player_id"))

        # Resolve the GT bbox — prefer explicit bbox in assignment, fall back to
        # looking up the detection by index in the detections JSON.
        gt_bbox: list[float] | None = _bbox_from_person(assign)
        if gt_bbox is None and det_frame:
            di_raw = assign.get("detection_index")
            if isinstance(di_raw, int) and 0 <= di_raw < len(persons):
                gt_bbox = _bbox_from_person(persons[di_raw])

        if gt_bbox is None:
            pair_list.append((gt_pid, None))
            continue

        matched_idx = _match_detection_index(gt_bbox, id_persons, used_pred)
        if matched_idx is None:
            pair_list.append((gt_pid, None))
            continue

        used_pred.add(matched_idx)
        pred_pid = _norm_player_id(id_persons[matched_idx].get("player_id"))
        pair_list.append((gt_pid, pred_pid))

    n_correct = sum(1 for gt, pred in pair_list if gt == pred and gt is not None)
    n_gt = len(pair_list)

    return FrameResult(
        frame_no=frame_no,
        n_gt=n_gt,
        n_correct=n_correct,
        assignments=pair_list,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _aggregate(results: list[FrameResult]) -> dict[str, Any]:
    all_pairs = [pair for r in results for pair in r.assignments]
    total = len(all_pairs)
    correct = sum(1 for gt, pred in all_pairs if gt == pred and gt is not None)
    null_pred = sum(1 for _, pred in all_pairs if pred is None)

    confusion: dict[str | None, dict[str | None, int]] = {
        pid: {p: 0 for p in ALL_LABELS} for pid in ALL_LABELS
    }
    for gt, pred in all_pairs:
        if gt in ALL_LABELS and pred in ALL_LABELS:
            confusion[gt][pred] += 1

    per_player: dict[str, dict[str, Any]] = {}
    for pid in PLAYER_IDS:
        tp = sum(1 for gt, pred in all_pairs if gt == pid and pred == pid)
        fp = sum(1 for gt, pred in all_pairs if gt != pid and pred == pid)
        fn = sum(1 for gt, pred in all_pairs if gt == pid and pred != pid)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_player[pid] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "null_pred_rate": null_pred / total if total > 0 else 0.0,
        "per_player": per_player,
        "confusion": confusion,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_confusion(confusion: dict, label: str) -> None:
    cols = [*PLAYER_IDS, None]
    col_names = [*PLAYER_IDS, "null"]
    click.echo(f"  {label} (rows=GT, cols=pred):")
    click.echo("         " + " ".join(f"{c:>5}" for c in col_names))
    for row_pid in cols:
        row_name = row_pid if row_pid else "null"
        row_vals = " ".join(f"{confusion[row_pid][c]:>5}" for c in cols)
        click.echo(f"    {row_name:<4} [ {row_vals} ]")


@click.command("eval-frame")
@click.option(
    "--video", "-v",
    required=False,
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Source video (used only to derive GT path; no video I/O performed).",
)
@click.option(
    "--ground-truth", "-g",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="GT annotation JSON (default: <video_stem>_gt.json).",
)
@click.option(
    "--strategy", "-s",
    default="ALL",
    show_default=True,
    type=click.Choice(["A", "B", "C", "ALL"], case_sensitive=False),
    help="Strategy to evaluate (ALL runs all three).",
)
@click.option(
    "--verbose", is_flag=True, default=False,
    help="Print per-frame results for each strategy.",
)
def eval_frame_cmd(
    video: Path | None,
    ground_truth: Path | None,
    strategy: str,
    verbose: bool,
) -> None:
    """Evaluate frame-level identification strategies against ground truth.

    Runs pluggable single-frame strategies on every confirmed GT annotation
    and reports per-strategy accuracy, per-player precision/recall, and a
    confusion matrix.  No video I/O; completes in seconds.
    """
    # Resolve GT path
    if ground_truth is None:
        if video is None:
            raise click.UsageError("Provide --video or --ground-truth.")
        ground_truth = video.with_name(video.stem + "_gt.json")
        if not ground_truth.exists():
            raise click.ClickException(
                f"Ground-truth file not found: {ground_truth}\n"
                "Run 'beach annotate-gt' first, or supply --ground-truth explicitly."
            )

    gt_data = json.loads(ground_truth.read_text())
    gt_annotations: list[dict] = gt_data.get("annotations", [])
    confirmed = [a for a in gt_annotations if a.get("confirmed", True)]
    if not confirmed:
        raise click.ClickException("No confirmed annotations found in GT file.")

    # Load detections
    det_rel = gt_data.get("detections", "")
    det_by_frame: dict[int, dict] = {}
    if det_rel:
        det_path = Path(det_rel)
        if not det_path.is_absolute():
            # The path stored in GT may be relative to cwd (project root) or to the
            # GT file's directory.  Try cwd first; fall back to GT-parent.
            cwd_candidate = Path.cwd() / det_path
            parent_candidate = (ground_truth.parent / det_path).resolve()
            if cwd_candidate.exists():
                det_path = cwd_candidate
            elif parent_candidate.exists():
                det_path = parent_candidate
        if det_path.exists():
            det_data = json.loads(det_path.read_text())
            for f in det_data.get("frames", []):
                if isinstance(f, dict) and isinstance(f.get("frame"), int):
                    det_by_frame[f["frame"]] = f
        else:
            raise click.ClickException(f"Detections file not found: {det_path}")

    click.echo(f"GT: {ground_truth}  ({len(confirmed)} confirmed annotations)")
    click.echo(f"Detections: {det_path}  ({len(det_by_frame)} frames)")

    # Build context from GT annotations (uses ALL confirmed annotations for templates)
    ctx = _build_context(confirmed, det_by_frame)

    click.echo(f"\nContext:")
    click.echo(f"  Court midpoint:  {ctx.court_midpoint:.0f} px")
    click.echo("  Colour templates (H, S, V):")
    for pid in PLAYER_IDS:
        tmpl = ctx.color_templates.get(pid, [])
        click.echo(f"    {pid}: H={tmpl[0]:.0f} S={tmpl[1]:.0f} V={tmpl[2]:.0f}")
    click.echo("  Zone templates (mean cx):")
    for pid in PLAYER_IDS:
        click.echo(f"    {pid}: {ctx.zone_templates.get(pid, 0):.0f} px")

    # Which strategies to run
    strats_to_run = list(STRATEGIES.keys()) if strategy.upper() == "ALL" else [strategy.upper()]

    click.echo("")

    for key in strats_to_run:
        fn, desc = STRATEGIES[key]
        click.echo(f"=== Strategy {key}: {desc} ===")

        results: list[FrameResult] = []
        for ann in sorted(confirmed, key=lambda a: a.get("frame", -1)):
            frame_no = ann.get("frame")
            det_frame = det_by_frame.get(frame_no) if isinstance(frame_no, int) else None
            result = _evaluate_frame(ann, det_frame, fn, ctx)
            results.append(result)

            if verbose:
                pairs_str = "  ".join(
                    f"{gt}→{'✓' if gt == pred else pred}"
                    for gt, pred in result.assignments
                )
                correct_marker = "✓" if result.n_correct == result.n_gt else f"{result.n_correct}/{result.n_gt}"
                click.echo(
                    f"  frame {frame_no:>5}  [{correct_marker:>3}]  {pairs_str}"
                )

        metrics = _aggregate(results)
        click.echo(
            f"\n  Accuracy: {metrics['accuracy']*100:.1f}%  "
            f"({metrics['correct']}/{metrics['total']} correct,  "
            f"null_pred={metrics['null_pred_rate']*100:.1f}%)"
        )
        click.echo("  Per-player:")
        for pid in PLAYER_IDS:
            from beach.eval_id import LABEL_NAMES
            r = metrics["per_player"][pid]
            click.echo(
                f"    {pid} ({LABEL_NAMES[pid]}): "
                f"precision={r['precision']*100:.1f}%  recall={r['recall']*100:.1f}%  "
                f"tp={r['tp']} fp={r['fp']} fn={r['fn']}"
            )

        _print_confusion(metrics["confusion"], "Confusion")
        click.echo("")
