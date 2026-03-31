from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

PLAYER_IDS = ["P1", "P2", "P3", "P4"]
ALL_LABELS = [*PLAYER_IDS, None]
LABEL_NAMES = {"P1": "Denny", "P2": "O-Love", "P3": "Ibu 800", "P4": "Bjirk"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level JSON object")
    return data


def _iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    # bbox = [x1, y1, x2, y2]
    ix1 = max(bbox_a[0], bbox_b[0])
    iy1 = max(bbox_a[1], bbox_b[1])
    ix2 = min(bbox_a[2], bbox_b[2])
    iy2 = min(bbox_a[3], bbox_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, bbox_a[2] - bbox_a[0]) * max(0.0, bbox_a[3] - bbox_a[1])
    area_b = max(0.0, bbox_b[2] - bbox_b[0]) * max(0.0, bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _norm_player_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        upper = value.strip().upper()
        if upper == "NULL" or upper == "NONE" or upper == "":
            return None
        if upper in PLAYER_IDS:
            return upper
    return None


def _bbox_from_person(person: dict[str, Any]) -> list[float] | None:
    # Support both {x1, y1, x2, y2} dict format and {bbox: [x1,y1,x2,y2]} list format.
    if "bbox" in person:
        b = person["bbox"]
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
            except (TypeError, ValueError):
                pass
    keys = ("x1", "y1", "x2", "y2")
    if not all(k in person for k in keys):
        return None
    try:
        return [float(person[k]) for k in keys]
    except (TypeError, ValueError):
        return None


def _resolve_detection_frame(
    gt_ann: dict[str, Any],
    gt_source_frame: dict[str, Any] | None,
) -> dict[str, Any] | None:
    ann_persons = gt_ann.get("persons")
    if isinstance(ann_persons, list):
        return {"persons": ann_persons}
    return gt_source_frame


def _assignment_bbox(
    assignment: dict[str, Any],
    frame_for_lookup: dict[str, Any] | None,
) -> list[float] | None:
    # Prefer explicit bbox in assignment when present.
    direct = _bbox_from_person(assignment)
    if direct is not None:
        return direct

    if frame_for_lookup is None:
        return None

    idx_raw = assignment.get("detection_index")
    if not isinstance(idx_raw, int):
        return None

    persons = frame_for_lookup.get("persons")
    if not isinstance(persons, list) or idx_raw < 0 or idx_raw >= len(persons):
        return None

    person = persons[idx_raw]
    if not isinstance(person, dict):
        return None
    return _bbox_from_person(person)


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _match_detection_index(
    gt_bbox: list[float],
    identified_persons: list[dict[str, Any]],
    used_pred_indices: set[int],
) -> int | None:
    best_iou_idx: int | None = None
    best_iou = 0.5

    for i, person in enumerate(identified_persons):
        if i in used_pred_indices or not isinstance(person, dict):
            continue
        pred_bbox = _bbox_from_person(person)
        if pred_bbox is None:
            continue
        score = _iou(gt_bbox, pred_bbox)
        if score > best_iou:
            best_iou = score
            best_iou_idx = i

    if best_iou_idx is not None:
        return best_iou_idx

    gx, gy = _bbox_center(gt_bbox)
    best_dist_idx: int | None = None
    best_dist = 20.0
    for i, person in enumerate(identified_persons):
        if i in used_pred_indices or not isinstance(person, dict):
            continue
        pred_bbox = _bbox_from_person(person)
        if pred_bbox is None:
            continue
        px, py = _bbox_center(pred_bbox)
        dist = ((gx - px) ** 2 + (gy - py) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_dist_idx = i

    return best_dist_idx


def _label_index(label: str | None) -> int:
    return ALL_LABELS.index(label)


def evaluate_identification(
    identified_path: Path,
    gt_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    identified = _load_json(identified_path)
    ground_truth = _load_json(gt_path)

    identified_frames = identified.get("frames")
    if not isinstance(identified_frames, list):
        raise ValueError(f"{identified_path}: missing/invalid 'frames' list")
    id_by_frame: dict[int, dict[str, Any]] = {}
    for frame in identified_frames:
        if not isinstance(frame, dict):
            continue
        frame_no = frame.get("frame")
        if isinstance(frame_no, int):
            id_by_frame[frame_no] = frame

    gt_annotations = ground_truth.get("annotations")
    if not isinstance(gt_annotations, list):
        raise ValueError(f"{gt_path}: missing/invalid 'annotations' list")

    source_det_by_frame: dict[int, dict[str, Any]] = {}
    det_rel = ground_truth.get("detections")
    if isinstance(det_rel, str) and det_rel.strip():
        det_path = Path(det_rel)
        if not det_path.is_absolute():
            det_path = (gt_path.parent / det_path).resolve()
        if det_path.exists():
            det_json = _load_json(det_path)
            det_frames = det_json.get("frames")
            if isinstance(det_frames, list):
                for frame in det_frames:
                    if isinstance(frame, dict) and isinstance(frame.get("frame"), int):
                        source_det_by_frame[frame["frame"]] = frame

    warnings: list[str] = []
    confirmed_annotations = [
        ann for ann in gt_annotations if isinstance(ann, dict) and ann.get("confirmed", True)
    ]
    confirmed_annotations.sort(key=lambda a: int(a.get("frame", -1)))

    if not confirmed_annotations:
        metrics = {
            "accuracy": 0.0,
            "null_rate_gt": 0.0,
            "null_rate_pred": 0.0,
            "per_player": {
                pid: {"precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0}
                for pid in PLAYER_IDS
            },
            "swap_rate": 0.0,
            "total_swaps": 0,
            "confusion_matrix": [[0 for _ in ALL_LABELS] for _ in ALL_LABELS],
            "n_annotated_frames": 0,
            "n_annotated_detections": 0,
        }
        return metrics, warnings

    gt_labels: list[str | None] = []
    pred_labels: list[str | None] = []
    pred_by_frame_and_player: dict[int, dict[str, str | None]] = {}

    for ann in confirmed_annotations:
        frame_no = ann.get("frame")
        if not isinstance(frame_no, int):
            warnings.append("GT annotation missing valid integer 'frame'; skipping annotation")
            continue

        assignments = ann.get("assignments")
        if not isinstance(assignments, list):
            warnings.append(f"Frame {frame_no}: missing/invalid 'assignments'; skipping frame")
            continue

        assignments = [a for a in assignments if isinstance(a, dict)]
        assignments.sort(key=lambda a: int(a.get("detection_index", -1)))

        id_frame = id_by_frame.get(frame_no)
        if id_frame is None:
            warnings.append(f"Frame {frame_no}: not found in identified JSON; counting as unmatched")

        id_persons = id_frame.get("persons") if isinstance(id_frame, dict) else None
        if not isinstance(id_persons, list):
            id_persons = []

        used_pred_indices: set[int] = set()
        gt_source_frame = source_det_by_frame.get(frame_no)
        frame_lookup = _resolve_detection_frame(ann, gt_source_frame)

        per_player_pred: dict[str, str | None] = {}

        for assign in assignments:
            gt_pid = _norm_player_id(assign.get("player_id"))
            gt_labels.append(gt_pid)

            pred_pid: str | None = None
            if id_frame is not None:
                gt_bbox = _assignment_bbox(assign, frame_lookup)
                if gt_bbox is None:
                    warnings.append(
                        f"Frame {frame_no}: assignment missing bbox reference "
                        "(no bbox fields and no valid detection_index)"
                    )
                else:
                    matched_idx = _match_detection_index(gt_bbox, id_persons, used_pred_indices)
                    if matched_idx is None:
                        warnings.append(
                            f"Frame {frame_no}: no identified detection matched GT detection "
                            f"index {assign.get('detection_index')}"
                        )
                    else:
                        used_pred_indices.add(matched_idx)
                        pred_person = id_persons[matched_idx]
                        if isinstance(pred_person, dict):
                            pred_pid = _norm_player_id(pred_person.get("player_id"))

            pred_labels.append(pred_pid)

            if gt_pid in PLAYER_IDS:
                per_player_pred[gt_pid] = pred_pid

        pred_by_frame_and_player[frame_no] = per_player_pred

    total = len(gt_labels)
    correct = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == pred)
    null_gt = sum(1 for gt in gt_labels if gt is None)
    null_pred = sum(1 for pred in pred_labels if pred is None)

    confusion = [[0 for _ in ALL_LABELS] for _ in ALL_LABELS]
    for gt, pred in zip(gt_labels, pred_labels):
        confusion[_label_index(gt)][_label_index(pred)] += 1

    per_player: dict[str, dict[str, float | int]] = {}
    for pid in PLAYER_IDS:
        tp = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == pid and pred == pid)
        fp = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt != pid and pred == pid)
        fn = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == pid and pred != pid)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_player[pid] = {
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    frame_ids = sorted(pred_by_frame_and_player.keys())
    total_swaps = 0
    for prev_f, curr_f in zip(frame_ids, frame_ids[1:]):
        prev = pred_by_frame_and_player.get(prev_f, {})
        curr = pred_by_frame_and_player.get(curr_f, {})
        for pid in PLAYER_IDS:
            # Swap is only defined when GT has that player in both frames.
            if pid not in prev or pid not in curr:
                continue
            if prev[pid] != curr[pid]:
                total_swaps += 1

    n_pairs = max(0, len(frame_ids) - 1)
    swap_rate = total_swaps / (n_pairs * len(PLAYER_IDS)) if n_pairs > 0 else 0.0

    metrics = {
        "accuracy": (correct / total) if total > 0 else 0.0,
        "null_rate_gt": (null_gt / total) if total > 0 else 0.0,
        "null_rate_pred": (null_pred / total) if total > 0 else 0.0,
        "per_player": per_player,
        "swap_rate": swap_rate,
        "total_swaps": total_swaps,
        "confusion_matrix": confusion,
        "n_annotated_frames": len(frame_ids),
        "n_annotated_detections": total,
    }
    return metrics, warnings


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _print_metrics(metrics: dict[str, Any], gt_path: Path, identified_path: Path) -> None:
    n_frames = metrics["n_annotated_frames"]
    n_det = metrics["n_annotated_detections"]
    correct = int(round(metrics["accuracy"] * n_det)) if n_det > 0 else 0

    click.echo("=== Player Identification Evaluation ===")
    click.echo(f"Ground truth: {gt_path} ({n_frames} annotated frames, {n_det} detections)")
    click.echo(f"Identified:   {identified_path}")
    click.echo("")

    click.echo(f"Overall accuracy:  {_pct(metrics['accuracy'])}  ({correct}/{n_det} correct)")
    click.echo(f"Null rate (GT):    {_pct(metrics['null_rate_gt'])}")
    click.echo(f"Null rate (pred):  {_pct(metrics['null_rate_pred'])}")
    click.echo("")

    click.echo("Per-player accuracy:")
    per_player = metrics["per_player"]
    for pid in PLAYER_IDS:
        row = per_player[pid]
        click.echo(
            f"  {pid} ({LABEL_NAMES[pid]}): "
            f"precision={_pct(row['precision'])}  recall={_pct(row['recall'])}"
        )

    click.echo("")
    click.echo("Temporal stability:")
    click.echo(
        f"  Identity swaps: {metrics['total_swaps']}  "
        f"(swap rate: {_pct(metrics['swap_rate'])})"
    )
    click.echo("")

    click.echo("Confusion matrix (rows=GT, cols=predicted):")
    cols = ["P1", "P2", "P3", "P4", "null"]
    click.echo("        " + " ".join(f"{c:>4}" for c in cols))

    matrix = metrics["confusion_matrix"]
    for idx, row_name in enumerate(cols):
        row_vals = " ".join(f"{v:>4}" for v in matrix[idx])
        click.echo(f"  {row_name:<4}[ {row_vals} ]")


@click.command("eval-id")
@click.option(
    "--identified",
    "-i",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Identified JSON file (output of beach identify).",
)
@click.option(
    "--ground-truth",
    "-g",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Ground-truth annotation JSON (default: <identified_stem minus _identified suffix>_gt.json).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional output path for metrics JSON.",
)
def eval_id_cmd(identified: Path, ground_truth: Path | None, output: Path | None) -> None:
    """Evaluate identified player IDs against frame-level ground truth annotations."""
    # Derive ground-truth path: strip optional '_identified' suffix, append '_gt'.
    if ground_truth is None:
        base = identified.stem
        if base.endswith("_identified"):
            base = base[: -len("_identified")]
        ground_truth = identified.with_name(base + "_gt.json")
        if not ground_truth.exists():
            raise click.ClickException(
                f"Ground-truth file not found: {ground_truth}\n"
                "Run 'beach annotate-gt' first, or supply --ground-truth explicitly."
            )
    """Evaluate identified player IDs against frame-level ground truth annotations."""
    try:
        metrics, warnings = evaluate_identification(identified, ground_truth)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        raise click.ClickException(str(exc))

    for warning in warnings:
        click.echo(f"WARNING: {warning}", err=True)

    if metrics["n_annotated_frames"] == 0:
        click.echo("WARNING: no confirmed ground-truth frames found; nothing to score.", err=True)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            click.echo(f"Metrics JSON written to: {output}")
        return

    _print_metrics(metrics, ground_truth, identified)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        click.echo("")
        click.echo(f"Metrics JSON written to: {output}")
