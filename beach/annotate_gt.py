from __future__ import annotations

import base64
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np

PLAYER_LABELS: dict[Optional[str], str] = {
    "P1": "P1 (Denny)",
    "P2": "P2 (O-Love)",
    "P3": "P3 (Ibu 800)",
    "P4": "P4 (Bjirk)",
    None: "null",
}

PLAYER_COLORS_RGB: dict[Optional[str], tuple[int, int, int]] = {
    "P1": (200, 200, 200),
    "P2": (50, 200, 255),
    "P3": (50, 200, 50),
    "P4": (50, 255, 200),
    None: (120, 120, 120),
}


HTML_PAGE = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Beach GT Annotator</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 16px; color: #1f2937; }
    .row { display: flex; gap: 12px; align-items: center; margin-bottom: 10px; }
    .grow { flex: 1; }
    button { padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 6px; background: #fff; cursor: pointer; }
    button:hover { background: #f9fafb; }
    .counter { font-weight: 600; }
    .progress-wrap { width: 100%; height: 14px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }
    .progress-fill { height: 100%; width: 0%; background: #16a34a; transition: width .15s; }
    .layout { display: grid; grid-template-columns: minmax(420px, 1fr) 420px; gap: 16px; }
    img { max-width: 100%; border: 1px solid #d1d5db; border-radius: 6px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px; border-bottom: 1px solid #e5e7eb; text-align: left; }
    tr.focused { background: #eff6ff; }
    .swatch { width: 18px; height: 18px; border-radius: 3px; border: 1px solid #374151; display: inline-block; }
    .summary { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; margin-top: 10px; }
    .ok-banner { display: none; margin: 10px 0; padding: 10px; border-radius: 8px; background: #dcfce7; color: #166534; font-weight: 700; }
    .muted { color: #6b7280; font-size: 13px; }
  </style>
</head>
<body>
  <h2>Ground Truth Annotator</h2>

  <div id=\"allDone\" class=\"ok-banner\">All frames confirmed!</div>

  <div class=\"row\">
    <button id=\"prevBtn\">← Prev</button>
    <button id=\"nextBtn\">Next →</button>
    <button id=\"confirmBtn\">Confirm (Enter)</button>
    <div class=\"counter\" id=\"counter\">(0 / 0)</div>
    <div class=\"grow muted\" id=\"meta\"></div>
  </div>

  <div class=\"row\">
    <div class=\"progress-wrap grow\"><div id=\"progressFill\" class=\"progress-fill\"></div></div>
    <div id=\"progressText\" class=\"counter\"></div>
  </div>

  <div class=\"layout\">
    <div>
      <img id=\"frameImg\" alt=\"frame\" />
    </div>

    <div>
      <table>
        <thead>
          <tr><th></th><th>Detection</th><th>H-ID</th><th>Player</th></tr>
        </thead>
        <tbody id=\"detBody\"></tbody>
      </table>
      <div id=\"summary\" class=\"summary\"></div>
      <div class=\"muted\" style=\"margin-top:8px\">Shortcuts: Tab focus row, 1/2/3/4 assign player, n=null, Enter=confirm+next, ←/→ navigation.</div>
    </div>
  </div>

  <script>
    const state = { keyframes: [], gt: null, idx: 0, focusedDet: 0 };

    function frameCount() { return state.keyframes.length; }

    function annByFrame(frameIdx) {
      return state.gt.annotations.find(a => a.frame === frameIdx);
    }

    function clampIdx(i) {
      if (frameCount() === 0) return 0;
      return Math.max(0, Math.min(frameCount() - 1, i));
    }

    function updateProgress() {
      const total = frameCount();
      const confirmed = state.gt.annotations.filter(a => !!a.confirmed).length;
      const pct = total ? (confirmed / total) * 100 : 0;
      document.getElementById('progressFill').style.width = pct.toFixed(2) + '%';
      document.getElementById('progressText').textContent = `${confirmed} / ${total} confirmed`;
      document.getElementById('allDone').style.display = (total > 0 && confirmed === total) ? 'block' : 'none';
    }

    async function saveState() {
      await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.gt),
      });
      updateProgress();
    }

    function currentKeyframe() {
      return state.keyframes[state.idx];
    }

    function currentAnnotation() {
      const kf = currentKeyframe();
      return annByFrame(kf.frame_idx);
    }

    function renderSummary(ann) {
      const bits = ann.assignments.map(a => {
        const pid = a.player_id || 'null';
        return `det ${a.detection_index} (${a.human_track_id || 'H?'}) → ${pid}`;
      });
      const text = bits.length ? bits.join('; ') : 'No detections in this frame.';
      document.getElementById('summary').textContent = text;
    }

    function setAssignment(detIdx, value) {
      const ann = currentAnnotation();
      const target = ann.assignments.find(a => a.detection_index === detIdx);
      if (!target) return;
      target.player_id = value;
      renderFrame();
      saveState();
    }

    function nextUnconfirmedFrom(start) {
      for (let i = start; i < frameCount(); i++) {
        const ann = annByFrame(state.keyframes[i].frame_idx);
        if (ann && !ann.confirmed) return i;
      }
      return clampIdx(start);
    }

    function confirmAndAdvance() {
      const ann = currentAnnotation();
      ann.confirmed = true;
      const next = nextUnconfirmedFrom(state.idx + 1);
      state.idx = (next === state.idx && state.idx < frameCount() - 1) ? state.idx + 1 : next;
      renderFrame();
      saveState();
    }

    function renderFrame() {
      if (frameCount() === 0) return;

      state.idx = clampIdx(state.idx);
      state.focusedDet = 0;

      const kf = currentKeyframe();
      const ann = currentAnnotation();

      document.getElementById('counter').textContent = `(${state.idx + 1} / ${frameCount()})`;
      document.getElementById('meta').textContent = `frame ${kf.frame_idx} · t=${kf.timestamp_sec.toFixed(2)}s${ann.confirmed ? ' · confirmed' : ''}`;
      document.getElementById('frameImg').src = `data:image/jpeg;base64,${kf.image_b64}`;

      const body = document.getElementById('detBody');
      body.innerHTML = '';

      for (const det of kf.detections) {
        const tr = document.createElement('tr');
        if (det.detection_index === state.focusedDet) tr.classList.add('focused');

        const sw = document.createElement('td');
        const swatch = document.createElement('span');
        swatch.className = 'swatch';
        swatch.style.background = `rgb(${det.color_rgb[0]},${det.color_rgb[1]},${det.color_rgb[2]})`;
        sw.appendChild(swatch);

        const detTd = document.createElement('td');
        detTd.textContent = String(det.detection_index);

        const hid = document.createElement('td');
        hid.textContent = det.human_track_id || 'null';

        const selectTd = document.createElement('td');
        const sel = document.createElement('select');
        sel.dataset.detIdx = String(det.detection_index);
        for (const [value, label] of [["", "null"], ["P1", "P1 (Denny)"], ["P2", "P2 (O-Love)"], ["P3", "P3 (Ibu 800)"], ["P4", "P4 (Bjirk)"]]) {
          const opt = document.createElement('option');
          opt.value = value;
          opt.textContent = label;
          sel.appendChild(opt);
        }
        const assignment = ann.assignments.find(a => a.detection_index === det.detection_index);
        sel.value = assignment && assignment.player_id ? assignment.player_id : '';
        sel.addEventListener('focus', () => {
          state.focusedDet = det.detection_index;
          renderFrame();
        });
        sel.addEventListener('change', (e) => {
          const v = e.target.value || null;
          setAssignment(det.detection_index, v);
        });

        selectTd.appendChild(sel);
        tr.appendChild(sw);
        tr.appendChild(detTd);
        tr.appendChild(hid);
        tr.appendChild(selectTd);
        body.appendChild(tr);
      }

      renderSummary(ann);
      updateProgress();
    }

    function focusedSelect() {
      return document.querySelector(`select[data-det-idx=\"${state.focusedDet}\"]`);
    }

    function assignFocused(pidOrNull) {
      const sel = focusedSelect();
      if (!sel) return;
      sel.value = pidOrNull || '';
      setAssignment(state.focusedDet, pidOrNull);
    }

    function go(delta) {
      state.idx = clampIdx(state.idx + delta);
      renderFrame();
    }

    document.getElementById('prevBtn').addEventListener('click', () => go(-1));
    document.getElementById('nextBtn').addEventListener('click', () => go(1));
    document.getElementById('confirmBtn').addEventListener('click', () => confirmAndAdvance());

    document.addEventListener('keydown', (e) => {
      if (e.target && e.target.tagName === 'INPUT') return;
      if (e.key === 'ArrowLeft') { e.preventDefault(); go(-1); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); go(1); }
      else if (e.key === 'Enter') { e.preventDefault(); confirmAndAdvance(); }
      else if (e.key === '1') { e.preventDefault(); assignFocused('P1'); }
      else if (e.key === '2') { e.preventDefault(); assignFocused('P2'); }
      else if (e.key === '3') { e.preventDefault(); assignFocused('P3'); }
      else if (e.key === '4') { e.preventDefault(); assignFocused('P4'); }
      else if (e.key.toLowerCase() === 'n') { e.preventDefault(); assignFocused(null); }
    });

    async function init() {
      const resp = await fetch('/api/state');
      const payload = await resp.json();
      state.keyframes = payload.keyframes;
      state.gt = payload.gt;
      renderFrame();
    }

    init();
  </script>
</body>
</html>
"""


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise click.ClickException(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON in {path}: {exc}") from exc


def _distance_sq(ax: float, ay: float, bx: float, by: float) -> float:
    return (ax - bx) ** 2 + (ay - by) ** 2


def _trim_evenly(frame_indices: list[int], max_count: int) -> list[int]:
    if len(frame_indices) <= max_count:
        return frame_indices
    picks = np.linspace(0, len(frame_indices) - 1, num=max_count, dtype=int)
    return [frame_indices[int(i)] for i in picks]


def select_keyframes(
    frames: list[dict],
    identified_frames: list[dict],
    fps: float = 50.0,
    interval_sec: float = 2.0,
) -> list[dict]:
    if not frames:
        return []

    frames_sorted = sorted(frames, key=lambda f: int(f["frame"]))
    by_frame = {int(f["frame"]): f for f in frames_sorted}
    indices = [int(f["frame"]) for f in frames_sorted]

    max_idx = indices[-1]
    step = max(1, int(round(interval_sec * fps)))

    exact4 = [f for f in frames_sorted if len(f.get("persons", [])) == 4]
    exact4_indices = [int(f["frame"]) for f in exact4]

    chosen: set[int] = set()

    def nearest(target: int, candidates: list[int]) -> Optional[int]:
        if not candidates:
            return None
        return min(candidates, key=lambda x: (abs(x - target), x))

    for boundary in range(0, max_idx + 1, step):
        best = nearest(boundary, exact4_indices)
        if best is None:
            best = nearest(boundary, indices)
        if best is not None:
            chosen.add(best)

    identified_sorted = sorted(identified_frames, key=lambda f: int(f["frame"]))
    for i in range(1, len(identified_sorted)):
        prev = identified_sorted[i - 1]
        curr = identified_sorted[i]
        prev_ids = {
            p.get("human_track_id")
            for p in prev.get("persons", [])
            if p.get("human_track_id") is not None
        }
        curr_ids = {
            p.get("human_track_id")
            for p in curr.get("persons", [])
            if p.get("human_track_id") is not None
        }
        if curr_ids - prev_ids and len(prev.get("persons", [])) >= 2:
            prev_idx = int(prev["frame"])
            if prev_idx in by_frame:
                chosen.add(prev_idx)

    selected = sorted(chosen)
    selected = _trim_evenly(selected, max_count=60)
    return [by_frame[idx] for idx in selected if idx in by_frame]


def preseed_keyframes(keyframes: list[dict], identified_frames: list[dict]) -> list[dict]:
    id_by_frame = {int(f["frame"]): f for f in identified_frames}
    seeded: list[dict] = []

    for frame in keyframes:
        frame_idx = int(frame["frame"])
        id_frame = id_by_frame.get(frame_idx, {})
        id_persons = id_frame.get("persons", []) if isinstance(id_frame, dict) else []
        unmatched = set(range(len(id_persons)))

        detections: list[dict] = []
        for det_idx, person in enumerate(frame.get("persons", [])):
            px = float(person.get("cx", 0.0))
            py = float(person.get("cy", 0.0))

            best_j = None
            best_d = float("inf")
            for j in unmatched:
                ip = id_persons[j]
                d2 = _distance_sq(px, py, float(ip.get("cx", 0.0)), float(ip.get("cy", 0.0)))
                if d2 < best_d:
                    best_d = d2
                    best_j = j

            seed: Optional[str] = None
            if best_j is not None and best_d < 25.0:
                seed = id_persons[best_j].get("player_id")
                unmatched.remove(best_j)
            elif det_idx < len(id_persons):
                seed = id_persons[det_idx].get("player_id")

            seed = seed if seed in {"P1", "P2", "P3", "P4"} else None
            color = PLAYER_COLORS_RGB.get(seed, PLAYER_COLORS_RGB[None])
            detections.append(
                {
                    "detection_index": det_idx,
                    "human_track_id": person.get("human_track_id"),
                    "bbox": [
                        int(round(float(person.get("x1", 0.0)))),
                        int(round(float(person.get("y1", 0.0)))),
                        int(round(float(person.get("x2", 0.0)))),
                        int(round(float(person.get("y2", 0.0)))),
                    ],
                    "player_id_seed": seed,
                    "color_rgb": [int(color[0]), int(color[1]), int(color[2])],
                }
            )

        seeded.append(
            {
                "frame_idx": frame_idx,
                "timestamp_sec": float(frame.get("timestamp_sec", frame_idx / 50.0)),
                "detections": detections,
            }
        )

    return seeded


def extract_annotated_frame(cap: cv2.VideoCapture, frame_idx: int, detections: list[dict]) -> str:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise click.ClickException(f"Could not read frame {frame_idx} from video")

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        rgb = tuple(int(v) for v in det.get("color_rgb", [120, 120, 120]))
        color_bgr = (rgb[2], rgb[1], rgb[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 5)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 3)

        label = str(det["detection_index"])
        tx = x1 + 6
        ty = y1 + 28
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise click.ClickException(f"Failed to encode frame {frame_idx} as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _build_new_gt(video_path: Path, detections_path: Path, keyframes: list[dict]) -> dict:
    return {
        "video": str(video_path),
        "detections": str(detections_path),
        "created": datetime.now(timezone.utc).isoformat(),
        "annotations": [
            {
                "frame": kf["frame_idx"],
                "timestamp_sec": kf["timestamp_sec"],
                "confirmed": False,
                "assignments": [
                    {
                        "detection_index": det["detection_index"],
                        "human_track_id": det.get("human_track_id"),
                        "bbox": det["bbox"],
                        "player_id": det.get("player_id_seed"),
                    }
                    for det in kf["detections"]
                ],
            }
            for kf in keyframes
        ],
    }


def _merge_existing_gt(gt: dict, existing: dict) -> dict:
    existing_by_frame = {
        int(a.get("frame")): a
        for a in existing.get("annotations", [])
        if isinstance(a, dict) and "frame" in a
    }

    for ann in gt["annotations"]:
        old = existing_by_frame.get(int(ann["frame"]))
        if not old:
            continue

        ann["confirmed"] = bool(old.get("confirmed", False))
        old_assign = {
            int(a.get("detection_index")): a
            for a in old.get("assignments", [])
            if isinstance(a, dict) and "detection_index" in a
        }
        for assignment in ann["assignments"]:
            prev = old_assign.get(int(assignment["detection_index"]))
            if prev is None:
                continue
            pid = prev.get("player_id")
            assignment["player_id"] = pid if pid in {"P1", "P2", "P3", "P4"} else None

    gt["created"] = existing.get("created", gt["created"])
    return gt


def _save_gt(gt: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")


def _json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _make_handler(state: dict, output_path: Path):
    class AnnotationHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            if self.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/api/state":
                _json_response(self, {"keyframes": state["keyframes"], "gt": state["gt"]})
                return

            if self.path == "/api/exit":
                _json_response(self, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return

            _json_response(self, {"error": "Not found"}, status=404)

        def do_POST(self):
            if self.path != "/api/save":
                _json_response(self, {"error": "Not found"}, status=404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("Expected object")
            except (json.JSONDecodeError, ValueError) as exc:
                _json_response(self, {"error": str(exc)}, status=400)
                return

            state["gt"] = payload
            _save_gt(state["gt"], output_path)
            _json_response(self, {"ok": True})

    return AnnotationHandler


def run_annotation_server(state: dict, output_path: Path, port: int) -> None:
    server = HTTPServer(("", port), _make_handler(state, output_path))
    try:
        server.serve_forever()
    finally:
        server.server_close()


@click.command("annotate-gt")
@click.option(
    "--detections",
    "-d",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Pass-1 detections JSON (default: <video_stem>_detections.json).",
)
@click.option(
    "--identified",
    "-i",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Pass-2 identified JSON (default: <video_stem>_identified.json).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output ground truth JSON path (default: <video_stem>_gt.json).",
)
@click.option(
    "--video",
    "-v",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to source video.",
)
@click.option("--port", type=int, default=7780, show_default=True)
@click.option("--fps", type=float, default=50.0, show_default=True)
def annotate_gt_cmd(
    detections: Path | None,
    identified: Path | None,
    output: Optional[Path],
    video: Path,
    port: int,
    fps: float,
) -> None:
    detections_path = detections or video.with_name(video.stem + "_detections.json")
    identified_path = identified or video.with_name(video.stem + "_identified.json")

    for label, path in (("detections", detections_path), ("identified", identified_path)):
        if not path.exists():
            raise click.ClickException(
                f"{label.capitalize()} file not found: {path}\n"
                f"Run the appropriate pass first, or supply --{label} explicitly."
            )

    det_json = _load_json(detections_path)
    id_json = _load_json(identified_path)

    det_frames = det_json.get("frames")
    id_frames = id_json.get("frames")
    if not isinstance(det_frames, list) or not isinstance(id_frames, list):
        raise click.ClickException("Both detections and identified JSON must contain top-level 'frames' array")

    output_path = output if output is not None else video.with_name(video.stem + "_gt.json")

    selected = select_keyframes(det_frames, id_frames, fps=fps, interval_sec=2.0)
    keyframes = preseed_keyframes(selected, id_frames)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise click.ClickException(f"Cannot open video: {video}")
    try:
        for kf in keyframes:
            kf["image_b64"] = extract_annotated_frame(cap, kf["frame_idx"], kf["detections"])
    finally:
        cap.release()

    gt = _build_new_gt(video, detections, keyframes)
    if output_path.exists():
        existing = _load_json(output_path)
        if isinstance(existing, dict):
            gt = _merge_existing_gt(gt, existing)

    state = {"keyframes": keyframes, "gt": gt}

    print(f"Annotation server running at http://localhost:{port}")
    print(f"GT will be saved to {output_path}")
    print("Press Ctrl+C to stop.")

    try:
        run_annotation_server(state, output_path, port)
    except KeyboardInterrupt:
        pass
    finally:
        _save_gt(state["gt"], output_path)
