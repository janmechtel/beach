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

from beach.paths import identified_path, identified_suffix

PLAYER_IDS: tuple[str, ...] = ("P1", "P2", "P3", "P4")
DEFAULT_PLAYER_NAMES: dict[str, str] = {
    "P1": "Denny",
    "P2": "O-Love",
    "P3": "Ibu 800",
    "P4": "Bjirk",
}
DEFAULT_PLAYER_COLORS_HEX: dict[str, str] = {
    "P1": "#000000",
    "P2": "#94a3b8",
    "P3": "#3b82f6",
    "P4": "#22c55e",
}
NULL_PLAYER_COLOR_RGB: tuple[int, int, int] = (120, 120, 120)
DEFAULT_PLAYERS_PATH = Path("output/players.json")


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
    .frame-wrap { position: relative; display: inline-block; width: 100%; border: 1px solid #d1d5db; border-radius: 6px; overflow: hidden; background: #0f172a; }
    #frameImg { width: 100%; display: block; }
    #frameOverlay { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px; border-bottom: 1px solid #e5e7eb; text-align: left; }
    tr.focused { background: #eff6ff; }
    tr:hover { background: #f8fafc; }
    .swatch { width: 18px; height: 18px; border-radius: 3px; border: 1px solid #374151; display: inline-block; }
    select { width: 100%; padding: 4px; }
    .summary { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; margin-top: 10px; }
    .ok-banner { display: none; margin: 10px 0; padding: 10px; border-radius: 8px; background: #dcfce7; color: #166534; font-weight: 700; }
    .muted { color: #6b7280; font-size: 13px; }
  </style>
</head>
<body>
  <h2>Ground Truth Annotator</h2>

  <div id=\"allDone\" class=\"ok-banner\">All frames confirmed!</div>

  <div class=\"row\">
    <button id=\"prevBtn\">← Prev frame</button>
    <button id=\"nextBtn\">Next frame →</button>
    <button id=\"confirmBtn\">Confirm frame</button>
    <div class=\"counter\" id=\"counter\">(0 / 0)</div>
    <div class=\"grow muted\" id=\"meta\"></div>
  </div>

  <div class=\"row\">
    <div class=\"progress-wrap grow\"><div id=\"progressFill\" class=\"progress-fill\"></div></div>
    <div id=\"progressText\" class=\"counter\"></div>
  </div>

  <div class=\"layout\">
    <div>
      <div class=\"frame-wrap\">
        <img id=\"frameImg\" alt=\"frame\" />
        <svg id=\"frameOverlay\"></svg>
      </div>
    </div>

    <div>
      <table>
        <thead>
          <tr><th></th><th>Player</th><th>H-ID</th><th>Manual pos</th></tr>
        </thead>
        <tbody id=\"playerBody\"></tbody>
      </table>
      <div id=\"summary\" class=\"summary\"></div>
      <div class="muted" style="margin-top:8px">Shortcuts: Tab next row, Shift+Tab previous row, Enter confirm frame + next, PageUp/PageDown prev/next frame, ←/→ cycle H-ID for focused row (assigned IDs are skipped for other players), 1/2/3/4 jump to P1..P4, n jump to null row, click frame to set manual position for focused player, Backspace/Delete clears focused manual position.</div>
    </div>
  </div>

  <script>
    const PLAYERS = __PLAYERS_JSON__;
    const PLAYER_INDEX_BY_KEY = new Map(PLAYERS.map((p, idx) => [String(p.key), idx]));

    const state = {
      keyframes: [],
      gt: null,
      idx: 0,
      focusedPlayerIdx: 0,
      selectedNullHid: null,
    };

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

    function assignmentForPlayer(playerKey) {
      if (playerKey === null) return null;
      const ann = currentAnnotation();
      return ann.assignments.find(a => a.player_id === playerKey) || null;
    }

    function ensureManualPositions(ann) {
      if (!ann.manual_positions || typeof ann.manual_positions !== 'object') {
        ann.manual_positions = {};
      }
      return ann.manual_positions;
    }

    function manualPositionForPlayer(playerKey) {
      if (playerKey === null) return null;
      const ann = currentAnnotation();
      const manual = ensureManualPositions(ann);
      const raw = manual[playerKey];
      if (!raw || typeof raw !== 'object') return null;
      const x = Number(raw.x);
      const y = Number(raw.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
      return { x, y };
    }

    function allHids() {
      const ann = currentAnnotation();
      const seen = new Set();
      const hids = [];
      for (const assignment of ann.assignments) {
        const hid = assignment.human_track_id;
        if (hid === null || hid === undefined) continue;
        const key = String(hid);
        if (seen.has(key)) continue;
        seen.add(key);
        hids.push(key);
      }
      return hids;
    }

    function unassignedHids() {
      const ann = currentAnnotation();
      return ann.assignments
        .filter(a => a.player_id === null && a.human_track_id !== null && a.human_track_id !== undefined)
        .map(a => String(a.human_track_id));
    }

    function availableHidsForPlayer(playerKey) {
      if (playerKey === null) return unassignedHids();
      const ann = currentAnnotation();
      const myIdx = PLAYERS.findIndex(p => p.key === playerKey);
      // HIDs already taken by players earlier in order (P1 before P2, etc.) are off-limits.
      const takenByPrior = new Set(
        ann.assignments
          .filter(a => {
            if (a.player_id === null || a.player_id === undefined) return false;
            if (a.human_track_id === null || a.human_track_id === undefined) return false;
            const otherIdx = PLAYERS.findIndex(p => p.key === a.player_id);
            return otherIdx !== -1 && otherIdx < myIdx;
          })
          .map(a => String(a.human_track_id))
      );
      return allHids().filter(hid => !takenByPrior.has(String(hid)));
    }

    function setPlayerAssignment(playerKey, hidOrNull) {
      const ann = currentAnnotation();

      if (playerKey === null) {
        state.selectedNullHid = hidOrNull;
        if (hidOrNull !== null) {
          const target = ann.assignments.find(a => String(a.human_track_id) === String(hidOrNull));
          if (target) target.player_id = null;
        }
        renderFrame();
        saveState();
        return;
      }

      // Clearing a player row explicitly removes its assignment.
      if (hidOrNull === null) {
        for (const a of ann.assignments) {
          if (a.player_id === playerKey) a.player_id = null;
        }
        renderFrame();
        saveState();
        const focused = focusedPlayerSelect();
        if (focused) focused.focus({ preventScroll: true });
        return;
      }

      const target = ann.assignments.find(a => String(a.human_track_id) === String(hidOrNull));
      if (!target) return;

      // One player maps to at most one H-ID: clear any old holder of this player.
      for (const a of ann.assignments) {
        if (a.player_id === playerKey) a.player_id = null;
      }
      target.player_id = playerKey;

      state.selectedNullHid = null;
      renderFrame();
      saveState();
      const focused = focusedPlayerSelect();
      if (focused) focused.focus({ preventScroll: true });
    }

    function setFocusedPlayerManualPosition(x, y) {
      const player = focusedPlayer();
      if (!player || player.key === null) return;
      const ann = currentAnnotation();
      const manual = ensureManualPositions(ann);
      manual[player.key] = { x: Math.round(x), y: Math.round(y) };
      renderFrame();
      saveState();
      const focused = focusedPlayerSelect();
      if (focused) focused.focus({ preventScroll: true });
    }

    function clearManualPosition(playerKey) {
      if (playerKey === null) return;
      const ann = currentAnnotation();
      const manual = ensureManualPositions(ann);
      if (!(playerKey in manual)) return;
      delete manual[playerKey];
      renderFrame();
      saveState();
      const focused = focusedPlayerSelect();
      if (focused) focused.focus({ preventScroll: true });
    }

    function clearFocusedManualPosition() {
      const player = focusedPlayer();
      if (!player || player.key === null) return;
      clearManualPosition(player.key);
    }

    function renderSummary() {
      const ann = currentAnnotation();
      const manual = ensureManualPositions(ann);
      const parts = [];
      for (const p of PLAYERS) {
        if (p.key === null) {
          const free = unassignedHids();
          parts.push(`null → [${free.length ? free.join(', ') : 'none'}]`);
          continue;
        }
        const own = assignmentForPlayer(p.key);
        if (own && own.human_track_id) {
          parts.push(`${p.key} → ${own.human_track_id}`);
          continue;
        }
        const pos = manual[p.key];
        if (pos && Number.isFinite(Number(pos.x)) && Number.isFinite(Number(pos.y))) {
          parts.push(`${p.key} → manual(${Math.round(Number(pos.x))}, ${Math.round(Number(pos.y))})`);
        } else {
          parts.push(`${p.key} → none`);
        }
      }
      document.getElementById('summary').textContent = parts.join(' · ');
    }

    function overlayColor(playerId) {
      const p = PLAYERS.find(x => x.key === playerId) || PLAYERS[PLAYERS.length - 1];
      return `rgb(${p.color[0]}, ${p.color[1]}, ${p.color[2]})`;
    }

    function renderOverlay() {
      const svg = document.getElementById('frameOverlay');
      const img = document.getElementById('frameImg');
      const kf = currentKeyframe();
      const ann = currentAnnotation();
      if (!kf || !ann) return;
      if (!img.naturalWidth || !img.naturalHeight) return;

      svg.setAttribute('viewBox', `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
      svg.innerHTML = '';

      const byDetIdx = new Map(ann.assignments.map(a => [a.detection_index, a]));
      for (const det of kf.detections) {
        const assign = byDetIdx.get(det.detection_index);
        const pid = assign ? assign.player_id : null;
        const [x1, y1, x2, y2] = det.bbox;
        const w = Math.max(1, x2 - x1);
        const h = Math.max(1, y2 - y1);
        const color = overlayColor(pid);

        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', String(x1));
        rect.setAttribute('y', String(y1));
        rect.setAttribute('width', String(w));
        rect.setAttribute('height', String(h));
        rect.setAttribute('fill', 'none');
        rect.setAttribute('stroke', color);
        rect.setAttribute('stroke-width', '3');
        svg.appendChild(rect);

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', String(x1 + 4));
        text.setAttribute('y', String(Math.max(16, y1 + 18)));
        text.setAttribute('font-size', '18');
        text.setAttribute('font-weight', '700');
        text.setAttribute('stroke', '#000');
        text.setAttribute('stroke-width', '3');
        text.setAttribute('paint-order', 'stroke');
        text.setAttribute('fill', '#fff');
        text.textContent = det.human_track_id || `det ${det.detection_index}`;
        svg.appendChild(text);
      }

      for (const player of PLAYERS) {
        if (player.key === null) continue;
        const own = assignmentForPlayer(player.key);
        if (own) continue;
        const manualPos = manualPositionForPlayer(player.key);
        if (!manualPos) continue;
        const color = overlayColor(player.key);

        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        marker.setAttribute('cx', String(manualPos.x));
        marker.setAttribute('cy', String(manualPos.y));
        marker.setAttribute('r', '10');
        marker.setAttribute('fill', 'none');
        marker.setAttribute('stroke', color);
        marker.setAttribute('stroke-width', '3');
        svg.appendChild(marker);

        const hLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        hLine.setAttribute('x1', String(manualPos.x - 14));
        hLine.setAttribute('y1', String(manualPos.y));
        hLine.setAttribute('x2', String(manualPos.x + 14));
        hLine.setAttribute('y2', String(manualPos.y));
        hLine.setAttribute('stroke', color);
        hLine.setAttribute('stroke-width', '3');
        svg.appendChild(hLine);

        const vLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        vLine.setAttribute('x1', String(manualPos.x));
        vLine.setAttribute('y1', String(manualPos.y - 14));
        vLine.setAttribute('x2', String(manualPos.x));
        vLine.setAttribute('y2', String(manualPos.y + 14));
        vLine.setAttribute('stroke', color);
        vLine.setAttribute('stroke-width', '3');
        svg.appendChild(vLine);

        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', String(manualPos.x + 12));
        label.setAttribute('y', String(Math.max(16, manualPos.y - 12)));
        label.setAttribute('font-size', '16');
        label.setAttribute('font-weight', '700');
        label.setAttribute('stroke', '#000');
        label.setAttribute('stroke-width', '3');
        label.setAttribute('paint-order', 'stroke');
        label.setAttribute('fill', '#fff');
        label.textContent = `${player.key} manual`;
        svg.appendChild(label);
      }
    }

    function updateFocusHighlight() {
      document.querySelectorAll('#playerBody tr').forEach((row, idx) => {
        row.classList.toggle('focused', idx === state.focusedPlayerIdx);
      });
    }

    function renderPlayerRows() {
      const body = document.getElementById('playerBody');
      body.innerHTML = '';

      for (let idx = 0; idx < PLAYERS.length; idx++) {
        const player = PLAYERS[idx];
        const tr = document.createElement('tr');
        tr.dataset.playerKey = String(player.key);
        if (idx === state.focusedPlayerIdx) tr.classList.add('focused');

        const sw = document.createElement('td');
        const swatch = document.createElement('span');
        swatch.className = 'swatch';
        swatch.style.background = `rgb(${player.color[0]},${player.color[1]},${player.color[2]})`;
        sw.appendChild(swatch);

        const playerTd = document.createElement('td');
        playerTd.textContent = player.label;

        const hidTd = document.createElement('td');
        const sel = document.createElement('select');
        sel.dataset.playerKey = String(player.key);

        const empty = document.createElement('option');
        empty.value = '';
        empty.textContent = '—';
        sel.appendChild(empty);

        const options = availableHidsForPlayer(player.key);
        for (const hid of options) {
          const opt = document.createElement('option');
          opt.value = String(hid);
          opt.textContent = String(hid);
          sel.appendChild(opt);
        }

        if (player.key === null) {
          const nullValue = state.selectedNullHid && options.includes(state.selectedNullHid) ? state.selectedNullHid : '';
          sel.value = nullValue;
        } else {
          const own = assignmentForPlayer(player.key);
          const ownHid = own && own.human_track_id ? String(own.human_track_id) : '';
          sel.value = ownHid;
        }

        sel.addEventListener('focus', () => {
          state.focusedPlayerIdx = idx;
          updateFocusHighlight();
        });
        sel.addEventListener('change', (e) => {
          const next = e.target.value || null;
          setPlayerAssignment(player.key, next);
        });

        tr.addEventListener('click', () => {
          state.focusedPlayerIdx = idx;
          updateFocusHighlight();
          sel.focus({ preventScroll: true });
        });

        const posTd = document.createElement('td');
        if (player.key === null) {
          posTd.textContent = '—';
        } else {
          const manualPos = manualPositionForPlayer(player.key);
          const btn = document.createElement('button');
          btn.type = 'button';
          if (manualPos) {
            btn.textContent = `Clear (${Math.round(manualPos.x)}, ${Math.round(manualPos.y)})`;
            btn.addEventListener('click', (e) => {
              e.stopPropagation();
              state.focusedPlayerIdx = idx;
              updateFocusHighlight();
              clearManualPosition(player.key);
            });
          } else {
            btn.textContent = 'Set by click';
            btn.disabled = false;
          }
          posTd.appendChild(btn);
        }

        hidTd.appendChild(sel);
        tr.appendChild(sw);
        tr.appendChild(playerTd);
        tr.appendChild(hidTd);
        tr.appendChild(posTd);
        body.appendChild(tr);
      }
    }

    function focusedPlayer() {
      return PLAYERS[state.focusedPlayerIdx] || PLAYERS[0];
    }

    function focusedPlayerSelect() {
      const player = focusedPlayer();
      return document.querySelector(`select[data-player-key="${String(player.key)}"]`);
    }

    function cycleFocusedHid(delta) {
      const player = focusedPlayer();
      const options = availableHidsForPlayer(player.key);
      if (!options.length) return;
      const current = player.key === null
        ? (state.selectedNullHid ? String(state.selectedNullHid) : null)
        : (() => {
            const own = assignmentForPlayer(player.key);
            return own && own.human_track_id !== null && own.human_track_id !== undefined ? String(own.human_track_id) : null;
          })();
      const curIdx = current === null ? -1 : options.findIndex(v => String(v) === current);
      let nextIdx = curIdx + delta;
      if (nextIdx < 0) nextIdx = options.length - 1;
      if (nextIdx >= options.length) nextIdx = 0;
      setPlayerAssignment(player.key, options[nextIdx]);
    }

    function focusDelta(delta) {
      state.focusedPlayerIdx = (state.focusedPlayerIdx + delta + PLAYERS.length) % PLAYERS.length;
      updateFocusHighlight();
      const sel = focusedPlayerSelect();
      if (sel) sel.focus({ preventScroll: true });
    }

    function nextUnconfirmedFrom(start) {
      for (let i = start; i < frameCount(); i++) {
        const ann = annByFrame(state.keyframes[i].frame_idx);
        if (ann && !ann.confirmed) return i;
      }
      return clampIdx(start);
    }

    function confirmAndAdvanceFrame() {
      const ann = currentAnnotation();
      ann.confirmed = true;
      const next = nextUnconfirmedFrom(state.idx + 1);
      state.idx = (next === state.idx && state.idx < frameCount() - 1) ? state.idx + 1 : next;
      state.focusedPlayerIdx = 0;
      state.selectedNullHid = null;
      renderFrame();
      saveState();
      const sel = focusedPlayerSelect();
      if (sel) sel.focus({ preventScroll: true });
    }

    function goFrame(delta) {
      state.idx = clampIdx(state.idx + delta);
      state.focusedPlayerIdx = 0;
      state.selectedNullHid = null;
      renderFrame();
      const sel = focusedPlayerSelect();
      if (sel) sel.focus({ preventScroll: true });
    }

    function setManualPositionFromFrameClick(event) {
      const player = focusedPlayer();
      if (!player || player.key === null) return;
      const img = document.getElementById('frameImg');
      if (!img.naturalWidth || !img.naturalHeight) return;
      const rect = img.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      const relX = (event.clientX - rect.left) / rect.width;
      const relY = (event.clientY - rect.top) / rect.height;
      if (relX < 0 || relX > 1 || relY < 0 || relY > 1) return;
      const x = Math.max(0, Math.min(img.naturalWidth - 1, relX * img.naturalWidth));
      const y = Math.max(0, Math.min(img.naturalHeight - 1, relY * img.naturalHeight));
      setFocusedPlayerManualPosition(x, y);
    }

    function firstUnconfirmedIdx() {
      for (let i = 0; i < frameCount(); i++) {
        const ann = annByFrame(state.keyframes[i].frame_idx);
        if (ann && !ann.confirmed) return i;
      }
      return 0;
    }

    function renderFrame() {
      if (frameCount() === 0) return;
      state.idx = clampIdx(state.idx);

      const kf = currentKeyframe();
      const ann = currentAnnotation();
      document.getElementById('counter').textContent = `(${state.idx + 1} / ${frameCount()})`;
      document.getElementById('meta').textContent = `frame ${kf.frame_idx} · t=${kf.timestamp_sec.toFixed(2)}s${ann.confirmed ? ' · confirmed' : ''}`;

      const img = document.getElementById('frameImg');
      img.onload = () => renderOverlay();
      img.src = `data:image/jpeg;base64,${kf.image_b64}`;

      renderPlayerRows();
      renderSummary();
      updateProgress();
      renderOverlay();
    }

    document.getElementById('prevBtn').addEventListener('click', () => goFrame(-1));
    document.getElementById('nextBtn').addEventListener('click', () => goFrame(1));
    document.getElementById('confirmBtn').addEventListener('click', () => confirmAndAdvanceFrame());
    document.getElementById('frameImg').addEventListener('click', (e) => setManualPositionFromFrameClick(e));

    document.addEventListener('keydown', (e) => {
      const tag = e.target && e.target.tagName ? e.target.tagName.toUpperCase() : '';
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;

      if (e.key === 'Tab') {
        e.preventDefault();
        focusDelta(e.shiftKey ? -1 : 1);
        return;
      }
      if (e.key === 'Enter') {
        e.preventDefault();
        confirmAndAdvanceFrame();
        return;
      }
      if (e.key === 'PageUp') {
        e.preventDefault();
        goFrame(-1);
        return;
      }
      if (e.key === 'PageDown') {
        e.preventDefault();
        goFrame(1);
        return;
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        cycleFocusedHid(-1);
        return;
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        cycleFocusedHid(1);
        return;
      }
      if (e.key === '1') { e.preventDefault(); state.focusedPlayerIdx = PLAYER_INDEX_BY_KEY.get('P1'); updateFocusHighlight(); focusedPlayerSelect()?.focus({ preventScroll: true }); return; }
      if (e.key === '2') { e.preventDefault(); state.focusedPlayerIdx = PLAYER_INDEX_BY_KEY.get('P2'); updateFocusHighlight(); focusedPlayerSelect()?.focus({ preventScroll: true }); return; }
      if (e.key === '3') { e.preventDefault(); state.focusedPlayerIdx = PLAYER_INDEX_BY_KEY.get('P3'); updateFocusHighlight(); focusedPlayerSelect()?.focus({ preventScroll: true }); return; }
      if (e.key === '4') { e.preventDefault(); state.focusedPlayerIdx = PLAYER_INDEX_BY_KEY.get('P4'); updateFocusHighlight(); focusedPlayerSelect()?.focus({ preventScroll: true }); return; }
      if (e.key.toLowerCase() === 'n') { e.preventDefault(); state.focusedPlayerIdx = PLAYER_INDEX_BY_KEY.get('null'); updateFocusHighlight(); focusedPlayerSelect()?.focus({ preventScroll: true }); return; }
      if (e.key.toLowerCase() === 'c') { e.preventDefault(); confirmAndAdvanceFrame(); }
      if (e.key === 'Backspace' || e.key === 'Delete') {
        e.preventDefault();
        clearFocusedManualPosition();
        return;
      }
    });

    async function init() {
      const resp = await fetch('/api/state');
      const payload = await resp.json();
      state.keyframes = payload.keyframes;
      state.gt = payload.gt;
      state.idx = firstUnconfirmedIdx();
      renderFrame();
      const sel = focusedPlayerSelect();
      if (sel) sel.focus({ preventScroll: true });
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


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        raise click.ClickException(f"Invalid hex color '{value}'; expected #RRGGBB")
    try:
        return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError as exc:
        raise click.ClickException(f"Invalid hex color '{value}'; expected #RRGGBB") from exc


def _default_players_data() -> dict[str, dict[str, str]]:
    team_by_pid = {"P1": "A", "P2": "A", "P3": "B", "P4": "B"}
    return {
        pid: {
            "name": DEFAULT_PLAYER_NAMES[pid],
            "color": DEFAULT_PLAYER_COLORS_HEX[pid],
            "description": "",
            "team": team_by_pid[pid],
            "position": "",
        }
        for pid in PLAYER_IDS
    }


def _load_or_init_players(players_path: Path) -> dict[str, dict[str, str]]:
    defaults = _default_players_data()
    if not players_path.exists():
        players_path.parent.mkdir(parents=True, exist_ok=True)
        players_path.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
        return defaults

    payload = _load_json(players_path)
    if not isinstance(payload, dict):
        raise click.ClickException(f"Invalid players file {players_path}: top-level JSON must be an object")

    merged: dict[str, dict[str, str]] = {}
    changed = False
    for pid in PLAYER_IDS:
        default_entry = defaults[pid].copy()
        raw_entry = payload.get(pid)
        if isinstance(raw_entry, dict):
            for key, value in raw_entry.items():
                if isinstance(value, str):
                    default_entry[key] = value
                else:
                    changed = True
        else:
            changed = True
        merged[pid] = default_entry
        if raw_entry != default_entry:
            changed = True

    if changed:
        players_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def _build_players_config(
    players_data: dict[str, dict[str, str]],
) -> tuple[list[dict], dict[Optional[str], tuple[int, int, int]]]:
    players_ui: list[dict] = []
    colors_by_player: dict[Optional[str], tuple[int, int, int]] = {None: NULL_PLAYER_COLOR_RGB}
    for pid in PLAYER_IDS:
        raw = players_data.get(pid, {})
        name = raw.get("name") or DEFAULT_PLAYER_NAMES[pid]
        color_hex = raw.get("color") or DEFAULT_PLAYER_COLORS_HEX[pid]
        color_rgb = _hex_to_rgb(color_hex)
        players_ui.append(
            {
                "key": pid,
                "label": f"{pid} ({name})",
                "color": [int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])],
            }
        )
        colors_by_player[pid] = color_rgb

    players_ui.append(
        {
            "key": None,
            "label": "null",
            "color": [
                int(NULL_PLAYER_COLOR_RGB[0]),
                int(NULL_PLAYER_COLOR_RGB[1]),
                int(NULL_PLAYER_COLOR_RGB[2]),
            ],
        }
    )
    return players_ui, colors_by_player


def _render_html_page(players_ui: list[dict]) -> bytes:
    return HTML_PAGE.replace("__PLAYERS_JSON__", json.dumps(players_ui)).encode("utf-8")


def _distance_sq(ax: float, ay: float, bx: float, by: float) -> float:
    return (ax - bx) ** 2 + (ay - by) ** 2


def _trim_evenly(frame_indices: list[int], max_count: int) -> list[int]:
    if len(frame_indices) <= max_count:
        return frame_indices
    picks = np.linspace(0, len(frame_indices) - 1, num=max_count, dtype=int)
    return [frame_indices[int(i)] for i in picks]


def select_first_frame(frames: list[dict]) -> list[dict]:
    """Return a single-element list with the earliest clean 4-player frame.

    Selection priority:
    1. Earliest frame with exactly 4 persons, all with unique non-null H-IDs.
    2. Earliest frame with any persons (fallback).
    3. Empty list when there are no detections at all.

    Used by ``beach annotate-gt --first-frame`` and ``beach run`` to get a
    minimal GT seed for the no-LLM rolling tracker.
    """
    frames_sorted = sorted(frames, key=lambda f: int(f["frame"]))

    for f in frames_sorted:
        persons = f.get("persons", [])
        if len(persons) == 4:
            hids = [p.get("human_track_id") for p in persons]
            if all(h is not None for h in hids) and len(set(hids)) == 4:
                return [f]

    # Fallback: first frame with any persons
    for f in frames_sorted:
        if f.get("persons"):
            return [f]

    return []


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

def include_existing_frames(
    selected_frames: list[dict],
    detections_frames: list[dict],
    existing_gt: dict,
    *,
    only_confirmed: bool = True,
) -> list[dict]:
    by_frame = {int(frame.get("frame")): frame for frame in detections_frames if isinstance(frame, dict) and "frame" in frame}
    chosen = {int(frame.get("frame")) for frame in selected_frames if isinstance(frame, dict) and "frame" in frame}

    for ann in existing_gt.get("annotations", []):
        if not isinstance(ann, dict) or "frame" not in ann:
            continue
        if only_confirmed and not bool(ann.get("confirmed", False)):
            continue
        frame_idx = int(ann.get("frame"))
        if frame_idx in by_frame:
            chosen.add(frame_idx)

    ordered = sorted(chosen)
    return [by_frame[idx] for idx in ordered if idx in by_frame]


def preseed_keyframes(
    keyframes: list[dict],
    identified_frames: list[dict],
    player_colors_rgb: dict[Optional[str], tuple[int, int, int]],
) -> list[dict]:
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
            color = player_colors_rgb.get(seed, player_colors_rgb[None])
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


def extract_annotated_frame(cap: cv2.VideoCapture, frame_idx: int) -> str:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise click.ClickException(f"Could not read frame {frame_idx} from video")

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
                "manual_positions": {},
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
        manual_positions: dict[str, dict[str, int]] = {}
        raw_manual = old.get("manual_positions")
        if isinstance(raw_manual, dict):
            for pid in PLAYER_IDS:
                pos = raw_manual.get(pid)
                if not isinstance(pos, dict):
                    continue
                x = pos.get("x")
                y = pos.get("y")
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    manual_positions[pid] = {"x": int(round(float(x))), "y": int(round(float(y)))}
        ann["manual_positions"] = manual_positions

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
                body = _render_html_page(state["players_ui"])
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
    "--detections", "-d",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Pass-1 detections JSON (default: <video_stem>_detections.json).",
)
@click.option(
    "--identified", "-i",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Pass-2 identified JSON (default: best available next to video).",
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output ground truth JSON path (default: <video_stem>_gt.json).",
)
@click.option(
    "--players",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_PLAYERS_PATH,
    show_default=True,
    help="players.json path used for labels/colors (created if missing).",
)
@click.option(
    "--video", "-v",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to source video.",
)
@click.option("--no-llm", is_flag=True, default=False, help="Identify strategy was --no-llm (affects identified file lookup).")
@click.option("--embeddings", is_flag=True, default=False, help="Identify strategy was --no-llm --embeddings.")
@click.option("--first-frame", "first_frame", is_flag=True, default=False, help="Annotate only the first clean 4-player frame (fast seed for --no-llm --seed-gt / beach run).")
@click.option("--port", type=int, default=7780, show_default=True)
@click.option("--fps", type=float, default=50.0, show_default=True)
def annotate_gt_cmd(
    detections: Path | None,
    identified: Path | None,
    output: Optional[Path],
    players: Path,
    video: Path,
    no_llm: bool,
    embeddings: bool,
    first_frame: bool,
    port: int,
    fps: float,
) -> None:
    detections_path = detections or video.with_name(video.stem + "_detections.json")
    if not detections_path.exists():
        raise click.ClickException(
            f"Detections file not found: {detections_path}\n"
            "Run 'beach track' first, or supply --detections explicitly."
        )

    # Resolve identified file: explicit > strategy-derived > any available > absent.
    if identified is not None:
        identified_file: Path | None = identified
    else:
        # Try the strategy the user indicated first, then the other variants.
        preferred = identified_path(video, no_llm=no_llm, embeddings=embeddings)
        all_suffixes = ("_identified_embeddings.json", "_identified_heuristic.json", "_identified.json")
        candidates = [preferred] + [
            video.with_name(video.stem + s) for s in all_suffixes if video.with_name(video.stem + s) != preferred
        ]
        identified_file = next((p for p in candidates if p.exists()), None)
        if identified_file is not None:
            click.echo(f"Using identified file: {identified_file}", err=True)
        else:
            click.echo(
                "No identified file found — frames will have no pre-labels. "
                "Run 'beach identify' first for faster annotation.",
                err=True,
            )

    det_json = _load_json(detections_path)
    id_frames: list[dict] = []
    if identified_file is not None:
        id_json = _load_json(identified_file)
        id_frames = id_json.get("frames") or []

    det_frames = det_json.get("frames")
    if not isinstance(det_frames, list):
        raise click.ClickException("Detections JSON must contain a top-level 'frames' array")

    output_path = output if output is not None else video.with_name(video.stem + "_gt.json")

    existing_gt: dict | None = None
    if output_path.exists():
        existing_payload = _load_json(output_path)
        if isinstance(existing_payload, dict):
            existing_gt = existing_payload

    players_data = _load_or_init_players(players)
    players_ui, player_colors_rgb = _build_players_config(players_data)

    if first_frame:
        selected = select_first_frame(det_frames)
        print("First-frame mode: annotating 1 seed frame.")
    else:
        selected = select_keyframes(det_frames, id_frames, fps=fps, interval_sec=2.0)
        if existing_gt is not None:
            selected = include_existing_frames(selected, det_frames, existing_gt, only_confirmed=True)
    keyframes = preseed_keyframes(selected, id_frames, player_colors_rgb)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise click.ClickException(f"Cannot open video: {video}")
    try:
        for kf in keyframes:
            kf["image_b64"] = extract_annotated_frame(cap, kf["frame_idx"])
    finally:
        cap.release()

    gt = _build_new_gt(video, detections_path, keyframes)
    if existing_gt is not None:
        gt = _merge_existing_gt(gt, existing_gt)

    state = {"keyframes": keyframes, "gt": gt, "players_ui": players_ui}

    print(f"Annotation server running at http://localhost:{port}")
    print(f"GT will be saved to {output_path}")
    print(f"Players file for labels/colors: {players}")
    print("Press Ctrl+C to stop.")

    try:
        run_annotation_server(state, output_path, port)
    except KeyboardInterrupt:
        pass
    finally:
        _save_gt(state["gt"], output_path)
