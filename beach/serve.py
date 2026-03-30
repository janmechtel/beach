"""Dev server for the beach volleyball viewer.

GET  /*                               — serves static files (viewer dist + data) with
                                        HTTP/1.1 Range support for video seeking.
GET  /api/videos                      — list available video stems from data/<stem>/
GET  /api/videos/<stem>/actions       — list action JSON files for a video
GET  /api/videos/<stem>/actions/<f>   — get a specific action JSON
POST /api/videos/<stem>/actions/<f>   — save edited actions back to disk
GET  /data/<stem>/<file>              — serves data directory files directly
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import click


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def _make_handler(root: Path, data_dir: Path, static_dir: Optional[Path]):
    """Return a request handler class bound to the given paths."""

    class Handler(BaseHTTPRequestHandler):
        # Force HTTP/1.1 so Range requests work (206 Partial Content).
        # SimpleHTTPRequestHandler defaults to 1.0 which breaks video seeking.
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter logs
            print(f"  {self.address_string()} {fmt % args}")

        # ------------------------------------------------------------------
        # Routing
        # ------------------------------------------------------------------

        def do_GET(self):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)

            if path == "/api/videos":
                self._api_list_videos()
            elif path.startswith("/api/videos/"):
                self._api_videos_route(path, method="GET")
            elif path.startswith("/data/"):
                # Serve files directly from data_dir
                rel = path[len("/data/"):]
                target = (data_dir / rel).resolve()
                if not str(target).startswith(str(data_dir)):
                    self._error(403, "Forbidden")
                    return
                self._serve_file(target)
            else:
                # Static file from viewer dist (or project root fallback)
                if static_dir and static_dir.exists():
                    # Map / to index.html
                    rel = path.lstrip("/") or "index.html"
                    target = (static_dir / rel).resolve()
                    if not str(target).startswith(str(static_dir)):
                        self._error(403, "Forbidden")
                        return
                    if not target.exists():
                        # SPA fallback: return index.html for unknown routes
                        target = static_dir / "index.html"
                    self._serve_file(target)
                else:
                    # Fallback: serve from project root (dev mode without built viewer)
                    rel = path.lstrip("/") or "index.html"
                    target = (root / rel).resolve()
                    if not str(target).startswith(str(root)):
                        self._error(403, "Forbidden")
                        return
                    self._serve_file(target)

        def do_POST(self):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path.startswith("/api/videos/"):
                self._api_videos_route(path, method="POST")
            # Legacy /save endpoint — kept for backward compat with old viewer.html
            elif path == "/save":
                self._legacy_save()
            else:
                self._error(404, "Not found")

        # ------------------------------------------------------------------
        # API handlers
        # ------------------------------------------------------------------

        def _api_list_videos(self):
            """GET /api/videos — list stems that have a data/<stem>/ directory."""
            stems = sorted(
                p.name for p in data_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            ) if data_dir.exists() else []
            self._json_response(stems)

        def _api_videos_route(self, path: str, method: str):
            """Route /api/videos/<stem>[/actions[/<filename>]] requests."""
            # Strip prefix
            rest = path[len("/api/videos/"):]
            parts = rest.split("/")

            if len(parts) == 1:
                # /api/videos/<stem> — not currently handled
                self._error(404, "Use /api/videos/<stem>/actions")
                return

            stem = parts[0]
            stem_dir = data_dir / stem

            if parts[1] != "actions":
                self._error(404, "Unknown route")
                return

            if len(parts) == 2:
                # /api/videos/<stem>/actions — list action JSON files
                if method != "GET":
                    self._error(405, "Method not allowed")
                    return
                self._api_list_actions(stem_dir)
            elif len(parts) == 3:
                filename = parts[2]
                if method == "GET":
                    self._api_get_action(stem_dir, filename)
                elif method == "POST":
                    self._api_save_action(stem_dir, filename)
                else:
                    self._error(405, "Method not allowed")
            else:
                self._error(404, "Not found")

        def _api_list_actions(self, stem_dir: Path):
            """GET /api/videos/<stem>/actions — list *.json files in stem dir."""
            if not stem_dir.exists():
                self._json_response([])
                return
            files = sorted(
                p.name for p in stem_dir.iterdir()
                if p.suffix == ".json" and not p.name.startswith(".")
            )
            self._json_response(files)

        def _api_get_action(self, stem_dir: Path, filename: str):
            """GET /api/videos/<stem>/actions/<filename>."""
            target = (stem_dir / filename).resolve()
            if not str(target).startswith(str(data_dir)):
                self._error(403, "Forbidden")
                return
            if not target.exists():
                self._error(404, f"{filename} not found")
                return
            self._serve_file(target)

        def _api_save_action(self, stem_dir: Path, filename: str):
            """POST /api/videos/<stem>/actions/<filename> — write actions JSON."""
            target = (stem_dir / filename).resolve()
            if not str(target).startswith(str(data_dir)):
                self._error(403, "Writes only allowed inside data/")
                return

            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
            except json.JSONDecodeError as exc:
                self._error(400, f"Invalid JSON: {exc}")
                return

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(data, indent=2))
            print(f"  saved {target.relative_to(data_dir)}  ({len(data) if isinstance(data, list) else '?'} items)")
            self._json_response({"ok": True})

        def _legacy_save(self):
            """POST /save — kept for backward compat; saves inside data_dir."""
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
                rel = Path(payload["path"])
                data = payload["data"]
            except (KeyError, ValueError) as exc:
                self._error(400, str(exc))
                return

            target = (root / rel).resolve()
            # Allow writes to output/ (legacy) or data/ (new layout)
            allowed = [str(root / "output"), str(data_dir)]
            if not any(str(target).startswith(a) for a in allowed):
                self._error(403, "Writes only allowed inside output/ or data/")
                return

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(data, indent=2))
            print(f"  saved {target.relative_to(root)}  ({len(data)} actions)")
            self._json_response({"ok": True})

        # ------------------------------------------------------------------
        # File serving with Range support
        # ------------------------------------------------------------------

        def _serve_file(self, path: Path):
            if not path.exists():
                self._error(404, "File not found")
                return
            if path.is_dir():
                self._error(403, "Directory listing not supported")
                return

            try:
                f = open(path, "rb")
            except OSError:
                self._error(404, "File not found")
                return

            try:
                file_size = os.fstat(f.fileno()).st_size
                range_header = self.headers.get("Range")
                content_type = self._guess_type(path)

                if range_header:
                    try:
                        unit, rng = range_header.split("=", 1)
                        if unit.strip() != "bytes":
                            raise ValueError("only bytes ranges are supported")
                        start_str, _, end_str = rng.partition("-")
                        start = int(start_str) if start_str else 0
                        end = int(end_str) if end_str else file_size - 1
                        end = min(end, file_size - 1)
                        if start > end or start < 0:
                            self._error(416, "Requested Range Not Satisfiable")
                            return
                        length = end - start + 1
                        f.seek(start)
                        self.send_response(206)
                        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                        self.send_header("Content-Length", str(length))
                        self.send_header("Accept-Ranges", "bytes")
                        self.send_header("Content-Type", content_type)
                        self.end_headers()
                        self._copy_bytes(f, length)
                    except (ValueError, IndexError):
                        self._error(400, "Malformed Range header")
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(file_size))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Type", content_type)
                    self.end_headers()
                    self._copy_bytes(f, file_size)
            finally:
                f.close()

        def _copy_bytes(self, f, n: int, chunk: int = 65536):
            remaining = n
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                self.wfile.write(data)
                remaining -= len(data)

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _guess_type(self, path: Path) -> str:
            ext = path.suffix.lower()
            types = {
                ".html": "text/html",
                ".js": "application/javascript",
                ".mjs": "application/javascript",
                ".css": "text/css",
                ".json": "application/json",
                ".mp4": "video/mp4",
                ".mov": "video/quicktime",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".svg": "image/svg+xml",
                ".ico": "image/x-icon",
                ".woff2": "font/woff2",
                ".woff": "font/woff",
            }
            return types.get(ext, "application/octet-stream")

        def _json_response(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code: int, message: str):
            body = json.dumps({"error": message}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def end_headers(self):
            # Allow the viewer (which may be on a different port in dev) to call the API
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

    return Handler


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------

@click.command("serve")
@click.option(
    "--port", "-p",
    default=8080,
    show_default=True,
    type=int,
    help="Port to listen on.",
)
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data"),
    show_default=True,
    help="Directory containing per-video data subdirectories.",
)
@click.option(
    "--viewer-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Path to built viewer dist/ directory (default: <project>/viewer/dist).",
)
def serve_cmd(port: int, data_dir: Path, viewer_dir: Optional[Path]) -> None:
    """Start the dev server for the beach volleyball viewer."""
    root = Path.cwd()

    # Resolve viewer dir: explicit arg > viewer/dist relative to cwd
    if viewer_dir is None:
        candidate = root / "viewer" / "dist"
        viewer_dir = candidate if candidate.exists() else None

    if not data_dir.is_absolute():
        data_dir = root / data_dir

    data_dir.mkdir(parents=True, exist_ok=True)

    handler_class = _make_handler(root, data_dir, viewer_dir)
    server = HTTPServer(("", port), handler_class)

    if viewer_dir and viewer_dir.exists():
        print(f"Viewer : {viewer_dir}")
    else:
        print("Viewer : not found (build with: cd viewer && npm run build)")

    print(f"Data   : {data_dir}")
    print(f"Serving on http://localhost:{port}/")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
