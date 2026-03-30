#!/usr/bin/env python3
"""
Minimal dev server for viewer.html.

GET  /*          — serves static files from the project root with HTTP range
                   support so browsers can seek into large video files.
POST /save       — body: {"path": "output/foo.json", "data": [...]}
                   writes data back to disk; path must be inside output/
"""
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = ROOT / "output"


class Handler(SimpleHTTPRequestHandler):
    # Force HTTP/1.1 so the browser can issue Range requests and we can respond
    # with 206 Partial Content.  SimpleHTTPRequestHandler uses HTTP/1.0 by
    # default, which causes Chrome/Firefox to refuse to seek large video files.
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter logs
        print(f"  {self.address_string()} {fmt % args}")

    def do_GET(self):
        """Serve static files; honour Range header for video seeking."""
        # Resolve path the same way SimpleHTTPRequestHandler does.
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            # Let the parent handle directory listings unchanged.
            super().do_GET()
            return

        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return

        try:
            file_size = os.fstat(f.fileno()).st_size
            range_header = self.headers.get("Range")

            if range_header:
                # Parse "bytes=start-end" (end is inclusive, may be absent).
                try:
                    unit, rng = range_header.split("=", 1)
                    if unit.strip() != "bytes":
                        raise ValueError("only bytes ranges are supported")
                    start_str, _, end_str = rng.partition("-")
                    start = int(start_str) if start_str else 0
                    end   = int(end_str)   if end_str   else file_size - 1
                    end   = min(end, file_size - 1)
                    if start > end or start < 0:
                        self.send_error(416, "Requested Range Not Satisfiable")
                        return
                    length = end - start + 1
                    f.seek(start)
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Type", self.guess_type(path))
                    self.end_headers()
                    self._copy_bytes(f, length)
                except (ValueError, IndexError):
                    self.send_error(400, "Malformed Range header")
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Type", self.guess_type(path))
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

    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            payload = json.loads(body)
            rel  = Path(payload["path"])   # e.g. "output/first30_....json"
            data = payload["data"]         # the actions array
        except (KeyError, ValueError) as e:
            self.send_error(400, str(e))
            return

        # Restrict writes to output/ — no path traversal
        target = (ROOT / rel).resolve()
        if not str(target).startswith(str(OUTPUT_DIR)):
            self.send_error(403, "writes only allowed inside output/")
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2))
        print(f"  saved {target.relative_to(ROOT)}  ({len(data)} actions)")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "12")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def end_headers(self):
        # Allow the page to call /save without CORS issues when opened via file://
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


if __name__ == "__main__":
    server = HTTPServer(("", PORT), Handler)
    print(f"Serving on http://localhost:{PORT}/viewer.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
