#!/usr/bin/env python3
"""
Minimal dev server for viewer.html.

GET  /*          — serves static files from the project root (same as python -m http.server)
POST /save       — body: {"path": "output/foo.json", "data": [...]}
                   writes data back to disk; path must be inside output/
"""
import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = ROOT / "output"


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter logs
        print(f"  {self.address_string()} {fmt % args}")

    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            payload = json.loads(body)
            rel = Path(payload["path"])          # e.g. "output/first30_....json"
            data = payload["data"]               # the actions array
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
        self.send_header("Access-Control-Allow-Origin", "*")
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
