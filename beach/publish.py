"""beach publish — generate static viewer manifests and optionally upload data to R2.

Reads publish.json at the repo root, selects JSON files per stem (fnmatch patterns)
plus the preferred mp4, writes viewer/public/api/ manifests, and optionally uploads
data files to R2 via wrangler.
"""

from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
VIEWER_PUBLIC_API = REPO_ROOT / "viewer" / "public" / "api"
PUBLISH_JSON = REPO_ROOT / "publish.json"


def _pick_video(stem: str, mp4_files: list[str]) -> str | None:
    """
    Mirror the TypeScript pickVideoFile priority from ActionViewer.tsx:
      1. Exact match: <stem>.mp4
      2. No "annotated" in the name
      3. Has "annotated" in the name
      4. Any mp4
    """
    if not mp4_files:
        return None
    exact = f"{stem}.mp4"
    if exact in mp4_files:
        return exact
    non_annotated = [f for f in mp4_files if "annotated" not in f.lower()]
    if non_annotated:
        return non_annotated[0]
    annotated = [f for f in mp4_files if "annotated" in f.lower()]
    if annotated:
        return annotated[0]
    return mp4_files[0]


def _collect_files(stem: str, patterns: list[str]) -> tuple[list[str], str | None]:
    """Return (json_filenames, video_filename) for a stem."""
    stem_dir = DATA_ROOT / stem
    if not stem_dir.is_dir():
        click.echo(f"  WARNING: data/{stem}/ not found — skipping", err=True)
        return [], None

    all_files = [f.name for f in stem_dir.iterdir() if f.is_file()]

    json_files = sorted(
        name for name in all_files
        if name.endswith(".json") and any(fnmatch.fnmatch(name, p) for p in patterns)
    )
    mp4_files = sorted(f for f in all_files if f.endswith(".mp4"))
    return json_files, _pick_video(stem, mp4_files)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    click.echo(f"  wrote {path.relative_to(REPO_ROOT)}")


def _upload(stem: str, filename: str) -> None:
    local_path = DATA_ROOT / stem / filename
    r2_key = f"beach-data/data/{stem}/{filename}"
    click.echo(f"  upload: {r2_key}")
    result = subprocess.run(
        ["npx", "wrangler", "r2", "object", "put", r2_key, "--file", str(local_path), "--remote"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        err = result.stderr.decode("utf-8").strip()
        raise RuntimeError(f"wrangler exited {result.returncode} for {r2_key}\n{err}")
    click.echo(f"  done: {r2_key}")


@click.command("publish")
@click.option("--dry-run", is_flag=True, help="Print plan without writing or uploading.")
@click.option("--upload", is_flag=True, help="Upload data files to R2 via wrangler.")
def publish_cmd(dry_run: bool, upload: bool) -> None:
    """Generate viewer manifests and optionally upload data to R2."""
    config = json.loads(PUBLISH_JSON.read_text(encoding="utf-8"))
    stems_config: dict[str, dict] = config["stems"]

    stems: list[str] = []
    stem_files: dict[str, list[str]] = {}

    for stem, stem_cfg in stems_config.items():
        click.echo(f"\n[{stem}]")
        json_files, video = _collect_files(stem, stem_cfg.get("json_patterns", []))

        all_files = list(json_files)
        if video:
            all_files.append(video)

        if not all_files:
            click.echo(f"  WARNING: no files selected for {stem}", err=True)
            continue

        stems.append(stem)
        stem_files[stem] = all_files

        for f in json_files:
            click.echo(f"  json: {f}")
        if video:
            click.echo(f"  video: {video}")
        else:
            click.echo(f"  WARNING: no mp4 found for {stem}", err=True)

    click.echo()

    if dry_run:
        click.echo("[dry-run] would write:")
        click.echo(f"  viewer/public/api/manifest.json  → {{stems: {stems}}}")
        for stem, files in stem_files.items():
            click.echo(f"  viewer/public/api/{stem}/actions.json → {files}")
        if upload:
            click.echo("[dry-run] would upload:")
            for stem, files in stem_files.items():
                for f in files:
                    click.echo(f"  beach-data/data/{stem}/{f}")
        return

    _write_json(VIEWER_PUBLIC_API / "manifest.json", {"stems": stems})
    for stem, files in stem_files.items():
        _write_json(VIEWER_PUBLIC_API / stem / "actions.json", files)

    if upload:
        import concurrent.futures
        click.echo("\nUploading to R2...")
        tasks = []
        for stem, files in stem_files.items():
            for filename in files:
                tasks.append((stem, filename))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_upload, s, f) for s, f in tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    click.echo(f"  ERROR: {e}", err=True)
                    sys.exit(1)
