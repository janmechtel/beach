"""beach split-rallies — Split a video + merged JSON into per-rally clips and JSONs.

Reads *_merged.json produced by `beach analytics` and, for each detected rally,
writes:

  <output_dir>/rally_<NN>/rally_<NN>.mp4          — video clip
  <output_dir>/rally_<NN>/rally_<NN>_merged.json  — merged data slice (timestamps rebased to 0)

The per-rally merged JSON is a self-contained version of the full merged JSON
that covers only the frames inside that rally window.  Timestamps are rebased
so that t=0 corresponds to the start of the clip — matching what Gemini will
see when you later run `beach analyze` on each clip.

If the full merged JSON contains a ``touches`` key (added by `beach analytics`
via detect-touches), touch events are sliced per rally and included in each
per-rally merged JSON with timestamps rebased to match the clip.

Metadata about the rally's position in the original video is preserved in a
top-level `source` block:

  {
    "rally_id": 3,
    "fps": 50.0,
    "total_frames": 742,
    "source": {
      "video": "GH021569_court.mp4",
      "start_sec": 41.2,
      "end_sec": 56.9,
      "start_frame": 2060,
      "end_frame": 2845
    },
    "rallies": [],          // empty — the clip IS the rally
    "touches": [...],       // touch events with timestamps rebased to 0
    "frames": [...]         // timestamps start from 0.0
  }

Usage
-----
    beach split-rallies --video videos/GH021569_court.mp4
    beach split-rallies --video videos/GH021569_court.mp4 --merged videos/GH021569_court_merged.json
    beach split-rallies --video videos/GH021569_court.mp4 --output-dir rallies/
    beach split-rallies --video videos/GH021569_court.mp4 --dry-run
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click


# ---------------------------------------------------------------------------
# Core function  (imported by detect_touches via split_rallies --detect-touches)
# ---------------------------------------------------------------------------

def split_rallies(
    video_path: Path,
    merged_path: Path,
    output_dir: Path,
    dry_run: bool = False,
    encode_preset: str = "fast",
    encode_crf: int = 18,
) -> list[Path]:
    """Split video and merged JSON into per-rally directories.

    Parameters
    ----------
    video_path:     Source video file.
    merged_path:    *_merged.json produced by `beach merge` / `beach analytics`.
                    If it contains a ``touches`` key those events are sliced per
                    rally and included in each per-rally merged JSON.
    output_dir:     Root directory for per-rally sub-directories.
    dry_run:        Print what would be done without writing anything.
    encode_preset:  libx264 preset (fast / medium / slow …).
    encode_crf:     libx264 CRF (18 = near-lossless, 23 = smaller file).

    Returns
    -------
    List of created rally directories.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH — please install it.")

    # --- Load merged JSON ---
    merged = json.loads(merged_path.read_text())
    rallies: list[dict] = merged.get("rallies", [])
    fps: float = merged.get("fps", 50.0)
    all_frames: list[dict] = merged.get("frames", [])
    all_touches: list[dict] | None = merged.get("touches")  # may be absent

    if not rallies:
        print("  No rallies found in merged JSON — nothing to split.")
        return []

    # Build a lookup: frame_index → frame dict (for fast slicing)
    frame_by_idx: dict[int, dict] = {f["frame"]: f for f in all_frames}

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Source video : {video_path}")
    print(f"  Merged JSON  : {merged_path}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Rallies      : {len(rallies)}")
    print()

    created_dirs: list[Path] = []

    for rally in rallies:
        rally_id: int = rally["rally_id"]
        start_sec: float = rally["start_sec"]
        end_sec: float = rally["end_sec"]
        start_frame: int = rally["start_frame"]
        end_frame: int = rally["end_frame"]
        duration_sec: float = end_sec - start_sec

        rally_name = f"rally_{rally_id:02d}"
        rally_dir = output_dir / rally_name
        clip_path = rally_dir / f"{rally_name}.mp4"
        json_path = rally_dir / f"{rally_name}_merged.json"

        print(f"  [{rally_id}] {start_sec:.2f}s – {end_sec:.2f}s  ({duration_sec:.2f}s)  → {rally_dir.name}/")

        if dry_run:
            print(f"       DRY RUN: would create {clip_path} and {json_path}")
            continue

        rally_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------
        # 1. Cut video clip with ffmpeg
        # ----------------------------------------------------------------
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", str(video_path),
            "-t", str(duration_sec),
            "-c:v", "libx264",
            "-crf", str(encode_crf),
            "-preset", encode_preset,
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(clip_path),
        ]
        print(f"       ffmpeg: {' '.join(ffmpeg_cmd)}")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"       ERROR: ffmpeg failed for rally {rally_id}:", file=sys.stderr)
            print(result.stderr[-2000:], file=sys.stderr)
            raise RuntimeError(f"ffmpeg failed (rally {rally_id})")
        print(f"       Video  → {clip_path.name}  ({clip_path.stat().st_size // 1024} KB)")

        # ----------------------------------------------------------------
        # 2. Slice frames + rebase timestamps
        # ----------------------------------------------------------------
        rally_frames: list[dict] = []
        for fidx in range(start_frame, end_frame + 1):
            frame = frame_by_idx.get(fidx)
            if frame is None:
                continue
            # Deep-copy the frame dict and rebase timestamp
            rebased = dict(frame)
            rebased["timestamp_sec"] = round(frame["timestamp_sec"] - start_sec, 4)
            rally_frames.append(rebased)

        # ----------------------------------------------------------------
        # 3. Write per-rally merged JSON  (includes touches if present)
        # ----------------------------------------------------------------
        rally_touches = None
        if all_touches is not None:
            rally_touches = [
                {**t, "timestamp_sec": round(t["timestamp_sec"] - start_sec, 4)}
                for t in all_touches
                if start_frame <= t["frame"] <= end_frame
            ]

        rally_merged = {
            "rally_id": rally_id,
            "fps": fps,
            "total_frames": len(rally_frames),
            "proximity_threshold_px": merged.get("proximity_threshold_px", 400),
            "source": {
                "video": video_path.name,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_frame": start_frame,
                "end_frame": end_frame,
            },
            # rally clip IS the full content — no sub-rallies
            "rallies": [],
            "touches": rally_touches if rally_touches is not None else [],
            "frames": rally_frames,
        }
        json_path.write_text(json.dumps(rally_merged, indent=2))
        n_touches = len(rally_touches) if rally_touches is not None else "n/a"
        print(f"       JSON   → {json_path.name}  ({len(rally_frames)} frames, {n_touches} touches)")

        created_dirs.append(rally_dir)

    print()
    if not dry_run:
        print(f"  Done — {len(created_dirs)} rally clip(s) written to {output_dir}/")
    return created_dirs


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@click.command("split-rallies")
@click.option(
    "--video", "-v",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Source video file.",
)
@click.option(
    "--merged", "-m",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Merged JSON (default: <video_stem>_merged.json sibling of --video).",
)
@click.option(
    "--output-dir", "-o",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Root output directory (default: <video_stem>_rallies/ sibling of --video).",
)
@click.option(
    "--dry-run",
    is_flag=True, default=False,
    help="Print what would be done without writing any files.",
)
@click.option(
    "--crf",
    default=18, show_default=True, type=int,
    help="libx264 CRF quality (18 = near-lossless, 23 = smaller).",
)
@click.option(
    "--preset",
    default="fast", show_default=True,
    type=click.Choice(["ultrafast", "superfast", "veryfast", "faster",
                        "fast", "medium", "slow", "slower", "veryslow"]),
    help="libx264 encoding preset.",
)
def split_rallies_cmd(
    video: Path,
    merged: Optional[Path],
    output_dir: Optional[Path],
    dry_run: bool,
    crf: int,
    preset: str,
) -> None:
    """Split a video + merged JSON into individual rally clips and JSON slices.

    Reads <video_stem>_merged.json (produced by `beach analytics`) and for each
    rally writes a trimmed video clip and a matching merged JSON with timestamps
    rebased to start at 0 — ready for per-rally action identification with
    `beach analyze`.

    If the merged JSON contains a ``touches`` key (added by `beach analytics`),
    touch events are automatically sliced into each per-rally merged JSON.
    """
    merged_path = merged or video.with_name(video.stem + "_merged.json")
    out_dir = output_dir or video.with_name(video.stem + "_rallies")

    if not merged_path.exists():
        raise click.ClickException(
            f"Merged JSON not found: {merged_path}\n"
            "Run 'beach analytics' first to generate it."
        )

    split_rallies(
        video_path=video,
        merged_path=merged_path,
        output_dir=out_dir,
        dry_run=dry_run,
        encode_preset=preset,
        encode_crf=crf,
    )
