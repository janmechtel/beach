"""Video splitting via FFmpeg.

Uses ``ffmpeg -ss … -to … -c copy`` for lossless, near-instant splitting
with no re-encoding.  Each point gets its own MP4 file named
``point_NNN.mp4`` inside the output directory.

FFmpeg is a system dependency — the module checks for it at call time and
raises a clear error if it is absent, rather than failing deep in a
subprocess call with an opaque message.

Notes on ``-c copy`` accuracy
------------------------------
Copy mode seeks to the nearest keyframe, which may be slightly before the
requested timestamp.  For beach volleyball points (each a distinct rally)
this is perfectly acceptable — the viewer will see a frame or two of the
previous rally at most, which is visually obvious and does not affect
analysis.  If sub-frame accuracy is needed in a future milestone, drop
``-c copy`` and add ``-vf select`` instead.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from beach.models import MatchMetadata, Point

logger = logging.getLogger(__name__)


def _require_ffmpeg() -> str:
    """Return the path to ffmpeg or raise if not found."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "ffmpeg not found on PATH.  Install it with:\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  macOS:         brew install ffmpeg\n"
            "  Arch:          sudo pacman -S ffmpeg"
        )
    return path


def _split_point(
    ffmpeg: str,
    source: Path,
    point: Point,
    output_path: Path,
    overwrite: bool,
) -> None:
    """Extract a single point clip with FFmpeg."""
    if output_path.exists() and not overwrite:
        logger.debug("Skipping existing clip: %s", output_path.name)
        return

    cmd = [
        ffmpeg,
        "-y",            # overwrite without asking
        "-ss", str(point.start),
        "-to", str(point.end),
        "-i", str(source),
        "-c", "copy",    # lossless: no re-encode
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]

    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed for point {point.index}:\n"
            f"  stdout: {result.stdout[-500:]}\n"
            f"  stderr: {result.stderr[-500:]}"
        )


def split_video(
    source: Path | str,
    metadata: MatchMetadata,
    output_dir: Path | str,
    overwrite: bool = False,
) -> list[Path]:
    """Split a video into per-point clips according to metadata.

    Parameters
    ----------
    source:
        Path to the original video file.
    metadata:
        Parsed ``MatchMetadata``; provides the cut timestamps and filenames.
    output_dir:
        Directory where clips will be written.  Created if absent.
    overwrite:
        If False (default), existing clips are skipped.  Set True to
        re-extract all clips even if they already exist.

    Returns
    -------
    list[Path]
        Absolute paths of all extracted clip files, in point order.

    Raises
    ------
    FileNotFoundError
        If the source video does not exist.
    RuntimeError
        If FFmpeg is not installed or exits non-zero for any clip.
    """
    source = Path(source)
    output_dir = Path(output_dir)

    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")

    ffmpeg = _require_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_paths: list[Path] = []

    for point in metadata.points:
        clip_path = output_dir / point.file
        logger.info(
            "Extracting point %d/%d: %.2f s – %.2f s → %s",
            point.index,
            len(metadata.points),
            point.start,
            point.end,
            point.file,
        )
        _split_point(ffmpeg, source, point, clip_path, overwrite)
        clip_paths.append(clip_path)

    return clip_paths


def write_metadata(metadata: MatchMetadata, output_dir: Path | str) -> Path:
    """Serialise metadata to ``<output_dir>/metadata.json``.

    Returns the path of the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "metadata.json"
    dest.write_text(metadata.model_dump_json(indent=2))
    logger.info("Wrote metadata → %s", dest)
    return dest
