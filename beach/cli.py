"""CLI entry points for the beach volleyball analysis toolkit.

Commands
--------
beach detect  — Run cut detection only; print/save JSON.
beach split   — Split a video given an existing metadata.json.
beach process — Detect cuts *and* split in one step (most common usage).

All commands write structured logging to stderr and actionable output
(JSON, file paths) to stdout so they compose cleanly with shell pipelines.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from beach.cut_detect import (
    DEFAULT_HIST_THRESHOLD,
    DEFAULT_MIN_GAP_SEC,
    DEFAULT_SAMPLE_EVERY,
    DEFAULT_THRESHOLD,
    detect_cuts,
)
from beach.models import MatchMetadata
from beach.split import split_video, write_metadata

# ---------------------------------------------------------------------------
# Logging setup — INFO to stderr by default; DEBUG when --verbose
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(levelname)s  %(message)s"


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level, format=_LOG_FORMAT)


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

_threshold_option = click.option(
    "--threshold",
    default=DEFAULT_THRESHOLD,
    show_default=True,
    type=float,
    help="Mean absolute pixel difference (0–255) to flag as a cut.",
)

_sample_option = click.option(
    "--sample-every",
    default=DEFAULT_SAMPLE_EVERY,
    show_default=True,
    type=int,
    help="Process every N-th frame.  Higher = faster, but may miss very short cuts.",
)

_gap_option = click.option(
    "--min-gap",
    default=DEFAULT_MIN_GAP_SEC,
    show_default=True,
    type=float,
    help="Minimum seconds between reported cuts (suppresses duplicates).",
)

_hist_option = click.option(
    "--use-hist",
    is_flag=True,
    default=False,
    help=(
        "Require histogram distance > --hist-threshold in addition to MAD threshold. "
        "Use when camera motion causes false positives."
    ),
)

_hist_threshold_option = click.option(
    "--hist-threshold",
    default=DEFAULT_HIST_THRESHOLD,
    show_default=True,
    type=float,
    help="Bhattacharyya histogram distance threshold (0–1), used with --use-hist.",
)

_verbose_option = click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Beach volleyball video analysis toolkit."""


# ---------------------------------------------------------------------------
# beach detect
# ---------------------------------------------------------------------------

@cli.command("detect")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=None,
    help=(
        "Directory to write metadata.json.  If omitted, JSON is printed to stdout."
    ),
)
@click.option(
    "--preview",
    is_flag=True,
    default=False,
    help="Print detected cut timestamps only (no JSON, no files).  For threshold tuning.",
)
@_threshold_option
@_sample_option
@_gap_option
@_hist_option
@_hist_threshold_option
@_verbose_option
def detect_cmd(
    video: Path,
    output: Path | None,
    preview: bool,
    threshold: float,
    sample_every: int,
    min_gap: float,
    use_hist: bool,
    hist_threshold: float,
    verbose: bool,
) -> None:
    """Detect hard cuts in VIDEO and emit metadata JSON.

    VIDEO is the path to the input MP4.

    Without --output, prints JSON to stdout so you can pipe it:

    \b
        beach detect game.mp4 | jq '.points | length'
    """
    _configure_logging(verbose)

    cuts = detect_cuts(
        video,
        threshold=threshold,
        sample_every=sample_every,
        min_gap_sec=min_gap,
        use_hist=use_hist,
        hist_threshold=hist_threshold,
    )

    if preview:
        click.echo(f"Detected {len(cuts)} cuts:")
        for c in cuts:
            click.echo(f"  t={c.timestamp:.3f}s  frame={c.frame}  score={c.score:.1f}")
        return

    # Build metadata (we need total duration — open video briefly)
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    total_duration = total_frames / fps

    match_id = video.stem
    metadata = MatchMetadata.from_cuts(video, cuts, total_duration, match_id)

    if output is not None:
        write_metadata(metadata, output)
        click.echo(str(output / "metadata.json"))
    else:
        click.echo(metadata.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# beach split
# ---------------------------------------------------------------------------

@cli.command("split")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--cuts", "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to metadata.json produced by 'beach detect'.",
)
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    help="Directory to write extracted clips.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing clip files.",
)
@_verbose_option
def split_cmd(
    video: Path,
    cuts: Path,
    output: Path,
    overwrite: bool,
    verbose: bool,
) -> None:
    """Split VIDEO into per-point clips using metadata from CUTS.

    CUTS is the metadata.json written by 'beach detect'.
    """
    _configure_logging(verbose)

    raw = json.loads(cuts.read_text())
    metadata = MatchMetadata.model_validate(raw)

    paths = split_video(video, metadata, output, overwrite=overwrite)
    for p in paths:
        click.echo(str(p))


# ---------------------------------------------------------------------------
# beach process  (detect + split in one step)
# ---------------------------------------------------------------------------

@cli.command("process")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    help="Directory to write clips and metadata.json.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing clip files.",
)
@_threshold_option
@_sample_option
@_gap_option
@_hist_option
@_hist_threshold_option
@_verbose_option
def process_cmd(
    video: Path,
    output: Path,
    overwrite: bool,
    threshold: float,
    sample_every: int,
    min_gap: float,
    use_hist: bool,
    hist_threshold: float,
    verbose: bool,
) -> None:
    """Detect cuts in VIDEO then split it into per-point clips.

    This is the most common workflow — equivalent to:

    \b
        beach detect game.mp4 --output out/game/
        beach split  game.mp4 --cuts out/game/metadata.json --output out/game/
    """
    _configure_logging(verbose)

    cuts = detect_cuts(
        video,
        threshold=threshold,
        sample_every=sample_every,
        min_gap_sec=min_gap,
        use_hist=use_hist,
        hist_threshold=hist_threshold,
    )

    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    total_duration = total_frames / fps

    match_id = video.stem
    metadata = MatchMetadata.from_cuts(video, cuts, total_duration, match_id)

    meta_path = write_metadata(metadata, output)
    click.echo(f"Metadata → {meta_path}", err=True)

    paths = split_video(video, metadata, output, overwrite=overwrite)
    click.echo(f"Extracted {len(paths)} clip(s) → {output}", err=True)

    # Emit metadata JSON to stdout for piping
    click.echo(metadata.model_dump_json(indent=2))
