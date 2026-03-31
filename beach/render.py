"""
beach render — Render an identified JSON + source video into an annotated MP4.

Reads a pass-2 identified JSON (output of `beach identify`) and the source
video, then writes a new video with per-player coloured bounding boxes and
labels identical to those produced by `beach identify --render-identified`.

Usage
-----
    beach render -v videos/output/GH021569_court_001.mp4 \\
                 -o videos/output/GH021569_court_001_identified_AB.json

Output defaults to <identified_json_stem>_rendered.mp4 next to the JSON.
Pass --output / -r to override.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from beach.identify import _render_identified


@click.command("render")
@click.option(
    "--video", "-v",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Source video file (same one passed to `beach track`).",
)
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Identified JSON produced by `beach identify`.",
)
@click.option(
    "--render-output", "-r",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination MP4 (default: <identified_json_stem>_rendered.mp4 next to JSON).",
)
def render_cmd(
    video: Path,
    output: Path,
    render_output: Optional[Path],
) -> None:
    """Render identified JSON overlay boxes onto the source video."""
    identified_path = output
    if not identified_path.exists():
        raise click.ClickException(f"Identified JSON not found: {identified_path}")

    data = json.loads(identified_path.read_text())
    if "frames" not in data:
        raise click.ClickException(
            f"{identified_path} does not look like an identified JSON "
            "(missing 'frames' key). Run `beach identify` first."
        )

    render_path = render_output or identified_path.with_name(
        identified_path.stem + "_rendered.mp4"
    )
    _render_identified(video, data["frames"], render_path)
