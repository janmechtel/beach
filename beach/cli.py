"""CLI entry points for the beach volleyball analysis toolkit.

Commands
--------
beach track         — Pass 1: YOLO person + ball detection with ByteTrack IDs.
beach identify      — Pass 2: Player identification via Gemini or heuristic.
beach analyze       — Pass 3: Action extraction via Gemini.
beach compare       — Compare candidate action JSON against ground truth.
beach annotate-gt   — Build frame-level ground truth for identification evaluation.
beach eval-id       — Score an identified JSON against ground truth.
beach serve         — Start the dev server for the viewer.
beach render        — Render identified JSON overlay onto source video.
"""

from __future__ import annotations

import click

# Subcommand modules — imported here so cli.add_command works.
from beach.track import track_cmd
from beach.identify import identify_cmd
from beach.analyze import analyze_cmd
from beach.compare import compare_cmd
from beach.annotate_gt import annotate_gt_cmd
from beach.eval_id import eval_id_cmd
from beach.serve import serve_cmd
from beach.render import render_cmd


@click.group()
def cli() -> None:
    """Beach volleyball video analysis toolkit."""


# Wire in subcommands from their respective modules.
cli.add_command(track_cmd)
cli.add_command(identify_cmd)
cli.add_command(analyze_cmd)
cli.add_command(compare_cmd)
cli.add_command(annotate_gt_cmd)
cli.add_command(eval_id_cmd)
cli.add_command(serve_cmd)
cli.add_command(render_cmd)
