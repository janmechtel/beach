"""CLI entry points for the beach volleyball analysis toolkit.

Commands
--------
beach run              — Full pipeline: track → annotate first frame → identify → render.
beach track            — Pass 1: YOLO person + ball detection with ByteTrack IDs.
beach identify         — Pass 2: Player identification via Gemini or heuristic.
beach analyze          — Pass 3: Action extraction via Gemini.
beach compare          — Compare candidate action JSON against ground truth.
beach annotate-gt      — Build frame-level ground truth for identification evaluation.
beach eval-id          — Score an identified JSON against ground truth.
beach eval-frame       — Score single-frame identification strategies against ground truth.
beach serve            — Start the dev server for the viewer.
beach render           — Render identified JSON overlay onto source video.
beach ball-track       — Run VballNet ball tracking → *_ball.csv.
beach merge            — Merge player bboxes + ball → *_merged.json.
beach detect-rallies   — Detect rally timings from merged JSON → *_rallies.json.
beach analytics-render — Render analytics overlay (players + ball + rally markers).
beach analytics        — Full analytics pipeline (ball-track → merge → rallies → render).
beach split-rallies    — Split video + merged JSON into per-rally clips + JSON slices.
beach fix-player-ids   — Fix LLM player IDs in action JSON using touches.json ground truth.
beach stitch-rallies   — Combine per-rally *_fixed.json files into one full-video action JSON.
beach publish          — Generate static viewer manifests; optionally upload to R2.
"""

from __future__ import annotations

import click

# Subcommand modules — imported here so cli.add_command works.
from beach.track import track_cmd
from beach.run import run_cmd
from beach.identify import identify_cmd
from beach.analyze import analyze_cmd
from beach.compare import compare_cmd
from beach.annotate_gt import annotate_gt_cmd
from beach.eval_id import eval_id_cmd
from beach.serve import serve_cmd
from beach.render import render_cmd
from beach.eval_frame import eval_frame_cmd
from beach.ball_track import ball_track_cmd
from beach.merge import merge_cmd
from beach.rallies import detect_rallies_cmd
from beach.analytics_render import analytics_render_cmd
from beach.analytics import analytics_cmd
from beach.split_rallies import split_rallies_cmd
from beach.fix_player_ids import fix_player_ids_cmd
from beach.stitch_rallies import stitch_rallies_cmd
from beach.publish import publish_cmd


@click.group()
def cli() -> None:
    """Beach volleyball video analysis toolkit."""


# Wire in subcommands from their respective modules.
cli.add_command(track_cmd)
cli.add_command(split_rallies_cmd)
cli.add_command(fix_player_ids_cmd)
cli.add_command(run_cmd)
cli.add_command(identify_cmd)
cli.add_command(analyze_cmd)
cli.add_command(compare_cmd)
cli.add_command(annotate_gt_cmd)
cli.add_command(eval_id_cmd)
cli.add_command(serve_cmd)
cli.add_command(render_cmd)
cli.add_command(eval_frame_cmd)
cli.add_command(ball_track_cmd)
cli.add_command(merge_cmd)
cli.add_command(detect_rallies_cmd)
cli.add_command(analytics_render_cmd)
cli.add_command(analytics_cmd)
cli.add_command(stitch_rallies_cmd)

cli.add_command(publish_cmd)