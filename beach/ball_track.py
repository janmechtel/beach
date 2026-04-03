"""beach ball-track — Run VballNet ball tracking on a video → *_ball.csv.

Delegates to the inference_onnx_seq_gray_v2.py script in the
fast-volleyball-tracking-inference repo using its own venv, then copies the
resulting CSV next to the source video.

The CSV already has coordinates in original video pixel space (the inference
script rescales X/Y from model-input resolution automatically).

Output CSV columns: Frame, Visibility, X, Y
  Frame       — 0-based frame index matching beach track / beach run output
  Visibility  — 1 = ball detected, 0 = not detected
  X, Y        — ball centre in original video pixels (-1 when Visibility=0)

Usage
-----
    beach ball-track --video videos/GH021569_court.mp4
    beach ball-track --video videos/GH021569_court.mp4 --skip-existing
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import click

# ---------------------------------------------------------------------------
# Repo / model locations — override via env vars for portability
# ---------------------------------------------------------------------------
_FVTI_ROOT = Path(
    os.environ.get(
        "FVTI_ROOT",
        Path(__file__).resolve().parents[2] / "fast-volleyball-tracking-inference",
    )
)
_FVTI_PYTHON = Path(
    os.environ.get("FVTI_PYTHON", _FVTI_ROOT / ".venv" / "bin" / "python")
)
_FVTI_SCRIPT = _FVTI_ROOT / "src" / "inference_onnx_seq_gray_v2.py"
_DEFAULT_MODEL = _FVTI_ROOT / "models" / "VballNetV2_seq9_grayscale_320_h288_w512.onnx"


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def run_ball_tracking(
    video_path: Path,
    output_csv: Path,
    model_path: Path = _DEFAULT_MODEL,
    confidence_threshold: float = 0.5,
    skip_existing: bool = False,
) -> Path:
    """Run VballNet inference on *video_path* and write CSV to *output_csv*.

    Returns the path to the written CSV.
    """
    if skip_existing and output_csv.exists():
        print(f"  ball CSV exists, skipping inference ({output_csv.name})")
        return output_csv

    if not _FVTI_PYTHON.exists():
        raise click.ClickException(
            f"fast-volleyball-tracking-inference Python not found: {_FVTI_PYTHON}\n"
            f"Set FVTI_ROOT or FVTI_PYTHON env vars to the correct path."
        )
    if not _FVTI_SCRIPT.exists():
        raise click.ClickException(f"Inference script not found: {_FVTI_SCRIPT}")
    if not model_path.exists():
        raise click.ClickException(f"Ball tracking model not found: {model_path}")

    print(f"  Running VballNet inference on {video_path.name} …")
    print(f"  Model : {model_path.name}")

    with tempfile.TemporaryDirectory(prefix="beach_ball_") as tmp:
        tmp_dir = Path(tmp)
        cmd = [
            str(_FVTI_PYTHON),
            str(_FVTI_SCRIPT),
            "--video_path", str(video_path),
            "--model_path", str(model_path),
            "--output_dir", str(tmp_dir),
            "--only_csv",
            "--confidence_threshold", str(confidence_threshold),
        ]
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise click.ClickException(
                f"Ball tracking inference failed (exit {result.returncode})."
            )

        # The script writes to <output_dir>/<video_stem>/ball.csv
        produced = tmp_dir / video_path.stem / "ball.csv"
        if not produced.exists():
            raise click.ClickException(
                f"Expected output CSV not found at {produced}. "
                "Check inference script output above."
            )

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, output_csv)

    print(f"  Ball CSV written: {output_csv}  "
          f"({output_csv.stat().st_size / 1024:.1f} KB)")
    return output_csv


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------
@click.command("ball-track")
@click.option(
    "--video", "-v",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input video file.",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output CSV path (default: <video_stem>_ball.csv next to video).",
)
@click.option(
    "--model",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help=f"VballNet ONNX model (default: {_DEFAULT_MODEL.name}).",
)
@click.option(
    "--confidence", "-c",
    default=0.5,
    type=float,
    show_default=True,
    help="Heatmap confidence threshold.",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="Skip if output CSV already exists.",
)
def ball_track_cmd(
    video: Path,
    output: Optional[Path],
    model: Optional[Path],
    confidence: float,
    skip_existing: bool,
) -> None:
    """Run VballNet ball tracking → *_ball.csv next to the video."""
    output_csv = output or video.with_name(video.stem + "_ball.csv")
    model_path = model or _DEFAULT_MODEL
    run_ball_tracking(
        video_path=video,
        output_csv=output_csv,
        model_path=model_path,
        confidence_threshold=confidence,
        skip_existing=skip_existing,
    )
