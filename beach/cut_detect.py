"""Cut detection via frame differencing (OpenCV).

Hard cuts in edited video produce large frame-to-frame pixel differences.
This module implements a simple, fast, parameter-free-by-default detector
that works well for sports footage where each rally is a hard cut with no
dissolves or fades.

Algorithm
---------
1. Sample every `sample_every`-th frame (default 2 — halves I/O cost;
   hard cuts span multiple frames so we cannot miss them).
2. Convert to grayscale; compute mean absolute difference (MAD) between
   consecutive sampled frames.
3. Flag frames where MAD exceeds `threshold` (default 30 / 255).
4. Suppress duplicate detections within `min_gap_sec` seconds
   (default 0.5 s) by keeping the highest-scoring detection in each cluster.
5. Return `CutResult` objects sorted by timestamp.

Tuning
------
- If false positives appear (camera shake, zooms): lower `threshold` or
  enable `use_hist` for histogram correlation as a secondary gate.
- If cuts are missed: raise `threshold`.
- Expose both via the CLI's `--threshold` flag.

The `use_hist` path computes Bhattacharyya distance on normalised
histograms as a second opinion.  A frame is flagged only when *both*
the MAD exceeds `threshold` AND the histogram distance exceeds
`hist_threshold`.  This is conservative and should eliminate motion blur
false positives without sacrificing recall on real hard cuts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

from beach.models import CutResult

logger = logging.getLogger(__name__)

# Sensible defaults derived from empirical experience with sports footage.
DEFAULT_THRESHOLD = 30.0      # MAD (0–255 scale)
DEFAULT_SAMPLE_EVERY = 2      # process every 2nd frame
DEFAULT_MIN_GAP_SEC = 0.5     # suppress duplicates within 0.5 s
DEFAULT_HIST_THRESHOLD = 0.3  # Bhattacharyya distance (0–1 scale)


def _sampled_frames(
    cap: cv2.VideoCapture,
    sample_every: int,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Yield (frame_index, gray_frame) for every `sample_every`-th frame."""
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:
            yield frame_idx, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_idx += 1


def _suppress_duplicates(
    candidates: list[CutResult],
    min_gap_sec: float,
    fps: float,
) -> list[CutResult]:
    """Cluster nearby detections and keep the highest-scoring one per cluster.

    Two detections are in the same cluster if they are within
    `min_gap_sec` seconds of each other.  Within a cluster the one with
    the highest `score` (largest pixel difference) is kept — it is most
    likely the true cut frame.
    """
    if not candidates:
        return []

    result: list[CutResult] = []
    cluster: list[CutResult] = [candidates[0]]

    for cut in candidates[1:]:
        if cut.timestamp - cluster[0].timestamp <= min_gap_sec:
            cluster.append(cut)
        else:
            result.append(max(cluster, key=lambda c: c.score))
            cluster = [cut]

    result.append(max(cluster, key=lambda c: c.score))
    return result


def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance between two grayscale frames.

    Returns a value in [0, 1]; higher = more different.
    Hard cuts typically score > 0.3; camera shake rarely exceeds 0.15.
    """
    hist_a = cv2.calcHist([a], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))


def detect_cuts(
    video_path: Path | str,
    threshold: float = DEFAULT_THRESHOLD,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    min_gap_sec: float = DEFAULT_MIN_GAP_SEC,
    use_hist: bool = False,
    hist_threshold: float = DEFAULT_HIST_THRESHOLD,
) -> list[CutResult]:
    """Detect hard cuts in a video file.

    Parameters
    ----------
    video_path:
        Path to the input video.
    threshold:
        Mean absolute difference (0–255) above which a frame transition
        is considered a cut.  Lower = more sensitive.
    sample_every:
        Process every N-th frame.  ``1`` = every frame (accurate but slow).
        ``2`` = every other frame (recommended default for 30fps footage).
    min_gap_sec:
        Minimum seconds between reported cuts.  Prevents one physical cut
        from generating multiple nearby detections.
    use_hist:
        If True, require *both* MAD > threshold AND histogram Bhattacharyya
        distance > hist_threshold.  More conservative; use when camera motion
        causes false positives.
    hist_threshold:
        Bhattacharyya distance threshold used when ``use_hist=True``.

    Returns
    -------
    list[CutResult]
        Sorted by timestamp.  The returned list does not include implicit
        boundaries at t=0 or t=duration — callers add those separately
        (see ``MatchMetadata.from_cuts``).

    Raises
    ------
    FileNotFoundError
        If the video file does not exist.
    ValueError
        If OpenCV cannot open the file (unsupported format, corruption, etc.)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")

    try:
        fps: float = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            # Some containers don't report FPS; fall back to 30.
            logger.warning("Could not read FPS from %s; assuming 30", video_path.name)
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            "Scanning %s: %.1f fps, %d frames, sample_every=%d, threshold=%.1f",
            video_path.name,
            fps,
            total_frames,
            sample_every,
            threshold,
        )

        candidates: list[CutResult] = []
        prev_gray: np.ndarray | None = None
        prev_idx: int = 0

        for frame_idx, gray in _sampled_frames(cap, sample_every):
            if prev_gray is None:
                prev_gray = gray
                prev_idx = frame_idx
                continue

            mad = float(np.mean(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16))))

            if mad >= threshold:
                if use_hist:
                    dist = _hist_distance(prev_gray, gray)
                    if dist < hist_threshold:
                        prev_gray = gray
                        prev_idx = frame_idx
                        continue

                timestamp = frame_idx / fps
                candidates.append(
                    CutResult(frame=frame_idx, timestamp=timestamp, score=mad)
                )
                logger.debug(
                    "Cut candidate at frame %d (t=%.3f s, MAD=%.1f)",
                    frame_idx,
                    timestamp,
                    mad,
                )

            prev_gray = gray
            prev_idx = frame_idx

    finally:
        cap.release()

    cuts = _suppress_duplicates(candidates, min_gap_sec, fps)
    logger.info("Detected %d cuts (from %d raw candidates)", len(cuts), len(candidates))
    return cuts
