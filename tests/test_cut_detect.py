"""Unit tests for cut detection.

The test fixture builds a synthetic MP4 in-process using OpenCV:
- Segment 0:  frames 0–29   solid red   (1 second @ 30fps)
- CUT
- Segment 1:  frames 30–59  solid blue  (1 second)
- CUT
- Segment 2:  frames 60–89  solid green (1 second)

Frame-to-frame difference within a segment is 0.
Frame-to-frame difference at a cut is ~255 (colour channels flip).

This gives two expected cuts: at frame 30 (t≈1.0 s) and frame 60 (t≈2.0 s).

Tests cover:
- Basic detection: correct count and approximate timestamps.
- Suppression: duplicates within min_gap are collapsed.
- Threshold sensitivity: very high threshold misses cuts.
- Invalid path: FileNotFoundError raised.
- MatchMetadata.from_cuts: correct segment count and durations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from beach.cut_detect import detect_cuts
from beach.models import CutResult, MatchMetadata


# ---------------------------------------------------------------------------
# Fixture: synthetic test video
# ---------------------------------------------------------------------------

def _make_test_video(path: Path, fps: int = 30, frames_per_segment: int = 30) -> float:
    """Write a 3-segment synthetic video; return total duration in seconds.

    Segments are solid BGR colours:  red, blue, green.
    Cuts between segments have maximum possible pixel difference.
    """
    width, height = 64, 64  # tiny — fast to write and read

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))

    colours = [
        (0, 0, 255),   # red   (BGR)
        (255, 0, 0),   # blue
        (0, 255, 0),   # green
    ]

    for colour in colours:
        frame = np.full((height, width, 3), colour, dtype=np.uint8)
        for _ in range(frames_per_segment):
            writer.write(frame)

    writer.release()

    total_frames = len(colours) * frames_per_segment
    return total_frames / fps


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, float]:
    """Return (video_path, total_duration) for the synthetic test video."""
    tmp = tmp_path_factory.mktemp("video")
    video_path = tmp / "test_game.mp4"
    duration = _make_test_video(video_path)
    return video_path, duration


# ---------------------------------------------------------------------------
# Tests: detect_cuts
# ---------------------------------------------------------------------------

class TestDetectCuts:
    def test_detects_correct_number_of_cuts(self, synthetic_video):
        video_path, _ = synthetic_video
        cuts = detect_cuts(video_path, threshold=20.0, sample_every=1)
        # Exactly 2 hard cuts (red→blue, blue→green)
        assert len(cuts) == 2

    def test_cut_timestamps_approximately_correct(self, synthetic_video):
        video_path, _ = synthetic_video
        cuts = detect_cuts(video_path, threshold=20.0, sample_every=1)
        # Cuts are at 1.0 s and 2.0 s; allow ±0.1 s tolerance for keyframe rounding
        assert abs(cuts[0].timestamp - 1.0) < 0.1
        assert abs(cuts[1].timestamp - 2.0) < 0.1

    def test_cuts_sorted_by_timestamp(self, synthetic_video):
        video_path, _ = synthetic_video
        cuts = detect_cuts(video_path, threshold=20.0, sample_every=1)
        timestamps = [c.timestamp for c in cuts]
        assert timestamps == sorted(timestamps)

    def test_cut_score_is_positive(self, synthetic_video):
        video_path, _ = synthetic_video
        cuts = detect_cuts(video_path, threshold=20.0, sample_every=1)
        assert all(c.score > 0 for c in cuts)

    def test_threshold_too_high_misses_all_cuts(self, synthetic_video):
        video_path, _ = synthetic_video
        # Threshold of 300 is above the max possible MAD (255) — nothing detected
        cuts = detect_cuts(video_path, threshold=300.0, sample_every=1)
        assert cuts == []

    def test_sample_every_2_still_detects_cuts(self, synthetic_video):
        """Sampling every 2nd frame must not cause cuts to be missed."""
        video_path, _ = synthetic_video
        cuts = detect_cuts(video_path, threshold=20.0, sample_every=2)
        assert len(cuts) == 2

    def test_min_gap_suppression(self, synthetic_video):
        """With a very large min_gap, multiple cuts collapse to one."""
        video_path, _ = synthetic_video
        # 10s gap collapses both cuts into one cluster starting at the first
        cuts = detect_cuts(video_path, threshold=20.0, sample_every=1, min_gap_sec=10.0)
        assert len(cuts) == 1

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            detect_cuts(tmp_path / "nonexistent.mp4")

    def test_invalid_file(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not a video")
        with pytest.raises(ValueError, match="could not open"):
            detect_cuts(bad)

    def test_returns_cut_result_objects(self, synthetic_video):
        video_path, _ = synthetic_video
        cuts = detect_cuts(video_path, threshold=20.0, sample_every=1)
        assert all(isinstance(c, CutResult) for c in cuts)


# ---------------------------------------------------------------------------
# Tests: MatchMetadata.from_cuts
# ---------------------------------------------------------------------------

class TestMatchMetadataFromCuts:
    def test_segment_count(self, synthetic_video):
        video_path, duration = synthetic_video
        cuts = detect_cuts(video_path, threshold=20.0, sample_every=1)
        meta = MatchMetadata.from_cuts(video_path, cuts, duration, "test_game")
        # N cuts → N+1 segments
        assert len(meta.points) == len(cuts) + 1

    def test_first_segment_starts_at_zero(self, synthetic_video):
        video_path, duration = synthetic_video
        cuts = detect_cuts(video_path, threshold=20.0, sample_every=1)
        meta = MatchMetadata.from_cuts(video_path, cuts, duration, "test_game")
        assert meta.points[0].start == 0.0

    def test_last_segment_ends_at_duration(self, synthetic_video):
        video_path, duration = synthetic_video
        cuts = detect_cuts(video_path, threshold=20.0, sample_every=1)
        meta = MatchMetadata.from_cuts(video_path, cuts, duration, "test_game")
        assert abs(meta.points[-1].end - duration) < 1e-6

    def test_segments_are_contiguous(self, synthetic_video):
        """end of segment N must equal start of segment N+1."""
        video_path, duration = synthetic_video
        cuts = detect_cuts(video_path, threshold=20.0, sample_every=1)
        meta = MatchMetadata.from_cuts(video_path, cuts, duration, "test_game")
        for a, b in zip(meta.points, meta.points[1:]):
            assert abs(a.end - b.start) < 1e-9

    def test_point_indices_are_1_based(self, synthetic_video):
        video_path, duration = synthetic_video
        cuts = detect_cuts(video_path, threshold=20.0, sample_every=1)
        meta = MatchMetadata.from_cuts(video_path, cuts, duration, "test_game")
        assert [p.index for p in meta.points] == list(range(1, len(meta.points) + 1))

    def test_filenames_follow_convention(self, synthetic_video):
        video_path, duration = synthetic_video
        cuts = detect_cuts(video_path, threshold=20.0, sample_every=1)
        meta = MatchMetadata.from_cuts(video_path, cuts, duration, "test_game")
        for point in meta.points:
            assert point.file == f"point_{point.index:03d}.mp4"

    def test_source_is_filename_only(self, synthetic_video):
        """source must be the filename, not the full path."""
        video_path, duration = synthetic_video
        cuts = detect_cuts(video_path, threshold=20.0, sample_every=1)
        meta = MatchMetadata.from_cuts(video_path, cuts, duration, "test_game")
        assert meta.source == video_path.name
        assert "/" not in meta.source

    def test_json_roundtrip(self, synthetic_video):
        """Metadata must survive a JSON serialise → deserialise cycle."""
        import json

        video_path, duration = synthetic_video
        cuts = detect_cuts(video_path, threshold=20.0, sample_every=1)
        meta = MatchMetadata.from_cuts(video_path, cuts, duration, "test_game")

        raw = json.loads(meta.model_dump_json())
        restored = MatchMetadata.model_validate(raw)

        assert restored == meta

    def test_no_cuts_produces_single_point(self, synthetic_video):
        """If no cuts are detected, the whole video is one point."""
        video_path, duration = synthetic_video
        meta = MatchMetadata.from_cuts(video_path, [], duration, "test_game")
        assert len(meta.points) == 1
        assert meta.points[0].start == 0.0
        assert abs(meta.points[0].end - duration) < 1e-6
