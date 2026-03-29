"""Data models for cut detection and match metadata.

All models serialise cleanly to/from JSON via Pydantic so downstream
milestones (Gemini analysis, viewer) can consume the same files.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, field_validator


class CutResult(BaseModel):
    """A single detected cut boundary.

    `frame` is the *first* frame of the new segment (0-indexed).
    `timestamp` is the corresponding time in seconds.
    `score` is the mean absolute pixel difference that triggered detection;
    higher = more abrupt cut.  Stored for post-hoc threshold tuning.
    """

    frame: int
    timestamp: float
    score: float

    @field_validator("frame")
    @classmethod
    def frame_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("frame index must be >= 0")
        return v

    @field_validator("timestamp")
    @classmethod
    def timestamp_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("timestamp must be >= 0")
        return v


class Point(BaseModel):
    """One rally / point in the match.

    `index` is 1-based (human-friendly).
    `start` / `end` are seconds from the beginning of the source video.
    `file` is the filename (not full path) of the extracted clip,
    relative to the output directory — making the metadata portable.
    """

    index: int
    start: float
    end: float
    file: str

    @property
    def duration(self) -> float:
        return self.end - self.start


class MatchMetadata(BaseModel):
    """Top-level metadata for a processed match video.

    Written as ``metadata.json`` alongside the extracted clips.
    """

    source: str          # original filename, no directory component
    match_id: str        # derived from source stem; used as output subdir name
    points: list[Point]

    @classmethod
    def from_cuts(
        cls,
        source_path: Path,
        cuts: list[CutResult],
        total_duration: float,
        match_id: str,
    ) -> "MatchMetadata":
        """Build metadata from a list of cut boundaries.

        The segment before the first cut and after the last cut are both
        included — the very start and end of the video are implicitly
        boundaries.
        """
        # Boundary timestamps: video start + each cut + video end
        boundaries = [0.0] + [c.timestamp for c in cuts] + [total_duration]

        points = [
            Point(
                index=i + 1,
                start=boundaries[i],
                end=boundaries[i + 1],
                file=f"point_{i + 1:03d}.mp4",
            )
            for i in range(len(boundaries) - 1)
        ]

        return cls(
            source=source_path.name,
            match_id=match_id,
            points=points,
        )
