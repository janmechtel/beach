"""Data models for the beach volleyball analysis pipeline.

All models serialise cleanly to/from JSON via Pydantic so downstream
milestones (Gemini analysis, viewer) can consume the same files.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator



# Exhaustive enums for Gemini-controlled fields — any deviation is a contract violation.
PlayerID = Literal["P1", "P2", "P3", "P4"]
ActionType = Literal[
    "Serve",
    "Reception",
    "Set",
    "Attack",
    "Dig",
    "Block",
    "Free Ball Sent",
    "Free Ball Received",
    "Not a touch",
]


class Action(BaseModel):
    """A single player action detected by Gemini in a video segment.

    `player_description` is an optional human-readable label (e.g. 'Denny (black
    tshirt)') added after validation for viewer compatibility; it is not emitted
    by Gemini and not constrained by the schema.
    """

    timestamp_sec: float
    player_id: PlayerID
    action: ActionType
    player_description: str | None = None

    @field_validator("timestamp_sec")
    @classmethod
    def timestamp_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("timestamp_sec must be >= 0")
        return v