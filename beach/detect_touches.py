"""beach detect_touches — Core touch-detection analysis functions.

These are pure functions operating on a ``frames_list`` (list of merged-frame
dicts).  They are called from ``beach.merge.build_merged`` so that the
``touches`` key is populated as part of the normal merge step.
"""

from __future__ import annotations

import math


# ── tuneable parameters ──────────────────────────────────────────────────────

# Max frame gap between two visible-ball detections to still count as consecutive
MAX_FRAME_GAP = 3

# Minimum ball speed (px/s) on both sides of an event to ignore noise / static ball
MIN_SPEED = 200

# Minimum direction change angle (degrees) to flag as a touch candidate
MIN_ANGLE_DEG = 40

# How many frames before the touch frame to search when assigning a player.
# Guards against bounding-box dropout at the exact contact frame.
PLAYER_LOOKBACK_FRAMES = 5

# ─────────────────────────────────────────────────────────────────────────────


def compute_velocities(frames_list: list[dict], max_gap: int = MAX_FRAME_GAP) -> dict:
    """Return frame_number → (vx, vy, speed, closest_player_id).

    Only computed between pairs of consecutive visible-ball frames within max_gap.
    """
    visible = [
        (f["frame"], f["timestamp_sec"], f["ball"]["x"], f["ball"]["y"], f["closest_player_id"])
        for f in frames_list
        if f["ball"]["visible"]
    ]

    vels: dict = {}
    for i in range(1, len(visible)):
        f0, f1 = visible[i - 1], visible[i]
        frame_gap = f1[0] - f0[0]
        if frame_gap > max_gap:
            continue
        dt = f1[1] - f0[1]
        if dt <= 0:
            continue
        vx = (f1[2] - f0[2]) / dt
        vy = (f1[3] - f0[3]) / dt
        speed = math.hypot(vx, vy)
        vels[f1[0]] = (vx, vy, speed, f1[4])

    return vels


def assign_touch_player(touch_frame: int, frames_list: list[dict],
                         lookback: int = PLAYER_LOOKBACK_FRAMES) -> str | None:
    """Find the player closest to the ball in [touch_frame-lookback .. touch_frame].

    Using a lookback window guards against tracking dropouts at the exact
    contact frame (player bbox occluded by the ball).
    """
    frame_lookup = {f["frame"]: f for f in frames_list}

    best_pid = None
    best_dist = float("inf")

    for fn in range(touch_frame - lookback, touch_frame + 1):
        fr = frame_lookup.get(fn)
        if fr is None or not fr["ball"]["visible"]:
            continue
        bx, by = fr["ball"]["x"], fr["ball"]["y"]
        for p in fr["players"]:
            pid = p.get("player_id")
            if pid is None:
                continue
            dist = math.hypot(p["cx"] - bx, p["cy"] - by)
            if dist < best_dist:
                best_dist = dist
                best_pid = pid

    return best_pid


def detect_touches(frames_list: list[dict], vels: dict,
                   min_speed: float = MIN_SPEED,
                   min_angle: float = MIN_ANGLE_DEG) -> list[dict]:
    """Walk consecutive velocity-bearing frames and flag large direction changes."""
    frame_lookup = {f["frame"]: f for f in frames_list}
    vel_frames = sorted(vels.keys())
    events: list[dict] = []

    for i in range(1, len(vel_frames)):
        fn0 = vel_frames[i - 1]
        fn1 = vel_frames[i]
        if fn1 - fn0 > MAX_FRAME_GAP:
            continue

        vx0, vy0, s0, _ = vels[fn0]
        vx1, vy1, s1, _ = vels[fn1]

        if s0 < min_speed or s1 < min_speed:
            continue

        cos_a = max(-1.0, min(1.0, (vx0 * vx1 + vy0 * vy1) / (s0 * s1)))
        angle = math.degrees(math.acos(cos_a))

        if angle < min_angle:
            continue

        fr = frame_lookup[fn1]
        bx, by = fr["ball"]["x"], fr["ball"]["y"]

        player_dists = {}
        for p in fr["players"]:
            d = math.hypot(bx - p["cx"], by - p["cy"])
            player_dists[p["player_id"]] = round(d, 1)

        closest = assign_touch_player(fn1, frames_list)

        events.append({
            "frame":            fn1,
            "time_sec":         round(fr["timestamp_sec"], 3),
            "ball_x":           round(bx, 1),
            "ball_y":           round(by, 1),
            "angle_change_deg": round(angle, 1),
            "speed_before":     round(s0),
            "speed_after":      round(s1),
            "closest_player":   closest,
            "player_dists":     player_dists,
            "dir_before":       (round(vx0 / s0, 3), round(vy0 / s0, 3)),
            "dir_after":        (round(vx1 / s1, 3), round(vy1 / s1, 3)),
        })

    return sorted(events, key=lambda e: e["frame"])


def deduplicate_events(events: list[dict], min_frame_gap: int = 10) -> list[dict]:
    """Keep only the strongest event within any cluster of nearby frames."""
    if not events:
        return []

    deduped = [events[0]]
    for ev in events[1:]:
        if ev["frame"] - deduped[-1]["frame"] < min_frame_gap:
            if ev["angle_change_deg"] > deduped[-1]["angle_change_deg"]:
                deduped[-1] = ev
        else:
            deduped.append(ev)

    return deduped


def format_touches(events: list[dict]) -> list[dict]:
    """Convert raw event dicts to the compact form stored in *_merged.json."""
    return [
        {
            "player":           ev["closest_player"],
            "frame":            ev["frame"],
            "timestamp_sec":    ev["time_sec"],
            "angle_change_deg": ev["angle_change_deg"],
            "speed_before":     ev["speed_before"],
            "speed_after":      ev["speed_after"],
        }
        for ev in events
    ]


def print_summary(events: list[dict], label: str) -> None:
    print(f"\n  {'─'*66}")
    print(f"  {'#':<4} {'Frame':<7} {'Time':>6}  {'Angle':>7}  {'Spd▶':>6} {'Spd◀':>6}  {'Player'}")
    print(f"  {'─'*66}")
    for i, ev in enumerate(events, 1):
        print(
            f"  {i:<4} {ev['frame']:<7} {ev['time_sec']:>6.2f}s  "
            f"{ev['angle_change_deg']:>6.1f}°  "
            f"{ev['speed_before']:>6}  {ev['speed_after']:>6}  "
            f"{str(ev['closest_player']):<8}"
        )
    print(f"  {'─'*66}")
    print(f"  {len(events)} touch event(s) — {label}")


