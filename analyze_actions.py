# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "google-genai",
#   "opencv-python-headless",
#   "pydantic",
# ]
# ///
"""
Analyze volleyball actions in first30.mp4 using Gemini video API.

Uploads the file, waits for processing, then prompts for structured JSON output.
Player identification uses up to three mechanisms depending on the model:
  1. Gemini response_schema with enum constraints (P1–P4, valid actions only).
  2. A seed frame (extracted with OpenCV) sent before the video so Gemini can
     anchor player identities to their visual appearance (pro models only).
  3. Team structure and sequencing rules in the prompt to reduce swap errors.

Upload caching:
  The uploaded Gemini file URI is persisted in output/.gemini_file_cache.json.
  On subsequent runs the cached file is reused if still ACTIVE (Gemini keeps
  uploaded files for ~48 h).  Re-upload happens automatically when the cache
  is missing, expired, or the file has left ACTIVE state.

Timestamp-seeded mode (--input):
  Pass a reference JSON (e.g. output/first30_Manual.json) to fix the timestamps
  and ask the model to classify only player_id + action at each moment.  This
  lets you run the same clip multiple times and measure whether identities
  converge.

Convergence mode (--runs N):
  Combines with --input to run N analyses on the same cached upload and print
  an agreement table showing how often each model run agrees on player_id and
  action per timestamp.
"""

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from compare_actions import compare, DEFAULT_REF, DEFAULT_TOL

import cv2
from beach.models import Action
from google import genai
from google.genai import types
from pydantic import ValidationError

VIDEO_PATH = Path("data/first30.mp4")
ANNOTATED_VIDEO_PATH = Path("data/first30_annotated.mp4")
PLAYERS_PATH = Path("output/players.json")
FILE_CACHE_PATH = Path("output/.gemini_file_cache.json")
API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyA1QrfSj0xqtlXnJzqJ12rHTJw0FVgvXp8")
_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview",
    "gemini-2.0-flash",
]


def pick_model() -> str:
    print("Select model:")
    for i, m in enumerate(_MODELS, 1):
        print(f"  {i}. {m}")
    while True:
        choice = input(f"Choice [1-{len(_MODELS)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(_MODELS):
            return _MODELS[int(choice) - 1]
        print("  Invalid — enter a number from the list.")


# Gemini schema: enum constraints prevent ID-suffix drift at the API level.
_ACTION_SCHEMA = types.Schema(
    type="ARRAY",
    items=types.Schema(
        type="OBJECT",
        properties={
            "timestamp_sec": types.Schema(type="NUMBER"),
            "player_id": types.Schema(
                type="STRING",
                enum=["P1", "P2", "P3", "P4"],
            ),
            "action": types.Schema(
                type="STRING",
                enum=[
                    "Serve",
                    "Reception",
                    "Set",
                    "Attack",
                    "Dig",
                    "Block",
                    "Free Ball Sent",
                    "Free Ball Received",
                ],
            ),
        },
        required=["timestamp_sec", "player_id", "action"],
    ),
)


def load_players(path: Path) -> dict[str, dict]:
    """Return the full player info dict keyed by player ID."""
    return json.loads(path.read_text())


def extract_seed_frame(video_path: Path, timestamp_sec: float = 4.0) -> bytes:
    """Extract a single JPEG frame from the video for player identification.

    Seeks to `timestamp_sec` so all four players are likely in frame.
    Returns raw JPEG bytes suitable for types.Part.from_bytes().
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(
            f"Could not extract frame at {timestamp_sec}s from {video_path}"
        )
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def make_seed_label(players: dict[str, dict]) -> str:
    """Build the reference-frame caption sent alongside the seed image.

    Describes each player's visual identity and starting position so Gemini
    can anchor IDs before watching the video.
    """
    lines = ["Reference frame showing all four players:"]
    for pid in ["P1", "P2", "P3", "P4"]:
        p = players[pid]
        lines.append(
            f"  {pid} ({p['name']}, {p['description']}): {p['position']}"
        )
    lines.append(
        "Use these visual identities consistently throughout the entire video."
    )
    return "\n".join(lines)


def make_prompt(players: dict[str, dict], annotated: bool = False) -> str:
    """Build the action-extraction prompt, including team structure and rules.

    Builds dynamically from players.json so the prompt stays in sync with the
    data file — no duplicated constants.
    """
    # Roster
    roster_lines = []
    for pid in ["P1", "P2", "P3", "P4"]:
        p = players[pid]
        roster_lines.append(
            f"  {pid}: {p['name']} — {p['description']}"
        )
    roster = "\n".join(roster_lines)

    # Teams derived from players.json "team" field
    team_a = [pid for pid, p in players.items() if p.get("team") == "A"]
    team_b = [pid for pid, p in players.items() if p.get("team") == "B"]
    team_a_names = " + ".join(
        f"{pid} ({players[pid]['name']})" for pid in sorted(team_a)
    )
    team_b_names = " + ".join(
        f"{pid} ({players[pid]['name']})" for pid in sorted(team_b)
    )

    annotation_notice = (
        "IMPORTANT: This video has coloured bounding boxes drawn around each player "
        "with their ID and name label (e.g. 'P1 Denny'). A bright yellow circle marks "
        "the ball when detected. Use these visual annotations to identify which player "
        "performs each action — the labels are ground truth.\n\n"
    ) if annotated else ""

    return f"""
{annotation_notice}You are a professional beach volleyball analyst. Watch this video carefully and identify every discrete player action.

Players — use ONLY the IDs P1, P2, P3, P4:
{roster}

Teams:
  Team A: {team_a_names}
  Team B: {team_b_names}

Volleyball sequencing rules (use to resolve ambiguous player identity):
- The team that did NOT serve makes the first contact (Reception) after each serve.
- Teams alternate ball possession across the net.
- Each team may make at most 3 contacts before sending the ball over.

For each action output a JSON object with exactly these fields:
- "timestamp_sec": float — when the action occurs (seconds from video start, e.g. 3.5)
- "player_id": string — must be exactly one of: P1, P2, P3, P4
- "action": string — must be exactly one of:
    Serve        — from outside the field; usually the player furthest from the net
    Reception    — after a serve or attack
    Set          — with hands or forearms, preparing for an attack
    Attack       — spike, smash, poke shot, rainbow, or cut shot
    Dig          — defending after an attack
    Block        — one or two arms close to the net blocking the ball
    Free Ball Sent     — easy ball sent over for lack of options
    Free Ball Received — very easy reception

Be exhaustive — capture every contact with the ball.
"""


def make_seeded_prompt(players: dict[str, dict], timestamps: list[float], annotated: bool = False) -> str:
    """Build a timestamp-seeded prompt when key moments are already known.

    The model only needs to identify who performed the action and what it was
    at each given timestamp — it does NOT discover new events.  This removes
    timing uncertainty from the analysis so repeated runs can be compared on
    player identity and action type alone.
    """
    # Roster
    roster_lines = []
    for pid in ["P1", "P2", "P3", "P4"]:
        p = players[pid]
        roster_lines.append(
            f"  {pid}: {p['name']} — {p['description']}"
        )
    roster = "\n".join(roster_lines)

    team_a = [pid for pid, p in players.items() if p.get("team") == "A"]
    team_b = [pid for pid, p in players.items() if p.get("team") == "B"]
    team_a_names = " + ".join(
        f"{pid} ({players[pid]['name']})" for pid in sorted(team_a)
    )
    team_b_names = " + ".join(
        f"{pid} ({players[pid]['name']})" for pid in sorted(team_b)
    )

    ts_list = "\n".join(f"  {t}" for t in timestamps)

    annotation_notice = (
        "IMPORTANT: This video has coloured bounding boxes drawn around each player "
        "with their ID and name label (e.g. 'P1 Denny'). A bright yellow circle marks "
        "the ball when detected. Use these visual annotations to identify which player "
        "performs each action — the labels are ground truth.\n\n"
    ) if annotated else ""

    return f"""
{annotation_notice}You are a professional beach volleyball analyst.

Players — use ONLY the IDs P1, P2, P3, P4:
{roster}

Teams:
  Team A: {team_a_names}
  Team B: {team_b_names}

Volleyball sequencing rules (use to resolve ambiguous player identity):
- The team that did NOT serve makes the first contact (Reception) after each serve.
- Teams alternate ball possession across the net.
- Each team may make at most 3 contacts before sending the ball over.

The following timestamps (seconds from video start) are the EXACT moments where
a ball contact occurs.  For EACH timestamp, watch that moment in the video and
identify the player and action type.

Timestamps to classify:
{ts_list}

For each timestamp output a JSON object with exactly these fields:
- "timestamp_sec": float — copy the timestamp exactly as given above
- "player_id": string — must be exactly one of: P1, P2, P3, P4
- "action": string — must be exactly one of:
    Serve        — from outside the field; usually the player furthest from the net
    Reception    — after a serve or attack
    Set          — with hands or forearms, preparing for an attack
    Attack       — spike, smash, poke shot, rainbow, or cut shot
    Dig          — defending after an attack
    Block        — one or two arms close to the net blocking the ball
    Free Ball Sent     — easy ball sent over for lack of options
    Free Ball Received — very easy reception

Return exactly {len(timestamps)} objects — one per timestamp, in order.
Do NOT add or remove any timestamps.
"""


# ---------------------------------------------------------------------------
# Gemini file cache
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    """Return the full cache dict (keyed by video filename stem).

    Backward-compatible: if the cache file contains the old flat format
    (with a top-level 'name' key) it is migrated in-memory to the new format.
    The migration is not written back to disk until the next save, which keeps
    the old entry alive for the raw video without an immediate re-upload.
    """
    if not FILE_CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(FILE_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    # Migrate old flat format: {"name": ..., "uri": ..., "expires_at": ...}
    if "name" in raw and "uri" in raw:
        return {VIDEO_PATH.stem: raw}
    return raw


def _save_cache(video_path: Path, file: types.File) -> None:
    """Persist the Gemini file entry for `video_path` under its stem key."""
    FILE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache()
    expires_at = getattr(file, "expiration_time", None)
    cache[video_path.stem] = {
        "name": file.name,
        "uri": file.uri,
        "expires_at": str(expires_at),
    }
    FILE_CACHE_PATH.write_text(json.dumps(cache, indent=2))

def get_or_upload_file(client: genai.Client, path: Path) -> types.File:
    """Return a ready Gemini File, reusing a cached upload when possible.

    The cache stores the Gemini file name and URI keyed by video filename stem.
    On each call we verify the file is still ACTIVE via files.get() before
    reusing it.  Any failure (expired, deleted, wrong state) triggers a fresh
    upload and cache update.
    """
    cache = _load_cache()
    entry = cache.get(path.stem)
    if entry:
        try:
            cached_file = client.files.get(name=entry["name"])
            if cached_file.state.name == "ACTIVE":
                print(f"Reusing cached Gemini file: {cached_file.uri}")
                return cached_file
            else:
                print(f"Cached file state is {cached_file.state.name} — re-uploading.")
        except Exception as exc:
            print(f"Cache lookup failed ({exc}) — re-uploading.")

    file = _upload_and_wait(client, path)
    _save_cache(path, file)
    return file

def _upload_and_wait(client: genai.Client, path: Path) -> types.File:
    print(f"Uploading {path} ({path.stat().st_size / 1024:.0f} KB)...")
    file = client.files.upload(file=path)
    print(f"  File URI: {file.uri}  state: {file.state}")

    # Poll until ACTIVE (video files need processing time)
    while file.state.name == "PROCESSING":
        time.sleep(3)
        file = client.files.get(name=file.name)
        print(f"  ...still processing: {file.state}")

    if file.state.name != "ACTIVE":
        raise RuntimeError(f"File upload failed with state: {file.state}")

    print(f"  Ready: {file.uri}")
    return file


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(
    client: genai.Client,
    file: types.File,
    players: dict[str, dict],
    seed_bytes: bytes | None,
    model: str,
    timestamps: list[float] | None = None,
    annotated: bool = False,
) -> tuple[list[Action], types.GenerateContentResponseUsageMetadata | None]:
    """Send the video (and optionally a seed frame) to Gemini and parse the response.

    When `timestamps` is provided the prompt asks the model to classify only
    those moments; otherwise it performs free discovery.

    The response_schema forces Gemini to emit valid JSON with only the four
    allowed player IDs and the eight allowed action strings — format drift is
    impossible.  Pydantic then validates each object as an Action, catching any
    schema mismatch that would otherwise silently corrupt downstream consumers.
    """
    if timestamps is not None:
        prompt = make_seeded_prompt(players, timestamps, annotated=annotated)
    else:
        prompt = make_prompt(players, annotated=annotated)

    print(f"Sending to {model}...")
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    # Seed frame — visual anchor for player identification (pro models only)
                    *(  # type: ignore[misc]
                        [
                            types.Part.from_bytes(data=seed_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(text=make_seed_label(players)),
                        ]
                        if seed_bytes is not None
                        else []
                    ),
                    types.Part.from_uri(file_uri=file.uri, mime_type="video/mp4"),
                    types.Part.from_text(text=prompt),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,  # deterministic for structured extraction
            response_mime_type="application/json",
            response_schema=_ACTION_SCHEMA,
        ),
    )

    raw_json = json.loads(response.text)

    # Validate every object; fail loudly rather than silently pass bad data.
    try:
        actions = [Action(**obj) for obj in raw_json]
    except ValidationError as exc:
        raise RuntimeError(
            f"Gemini response failed Action validation:\n{exc}\n\nRaw response:\n{response.text}"
        ) from exc

    # Enrich with human-readable player descriptions for viewer compatibility.
    for action in actions:
        p = players.get(action.player_id, {})
        action.player_description = f"{p.get('name', action.player_id)} ({p.get('description', '')})"

    return actions, response.usage_metadata


# ---------------------------------------------------------------------------
# Convergence reporting
# ---------------------------------------------------------------------------

def _convergence_report(all_runs: list[list[Action]], timestamps: list[float]) -> None:
    """Print a table showing per-timestamp agreement across runs.

    For each timestamp shows which player_id and action each run returned, plus
    a consensus (the majority value) and the agreement rate.
    """
    n = len(all_runs)
    print(f"\n{'═' * 72}")
    print(f"  CONVERGENCE REPORT — {n} runs")
    print(f"{'═' * 72}")
    print(f"  {'ts':>5}  {'consensus player':>17}  {'agr':>4}  {'consensus action':>18}  {'agr':>4}")
    print(f"  {'─'*5}  {'─'*17}  {'─'*4}  {'─'*18}  {'─'*4}")

    for ts in timestamps:
        players_at_ts: list[str] = []
        actions_at_ts: list[str] = []

        for run in all_runs:
            # Find the action in this run closest to the target timestamp
            # (in seeded mode they should be exact, but tolerate float drift)
            match = min(run, key=lambda a: abs(a.timestamp_sec - ts), default=None)
            if match and abs(match.timestamp_sec - ts) <= 0.5:
                players_at_ts.append(match.player_id)
                actions_at_ts.append(match.action)

        if not players_at_ts:
            print(f"  {ts:>5.1f}  {'(no data)':>17}  {'—':>4}  {'(no data)':>18}  {'—':>4}")
            continue

        player_counter = Counter(players_at_ts)
        action_counter = Counter(actions_at_ts)
        best_player, player_votes = player_counter.most_common(1)[0]
        best_action, action_votes = action_counter.most_common(1)[0]
        player_agr = f"{player_votes}/{len(players_at_ts)}"
        action_agr = f"{action_votes}/{len(actions_at_ts)}"

        print(f"  {ts:>5.1f}  {best_player:>17}  {player_agr:>4}  {best_action:>18}  {action_agr:>4}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze beach volleyball actions using Gemini video API.",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        metavar="JSON",
        help=(
            "Reference JSON file (e.g. output/first30_Manual.json) whose timestamps "
            "are fed to the model.  The model classifies player_id + action only — "
            "it does not discover new events.  Enables convergence comparison."
        ),
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=1,
        metavar="N",
        help="Number of analysis runs on the same cached upload (default: 1). "
             "Use with --input to measure convergence.",
    )
    parser.add_argument(
        "--annotated", "-a",
        action="store_true",
        default=False,
        help=(
            "Use the YOLO-annotated video (data/first30_annotated.mp4) instead of "
            "the raw video.  Run annotate_video.py first to generate it.  The prompt "
            "is updated to tell Gemini about the bounding-box labels."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = pick_model()
    players = load_players(PLAYERS_PATH)
    client = genai.Client(api_key=API_KEY)

    # Load timestamps from input file when given
    timestamps: list[float] | None = None
    if args.input is not None:
        ref_events: list[dict] = json.loads(args.input.read_text())
        timestamps = [ev["timestamp_sec"] for ev in ref_events]
        print(f"Seeded mode: {len(timestamps)} timestamps from {args.input}")

    # Select video source — annotated or raw
    video_path = ANNOTATED_VIDEO_PATH if args.annotated else VIDEO_PATH
    if args.annotated and not video_path.exists():
        raise SystemExit(
            f"{video_path} not found.  Run `uv run annotate_video.py` first."
        )

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    print("=== PROMPT ===")
    if timestamps is not None:
        print(make_seeded_prompt(players, timestamps, annotated=args.annotated))
    else:
        print(make_prompt(players, annotated=args.annotated))
    print("=== END ===")

    use_seed = "pro" in model
    seed_bytes: bytes | None = None
    if use_seed:
        # Always extract seed frame from the raw video (consistent visual anchor)
        seed_bytes = extract_seed_frame(VIDEO_PATH)
        print(f"Seed frame extracted: {len(seed_bytes):,} bytes")

    # Reuse cached Gemini file if still ACTIVE — no re-upload for repeated runs.
    file = get_or_upload_file(client, video_path)

    all_runs: list[list[Action]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_thoughts_tokens = 0

    for run_idx in range(args.runs):
        if args.runs > 1:
            print(f"\n--- Run {run_idx + 1}/{args.runs} ---")

        actions, usage = analyze(
            client, file, players, seed_bytes, model, timestamps,
            annotated=args.annotated,
        )

        if usage:
            total_input_tokens += usage.prompt_token_count or 0
            total_output_tokens += usage.candidates_token_count or 0
            total_thoughts_tokens += usage.thoughts_token_count or 0

        all_runs.append(actions)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = video_path.stem  # first30 or first30_annotated
        model_tag = model.replace("/", "-")
        run_tag = f"_run{run_idx + 1}" if args.runs > 1 else ""
        mode_tag = "_seeded" if timestamps is not None else ""
        output_path = Path(f"output/{stem}_{model_tag}{mode_tag}{run_tag}_{timestamp}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = [a.model_dump(exclude_none=False) for a in actions]
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nFound {len(actions)} actions → {output_path}")
        print(json.dumps(output_data, indent=2))

        # Compare against manual ground truth after each run
        if DEFAULT_REF.exists():
            compare(DEFAULT_REF, output_path, DEFAULT_TOL)
        else:
            print(f"\n(Skipping comparison — reference not found: {DEFAULT_REF})")

    if args.runs > 1:
        total_tokens = total_input_tokens + total_output_tokens + total_thoughts_tokens
        print(
            f"\nTotal tokens across {args.runs} runs — "
            f"input: {total_input_tokens:,}  output: {total_output_tokens:,}  "
            f"thoughts: {total_thoughts_tokens:,}  total: {total_tokens:,}"
        )

    if timestamps is not None and args.runs > 1:
        _convergence_report(all_runs, timestamps)


if __name__ == "__main__":
    main()
