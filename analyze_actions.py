"""
Analyze volleyball actions in first30.mp4 using Gemini video API.
Uploads the file, waits for processing, then prompts for structured JSON output.
"""

import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types

VIDEO_PATH = Path("data/first30.mp4")
API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyA1QrfSj0xqtlXnJzqJ12rHTJw0FVgvXp8")

PROMPT = """
You are a beach volleyball analyst. Watch this video carefully and identify every discrete player action.

For each action, output a JSON object with exactly these fields:
- "timestamp_sec": float — when the action occurs (seconds from video start, e.g. 3.5)
- "player_id": string — identify the player by their jersey number, color, or position (e.g. "P1_yellow", "P2_blue", "P3_red", "P4_white"). Use consistent IDs across the whole video.
- "action": string — must be exactly one of:
    Serve
    Reception
    Set
    Attack
    Dig
    Block
    Free Ball Sent
    Free Ball Received

Return ONLY a JSON array of these objects, no prose, no markdown fences.
Example:
[
  {"timestamp_sec": 1.2, "player_id": "P1_yellow", "action": "Serve"},
  {"timestamp_sec": 2.8, "player_id": "P3_blue", "action": "Reception"}
]

Be exhaustive — capture every contact with the ball.
"""


def upload_and_wait(client: genai.Client, path: Path) -> types.File:
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


def analyze(client: genai.Client, file: types.File) -> list[dict]:
    print("Sending to Gemini 2.5 Flash...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=file.uri, mime_type="video/mp4"),
                    types.Part.from_text(text=PROMPT),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,  # deterministic for structured extraction
        ),
    )

    raw = response.text.strip()
    # Strip markdown fences if model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()

    return json.loads(raw)


def main() -> None:
    client = genai.Client(api_key=API_KEY)

    file = upload_and_wait(client, VIDEO_PATH)

    try:
        actions = analyze(client, file)
    finally:
        # Clean up uploaded file to avoid storage accumulation
        client.files.delete(name=file.name)
        print(f"  Deleted remote file {file.name}")

    output_path = Path("output/actions.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(actions, indent=2))

    print(f"\nFound {len(actions)} actions → {output_path}")
    print(json.dumps(actions, indent=2))


if __name__ == "__main__":
    main()
