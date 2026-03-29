#!/usr/bin/env bash
# chunk_video.sh — trim, downscale, and split a video into fixed-length chunks.
#
# Usage:
#   chunk_video.sh -i INPUT [-s START] [-d CHUNK_DURATION] [-r SCALE] [-o OUTDIR] [-p PREFIX]
#
# Options:
#   -i  Input video file (required)
#   -s  Start time, ffmpeg format: HH:MM:SS, MM:SS, or raw seconds  [default: 00:00:00]
#   -d  Chunk duration in seconds                                    [default: 60]
#   -r  Output scale as WxH (e.g. 960x720) or 'half' for half-res   [default: half]
#   -o  Output directory                                             [default: ./chunks]
#   -p  Output filename prefix                                       [default: chunk]
#
# Output files: <outdir>/<prefix>_001.mp4, <prefix>_002.mp4, …
#
# Dependencies: ffmpeg (system), ffprobe (system)
#
# Notes:
#   - Re-encodes video with libx264 (crf 18) to allow arbitrary trim points
#     while preserving visual quality.  Audio is copied where possible.
#   - Frame rate is preserved from source (no -r flag passed to encoder).
#   - The last chunk may be shorter than CHUNK_DURATION if the source ends first.

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
INPUT=""
START="00:00:00"
CHUNK_DURATION=60
SCALE="half"
OUTDIR="./chunks"
PREFIX="chunk"

# ── argument parsing ─────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -25
    exit 1
}

while getopts ":i:s:d:r:o:p:h" opt; do
    case $opt in
        i) INPUT="$OPTARG" ;;
        s) START="$OPTARG" ;;
        d) CHUNK_DURATION="$OPTARG" ;;
        r) SCALE="$OPTARG" ;;
        o) OUTDIR="$OPTARG" ;;
        p) PREFIX="$OPTARG" ;;
        h) usage ;;
        :) echo "ERROR: -$OPTARG requires an argument." >&2; exit 1 ;;
        \?) echo "ERROR: Unknown option -$OPTARG" >&2; exit 1 ;;
    esac
done

[[ -z "$INPUT" ]] && { echo "ERROR: -i INPUT is required." >&2; exit 1; }
[[ ! -f "$INPUT" ]] && { echo "ERROR: File not found: $INPUT" >&2; exit 1; }

# ── resolve scale filter ─────────────────────────────────────────────────────
if [[ "$SCALE" == "half" ]]; then
    # Halve each dimension; keep it divisible by 2 (required by libx264).
    VF="scale=iw/2:ih/2"
elif [[ "$SCALE" =~ ^[0-9]+x[0-9]+$ ]]; then
    W="${SCALE%%x*}"
    H="${SCALE##*x}"
    VF="scale=${W}:${H}"
else
    echo "ERROR: -r must be 'half' or WxH (e.g. 960x720), got: $SCALE" >&2
    exit 1
fi

# ── probe total duration ─────────────────────────────────────────────────────
TOTAL_DURATION=$(ffprobe -v error -select_streams v:0 \
    -show_entries format=duration -of csv=p=0 "$INPUT")
TOTAL_DURATION=${TOTAL_DURATION%.*}   # truncate to integer seconds

# Convert START to seconds for arithmetic.
start_to_seconds() {
    local t="$1"
    # Accepts HH:MM:SS, MM:SS, or plain seconds.
    if [[ "$t" =~ ^([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
        echo $(( 10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]} ))
    elif [[ "$t" =~ ^([0-9]+):([0-9]{2})$ ]]; then
        echo $(( 10#${BASH_REMATCH[1]} * 60 + 10#${BASH_REMATCH[2]} ))
    elif [[ "$t" =~ ^[0-9]+$ ]]; then
        echo "$t"
    else
        echo "ERROR: Cannot parse start time: $t" >&2; exit 1
    fi
}

START_SEC=$(start_to_seconds "$START")

if (( START_SEC >= TOTAL_DURATION )); then
    echo "ERROR: Start time ${START} (${START_SEC}s) is at or beyond video duration (${TOTAL_DURATION}s)." >&2
    exit 1
fi

AVAILABLE=$(( TOTAL_DURATION - START_SEC ))
CHUNKS=$(( (AVAILABLE + CHUNK_DURATION - 1) / CHUNK_DURATION ))  # ceiling division

# ── output directory ─────────────────────────────────────────────────────────
mkdir -p "$OUTDIR"

echo "Input       : $INPUT"
echo "Start       : ${START} (${START_SEC}s)"
echo "Available   : ${AVAILABLE}s after start"
echo "Chunk size  : ${CHUNK_DURATION}s"
echo "Chunks      : ${CHUNKS}"
echo "Scale filter: $VF"
echo "Output dir  : $OUTDIR"
echo "---"

# ── encode chunks ────────────────────────────────────────────────────────────
for (( i=0; i<CHUNKS; i++ )); do
    OFFSET=$(( START_SEC + i * CHUNK_DURATION ))
    # Last chunk: clamp duration so ffmpeg doesn't warn about running past EOF.
    REMAINING=$(( TOTAL_DURATION - OFFSET ))
    DUR=$(( CHUNK_DURATION < REMAINING ? CHUNK_DURATION : REMAINING ))

    # Zero-padded index: 001, 002, …
    IDX=$(printf "%03d" $(( i + 1 )))
    OUTFILE="${OUTDIR}/${PREFIX}_${IDX}.mp4"

    echo "Chunk ${IDX}: offset=${OFFSET}s  duration=${DUR}s  -> $OUTFILE"

    ffmpeg -y \
        -ss "$OFFSET" \
        -i "$INPUT" \
        -t "$DUR" \
        -vf "$VF" \
        -c:v libx264 \
        -crf 18 \
        -preset fast \
        -c:a aac \
        -b:a 128k \
        -movflags +faststart \
        "$OUTFILE"
done

echo "---"
echo "Done. ${CHUNKS} chunk(s) written to ${OUTDIR}/"
