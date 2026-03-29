#!/usr/bin/env bash
# chunk_video.sh — trim, crop, downscale, and optionally split a video.
#
# Usage:
#   chunk_video.sh -i INPUT [-s START] [-d CHUNK_DURATION] [-r SCALE] \
#                  [-C top:bottom:left:right] [-o OUTDIR] [-p PREFIX]
#
# Options:
#   -i  Input video file (required)
#   -s  Start time: HH:MM:SS, MM:SS, or raw seconds          [default: 00:00:00]
#   -d  Chunk duration in seconds; 0 = single output file    [default: 60]
#   -r  Output scale: WxH (e.g. 1280x720), 'half', or 'none' [default: none]
#   -C  Crop margins as percentages: top:bottom:left:right   [default: no crop]
#       Example: 33:15:10:10  removes 33% top, 15% bottom, 10% each side
#       Tested default for GoPro overhead court cam (1920x1440): 33:15:10:10
#       → crops to 1536x748, eliminates ceiling/near-baseline dead zones and side walls
#   -o  Output directory                                      [default: ./chunks]
#   -p  Output filename prefix                                [default: chunk]
#
# Output:
#   Single file (-d 0): <outdir>/<prefix>.mp4
#   Chunks      (-d N): <outdir>/<prefix>_001.mp4, _002.mp4, …
#
# Dependencies: ffmpeg ≥4, ffprobe (system install)
#
# Notes:
#   - Input seek (-ss before -i) is fast: ffmpeg seeks in the container without
#     decoding skipped frames. Re-encode is required because trim points are
#     unlikely to land on keyframes.
#   - Crop percentages are computed against the source frame dimensions probed
#     at runtime, so the filter is pixel-exact regardless of source resolution.
#   - libx264 requires width and height divisible by 2; the crop calculation
#     rounds outward (shrinks the crop region) to satisfy this constraint.
#   - CRF 18 is visually near-lossless for sports content. Use 23 for ~40%
#     smaller files if fine detail is not needed.

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
INPUT=""
START="03:20"
CHUNK_DURATION=0
SCALE="none"
CROP="33:15:10:10"  # GoPro overhead court cam (1920x1440) → 1536x748
OUTDIR="./chunks"
PREFIX="GH021569_court"

# ── argument parsing ──────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -32
    exit 1
}

while getopts ":i:s:d:r:C:o:p:h" opt; do
    case $opt in
        i) INPUT="$OPTARG" ;;
        s) START="$OPTARG" ;;
        d) CHUNK_DURATION="$OPTARG" ;;
        r) SCALE="$OPTARG" ;;
        C) CROP="$OPTARG" ;;
        o) OUTDIR="$OPTARG" ;;
        p) PREFIX="$OPTARG" ;;
        h) usage ;;
        :) echo "ERROR: -$OPTARG requires an argument." >&2; exit 1 ;;
        \?) echo "ERROR: Unknown option -$OPTARG" >&2; exit 1 ;;
    esac
done

[[ -z "$INPUT" ]] && { echo "ERROR: -i INPUT is required." >&2; exit 1; }
[[ ! -f "$INPUT" ]] && { echo "ERROR: File not found: $INPUT" >&2; exit 1; }

# ── probe source dimensions and duration ──────────────────────────────────────
read -r SRC_W SRC_H < <(ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height -of csv=p=0 "$INPUT" | tr ',' ' ')

TOTAL_DURATION=$(ffprobe -v error -select_streams v:0 \
    -show_entries format=duration -of csv=p=0 "$INPUT")
TOTAL_DURATION=${TOTAL_DURATION%.*}   # truncate to integer seconds

# ── resolve crop filter ───────────────────────────────────────────────────────
# crop=w:h:x:y  (ffmpeg convention: x,y is top-left origin)
CROP_FILTER=""
if [[ -n "$CROP" ]]; then
    # Validate format: four colon-separated integers
    if ! [[ "$CROP" =~ ^([0-9]+):([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        echo "ERROR: -C must be top:bottom:left:right as integers (e.g. 25:20:5:5)." >&2
        exit 1
    fi
    PCT_TOP="${BASH_REMATCH[1]}"
    PCT_BOT="${BASH_REMATCH[2]}"
    PCT_LEFT="${BASH_REMATCH[3]}"
    PCT_RIGHT="${BASH_REMATCH[4]}"

    # Pixel offsets (floor division keeps us inside the frame)
    CX=$(( SRC_W * PCT_LEFT  / 100 ))
    CY=$(( SRC_H * PCT_TOP   / 100 ))
    CW=$(( SRC_W - CX - SRC_W * PCT_RIGHT / 100 ))
    CH=$(( SRC_H - CY - SRC_H * PCT_BOT   / 100 ))

    # libx264 requires even dimensions — round down to nearest even pixel
    CW=$(( CW & ~1 ))
    CH=$(( CH & ~1 ))

    CROP_FILTER="crop=${CW}:${CH}:${CX}:${CY}"
    echo "Crop        : ${PCT_TOP}%T/${PCT_BOT}%B/${PCT_LEFT}%L/${PCT_RIGHT}%R  →  ${CW}x${CH}+${CX}+${CY}"
fi

# ── resolve scale filter ──────────────────────────────────────────────────────
SCALE_FILTER=""
if [[ "$SCALE" == "none" ]]; then
    : # no scaling
elif [[ "$SCALE" == "half" ]]; then
    # Halve post-crop dimensions; iw/ih refer to the filter input at this stage.
    SCALE_FILTER="scale=iw/2:ih/2"
elif [[ "$SCALE" =~ ^[0-9]+x[0-9]+$ ]]; then
    W="${SCALE%%x*}"
    H="${SCALE##*x}"
    SCALE_FILTER="scale=${W}:${H}"
else
    echo "ERROR: -r must be 'none', 'half', or WxH (e.g. 960x720), got: $SCALE" >&2
    exit 1
fi

# Combine crop + scale into a single -vf string (empty filters omitted)
VF_PARTS=()
[[ -n "$CROP_FILTER"  ]] && VF_PARTS+=("$CROP_FILTER")
[[ -n "$SCALE_FILTER" ]] && VF_PARTS+=("$SCALE_FILTER")

if (( ${#VF_PARTS[@]} > 0 )); then
    VF=$(IFS=,; echo "${VF_PARTS[*]}")
    VF_ARGS=(-vf "$VF")
else
    VF_ARGS=()
fi

# ── convert START to seconds ──────────────────────────────────────────────────
start_to_seconds() {
    local t="$1"
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
    echo "ERROR: Start ${START} (${START_SEC}s) is at or beyond video duration (${TOTAL_DURATION}s)." >&2
    exit 1
fi

AVAILABLE=$(( TOTAL_DURATION - START_SEC ))

# ── output directory ──────────────────────────────────────────────────────────
mkdir -p "$OUTDIR"

# ── shared ffmpeg encode function ─────────────────────────────────────────────
# Usage: encode_segment OFFSET DURATION OUTFILE
encode_segment() {
    local offset="$1" dur="$2" outfile="$3"
    ffmpeg -y \
        -ss "$offset" \
        -i "$INPUT" \
        -t "$dur" \
        "${VF_ARGS[@]}" \
        -c:v libx264 \
        -crf 18 \
        -preset fast \
        -c:a aac \
        -b:a 128k \
        -movflags +faststart \
        "$outfile"
}

# ── single-file mode (CHUNK_DURATION == 0) ────────────────────────────────────
if (( CHUNK_DURATION == 0 )); then
    OUTFILE="${OUTDIR}/${PREFIX}.mp4"
    echo "Input       : $INPUT  (${SRC_W}x${SRC_H}, ${TOTAL_DURATION}s)"
    echo "Start       : ${START} (${START_SEC}s)"
    echo "Duration    : ${AVAILABLE}s"
    [[ -n "${VF_ARGS[*]+set}" && ${#VF_ARGS[@]} -gt 0 ]] && echo "Filters     : $VF"
    echo "Output      : $OUTFILE"
    echo "---"
    encode_segment "$START_SEC" "$AVAILABLE" "$OUTFILE"
    echo "---"
    echo "Done. Written to $OUTFILE"
    exit 0
fi

# ── chunk mode ────────────────────────────────────────────────────────────────
CHUNKS=$(( (AVAILABLE + CHUNK_DURATION - 1) / CHUNK_DURATION ))  # ceiling

echo "Input       : $INPUT  (${SRC_W}x${SRC_H}, ${TOTAL_DURATION}s)"
echo "Start       : ${START} (${START_SEC}s)"
echo "Available   : ${AVAILABLE}s"
echo "Chunk size  : ${CHUNK_DURATION}s"
echo "Chunks      : ${CHUNKS}"
[[ ${#VF_ARGS[@]} -gt 0 ]] && echo "Filters     : $VF"
echo "Output dir  : $OUTDIR"
echo "---"

for (( i=0; i<CHUNKS; i++ )); do
    OFFSET=$(( START_SEC + i * CHUNK_DURATION ))
    REMAINING=$(( TOTAL_DURATION - OFFSET ))
    DUR=$(( CHUNK_DURATION < REMAINING ? CHUNK_DURATION : REMAINING ))
    IDX=$(printf "%03d" $(( i + 1 )))
    OUTFILE="${OUTDIR}/${PREFIX}_${IDX}.mp4"
    echo "Chunk ${IDX}: offset=${OFFSET}s  duration=${DUR}s  -> $OUTFILE"
    encode_segment "$OFFSET" "$DUR" "$OUTFILE"
done

echo "---"
echo "Done. ${CHUNKS} chunk(s) written to ${OUTDIR}/"
