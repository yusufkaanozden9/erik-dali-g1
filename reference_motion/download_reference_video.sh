#!/usr/bin/env bash
# Optional: (re)download and trim the reference clip on a fresh machine (e.g. the GPU box).
# For this project's current clip, the source video was supplied directly by the user
# (from ~/Downloads) rather than downloaded here — see NOTES.md. This script exists so the
# same clip (or a different source, via $1) can be reproduced/re-trimmed elsewhere.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$ROOT_DIR/raw"
mkdir -p "$RAW_DIR"

URL="${1:-}"
START="${2:-00:04:53}"
END="${3:-00:05:16}"
OUT_NAME="${4:-erik_dali_clip.mp4}"

if [ -z "$URL" ]; then
  cat <<'EOF'
Usage: ./download_reference_video.sh <youtube_url> [start=00:04:53] [end=00:05:16] [out_name=erik_dali_clip.mp4]

Requires yt-dlp and ffmpeg on PATH.
EOF
  exit 1
fi

command -v yt-dlp >/dev/null 2>&1 || { echo "[error] yt-dlp not found (pip install yt-dlp)" >&2; exit 1; }
command -v ffmpeg  >/dev/null 2>&1 || { echo "[error] ffmpeg not found" >&2; exit 1; }

TMP_FULL="$RAW_DIR/_full_download.mp4"
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/mp4" -o "$TMP_FULL" "$URL"

ffmpeg -y -i "$TMP_FULL" -ss "$START" -to "$END" \
  -c:v libx264 -crf 18 -c:a aac \
  "$RAW_DIR/$OUT_NAME"

rm -f "$TMP_FULL"
echo "[done] $RAW_DIR/$OUT_NAME"
