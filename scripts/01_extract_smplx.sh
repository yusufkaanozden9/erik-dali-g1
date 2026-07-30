#!/usr/bin/env bash
# Stage 1: monocular video -> SMPL-X human motion, via GVHMR.
# Input: reference_motion/raw/erik_dali_reference.mp4 (static camera, single
#        dancer, full body visible — see reference_motion/NOTES.md for the sanity check).
# Output: GVHMR's own output dir (per its demo.py conventions), containing the SMPL-X
#         motion sequence consumed by scripts/02_retarget_to_g1.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GVHMR_DIR="$ROOT_DIR/third_party/gvhmr"
CLIP="$ROOT_DIR/reference_motion/raw/erik_dali_reference.mp4"

if [ ! -f "$CLIP" ]; then
  echo "[error] $CLIP not found. See reference_motion/NOTES.md." >&2
  exit 1
fi

cd "$GVHMR_DIR"

# -s skips visual odometry: appropriate here because the source clip is a static tripod
# shot (confirmed in reference_motion/NOTES.md). If you swap in a clip with camera motion,
# drop -s so GVHMR runs full world-grounded odometry instead.
conda run -n gvhmr python tools/demo/demo.py --video="$CLIP" -s

echo "[done] Check $GVHMR_DIR/outputs/ (or wherever demo.py printed its output path) for the SMPL-X motion."
echo "       Pass that path as --smplx_file to scripts/02_retarget_to_g1.sh."
