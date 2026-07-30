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
#
# NOTE: demo.py always tries to render a preview video afterwards, which needs the plain
# (not SMPL-X) SMPL body model at inputs/checkpoints/body_models/smpl/ — a third, separately
# gated download (smpl.is.tue.mpg.de) we don't otherwise need. Verified (2026-07-30): the
# actual motion prediction is saved to outputs/demo/<name>/hmr4d_results.pt BEFORE the
# render step runs, so demo.py crashing on the missing SMPL model afterwards is fine — the
# `|| true` below lets this script succeed regardless.
conda run -n gvhmr python tools/demo/demo.py --video="$CLIP" -s || true

RESULT="$GVHMR_DIR/outputs/demo/$(basename "$CLIP" .mp4)/hmr4d_results.pt"
if [ -f "$RESULT" ]; then
  echo "[done] $RESULT"
  echo "       Pass this path as --gvhmr_pred_file to scripts/02_retarget_to_g1.sh."
else
  echo "[error] $RESULT not found — check the output above for a real failure (not the render crash)." >&2
  exit 1
fi
