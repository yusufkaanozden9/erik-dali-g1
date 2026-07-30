#!/usr/bin/env bash
# Stage 3: GMR's retargeted CSV (29-DoF G1 joint layout) -> 23-DoF npz for mjlab training.
#
# GMR retargets to the full 29-DoF G1 only (see 02_retarget_to_g1.sh). unitree_rl_mjlab's
# 23-DoF task expects a CSV with fewer DoF columns (no wrist roll/pitch/yaw per arm). Before
# running this, diff your retargeted CSV's column count against the shipped example at
#   third_party/unitree_rl_mjlab/src/assets/motions/g1_23dof/dance1_subject2.csv
# to confirm the column layout / ordering matches what --robot g1_23dof expects. If it
# doesn't, the wrist DoF columns need to be dropped from the 29-DoF CSV before this step —
# do that with a short pandas/numpy snippet, comparing g1_constants.py vs
# g1_23dof_constants.py in third_party/unitree_rl_mjlab/src/assets/robots/unitree_g1/ for
# the exact joint-name mapping.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MJLAB_DIR="$ROOT_DIR/third_party/unitree_rl_mjlab"

CSV_FILE="${1:?Usage: $0 <path_to_retargeted_csv> [output_name=erik_dali.npz] [input_fps=30]}"
OUT_NAME="${2:-erik_dali.npz}"
INPUT_FPS="${3:-30}"

cd "$MJLAB_DIR"

conda run -n unitree_rl_mjlab python scripts/csv_to_npz.py \
  --input-file "$CSV_FILE" \
  --output-name "$OUT_NAME" \
  --input-fps "$INPUT_FPS" \
  --output-fps 50 \
  --robot g1_23dof

echo "[done] Output under $MJLAB_DIR (per csv_to_npz.py's own convention, typically src/assets/motions/g1_23dof/$OUT_NAME)"
