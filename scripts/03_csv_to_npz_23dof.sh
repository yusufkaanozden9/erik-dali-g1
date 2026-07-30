#!/usr/bin/env bash
# Stage 3: GMR's retargeted CSV (29-DoF G1 joint layout) -> 23-DoF npz for mjlab training.
#
# GMR retargets to the full 29-DoF G1 only (see 02_retarget_to_g1.sh). unitree_rl_mjlab's
# 23-DoF task expects 23 joint columns, not 29. VERIFIED (2026-07-30) which 6 to drop by
# loading mjlab's own compiled g1_23dof MJCF and reading its joint order directly (not
# guessed): 23-DoF drops waist_roll, waist_pitch (only waist_yaw remains), and each arm's
# wrist_pitch + wrist_yaw (only wrist_roll remains). In terms of the 29-DoF GMR motor-ID
# order (root take cols 0-6, dof_pos starts at col 7):
#   keep 29-DoF joint indices [0-12, 15-19, 22-26]  ->  drop [13,14, 20,21, 27,28]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MJLAB_DIR="$ROOT_DIR/third_party/unitree_rl_mjlab"

CSV_FILE="${1:?Usage: $0 <path_to_29dof_retargeted_csv> [output_name=erik_dali.npz] [input_fps=30] [device=cpu]}"
OUT_NAME="${2:-erik_dali.npz}"
INPUT_FPS="${3:-30}"
DEVICE="${4:-cpu}"   # csv_to_npz.py defaults to cuda:0; pass "cuda:0" here on the GPU box.

CSV_23DOF="${CSV_FILE%.csv}_23dof.csv"

python3 - "$CSV_FILE" "$CSV_23DOF" <<'PYEOF'
import sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
motion = np.loadtxt(src, delimiter=",")
assert motion.shape[1] == 36, f"expected 36 cols (7 root + 29 dof), got {motion.shape[1]}"
root = motion[:, :7]
dof29 = motion[:, 7:]
keep_29dof_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,19,22,23,24,25,26]
dof23 = dof29[:, keep_29dof_idx]
motion23 = np.concatenate([root, dof23], axis=1)
np.savetxt(dst, motion23, delimiter=",")
print(f"[23dof csv] {motion.shape} -> {motion23.shape}, saved to {dst}")
PYEOF

cd "$MJLAB_DIR"

conda run -n unitree_rl_mjlab python scripts/csv_to_npz.py \
  --input-file "$CSV_23DOF" \
  --output-name "$OUT_NAME" \
  --input-fps "$INPUT_FPS" \
  --output-fps 50 \
  --robot g1_23dof \
  --device "$DEVICE"

echo "[done] Output under $MJLAB_DIR (per csv_to_npz.py's own convention, typically src/assets/motions/g1_23dof/$OUT_NAME)"
