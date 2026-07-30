#!/usr/bin/env bash
# Stage 2: GVHMR's SMPL-X motion prediction -> Unitree G1 joint-space motion, via GMR.
#
# VERIFIED (2026-07-30): GMR ships a script specifically for this GVHMR->robot handoff,
# scripts/gvhmr_to_robot.py, which takes GVHMR's own hmr4d_results.pt directly — no need to
# go through smplx_to_robot.py or re-export SMPL-X params by hand. Only targets the full
# 29-DoF G1 (--robot unitree_g1, GMR's only G1 IK config); the 23-DoF motion used downstream
# is derived from this by scripts/03_csv_to_npz_23dof.sh.
#
# Requires a display: GMR's scripts open a GLFW window even in --record_video mode, which
# fails outright on a headless box ("X11: The DISPLAY environment variable is missing").
# Run under xvfb-run (apt install -y xvfb, once).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GMR_DIR="$ROOT_DIR/third_party/gmr"

GVHMR_PRED_FILE="${1:?Usage: $0 <path_to_hmr4d_results.pt_from_stage_1> [output_basename=erik_dali]}"
OUT_NAME="${2:-erik_dali}"

OUT_DIR="$ROOT_DIR/reference_motion/retargeted"
mkdir -p "$OUT_DIR"

cd "$GMR_DIR"

xvfb-run -a conda run -n gmr python scripts/gvhmr_to_robot.py \
  --gvhmr_pred_file "$GVHMR_PRED_FILE" \
  --robot unitree_g1 \
  --save_path "$OUT_DIR/$OUT_NAME.pkl" \
  --rate_limit

conda run -n gmr python scripts/batch_gmr_pkl_to_csv.py --folder "$OUT_DIR"

echo "[done] $OUT_DIR/$OUT_NAME.pkl and $OUT_DIR/csv/$OUT_NAME.csv"
echo "       Sanity-check the pkl first with scripts/04_kinematic_sanity_check.sh, then feed"
echo "       the csv into scripts/03_csv_to_npz_23dof.sh."
