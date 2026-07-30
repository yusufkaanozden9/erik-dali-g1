#!/usr/bin/env bash
# Stage 2: SMPL-X human motion -> Unitree G1 joint-space motion, via GMR.
#
# GMR only ships a single G1 IK config (general_motion_retargeting/ik_configs/smplx_to_g1.json),
# targeting the full 29-DoF G1 (--robot unitree_g1) — there is no separate 23-DoF retargeting
# config. The 23-DoF motion used downstream is derived from this 29-DoF output by
# scripts/03_csv_to_npz_23dof.sh (which drops the wrist/hand DoF columns mjlab's 23-DoF model
# doesn't have).
#
# VERIFIED (2026-07-30) against the actual installed GMR CLI — the earlier plan assumed a
# `--save_as_csv` flag on smplx_to_robot.py; that doesn't exist. CSV export is a SEPARATE
# script that batch-converts a folder of .pkl outputs:
#   scripts/smplx_to_robot.py        --smplx_file ... --robot unitree_g1 --save_path <dir>/erik_dali.pkl
#   scripts/batch_gmr_pkl_to_csv.py  --folder <dir>   -> writes <dir>/csv/erik_dali.csv
# Confirmed by reading batch_gmr_pkl_to_csv.py: it writes columns
# [root_pos(3), root_rot_xyzw(4), dof_pos(N)] — exactly what mjlab's csv_to_npz.py expects
# (root pos/quat + per-joint angles), and downsamples to 30fps if the source is faster.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GMR_DIR="$ROOT_DIR/third_party/gmr"

SMPLX_FILE="${1:?Usage: $0 <path_to_smplx_motion_from_stage_1> [output_basename=erik_dali]}"
OUT_NAME="${2:-erik_dali}"

OUT_DIR="$ROOT_DIR/reference_motion/retargeted"
mkdir -p "$OUT_DIR"

cd "$GMR_DIR"

conda run -n gmr python scripts/smplx_to_robot.py \
  --smplx_file "$SMPLX_FILE" \
  --robot unitree_g1 \
  --save_path "$OUT_DIR/$OUT_NAME.pkl" \
  --rate_limit

conda run -n gmr python scripts/batch_gmr_pkl_to_csv.py --folder "$OUT_DIR"

echo "[done] $OUT_DIR/$OUT_NAME.pkl and $OUT_DIR/csv/$OUT_NAME.csv"
echo "       Sanity-check the pkl first with scripts/04_kinematic_sanity_check.sh, then feed"
echo "       the csv into scripts/03_csv_to_npz_23dof.sh."
