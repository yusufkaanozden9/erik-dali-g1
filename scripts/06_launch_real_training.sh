#!/usr/bin/env bash
# Stage 6: the real training run, on the Erik Dalı motion, at full scale, on a GPU box.
# Verified working (2026-07-30) on a rented RTX 4090: 0.96s/iteration at
# --env.scene.num-envs=4096, ~102k env-steps/sec. At that rate:
#   1,000 iters  ~16 min
#   5,000 iters  ~1h20m
#   30,001 iters (mjlab's own default) ~8h
# Cost is trivial relative to iteration count on a ~$0.30-0.35/hr spot GPU — time is the
# real budget here, not money.
#
# PYTHONUNBUFFERED=1 / python -u is NOT optional: without it, stdout is fully
# block-buffered once redirected to a log file, so nothing appears for a long time even
# though training is genuinely progressing at the expected rate. This cost real GPU-minutes
# once (killed a run that looked "stuck" after 20 silent minutes, only to find via a
# smaller unbuffered re-run that it had been training correctly the whole time).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MJLAB_DIR="$ROOT_DIR/third_party/unitree_rl_mjlab"
NUM_ENVS="${1:-4096}"
MAX_ITERATIONS="${2:-5000}"
SAVE_INTERVAL="${3:-250}"
LOG_FILE="${4:-/workspace/real_train.log}"

cd "$MJLAB_DIR"

echo "[info] motion file: src/assets/motions/g1_23dof/erik_dali.npz (from scripts/03_csv_to_npz_23dof.sh)"
echo "[info] logging to $LOG_FILE, unbuffered — tail -f it to watch progress"

PYTHONUNBUFFERED=1 nohup conda run --no-capture-output -n unitree_rl_mjlab python -u scripts/train.py \
  Unitree-G1-23Dof-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/g1_23dof/erik_dali.npz \
  --env.scene.num-envs="$NUM_ENVS" \
  --agent.max-iterations="$MAX_ITERATIONS" \
  --agent.save-interval="$SAVE_INTERVAL" \
  --agent.logger tensorboard \
  --gpu-ids '[0]' > "$LOG_FILE" 2>&1 &
disown

echo "[done] launched in background (survives SSH disconnect via disown), pid $!"
echo "       checkpoints land in logs/rsl_rl/g1_23dof_tracking/<timestamp>/model_*.pt"
