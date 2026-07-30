#!/usr/bin/env bash
# Stage 5 (smoke test only — NOT real training): prove the RL training entrypoint works
# — task registration, CLI parsing, MuJoCo scene construction, a few PPO update steps —
# using unitree_rl_mjlab's own BUNDLED example motion (dance1_subject2), not Erik Dalı data.
# This can run before stages 1-4 finish, and is CPU-tolerant (just slow) so it's the one
# thing in this pipeline verifiable without a rented GPU. Verified working on macOS/CPU
# (no CUDA) with these exact flags — see docs/PIPELINE.md "Smoke test — verified" for the
# install fixes (mujoco pin, scipy) this needed.
#
# Real training (thousands of iterations, --env.scene.num-envs=4096) is a deliberate,
# separate step — see docs/PIPELINE.md "Real training run". Don't scale this up in place;
# launch that on the GPU box instead.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MJLAB_DIR="$ROOT_DIR/third_party/unitree_rl_mjlab"
NUM_ENVS="${1:-4}"
MAX_ITERATIONS="${2:-5}"

cd "$MJLAB_DIR"

# PYTHONUNBUFFERED=1 / python -u matters: without it, stdout is fully block-buffered
# once redirected to a file/log, so nothing appears for a long time even though training
# is genuinely progressing — this cost real GPU-minutes once during the actual training
# run before we figured it out (see docs/PIPELINE.md "Real training run — verified").
PYTHONUNBUFFERED=1 conda run --no-capture-output -n unitree_rl_mjlab python -u scripts/train.py Unitree-G1-23Dof-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/g1_23dof/dance1_subject2.npz \
  --env.scene.num-envs="$NUM_ENVS" \
  --agent.max-iterations="$MAX_ITERATIONS" \
  --gpu-ids None

echo "[done] If this completed without crashing, the training entrypoint is confirmed working."
echo "       Swap --motion-file to the Erik Dalı npz from stage 3, drop --gpu-ids None, and use"
echo "       --env.scene.num-envs=4096 for the real run, on a GPU box (see docs/PIPELINE.md)."
