#!/usr/bin/env bash
# Re-apply the Erik Dalı deploy overlay onto a fresh third_party/unitree_rl_mjlab checkout.
#
# The vendored clone is gitignored and detached-HEAD, so every change made to it for the
# hardware bring-up (FSM transition-reason logging, the lowcmd-channel guard, the
# Mimic_ErikDali FSM state) is lost the moment setup_envs.sh re-clones it. This script
# puts all of it back. See README.md in this directory for what each piece does and why.
#
# Usage:  ./deploy_overlay/apply.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY="$REPO_ROOT/deploy_overlay"
VENDORED="$REPO_ROOT/third_party/unitree_rl_mjlab"
BASE_COMMIT="1425b15f73bd4095f0df53709d7c389c3eb9e790"
MIMIC_DIR="$VENDORED/deploy/robots/g1_23dof/config/policy/mimic/erik_dali"

[ -d "$VENDORED" ] || { echo "ERROR: $VENDORED missing — run ./setup_envs.sh first." >&2; exit 1; }

# --- 1. C++ source patch -----------------------------------------------------------
head_commit="$(git -C "$VENDORED" rev-parse HEAD)"
if [ "$head_commit" != "$BASE_COMMIT" ]; then
  echo "WARNING: vendored clone is at $head_commit, patch was made against $BASE_COMMIT."
  echo "         If the patch fails below, re-derive it by hand against the new upstream."
fi

if git -C "$VENDORED" apply --reverse --check "$OVERLAY/unitree_rl_mjlab.patch" 2>/dev/null; then
  echo "[skip] C++ patch already applied"
elif git -C "$VENDORED" apply --check "$OVERLAY/unitree_rl_mjlab.patch" 2>/dev/null; then
  git -C "$VENDORED" apply "$OVERLAY/unitree_rl_mjlab.patch"
  echo "[ok]   C++ patch applied (5 files)"
else
  echo "ERROR: unitree_rl_mjlab.patch does not apply cleanly to this checkout." >&2
  exit 1
fi

# --- 2. Config files ---------------------------------------------------------------
mkdir -p "$MIMIC_DIR/params" "$MIMIC_DIR/exported"
(cd "$OVERLAY/files" && find . -type f -print0) | while IFS= read -r -d '' f; do
  install -m 644 "$OVERLAY/files/$f" "$VENDORED/${f#./}"
  echo "[ok]   ${f#./}"
done

# --- 3. Trained artifacts ----------------------------------------------------------
# Binaries stay out of git (see .gitignore); they are copied from trained_model/, which
# is where the GPU-box run's outputs were brought back to.
for src in "$REPO_ROOT/trained_model/policy.onnx:$MIMIC_DIR/exported/policy.onnx" \
           "$REPO_ROOT/trained_model/erik_dali.npz:$MIMIC_DIR/params/erik_dali.npz"; do
  from="${src%%:*}"; to="${src##*:}"
  if [ -f "$from" ]; then
    install -m 644 "$from" "$to"
    echo "[ok]   $(basename "$to")"
  else
    echo "WARNING: $from missing — copy the trained checkpoint back before deploying." >&2
  fi
done

echo
echo "Overlay applied. Sanity check:"
echo "  grep -n Mimic_ErikDali $VENDORED/deploy/robots/g1_23dof/config/config.yaml"
