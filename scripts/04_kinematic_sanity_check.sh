#!/usr/bin/env bash
# Stage 4: kinematic-only sanity check of the retargeted motion, BEFORE spending any
# training compute — catches self-collision, joint-limit, and foot-sliding problems early
# (PDF's step 3 / "3. Önce yalnızca kinematik simülasyon").
#
# GMR's own visualizer runs the retarget in isolation (no mjlab task/RL involved yet) and
# is real-time on CPU per GMR's docs, so this step doesn't need the GPU box.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GMR_DIR="$ROOT_DIR/third_party/gmr"

PKL_FILE="${1:?Usage: $0 <path_to_retargeted_pkl_from_stage_2>}"

cd "$GMR_DIR"
conda run -n gmr python scripts/vis_robot_motion.py \
  --robot unitree_g1 \
  --robot_motion_path "$PKL_FILE"

cat <<'EOF'

Watch for (per the PDF's checklist):
  - self-collision (limbs clipping through the torso/each other)
  - joint angles/speeds pinned at their limits
  - feet sliding through the floor, or the reference foot leaving the ground when it
    shouldn't
  - center-of-mass leaving the support polygon
  - abrupt frame-to-frame jumps
  - start/end pose not close to a safe standing pose

If it looks physically implausible, go back and hand-fix the offending segment in GMR's
per-joint output (or re-trim the source clip) before spending training compute on it.
EOF
