#!/usr/bin/env bash
# Clones and pins the three upstream repos this pipeline depends on, then creates one
# conda env per repo (their dependency stacks conflict, same reasoning the sibling
# unibot_submission project uses for its dual-venv split — see its vast_ai/setup_envs.sh).
#
# Target platform: Ubuntu 22.04/24.04 + NVIDIA GPU (driver 550+). Not tested on macOS —
# GVHMR/GMR/mjlab all assume CUDA is available for anything beyond a slow CPU smoke test.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="$ROOT_DIR/third_party"
mkdir -p "$THIRD_PARTY_DIR"

# Pinned commits (HEAD of each repo's default branch as of 2026-07-30). Bump deliberately —
# don't float on `main`/`master`, these projects are under active development.
GVHMR_REPO="https://github.com/zju3dv/GVHMR.git"
GVHMR_COMMIT="6ec3ca39336c50492c0fae65fba2fb831fc7d866"

GMR_REPO="https://github.com/YanjieZe/GMR.git"
GMR_COMMIT="bb1bbe40774794fceb2a7c579a3464a28e68c844"

MJLAB_REPO="https://github.com/unitreerobotics/unitree_rl_mjlab.git"
MJLAB_COMMIT="1425b15f73bd4095f0df53709d7c389c3eb9e790"

clone_pinned() {
  local repo="$1" commit="$2" dest="$3"
  if [ -d "$dest/.git" ]; then
    echo "[skip] $dest already cloned"
    return
  fi
  git clone "$repo" "$dest"
  git -C "$dest" checkout "$commit"
}

echo "== Cloning upstream repos =="
clone_pinned "$GVHMR_REPO" "$GVHMR_COMMIT" "$THIRD_PARTY_DIR/gvhmr"
clone_pinned "$GMR_REPO"   "$GMR_COMMIT"   "$THIRD_PARTY_DIR/gmr"
clone_pinned "$MJLAB_REPO" "$MJLAB_COMMIT" "$THIRD_PARTY_DIR/unitree_rl_mjlab"

echo "== System dependencies =="
if command -v apt >/dev/null 2>&1; then
  sudo apt update
  # ffmpeg: video decode for GVHMR's demo.py and our own trim step.
  # libyaml-cpp-dev/libboost-all-dev/libeigen3-dev/libspdlog-dev/libfmt-dev: unitree_rl_mjlab's
  # C++ deploy/ target (sim2real controller) — not needed for training-only work, but cheap
  # to install up front per the repo's own setup doc.
  sudo apt install -y ffmpeg libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
else
  echo "[warn] apt not found — install ffmpeg + the unitree_rl_mjlab C++ deps manually" >&2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda not found. Install Miniconda first:" >&2
  echo "  https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/doc/setup_en.md" >&2
  exit 1
fi

echo "== gvhmr conda env (python 3.10) =="
conda create -y -n gvhmr python=3.10
conda run -n gvhmr pip install -r "$THIRD_PARTY_DIR/gvhmr/requirements.txt"
conda run -n gvhmr pip install -e "$THIRD_PARTY_DIR/gvhmr"

echo "== gmr conda env (python 3.10) =="
conda create -y -n gmr python=3.10
conda run -n gmr pip install -e "$THIRD_PARTY_DIR/gmr"
conda run -n gmr pip install PyQt6 PyQt6-Qt6 PyQt6-sip
conda install -y -n gmr -c conda-forge libstdcxx-ng

echo "== unitree_rl_mjlab conda env (python 3.11) =="
conda create -y -n unitree_rl_mjlab python=3.11
conda run -n unitree_rl_mjlab pip install -e "$THIRD_PARTY_DIR/unitree_rl_mjlab"
# Verified on macOS/CPU (2026-07-30): the repo's setup.py pins mujoco-warp==3.5.0 but
# doesn't pin `mujoco` itself, so pip grabs the newest mujoco — which has since renamed/
# removed an enum (mjENBL_MULTICCD) mujoco-warp 3.5.0 still expects, breaking every import.
# Pin mujoco back down to match. Also: `scipy` is imported by mjlab's terrain code but
# missing from its own dependency list.
conda run -n unitree_rl_mjlab pip install "mujoco==3.5.0" scipy

cat <<'EOF'

== Manual step required: SMPL-X body models ==
Both GVHMR and GMR need the SMPL-X body models, which are license-gated (free registration
required, cannot be scripted):

  1. Register at https://smpl-x.is.tue.mpg.de/
  2. Download SMPLX_NEUTRAL.pkl, SMPLX_FEMALE.pkl, SMPLX_MALE.pkl
  3. Place them in:
       third_party/gvhmr/body_models/smplx/
       third_party/gmr/assets/body_models/smplx/
     (check each repo's own INSTALL.md for the exact expected path/filename — these move
     between versions.)

GVHMR also needs its own pretrained checkpoints (HMR2, ViTPose, DPVO, YOLO) — see
third_party/gvhmr/docs/INSTALL.md for the download links, several are also gated.

Once those are in place, run the scripts/ pipeline in order (01 → 05).

Note: scripts/05_smoke_test_training.sh (bundled example motion, no SMPL-X needed) was
verified working end-to-end on macOS/CPU (arm64) with this exact setup — task registration,
scene construction, a few PPO iterations, checkpoint + ONNX export all confirmed. On a
machine with no CUDA GPU, pass --gpu-ids None (train.py) / --device cpu (csv_to_npz.py) —
both scripts already do this by default via their positional/flag args, see their headers.
EOF
