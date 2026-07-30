# Erik Dalı → Unitree G1 (23-DoF)

Teach a Unitree G1 (23-DoF variant) to perform the Turkish folk dance **Erik Dalı**, via
motion retargeting + reinforcement-learning-based motion imitation.

This project glues together three existing open-source projects rather than reimplementing
any of it:

| Stage | Tool | Repo |
|---|---|---|
| Monocular video → SMPL-X human motion | GVHMR | https://github.com/zju3dv/GVHMR |
| SMPL-X → G1 joint-space motion (retargeting) | GMR | https://github.com/YanjieZe/GMR |
| RL motion-imitation training (MuJoCo/mjlab) | unitree_rl_mjlab | https://github.com/unitreerobotics/unitree_rl_mjlab |

```
video clip (reference_motion/raw/)
        │  GVHMR (01_extract_smplx.sh)
        ▼
SMPL-X motion
        │  GMR (02_retarget_to_g1.sh)
        ▼
G1 joint-space motion (.csv, mjlab-compatible)
        │  mjlab csv_to_npz.py (03_csv_to_npz_23dof.sh)
        ▼
erik_dali.npz
        │  GMR vis_robot_motion.py (04_kinematic_sanity_check.sh)
        ▼
kinematic sanity check (no physics/RL yet)
        │  mjlab train.py, Unitree-G1-23Dof-Tracking-No-State-Estimation
        ▼
PPO-trained tracking policy  ──►  sim2sim (MuJoCo↔Isaac) ──►  staged real-G1 rollout
```

See `docs/PIPELINE.md` for the full write-up (grounded, command-accurate version of the
original planning doc), including the stages this repo does **not** run (real training,
domain randomization, hardware rollout — all deferred to a rented GPU box / real hardware).

## Status

- [x] Reference clip sourced + visually sanity-checked (`reference_motion/NOTES.md`)
- [x] Pipeline scaffold + scripts written
- [x] `unitree_rl_mjlab` + `mjlab`/`mujoco-warp` install verified on macOS/CPU (arm64) —
      needed 2 install fixes not in the upstream repo's own docs, see
      `docs/PIPELINE.md` "Smoke test — verified"
- [x] Stage 5 smoke test (`05_smoke_test_training.sh`) — **verified working**: full PPO
      loop ran on the bundled example motion, checkpoint + ONNX export confirmed
- [x] GMR conda env installed + CLI verified on macOS/CPU (`smplx_to_robot.py`,
      `batch_gmr_pkl_to_csv.py`, `vis_robot_motion.py` — corrected `scripts/02_*.sh`, the
      original plan's `--save_as_csv` flag guess didn't exist; real flow is two scripts)
- [x] GVHMR install feasibility checked — **confirmed impossible on macOS**: its
      `requirements.txt` hard-pins `torch==2.3.0+cu121` and a `pytorch3d` wheel with
      `linux_x86_64` baked into the URL itself. Stage 1 (video → SMPL-X) can only run on
      the Linux+GPU box — see `docs/PIPELINE.md`.
- [x] SMPL-X body models downloaded + verified for GMR (`third_party/gmr/assets/body_models/smplx/`)
      — user registered at smpl-x.is.tue.mpg.de and downloaded `models_smplx_v1_1.zip`
      themselves (license-gated, can't be scripted). First attempt actually downloaded the
      wrong model (plain **SMPL**, not SMPL-X — same-looking filenames, missing
      hand/expression blendshapes); caught it by trying to load the file and re-downloaded
      the correct package. Confirmed with a real forward pass: 10475 vertices, 127 joints.
      GVHMR needs its own copy of these — not yet placed, see below.
- [ ] Stages 1–4 on the actual Erik Dalı clip (GVHMR → GMR → csv_to_npz → kinematic check)
- [ ] Real training run on Erik Dalı data with `--env.scene.num-envs=4096` on a GPU box

## Quickstart (on a Linux box with an NVIDIA GPU)

```bash
./setup_envs.sh                        # clones GVHMR/GMR/unitree_rl_mjlab, creates 3 conda envs
# ... register at https://smpl-x.is.tue.mpg.de/ and drop SMPL-X .pkl files where setup_envs.sh points ...
./scripts/01_extract_smplx.sh
./scripts/02_retarget_to_g1.sh
./scripts/03_csv_to_npz_23dof.sh
./scripts/04_kinematic_sanity_check.sh
./scripts/05_smoke_test_training.sh    # bundled example motion, tiny env count — proves the chain works
```

Real training (`--env.scene.num-envs=4096`, hours of GPU time) is a separate, deliberate
step — see `docs/PIPELINE.md` §"Real training run".
