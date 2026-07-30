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
SMPL-X motion (hmr4d_results.pt)
        │  GMR gvhmr_to_robot.py (02_retarget_to_g1.sh)
        ▼
G1 joint-space motion (29-DoF .pkl → .csv, mjlab-compatible)
        │  select 23-DoF joint columns, mjlab csv_to_npz.py (03_csv_to_npz_23dof.sh)
        ▼
erik_dali.npz (23-DoF)
        │  GMR vis_robot_motion.py --record_video (04_kinematic_sanity_check.sh)
        ▼
kinematic sanity check (no physics/RL yet)
        │  mjlab train.py, Unitree-G1-23Dof-Tracking-No-State-Estimation (06_launch_real_training.sh)
        ▼
PPO-trained tracking policy  ──►  sim2sim (MuJoCo↔Isaac) ──►  staged real-G1 rollout
```

See `docs/PIPELINE.md` for the full write-up (grounded, command-accurate version of the
original planning doc) — every stage below has now actually been run end-to-end, not just
planned.

## Status — full pipeline verified end-to-end (2026-07-30, rented RTX 4090)

- [x] Reference clip sourced + visually sanity-checked (`reference_motion/NOTES.md`)
- [x] All three conda envs (`gvhmr`, `gmr`, `unitree_rl_mjlab`) installed on a real Linux+GPU
      box — several install fixes not in the upstream repos' own docs, all folded into
      `setup_envs.sh` (conda ToS gate, chumpy build-isolation, `mujoco`/`warp-lang`
      version pins, missing `scipy`)
- [x] SMPL-X body models registered + downloaded by the user, placed for both GVHMR and GMR,
      verified with a real forward pass (10475 vertices / 127 joints). Caught and fixed a
      real gotcha: the SMPL-X site's downloads page also hosts the *plain SMPL* model under
      a similar-looking package — first attempt grabbed the wrong one.
- [x] GVHMR's own pretrained checkpoints (gvhmr, hmr2, vitpose, yolo — ~5.5GB) downloaded via
      a HuggingFace mirror after Google Drive's anonymous-download quota was hit
- [x] **Stage 1** — GVHMR ran on the actual reference clip, produced real SMPL-X motion
      (`hmr4d_results.pt`); demo.py's preview-render step was allowed to fail (needs a
      *third*, separately-gated plain-SMPL model we don't otherwise need) since the motion
      data is already saved to disk by that point
- [x] **Stage 2** — GMR's `gvhmr_to_robot.py` retargeted the SMPL-X motion straight to the
      full 29-DoF G1 (needed `xvfb-run` — GMR's scripts open a GLFW window even when just
      recording video, which fails outright on a headless box)
- [x] **Stage 3** — derived the exact 23-DoF joint subset (23 of 29 columns; dropped
      waist-roll, waist-pitch, and both wrists' pitch/yaw — verified against mjlab's own
      compiled MJCF joint order, not guessed) and converted to `erik_dali.npz`
      (1180 frames, 23 joints) — genuine, correctly-shaped training data
- [x] **Stage 4** — recorded and visually inspected a sanity-check video of the retargeted
      G1 motion: no self-collision, feet planted, arm poses clearly match the source
      choreography (both-arms-raised, one-arm-behind poses visible)
- [x] **Stage 5** smoke test — verified working on macOS/CPU with the bundled example motion
- [x] **Stage 6** — real PPO training launched on the actual Erik Dalı data,
      `--env.scene.num-envs=4096`. Measured throughput: 0.96s/iteration, ~102k
      env-steps/sec on a single RTX 4090. 5000-iteration run ≈ 1h20m, ≈ $0.45 in GPU time.

See `docs/PIPELINE.md` for the full list of install gotchas found along the way (each one
real, verified, not assumed from the upstream docs) and what's still open (curriculum /
domain randomization tuning, sim2sim cross-check, staged real-hardware rollout).

## Quickstart (on a Linux box with an NVIDIA GPU)

```bash
./setup_envs.sh                        # clones GVHMR/GMR/unitree_rl_mjlab, creates 3 conda envs
# ... register at https://smpl-x.is.tue.mpg.de/ (SMPL-X, not plain SMPL!) and place the
#     .npz files where setup_envs.sh points; download GVHMR's checkpoints per its printed
#     instructions ...
./scripts/01_extract_smplx.sh
./scripts/02_retarget_to_g1.sh
./scripts/03_csv_to_npz_23dof.sh
./scripts/04_kinematic_sanity_check.sh
./scripts/06_launch_real_training.sh   # the real run — see docs/PIPELINE.md for timing/cost
```

`scripts/05_smoke_test_training.sh` (bundled example motion, tiny env count, CPU-tolerant)
is the quick way to confirm a fresh install's training entrypoint works before committing
GPU time to the real run.
