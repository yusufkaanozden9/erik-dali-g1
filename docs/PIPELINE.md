# Pipeline: Erik Dalı → Unitree G1 (23-DoF)

Grounded, command-accurate version of the original planning doc, verified against the
actual upstream repos (GVHMR, GMR, unitree_rl_mjlab) rather than taken on faith.

## 1. Reference motion

Source: user-supplied video, trimmed to a clean 23s single-dancer segment.
Details + visual sanity check: `reference_motion/NOTES.md`.

This is a **video-based pose estimate**, not real mocap (Xsens/OptiTrack/LAFAN1-grade).
Lower fidelity than clean mocap, per the original plan's own caveat — acceptable for a
first pass; can be replaced with real mocap later without changing anything downstream of
stage 1.

## 2. Retargeting (executed via scripts, not yet run in this repo — needs Linux+GPU box)

1. `scripts/01_extract_smplx.sh` — GVHMR: video → SMPL-X motion.
2. `scripts/02_retarget_to_g1.sh` — GMR: SMPL-X → full 29-DoF G1 joint motion
   (`--robot unitree_g1`, GMR's only G1 IK config).
3. `scripts/03_csv_to_npz_23dof.sh` — mjlab's `csv_to_npz.py --robot g1_23dof`: 29-DoF CSV
   → 23-DoF npz, dropping the wrist DoFs the 23-DoF model lacks. **Verify the column mapping**
   against the shipped `third_party/unitree_rl_mjlab/src/assets/motions/g1_23dof/dance1_subject2.csv`
   before trusting the output — see the script's own header comment.
4. `scripts/04_kinematic_sanity_check.sh` — GMR's visualizer, CPU-only, catches
   self-collision / joint-limit / foot-sliding problems before any training compute is spent.

## 3. RL motion-imitation training

`unitree_rl_mjlab` already ships the exact task needed:

```bash
python scripts/train.py Unitree-G1-23Dof-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1_23dof/erik_dali.npz \
  --env.scene.num-envs=4096
```

What this does under the hood (asymmetric actor-critic PPO, per `src/tasks/tracking/`):
- **Actor**: only observations available on the real robot (joint angles/velocities,
  base angular velocity, gravity direction, previous action, reference phase).
- **Critic**: gets privileged simulation-only state during training (not deployed).
- **Action**: policy outputs a small residual on top of the reference joint targets
  (`q_target = q_ref + scale * action`), not raw torque — the reference motion carries most
  of the trajectory, RL cleans up balance/contact/physical-feasibility errors.
- Reward mixes joint tracking, body orientation, foot tracking/contact, balance, and
  penalties (torque, jerk, joint-limit proximity, foot slip, unwanted self-collision).

Training outputs land in `logs/rsl_rl/<robot>_tracking/<date_time>/model_<iter>.pt`.

### Smoke test (done here, no GPU needed)

`scripts/05_smoke_test_training.sh` runs the same task on unitree_rl_mjlab's own bundled
`dance1_subject2` example with `--env.scene.num-envs=4` for a handful of iterations — proves
task registration / CLI / MuJoCo scene construction work, without waiting on the retargeting
chain or spending real GPU time.

### Real training run (NOT executed in this repo — deferred)

- 4096 parallel envs needs an NVIDIA GPU with real VRAM headroom; drop to 2048/1024/512 if
  it OOMs (affects sample throughput and PPO batch behavior, not feasibility).
- Follow the same operational pattern `unibot_submission/vast_ai/` already uses for its own
  (unrelated) training runs: rent a GPU box, run `setup_envs.sh`, launch training, monitor,
  tear down. Would need a `vast_ai/`-equivalent directory here with an adapted
  `launch_training.sh` — not written yet, since no training has actually been launched.
- Curriculum (per the original plan): start the policy on a static reference pose, add arm
  gestures, then small weight shifts, then foot lifts, ramping tempo from 50% → 100%,
  finally enabling turns and stronger domain randomization. `unitree_rl_mjlab`'s task config
  (`src/tasks/tracking/config/g1_23dof/env_cfgs.py`) is where this would be implemented —
  not yet touched.
- Domain randomization (friction, mass, PD gains, motor delay, IMU/encoder noise, small
  external pushes) — same file, not yet touched.

## 4. Evaluation

```bash
python scripts/play.py Unitree-G1-23Dof-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1_23dof/erik_dali.npz \
  --checkpoint_file=logs/rsl_rl/g1_23dof_tracking/<run>/model_<iter>.pt
```

Track more than total reward: MPJPE, joint-angle error, foot-slide distance, fall rate,
episode-completion rate, peak/RMS torque, joint-velocity-limit violations, real-time
inference latency. Success = many episodes across many randomization seeds, not one lucky
rollout.

## 5. Sim2sim / sim2real (deferred — needs GPU box + physical G1)

- Sim2sim: if trained on Isaac Lab/PhysX, re-validate in MuJoCo (or vice versa) with
  different timestep/contact/friction settings — checks the policy isn't overfit to one
  physics engine's quirks.
- Sim2real staged rollout (`unitree_rl_mjlab`'s own recommended order, `deploy/robots/g1_23dof/`):
  1. Robot suspended, `zero-torque` mode, enter debug mode (`L2+R2`).
  2. Feet fully off the ground, 10–20% motion amplitude, policy output logged, one operator,
     e-stop/damping ready, no bystanders.
  3. Feet lightly touching ground.
  4. Free-standing test only after torque/temperature/latency/joint-target behavior has been
     reviewed.
  5. Never send SDK control commands and the built-in motion controller at the same time —
     the two fight each other and cause shaking (per `unitree_rl_mjlab`'s own deployment
     README).

None of this is runnable without physical G1 hardware; documented here so the eventual
hardware step follows the same staged-rollout discipline the original plan called for.
