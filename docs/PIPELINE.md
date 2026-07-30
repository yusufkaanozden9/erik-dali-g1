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

1. `scripts/01_extract_smplx.sh` — GVHMR: video → SMPL-X motion. **Not yet installed/run**
   (see "What's actually been verified" below — blocked on license-gated models, and its
   own dependency stack (DPVO, HMR2, detectron2-style CV libs) likely needs CUDA to build,
   so this stage specifically may need the Linux GPU box even just to install).
2. `scripts/02_retarget_to_g1.sh` — GMR: SMPL-X → full 29-DoF G1 joint motion
   (`--robot unitree_g1`, GMR's only G1 IK config — confirmed via `smplx_to_robot.py --help`,
   full robot list: unitree_g1, unitree_g1_with_hands, unitree_h1, unitree_h1_2,
   booster_t1[_29dof], stanford_toddy, fourier_n1, engineai_pm01, kuavo_s45, hightorque_hi,
   galaxea_r1pro, berkeley_humanoid_lite, booster_k1, pnd_adam_lite, openloong, tienkung,
   fourier_gr3). Two-step, not one: `smplx_to_robot.py --save_path <dir>/erik_dali.pkl`,
   then `batch_gmr_pkl_to_csv.py --folder <dir>` writes `<dir>/csv/erik_dali.csv` — verified
   by reading the converter's source, which writes `[root_pos(3), root_rot_xyzw(4),
   dof_pos(N)]`, exactly matching what mjlab's `csv_to_npz.py` parses.
3. `scripts/03_csv_to_npz_23dof.sh` — mjlab's `csv_to_npz.py --robot g1_23dof --device cpu`:
   29-DoF CSV → 23-DoF npz, dropping the wrist DoFs the 23-DoF model lacks. **Verify the
   column mapping** against the shipped
   `third_party/unitree_rl_mjlab/src/assets/motions/g1_23dof/dance1_subject2.csv` before
   trusting the output — see the script's own header comment. CLI itself (`--device`, all
   other flags) confirmed working — this exact script converted the bundled example motion
   successfully during the stage-5 smoke test below.
4. `scripts/04_kinematic_sanity_check.sh` — GMR's visualizer (`vis_robot_motion.py --robot
   --robot_motion_path`, confirmed via `--help`), CPU-only, catches self-collision /
   joint-limit / foot-sliding problems before any training compute is spent.

### What's actually been verified vs. still assumed (2026-07-30)

- **Installed + CLI-verified on macOS/CPU**: `unitree_rl_mjlab` (fully — see smoke test
  below) and `GMR` (`pip install -e .` + PyQt6 succeeded; `smplx_to_robot.py`,
  `batch_gmr_pkl_to_csv.py`, `vis_robot_motion.py` all checked against real `--help` output,
  not assumed from docs).
- **SMPL-X body models**: downloaded and verified for GMR (`third_party/gmr/assets/body_models/smplx/`,
  `SMPLX_{NEUTRAL,FEMALE,MALE}.npz`). Gotcha worth flagging: the SMPL-X download site also
  hosts the plain (older) **SMPL** model under a similarly-named download, and a first
  attempt grabbed that by mistake — files were named `SMPLX_NEUTRAL.npz` etc. but only had
  11 npz keys (`J`, `posedirs`, `shapedirs`, `v_template`, `weights`, ...), missing hand PCA
  (`hands_componentsl/r`) entirely, so `smplx.create(..., model_type='smplx')` failed with
  `AttributeError: 'Struct' object has no attribute 'hands_componentsl'`. The correct
  package is `models_smplx_v1_1.zip` (~870MB) from the SMPL-X-specific section of the
  downloads page; genuine SMPL-X npz files have 22 keys and load into a model that produces
  10475 vertices / 127 joints on a forward pass — verified that actually works here.
- **`GVHMR`: confirmed impossible to install on macOS, not just "likely CUDA-only".** Its
  `requirements.txt` hard-pins `torch==2.3.0+cu121` via `--extra-index-url
  https://download.pytorch.org/whl/cu121` (a CUDA-12.1-specific wheel that doesn't exist for
  macOS at all) and, worse, pins `pytorch3d` directly to a wheel URL with a
  `linux_x86_64`platform tag baked into the filename
  (`pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl`) — pip will refuse that on any other
  platform outright, no patching-around-the-edges fix. Checked this by reading the actual
  requirements.txt rather than attempting the install; not worth the rabbit hole of
  hand-patching CUDA-pinned deps just to prove what's already evident from the pins
  themselves. **Stage 1 is a hard Linux+NVIDIA-GPU-box-only step.** Everything downstream of
  it (stages 2-5, all now installed and either verified or ready) can be prepared in advance,
  but the actual Erik Dalı SMPL-X extraction has to happen there.

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

### Smoke test — verified (2026-07-30, macOS arm64, CPU only)

`scripts/05_smoke_test_training.sh` was actually run end-to-end on this machine (no GPU,
no CUDA) against unitree_rl_mjlab's own bundled `dance1_subject2` example. Confirmed
working: task registration (`Unitree-G1-23Dof-Tracking-No-State-Estimation` is a real,
listed task — `python scripts/list_envs.py` shows 29 registered tasks total), CSV→npz
conversion, MuJoCo/mjlab scene construction (124-dim actor obs, 256-dim critic obs, 23-dim
action space — matches the 23-DoF G1 exactly), the full PPO update loop (3 iterations,
~99 steps/s on CPU with 4 parallel envs), and checkpoint + ONNX export
(`model_2.pt`, `policy.onnx`).

Install fixes needed beyond the plain `pip install -e .` (now folded into `setup_envs.sh`):
- **`mujoco` version pin.** `unitree_rl_mjlab`'s `setup.py` pins `mujoco-warp==3.5.0` but
  not `mujoco` itself, so pip installs the newest `mujoco` (3.11.0 at time of writing) —
  which has renamed/removed `mujoco.mjtEnableBit.mjENBL_MULTICCD`, an enum
  `mujoco-warp==3.5.0` still references at import time. Fix: `pip install mujoco==3.5.0`.
- **Missing `scipy`.** `mjlab.terrains.heightfield_terrains` imports `scipy.interpolate`
  but `scipy` isn't in mjlab's own dependency list. Fix: `pip install scipy`.

CLI flags actually differ slightly from what the earlier plan assumed (corrected in the
scripts):
- `scripts/train.py`: `--motion-file` (hyphenated, not `--motion_file`), iteration count is
  `--agent.max-iterations` (not `--max_iterations`), and `--gpu-ids` defaults to `'[0]'` —
  pass `--gpu-ids None` to force CPU. Default logger is `wandb`, which errors without a
  configured API key; pass `--agent.logger tensorboard` to avoid that.
- `scripts/csv_to_npz.py`: `--device` defaults to `cuda:0` — pass `--device cpu` on a
  machine without CUDA.

Verified command (run from `third_party/unitree_rl_mjlab/`, `unitree_rl_mjlab` conda env):

```bash
python scripts/csv_to_npz.py --input-file src/assets/motions/g1_23dof/dance1_subject2.csv \
  --output-name dance1_subject2.npz --input-fps 30 --output-fps 50 --robot g1_23dof --device cpu

python scripts/train.py Unitree-G1-23Dof-Tracking-No-State-Estimation \
  --motion-file=src/assets/motions/g1_23dof/dance1_subject2.npz \
  --env.scene.num-envs=4 --agent.max-iterations=3 --agent.logger tensorboard --gpu-ids None
```

This proves the training entrypoint works, without waiting on the retargeting chain
(stages 1-2, blocked on the license-gated SMPL-X body models) or spending real GPU time.

### Real training run — verified (2026-07-30, rented RTX 4090)

Actually launched, on the real `erik_dali.npz` motion (from the user's own reference clip,
retargeted through GVHMR→GMR, converted to 23-DoF): `scripts/06_launch_real_training.sh`,
`--env.scene.num-envs=4096 --agent.max-iterations=5000`.

Measured throughput: **0.96s/iteration, ~102k env-steps/sec** at 4096 envs on a single
RTX 4090. At that rate: 1,000 iters ≈ 16 min, 5,000 iters ≈ 1h20m, mjlab's own default
30,001 iters ≈ 8h. GPU rental cost is trivial at any of these (~$0.30-0.35/hr spot pricing)
— wall-clock time is the actual budget constraint, not money.

**Gotcha that cost real GPU-minutes**: without `PYTHONUNBUFFERED=1` / `python -u`, stdout
is fully block-buffered once redirected to a log file (`... > file.log 2>&1 &`) — training
was genuinely running correctly the whole time, but the log stayed empty for 15-20+ minutes,
looking exactly like a hang. Killed a real run over this before diagnosing it with a smaller
unbuffered re-run. `scripts/06_launch_real_training.sh` and the updated
`scripts/05_smoke_test_training.sh` both set this correctly now.

Also needed on this rented box specifically (not present in the original setup_envs.sh):
- Anaconda's channel Terms-of-Service gate blocks non-interactive `conda create` on fresh
  installs — run `conda tos accept --override-channels --channel
  https://repo.anaconda.com/pkgs/main` (and `.../pkgs/r`) once, first.
- `warp-lang` has no upper version bound in `mjlab==1.2.0`'s own dependency pin
  (`warp-lang>=1.12.0`), so plain `pip install` grabs the newest (1.15.0 as of writing),
  which breaks `mjlab.sim.Simulation`'s CUDA-graph detection
  (`AttributeError: module 'warp' has no attribute 'context'`) — this only shows up when
  actually using a CUDA device, so the earlier CPU-only smoke test never hit it. Fix:
  `pip install warp-lang==1.12.1` (mjlab's declared minimum).

Curriculum and domain randomization (start on a static pose, ramp tempo, ramp
randomization strength) as described in the original plan were **not** applied for this
first run — it trains directly on the full-tempo, full-amplitude Erik Dalı clip from
iteration 0. Revisit `src/tasks/tracking/config/g1_23dof/env_cfgs.py` if the resulting
policy struggles with the faster/more dynamic sections of the dance (see
`reference_motion/NOTES.md`'s note about motion blur on fast segments).

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
