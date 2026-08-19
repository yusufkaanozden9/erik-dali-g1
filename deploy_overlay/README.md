# Deploy overlay — everything the vendored clone would otherwise eat

`third_party/unitree_rl_mjlab` is gitignored (it's pulled by `setup_envs.sh`) and sits on a
detached HEAD. Every change the hardware bring-up needed was made *inside* that clone, so a
single `setup_envs.sh` re-run — or a `git checkout .` in there — silently destroys all of it,
including the one thing that cost real robot time to learn. This directory is the tracked
copy. Re-apply with:

```bash
./deploy_overlay/apply.sh
```

Patch base commit: `1425b15f73bd4095f0df53709d7c389c3eb9e790` ("Fix the warnings during
rough-terrain training", 2026-04-13).

## What's in here

| Path | What |
|---|---|
| `unitree_rl_mjlab.patch` | 5 C++ files — FSM transition-reason logging + the lowcmd-channel guard |
| `files/.../config/config.yaml` | `Mimic_ErikDali` FSM state (id 6) + the `time_end: 5.0` safety cap |
| `files/.../mimic/erik_dali/params/deploy.yaml` | byte-identical copy of `dance1_subject2`'s |

`policy.onnx` and `erik_dali.npz` are deliberately **not** here — binaries stay out of git per
`.gitignore`. `apply.sh` copies them from `trained_model/`.

## The hardware run this all came out of (2026-07-31)

First real-robot test, stage 2 of the staged rollout: **suspended, feet dangling**. The robot
ran ~8s of the dance cleanly, then diverged badly by ~11s. `bad_orientation` never fired — the
operator stopped it by hand.

Why: a position-controlled tracking policy suspended in the air has no ground reaction to push
against, so its corrections accumulate instead of settling. **The 100% pass rate from
`scripts/07_evaluate_policy.py` does not cover this regime** — every episode in that evaluation
was ground-contact. Suspended is *outside* the training distribution, not a weaker version of it.

Two things that made the run harder to read than it should have been, both fixed below: nothing
in the logs said *why* the FSM did or didn't change state, and the "another process owns the
lowcmd channel" guard printed a warning and then carried on anyway.

## The changes, one by one

**`config.yaml` — `Mimic_ErikDali`, FSM id 6.** Entered with `RB + X`, `LT + B` returns to
`Passive` (emergency zero-torque). Same generic `State_Mimic` C++ class the vendored
`dance1_subject2` example uses — there is no separate Unitree "dance mode" API, it's
config-driven and needs no rebuild.

**`config.yaml` — `time_end: 5.0`.** The dance stops itself after 5s and returns to `FixStand`,
so the *next* test (feet on the ground) is bounded by config rather than by operator reaction
time. Raise in steps — 5 → 10 → 23.6 — as each length is seen to behave. 23.6s is the motion's
own full duration.

**`config.yaml` — `end_state: FixStand`.** `State_Mimic` defaults `end_state` to `Velocity` and
resolves it through an unguarded `FSMStringMap.right.at()`, which throws when `Velocity` is
disabled — and it *is* disabled here, along with `Mimic_Dance1_subject2`, because upstream ships
their configs but no trained weights. `end_state` must name an enabled state.

**`BaseState.h` / `CtrlFSM.h` / `FSMState.h` / `State_Mimic.cpp` — labelled FSM checks.**
Upstream logs only `changed from X to Y`, which is ambiguous exactly when it matters: operator
e-stop, comms dropout, motion finishing, and `bad_orientation` tripping all land in `Passive` and
print the same line. `register_check(pred, target, label)` carries a reason string alongside each
check, and the transition log becomes `... [reason: bad_orientation (policy lost the torso
upright, threshold 1.0)]`. Labels live in a vector parallel to `registered_checks` rather than
widening that type, so unlabelled registrations elsewhere still compile and just report
`unlabelled`.

**`main.cpp` — re-enabled `exit()` on the lowcmd-channel guard.** Upstream ships this `exit(0)`
commented out, so the guard warns and then runs anyway, leaving two controllers publishing lowcmd
at once — the exact failure unitree_rl_mjlab's own deployment README says produces shaking. Seen
live on 2026-07-31. Now `exit(1)`, non-zero so a wrapper/service can tell it from a clean
shutdown. If it trips: the robot isn't in debug mode — put it in zero-torque, press `L2+R2`,
start again.

## Three defects in the deploy path, found by sim2sim (2026-08-19)

`scripts/08_sim2sim_fsm.py` runs these exact artifacts against MuJoCo. Its observation
pipeline was cross-checked term-by-term against mjlab's own training env and matches to
1e-3 (the anchor term to 0.000000), so what follows is about the deploy code, not the
harness.

**1. `bad_orientation` cannot fire. At all.** `deploy/include/isaaclab/envs/mdp/terminations.h`:

```cpp
inline bool bad_orientation(ManagerBasedRLEnv* env, float limit_angle = 1.0)
{
    auto & data = asset->data.projected_gravity_b;
//    return std::fabs(std::acos(-data[2])) > limit_angle;
    return false;
}
```

The real test is commented out and the function returns `false` unconditionally. This is
unmodified upstream, and every robot in the repo registers it as their tip-over fallback.
So on 2026-07-31 `bad_orientation` did not "fail to fire" because the robot stayed within
threshold — **the check does not exist at runtime**. The robot could be fully inverted and
the FSM would stay in Mimic. Right now the only thing between a diverging policy and the
hardware is the operator's reaction time. Fix before the feet-on-the-ground test.

**2. Mimic gains are applied positionally, not through `joint_ids_map`.** `State_Mimic::enter`
(and `State_RLBase::enter`) do `for (i < joint_stiffness.size()) motor_cmd[i].kp() = ...`,
while `run()` writes targets through `joint_ids_map`. Every other robot in the repo has an
identity map, so this is invisible there. The 23-DoF G1 does not: its map skips waist
roll/pitch and the wrist pitch/yaw pairs, so gains land on the wrong motors and **motors
23-26 -- right shoulder roll, right shoulder yaw, right elbow, right wrist roll -- are never
reassigned at all**. They dance with FixStand's `kp=40, kd=10` instead of the policy's
`14.3 / 0.9`: 2.8x stiff, 11x over-damped. In sim they show up as the worst-tracking joints
after the ankles, and drop out of the top offenders entirely under `--fix-gain-map`.

**3. `init_quat` is built from the wrong reference frame.** `State_Mimic::enter` computes

```cpp
auto ref_yaw = yawQuaternion(motion->root_quaternion()).toRotationMatrix();   // PELVIS
auto robot_yaw = yawQuaternion(robot_quat_w(env.get())).toRotationMatrix();   // TORSO
init_quat = robot_yaw * ref_yaw.transpose();
```

but the observation it feeds uses `motion_anchor_quat_w`, which is the reference **torso**
(pelvis o waist_yaw). Mixing the two leaves a constant yaw error equal to the reference's
waist yaw at `time_start` -- for erik_dali that is -0.258 rad, **-14.8 degrees**. Measured
directly: place the robot exactly on the reference and `motion_anchor_ori_b` reads
`[0.967, 0.255, -0.255, 0.967, ...]` where training defines identity; with the reference
torso used on both sides it reads exactly `[1, 0, 0, 1, 0, 0]`. The policy is therefore told,
every tick for the whole dance, that its torso is yawed 15 degrees off the reference, and
spends control authority correcting an error that is not there. `--fix-anchor-yaw`.

Whether `dance1_subject2` hides this depends on its own frame-0 waist yaw, which is a
plausible reason it survived upstream.

**Still open:** even with 2 and 3 corrected the policy does not survive the dance in the
sim2sim scene -- it tips within ~1s of the support being released. Unattributed so far.
`scene_g1_23dof.xml` and mjlab's own `g1_23dof.xml` disagree on joint armature (0.01 vs
0.0036), damping (0.05 vs 0), frictionloss (0.2 vs 0) and timestep (0.002 vs 0.005), so a
sim2sim model gap is the leading candidate, but `--match-training-model` makes MuJoCo's
solver too slow to A/B in reasonable time and the question is unresolved. Do not read the
current sim behaviour as a prediction of hardware behaviour in either direction.

Related: **FixStand does not balance.** It is a fixed-pose PD, so a free-standing G1 topples
in ~1.5s in sim even at 20x its configured gains. The harness's `--support` models the
gantry/operator holding the robot until the policy takes over. Worth knowing before planning
a feet-on-the-ground test around FixStand holding the robot up between runs.

## Next hardware step

Feet on the ground, 5s, per `docs/PIPELINE.md` §5. Not yet done. The deploy binary
(`deploy/robots/g1_23dof`) has not been built either.
