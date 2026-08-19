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

**1. `bad_orientation` could not fire at all — now fixed in this overlay.** Upstream's
`deploy/include/isaaclab/envs/mdp/terminations.h` was:

```cpp
inline bool bad_orientation(ManagerBasedRLEnv* env, float limit_angle = 1.0)
{
    auto & data = asset->data.projected_gravity_b;
//    return std::fabs(std::acos(-data[2])) > limit_angle;
    return false;
}
```

The real test commented out, `false` returned unconditionally — and every robot in the repo
registers this as its tip-over fallback. So on 2026-07-31 `bad_orientation` did not fail to
fire because the robot stayed within threshold: **the check did not exist at runtime**. The
robot could have gone fully inverted and the FSM would have stayed in Mimic. Until this was
fixed, the operator's reaction time was the only tip-over protection there was.

The patch re-enables it with two guards, because a false positive here is not free — the
transition lands in `Passive`, which is zero-stiffness, so a spurious trip drops a robot that
was doing fine:

- `projected_gravity_b` is derived from the IMU quaternion, so before the first valid
  lowstate (or on a corrupt frame) it is not a unit vector. Anything that is not one is
  treated as *no reading* rather than as a reading of zero tilt. The acos argument is also
  clamped: NaN would compare false and put us straight back to the silent failure.
- Three consecutive violations are required before acting. A genuine tip-over holds for far
  longer; a single bad frame does not.

`deploy_overlay/test_bad_orientation.cpp` exercises the logic standalone — upright, 45 deg
(under the 57 deg limit), 70 deg, fully inverted, all-zero / NaN / un-normalised readings,
a single bad frame between good ones, and the hold window itself. Eleven cases, all passing
under `g++ -std=c++17 -Wall -Wextra -Werror`:

```bash
g++ -std=c++17 -Wall -Wextra -Werror -O2 -o /tmp/t deploy_overlay/test_bad_orientation.cpp && /tmp/t
```

The standalone test exists because the deploy binary needs unitree_sdk2, DDS and cnpy and
only builds on the robot — **this change has not been compiled in situ or run on hardware.**
Build it on the Jetson before trusting it. `scripts/08_sim2sim_fsm.py` now defaults to the
restored guard and reproduces the behaviour end to end (start the dance with the robot on
the floor and the FSM drops to Passive within three checks); `--no-tipover-guard` restores
upstream's dead-code behaviour for comparison.

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

**Resolved (was open): the harness, not the policy.** The tipping reported here earlier was
a fidelity bug in `08_sim2sim_fsm.py`, since fixed: it computed the motor torque once per
50Hz control period and held it across all ten physics steps. On the robot the target `q`
goes out at 50Hz but each motor closes its own PD loop continuously at 500Hz-2kHz, so
holding torque constant is a zero-order hold on *torque* instead of on *position* --
it strips the damping term of its authority exactly when a joint is moving fastest.
Recomputing the PD every physics step, target held, changed the outcome completely:

| | before | after |
|---|---|---|
| max torso tilt over the dance | 118 deg | **13.9 deg** |
| worst joint tracking error | 40-57 deg | **7-16 deg** |
| outcome | tipped ~1s after release | **completes all 23.6s upright** |

Worth internalising rather than filing away: most of this policy's apparent stability is
supplied by the motor loop running far faster than the policy. Anything that degrades that
loop on hardware -- a slow control thread, a dropped lowcmd cycle, motors in a mode where
they hold torque rather than track position -- takes away the same margin this bug did.

**That also resizes defects 2 and 3.** With the harness fixed, toggling either changes max
tilt by a few tenths of a degree (13.9 baseline, 13.4 anchor-fixed, 13.3 gain-map-fixed,
13.2 both). They are real defects in the deployed code and worth fixing, but they are not
what makes or breaks the run, and the earlier framing here overstated them. Defect 1 is the
one that matters, and it is a safety hole rather than a performance one.

**Suspended vs on the ground.** `--start hold` reproduces the 2026-07-31 setup. Suspended,
joint tracking error runs 23-35 deg on the hips and ankles against 12-16 deg with the feet
loaded -- the legs have nothing to push against, so the same policy tracks two to three
times worse. That is the mechanism the hardware run showed, reproduced in sim, and it stands
independently of how far the torso tips (the pelvis is pinned, so tilt reads zero by
construction).

Related: **FixStand does not balance.** It is a fixed-pose PD, so a free-standing G1 topples
in ~1.5s in sim even at 20x its configured gains. The harness's `--support` models the
gantry/operator holding the robot until the policy takes over. Worth knowing before planning
a feet-on-the-ground test around FixStand holding the robot up between runs.

## Standing when the dance ends — fixed (2026-08-19)

Run the whole sequence and watch past `time_end`, which nothing had done before:

```bash
conda run -n unitree_rl_mjlab python scripts/08_sim2sim_fsm.py \
  --start supine --support --passive-secs 2 --stand-secs 4 --time-end 23.6 --duration 45
```

The robot is lifted off the floor and dances all 23.6s upright at **13.7 deg** peak torso
tilt. What happened next depended entirely on whether the FSM handed back:

| when the motion ends | peak torso tilt after | lowest pelvis | |
|---|---|---|---|
| hand back to `end_state` | 96 deg | 0.09 m | goes down |
| `hold_after_end: true` | **6 deg** | **0.78 m** | **stays up, indefinitely** |

Handing back drops the robot whatever it hands back to, and that is a property of the FSM,
not of the policy. `FixStand` is a fixed-pose PD with no balance authority — on its own it
topples a free-standing G1 in ~1.5s even at 20x its configured gains. `Passive` is zero
stiffness by definition. `Velocity`, the one state that could catch the robot, ships without
weights and is disabled. **No enabled end state leaves this robot standing.**

So the overlay stops handing back. `hold_after_end` makes `State_Mimic` skip registering the
timeout transition entirely; `MotionLoader_::update` already clamps the phase to
`[0, duration]`, so the policy simply goes on tracking the final frame and keeps balancing
around it — it is the only thing in this FSM that can. The final frame of erik_dali is a
plausible stance to hold: the legs are within 20 deg of the FixStand pose, root height
0.833m, joint velocities under 0.4 rad/s (the arms finish raised, which is the choreography).

Checked at `time_end` 5, 10, 15 and 23.6 — that is, mid-dance freezes as well as the motion's
own end — all four stayed up, so the staged 5 -> 10 -> 23.6 ramp works with this on.

The exit is deliberate now rather than automatic: `bad_orientation` still fires if the robot
goes over, and the operator still has `LT+B`. Both land in `Passive`, which is zero-torque —
**support the robot before ending the run.**

Recorded: `reference_motion/retargeted/sim2sim_full_sequence.mp4` — floor, lift, the full
dance, and still standing after it. 45s at 50fps. (Local only; `*.mp4` is gitignored.)

### Correction to an earlier number here

The first version of this section reported the handover fall as "peak tilt 90 deg, pelvis to
0.160 m". That measurement was contaminated: `--support` was re-engaging after the dance and
teleporting the robot back to its *start* pose, and 0.160 m is exactly the supine spawn
height. `--support` is now bring-up only, and `--catch` is the separate switch for an
operator taking the robot at the handover. The corrected figures are in the table above and
the conclusion is unchanged — handing back does drop the robot — but the number was wrong.

## Next hardware step

Feet on the ground, 5s, per `docs/PIPELINE.md` §5. Not yet done. The deploy binary
(`deploy/robots/g1_23dof`) has not been built either.
