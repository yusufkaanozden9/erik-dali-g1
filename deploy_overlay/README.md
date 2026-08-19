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

## Next hardware step

Feet on the ground, 5s, per `docs/PIPELINE.md` §5. Not yet done. The deploy binary
(`deploy/robots/g1_23dof`) has not been built either.
