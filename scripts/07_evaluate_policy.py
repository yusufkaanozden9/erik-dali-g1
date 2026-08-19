"""Stage 7: headless, multi-seed robustness evaluation of a trained tracking policy.

`scripts/play.py` always ends in an interactive viewer, so it answers "does this
look right?" but not "how often does it fall?". docs/PIPELINE.md §4 asks for the
latter before any free-standing hardware test: *"Success = many episodes across
many randomization seeds, not one lucky rollout."*

This runs the policy headless across many parallel environments and reports, per
scenario, how many episodes survived the full motion and how the rest failed.

Two scenarios, because they answer different questions:

  nominal  push_robot disabled. Startup randomization (CoM offset, encoder bias,
           foot friction) and observation noise stay on. This is the closest
           analogue to the real robot standing on a flat floor with nobody
           touching it — the number that matters for "can it dance free-standing".

  stress   push_robot enabled: a random velocity impulse every 1-3s, the same
           disturbance used during training. This is a robustness margin, not a
           pass/fail gate — a policy that only fails here is still deployable,
           it just has little headroom.

Both start every episode at t=0 of the motion (`sampling_mode="start"`, RSI
pose/velocity randomization cleared), matching the real FSM where the dance is
entered from a fixed FixStand pose rather than a random point mid-motion.

Terminations (from src/tasks/tracking/tracking_env_cfg.py) are reported
separately rather than lumped into one "fail" count, since they mean different
things on hardware:

  anchor_pos    torso Z is >0.25m off the reference -> the robot collapsed
  anchor_ori    torso orientation is >0.8 off        -> the robot tipped over
  ee_body_pos   an ankle/wrist Z is >0.25m off       -> tracking diverged
  time_out      episode reached the end              -> SUCCESS, not a failure

Usage (from the repo root, in the `unitree_rl_mjlab` conda env):

    conda run --no-capture-output -n unitree_rl_mjlab python -u \
      scripts/07_evaluate_policy.py --num-envs 64

Runs on CPU by default; pass --device cuda:0 on a GPU box.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MJLAB_DIR = REPO_ROOT / "third_party" / "unitree_rl_mjlab"
# mjlab's task registry lives under `src.tasks`, which is only importable with the
# vendored checkout on sys.path *and* as the working directory (its asset paths are
# resolved relative to cwd) — so do both here rather than requiring the caller to cd.
sys.path.insert(0, str(MJLAB_DIR))
os.chdir(MJLAB_DIR)

import torch  # noqa: E402

TASK_ID = "Unitree-G1-23Dof-Tracking-No-State-Estimation"
# Anything that is not `time_out`. Ordered most- to least-severe for reporting.
FAILURE_TERMS = ("anchor_ori", "anchor_pos", "ee_body_pos")


def build_env(motion_file: Path, episode_length_s: float, device: str, num_envs: int, pushes: bool):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.tasks.tracking.mdp import MotionCommandCfg

    # play=False keeps the training config's domain randomization and terminations;
    # play=True would strip exactly the noise this evaluation exists to measure.
    env_cfg = load_env_cfg(TASK_ID, play=False)
    agent_cfg = load_rl_cfg(TASK_ID)

    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_file)
    motion_cmd.sampling_mode = "start"
    # Clear reference-state-initialization jitter: on hardware the dance always
    # begins from the same FixStand pose, so a randomized start would measure a
    # situation that cannot occur.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = episode_length_s
    if not pushes:
        env_cfg.events.pop("push_robot", None)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions), agent_cfg


def load_policy(env, agent_cfg, checkpoint: Path, device: str):
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_runner_cls

    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    return runner.get_inference_policy(device=device)


def run_scenario(name: str, checkpoint: Path, motion_file: Path, duration_s: float,
                 num_envs: int, device: str, pushes: bool, seed: int) -> dict:
    print(f"\n{'=' * 74}\n  SENARYO: {name}  (itme {'ACIK' if pushes else 'KAPALI'}, "
          f"{num_envs} env, {duration_s:.1f}s)\n{'=' * 74}", flush=True)

    env, agent_cfg = build_env(motion_file, duration_s, device, num_envs, pushes)
    env.seed(seed)
    policy = load_policy(env, agent_cfg, checkpoint, device)

    obs, _ = env.reset()
    term_mgr = env.unwrapped.termination_manager
    max_steps = int(duration_s / env.unwrapped.step_dt)

    # Warm-up step, discarded. Straight after `reset()` the motion command's
    # reference buffers (`body_pos_relative_w`) are still all-zero — they are only
    # filled by `_update_command()` during the first `step()`. The terminations are
    # evaluated against that zero reference, so every env trips `ee_body_pos` with a
    # bogus ~0.9m "error" (the robot's true body height measured against zero) and
    # mjlab auto-resets it. Measured immediately afterwards the real tracking error
    # is 0.004-0.015m. Stepping once before recording skips the artifact; the
    # auto-reset has already put every env back at t=0, so nothing is lost.
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))

    # Record only each env's *first* failure: mjlab auto-resets a terminated env,
    # so without this an early faller would keep contributing later episodes and
    # skew the counts.
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    fail_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    fail_kind = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    for step in range(max_steps):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))

        for kind, term_name in enumerate(FAILURE_TERMS):
            newly = term_mgr.get_term(term_name) & alive
            if newly.any():
                fail_step[newly] = step
                fail_kind[newly] = kind
                alive &= ~newly

        if step % 250 == 0 or step == max_steps - 1:
            print(f"    adim {step + 1:>4}/{max_steps}  ayakta: {int(alive.sum())}/{num_envs}", flush=True)
        if not alive.any():
            print("    (tum env'ler dustu, erken bitiriliyor)", flush=True)
            break

    survived = int(alive.sum())
    result = {
        "name": name, "pushes": pushes, "num_envs": num_envs,
        "survived": survived, "rate": survived / num_envs,
        "by_term": {t: int((fail_kind == i).sum()) for i, t in enumerate(FAILURE_TERMS)},
        "fail_times": (fail_step[fail_kind >= 0].float() * env.unwrapped.step_dt).cpu().numpy(),
    }
    env.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "trained_model" / "model_4999.pt")
    p.add_argument("--motion-file", type=Path, default=REPO_ROOT / "trained_model" / "erik_dali.npz")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenario", choices=("both", "nominal", "stress"), default="both")
    args = p.parse_args()

    for path, label in ((args.checkpoint, "checkpoint"), (args.motion_file, "motion file")):
        if not path.exists():
            sys.exit(f"[HATA] {label} bulunamadi: {path}")

    import numpy as np
    duration_s = float(np.load(args.motion_file)["joint_pos"].shape[0] / 50.0)

    print(f"checkpoint : {args.checkpoint}")
    print(f"motion     : {args.motion_file}  ({duration_s:.2f}s)")
    print(f"device     : {args.device}   env sayisi: {args.num_envs}   seed: {args.seed}")

    import mjlab.tasks  # noqa: F401  (populates the task registry)
    import src.tasks  # noqa: F401

    scenarios = [("nominal", False), ("stress", True)]
    if args.scenario != "both":
        scenarios = [s for s in scenarios if s[0] == args.scenario]

    results = [
        run_scenario(name, args.checkpoint, args.motion_file, duration_s,
                     args.num_envs, args.device, pushes, args.seed)
        for name, pushes in scenarios
    ]

    print(f"\n{'=' * 74}\n  OZET\n{'=' * 74}")
    for r in results:
        print(f"\n  {r['name'].upper()}  (itme {'acik' if r['pushes'] else 'kapali'})")
        print(f"    tam hareketi tamamlayan : {r['survived']}/{r['num_envs']}  (%{100 * r['rate']:.1f})")
        for term, n in r["by_term"].items():
            if n:
                label = {"anchor_ori": "govde devrildi", "anchor_pos": "govde cokti",
                         "ee_body_pos": "ayak/el takibi koptu"}[term]
                print(f"    {term:<14} {n:>3} env  ({label})")
        if len(r["fail_times"]):
            ft = r["fail_times"]
            print(f"    ilk dusme  t={ft.min():.1f}s   ortanca t={float(__import__('numpy').median(ft)):.1f}s")

    nominal = next((r for r in results if r["name"] == "nominal"), None)
    if nominal:
        rate = nominal["rate"]
        print(f"\n  DEGERLENDIRME (nominal senaryo, %{100 * rate:.1f} basari):")
        if rate >= 0.95:
            print("    Serbest ayakta denemek icin sim tarafi ikna edici. Yine de once")
            print("    asama 3 (ayaklar yere hafif degiyor) yapilmali.")
        elif rate >= 0.8:
            print("    Sinirda. Askida/destekli denemeye devam; serbest ayakta oncesi")
            print("    daha uzun egitim (mjlab varsayilani 30.001 iter) dusunulmeli.")
        else:
            print("    SERBEST AYAKTA DENEMEYIN. Policy simulasyonda bile guvenilir")
            print("    degil; once egitim suresi/curriculum/domain randomization")
            print("    (docs/PIPELINE.md §3 sonu) revize edilmeli.")


if __name__ == "__main__":
    main()
