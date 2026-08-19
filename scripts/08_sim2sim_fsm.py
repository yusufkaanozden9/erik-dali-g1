"""Stage 8: sim2sim of the *deploy* FSM in MuJoCo — Passive -> FixStand -> Mimic_ErikDali.

`scripts/play.py` runs the training-time policy inside mjlab's training env: the robot is
spawned already standing at the reference pose, observations come from mjlab's own pipeline,
and nothing about the deployed artifacts is exercised. That answers "did training work". It
does not answer "does the thing we actually ship to the robot behave", which is the question
that matters before the next hardware test.

This script runs the deployed artifacts instead:

  * policy.onnx          — the exported network, not the .pt checkpoint
  * params/deploy.yaml   — joint_ids_map, per-joint stiffness/damping, action scale/offset,
                           and the observation list, parsed the same way
                           isaaclab::ManagerBasedRLEnv parses it
  * config/config.yaml   — the FSM: Passive gains, FixStand kp/kd/ts/qs, Mimic time_end

and reimplements the deploy C++ FSM (State_Passive / State_FixStand / State_Mimic) against
MuJoCo's `scene_g1_23dof.xml`, which is unitree_rl_mjlab's own sim2sim scene: its sensors are
laid out exactly like a G1 LowState message (29 motor pos, 29 motor vel, 29 motor torque, IMU
quaternion, IMU gyro). So the robot is observed through the same signals the real one reports,
not through simulator ground truth.

Crucially, the MJCF's actuator order *is* the real G1's 29-motor order (0-11 legs, 12-14
waist, 15-21 left arm, 22-28 right arm), so `joint_ids_map` indexes it directly — the same
index arithmetic the deploy code does, with the same off-by-mapping consequences (see
--fix-gain-map below).

Stages, mirroring what the operator does on hardware:

  Passive     kp=0, kd from config; motor targets track measured position. The robot is limp.
              This is where it lies on the floor.
  FixStand    kp/kd from config; linear interpolation from the pose held at entry to
              config's qs[1] over ts seconds. On hardware this is `LT + up`.
  Mimic       kp/kd from deploy.yaml; policy.onnx at 50Hz; q_target = action*scale + offset
              written through joint_ids_map. On hardware this is `RB + X`.

Usage (repo root, `unitree_rl_mjlab` conda env — mjpython is required for the viewer on macOS):

    conda run -n unitree_rl_mjlab mjpython scripts/08_sim2sim_fsm.py
    conda run -n unitree_rl_mjlab mjpython scripts/08_sim2sim_fsm.py --start hold
    conda run -n unitree_rl_mjlab python  scripts/08_sim2sim_fsm.py --headless --duration 20

Keys in the viewer: [1] force Passive  [2] force FixStand  [3] force Mimic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MJLAB_DIR = REPO_ROOT / "third_party" / "unitree_rl_mjlab"
DEPLOY_DIR = MJLAB_DIR / "deploy" / "robots" / "g1_23dof"
SCENE_XML = MJLAB_DIR / "src/assets/robots/unitree_g1/xmls/scene_g1_23dof.xml"

N_MOTORS = 29  # the G1's lowcmd motor array, which the MJCF actuator order matches
MOTION_FPS = 50.0  # State_Mimic::MotionLoader_ hardcodes dt = 1/50
WAIST_YAW_MOTOR = 12  # State_Mimic composes torso yaw from this motor


# ----------------------------------------------------------------------------------
# Quaternion helpers. MuJoCo and the deploy code both use (w, x, y, z).
# ----------------------------------------------------------------------------------
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  aw, ax, ay, az = a
  bw, bx, by, bz = b
  return np.array([
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ])


def quat_conj(q: np.ndarray) -> np.ndarray:
  return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_z(angle: float) -> np.ndarray:
  return np.array([np.cos(angle * 0.5), 0.0, 0.0, np.sin(angle * 0.5)])


def yaw_quat(q: np.ndarray) -> np.ndarray:
  """isaaclab::yawQuaternion — strip everything but rotation about world Z."""
  w, x, y, z = q
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return quat_from_z(yaw)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
  m = np.zeros(9)
  mujoco.mju_quat2Mat(m, np.ascontiguousarray(q, dtype=np.float64))
  return m.reshape(3, 3)


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, np.ascontiguousarray(mat.reshape(9), dtype=np.float64))
  return q


# ----------------------------------------------------------------------------------
# Motion — a direct port of State_Mimic::MotionLoader_
# ----------------------------------------------------------------------------------
class MotionLoader:
  def __init__(self, npz_path: Path):
    data = np.load(npz_path)
    self.dt = 1.0 / MOTION_FPS
    self.joint_pos = data["joint_pos"].astype(np.float64)
    self.joint_vel = data["joint_vel"].astype(np.float64)
    # body index 0 is the root; the C++ loader reads the first body's pose per frame.
    self.root_pos = data["body_pos_w"][:, 0, :].astype(np.float64)
    self.root_quat = data["body_quat_w"][:, 0, :].astype(np.float64)
    self.num_frames = self.joint_pos.shape[0]
    self.duration = self.num_frames * self.dt
    self.frame = 0

  def update(self, t: float) -> None:
    phase = float(np.clip(t, 0.0, self.duration))
    self.frame = min(int(np.floor(phase / self.dt)), self.num_frames - 1)

  def anchor_quat(self) -> np.ndarray:
    """motion_anchor_quat_w: reference root quat composed with the reference waist yaw."""
    return quat_mul(self.root_quat[self.frame], quat_from_z(self.joint_pos[self.frame][WAIST_YAW_MOTOR]))


# ----------------------------------------------------------------------------------
# The robot as the deploy code sees it: a LowState message read off MuJoCo's sensors.
# ----------------------------------------------------------------------------------
class LowState:
  """scene_g1_23dof.xml's sensor block is laid out exactly like a G1 LowState."""

  def __init__(self, model: mujoco.MjModel):
    adr = {}
    for i in range(model.nsensor):
      adr[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)] = (
        model.sensor_adr[i], model.sensor_dim[i]
      )
    for required in ("imu_quat", "imu_gyro"):
      if required not in adr:
        raise RuntimeError(f"{SCENE_XML.name} has no '{required}' sensor")
    self._quat_adr = adr["imu_quat"][0]
    self._gyro_adr = adr["imu_gyro"][0]
    # Motor pos/vel/torque sensors come first, one block each, in actuator order.
    self._q_adr, self._dq_adr = 0, N_MOTORS

  def read(self, data: mujoco.MjData):
    self.q = data.sensordata[self._q_adr:self._q_adr + N_MOTORS].copy()
    self.dq = data.sensordata[self._dq_adr:self._dq_adr + N_MOTORS].copy()
    self.imu_quat = data.sensordata[self._quat_adr:self._quat_adr + 4].copy()
    self.gyro = data.sensordata[self._gyro_adr:self._gyro_adr + 3].copy()
    return self

  def torso_quat(self) -> np.ndarray:
    """robot_quat_w: pelvis IMU quat composed with the measured waist yaw."""
    return quat_mul(self.imu_quat, quat_from_z(self.q[WAIST_YAW_MOTOR]))

  def projected_gravity(self) -> np.ndarray:
    g = np.array([0.0, 0.0, -1.0])
    return quat_to_mat(quat_conj(self.imu_quat)) @ g


# ----------------------------------------------------------------------------------
# FSM
# ----------------------------------------------------------------------------------
class DeployFSM:
  PASSIVE, FIXSTAND, MIMIC = "Passive", "FixStand", "Mimic_ErikDali"

  def __init__(self, cfg: dict, deploy_cfg: dict, motion: MotionLoader,
               session: ort.InferenceSession, args: argparse.Namespace):
    self.cfg, self.deploy_cfg, self.motion, self.session, self.args = (
      cfg, deploy_cfg, motion, session, args
    )
    fsm = cfg["FSM"]
    self.passive_kd = np.asarray(fsm["Passive"]["kd"], dtype=np.float64)
    self.stand_kp = np.asarray(fsm["FixStand"]["kp"], dtype=np.float64)
    self.stand_kd = np.asarray(fsm["FixStand"]["kd"], dtype=np.float64)
    self.stand_ts = np.asarray(fsm["FixStand"]["ts"], dtype=np.float64)
    self.stand_qs = [np.asarray(q, dtype=np.float64) for q in fsm["FixStand"]["qs"]]

    mimic = fsm[self.MIMIC]
    self.step_dt = float(deploy_cfg["step_dt"])
    self.joint_ids_map = np.asarray(deploy_cfg["joint_ids_map"], dtype=int)
    self.n_policy = len(self.joint_ids_map)
    self.default_joint_pos = np.asarray(deploy_cfg["default_joint_pos"], dtype=np.float64)
    self.stiffness = np.asarray(deploy_cfg["stiffness"], dtype=np.float64)
    self.damping = np.asarray(deploy_cfg["damping"], dtype=np.float64)
    act = deploy_cfg["actions"]["JointPositionAction"]
    self.act_scale = np.asarray(act["scale"], dtype=np.float64)
    self.act_offset = np.asarray(act["offset"], dtype=np.float64)
    self.obs_terms = list(deploy_cfg["observations"].keys())

    self.time_start = float(np.clip(mimic.get("time_start", 0.0), 0.0, motion.duration))
    time_end = args.time_end if args.time_end is not None else mimic.get("time_end", motion.duration)
    self.time_end = float(np.clip(time_end, 0.0, motion.duration))
    self.end_state = mimic.get("end_state", "Velocity")

    # Motor effort limits. G1_23DOF_ACTION_SCALE is defined as 0.25 * effort / stiffness,
    # so the limits the training actuators enforced are recoverable exactly from the two
    # numbers deploy.yaml already carries.
    self.effort_limit = np.full(N_MOTORS, np.inf)
    self.effort_limit[self.joint_ids_map] = self.act_scale * self.stiffness / 0.25

    self.state = None
    self.kp = np.zeros(N_MOTORS)
    self.kd = np.zeros(N_MOTORS)
    self.q_des = np.zeros(N_MOTORS)
    self.raw_action = np.zeros(self.n_policy)
    self.episode_length = 0
    self.t_enter = 0.0
    self.init_quat = np.array([1.0, 0.0, 0.0, 0.0])
    self._tipover_warned = False
    self.track_err = np.zeros(self.n_policy)   # sum |q_measured - q_commanded|
    self.track_n = 0
    self.max_tilt = 0.0
    self.mimic_kp = np.zeros(N_MOTORS)   # snapshot: the report runs after Mimic exits
    self.mimic_kd = np.zeros(N_MOTORS)

  # -- transitions -----------------------------------------------------------------
  def enter(self, state: str, low: LowState, t: float, reason: str) -> None:
    if self.state is not None:
      print(f"FSM: Change state from {self.state} to {state} [reason: {reason}]")
    else:
      print(f"FSM: enter {state} [reason: {reason}]")
    self.state = state
    self.t_enter = t
    self._tipover_warned = False

    if state == self.PASSIVE:
      self.kp[:] = 0.0
      self.kd[:] = self.passive_kd
      self.q_des[:] = low.q
    elif state == self.FIXSTAND:
      self.kp[:] = self.stand_kp
      self.kd[:] = self.stand_kd
      self.stand_qs[0] = low.q.copy()  # State_FixStand::enter overwrites qs[0] with q0
    elif state == self.MIMIC:
      if self.args.fix_gain_map:
        # What the code means to do: gains follow joint_ids_map, like the targets in run().
        self.kp[self.joint_ids_map] = self.stiffness
        self.kd[self.joint_ids_map] = self.damping
      else:
        # What State_Mimic::enter actually does: `for i < joint_stiffness.size()` writes
        # motor_cmd[i], positionally. Harmless when joint_ids_map is identity (every other
        # robot in the repo); on the 23-DoF G1 it is not identity, so motors 23-26 --
        # right shoulder roll/yaw, right elbow, right wrist roll -- are never reassigned
        # and dance with FixStand's kp/kd instead of the policy's.
        n = len(self.stiffness)
        self.kp[:n] = self.stiffness
        self.kd[:n] = self.damping
      self.mimic_kp[:] = self.kp
      self.mimic_kd[:] = self.kd
      self.raw_action[:] = 0.0
      self.episode_length = 0
      self.motion.update(self.time_start)
      # State_Mimic::enter: init_quat aligns the reference's yaw with the robot's yaw.
      robot_yaw = quat_to_mat(yaw_quat(low.torso_quat()))
      ref_anchor = (self.motion.anchor_quat() if self.args.fix_anchor_yaw
                    else self.motion.root_quat[self.motion.frame])
      ref_yaw = quat_to_mat(yaw_quat(ref_anchor))
      self.init_quat = mat_to_quat(robot_yaw @ ref_yaw.T)
      self.q_des[:] = low.q

  def check_transitions(self, low: LowState, t: float) -> tuple[str, str] | None:
    if self.state != self.MIMIC:
      return None
    if self.episode_length * self.step_dt > self.time_end:
      return self.end_state, "motion finished (reached time_end)"

    # bad_orientation, computed for real. Upstream's isaaclab::mdp::bad_orientation is
    # `return false;` with the real test commented out (deploy/include/isaaclab/envs/mdp/
    # terminations.h), so on hardware this fallback cannot fire no matter how far the robot
    # tips. Reported here either way; --tipover-guard makes it act.
    tilt = abs(np.arccos(np.clip(-low.projected_gravity()[2], -1.0, 1.0)))
    self.max_tilt = max(self.max_tilt, tilt)
    if tilt > self.args.tipover_limit:
      if self.args.tipover_guard:
        return self.PASSIVE, f"bad_orientation (tilt {np.degrees(tilt):.0f} deg)"
      if not self._tipover_warned:
        self._tipover_warned = True
        print(f"  [!] tilt {np.degrees(tilt):.0f} deg exceeds the {np.degrees(self.args.tipover_limit):.0f} deg "
              f"limit at t={t - self.t_enter:.2f}s into the dance -- upstream's "
              f"bad_orientation returns false, so the deployed binary does NOT react.")
    return None

  # -- per-control-step ------------------------------------------------------------
  def observe(self, low: LowState) -> np.ndarray:
    """Build the 124-dim observation in deploy.yaml's declared order."""
    q = low.q[self.joint_ids_map]
    dq = low.dq[self.joint_ids_map]
    frame = self.motion.frame
    parts = {
      "motion_command": np.concatenate([self.motion.joint_pos[frame], self.motion.joint_vel[frame]]),
      "motion_anchor_ori_b": self._anchor_ori(low),
      "base_ang_vel": low.gyro,
      "joint_pos_rel": q - self.default_joint_pos,
      "joint_vel_rel": dq,
      "last_action": self.raw_action,
    }
    missing = [t for t in self.obs_terms if t not in parts]
    if missing:
      raise RuntimeError(f"deploy.yaml declares observation terms this script does not implement: {missing}")
    return np.concatenate([parts[t] for t in self.obs_terms]).astype(np.float32)

  def _anchor_ori(self, low: LowState) -> np.ndarray:
    rot_ = quat_mul(quat_conj(quat_mul(self.init_quat, self.motion.anchor_quat())), low.torso_quat())
    rot = quat_to_mat(rot_).T
    return np.array([rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1], rot[2, 0], rot[2, 1]])

  def control_step(self, low: LowState, t: float) -> None:
    """One 50Hz tick: what the deploy binary's policy thread + run() do together."""
    if self.state == self.PASSIVE:
      self.q_des[:] = low.q  # State_Passive::run mirrors measured position
    elif self.state == self.FIXSTAND:
      self.q_des[:] = self._interpolate(t - self.t_enter)
    elif self.state == self.MIMIC:
      self.episode_length += 1
      self.motion.update(self.episode_length * self.step_dt + self.time_start)
      obs = self.observe(low)
      action = self.session.run(None, {self.session.get_inputs()[0].name: obs[None, :]})[0]
      self.raw_action = np.asarray(action, dtype=np.float64).reshape(-1)
      processed = self.raw_action * self.act_scale + self.act_offset
      # Error against the target the *previous* tick commanded: how well the motor-level
      # PD actually delivered what the policy asked for, which is what a wrong kp/kd shows up as.
      self.track_err += np.abs(low.q[self.joint_ids_map] - self.q_des[self.joint_ids_map])
      self.track_n += 1
      self.q_des[self.joint_ids_map] = processed

  def _interpolate(self, dt: float) -> np.ndarray:
    """LinearInterpolator.h: piecewise-linear through (ts, qs), clamped at both ends."""
    if dt <= self.stand_ts[0]:
      return self.stand_qs[0]
    if dt >= self.stand_ts[-1]:
      return self.stand_qs[-1]
    i = int(np.searchsorted(self.stand_ts, dt)) - 1
    span = self.stand_ts[i + 1] - self.stand_ts[i]
    a = 0.0 if span <= 0 else (dt - self.stand_ts[i]) / span
    return (1.0 - a) * self.stand_qs[i] + a * self.stand_qs[i + 1]

  def torque(self, low: LowState) -> np.ndarray:
    """The motor-level PD every G1 joint runs at 500Hz: tau = kp(q*-q) + kd(0-dq)."""
    tau = self.kp * (self.q_des - low.q) + self.kd * (0.0 - low.dq)
    return np.clip(tau, -self.effort_limit, self.effort_limit)


# ----------------------------------------------------------------------------------
# Start poses
# ----------------------------------------------------------------------------------
def place_robot(model: mujoco.MjModel, data: mujoco.MjData, start: str,
                stand_pose: np.ndarray) -> None:
  """Put the robot where the hardware procedure would have it before g1_ctrl starts.

  `stand_pose` is config.yaml's FixStand qs[1] -- a 29-vector in *motor* order, which is
  also the MJCF's actuator order, so it maps onto joints via each actuator's transmission.
  """
  mujoco.mj_resetData(model, data)
  if start in ("stand", "hold"):
    for motor in range(N_MOTORS):
      jid = int(model.actuator_trnid[motor, 0])
      data.qpos[model.jnt_qposadr[jid]] = stand_pose[motor]
    data.qpos[:3] = [0.0, 0.0, 0.793 if start == "stand" else 1.30]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
  elif start in ("supine", "prone"):
    # Lying on the floor. Pitch +-90 deg about Y puts the pelvis on its back / front.
    ang = np.pi / 2 if start == "supine" else -np.pi / 2
    data.qpos[:3] = [0.0, 0.0, 0.16]
    data.qpos[3:7] = [np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0]
  else:
    raise ValueError(f"unknown start pose: {start}")
  mujoco.mj_forward(model, data)
  if start == "stand":
    soles = [g for g in range(model.ngeom)
             if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_size[g][0] < 0.01
             and "ankle_roll" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "")]
    if soles:
      clearance = min(data.geom_xpos[g][2] - model.geom_size[g][0] for g in soles)
      data.qpos[2] -= clearance  # put the soles on the floor, not 0.9cm above it
      mujoco.mj_forward(model, data)


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--start", choices=["supine", "prone", "stand", "hold"], default="supine",
                 help="initial pose. 'supine'/'prone' = lying on the floor (default); "
                      "'hold' = pelvis pinned, reproducing the suspended 2026-07-31 test")
  p.add_argument("--passive-secs", type=float, default=None,
                 help="dwell in Passive before FixStand (default: 0 when starting from "
                      "'stand', since Passive is limp and an unsupported robot just falls; "
                      "2s otherwise)")
  p.add_argument("--stand-secs", type=float, default=4.0, help="dwell in FixStand before Mimic")
  p.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = run until closed)")
  p.add_argument("--headless", action="store_true", help="no viewer (plain python, no mjpython needed)")
  p.add_argument("--realtime", action="store_true", help="throttle to wall-clock in headless mode")
  p.add_argument("--fix-gain-map", action="store_true",
                 help="apply Mimic gains through joint_ids_map instead of positionally, "
                      "i.e. what State_Mimic::enter appears to intend")
  p.add_argument("--support", action="store_true",
                 help="hold the pelvis while in Passive/FixStand and let go the instant Mimic\n                      starts -- the gantry/operator that supports the robot during bring-up.\n                      Needed because FixStand is a fixed-pose PD, not a balance controller:\n                      free-standing, the G1 topples in ~1.5s in sim even at 20x its gains,\n                      so without this the policy inherits an already-falling robot.")
  p.add_argument("--fix-anchor-yaw", action="store_true",
                 help="build init_quat from the reference TORSO yaw instead of the reference\n                      PELVIS yaw. State_Mimic::enter uses motion->root_quaternion() (pelvis)\n                      while the observation it feeds uses motion_anchor_quat_w (torso =\n                      pelvis o waist_yaw), so the anchor observation carries a constant yaw\n                      bias equal to the reference waist yaw at time_start -- -14.8 deg for\n                      erik_dali. With this flag the anchor reads exactly identity when the\n                      robot is on the reference, as training defines it.")
  p.add_argument("--match-training-model", action="store_true",
                 help="override the sim2sim scene's joint armature/damping/frictionloss and\n                      timestep with the values the policy was TRAINED against. Unitree's\n                      scene_g1_23dof.xml and mjlab's own g1_23dof asset disagree on all four\n                      (armature 0.01 vs 0.0036, damping 0.05 vs 0, frictionloss 0.2 vs 0,\n                      dt 0.002 vs 0.005). Use this to separate a sim2sim model gap from a\n                      genuine policy failure.")
  p.add_argument("--tipover-guard", action="store_true",
                 help="actually fall back to Passive on bad_orientation (upstream cannot: "
                      "its bad_orientation is hardcoded to false)")
  p.add_argument("--tipover-limit", type=float, default=1.0, help="bad_orientation limit, radians")
  p.add_argument("--time-end", type=float, default=None,
                 help="override config.yaml's Mimic time_end (which is capped at 5s for the "
                      "next hardware test); pass 23.6 to watch the whole dance")
  p.add_argument("--replay", action="store_true",
                 help="kinematic playback of the reference motion itself -- no policy, no\n                      physics, joints and root driven straight from the npz. This is what the\n                      retargeting produced, i.e. the target the policy is chasing.")
  p.add_argument("--policy", type=Path, default=None, help="override policy.onnx path")
  p.add_argument("--motion", type=Path, default=None, help="override motion npz path")
  args = p.parse_args()
  if args.passive_secs is None:
    args.passive_secs = 0.0 if args.start == 'stand' else 2.0

  mimic_dir = DEPLOY_DIR / "config/policy/mimic/erik_dali"
  policy_path = args.policy or (mimic_dir / "exported/policy.onnx")
  motion_path = args.motion or (mimic_dir / "params/erik_dali.npz")
  for path in (SCENE_XML, DEPLOY_DIR / "config/config.yaml", mimic_dir / "params/deploy.yaml",
               policy_path, motion_path):
    if not path.exists():
      print(f"ERROR: missing {path}\nRun ./deploy_overlay/apply.sh first.", file=sys.stderr)
      return 1

  cfg = yaml.safe_load((DEPLOY_DIR / "config/config.yaml").read_text())
  deploy_cfg = yaml.safe_load((mimic_dir / "params/deploy.yaml").read_text())
  motion = MotionLoader(motion_path)
  session = ort.InferenceSession(str(policy_path))

  model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
  if args.match_training_model:
    model.dof_armature[6:] = 0.0036
    model.dof_damping[6:] = 0.0
    model.dof_frictionloss[6:] = 0.0
    model.opt.timestep = 0.005
  data = mujoco.MjData(model)
  low = LowState(model)
  fsm = DeployFSM(cfg, deploy_cfg, motion, session, args)

  obs_dim = session.get_inputs()[0].shape[-1]
  print(f"policy   : {policy_path.relative_to(REPO_ROOT)}  obs {obs_dim} -> act {session.get_outputs()[0].shape[-1]}")
  print(f"motion   : {motion_path.name}  {motion.num_frames} frames, {motion.duration:.2f}s")
  print(f"gains    : Mimic gains applied {'through joint_ids_map (corrected)' if args.fix_gain_map else 'positionally (as deployed)'}")
  print(f"anchor   : init_quat from reference {'torso (corrected)' if args.fix_anchor_yaw else 'pelvis (as deployed)'}")
  print(f"start    : {args.start}   physics {1 / model.opt.timestep:.0f}Hz, policy {1 / fsm.step_dt:.0f}Hz")
  print(f"schedule : Passive {args.passive_secs}s -> FixStand {args.stand_secs}s -> {fsm.MIMIC} "
        f"(time_end {fsm.time_end:.1f}s -> {fsm.end_state})\n")

  stand_pose = np.asarray(cfg['FSM']['FixStand']['qs'][1], dtype=np.float64)
  # Where 'standing' actually is for this model, measured rather than assumed.
  place_robot(model, data, 'stand', stand_pose)
  stand_root = data.qpos[:7].copy()
  place_robot(model, data, args.start, stand_pose)
  start_root = data.qpos[:7].copy()
  supported = args.start == "hold" or args.support
  low.read(data)
  fsm.enter(fsm.PASSIVE, low, 0.0, "startup")

  substeps = max(1, int(round(fsm.step_dt / model.opt.timestep)))
  forced: list[str] = []

  def key_callback(keycode: int) -> None:
    forced.append({49: fsm.PASSIVE, 50: fsm.FIXSTAND, 51: fsm.MIMIC}.get(keycode, ""))

  def tick() -> bool:
    """One 50Hz control period. Returns False when the run should stop."""
    t = data.time
    if args.replay:
      motion.update(t)
      f = motion.frame
      for i, motor in enumerate(fsm.joint_ids_map):
        jid = int(model.actuator_trnid[int(motor), 0])
        data.qpos[model.jnt_qposadr[jid]] = motion.joint_pos[f][i]
      data.qpos[:3] = motion.root_pos[f]
      data.qpos[3:7] = motion.root_quat[f]
      data.qvel[:] = 0.0
      mujoco.mj_forward(model, data)
      data.time += fsm.step_dt
      return not (args.duration and data.time >= args.duration) and t < motion.duration
    low.read(data)

    while forced:
      target = forced.pop()
      if target:
        fsm.enter(target, low, t, "operator (keyboard)")

    nxt = fsm.check_transitions(low, t)
    if nxt is not None:
      fsm.enter(nxt[0], low, t, nxt[1])
    elif fsm.state == fsm.PASSIVE and t - fsm.t_enter >= args.passive_secs and not forced:
      fsm.enter(fsm.FIXSTAND, low, t, f"scheduled (passive dwell {args.passive_secs}s)")
    elif fsm.state == fsm.FIXSTAND and t - fsm.t_enter >= args.stand_secs:
      fsm.enter(fsm.MIMIC, low, t, f"scheduled (stand dwell {args.stand_secs}s)")

    fsm.control_step(low, t)
    for _ in range(substeps):
      # The motor-level PD is a *continuous* loop on the real robot: lowcmd carries the
      # target q, and each motor closes the loop on it at ~500Hz-2kHz on its own. Holding
      # one torque across the whole 20ms control period instead is a zero-order hold on
      # torque rather than on position, which removes most of the damping term's authority
      # exactly when the joint is moving fastest. Recompute per physics step, target held.
      data.ctrl[:] = fsm.torque(low.read(data))
      mujoco.mj_step(model, data)
      # 'hold' pins throughout (the suspended test). --support instead *raises* the robot
      # from wherever it started to standing across the FixStand phase and lets go the
      # moment Mimic begins -- the gantry/operator, not a get-up policy. Without it a
      # robot that starts on the floor stays on the floor: FixStand only drives joints to
      # a pose, it has no notion of standing up.
      if supported and (args.start == 'hold' or fsm.state != fsm.MIMIC):
        if args.start == 'hold':
          data.qpos[:7] = start_root
        elif fsm.state == fsm.FIXSTAND and args.stand_secs > 0:
          a = float(np.clip((t - fsm.t_enter) / args.stand_secs, 0.0, 1.0))
          data.qpos[:3] = (1 - a) * start_root[:3] + a * stand_root[:3]
          q = (1 - a) * start_root[3:7] + a * stand_root[3:7]
          data.qpos[3:7] = q / np.linalg.norm(q)
        else:
          data.qpos[:7] = start_root
        data.qvel[:6] = 0.0
    return not (args.duration and data.time >= args.duration)

  if args.headless:
    import time as _time
    while tick():
      if args.realtime:
        _time.sleep(fsm.step_dt)
  else:
    import time as _time
    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
      while viewer.is_running():
        wall = _time.perf_counter()
        if not tick():
          break
        viewer.sync()
        lag = fsm.step_dt - (_time.perf_counter() - wall)
        if lag > 0:
          _time.sleep(lag)

  print(f"\nended in {fsm.state} at t={data.time:.2f}s")
  if fsm.track_n:
    err = np.degrees(fsm.track_err / fsm.track_n)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(m)) for m in fsm.joint_ids_map]
    print(f"max torso tilt during the dance: {np.degrees(fsm.max_tilt):.1f} deg")
    print(f"mean |measured - commanded| over {fsm.track_n} policy ticks, worst 6 joints:")
    for i in np.argsort(err)[::-1][:6]:
      motor = int(fsm.joint_ids_map[i])
      flag = ("  <-- gains not reassigned by State_Mimic::enter"
              if motor >= len(fsm.stiffness) and not args.fix_gain_map else "")
      print(f"    motor {motor:2d} {names[i]:<22} {err[i]:6.2f} deg  (kp {fsm.mimic_kp[motor]:5.1f} kd {fsm.mimic_kd[motor]:4.1f}){flag}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
