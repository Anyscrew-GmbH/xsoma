"""Holosoma's G1 29-DoF velocity-tracking task (amazon-far/holosoma 79ba714, presets `g1_29dof` and `g1_29dof_fast_sac`)
as pure JAX functions of the robot's state, shared by the simulator (`sim.py`) and by imagination in the world model.

Every number below is Holosoma's; "spec N" cites section N of docs/holosoma_task.md, which cites Holosoma's source line by
line. Joint order is Holosoma's config order, which is also the MJCF's joint order, so `q = qpos[7:]`. The constants of the
task (PD gains, commands, rewards, penalty curriculum, randomization ranges, reset values) sit next to the functions that
use them.

A robot state is a `Robot`: pelvis position and rotation, the pelvis frame's world linear velocity (MuJoCo's free-joint
convention, spec 2.3), the body-frame angular velocity, joint positions and velocities, and the frames the task reads (foot
contact points and ankle-roll links). The simulator fills it from MuJoCo's kinematics, imagination from forward kinematics.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# ------------------------------------------------------------------ robot and control (spec 1.4-1.6)
# Another robot: the environment variable XSOMA_ROBOT_DIR (train.py --robot_dir) names a directory with robot.xml (the MJCF,
# scene.py) and robot.json, whose values replace the G1's wherever a constant below is wrapped in `_robot` (README, Another robot)
ROBOT_DIR = os.environ.get("XSOMA_ROBOT_DIR")
if ROBOT_DIR:
    with open(os.path.join(ROBOT_DIR, "robot.json")) as _f:
        _ROBOT = json.load(_f)


def _robot(key, g1):
    """The value of `key` in robot.json, in the type of the G1's value: a tuple of names, a float32 array or a scalar."""
    if not ROBOT_DIR:
        return g1
    v = _ROBOT[key]
    return tuple(v) if isinstance(g1, tuple) else np.asarray(v, np.float32) if isinstance(v, list) else v


DOF = _robot("dof", ("left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
       "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
       "waist_yaw", "waist_roll", "waist_pitch",
       "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
       "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch",
       "right_wrist_yaw"))  # fmt: skip
J = len(DOF)
LEG = np.asarray([-0.312, 0.0, 0.0, 0.669, -0.363, 0.0])
DEFAULT_Q = _robot("default_q", np.concatenate([LEG, LEG, np.zeros(3), [0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0], [0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0]]).astype(np.float32))

# (kp, kd) of each motor type, from its armature I: kp = I w^2, kd = 2 zeta I w, with w = 2 pi 10 rad/s and zeta = 2 (spec 1.5)
_HIP_P = (40.179238471, 2.557889765)
_HIP_R = (99.098427777, 6.308801854)
_ANKLE = (28.501246196, 1.814445687)
_ARM = (14.250623098, 0.907222843)
_WRIST = (16.778327481, 1.068141502)
# per leg joint: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll (the knee has the same gains as the hip roll)
_LEG_GAINS = [_HIP_P, _HIP_R, _HIP_P, _HIP_R, _ANKLE, _ANKLE]
_ARM_GAINS = [_ARM, _ARM, _ARM, _ARM, _ARM, _WRIST, _WRIST]
_GAINS = _LEG_GAINS * 2 + [_HIP_P, _ANKLE, _ANKLE] + _ARM_GAINS * 2  # waist yaw like hip_pitch, roll / pitch like ankles
KP = _robot("kp", np.asarray([g[0] for g in _GAINS], np.float32))
KD = _robot("kd", np.asarray([g[1] for g in _GAINS], np.float32))
# torque limits (N m) in Holosoma's config; the MJCF's motors clamp the hip roll to 88 on top of this (see sim.py)
EFFORT = _robot("effort", np.asarray([88, 139, 88, 139, 50, 50] * 2 + [88, 50, 50] + [25, 25, 25, 25, 25, 5, 5] * 2, np.float32))
# joint target = DEFAULT_Q + ACTION_SCALE * action (a robot file may give one scale per joint); the action is clipped first
ACTION_SCALE, ACTION_CLIP = _robot("action_scale", 0.25), 100.0
SIM_DT, DECIMATION = 0.005, 4  # physics step, and physics steps per policy step
DT = SIM_DT * DECIMATION  # 50 Hz policy
MAX_EPISODE = 1000  # time-out once the step counter exceeds it: an un-terminated episode has 1001 policy steps (spec 1.6)

# bodies of the compiled scene (prefix "robot_"): frames the task reads and the bodies whose ground contact ends an episode
FOOT_POINTS = _robot("foot_points", ("left_foot_contact_point", "right_foot_contact_point"))
FEET = _robot("feet", ("left_ankle_roll_link", "right_ankle_roll_link"))
# m: feet_cross penalizes the left foot being less than this to the left of the right one (the stance width itself follows from
# the robot's MJCF and default pose: 0.24 m for the G1)
FEET_CROSS_MARGIN = 0.10
TERMINATION_BODIES = _robot("termination_bodies", ("pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "right_hip_pitch_link",
                      "right_hip_roll_link", "right_hip_yaw_link", "left_shoulder_pitch_link", "left_shoulder_roll_link",
                      "left_shoulder_yaw_link", "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link"))  # fmt: skip
CONTACT_FORCE = 1.0  # N: a net contact force above this on a termination body ends the episode (spec 7)
# robot.json's optional `self_collision` (the G1 has none): two groups of body names whose collision geoms collide, group with
# group (scene.py); these contacts are simulated but do not count for the termination (sim.py)
SELF_COLLISION = _ROBOT.get("self_collision") if ROBOT_DIR else None
SELF_CLEARANCE_MARGIN = 0.02  # m: self_clearance penalizes every pair of the two groups' colliders that is closer than this
FOOT_CLEARANCE_HEIGHT = 0.06  # m: foot_clearance penalizes a foot that moves sideways or forward lower than this over the terrain
TOUCHDOWN_HEIGHT = 0.03  # m: touchdown_speed penalizes a foot that moves down lower than this over the terrain
TURN_STEP_HEIGHT = 0.02  # m: turn_stance counts a foot as standing whose lowest sole point is lower than this over the terrain
TURN_STEP_WIDTH = 0.01  # m: the width of turn_stance's sigmoid (its gradient; wide enough to reach a foot flat on the ground)
G = 9.81  # m/s^2


class Robot(NamedTuple):
    p: jax.Array  # (N, 3) pelvis position
    R: jax.Array  # (N, 3, 3) pelvis rotation (body to world)
    v: jax.Array  # (N, 3) world velocity of the pelvis frame origin
    w: jax.Array  # (N, 3) angular velocity in the pelvis frame
    q: jax.Array  # (N, J)
    qd: jax.Array  # (N, J)
    foot_p: jax.Array  # (N, 2, 3) world positions of the foot contact points (left, right)
    feet_p: jax.Array  # (N, 2, 3) world positions of the ankle-roll links (left, right)
    feet_R: jax.Array  # (N, 2, 3, 3) rotations of the ankle-roll links


def torques(q, qd, a_applied, kp_scale, kd_scale):
    """PD torques of one 5 ms physics substep (spec 1.5): kp_scale * KP * (target - q) - kd_scale * KD * qd, clipped to the
    effort limits. The joint target is q0 + 0.25 a (ACTION_SCALE), with q0 the default pose."""
    tau = kp_scale * KP * (ACTION_SCALE * a_applied + DEFAULT_Q - q) - kd_scale * KD * qd
    return jnp.clip(tau, -EFFORT, EFFORT)


# ------------------------------------------------------------------ commands and gait (spec 5)
CMD_RANGE = 1.0  # vx, vy, wz ~ U[-1, 1] (body frame)
STAND_PROB = 0.2  # probability that a drawn command is zero in all three components (the robot should stand)
RESAMPLE = int(10.0 / DT)  # commands change every 10 s = 500 steps
GAIT_FREQ = (0.8, 1.2)  # Hz: 1 / gait_period (a period of 1 s), plus or minus a width of 0.2
STAND_PHASE = math.pi  # the gait phase of both feet while standing


def sample_commands(key, n, scale=None):
    """Draw n commands [vx, vy, wz] and gait frequencies; every resample redraws both (spec 5). `scale`, which is not
    Holosoma's (train.py --cmd_scale): three factors on the ranges of vx, vy and wz; a fourth entry replaces STAND_PROB; a
    fifth is the probability with which each component of a command is set to zero on its own (commands along one or two
    axes, as an operator's stick gives them, are next to never drawn otherwise); a sixth and a seventh entry (s, f): a share s
    of the commands is scaled by f on top (slow commands, which are rare in the uniform draw otherwise)."""
    kc, ks, kf = jax.random.split(key, 3)
    cmd = jax.random.uniform(kc, (n, 3), minval=-CMD_RANGE, maxval=CMD_RANGE)
    stand = STAND_PROB
    if scale is not None:
        cmd = cmd * jnp.asarray(scale[:3], jnp.float32)
        stand = scale[3] if len(scale) > 3 else stand
        if len(scale) > 4:
            cmd = jnp.where(jax.random.uniform(jax.random.fold_in(kc, 1), (n, 3)) < scale[4], 0.0, cmd)
        if len(scale) > 5:
            cmd = jnp.where(jax.random.uniform(jax.random.fold_in(kc, 2), (n, 1)) < scale[5], scale[6] * cmd, cmd)
    cmd = jnp.where(jax.random.uniform(ks, (n, 1)) <= stand, 0.0, cmd)
    return cmd, jax.random.uniform(kf, (n,), minval=GAIT_FREQ[0], maxval=GAIT_FREQ[1])


def sample_phase_offsets(key, n):
    """Left offset U[-pi, pi], right offset the left one shifted by pi (spec 5)."""
    left = jax.random.uniform(key, (n,), minval=-math.pi, maxval=math.pi)
    return jnp.stack([left, jnp.fmod(left + 2 * math.pi, 2 * math.pi) - math.pi], -1)


def gait_phase(episode_step, freq, offsets, cmd):
    """Gait phase phi of each foot from the episode counter t: fmod(t * 2 pi dt f + offset + pi, 2 pi) - pi. Both feet are held
    at pi (standing) when the command is zero."""
    phase_dt = 2 * math.pi * DT * freq
    phi = jnp.fmod(episode_step[:, None] * phase_dt[:, None] + offsets + math.pi, 2 * math.pi) - math.pi
    stand = (jnp.linalg.norm(cmd[:, :2], axis=-1) < 0.01) & (jnp.abs(cmd[:, 2]) < 0.01)
    return jnp.where(stand[:, None], STAND_PHASE, phi)


def swing_height(phi, height: float = 0.09):
    """Desired foot height: cubic Bezier rise to 0.09 m at phi = 0 and back to 0 at +-pi (spec 6)."""
    x = (phi + math.pi) / (2 * math.pi)

    def bezier(t):
        return t**3 + 3 * t**2 * (1 - t)

    return jnp.where(x <= 0.5, height * bezier(2 * x), height - height * bezier(2 * x - 1))


# ------------------------------------------------------------------ observations (spec 3)
GRAVITY = np.asarray([0.0, 0.0, -1.0], np.float32)
ACTOR_DIM, CRITIC_DIM = 3 * J + 13, 3 * J + 16  # 100 and 103 for the G1's 29 joints
# the terms of the actor observation in order, with their sizes (Holosoma sorts its observation terms alphabetically)
ACTOR_LAYOUT = (("actions", J), ("base_ang_vel", 3), ("command_ang_vel", 1), ("command_lin_vel", 2), ("cos_phase", 2), ("dof_pos", J),
                ("dof_vel", J), ("projected_gravity", 3), ("sin_phase", 2))  # fmt: skip
OBS_CLIP = 100.0


def projected_gravity(R):
    """The direction of gravity in the pelvis frame: R^T (0, 0, -1)."""
    return jnp.einsum("...ji,j->...i", R, GRAVITY)


def observations(robot: Robot, a_last, cmd, phase, key=None):
    """(actor, critic) observations. Terms in alphabetical order; uniform noise (actor only, raw units) before scaling.

    actor: actions, base_ang_vel x0.25, command_ang_vel, command_lin_vel, cos_phase, dof_pos (+-0.01), dof_vel x0.05 (+-0.1),
    projected_gravity, sin_phase; critic: the same without noise plus base_lin_vel x2.0 after base_ang_vel.
    """
    dq = robot.q - DEFAULT_Q
    qd = robot.qd
    if key is not None:
        k1, k2 = jax.random.split(key)
        dq_n = dq + jax.random.uniform(k1, dq.shape, minval=-0.01, maxval=0.01)
        qd_n = qd + jax.random.uniform(k2, qd.shape, minval=-0.1, maxval=0.1)
    else:
        dq_n, qd_n = dq, qd
    g = projected_gravity(robot.R)
    v_b = jnp.einsum("nji,nj->ni", robot.R, robot.v)
    common = lambda dq_, qd_: (jnp.cos(phase), dq_, 0.05 * qd_, g, jnp.sin(phase))  # noqa: E731
    head = (a_last, 0.25 * robot.w)
    cmds = (cmd[:, 2:3], cmd[:, :2])
    actor = jnp.concatenate(head + cmds + common(dq_n, qd_n), -1)
    critic = jnp.concatenate(head + (2.0 * v_b,) + cmds + common(dq, qd), -1)
    return jnp.clip(actor, -OBS_CLIP, OBS_CLIP), jnp.clip(critic, -OBS_CLIP, OBS_CLIP)


# ------------------------------------------------------------------ rewards (spec 6)
# weight of each joint's squared offset from the default pose, in DOF order: the six leg joints twice, then 3 waist, 14 arm joints
POSE_W = _robot("pose_w", np.asarray([0.01, 1.0, 5.0, 0.01, 5.0, 5.0] * 2 + [50.0] * 3 + [50.0] * 14, np.float32))
# the terms whose weights the penalty curriculum scales
PENALTIES = ("ang_vel_xy", "orientation", "action_rate", "pose", "close_feet_xy", "feet_ori")


@dataclasses.dataclass(frozen=True)
class RewardCfg:
    """Weights of r = sum_i w_i dt r_i (no clipping); the penalty terms' weights are scaled by the penalty curriculum."""

    tracking_lin_vel: float = 2.0
    tracking_ang_vel: float = 1.5
    ang_vel_xy: float = -1.0
    orientation: float = -10.0
    action_rate: float = -2.0
    feet_phase: float = 5.0
    pose: float = -0.5
    close_feet_xy: float = -10.0
    feet_ori: float = -5.0
    alive: float = 1.0


PPO_REWARD = RewardCfg()
FAST_SAC_REWARD = RewardCfg(alive=10.0)  # the presets differ only in the alive weight


def feet_dy(robot: Robot):
    """How far the left foot is to the left of the right one, in the pelvis heading's frame (m): the stance width, negative when
    the feet are crossed. Left and right are the order of FEET."""
    yaw = jnp.arctan2(robot.R[:, 1, 0], robot.R[:, 0, 0])
    dx = robot.feet_p[:, 0, 0] - robot.feet_p[:, 1, 0]
    dy = robot.feet_p[:, 0, 1] - robot.feet_p[:, 1, 1]
    return jnp.cos(yaw) * dy - jnp.sin(yaw) * dx


def feet_cross(robot: Robot):
    """The feet-crossing penalty term, which is not Holosoma's (train.py --feet_cross): max(0, FEET_CROSS_MARGIN - feet_dy)."""
    return jnp.maximum(0.0, FEET_CROSS_MARGIN - feet_dy(robot))


def self_clearance(distance):
    """The self-clearance penalty term, which is not Holosoma's (train.py --self_clearance): over the pairs of `self_collision`
    colliders, with their signed distances (sim.SelfDistance), the sum of max(0, SELF_CLEARANCE_MARGIN - distance)."""
    return jnp.maximum(0.0, SELF_CLEARANCE_MARGIN - distance).sum(-1)


def foot_clearance(sole_h, step_xy, height=FOOT_CLEARANCE_HEIGHT):
    """The swing-clearance penalty term, which is not Holosoma's and needs no gait clock (train.py --foot_clearance): per foot,
    how far its lowest sole point is below `height` over the terrain (`sole_h`, (N, 2)) times its horizontal speed, from the
    displacement `step_xy` (N, 2, 2) of its contact point in one policy step; summed over the feet. Zero for a foot that
    stands still and for one lifted above `height`, however it moves."""
    speed = jnp.sqrt(jnp.sum(jnp.square(step_xy), -1) + 1e-12) / DT  # the 1e-12 keeps the gradient finite at rest
    return jnp.sum(jnp.maximum(0.0, height - sole_h) * speed, -1)


def stand_still(robot: Robot, feet_step, cmd):
    """The standing penalty term, which is not Holosoma's (train.py --stand_still): under a zero command, the base's planar speed
    (m/s) plus its angular speed (rad/s) plus the speed of each foot contact point, from its displacement `feet_step` (N, 2, 3)
    in one policy step (m/s); zero under any other command. Norms, not squares: a slow drift is pulled to zero as hard as a step."""
    norm = lambda x: jnp.sqrt(jnp.sum(jnp.square(x), -1) + 1e-12)  # noqa: E731  (the 1e-12 keeps the gradient finite at rest)
    return jnp.where((jnp.abs(cmd) < 0.01).all(-1), norm(robot.v[:, :2]) + norm(robot.w) + norm(feet_step).sum(-1) / DT, 0.0)


def track_l1(robot: Robot, cmd, rel=0.0):
    """The plain tracking error, which is not Holosoma's (train.py --track_l1): the distance between the commanded and the
    base's planar velocity (m/s, base frame) plus that of the yaw rates (rad/s). The tracking rewards' exp(-error^2 / 0.25)
    is flat near zero: standing still under a command of 0.2 m/s loses 15 % of it. `rel` > 0 (train.py --track_rel): the
    error is divided by the command's size, the norm of its three components taken as at least `rel`, so that standing still
    costs at least 1 under every command of that size or more; a command of size 1 and a zero command keep the plain error."""
    v_b = jnp.einsum("nji,nj->ni", robot.R, robot.v)
    size = lambda x: jnp.sqrt(jnp.sum(jnp.square(x), -1) + 1e-12)  # noqa: E731
    err = size(cmd[:, :2] - v_b[:, :2]) + jnp.abs(cmd[:, 2] - robot.w[:, 2])
    return err / jnp.where(size(cmd) < 0.01, 1.0, jnp.maximum(size(cmd), rel)) if rel else err


def torque_limit(q, qd, target, soft=0.9):
    """The torque-limit penalty term, which is not Holosoma's (train.py --torque_limit): per joint (N, J), how far the PD law's
    torque with the nominal gains, KP (target - q) - KD qd before the clip, exceeds `soft` x EFFORT, in units of EFFORT:
    max(0, |tau| / EFFORT - soft). The simulator clips the torque, so beyond EFFORT a larger request changes nothing there;
    this term is what tells the policy. Linear, not squared: every step near the limit is pulled down as hard as a rare spike."""
    return jnp.maximum(0.0, jnp.abs(KP * (target - q) - KD * qd) / EFFORT - soft)


def joint_range(q, lo, hi):
    """The joint-range penalty term, which is not Holosoma's (train.py --range_penalty): per joint (N, n), how far q lies outside
    [lo, hi] (rad), linear. A range inside the model's own lets a policy learn to stay off limits that something else than the
    model enforces, without a hard stop in the simulation that it could learn to lean on."""
    return jnp.maximum(0.0, q - hi) + jnp.maximum(0.0, lo - q)


def joint_vel(qd):
    """The joint-speed penalty term, which is not Holosoma's (train.py --joint_vel): the sum over the joints of the squared joint
    speeds (rad/s)^2, (N, J) -> (N,)."""
    return jnp.sum(jnp.square(qd), -1)


def touchdown_speed(sole_h, feet_dz, height=TOUCHDOWN_HEIGHT):
    """The touchdown-speed penalty term, which is not Holosoma's (train.py --touchdown_speed): per foot whose lowest sole point
    ends the step less than `height` over the terrain (`sole_h`, (N, 2)), the square of its downward speed (m/s), from the
    vertical displacement `feet_dz` (N, 2) of its contact point in one policy step; summed over the feet. Zero for a foot that
    rises, stands or is higher: a foot that lands is charged for the speed it closes the last centimeters with."""
    return jnp.sum(jnp.where(sole_h < height, jnp.square(jnp.maximum(0.0, -feet_dz / DT)), 0.0), -1)


def turn_stance(st, sole_h):
    """Seconds each foot has stood (N, 2), one policy step on (train.py --turn_step): a foot whose lowest sole point `sole_h` is
    under TURN_STEP_HEIGHT over the terrain stands (DT added), a higher one is reset to zero. The value counts hard; its gradient
    is that of a sigmoid of width TURN_STEP_WIDTH (straight through), so lifting a standing foot lowers it."""
    soft = jax.nn.sigmoid((TURN_STEP_HEIGHT - sole_h) / TURN_STEP_WIDTH)
    return ((sole_h < TURN_STEP_HEIGHT) + soft - jax.lax.stop_gradient(soft)) * (st + DT)


def turn_step(st, wz, stance):
    """The step-rate penalty term, which is not Holosoma's (train.py --turn_step): |yaw-rate command| `wz` (N,) x the seconds by
    which each foot's stance `st` (N, 2; turn_stance) exceeds `stance`, summed over the feet. Turning asks each foot to lift at
    least every `stance` seconds; straight walking and standing (wz = 0) are not charged."""
    return jnp.abs(wz) * jnp.maximum(0.0, st - stance).sum(-1)


def walk_step(st, cmd, stance):
    """The cadence penalty term, which is not Holosoma's (train.py --walk_step): under any nonzero planar or yaw command, the
    seconds by which each foot's stance `st` (N, 2; turn_stance) exceeds `stance`, summed over the feet, whatever the command's
    size: slow commands are walked in steps (a foot lifts at least every `stance` seconds), not by leaning. Standing is free."""
    return jnp.any(cmd != 0, -1) * jnp.maximum(0.0, st - stance).sum(-1)


def balance(robot: Robot, margin):
    """The balance penalty term, which is not Holosoma's (train.py --balance): the squared horizontal distance (m) beyond
    `margin` from the pelvis (a proxy of the center of mass) to the segment between the two feet's contact points, whether they
    stand or swing: leaning ahead of (or behind, beside) both feet costs, and stepping a foot under the pelvis is what removes
    it. (A single-stance support point would charge every step: the pelvis sits half a stance width beside the stance foot.)"""
    p, a, b = robot.p[:, :2], robot.foot_p[:, 0, :2], robot.foot_p[:, 1, :2]
    ab = b - a
    t = jnp.clip(jnp.sum((p - a) * ab, -1) / (jnp.sum(ab * ab, -1) + 1e-6), 0.0, 1.0)
    d = jnp.sqrt(jnp.sum(jnp.square(p - a - t[:, None] * ab), -1) + 1e-12)
    return jnp.square(jnp.maximum(0.0, d - margin))


def step_length(robot: Robot, cmd, base, per_speed, width):
    """The step-length penalty term, which is not Holosoma's (train.py --step_length): in the pelvis heading's frame, the squared
    excess of the feet's fore-aft spread over base + per_speed x |vx command| and of their lateral spread over width + base +
    per_speed x |vy command| (m; per_speed in s): small steps for slow commands, longer ones as the commanded speed rises."""
    yaw = jnp.arctan2(robot.R[:, 1, 0], robot.R[:, 0, 0])
    d = robot.feet_p[:, 0, :2] - robot.feet_p[:, 1, :2]
    fx = jnp.cos(yaw) * d[:, 0] + jnp.sin(yaw) * d[:, 1]
    fy = jnp.cos(yaw) * d[:, 1] - jnp.sin(yaw) * d[:, 0]
    ex = jnp.maximum(0.0, jnp.abs(fx) - base - per_speed * jnp.abs(cmd[:, 0]))
    ey = jnp.maximum(0.0, jnp.abs(fy) - width - base - per_speed * jnp.abs(cmd[:, 1]))
    return jnp.square(ex) + jnp.square(ey)


def impact_accel(v0, v1):
    """The impact penalty term, which is not Holosoma's (train.py --impact_accel): the square of the base's upward acceleration
    over one policy step in units of g, from its world velocities `v0`, `v1` (N, 3) at the step's start and end; zero while it
    accelerates down. A proxy of the ground force above the robot's weight (net vertical force = mass x (a_z + g) of the center
    of mass), for a model that predicts no contact forces; as an IMU on the base measures it."""
    return jnp.square(jnp.maximum(0.0, (v1[:, 2] - v0[:, 2]) / (DT * G)))


EXT_SPEED = 2.0  # rad/s: a joint that something else drives moves to each goal at a speed of U[0, EXT_SPEED]
EXT_PERIOD = 2.0  # s: the mean time between two goals


def ext_step(key, x, goal, speed, lo, hi):
    """One policy step of the targets (rad) of joints that something else than the policy drives (train.py --ext_joints):
    (x, goal, speed), each (N, n). Every joint moves towards its goal at its speed; with probability DT / EXT_PERIOD an env
    draws new goals in [lo, hi] and new speeds for all of its joints."""
    # ponytail: constant speed, no acceleration limit; smooth the ramps if the drives should not see velocity steps
    k_n, k_g, k_s = jax.random.split(key, 3)
    new = jax.random.uniform(k_n, x.shape[:-1] + (1,)) < DT / EXT_PERIOD
    goal = jnp.where(new, jax.random.uniform(k_g, x.shape, minval=lo, maxval=hi), goal)
    speed = jnp.where(new, EXT_SPEED * jax.random.uniform(k_s, x.shape), speed)
    return x + jnp.clip(goal - x, -speed * DT, speed * DT), goal, speed


def reward_terms(robot: Robot, foot_heights, a, a_prev, cmd, phase):
    """Unweighted terms of one step. foot_heights (N, 2): foot contact points over the terrain; phase / cmd: the ones the
    policy observed when it chose `a` (spec 6, timing)."""
    v_b = jnp.einsum("nji,nj->ni", robot.R, robot.v)  # velocity in the pelvis frame
    g = projected_gravity(robot.R)
    lateral = jnp.abs(feet_dy(robot))  # the feet's sideways distance, in the pelvis heading's frame
    g_feet = jnp.einsum("nfji,j->nfi", robot.feet_R, GRAVITY)  # gravity in each foot's frame (zero x and y when flat)
    err_h = jnp.sum(jnp.square(foot_heights - swing_height(phase)), -1)  # squared foot height error against the gait reference
    return {
        "tracking_lin_vel": jnp.exp(-jnp.sum(jnp.square(cmd[:, :2] - v_b[:, :2]), -1) / 0.25),
        "tracking_ang_vel": jnp.exp(-jnp.square(cmd[:, 2] - robot.w[:, 2]) / 0.25),
        "ang_vel_xy": jnp.sum(jnp.square(robot.w[:, :2]), -1),
        "orientation": jnp.sum(jnp.square(g[:, :2]), -1),
        "action_rate": jnp.sum(jnp.square(a_prev - a), -1),
        "feet_phase": jnp.exp(-err_h / 0.008),
        "pose": jnp.sum(POSE_W * jnp.square(robot.q - DEFAULT_Q), -1),
        "close_feet_xy": (lateral < 0.15).astype(jnp.float32),
        "feet_ori": jnp.sum(jnp.linalg.norm(g_feet[..., :2], axis=-1), -1),
        "alive": jnp.ones(robot.q.shape[0]),
    }


TERMS = tuple(f.name for f in dataclasses.fields(RewardCfg))  # the reward terms, in the order of every per-term array


def weighted(terms, cfg: RewardCfg, penalty_scale):
    """The summands w_i dt r_i of the reward (Holosoma logs their episode sums as Episode/rew_<term>)."""
    w = dataclasses.asdict(cfg)
    return {k: DT * w[k] * (penalty_scale if k in PENALTIES else 1.0) * terms[k] for k in TERMS}


def reward(terms, cfg: RewardCfg, penalty_scale):
    """The reward of one step: the sum of the weighted terms."""
    return sum(weighted(terms, cfg, penalty_scale).values())


# ------------------------------------------------------------------ curriculum (spec 10)
@dataclasses.dataclass(frozen=True)
class CurriculumCfg:
    """Penalty scale s, shared by all envs, updated once per policy step from the average episode length: s *= 1 - degree
    below 150 steps, s *= 1 + degree above 750, clipped to [min, max]. The average is an EMA over finished episodes with
    weight min(n / K, 1), K = 1000 * num_envs / 4096."""

    initial: float = 0.1
    min: float = 0.0
    max: float = 1.0
    degree: float = 0.00025
    down: float = 150.0
    up: float = 750.0


PPO_CURRICULUM = CurriculumCfg()
FAST_SAC_CURRICULUM = CurriculumCfg(initial=0.5, min=0.5, degree=0.001)


def curriculum_step(cfg: CurriculumCfg, scale, avg_len, finished_lengths, finished, num_envs: int):
    """Average-episode tracker on this step's finished episodes, then the per-step penalty-scale update."""
    k = max(1, int(1000 * num_envs / 4096))
    n = finished.sum()
    w = jnp.minimum(n / k, 1.0)
    mean = jnp.sum(jnp.where(finished, finished_lengths, 0.0)) / jnp.maximum(n, 1)
    avg_len = jnp.where(n > 0, avg_len * (1 - w) + mean * w, avg_len)
    scale = jnp.where(avg_len < cfg.down, scale * (1 - cfg.degree), jnp.where(avg_len > cfg.up, scale * (1 + cfg.degree), scale))
    return jnp.clip(scale, cfg.min, cfg.max), avg_len


# ------------------------------------------------------------------ domain randomization (spec 9)
# "setup only": drawn once per env when the simulator is built; the others are redrawn at every reset or on a schedule
MASS_BODIES = _robot("mass_bodies", ("pelvis", "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link", "left_knee_link",
                                    "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link", "right_knee_link"))  # fmt: skip
MASS_SCALE = (0.9, 1.2)  # the masses of MASS_BODIES times U, inertia scaled by the same ratio (setup only)
TORSO = _robot("torso", "torso_link")  # the body of the next two
TORSO_ADDED_MASS = (-1.0, 3.0)  # kg on torso_link (setup only)
TORSO_COM = 0.05  # +-m per axis on torso_link (setup only)
FRICTION = (0.5, 1.25)  # range of the friction coefficient; each env gets one of 64 equally spaced values (setup only)
FRICTION_BUCKETS = 64
GAIN_SCALE = (0.9, 1.1)  # kp and kd per (env, joint), every reset
MAX_DELAY = 1  # action delay in policy steps, U{0, 1} per env at every reset
PUSH_INTERVAL = (5.0, 10.0)  # s between pushes, per env; a push overwrites the root's world (vx, vy) with U[-1, 1] each
PUSH_VEL = 1.0

# A rope that holds the robot up by its torso at the start of an episode (sim.SimCfg.support, train.py --support; not Holosoma's):
# it pulls straight up on the torso body (at its center of mass, or at sim.SimCfg.support_point), taut while the pelvis (or that
# point) is below the rope's top; after a hold it is paid out, then taken off
SUPPORT_K, SUPPORT_C = 5e4, 1e3  # the rope's stiffness (N/m) and damping (N s/m)
SUPPORT_HOLD = (0.0, 8.0)  # s, U per episode: the hold before the pay-out
SUPPORT_SPEED = 0.05  # m/s: the pay-out
SUPPORT_SLACK = 0.05  # m: the rope is taken off once its top is this far below the standing height
SUPPORT_OFF = -1.0  # the top (relative to the standing height) of an env without a rope


def support_force(top, h, vz):
    """The rope's tension (N), never negative: K (top - h) - C vz while the pelvis height h is below the top (both in m, in one
    frame), zero above; vz is the pelvis' vertical velocity (m/s)."""
    return jnp.where(h < top, jnp.maximum(0.0, SUPPORT_K * (top - h) - SUPPORT_C * vz), 0.0)


def support_pay_out(top, rate):
    """One policy step of the rope: its top (m, relative to the standing height) lowered by rate (m/s) x DT, and once it is
    SUPPORT_SLACK below the standing height the rope taken off (top SUPPORT_OFF, rate 0). Returns (top, rate)."""
    top = top - rate * DT
    off = top < -SUPPORT_SLACK
    return jnp.where(off, SUPPORT_OFF, top), jnp.where(off, 0.0, rate)


def _payload(entries):
    """robot.json's optional `payload` (the G1 has none) -> the bodies' names and, per body, the ranges (lo, hi) of its mass (kg)
    and CoM (m, body frame) as (n, 4) arrays. The second body of an entry is the first one's mirror image: its y range is mirrored."""
    names, lo, hi = [], [], []
    for e in entries:
        a, b = [e["mass"][0], *e["com_lo"]], [e["mass"][1], *e["com_hi"]]
        assert len(e["bodies"]) in (1, 2) and len(a) == len(b) == 4, f"payload {e}: one body or a mirror pair, CoM as [x, y, z]"
        assert 0 < a[0] and all(x <= y for x, y in zip(a, b)), f"payload {e['bodies']}: ranges must be lo <= hi, mass > 0"
        names += e["bodies"]
        lo += [a, [a[0], a[1], -b[2], a[3]]][: len(e["bodies"])]
        hi += [b, [b[0], b[1], -a[2], b[3]]][: len(e["bodies"])]
    return tuple(names), np.asarray(lo, np.float32).reshape(-1, 4), np.asarray(hi, np.float32).reshape(-1, 4)


# welded bodies of the MJCF (no joint, no collider) whose mass and CoM are drawn per env, inertia by the mass ratio (setup only)
PAYLOAD_BODIES, PAYLOAD_LO, PAYLOAD_HI = _payload(_ROBOT.get("payload", []) if ROBOT_DIR else [])

# ------------------------------------------------------------------ resets (spec 8)
SPAWN_HEIGHT = _robot("spawn_height", 0.8)  # pelvis height above the tile origin (the terrain at the spawn point is not looked up)
SPAWN_XY = 1.0  # U[-1, 1] m offset from the tile origin
SPAWN_VEL = 0.5  # all six root velocities U[-0.5, 0.5] (world frame)
SPAWN_Q = (0.5, 1.5)  # q = q0 * U, joints with q0 = 0 start at 0; qd = 0
