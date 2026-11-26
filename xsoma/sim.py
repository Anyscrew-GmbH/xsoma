"""Holosoma's G1 velocity task (task.py) on MuJoCo Warp through MJX: N worlds, one jitted function per policy step.

This is the simulator of training (collection) and of evaluation. Imagination does not run it, but shares its `view_state`
(forward kinematics and terrain heights), so both read the task's quantities from a state in the same way.

A step, in the order of Holosoma's BaseTask.step (docs/holosoma_task.md, spec 0):
  1. the raw action is stored; clip(a, +-100), delayed by the env's 0 or 1 policy steps, drives the PD law
  2. 4 physics substeps of 5 ms, the PD torques recomputed from the current q, qd before each; MuJoCo clamps them to the
     motors' ctrlrange (hip roll at 88 N m, as on Holosoma's MuJoCo backends; the task's own clip is 139)
  3. episode counter + 1; terminations: a net contact force above 1 N on a pelvis, hip or shoulder link after the last
     substep, or a counter past 1000 (time-out); rewards of the final state, with the command and gait phase the policy saw
     and the current penalty scale
  4. finished envs reset (pose, velocities, PD gain scales, delay, command, gait offsets, push schedule); the average episode
     length and the penalty scale update (every step, one scale for all envs)
  5. the gait phase from the episode counter and the current command, then commands resample every 500 steps (on a reset
     step twice: the first draw only decides that step's stand phase)
  6. pushes overwrite the root's world xy velocity; observations of the new state (actor with noise, critic without)
The start-up randomization (link masses and inertias, torso mass and CoM, friction) lives in per-world model fields.

Deviations from Holosoma on its MuJoCo Warp backend:
  - The terrain is one heightfield per tile, because MuJoCo Warp's collision with Holosoma's single heightfield is numerically
    unstable far from its center (scene.py). It is not stretched by n / (n - 1), so physics and foot heights share one surface
    (terrain.py).
  - The explicit sole-floor pairs become equivalent priority contacts (scene.py).
  - `sole_friction` (default on) randomizes the soles' friction too. Holosoma's MuJoCo backends leave it at 0.8, because their
    randomization writes geom_friction, which explicit pairs ignore.
  - A time-out on a step that also ends in contact counts as a termination.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import warp as wp
from mujoco import mjx

from . import scene, task
from . import terrain as terrain_mod
from .kin import Kinematics
from .rot import mv3, quat_to_mat, skew, tr3


@dataclasses.dataclass(frozen=True)
class SimCfg:
    num_envs: int = 4096
    terrain: str = "mix"  # mix: Holosoma's terrain_locomotion_mix (heightfield) | plane
    terrain_scale: float = 1.0  # Holosoma's terrain scale_factor (its MuJoCo Warp PPO runs need 0.5)
    foot: str = "capsules"  # capsules: Holosoma's MJCF soles | box: RL-X's box feet
    preset: str = "ppo"  # rewards and penalty curriculum: ppo (g1_29dof) | fast_sac (g1_29dof_fast_sac)
    sole_friction: bool = True  # friction randomization reaches the soles (Holosoma's MuJoCo backends: fixed 0.8)
    random_start: bool = False  # PPO's init_at_random_ep_len: initial episode counters uniform in [0, 1000)
    max_episode: int = task.MAX_EPISODE  # time-out once the episode counter exceeds it (evaluate.py --steps runs longer episodes)
    cmd_scale: tuple | None = None  # three factors on the command ranges of vx, vy and wz (task.sample_commands); None: Holosoma's
    push_vel: float = task.PUSH_VEL  # a push sets the root's world (vx, vy) to U[-push_vel, push_vel] each; 0: no pushes
    support: float = 0.0  # share of the episodes that start on a rope (task.support_force; not Holosoma's), see Sim._support; 0: none
    support_gap: tuple = (-0.02, 0.05)  # m, U per such episode: the lowest sole point over the terrain while the rope holds the
    #                                     nominal robot in the default pose; negative: on the terrain, the rope carrying part of the
    #                                     weight (from minus the rope's stretch under the weight on, none: slack)
    support_point: tuple | None = None  # m, the rope's attachment in the frame of the rope's body (task.TORSO), the rope vertical
    #                                     above it (an overhead crane), see Sim._rope; None: at that body's center of mass, the rope's
    #                                     length taken at the root
    nconmax: int = 96  # contacts per world (the size of Holosoma's MuJoCo Warp contact pool)
    njmax: int = 576  # constraint rows per world (Holosoma: max(6 x 96, 4 nv))
    naccdmax: int = 0  # heightfield (CCD) contacts per world; 0: nconmax
    iterations: int = 100  # Newton iterations and line-search iterations: MuJoCo's defaults, which Holosoma keeps
    ls_iterations: int = 50
    ccd_tolerance: float = 1e-6  # GJK / EPA tolerance of the heightfield collisions (MuJoCo's default)
    tiles: bool = True  # one heightfield per terrain tile (scene.py: a single 160 x 240 m field is numerically unstable)
    seed: int = 42  # terrain layout and env origins; the start-up randomization uses the key passed to init
    friction_surfaces: tuple = ()  # ((lo, hi), ...): each episode draws a surface uniformly, then the friction coefficient
    #                                uniformly in its range (Sim._mu); (): task.FRICTION's buckets once per env at start-up
    start_share: float = 0.0  # share of the fresh episodes that start in the pose of start_joints (Sim._fresh; episodes on the rope keep its pose)
    start_joints: tuple | None = None  # ((joint index, lo, hi), ...): those joints start at their default position + U[lo, hi]
    start_cmd: tuple | None = None  # ((lo, hi) of vx, vy, wz): those episodes' command; None: drawn as any other


def surfaces(spec):
    """--friction_surfaces (train.py, evaluate.py) "lo:hi,..." as ((lo, hi), ...)."""
    out = tuple(tuple(float(x) for x in e.split(":")) for e in spec.split(",") if e)
    assert all(len(e) == 2 and 0 <= e[0] <= e[1] for e in out), f"--friction_surfaces {spec!r}: lo:hi,... with 0 <= lo <= hi"
    return out


wp.config.log_level = wp.LOG_WARNING  # no line per loaded MuJoCo Warp kernel

WORLD_FIELDS = ("body_mass", "body_inertia", "body_ipos", "geom_friction")  # the model fields that differ between envs
# |qvel| beyond this (m/s, rad/s) counts as a physics failure; under hard random actions, 99.99 % of the speeds stay below 200
# (tools/stress_sim.py)
DIVERGED = 1e3


class World(NamedTuple):
    """Constants of the run. Per env: the tile origin and the start-up randomization (the model fields that differ between
    envs). Shared by all envs: the terrain heights and the MuJoCo Warp model. A World is passed into the compiled programs as an
    argument, not baked in as constants, so every seed's terrain runs the same programs (one compilation, cached across runs)
    and the programs stay small."""

    origin: jax.Array  # (N, 3)
    body_mass: jax.Array  # (N, nbody)
    body_inertia: jax.Array  # (N, nbody, 3)
    body_ipos: jax.Array  # (N, nbody, 3)
    geom_friction: jax.Array  # (N, ngeom, 3)
    heights: jax.Array  # (nx, ny) terrain heights in meters, shared ((1, 1) zeros for the plane)
    model: mjx.Model  # the MuJoCo Warp model, shared (its per-world fields come from the fields above)


class State(NamedTuple):
    """The state of all envs: the physics data, the World, and the task's own state (last action, PD gain scales, action delay,
    command, gait, push schedule, penalty curriculum). A pytree that is carried through the compiled programs."""

    data: mjx.Data
    world: World
    obs: jax.Array  # (N, 100) actor observation (with noise)
    critic: jax.Array  # (N, 103) critic observation
    a_last: jax.Array  # (N, J) last raw action, zero after a reset
    kp: jax.Array  # (N, J) PD gain scales of the episode
    kd: jax.Array  # (N, J) the same for the damping gains
    delay: jax.Array  # (N,) action delay in policy steps, 0 or 1
    cmd: jax.Array  # (N, 3) vx, vy, wz command (body frame)
    freq: jax.Array  # (N,) gait frequency
    offset: jax.Array  # (N, 2) gait phase offsets
    phase: jax.Array  # (N, 2) gait phase in the current observation
    t: jax.Array  # (N,) episode counter
    push_t: jax.Array  # (N,) steps since the last push or reset
    push_at: jax.Array  # (N,) push interval in steps
    scale: jax.Array  # () penalty curriculum scale
    avg_len: jax.Array  # () average episode length of the curriculum
    ep_sums: jax.Array  # (N, len(task.TERMS)) running episode sums of the weighted reward terms
    key: jax.Array  # random key, split anew in every step
    sup: jax.Array = None  # (N,) with SimCfg.support: the rope's top relative to the standing height (m), task.SUPPORT_OFF: no rope
    sup_wait: jax.Array = None  # (N,) with SimCfg.support: steps until the rope is paid out


class Transition(NamedTuple):
    """What one step records: the states before (qpos, qvel) and after (qpos1, qvel1) the physics, the PD target, the reward,
    the end-of-episode flags and statistics. The world model trains on these transitions and imagination starts from them."""

    qpos: jax.Array  # (N, nq) state the action was taken in (after the previous step's resets and pushes)
    qvel: jax.Array
    target: jax.Array  # (N, J) the PD law's action input (clipped, delayed)
    qpos1: jax.Array  # (N, nq) state after the physics substeps, before resets
    qvel1: jax.Array
    reward: jax.Array  # (N,)
    terminated: jax.Array  # (N,) contact on a termination body
    truncated: jax.Array  # (N,) time-out without contact
    diverged: jax.Array  # (N,) physics failure (non-finite or impossible velocities): terminated, not a valid transition
    parts: jax.Array  # (N, len(task.TERMS)) the reward's weighted terms
    critic_final: jax.Array  # (N, 103) critic observation of the final state (for time-out bootstraps; XSoma has no critic)
    clearance: jax.Array  # (N, n) terrain clearance of the termination colliders (imagination's termination geometry)
    stats: dict
    force: jax.Array = None  # (N, n) net contact force (N) on each termination collider: which body ended the episode


class View(NamedTuple):
    """What the task reads from a state: its robot frames, and the heights over the terrain that rewards and terminations use."""

    robot: task.Robot
    foot_h: jax.Array  # (..., 2) foot contact points over the terrain (Holosoma's feet heights)
    clearance: jax.Array  # (..., n) termination colliders over the terrain (radius included); negative when penetrating
    soles: jax.Array  # (..., k) sole collider points over the terrain (radius included)
    height: jax.Array  # (...,) pelvis over the terrain


def unflatten(qpos, qvel):
    """MuJoCo's (qpos, qvel) of the free-root robot -> the state tuple S = (p, R, v, w, q, qd) that the world model and the views
    use: pelvis position and rotation, linear velocity of the pelvis origin and angular velocity (both in the world frame), joint
    positions and velocities."""
    R = quat_to_mat(qpos[..., 3:7])
    return qpos[..., :3], R, qvel[..., :3], mv3(R, qvel[..., 3:6]), qpos[..., 7:], qvel[..., 6:]


def _points(m: mujoco.MjModel, geoms, frames, bottom: bool):
    """Reduce geoms to points with a radius, so that a point's height over the terrain minus its radius tells how far the geom
    is above the terrain there: a sphere is its center, a capsule its two end centers, a box its corners (radius 0): the four
    bottom ones with `bottom` (a sole), else all eight (a falling body may land on any of them).
    Returns the points in their body frames, the index of each point's body in `frames`, and the radii."""
    local, frame, radius = [], [], []
    for g in geoms:
        R = np.asarray(quat_to_mat(np.asarray(m.geom_quat[g], np.float32)))
        kind, size, pos = int(m.geom_type[g]), m.geom_size[g], m.geom_pos[g]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            pts, r = [pos], size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            pts, r = [pos + s * size[1] * R[:, 2] for s in (1, -1)], size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            pts, r = [pos + R @ (np.asarray([sx, sy, sz]) * size) for sz in ((-1,) if bottom else (-1, 1)) for sx in (1, -1) for sy in (1, -1)], 0.0
        else:
            raise ValueError(f"collider geom {m.geom(g).name} has unsupported type {kind}")
        local += pts
        frame += [frames.index(int(m.geom_bodyid[g]))] * len(pts)
        radius += [float(r)] * len(pts)
    return np.asarray(local, np.float32), np.asarray(frame), np.asarray(radius, np.float32)


def _segment_distance(a0, a1, b0, b1):
    """Distance between the segments a0-a1 and b0-b1 (..., 3). The gradient holds the closest points constant: they minimize
    the distance, so it is the exact one, and it stays finite for parallel segments."""
    d1, d2, r = a1 - a0, b1 - b0, a0 - b0
    dot = lambda x, y: jnp.sum(x * y, -1)  # noqa: E731
    a, e, b, c, f = dot(d1, d1), dot(d2, d2), dot(d1, d2), dot(d1, r), dot(d2, r)
    s = jnp.clip((b * f - c * e) / jnp.maximum(a * e - b * b, 1e-12), 0.0, 1.0)  # parallel: any point of a, improved below
    t = jnp.clip((b * s + f) / e, 0.0, 1.0)
    s = jnp.clip((b * t - c) / a, 0.0, 1.0)
    s, t = jax.lax.stop_gradient((s[..., None], t[..., None]))
    return jnp.sqrt(jnp.sum(jnp.square(r + s * d1 - t * d2), -1) + 1e-12)


class SelfDistance:
    """Signed distances (m) between every geom of `a` and every geom of `b` (the colliders of robot.json's `self_collision`
    groups; pair k is `geoms[ia[k]]` with `geoms[ib[k]]`), negative when they overlap: a differentiable function of a state S. A
    capsule is itself, a segment with a radius. A box is the capsule along its longest axis from face to face, with the
    half-diagonal of the box's cross-section as radius: it contains the box, so the distance is never above the true one, and
    is below it by at most `slack` per pair (per box: the radius at the capsule's end caps; across the box, the radius minus
    the box's half-size in that direction)."""

    def __init__(self, m: mujoco.MjModel, a, b):
        self.geoms = geoms = [int(g) for g in (*a, *b)]
        bodies = sorted({int(m.geom_bodyid[g]) for g in geoms})
        self.kin = Kinematics(m, [m.body(i).name for i in bodies])
        ends, radius, slack = [], [], []
        for g in geoms:
            R = np.asarray(quat_to_mat(np.asarray(m.geom_quat[g], np.float32)))
            kind, size = int(m.geom_type[g]), m.geom_size[g]
            if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                axis, half, r = 2, size[1], size[0]
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:  # ponytail: one fat capsule; several thin ones if a flat box's faces matter
                axis = int(np.argmax(size))
                half, r = size[axis], float(np.linalg.norm(np.delete(size, axis)))
            else:
                raise ValueError(f"self_collision geom {m.geom(g).name} has unsupported type {kind}")
            ends.append([m.geom_pos[g] + s * half * R[:, axis] for s in (1, -1)])
            radius.append(r)
            slack.append(r * (kind == mujoco.mjtGeom.mjGEOM_BOX))
        self.frame = np.asarray([bodies.index(int(m.geom_bodyid[g])) for g in geoms])
        self.ends = np.asarray(ends, np.float32)  # (n, 2, 3), in the body frames
        self.ia, self.ib = (x.ravel() for x in np.meshgrid(np.arange(len(a)), len(a) + np.arange(len(b)), indexing="ij"))
        self.radius = (np.asarray(radius)[self.ia] + np.asarray(radius)[self.ib]).astype(np.float32)
        self.slack = np.asarray(slack)[self.ia] + np.asarray(slack)[self.ib]

    def __call__(self, S):
        p, R, _, _, q, _ = S
        P, Rs = self.kin.forward(p, R, q)
        e = P[..., self.frame, None, :] + mv3(Rs[..., self.frame, None, :, :], self.ends)
        a, b = e[..., self.ia, :, :], e[..., self.ib, :, :]
        return _segment_distance(a[..., 0, :], a[..., 1, :], b[..., 0, :], b[..., 1, :]) - self.radius


class Sim:
    """N parallel envs of Holosoma's task: `init` resets all of them and `step` advances them by one policy step. `view` and
    `view_state` give the task's quantities (frames, terrain heights, clearances) of any state, also an imagined one."""

    def __init__(self, cfg: SimCfg = SimCfg()):
        self.cfg = c = cfg
        self.N = N = c.num_envs
        self.reward_cfg = {"ppo": task.PPO_REWARD, "fast_sac": task.FAST_SAC_REWARD}[c.preset]
        self.curriculum_cfg = {"ppo": task.PPO_CURRICULUM, "fast_sac": task.FAST_SAC_CURRICULUM}[c.preset]
        if c.terrain == "mix":
            self.terrain = terrain_mod.generate(terrain_mod.TerrainCfg.scaled(c.terrain_scale, c.seed))
            self.heights = np.asarray(self.terrain.heights_m, np.float32)
            self.x0, self.hs = self.terrain.x0, self.terrain.cfg.horizontal_scale
        elif c.terrain == "plane":
            self.terrain, self.heights = None, np.zeros((1, 1), np.float32)
        else:
            raise ValueError(f"terrain must be 'mix' or 'plane', not {c.terrain!r}")
        self.m = m = scene.build(c.foot, self.terrain, tiles=c.tiles, iterations=c.iterations, ls_iterations=c.ls_iterations,
                                 ccd_tolerance=c.ccd_tolerance, self_collision=task.SELF_COLLISION)  # fmt: skip
        # robot.json's payload bodies, checked first: a jointed one would fail the joint-order check below less clearly
        names = [m.body(i).name for i in range(m.nbody)]
        for b in task.PAYLOAD_BODIES:
            assert f"robot_{b}" in names, f"payload body {b!r} is not a body of robot.xml"
            i = names.index(f"robot_{b}")
            assert m.body_jntnum[i] == 0, f"payload body {b!r} has a joint: it must be welded to its parent"
            assert m.body_mass[i] > 0, f"payload body {b!r} needs a nominal mass (its inertia is scaled by mass / nominal mass)"
            assert not ((m.geom_bodyid == i) & ((m.geom_contype | m.geom_conaffinity) > 0)).any(), f"payload body {b!r} has a collider"
            assert b not in task.MASS_BODIES + (task.TORSO,), f"payload body {b!r} is randomized already (mass_bodies / torso)"
        self.payload = np.asarray([names.index(f"robot_{b}") for b in task.PAYLOAD_BODIES], int)
        joints = tuple(m.joint(j).name.removeprefix("robot_").removesuffix("_joint") for j in range(1, m.njnt))
        assert joints == task.DOF, f"joint order {joints} != task.DOF"
        assert [int(m.actuator_trnid[i, 0]) for i in range(m.nu)] == list(range(1, m.njnt)), "one motor per joint, in joint order"
        assert (m.geom_condim[m.geom_contype > 0] == 3).all() and m.npair == 0, "contacts are dynamic and read as pyramidal condim 3"

        # termination colliders (the geoms of the termination bodies) and sole colliders (the robot side of the explicit pairs),
        # each reduced to points with a radius (see _points)
        tb = [m.body(f"robot_{b}").id for b in task.TERMINATION_BODIES]
        tg = [g for g in range(m.ngeom) if m.geom_bodyid[g] in tb]
        assert len({int(m.geom_bodyid[g]) for g in tg}) == len(tg), "one collider per termination body"
        self.collider_geoms = np.asarray(tg)
        self.term_slot = np.full(m.ngeom, -1, np.int32)
        self.term_slot[tg] = np.arange(len(tg))
        self.sole_geoms = sole_geoms = scene.sole_geoms(m, [f"robot_{b}" for b in task.FEET])
        # the bodies whose poses forward kinematics returns: the two foot contact points, the two ankle-roll links (the task's own
        # frames, in this order), then the bodies of the termination colliders
        frames = [m.body(f"robot_{b}").id for b in task.FOOT_POINTS + task.FEET] + [int(m.geom_bodyid[g]) for g in tg]
        self.coll_local, self.coll_frame, self.coll_radius = _points(m, tg, frames, bottom=False)
        self.sole_local, self.sole_frame, self.sole_radius = _points(m, sole_geoms, frames, bottom=True)
        self.kin = Kinematics(m, [m.body(b).name for b in frames])
        P, Rs = self.kin.forward(jnp.zeros(3), jnp.eye(3), jnp.asarray(task.DEFAULT_Q))  # the default pose, pelvis at the origin
        y = np.asarray(P)[2:4, 1]
        assert y[0] > y[1], "task.FEET must name the left foot first (task.feet_dy)"
        # the standing height (the pelvis over the lowest sole point in the default pose) and the rope's stretch under the
        # nominal weight (SimCfg.support)
        self.stand_h = float(-(np.asarray(P[self.sole_frame] + mv3(Rs[self.sole_frame], self.sole_local))[:, 2] - self.sole_radius).min())
        self.hang = float(m.body_mass.sum() * 9.81 / task.SUPPORT_K)
        if task.SELF_COLLISION is not None:  # the two groups' colliders are the geoms with scene.py's contact bit 4
            self.self_distance = SelfDistance(m, np.flatnonzero(m.geom_contype & 4), np.flatnonzero(m.geom_conaffinity & 4))
        self.mass_ids = np.asarray([m.body(f"robot_{b}").id for b in task.MASS_BODIES])
        self.torso = m.body(f"robot_{task.TORSO}").id
        if c.support_point is not None:  # Sim._rope
            self._hang_pose(m)

        self.mx = mjx.put_model(m, impl="warp")
        self.dx = mjx.make_data(m, impl="warp", naconmax=N * c.nconmax, njmax=c.njmax, naccdmax=N * (c.naccdmax or c.nconmax))
        # vmap axes of the model: only the start-up randomized fields differ per world, the rest is shared
        self.axes = jax.tree.map(lambda _: None, self.mx).tree_replace({k: 0 for k in WORLD_FIELDS})

    # ------------------------------------------------------------------ geometry shared with imagination
    def height(self, heights, x, y):
        """Terrain height at world (x, y), zero on the plane."""
        if self.terrain is None:
            return jnp.zeros_like(x)
        return terrain_mod.height(heights, self.x0, self.hs, x, y)

    def view(self, qpos, qvel, heights) -> View:
        return self.view_state(unflatten(qpos, qvel), heights)

    def view_state(self, S, heights) -> View:
        """The task's view of states S = (p, R, v, w, q, qd) (world-frame v and w): forward kinematics of the frames the task
        reads and the terrain under the robot. Shared by the simulator and imagination (differentiable)."""
        p, R, v, w, q, qd = S
        P, Rs = self.kin.forward(p, R, q)
        rb = task.Robot(p, R, v, mv3(tr3(R), w), q, qd, P[..., 0:2, :], P[..., 2:4, :], Rs[..., 2:4, :, :])

        def clearance(local, frame, radius):  # height of each collider point over the terrain, minus its radius
            pts = P[..., frame, :] + mv3(Rs[..., frame, :, :], local)
            return pts[..., 2] - radius - self.height(heights, pts[..., 0], pts[..., 1])

        return View(rb, rb.foot_p[..., 2] - self.height(heights, rb.foot_p[..., 0], rb.foot_p[..., 1]),
                    clearance(self.coll_local, self.coll_frame, self.coll_radius), clearance(self.sole_local, self.sole_frame, self.sole_radius),
                    p[..., 2] - self.height(heights, p[..., 0], p[..., 1]))  # fmt: skip

    # ------------------------------------------------------------------ setup
    def world(self, key) -> World:
        """Start-up randomization (spec 9): link masses x U[0.9, 1.2] (inertia by the same ratio), torso mass + U[-1, 3] kg
        (inertia by the ratio) and CoM + U[-0.05, 0.05] m per axis, friction from 64 quantiles of U[0.5, 1.25] per env for
        every geom (robot and terrain; the soles only with sole_friction); robot.json's payload bodies: mass and CoM uniform in
        their ranges, inertia by the mass ratio."""
        m, N, c = self.m, self.N, self.cfg
        k_link, k_add, k_com, k_mu = jax.random.split(key, 4)
        tile = lambda x: jnp.tile(jnp.asarray(x, jnp.float32)[None], (N,) + (1,) * np.ndim(x))  # noqa: E731
        mass, inertia, ipos = tile(m.body_mass), tile(m.body_inertia), tile(m.body_ipos)
        ids, torso = self.mass_ids, self.torso
        s = jax.random.uniform(k_link, (N, len(ids)), minval=task.MASS_SCALE[0], maxval=task.MASS_SCALE[1])
        mass, inertia = mass.at[:, ids].multiply(s), inertia.at[:, ids].multiply(s[..., None])
        add = jax.random.uniform(k_add, (N,), minval=task.TORSO_ADDED_MASS[0], maxval=task.TORSO_ADDED_MASS[1])
        ratio = (m.body_mass[torso] + add) / m.body_mass[torso]
        mass, inertia = mass.at[:, torso].add(add), inertia.at[:, torso].multiply(ratio[:, None])
        ipos = ipos.at[:, torso].add(jax.random.uniform(k_com, (N, 3), minval=-task.TORSO_COM, maxval=task.TORSO_COM))
        if len(p := self.payload):  # a key of its own, so that the draws above stay those of a robot without payload
            u = jax.random.uniform(jax.random.fold_in(key, 1), (N, len(p), 4), minval=task.PAYLOAD_LO, maxval=task.PAYLOAD_HI)
            inertia = inertia.at[:, p].multiply(u[..., :1] / m.body_mass[p][:, None])
            mass, ipos = mass.at[:, p].set(u[..., 0]), ipos.at[:, p].set(u[..., 1:])
        geom_friction = self._friction(tile(m.geom_friction), self._mu(k_mu))
        if self.terrain is None:
            origin = np.zeros((N, 3), np.float32)
        else:
            origin = terrain_mod.sample_env_origins(self.terrain, N, np.random.RandomState(c.seed)).astype(np.float32)
        return World(jnp.asarray(origin), mass, inertia, ipos, geom_friction, jnp.asarray(self.heights), self.mx)

    def _mu(self, key):
        """(N,) sliding friction coefficients: with SimCfg.friction_surfaces a surface drawn uniformly, then U[lo, hi] of it; else
        one of task.FRICTION_BUCKETS equally spaced values of task.FRICTION."""
        if self.cfg.friction_surfaces:
            k_s, k_u = jax.random.split(key)
            lo, hi = jnp.asarray(self.cfg.friction_surfaces, jnp.float32)[jax.random.randint(k_s, (self.N,), 0, len(self.cfg.friction_surfaces))].T
            return lo + (hi - lo) * jax.random.uniform(k_u, (self.N,))
        lo, hi = task.FRICTION
        return lo + (hi - lo) * (jax.random.randint(key, (self.N,), 0, task.FRICTION_BUCKETS) + 0.5) / task.FRICTION_BUCKETS

    def _friction(self, friction, mu):
        """geom_friction (N, ngeom, 3) with the sliding coefficient of every geom set to its env's mu, the soles' only with
        sole_friction. Every geom of a contact then has mu: the soles by their priority (scene.py), any other pair as MuJoCo's
        maximum of the two. Torsional and rolling coefficients stay the model's (the soles' contacts have condim 3: unused)."""
        keep = np.isin(np.arange(self.m.ngeom), self.sole_geoms) & (not self.cfg.sole_friction)  # Holosoma's MuJoCo backends: soles stay at 0.8
        return friction.at[:, :, 0].set(jnp.where(keep, friction[:, :, 0], mu[:, None]))

    def _new_surface(self, w: World, done, key) -> World:
        """`w` with a friction coefficient drawn anew (_mu) for the envs where `done` is set."""
        new = self._friction(w.geom_friction, self._mu(key))
        return w._replace(geom_friction=jnp.where(done[:, None, None], new, w.geom_friction))

    def context(self, s: State):
        """(N, 72) dynamics context xi of each env: its PD gain scales, link mass scales, torso added mass and CoM offset,
        friction, then per payload body (robot.json) its added mass and CoM offset (4 more each). These differ between envs but
        cannot be read off a state, so the world model gets them as an extra input."""
        m, w, t, p = self.m, s.world, self.torso, self.payload
        xi = jnp.concatenate((s.kp, s.kd, w.body_mass[:, self.mass_ids] / m.body_mass[self.mass_ids], (w.body_mass[:, t] - m.body_mass[t])[:, None],
                              w.body_ipos[:, t] - m.body_ipos[t], w.geom_friction[:, 0, :1]), -1)  # fmt: skip
        if len(p):
            pay = jnp.concatenate(((w.body_mass[:, p] - m.body_mass[p])[..., None], w.body_ipos[:, p] - m.body_ipos[p]), -1)
            xi = jnp.concatenate((xi, pay.reshape(len(pay), -1)), -1)
        if self.cfg.support:  # the rope's top and the speed at which it is paid out: the last 2 (imagination pays it out too)
            xi = jnp.concatenate((xi, s.sup[:, None], self._rate(s)[:, None]), -1)
        return xi

    def _rate(self, s: State):
        """(N,) the speed (m/s) at which each env's rope is being paid out: zero without a rope and while it holds."""
        return jnp.where((s.sup > task.SUPPORT_OFF) & (s.sup_wait <= 0), task.SUPPORT_SPEED, 0.0)

    def init(self, key) -> State:
        """Reset every env, then take one zero-action step, as Holosoma does at start-up (the curriculum updates in both)."""
        N, J = self.N, task.J
        k_world, k_reset, k_t, key = jax.random.split(key, 4)
        world = self.world(k_world)
        data = jax.vmap(lambda q: self.dx.replace(qpos=q))(jnp.zeros((N, self.m.nq), jnp.float32))  # placeholder, reset below
        zf, zi = jnp.zeros((N,), jnp.float32), jnp.zeros((N,), jnp.int32)
        c0 = self.curriculum_cfg
        s = State(data, world, jnp.zeros((N, task.ACTOR_DIM)), jnp.zeros((N, task.CRITIC_DIM)), jnp.zeros((N, J)), jnp.ones((N, J)),
                  jnp.ones((N, J)), zi, jnp.zeros((N, 3)), zf, jnp.zeros((N, 2)), jnp.zeros((N, 2)), zi, zi, zi,
                  jnp.asarray(c0.initial, jnp.float32), jnp.zeros((), jnp.float32), jnp.zeros((N, len(task.TERMS))), key)  # fmt: skip
        if self.cfg.support:
            s = s._replace(sup=jnp.full((N,), task.SUPPORT_OFF, jnp.float32), sup_wait=zi)
        s = self._reset(s, jnp.ones((N,), bool), k_reset)
        # the penalty curriculum updates once here, as in Holosoma's reset_all, and once more in the step
        scale, avg_len = task.curriculum_step(self.curriculum_cfg, s.scale, s.avg_len, jnp.zeros((N,)), jnp.zeros((N,), bool), N)
        s, _ = self.step(s._replace(scale=scale, avg_len=avg_len), jnp.zeros((N, J)))
        if self.cfg.random_start:  # episodes then end at different times
            s = s._replace(t=jax.random.randint(k_t, (N,), 0, self.cfg.max_episode))
        return s

    # ------------------------------------------------------------------ one policy step
    def physics(self, world: World, data, target, kp, kd, top=None):
        """Advance every world by one policy step: DECIMATION physics substeps, with the PD torques recomputed from the current
        joint state before each one (spec 1.5). `top` (N,), with SimCfg.support: the world height of each env's rope top; the
        rope's tension (task.support_force of the pelvis height) pulls the torso up, recomputed before each substep too."""
        mx = world.model.tree_replace({k: getattr(world, k) for k in WORLD_FIELDS})

        def one(m_, d, tg, kp_, kd_, top_):
            def substep(_, dd):
                dd = dd.replace(ctrl=task.torques(dd.qpos[7:], dd.qvel[6:], tg, kp_, kd_))
                if top_ is not None and self.cfg.support_point is not None:
                    dd = dd.replace(xfrc_applied=dd.xfrc_applied.at[self.torso].set(self._rope(m_.body_ipos[self.torso], dd.qpos, dd.qvel, top_)))
                elif top_ is not None:
                    dd = dd.replace(xfrc_applied=dd.xfrc_applied.at[self.torso, 2].set(task.support_force(top_, dd.qpos[2], dd.qvel[2])))
                return mjx.step(m_, dd)

            return jax.lax.fori_loop(0, task.DECIMATION, substep, d)

        return jax.vmap(one, in_axes=(self.axes, 0, 0, 0, 0, None if top is None else 0))(mx, data, target, kp, kd, top)

    def contact_termination(self, data):
        """Net contact force above 1 N on a termination body after the last substep."""
        return (jnp.linalg.norm(self.collider_forces(data), axis=-1) > task.CONTACT_FORCE).any(-1)

    def collider_forces(self, data):
        """(N, n, 3) world-frame net contact force on each termination collider (one per body) after the last substep, rebuilt
        from the solver's constraint forces: every contact of the (global) contact pool is decoded from its 4 pyramid rows into
        a force in its contact frame, turned into the world frame, and summed per (world, collider). With robot.json's
        `self_collision`, only the contacts with a terrain geom (a geom of the world body) are summed."""
        im, N = data._impl, self.N
        n_slots = int(self.term_slot.max()) + 1
        valid = jnp.arange(im.contact__dist.shape[0]) < im.nacon[0]
        w, rows = im.contact__worldid, im.contact__efc_address
        p = im.efc__force[jnp.clip(w, 0, N - 1)[:, None], jnp.maximum(rows, 0)] * (rows >= 0)
        mu = im.contact__friction
        f = jnp.stack((p.sum(-1), (p[:, 0] - p[:, 1]) * mu[:, 0], (p[:, 2] - p[:, 3]) * mu[:, 1]), -1)  # contact frame
        F = (im.contact__frame * f[:, :, None]).sum(1)  # world frame: frame^T f
        slot_of = jnp.asarray(self.term_slot)
        slot = jnp.maximum(slot_of[im.contact__geom[:, 0]], slot_of[im.contact__geom[:, 1]])
        hit = valid & (slot >= 0)
        if task.SELF_COLLISION is not None:  # a contact between two robot geoms is not a fall: only terrain contacts count
            hit &= jnp.asarray(self.m.geom_bodyid == 0)[im.contact__geom].any(-1)
        idx = jnp.clip(w, 0, N - 1) * n_slots + jnp.maximum(slot, 0)
        acc = jnp.zeros((N * n_slots, 3)).at[idx].add(jnp.where(hit[:, None], F, 0.0))
        return acc.reshape(N, n_slots, 3)

    def _fresh(self, key, origin):
        """Fresh values for every env, which `_reset` applies where an episode ended (spec 8 and 9): pose, velocities, PD gain
        scales, action delay, command and gait frequency, gait offsets and push interval."""
        N, J = self.N, task.J
        k = jax.random.split(key, 9)
        q = task.DEFAULT_Q * jax.random.uniform(k[0], (N, J), minval=task.SPAWN_Q[0], maxval=task.SPAWN_Q[1])
        xy = origin[:, :2] + jax.random.uniform(k[1], (N, 2), minval=-task.SPAWN_XY, maxval=task.SPAWN_XY)
        quat = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0, 0.0]), (N, 4))
        qpos = jnp.concatenate((xy, origin[:, 2:3] + task.SPAWN_HEIGHT, quat, q), -1)
        qvel = jnp.concatenate((jax.random.uniform(k[2], (N, 6), minval=-task.SPAWN_VEL, maxval=task.SPAWN_VEL), jnp.zeros((N, J))), -1)
        kp = jax.random.uniform(k[3], (N, J), minval=task.GAIN_SCALE[0], maxval=task.GAIN_SCALE[1])
        kd = jax.random.uniform(k[4], (N, J), minval=task.GAIN_SCALE[0], maxval=task.GAIN_SCALE[1])
        delay = jax.random.randint(k[5], (N,), 0, task.MAX_DELAY + 1)
        cmd, freq = task.sample_commands(k[6], N, self.cfg.cmd_scale)
        if self.cfg.start_share:  # SimCfg.start_share: another starting pose, and command, for a share of the episodes
            k_on, k_q, k_c = jax.random.split(jax.random.fold_in(key, 9), 3)
            on = (jax.random.uniform(k_on, (N,)) < self.cfg.start_share)[:, None]
            i, lo, hi = (np.asarray(x) for x in zip(*self.cfg.start_joints))
            q_s = np.broadcast_to(task.DEFAULT_Q, (J,))[i] + jax.random.uniform(k_q, (N, len(i)), minval=lo, maxval=hi)
            qpos = qpos.at[:, 7 + i].set(jnp.where(on, q_s, qpos[:, 7 + i]))
            if self.cfg.start_cmd:
                lo, hi = np.asarray(self.cfg.start_cmd, np.float32).T
                cmd = jnp.where(on, jax.random.uniform(k_c, (N, 3), minval=lo, maxval=hi), cmd)
        return qpos, qvel, kp, kd, delay, cmd, freq, task.sample_phase_offsets(k[7], N), self._push_interval(k[8])

    def _push_interval(self, key):
        """Steps until each env's next push, drawn from PUSH_INTERVAL (in seconds)."""
        u = jax.random.uniform(key, (self.N,), minval=task.PUSH_INTERVAL[0], maxval=task.PUSH_INTERVAL[1])
        return (u / task.DT).astype(jnp.int32)

    def _reset(self, s: State, done, key) -> State:
        """Replace the state of the envs where `done` is set by freshly drawn values."""
        qpos, qvel, kp, kd, delay, cmd, freq, offset, push_at = self._fresh(key, s.world.origin)
        on, on2 = done[:, None], done  # the reset mask, shaped for fields with a trailing axis and for per-env scalars
        if self.cfg.support:
            qpos, qvel, cmd, sup, wait = self._support(jax.random.fold_in(key, 3), qpos, qvel, cmd, s.world.heights)
            s = s._replace(sup=jnp.where(on2, sup, s.sup), sup_wait=jnp.where(on2, wait, s.sup_wait))
        if self.cfg.friction_surfaces:  # a new surface per episode
            s = s._replace(world=self._new_surface(s.world, done, jax.random.fold_in(key, 4)))
        d = s.data
        # the solver's warm start (the previous acceleration) belongs to the old state, so it is cleared too
        d = d.replace(qpos=jnp.where(on, qpos, d.qpos), qvel=jnp.where(on, qvel, d.qvel), qacc_warmstart=jnp.where(on, 0.0, d.qacc_warmstart))
        return s._replace(data=d, a_last=jnp.where(on, 0.0, s.a_last), kp=jnp.where(on, kp, s.kp), kd=jnp.where(on, kd, s.kd),
                          delay=jnp.where(on2, delay, s.delay), cmd=jnp.where(on, cmd, s.cmd), freq=jnp.where(on2, freq, s.freq),
                          offset=jnp.where(on, offset, s.offset), t=jnp.where(on2, 0, s.t), push_t=jnp.where(on2, 0, s.push_t),
                          push_at=jnp.where(on2, push_at, s.push_at), ep_sums=jnp.where(on, 0.0, s.ep_sums))  # fmt: skip

    def _hang_pose(self, m):
        """SimCfg.support_point: the pose in which the nominal robot rests on the rope, the default joint pose turned about the
        horizontal axis that puts the point straight above the robot's center of mass (with the root at the origin): its root
        quaternion (sup_quat), the root's height over the lowest sole point in it (sup_stand), and the point's height in it
        minus the standing height difference (sup_lift: the rope's length is taken at the point's height - sup_lift, so the rope's
        top keeps its meaning of SimCfg.support)."""
        d = mujoco.MjData(m)
        d.qpos[:7], d.qpos[7:] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), task.DEFAULT_Q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        self.sup_kin, self.sup_local = Kinematics(m, [m.body(self.torso).name]), np.asarray(self.cfg.support_point, np.float32)
        point = lambda R: np.asarray(self._point(jnp.zeros(3), jnp.asarray(R, jnp.float32), jnp.asarray(task.DEFAULT_Q), np.zeros(3))[0])  # noqa: E731
        r = point(np.eye(3)) - d.subtree_com[1]
        axis = np.cross(r, (0.0, 0.0, 1.0))
        half = np.arctan2(np.linalg.norm(axis), r[2]) / 2
        quat = np.concatenate(([np.cos(half)], np.sin(half) * axis / max(np.linalg.norm(axis), 1e-12)))
        R = np.asarray(quat_to_mat(jnp.asarray(quat, jnp.float32)))
        P, Rs = self.kin.forward(jnp.zeros(3), jnp.asarray(R, jnp.float32), jnp.asarray(task.DEFAULT_Q))
        self.sup_quat, self.sup_stand = quat.astype(np.float32), float(-(np.asarray(P[self.sole_frame] + mv3(Rs[self.sole_frame], self.sole_local))[:, 2] - self.sole_radius).min())
        self.sup_lift = float(point(R)[2] + self.sup_stand - self.stand_h)

    def _point(self, p, R, q, ipos):
        """World positions of the rope's point and of the rope body's center of mass (body frame offset ipos), root pose (p, R)."""
        Pb, Rb = self.sup_kin.forward(p, R, q)
        return Pb[0] + mv3(Rb[0], jnp.asarray(self.sup_local)), Pb[0] + mv3(Rb[0], ipos)

    def _rope(self, ipos, qpos, qvel, top):
        """SimCfg.support_point: the rope's wrench on its body (6,: force, torque about the center of mass ipos, as xfrc_applied):
        the tension task.support_force of the point's height - sup_lift and its vertical velocity, pulling straight up at it."""
        R = quat_to_mat(qpos[3:7])
        (pt, com), (v, _) = jax.jvp(lambda p, R, q: self._point(p, R, q, ipos), (qpos[:3], R, qpos[7:]), (qvel[:3], R @ skew(qvel[3:6]), qvel[6:]))
        T = task.support_force(top, pt[2] - self.sup_lift, v[2])
        f = jnp.stack((jnp.zeros_like(T), jnp.zeros_like(T), T))
        return jnp.concatenate((f, jnp.cross(pt - com, f)))

    def _support(self, key, qpos, qvel, cmd, heights):
        """SimCfg.support for fresh envs: a share of them starts on the rope, at rest in the default pose (upright; with
        SimCfg.support_point turned as it hangs, _hang_pose) with the lowest sole point support_gap over the terrain under the
        pelvis (on it for a negative gap), and without a command (an operator gives none while the robot hangs); the rope's top
        is where it holds the nominal robot at that gap, and it is paid out after a hold of task.SUPPORT_HOLD. Returns qpos, qvel, cmd, the rope's top and the steps until the pay-out."""
        N, (lo, hi) = self.N, self.cfg.support_gap
        k_on, k_gap, k_hold = jax.random.split(key, 3)
        on = (jax.random.uniform(k_on, (N,)) < self.cfg.support)[:, None]
        gap = jax.random.uniform(k_gap, (N,), minval=lo, maxval=hi)
        hung = self.cfg.support_point is not None  # then at rest in the hanging pose (_hang_pose)
        z = self.height(heights, qpos[:, 0], qpos[:, 1]) + (self.sup_stand if hung else self.stand_h) + jnp.maximum(gap, 0.0)
        quat = jnp.broadcast_to(jnp.asarray(self.sup_quat), (N, 4)) if hung else qpos[:, 3:7]
        pose = jnp.concatenate((qpos[:, :2], z[:, None], quat, jnp.broadcast_to(jnp.asarray(task.DEFAULT_Q), (N, task.J))), -1)
        hold = jax.random.uniform(k_hold, (N,), minval=task.SUPPORT_HOLD[0], maxval=task.SUPPORT_HOLD[1])
        return (jnp.where(on, pose, qpos), jnp.where(on, 0.0, qvel), jnp.where(on, 0.0, cmd), jnp.where(on[:, 0], gap + self.hang, task.SUPPORT_OFF),
                (hold / task.DT).astype(jnp.int32))  # fmt: skip

    def step(self, s: State, action) -> tuple[State, Transition]:
        """One policy step of all envs, in the order of the module docstring. Returns the next state, with the observations
        for the next action, and the step's Transition."""
        N = self.N
        key, k_reset, k_cmd, k_push, k_push_at, k_obs = jax.random.split(s.key, 6)
        a = action.astype(jnp.float32)
        target = jnp.clip(jnp.where(s.delay[:, None] == 0, a, s.a_last), -task.ACTION_CLIP, task.ACTION_CLIP)
        rope = {}
        if self.cfg.support:  # the rope's top in world height, over the terrain under the pelvis
            rope["top"] = self.height(s.world.heights, s.data.qpos[:, 0], s.data.qpos[:, 1]) + self.stand_h + s.sup
        data = self.physics(s.world, s.data, target, s.kp, s.kd, **rope)
        s1_qpos, s1_qvel = data.qpos, data.qvel
        t = s.t + 1
        v = self.view(s1_qpos, s1_qvel, s.world.heights)
        rb, foot_h, clearance = v.robot, v.foot_h, v.clearance
        # a physics failure (MuJoCo Warp's heightfield contacts can blow up under violent motion) ends the episode unrewarded
        diverged = ~(jnp.isfinite(s1_qpos).all(-1) & jnp.isfinite(s1_qvel).all(-1)) | (jnp.abs(s1_qvel) > DIVERGED).any(-1)
        force = jnp.linalg.norm(self.collider_forces(data), axis=-1)
        terminated = (force > task.CONTACT_FORCE).any(-1) | diverged
        timeout = t > self.cfg.max_episode
        done = terminated | timeout
        parts = task.weighted(task.reward_terms(rb, foot_h, a, s.a_last, s.cmd, s.phase), self.reward_cfg, s.scale)
        parts = jnp.where(diverged[:, None], 0.0, jnp.stack([parts[k] for k in task.TERMS], -1))
        reward = parts.sum(-1)
        _, critic_final = task.observations(rb, a, s.cmd, s.phase)
        ep_sums = s.ep_sums + parts
        fin = done.astype(jnp.float32)
        # term_geom: how often imagination's geometric termination (a collider below the terrain) agrees with the contact one;
        # nacon: the share of the contact pool in use
        stats = {"reward": reward.mean(), "done": fin.sum(), "terminated": terminated.sum().astype(jnp.float32),
                 "len_sum": (fin * t).sum(), "rew_sums": (fin[:, None] * ep_sums).sum(0) / (self.cfg.max_episode * task.DT),
                 "term_geom": (((clearance.min(-1) < 0) == terminated) | diverged).mean(), "diverged": diverged.sum().astype(jnp.float32),
                 "nacon": data._impl.nacon[0] / data._impl.contact__dist.shape[0]}  # fmt: skip
        s1 = s._replace(data=data, a_last=a, t=t, ep_sums=ep_sums)
        if self.cfg.support:  # the rope is paid out after its hold, then taken off
            s1 = s1._replace(sup=task.support_pay_out(s.sup, self._rate(s))[0], sup_wait=s.sup_wait - 1)

        # resets and the curriculum
        scale, avg_len = task.curriculum_step(self.curriculum_cfg, s.scale, s.avg_len, t.astype(jnp.float32), done, N)
        s1 = self._reset(s1, done, k_reset)._replace(scale=scale, avg_len=avg_len)

        # gait phase with the current command, then command resampling
        phase = task.gait_phase(s1.t, s1.freq, s1.offset, s1.cmd)
        resample = s1.t % task.RESAMPLE == 0
        cmd2, freq2 = task.sample_commands(k_cmd, N, self.cfg.cmd_scale)
        cmd, freq = jnp.where(resample[:, None], cmd2, s1.cmd), jnp.where(resample, freq2, s1.freq)
        if self.cfg.support:  # no command on the rope
            cmd = jnp.where((s1.sup > task.SUPPORT_OFF)[:, None], 0.0, cmd)
            stats["support"] = (s1.sup > task.SUPPORT_OFF).mean()

        # pushes
        push_t = s1.push_t + 1
        push = (push_t == s1.push_at) & (self.cfg.push_vel > 0)  # push_vel 0: no pushes (a zeroed velocity would be one)
        qvel = s1.data.qvel
        vxy = self.cfg.push_vel * jax.random.uniform(k_push, (N, 2), minval=-1.0, maxval=1.0)
        qvel = qvel.at[:, :2].set(jnp.where(push[:, None], vxy, qvel[:, :2]))
        push_at = jnp.where(push, self._push_interval(k_push_at), s1.push_at)
        data = s1.data.replace(qvel=qvel)

        qp = data.qpos
        # the robot after resets and pushes, with only the fields that the observations read (no frames)
        view = task.Robot(qp[:, :3], quat_to_mat(qp[:, 3:7]), qvel[:, :3], qvel[:, 3:6], qp[:, 7:], qvel[:, 6:], None, None, None)
        obs, critic = task.observations(view, s1.a_last, cmd, phase, k_obs)
        stats.update({"scale": scale, "avg_len": avg_len, "pushes": push.sum().astype(jnp.float32)})
        s2 = s1._replace(data=data, obs=obs, critic=critic, cmd=cmd, freq=freq, phase=phase, push_t=jnp.where(push, 0, push_t), push_at=push_at,
                         key=key)  # fmt: skip
        tr = Transition(s.data.qpos, s.data.qvel, target, s1_qpos, s1_qvel, reward, terminated, timeout & ~terminated, diverged, parts,
                        critic_final, clearance, stats, force)  # fmt: skip
        return s2, tr
