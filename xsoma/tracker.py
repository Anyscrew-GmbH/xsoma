"""A motion-tracking policy as the velocity task's policy, without further training.

A tracker policy trained with goal masking (XMimic's `goal_mask_share`) accepts, besides its reference goals, a command-only
goal: per future frame only the planar velocity and the yaw rate are set, every other entry holds a neutral value, and a flag
is appended. Its observation then relates the base to a commanded heading instead of a reference motion. The task's command
(vx, vy, wz) provides exactly that, so this module builds the tracker's observation, history and goal from the simulator's state
and maps the tracker's action to the simulator's:

    python -m xsoma.evaluate <tracker ckpt.npz> --tracker_meta <meta.json>     # XSOMA_ROBOT_DIR: the tracker's robot

`meta.json` is the tracker's robot description (its `joint_names`, `default_joint_pos` and `action_scale` are read). It must
describe the robot of the task: the same joints (the tracker's names end in `_joint`), in any order, and after the permutation
the same default pose and action scale, so that an action a means the joint target q0 + scale * a in both.

What the tracker gets (XMimic's mdp.py, sim.py and motion.py define it):
  observation       q - q0 and qd in the tracker's joint order, the base's angular velocity and the direction of gravity in the
                    base frame, the previous action, and the first two columns of R_base^T R_z(heading). Training adds
                    uniform noise per block (NOISE), the evaluation none
  history           the last H observations, oldest first; at an episode's first step all H are that step's observation
  goal              K future frames of (root displacement 3, rotation 6, velocity 3, angular velocity 3, q, 0.1 qd): v x, v y
                    and w z are the command, held over the K frames; the rest is neutral (zeros, the identity, q0); then a 1
  heading           "hold": the base's yaw at the episode's first step, then the commanded yaw rate integrated at the policy
                    rate. "rate": the base's own yaw at every step, so the relative rotation carries the tilt only and the
                    commanded yaw rate reaches the policy through the goal alone
  previous action   the simulator's last raw action, which is zero after a reset. The simulator's 0 or 1 step of action delay
                    acts on the tracker's action as on any policy's
The simulator clips actions at +-100 (task.ACTION_CLIP) and does not squash them; the tracker's raw action passes unchanged.

The forward pass (`act`) is XMimic's policy (xmimic/nets.py, pi_encode and pi_apply; MIT, by the same author), copied here so
that XSoma does not import XMimic.

train.py fine-tunes the encoder and the decoder on the task (`--init_tracker`, with the normalizers as loaded) and writes its
checkpoints in the layout read here, the live policy next to the averaged one under the prefix `live.`.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from . import nets, task
from .rot import axis_angle_to_mat, mat_to_6d, mm3, quat_to_mat, tr3

FRAME = 15  # a goal frame's entries before its joints: root displacement 3, rotation 6, velocity 3, angular velocity 3
CMD = np.asarray((9, 10, 14))  # v x, v y, w z in a goal frame
Z = np.asarray([0.0, 0.0, 1.0], np.float32)
# the tracker's training values: uniform noise within +- these, per observation block (q, qd, angular velocity, gravity, previous
# action, relative rotation)
NOISE = (0.01, 0.5, 0.2, 0.05, 0.0, 0.05)


def load(ckpt, meta, prefix=""):
    """The tracker as one pytree of arrays, from a checkpoint as XMimic writes it and the tracker's robot description: `enc` and
    `dec` (layers), `obs_norm` and `goal_norm` (the normalizers, frozen), `q0` and `alpha` (default pose and action scale in the
    tracker's joint order), `to_tracker` (x_tracker = x_sim[..., to_tracker]) and `to_sim`, its inverse. `prefix` "live." reads
    the live policy of a checkpoint that train.py wrote."""
    z = np.load(ckpt)
    args = json.loads(str(z["meta"]))["args"]
    assert args.get("goal_mask_share", 0.0) > 0, f"{ckpt} was trained without goal masking: it does not accept a command-only goal"
    with open(meta) as f:
        meta = json.load(f)
    names = [n.removesuffix("_joint") for n in meta["joint_names"]]
    assert sorted(names) == sorted(task.DOF), f"the tracker's joints are not the robot's: {sorted(set(names) ^ set(task.DOF))} differ"
    to_tracker = np.asarray([task.DOF.index(n) for n in names])
    q0, alpha = np.asarray(meta["default_joint_pos"], np.float32), np.asarray(meta["action_scale"], np.float32)
    for what, a, b in (("default joint pose", q0, task.DEFAULT_Q), ("action scale", alpha, task.ACTION_SCALE)):
        b = np.broadcast_to(b, (task.J,))[to_tracker]
        bad = np.nonzero(np.abs(a - b) > 1e-5)[0]
        assert not len(bad), f"the {what} of the tracker and of the robot differ at {[names[i] for i in bad]}: {a[bad]} and {b[bad]}"
    J, K, H = task.J, args["goal_frames"], args["hist"]
    p = {net: [{k: jnp.asarray(z[f"{prefix}{net}.{i}.{k}"]) for k in ("w", "b")} for i in range(sum(f.startswith(prefix + net + ".") and f.endswith(".w") for f in z.files))]
         for net in ("enc", "dec")}  # fmt: skip
    p |= {n: {k: jnp.asarray(z[f"{n}.{k}"]) for k in ("mean", "var")} for n in ("obs_norm", "goal_norm")}
    sizes = (p["enc"][0]["w"].shape[0], p["dec"][0]["w"].shape[0], p["dec"][-1]["w"].shape[1])
    expected = (K * (FRAME + 2 * J) + 1, p["enc"][-1]["w"].shape[1] + H * (3 * J + 12), J)
    assert sizes == expected, f"{ckpt}: goal, decoder input and action sizes {sizes}, but {expected} for {J} joints, {K} goal frames and a history of {H}"
    return p | {"q0": jnp.asarray(q0), "alpha": jnp.asarray(alpha), "to_tracker": jnp.asarray(to_tracker), "to_sim": jnp.asarray(np.argsort(to_tracker))}


def observation(p, q, qd, R, w, a_prev, heading, key=None, noise=1.0):
    """The tracker's observation (..., 3 J + 12) of robots with the joint state q, qd (the simulator's order), the base rotation
    R, the base's angular velocity w in the base frame, the previous tracker action (the tracker's order) and the commanded
    heading (rad; None: the base's own yaw). With `key`, uniform noise within +- `noise` x NOISE is added."""
    i = p["to_tracker"]
    if heading is None:
        heading = jnp.arctan2(R[..., 1, 0], R[..., 0, 0])
    rel = mat_to_6d(mm3(tr3(R), axis_angle_to_mat(Z, heading)))
    o = jnp.concatenate((q[..., i] - p["q0"], qd[..., i], w, task.projected_gravity(R), a_prev, rel), -1)
    if key is None:
        return o
    J = q.shape[-1]
    return o + (2 * jax.random.uniform(key, o.shape) - 1) * noise * np.repeat(np.asarray(NOISE, np.float32), (J, J, 3, 3, J, 6))


def goal(p, cmd):
    """The command-only goal (..., K (15 + 2 J) + 1) of the commands (..., 3) = v x, v y, w z."""
    J = p["q0"].shape[0]
    K = (p["enc"][0]["w"].shape[0] - 1) // (FRAME + 2 * J)
    frame = jnp.concatenate((jnp.asarray([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], jnp.float32), p["q0"], jnp.zeros(J)))
    g = jnp.broadcast_to(frame, cmd.shape[:-1] + (K, FRAME + 2 * J)).at[..., CMD].set(cmd[..., None, :])
    return jnp.concatenate((g.reshape(cmd.shape[:-1] + (-1,)), jnp.ones(cmd.shape[:-1] + (1,))), -1)


def act(p, hist, g, out_bf16=False):
    """The tracker's action (its joint order) for the observation histories (..., H, 3 J + 12) and the goals g:
    decoder(tanh(encoder(g)), history) + (the goal's next joint positions - q0) / alpha. The last term is zero for a
    command-only goal. `out_bf16`: see nets.mlp_apply."""
    J = p["q0"].shape[0]
    z = jnp.tanh(nets.mlp_apply(p["enc"], nets.rn_apply(p["goal_norm"], g), jax.nn.elu, out_bf16=out_bf16))
    o = nets.rn_apply(p["obs_norm"], hist).reshape(hist.shape[:-2] + (-1,))
    return nets.mlp_apply(p["dec"], jnp.concatenate((z, o), -1), jax.nn.elu, out_bf16=out_bf16) + (g[..., FRAME : FRAME + J] - p["q0"]) / p["alpha"]


# the sagittal mirror M = diag(1, -1, 1) of the observation's base blocks: the angular velocity is a pseudo-vector (-M w), gravity
# a vector (M g), the relative rotation X becomes M X M (its first two columns, row-major as rot.mat_to_6d); and of a command
MIRROR_W, MIRROR_G, MIRROR_REL = np.float32([-1, 1, -1]), np.float32([1, -1, 1]), np.float32([1, -1, -1, 1, 1, -1])
MIRROR_CMD = np.float32([1, -1, -1])  # v x, v y, w z


def mirror(meta):
    """The left/right mirror image of the tracker's joints (its order) from its robot description (`kin_joint_idx`, `kin_axes`, as
    XMimic's algorithm.Mirror reads them): (perm, sign), joint i of the mirror image = sign[i] x joint perm[i]. A joint `left_*`
    pairs with `right_*`, any other with itself. A rotation about the axis a mirrors to one about -M a, so the sign is -(a_partner .
    M a). The default pose and the action scale must be their own mirror images, so that mirroring q - q0 and the actions is exact."""
    with open(meta) as f:
        meta = json.load(f)
    names = list(meta["joint_names"])
    axis = {names[j]: np.asarray(a) for j, a in zip(meta["kin_joint_idx"], meta["kin_axes"]) if j >= 0}
    swap = lambda n: n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")  # noqa: E731
    perm = np.asarray([names.index(swap(n)) for n in names])
    s = np.asarray([-float(axis[names[p]] @ (axis[n] * [1.0, -1.0, 1.0])) for n, p in zip(names, perm)])
    assert np.allclose(np.abs(s), 1, atol=1e-4), f"joints whose partner is not their mirror image: {[n for n, x in zip(names, s) if abs(abs(x) - 1) > 1e-4]}"
    sign = np.sign(s).astype(np.float32)
    for what in ("default_joint_pos", "action_scale"):
        x = np.asarray(meta[what], np.float32)
        x_m = x[perm] if what == "action_scale" else x[perm] * sign
        assert np.allclose(x_m, x, atol=1e-5), f"the {what} is not its own mirror image"
    return perm, sign


def mirror_obs(m, o):
    """The observation (..., 3 J + 12; also a history of them) of the mirror image of the robot, `m` = mirror(meta): the joint blocks
    (q - q0, qd, previous action) swapped with their signs, the base blocks as MIRROR_W, MIRROR_G, MIRROR_REL."""
    perm, sign = m
    J = len(perm)
    j = lambda x: x[..., perm] * sign  # noqa: E731
    return jnp.concatenate((j(o[..., :J]), j(o[..., J : 2 * J]), o[..., 2 * J : 2 * J + 3] * MIRROR_W, o[..., 2 * J + 3 : 2 * J + 6] * MIRROR_G,
                            j(o[..., 2 * J + 6 : 3 * J + 6]), o[..., 3 * J + 6 :] * MIRROR_REL), -1)  # fmt: skip


def memory(p, n):
    """What the tracker carries through an episode, for n envs: the observation history, the commanded heading and the episode
    counter of its last step. The initial counter is above any episode's, so `step` starts every env anew."""
    J = p["q0"].shape[0]
    H = (p["dec"][0]["w"].shape[0] - p["enc"][-1]["w"].shape[1]) // (3 * J + 12)
    return {"hist": jnp.zeros((n, H, 3 * J + 12)), "heading": jnp.zeros((n,)), "t": jnp.full((n,), jnp.iinfo(jnp.int32).max)}


def step(p, mem, s, hold=True, key=None, noise=1.0):
    """One policy step on the simulator state s: (the action for Sim.step, the memory for the next step). An env whose episode
    counter did not advance has been reset: its history is refilled and its heading restarts at the base's yaw. `hold` False:
    the observation refers to the base's own yaw instead of the heading; `key`, `noise`: see `observation`."""
    fresh = s.t <= mem["t"]
    qpos, qvel = s.data.qpos, s.data.qvel
    R = quat_to_mat(qpos[:, 3:7])
    heading = jnp.where(fresh, jnp.arctan2(R[:, 1, 0], R[:, 0, 0]), mem["heading"])
    o = observation(p, qpos[:, 7:], qvel[:, 6:], R, qvel[:, 3:6], s.a_last[:, p["to_tracker"]], heading if hold else None, key, noise)
    hist = jnp.where(fresh[:, None, None], o[:, None], jnp.concatenate((mem["hist"][:, 1:], o[:, None]), 1))
    a = act(p, hist, goal(p, s.cmd))
    return a[:, p["to_sim"]], {"hist": hist, "heading": heading + task.DT * s.cmd[:, 2], "t": s.t}


def history(p, qpos, qvel, a_last, heading, done, key=None, noise=1.0):
    """The observation histories (..., H, 3 J + 12) that `step` held at the last of H consecutive steps of an env, rebuilt from
    what a replay buffer keeps of every step (..., H, .): the simulator's qpos and qvel, the last raw action, the commanded
    heading of the step's observation and whether the step ended its episode. Steps of an earlier episode are replaced by the
    first step of the last step's episode, as `step` fills a new history. `heading` None, `key`, `noise`: see `observation`;
    the noise is drawn anew, once per stored step."""
    o = observation(p, qpos[..., 7:], qvel[..., 6:], quat_to_mat(qpos[..., 3:7]), qvel[..., 3:6], a_last[..., p["to_tracker"]], heading, key, noise)
    H = o.shape[-2]
    first = jnp.max(jnp.arange(1, H) * (done[..., :-1] > 0), -1, initial=0)  # the window index where the last step's episode starts
    return jnp.take_along_axis(o, jnp.maximum(jnp.arange(H), first[..., None])[..., None], -2)
