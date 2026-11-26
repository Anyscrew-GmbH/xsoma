"""The left/right symmetry term (train.py --symmetry, tracker.mirror, tracker.mirror_obs) and the starting-pose option
(--start_share, --start_joints, --start_cmd; sim.Sim._fresh) (run: python tests/test_symmetry.py, or pytest). CPU only.

  map       on a made-up robot of two roll joints, two pitch joints (left_ and right_) and a yaw joint on the centerline: the
            partners and signs follow the joint axes (roll and yaw flip, pitch keeps its sign); a default pose that is not its
            own mirror image is refused
  obs       the tracker's observation of the mirror image of a robot state (M = diag(1, -1, 1): base rotation M R M, angular
            velocity -M w, joints swapped with their signs, heading negated) equals mirror_obs of the observation of the state
  loss      on test_finetune's stand-in: without the option no term is computed; with weight W the loss gains W x the survival-
            weighted mean of |a_m - mirror(a)|^2, rebuilt from the recorded calls: a on the history held constant, a_m on its
            hand-mirrored copy and the goal of the command (vx, -vy, -wz); policy updates with a large weight lower the term
  start     the real Sim (bundled G1, plane): about `start_share` of the fresh envs start with the given joints at the default
            position + U[lo, hi] and the command in the given ranges; every other value, and every other env, is drawn as
            without the option
"""

import dataclasses
import json
import os
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import test_finetune as tf  # noqa: E402
import test_tracker  # noqa: E402

from xsoma import task, tracker, train  # noqa: E402
from xsoma.rot import axis_angle_to_mat  # noqa: E402

# the made-up robot, in the simulator's order; the tracker lists its joints in ORDER
DOF = ("left_roll", "left_pitch", "right_roll", "right_pitch", "waist_yaw")
AXES = {"left_roll": (1, 0, 0), "right_roll": (1, 0, 0), "left_pitch": (0, 1, 0), "right_pitch": (0, 1, 0), "waist_yaw": (0, 0, 1)}
Q0 = np.float32([0.1, 0.3, -0.1, 0.3, 0.0])  # its own mirror image: roll flips, pitch keeps its sign
SCALE = np.float32([0.2, 0.4, 0.2, 0.4, 0.3])
ORDER = test_tracker.ORDER
ROBOT = dict(tf.ROBOT, DOF=DOF, DEFAULT_Q=Q0, ACTION_SCALE=SCALE)
NAMES = [DOF[i] for i in ORDER]  # the tracker's order
PERM = np.asarray([NAMES.index(n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")) for n in NAMES])
SIGN = np.float32([1.0 if "pitch" in n else -1.0 for n in NAMES])
W_SIGN, G_SIGN, REL_SIGN = np.float32([-1, 1, -1]), np.float32([1, -1, 1]), np.float32([1, -1, -1, 1, 1, -1])  # derived by hand


def make(d, q0=Q0):
    """test_tracker's random tracker for the made-up robot, its meta.json with the joint axes (kin_joint_idx, kin_axes)."""
    with mock.patch.object(test_tracker, "ROBOT", ROBOT):
        p = test_tracker.make(d, names=[n + "_joint" for n in NAMES], q0=Q0[ORDER], scale=SCALE[ORDER])
    path = os.path.join(d, "meta.json")
    with open(path) as f:
        meta = json.load(f)
    meta |= {"default_joint_pos": q0[ORDER].tolist(), "kin_joint_idx": [-1] + list(range(len(NAMES))), "kin_axes": [[0, 0, 0]] + [list(AXES[n]) for n in NAMES]}
    with open(path, "w") as f:
        json.dump(meta, f)
    return p, path


def mirror_hist(o):
    """mirror_obs by hand: joint blocks swapped with SIGN, base blocks with the hand-derived signs."""
    J = len(NAMES)
    jm = lambda x: x[..., PERM] * SIGN  # noqa: E731
    return np.concatenate((jm(o[..., :J]), jm(o[..., J : 2 * J]), o[..., 2 * J : 2 * J + 3] * W_SIGN, o[..., 2 * J + 3 : 2 * J + 6] * G_SIGN,
                           jm(o[..., 2 * J + 6 : 3 * J + 6]), o[..., 3 * J + 6 :] * REL_SIGN), -1)  # fmt: skip


def check_map(d):
    _, path = make(d)
    perm, sign = tracker.mirror(path)
    assert perm.tolist() == PERM.tolist() and sign.tolist() == SIGN.tolist(), (perm, sign)
    try:
        make(d, q0=np.float32([0.1, 0.3, 0.1, 0.3, 0.0]))
        tracker.mirror(path)
        raise RuntimeError("an asymmetric default pose was accepted")
    except AssertionError:
        pass
    print(f"map: partners {perm.tolist()}, signs {sign.tolist()} (roll and yaw flip, pitch keeps), an asymmetric default pose refused")


def check_obs(d):
    p, path = make(d)
    m = tracker.mirror(path)
    rng, N = np.random.default_rng(1), 16
    to_t, to_s = np.asarray(p["to_tracker"]), np.asarray(p["to_sim"])
    q, qd, a = (rng.normal(size=(N, 5)).astype(np.float32) for _ in range(3))  # q, qd in the simulator's order, a in the tracker's
    ax = rng.normal(size=(N, 3))
    R = np.asarray(axis_angle_to_mat(jnp.asarray(ax / np.linalg.norm(ax, axis=-1, keepdims=True), jnp.float32), jnp.asarray(rng.uniform(0, 3, N), jnp.float32)))
    w, heading = rng.normal(size=(N, 3)).astype(np.float32), rng.uniform(-3, 3, N).astype(np.float32)
    M = np.float32([1, -1, 1])
    jt = lambda x: x[:, to_t][:, PERM] * SIGN  # noqa: E731  # simulator order -> mirrored, tracker order
    for h in (heading, None):
        o = tracker.observation(p, q, qd, R, w, a, h)
        o_m = tracker.observation(p, jt(q)[:, to_s], jt(qd)[:, to_s], R * M[:, None] * M, -w * M, a[:, PERM] * SIGN, None if h is None else -h)
        assert np.allclose(o_m, tracker.mirror_obs(m, o), atol=1e-5) and np.allclose(mirror_hist(np.asarray(o)), o_m, atol=1e-5)
    print("obs: the observation of the mirrored state = mirror_obs of the observation, with a held heading and with the base's own yaw")


def check_loss(d):
    _, path = make(d)
    xs = tf.build(init_tracker=os.path.join(d, "ckpt.npz"), tracker_meta=path, gait_clock=0, pi_gn_gate=0.0)
    carry, start, _ = tf.collect(xs)
    args = (carry["pi"], carry["wm"], carry["wm_b"], start, jax.random.split(jax.random.PRNGKey(3), xs.cfg.horizon), carry["env"].scale, carry["env"].world.heights)
    plain, st = xs.imagine(*args)
    assert "sym" not in st and xs.mirror is None
    W = 0.5
    with mock.patch.object(train, "Sim", tf.FakeSim):
        xw = train.XSoma(dataclasses.replace(xs.cfg, symmetry=W))
    xw.init(jax.random.PRNGKey(0))
    calls, ends, act, term = [], [], tracker.act, train.XSoma.termination

    def rec_act(p, hist, g, **kw):
        calls.append((np.asarray(hist), np.asarray(g), np.asarray(act(p, hist, g, **kw))))
        return calls[-1][2]

    def rec_end(self, clearance):
        ends.append(np.asarray(term(self, clearance)))
        return ends[-1]

    with jax.disable_jit(), mock.patch.object(tracker, "act", rec_act), mock.patch.object(train.XSoma, "termination", rec_end):
        loss, st = xw.imagine(*args)
    # per step three calls: the policy's action, then the policy's on the history held constant and on its mirror image
    fix, mir = calls[1::3], calls[2::3]
    for (h, g, _), (hm, gm, _) in zip(fix, mir):
        cmd = g[:, tracker.CMD]  # the first goal frame's v x, v y, w z
        assert np.allclose(hm, mirror_hist(h), atol=1e-6) and np.allclose(gm, tracker.goal({**xw.frozen, **carry["pi"]}, cmd * np.float32([1, -1, -1])), atol=1e-6)
    a, a_m = np.stack([c[2] for c in fix]), np.stack([c[2] for c in mir])
    alive = np.cumprod(1 - np.stack(ends), 0)
    w = np.concatenate((np.ones_like(alive[:1]), alive[:-1]))
    sym = (w * ((a_m - a[..., PERM] * SIGN) ** 2).sum(-1)).mean()
    assert sym > 0 and np.isclose(float(st["sym"]), sym, rtol=1e-4) and np.isclose(float(loss), float(plain) + W * sym, rtol=1e-4), (float(st["sym"]), sym)
    with mock.patch.object(train, "Sim", tf.FakeSim):
        xb = train.XSoma(dataclasses.replace(xs.cfg, symmetry=1e3, pi_lr=3e-4))
    xb.init(jax.random.PRNGKey(0))  # sets the optimizer's masks
    upd, sub = jax.jit(xb.policy_update), carry
    for _ in range(8):  # the key of `args`: the same imagined noise as the measurement's
        sub = {**sub, **upd(sub, start, jax.random.PRNGKey(3))[0]}
    after = float(xb.imagine(sub["pi"], *args[1:])[1]["sym"])
    before = float(xb.imagine(*args)[1]["sym"])
    assert after < 0.5 * before, (before, after)
    print(f"loss: no term without the option; = plain + {W} x {sym:.4f}, the survival-weighted mean of |a_m - mirror(a)|^2 of the recorded calls "
          f"on hand-mirrored inputs; 8 updates at weight 1e3 lower it from {before:.4f} to {after:.4f}")


def test_symmetry():
    with mock.patch.multiple(task, **ROBOT), tempfile.TemporaryDirectory() as d:
        check_map(d)
        check_obs(d)
        check_loss(d)


def test_start():
    from xsoma.sim import Sim, SimCfg

    N, share = 1000, 0.4
    names = ("left_hip_roll", "left_knee", "right_hip_pitch")
    spec = tuple((i, lo, hi) for i, (lo, hi) in ((task.DOF.index(next(d for d in task.DOF if n in d)), r) for n, r in zip(names, ((-0.1, -0.05), (0.1, 0.2), (0.0, 0.15)))))
    cmd_r = ((0.0, 0.0), (-0.3, 0.3), (0.0, 0.0))
    plain = Sim(SimCfg(num_envs=N, terrain="plane"))
    sim = Sim(SimCfg(num_envs=N, terrain="plane", start_share=share, start_joints=spec, start_cmd=cmd_r))
    origin, key = jnp.zeros((N, 3)), jax.random.PRNGKey(5)
    a, b = ([np.asarray(x) for x in s._fresh(key, origin)] for s in (plain, sim))
    on = (a[0] != b[0]).any(-1)
    i = np.asarray([s[0] for s in spec])
    dq = b[0][on][:, 7 + i] - np.broadcast_to(task.DEFAULT_Q, (task.J,))[i]
    assert abs(on.mean() - share) < 0.05 and (dq >= [s[1] - 1e-6 for s in spec]).all() and (dq <= [s[2] + 1e-6 for s in spec]).all()
    rest = np.setdiff1d(np.arange(b[0].shape[1]), 7 + i)
    assert (a[0][:, rest] == b[0][:, rest]).all() and (a[0][~on] == b[0][~on]).all(), "values outside the option changed"
    c = b[5][on]
    assert (c[:, [0, 2]] == 0).all() and (np.abs(c[:, 1]) <= 0.3).all() and c[:, 1].std() > 0.1 and (a[5][~on] == b[5][~on]).all()
    assert all((x == y).all() for x, y in zip(a[1:5] + a[6:], b[1:5] + b[6:])), "other draws changed"
    print(f"start: {on.mean():.2f} of the fresh envs (share {share}) start with {names} at the default + U[lo, hi] and a sideways command up to 0.3; the rest as drawn without it")


if __name__ == "__main__":
    assert not os.environ.get("XSOMA_ROBOT_DIR"), "run this test without XSOMA_ROBOT_DIR"
    test_symmetry()
    test_start()
    print("\nall checks passed")
