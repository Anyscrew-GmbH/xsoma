"""The tracker policy's interface (xsoma/tracker.py) on a made-up robot, without a simulator (run: python tests/test_tracker.py,
or pytest): a robot of 5 joints with a random default pose and random action scales, whose tracker lists the joints in another
order, and a tracker checkpoint with random weights.

  load         the permutation between the two joint orders and its inverse; a checkpoint without goal masking, another joint
               set, another default pose or action scale are refused
  goal         K frames of the neutral values with the command at v x, v y, w z, then the flag
  observation  every block where the tracker expects it; gravity and the relative rotation for an upright robot
  noise        none without a key or at scale 0; at scale 1 every block moves within its training scale (tracker.NOISE)
  heading      without a commanded heading ("rate") the relative rotation is the base's tilt alone, whatever its yaw
  step         the heading starts at the base's yaw and integrates the commanded yaw rate; the history fills at an episode's
               first step and then shifts; a reset starts both anew; the action comes back in the simulator's joint order
"""

import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from xsoma import nets, task, tracker  # noqa: E402

J, K, H, ZDIM = 5, 3, 4, 6
RNG = np.random.default_rng(0)
DOF = ("hip", "knee", "ankle", "waist", "elbow")  # the simulator's order
ORDER = [3, 0, 4, 2, 1]  # the tracker's joint i is the simulator's joint ORDER[i]
Q0, SCALE = RNG.uniform(-0.5, 0.5, J).astype(np.float32), RNG.uniform(0.05, 0.5, J).astype(np.float32)  # in the simulator's order
ROBOT = dict(DOF=DOF, J=J, DEFAULT_Q=Q0, ACTION_SCALE=SCALE)


def make(d, share=0.5, names=None, q0=None, scale=None):
    """Write a tracker checkpoint with random weights and the tracker's robot description to the directory d; load them."""
    enc = nets.mlp_init(jax.random.PRNGKey(0), [K * (15 + 2 * J) + 1, 16, ZDIM])
    dec = nets.mlp_init(jax.random.PRNGKey(1), [ZDIM + H * (3 * J + 12), 16, J])
    flat = {f"{n}.{i}.{k}": np.asarray(v) for n, net in (("enc", enc), ("dec", dec)) for i, layer in enumerate(net) for k, v in layer.items()}
    for n, dim in (("obs_norm", 3 * J + 12), ("goal_norm", K * (15 + 2 * J) + 1)):
        flat |= {f"{n}.mean": RNG.normal(size=dim).astype(np.float32), f"{n}.var": RNG.uniform(0.5, 2, dim).astype(np.float32), f"{n}.count": np.float32(1e6)}
    np.savez(os.path.join(d, "ckpt.npz"), **flat, meta=json.dumps({"args": {"goal_mask_share": share, "goal_frames": K, "hist": H}}))
    meta = {"joint_names": names or [DOF[i] + "_joint" for i in ORDER], "default_joint_pos": (Q0[ORDER] if q0 is None else q0).tolist(),
            "action_scale": (SCALE[ORDER] if scale is None else scale).tolist()}  # fmt: skip
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    with mock.patch.multiple(task, **ROBOT):
        return tracker.load(os.path.join(d, "ckpt.npz"), os.path.join(d, "meta.json"))


def yaw_quat(yaw):
    return np.stack((np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)), -1)


def test_tracker():
    with tempfile.TemporaryDirectory() as d:
        p = make(d)
        refused = 0
        for bad in (dict(share=0.0), dict(names=[n + "_joint" for n in ("hip", "knee", "ankle", "waist", "wrist")]), dict(q0=Q0), dict(scale=SCALE)):
            try:
                make(d, **bad)
            except AssertionError:
                refused += 1
        assert refused == 4, refused
    N = 7
    to_t, to_s = np.asarray(p["to_tracker"]), np.asarray(p["to_sim"])
    x = RNG.normal(size=(N, J)).astype(np.float32)
    assert to_t.tolist() == ORDER and (x[:, to_t][:, to_s] == x).all() and (x[:, to_s][:, to_t] == x).all()
    assert np.allclose(p["q0"], Q0[to_t]) and np.allclose(p["alpha"], SCALE[to_t])
    print(f"load: the tracker's joints are the simulator's {to_t.tolist()}, the inverse restores them; {refused} unfit trackers refused")

    cmd = RNG.uniform(-1, 1, (N, 3)).astype(np.float32)
    g = np.asarray(tracker.goal(p, cmd))
    assert g.shape == (N, K * (15 + 2 * J) + 1) and (g[:, -1] == 1).all()
    fr = g[:, :-1].reshape(N, K, 15 + 2 * J)
    neutral = np.concatenate(([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], Q0[to_t], np.zeros(J))).astype(np.float32)
    rest = np.setdiff1d(np.arange(15 + 2 * J), (9, 10, 14))
    assert (fr[..., (9, 10, 14)] == cmd[:, None]).all() and (fr[..., rest] == neutral[rest]).all()
    print(f"goal: {K} frames of {15 + 2 * J}, the command at entries 9, 10 and 14, neutral elsewhere, flag 1")

    q, qd, w, a_prev = (RNG.normal(size=(N, n)).astype(np.float32) for n in (J, J, 3, J))
    yaw = RNG.uniform(-3, 3, N).astype(np.float32)
    R = np.asarray(tracker.quat_to_mat(yaw_quat(yaw)))
    o = np.asarray(tracker.observation(p, q, qd, R, w, a_prev, yaw))
    blocks = np.split(o, np.cumsum([J, J, 3, 3, J]), -1)
    assert o.shape == (N, 3 * J + 12)
    for name, got, want in zip(("q - q0", "qd", "angular velocity", "gravity", "previous action", "relative rotation"), blocks,
                               ((q - Q0)[:, to_t], qd[:, to_t], w, [0, 0, -1], a_prev, [1, 0, 0, 1, 0, 0])):  # fmt: skip
        assert np.allclose(got, want, atol=1e-6), name
    quarter = np.asarray(tracker.observation(p, q, qd, R, w, a_prev, yaw + np.pi / 2))[:, -6:]  # the heading a quarter turn to the left
    assert np.allclose(quarter, [0, -1, 1, 0, 0, 0], atol=1e-6)
    pitch = np.asarray([[np.cos(0.3), 0, np.sin(0.3)], [0, 1, 0], [-np.sin(0.3), 0, np.cos(0.3)]], np.float32)  # nose down by 0.3 rad
    assert np.allclose(tracker.observation(p, q[0], qd[0], pitch, w[0], a_prev[0], np.float32(0))[2 * J + 3 : 2 * J + 6], [np.sin(0.3), 0, -np.cos(0.3)], atol=1e-6)
    print("observation: q - q0, qd, angular velocity, gravity, previous action and relative rotation where the tracker expects them")

    key = jax.random.PRNGKey(2)
    dev = [np.asarray(tracker.observation(p, q, qd, R, w, a_prev, yaw, key, n)) - o for n in (0.0, 1.0, 2.0)]
    scale = np.repeat(np.asarray(tracker.NOISE, np.float32), (J, J, 3, 3, J, 6))  # per entry
    assert (dev[0] == 0).all() and np.allclose(dev[2], 2 * dev[1], atol=1e-6) and np.allclose(dev[1], (2 * np.asarray(jax.random.uniform(key, o.shape)) - 1) * scale, atol=1e-6)
    top = [float(np.abs(b).max()) for b in np.split(dev[1], np.cumsum([J, J, 3, 3, J]), -1)]
    assert all(0.5 * n <= t <= 1.0001 * n for t, n in zip(top, tracker.NOISE)), top  # the previous action's scale is 0
    print(f"noise: scale 0 leaves the observation as it was; scale 1 moves the blocks by at most {[round(t, 3) for t in top]}, within {tracker.NOISE}")

    roll = np.asarray([[1, 0, 0], [0, np.cos(0.2), np.sin(0.2)], [0, -np.sin(0.2), np.cos(0.2)]], np.float32)
    tilted = R @ pitch @ roll  # the base's yaw, then a tilt
    rate, hold = (np.asarray(tracker.observation(p, q, qd, tilted, w, a_prev, h)) for h in (None, yaw + 0.5))
    assert np.allclose(rate[:, -6:], (pitch @ roll).T[:, :2].reshape(-1), atol=1e-6) and (rate[:, :-6] == hold[:, :-6]).all()
    assert np.abs(hold[:, -6:] - rate[:, -6:]).max() > 0.1 and np.allclose(tracker.observation(p, q, qd, R, w, a_prev, None)[:, -6:], [1, 0, 0, 1, 0, 0], atol=1e-6)
    print(f"heading: without a commanded heading the relative rotation is the transposed tilt for {N} yaws, the identity when upright; the rest is unchanged")

    def state(t, yaw, a_last):  # the fields of sim.State that the tracker reads
        qpos = np.concatenate((np.zeros((N, 3)), yaw_quat(yaw), q), -1).astype(np.float32)
        return SimpleNamespace(t=np.asarray(t, np.int32), data=SimpleNamespace(qpos=qpos, qvel=np.concatenate((np.zeros((N, 3)), w, qd), -1)), a_last=a_last, cmd=cmd)

    mem = tracker.memory(p, N)
    a, mem = tracker.step(p, mem, state(np.full(N, 1), yaw, np.zeros((N, J), np.float32)))  # as after Sim.init: counters at 1
    first = np.asarray(tracker.observation(p, q, qd, R, w, np.zeros((N, J), np.float32), yaw))
    assert a.shape == (N, J) and np.isfinite(a).all() and np.allclose(mem["hist"], first[:, None], atol=1e-6)
    assert np.allclose(a, np.asarray(tracker.act(p, mem["hist"], tracker.goal(p, cmd)))[:, to_s])
    assert np.allclose(mem["heading"], yaw + task.DT * cmd[:, 2], atol=1e-6)
    a2, mem2 = tracker.step(p, mem, state(np.full(N, 2), yaw + 1.0, np.asarray(a)))
    second = np.asarray(tracker.observation(p, q, qd, np.asarray(tracker.quat_to_mat(yaw_quat(yaw + 1.0))), w, np.asarray(a)[:, to_t], yaw + task.DT * cmd[:, 2]))
    assert (np.asarray(mem2["hist"][:, :-1]) == np.asarray(mem["hist"][:, 1:])).all() and np.allclose(mem2["hist"][:, -1], second, atol=1e-6)
    assert np.allclose(mem2["heading"], yaw + 2 * task.DT * cmd[:, 2], atol=1e-6)
    t3 = np.where(np.arange(N) < 3, 0, 3)  # the first three envs were reset
    _, mem3 = tracker.step(p, mem2, state(t3, yaw - 1.0, np.zeros((N, J), np.float32)))
    h3 = np.asarray(mem3["hist"])
    assert (h3[:3] == h3[:3, -1:]).all() and (h3[3:, :-1] == np.asarray(mem2["hist"])[3:, 1:]).all()
    assert np.allclose(mem3["heading"], np.where(np.arange(N) < 3, yaw - 1.0, yaw + 2 * task.DT * cmd[:, 2]) + task.DT * cmd[:, 2], atol=1e-6)
    # training's options in a step: the noisy observation at the base's own yaw enters the history; the heading integrates as before
    a_r, mem_r = tracker.step(p, mem, state(np.full(N, 2), yaw + 1.0, np.asarray(a)), False, key, 1.0)
    want = np.asarray(tracker.observation(p, q, qd, np.asarray(tracker.quat_to_mat(yaw_quat(yaw + 1.0))), w, np.asarray(a)[:, to_t], None, key, 1.0))
    assert (np.asarray(mem_r["hist"][:, -1]) == want).all() and np.abs(want[:, -6:] - [1, 0, 0, 1, 0, 0]).max() <= 1.0001 * tracker.NOISE[5]
    assert np.abs(want - second).max() > 0.1 and (np.asarray(mem_r["heading"]) == np.asarray(mem2["heading"])).all()
    assert np.allclose(a_r, np.asarray(tracker.act(p, mem_r["hist"], tracker.goal(p, cmd)))[:, to_s])
    print(f"step: heading from the base's yaw plus {task.DT} s of the commanded yaw rate per step; a history of {H} filled, shifted and reset; "
          "with noise and without a commanded heading the action follows that observation")  # fmt: skip


if __name__ == "__main__":
    test_tracker()
    print("\nall checks passed")
