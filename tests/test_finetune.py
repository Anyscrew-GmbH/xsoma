"""Fine-tuning a tracker policy (train.py --init_tracker) on the made-up robot of test_tracker.py, with a stand-in for the
simulator: random states instead of physics, episodes of at most 7 steps (run: python tests/test_finetune.py, or pytest). The
replay buffer, the start sampling, imagination, the policy update and the checkpoints are train.py's own.

  starts      the observation history of every sampled start, rebuilt from the replay buffer, is the one the tracker held at
              that step of the collection, also right after an episode's start and after the buffer wrapped around
  imagine     imagination's first step calls the policy with that history and the command's goal, and the world model with the
              policy's action in the simulator's joint order (the previous action in a delayed env); its second step sees the
              predicted state, the first action, the heading advanced by the commanded yaw rate and the shifted history
  update      training's compiled programs run (collection, world-model fit, policy update); the anchor is zero at the first
              update and positive after it; only the encoder and the decoder change
  options     without --anchor no reference policy is kept; --gait_clock 0 sets the weight of feet_phase to zero in
              imagination's reward only, and the loss then ignores the gait clock
  checkpoint  a saved checkpoint loads as a tracker (the averaged and the live policy, same actions), lists as a run's
              checkpoint for evaluate.py and is accepted by --init_tracker again
  heading     all of the above once more with --tracker_heading rate: no history then holds a yaw error
  evaluation  evaluate.py on the run directory takes the heading mode from the run's config.json unless one is given, and writes
              the feet's crossing; on a hand-made sequence of foot positions it counts the crossed steps before the fall
  noise       with --tracker_noise (default 1) the history holds the noisy observation, within the tracker's scales; a rebuilt
              history differs from the held one by two draws of that noise only; the anchor's reference sees the same inputs
  feet        task.feet_cross is 0 for a normal stance and positive for close or swapped feet, at any yaw; with --feet_cross
              imagination's loss gains weight x dt x penalty scale x the discounted term, without it the term is not computed
  default     XSoma's own policy trains on the same stand-in (the path the options leave untouched)
"""

import contextlib
import dataclasses
import io
import json
import os
import sys
import tempfile
from types import SimpleNamespace
from typing import NamedTuple
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from test_tracker import ROBOT, H, J, make, yaw_quat  # noqa: E402

from xsoma import evaluate, nets, task, tracker, train  # noqa: E402
from xsoma.rot import mv3, quat_to_mat, tr3  # noqa: E402
from xsoma.sim import SimCfg, State, Transition, View, unflatten  # noqa: E402

ROBOT = dict(ROBOT, ACTOR_DIM=3 * J + 13, CRITIC_DIM=3 * J + 16, POSE_W=np.ones(J, np.float32))
N, T, STEPS, EP = 6, 24, 40, 7  # envs, replay-buffer length, collected steps, the stand-in's time-out
SMALL = dict(num_envs=N, buffer_steps=T, warmup_steps=H, wm_only_iters=0, wm_hidden="16,16", wm_window=4, wm_batch=8, wm_batches=1, horizon=5,
             pi_batch=16, pi_updates=1, imag_unroll=1, wm_eval=0, jax_cache="")  # fmt: skip


class Data(NamedTuple):
    qpos: jax.Array
    qvel: jax.Array


class World(NamedTuple):
    heights: jax.Array


class FakeSim:
    """What train.XSoma uses of sim.Sim, without physics: the base drifts at random, the joints move towards their targets, an
    episode ends at random or after EP steps, and two made-up points under the pelvis stand for the feet."""

    def __init__(self, cfg):
        self.cfg, self.N = cfg, cfg.num_envs
        self.reward_cfg = task.FAST_SAC_REWARD
        self.sole_radius = np.zeros(2)

    def view_state(self, S, heights):
        p, R, v, w, q, qd = S
        feet = p[..., None, :] + mv3(R[..., None, :, :], jnp.asarray([[0, 0.1, -0.7], [0, -0.1, -0.7]]) + 0.1 * jnp.stack((q[..., :3], q[..., 2:]), -2))
        rb = task.Robot(p, R, v, mv3(tr3(R), w), q, qd, feet, feet, jnp.broadcast_to(R[..., None, :, :], feet.shape + (3,)))
        h = p[..., 2] - heights.mean()
        return View(rb, feet[..., 2], h[..., None] - 0.3, feet[..., 2], h)

    def view(self, qpos, qvel, heights):
        return self.view_state(unflatten(qpos, qvel), heights)

    def context(self, s):
        return s.kp

    def _fresh(self, key):
        k = jax.random.split(key, 4)
        yaw = jax.random.uniform(k[0], (self.N,), minval=-3.0, maxval=3.0)
        quat = jnp.stack((jnp.cos(yaw / 2), 0 * yaw, 0 * yaw, jnp.sin(yaw / 2)), -1)
        qpos = jnp.concatenate((0.1 * jax.random.normal(k[1], (self.N, 2)), jnp.full((self.N, 1), 0.8), quat, task.DEFAULT_Q + 0.1 * jax.random.normal(k[2], (self.N, J))), -1)
        return qpos, 0.3 * jax.random.normal(k[3], (self.N, 6 + J))

    def init(self, key):
        n = self.N
        k = jax.random.split(key, 5)
        cmd, freq = task.sample_commands(k[1], n)
        zi = jnp.zeros((n,), jnp.int32)
        # the episode counters start at 1, as after Sim.init's zero-action step; every other env delays its actions
        return State(Data(*self._fresh(k[0])), World(jnp.zeros((1, 1))), jnp.zeros((n, task.ACTOR_DIM)), jnp.zeros((n, task.CRITIC_DIM)), jnp.zeros((n, J)),
                     jax.random.uniform(k[2], (n, J), minval=0.9, maxval=1.1), jnp.ones((n, J)), jnp.arange(n) % 2, cmd, freq,
                     task.sample_phase_offsets(k[3], n), jnp.zeros((n, 2)), zi + 1, zi, zi + 0, jnp.float32(0.5), jnp.float32(0.0), jnp.zeros((n, len(task.TERMS))), k[4])  # fmt: skip

    def step(self, s, action):
        n = self.N
        key, kq, kv, kt, kr, kc, ko = jax.random.split(s.key, 7)
        a = action.astype(jnp.float32)
        target = jnp.clip(jnp.where(s.delay[:, None] == 0, a, s.a_last), -task.ACTION_CLIP, task.ACTION_CLIP)
        qpos, qvel = s.data
        q1 = qpos[:, 7:] + 0.3 * (train.target_q(target) - qpos[:, 7:])
        qpos1 = jnp.concatenate((qpos[:, :7] + 0.02 * jax.random.normal(kq, (n, 7)), q1), -1)
        qvel1 = 0.9 * qvel + 0.1 * jax.random.normal(kv, qvel.shape)
        t = s.t + 1
        terminated = jax.random.uniform(kt, (n,)) < 0.08
        done = terminated | (t > EP)
        v = self.view_state(unflatten(qpos1, qvel1), s.world.heights)
        parts = task.weighted(task.reward_terms(v.robot, v.foot_h, a, s.a_last, s.cmd, s.phase), self.reward_cfg, s.scale)
        parts = jnp.stack([parts[k] for k in task.TERMS], -1)
        on = done[:, None]
        fq, fv = self._fresh(kr)
        data = Data(jnp.where(on, fq, qpos1), jnp.where(on, fv, qvel1))
        a_last, t1, cmd = jnp.where(on, 0.0, a), jnp.where(done, 0, t), jnp.where(on, task.sample_commands(kc, n)[0], s.cmd)
        phase = task.gait_phase(t1, s.freq, s.offset, cmd)
        rb = task.Robot(data.qpos[:, :3], quat_to_mat(data.qpos[:, 3:7]), data.qvel[:, :3], data.qvel[:, 3:6], data.qpos[:, 7:], data.qvel[:, 6:], None, None, None)
        obs, critic = task.observations(rb, a_last, cmd, phase, ko)
        s1 = s._replace(data=data, obs=obs, critic=critic, a_last=a_last, t=t1, cmd=cmd, phase=phase, key=key)
        stats = {"done": done.sum().astype(jnp.float32), "reward": parts.sum(-1).mean()}
        return s1, Transition(qpos, qvel, target, qpos1, qvel1, parts.sum(-1), terminated, done & ~terminated, jnp.zeros((n,), bool), parts, critic, v.clearance, stats)


def build(**options):
    """train.XSoma on the stand-in simulator, with small sizes."""
    with mock.patch.object(train, "Sim", FakeSim):
        return train.XSoma(dataclasses.replace(train.Config(), **SMALL, **options))


def leaves(tree):
    return np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(tree)])


def collect(xs):
    """STEPS collection steps from a new carry (STEPS > T, so the buffer wraps around), then a batch of starts and, for each, the
    history that the tracker held at that step of the collection."""
    carry = xs.init(jax.random.PRNGKey(0))
    step, held = jax.jit(xs.collect_step), {}
    for k in jax.random.split(jax.random.PRNGKey(1), STEPS):
        qpos = np.asarray(carry["env"].data.qpos)
        carry, _ = step(carry, k)
        held.update({qpos[e].tobytes(): np.asarray(carry["mem"]["hist"][e]) for e in range(N)})
    start = jax.jit(xs.sample_starts)(carry["buf"], jax.random.PRNGKey(2))
    return carry, start, np.stack([held[np.asarray(q).tobytes()] for q in start["s_qpos"]])


def check_finetune(d, heading):
    ckpt, meta = os.path.join(d, "ckpt.npz"), os.path.join(d, "meta.json")
    p = make(d)
    # no gradient gate: the anchor's gradient first appears at the second update, where the gate would skip this toy's step
    xs = build(init_tracker=ckpt, tracker_meta=meta, anchor=0.5, gait_clock=0, pi_gn_gate=0.0, tracker_noise=0.0, tracker_heading=heading)
    carry, start, held = collect(xs)
    assert set(carry["pi"]) == {"enc", "dec"} and "heading" in carry["buf"] and carry["mem"]["hist"].shape == (N, H, 3 * J + 12)
    hist = np.asarray(start["hist"])
    err = np.abs(hist - held).max()
    filled = int((np.abs(hist[:, 0] - hist[:, 1]).max(-1) == 0).sum())  # histories that reach back to their episode's first step
    yaw_err = np.abs(hist[..., -5]).max()  # entry (0, 1) of the relative rotation: zero when the heading is the base's yaw
    assert err < 1e-5 and 0 < filled < hist.shape[0] and (yaw_err > 1e-2 if heading == "hold" else yaw_err < 1e-6), (err, filled, yaw_err)
    print(f"starts: {hist.shape[0]} histories of {H} rebuilt from a buffer of {T} steps after {STEPS}: max |rebuilt - held| = {err:.1e}; {filled} reach their "
          f"episode's start; largest yaw entry of the relative rotation {yaw_err:.1e}")  # fmt: skip

    # training's own programs (the carry is donated to each): a collection step, a world-model fit and two policy updates
    pi0, frozen = jax.tree.map(np.asarray, carry["pi"]), jax.tree.map(np.asarray, xs.frozen)
    progs = xs.compile_programs(carry)
    carry, _ = progs["collect"](carry)
    carry, _ = progs["wm"](carry)
    carry, st1 = progs["pi"](carry)
    moved = {net: float(np.abs(leaves(carry["pi"][net]) - leaves(pi0[net])).max()) for net in ("enc", "dec")}
    carry, st2 = progs["pi"](carry)
    assert all(np.isfinite(float(v)) for st in (st1, st2) for v in st.values()) and float(st1["gn"]) > 0 and float(st1["gn_gated"]) == float(st2["gn_gated"]) == 0
    assert float(st1["anchor"]) == 0 and float(st2["anchor"]) > 0 and min(moved.values()) > 0, (st1, st2, moved)
    assert (leaves(carry["pi0"]) == leaves(pi0)).all() and (leaves(xs.frozen) == leaves(frozen)).all()
    print(f"update: imagined return {float(st1['G']):.3f}, gradient norm {float(st1['gn']):.3f}; anchor {float(st1['anchor']):.1e} at the initial "
          f"tracker, {float(st2['anchor']):.1e} after one update (encoder moved by {moved['enc']:.1e}, decoder by {moved['dec']:.1e})")  # fmt: skip

    # the options: the loss with and without the anchor, and with another gait clock
    args = (carry["pi"], carry["wm"], carry["wm_b"], start, jax.random.split(jax.random.PRNGKey(3), xs.cfg.horizon), carry["env"].scale, carry["env"].world.heights)
    loss, st = xs.imagine(*args, pi0=carry["pi0"])
    plain, st_plain = xs.imagine(*args)
    assert "anchor" not in st_plain and float(st["anchor"]) > 0 and np.isclose(float(loss), float(plain) + 0.5 * float(st["anchor"]), rtol=1e-6)
    clock = {"phase": start["phase"] + 1.0, "freq": 2 * start["freq"], "offset": start["offset"] - 1.0}
    assert float(xs.imagine(*args[:3], {**start, **clock}, *args[4:])[0]) == float(plain)
    weights, sim_w = dataclasses.asdict(xs.reward_cfg), dataclasses.asdict(xs.sim.reward_cfg)
    assert weights == {**sim_w, "feet_phase": 0.0} and sim_w["feet_phase"] == 5.0
    off = build(init_tracker=ckpt, tracker_meta=meta)
    c_off = off.init(jax.random.PRNGKey(0))
    assert "pi0" not in c_off and off.reward_cfg is off.sim.reward_cfg
    assert float(off.imagine(*args[:3], {**start, **clock}, *args[4:])[0]) != float(off.imagine(*args)[0])
    print(f"options: loss {float(loss):.4f} = {float(plain):.4f} + 0.5 x anchor {float(st['anchor']):.2e}; --gait_clock 0: feet_phase weighs 0 in "
          "imagination and the loss ignores the clock; defaults: no reference policy, the simulator's reward")  # fmt: skip

    # imagination's first two steps, run eagerly, with what the policy and the world model are called with
    seen, moved, act, wm_step = [], [], tracker.act, nets.wm_step

    def rec_act(p, hist, g, **kw):
        seen.append((hist, g))
        return act(p, hist, g, **kw)

    def rec_wm(wm, wm_b, S, h, soles, tgt, *rest, **kw):
        moved.append((tgt, wm_step(wm, wm_b, S, h, soles, tgt, *rest, **kw)))
        return moved[-1][1]

    with jax.disable_jit(), mock.patch.object(tracker, "act", rec_act), mock.patch.object(nets, "wm_step", rec_wm):
        xs.imagine(*args)
    pf = {**xs.frozen, **carry["pi"]}
    (h0, g0), (h1, _), (tgt, S1) = seen[0], seen[1], moved[0]
    a0 = tracker.act(pf, h0, g0, out_bf16=True)  # as imagination computes it (mm_bf16 "imag")
    applied = np.where(np.asarray(start["delay"])[:, None] == 0, np.asarray(a0)[:, np.asarray(pf["to_sim"])], np.asarray(start["a_last"]))
    o1 = tracker.observation(pf, S1[4], S1[5], S1[1], mv3(tr3(S1[1]), S1[3]), a0, start["heading"] + task.DT * start["cmd"][:, 2] if heading == "hold" else None)
    errs = [float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in (
        (h0, held), (g0, tracker.goal(pf, start["cmd"])),
        ((np.asarray(tgt) - task.DEFAULT_Q) / task.ACTION_SCALE, applied), (h1, jnp.concatenate((h0[:, 1:], o1[:, None]), 1)))]  # fmt: skip
    assert len(seen) == xs.cfg.horizon and max(errs) < 1e-4 and 0 < np.asarray(start["delay"]).sum() < len(applied), errs
    print(f"imagine: step 1 acts on the held history ({errs[0]:.1e}) and the command's goal ({errs[1]:.1e}) and applies the action in the simulator's "
          f"order ({errs[2]:.1e}); step 2 sees the predicted state, that action and {'the advanced heading' if heading == 'hold' else 'its own yaw'} ({errs[3]:.1e})")  # fmt: skip

    # checkpoints: both policies come back as trackers, the file is a run's checkpoint and an initial tracker again
    os.makedirs(os.path.join(d, "run", "ckpt"))
    path = os.path.join(d, "run", "ckpt", "0000")
    train.save_ckpt(path, carry["avg"], carry["pi"], {"it": 2, "train_s": 1.0, "wall_s": 2.0, "env_steps": 3.0}, xs.ckpt_extra)
    oh, g = jax.random.normal(jax.random.PRNGKey(4), (3, H, 3 * J + 12)), tracker.goal(p, jax.random.normal(jax.random.PRNGKey(5), (3, 3)))
    for which in ("avg", "live"):
        kind, _, q = evaluate.load(path + ".npz", which, meta)
        want = tracker.act({**xs.frozen, **carry["pi" if which == "live" else "avg"]}, oh, g)
        assert kind == "tracker" and jax.tree.structure(q) == jax.tree.structure(p) and (np.asarray(tracker.act(q, oh, g)) == np.asarray(want)).all()
    assert all(np.abs(np.asarray(tracker.act(q, oh, g)) - np.asarray(tracker.act({**xs.frozen, **other}, oh, g))).max() > 0 for other in (pi0, carry["avg"]))
    assert evaluate.checkpoints(os.path.join(d, "run")) == [(path + ".npz", 2.0, 3.0)]
    again = build(init_tracker=path + ".npz", tracker_meta=meta).init(jax.random.PRNGKey(0))
    assert (leaves(again["pi"]) == leaves(carry["avg"])).all() and json.loads(str(np.load(path + ".npz")["meta"])) == json.loads(str(np.load(ckpt)["meta"]))
    print("checkpoint: the averaged and the live policy load as trackers with the saved actions; evaluate.py lists the run's checkpoint; --init_tracker takes it")

    # evaluate.py on the run directory, with the stand-in simulator; the "hold" run has no config.json
    run, opts = os.path.join(d, "run"), ["--n", str(N), "--tracker_meta", meta]
    if heading == "rate":
        with open(os.path.join(run, "config.json"), "w") as f:
            json.dump({"tracker_heading": heading}, f)
    with mock.patch.object(evaluate, "make_sim", lambda n, *a, **k: FakeSim(SimCfg(num_envs=n))), contextlib.redirect_stdout(io.StringIO()) as out:
        evaluate.main([run] + opts)
        if heading == "rate":
            evaluate.main([run, "--tracker_heading", "hold", "--out", "hold"] + opts)
    want = [heading] + ["hold"] * (heading == "rate")
    recs = [json.loads(open(os.path.join(run, name + ".jsonl")).read()) for name in ("eval", "hold")[: len(want)]]
    assert [r["tracker_heading"] for r in recs] == want and len({r["feet_dy"] for r in recs}) == len(want) == out.getvalue().count("feet crossed")
    assert all(0 <= r["feet_crossed"] <= 1 and 0 < r["feet_dy"] < 0.5 for r in recs), recs
    print(f"evaluation: heading {heading} unless given, as the run's config.json " + ("says" if heading == "rate" else "is missing") + "; "
          + out.getvalue().splitlines()[0].split(" | ")[2])  # fmt: skip


class Seq(NamedTuple):
    t: jax.Array
    scale: jax.Array
    obs: jax.Array
    cmd: jax.Array
    world: World


class SeqSim:
    """What evaluate() uses of a simulator, with feet that follow a table: after step i of env e the left foot is DY[e, i] to the
    left of the right one and 0.3 m ahead of it, in the frame of a base turned by YAW[e]. Env 1 falls in its third step."""

    N = 3
    DY = np.asarray([[0.2, 0.2, -0.1, -0.1, 0.2, 0.2], [0.2, -0.05, 0.2, -0.3, -0.3, -0.3], [0.25] * 6], np.float32)
    YAW = np.asarray([0.0, 1.0, -2.5], np.float32)

    def init(self, key):
        return Seq(jnp.zeros((3,), jnp.int32), jnp.ones(()), jnp.zeros((3, 1)), jnp.zeros((3, 3)), World(jnp.zeros((1, 1))))

    def step(self, s, a):
        z = jnp.zeros((3,), bool)
        return s._replace(t=s.t + 1), SimpleNamespace(parts=jnp.zeros((3, len(task.TERMS))), qpos1=s.t, qvel1=None, terminated=z.at[1].set(s.t[1] == 2), truncated=z)

    def view(self, step, qvel, heights):
        R = quat_to_mat(jnp.asarray(yaw_quat(self.YAW)))
        left = mv3(R, jnp.stack((jnp.full((3,), 0.3), jnp.asarray(self.DY)[jnp.arange(3), step], jnp.zeros((3,))), -1))
        return SimpleNamespace(robot=task.Robot(None, R, jnp.zeros((3, 3)), jnp.zeros((3, 3)), None, None, None, jnp.stack((left, 0 * left), 1), None), soles=jnp.zeros((3, 1)))


def check_eval():
    _, _, fell, _, _, _, crossed, dy, *_ = evaluate.evaluate(SeqSim(), lambda obs: obs, jax.random.PRNGKey(0), steps=6)
    assert np.asarray(fell).tolist() == [False, True, False]
    assert np.allclose(crossed, [2 / 6, 1 / 3, 0]) and np.allclose(dy, [SeqSim.DY[0].mean(), SeqSim.DY[1, :3].mean(), 0.25], atol=1e-6)
    print(f"evaluation: feet crossed in {[round(float(x), 3) for x in crossed]} of the steps before a fall or the end, mean distance {[round(float(x), 3) for x in dy]} m")


def check_noise(d):
    ckpt, meta = os.path.join(d, "ckpt.npz"), os.path.join(d, "meta.json")
    make(d)
    quiet, noisy = build(init_tracker=ckpt, tracker_meta=meta, tracker_noise=0.0), build(init_tracker=ckpt, tracker_meta=meta)
    assert noisy.cfg.tracker_noise == 1.0
    blocks, bound = np.cumsum([J, J, 3, 3, J]), np.asarray(tracker.NOISE)
    top = lambda dev: np.asarray([np.abs(b).max() for b in np.split(dev, blocks, -1)])  # noqa: E731
    # one collection step of both from the same state: the history, filled at an episode's start, holds one noisy observation
    carry, key = [xs.init(jax.random.PRNGKey(0)) for xs in (quiet, noisy)][0], jax.random.PRNGKey(1)
    dev = np.asarray(noisy.collect_step(carry, key)[0]["mem"]["hist"]) - np.asarray(quiet.collect_step(carry, key)[0]["mem"]["hist"])
    assert (dev == dev[:, :1]).all() and (0.5 * bound <= top(dev)).all() and (top(dev) <= 1.0001 * bound + 1e-6).all(), top(dev)
    # a rebuilt history: the held one but for two draws of the noise; steps before the episode's start share the first step's
    carry, start, held = collect(noisy)
    hist = np.asarray(start["hist"])
    filled = int((np.abs(hist[:, 0] - hist[:, 1]).max(-1) == 0).sum())
    assert filled > 0 and (1.2 * bound <= top(hist - held)).all() and (top(hist - held) <= 2.0001 * bound + 1e-6).all(), top(hist - held)
    # imagination: another loss than without noise; the initial tracker as the anchor's reference acts on the same noisy inputs
    args = (carry["pi"], carry["wm"], carry["wm_b"], start, jax.random.split(jax.random.PRNGKey(3), noisy.cfg.horizon), carry["env"].scale, carry["env"].world.heights)
    loss, st = noisy.imagine(*args, pi0=carry["pi"])
    assert float(st["anchor"]) == 0 and float(loss) != float(quiet.imagine(*args)[0])
    print(f"noise: a collected observation moves by at most {[round(float(x), 3) for x in top(dev)]} per block, a rebuilt history differs from the held one by at "
          f"most {[round(float(x), 3) for x in top(hist - held)]} ({filled} of {hist.shape[0]} repeat their first step); in imagination the anchor stays 0 at the initial tracker")  # fmt: skip


def check_feet():
    R = quat_to_mat(jnp.asarray(yaw_quat(np.linspace(-3, 3, 7)), jnp.float32))
    # the feet at these sideways positions in the base's frame (left, right), 0.1 m apart in its forward direction, for 7 yaws
    robot = lambda left, right: task.Robot(None, R, None, None, None, None, None, 1.0 + mv3(R[:, None], jnp.asarray([[0.05, left, 0], [-0.05, right, 0]])), None)  # noqa: E731
    for name, (left, right), d, pen in (("stance", (0.12, -0.12), 0.24, 0.0), ("close", (0.03, -0.03), 0.06, 0.04), ("swapped", (-0.12, 0.12), -0.24, 0.34)):
        assert np.allclose(task.feet_dy(robot(left, right)), d, atol=1e-6) and np.allclose(task.feet_cross(robot(left, right)), pen, atol=1e-6), name
    assert task.FEET_CROSS_MARGIN == 0.10

    # imagination, run eagerly with a record of the term and of the terminations; a margin that the stand-in's feet are within
    plain, pen = build(), build(feet_cross=3.0)
    carry, step = plain.init(jax.random.PRNGKey(0)), jax.jit(plain.collect_step)
    for k in jax.random.split(jax.random.PRNGKey(1), 3 * H):
        carry, _ = step(carry, k)
    c, scale = pen.cfg, carry["env"].scale
    args = (carry["pi"], carry["wm"], carry["wm_b"], plain.sample_starts(carry["buf"], jax.random.PRNGKey(2)), jax.random.split(jax.random.PRNGKey(3), c.horizon), scale, carry["env"].world.heights)
    terms, ends, term, end = [], [], task.feet_cross, train.XSoma.termination

    def rec_term(robot):
        terms.append(term(robot))
        return terms[-1]

    def rec_end(self, clearance):
        ends.append(end(self, clearance))
        return ends[-1]

    with jax.disable_jit(), mock.patch.object(task, "FEET_CROSS_MARGIN", 0.3), mock.patch.object(task, "feet_cross", rec_term), mock.patch.object(train.XSoma, "termination", rec_end):
        loss0 = float(plain.imagine(*args)[0])
        assert not terms and len(ends) == c.horizon
        del ends[:]
        loss = float(pen.imagine(*args)[0])
    alive = np.cumprod(1 - np.stack(ends), 0)
    disc = (c.gamma ** np.arange(c.horizon))[:, None] * np.concatenate((np.ones_like(alive[:1]), alive[:-1]))
    want = 3.0 * task.DT * float(scale) * (disc * np.stack(terms)).sum(0).mean()
    assert len(terms) == c.horizon and want > 1e-3 and np.isclose(loss - loss0, want, rtol=1e-3), (loss, loss0, want)
    print(f"feet: stance 0.24 m: no penalty; 0.06 m: 0.04; swapped: 0.34, at any yaw; --feet_cross 3: imagination's loss {loss:.4f} = {loss0:.4f} + 3 x "
          f"{task.DT} s x scale {float(scale)} x the discounted term ({want:.4f}); without it the term is not computed")  # fmt: skip


def check_default():
    xs = build()
    carry = xs.init(jax.random.PRNGKey(0))
    assert isinstance(carry["pi"], list) and not {"mem", "pi0"} & set(carry) and "heading" not in carry["buf"] and xs.ckpt_extra is None
    progs = xs.compile_programs(carry)
    for _ in range(3 * H):
        carry, _ = progs["collect"](carry)
    carry, _ = progs["wm"](carry)
    carry, st = progs["pi"](carry)
    assert sorted(st) == ["G", "gn", "gn_gated", "r", "surv", "wm_gated"] and all(np.isfinite(float(v)) for v in st.values())
    print(f"default: XSoma's own policy on the stand-in, imagined return {float(st['G']):.3f}, gradient norm {float(st['gn']):.3f}")


def test_finetune():
    with mock.patch.multiple(task, **ROBOT), tempfile.TemporaryDirectory() as d:
        for heading in ("hold", "rate"):
            with tempfile.TemporaryDirectory() as run:
                check_finetune(run, heading)
        check_eval()
        check_noise(d)
        check_feet()
        check_default()


if __name__ == "__main__":
    test_finetune()
    print("\nall checks passed")
