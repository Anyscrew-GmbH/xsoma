"""The clearance between robot.json's `self_collision` groups (run: python tests/test_self_clearance.py, or pytest): the signed
distances of sim.SelfDistance, train.py's penalty and imagined termination on them, and the evaluation's measure; also the
evaluation's measure of the soles under the terrain, which every robot gets.

On the bundled G1 with its two legs as the groups (test_self_collision.py), against MuJoCo's CPU engine:
  proxy     on random leg poses, half of them with the legs pushed together: a capsule pair's distance is mj_geomDistance; with
            box soles (scene.py's foot="box") a pair with a box is never above MuJoCo's distance, never positive while the shapes
            overlap, and below MuJoCo's by at most the box capsules' radii while they are apart. The errors are printed
  stance    the simulator finds the 81 pairs by the contact bits; they are apart in the default stance and overlap with the
            legs crossed, by the depth and in the number of MuJoCo's contacts
  gradient  while the legs touch, descending the penalty's gradient in the hip rolls moves them apart. Once a leg's center line
            has passed through the other's, the nearest way out is onwards: there the gradient does not separate them (printed)
  startup   train.py refuses the two options for a robot without the key
  soles     the G1 without the key, standing tilted on a plane: with its lower foot 2 cm under the surface (by MuJoCo's geom
            poses) the simulator's view gives 0.02 m, resting on it 0; evaluate() on a hand-made sequence of such states gives
            each episode's largest depth, the share of steps deeper than 1 cm and the 99th percentile, up to the fall
On the stand-in simulator of test_finetune.py, with made-up distances of two pairs:
  imagine   without the options the distances are not computed; --self_clearance adds weight x dt x penalty scale x the
            discounted term to the loss; --self_clearance_term ends a rollout at the steps that are deeper than the threshold
            and only there, through the mask of the terrain termination; the compiled policy update runs with both
  metric    evaluate() on a hand-made sequence counts the overlapping steps before the fall and averages the smallest distance;
            a run's record and [eval] line carry both, and neither without the key
"""

import contextlib
import io
import json
import os
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import test_finetune as tf  # noqa: E402
from test_self_collision import LEGS, pose  # noqa: E402

from xsoma import evaluate, scene, task, train  # noqa: E402
from xsoma.sim import SelfDistance, Sim, SimCfg, unflatten  # noqa: E402

KEY = [["a"], ["b"]]  # a stand-in for the key: the stand-in simulators below bring their own distances


def g1():
    """The real simulator of the G1 with its legs as the groups (construction only: its physics needs a GPU)."""
    with mock.patch.object(task, "SELF_COLLISION", LEGS):
        return Sim(SimCfg(num_envs=1, terrain="plane"))


def state(q, root=(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)):
    q = jnp.asarray(q, jnp.float32)
    return unflatten(jnp.concatenate((jnp.broadcast_to(jnp.asarray(root, jnp.float32), q.shape[:-1] + (7,)), q), -1), jnp.zeros(q.shape[:-1] + (6 + task.J,)))


def crossed(roll):
    """The default pose with both hip rolls turned inward by `roll` (test_self_collision.pose)."""
    return jnp.asarray(task.DEFAULT_Q).at[task.DOF.index("left_hip_roll")].add(-roll).at[task.DOF.index("right_hip_roll")].add(roll)


def test_proxy(n=400):
    rng = np.random.default_rng(0)
    legs = [i for i, name in enumerate(task.DOF) if name.split("_")[1] in ("hip", "knee", "ankle")]
    for foot in ("capsules", "box"):
        if foot == "capsules":
            sim = g1()
            m, sd = sim.m, sim.self_distance
        else:  # the box soles carry no contact bits (scene.py), so the groups are picked by body here
            m = scene.build(foot="box")
            pick = lambda group: [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.removeprefix("robot_") in group]  # noqa: E731
            sd = SelfDistance(m, pick(LEGS[0]), pick(LEGS[1]))
            assert len(sd.ia) == 9 and (sd.slack > 0).sum() == 5 and np.isclose(sd.slack.max(), 2 * np.hypot(*scene.BOX_FOOT["size"][1:]))
        # half of the poses anywhere between the default pose and a uniform draw in the joint ranges, half near the default
        # pose with the hips rolled inward by up to 0.6 rad; any root pose
        q, h = np.tile(task.DEFAULT_Q, (n, 1)), n // 2
        lo, hi = m.jnt_range[1:, 0][legs], m.jnt_range[1:, 1][legs]
        q[:h, legs] += rng.uniform(0, 1, (h, 1)) * (rng.uniform(lo, hi, (h, len(legs))) - q[:h, legs])
        q[h:, legs] += rng.uniform(-0.3, 0.3, (n - h, len(legs)))
        q[h:, task.DOF.index("left_hip_roll")] -= rng.uniform(0, 0.6, n - h)
        q[h:, task.DOF.index("right_hip_roll")] += rng.uniform(0, 0.6, n - h)
        quat = rng.normal(size=(n, 4))
        root = np.concatenate((rng.uniform(-2, 2, (n, 3)), quat / np.linalg.norm(quat, axis=1, keepdims=True)), 1)
        got = np.asarray(jax.jit(sd)(state(q, root)))
        ref, d = np.zeros_like(got), mujoco.MjData(m)
        for i in range(n):
            d.qpos[:] = np.concatenate((root[i], q[i]))
            mujoco.mj_kinematics(m, d)
            ref[i] = [mujoco.mj_geomDistance(m, d, sd.geoms[a], sd.geoms[b], 10.0, None) for a, b in zip(sd.ia, sd.ib)]
        err, box, apart = ref - got, np.broadcast_to(sd.slack > 0, got.shape), ref > 0
        # float32 kinematics against MuJoCo's float64, at up to 3.5 m from the origin
        assert np.abs(err[~box]).max() < 1e-5, f"{foot}: capsule pairs differ from MuJoCo by {np.abs(err[~box]).max()}"
        assert ((ref < 0).any(1).mean() > 0.2) and (ref[~box] < 0).sum() > 100, "too few overlapping poses to test on"
        if box.any():
            # the capsule contains its box and lies within the box grown by the capsule's radius, so 0 <= MuJoCo - proxy <= slack
            # (the sum of the pair's box radii); the upper bound is MuJoCo's own depth convention once the shapes overlap
            assert err[box].min() > -1e-5 and (got[box & ~apart] < 1e-5).all(), "a box's capsule must contain the box"
            assert (err - sd.slack)[box & apart].max() < 1e-5, "a box's capsule is too far from its box"
            assert (box & ~apart).sum() > 10
        pct = lambda x: ", ".join(f"{p} % {1000 * v:.1f}" for p, v in zip((50, 95, 100), np.percentile(x, (50, 95, 100))))  # noqa: E731
        print(f"proxy, {foot} soles: {n} poses x {len(sd.ia)} pairs, the legs overlap in {(ref.min(1) < 0).mean():.2f} of the poses by MuJoCo and in "
              f"{(got.min(1) < 0).mean():.2f} by the proxy; capsule pairs: max |error| {np.abs(err[~box]).max():.1e} m"
              + (f"; pairs with a box, MuJoCo - proxy in mm: apart {pct(err[box & apart])}; overlapping {pct(err[box & ~apart])}; bound "
                 f"{1000 * sd.slack[sd.slack > 0].min():.1f} (one box) / {1000 * sd.slack.max():.1f} (two)" if box.any() else ""))  # fmt: skip


def test_stance():
    sim = g1()
    m, sd = sim.m, sim.self_distance
    names = [m.geom(g).name.removeprefix("robot_") for g in sd.geoms]
    assert len(sd.ia) == 81 and {m.body(m.geom_bodyid[g]).name.removeprefix("robot_") for g in sd.geoms} == set(LEGS[0] + LEGS[1])
    out = {}
    for roll in (0.0, 0.15, 0.45):
        dist, d = np.asarray(sim.self_distance(state(crossed(roll)))), pose(m, roll, z=1.0)
        depth = [c.dist for c in d.contact[: d.ncon]]
        # MuJoCo's contacts are leg x leg only in the air: as many as overlapping pairs, the deepest as deep
        assert (dist < 0).sum() == d.ncon and (roll == 0) == (d.ncon == 0) and np.isclose(min(depth, default=0.0), min(dist.min(), 0.0), atol=1e-5)
        k = dist.argmin()
        out[roll] = f"{dist.min():+.4f} m ({names[sd.ia[k]]} x {names[sd.ib[k]]}, {(dist < 0).sum()} pairs overlap)"
    assert float(task.self_clearance(jnp.asarray(dist))) > 0 == float(task.self_clearance(sim.self_distance(state(crossed(0.0)))))
    print(f"stance: 81 pairs; smallest distance in the default stance {out[0.0]}, hips rolled inward by 0.15 rad {out[0.15]}, by 0.45 rad {out[0.45]}; "
          "overlaps and depths are those of MuJoCo's contacts")  # fmt: skip


def test_gradient():
    sim = g1()
    left, right = task.DOF.index("left_hip_roll"), task.DOF.index("right_hip_roll")
    pen = lambda q: task.self_clearance(sim.self_distance(state(q)))  # noqa: E731
    gap = lambda q: float(sim.self_distance(state(q)).min())  # noqa: E731
    out = []
    for roll in (0.13, 0.15, 0.2):  # within the margin, overlapping by 1 cm, by 3.5 cm
        q = crossed(roll)
        p, g = jax.value_and_grad(pen)(q)
        assert p > 0 and np.isfinite(np.asarray(g)).all() and g[left] < 0 < g[right], f"roll {roll}: the gradient must roll both hips outward"
        hips = jnp.zeros_like(g).at[jnp.asarray([left, right])].set(g[jnp.asarray([left, right])])
        q1 = q - 0.02 * hips / jnp.linalg.norm(hips)
        assert pen(q1) < p and gap(q1) > gap(q), f"roll {roll}: a step down the gradient must separate the legs"
        out.append(f"{roll} rad: distance {gap(q):+.3f} m, penalty {float(p):.3f}, d/d(left, right hip roll) {float(g[left]):+.1f}, {float(g[right]):+.1f}")
    g = jax.grad(pen)(crossed(0.25))
    assert float(pen(jnp.asarray(task.DEFAULT_Q))) == 0 and not np.asarray(jax.grad(pen)(jnp.asarray(task.DEFAULT_Q))).any()
    assert np.isfinite(np.asarray(jax.grad(pen)(crossed(0.8)))).all(), "parallel or coincident center lines must keep the gradient finite"
    print("gradient: hips rolled inward by " + "; ".join(out) + f": descending it rolls both hips outward. At 0.25 rad the shins' center "
          f"lines have passed through each other and it is {float(g[left]):+.1f}, {float(g[right]):+.1f}: onwards, not back")  # fmt: skip


def test_startup():
    for flag in ("--self_clearance", "--self_clearance_term"):
        for value, ok in (("0.5", task.SELF_COLLISION is not None), ("-1", False)):
            try:
                with mock.patch.object(train, "XSoma", side_effect=KeyboardInterrupt), tempfile.TemporaryDirectory() as d:
                    train.main([flag, value, "--log_dir", d])
            except AssertionError as e:
                assert not ok and ("not negative" if value == "-1" else "self_collision") in str(e), e
            except KeyboardInterrupt:
                assert ok, f"{flag} {value}: not refused"
    print("startup: --self_clearance and --self_clearance_term are refused for a robot without the key, and when negative")


class SoleSeq(tf.SeqSim):
    """test_finetune's sequence simulator (env 1 falls in its third step) around the real G1 on a plane: after step i of env e
    the robot stands rolled by 0.1 rad, the lowest point of its soles SINK[e, i] m under the surface (negative: above it)."""

    SINK = np.asarray([[0.0, 0.02, 0.005, -0.05, 0.02, 0.012], [0.03, 0.0, 0.0, 0.5, 0.5, 0.5], [0.0] * 6], np.float32)

    def __init__(self):
        self.sim = Sim(SimCfg(num_envs=1, terrain="plane"))
        self.stand = np.concatenate(([0, 0, 1.0, np.cos(0.05), np.sin(0.05), 0, 0], task.DEFAULT_Q)).astype(np.float32)
        self.stand[2] -= float(self.view(jnp.asarray(self.stand), jnp.zeros(6 + task.J), jnp.zeros((1, 1))).soles.min())

    def qpos(self, sink):
        return jnp.asarray(self.stand) - jnp.asarray(sink)[..., None] * (jnp.arange(len(self.stand)) == 2)

    def step(self, s, a):
        s1, tr = super().step(s, a)
        tr.qpos1, tr.qvel1 = self.qpos(jnp.asarray(self.SINK)[jnp.arange(3), s.t]), jnp.zeros((3, 6 + task.J))
        return s1, tr

    def view(self, qpos, qvel, heights):
        return self.sim.view(qpos, qvel, heights)


def test_soles():
    seq = SoleSeq()
    m, d, feet = seq.sim.m, mujoco.MjData(seq.sim.m), {}
    for sink in (0.02, 0.0):
        d.qpos[:] = np.asarray(seq.qpos(sink))
        mujoco.mj_kinematics(m, d)
        for g in seq.sim.sole_geoms:  # a capsule's lowest point, per foot
            low = d.geom_xpos[g, 2] - m.geom_size[g, 0] - abs(d.geom_xmat[g, 8]) * m.geom_size[g, 1]
            feet[m.geom_bodyid[g]] = min(low, feet.get(m.geom_bodyid[g], np.inf))
        depth = -float(seq.view(seq.qpos(sink), jnp.zeros(6 + task.J), jnp.zeros((1, 1))).soles.min())
        lower, upper = sorted(feet.values())
        assert abs(depth - sink) < 1e-6 and abs(-lower - sink) < 1e-5 and upper > lower + 0.01, (sink, depth, feet)
        feet.clear()
    out = evaluate.evaluate(seq, lambda obs: obs, jax.random.PRNGKey(0), steps=6)
    alive = np.maximum(np.concatenate((seq.SINK[0], seq.SINK[1, :3], seq.SINK[2])), 0)  # the depths that count
    assert len(out) == 11 and np.asarray(out[2]).tolist() == [False, True, False] and evaluate.SOLE_SINK == 0.01
    assert np.allclose(out[8], [3 / 6, 1 / 3, 0]) and np.allclose(out[9], [0.02, 0.03, 0], atol=1e-6) and np.isclose(out[10], np.percentile(alive, 99), atol=1e-6)
    print(f"soles: the tilted G1's lower foot 0.02 m under the plane by MuJoCo's geoms: the view's deepest sole point is 0.02 m under it, resting: "
          f"{abs(depth):.1e} m; on the sequence an episode's largest depth is {[round(float(x), 3) for x in out[9]]} m before a fall or the end, deeper "
          f"than 1 cm in {[round(float(x), 3) for x in out[8]]} of the steps, 99th percentile {float(out[10]):.4f} m")  # fmt: skip


class GapSim(tf.FakeSim):
    """test_finetune's stand-in with two made-up collider pairs, whose distances follow the first two joints."""

    calls = 0

    def self_distance(self, S):
        GapSim.calls += 1
        return 0.2 * (S[4][..., :2] - jnp.asarray(task.DEFAULT_Q[:2]))


def build(**options):
    with mock.patch.object(train, "Sim", GapSim):
        return train.XSoma(train.dataclasses.replace(train.Config(), **tf.SMALL, **options))


def check_imagine():
    W, THR = 3.0, 0.01
    plain, pen, end, both = build(), build(self_clearance=W), build(self_clearance_term=THR), build(self_clearance=W, self_clearance_term=THR)
    carry, step = plain.init(jax.random.PRNGKey(0)), jax.jit(plain.collect_step)
    for k in jax.random.split(jax.random.PRNGKey(1), 3 * tf.H):
        carry, _ = step(carry, k)
    c, scale = plain.cfg, carry["env"].scale
    start = plain.sample_starts(carry["buf"], jax.random.PRNGKey(2))
    # a terrain 0.5 m high, which the stand-in's termination collider (0.3 m under a pelvis at about 0.8 m) touches in some rollouts
    args = (carry["pi"], carry["wm"], carry["wm_b"], start, jax.random.split(jax.random.PRNGKey(3), c.horizon), scale, jnp.full((1, 1), 0.5))
    log = {"gap": [], "term": [], "reward": [], "end": []}

    def record(name, f):
        def g(*a):
            log[name].append(f(*a))
            return log[name][-1]

        return g

    def run(xs):
        for v in log.values():
            del v[:]
        GapSim.calls = 0
        with jax.disable_jit(), mock.patch.object(GapSim, "self_distance", record("gap", GapSim.self_distance)), \
                mock.patch.object(task, "self_clearance", record("term", task.self_clearance)), mock.patch.object(task, "reward", record("reward", task.reward)), \
                mock.patch.object(train.XSoma, "termination", record("end", train.XSoma.termination)):  # fmt: skip
            loss = float(xs.imagine(*args)[0])
        return loss, {k: np.stack(v) if v else None for k, v in log.items()}

    disc = (c.gamma ** np.arange(c.horizon))[:, None]
    before = lambda ended: np.concatenate((np.ones_like(ended[:1]), np.cumprod(1 - ended, 0)[:-1]))  # noqa: E731, alive before each step
    loss0, l0 = run(plain)
    assert GapSim.calls == 0 and l0["gap"] is None and l0["term"] is None and len(l0["end"]) == c.horizon, "weight 0: nothing may be computed"

    loss, l = run(pen)
    assert task.SELF_CLEARANCE_MARGIN == 0.02 and np.allclose(l["term"], np.maximum(0, 0.02 - l["gap"]).sum(-1), atol=1e-7)
    want = W * task.DT * float(scale) * (disc * before(l["end"]) * l["term"]).sum(0).mean()
    assert len(l["end"]) == c.horizon and (l["gap"] > 0.02).any() and want > 1e-4 and np.isclose(loss - loss0, want, rtol=1e-3), (loss, loss0, want)

    loss_e, l = run(end)
    terrain, deep = l["end"][0::2], l["end"][1::2]  # per step: the terrain's flag, then the clearance's
    low = l["gap"].min(-1)
    assert l["term"] is None and len(l["end"]) == 2 * c.horizon and np.array_equal(terrain, l0["end"]) and np.allclose(deep, low < -THR, atol=1e-6)
    cases = ((deep > 0.5) & (terrain < 0.5), (deep < 0.5) & (terrain > 0.5), (low < 0) & (low > -THR), low > 0)
    assert all(x.any() for x in cases), f"the rollouts need every case: {[int(x.sum()) for x in cases]}"
    want_e = -(disc * before(1 - (1 - terrain) * (1 - deep)) * l["reward"]).sum(0).mean()
    assert np.isclose(loss_e, want_e, rtol=1e-5) and abs(loss_e - loss0) > 1e-3, (loss_e, want_e, loss0)
    flag = np.asarray(end.termination(jnp.asarray([[-0.0101, 0.3], [-0.0099, 0.3], [0.3, -0.5]]) + THR))
    assert np.allclose(flag, [1, 0, 1], atol=1e-6)

    both.init(jax.random.PRNGKey(0))  # sets the optimizer's masks
    sub, st = jax.jit(both.policy_update)(carry, start, jax.random.PRNGKey(4))
    moved = np.abs(tf.leaves(sub["pi"]) - tf.leaves(carry["pi"])).max()
    assert np.isfinite(float(st["gn"])) and float(st["gn"]) > 0 and moved > 0
    print(f"imagine: no option: the distances are not computed; --self_clearance {W}: loss {loss:.4f} = {loss0:.4f} + {W} x {task.DT} s x scale "
          f"{float(scale)} x the discounted term ({want:.4f}); --self_clearance_term {THR}: {int(cases[0].sum())} of {deep.size} imagined steps are deeper, above "
          f"the terrain, and end their rollouts as the {int((terrain > 0.5).sum())} below it do, {int(cases[2].sum())} shallower overlaps do not, loss {loss_e:.4f}; both, compiled: gradient norm {float(st['gn']):.3f}")  # fmt: skip
    return carry


class GapSeq(tf.SeqSim):
    """test_finetune's sequence simulator (env 1 falls in its third step) with the distances of two collider pairs from a table."""

    GAP = np.asarray([[[0.05, 0.03], [0.05, -0.01], [-0.02, 0.03], [0.01, 0.02], [0.05, 0.03], [0.05, 0.03]],
                      [[0.04, 0.04], [-0.03, 0.04], [0.04, 0.04], [-0.5, -0.5], [-0.5, -0.5], [-0.5, -0.5]], [[0.1, 0.2]] * 6], np.float32)  # fmt: skip

    def step(self, s, a):
        s1, tr = super().step(s, a)
        tr.qpos1 = jnp.zeros((3, 8)).at[:, 0].set(s.t).at[:, 3].set(1.0)  # the step's number in place of x
        tr.qvel1 = jnp.zeros((3, 7))
        return s1, tr

    def view(self, qpos, qvel, heights):
        return super().view(qpos[:, 0].astype(int), qvel, heights)

    def self_distance(self, S):
        return jnp.asarray(self.GAP)[jnp.arange(3), S[0][:, 0].astype(int)]


def check_metric(carry):
    args = (GapSeq(), lambda obs: obs, jax.random.PRNGKey(0))
    old = evaluate.evaluate(*args, steps=6)
    with mock.patch.object(task, "SELF_COLLISION", KEY):
        new = evaluate.evaluate(*args, steps=6)
    assert len(old) == 11 and len(new) == 13 and all(np.array_equal(a, b) for a, b in zip(old, new))
    low = GapSeq.GAP.min(-1)
    assert np.allclose(new[11], [2 / 6, 1 / 3, 0]) and np.allclose(new[12], [low[0].mean(), low[1, :3].mean(), 0.1], atol=1e-7)

    lines = {}
    for key in (None, KEY):
        with tempfile.TemporaryDirectory() as run, mock.patch.object(task, "SELF_COLLISION", key), contextlib.redirect_stdout(io.StringIO()) as out, \
                mock.patch.object(evaluate, "make_sim", lambda n, *a: GapSim(SimCfg(num_envs=n))):  # fmt: skip
            os.makedirs(os.path.join(run, "ckpt"))
            train.save_ckpt(os.path.join(run, "ckpt", "0000"), carry["avg"], carry["pi"], {"it": 2, "train_s": 1.0, "wall_s": 2.0, "env_steps": 3.0})
            evaluate.main([run, "--n", str(tf.N)])
            rec = json.loads(open(os.path.join(run, "eval.jsonl")).read())
        lines[bool(key)] = (rec, out.getvalue().strip())
    (rec0, line0), (rec, line) = lines[False], lines[True]
    assert set(rec) - set(rec0) == {"legs_overlap", "legs_dist"} and {"sole_sink_max", "sole_sink_frac", "sole_sink_p99"} < set(rec0) and all(rec[k] == v for k, v in rec0.items() if k != "ckpt")
    assert 0 < rec["legs_overlap"] <= 1 and -0.2 < rec["legs_dist"] < 0.2 and "legs" not in line0 and line.count(" | legs overlap ") == 1
    assert line.split(" | legs overlap ")[0].split(": ", 1)[1] == line0.split(": ", 1)[1]
    print(f"metric: on the sequence, overlap in {[round(float(x), 3) for x in new[11]]} of the steps before a fall or the end, mean smallest distance "
          f"{[round(float(x), 3) for x in new[12]]} m; the other outputs unchanged; a run's record gains legs_overlap and legs_dist: [eval] ... | "
          + line.split(" | ")[-1])  # fmt: skip


def test_standin():
    with mock.patch.multiple(task, **tf.ROBOT, SELF_COLLISION=KEY):
        carry = check_imagine()
    with mock.patch.multiple(task, **tf.ROBOT):
        check_metric(carry)


if __name__ == "__main__":
    assert not os.environ.get("XSOMA_ROBOT_DIR"), "run this test without XSOMA_ROBOT_DIR"
    test_proxy()
    test_stance()
    test_gradient()
    test_startup()
    test_soles()
    test_standin()
    print("\nall checks passed")
