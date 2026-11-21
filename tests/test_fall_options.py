"""Options against falls (run: python tests/test_fall_options.py, or pytest): train.py's --foot_clearance, --cmd_scale,
--push_vel, --term_penalty, --fall_starts, --fall_lead and --tilt_term, and evaluate.py's --cmd_scale and --push_vel. CPU only.

  term      task.foot_clearance: (height - lowest sole point) x horizontal speed for a low foot that moves, nothing for one
            above the height, next to nothing at rest, where the gradient stays finite; lifting a low moving foot lowers it
  curb      the real simulator of the G1 on the terrain, hovering 3 cm over flat ground and moving sideways at 1 m/s: the
            term is 2 x 0.03; with the ground 5 cm higher under the left foot only, that foot's share grows by 0.05 and the
            right one's stays; 10 cm up it is zero
  imagine   on test_finetune's stand-in: without the option the term is not computed; with it the term gets each step's soles
            after the step and the feet's displacement in it, and the loss gains weight x dt x penalty scale x the
            discounted term; the compiled policy update runs
  commands  task.sample_commands with factors is the plain draw times the factors, with the same gait frequencies. train.py
            passes --cmd_scale and --push_vel to its simulator and draws imagination's commands with the factors;
            evaluate.py --cmd_scale reaches its simulator and the record, which has no such entry without it
  step      the real Sim.step of the G1 (test_falls.py's hand-made physics) with `cmd_scale`: every command, at the start,
            after a reset and after a resample, is the plain simulator's times the factors; with `push_vel` 2: every push
            leaves twice the plain one's velocity; with `push_vel` 0 there is no push and no velocity is overwritten
  falls     buffer._before_falls on a hand-made ring buffer that has wrapped: every redrawn start lies in the K steps up to a
            fall of its env and is not younger than asked, all such transitions get drawn, the rest stay as drawn, and so
            does everything while the buffer holds no fall; sample_starts with the share on the stand-in's buffer keeps it.
            --fall_lead narrows that to its number of steps. --term_penalty P: the loss gains P x the discounted share of
            rollouts at the step that ends them. --tilt_term: every imagined step asks for a second termination, of the
            cosine of the new state's tilt minus the angle's cosine, and a rollout whose base starts tilted further ends at once
"""

import contextlib
import dataclasses
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
import numpy as np  # noqa: E402
import test_falls as tfalls  # noqa: E402
import test_finetune as tf  # noqa: E402

from xsoma import buffer as rb  # noqa: E402
from xsoma import evaluate, task, train  # noqa: E402
from xsoma.sim import SimCfg, unflatten  # noqa: E402

F = (1.0, 0.5, 0.0)  # command factors: sideways halved, no turning


def test_term():
    sole = jnp.asarray([[0.02, 0.10], [0.0, 0.02], [0.0, 0.0]])
    step = jnp.asarray([[[0.03, 0.04], [0.5, 0.5]], [[0.0, 0.0], [0.0, -0.02]], [[0.0, 0.0], [0.0, 0.0]]])  # 2.5 m/s and fast; rest and 1 m/s; rest
    rest = 1e-6 / task.DT
    assert task.FOOT_CLEARANCE_HEIGHT == 0.06 and np.allclose(task.foot_clearance(sole, step), [0.04 * 2.5, 0.06 * rest + 0.04, 0.12 * rest], atol=1e-6)
    assert np.allclose(task.foot_clearance(sole, step, 0.03), [0.01 * 2.5, 0.03 * rest + 0.01, 0.06 * rest], atol=1e-6)
    g_sole, g_step = jax.grad(lambda a, b: task.foot_clearance(a, b).sum(), (0, 1))(sole, step)
    assert np.allclose(g_sole, [[-2.5, 0.0], [-rest, -1.0], [-rest, -rest]], atol=1e-4) and np.isfinite(g_step).all() and np.allclose(g_step[2], 0.0)
    print("term: (0.06 m - sole) x speed for a low moving foot, 0 above 0.06 m, 3e-6 at rest with a finite gradient; lifting the moving foot lowers it")


def test_curb():
    xs = train.XSoma(dataclasses.replace(train.Config(), **tf.SMALL, foot_clearance=1.0))
    sim, y = xs.sim, 2.05  # the pelvis over the middle of the pixel between y = 2.0 and 2.1, a foot on either side
    flat = jnp.zeros(sim.heights.shape)
    curb = flat.at[:, round((2.1 - sim.x0) / sim.hs) :].set(0.05)  # 5 cm higher from y = 2.1 on, where the left foot is

    def view(heights, z, dy=0.0):
        root = jnp.asarray([2.0, y + dy, z, 1.0, 0.0, 0.0, 0.0])
        return sim.view_state(unflatten(jnp.concatenate((root, jnp.asarray(task.DEFAULT_Q)))[None], jnp.zeros((1, 6 + task.J))), heights)

    z = 1.0 - float(view(flat, 1.0).soles.min()) + 0.03  # the soles' lowest points 3 cm over the flat ground
    term = lambda heights, z: float(xs.foot_clearance(view(heights, z, -0.02), view(heights, z))[0])  # noqa: E731  (0.02 m per step: 1 m/s)
    low = lambda heights: np.asarray(jnp.where(xs.sole_foot, view(heights, z).soles[:, None], jnp.inf).min(-1))[0]  # noqa: E731
    assert xs.sole_foot.shape == (2, len(sim.sole_radius)) and (xs.sole_foot.sum(0) == 1).all() and xs.sole_foot.sum(1).min() > 0
    assert np.allclose(low(flat), 0.03, atol=2e-3) and np.allclose(low(curb), [-0.02, 0.03], atol=2e-3), (low(flat), low(curb))
    assert np.isclose(term(flat, z), 2 * 0.03, atol=4e-3) and np.isclose(term(curb, z) - term(flat, z), 0.05, atol=1e-4) and term(curb, z + 0.1) < 1e-4
    print(f"curb: the G1 3 cm over the ground at 1 m/s sideways: {term(flat, z):.3f}; with a 5 cm step up under its left foot {term(curb, z):.3f} "
          f"(lowest sole points {[round(float(x), 3) for x in low(curb)]} m over the terrain, left then right); 10 cm higher: {term(curb, z + 0.1):.5f}")  # fmt: skip


def check_imagine():
    W, HEIGHT = 3.0, 0.3  # a height that the stand-in's feet (about 0.1 m, never over a terrain) are under
    with mock.patch.object(tf.FakeSim, "sole_frame", np.asarray([2, 3]), create=True):
        plain, pen = tf.build(), tf.build(foot_clearance=W, foot_clearance_height=HEIGHT)
    carry, step = plain.init(jax.random.PRNGKey(0)), jax.jit(plain.collect_step)
    for k in jax.random.split(jax.random.PRNGKey(1), 3 * tf.H):
        carry, _ = step(carry, k)
    c, scale = pen.cfg, carry["env"].scale
    start = plain.sample_starts(carry["buf"], jax.random.PRNGKey(2))
    args = (carry["pi"], carry["wm"], carry["wm_b"], start, jax.random.split(jax.random.PRNGKey(3), c.horizon), scale, carry["env"].world.heights)
    log = {"term": [], "end": [], "view": []}

    def record(name, f):
        def g(*a):
            log[name].append((a, f(*a)))
            return log[name][-1][1]

        return g

    with jax.disable_jit(), mock.patch.object(task, "foot_clearance", record("term", task.foot_clearance)), \
            mock.patch.object(train.XSoma, "termination", record("end", train.XSoma.termination)), mock.patch.object(tf.FakeSim, "view_state", record("view", tf.FakeSim.view_state)):  # fmt: skip
        loss0 = float(plain.imagine(*args)[0])
        assert not log["term"] and len(log["end"]) == c.horizon and len(log["view"]) == c.horizon + 1, "weight 0: the term may not be computed"
        del log["end"][:], log["view"][:]
        loss = float(pen.imagine(*args)[0])
    views, terms = [v for _, v in log["view"]], np.stack([t for _, t in log["term"]])
    for i, ((sole, moved, height), _) in enumerate(log["term"]):  # step i runs from view i to view i + 1
        assert height == HEIGHT and np.array_equal(sole, views[i + 1].soles) and np.array_equal(moved, (views[i + 1].robot.foot_p - views[i].robot.foot_p)[..., :2])
    alive = np.cumprod(1 - np.stack([e for _, e in log["end"]]), 0)
    disc = (c.gamma ** np.arange(c.horizon))[:, None] * np.concatenate((np.ones_like(alive[:1]), alive[:-1]))
    want = W * task.DT * float(scale) * (disc * terms).sum(0).mean()
    assert len(terms) == c.horizon and want > 1e-3 and np.isclose(loss - loss0, want, rtol=1e-3), (loss, loss0, want)
    pen.init(jax.random.PRNGKey(0))  # sets the optimizer's masks
    sub, st = jax.jit(pen.policy_update)(carry, start, jax.random.PRNGKey(4))
    assert np.isfinite(float(st["gn"])) and float(st["gn"]) > 0 and np.abs(tf.leaves(sub["pi"]) - tf.leaves(carry["pi"])).max() > 0
    print(f"imagine: no option: the term is not computed; --foot_clearance {W}: loss {loss:.4f} = {loss0:.4f} + {W} x {task.DT} s x scale {float(scale)} x the "
          f"discounted term ({want:.4f}), on each step's soles after it and the feet's displacement in it; compiled: gradient norm {float(st['gn']):.3f}")  # fmt: skip
    return carry, start, args


def test_before_falls():
    T, N, K, lo = 12, 3, 3, 2
    term = np.zeros((T, N), np.float32)
    ptr = 5  # the oldest step is row 5: age a is row (5 + a) % 12
    falls = [(4, 0), (9, 2), (11, 2), (0, 1)]  # (age, env); the last one is younger than `lo` allows a start to be
    for a, e in falls:
        term[(ptr + a) % T, e] = 1
    buf = {"term": jnp.asarray(term), "s_qpos": jnp.zeros((T, N, 1)), "ptr": jnp.asarray(ptr), "size": jnp.asarray(T)}
    want = {(a - k, e) for a, e in falls for k in range(K) if a - k >= lo}
    age0, env0 = jnp.full((400,), 7), jnp.full((400,), 1)
    age, env = (np.asarray(x) for x in rb._before_falls(buf, jax.random.PRNGKey(0), age0, env0, 0.75, K, lo))
    assert set(zip(age[:300].tolist(), env[:300].tolist())) == want and (age[300:] == 7).all() and (env[300:] == 1).all(), (set(zip(age.tolist(), env.tolist())), want)
    none = rb._before_falls({**buf, "term": jnp.zeros((T, N))}, jax.random.PRNGKey(0), age0, env0, 0.75, K, lo)
    half = rb._before_falls({**buf, "size": jnp.asarray(4), "ptr": jnp.asarray(4)}, jax.random.PRNGKey(0), age0, env0, 1.0, K)  # rows 0..3 filled only
    assert (np.asarray(none[0]) == 7).all() and rb._before_falls(buf, None, age0, env0, 0.0, K) == (age0, env0)
    filled = {(a, e) for a in range(4) for e in range(N) if term[a : a + K, e][: 4 - a].any()}
    assert filled and set(zip(np.asarray(half[0]).tolist(), np.asarray(half[1]).tolist())) == filled
    print(f"falls: of 400 starts, the first 300 are redrawn onto the {len(want)} transitions up to {K - 1} steps before a fall (age >= {lo}), all of them hit; "
          "the rest, and all of them without a fall in the buffer, stay as drawn; a buffer that is not full counts its filled rows only")  # fmt: skip


def check_falls(carry, start, args):
    # sample_starts with the share: every start of the stand-in's buffer then lies within `horizon` steps before a fall
    xs = tf.build(fall_starts=1.0)
    st = xs.sample_starts(carry["buf"], jax.random.PRNGKey(2))
    buf = {k: np.asarray(v) for k, v in carry["buf"].items()}
    order = (int(buf["ptr"]) - int(buf["size"]) + np.arange(int(buf["size"]))) % buf["term"].shape[0]
    hits = 0
    for q in np.asarray(st["s_qpos"]):
        (a,), (e,) = np.nonzero((buf["s_qpos"][order] == q).all(-1))[:2]
        hits += bool(buf["term"][order][a : a + xs.cfg.horizon, e].any())
    plain = np.mean([bool(buf["term"][order][a : a + xs.cfg.horizon, e].any()) for a in range(len(order)) for e in range(tf.N)])
    assert hits == len(st["s_qpos"]) and 0 < plain < 0.9, (hits, plain)
    for q in np.asarray(tf.build(fall_starts=1.0, fall_lead=2).sample_starts(carry["buf"], jax.random.PRNGKey(2))["s_qpos"]):
        (a,), (e,) = np.nonzero((buf["s_qpos"][order] == q).all(-1))[:2]
        assert buf["term"][order][a : a + 2, e].any()

    # --tilt_term: a second termination per step, on the cosine of the new state's tilt
    ANGLE, tilt, seen, states, real, wm_step = 0.7, tf.build(tilt_term=0.7), [], [], train.XSoma.termination, train.nets.wm_step
    with jax.disable_jit(), mock.patch.object(train.XSoma, "termination", lambda self, x: seen.append(x) or real(self, x)), \
            mock.patch.object(train.nets, "wm_step", lambda *a, **kw: states.append(wm_step(*a, **kw)) or states[-1]):  # fmt: skip
        tilt.imagine(*args)
        flat = {**args[3], "s_qpos": args[3]["s_qpos"].at[:, 3:7].set(jnp.asarray([0.0, 1.0, 0.0, 0.0]))}  # upside down
        assert float(tilt.imagine(*args[:3], flat, *args[4:])[1]["surv"]) == 0.0
    assert len(seen) >= 2 * tilt.cfg.horizon and all(np.allclose(seen[2 * i + 1][:, 0], states[i][1][:, 2, 2] - np.cos(ANGLE), atol=1e-6) for i in range(tilt.cfg.horizon))

    # --term_penalty: rollouts 0, 2, 4, ... end at their third step
    W, pen, calls = 5.0, tf.build(term_penalty=5.0), []

    def end(self, clearance):
        calls.append(0)
        return jnp.where((len(calls) == 3) & (jnp.arange(clearance.shape[0]) % 2 == 0), 1.0, 0.0)

    with jax.disable_jit(), mock.patch.object(train.XSoma, "termination", end):
        loss0 = float(tf.build().imagine(*args)[0])
        del calls[:]
        loss = float(pen.imagine(*args)[0])
    want = W * pen.cfg.gamma**2 * 0.5
    assert np.isclose(loss - loss0, want, rtol=1e-4), (loss, loss0, want)
    print(f"falls: --fall_starts 1: all {hits} starts lie within {xs.cfg.horizon} steps before a fall (uniform: {plain:.2f} of the buffer); --term_penalty {W}: "
          f"with half the rollouts ending at step 3 the loss grows by {loss - loss0:.4f} = {W} x gamma^2 x 0.5; --fall_lead 2: within 2 steps; "
          f"--tilt_term {ANGLE}: a second termination per step on cos(tilt) - cos({ANGLE}), and rollouts that start upside down all end")  # fmt: skip


def check_commands(carry, start, args):
    cmd, freq = task.sample_commands(jax.random.PRNGKey(5), 4000)
    cmd_f, freq_f = task.sample_commands(jax.random.PRNGKey(5), 4000, F)
    stand = np.asarray(cmd == 0).all(-1)
    assert np.array_equal(cmd_f, cmd * np.asarray(F, np.float32)) and np.array_equal(freq_f, freq) and 0.15 < stand.mean() < 0.25 and np.abs(cmd_f[:, 1]).max() > 0.49

    # the stand-in passes on what train.py gives its simulator, and imagination draws with the factors at both of its places
    scaled = tf.build(cmd_scale="1,0.5,0", push_vel=1.5, cmd_fresh=1.0)
    assert scaled.sim.cfg.cmd_scale == F and scaled.sim.cfg.push_vel == 1.5 and tf.build().sim.cfg.cmd_scale is None and tf.build().sim.cfg.push_vel == task.PUSH_VEL
    calls, draw = [], task.sample_commands
    with jax.disable_jit(), mock.patch.object(task, "sample_commands", lambda key, n, scale=None: calls.append(scale) or draw(key, n, scale)):
        fresh = scaled.sample_starts(carry["buf"], jax.random.PRNGKey(2))
        scaled.imagine(*args)
    assert calls == [F] * (1 + scaled.cfg.horizon) and np.abs(fresh["cmd"][:, 1]).max() <= 0.5 and not np.asarray(fresh["cmd"][:, 2]).any() and np.abs(start["cmd"][:, 2]).max() > 0.5

    # evaluate.py: the option reaches the simulator and the record
    recs = []
    for options in ([], ["--cmd_scale", "1", "0.5", "0"], ["--push_vel", "0"]):
        seen = []
        with tempfile.TemporaryDirectory() as run, contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(evaluate, "make_sim", lambda n, *a, **kw: seen.append(a) or tf.FakeSim(SimCfg(num_envs=n, cmd_scale=a[-1] and tuple(a[-1]), **kw))):  # fmt: skip
            os.makedirs(os.path.join(run, "ckpt"))
            train.save_ckpt(os.path.join(run, "ckpt", "0000"), carry["avg"], carry["pi"], {"it": 2, "train_s": 1.0, "wall_s": 2.0, "env_steps": 3.0})
            evaluate.main([run, "--n", str(tf.N), *options])
            recs.append(json.loads(open(os.path.join(run, "eval.jsonl")).read()))
        assert seen[0][-1] == (list(F) if "--cmd_scale" in options else None)
    assert "cmd_scale" not in recs[0] and recs[1]["cmd_scale"] == list(F) and set(recs[1]) - set(recs[0]) == {"cmd_scale"}
    assert recs[2]["push_vel"] == 0 and set(recs[2]) - set(recs[0]) == {"push_vel"}
    print(f"commands: factors {F}: the plain draw times the factors, {stand.mean():.2f} of them stand; train.py passes them and push_vel to its simulator and draws "
          f"imagination's {len(calls)} times with them; evaluate.py --cmd_scale reaches its simulator and the record")  # fmt: skip


def test_step():
    # commands at the start, after robot 2's reset at step 40 and at every resample; pushes
    act = lambda obs: jnp.zeros((tfalls.N, task.J))  # noqa: E731
    with mock.patch.multiple(task, RESAMPLE=10, PUSH_INTERVAL=(0.1, 0.3)):
        (_, plain), (_, cmds), (_, push), (_, calm) = (tfalls.record(tfalls.FallSim(SimCfg(num_envs=tfalls.N, terrain="plane", **kw)), act, jax.random.PRNGKey(6), 45)
                                                       for kw in ({}, {"cmd_scale": F}, {"push_vel": 2.0}, {"push_vel": 0.0}))  # fmt: skip
    changes = int((np.abs(np.diff(plain["cmd"], axis=0)).max(-1) > 0).sum())
    assert np.array_equal(cmds["cmd"], plain["cmd"] * np.asarray(F, np.float32)) and plain["end"][:, 2].any() and changes >= 8 and np.abs(plain["cmd"][..., 2]).max() > 0.5
    hit = plain["pushed"]
    assert hit.sum() >= 8 and np.array_equal(push["pushed"], hit) and np.array_equal(cmds["qvel_next"], plain["qvel_next"]) and np.array_equal(push["cmd"], plain["cmd"])
    assert np.array_equal(push["qvel_next"][hit][:, :2], 2 * plain["qvel_next"][hit][:, :2]) and np.array_equal(push["qvel_next"][~hit], plain["qvel_next"][~hit])
    assert 0.5 < np.abs(plain["qvel_next"][hit][:, :2]).max() <= 1.0
    first = np.argmax(hit.any(1))  # up to the first push the run without pushes is the plain one; at it, its velocity is the physics' own
    assert not calm["pushed"][1:].any() and np.array_equal(calm["qvel_next"][:first], plain["qvel_next"][:first]) and np.array_equal(calm["qvel_next"][first], calm["qvel"][first])
    assert evaluate.make_sim(2, 0.5, False, "fast_sac", "plane", push_vel=0.0).cfg.push_vel == 0 and evaluate.make_sim(2, 0.5, False, "fast_sac", "plane").cfg.push_vel == task.PUSH_VEL
    assert evaluate.make_sim(2, 0.5, False, "fast_sac", "plane", 1001, [1.0, 0.5, 0.0]).cfg.cmd_scale == F and evaluate.make_sim(2, 0.5, False, "fast_sac", "plane").cfg.cmd_scale is None
    print(f"step: the real step with cmd_scale {F}: {changes} command changes in 45 steps of 4 robots, each the plain one's times the factors; push_vel 2: "
          f"{int(hit.sum())} pushes, each twice the plain one's velocity; push_vel 0: none")  # fmt: skip


def test_standin():
    with mock.patch.multiple(task, **tf.ROBOT):
        out = check_imagine()
        check_falls(*out)
        check_commands(*out)


if __name__ == "__main__":
    assert not os.environ.get("XSOMA_ROBOT_DIR"), "run this test without XSOMA_ROBOT_DIR"
    test_term()
    test_before_falls()
    test_curb()
    test_standin()
    test_step()
    print("\nall checks passed")
