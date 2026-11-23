"""Options for standing while something else than the policy drives some joints (run: python tests/test_stand_options.py, or
pytest): train.py's --ext_joints, --ext_share, --ext_blind, --stand_still and --track_l1, and the fourth and fifth entries of --cmd_scale. CPU only.

  terms     task.stand_still: under a zero command the base's planar and angular speed plus the feet's speeds, nothing under
            another command, a finite gradient at rest. task.ext_step: a target never moves faster than EXT_SPEED and never
            away from its goal, goals lie in their range and are redrawn once per EXT_PERIOD on average. A fourth entry of
            the command factors is the share of zero commands; the other commands stay as drawn
  standin   on test_finetune's stand-in with every command zero. Without the options nothing of them exists. With
            --ext_joints: an env's joints are taken over from its first new command on; from then on their applied targets are
            task.ext_step's, continuous with the policy's own before, and every other action is the policy's; the replay buffer
            marks those steps. Imagination continues such a rollout's targets from its start's last action at EXT_SPEED at
            most, leaves every other rollout to the policy, and gives the reward the default pose and no action for those
            joints. --stand_still W, --track_l1 W: the loss gains W x dt x penalty scale x the discounted term; the compiled
            update runs
  blind     a tracker policy with --ext_blind 1 on the same stand-in: in collection and in imagination its observation holds
            the default pose and zero velocity for those joints while they are driven from outside, and its own last action
            instead of the applied one; the histories rebuilt from the replay buffer are the ones it held; training's compiled
            programs run
"""

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

from xsoma import task, tracker, train  # noqa: E402

STEP = task.EXT_SPEED * task.DT + 1e-6  # rad: the most a target moves in one policy step


def test_terms():
    rb = task.Robot(None, jnp.broadcast_to(jnp.eye(3), (2, 3, 3)), jnp.asarray([[0.3, 0.4, 9.0]] * 2), jnp.asarray([[0.0, 0.0, 2.0]] * 2), *[None] * 5)
    assert np.allclose(task.track_l1(rb, jnp.asarray([[0.0, 0.0, 0.0], [0.3, 0.0, 0.5]])), [0.5 + 2.0, 0.4 + 1.5], atol=1e-5)
    # relative to the command's size, floored at 0.2: a zero command keeps the plain error, another one divides by its norm
    assert np.allclose(task.track_l1(rb, jnp.asarray([[0.0, 0.0, 0.0], [0.3, 0.0, 0.5]]), 0.2), [0.5 + 2.0, (0.4 + 1.5) / np.hypot(0.3, 0.5)], atol=1e-4)
    still = task.Robot(None, jnp.broadcast_to(jnp.eye(3), (4, 3, 3)), jnp.zeros((4, 3)), jnp.zeros((4, 3)), *[None] * 5)  # standing still costs 1 under a command along one axis of the floor's size or more, less below it
    assert np.allclose(task.track_l1(still, jnp.asarray([[0.2, 0.0, 0.0], [0.0, -0.9, 0.0], [0.0, 0.0, 0.5], [0.05, 0.0, 0.0]]), 0.2), [1.0, 1.0, 1.0, 0.25], atol=1e-4)
    feet = jnp.asarray([[[0.0, 0.0, 0.02], [0.006, 0.008, 0.0]]] * 2)
    cmd = jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    assert np.allclose(task.stand_still(rb, feet, cmd), [0.5 + 2.0 + (0.02 + 0.01) / task.DT, 0.0], atol=1e-4)
    assert np.isfinite(jax.grad(lambda f: task.stand_still(rb._replace(v=0 * rb.v, w=0 * rb.w), f, cmd).sum())(0 * feet)).all()

    lo, hi = np.asarray([-1.0, 0.0], np.float32), np.asarray([1.0, 2.0], np.float32)
    x = goal = jnp.full((2000, 2), 0.5)
    speed, new, step = 0 * x, 0.0, jax.jit(task.ext_step)
    for k in jax.random.split(jax.random.PRNGKey(0), 300):
        x1, goal1, speed = step(k, x, goal, speed, lo, hi)
        drawn = np.asarray(goal1 != goal).any(-1)
        assert np.abs(x1 - x).max() <= STEP and (np.abs(goal1 - x1) <= np.abs(goal1 - x) + 1e-6).all() and (goal1[drawn] >= lo).all() and (goal1[drawn] <= hi).all()
        new, x, goal = new + drawn.mean() / 300, x1, goal1
    assert abs(new - task.DT / task.EXT_PERIOD) < 2e-3 and np.asarray(x != 0.5).any(-1).mean() > 0.9 and speed.max() <= task.EXT_SPEED

    c3, f3 = task.sample_commands(jax.random.PRNGKey(5), 4000, (1.0, 0.5, 0.0))
    c4, f4 = task.sample_commands(jax.random.PRNGKey(5), 4000, (1.0, 0.5, 0.0, 0.6))
    z3, z4 = (np.asarray(c == 0).all(-1) for c in (c3, c4))
    assert 0.15 < z3.mean() < 0.25 and 0.55 < z4.mean() < 0.65 and np.array_equal(c4[~z4], c3[~z4]) and np.array_equal(f3, f4)
    # a fifth entry: each component is zero with that probability on its own; the components that stay are as drawn
    c5, f5 = task.sample_commands(jax.random.PRNGKey(5), 4000, (1.0, 0.5, 1.0, 0.2, 0.3))
    c0, _ = task.sample_commands(jax.random.PRNGKey(5), 4000, (1.0, 0.5, 1.0, 0.2))
    drop = np.asarray(c5 == 0) & np.asarray(c0 != 0)
    assert 0.25 < drop[np.asarray(c0 != 0).all(-1)].mean() < 0.35 and np.array_equal(np.asarray(c5)[~drop], np.asarray(c0)[~drop]) and np.array_equal(f5, f3)
    assert 0.035 < (drop[:, 1] & drop[:, 2] & ~drop[:, 0]).mean() < 0.065  # along x alone: 0.8 x 0.3 x 0.3 x 0.7 = 0.05
    print(f"terms: stand_still = planar + angular + feet speeds under a zero command only; ext_step: at most {task.EXT_SPEED} rad/s, towards goals in "
          f"their range, new goals in {new:.4f} of the steps; a fourth command factor 0.6: {z4.mean():.2f} of the commands are zero (else {z3.mean():.2f})")  # fmt: skip


def test_standin():
    W, SPEC = 3.0, "knee:-0.2:0.4,elbow:0.5:0.6"
    with mock.patch.multiple(task, **tf.ROBOT, STAND_PROB=1.0):  # every command is zero
        plain, ext = tf.build(action_noise=0.0), tf.build(action_noise=0.0, ext_joints=SPEC, ext_share=1.0)
        i, q0, scale = ext.ext[0], np.asarray(task.DEFAULT_Q), np.asarray(task.ACTION_SCALE)
        carry0 = plain.init(jax.random.PRNGKey(0))
        assert plain.ext is None and "ext" not in carry0 and "ext" not in carry0["buf"] and i.tolist() == [1, 4]

        # collection: the applied actions against the policy's own and against the generator's state
        carry, own, applied, xs, on, act = ext.init(jax.random.PRNGKey(0)), [], [], [], [], train.nets.pi_apply
        xs.append(np.asarray(carry["ext"]["x"]))
        with jax.disable_jit(), mock.patch.object(train.nets, "pi_apply", lambda *a, **kw: own.append(act(*a, **kw)) or own[-1]), \
                mock.patch.object(tf.FakeSim, "step", lambda self, s, a, f=tf.FakeSim.step: applied.append(a) or f(self, s, a)):  # fmt: skip
            for k in jax.random.split(jax.random.PRNGKey(1), 20):
                carry, _ = ext.collect_step(carry, k)
                xs.append(np.asarray(carry["ext"]["x"]))
                on.append(np.asarray(carry["buf"]["ext"])[(int(carry["buf"]["ptr"]) - 1) % tf.T] > 0)
        own, applied, xs, on = np.asarray(own), np.asarray(applied), np.asarray(xs), np.asarray(on)
        rest = np.delete(np.arange(tf.J), i)
        assert 0.2 < on.mean() < 1 and not on[0].any() and (np.diff(on.astype(int), axis=0) >= 0).all(), "taken over from an env's first reset on"
        assert np.allclose(applied[..., rest], own[..., rest], atol=1e-6) and np.allclose(applied[~on], own[~on], atol=1e-5)
        assert np.allclose(q0[i] + scale[i] * applied[..., i], xs[1:], atol=1e-5) and np.abs(np.diff(xs, axis=0))[on].max() <= STEP
        assert np.abs(xs[1:][on] - (q0[i] + scale[i] * own[..., i])[on]).max() > 0.05, "the targets leave the policy's own"

        # imagination: the targets the world model gets, what the reward sees, and the stand_still term
        start = ext.sample_starts(carry["buf"], jax.random.PRNGKey(2))
        e = np.asarray(start["ext"]) > 0
        assert 0 < e.sum() < len(e)
        args = (carry["pi"], carry["wm"], carry["wm_b"], start, jax.random.split(jax.random.PRNGKey(3), ext.cfg.horizon), carry["env"].scale, carry["env"].world.heights)
        log = {"wm": [], "rew": [], "term": [], "end": []}

        def record(name, f):
            def g(*a, **kw):
                log.setdefault(name, []).append((a, f(*a, **kw)))
                return log[name][-1][1]

            return g

        with jax.disable_jit(), mock.patch.object(train.nets, "wm_step", record("wm", train.nets.wm_step)), mock.patch.object(task, "reward_terms", record("rew", task.reward_terms)), \
                mock.patch.object(task, "stand_still", record("term", task.stand_still)), mock.patch.object(task, "track_l1", record("term", task.track_l1)):  # fmt: skip
            loss0 = float(ext.imagine(*args)[0])
            assert not log["term"], "weight 0: the terms may not be computed"
            tgt = np.stack([np.asarray(train.target_q(start["a_last"]))] + [np.asarray(a[5]) for a, _ in log["wm"]])[..., i]
            assert np.abs(np.diff(tgt, axis=0))[:, e].max() <= STEP and np.abs(tgt[-1] - tgt[0])[e].max() > 0.02
            for (rb, _, a, a_prev, _, _), _ in log["rew"]:
                assert (np.asarray(rb.q)[e][:, i] == q0[i]).all() and not np.asarray(a)[e][:, i].any() and not np.asarray(a_prev)[e][:, i].any()
                assert (np.asarray(rb.q)[~e][:, i] != q0[i]).all() and np.asarray(a)[~e][:, i].all()
        for name in ("stand_still", "track_l1"):
            pen = tf.build(ext_joints=SPEC, ext_share=1.0, **{name: W})
            del log["end"][:]
            with jax.disable_jit(), mock.patch.object(task, name, record(name, getattr(task, name))), mock.patch.object(train.XSoma, "termination", record("end", train.XSoma.termination)):
                loss = float(pen.imagine(*args)[0])
            terms = np.stack([t for _, t in log.pop(name)])
            alive = np.cumprod(1 - np.stack([x for _, x in log["end"]]), 0)
            disc = (pen.cfg.gamma ** np.arange(pen.cfg.horizon))[:, None] * np.concatenate((np.ones_like(alive[:1]), alive[:-1]))
            want = W * task.DT * float(args[5]) * (disc * terms).sum(0).mean()
            assert len(terms) == pen.cfg.horizon and want > 1e-3 and np.isclose(loss - loss0, want, rtol=1e-3), (name, loss, loss0, want)
        pen.init(jax.random.PRNGKey(0))  # sets the optimizer's masks
        sub, st = jax.jit(pen.policy_update)(carry, start, jax.random.PRNGKey(4))
        assert np.isfinite(float(st["gn"])) and float(st["gn"]) > 0 and np.abs(tf.leaves(sub["pi"]) - tf.leaves(carry["pi"])).max() > 0
    print(f"standin: --ext_joints {SPEC}: {on.mean():.2f} of the collected steps taken over, targets continuous and at most {task.EXT_SPEED} rad/s, the "
          f"other actions the policy's; imagination continues them in {int(e.sum())} of {len(e)} rollouts and keeps them out of the reward; "
          f"--stand_still and --track_l1 {W}: the loss gains weight x dt x scale x the discounted term; compiled: gradient norm {float(st['gn']):.3f}")  # fmt: skip


def test_blind():
    SPEC = "knee:-0.2:0.4,elbow:0.5:0.6"
    with tempfile.TemporaryDirectory() as d, mock.patch.multiple(task, **tf.ROBOT, STAND_PROB=1.0):
        p = tf.make(d)
        xs = tf.build(init_tracker=os.path.join(d, "ckpt.npz"), tracker_meta=os.path.join(d, "meta.json"), tracker_noise=0.0, tracker_heading="rate", action_noise=0.0,
                      ext_joints=SPEC, ext_share=1.0, ext_blind=1)  # fmt: skip
        i, q0, to_sim = xs.ext[0], np.asarray(task.DEFAULT_Q), np.asarray(p["to_sim"])
        carry, seen, obs, n = xs.init(jax.random.PRNGKey(0)), [], tracker.observation, 0
        with jax.disable_jit(), mock.patch.object(tracker, "observation", lambda *a, **kw: seen.append(a) or obs(*a, **kw)):
            for k in jax.random.split(jax.random.PRNGKey(1), 12):
                own, t = np.asarray(carry["ext"]["own"]), np.asarray(carry["env"].t)
                carry, _ = xs.collect_step(carry, k)
                on = np.asarray(carry["buf"]["ext"])[(int(carry["buf"]["ptr"]) - 1) % tf.T] > 0
                _, q, qd, _, _, a_prev = seen[-1][:6]
                assert (np.asarray(q)[on][:, i] == q0[i]).all() and not np.asarray(qd)[on][:, i].any() and (np.asarray(q)[~on][:, i] != q0[i]).all()
                assert np.array_equal(np.asarray(a_prev)[:, to_sim][on & (t > 0)][:, i], own[on & (t > 0)])
                n += int(on.sum())
        assert n > 10
        carry, start, held = tf.collect(xs)  # the rebuilt histories against the held ones, from a buffer that wrapped
        e = np.asarray(start["ext"]) > 0
        assert np.abs(np.asarray(start["hist"]) - held).max() < 1e-5 and 0 < e.sum() and not np.asarray(start["hist"])[e][:, -1][:, to_sim[i]].any(), np.abs(np.asarray(start["hist"]) - held).max()
        del seen[:]
        with jax.disable_jit(), mock.patch.object(tracker, "observation", lambda *a, **kw: seen.append(a) or obs(*a, **kw)):
            xs.imagine(carry["pi"], carry["wm"], carry["wm_b"], start, jax.random.split(jax.random.PRNGKey(3), xs.cfg.horizon), carry["env"].scale, carry["env"].world.heights)
        assert len(seen) == xs.cfg.horizon and all((np.asarray(a[1])[e][:, i] == q0[i]).all() and not np.asarray(a[2])[e][:, i].any() for a in seen)
        progs = xs.compile_programs(carry)
        for name in ("collect", "wm", "pi"):
            carry, st = progs[name](carry)
        assert np.isfinite(float(st["gn"])) and "own" in carry["ext"] and "ext_own" in carry["buf"]
    print(f"blind: --ext_blind 1: in {n} collected steps and in every imagined step of {int(e.sum())} rollouts the tracker is fed the default pose, no velocity "
          "and its own last action for the joints driven from outside; rebuilt histories equal the held ones; the compiled programs run")


if __name__ == "__main__":
    assert not os.environ.get("XSOMA_ROBOT_DIR"), "run this test without XSOMA_ROBOT_DIR"
    test_terms()
    test_standin()
    test_blind()
    print("\nall checks passed")
