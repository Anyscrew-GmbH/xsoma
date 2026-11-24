"""The two impact penalties (run: python tests/test_impact.py, or pytest): task.touchdown_speed with train.py's --touchdown_speed
and --touchdown_height, and task.impact_accel with --impact_accel. CPU only.

  touchdown  task.touchdown_speed on hand-made feet: the squared downward speed of a foot below the height, zero for a foot
             that rises or is higher, a finite gradient
  accel      task.impact_accel on hand-made velocities: the squared upward acceleration in g, zero downward
  standin    on test_finetune's stand-in, per option: without it the term is not computed; with weight W the loss gains
             W x dt x penalty scale x the discounted term, computed from the views each imagined step starts and ends in
"""

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import test_finetune as tf  # noqa: E402

from xsoma import task, train  # noqa: E402


def test_touchdown():
    h, dz = jnp.asarray([[0.01, 0.01], [0.05, 0.0]]), jnp.asarray([[-0.006, 0.004], [-0.01, 0.0]])
    v = 0.006 / task.DT
    assert np.allclose(task.touchdown_speed(h, dz, 0.03), [v * v, 0.0], rtol=1e-5)
    g = jax.grad(lambda x: task.touchdown_speed(h, x, 0.03).sum())(dz)
    assert np.allclose(g, [[-2 * v / task.DT, 0.0], [0.0, 0.0]], rtol=1e-5)
    print(f"touchdown: a foot 1 cm up moving down {v:.2f} m/s costs {v * v:.3f}; rising, standing or 5 cm up: 0")


def test_accel():
    v0, v1 = jnp.zeros((3, 3)), jnp.asarray([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1], [0.5, 0.0, 0.0]])
    a = 0.1 / task.DT / task.G
    assert np.allclose(task.impact_accel(v0, v1), [a * a, 0.0, 0.0], rtol=1e-5)
    print(f"accel: +0.1 m/s in one step = {a:.2f} g costs {a * a:.3f}; down or sideways: 0")


def check_standin(name, fn, opts, terms_of):
    """The loss with `opts` minus the loss without equals W x dt x scale x the discounted terms that `terms_of(views)` rebuilds
    from the recorded views (start, end) of each imagined step, and the term `fn` is computed only with the option."""
    W = next(iter(opts.values()))
    with mock.patch.multiple(task, **tf.ROBOT), mock.patch.object(tf.FakeSim, "sole_frame", np.asarray([2, 3]), create=True):  # a sole point per foot
        plain = tf.build(action_noise=0.0)
        carry, step = plain.init(jax.random.PRNGKey(0)), jax.jit(plain.collect_step)
        for k in jax.random.split(jax.random.PRNGKey(1), tf.STEPS):
            carry, _ = step(carry, k)
        start = plain.sample_starts(carry["buf"], jax.random.PRNGKey(2))
        args = (carry["pi"], carry["wm"], carry["wm_b"], start, jax.random.split(jax.random.PRNGKey(3), plain.cfg.horizon), carry["env"].scale, carry["env"].world.heights)
        log = {"term": [], "end": [], "view": []}

        def record(key, f):
            def g(*a, **kw):
                log[key].append((a, f(*a, **kw)))
                return log[key][-1][1]

            return g

        with jax.disable_jit(), mock.patch.object(task, fn, record("term", getattr(task, fn))):
            loss0 = float(plain.imagine(*args)[0])
        assert not log["term"], "weight 0: the term may not be computed"
        pen = tf.build(action_noise=0.0, **opts)
        with jax.disable_jit(), mock.patch.object(task, fn, record("term", getattr(task, fn))), \
                mock.patch.object(train.XSoma, "termination", record("end", train.XSoma.termination)), \
                mock.patch.object(pen.sim, "view_state", record("view", pen.sim.view_state)):  # fmt: skip
            loss = float(pen.imagine(*args)[0])
        views = [v for _, v in log["view"]]  # the start's view, then each step's end view
        terms = np.stack([t for _, t in log["term"]])
        assert len(terms) == pen.cfg.horizon and len(views) == pen.cfg.horizon + 1
        assert np.allclose(terms, np.stack([terms_of(pen, views[i], views[i + 1]) for i in range(pen.cfg.horizon)]), atol=1e-6)
        alive = np.cumprod(1 - np.stack([x for _, x in log["end"]]), 0)
        disc = (pen.cfg.gamma ** np.arange(pen.cfg.horizon))[:, None] * np.concatenate((np.ones_like(alive[:1]), alive[:-1]))
        want = W * task.DT * float(args[5]) * (disc * terms).sum(0).mean()
        assert want > 1e-6 and np.isclose(loss - loss0, want, rtol=1e-3), (loss, loss0, want)
        pen.init(jax.random.PRNGKey(0))  # sets the optimizer's masks
        sub, st = jax.jit(pen.policy_update)(carry, start, jax.random.PRNGKey(4))
        assert np.isfinite(float(st["gn"])) and float(st["gn"]) > 0 and np.abs(tf.leaves(sub["pi"]) - tf.leaves(carry["pi"])).max() > 0
    print(f"standin {name}: {opts}: the loss gains weight x dt x scale x the discounted term ({want:.5f}); {(terms > 0).mean():.2f} of the "
          f"imagined steps have a term; compiled: gradient norm {float(st['gn']):.3f}")


def test_standin_touchdown():
    def terms_of(pen, v, v1):  # the feet's lowest sole points at the step's end, their contact points' vertical displacement
        low = np.where(pen.sole_foot, np.asarray(v1.soles)[..., None, :], np.inf).min(-1)
        dz = np.asarray(v1.robot.foot_p - v.robot.foot_p)[..., 2]
        return (np.where(low < 0.5, np.maximum(0.0, -dz / task.DT) ** 2, 0.0)).sum(-1)

    check_standin("touchdown", "touchdown_speed", {"touchdown_speed": 3.0, "touchdown_height": 0.5}, terms_of)


def test_standin_accel():
    def terms_of(pen, v, v1):
        return np.maximum(0.0, np.asarray(v1.robot.v - v.robot.v)[:, 2] / (task.DT * task.G)) ** 2

    check_standin("accel", "impact_accel", {"impact_accel": 3.0}, terms_of)


if __name__ == "__main__":
    assert not os.environ.get("XSOMA_ROBOT_DIR"), "run this test without XSOMA_ROBOT_DIR"
    test_touchdown()
    test_accel()
    test_standin_touchdown()
    test_standin_accel()
    print("\nall checks passed")
