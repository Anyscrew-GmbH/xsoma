"""The joint-range penalty and the slow-command entries of --cmd_scale (run: python tests/test_range_penalty.py, or pytest):
task.joint_range, train.py's --range_penalty and --range_joints, and task.sample_commands' sixth and seventh entries. CPU only.

  term      task.joint_range on hand-made poses: per joint the distance outside [lo, hi], zero inside, a finite gradient
  commands  a sixth and seventh entry (s, f): a share s of the commands is the same draw times f, the rest unchanged
  standin   on test_finetune's stand-in: without the option the term is not computed; with --range_penalty W the loss gains
            W x dt x penalty scale x the discounted term summed over the named joints, on the pose each imagined step ends in
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


def test_term():
    q, lo, hi = jnp.asarray([[0.3, -0.5], [0.1, 0.0]]), np.asarray([-0.2, -0.4], np.float32), np.asarray([0.2, 0.1], np.float32)
    assert np.allclose(task.joint_range(q, lo, hi), [[0.1, 0.1], [0.0, 0.0]], atol=1e-6)
    assert np.allclose(jax.grad(lambda x: task.joint_range(x, lo, hi).sum())(q), [[1.0, -1.0], [0.0, 0.0]])
    print("term: the distance outside [lo, hi] per joint, zero inside, gradient +-1 outside")


def test_commands():
    c5, f5 = task.sample_commands(jax.random.PRNGKey(5), 4000, (1.0, 0.5, 1.0, 0.2, 0.3))
    c7, f7 = task.sample_commands(jax.random.PRNGKey(5), 4000, (1.0, 0.5, 1.0, 0.2, 0.3, 0.4, 0.25))
    c5, c7 = np.asarray(c5), np.asarray(c7)
    scaled = ~np.isclose(c5, c7).all(-1)
    assert np.allclose(c7[scaled], 0.25 * c5[scaled]) and np.array_equal(f5, f7)
    assert 0.35 < scaled[(c5 != 0).any(-1)].mean() < 0.45
    print(f"commands: (0.4, 0.25) scales {scaled[(c5 != 0).any(-1)].mean():.2f} of the nonzero commands by 0.25, the rest unchanged")


def test_standin():
    W, SPEC = 3.0, "knee:-0.02:0.02,waist:-0.01:0.01"
    with mock.patch.multiple(task, **tf.ROBOT):
        plain = tf.build(action_noise=0.0)
        carry, step = plain.init(jax.random.PRNGKey(0)), jax.jit(plain.collect_step)
        for k in jax.random.split(jax.random.PRNGKey(1), tf.STEPS):
            carry, _ = step(carry, k)
        start = plain.sample_starts(carry["buf"], jax.random.PRNGKey(2))
        args = (carry["pi"], carry["wm"], carry["wm_b"], start, jax.random.split(jax.random.PRNGKey(3), plain.cfg.horizon), carry["env"].scale, carry["env"].world.heights)
        log = {"term": [], "end": [], "wm": []}

        def record(name, f):
            def g(*a, **kw):
                log[name].append((a, f(*a, **kw)))
                return log[name][-1][1]

            return g

        with jax.disable_jit(), mock.patch.object(task, "joint_range", record("term", task.joint_range)):
            loss0 = float(plain.imagine(*args)[0])
        assert not log["term"] and plain.rng is None, "weight 0: the term may not be computed"
        pen = tf.build(action_noise=0.0, range_penalty=W, range_joints=SPEC)
        assert pen.rng[0].tolist() == [1, 3] and np.allclose(pen.rng[1], [-0.02, -0.01]) and np.allclose(pen.rng[2], [0.02, 0.01])
        with jax.disable_jit(), mock.patch.object(task, "joint_range", record("term", task.joint_range)), \
                mock.patch.object(train.XSoma, "termination", record("end", train.XSoma.termination)), mock.patch.object(train.nets, "wm_step", record("wm", train.nets.wm_step)):  # fmt: skip
            loss = float(pen.imagine(*args)[0])
        for ((q, lo, hi), _), (_, s) in zip(log["term"], log["wm"]):  # the pose each step ends in, the named joints only
            assert np.allclose(q, s[4][:, [1, 3]]) and np.allclose(lo, [-0.02, -0.01])
        terms = np.stack([t for _, t in log["term"]]).sum(-1)
        alive = np.cumprod(1 - np.stack([x for _, x in log["end"]]), 0)
        disc = (pen.cfg.gamma ** np.arange(pen.cfg.horizon))[:, None] * np.concatenate((np.ones_like(alive[:1]), alive[:-1]))
        want = W * task.DT * float(args[5]) * (disc * terms).sum(0).mean()
        assert len(terms) == pen.cfg.horizon and want > 1e-4 and np.isclose(loss - loss0, want, rtol=1e-3), (loss, loss0, want)
        pen.init(jax.random.PRNGKey(0))  # sets the optimizer's masks
        sub, st = jax.jit(pen.policy_update)(carry, start, jax.random.PRNGKey(4))
        assert np.isfinite(float(st["gn"])) and float(st["gn"]) > 0 and np.abs(tf.leaves(sub["pi"]) - tf.leaves(carry["pi"])).max() > 0
    print(f"standin: --range_penalty {W} --range_joints {SPEC}: the loss gains weight x dt x scale x the discounted term ({want:.4f}) on the pose each "
          f"step ends in; {(terms > 0).mean():.2f} of the imagined steps have a term; compiled: gradient norm {float(st['gn']):.3f}")


if __name__ == "__main__":
    assert not os.environ.get("XSOMA_ROBOT_DIR"), "run this test without XSOMA_ROBOT_DIR"
    test_term()
    test_commands()
    test_standin()
    print("\nall checks passed")
