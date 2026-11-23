"""The torque-limit penalty (run: python tests/test_torque_limit.py, or pytest): task.torque_limit and train.py's --torque_limit
and --torque_soft. CPU only.

  term      task.torque_limit on hand-made states and targets: per joint max(0, |KP (target - q) - KD qd| / EFFORT - soft), of
            either sign of the torque, zero below the soft limit; a finite gradient, also at zero torque
  standin   on test_finetune's stand-in: without the option the term is not computed; with --torque_limit W the loss gains
            W x dt x penalty scale x the discounted term summed over the joints, the term taken from the state each imagined
            step starts from and its PD target; the compiled update runs
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

GAINS = dict(KP=np.full(tf.J, 20.0, np.float32), KD=np.full(tf.J, 1.0, np.float32), EFFORT=np.full(tf.J, 2.0, np.float32))


def test_term():
    with mock.patch.multiple(task, KP=np.asarray([10.0, 20.0], np.float32), KD=np.asarray([1.0, 2.0], np.float32), EFFORT=np.asarray([5.0, 10.0], np.float32)):
        q, qd = jnp.asarray([[0.0, 0.1], [0.2, 0.0]]), jnp.asarray([[0.0, 3.0], [-6.0, 0.0]])
        target = jnp.asarray([[0.5, 0.0], [0.0, 0.43]])
        # torques: row 0: 5 (= the limit), 20 x -0.1 - 6 = -8 (0.8 of it); row 1: -2 + 6 = 4 (0.8), 8.6 (0.86)
        assert np.allclose(task.torque_limit(q, qd, target), [[0.1, 0.0], [0.0, 0.0]], atol=1e-5)
        assert np.allclose(task.torque_limit(q, qd, target, 0.5), [[0.5, 0.3], [0.3, 0.36]], atol=1e-5)
        g = jax.grad(lambda t: task.torque_limit(q, qd, t, 0.5).sum())(target)
        assert np.allclose(g, [[10 / 5, -20 / 10], [10 / 5, 20 / 10]], atol=1e-5), g  # d|tau| / d target / EFFORT, signed as tau
        assert np.isfinite(jax.grad(lambda t: task.torque_limit(0 * q, 0 * qd, t, 0.5).sum())(0 * target)).all()
    print("term: max(0, |tau| / EFFORT - soft) per joint for torques of either sign, zero below the soft limit, finite gradient at zero torque")


def test_standin():
    W = 3.0
    with mock.patch.multiple(task, **tf.ROBOT, **GAINS):
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

        with jax.disable_jit(), mock.patch.object(task, "torque_limit", record("term", task.torque_limit)):
            loss0 = float(plain.imagine(*args)[0])
        assert not log["term"], "weight 0: the term may not be computed"
        pen = tf.build(action_noise=0.0, torque_limit=W, torque_soft=0.5)
        with jax.disable_jit(), mock.patch.object(task, "torque_limit", record("term", task.torque_limit)), \
                mock.patch.object(train.XSoma, "termination", record("end", train.XSoma.termination)), mock.patch.object(train.nets, "wm_step", record("wm", train.nets.wm_step)):  # fmt: skip
            loss = float(pen.imagine(*args)[0])
        # the term's inputs: the state the step starts from (the start, then each world-model step's result) and its PD target
        S = [train.unflatten(start["s_qpos"], start["s_qvel"])] + [s for _, s in log["wm"][:-1]]
        for ((q, qd, target, soft), _), (wa, _), s in zip(log["term"], log["wm"], S):
            assert soft == 0.5 and np.array_equal(target, wa[5]) and np.allclose(q, s[4]) and np.allclose(qd, s[5])
        terms = np.stack([t for _, t in log["term"]]).sum(-1)
        alive = np.cumprod(1 - np.stack([x for _, x in log["end"]]), 0)
        disc = (pen.cfg.gamma ** np.arange(pen.cfg.horizon))[:, None] * np.concatenate((np.ones_like(alive[:1]), alive[:-1]))
        want = W * task.DT * float(args[5]) * (disc * terms).sum(0).mean()
        assert len(terms) == pen.cfg.horizon and want > 1e-3 and np.isclose(loss - loss0, want, rtol=1e-3), (loss, loss0, want)
        pen.init(jax.random.PRNGKey(0))  # sets the optimizer's masks
        sub, st = jax.jit(pen.policy_update)(carry, start, jax.random.PRNGKey(4))
        assert np.isfinite(float(st["gn"])) and float(st["gn"]) > 0 and np.abs(tf.leaves(sub["pi"]) - tf.leaves(carry["pi"])).max() > 0
    print(f"standin: --torque_limit {W} --torque_soft 0.5: the loss gains weight x dt x scale x the discounted term ({want:.4f}), from each step's "
          f"start state and PD target; {(terms > 0).mean():.2f} of the imagined steps have a term; compiled: gradient norm {float(st['gn']):.3f}")


if __name__ == "__main__":
    assert not os.environ.get("XSOMA_ROBOT_DIR"), "run this test without XSOMA_ROBOT_DIR"
    test_term()
    test_standin()
    print("\nall checks passed")
