"""The KL bound to the initial tracker (train.py --kl_bound, --kl_beta, --kl_beta_lr) and the joint-speed penalty (--joint_vel), on
test_finetune's stand-in (run: python tests/test_kl_bound.py, or pytest). CPU only.

  off        without --kl_bound no weight is kept and no KL is computed; without --anchor and --kl_bound no reference policy
  kl         imagination's KL, rebuilt from the recorded actions of the policy and of the initial tracker on the same inputs and
             the recorded terminations: the survival-weighted mean of |a - a0|^2 / (2 action_noise^2); zero at the initial
             tracker; the loss gains kl_beta x max(0, KL - eps), nothing for a bound above the KL
  dual       a compiled policy update moves the weight by kl_beta_lr x clip(KL / eps - 1, -1, 1), never below zero, not after a
             step that the gradient gate skips, and keeps it with kl_beta_lr 0; training's programs log kl, kl_term and kl_beta
  joint_vel  without the option the term is not computed; with weight W the loss gains W x dt x penalty scale x the discounted sum
             of the squared joint speeds of each imagined step's end state
"""

import dataclasses
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
import test_impact  # noqa: E402

from xsoma import task, tracker, train  # noqa: E402


def setup(d, **opts):
    """A tracker fine-tune on the stand-in with `opts`, its carry after the collection, a batch of starts and imagine's arguments."""
    tf.make(d)
    xs = tf.build(init_tracker=os.path.join(d, "ckpt.npz"), tracker_meta=os.path.join(d, "meta.json"), gait_clock=0, pi_gn_gate=0.0, **opts)
    carry, start, _ = tf.collect(xs)
    args = (carry["pi"], carry["wm"], carry["wm_b"], start, jax.random.split(jax.random.PRNGKey(3), xs.cfg.horizon), carry["env"].scale, carry["env"].world.heights)
    return xs, carry, start, args


def rebuild(xs, **opts):
    """The learner `xs` with other options, on the stand-in."""
    with mock.patch.object(train, "Sim", tf.FakeSim):
        return train.XSoma(dataclasses.replace(xs.cfg, **opts))


def moved(pi, k=0.05):
    """The policy with noise on every weight: a policy away from the initial one."""
    leaves, tree = jax.tree.flatten(pi)
    keys = jax.random.split(jax.random.PRNGKey(7), len(leaves))
    return jax.tree.unflatten(tree, [x + k * jax.random.normal(kk, x.shape) for x, kk in zip(leaves, keys)])


def check_off(d):
    xs, carry, _, args = setup(d)
    assert "pi0" not in carry and "kl_beta" not in carry and "kl" not in xs.imagine(*args)[1]
    xs, carry, _, _ = setup(d, anchor=1.0)
    assert "pi0" in carry and "kl_beta" not in carry
    print("off: no reference policy without --anchor and --kl_bound, no KL weight without --kl_bound")


def check_kl(d):
    eps, beta, sigma = 0.01, 2.0, 0.3
    xs, carry, start, args = setup(d, kl_bound=eps, kl_beta=beta, action_noise=sigma)
    assert "pi0" in carry and np.isclose(float(carry["kl_beta"]), beta)
    plain = float(xs.imagine(*args)[0])
    at0 = xs.imagine(*args, pi0=carry["pi0"], kl_beta=carry["kl_beta"])
    assert float(at0[1]["kl"]) == 0 and float(at0[1]["kl_term"]) == 0 and np.isclose(float(at0[0]), plain, rtol=1e-6)
    pi = moved(carry["pi"])
    args1 = (pi,) + args[1:]
    acts, ends, act, term = [], [], tracker.act, train.XSoma.termination

    def rec_act(p, hist, g, **kw):
        acts.append(np.asarray(act(p, hist, g, **kw)))
        return acts[-1]

    def rec_end(self, clearance):
        ends.append(np.asarray(term(self, clearance)))
        return ends[-1]

    with jax.disable_jit(), mock.patch.object(tracker, "act", rec_act), mock.patch.object(train.XSoma, "termination", rec_end):
        loss, st = xs.imagine(*args1, pi0=carry["pi0"], kl_beta=carry["kl_beta"])
    # per step three calls: the policy's action, then the initial tracker's and the policy's on the inputs held constant
    a0, a = np.stack(acts[1::3]), np.stack(acts[2::3])
    alive = np.cumprod(1 - np.stack(ends), 0)
    w = np.concatenate((np.ones_like(alive[:1]), alive[:-1]))
    kl = (w * ((a - a0) ** 2).sum(-1)).sum() / w.sum() / (2 * sigma**2)
    plain1 = float(xs.imagine(*args1)[0])
    assert kl > eps and np.isclose(float(st["kl"]), kl, rtol=1e-4), (float(st["kl"]), kl)
    assert np.isclose(float(loss) - plain1, beta * (kl - eps), rtol=1e-3) and np.isclose(float(st["kl_term"]), beta * (kl - eps), rtol=1e-4)
    loose = rebuild(xs, kl_bound=2 * kl)
    loose.init(jax.random.PRNGKey(0))
    l2, s2 = loose.imagine(*args1, pi0=carry["pi0"], kl_beta=carry["kl_beta"])
    assert float(s2["kl_term"]) == 0 and np.isclose(float(l2), plain1, rtol=1e-6)
    print(f"kl: 0 at the initial tracker; {kl:.4f} after moving the weights, = the survival-weighted mean of |a - a0|^2 / (2 x {sigma}^2) of the "
          f"recorded actions; the loss gains {beta} x (KL - {eps}), nothing with the bound at 2 x KL")


def check_dual(d):
    eps, lr = 0.01, 0.5
    for beta in (2.0, 0.1):
        xs, carry, start, _ = setup(d, kl_bound=eps, kl_beta=beta, kl_beta_lr=lr)
        xs.init(jax.random.PRNGKey(0))  # sets the optimizer's masks
        carry = {**carry, "pi": moved(carry["pi"])}
        out, st = jax.jit(xs.policy_update)(carry, start, jax.random.PRNGKey(4))
        want = max(0.0, beta + lr * min(1.0, float(st["kl"]) / eps - 1))
        assert np.isclose(float(st["kl_beta"]), beta) and np.isclose(float(out["kl_beta"]), want, rtol=1e-5), (float(out["kl_beta"]), want)
        gated = rebuild(xs, pi_gn_gate=1.0)  # a step the gradient gate skips leaves the weight as it is
        gated.init(jax.random.PRNGKey(0))
        out, st = jax.jit(gated.policy_update)({**carry, "gn_ema": jnp.float32(1e-6)}, start, jax.random.PRNGKey(4))
        assert float(st["gn_gated"]) == 1 and np.isclose(float(out["kl_beta"]), beta)
        far = rebuild(xs, kl_bound=1e3)  # KL far under the bound: the weight shrinks, not below zero
        far.init(jax.random.PRNGKey(0))
        out, st = jax.jit(far.policy_update)(carry, start, jax.random.PRNGKey(4))
        assert np.isclose(float(out["kl_beta"]), max(0.0, beta - lr * (1 - float(st["kl"]) / 1e3)), rtol=1e-5, atol=1e-7)
    xs, carry, _, _ = setup(d, kl_bound=eps, kl_beta=beta)  # fixed weight
    progs = xs.compile_programs(carry)
    carry, _ = progs["collect"](carry)
    carry, _ = progs["wm"](carry)
    carry, st1 = progs["pi"](carry)
    carry, st2 = progs["pi"](carry)
    assert {"kl", "kl_term", "kl_beta", "anchor"} <= set(st2) and all(np.isfinite(float(v)) for v in st2.values())
    assert float(st1["kl"]) == 0 and float(st2["kl"]) > 0 and np.isclose(float(carry["kl_beta"]), beta)
    print(f"dual: the weight moves by kl_beta_lr x clip(KL / eps - 1, -1, 1) up and down, not below zero, not after a gated step, and stays without kl_beta_lr; training's programs "
          f"log kl {float(st2['kl']):.2e} after one update, kl_term, kl_beta")


def test_kl_bound():
    with mock.patch.multiple(task, **tf.ROBOT), tempfile.TemporaryDirectory() as d:
        check_off(d)
        check_kl(d)
        check_dual(d)


def test_joint_vel():
    def terms_of(pen, v, v1):
        return (np.asarray(v1.robot.qd) ** 2).sum(-1)

    test_impact.check_standin("joint_vel", "joint_vel", {"joint_vel": 3.0}, terms_of)


if __name__ == "__main__":
    assert not os.environ.get("XSOMA_ROBOT_DIR"), "run this test without XSOMA_ROBOT_DIR"
    test_kl_bound()
    test_joint_vel()
    print("\nall checks passed")
