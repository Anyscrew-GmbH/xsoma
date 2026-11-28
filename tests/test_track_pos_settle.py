"""The displacement tracking and settle penalties (run: python tests/test_track_pos_settle.py, or pytest). CPU only.

  track_pos  task.track_pos_error / task.track_pos: standing still under a steady command costs about 1 after a few tau, walking
             it in bursts with pauses much less, a heading-aligned walk at the command nothing; with --track_pos the replay buffer
             holds collection's error, a redrawn command zeroes it, and a start's error costs from the first imagined step
  settle     task.settle: zero in the default stance and under any nonzero command, the stance error past the margin otherwise,
             lowered by a step back under the pelvis; in imagination it costs under a zero command, and with --stand_still a
             stance off the default waives the feet's motion
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
from test_balance_step import robot  # noqa: E402

from xsoma import task  # noqa: E402


def mean_cost(v_of_t, cmd, tau=1.0, rel=0.05, seconds=6.0, yaw=0.0):
    """The mean of track_pos over the last 3 s of a walk whose pelvis moves at v_of_t(t) (m/s, along its heading)."""
    d, cmd, costs = jnp.zeros((1, 2)), jnp.asarray([cmd], jnp.float32), []
    for i in range(int(seconds / task.DT)):
        r = robot([0.0, 0.0, 0.7], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0], yaw)
        v = v_of_t(i * task.DT)
        r = r._replace(v=jnp.asarray([[np.cos(yaw) * v, np.sin(yaw) * v, 0.0]], jnp.float32))
        d = task.track_pos_error(d, r, cmd, tau)
        costs.append(float(task.track_pos(d, cmd, tau, rel)[0]))
    return float(np.mean(costs[-int(3.0 / task.DT):]))


def test_track_pos():
    cmd = [0.1, 0.0, 0.0]
    still = mean_cost(lambda t: 0.0, cmd)
    bursts = mean_cost(lambda t: 0.2 if t % 1.0 < 0.5 else 0.0, cmd)  # 0.2 m/s for 0.5 s, then a 0.5 s pause: 0.1 m/s on average
    exact = mean_cost(lambda t: 0.1, cmd, yaw=2.0)  # at the command, the pelvis facing another way: the heading frame
    assert 0.9 < still < 1.0 and bursts < 0.3 * still and exact < 1e-3, (still, bursts, exact)
    # track_l1 with rel charges the same bursts as much as standing still (|0.2 - 0.1| = |0 - 0.1|)
    print(f"track_pos: standing still under 0.1 m/s costs {still:.2f}, bursts of 0.2 m/s with pauses {bursts:.2f}, the command itself {exact:.4f}")


def test_settle():
    ref = task.stance_offsets(robot([0.0, 0.0, 0.7], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0]))[0]
    zero, walk = jnp.zeros((1, 3)), jnp.asarray([[0.1, 0.0, 0.0]])
    home = robot([0.5, 0.2, 0.7], [0.5, 0.3, 0.0], [0.5, 0.1, 0.0])
    split = robot([0.1, 0.0, 0.7], [0.2, 0.1, 0.0], [0.0, -0.1, 0.0])  # staggered 0.2 m, pelvis over the midpoint
    assert float(task.settle(home, zero, ref, 0.03)[0]) < 1e-5
    assert np.isclose(float(task.settle(split, zero, ref, 0.03)[0]), 0.17, atol=1e-5)
    assert float(task.settle(split, walk, ref, 0.03)[0]) == 0.0
    turned = robot([0.0, 0.0, 0.7], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], yaw=np.pi / 2)  # the default stance seen from a pelvis facing +y
    assert float(task.settle(turned, zero, ref, 0.03)[0]) < 1e-5
    lean = robot([0.15, 0.0, 0.7], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0])  # pelvis 0.15 m ahead of square feet
    assert np.isclose(float(task.settle(lean, zero, ref, 0.03)[0]), 0.12, atol=1e-5)
    g = jax.grad(lambda x: task.settle(robot([0.1, 0.0, 0.7], [x, 0.1, 0.0], [0.0, -0.1, 0.0]), zero, ref, 0.03)[0])(0.2)
    assert g > 0, g  # bringing the front foot back lowers it
    ff = jnp.ones((1, 2, 3)) * 0.01
    assert float(task.stand_still(home, ff, zero, 0.0)[0]) < float(task.stand_still(home, ff, zero)[0])
    print("settle: 0 in the default stance (heading frame) and under a command; stagger and lean past the margin cost; gradient finite")


def standin(**options):
    """test_finetune's stand-in after collection: the XSoma, its carry and imagination's arguments."""
    xs = tf.build(action_noise=0.0, **options)
    carry, step = xs.init(jax.random.PRNGKey(0)), jax.jit(xs.collect_step)
    for k in jax.random.split(jax.random.PRNGKey(1), tf.STEPS):
        carry, _ = step(carry, k)
    start = xs.sample_starts(carry["buf"], jax.random.PRNGKey(2))
    rest = (jax.random.split(jax.random.PRNGKey(3), xs.cfg.horizon), carry["env"].scale, carry["env"].world.heights)
    return xs, carry, start, rest


def loss(xs, carry, start, rest):
    return float(xs.imagine(carry["pi"], carry["wm"], carry["wm_b"], start, *rest)[0])


def test_carry():
    with mock.patch.multiple(task, **tf.ROBOT):
        xs, carry, start, rest = standin(track_pos=3.0, cmd_fresh=0.0)
        buf = np.asarray(carry["buf"]["track_d"])
        assert buf.shape[-1] == 2 and np.abs(buf).max() > 0, buf.shape
        assert start["track_d"].shape == (xs.cfg.pi_batch, 2)
        moving = {**start, "cmd": jnp.full_like(start["cmd"], 0.3)}
        lo, hi = (loss(xs, carry, {**moving, "track_d": jnp.full_like(start["track_d"], x)}, rest) for x in (0.0, 5.0))
        assert hi > lo + 3.0 * task.DT * 5.0, (lo, hi)  # an error of 7 m against 0.3 m/s x 1 s: about 23 per step from the first
        sub, st = jax.jit(xs.policy_update)(carry, start, jax.random.PRNGKey(4))
        assert np.isfinite(float(st["gn"])) and float(st["gn"]) > 0
        fresh = tf.build(action_noise=0.0, track_pos=3.0, cmd_fresh=1.0)
        assert float(jnp.abs(fresh.sample_starts(carry["buf"], jax.random.PRNGKey(2))["track_d"]).max()) == 0.0
    print(f"track_pos carry: the buffer holds collection's error (max {np.abs(buf).max():.3f} m), a start's error costs from the first "
          f"imagined step ({lo:.3f} -> {hi:.3f}), a redrawn command zeroes it; gradient norm {float(st['gn']):.2e}")


def test_settle_imagined():
    with mock.patch.multiple(task, **tf.ROBOT):
        plain, carry, start, rest = standin(cmd_fresh=0.0)
        still = {**start, "cmd": jnp.zeros_like(start["cmd"])}
        base = loss(plain, carry, still, rest)
        pen = tf.build(action_noise=0.0, settle=3.0, settle_margin=0.0)
        assert loss(pen, carry, still, rest) > base + 1e-4
        assert np.isclose(loss(pen, carry, {**still, "cmd": jnp.full_like(start["cmd"], 0.3)}, rest),
                          loss(plain, carry, {**still, "cmd": jnp.full_like(start["cmd"], 0.3)}, rest), rtol=1e-5)
        pen.init(jax.random.PRNGKey(0))  # sets the optimizer's masks
        sub, st = jax.jit(pen.policy_update)(carry, start, jax.random.PRNGKey(4))
        assert np.isfinite(float(st["gn"])) and float(st["gn"]) > 0
        # stand_still's feet part is waived where the stance error is above the margin (settle's weight negligible here)
        ss = loss(tf.build(action_noise=0.0, stand_still=3.0), carry, still, rest)
        waived = loss(tf.build(action_noise=0.0, stand_still=3.0, settle=1e-9, settle_margin=0.0), carry, still, rest)
        kept = loss(tf.build(action_noise=0.0, stand_still=3.0, settle=1e-9, settle_margin=100.0), carry, still, rest)
        assert waived < ss - 1e-4 and np.isclose(kept, ss, rtol=1e-5), (ss, waived, kept)
    print(f"settle imagined: costs under a zero command only; stand_still without the feet past the margin ({ss:.3f} -> {waived:.3f}); "
          f"gradient norm {float(st['gn']):.2e}")


if __name__ == "__main__":
    test_track_pos()
    test_settle()
    test_carry()
    test_settle_imagined()
