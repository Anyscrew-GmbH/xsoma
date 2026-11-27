"""The balance, cadence and step-length penalties (run: python tests/test_balance_step.py, or pytest). CPU only.

  balance      task.balance: zero over the segment between the feet; grows past the margin; a step under the pelvis lowers it
  walk_step    task.walk_step: stance beyond the threshold under any nonzero command, zero while standing
  step_length  task.step_length: zero within base + per_speed x |command|, the squared excess beyond; the width is not charged
  stance_carry task.stance_clock counts a standing foot's time and resets a lifted foot, a reset or a new command; with
               --stance_carry the replay buffer and imagination's starts hold the clocks, a redrawn command zeroes them, and a
               start's clock past the stance limit costs walk_step from the first imagined step
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from xsoma import task  # noqa: E402


def robot(p, left, right, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    R = jnp.asarray([[[c, -s, 0], [s, c, 0], [0, 0, 1]]], jnp.float32)
    feet = jnp.asarray([[left, right]], jnp.float32)
    return task.Robot(p=jnp.asarray([p], jnp.float32), R=R, v=jnp.zeros((1, 3)), w=jnp.zeros((1, 3)), q=jnp.zeros((1, task.J)),
                      qd=jnp.zeros((1, task.J)), foot_p=feet, feet_p=feet, feet_R=jnp.tile(jnp.eye(3), (1, 2, 1, 1)))


def test_balance():
    over = robot([0.0, 0.0, 0.7], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0])
    ahead = robot([0.2, 0.0, 0.7], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0])
    stepped = robot([0.2, 0.0, 0.7], [0.25, 0.1, 0.0], [0.0, -0.1, 0.0])  # the left foot stepped ahead of the pelvis
    assert float(task.balance(over, 0.03)[0]) == 0.0
    assert np.isclose(float(task.balance(ahead, 0.03)[0]), 0.17**2, atol=1e-5)
    assert float(task.balance(stepped, 0.03)[0]) < float(task.balance(ahead, 0.03)[0])
    g = jax.grad(lambda x: task.balance(robot([x, 0.0, 0.7], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0]), 0.03)[0])(0.2)
    assert g > 0, g
    print("balance: 0 between the feet, (d - margin)^2 ahead of both, less after a step under the pelvis; gradient finite")


def test_walk_step():
    st = jnp.asarray([[0.6, 0.1], [0.6, 0.1]])
    cmd = jnp.asarray([[0.05, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(task.walk_step(st, cmd, 0.4), [0.2, 0.0], atol=1e-6)
    print("walk_step: any nonzero command charges the stance excess at full weight, standing is free")


def test_step_length():
    r = robot([0.0, 0.0, 0.7], [0.3, 0.1, 0.0], [0.0, -0.1, 0.0])  # 0.3 fore-aft, 0.2 lateral (= width)
    slow, fast = jnp.asarray([[0.1, 0.0, 0.0]]), jnp.asarray([[0.6, 0.0, 0.0]])
    assert np.isclose(float(task.step_length(r, slow, 0.1, 0.5, 0.2)[0]), (0.3 - 0.15) ** 2, atol=1e-5)
    assert float(task.step_length(r, fast, 0.1, 0.5, 0.2)[0]) == 0.0
    turned = robot([0.0, 0.0, 0.7], [0.1, 0.3, 0.0], [-0.1, 0.0, 0.0], yaw=np.pi / 2)  # the same feet seen from a pelvis facing +y
    assert np.isclose(float(task.step_length(turned, slow, 0.1, 0.5, 0.2)[0]), (0.3 - 0.15) ** 2, atol=1e-5)
    print("step_length: the fore-aft spread past base + per_speed x |vx| is charged, in the heading frame; within it free")


def test_standin():
    """On test_finetune's stand-in: with weight W the loss gains W x dt x scale x the discounted term, from the imagined views."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_impact as ti

    ti.check_standin("balance", "balance", {"balance": 3.0, "balance_margin": 0.0}, lambda pen, v, v1: np.asarray(task.balance(v1.robot, 0.0)))
    zero = np.zeros((1, 3), np.float32)  # step_per_speed 0: the term does not depend on the command
    ti.check_standin("step_length", "step_length", {"step_length": 3.0, "step_base": 0.0, "step_per_speed": 0.0, "step_width": 0.0},
                     lambda pen, v, v1: np.asarray(task.step_length(v1.robot, zero, 0.0, 0.0, 0.0)))


def test_stance_carry():
    st = jnp.asarray([[0.3, 0.3], [0.3, 0.3]])
    h = jnp.asarray([[0.0, 0.05], [0.0, 0.0]])  # env 0: the right foot lifted; env 1: both stand, but a new command
    assert np.allclose(task.stance_clock(st, h, jnp.asarray([True, False])), [[0.3 + task.DT, 0.0], [0.0, 0.0]], atol=1e-6)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from unittest import mock

    import test_finetune as tf

    with mock.patch.multiple(task, **tf.ROBOT), mock.patch.object(tf.FakeSim, "sole_frame", np.asarray([2, 3]), create=True), \
            mock.patch.object(task, "TURN_STEP_HEIGHT", 10.0):  # fmt: skip  (every foot of the stand-in stands)
        xs = tf.build(action_noise=0.0, walk_step=3.0, turn_step_stance=0.5, stance_carry=1, cmd_fresh=0.0)
        carry, step = xs.init(jax.random.PRNGKey(0)), jax.jit(xs.collect_step)
        for k in jax.random.split(jax.random.PRNGKey(1), tf.STEPS):
            carry, _ = step(carry, k)
        clk = np.asarray(carry["buf"]["stance"])
        assert clk.shape[-1] == 2 and clk.max() > 2 * task.DT and np.allclose(clk / task.DT, np.round(clk / task.DT), atol=1e-3), clk.max()
        start = xs.sample_starts(carry["buf"], jax.random.PRNGKey(2))
        assert start["stance"].shape == (xs.cfg.pi_batch, 2)
        args = (carry["pi"], carry["wm"], carry["wm_b"])
        rest = (jax.random.split(jax.random.PRNGKey(3), xs.cfg.horizon), carry["env"].scale, carry["env"].world.heights)
        moving = {**start, "cmd": jnp.full_like(start["cmd"], 0.3)}
        loss = [float(xs.imagine(*args, {**moving, "stance": jnp.full_like(start["stance"], x)}, *rest)[0]) for x in (0.0, 5.0)]
        # 5 s on the clock: each foot is charged 4.5 s and more from the first step (stance 0.5); from zero: nothing in 5 steps
        assert loss[1] > loss[0] + 3.0 * task.DT * 0.9, loss
        fresh = tf.build(action_noise=0.0, walk_step=3.0, turn_step_stance=0.5, stance_carry=1, cmd_fresh=1.0)
        assert float(jnp.abs(fresh.sample_starts(carry["buf"], jax.random.PRNGKey(2))["stance"]).max()) == 0.0
    print(f"stance_carry: clocks count and reset; buffer holds them (max {clk.max():.2f} s), a long clock costs from the first "
          f"imagined step ({loss[0]:.3f} -> {loss[1]:.3f}), a redrawn command zeroes it")


if __name__ == "__main__":
    test_balance()
    test_walk_step()
    test_step_length()
    test_standin()
    test_stance_carry()
