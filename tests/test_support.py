"""The rope that holds the robot up at the start of an episode (run: python tests/test_support.py, or pytest): task.support_force,
task.support_pay_out, sim.Sim._support and train.py's --support. CPU only.

  rope      the tension K (top - h) - C vz below the top, zero above it and never negative; a pay-out step lowers the top by
            rate x dt and takes the rope off (top SUPPORT_OFF, rate 0) once it is SUPPORT_SLACK below the standing height
  reset     the real Sim (bundled G1, plane): about `support` of the fresh envs start on the rope, at rest in the default pose,
            upright, without a command, the lowest sole point max(gap, 0) over the plane, the top gap + the stretch under the
            nominal weight, the hold in SUPPORT_HOLD; the others are left as drawn. On MuJoCo's CPU engine with the joints held
            at the default pose and the rope's tension on the torso before every 5 ms step: hanging with a gap of 3 cm, the soles
            stay 1 to 3.5 cm over the plane (it swings and tilts on the rope) and the rope carries the weight; at a gap of minus half the stretch the feet stand and
            the rope carries part of the weight. (MuJoCo Warp's physics does not run on the CPU: Sim.physics itself is not run.)
  standin   on test_finetune's stand-in with a context whose last two entries are a rope: imagination's starts on the rope get a
            zero command (and the standing gait phase), every imagined step's world model sees the rope paid out as
            support_pay_out does, and on the rope the commands stay zero at a resample; without the option the context stays
            as sampled; the compiled update runs
"""

import dataclasses
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from xsoma import task  # noqa: E402

K, C, DT, OFF, SLACK, SPEED = task.SUPPORT_K, task.SUPPORT_C, task.DT, task.SUPPORT_OFF, task.SUPPORT_SLACK, task.SUPPORT_SPEED


def test_rope():
    top, h, vz = jnp.asarray([1.0, 1.0, 1.0, 1.0]), jnp.asarray([0.99, 1.01, 0.99, 0.99]), jnp.asarray([0.0, 0.0, 0.2, -0.1])
    assert np.allclose(task.support_force(top, h, vz), [K * 0.01, 0.0, max(0.0, K * 0.01 - C * 0.2), K * 0.01 + C * 0.1], rtol=1e-4)
    t, r = task.support_pay_out(jnp.asarray([0.03, -SLACK + 0.0005, OFF, 0.02]), jnp.asarray([SPEED, SPEED, 0.0, 0.0]))
    assert np.allclose(t, [0.03 - SPEED * DT, OFF, OFF, 0.02]) and np.allclose(r, [SPEED, 0.0, 0.0, 0.0]), (t, r)
    print(f"rope: tension K (top - h) - C vz, zero above the top and never negative; pay-out {SPEED * DT * 1000:.1f} mm per step, off {SLACK} m below")


def test_reset():
    import mujoco

    from xsoma.sim import Sim, SimCfg

    N, share, gap_range = 1000, 0.3, (-0.02, 0.05)
    sim = Sim(SimCfg(num_envs=N, terrain="plane", support=share, support_gap=gap_range))
    qpos = jax.random.uniform(jax.random.PRNGKey(0), (N, sim.m.nq))
    qvel, cmd = jnp.ones((N, sim.m.nv)), jnp.ones((N, 3))
    q, v, c, top, wait = (np.asarray(x) for x in sim._support(jax.random.PRNGKey(1), qpos, qvel, cmd, jnp.zeros((1, 1))))
    on = top > OFF
    gap = top[on] - sim.hang
    assert abs(on.mean() - share) < 0.05 and (gap >= gap_range[0] - 1e-6).all() and (gap <= gap_range[1] + 1e-6).all(), (on.mean(), gap.min(), gap.max())
    assert np.allclose(q[on, 2], sim.stand_h + np.maximum(gap, 0)) and np.allclose(q[on, 7:], task.DEFAULT_Q) and np.allclose(q[on, :2], qpos[on, :2])
    assert (v[on] == 0).all() and (c[on] == 0).all() and np.array_equal(q[~on], qpos[~on]) and (v[~on] == 1).all() and (c[~on] == 1).all()
    assert (wait >= 0).all() and (wait <= task.SUPPORT_HOLD[1] / DT).all()

    m = sim.m
    d = mujoco.MjData(m)
    soles, floor = list(sim.sole_geoms), [i for i in range(m.ngeom) if m.geom_bodyid[i] == 0]
    weight = m.body_mass.sum() * 9.81

    def hang(gap, seconds=2.0):
        """The default pose held by the PD law, the rope's top where it holds the nominal robot at `gap`; returns the lowest
        sole point over the plane and the rope's tension at the end."""
        mujoco.mj_resetData(m, d)
        d.qpos[2], d.qpos[3:7], d.qpos[7:] = sim.stand_h + max(gap, 0.0), [1, 0, 0, 0], task.DEFAULT_Q
        top = sim.stand_h + gap + sim.hang
        for _ in range(round(seconds / m.opt.timestep)):
            d.ctrl[:] = task.torques(d.qpos[7:], d.qvel[6:], np.zeros(task.J), 1.0, 1.0)
            d.xfrc_applied[sim.torso, 2] = tension = float(task.support_force(top, d.qpos[2], d.qvel[2]))
            mujoco.mj_step(m, d)
        mujoco.mj_kinematics(m, d)
        low = min(mujoco.mj_geomDistance(m, d, g, f, 1.0, None) for g in soles for f in floor)
        return low, tension

    assert abs(m.opt.timestep - 0.005) < 1e-9
    low, t_hang = hang(0.03)
    assert 0.01 < low < 0.035 and abs(t_hang / weight - 1) < 0.03, (low, t_hang / weight)  # the robot tilts on the rope, the joints give a little
    low2, t_part = hang(-sim.hang / 2)
    assert low2 < 0.003 and 0.1 < t_part / weight < 0.9, (low2, t_part / weight)
    print(f"reset: {on.mean():.3f} of {N} fresh envs on the rope (share {share}), gaps {gap.min():+.3f} to {gap.max():+.3f} m; standing height "
          f"{sim.stand_h:.3f} m, stretch {sim.hang * 1000:.1f} mm; CPU physics, 2 s: gap 3 cm -> soles {low * 100:.2f} cm up, rope "
          f"{t_hang / weight:.2f} x weight; gap -{sim.hang * 500:.1f} mm -> soles {low2 * 100:.2f} cm, rope {t_part / weight:.2f} x weight")


def test_standin():
    import test_finetune as tf

    from xsoma import train

    tops = np.asarray([OFF, 0.03, -SLACK + 0.0015, 0.0, OFF, 0.01], np.float32)  # per env: the rope's top and its pay-out speed
    rates = np.asarray([0.0, 0.0, SPEED, SPEED, 0.0, 0.0], np.float32)
    assert len(tops) == tf.N

    class RopeSim(tf.FakeSim):
        def context(self, s):
            return jnp.concatenate((s.kp, jnp.broadcast_to(jnp.asarray(tops)[:, None], (self.N, 1)), jnp.broadcast_to(jnp.asarray(rates)[:, None], (self.N, 1))), -1)

    def build(**options):
        with mock.patch.object(train, "Sim", RopeSim):
            return train.XSoma(dataclasses.replace(train.Config(), **tf.SMALL, **options))

    with mock.patch.multiple(task, **tf.ROBOT):
        out = {}
        for support in (0.0, 0.5):
            xs = build(action_noise=0.0, support=support, cmd_fresh=0.0)
            carry, step = xs.init(jax.random.PRNGKey(0)), jax.jit(xs.collect_step)
            for k in jax.random.split(jax.random.PRNGKey(1), tf.STEPS):
                carry, _ = step(carry, k)
            start = xs.sample_starts(carry["buf"], jax.random.PRNGKey(2))
            xi0 = np.asarray(start["xi"])
            seen = {"wm": [], "obs": []}

            def record(name, f):
                def g(*a, **kw):
                    seen[name].append(np.asarray(a[6] if name == "wm" else a[2]))
                    return f(*a, **kw)

                return g

            args = (carry["pi"], carry["wm"], carry["wm_b"], start, jax.random.split(jax.random.PRNGKey(3), xs.cfg.horizon), carry["env"].scale, carry["env"].world.heights)
            with jax.disable_jit(), mock.patch.object(task, "RESAMPLE", 2), mock.patch.object(train.nets, "wm_step", record("wm", train.nets.wm_step)), \
                    mock.patch.object(task, "observations", record("obs", task.observations)):  # fmt: skip
                xs.imagine(*args)
            out[support] = (xi0, np.asarray(start["cmd"]), np.asarray(start["phase"]), seen)
            if support:
                xs.init(jax.random.PRNGKey(0))  # sets the optimizer's masks
                sub, st = jax.jit(xs.policy_update)(carry, start, jax.random.PRNGKey(4))
                assert np.isfinite(float(st["gn"])) and np.abs(tf.leaves(sub["pi"]) - tf.leaves(carry["pi"])).max() > 0

    xi0, cmd, phase, seen = out[0.0]
    assert all(np.array_equal(x, xi0) for x in seen["wm"]), "without --support the context stays as sampled"
    assert np.abs(cmd).max() > 0.1
    xi0, cmd, phase, seen = out[0.5]
    on = xi0[:, -2] > OFF
    assert 0 < on.sum() < len(on) and (cmd[on] == 0).all() and np.allclose(phase[on], task.STAND_PHASE) and np.abs(cmd[~on]).max() > 0.1
    top, rate, off_seen = xi0[:, -2].copy(), xi0[:, -1].copy(), False
    for k, (x, c) in enumerate(zip(seen["wm"], seen["obs"])):
        assert np.allclose(x[:, -2], top, atol=1e-6) and np.allclose(x[:, -1], rate) and np.array_equal(x[:, :-2], xi0[:, :-2]), k
        live = top > OFF
        assert (c[live] == 0).all() and (k < 2 or np.abs(c[~live]).max() > 0.1), k  # resamples every 2nd step (RESAMPLE patched)
        top, rate = top - rate * DT, rate
        rate, top = np.where(top < -SLACK, 0.0, rate), np.where(top < -SLACK, OFF, top)
        off_seen |= bool(((xi0[:, -2] > OFF) & (top == OFF)).any())
    assert off_seen, "a rope taken off during the rollout"
    print(f"standin: {on.sum()} of {len(on)} starts on the rope get a zero command and the standing phase; over {len(seen['wm'])} imagined steps the "
          f"world model sees the rope paid out ({SPEED * DT * 1000:.1f} mm per step, taken off below -{SLACK} m) and the commands on it stay zero "
          f"at resamples; without the option the context is constant; compiled update: gradient norm {float(st['gn']):.3f}")


if __name__ == "__main__":
    test_rope()
    test_reset()
    test_standin()
