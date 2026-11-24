"""The step-rate penalty while turning and the friction surfaces (run: python tests/test_turn_friction.py, or pytest). CPU only.

  stance     task.turn_stance on hand-made sole heights: counts a standing foot's seconds, resets a lifted one, and its gradient
             says that raising a standing foot lowers its stance time
  turn_step  task.turn_step: |yaw command| x the stance beyond the threshold, zero without a yaw command
  standin    on test_finetune's stand-in: without --turn_step the term is not computed; with weight W the loss gains
             W x dt x scale x the discounted term, whose stance clocks are rebuilt from the imagined views (from zero per rollout)
  surfaces   Sim._mu with --friction_surfaces: every draw inside one of the ranges, each surface drawn about equally often;
             Sim._friction sets every geom (the soles only with sole_friction); Sim._new_surface redraws only the ended episodes
"""

import os
import sys
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import test_impact as ti  # noqa: E402

from xsoma import task  # noqa: E402
from xsoma.sim import Sim, SimCfg, World, surfaces  # noqa: E402


def test_stance():
    st = jnp.zeros((1, 2))
    for h in ([0.0, 0.05], [0.0, 0.0], [0.01, 0.0], [0.05, 0.0]):
        st = task.turn_stance(st, jnp.asarray([h]))
    assert np.allclose(st, [[0.0, 3 * task.DT]], atol=1e-6), st
    g = jax.grad(lambda h: task.turn_stance(jnp.asarray([[0.1, 0.1]]), h).sum())(jnp.asarray([[0.0, 0.05]]))
    assert np.isfinite(g).all() and g[0, 0] < 0, g
    print(f"stance: counts a standing foot ({float(st[0, 1]):.2f} s after 3 steps), resets a lifted one; d stance / d height {float(g[0, 0]):.1f}")


def test_turn_step():
    st = jnp.asarray([[0.5, 0.1], [0.5, 0.5]])
    assert np.allclose(task.turn_step(st, jnp.asarray([-0.5, 0.0]), 0.4), [0.5 * 0.1, 0.0], atol=1e-6)
    print("turn_step: |wz| x stance beyond the threshold; no yaw command, no cost")


def test_standin_turn_step():
    seen = []  # (stance, wz) of each recorded call of task.turn_step with the option on
    real = task.turn_step

    def spy(st, wz, stance):
        if not isinstance(st, jax.core.Tracer):  # the compiled update at the end traces it too
            seen.append((np.asarray(st), np.asarray(wz)))
        return real(st, wz, stance)

    st = {}

    def terms_of(pen, v, v1):  # the stance clocks from the views, hard counted, from zero at the rollout's start
        i = st.setdefault("i", 0)
        low = np.where(pen.sole_foot, np.asarray(v1.soles)[..., None, :], np.inf).min(-1)
        clock = (low < task.TURN_STEP_HEIGHT) * (st.get("clock", 0.0) + task.DT)
        st.update(i=i + 1, clock=clock)
        k = len(seen) - pen.cfg.horizon + i  # the option-on run's calls are the last horizon ones
        assert np.allclose(seen[k][0], clock, atol=1e-5)
        return np.abs(seen[k][1]) * np.maximum(0.0, clock - 0.02).sum(-1)

    with mock.patch.object(task, "turn_step", spy), mock.patch.object(task, "TURN_STEP_HEIGHT", 0.5):
        ti.check_standin("turn_step", "turn_step", {"turn_step": 3.0, "turn_step_stance": 0.02}, terms_of)


def fake_sim(n, surfaces_, sole_friction=True):
    m = SimpleNamespace(ngeom=4, geom_friction=np.tile([[0.7, 0.005, 0.0001]], (4, 1)).astype(np.float32))
    sim = SimpleNamespace(N=n, m=m, sole_geoms=np.asarray([2, 3]), cfg=SimCfg(num_envs=n, friction_surfaces=surfaces_, sole_friction=sole_friction))
    sim._mu, sim._friction = (lambda *a: Sim._mu(sim, *a)), (lambda *a: Sim._friction(sim, *a))
    return sim


def test_surfaces():
    ranges = surfaces("0.3:0.5,0.8:1.2")
    assert ranges == ((0.3, 0.5), (0.8, 1.2))
    sim = fake_sim(4000, ranges)
    mu = np.asarray(Sim._mu(sim, jax.random.PRNGKey(0)))
    low = (mu >= 0.3) & (mu <= 0.5)
    assert (low | ((mu >= 0.8) & (mu <= 1.2))).all() and 0.45 < low.mean() < 0.55, low.mean()
    assert mu[low].min() < 0.32 and mu[low].max() > 0.48  # uniform inside a range
    f = np.asarray(Sim._friction(sim, jnp.tile(sim.m.geom_friction[None], (4000, 1, 1)), jnp.asarray(mu)))
    assert np.allclose(f[:, :, 0], mu[:, None]) and np.allclose(f[:, :, 1:], sim.m.geom_friction[None, :, 1:])
    keep = np.asarray(Sim._friction(fake_sim(4000, ranges, False), jnp.asarray(f), jnp.asarray(mu) + 1))
    assert np.allclose(keep[:, 2:, 0], f[:, 2:, 0]) and np.allclose(keep[:, :2, 0], mu[:, None] + 1)  # soles kept without sole_friction
    sim = fake_sim(8, ranges)
    w = World(None, None, None, None, jnp.asarray(f[:8]), None, None)
    done = jnp.asarray([True, False] * 4)
    w1 = Sim._new_surface(sim, w, done, jax.random.PRNGKey(1))
    assert np.allclose(w1.geom_friction[~np.asarray(done)], f[:8][~np.asarray(done)]) and not np.allclose(w1.geom_friction[np.asarray(done)], f[:8][np.asarray(done)])
    print(f"surfaces: {ranges}: {low.mean():.2f} of the draws on the first; every geom set (soles with sole_friction); only ended episodes redrawn")


if __name__ == "__main__":
    assert not os.environ.get("XSOMA_ROBOT_DIR"), "run this test without XSOMA_ROBOT_DIR"
    test_stance()
    test_turn_step()
    test_surfaces()
    test_standin_turn_step()
    print("\nall checks passed")
