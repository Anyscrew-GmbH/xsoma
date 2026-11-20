"""evaluate.py's long episodes and its records of falls (run: python tests/test_falls.py, or pytest): --steps, --falls and
--episodes.

CPU only, where MuJoCo Warp does not step. The simulator is the real Sim of the bundled G1 on the scaled terrain, with its own
step (commands, pushes, time-out, resets), views, context and tiles; only its physics and its contact forces are hand-made:
every robot glides at its own velocity while it turns and leans, and its joints follow their targets. Robot 1 touches the
ground with one termination body at step 530 and robot 2 with another at step 40, robot 3's physics fails at step 450 and
robot 0 reaches the time-out.

  window    evaluate.main with --steps 620 --falls --episodes, against a recording of every step of the same episodes: a fall's
            record holds the recording's last 100 steps before the fall (39 for robot 2, which has no more), the step, the
            body and its force or the physics failure, the pushes and the command change in the window; the robot without a
            fall has no record. An episode's summary holds its fall or -1, pushes and randomization, per command the ground classes
            and the signals' means, the steps after a change of ground, and without a fall the range of each signal in its last
            100 steps
  steps     --steps moves the time-out of the real step: robot 0's episode has 620 steps, and 300 with --steps 300
  metrics   with the records, the evaluation's numbers are those without them; with `self_collision`, a fall's record gains
            the smallest distance of the two groups
  default   without the three options, the lowered text of evaluate() on the real Sim, and main()'s line and record on
            test_finetune's stand-in, are those of the commit BASE (skipped where the checkout does not have it)
"""

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, sys.argv[2] if sys.argv[1:2] == ["fingerprint"] else REPO)  # a child of test_default: the package of another tree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from xsoma import evaluate, task  # noqa: E402
from xsoma import terrain as terrain_mod  # noqa: E402
from xsoma.sim import Sim, SimCfg  # noqa: E402

BASE = "d7f6e4391e505421ea65a73846c2012ad29903c0"  # the last commit without the options: what the default evaluation is held to
N, STEPS = 4, 620
FALL = np.asarray([-1, 530, 40, -1])  # the step at which the robot's termination body SLOT touches the ground with FORCE
SLOT, FORCE = np.asarray([0, 4, 0, 0]), 50.0
FAIL = np.asarray([-1, -1, -1, 450])  # the step at which the robot's physics fails
VEL = np.asarray([[1.0, 0.3], [-0.5, 0.8], [0.2, -1.0], [0.0, 0.0]], np.float32)  # m/s, world


class FallSim(Sim):
    """The real Sim with hand-made physics and contact forces (module docstring). `data.time` counts the policy steps."""

    def physics(self, world, data, target, kp, kd):
        t = data.time + 1
        roll, pitch, yaw = 0.001 * t * (jnp.arange(self.N) == 1), 0.0005 * t, 0.01 * t
        cr, sr, cp, sp, cy, sy = (f(x / 2) for x in (roll, pitch, yaw) for f in (jnp.cos, jnp.sin))
        quat = jnp.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), -1)
        q = data.qpos[:, 7:] + 0.1 * (task.DEFAULT_Q + task.ACTION_SCALE * target - data.qpos[:, 7:])
        qpos = jnp.concatenate((data.qpos[:, :2] + task.DT * VEL, data.qpos[:, 2:3], quat, q), -1)
        vx = jnp.where(t == jnp.asarray(FAIL), 2e3, VEL[:, 0])  # beyond sim.DIVERGED
        qvel = jnp.concatenate((vx[:, None], VEL[:, 1:], jnp.zeros((self.N, 3)), jnp.full((self.N, 1), 0.5), (q - data.qpos[:, 7:]) / task.DT), -1)
        return data.replace(qpos=qpos, qvel=qvel, time=t)

    def collider_forces(self, data):
        hit = (data.time == jnp.asarray(FALL))[:, None] & (jnp.arange(len(self.collider_geoms)) == jnp.asarray(SLOT)[:, None])
        return hit[..., None] * jnp.asarray([0.0, 0.0, FORCE])


def policy(path):
    """Write a small PPO-type actor with random weights as a checkpoint for evaluate.py, and return its parameters."""
    rng, sizes = np.random.RandomState(0), (task.ACTOR_DIM, 8, 8, 8, task.J)
    p = {f"w{i}": 0.3 * rng.randn(a, b).astype(np.float32) for i, (a, b) in enumerate(zip(sizes, sizes[1:]))}
    p |= {f"b{i}": 0.3 * rng.randn(b).astype(np.float32) for i, b in enumerate(sizes[1:])}
    np.savez(path, kind="ppo", **p)
    return {k: jnp.asarray(v) for k, v in p.items()}


def record(sim, act, key, steps):
    """Every step of every env as evaluate() takes them, one compiled step at a time: (the initial state, arrays (steps, N, .))."""
    s0 = sim.init(key)

    @jax.jit
    def one(s):
        s1, tr = sim.step(s, act(s.obs))
        v = sim.view(tr.qpos1, tr.qvel1, s.world.heights)
        out = {"t": s.t + 1, "cmd": s.cmd, "qpos": tr.qpos1, "qvel": tr.qvel1, "end": tr.terminated | tr.truncated, "fell": tr.terminated, "target": tr.target,
               "kp": s.kp, "kd": s.kd, "height": v.height, "foot_h": v.foot_h, "feet": v.robot.feet_p, "pushed": s1.push_t == 0, "qvel_next": s1.data.qvel}  # fmt: skip
        return s1._replace(scale=jnp.ones_like(s1.scale)), out

    s, rows = s0._replace(scale=jnp.ones_like(s0.scale)), []
    for _ in range(steps):
        s, out = one(s)
        rows.append(jax.device_get(out))
    return jax.device_get(s0), {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def channels(sim, rec, e):
    """What a fall record keeps of every step of env e, from the recording and by formulas of this test's own."""
    q, v = rec["qpos"][:, e].astype(np.float64), rec["qvel"][:, e].astype(np.float64)
    w, x, y, z = q[:, 3:7].T
    R = np.zeros((len(q), 9))
    for i in range(len(q)):
        mujoco.mju_quat2Mat(R[i], q[i, 3:7])
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    feet = rec["feet"][:, e, 0] - rec["feet"][:, e, 1]
    tau = rec["kp"][:, e] * task.KP * (task.ACTION_SCALE * rec["target"][:, e] + task.DEFAULT_Q - q[:, 7:]) - rec["kd"][:, e] * task.KD * v[:, 6:]
    tau = np.abs(np.clip(tau, -task.EFFORT, task.EFFORT)) / task.EFFORT
    t, c = sim.terrain, sim.terrain.cfg
    ground = []
    for px, py in q[:, :2]:
        i, j = int(np.floor(px / c.tile)), int(np.floor(py / c.tile))
        ground.append(f"{terrain_mod.TYPES[t.types[i, j]]} {t.difficulty[i, j]:g}" if 0 <= i < c.num_rows and 0 <= j < c.num_cols else "border")
    return {"cmd": rec["cmd"][:, e], "v": np.einsum("tji,tj->ti", R.reshape(-1, 3, 3), v[:, :3]), "w": v[:, 3:6],
            "rpy": np.stack((np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), np.arcsin(2 * (w * y - z * x)), yaw), -1), "height": rec["height"][:, e],
            "foot_h": rec["foot_h"][:, e], "feet_dy": np.cos(yaw) * feet[:, 1] - np.sin(yaw) * feet[:, 0], "torque": tau.max(-1),
            "torque_joint": [task.DOF[i] for i in tau.argmax(-1)], "xy": q[:, :2], "ground": ground}  # fmt: skip


def run(d, ckpt, *options):
    """evaluate.main on FallSim with the options: (its simulator, the fall records, the episode summaries, the printed line)."""
    sims, make = [], evaluate.make_sim
    falls, episodes = os.path.join(d, "falls.jsonl"), os.path.join(d, "episodes.jsonl")
    with mock.patch.object(evaluate, "Sim", FallSim), mock.patch.object(evaluate, "make_sim", lambda *a: sims.append(make(*a)) or sims[-1]), \
            contextlib.redirect_stdout(io.StringIO()) as out:  # fmt: skip
        evaluate.main([ckpt, "--n", str(N), *options, "--falls", falls, "--episodes", episodes])
    return sims[0], *([json.loads(x) for x in open(f)] for f in (falls, episodes)), out.getvalue()


def test_window():
    W = evaluate.WINDOW
    close = lambda a, b: np.allclose(np.asarray(a, np.float64), np.asarray(b, np.float64), atol=3e-4)  # noqa: E731  (the records keep 4 decimals)
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "actor.npz")
        params = policy(ckpt)
        sim, falls, eps, line = run(d, ckpt, "--steps", str(STEPS))
        short = run(d, ckpt, "--steps", "300")[2]
    assert sim.cfg.max_episode == STEPS - 1 and "no fall 0.250" in line, line
    act = jax.tree_util.Partial(evaluate.act_fn("ppo"), params)
    s0, rec = record(sim, act, jax.random.PRNGKey(1), STEPS + 1)
    n = rec["end"].argmax(0) + 1  # every env's alive steps
    last = rec["t"][n - 1, np.arange(N)]
    assert last.tolist() == [STEPS, 530, 40, 450] and rec["fell"][n - 1, np.arange(N)].tolist() == [False, True, True, True]

    # the fall records
    assert [f["env"] for f in falls] == [1, 2, 3] and all(f["ckpt"] == ckpt for f in falls + eps)
    bodies = [sim.m.body(sim.m.geom_bodyid[g]).name.removeprefix("robot_") for g in sim.collider_geoms]
    in_window = 0
    for f in falls:
        e = f["env"]
        k = min(n[e], W)
        rows = slice(n[e] - k, n[e])
        steps = rec["t"][rows, e]
        assert (f["step"], f["first"], len(f["height"])) == (last[e], steps[0], k) and k == (39 if e == 2 else W)
        for name, want in channels(sim, rec, e).items():
            want = want[rows]
            assert f[name] == want if isinstance(want, list) else close(f[name], want), (e, name)
        pushes = [[int(t), *(rec["qvel_next"][i, e, :2] - rec["qvel"][i, e, :2])] for i, t in zip(range(rows.start, rows.stop), steps) if rec["pushed"][i, e]]
        assert len(f["pushes"]) == len(pushes) and close(f["pushes"], pushes) and f["resample"] == [t for t in steps if t % task.RESAMPLE == 0]
        in_window += len(pushes)
        assert f["diverged"] == (e == 3) and f["contact"] == ({} if e == 3 else {bodies[SLOT[e]]: FORCE}) and "legs_dist" not in f
    assert falls[0]["resample"] == [500] and falls[0]["cmd"][0] != falls[0]["cmd"][-1] and len(falls[2]["pushes"]) == in_window == 1
    assert any(r > 0.05 for r in falls[0]["rpy"][-1][:2]) and len(set(falls[0]["ground"]) | set(falls[2]["ground"])) > 1
    # the classes of ground: a tile, the border around the tiles, and beyond the border's boxes
    names, at = evaluate.ground(sim)
    c, (i, j) = sim.terrain.cfg, (1, 2)
    far = c.num_rows * c.tile + c.border
    x, y = jnp.asarray([(i + 0.5) * c.tile, -0.05, -c.border + 0.05, -c.border - 0.05, far - 0.2, far, 1.0, 1.0]), jnp.asarray([(j + 0.5) * c.tile] + [1.0] * 7)
    got = [names[int(g)] for g in at(x, y, jnp.asarray([0.8] * 6 + [-0.9, -1.1]))]  # the last one is 1.1 m under the surface of a tile
    here = [f"{terrain_mod.TYPES[sim.terrain.types[a, b]]} {sim.terrain.difficulty[a, b]:g}" for a, b in ((i, j), (0, 0))]
    assert got == [here[0], "border", "border", "off", "border", "off", here[1], "off"], got

    # the episode summaries
    t = sim.terrain
    tiles = (terrain_mod.sample_env_origins(t, N, np.random.RandomState(sim.cfg.seed))[:, :2] // t.cfg.tile).astype(int)
    assert [x["env"] for x in eps] == list(range(N)) and [x["fall"] for x in eps] == [-1, 530, 40, 450] and [x["len"] for x in eps] == last.tolist()
    for x in eps:
        e = x["env"]
        alive = slice(0, n[e])
        pushed = np.flatnonzero(rec["pushed"][alive, e])
        assert x["pushes"] == len(pushed) and x["last_push"] == (rec["t"][pushed[-1], e] if len(pushed) else -1)
        assert close(x["cmd_speed"], np.linalg.norm(rec["cmd"][alive, e, :2], axis=-1).mean())
        firsts = [i for i in range(n[e]) if i == 0 or (rec["t"][i, e] - 1) % task.RESAMPLE == 0]  # the first step of every command
        assert close(x["cmds"], rec["cmd"][firsts, e]) and len(x["cmds"]) == (last[e] - 1) // task.RESAMPLE + 1
        truth = channels(sim, rec, e)
        ground, seg = truth["ground"][alive], (rec["t"][alive, e] - 1) // task.RESAMPLE  # the number of every step's command
        assert x["on"] == [{g: c for g in set(ground) if (c := sum(a == g and b == k for a, b in zip(ground, seg)))} for k in range(len(x["cmds"]))]
        assert set(x["mean"]) == set(evaluate.SIGNALS) - {"legs_dist"} and all(close(x["mean"][name], [truth[name][alive][seg == k].mean(0) for k in range(len(x["cmds"]))]) for name in x["mean"])
        change = np.flatnonzero([a != b for a, b in zip([ground[0]] + ground, ground)])  # the steps whose ground is another class than the step before
        assert x["since_change"] == (n[e] - 1 - change[-1] if len(change) else -1) and x["changed"] == len({i + k for i in change for k in range(W) if i + k < n[e]})
        if e == 0:  # no fall: the range of every signal in its last 100 steps
            want = {k: v[n[e] - W : n[e]] for k, v in channels(sim, rec, e).items() if k in evaluate.SIGNALS}
            assert set(x["normal"]) == set(want) and all(close(x["normal"][k], [v.min(0), v.max(0)]) for k, v in want.items())
        else:
            assert "normal" not in x
        i, j = tiles[e]
        assert (x["tile"], x["terrain"], x["level"]) == ([i, j], terrain_mod.TYPES[t.types[i, j]], t.difficulty[i, j])
        c, w = x["context"], s0.world
        assert close(c["kp"], s0.kp[e]) and close(c["kd"], s0.kd[e]) and c["delay"] == s0.delay[e] and close(c["friction"], w.geom_friction[e, 0, 0])
        assert close(c["mass"], w.body_mass[e, sim.mass_ids] / sim.m.body_mass[sim.mass_ids]) and close(c["torso_mass"], w.body_mass[e, sim.torso] - sim.m.body_mass[sim.torso])
        assert close(c["torso_com"], w.body_ipos[e, sim.torso] - sim.m.body_ipos[sim.torso]) and c["payload"] == [] and {f["env"]: f["context"] for f in falls}.get(e, c) == c
    assert sum(x["pushes"] for x in eps) >= 3 and len(eps[0]["on"][0]) > 1 and len(eps[0]["cmds"]) == 2 and eps[0]["changed"] > W
    print(f"window: robots 1, 2 and 3 fall at steps {[f['step'] for f in falls]}: their records hold the recording's last {[len(f['height']) for f in falls]} steps "
          f"({in_window} pushes and the command change of step 500 among them), body and force {falls[0]['contact']}, robot 3 a physics failure; "
          f"robot 0 has none. Summaries: pushes {[x['pushes'] for x in eps]}, commands {[len(x['cmds']) for x in eps]}, robot 0 on {eps[0]['on']}, "
          f"{eps[0]['changed']} of its steps within {W} after the ground changed")  # fmt: skip

    # --steps: the time-out of the real step
    assert (eps[0]["len"], eps[0]["fall"]) == (STEPS, -1) and [(x["len"], x["fall"]) for x in short] == [(300, -1), (300, -1), (40, 40), (300, -1)]
    print(f"steps: robot 0 reaches the time-out after {eps[0]['len']} steps with --steps {STEPS} and after {short[0]['len']} with --steps 300")

    # the same numbers with and without the records; the smallest distance of two made-up collider pairs
    key = jax.random.PRNGKey(1)
    plain = jax.jit(lambda k: evaluate.evaluate(sim, act, k, STEPS + 1))(key)
    full = jax.jit(lambda k: evaluate.evaluate(sim, act, k, STEPS + 1, window=W))(key)
    assert len(plain) == 11 and len(full) == 12 and all(np.allclose(a, b, rtol=1e-5, atol=1e-6) for a, b in zip(plain, full))
    sim.self_distance = lambda S: jnp.stack((S[0][:, 0] - 14.0, 0.01 * S[0][:, 1]), -1)
    with mock.patch.object(task, "SELF_COLLISION", [["a"], ["b"]]):
        legs = jax.device_get(jax.jit(lambda k: evaluate.evaluate(sim, act, k, STEPS + 1, window=W))(key))
        gaps = evaluate.records(sim, legs[1], legs[2], legs[-1])[1]
    assert len(legs) == 14 and all(np.allclose(a, b, rtol=1e-5, atol=1e-6) for a, b in zip(plain, legs))
    for f, g in zip(falls, gaps):
        rows = slice(n[f["env"]] - len(f["height"]), n[f["env"]])
        xy = rec["qpos"][rows, f["env"], :2]
        assert close(g["legs_dist"], np.minimum(xy[:, 0] - 14.0, 0.01 * xy[:, 1])) and set(g) - set(f) == {"legs_dist"}
    print(f"metrics: the 11 numbers of the evaluation are the same with the records; with two collider groups a fall's record gains their "
          f"smallest distance ({gaps[0]['legs_dist'][-1]} m at robot 1's fall)")  # fmt: skip


def fingerprint(root):
    """Of the package under `root`, with default options: the lowered text of evaluate() on the real Sim, without and with a
    policy memory, and main()'s output on test_finetune's stand-in."""
    import test_finetune as tf

    assert os.path.abspath(evaluate.__file__).startswith(os.path.abspath(root) + os.sep), evaluate.__file__
    sim = evaluate.make_sim(2, 0.5, False, "fast_sac", "mix")
    zero = jnp.zeros((2, task.J))
    out = {"plain": jax.jit(lambda k: evaluate.evaluate(sim, lambda obs: zero + obs[:, :1], k)),
           "memory": jax.jit(lambda k: evaluate.evaluate(sim, lambda mem, s: (zero + mem, mem + s.t[0]), k, mem=jnp.zeros(())))}  # fmt: skip
    out = {k: hashlib.sha256(f.lower(jax.random.PRNGKey(0)).as_text().encode()).hexdigest() for k, f in out.items()}
    with tempfile.TemporaryDirectory() as d, mock.patch.multiple(task, **tf.ROBOT), contextlib.redirect_stdout(io.StringIO()) as line, \
            mock.patch.object(evaluate, "make_sim", lambda n, *a: tf.FakeSim(SimCfg(num_envs=n))):  # fmt: skip
        os.makedirs(os.path.join(d, "ckpt"))
        policy(os.path.join(d, "ckpt", "0000.npz"))
        with open(os.path.join(d, "ckpt", "0000.json"), "w") as f:
            json.dump({"wall_s": 2.0, "train_s": 1.0, "env_steps": 3.0}, f)
        evaluate.main([d, "--n", str(tf.N)])
        out["record"] = open(os.path.join(d, "eval.jsonl")).read().replace(d, "")
    return out | {"line": line.getvalue().replace(d, "")}


def test_default():
    with tempfile.TemporaryDirectory() as head:
        subprocess.run(f"git -C '{REPO}' archive {BASE} | tar -x -C '{head}'", shell=True, capture_output=True)
        if not os.listdir(head):
            print(f"default: skipped, this checkout does not have commit {BASE[:7]}")
            return
        prints = [subprocess.run([sys.executable, os.path.abspath(__file__), "fingerprint", root], capture_output=True, text=True) for root in (head, REPO)]
    for r in prints:
        assert r.returncode == 0, r.stderr[-2000:]
    old, new = (json.loads(r.stdout.splitlines()[-1]) for r in prints)
    assert old == new and len(old["record"]) > 100 and "no fall" in old["line"], {k: (old[k], new[k]) for k in old if old[k] != new[k]}
    print(f"default: without the options, evaluate() on the real Sim lowers to the text of commit {BASE[:7]} (sha256 {new['plain'][:12]}, with a policy "
          f"memory {new['memory'][:12]}), and main() on the stand-in prints the same line and writes the same record")  # fmt: skip


if __name__ == "__main__":
    assert not os.environ.get("XSOMA_ROBOT_DIR"), "run this test without XSOMA_ROBOT_DIR"
    if sys.argv[1:2] == ["fingerprint"]:
        print(json.dumps(fingerprint(sys.argv[2])))
        sys.exit()
    test_window()
    test_default()
    print("\nall checks passed")
