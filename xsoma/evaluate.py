"""Deterministic evaluation of policies in the simulator. Every env runs one full episode (up to 1001 steps, 20 s; --steps) with the
policy's mean action, on training's terrain and with training's randomization (masses, friction, PD gains, delay, pushes). All
envs start at step 0.

Per policy it reports Holosoma's logged metrics (Episode/rew_<term> = the episode's sum of w dt r / 20 s, averaged over episodes;
penalties at full weight, penalty scale 1), the mean episode length, the share of episodes that reach the time-out without a
fall (`success`), the mean velocity-tracking errors while the robot is alive, and of the feet (task.feet_dy: how far the left one
is to the left of the right one) the share of an episode's alive steps in which they are crossed and the mean distance, both
averaged over episodes. Of the soles it reports how far their deepest collider point is under the terrain surface
(sim.View.soles), in meters: each episode's largest depth, averaged over episodes (`sole_sink_max`), the share of alive steps
in which it exceeds SOLE_SINK (`sole_sink_frac`) and the depth that 99 % of all alive steps stay within (`sole_sink_p99`). For
a robot with `self_collision` groups (task.py) it adds, from the signed distances between the two groups' colliders
(sim.SelfDistance), the share of alive steps in which any pair overlaps (`legs_overlap`) and the mean smallest distance
(`legs_dist`). The policies are XSoma checkpoints (train.save_ckpt), Holosoma's PPO and FastSAC actors (converted by
tools/holosoma_ckpt.py) and tracker policies trained with goal
masking, driven by the task's commands (tracker.py), all evaluated alike.

    python -m xsoma.evaluate <run dir or checkpoint> [...] [--n 2048] [--scale 0.5] [--sole_friction 0] [--preset fast_sac]
                             [--steps 1001] [--falls falls.jsonl] [--episodes episodes.jsonl] [--cmd_scale 1 1 1]
    python -m xsoma.evaluate <tracker checkpoint> --tracker_meta <the tracker's meta.json> [--tracker_heading rate] [the same options]

A run directory is evaluated checkpoint by checkpoint (XSoma: ckpt/*.npz, with --tracker_meta if the run fine-tuned a tracker;
Holosoma: hs/**/model_*.npz), each with its wall-clock time since the run's launch, and the records are appended to
<run>/eval.jsonl. The defaults are the setting that Holosoma's MuJoCo Warp baselines train in: terrain scale 0.5, sole friction
fixed at 0.8. train.py calls this module after training.

`--steps` sets the episode's length (the time-out) in policy steps; a record then names it (`episode`). `--cmd_scale x y z`
scales the command ranges of vx, vy and wz; a record then names the factors (`cmd_scale`). `--push_vel` sets the size of the
pushes (0: none); a record then names it (`push_vel`). `--falls` and
`--episodes` write JSON lines that tell when and why robots fall (both files start empty; every line names its checkpoint; a
step is numbered by the episode counter after it, as `len` is, and lasts task.DT):
  --episodes  one line per env: `fall` (the step of the fall, -1 without one), the tile it started on (`tile` row and column,
              `terrain` type, `level` = the tile's difficulty), `pushes` received and the step of the last one, the mean
              commanded planar speed, and `context`: the episode's randomization (sim.Sim.context by name, and the action
              delay). Per command, which changes every task.RESAMPLE steps: `cmds`, `on` (the steps spent on each class of
              ground: a tile type with its difficulty, the flat "border", or "off" the terrain, where nothing holds the robot
              and it falls without an end) and `mean` (of each of SIGNALS). `changed`: the steps within WINDOW steps after the
              ground's class changed, `since_change`: the steps since the last change at the end (-1: none). Without a fall
              also `normal`: the smallest and the largest value of each of SIGNALS in the episode's last WINDOW steps, which
              tells what the fall records' signals look like in ordinary walking
  --falls     one line per fall: the same start tile and context, `diverged` (a physics failure, not a contact), `contact` (the
              termination bodies whose contact force ended the episode, in N) and the last WINDOW steps, from step `first` on,
              as one list per CHANNELS entry: the command, the base's linear and angular velocity in its frame, roll, pitch and
              yaw, the pelvis and the two foot points over the terrain, task.feet_dy, the largest PD torque of the step's final
              state over its limit (task.EFFORT) and its joint, the base's x and y, the ground class under it and, with
              `self_collision`, the smallest distance between the two groups; `pushes`: (step, change of the world x and y
              velocity) of the pushes in the window; `resample`: the steps after which the command was redrawn
The window is a ring of WINDOW steps per env on the device that stops when the env's episode ends: 100 x envs x 25 numbers,
one row written per step; the sums per command are envs x commands x about 30 numbers.
"""

import argparse
import glob
import json
import os

if __name__ == "__main__":  # standalone: JAX allocates on demand, so MuJoCo Warp's own buffers (more contacts when robots fall) fit
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from . import nets, task, tracker, wandb_log  # noqa: E402
from . import terrain as terrain_mod  # noqa: E402
from .sim import Sim, SimCfg, surfaces, unflatten  # noqa: E402

SOLE_SINK = 0.01  # m: sole_sink_frac counts the steps in which a sole collider point is further than this below the terrain
WINDOW = 100  # steps before a fall that its record keeps (2 s)
# what a fall record keeps of each of those steps, with the number of values; legs_dist only with task.SELF_COLLISION
CHANNELS = (("v", 3), ("w", 3), ("rpy", 3), ("height", 1), ("foot_h", 2), ("feet_dy", 1), ("torque", 1), ("cmd", 3), ("torque_joint", 1),
            ("push", 3), ("xy", 2), ("ground", 1), ("legs_dist", 1))  # fmt: skip
SIGNALS = ("v", "w", "rpy", "height", "foot_h", "feet_dy", "torque", "legs_dist")  # the channels that an episode summary keeps means and ranges of


def ground(sim):
    """(names, f): the classes of ground under a robot, which are every tile type at every difficulty, the flat border and
    "off" the terrain, and f(x, y, h) -> the index of the class at the world point for a pelvis h over the terrain."""
    t = sim.terrain
    if t is None:
        return ("plane",), lambda x, y, h: jnp.zeros(x.shape, jnp.int32)
    c, levels = t.cfg, np.unique(t.difficulty)
    names = tuple(f"{k} {d:g}" for k in terrain_mod.TYPES for d in levels) + ("border", "off")
    cls = jnp.asarray(len(levels) * t.types + np.searchsorted(levels, t.difficulty))
    end = t.x0 + (np.asarray(t.heights.shape) - 1) * c.horizontal_scale  # where the border boxes end (scene.py)

    def f(x, y, h):
        i, j = jnp.floor(x / c.tile).astype(jnp.int32), jnp.floor(y / c.tile).astype(jnp.int32)
        held = (h > -1.0) & (x >= t.x0) & (x <= end[0]) & (y >= t.x0) & (y <= end[1])  # 1 m under the surface: it fell off the edge
        tile = held & (i >= 0) & (i < c.num_rows) & (j >= 0) & (j < c.num_cols)
        return jnp.where(tile, cls[jnp.clip(i, 0, c.num_rows - 1), jnp.clip(j, 0, c.num_cols - 1)], len(names) - 1 - held)

    return names, f


def evaluate(sim: Sim, act, key, steps: int = task.MAX_EPISODE + 2, mem=None, window: int = 0):
    """Run every env for one episode with the deterministic policy `act(obs) -> action`; with `mem` (a policy's initial memory)
    it is `act(mem, state) -> (action, mem)`. The default number of steps lets every env reach the time-out. Returns, per env,
    for its first episode only: the sums of the weighted reward terms, the length, whether it ended in a fall, whether it ended
    at all, and while alive the mean velocity and yaw-rate tracking errors, the share of steps with the feet crossed, the mean
    of task.feet_dy and the share of steps with a sole point more than SOLE_SINK below the terrain; then the largest depth of a
    sole point below the terrain and, as one number for all envs, the 99th percentile of that depth over the alive steps; with
    task.SELF_COLLISION also the share of alive steps in which the two groups overlap and the mean of their smallest distance
    (sim.SelfDistance). With `window` > 0 a dict follows, of which `records` makes the episode summaries and the fall
    records: `ring` (window, N, .), the CHANNELS of every env's last `window` alive steps (alive step k in row k % window),
    and per env the alive `steps`, the contact `force` per termination collider at the end, whether it `diverged`, the
    `pushes`, the step of the `last_push`, the summed commanded `speed`, per task.RESAMPLE steps the command (`cmds`), the
    steps `on` each class of ground and the `sums` of the SIGNALS, the `ground` class at the end, the steps `since` it changed
    and those within `window` after a change (`changed`), and the first episode's context `xi` and action `delay`."""
    N = sim.N
    s = sim.init(key)
    s = s._replace(scale=jnp.ones_like(s.scale))  # penalties at full weight

    def body(carry, i):
        s, mem, sums, ended, length, fell, sink, err_v, err_w, crossed, dy, deep, legs, alive_steps, log = carry
        a, mem = (act(s.obs), None) if mem is None else act(mem, s)
        s1, tr = sim.step(s, a)
        live = ~ended  # the env's first episode is still running; what happens after its reset does not count
        sums = sums + jnp.where(live[:, None], tr.parts, 0.0)
        v = sim.view(tr.qpos1, tr.qvel1, s.world.heights)  # the task's view after the physics, before any reset
        R = v.robot
        v_b = jnp.einsum("nji,nj->ni", R.R, R.v)
        err_v = err_v + jnp.where(live, jnp.linalg.norm(s.cmd[:, :2] - v_b[:, :2], axis=-1), 0.0)
        err_w = err_w + jnp.where(live, jnp.abs(s.cmd[:, 2] - R.w[:, 2]), 0.0)
        d = task.feet_dy(R)
        crossed, dy = crossed + (live & (d < 0)), dy + jnp.where(live, d, 0.0)
        depth = jnp.maximum(-v.soles.min(-1), 0.0)  # of the deepest sole point under the terrain
        sink, deep = jnp.maximum(sink, jnp.where(live, depth, 0.0)), deep + (live & (depth > SOLE_SINK))
        if legs:
            gap = sim.self_distance(unflatten(tr.qpos1, tr.qvel1)).min(-1)
            legs = (legs[0] + (live & (gap < 0)), legs[1] + jnp.where(live, gap, 0.0))
        alive_steps = alive_steps + live
        end = (tr.terminated | tr.truncated) & live
        length = jnp.where(end, s.t + 1, length)
        fell = fell | (tr.terminated & live)
        if window:
            pushed = live & (s1.push_t == 0)  # only a push sets the counter to 0 (a reset leaves it at 1)
            tau = jnp.abs(task.torques(tr.qpos1[:, 7:], tr.qvel1[:, 6:], tr.target, s.kp, s.kd)) / task.EFFORT
            rpy = (jnp.arctan2(R.R[:, 2, 1], R.R[:, 2, 2]), -jnp.arcsin(jnp.clip(R.R[:, 2, 0], -1.0, 1.0)), jnp.arctan2(R.R[:, 1, 0], R.R[:, 0, 0]))
            g = ground_at(R.p[:, 0], R.p[:, 1], v.height)
            since = jnp.where(g == log["ground"], log["since"] + 1, 0)
            col = lambda *xs: [x[:, None].astype(jnp.float32) for x in xs]  # noqa: E731
            push = jnp.where(pushed[:, None], s1.data.qvel[:, :2] - tr.qvel1[:, :2], 0.0)
            row = jnp.concatenate((v_b, R.w, *col(*rpy, v.height), v.foot_h, *col(d, tau.max(-1)), s.cmd, *col(tau.argmax(-1), pushed), push, R.p[:, :2],
                                   *col(g, *((gap,) if legs else ()))), -1)  # fmt: skip
            k, seg = i % window, (jnp.arange(N), s.t // task.RESAMPLE)
            log = {"ring": log["ring"].at[k].set(jnp.where(live[:, None], row, log["ring"][k])),  # an ended env's rows stay
                   "force": jnp.where(end[:, None], tr.force, log["force"]), "diverged": log["diverged"] | (end & tr.diverged),
                   "pushes": log["pushes"] + pushed, "last_push": jnp.where(pushed, s.t + 1, log["last_push"]),
                   "speed": log["speed"] + jnp.where(live, jnp.linalg.norm(s.cmd[:, :2], axis=-1), 0.0),
                   "cmds": log["cmds"].at[seg].set(jnp.where(live[:, None], s.cmd, log["cmds"][seg])),
                   "on": log["on"].at[seg].add(live[:, None] & (g[:, None] == jnp.arange(len(names)))),
                   "sums": log["sums"].at[seg].add(jnp.where(live[:, None], row[:, signals], 0.0)), "ground": jnp.where(live, g, log["ground"]),
                   "since": jnp.where(live, since, log["since"]), "changed": log["changed"] + (live & (since < window))}  # fmt: skip
        # a step updates the penalty curriculum, so the scale is set back to 1
        return (s1._replace(scale=jnp.ones_like(s1.scale)), mem, sums, ended | end, length, fell, sink, err_v, err_w, crossed, dy, deep, legs, alive_steps, log), jnp.where(live, depth, jnp.nan)

    z, log = jnp.zeros((N,)), ()
    if window:
        names, ground_at = ground(sim)
        width, S = sum(n for _, n in CHANNELS) - (not task.SELF_COLLISION), steps // task.RESAMPLE + 1
        signals = np.flatnonzero(np.repeat([k in SIGNALS for k, _ in CHANNELS], [n for _, n in CHANNELS])[:width])  # their columns of a row
        log = {"ring": jnp.zeros((window, N, width)), "force": jnp.zeros((N, len(sim.collider_geoms))), "diverged": jnp.zeros((N,), bool), "pushes": z,
               "last_push": z - 1, "speed": z, "cmds": jnp.zeros((N, S, 3)), "on": jnp.zeros((N, S, len(names))), "sums": jnp.zeros((N, S, len(signals))),
               "ground": ground_at(s.data.qpos[:, 0], s.data.qpos[:, 1], z), "since": jnp.full((N,), 10**6), "changed": z}  # fmt: skip
    init = (s, mem, jnp.zeros((N, len(task.TERMS))), jnp.zeros((N,), bool), z, jnp.zeros((N,), bool), z, z, z, z, z, z, (z, z) if task.SELF_COLLISION else (), z, log)
    out, depth = jax.lax.scan(body, init, jnp.arange(steps) if window else None, length=steps)
    _, _, sums, ended, length, fell, sink, *means, legs, alive_steps, log = out
    means, legs = (tuple(x / jnp.maximum(alive_steps, 1) for x in xs) for xs in (means, legs))
    out = (sums, length, fell, ended, *means, sink, jnp.nanpercentile(depth, 99.0), *legs)
    # `s` is still the initial state: the first episode's randomization
    return (*out, log | {"steps": alive_steps, "xi": sim.context(s), "delay": s.delay}) if window else out


def records(sim, length, fell, log):
    """(episode summaries, fall records) of one evaluation as lists of dicts (the module docstring describes them), from
    `evaluate`'s lengths, falls and its dict."""
    m, t = sim.m, sim.terrain
    r = lambda x: np.round(np.asarray(x, np.float64), 4).tolist()  # noqa: E731
    one = lambda x: x.squeeze(-1 if x.shape[-1] == 1 else ())  # noqa: E731
    cuts = np.cumsum([n for k, n in CHANNELS if k in SIGNALS])[:-1]
    names = ground(sim)[0]
    bodies = [m.body(m.geom_bodyid[g]).name.removeprefix("robot_") for g in sim.collider_geoms]
    xi = np.split(log["xi"], np.cumsum([task.J, task.J, len(sim.mass_ids), 1, 3, 1]), -1)
    assert xi[6].shape[-1] == 4 * len(sim.payload), "the layout of Sim.context changed"
    if t is not None:  # the tiles that Sim.world starts the envs on
        tiles = (terrain_mod.sample_env_origins(t, sim.N, np.random.RandomState(sim.cfg.seed))[:, :2] // t.cfg.tile).astype(int)
    chan = dict(zip((k for k, _ in CHANNELS), np.split(log["ring"], np.cumsum([n for _, n in CHANNELS])[:-1], -1)))
    eps, falls = [], []
    for e in range(sim.N):
        n, last = int(log["steps"][e]), int(length[e])
        start = {"terrain": "plane", "level": 0.0}
        if t is not None:
            start = {"tile": tiles[e].tolist(), "terrain": terrain_mod.TYPES[t.types[tuple(tiles[e])]], "level": float(t.difficulty[tuple(tiles[e])])}
        ctx = {"kp": r(xi[0][e]), "kd": r(xi[1][e]), "mass": r(xi[2][e]), "torso_mass": r(xi[3][e, 0]), "torso_com": r(xi[4][e]),
               "friction": r(xi[5][e, 0]), "payload": r(xi[6][e].reshape(-1, 4)), "delay": int(log["delay"][e])}  # fmt: skip
        on = log["on"][e, : (last - 1) // task.RESAMPLE + 1]
        mean = np.split(log["sums"][e, : len(on)] / np.maximum(on.sum(-1, keepdims=True), 1), cuts, -1)
        eps.append({"env": e, **start, "fall": last if fell[e] else -1, "len": last, "pushes": int(log["pushes"][e]), "last_push": int(log["last_push"][e]),
                    "cmd_speed": r(log["speed"][e] / max(n, 1)), "cmds": r(log["cmds"][e, : len(on)]), "on": [{k: int(x) for k, x in zip(names, o) if x} for o in on],
                    "mean": {k: r(one(x)) for k, x in zip(SIGNALS, mean) if x.shape[-1]}, "changed": int(log["changed"][e]),
                    "since_change": int(log["since"][e]) if log["since"][e] < 10**6 else -1, "context": ctx})  # fmt: skip
        k = min(n, len(log["ring"]))
        w = {name: one(c[np.arange(n - k, n) % len(c), e]) for name, c in chan.items() if c.shape[-1]}
        if not fell[e]:
            eps[-1]["normal"] = {name: [r(w[name].min(0)), r(w[name].max(0))] for name in SIGNALS if name in w}
            continue
        step = np.arange(last - k + 1, last + 1)
        falls.append({"env": e, "step": last, **start, "diverged": bool(log["diverged"][e]),
                      "contact": {b: r(f) for b, f in zip(bodies, log["force"][e]) if f > task.CONTACT_FORCE}, "first": int(step[0]),
                      **{name: r(w[name]) for name in ("cmd", "xy") + SIGNALS if name in w},
                      "torque_joint": [task.DOF[int(j)] for j in w["torque_joint"]], "ground": [names[int(g)] for g in w["ground"]],
                      "pushes": [[int(i), *r(p[1:])] for i, p in zip(step, w["push"]) if p[0]], "resample": step[step % task.RESAMPLE == 0].tolist(),
                      "context": ctx})  # fmt: skip
    return eps, falls


def load(path, which="avg", tracker_meta=None):
    """(kind, normalize, parameters) of an XSoma checkpoint, a converted Holosoma PPO / FastSAC actor or a tracker checkpoint
    (with `tracker_meta`, the tracker's robot description; `which` applies if train.py wrote the checkpoint)."""
    z = np.load(path)
    if "enc.0.w" in z.files:
        assert tracker_meta, f"{path} is a tracker checkpoint: give --tracker_meta"
        return "tracker", False, tracker.load(path, tracker_meta, "live." if which == "live" and "live.enc.0.w" in z.files else "")
    if "kind" not in z.files:
        return "xsoma", False, load_policy(path, which)
    params = {k: jnp.asarray(z[k]) for k in z.files if k not in ("kind", "iteration", "normalize", "wall_s")}
    return str(z["kind"]), bool(z["normalize"]) if "normalize" in z.files else False, params


def act_fn(kind, normalize=False, hold=True):
    """(parameters, obs) -> deterministic action of a policy type (`kind` is "xsoma", "ppo" or "fast_sac"). The parameters are an
    argument, so one compilation serves a policy type. "tracker": (parameters, memory, state) -> (action, memory), with the
    heading held or (`hold` False) the base's own yaw (tracker.step)."""
    if kind == "xsoma":
        return nets.pi_apply
    if kind == "tracker":
        return lambda p, mem, s: tracker.step(p, mem, s, hold)

    def dot(p, h, i):
        return jnp.dot(h, p[f"w{i}"], precision=jax.lax.Precision.HIGHEST) + p[f"b{i}"]

    if kind == "ppo":  # Holosoma PPO's actor mean

        def act(p, o):
            h = o
            for i in range(3):
                h = jax.nn.elu(dot(p, h, i))
            return dot(p, h, 3)

        return act

    def act(p, o):  # Holosoma FastSAC: observation normalizer, LayerNorm-SiLU MLP, tanh scaled per joint
        h = (o - p["mean"]) / (p["std"] + 0.01) if normalize else o
        for i in range(3):
            h = dot(p, h, i)
            mu, var = h.mean(-1, keepdims=True), h.var(-1, keepdims=True)
            h = jax.nn.silu((h - mu) * jax.lax.rsqrt(var + 1e-5) * p[f"g{i}"] + p[f"h{i}"])
        return jnp.tanh(dot(p, h, 3)) * p["scale"] + p["bias"]

    return act


def checkpoints(run):
    """(path, wall-clock seconds since launch, env steps or iteration) of a run directory's checkpoints."""
    out = []
    for p in sorted(glob.glob(os.path.join(run, "ckpt", "*.npz"))):  # XSoma
        meta = json.load(open(p[: -len(".npz")] + ".json"))
        out.append((p, meta.get("wall_s", meta["train_s"]), meta["env_steps"]))
    if not out:  # Holosoma: model_*.pt converted to .npz (tools/holosoma_ckpt.py stores the time since launch)
        for p in sorted(glob.glob(os.path.join(run, "hs", "**", "model_*.npz"), recursive=True)):
            z = np.load(p)
            if "wall_s" in z.files:
                wall = float(z["wall_s"])
            else:  # no stored time: the .pt file's modification time minus the launch time that <run>/t_start holds
                wall = os.path.getmtime(p[: -len(".npz")] + ".pt") - float(open(os.path.join(run, "t_start")).read().split()[0])
            out.append((p, wall, int(z["iteration"])))
    return out


def load_policy(path, which="avg"):
    """The layers (a list of parameter dicts) of the `avg` (averaged) or the `live` policy in an XSoma checkpoint."""
    z = np.load(path)
    n = 1 + max(int(k.split("/")[1]) for k in z.files if k.startswith(which + "/"))
    return [{k.split("/")[2]: jnp.asarray(z[k]) for k in z.files if k.startswith(f"{which}/{i}/")} for i in range(n)]


def make_sim(n=2048, scale=0.5, sole_friction=False, preset="fast_sac", terrain="mix", steps=task.MAX_EPISODE + 1, cmd_scale=None, push_vel=task.PUSH_VEL, friction_surfaces=()):
    """The evaluation simulator; the defaults are the setting that Holosoma's MuJoCo Warp baselines train in (terrain scale 0.5,
    sole friction fixed at 0.8). `steps`: the policy steps of an episode that does not end in a fall. `cmd_scale`: three
    factors on the command ranges of vx, vy and wz. `push_vel`: the size of the pushes (m/s), 0 for none."""
    return Sim(SimCfg(num_envs=n, terrain=terrain, terrain_scale=scale, preset=preset, sole_friction=sole_friction, random_start=False,
                      max_episode=steps - 1, cmd_scale=cmd_scale and tuple(cmd_scale), push_vel=push_vel,
                      friction_surfaces=friction_surfaces))  # fmt: skip


def run_all(sim: Sim, jobs, seed=1, which="avg", out=None, tracker_meta=None, tracker_heading=None, falls=None, episodes=None):
    """Evaluate (path, wall_s, steps, run) jobs; one compilation per policy type. Yields the result records; a job with a run
    directory appends its record to <run>/<out>.jsonl. `tracker_heading` None: a tracker gets the heading mode that its run's
    config.json names, else "hold". `falls`, `episodes`: files that every job's fall records and episode summaries are appended
    to (module docstring)."""
    progs, key, T = {}, jax.random.PRNGKey(seed), sim.cfg.max_episode
    window = WINDOW if falls or episodes else 0
    for path, wall, steps, run in jobs:
        kind, normalize, params = load(path, which, tracker_meta)
        heading = None
        if kind == "tracker":
            cfg = os.path.join(run or "", "config.json")
            heading = tracker_heading or (json.load(open(cfg)).get("tracker_heading") if run and os.path.exists(cfg) else None) or "hold"
        if (kind, normalize, heading) not in progs:
            act = act_fn(kind, normalize, heading != "rate")
            # the parameters enter as arguments, so every checkpoint of a policy type reuses one compilation
            mem = (lambda p: tracker.memory(p, sim.N)) if kind == "tracker" else (lambda p: None)
            progs[(kind, normalize, heading)] = jax.jit(lambda p, k, act=act, mem=mem: evaluate(sim, jax.tree_util.Partial(act, p), k, T + 2, mem(p), window))
        sums, length, fell, ended, ev, ew, crossed, dy, deep, sink, p99, *legs = jax.device_get(progs[(kind, normalize, heading)](params, key))
        if window:
            for name, recs in zip((episodes, falls), records(sim, length, fell, legs.pop())):
                if name:
                    with open(name, "a") as f:
                        f.writelines(json.dumps({"ckpt": path, **x}) + "\n" for x in recs)
        rew = sums.mean(0) / (T * task.DT)  # the episode sums per second, as Holosoma logs them
        rec = {"ckpt": path, "kind": kind, "wall_s": wall, "steps": steps, "n": sim.N, "scale": sim.cfg.terrain_scale, "sole_friction": sim.cfg.sole_friction,
               "len": float(length.mean()), "success": float(1 - fell.mean()), "ended": float(ended.mean()), "err_v": float(ev.mean()),
               "err_w": float(ew.mean()), "feet_crossed": float(crossed.mean()), "feet_dy": float(dy.mean()),
               "sole_sink_max": float(sink.mean()), "sole_sink_frac": float(deep.mean()), "sole_sink_p99": float(p99),
               **({"legs_overlap": float(legs[0].mean()), "legs_dist": float(legs[1].mean())} if legs else {}),
               **({"tracker_heading": heading} if heading else {}), **({"episode": T + 1} if T != task.MAX_EPISODE else {}),
               **({"cmd_scale": list(sim.cfg.cmd_scale)} if sim.cfg.cmd_scale else {}), **({"push_vel": sim.cfg.push_vel} if sim.cfg.push_vel != task.PUSH_VEL else {}),
               **{f"rew_{k}": float(v) for k, v in zip(task.TERMS, rew)}}  # fmt: skip
        if run and out:
            with open(os.path.join(run, out + ".jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            wandb_log.log(rec, prefix="eval/")
        yield rec


def main(argv=None):
    p = argparse.ArgumentParser(description="Deterministic evaluation of XSoma runs, Holosoma runs or single checkpoints.")
    p.add_argument("paths", nargs="+", help="run directories or .npz checkpoints")
    p.add_argument("--n", type=int, default=2048, help="envs, one episode each")
    p.add_argument("--scale", type=float, default=0.5, help="terrain scale")
    p.add_argument("--sole_friction", type=int, default=0, help="1: the soles' friction is randomized as well")
    p.add_argument("--preset", default="fast_sac", help="reward preset (the logged metrics; both presets weigh tracking alike)")
    p.add_argument("--terrain", default="mix")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--which", default="avg", help="XSoma checkpoints: avg (the averaged policy) or live")
    p.add_argument("--out", default="eval", help="records go to <run>/<out>.jsonl")
    p.add_argument("--tracker_meta", default=None, help="tracker checkpoints: the meta.json of the tracker's robot (tracker.py)")
    p.add_argument("--tracker_heading", default=None, choices=("hold", "rate"),
                   help="tracker checkpoints: the heading mode (train.py); default: the one in the run directory's config.json, else hold")  # fmt: skip
    p.add_argument("--steps", type=int, default=task.MAX_EPISODE + 1, help="policy steps of an episode (the time-out); 50 per second")
    p.add_argument("--cmd_scale", type=float, nargs=3, default=None, help="factors on the command ranges of vx, vy and wz (default: U[-1, 1] each)")
    p.add_argument("--push_vel", type=float, default=task.PUSH_VEL, help="size of the pushes in m/s; 0: no pushes")
    p.add_argument("--friction_surfaces", default="", help="lo:hi,...: friction per episode as train.py's option (with --sole_friction 1 on the soles too)")
    p.add_argument("--falls", default=None, help="write every fall with its last 2 s to this JSON-lines file")
    p.add_argument("--episodes", default=None, help="write a summary of every episode to this JSON-lines file")
    a = p.parse_args(argv)
    sim = make_sim(a.n, a.scale, bool(a.sole_friction), a.preset, a.terrain, a.steps, a.cmd_scale, push_vel=a.push_vel,
                   friction_surfaces=surfaces(a.friction_surfaces))
    for name in (a.falls, a.episodes):
        if name:
            open(name, "w").close()
    jobs = []
    for path in a.paths:
        if os.path.isdir(path):
            jobs += [(c, t, steps, path) for c, t, steps in checkpoints(path)]
        else:
            jobs.append((path, None, None, None))
    for rec in run_all(sim, jobs, a.seed, a.which, a.out, a.tracker_meta, a.tracker_heading, a.falls, a.episodes):
        t = f"{rec['wall_s'] / 60:.1f} min" if rec["wall_s"] is not None else ""
        print(f"[eval] {rec['ckpt']} {t}: length {rec['len']:.0f} no fall {rec['success']:.3f} velocity error {rec['err_v']:.3f} m/s "
              f"yaw-rate error {rec['err_w']:.3f} rad/s | Holosoma's tracking terms lin {rec['rew_tracking_lin_vel']:.3f} "
              f"ang {rec['rew_tracking_ang_vel']:.3f} | feet crossed {rec['feet_crossed']:.3f} of the steps, mean distance {rec['feet_dy']:.3f} m | "
              f"soles under the terrain: an episode's deepest {1000 * rec['sole_sink_max']:.1f} mm, 99 % of the steps within {1000 * rec['sole_sink_p99']:.1f} mm, "
              f"deeper than {1000 * SOLE_SINK:.0f} mm in {rec['sole_sink_frac']:.4f} of the steps"
              + (f" | legs overlap {rec['legs_overlap']:.3f} of the steps, mean smallest distance {rec['legs_dist']:.3f} m" if "legs_overlap" in rec else ""), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
