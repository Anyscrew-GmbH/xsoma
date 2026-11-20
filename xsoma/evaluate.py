"""Deterministic evaluation of policies in the simulator. Every env runs one full episode (up to 1001 steps, 20 s) with the
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
    python -m xsoma.evaluate <tracker checkpoint> --tracker_meta <the tracker's meta.json> [--tracker_heading rate] [the same options]

A run directory is evaluated checkpoint by checkpoint (XSoma: ckpt/*.npz, with --tracker_meta if the run fine-tuned a tracker;
Holosoma: hs/**/model_*.npz), each with its wall-clock time since the run's launch, and the records are appended to
<run>/eval.jsonl. The defaults are the setting that Holosoma's MuJoCo Warp baselines train in: terrain scale 0.5, sole friction
fixed at 0.8. train.py calls this module after training.
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

from . import nets, task, tracker  # noqa: E402
from .sim import Sim, SimCfg, unflatten  # noqa: E402

SOLE_SINK = 0.01  # m: sole_sink_frac counts the steps in which a sole collider point is further than this below the terrain


def evaluate(sim: Sim, act, key, steps: int = task.MAX_EPISODE + 2, mem=None):
    """Run every env for one episode with the deterministic policy `act(obs) -> action`; with `mem` (a policy's initial memory)
    it is `act(mem, state) -> (action, mem)`. The default number of steps lets every env reach the time-out. Returns, per env,
    for its first episode only: the sums of the weighted reward terms, the length, whether it ended in a fall, whether it ended
    at all, and while alive the mean velocity and yaw-rate tracking errors, the share of steps with the feet crossed, the mean
    of task.feet_dy and the share of steps with a sole point more than SOLE_SINK below the terrain; then the largest depth of a
    sole point below the terrain and, as one number for all envs, the 99th percentile of that depth over the alive steps; with
    task.SELF_COLLISION also the share of alive steps in which the two groups overlap and the mean of their smallest distance
    (sim.SelfDistance)."""
    N = sim.N
    s = sim.init(key)
    s = s._replace(scale=jnp.ones_like(s.scale))  # penalties at full weight

    def body(carry, _):
        s, mem, sums, ended, length, fell, sink, err_v, err_w, crossed, dy, deep, legs, alive_steps = carry
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
        # a step updates the penalty curriculum, so the scale is set back to 1
        return (s1._replace(scale=jnp.ones_like(s1.scale)), mem, sums, ended | end, length, fell, sink, err_v, err_w, crossed, dy, deep, legs, alive_steps), jnp.where(live, depth, jnp.nan)

    z = jnp.zeros((N,))
    init = (s, mem, jnp.zeros((N, len(task.TERMS))), jnp.zeros((N,), bool), z, jnp.zeros((N,), bool), z, z, z, z, z, z, (z, z) if task.SELF_COLLISION else (), z)
    out, depth = jax.lax.scan(body, init, None, length=steps)
    _, _, sums, ended, length, fell, sink, *means, legs, alive_steps = out
    means, legs = (tuple(x / jnp.maximum(alive_steps, 1) for x in xs) for xs in (means, legs))
    return (sums, length, fell, ended, *means, sink, jnp.nanpercentile(depth, 99.0), *legs)


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


def make_sim(n=2048, scale=0.5, sole_friction=False, preset="fast_sac", terrain="mix"):
    """The evaluation simulator; the defaults are the setting that Holosoma's MuJoCo Warp baselines train in (terrain scale 0.5,
    sole friction fixed at 0.8)."""
    return Sim(SimCfg(num_envs=n, terrain=terrain, terrain_scale=scale, preset=preset, sole_friction=sole_friction, random_start=False))


def run_all(sim: Sim, jobs, seed=1, which="avg", out=None, tracker_meta=None, tracker_heading=None):
    """Evaluate (path, wall_s, steps, run) jobs; one compilation per policy type. Yields the result records; a job with a run
    directory appends its record to <run>/<out>.jsonl. `tracker_heading` None: a tracker gets the heading mode that its run's
    config.json names, else "hold"."""
    progs, key = {}, jax.random.PRNGKey(seed)
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
            progs[(kind, normalize, heading)] = jax.jit(lambda p, k, act=act, mem=mem: evaluate(sim, jax.tree_util.Partial(act, p), k, mem=mem(p)))
        sums, length, fell, ended, ev, ew, crossed, dy, deep, sink, p99, *legs = jax.device_get(progs[(kind, normalize, heading)](params, key))
        rew = sums.mean(0) / (task.MAX_EPISODE * task.DT)  # the episode sums per second, as Holosoma logs them
        rec = {"ckpt": path, "kind": kind, "wall_s": wall, "steps": steps, "n": sim.N, "scale": sim.cfg.terrain_scale, "sole_friction": sim.cfg.sole_friction,
               "len": float(length.mean()), "success": float(1 - fell.mean()), "ended": float(ended.mean()), "err_v": float(ev.mean()),
               "err_w": float(ew.mean()), "feet_crossed": float(crossed.mean()), "feet_dy": float(dy.mean()),
               "sole_sink_max": float(sink.mean()), "sole_sink_frac": float(deep.mean()), "sole_sink_p99": float(p99),
               **({"legs_overlap": float(legs[0].mean()), "legs_dist": float(legs[1].mean())} if legs else {}),
               **({"tracker_heading": heading} if heading else {}), **{f"rew_{k}": float(v) for k, v in zip(task.TERMS, rew)}}  # fmt: skip
        if run and out:
            with open(os.path.join(run, out + ".jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
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
    a = p.parse_args(argv)
    sim = make_sim(a.n, a.scale, bool(a.sole_friction), a.preset, a.terrain)
    jobs = []
    for path in a.paths:
        if os.path.isdir(path):
            jobs += [(c, t, steps, path) for c, t, steps in checkpoints(path)]
        else:
            jobs.append((path, None, None, None))
    for rec in run_all(sim, jobs, a.seed, a.which, a.out, a.tracker_meta, a.tracker_heading):
        t = f"{rec['wall_s'] / 60:.1f} min" if rec["wall_s"] is not None else ""
        print(f"[eval] {rec['ckpt']} {t}: length {rec['len']:.0f} no fall {rec['success']:.3f} velocity error {rec['err_v']:.3f} m/s "
              f"yaw-rate error {rec['err_w']:.3f} rad/s | Holosoma's tracking terms lin {rec['rew_tracking_lin_vel']:.3f} "
              f"ang {rec['rew_tracking_ang_vel']:.3f} | feet crossed {rec['feet_crossed']:.3f} of the steps, mean distance {rec['feet_dy']:.3f} m | "
              f"soles under the terrain: an episode's deepest {1000 * rec['sole_sink_max']:.1f} mm, 99 % of the steps within {1000 * rec['sole_sink_p99']:.1f} mm, "
              f"deeper than {1000 * SOLE_SINK:.0f} mm in {rec['sole_sink_frac']:.4f} of the steps"
              + (f" | legs overlap {rec['legs_overlap']:.3f} of the steps, mean smallest distance {rec['legs_dist']:.3f} m" if "legs_overlap" in rec else ""), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
