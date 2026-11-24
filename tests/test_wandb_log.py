"""Weights & Biases logging (xsoma/wandb_log.py), with a stand-in for the wandb package (run: python tests/test_wandb_log.py, or
pytest; needs no wandb installation and no network).

  off      without --wandb_project (and WANDB_PROJECT) a run does not start wandb
  run      a short train.py run on test_finetune's stand-in simulator sends each log.jsonl record's numbers at its iteration,
           then each eval.jsonl record's numbers under eval/; the settings switch off console, code, git, metadata and system
           metrics; the flags reach wandb.init
  config   the configuration sent holds no option that names a file or directory (no key naming one, no value that looks
           like a path), and keeps the hyperparameters
  fallback a wandb that cannot start online (login, network) leaves one warning and an offline run; one that cannot start at
           all, or fails while logging, leaves a warning and training goes on to its end
  real     with the wandb package installed: an offline run's record (what `wandb sync` uploads) holds the numbers and the
           filtered configuration, not a path, console line, program or host name; no output, requirements, metadata or code file
"""

import contextlib
import glob
import io
import json
import os
import socket
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp  # noqa: E402
import test_finetune as tf  # noqa: E402

from xsoma import evaluate, task, train, wandb_log  # noqa: E402
from xsoma.sim import SimCfg  # noqa: E402

FLAGS = [x for k, v in tf.SMALL.items() for x in (f"--{k}", str(v))] + ["--minutes", "0.002", "--log_every", "2", "--eval_n", str(tf.N)]


class LogSim(tf.FakeSim):
    """test_finetune's stand-in, with the remaining per-step statistics of sim.Sim that train.py logs."""

    def step(self, s, action):
        s1, tr = super().step(s, action)
        z = jnp.float32(0)
        more = {"terminated": z, "len_sum": tr.stats["done"] * 3, "rew_sums": jnp.ones(len(task.TERMS)), "term_geom": z, "diverged": z, "nacon": z,
                "scale": s.scale, "avg_len": s.avg_len, "pushes": z}  # fmt: skip
        return s1, tr._replace(stats={**tr.stats, **more})


class FakeWandb:
    """wandb.Settings, wandb.init and run.log, recorded."""

    def __init__(self, fail=()):
        self.init_kw, self.logged, self.metrics, self.fail = None, [], {}, fail  # fail: modes whose init raises, or "log"

    def Settings(self, **kw):
        return kw

    def init(self, **kw):
        if kw["mode"] in self.fail:
            raise RuntimeError("stand-in: cannot start")
        self.init_kw = kw
        return self

    def define_metric(self, name, step_metric):
        self.metrics[name] = step_metric

    def log(self, data, step=None):
        if "log" in self.fail:
            raise RuntimeError("stand-in: cannot log")
        self.logged.append((step, data))


def run(flags, fail=()):
    """train.main with `flags` on the stand-in simulator: (the fake wandb, log.jsonl records, eval.jsonl records, the output)."""
    fake, wandb_log._run, out = FakeWandb(fail), None, io.StringIO()
    with mock.patch.dict(sys.modules, {"wandb": fake}), mock.patch.multiple(task, **tf.ROBOT), mock.patch.object(train, "Sim", LogSim), \
            mock.patch.object(evaluate, "make_sim", lambda n, *a, **k: LogSim(SimCfg(num_envs=n))), \
            tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(out):  # fmt: skip
        train.main(FLAGS + ["--log_dir", d] + flags)
        recs, evals = ([json.loads(line) for line in open(os.path.join(d, name))] for name in ("log.jsonl", "eval.jsonl"))
        assert fake.init_kw is None or fake.init_kw["dir"] == d
    wandb_log._run = None
    return fake, recs, evals, out.getvalue()


def test_wandb_log():
    with mock.patch.dict(os.environ, {"WANDB_PROJECT": ""}):
        fake, recs, _, _ = run([])
    assert fake.init_kw is None and not fake.logged and recs
    print(f"off: {len(recs)} log records, wandb not started")

    fake, recs, evals, _ = run(["--wandb_project", "proj", "--wandb_entity", "team", "--wandb_name", "run-a", "--wandb_mode", "offline"])
    kw = fake.init_kw
    assert (kw["project"], kw["entity"], kw["name"], kw["mode"]) == ("proj", "team", "run-a", "offline")
    assert kw["settings"]["console"] == "off" and not kw["settings"]["save_code"] and kw["settings"]["disable_git"] and kw["settings"]["x_disable_meta"]
    assert kw["settings"]["x_disable_stats"] and not kw["settings"]["x_save_requirements"] and fake.metrics == {"eval/*": "eval/wall_s"}
    train_logs, eval_logs = fake.logged[: len(recs)], fake.logged[len(recs) :]
    assert [s for s, _ in train_logs] == [r["it"] for r in recs] and evals and len(eval_logs) == len(evals)
    for (_, sent), rec in zip(train_logs, recs):
        assert sent == {k: v for k, v in rec.items() if isinstance(v, (int, float))} and "wm" in sent and "rew_tracking_lin_vel" in sent
    for (step, sent), rec in zip(eval_logs, evals):
        assert step is None and sent == {"eval/" + k: v for k, v in rec.items() if isinstance(v, (int, float))} and "eval/success" in sent
        assert not any(isinstance(v, str) for v in sent.values()) and "eval/ckpt" not in sent
    print(f"run: {len(recs)} log records at iterations {[s for s, _ in train_logs]} and {len(evals)} evaluation record(s) sent, {len(sent)} numbers each")

    config = kw["config"]
    names_file = lambda s: wandb_log.PATH_KEY.search(s) or wandb_log.PATH_VALUE.search(s)  # noqa: E731
    assert not any(names_file(k) or (isinstance(v, str) and names_file(v)) for k, v in config.items()), config
    assert not {"robot_dir", "init_tracker", "tracker_meta", "log_dir", "jax_cache"} & set(config) and not any(k.startswith("wandb") for k in config)
    assert {"pi_lr", "pi_clip", "horizon", "num_envs", "support_gap", "seed"} <= set(config) and config["num_envs"] == tf.N
    paths = {"robot_dir": "robots/x", "init_tracker": "a.npz", "tracker_meta": "m.json", "notes": "~/x", "home": "C:\\runs", "pi_lr": 1e-3}
    assert wandb_log.hparams(paths) == {"pi_lr": 1e-3}
    print(f"config: {len(config)} options sent, none naming a file or directory; dropped: "
          f"{sorted(set(train.Config.__dataclass_fields__) - set(config))}")


def test_fallback():
    online = ["--wandb_project", "proj", "--wandb_mode", "online"]
    fake, recs, evals, out = run(online, fail=("online",))
    assert fake.init_kw["mode"] == "offline" and out.count("[wandb] warning") == 1 and "logging offline to" in out
    assert len(fake.logged) == len(recs) + len(evals) and recs[-1]["train_s"] >= 0.12 and evals
    for fail in (("online", "offline"), ("log",)):
        fake, recs, evals, out = run(online, fail=fail)
        assert not fake.logged and out.count("[wandb] warning") == len(fail) and recs[-1]["train_s"] >= 0.12 and evals, out
    print("fallback: online start failed -> one warning, offline run with every record; no start at all, or a failing log -> "
          "warnings, training and evaluation to their end")


def test_real():
    try:
        import wandb  # noqa: F401
    except ImportError:
        print("real: wandb not installed, skipped")
        return
    wandb_log._run = None
    with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"WANDB_SILENT": "true"}):
        config = {"wandb_project": "proj", "wandb_mode": "offline", "lr": 0.001, "data_dir": "x/y", "init_ckpt": "a.npz"}
        wandb_log.start(config, d)
        print("console output that must stay local")
        for it in (1, 2):
            wandb_log.log({"it": it, "loss": 1.0 / it, "name": "x/y"}, step=it)
        wandb_log._run.finish()
        wandb_log._run = None
        (record,) = glob.glob(os.path.join(d, "wandb", "offline-run-*", "run-*.wandb"))  # what `wandb sync` uploads
        run_dir = os.path.dirname(record)
        others = [f for f in glob.glob(os.path.join(run_dir, "**"), recursive=True) if os.path.isfile(f) and f != record]
        assert all(os.path.relpath(f, run_dir).startswith("logs" + os.sep) for f in others), others  # wandb's local debug logs only
        data = open(record, "rb").read()
        sent = [b"lr", b"loss", b"0.001"]
        kept = [b"x/y", b"a.npz", b"data_dir", b"init_ckpt", b"must stay local", b"test_wandb_log", socket.gethostname().encode()]
        assert all(x in data for x in sent) and not [x for x in kept if x in data], [x for x in kept if x in data]
    print(f"real: wandb {wandb.__version__} offline run: its record holds the numbers and the filtered configuration; no path, console line, "
          "program name or host name, and no output, requirements, metadata or code file")


if __name__ == "__main__":
    test_wandb_log()
    test_fallback()
    test_real()
    print("\nall checks passed")
