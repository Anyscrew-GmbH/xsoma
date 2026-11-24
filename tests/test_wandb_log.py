"""Weights & Biases logging (xsoma/wandb_log.py), with a stand-in for the wandb package (run: python tests/test_wandb_log.py, or
pytest; needs no wandb installation and no network).

  off      without --wandb_project (and WANDB_PROJECT) a run does not start wandb
  run      a short train.py run on test_finetune's stand-in simulator sends each log.jsonl record's numbers at its iteration,
           then each eval.jsonl record's numbers under eval/; the settings switch off console, code, git, metadata and system
           metrics; the flags reach wandb.init
  config   the configuration sent holds no option that names a file or directory (no key naming one, no value that looks
           like a path), and keeps the hyperparameters
"""

import contextlib
import io
import json
import os
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

    def __init__(self):
        self.init_kw, self.logged, self.metrics = None, [], {}

    def Settings(self, **kw):
        return kw

    def init(self, **kw):
        self.init_kw = kw
        return self

    def define_metric(self, name, step_metric):
        self.metrics[name] = step_metric

    def log(self, data, step=None):
        self.logged.append((step, data))


def run(flags):
    """train.main with `flags` on the stand-in simulator: (the fake wandb, log.jsonl records, eval.jsonl records)."""
    fake, wandb_log._run = FakeWandb(), None
    with mock.patch.dict(sys.modules, {"wandb": fake}), mock.patch.multiple(task, **tf.ROBOT), mock.patch.object(train, "Sim", LogSim), \
            mock.patch.object(evaluate, "make_sim", lambda n, *a, **k: LogSim(SimCfg(num_envs=n))), \
            tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):  # fmt: skip
        train.main(FLAGS + ["--log_dir", d] + flags)
        recs, evals = ([json.loads(line) for line in open(os.path.join(d, name))] for name in ("log.jsonl", "eval.jsonl"))
        assert fake.init_kw is None or fake.init_kw["dir"] == d
    wandb_log._run = None
    return fake, recs, evals


def test_wandb_log():
    with mock.patch.dict(os.environ, {"WANDB_PROJECT": ""}):
        fake, recs, _ = run([])
    assert fake.init_kw is None and not fake.logged and recs
    print(f"off: {len(recs)} log records, wandb not started")

    fake, recs, evals = run(["--wandb_project", "proj", "--wandb_entity", "team", "--wandb_name", "run-a", "--wandb_mode", "offline"])
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


if __name__ == "__main__":
    test_wandb_log()
    print("\nall checks passed")
