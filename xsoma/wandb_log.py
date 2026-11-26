"""Optional Weights & Biases logging of a run's scalar metrics and hyperparameters.

Off unless --wandb_project (or the environment variable WANDB_PROJECT) names a project; only then is wandb imported. The
other options (--wandb_entity, --wandb_name, --wandb_mode online | offline | disabled) fall back to wandb's own environment
variables when empty. What is sent: the numbers of the log records and the configuration without the options that name a
file or directory. What is not: console output, code, git state, the command line, the host name, system metadata and
metrics, files.
"""

import os
import re

PATH_KEY = re.compile(r"(^|_)(dir|path|file|meta|motion|clips|data|dataset|cache|init)(_|$)", re.IGNORECASE)  # options naming files
PATH_VALUE = re.compile(r"[/\\~]|\.[A-Za-z]\w*$")  # a path separator, a home directory or a file extension

_run = None


def hparams(config: dict) -> dict:
    """`config` without the entries whose key or value names a file or directory, and without the wandb options."""
    return {k: v for k, v in config.items()
            if not k.startswith("wandb_") and (isinstance(v, (int, float)) or not PATH_KEY.search(k))  # a number never names a file
            and not (isinstance(v, str) and PATH_VALUE.search(v))}  # fmt: skip


def start(config: dict, log_dir: str):
    """Start a run if a project is given; wandb keeps its local files in `log_dir`. If wandb cannot start (login, network), the
    run is kept offline in <log_dir>/wandb (upload it later with `wandb sync`); W&B never stops training."""
    global _run
    project = config.get("wandb_project") or os.environ.get("WANDB_PROJECT")
    if not project:
        return
    try:
        import wandb
    except ImportError as e:
        print(f"[wandb] warning: not logging, {e!r}", flush=True)
        return
    settings = wandb.Settings(console="off", save_code=False, disable_code=True, disable_git=True, x_disable_meta=True,
                              x_disable_stats=True, x_disable_machine_info=True, x_save_requirements=False, host="")  # fmt: skip
    kw = dict(project=project, entity=config.get("wandb_entity") or None, name=config.get("wandb_name") or None, dir=log_dir,
              config=hparams(config), settings=settings)  # fmt: skip
    for mode in dict.fromkeys((config.get("wandb_mode") or None, "offline")):
        try:
            _run = wandb.init(mode=mode, **kw)
            _run.define_metric("eval/*", step_metric="eval/wall_s")  # evaluation records: against their checkpoint's time
            return
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] warning: start {'failed' if mode == 'offline' else 'failed, logging offline to ' + log_dir + '/wandb'}: {e!r}", flush=True)
    _run = None


# W&B sections: overview/ (is it learning to walk), rewards/ (the environment's reward and its terms), actor/ (policy loss,
# gradients, KL), critic/ (value), world_model/ (model losses and errors), metrics/ (speed, steps, learning rates, health).
# A key not listed: rew_* -> rewards/, else metrics/. Overview keys are also logged in their own section.
OVERVIEW = {"reward", "ep_len", "avg_len", "surv", "term_env", "rew_tracking_lin_vel", "rew_tracking_ang_vel",  # fine-tuning
            "ep_ret", "sim_r", "term_frac"}  # imitation
GROUPS = {"rewards": {"reward", "r", "sim_r", "ir", "ep_ret"},
          "actor": {"G", "pi", "pi_lr", "gn", "gn_gated", "scale", "kl", "kl_term", "kl_beta", "support", "sym"},
          "critic": {"v", "q"},
          "world_model": {"wm", "wm8", "wm_gated", "h_err8", "h_errK", "term_wm", "diverged", "anchor"}}
_last = None  # (wall seconds, steps) of the previous record, for steps per second


def sections(record: dict) -> dict:
    """{section/key: number} for a training record, plus metrics/steps_per_s from the previous record."""
    global _last
    out = {}
    for k, v in record.items():
        if not isinstance(v, (int, float)):
            continue
        g = next((g for g, ks in GROUPS.items() if k in ks), "rewards" if k.startswith("rew_") else "metrics")
        out[f"{g}/{k}"] = v
        if k in OVERVIEW:
            out[f"overview/{k}"] = v
    wall = record.get("wall_s", 60.0 * record["wall_min"] if isinstance(record.get("wall_min"), (int, float)) else None)
    steps = record.get("env_steps", record.get("sim_steps"))
    if isinstance(wall, (int, float)) and isinstance(steps, (int, float)):
        if _last is not None and wall > _last[0]:
            out["metrics/steps_per_s"] = (steps - _last[1]) / (wall - _last[0])
        _last = (wall, steps)
    return out


def log(record: dict, step=None, prefix=""):
    """The record's numbers (no strings, lists or None), if a run was started: training records in their sections (see
    GROUPS), records with a prefix (evaluation: "eval/") under it as they are."""
    global _run
    if _run is not None:
        try:
            nums = {prefix + k: v for k, v in record.items() if isinstance(v, (int, float))} if prefix else sections(record)
            _run.log(nums, step=step)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] warning: logging stopped: {e!r}", flush=True)
            _run = None
