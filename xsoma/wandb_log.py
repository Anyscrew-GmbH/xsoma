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
            if not k.startswith("wandb_") and not PATH_KEY.search(k) and not (isinstance(v, str) and PATH_VALUE.search(v))}  # fmt: skip


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


def log(record: dict, step=None, prefix=""):
    """The record's numbers (no strings, lists or None), if a run was started."""
    global _run
    if _run is not None:
        try:
            _run.log({prefix + k: v for k, v in record.items() if isinstance(v, (int, float))}, step=step)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] warning: logging stopped: {e!r}", flush=True)
            _run = None
