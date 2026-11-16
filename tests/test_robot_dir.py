"""The robot directory (XSOMA_ROBOT_DIR, train.py --robot_dir) on the bundled G1 (run: python tests/test_robot_dir.py, or pytest):
the G1's values written to a robot.json, next to its MJCF as robot.xml, give the same task constants and the same compiled scene
as the built-in default.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# robot.json's keys and the constants of task.py they fill
KEYS = {"dof": "DOF", "default_q": "DEFAULT_Q", "kp": "KP", "kd": "KD", "effort": "EFFORT", "action_scale": "ACTION_SCALE",
        "foot_points": "FOOT_POINTS", "feet": "FEET", "termination_bodies": "TERMINATION_BODIES", "pose_w": "POSE_W",
        "mass_bodies": "MASS_BODIES", "torso": "TORSO", "spawn_height": "SPAWN_HEIGHT"}  # fmt: skip


def fingerprint():
    """What the selected robot gives: every robot constant of task.py (value and type), the sizes that follow from the joint
    count, and a hash of the compiled scene."""
    import mujoco
    import numpy as np

    from xsoma import scene, task

    m = scene.build()
    buf = np.zeros(mujoco.mj_sizeModel(m), np.uint8)
    mujoco.mj_saveModel(m, None, buf)
    values = {k: np.asarray(getattr(task, c)).tolist() for k, c in KEYS.items()}
    types = {k: f"{type(getattr(task, c)).__name__} {np.asarray(getattr(task, c)).dtype}" for k, c in KEYS.items()}
    return {"values": values, "types": types, "dims": [task.J, task.ACTOR_DIM, task.CRITIC_DIM], "scene": hashlib.sha256(buf).hexdigest(),
            "source": [task.ROBOT_DIR, os.path.basename(scene.G1_XML)]}  # fmt: skip


def test_g1_as_robot_dir():
    from xsoma import scene

    ref = fingerprint()
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(scene.G1_XML, os.path.join(d, "robot.xml"))
        with open(os.path.join(d, "robot.json"), "w") as f:
            json.dump(ref["values"], f)
        env = {k: v for k, v in os.environ.items() if k != "XSOMA_G1_XML"} | {"XSOMA_ROBOT_DIR": d}
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "fingerprint"], env=env, capture_output=True, text=True, check=True).stdout
        got = json.loads(out.splitlines()[-1])
        assert got.pop("source") == [d, "robot.xml"] and ref.pop("source")[0] is None, "the child must have read the directory"
    for k in ref:
        assert got[k] == ref[k], f"{k}: {got[k]} != {ref[k]}"
    print(f"the G1 as a robot directory: {len(KEYS)} task constants, {ref['dims']} sizes and the compiled scene equal the default's")


if __name__ == "__main__":
    if sys.argv[1:] == ["fingerprint"]:  # the child process, with XSOMA_ROBOT_DIR set
        print(json.dumps(fingerprint()))
    else:
        assert not os.environ.get("XSOMA_ROBOT_DIR"), "run this test without XSOMA_ROBOT_DIR: it compares against the default robot"
        test_g1_as_robot_dir()
        print("\nall checks passed")
