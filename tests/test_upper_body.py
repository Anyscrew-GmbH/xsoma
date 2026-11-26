"""The upper-body tilt and sway penalties (run: python tests/test_upper_body.py, or pytest). CPU only.

  lateral_tilt  task.lateral_tilt: the up axis of a body along the pelvis heading's lateral axis; 0 upright or pitched, sin(roll) leaning
  upper_tilt    task.upper_tilt: its square, the same whatever way the pelvis faces
  upper_rate    task.upper_rate: the squared change of the tilt over a policy step / DT; a steady lean is free
  standin       train.py --upper_tilt / --upper_rate on test_finetune's stand-in, the body's pose by forward kinematics of a small model
"""

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from xsoma import task  # noqa: E402

# a free root and one body on a hinge about the root's x axis, 0.3 m above it
SMALL = """<mujoco><worldbody><body name="robot_root"><freejoint/><geom size="0.1"/>
<body name="robot_upper" pos="0 0 0.3"><joint name="roll" axis="1 0 0"/><geom size="0.1"/></body></body></worldbody></mujoco>"""


def rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def tilt(Rb, R):
    return np.asarray(task.lateral_tilt(jnp.asarray(Rb[None], jnp.float32), jnp.asarray(R[None], jnp.float32)))[0]


def test_lateral_tilt():
    lean = rot("x", -0.2)  # a rotation about -x leans the up axis toward +y (left)
    assert abs(tilt(np.eye(3), np.eye(3))) < 1e-7 and abs(tilt(rot("y", 0.3), np.eye(3))) < 1e-7  # upright, or pitched only
    assert np.isclose(tilt(lean, np.eye(3)), np.sin(0.2), atol=1e-6)
    yaw = rot("z", np.pi / 2)  # facing +y, the same lean in that heading's frame
    assert np.isclose(tilt(yaw @ lean, yaw), np.sin(0.2), atol=1e-6)
    assert np.isclose(float(task.upper_tilt(jnp.asarray((yaw @ lean)[None], jnp.float32), jnp.asarray(yaw[None], jnp.float32))[0]), np.sin(0.2) ** 2, atol=1e-6)
    print("lateral_tilt: 0 upright or pitched, sin(roll) leaning left, in the heading's frame; upper_tilt its square")


def test_upper_rate():
    f = lambda a, b: float(task.upper_rate(*(jnp.asarray(x[None], jnp.float32) for x in (rot("x", -a), np.eye(3), rot("x", -b), np.eye(3))))[0])  # noqa: E731
    assert f(0.1, 0.1) < 1e-8  # a steady lean
    assert np.isclose(f(0.1, 0.12), ((np.sin(0.12) - np.sin(0.1)) / task.DT) ** 2, rtol=1e-3)
    Rx = lambda b: jnp.asarray([[1.0, 0.0, 0.0], [0.0, jnp.cos(b), jnp.sin(b)], [0.0, -jnp.sin(b), jnp.cos(b)]])[None]  # noqa: E731  (about -x by b)
    g = jax.grad(lambda b: task.upper_rate(Rx(0.0), jnp.eye(3)[None], Rx(b), jnp.eye(3)[None])[0])(0.05)
    assert np.isfinite(g) and g > 0, g
    print("upper_rate: ((tilt1 - tilt0) / DT)^2, a steady lean free; gradient finite")


def test_standin():
    """On test_finetune's stand-in with a small model's kinematics: the loss gains W x dt x scale x the discounted term."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_finetune as tf
    import test_impact as ti

    def upper(R, q):  # the body's world rotation: the root's, then the hinge about x by q[..., 0]
        c, s = np.cos(q[:, 0]), np.sin(q[:, 0])
        o, z = np.ones_like(c), np.zeros_like(c)
        return np.asarray(R) @ np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)

    with mock.patch.object(tf.FakeSim, "m", mujoco.MjModel.from_xml_string(SMALL), create=True):
        ti.check_standin("upper_tilt", "upper_tilt", {"upper_tilt": 3.0, "upper_body": "upper"},
                         lambda pen, v, v1: np.asarray(task.upper_tilt(upper(v1.robot.R, np.asarray(v1.robot.q)), v1.robot.R)))
        ti.check_standin("upper_rate", "upper_rate", {"upper_rate": 3.0, "upper_body": "upper"},
                         lambda pen, v, v1: np.asarray(task.upper_rate(upper(v.robot.R, np.asarray(v.robot.q)), v.robot.R, upper(v1.robot.R, np.asarray(v1.robot.q)), v1.robot.R)))


if __name__ == "__main__":
    test_lateral_tilt()
    test_upper_rate()
    test_standin()
