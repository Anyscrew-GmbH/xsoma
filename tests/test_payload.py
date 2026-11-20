"""robot.json's `payload` (run: python tests/test_payload.py, or pytest) on the bundled G1 as a robot directory, with two welded
bodies added under the wrists: the loader accepts it and rejects a missing body, a jointed one, one with a collider and an
unordered range; the start-up randomization draws each body's mass and CoM in its range (y mirrored for the second body), scales
its inertia by the mass ratio and leaves every other draw as it is; the context grows by 4 entries per body, holding the added
mass and the CoM offset. Runs on the CPU (the world's draws need no physics step).
"""

import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N = 512
NOMINAL = {"mass": 1.0, "pos": [0.0, 0.02, -0.01], "inertia": [1e-3, 2e-3, 3e-3]}
PAYLOAD = [{"bodies": ["left_payload", "right_payload"], "mass": [0.4, 1.6], "com_lo": [-0.01, 0.01, -0.12], "com_hi": [0.02, 0.03, -0.03]}]


def probe():
    """The child process: build the simulator, draw the worlds and print the context and the payload bodies' model fields."""
    import types

    import jax
    import jax.numpy as jnp
    import numpy as np

    from xsoma import task
    from xsoma.sim import Sim, SimCfg

    sim = Sim(SimCfg(num_envs=N, terrain="plane"))
    w = sim.world(jax.random.PRNGKey(3))
    xi = sim.context(types.SimpleNamespace(kp=jnp.ones((N, task.J)), kd=jnp.ones((N, task.J)), world=w))
    p = sim.payload
    f = lambda x: np.asarray(x).tolist()  # noqa: E731
    return {"xi": f(xi), "mass": f(w.body_mass[:, p]), "ipos": f(w.body_ipos[:, p]), "inertia": f(w.body_inertia[:, p]),
            "friction": f(w.geom_friction[:, 0, 0]), "lo": f(task.PAYLOAD_LO), "hi": f(task.PAYLOAD_HI)}  # fmt: skip


def robot_dir(d, payload=PAYLOAD, extra=""):
    """The G1 as a robot directory in d, with a welded payload body under each wrist (`extra` goes into the left one)."""
    from test_robot_dir import fingerprint

    from xsoma import scene

    xml = open(scene.G1_XML).read()
    for side, y in (("left", 1), ("right", -1)):
        n = NOMINAL
        body = (f'<body name="{side}_payload" pos="0.1 0 0"><inertial pos="{n["pos"][0]} {y * n["pos"][1]} {n["pos"][2]}" mass="{n["mass"]}" '
                f'diaginertia="{" ".join(map(str, n["inertia"]))}"/>{extra if side == "left" else ""}</body>')  # fmt: skip
        xml = re.sub(rf'(<body name="{side}_wrist_yaw_link"[^>]*>)', lambda m: m.group(1) + body, xml, count=1)
    with open(os.path.join(d, "robot.xml"), "w") as f:
        f.write(xml)
    values = fingerprint()["values"] | ({"payload": payload} if payload is not None else {})
    with open(os.path.join(d, "robot.json"), "w") as f:
        json.dump(values, f)
    return d


def run(d):
    env = {k: v for k, v in os.environ.items() if k != "XSOMA_G1_XML"} | {"XSOMA_ROBOT_DIR": d, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([sys.executable, os.path.abspath(__file__), "probe"], env=env, capture_output=True, text=True)


def test_payload():
    import numpy as np

    with tempfile.TemporaryDirectory() as root:
        bad = {"a missing body": ([{**PAYLOAD[0], "bodies": ["left_payload", "no_such_body"]}], "", "is not a body of robot.xml"),
               "a jointed body": (PAYLOAD, '<joint name="left_payload_joint"/>', "has a joint"),
               "a collider": (PAYLOAD, '<geom type="sphere" size="0.02" contype="1" conaffinity="1"/>', "has a collider"),
               "an unordered range": ([{**PAYLOAD[0], "mass": [1.6, 0.4]}], "", "ranges must be lo <= hi")}  # fmt: skip
        for case, (payload, extra, msg) in bad.items():
            r = run(robot_dir(tempfile.mkdtemp(dir=root), payload, extra))
            assert r.returncode != 0 and msg in r.stderr, f"{case}: not rejected with '{msg}':\n{r.stderr[-2000:]}"
        print(f"rejected: {', '.join(bad)}")

        outs = {}
        for name, payload in (("with", PAYLOAD), ("without", None)):
            r = run(robot_dir(tempfile.mkdtemp(dir=root), payload))
            assert r.returncode == 0, r.stderr[-3000:]
            outs[name] = {k: np.asarray(v) for k, v in json.loads(r.stdout.splitlines()[-1]).items()}
    a, b = outs["with"], outs["without"]
    nb = len(PAYLOAD[0]["bodies"])

    lo, hi = a["lo"], a["hi"]
    e = PAYLOAD[0]
    assert np.allclose(lo, [[e["mass"][0], *e["com_lo"]], [e["mass"][0], e["com_lo"][0], -e["com_hi"][1], e["com_lo"][2]]])
    assert np.allclose(hi, [[e["mass"][1], *e["com_hi"]], [e["mass"][1], e["com_hi"][0], -e["com_lo"][1], e["com_hi"][2]]])
    drawn = np.concatenate((a["mass"][..., None], a["ipos"]), -1)  # (N, bodies, 4)
    assert (drawn >= lo).all() and (drawn <= hi).all(), "a draw outside its range"
    assert (drawn.min(0) < lo + 0.05 * (hi - lo)).all() and (drawn.max(0) > hi - 0.05 * (hi - lo)).all(), "draws do not fill their ranges"
    assert (drawn[:, 1, 2] < 0).all() and (drawn[:, 0, 2] > 0).all(), "the second body's y is not mirrored"
    assert np.allclose(a["inertia"], a["mass"][..., None] / NOMINAL["mass"] * np.asarray(NOMINAL["inertia"]), rtol=1e-5), "inertia not by the mass ratio"
    print(f"drawn: mass {drawn[..., 0].min():.3f} to {drawn[..., 0].max():.3f} kg, CoM y of the second body {drawn[:, 1, 2].min():.3f} to "
          f"{drawn[:, 1, 2].max():.3f} m, inertia = nominal x mass ratio")  # fmt: skip

    assert b["xi"].shape == (N, 72) and b["mass"].size == 0, b["xi"].shape
    assert a["xi"].shape == (N, 72 + 4 * nb), a["xi"].shape
    assert np.array_equal(a["xi"][:, :72], b["xi"]) and np.array_equal(a["friction"], b["friction"]), "the payload changed the other draws"
    pos = np.asarray([NOMINAL["pos"], np.asarray(NOMINAL["pos"]) * [1, -1, 1]])
    want = np.concatenate(((a["mass"] - NOMINAL["mass"])[..., None], a["ipos"] - pos), -1).reshape(N, -1)
    assert np.allclose(a["xi"][:, 72:], want, atol=1e-6), "context tail != added mass and CoM offset per body"
    print(f"context: {b['xi'].shape[1]} without the payload, {a['xi'].shape[1]} with it; the first 72 and the friction draws equal")


if __name__ == "__main__":
    if sys.argv[1:] == ["probe"]:  # the child process, with XSOMA_ROBOT_DIR set
        print(json.dumps(probe()))
    else:
        assert not os.environ.get("XSOMA_ROBOT_DIR"), "run this test without XSOMA_ROBOT_DIR"
        test_payload()
        print("\nall checks passed")
