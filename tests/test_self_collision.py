"""robot.json's `self_collision` (run: python tests/test_self_collision.py, or pytest) on the bundled G1, with its two legs as the
groups. On MuJoCo's CPU engine:

  bits      under MuJoCo's rule, over every pair of geoms: the pairs that collide are robot x terrain, as without the key, plus
            left leg x right leg; nothing else of the compiled scene differs
  contacts  legs crossed in the air: contacts between the legs only, none without the key; standing (legs apart and crossed,
            priority soles and Holosoma's explicit sole pairs): the terrain contacts and their parameters are those without the key
  loader    one group, an empty group, a body in both groups, a missing body, a body without a collision geom: refused
  forces    the simulator's `collider_forces` and `contact_termination` (the real Sim with the G1 as a robot directory, in a child
            process) sum the terrain contacts only: crossed legs press on each other without ending the episode, a pelvis on
            the ground ends it. MuJoCo Warp's physics does not run on the CPU, so the contact pool is hand-made: the contacts and
            constraint forces of MuJoCo's CPU engine in the pool's layout, with unused entries behind them
"""

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

LEGS = [[f"{side}_{b}" for b in ("hip_roll_link", "knee_link", "ankle_roll_link")] for side in ("left", "right")]  # thigh, shin, foot
ON_BACK = (np.cos(np.pi / 4), 0.0, -np.sin(np.pi / 4), 0.0)  # pitched back by 90 degrees


def pose(m, roll, z=None, quat=(1.0, 0.0, 0.0, 0.0)):
    """The default pose with both hip rolls turned inward by `roll` (0.45 rad crosses the legs: thigh on thigh, shin on shin),
    the pelvis at height z or, with z=None, the lowest sole point 4 mm into the plane. Returns the data after mj_forward."""
    from xsoma import scene, task

    d = mujoco.MjData(m)
    d.qpos[:3], d.qpos[3:7], d.qpos[7:] = [0.0, 0.0, 1.0 if z is None else z], quat, task.DEFAULT_Q
    d.qpos[7 + task.DOF.index("left_hip_roll")], d.qpos[7 + task.DOF.index("right_hip_roll")] = -roll, roll
    if z is None:
        mujoco.mj_kinematics(m, d)
        d.qpos[2] -= min(d.geom_xpos[g, 2] - m.geom_size[g, 0] - abs(d.geom_xmat[g, 8]) * m.geom_size[g, 1] for g in scene.sole_geoms(m)) + 0.004
    mujoco.mj_forward(m, d)
    return d


def contacts(m, d, terrain):
    """The contacts with the terrain (or, terrain=False, between robot geoms): names, geometry and parameters, sorted."""
    rows = [(m.geom(c.geom1).name, m.geom(c.geom2).name, c.dist, c.dim, *c.pos, *c.frame, *c.friction, *c.solref, *c.solimp)
            for c in d.contact[: d.ncon] if (m.geom_bodyid[[c.geom1, c.geom2]] == 0).any() == terrain]  # fmt: skip
    return sorted(rows)


def groups(m):
    """Per geom: "T" terrain, "A" / "B" the collision geoms of LEGS' bodies, "R" the rest of the robot."""
    body = [m.body(b).name.removeprefix("robot_") for b in m.geom_bodyid]
    return ["T" if b == "world" else "A" if b in LEGS[0] else "B" if b in LEGS[1] else "R" for b in body]


def test_bits():
    from xsoma import scene

    a, b = scene.build(), scene.build(self_collision=LEGS)
    collide = lambda m, i, j: bool((m.geom_contype[i] & m.geom_conaffinity[j]) | (m.geom_contype[j] & m.geom_conaffinity[i]))  # noqa: E731
    g = groups(b)
    assert g.count("A") == g.count("B") == 9 and g.count("T") == 1, g
    for i in range(b.ngeom):
        for j in range(i + 1, b.ngeom):
            terrain = (g[i] == "T") != (g[j] == "T")
            assert collide(a, i, j) == terrain, (b.geom(i).name, b.geom(j).name)
            assert collide(b, i, j) == (terrain or {g[i], g[j]} == {"A", "B"}), (b.geom(i).name, b.geom(j).name)
    bits = {k: sorted({(int(b.geom_contype[i]), int(b.geom_conaffinity[i])) for i in range(b.ngeom) if g[i] == k}) for k in "TABR"}
    assert bits == {"T": [(2, 1)], "A": [(5, 2)], "B": [(1, 6)], "R": [(1, 2)]}, bits
    for f in ("geom_contype", "geom_conaffinity", "body_contype", "body_conaffinity"):  # the bodies': the OR over their geoms
        getattr(b, f)[:] = getattr(a, f)
    saved = []
    for m in (a, b):
        saved.append(np.zeros(mujoco.mj_sizeModel(m), np.uint8))
        mujoco.mj_saveModel(m, None, saved[-1])
    assert np.array_equal(*saved), "the key changed more than the contact bits"
    print(f"bits (contype, conaffinity): {bits}; of {b.ngeom * (b.ngeom - 1) // 2} geom pairs, robot x terrain collide as without the key, "
          f"plus the {g.count('A') * g.count('B')} of A x B; the rest of the model is identical")  # fmt: skip


def test_contacts():
    from xsoma import scene

    for pairs in (False, True):
        a, b = scene.build(pairs=pairs), scene.build(pairs=pairs, self_collision=LEGS)
        assert a.npair == b.npair == (14 if pairs else 0)
        for f in ("pair_geom1", "pair_geom2", "pair_dim", "pair_friction", "pair_solref", "pair_solimp", "pair_margin", "pair_gap"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), f
        g = dict(zip((b.geom(i).name for i in range(b.ngeom)), groups(b)))
        air = contacts(b, pose(b, 0.45, z=1.0), terrain=False)
        assert pose(a, 0.45, z=1.0).ncon == 0 and len(air) == pose(b, 0.45, z=1.0).ncon > 0
        assert all({g[r[0]], g[r[1]]} == {"A", "B"} for r in air), "a contact within a group or outside the groups"
        n = {}
        for roll in (0.0, 0.45):
            da, db = pose(a, roll), pose(b, roll)
            ta, tb, legs = contacts(a, da, True), contacts(b, db, True), contacts(b, db, False)
            assert len(ta) == da.ncon > 0 and ta == tb, "the terrain contacts changed"
            assert len(legs) == (2 if roll else 0) and all({g[r[0]], g[r[1]]} == {"A", "B"} for r in legs)
            n[roll] = (len(tb), len(legs))
        print(f"{'explicit sole pairs' if pairs else 'priority soles'}: crossed in the air {len(air)} contacts, all leg x leg; standing "
              f"{n[0.0][0]} terrain contacts, crossed {n[0.45][0]} + {n[0.45][1]} leg x leg: the terrain contacts equal those without the key")  # fmt: skip


def test_loader():
    from xsoma import scene

    left, right = LEGS
    bad = {"one group": ([left], "two non-empty groups"), "an empty group": ([left, []], "two non-empty groups"),
           "a body in both groups": ([left, right + left[:1]], "['left_hip_roll_link'] are in both groups"),
           "a missing body": ([left, ["no_such_body"]], "'no_such_body' is not a body"),
           "a body without a collision geom": ([left, ["right_hip_pitch_link"]], "'right_hip_pitch_link' has no collision geom")}  # fmt: skip
    for case, (value, msg) in bad.items():
        try:
            scene.build(self_collision=value)
        except AssertionError as e:
            assert msg in str(e), f"{case}: {e}"
        else:
            raise AssertionError(f"{case}: not rejected")
    print(f"rejected: {', '.join(bad)}")


def probe():
    """The child process: the simulator's collider forces on MuJoCo's CPU contacts of two poses, next to the sums of
    mj_contactForce over each termination collider's terrain contacts and over all its contacts."""
    import types
    from unittest import mock

    import jax.numpy as jnp

    from xsoma import task
    from xsoma.sim import Sim, SimCfg

    sim = Sim(SimCfg(num_envs=1, terrain="plane"))
    m, out = sim.m, {}
    assert task.SELF_COLLISION == LEGS
    for name, d in (("crossed", pose(m, 0.45)), ("fallen", pose(m, 0.45, z=0.065, quat=ON_BACK))):
        n, pad, f6 = d.ncon, 5, np.zeros(6)
        ref = {k: np.zeros((len(sim.collider_geoms), 3)) for k in ("terrain", "all")}
        for i, c in enumerate(d.contact[:n]):
            mujoco.mj_contactForce(m, d, i, f6)
            for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if sim.term_slot[g] >= 0:
                    ref["all"][sim.term_slot[g]] += c.frame.reshape(3, 3).T @ f6[:3]
                    ref["terrain"][sim.term_slot[g]] += (m.geom_bodyid[other] == 0) * (c.frame.reshape(3, 3).T @ f6[:3])
        fill = lambda x, v: jnp.asarray(np.concatenate((x, np.full((pad, *np.shape(x)[1:]), v, np.asarray(x).dtype))))  # noqa: E731
        con = d.contact[:n]
        rows = np.where(con.efc_address[:, None] >= 0, con.efc_address[:, None] + np.arange(4), -1)  # 4 pyramid rows per contact
        impl = types.SimpleNamespace(nacon=jnp.asarray([n]), contact__dist=fill(con.dist, 0.0), contact__worldid=fill(np.zeros(n, np.int32), 0),
                                     contact__efc_address=fill(rows, 0), efc__force=jnp.asarray(d.efc_force[None], jnp.float32),
                                     contact__friction=fill(con.friction, 1.0), contact__frame=fill(con.frame.reshape(n, 3, 3), 1.0),
                                     contact__geom=fill(con.geom, 1))  # fmt: skip
        data = types.SimpleNamespace(_impl=impl)
        with mock.patch.object(task, "SELF_COLLISION", None):  # the decoding of a robot without the key
            unmasked = sim.collider_forces(data)
        out[name] = {"forces": np.asarray(sim.collider_forces(data))[0].tolist(), "unmasked": np.asarray(unmasked)[0].tolist(),
                     "terminated": bool(sim.contact_termination(data)[0]), "terrain": ref["terrain"].tolist(), "all": ref["all"].tolist(),
                     "bodies": [m.body(m.geom_bodyid[g]).name.removeprefix("robot_") for g in sim.collider_geoms]}  # fmt: skip
    return out


def test_forces():
    from test_robot_dir import fingerprint

    from xsoma import scene, task

    with tempfile.TemporaryDirectory() as d:
        with open(scene.G1_XML) as src, open(os.path.join(d, "robot.xml"), "w") as dst:
            dst.write(src.read())
        with open(os.path.join(d, "robot.json"), "w") as f:
            json.dump(fingerprint()["values"] | {"self_collision": LEGS}, f)
        env = {k: v for k, v in os.environ.items() if k != "XSOMA_G1_XML"} | {"XSOMA_ROBOT_DIR": d, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "probe"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.splitlines()[-1])
    for name, o in out.items():
        forces, unmasked, terrain, every = (np.asarray(o[k]) for k in ("forces", "unmasked", "terrain", "all"))
        norm = lambda x: dict(zip(o["bodies"], np.linalg.norm(x, axis=-1).round(1).tolist()))  # noqa: E731
        thighs = [o["bodies"].index(f"{side}_hip_roll_link") for side in ("left", "right")]
        np.testing.assert_allclose(forces, terrain, rtol=1e-4, atol=1e-2, err_msg=f"{name}: not the terrain contacts' forces")
        assert (np.linalg.norm(every[thighs], axis=-1) > 100 * task.CONTACT_FORCE).all(), f"{name}: the thighs do not press on each other"
        assert (np.linalg.norm(unmasked[thighs], axis=-1) > task.CONTACT_FORCE).any(), f"{name}: without the mask the leg contact must count"
        assert o["terminated"] == (name == "fallen") == bool((np.linalg.norm(terrain, axis=-1) > task.CONTACT_FORCE).any()), name
        hit = {b: v for b, v in norm(forces).items() if v > 0}
        print(f"{name}: thigh on thigh {norm(every)['left_hip_roll_link']} N (MuJoCo), counted {norm(forces)['left_hip_roll_link']} N, without the "
              f"mask {max(norm(unmasked)[f'{s}_hip_roll_link'] for s in ('left', 'right'))} N; terrain forces {hit or 'none'}; terminated {o['terminated']}")  # fmt: skip


if __name__ == "__main__":
    if sys.argv[1:] == ["probe"]:  # the child process, with XSOMA_ROBOT_DIR set
        print(json.dumps(probe()))
    else:
        assert not os.environ.get("XSOMA_ROBOT_DIR"), "run this test without XSOMA_ROBOT_DIR"
        test_bits()
        test_contacts()
        test_loader()
        test_forces()
        print("\nall checks passed")
