"""Tests für Function Objects, VTK-Export, Speicherstände und Case-Dateien."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from cfd2d import cases, functions


def _cyl_spec(**extra):
    spec = {"fall": "cylinder", "parameter": {"re": 100, "nx": 88, "ny": 17, "t_end": 0.2}}
    spec.update(extra)
    return spec


def test_probes_and_line_sample(tmp_path):
    case = cases.lid_driven_cavity(re=100, n=16)
    pr = functions.Probes([(0.5, 0.5), (0.5, 0.9)], fields=("u", "pressure", "vorticity"))
    ls = functions.LineSample((0.5, 0.0), (0.5, 1.0), n=21, fields=("u", "v"), label="mitte")
    case.sim.function_objects += [pr, ls]
    case.sim.run(0.5, verbose=False)
    files = pr.write(tmp_path) + ls.write(tmp_path)
    assert all(f.exists() for f in files)
    assert len(pr.t) == case.sim.n
    # Linienprobe trifft die Wandwerte: u = 0 unten, u = 1 oben (Deckel)
    assert ls.result["u"][0] == pytest.approx(0.0, abs=1e-12)
    assert ls.result["u"][-1] == pytest.approx(1.0, abs=1e-12)


def test_vtk_file_structure(tmp_path):
    case = cases.build_case(_cyl_spec(temperatur={"alpha": 1e-3, "T_obstacle": 1.0}))
    case.sim.run(0.05, verbose=False)
    path = functions.write_vtk(case.sim, tmp_path / "f.vtk")
    raw = path.read_bytes()
    assert raw.startswith(b"# vtk DataFile Version 3.0")
    assert b"DIMENSIONS 89 18 1" in raw
    assert b"VECTORS U float" in raw and b"SCALARS T float 1" in raw
    # Größe des Vektorfeldes: nx*ny*3 float32
    i = raw.index(b"VECTORS U float\n") + len(b"VECTORS U float\n")
    vec = np.frombuffer(raw[i:i + 88 * 17 * 12], dtype=">f4").reshape(-1, 3)
    uc, _ = case.sim.cell_velocity()
    assert np.allclose(vec[:, 0], np.nan_to_num(uc).T.ravel(), atol=1e-5)


@pytest.mark.parametrize("extra", [
    {},
    {"turbulenz": {"modell": "kOmegaSST"}},
    {"skalar": {"name": "c", "diffusivity": 1e-3, "inflow_band": [0.15, 0.25]}},
])
def test_checkpoint_restart_is_exact(tmp_path, extra):
    spec = _cyl_spec(**extra)
    ref = cases.build_case(spec)
    for _ in range(80):
        ref.sim.step()

    a = cases.build_case(spec)
    for _ in range(40):
        a.sim.step()
    functions.save_checkpoint(a.sim, tmp_path / "cp.npz")
    b = cases.build_case(spec)
    functions.load_checkpoint(b.sim, tmp_path / "cp.npz")
    for _ in range(40):
        b.sim.step()
    assert b.sim.n == ref.sim.n
    assert np.allclose(b.sim.u, ref.sim.u, atol=1e-12, rtol=0)
    for name in ref.sim.scalars:
        assert np.allclose(b.sim.scalars[name].phi, ref.sim.scalars[name].phi, atol=1e-12)


def test_checkpoint_restart_vof(tmp_path):
    spec = {"fall": "dam_break", "parameter": {"n": 24, "t_end": 0.1}}
    ref = cases.build_case(spec)
    for _ in range(30):
        ref.sim.step()
    a = cases.build_case(spec)
    for _ in range(15):
        a.sim.step()
    functions.save_checkpoint(a.sim, tmp_path / "cp.npz")
    b = cases.build_case(spec)
    functions.load_checkpoint(b.sim, tmp_path / "cp.npz")
    for _ in range(15):
        b.sim.step()
    va = b.sim.model("multiphase").alpha
    vr = ref.sim.model("multiphase").alpha
    assert np.allclose(va, vr, atol=1e-10)


def test_checkpoint_wrong_geometry_rejected(tmp_path):
    a = cases.lid_driven_cavity(re=100, n=16)
    functions.save_checkpoint(a.sim, tmp_path / "cp.npz")
    b = cases.lid_driven_cavity(re=100, n=20)
    with pytest.raises(ValueError):
        functions.load_checkpoint(b.sim, tmp_path / "cp.npz")


def test_case_json_roundtrip_and_report(tmp_path):
    from cfd2d import report

    spec = _cyl_spec(function_objects=[{"typ": "sonden", "punkte": [[0.5, 0.2]]},
                                       {"typ": "mittelwerte", "t_start": 0.0},
                                       {"typ": "vtk", "intervall": 0.1},
                                       {"typ": "checkpoint", "intervall": 0.1}])
    cases.save_case(spec, tmp_path / "case.json")
    loaded = cases.load_case(tmp_path / "case.json")
    assert loaded == spec
    case = cases.build_case(loaded)
    out = tmp_path / "out"
    case.sim.function_objects = functions.build_function_objects(loaded["function_objects"], out)
    reason = case.sim.run(case.t_end, verbose=False)
    report.create_report(case, out, reason)
    for f in ("case.json", "sonden.csv", "mittelwerte.npz", "checkpoint.npz",
              "vtk/feld.vtk.series", "vtk/feld_00000.vtk"):
        assert (out / f).exists(), f
