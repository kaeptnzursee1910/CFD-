"""Tests für cfd2d: Erhaltungseigenschaften, Validierung, Geometrie, Bedienung."""

import builtins
import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from cfd2d import cases, geometry, validation
from cfd2d.boundary import BoundaryCondition as BC
from cfd2d.grid import Grid
from cfd2d.solver import Simulation


# ---------------------------------------------------------------------- #
# Grundlegende Eigenschaften
# ---------------------------------------------------------------------- #
def test_divergence_free_cavity():
    case = cases.lid_driven_cavity(re=100, n=24)
    for _ in range(50):
        case.sim.step()
    assert case.sim.max_divergence() < 1e-10


def test_divergence_free_with_obstacles_and_outflow():
    case = cases.custom_obstacles([{"typ": "kreis", "x": 1.0, "y": 0.5, "r": 0.15},
                                   {"typ": "naca", "code": "2412", "x": 2.0, "y": 0.4,
                                    "sehne": 0.6, "winkel": 10}],
                                  lx=4, ly=1, nx=80, ny=20, re=100)
    for _ in range(100):
        case.sim.step()
    assert case.sim.max_divergence() < 1e-10
    # Massenerhaltung: Einstrom = Ausstrom
    g = case.sim.grid
    q_in = case.sim.u[0, 1:-1].sum() * g.dy
    q_out = case.sim.u[g.nx, 1:-1].sum() * g.dy
    assert q_out == pytest.approx(q_in, rel=1e-8)


def test_zero_velocity_in_solid_normal_faces():
    case = cases.cylinder_flow(re=20, nx=110, ny=21, t_end=0.1)
    case.sim.run(0.1, verbose=False)
    s = case.sim
    fl = s.grid.fluid
    one_sided = (fl[:-1, :] ^ fl[1:, :])[1:-1, 1:-1]  # nur innere Flächen
    assert one_sided.any()
    assert np.abs(s.u[1:-1, 1:-1][one_sided]).max() == 0.0


# ---------------------------------------------------------------------- #
# Validierung
# ---------------------------------------------------------------------- #
def test_cavity_re100_matches_ghia():
    case = cases.lid_driven_cavity(re=100, n=32)
    case.sim.run(40, steady_tol=1e-5, verbose=False)
    err = validation.cavity_centerline_errors(case.sim, 100)
    assert err["u_max_fehler"] < 0.03
    assert err["v_max_fehler"] < 0.03


def test_poiseuille_profile_and_pressure_gradient():
    case = cases.channel_flow(re=20, length=4, ny=16, profile="uniform")
    case.sim.run(case.t_end, steady_tol=1e-6, verbose=False)
    s = case.sim
    g = s.grid
    y = g.yc
    u = s.sample_u(np.full_like(y, 3.6), y)
    ua = validation.poiseuille_profile(y)
    assert np.abs(u - ua).max() < 0.02
    p = s.cell_pressure()[:, g.ny // 2]
    i1, i2 = int(0.6 * g.nx), int(0.9 * g.nx)
    dpdx = (p[i2] - p[i1]) / (g.xc[i2] - g.xc[i1])
    assert dpdx == pytest.approx(validation.poiseuille_pressure_gradient(s.nu), rel=0.03)


def test_cylinder_re20_drag_in_plausible_range():
    case = cases.cylinder_flow(re=20, nx=220, ny=41, t_end=3.0)
    case.sim.run(case.t_end, verbose=False)
    fx, fy = case.sim.obstacle_forces()
    cd, cl = validation.force_coefficients(fx, fy, 1.0, 0.1)
    # Referenz 5.57-5.59; Treppenstufen-Geometrie auf grobem Gitter: ±10 %
    assert 5.0 < cd < 6.1
    assert abs(cl) < 0.05


def test_step_has_recirculation():
    case = cases.backward_facing_step(re=200, length=8, ny=16, t_end=30)
    case.sim.run(case.t_end, steady_tol=1e-4, verbose=False)
    xr = validation.reattachment_length(case.sim, case.info["x_stufe"])
    assert xr is not None and 1.0 < xr < 5.0


def test_euler_scheme_and_cg_solver():
    case = cases.lid_driven_cavity(re=100, n=16, scheme="euler", pressure_solver="cg",
                                   gamma=0.5)
    for _ in range(30):
        case.sim.step()
    assert case.sim.max_divergence() < 1e-6


def test_strouhal_from_synthetic_signal():
    t = np.linspace(0, 20, 4000)
    cl = np.sin(2 * np.pi * 3.0 * t)
    st = validation.strouhal_number(t, cl, D=0.1, U=1.0)
    assert st == pytest.approx(0.3, rel=0.01)


# ---------------------------------------------------------------------- #
# Geometrie
# ---------------------------------------------------------------------- #
def test_circle_area():
    g = Grid(200, 200, 1.0, 1.0)
    m = geometry.circle(g, 0.5, 0.5, 0.25)
    assert m.sum() * g.dx * g.dy == pytest.approx(np.pi * 0.25**2, rel=0.01)


def test_naca_thickness():
    pts = geometry.naca4_coordinates("0012", chord=1.0)
    assert pts[:, 1].max() - pts[:, 1].min() == pytest.approx(0.12, rel=0.02)
    g = Grid(200, 100, 2.0, 1.0)
    m = geometry.naca4(g, "0012", 0.5, 0.5, 1.0, 0.0)
    # Fläche eines NACA 0012 ≈ 0.0822 c²
    assert m.sum() * g.dx * g.dy == pytest.approx(0.0822, rel=0.05)


def test_obstacle_from_image(tmp_path):
    from PIL import Image

    img = np.full((50, 100), 255, dtype=np.uint8)
    img[10:40, 20:40] = 0  # schwarzes Rechteck
    path = tmp_path / "hindernis.png"
    Image.fromarray(img).save(path)
    g = Grid(100, 50, 2.0, 1.0)
    m = geometry.from_image(g, path)
    assert m.sum() == pytest.approx(30 * 20, rel=0.05)
    # Bildzeile 10-40 von oben entspricht y in [0.2, 0.8]
    ys = g.yc[np.where(m.any(axis=0))[0]]
    assert ys.min() == pytest.approx(0.2, abs=0.03) and ys.max() == pytest.approx(0.8, abs=0.03)


def test_enclosed_fluid_removed():
    g = Grid(40, 40, 1.0, 1.0)
    ring = geometry.circle(g, 0.5, 0.5, 0.3) & ~geometry.circle(g, 0.5, 0.5, 0.2)
    g.add_obstacle(ring)
    removed = g.finalize()
    assert removed > 0
    assert g.solid[20, 20]


def test_invalid_boundary_condition():
    with pytest.raises(ValueError):
        BC("wand")


def test_instability_detected():
    g = Grid(16, 16, 1.0, 1.0)
    bcs = {"left": BC("noslip"), "right": BC("noslip"), "bottom": BC("noslip"),
           "top": BC("noslip", 1.0)}
    sim = Simulation(g, bcs, nu=1e-2, dt=1.0)  # viel zu großer Zeitschritt
    with pytest.raises(FloatingPointError):
        for _ in range(200):
            sim.step()


# ---------------------------------------------------------------------- #
# Bedienung und Ausgabe
# ---------------------------------------------------------------------- #
def _feed(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(it))


def test_interactive_cavity_run(monkeypatch, tmp_path):
    from cfd2d import interactive

    monkeypatch.setattr("cfd2d.visualization.display_available", lambda: False)
    out = tmp_path / "cavity"
    answers = [
        "1",          # Hauptmenü: Cavity
        "100",        # Re
        "16",         # Zellen
        "2",          # t_end
        "",           # Toleranz (Standard)
        "",           # Turbulenzmodell: laminar
        "n",          # Temperatur
        "n",          # Skalar
        "n",          # Numerik anpassen? nein
        "n",          # Function Objects
        str(out),     # Ausgabeordner
        "j",          # GIF speichern
        "2",          # Feld: Wirbelstärke
        "n",          # Stromlinien
        "20",         # Update alle 20 Schritte
        "n",          # Case-Datei extra speichern
        "",           # starten
        "n",          # keine weitere Simulation
    ]
    _feed(monkeypatch, answers)
    interactive.main(output_root=tmp_path)
    for f in ("felder.png", "konvergenz.png", "validierung_mittellinien.png", "ergebnis.npz",
              "zusammenfassung.json", "animation.gif", "case.json", "checkpoint.npz"):
        assert (out / f).exists(), f
    summary = json.loads((out / "zusammenfassung.json").read_text(encoding="utf-8"))
    assert summary["reynolds"] == 100


def test_interactive_custom_with_obstacles(monkeypatch, tmp_path):
    from cfd2d import interactive

    monkeypatch.setattr("cfd2d.visualization.display_available", lambda: False)
    out = tmp_path / "custom"
    answers = [
        "5",                      # eigene Hindernisse
        "3", "1", "20", "", "",   # lx, ly, ny, nx, U
        "n",                      # Randbedingungen einzeln? nein
        "1", "2",                 # Blockprofil, haftende Wände
        "1", "0,8", "0.5", "0.15",  # Kreis (mit Dezimalkomma)
        "5", "1.5 0.3; 2 0.5; 1.5 0.7",  # Polygon
        "7",                      # fertig
        "", "50", "0.5",          # L_ref automatisch, Re, t_end
        "n",                      # kein vorzeitiger Abbruch
        "", "n", "n",             # laminar, keine Temperatur, kein Skalar
        "n", "n",                 # Numerik, Function Objects
        str(out),
        "n",                      # GIF
        "n", "", "n",
    ]
    _feed(monkeypatch, answers)
    interactive.main(output_root=tmp_path)
    assert (out / "kraftbeiwerte.png").exists()
    summary = json.loads((out / "zusammenfassung.json").read_text(encoding="utf-8"))
    assert len(summary["hindernisse"]) == 2


def test_interactive_physics_functions_and_restart(monkeypatch, tmp_path):
    from cfd2d import interactive

    monkeypatch.setattr("cfd2d.visualization.display_available", lambda: False)
    out = tmp_path / "zyl"
    answers = [
        "2", "100", "4", "40", "", "", "0.2",   # Zylinder, eigenes Gitter 40 x (auto) -> nx
        "2", "", "",                            # k-omega SST, I, ratio
        "j", "", "", "j", "1",                  # Temperatur: Pr, T_in, beheizt, T=1
        "n", "n", "n", "n", "n",                # keine Wandtemperaturen (4 Seiten), kein Auftrieb
        "",                                     # Limiter (van Leer)
        "j", "", "j", "0.15", "0.25", "",       # Skalar mit Streifen
        "n",                                    # Numerik
        "j", "1", "0.5 0.2", "", "",            # Sonde
        "5", "0.1",                             # Checkpoint
        "6",                                    # fertig
        str(out), "n", "n", "", "j",            # Ausgabe, GIF nein, JSON nein, start, weitere
        "11", str(out), "0.3", "n", "n",        # Fortsetzen bis t = 0.3
    ]
    _feed(monkeypatch, answers)
    interactive.main(output_root=tmp_path)
    for f in ("case.json", "checkpoint.npz", "sonden.csv", "felder_modelle.png"):
        assert (out / f).exists(), f
    spec = json.loads((out / "case.json").read_text(encoding="utf-8"))
    assert spec["turbulenz"]["modell"] == "kOmegaSST"
    assert spec["temperatur"]["T_obstacle"] == 1.0
    summary = json.loads((out / "zusammenfassung.json").read_text(encoding="utf-8"))
    assert summary["endzeit"] == pytest.approx(0.3)


def test_input_validation_retries(monkeypatch):
    from cfd2d import interactive

    _feed(monkeypatch, ["abc", "-5", "7"])
    assert interactive.ask_float("Wert", 1.0, lo=0) == 7.0
    _feed(monkeypatch, ["vielleicht", "ja"])
    assert interactive.ask_bool("Frage", False) is True


def test_cli_non_interactive(tmp_path):
    from cfd2d.cli import main

    out = tmp_path / "cli"
    assert main(["--fall", "channel", "--re", "10", "--ny", "8", "--laenge", "3",
                 "--t-ende", "1", "--ausgabe", str(out)]) == 0
    assert (out / "profile.png").exists()
