"""Tests der Zusatzmodelle: Skalartransport, Temperatur/Auftrieb, periodische Ränder."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from cfd2d import cases, geometry
from cfd2d.boundary import BoundaryCondition as BC
from cfd2d.grid import Grid
from cfd2d.models import PassiveScalar, ScalarBC
from cfd2d.solver import Simulation


def test_periodic_channel_flow_rate_control():
    g = Grid(8, 20, 1.0, 1.0)
    bcs = dict(left=BC("periodic"), right=BC("periodic"), bottom=BC("noslip"), top=BC("noslip"))
    s = Simulation(g, bcs, nu=0.05, flow_rate=1.0)
    s.run(20, steady_tol=1e-7, verbose=False)
    uc, _ = s.cell_velocity()
    assert np.abs(uc[3] - 6 * g.yc * (1 - g.yc)).max() < 0.01
    assert s.flow_force == pytest.approx(12 * 0.05, rel=0.01)  # dp/dx = 12 nu U / H^2
    assert s.bulk_velocity() == pytest.approx(1.0, abs=1e-10)


def test_periodic_with_obstacle_divergence_free():
    g = Grid(32, 32, 1.0, 1.0)
    g.add_obstacle(geometry.circle(g, 0.5, 0.5, 0.15))
    bcs = dict(left=BC("periodic"), right=BC("periodic"), bottom=BC("slip"), top=BC("slip"))
    s = Simulation(g, bcs, nu=0.02, flow_rate=1.0)
    for _ in range(100):
        s.step()
    assert s.max_divergence() < 1e-10
    assert s.flow_force > 0


@pytest.mark.parametrize("limiter", ["upwind", "vanleer", "minmod", "superbee", "mc"])
def test_tvd_scalar_bounded_and_conservative(limiter):
    # Periodischer Kanal mit Pfropfenströmung: Rechteckpuls wird transportiert
    g = Grid(64, 8, 1.0, 0.125)
    bcs = dict(left=BC("periodic"), right=BC("periodic"), bottom=BC("slip"), top=BC("slip"))
    X, _ = g.cell_centers()
    c0 = ((X > 0.2) & (X < 0.4)).astype(float)
    sc = PassiveScalar("c", diffusivity=0.0, limiter=limiter, initial=c0)
    s = Simulation(g, bcs, nu=1e-3, models=[sc], initial_velocity=(1.0, 0.0))
    total0 = sc.field.total()
    s.run(0.5, verbose=False)
    c = sc.field.cell()
    assert c.min() >= -1e-12 and c.max() <= 1 + 1e-12
    assert sc.field.total() == pytest.approx(total0, rel=1e-10)
    # Schwerpunkt um 0.5 verschoben (0.3 -> 0.8)
    xc = (c * X).sum() / c.sum()
    assert xc == pytest.approx(0.8, abs=0.02)


def test_scalar_diffusion_steady_linear_profile():
    # Ruhendes Fluid, links c = 1, rechts c = 0: lineares Profil
    g = Grid(20, 4, 1.0, 0.2)
    bcs = {s: BC("slip") for s in ("left", "right", "bottom", "top")}
    sc = PassiveScalar("c", diffusivity=0.1, bcs={"left": ScalarBC("fixed", 1.0),
                                                   "right": ScalarBC("fixed", 0.0)})
    s = Simulation(g, bcs, nu=0.1, models=[sc])
    s.run(8.0, verbose=False)
    assert np.abs(sc.field.cell()[:, 2] - (1 - g.xc)).max() < 1e-3


def test_heated_cavity_de_vahl_davis():
    case = cases.heated_cavity(ra=1e4, n=32)
    case.sim.run(case.t_end, steady_tol=case.steady_tol, verbose=False)
    th = case.sim.model("thermal")
    assert th.nusselt("left") == pytest.approx(2.243, rel=0.03)
    assert th.nusselt("left") == pytest.approx(th.nusselt("right"), rel=1e-3)


@pytest.mark.parametrize("ra,convects", [(1200.0, False), (5000.0, True)])
def test_rayleigh_benard_onset(ra, convects):
    case = cases.rayleigh_benard(ra=ra, ny=12, t_end=250)
    case.sim.run(case.t_end, steady_tol=1e-6, verbose=False)
    nu = case.sim.model("thermal").nusselt("bottom")
    assert (nu > 1.2) == convects
    if not convects:
        assert nu == pytest.approx(1.0, abs=0.01)


def test_heated_obstacle_forced_convection():
    case = cases.custom_obstacles([{"typ": "kreis", "x": 1.0, "y": 0.5, "r": 0.15}],
                                  lx=3, ly=1, nx=60, ny=20, re=40, t_end=2.0)
    sim = case.sim
    from cfd2d.models import BoussinesqTemperature

    th = BoussinesqTemperature(alpha=0.01, buoyancy=False, obstacle=ScalarBC("fixed", 1.0))
    sim.add_model(th)
    sim.run(case.t_end, verbose=False)
    T = th.field.cell()
    assert np.nanmax(T) <= 1.0 + 1e-9 and np.nanmin(T) >= -1e-9
    # Nachlauf ist wärmer als die Anströmung
    assert np.nanmean(T[45:, :]) > np.nanmean(T[:10, :])


def test_turbulent_channel_sst_against_dean():
    case = cases.turbulent_channel(re_tau=120, y_plus_1=2.0, t_end=250)
    case.sim.run(case.t_end, steady_tol=1e-5, verbose=False)
    from cfd2d import report

    res = report.report_turbulent_channel(case, __import__("pathlib").Path(
        __import__("tempfile").mkdtemp()))
    assert abs(res["cf_abweichung_rel"]) < 0.15
    assert res["u_plus_mitte"] > 14  # deutlich turbulentes, vollere Profil als laminar
    turb = case.sim.model("turbulence")
    assert turb.k.phi[1:-1, 1:-1].min() >= 0
    assert case.sim.nu_t[2, 0] == 0.0  # nu_t = 0 an der Wand


def test_smagorinsky_runs_and_vanishes_at_walls():
    case = cases.add_turbulence(cases.lid_driven_cavity(re=1000, n=32), "Smagorinsky")
    for _ in range(100):
        case.sim.step()
    nut = case.sim.nu_t
    assert nut[1:-1, 1:-1].max() > 0
    assert case.sim.max_divergence() < 1e-10


def test_add_scalar_with_band_and_sst_on_step():
    case = cases.backward_facing_step(re=2000, length=6, ny=16, t_end=1.0)
    cases.add_turbulence(case, "kOmegaSST")
    cases.add_scalar(case, "c", 1e-3, inflow_band=(0.5, 0.75))
    case.sim.run(case.t_end, verbose=False)
    c = case.sim.scalars["c"].cell()
    assert np.nanmax(c) <= 1 + 1e-9 and np.nanmin(c) >= -1e-9
    assert np.nanmax(c) > 0.5


def test_vof_hydrostatic_rest():
    from cfd2d.vof import VOF

    g = Grid(20, 20, 0.2, 0.2)
    X, Y = g.cell_centers()
    alpha0 = (Y < 0.1).astype(float)
    bcs = {s: BC("noslip") for s in ("left", "right", "bottom")}
    bcs["top"] = BC("outflow")
    vof = VOF(alpha0, sigma=0.0)
    sim = Simulation(g, bcs, nu=1e-6, models=[vof])
    sim.run(0.2, verbose=False)
    uc, vc = sim.cell_velocity()
    assert np.nanmax(np.hypot(uc, vc)) < 1e-6  # keine Scheinströmungen
    p_bottom = sim.cell_pressure()[10, 0]
    assert p_bottom == pytest.approx(1000 * 9.81 * (0.1 - 0.005), rel=0.02)
    assert abs(vof.volume() / vof.volume0 - 1) < 1e-10


def test_dam_break_runs_conserves_volume():
    case = cases.dam_break(n=32, obstacle=True, t_end=0.15)
    case.sim.run(case.t_end, verbose=False)
    vof = case.sim.model("multiphase")
    assert abs(vof.volume() / vof.volume0 - 1) < 1e-3
    a = vof.alpha[1:-1, 1:-1]
    assert a.min() >= 0 and a.max() <= 1
    # die Front hat sich nach rechts bewegt
    assert case.sim.history.extra["front_x"][-1] > case.info["a"] * 1.3
    assert case.sim.max_divergence() < 1e-8
