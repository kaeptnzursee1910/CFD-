"""Vordefinierte Testfälle.

Alle Fälle sind dimensionslos mit Dichte 1. Die Reynolds-Zahl
``Re = U_ref * L_ref / nu`` legt die Viskosität fest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry
from .boundary import BoundaryCondition as BC
from .models import BoussinesqTemperature, PassiveScalar, ScalarBC
from .grid import Grid
from .solver import Simulation


@dataclass
class Case:
    """Ein vollständig aufgesetzter Testfall."""

    key: str
    title: str
    sim: Simulation
    t_end: float
    steady_tol: float | None
    U_ref: float
    L_ref: float
    re: float
    info: dict = field(default_factory=dict)

    def describe(self) -> str:
        s = [f"=== {self.title} ===", f"Reynolds-Zahl Re = {self.re:g} "
             f"(U_ref = {self.U_ref:g}, L_ref = {self.L_ref:g})",
             f"Endzeit t_end = {self.t_end:g}"
             + (f", stationäres Abbruchkriterium: Residuum < {self.steady_tol:g}"
                if self.steady_tol else ""),
             self.sim.summary()]
        return "\n".join(s)


def _numerics(kw: dict) -> dict:
    """Filtert die numerischen Einstellungen aus den Keyword-Argumenten."""
    keys = ("scheme", "gamma", "tau", "dt", "pressure_solver")
    return {k: kw[k] for k in keys if k in kw and kw[k] is not None}


# ---------------------------------------------------------------------- #
# 1) Lid-Driven Cavity
# ---------------------------------------------------------------------- #
def lid_driven_cavity(re: float = 100.0, n: int = 64, t_end: float = 50.0,
                      steady_tol: float | None = 1e-5, **num) -> Case:
    """Quadratische Box mit bewegtem Deckel (U = 1, L = 1).

    Validierung gegen Ghia, Ghia & Shin (1982) für Re = 100, 400, 1000.
    """
    grid = Grid(n, n, 1.0, 1.0)
    bcs = {"left": BC("noslip"), "right": BC("noslip"), "bottom": BC("noslip"),
           "top": BC("noslip", velocity=1.0)}
    sim = Simulation(grid, bcs, nu=1.0 / re, **_numerics(num))
    return Case("cavity", "Lid-Driven Cavity", sim, t_end, steady_tol, 1.0, 1.0, re)


# ---------------------------------------------------------------------- #
# 2) Zylinderumströmung (Benchmark nach Schäfer & Turek 1996)
# ---------------------------------------------------------------------- #
CYLINDER_REFERENCE = {
    # Schäfer, Turek et al. (1996), "Benchmark Computations of Laminar Flow Around a Cylinder"
    20: {"cd": (5.57, 5.59), "cl": (0.0104, 0.0110), "typ": "stationär (Test 2D-1)"},
    100: {"cd_max": (3.22, 3.24), "cl_max": (0.99, 1.01), "st": (0.295, 0.305),
          "typ": "periodisch (Test 2D-2)"},
}


def cylinder_flow(re: float = 100.0, nx: int = 440, ny: int = 82, t_end: float = 8.0,
                  profile: str = "parabolic", length: float = 2.2, **num) -> Case:
    """Kanal 2.2 x 0.41 mit Zylinder (D = 0.1) bei (0.2, 0.2).

    Die mittlere Einströmgeschwindigkeit ist U = 1, also ``Re = U * D / nu``.
    Mit parabolischem Profil und Re = 20 bzw. 100 entspricht das den
    Benchmarks 2D-1 und 2D-2 von Schäfer & Turek.
    """
    H, D, xc, yc = 0.41, 0.1, 0.2, 0.2
    grid = Grid(nx, ny, length, H)
    grid.add_obstacle(geometry.circle(grid, xc, yc, D / 2))
    bcs = {"left": BC("inflow", velocity=1.0, profile=profile), "right": BC("outflow"),
           "bottom": BC("noslip"), "top": BC("noslip")}
    sim = Simulation(grid, bcs, nu=1.0 * D / re, compute_forces=True,
                     initial_velocity=(0.0, 0.0), **_numerics(num))
    info = {"D": D, "xc": xc, "yc": yc, "H": H}
    ref = CYLINDER_REFERENCE.get(int(round(re))) if profile == "parabolic" and length == 2.2 else None
    if ref:
        info["referenz"] = ref
    return Case("cylinder", "Zylinderumströmung (Schäfer-Turek-Kanal)", sim, t_end, None,
                1.0, D, re, info)


# ---------------------------------------------------------------------- #
# 3) Kanalströmung (Poiseuille)
# ---------------------------------------------------------------------- #
def channel_flow(re: float = 20.0, length: float = 6.0, ny: int = 32, nx: int | None = None,
                 profile: str = "uniform", t_end: float = 40.0,
                 steady_tol: float | None = 1e-5, **num) -> Case:
    """Ebener Kanal der Höhe H = 1 mit mittlerer Geschwindigkeit U = 1.

    Mit Blockprofil am Einlass entwickelt sich die Strömung über die
    Einlauflänge L_e ≈ 0.05 Re H zum parabolischen Poiseuille-Profil
    ``u(y) = 6 U y (H - y) / H^2``. Das Druckgefälle dort ist
    ``dp/dx = -12 nu U / H^2``.
    """
    if nx is None:
        nx = int(round(length * ny))
    grid = Grid(nx, ny, length, 1.0)
    bcs = {"left": BC("inflow", velocity=1.0, profile=profile), "right": BC("outflow"),
           "bottom": BC("noslip"), "top": BC("noslip")}
    sim = Simulation(grid, bcs, nu=1.0 / re, initial_velocity=(1.0, 0.0), **_numerics(num))
    info = {"H": 1.0, "einlauflaenge": 0.05 * re}
    return Case("channel", "Kanalströmung (Poiseuille)", sim, t_end, steady_tol, 1.0, 1.0, re, info)


# ---------------------------------------------------------------------- #
# 4) Rückwärtsgewandte Stufe
# ---------------------------------------------------------------------- #
def backward_facing_step(re: float = 400.0, length: float = 15.0, ny: int = 40,
                         nx: int | None = None, inlet_length: float = 1.0,
                         t_end: float = 100.0, steady_tol: float | None = 1e-5,
                         **num) -> Case:
    """Kanal der Höhe H = 1 mit Stufenhöhe h = H/2 (Erweiterungsverhältnis 2).

    Einströmung parabolisch über der Stufe mit mittlerer Geschwindigkeit
    U = 1. Die Reynolds-Zahl ``Re = U * H / nu`` ist mit dem hydraulischen
    Durchmesser des Einlasses (2 * H/2 = H) gebildet, wie bei Gartling (1990).
    Ausgewertet wird die Wiederanlegelänge der Ablöseblase hinter der Stufe.
    """
    H, h = 1.0, 0.5
    if nx is None:
        nx = int(round(length / H * ny))
    grid = Grid(nx, ny, length, H)
    if inlet_length > 0:
        grid.add_obstacle(geometry.rectangle(grid, 0.0, 0.0, inlet_length, h))
    bcs = {"left": BC("inflow", velocity=1.0, profile="parabolic"), "right": BC("outflow"),
           "bottom": BC("noslip"), "top": BC("noslip")}
    sim = Simulation(grid, bcs, nu=1.0 * H / re, **_numerics(num))
    info = {"H": H, "h": h, "x_stufe": inlet_length}
    return Case("step", "Rückwärtsgewandte Stufe (Backward-Facing Step)", sim, t_end, steady_tol,
                1.0, H, re, info)


# ---------------------------------------------------------------------- #
# 5) Eigene Hindernisse
# ---------------------------------------------------------------------- #
def custom_obstacles(obstacles: list[dict], lx: float = 4.0, ly: float = 1.0, nx: int = 320,
                     ny: int = 80, re: float = 100.0, U: float = 1.0, L_ref: float | None = None,
                     inflow_profile: str = "uniform", walls: str = "slip", t_end: float = 10.0,
                     bcs: dict | None = None, steady_tol: float | None = None,
                     **num) -> Case:
    """Freie Geometrie: Kanal mit beliebigen Hindernissen.

    Parameters
    ----------
    obstacles : list of dict
        Beschreibungen für :func:`cfd2d.geometry.build_obstacle`.
    walls : {"slip", "noslip"}
        Art der Wände oben und unten (wird ignoriert, wenn ``bcs`` gesetzt ist).
    bcs : dict, optional
        Eigene Randbedingungen für alle vier Seiten.
    L_ref : float, optional
        Charakteristische Länge für Re und die Kraftbeiwerte. Standard ist die
        größte Ausdehnung der Hindernisse quer zur Strömung (oder ly, wenn es
        keine Hindernisse gibt).
    """
    grid = Grid(nx, ny, lx, ly)
    for spec in obstacles:
        grid.add_obstacle(geometry.build_obstacle(grid, spec))
    if L_ref is None:
        solid = grid.solid
        if solid.any():
            rows = solid.any(axis=0)
            L_ref = rows.sum() * grid.dy
        else:
            L_ref = ly
    if bcs is None:
        bcs = {"left": BC("inflow", velocity=U, profile=inflow_profile), "right": BC("outflow"),
               "bottom": BC(walls), "top": BC(walls)}
    sim = Simulation(grid, bcs, nu=U * L_ref / re, compute_forces=bool(obstacles),
                     **_numerics(num))
    info = {"hindernisse": obstacles, "L_ref": L_ref}
    return Case("custom", "Eigene Hindernisse", sim, t_end, steady_tol, U, L_ref, re, info)


# ---------------------------------------------------------------------- #
# 6) Natürliche Konvektion: seitlich beheizte Cavity (de Vahl Davis)
# ---------------------------------------------------------------------- #
def heated_cavity(ra: float = 1e4, pr: float = 0.71, n: int = 48, t_end: float = 500.0,
                  steady_tol: float | None = 1e-6, limiter: str = "vanleer", **num) -> Case:
    """Quadratische Box: links warm (T = +0.5), rechts kalt (T = -0.5), oben und unten adiabat.

    Freifall-Skalierung: U = sqrt(g beta dT L), nu = sqrt(Pr/Ra), alpha = 1/sqrt(Ra Pr).
    Referenz: de Vahl Davis (1983) für Ra = 1e3 bis 1e6.
    """
    grid = Grid(n, n, 1.0, 1.0)
    bcs = {s: BC("noslip") for s in ("left", "right", "bottom", "top")}
    thermal = BoussinesqTemperature(
        alpha=1.0 / np.sqrt(ra * pr), limiter=limiter,
        bcs={"left": ScalarBC("fixed", 0.5), "right": ScalarBC("fixed", -0.5),
             "bottom": ScalarBC("zerogradient"), "top": ScalarBC("zerogradient")})
    sim = Simulation(grid, bcs, nu=np.sqrt(pr / ra), models=[thermal], **_numerics(num))
    info = {"ra": ra, "pr": pr, "geschw_skala_alpha": np.sqrt(ra * pr)}
    return Case("heated_cavity", "Beheizte Cavity (natürliche Konvektion)", sim, t_end,
                steady_tol, 1.0, 1.0, float(np.sqrt(ra / pr)), info)


# ---------------------------------------------------------------------- #
# 7) Rayleigh-Bénard-Konvektion
# ---------------------------------------------------------------------- #
def rayleigh_benard(ra: float = 1e4, pr: float = 0.71, aspect: float = 4.032, ny: int = 24,
                    nx: int | None = None, t_end: float = 300.0, steady_tol: float | None = 1e-6,
                    seed: int = 1, limiter: str = "vanleer", **num) -> Case:
    """Fluidschicht der Höhe 1, unten warm (+0.5), oben kalt (-0.5), periodisch in x.

    Unterhalb von Ra_krit ≈ 1708 bleibt die Schicht in Ruhe (Nu = 1); darüber
    bilden sich Konvektionsrollen. Das Standard-Seitenverhältnis 4.032 entspricht
    zwei kritischen Wellenlängen.
    """
    if nx is None:
        nx = int(round(aspect * ny))
    grid = Grid(nx, ny, aspect, 1.0)
    bcs = {"left": BC("periodic"), "right": BC("periodic"), "bottom": BC("noslip"),
           "top": BC("noslip")}
    rng = np.random.default_rng(seed)
    X, Y = grid.cell_centers()
    T0 = 0.5 - Y + 0.01 * rng.standard_normal(X.shape) * np.sin(np.pi * Y)
    thermal = BoussinesqTemperature(
        alpha=1.0 / np.sqrt(ra * pr), initial=T0, limiter=limiter,
        bcs={"bottom": ScalarBC("fixed", 0.5), "top": ScalarBC("fixed", -0.5)})
    sim = Simulation(grid, bcs, nu=np.sqrt(pr / ra), models=[thermal], **_numerics(num))
    info = {"ra": ra, "pr": pr, "ra_krit": 1707.76}
    return Case("rayleigh_benard", "Rayleigh-Bénard-Konvektion", sim, t_end, steady_tol, 1.0, 1.0,
                float(np.sqrt(ra / pr)), info)


# ---------------------------------------------------------------------- #
# 8) Turbulente Kanalströmung (RANS k-omega SST)
# ---------------------------------------------------------------------- #
def dean_cf(re_b: float) -> float:
    """Reibungsbeiwert nach Dean (1978): Cf = 0.073 Re_b^(-1/4), Re_b = U_b * 2h / nu."""
    return 0.073 * re_b ** -0.25


def turbulent_channel(re_tau: float = 180.0, ny: int | None = None, y_plus_1: float = 1.0,
                      model: str = "kOmegaSST", t_end: float = 400.0,
                      steady_tol: float | None = 1e-6, **num) -> Case:
    """Periodischer Kanal (Halbhöhe h = 1, Wände bei y = 0 und y = 2), U_b = 1.

    Re_b wird aus Re_tau über die Dean-Korrelation Re_tau = 0.09 Re_b^0.88 bestimmt,
    ny so gewählt, dass die erste Zelle bei y+ ≈ y_plus_1 liegt.
    """
    from .turbulence import TURBULENCE_MODELS

    re_b = (re_tau / 0.09) ** (1 / 0.88)
    nu = 2.0 / re_b
    if ny is None:
        ny = int(np.ceil(re_tau / y_plus_1))
        ny += ny % 2
    grid = Grid(4, ny, 0.5, 2.0)
    bcs = {"left": BC("periodic"), "right": BC("periodic"), "bottom": BC("noslip"),
           "top": BC("noslip")}
    cls = TURBULENCE_MODELS[model]
    models = [cls(intensity=0.05, viscosity_ratio=50.0)] if model == "kOmegaSST" else (
        [cls()] if cls else [])
    sim = Simulation(grid, bcs, nu=nu, flow_rate=1.0, initial_velocity=(1.0, 0.0),
                     models=models, **_numerics(num))
    info = {"re_tau_soll": re_tau, "re_b": re_b, "cf_dean": dean_cf(re_b), "modell": model}
    return Case("turbulent_channel", "Turbulente Kanalströmung (RANS)", sim, t_end, steady_tol,
                1.0, 2.0, re_b, info)


# ---------------------------------------------------------------------- #
# Zusatzphysik für beliebige Fälle
# ---------------------------------------------------------------------- #
def add_turbulence(case: Case, model: str = "kOmegaSST", **kw) -> Case:
    """Ergänzt einen Fall um ein Turbulenzmodell (``kOmegaSST`` oder ``Smagorinsky``)."""
    from .turbulence import TURBULENCE_MODELS

    cls = TURBULENCE_MODELS[model]
    if cls is None:
        return case
    if model == "kOmegaSST":
        kw.setdefault("U_ref", case.U_ref)
    case.sim.add_model(cls(**kw))
    case.info["turbulenz"] = model
    return case


def add_temperature(case: Case, alpha: float, T_inflow: float = 0.0, T_obstacle: float | None = None,
                    wall_temperatures: dict | None = None, buoyancy: bool = False,
                    gravity=(0.0, -1.0), limiter: str = "vanleer") -> Case:
    """Ergänzt Temperaturtransport (optional mit Auftrieb).

    wall_temperatures: {seite: T} für Wände mit fester Temperatur, die übrigen Wände sind adiabat.
    """
    from .models import default_scalar_bcs

    bcs = default_scalar_bcs(case.sim, T_inflow)
    for side, T in (wall_temperatures or {}).items():
        bcs[side] = ScalarBC("fixed", T)
    obs = ScalarBC("fixed", T_obstacle) if T_obstacle is not None else ScalarBC("zerogradient")
    case.sim.add_model(BoussinesqTemperature(alpha, bcs, gravity=gravity, obstacle=obs,
                                             limiter=limiter, buoyancy=buoyancy,
                                             initial=T_inflow))
    return case


def add_scalar(case: Case, name: str = "c", diffusivity: float = 1e-3, inflow_value=1.0,
               inflow_band: tuple | None = None, obstacle_value: float | None = None,
               limiter: str = "vanleer") -> Case:
    """Ergänzt einen passiven Skalar.

    inflow_band: (s0, s1) Abschnitt der Einströmränder, in dem der Skalar den Wert
    ``inflow_value`` hat (sonst 0), z. B. ein Farbstoffstreifen.
    """
    from .models import default_scalar_bcs

    bcs = default_scalar_bcs(case.sim, 0.0)
    for side, bc in bcs.items():
        if bc.kind == "fixed":
            if inflow_band is None:
                bc.value = inflow_value
            else:
                a, b = inflow_band
                bc.value = lambda s, a=a, b=b: np.where((s >= a) & (s <= b), inflow_value, 0.0)
    obs = ScalarBC("fixed", obstacle_value) if obstacle_value is not None else None
    case.sim.add_model(PassiveScalar(name, diffusivity, bcs, obs, limiter))
    return case


CASES = {
    "cavity": lid_driven_cavity,
    "cylinder": cylinder_flow,
    "channel": channel_flow,
    "step": backward_facing_step,
    "custom": custom_obstacles,
    "heated_cavity": heated_cavity,
    "rayleigh_benard": rayleigh_benard,
    "turbulent_channel": turbulent_channel,
}
