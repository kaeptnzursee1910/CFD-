"""Physikalische Zusatzmodelle (OpenFOAM-Analogie: Transportgleichungen und Quellterme).

Ein Modell wird mit ``Simulation(..., models=[...])`` oder ``sim.add_model(m)``
eingebunden und klinkt sich über diese Hooks in jeden Zeitschritt ein:

``attach(sim)``                 einmalig: Felder anlegen
``pre_momentum(sim)``           vor der Impulsgleichung (z. B. Turbulenzviskosität setzen)
``momentum_source(sim, hu, hv)`` Quellterme der Impulsgleichung (z. B. Auftrieb)
``post_step(sim, dt)``          nach der Projektion (z. B. Skalare transportieren)
``dt_limit(sim)``               zusätzliche Zeitschrittgrenze
``monitors(sim)``               Kennzahlen für die Zeitverläufe (dict)
``fields(sim)``                 Zellfelder für Visualisierung und Export (dict)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import kernels
from .boundary import SIDES


# ---------------------------------------------------------------------- #
# Randbedingungen für Skalare
# ---------------------------------------------------------------------- #
@dataclass
class ScalarBC:
    """Randbedingung für ein Skalarfeld.

    kind:
        ``fixed``         fester Wert (OpenFOAM: fixedValue)
        ``zerogradient``  Gradient null (zeroGradient), z. B. adiabate Wand oder Ausströmung
        ``flux``          fester Fluss ins Gebiet (fixedGradient), z. B. Wärmestrom
    value:
        Zahl, Array entlang der Seite oder Funktion f(s) der Randkoordinate.
    """

    kind: str = "zerogradient"
    value: float | np.ndarray | object = 0.0

    _CODES = {"zerogradient": 0, "fixed": 1, "flux": 2}

    def __post_init__(self):
        if self.kind not in self._CODES:
            raise ValueError(f"Unbekannte Skalar-Randbedingung {self.kind!r}, "
                             f"erlaubt: {list(self._CODES)}")

    @property
    def code(self) -> int:
        return self._CODES[self.kind]

    def describe(self) -> str:
        names = {"zerogradient": "Gradient null", "fixed": "fester Wert", "flux": "fester Fluss"}
        v = self.value
        vs = f" = {v:g}" if isinstance(v, (int, float)) else ""
        return names[self.kind] + (vs if self.kind != "zerogradient" else "")


def default_scalar_bcs(sim, inflow_value: float = 0.0) -> dict:
    """Plausible Standardränder: Einströmung = fester Wert, sonst Gradient null."""
    out = {}
    for side in SIDES:
        kind = sim.bcs[side].kind
        if kind == "inflow":
            out[side] = ScalarBC("fixed", inflow_value)
        else:
            out[side] = ScalarBC("zerogradient")
    return out


class ScalarField:
    """Transportgleichung ``d(phi)/dt + div(u phi) = div(D grad phi) + S``.

    Die Konvektion wird mit einem TVD-Schema (Limiter wählbar) und die
    Diffusion zentral diskretisiert, die Zeitintegration ist explizit (Euler).
    Quellterme werden als ``S = S_exp - S_imp * phi`` übergeben; der implizite
    Anteil wird punktweise implizit behandelt (stabil für Senken).
    """

    def __init__(self, name: str, diffusivity: float, bcs: dict | None = None,
                 obstacle: ScalarBC | None = None, limiter: str = "vanleer",
                 initial: float | np.ndarray = 0.0, source=None,
                 lower: float | None = None, upper: float | None = None,
                 label: str | None = None):
        if limiter not in kernels.LIMITERS:
            raise ValueError(f"Unbekannter Limiter {limiter!r}, erlaubt: {list(kernels.LIMITERS)}")
        self.name = name
        self.label = label or name
        self.D0 = float(diffusivity)
        self.bcs_in = bcs
        self.obstacle = obstacle or ScalarBC("zerogradient")
        self.limiter = limiter
        self.initial = initial
        self.source = source
        self.lower = lower
        self.upper = upper
        self.sim = None

    # ------------------------------------------------------------------ #
    def attach(self, sim) -> None:
        self.sim = sim
        g = sim.grid
        nx, ny = g.nx, g.ny
        self.phi = np.zeros((nx + 2, ny + 2))
        if np.isscalar(self.initial):
            self.phi[:] = float(self.initial)
        else:
            init = np.asarray(self.initial, dtype=float)
            if init.shape == (nx, ny):
                self.phi[1:-1, 1:-1] = init
            elif init.shape == self.phi.shape:
                self.phi[:] = init
            else:
                raise ValueError(f"Anfangsfeld für {self.name} hat falsche Form {init.shape}")
        self.diff = np.full((nx + 2, ny + 2), self.D0)
        self.rhs = np.zeros_like(self.phi)

        bcs = default_scalar_bcs(sim)
        bcs.update(self.bcs_in or {})
        self.bcs = bcs
        coords = {"left": g.yc, "right": g.yc, "bottom": g.xc, "top": g.xc}
        self.bt = {}
        self.bv = {}
        for side in SIDES:
            bc = bcs[side]
            s = coords[side]
            self.bt[side] = np.full(len(s), bc.code, dtype=np.int64)
            val = bc.value
            if callable(val):
                arr = np.asarray(val(s), dtype=float)
            else:
                arr = np.broadcast_to(np.asarray(val, dtype=float), s.shape).copy()
            self.bv[side] = arr
        self.fill_ghosts()

    # ------------------------------------------------------------------ #
    def fill_ghosts(self) -> None:
        """Setzt Geister- und Festkörperzellen passend zu den Randbedingungen.

        Für den Transport selbst nicht nötig (der Kernel wertet die Ränder
        direkt aus), aber für Gradienten (Turbulenzmodelle) und Visualisierung.
        """
        phi = self.phi
        g = self.sim.grid
        nx, ny = g.nx, g.ny
        D = np.maximum(self.diff, 1e-300)

        def ghost(inner, code, val, h, Dloc):
            out = inner.copy()
            out[code == 1] = 2 * val[code == 1] - inner[code == 1]
            m = code == 2
            out[m] = inner[m] + val[m] * h / Dloc[m]
            return out

        if self.sim.periodic:
            phi[0, :] = phi[nx, :]
            phi[nx + 1, :] = phi[1, :]
        else:
            phi[0, 1:-1] = ghost(phi[1, 1:-1], self.bt["left"], self.bv["left"], g.dx, D[1, 1:-1])
            phi[nx + 1, 1:-1] = ghost(phi[nx, 1:-1], self.bt["right"], self.bv["right"], g.dx,
                                      D[nx, 1:-1])
        phi[1:-1, 0] = ghost(phi[1:-1, 1], self.bt["bottom"], self.bv["bottom"], g.dy, D[1:-1, 1])
        phi[1:-1, ny + 1] = ghost(phi[1:-1, ny], self.bt["top"], self.bv["top"], g.dy,
                                  D[1:-1, ny])

        # Festkörperzellen im Inneren: Mittel der Fluidnachbarn (bzw. gespiegelt bei festem Wert)
        solid = g.interior_solid_padded()
        if solid.any():
            fl = g.fluid.astype(float)
            pf = np.where(g.fluid, phi, 0.0)
            s = np.zeros_like(phi)
            c = np.zeros_like(phi)
            s[1:-1, 1:-1] = pf[2:, 1:-1] + pf[:-2, 1:-1] + pf[1:-1, 2:] + pf[1:-1, :-2]
            c[1:-1, 1:-1] = fl[2:, 1:-1] + fl[:-2, 1:-1] + fl[1:-1, 2:] + fl[1:-1, :-2]
            near = solid & (c > 0)
            avg = np.where(c > 0, s / np.maximum(c, 1), 0.0)
            if self.obstacle.kind == "fixed":
                phi[near] = 2 * float(self.obstacle.value) - avg[near]
                phi[solid & ~near] = float(self.obstacle.value)
            else:
                phi[near] = avg[near]

    # ------------------------------------------------------------------ #
    def step(self, dt: float) -> None:
        sim = self.sim
        g = sim.grid
        kernels.scalar_transport(
            self.phi, sim.u, sim.v, g.fluid, g.nx, g.ny, g.dx, g.dy, self.diff,
            kernels.LIMITERS[self.limiter], sim.periodic,
            self.bt["left"], self.bv["left"], self.bt["right"], self.bv["right"],
            self.bt["bottom"], self.bv["bottom"], self.bt["top"], self.bv["top"],
            self.obstacle.code, float(self.obstacle.value) if self.obstacle.kind == "fixed" else 0.0,
            self.rhs)
        inner = (slice(1, -1), slice(1, -1))
        fl = g.fluid_interior
        phi_i = self.phi[inner]
        new = phi_i + dt * self.rhs[inner]
        if self.source is not None:
            s_exp, s_imp = self.source(sim, self)
            new = (new + dt * s_exp) / (1.0 + dt * np.maximum(s_imp, 0.0))
        if self.lower is not None:
            new = np.maximum(new, self.lower)
        if self.upper is not None:
            new = np.minimum(new, self.upper)
        self.phi[inner] = np.where(fl, new, phi_i)
        self.fill_ghosts()

    def dt_limit(self) -> float:
        g = self.sim.grid
        dmax = float(self.diff[g.fluid].max()) if g.fluid.any() else self.D0
        if dmax <= 0:
            return np.inf
        return 0.5 / (dmax * (1.0 / g.dx**2 + 1.0 / g.dy**2))

    def cell(self) -> np.ndarray:
        """Werte in den Zellmittelpunkten, Form (nx, ny), NaN im Festkörper."""
        return np.where(self.sim.grid.solid, np.nan, self.phi[1:-1, 1:-1])

    def total(self) -> float:
        """Integral über das Fluidgebiet."""
        g = self.sim.grid
        return float(np.nansum(self.cell()) * g.dx * g.dy)

    def wall_flux(self, side: str) -> np.ndarray:
        """Diffusiver Fluss ``-D dphi/dn`` aus dem Gebiet heraus an den Randflächen einer Seite.

        Für die Temperatur ist das der Wandwärmestrom (positiv = Wärme verlässt das Fluid
        an dieser Wand, negativ = Wand heizt). Nur bei festem Wert oder festem Fluss
        ungleich null.
        """
        g = self.sim.grid
        phi = self.phi
        code = self.bt[side]
        val = self.bv[side]
        if side == "left":
            inner, D, h = phi[1, 1:-1], self.diff[1, 1:-1], g.dx
        elif side == "right":
            inner, D, h = phi[g.nx, 1:-1], self.diff[g.nx, 1:-1], g.dx
        elif side == "bottom":
            inner, D, h = phi[1:-1, 1], self.diff[1:-1, 1], g.dy
        else:
            inner, D, h = phi[1:-1, g.ny], self.diff[1:-1, g.ny], g.dy
        q = np.zeros_like(inner)
        m = code == 1
        q[m] = -D[m] * (val[m] - inner[m]) / (0.5 * h)  # Fluss aus dem Fluid in die Wand
        m2 = code == 2
        q[m2] = -val[m2]
        return q


# ---------------------------------------------------------------------- #
# Basisklasse
# ---------------------------------------------------------------------- #
class Model:
    kind = "model"
    title = "Modell"

    def attach(self, sim) -> None:
        self.sim = sim

    def pre_momentum(self, sim) -> None:
        pass

    def momentum_source(self, sim, hu, hv) -> None:
        pass

    def post_step(self, sim, dt) -> None:
        pass

    def dt_limit(self, sim):
        return None

    def monitors(self, sim) -> dict:
        return {}

    def fields(self, sim) -> dict:
        """{schlüssel: (beschriftung, zellfeld)}"""
        return {}

    def describe(self) -> str:
        return self.title


# ---------------------------------------------------------------------- #
# Passiver Skalar (scalarTransportFoam)
# ---------------------------------------------------------------------- #
class PassiveScalar(Model):
    """Passiver Skalar, z. B. Konzentration eines Farbstoffs oder Schadstoffs.

    Mit ``turbulent_schmidt`` wird bei Turbulenzmodellen die turbulente
    Diffusivität ``nu_t / Sc_t`` addiert.
    """

    kind = "scalar"

    def __init__(self, name: str = "c", diffusivity: float = 1e-3, bcs: dict | None = None,
                 obstacle: ScalarBC | None = None, limiter: str = "vanleer",
                 initial=0.0, source=None, turbulent_schmidt: float = 0.7,
                 label: str | None = None):
        self.field = ScalarField(name, diffusivity, bcs, obstacle, limiter, initial, source,
                                 label=label or f"Skalar {name}")
        self.title = f"Passiver Skalar '{name}' (D = {diffusivity:g}, {limiter})"
        self.sct = turbulent_schmidt
        self.name = name

    def attach(self, sim) -> None:
        super().attach(sim)
        self.field.attach(sim)
        sim.scalars[self.name] = self.field

    def post_step(self, sim, dt) -> None:
        if sim.variable_viscosity:
            self.field.diff[:] = self.field.D0 + sim.nu_t / self.sct
        self.field.step(dt)

    def dt_limit(self, sim):
        lim = self.field.dt_limit()
        return lim if np.isfinite(lim) else None

    def monitors(self, sim) -> dict:
        return {f"{self.name}_integral": self.field.total()}

    def fields(self, sim) -> dict:
        return {self.name: (self.field.label, self.field.cell())}


# ---------------------------------------------------------------------- #
# Temperatur mit Boussinesq-Auftrieb (buoyantBoussinesqPimpleFoam)
# ---------------------------------------------------------------------- #
class BoussinesqTemperature(Model):
    """Temperaturtransport mit Auftrieb in Boussinesq-Näherung.

    Impulsquelle: ``f = -beta (T - T_ref) g``. Bei der Freifall-Skalierung
    (U = sqrt(g beta dT L)) ist ``beta*|g| = 1``, und es gilt ``nu = sqrt(Pr/Ra)``
    sowie ``alpha = 1/sqrt(Ra Pr)``.

    Parameters
    ----------
    alpha : float
        Temperaturleitfähigkeit.
    gravity : (gx, gy)
        Schwerebeschleunigung mal Ausdehnungskoeffizient ``beta*g``. Standard:
        (0, -1), also Auftrieb nach oben für warmes Fluid.
    T_ref : float
        Bezugstemperatur.
    obstacle : ScalarBC
        Thermische Bedingung an Hindernissen (``fixed`` = beheizter Körper).
    prandtl_turbulent : float
        Turbulente Prandtl-Zahl (nur mit Turbulenzmodell).
    """

    kind = "thermal"

    def __init__(self, alpha: float, bcs: dict | None = None, gravity=(0.0, -1.0),
                 T_ref: float = 0.0, initial=0.0, obstacle: ScalarBC | None = None,
                 limiter: str = "vanleer", prandtl_turbulent: float = 0.85,
                 buoyancy: bool = True):
        self.field = ScalarField("T", alpha, bcs, obstacle, limiter, initial, label="Temperatur T")
        self.alpha = float(alpha)
        self.gravity = (float(gravity[0]), float(gravity[1]))
        self.T_ref = float(T_ref)
        self.prt = prandtl_turbulent
        self.buoyancy = buoyancy
        self.title = (f"Temperatur (alpha = {alpha:.4g}"
                      + (f", Auftrieb beta*g = {self.gravity}" if buoyancy else ", ohne Auftrieb")
                      + ")")

    @property
    def T(self) -> np.ndarray:
        return self.field.phi

    def attach(self, sim) -> None:
        super().attach(sim)
        self.field.attach(sim)
        sim.scalars["T"] = self.field

    def momentum_source(self, sim, hu, hv) -> None:
        if not self.buoyancy:
            return
        T = self.field.phi
        gx, gy = self.gravity
        if gx:
            Tf = 0.5 * (T[:-1, :] + T[1:, :]) - self.T_ref  # (nx+1, ny+2)
            hu[sim.u_fluid] += -gx * Tf[sim.u_fluid]
        if gy:
            Tf = 0.5 * (T[:, :-1] + T[:, 1:]) - self.T_ref  # (nx+2, ny+1)
            hv[sim.v_fluid] += -gy * Tf[sim.v_fluid]

    def post_step(self, sim, dt) -> None:
        if sim.variable_viscosity:
            self.field.diff[:] = self.alpha + sim.nu_t / self.prt
        self.field.step(dt)

    def dt_limit(self, sim):
        lim = self.field.dt_limit()
        return lim if np.isfinite(lim) else None

    def nusselt(self, side: str, L: float = 1.0, dT: float = 1.0) -> float | None:
        """Mittlere Nusselt-Zahl ``Nu = q L / (alpha dT)`` an einer Seite mit fester Temperatur."""
        if not np.any(self.field.bt[side] == 1):
            return None
        q = self.field.wall_flux(side)
        return abs(float(np.mean(q))) * L / (self.alpha * dT) if q.size else None

    def local_nusselt(self, side: str, L: float = 1.0, dT: float = 1.0) -> np.ndarray:
        return np.abs(self.field.wall_flux(side)) * L / (self.alpha * dT)

    def monitors(self, sim) -> dict:
        out = {"T_mittel": float(np.nanmean(self.field.cell()))}
        for side in SIDES:
            nu = self.nusselt(side)
            if nu is not None:
                out[f"Nu_{side}"] = nu
        return out

    def fields(self, sim) -> dict:
        return {"T": ("Temperatur T", self.field.cell())}
