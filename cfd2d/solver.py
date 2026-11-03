"""Löser für die inkompressiblen Navier-Stokes-Gleichungen in 2D.

Gleichungen (dimensionslos, Dichte rho = 1):

    du/dt + (u·grad) u = -grad p + nu Lap(u)
    div u = 0

Verfahren: Projektionsmethode nach Chorin auf versetztem MAC-Gitter.

1. Randbedingungen auf u und v setzen
2. Zeitschritt dt aus CFL- und Diffusionsbedingung bestimmen
3. Vorläufige Geschwindigkeit ``F = u + dt * H(u)``, wobei H Konvektion und
   Diffusion enthält. Zeitschema: explizites Euler-Verfahren oder
   Adams-Bashforth 2. Ordnung.
4. Druck-Poisson-Gleichung lösen: ``Lap(p) = div(F) / dt``
5. Projektion: ``u = F - dt * grad p``. Danach ist u (bis auf
   Maschinengenauigkeit) divergenzfrei.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import kernels
from .boundary import BoundaryCondition, DomainBoundary
from .grid import Grid
from .pressure import PressureSolver


class StopSimulation(Exception):
    """Wird ausgelöst, um eine laufende Simulation sauber zu beenden."""


@dataclass
class History:
    """Zeitverläufe von Monitorgrößen."""

    step: list = field(default_factory=list)
    t: list = field(default_factory=list)
    dt: list = field(default_factory=list)
    residual: list = field(default_factory=list)
    divergence: list = field(default_factory=list)
    kinetic_energy: list = field(default_factory=list)
    fx: list = field(default_factory=list)
    fy: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def as_arrays(self) -> dict:
        out = {k: np.asarray(v) for k, v in self.__dict__.items() if k != "extra"}
        for k, v in self.extra.items():
            out[k] = np.asarray(v)
        return out


class Simulation:
    """2D-Simulation inkompressibler Strömung.

    Parameters
    ----------
    grid : Grid
        Gitter mit Hindernismaske. ``grid.finalize()`` wird intern aufgerufen.
    bcs : dict
        Randbedingungen je Seite (``left``, ``right``, ``bottom``, ``top``).
    nu : float
        Kinematische Viskosität (dimensionslos: 1/Re bei U = L = 1).
    scheme : {"ab2", "euler"}
        Zeitintegration der expliziten Terme.
    gamma : float or "auto"
        Upwind-Anteil der Konvektion (0 = zentral, 1 = Donor-Cell).
        Bei ``"auto"`` wird gamma in jedem Schritt aus der lokalen
        Courant-Zahl bestimmt (Stabilitätsbedingung nach Hirt).
    tau : float
        Sicherheitsfaktor für den Zeitschritt, liegt in (0, 1].
    dt : float or None
        Fester Zeitschritt. Bei ``None`` wird er adaptiv gewählt.
    pressure_solver : {"direct", "cg"}
        Direkte LU-Zerlegung (schnell, mehr Speicher) oder vorkonditionierte
        konjugierte Gradienten.
    compute_forces : bool
        Kräfte auf innere Hindernisse in jedem Schritt aufzeichnen.
    models : list, optional
        Physikalische Zusatzmodelle (Turbulenz, Temperatur, Skalare, ...), siehe
        :mod:`cfd2d.models`. Sie werden über Hooks in jeden Zeitschritt eingebunden.
    body_force : (fx, fy)
        Konstante Volumenkraft pro Masse, z. B. ein vorgegebenes Druckgefälle.
    flow_rate : float, optional
        Soll-Geschwindigkeit (Mittelwert von u) für periodische Kanäle. Die
        antreibende Kraft wird wie bei OpenFOAM ``meanVelocityForce`` in jedem
        Schritt nachgeregelt.
    """

    def __init__(self, grid: Grid, bcs: dict[str, BoundaryCondition], nu: float,
                 scheme: str = "ab2", gamma: float | str = "auto", tau: float = 0.5,
                 dt: float | None = None, pressure_solver: str = "direct",
                 compute_forces: bool = False, initial_velocity: tuple = (0.0, 0.0),
                 models: list | None = None, body_force: tuple = (0.0, 0.0),
                 flow_rate: float | None = None):
        if nu <= 0:
            raise ValueError("Die Viskosität muss positiv sein.")
        if scheme not in ("ab2", "euler"):
            raise ValueError("scheme muss 'ab2' oder 'euler' sein.")
        if not (0 < tau <= 1):
            raise ValueError("tau muss in (0, 1] liegen.")
        if gamma != "auto" and not (0.0 <= float(gamma) <= 1.0):
            raise ValueError("gamma muss in [0, 1] liegen oder 'auto' sein.")

        self.removed_cells = grid.finalize()
        self.grid = grid
        self.nu = float(nu)
        self.scheme = scheme
        self.gamma_setting = gamma
        self.tau = float(tau)
        self.dt_fixed = dt
        self.compute_forces = compute_forces

        nx, ny = grid.nx, grid.ny
        self.boundary = DomainBoundary(grid, bcs)
        self.bcs = bcs
        self.pressure = PressureSolver(grid, self.boundary, method=pressure_solver)

        self.u = np.zeros((nx + 1, ny + 2))
        self.v = np.zeros((nx + 2, ny + 1))
        self.p = np.zeros((nx + 2, ny + 2))
        self.F = np.zeros_like(self.u)
        self.G = np.zeros_like(self.v)
        self.hu = np.zeros_like(self.u)
        self.hv = np.zeros_like(self.v)
        self.hu_old: np.ndarray | None = None
        self.hv_old: np.ndarray | None = None
        self.dt_old: float | None = None

        self.u_fluid = grid.u_fluid_faces()
        self.v_fluid = grid.v_fluid_faces()
        self.periodic = self.boundary.periodic
        if self.periodic:
            self.u_fluid[0, :] = False  # Fläche 0 ist eine Kopie von Fläche nx

        # Viskositätsfeld (molekular + turbulent), Volumenkräfte, Modelle
        self.nu_t = np.zeros((nx + 2, ny + 2))
        self.nu_eff = np.full((nx + 2, ny + 2), self.nu)
        self.variable_viscosity = False
        # Variable Dichte (VOF): rho-Zellfeld; nu_eff enthält dann die dynamische Viskosität mu
        self.variable_density = False
        self.rho = np.ones((nx + 2, ny + 2))
        self.body_force = (float(body_force[0]), float(body_force[1]))
        self.flow_rate = flow_rate
        self.flow_force = 0.0  # nachgeregeltes Druckgefälle bei flow_rate
        self.scalars: dict = {}
        self.function_objects: list = []
        self.models = []
        for m in models or []:
            self.add_model(m)

        # Anfangsbedingung
        u0, v0 = initial_velocity
        if u0:
            self.u[self.u_fluid] = u0
        if v0:
            self.v[self.v_fluid] = v0

        self.t = 0.0
        self.n = 0
        self.dt = 0.0
        self.gamma = 0.0 if gamma == "auto" else float(gamma)
        self.residual = np.inf
        self.history = History()
        self.wall_time = 0.0

        self._apply_bc()
        self._project_initial()

    # ------------------------------------------------------------------ #
    # Randbedingungen
    # ------------------------------------------------------------------ #
    def add_model(self, model) -> None:
        """Bindet ein Zusatzmodell ein (Turbulenz, Temperatur, Skalar ...)."""
        model.attach(self)
        self.models.append(model)

    def model(self, kind: str):
        """Liefert das erste Modell eines Typs (``model.kind``) oder None."""
        for m in self.models:
            if getattr(m, "kind", None) == kind:
                return m
        return None

    def _apply_bc(self) -> None:
        self.boundary.apply(self.u, self.v)
        kernels.obstacle_bc(self.u, self.v, self.grid.fluid, self.grid.nx, self.grid.ny,
                            self.periodic)

    def _project_initial(self) -> None:
        """Macht das Anfangsfeld divergenzfrei (ein Projektionsschritt mit dt = 1)."""
        if not (np.any(self.u) or np.any(self.v)):
            return
        self.F[:] = self.u
        self.G[:] = self.v
        self.boundary.extrapolate_outflow_predictor(self.F, self.G)
        self._solve_pressure_and_correct(1.0)
        self.p[:] = 0.0
        self._apply_bc()

    # ------------------------------------------------------------------ #
    # Zeitschritt
    # ------------------------------------------------------------------ #
    def compute_dt(self) -> float:
        if self.dt_fixed:
            return float(self.dt_fixed)
        g = self.grid
        umax = max(np.abs(self.u).max(), 1e-12)
        vmax = max(np.abs(self.v).max(), 1e-12)
        if self.variable_density:
            nu_max = float((self.nu_eff / self.rho)[self.grid.fluid].max())
        else:
            nu_max = float(self.nu_eff.max()) if self.variable_viscosity else self.nu
        dt_visc = 0.5 / (nu_max * (1.0 / g.dx**2 + 1.0 / g.dy**2))
        dt_conv = min(g.dx / umax, g.dy / vmax)
        if self.scheme == "ab2":
            # AB2 ist auf der negativen reellen Achse nur bis lambda*dt = -1 stabil
            # (Euler bis -2), daher halbes Diffusionslimit. Für Konvektion ist es etwas restriktiver.
            dt_visc *= 0.5
            dt_conv *= 0.8
        dt = self.tau * min(dt_visc, dt_conv)
        for m in self.models:
            lim = m.dt_limit(self)
            if lim is not None:
                dt = min(dt, self.tau * lim)
        return dt

    def _auto_gamma(self, dt: float) -> float:
        """Automatischer Upwind-Anteil.

        * Euler: Stabilitätsbedingung nach Hirt, gamma >= max(|u| dt/dx, |v| dt/dy).
        * AB2: Das Verfahren ist mit zentralen Differenzen deutlich stabiler.
          gamma wächst mit der Zell-Reynolds-Zahl Re_h = |u|max h / nu und
          dämpft so nur dort, wo zentrale Differenzen oszillieren würden.
          Bei gut aufgelösten Rechnungen bleibt es klein, nahe an 2. Ordnung.
        """
        g = self.grid
        umax = np.abs(self.u).max()
        vmax = np.abs(self.v).max()
        if self.scheme == "euler":
            cfl = max(umax * dt / g.dx, vmax * dt / g.dy)
            return float(min(1.0, max(0.05, 1.2 * cfl)))
        if self.variable_density:
            nu_ref = float((self.nu_eff / self.rho)[self.grid.fluid].min())
        else:
            nu_ref = float(self.nu_eff[self.grid.fluid].min()) if self.variable_viscosity else self.nu
        re_cell = max(umax * g.dx, vmax * g.dy) / nu_ref
        return float(min(1.0, max(0.05, 0.015 * re_cell)))

    # ------------------------------------------------------------------ #
    # Druck und Projektion
    # ------------------------------------------------------------------ #
    def _solve_pressure_and_correct(self, dt: float) -> None:
        g = self.grid
        nx, ny = g.nx, g.ny
        dx, dy = g.dx, g.dy
        F, G = self.F, self.G

        if self.periodic:
            F[0, :] = F[nx, :]
        div = (F[1:, 1:-1] - F[:-1, 1:-1]) / dx + (G[1:-1, 1:] - G[1:-1, :-1]) / dy
        if self.variable_density:
            # Flächenkoeffizienten 1/rho (harmonisch gemittelte Dichte wäre die Alternative)
            bx = 2.0 / (self.rho[:-1, :] + self.rho[1:, :])  # (nx+1, ny+2)
            by = 2.0 / (self.rho[:, :-1] + self.rho[:, 1:])  # (nx+2, ny+1)
            if self.periodic:
                bx[nx, :] = 2.0 / (self.rho[nx, :] + self.rho[1, :])
            p_int = self.pressure.solve_variable(div / dt, bx[:, 1:-1], by[1:-1, :])
        else:
            bx = by = None
            p_int = self.pressure.solve(div / dt)
        p = self.p
        p[:] = 0.0
        p[1:-1, 1:-1] = p_int
        if self.periodic:
            p[0, :] = p[nx, :]
            p[nx + 1, :] = p[1, :]

        def cx(sl):
            return 1.0 if bx is None else bx[sl]

        def cy(sl):
            return 1.0 if by is None else by[sl]

        # Innere Fluidflächen
        uf = self.u_fluid[1:nx, 1:ny + 1]
        corr = F[1:nx, 1:ny + 1] - dt * cx((slice(1, nx), slice(1, ny + 1))) * (
            p[2:nx + 1, 1:ny + 1] - p[1:nx, 1:ny + 1]) / dx
        self.u[1:nx, 1:ny + 1] = np.where(uf, corr, F[1:nx, 1:ny + 1])
        vf = self.v_fluid[1:nx + 1, 1:ny]
        corr = G[1:nx + 1, 1:ny] - dt * cy((slice(1, nx + 1), slice(1, ny))) * (
            p[1:nx + 1, 2:ny + 1] - p[1:nx + 1, 1:ny]) / dy
        self.v[1:nx + 1, 1:ny] = np.where(vf, corr, G[1:nx + 1, 1:ny])

        if self.periodic:
            uf = self.u_fluid[nx, 1:ny + 1]
            corr = F[nx, 1:ny + 1] - dt * cx((nx, slice(1, ny + 1))) * (
                p[1, 1:ny + 1] - p[nx, 1:ny + 1]) / dx
            self.u[nx, 1:ny + 1] = np.where(uf, corr, F[nx, 1:ny + 1])
            self.u[0, :] = self.u[nx, :]
            return

        # Randflächen: vorgegebene Geschwindigkeit, an Ausströmrändern mit Druckkorrektur
        d = self.pressure.dirichlet
        self.u[0, 1:ny + 1] = np.where(
            d["left"], F[0, 1:ny + 1] - 2 * dt * cx((0, slice(1, ny + 1))) * p[1, 1:ny + 1] / dx,
            F[0, 1:ny + 1])
        self.u[nx, 1:ny + 1] = np.where(
            d["right"], F[nx, 1:ny + 1] + 2 * dt * cx((nx, slice(1, ny + 1))) * p[nx, 1:ny + 1] / dx,
            F[nx, 1:ny + 1])
        self.v[1:nx + 1, 0] = np.where(
            d["bottom"], G[1:nx + 1, 0] - 2 * dt * cy((slice(1, nx + 1), 0)) * p[1:nx + 1, 1] / dy,
            G[1:nx + 1, 0])
        self.v[1:nx + 1, ny] = np.where(
            d["top"], G[1:nx + 1, ny] + 2 * dt * cy((slice(1, nx + 1), ny)) * p[1:nx + 1, ny] / dy,
            G[1:nx + 1, ny])

    # ------------------------------------------------------------------ #
    # Ein Zeitschritt
    # ------------------------------------------------------------------ #
    def step(self) -> None:
        g = self.grid
        t0 = time.perf_counter()
        self._apply_bc()
        for m in self.models:
            m.pre_momentum(self)
        if self.variable_viscosity and not self.variable_density:
            np.add(self.nu, self.nu_t, out=self.nu_eff)

        dt = self.compute_dt()
        self.gamma = self._auto_gamma(dt) if self.gamma_setting == "auto" else float(self.gamma_setting)

        kernels.momentum_tendencies(self.u, self.v, g.fluid, g.nx, g.ny, g.dx, g.dy,
                                    self.nu_eff, self.variable_viscosity, self.gamma,
                                    self.periodic, self.hu, self.hv, self.rho,
                                    self.variable_density)
        for m in self.models:
            m.momentum_source(self, self.hu, self.hv)

        if self.scheme == "ab2" and self.hu_old is not None:
            r = dt / self.dt_old
            a, b = 1.0 + 0.5 * r, -0.5 * r
            Hu = a * self.hu + b * self.hu_old
            Hv = a * self.hv + b * self.hv_old
        else:
            Hu, Hv = self.hu, self.hv

        u_prev = self.u.copy()
        v_prev = self.v.copy()

        self.F[:] = self.u
        self.G[:] = self.v
        self.F[self.u_fluid] += dt * Hu[self.u_fluid]
        self.G[self.v_fluid] += dt * Hv[self.v_fluid]
        # Volumenkraft (konstant oder geregelt) explizit, nicht über AB2: Sonst bildet der
        # Volumenstromregler mit der AB2-Extrapolation eine ungedämpfte 2-Schritt-Schwingung.
        fx = self.body_force[0] + self.flow_force
        fy = self.body_force[1]
        if fx:
            self.F[self.u_fluid] += dt * fx
        if fy:
            self.G[self.v_fluid] += dt * fy
        self.boundary.extrapolate_outflow_predictor(self.F, self.G)

        self._solve_pressure_and_correct(dt)
        self._apply_bc()
        if self.flow_rate is not None:
            self._control_flow_rate(dt)

        for m in self.models:
            m.post_step(self, dt)

        if self.scheme == "ab2":
            if self.hu_old is None:
                self.hu_old = np.empty_like(self.hu)
                self.hv_old = np.empty_like(self.hv)
            self.hu_old[:] = self.hu
            self.hv_old[:] = self.hv
        self.dt_old = dt

        self.t += dt
        self.n += 1
        self.dt = dt

        du = np.abs(self.u[self.u_fluid] - u_prev[self.u_fluid]).max(initial=0.0)
        dv = np.abs(self.v[self.v_fluid] - v_prev[self.v_fluid]).max(initial=0.0)
        self.residual = max(du, dv) / dt
        self.wall_time += time.perf_counter() - t0

        if not np.isfinite(self.residual) or self.residual > 1e8:
            raise FloatingPointError(
                f"Simulation instabil in Schritt {self.n} (t = {self.t:.4g}). "
                "Zeitschritt verkleinern (tau), Gitter verfeinern oder gamma erhöhen."
            )

        self._record()
        for fo in self.function_objects:
            fo.execute(self)

    def bulk_velocity(self) -> float:
        """Mittlere Geschwindigkeit u über alle Fluidflächen (periodischer Kanal)."""
        m = self.u_fluid[1:self.grid.nx + 1, 1:-1]
        return float(self.u[1:self.grid.nx + 1, 1:-1][m].mean())

    def _control_flow_rate(self, dt: float) -> None:
        """Regelt den Volumenstrom auf ``flow_rate`` (wie OpenFOAM meanVelocityForce)."""
        du = self.flow_rate - self.bulk_velocity()
        self.u[self.u_fluid] += du
        if self.periodic:
            self.u[0, :] = self.u[self.grid.nx, :]
        self.flow_force += du / dt
        if self.grid.solid.any():
            # Mit Hindernissen ist eine gleichmäßige Korrektur nicht divergenzfrei: neu projizieren
            self.F[:] = self.u
            self.G[:] = self.v
            self.boundary.extrapolate_outflow_predictor(self.F, self.G)
            p_save = self.p.copy()
            self._solve_pressure_and_correct(1.0)
            self.p[:] = p_save
        self._apply_bc()

    def _record(self) -> None:
        h = self.history
        h.step.append(self.n)
        h.t.append(self.t)
        h.dt.append(self.dt)
        h.residual.append(self.residual)
        h.divergence.append(self.max_divergence())
        h.kinetic_energy.append(self.kinetic_energy())
        if self.compute_forces:
            fx, fy = self.obstacle_forces()
            h.fx.append(fx)
            h.fy.append(fy)
        if self.flow_rate is not None:
            h.extra.setdefault("antriebskraft", []).append(self.flow_force)
        for m in self.models:
            for k, val in (m.monitors(self) or {}).items():
                h.extra.setdefault(k, []).append(val)

    # ------------------------------------------------------------------ #
    # Simulationsschleife
    # ------------------------------------------------------------------ #
    def run(self, t_end: float, max_steps: int | None = None, steady_tol: float | None = None,
            callback=None, callback_every: int = 50, verbose: bool = True,
            print_every: int = 200) -> str:
        """Rechnet bis ``t_end`` oder bis Konvergenz (``residual < steady_tol``).

        ``callback(sim)`` wird alle ``callback_every`` Schritte aufgerufen. Liefert
        der Callback ``False`` oder löst er :class:`StopSimulation` aus, endet
        die Rechnung.

        Returns
        -------
        str
            Grund für das Ende: ``"t_end"``, ``"konvergiert"``, ``"max_steps"``,
            ``"abgebrochen"``.
        """
        reason = "t_end"
        t_start = time.perf_counter()
        try:
            while self.t < t_end - 1e-12:
                if max_steps is not None and self.n >= max_steps:
                    reason = "max_steps"
                    break
                if self.dt_fixed is None:
                    # letzten Schritt so wählen, dass genau t_end erreicht wird
                    dt_next = self.compute_dt()
                    if self.t + dt_next > t_end:
                        self.dt_fixed = t_end - self.t
                        try:
                            self.step()
                        finally:
                            self.dt_fixed = None
                    else:
                        self.step()
                else:
                    self.step()

                if verbose and self.n % print_every == 0:
                    print(self.status_line())
                if callback is not None and self.n % callback_every == 0:
                    if callback(self) is False:
                        reason = "abgebrochen"
                        break
                if steady_tol is not None and self.n > 10 and self.residual < steady_tol:
                    reason = "konvergiert"
                    break
        except (KeyboardInterrupt, StopSimulation):
            reason = "abgebrochen"
        if callback is not None:
            try:
                callback(self)
            except StopSimulation:
                pass
        if verbose:
            print(self.status_line())
            el = time.perf_counter() - t_start
            print(f"Ende: {reason}. {self.n} Schritte, Rechenzeit {el:.1f} s "
                  f"({1e3 * self.wall_time / max(self.n, 1):.2f} ms/Schritt)")
        return reason

    def status_line(self) -> str:
        s = (f"Schritt {self.n:7d} | t = {self.t:9.4f} | dt = {self.dt:.3e} | "
             f"Residuum = {self.residual:.3e} | max|div u| = {self.max_divergence():.2e}")
        if self.compute_forces and self.history.fx:
            s += f" | Fx = {self.history.fx[-1]:+.4f} | Fy = {self.history.fy[-1]:+.4f}"
        return s

    # ------------------------------------------------------------------ #
    # Auswertungsgrößen
    # ------------------------------------------------------------------ #
    def divergence(self) -> np.ndarray:
        g = self.grid
        div = ((self.u[1:, 1:-1] - self.u[:-1, 1:-1]) / g.dx
               + (self.v[1:-1, 1:] - self.v[1:-1, :-1]) / g.dy)
        return np.where(g.fluid_interior, div, 0.0)

    def max_divergence(self) -> float:
        return float(np.abs(self.divergence()).max())

    def cell_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Geschwindigkeiten in den Zellmittelpunkten, Form (nx, ny), NaN im Festkörper."""
        uc = 0.5 * (self.u[:-1, 1:-1] + self.u[1:, 1:-1])
        vc = 0.5 * (self.v[1:-1, :-1] + self.v[1:-1, 1:])
        solid = self.grid.solid
        uc = np.where(solid, np.nan, uc)
        vc = np.where(solid, np.nan, vc)
        return uc, vc

    def cell_pressure(self) -> np.ndarray:
        return np.where(self.grid.solid, np.nan, self.p[1:-1, 1:-1])

    def kinetic_energy(self) -> float:
        uc, vc = self.cell_velocity()
        return float(0.5 * np.nansum(uc**2 + vc**2) * self.grid.dx * self.grid.dy)

    def obstacle_forces(self) -> tuple[float, float]:
        """Kraft (Fx, Fy) des Fluids auf alle inneren Hindernisse (Dichte 1).

        Druckkraft und Wandschubspannung werden über alle Flächen zwischen
        Fluid- und Hinderniszellen aufsummiert (Treppenstufen-Approximation).
        """
        g = self.grid
        fl = g.fluid
        solid = ~fl
        solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = False  # Gebietsrand ausnehmen
        f = fl[1:-1, 1:-1]
        east = f & solid[2:, 1:-1]
        west = f & solid[:-2, 1:-1]
        north = f & solid[1:-1, 2:]
        south = f & solid[1:-1, :-2]

        p = self.p[1:-1, 1:-1]
        uc = 0.5 * (self.u[:-1, 1:-1] + self.u[1:, 1:-1])
        vc = 0.5 * (self.v[1:-1, :-1] + self.v[1:-1, 1:])
        mu = self.nu

        fx = (p[east].sum() - p[west].sum()) * g.dy
        fy = (p[north].sum() - p[south].sum()) * g.dx
        fx += mu * (uc[north].sum() + uc[south].sum()) / (0.5 * g.dy) * g.dx
        fy += mu * (vc[east].sum() + vc[west].sum()) / (0.5 * g.dx) * g.dy
        return float(fx), float(fy)

    def vorticity(self) -> np.ndarray:
        """Wirbelstärke omega = dv/dx - du/dy in den Gitterknoten, Form (nx+1, ny+1)."""
        g = self.grid
        return ((self.v[1:, :] - self.v[:-1, :]) / g.dx
                - (self.u[:, 1:] - self.u[:, :-1]) / g.dy)

    def cell_vorticity(self) -> np.ndarray:
        w = self.vorticity()
        wc = 0.25 * (w[:-1, :-1] + w[1:, :-1] + w[:-1, 1:] + w[1:, 1:])
        return np.where(self.grid.solid, np.nan, wc)

    def streamfunction(self) -> np.ndarray:
        """Stromfunktion psi in den Gitterknoten, Form (nx+1, ny+1).

        Es gilt u = dpsi/dy und v = -dpsi/dx, mit psi(0, 0) = 0.
        """
        g = self.grid
        fl = g.fluid
        # Nur Flächen mit mindestens einer angrenzenden Fluidzelle tragen zum Fluss bei.
        # Die gespiegelten Hilfswerte im Festkörper werden ignoriert, Körper sind Stromlinien.
        u_open = (fl[:-1, 1:-1] | fl[1:, 1:-1])
        v_open = (fl[1:-1, 0] | fl[1:-1, 1])
        u = np.where(u_open, self.u[:, 1:-1], 0.0)
        v0 = np.where(v_open, self.v[1:-1, 0], 0.0)
        psi = np.zeros((g.nx + 1, g.ny + 1))
        psi[1:, 0] = -np.cumsum(v0) * g.dx
        psi[:, 1:] = psi[:, :1] + np.cumsum(u, axis=1) * g.dy
        return psi

    def sample_u(self, x, y) -> np.ndarray:
        """Interpoliert u bilinear an beliebigen Punkten (inklusive Wandwerten über Geisterzellen)."""
        from scipy.interpolate import RegularGridInterpolator

        g = self.grid
        xs = np.arange(g.nx + 1) * g.dx
        ys = (np.arange(g.ny + 2) - 0.5) * g.dy
        f = RegularGridInterpolator((xs, ys), self.u, bounds_error=False, fill_value=None)
        return f(np.column_stack([np.ravel(x), np.ravel(y)]))

    def sample_v(self, x, y) -> np.ndarray:
        from scipy.interpolate import RegularGridInterpolator

        g = self.grid
        xs = (np.arange(g.nx + 2) - 0.5) * g.dx
        ys = np.arange(g.ny + 1) * g.dy
        f = RegularGridInterpolator((xs, ys), self.v, bounds_error=False, fill_value=None)
        return f(np.column_stack([np.ravel(x), np.ravel(y)]))

    def summary(self) -> str:
        g = self.grid
        lines = [
            f"Gitter: {g.nx} x {g.ny} Zellen, Gebiet {g.lx:g} x {g.ly:g}, "
            f"dx = {g.dx:.4g}, dy = {g.dy:.4g}",
            f"Fluidzellen: {g.fluid_interior.sum()} / {g.nx * g.ny}"
            + (f" ({self.removed_cells} eingeschlossene Zellen entfernt)" if self.removed_cells else ""),
            f"Viskosität nu = {self.nu:.4g}",
            f"Zeitschema: {'Adams-Bashforth 2' if self.scheme == 'ab2' else 'Euler explizit'}, "
            f"tau = {self.tau}, gamma = {self.gamma_setting}",
            f"Drucklöser: {'LU direkt' if self.pressure.method == 'direct' else 'CG (Jacobi)'}",
            "Randbedingungen:",
            self.boundary.describe(),
        ]
        return "\n".join(lines)
