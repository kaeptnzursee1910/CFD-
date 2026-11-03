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

    def as_arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


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
    """

    def __init__(self, grid: Grid, bcs: dict[str, BoundaryCondition], nu: float,
                 scheme: str = "ab2", gamma: float | str = "auto", tau: float = 0.5,
                 dt: float | None = None, pressure_solver: str = "direct",
                 compute_forces: bool = False, initial_velocity: tuple = (0.0, 0.0)):
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
    def _apply_bc(self) -> None:
        self.boundary.apply(self.u, self.v)
        kernels.obstacle_bc(self.u, self.v, self.grid.fluid, self.grid.nx, self.grid.ny)

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
        dt_visc = 0.5 / (self.nu * (1.0 / g.dx**2 + 1.0 / g.dy**2))
        dt_conv = min(g.dx / umax, g.dy / vmax)
        if self.scheme == "ab2":
            # AB2 ist auf der negativen reellen Achse nur bis lambda*dt = -1 stabil
            # (Euler bis -2), daher halbes Diffusionslimit. Für Konvektion ist es etwas restriktiver.
            dt_visc *= 0.5
            dt_conv *= 0.8
        return self.tau * min(dt_visc, dt_conv)

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
        re_cell = max(umax * g.dx, vmax * g.dy) / self.nu
        return float(min(1.0, max(0.05, 0.015 * re_cell)))

    # ------------------------------------------------------------------ #
    # Druck und Projektion
    # ------------------------------------------------------------------ #
    def _solve_pressure_and_correct(self, dt: float) -> None:
        g = self.grid
        nx, ny = g.nx, g.ny
        dx, dy = g.dx, g.dy
        F, G = self.F, self.G

        div = (F[1:, 1:-1] - F[:-1, 1:-1]) / dx + (G[1:-1, 1:] - G[1:-1, :-1]) / dy
        p_int = self.pressure.solve(div / dt)
        p = self.p
        p[:] = 0.0
        p[1:-1, 1:-1] = p_int

        # Innere Fluidflächen
        uf = self.u_fluid[1:nx, 1:ny + 1]
        corr = F[1:nx, 1:ny + 1] - dt * (p[2:nx + 1, 1:ny + 1] - p[1:nx, 1:ny + 1]) / dx
        self.u[1:nx, 1:ny + 1] = np.where(uf, corr, F[1:nx, 1:ny + 1])
        vf = self.v_fluid[1:nx + 1, 1:ny]
        corr = G[1:nx + 1, 1:ny] - dt * (p[1:nx + 1, 2:ny + 1] - p[1:nx + 1, 1:ny]) / dy
        self.v[1:nx + 1, 1:ny] = np.where(vf, corr, G[1:nx + 1, 1:ny])

        # Randflächen: vorgegebene Geschwindigkeit, an Ausströmrändern mit Druckkorrektur
        d = self.pressure.dirichlet
        self.u[0, 1:ny + 1] = np.where(d["left"], F[0, 1:ny + 1] - 2 * dt * p[1, 1:ny + 1] / dx,
                                       F[0, 1:ny + 1])
        self.u[nx, 1:ny + 1] = np.where(d["right"], F[nx, 1:ny + 1] + 2 * dt * p[nx, 1:ny + 1] / dx,
                                        F[nx, 1:ny + 1])
        self.v[1:nx + 1, 0] = np.where(d["bottom"], G[1:nx + 1, 0] - 2 * dt * p[1:nx + 1, 1] / dy,
                                       G[1:nx + 1, 0])
        self.v[1:nx + 1, ny] = np.where(d["top"], G[1:nx + 1, ny] + 2 * dt * p[1:nx + 1, ny] / dy,
                                        G[1:nx + 1, ny])

    # ------------------------------------------------------------------ #
    # Ein Zeitschritt
    # ------------------------------------------------------------------ #
    def step(self) -> None:
        g = self.grid
        t0 = time.perf_counter()
        self._apply_bc()

        dt = self.compute_dt()
        self.gamma = self._auto_gamma(dt) if self.gamma_setting == "auto" else float(self.gamma_setting)

        kernels.momentum_tendencies(self.u, self.v, g.fluid, g.nx, g.ny, g.dx, g.dy,
                                    self.nu, self.gamma, self.hu, self.hv)

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
        self.boundary.extrapolate_outflow_predictor(self.F, self.G)

        self._solve_pressure_and_correct(dt)
        self._apply_bc()

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
