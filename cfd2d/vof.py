"""Zweiphasenströmung mit der Volume-of-Fluid-Methode (OpenFOAM-Analogie: ``interFoam``).

Der Volumenanteil ``alpha`` (1 = Phase 1, z. B. Wasser; 0 = Phase 2, z. B. Luft)
wird mit einem beschränkten TVD-Schema (van Leer) transportiert. Dazu kommt
ein Interface-Kompressionsfluss ``div(u_r alpha (1 - alpha))`` mit
``u_r = c_alpha |u| n`` wie bei interFoam, der die Grenzfläche scharf hält.

Dichte und Viskosität sind gemischt: ``rho = alpha rho1 + (1 - alpha) rho2``.
Der Löser rechnet dann die Impulsgleichung mit ``(1/rho) div(mu (grad u + grad u^T))``
und die Druckgleichung ``div((1/rho) grad p) = div(u*) / dt``.
Oberflächenspannung wird nach dem CSF-Modell (Brackbill 1992) als Volumenkraft
``sigma kappa grad(alpha)`` eingebracht.
"""

from __future__ import annotations

import numpy as np

from .kernels import njit
from .models import Model


@njit(cache=True)
def _vl(phi_u, phi_c, phi_d, has_u):
    if not has_u:
        return phi_c
    den = phi_d - phi_c
    if abs(den) < 1e-300:
        return phi_c
    r = (phi_c - phi_u) / den
    psi = 0.0 if r <= 0.0 else 2.0 * r / (1.0 + r)
    return phi_c + 0.5 * psi * den


@njit(cache=True)
def vof_advect(a, u, v, fluid, nx, ny, dx, dy, dt, c_alpha, periodic, a_in, out):
    """Ein expliziter Transportschritt für alpha (konservativ, mit Kompression).

    ``a_in``: alpha, das an offenen Rändern einströmt (z. B. 0 = Luft).
    """
    for i in range(nx + 2):
        for j in range(ny + 2):
            out[i, j] = a[i, j]

    # --- x-Flächen ---
    for i in range(0, nx + 1):
        for j in range(1, ny + 1):
            uf = u[i, j]
            if i == 0 or (i == nx and not periodic):
                if i == 0 and periodic:
                    continue
                P = 1 if i == 0 else nx
                if not fluid[P, j]:
                    continue
                into = uf if i == 0 else -uf
                ab = a_in if into > 0 else a[P, j]
                out[P, j] += dt * into * ab / dx
                continue
            L = i
            R = i + 1 if i < nx else 1
            if not (fluid[L, j] and fluid[R, j]):
                continue
            if uf >= 0.0:
                C, D, Ui = L, R, L - 1
                if Ui < 1:
                    Ui = nx if periodic else -1
            else:
                C, D, Ui = R, L, R + 1
                if Ui > nx:
                    Ui = 1 if periodic else -1
            has_u = Ui >= 1 and fluid[Ui, j]
            af = _vl(a[Ui, j] if has_u else 0.0, a[C, j], a[D, j], has_u)
            flux = uf * af
            # Kompression: Normalenrichtung aus dem Gradienten an der Fläche
            gx = (a[R, j] - a[L, j]) / dx
            jn = j + 1 if j < ny else j
            js = j - 1 if j > 1 else j
            gy = 0.25 * ((a[L, jn] - a[L, js]) + (a[R, jn] - a[R, js])) / dy
            gm = np.sqrt(gx * gx + gy * gy)
            if gm > 1e-12 and c_alpha > 0.0:
                ur = c_alpha * abs(uf) * gx / gm
                ac = a[L, j] if ur >= 0.0 else a[R, j]
                ad = a[R, j] if ur >= 0.0 else a[L, j]
                # alpha*(1-alpha): stromauf alpha, stromab (1-alpha), bleibt dadurch beschränkt
                flux += ur * ac * (1.0 - ad)
            out[L, j] -= dt * flux / dx
            out[R, j] += dt * flux / dx

    # --- y-Flächen ---
    for i in range(1, nx + 1):
        for j in range(0, ny + 1):
            vf = v[i, j]
            if j == 0 or j == ny:
                P = 1 if j == 0 else ny
                if not fluid[i, P]:
                    continue
                into = vf if j == 0 else -vf
                ab = a_in if into > 0 else a[i, P]
                out[i, P] += dt * into * ab / dy
                continue
            B = j
            T = j + 1
            if not (fluid[i, B] and fluid[i, T]):
                continue
            if vf >= 0.0:
                C, D, Uj = B, T, B - 1
            else:
                C, D, Uj = T, B, T + 1
            has_u = 1 <= Uj <= ny and fluid[i, Uj]
            af = _vl(a[i, Uj] if has_u else 0.0, a[i, C], a[i, D], has_u)
            flux = vf * af
            gy = (a[i, T] - a[i, B]) / dy
            ie = i + 1 if (i < nx or periodic) else i
            iw = i - 1 if (i > 1 or periodic) else i
            if ie > nx:
                ie = 1
            if iw < 1:
                iw = nx
            gx = 0.25 * ((a[ie, B] - a[iw, B]) + (a[ie, T] - a[iw, T])) / dx
            gm = np.sqrt(gx * gx + gy * gy)
            if gm > 1e-12 and c_alpha > 0.0:
                ur = c_alpha * abs(vf) * gy / gm
                ac = a[i, B] if ur >= 0.0 else a[i, T]
                ad = a[i, T] if ur >= 0.0 else a[i, B]
                flux += ur * ac * (1.0 - ad)
            out[i, B] -= dt * flux / dy
            out[i, T] += dt * flux / dy

    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            if fluid[i, j]:
                if out[i, j] < 0.0:
                    out[i, j] = 0.0
                elif out[i, j] > 1.0:
                    out[i, j] = 1.0


class VOF(Model):
    """Zweiphasenmodell (interFoam-ähnlich).

    Parameters
    ----------
    alpha0 : ndarray (nx, ny)
        Anfangsverteilung von Phase 1 (0..1).
    rho1, rho2 : float
        Dichten von Phase 1 (Wasser) und Phase 2 (Luft).
    nu1, nu2 : float
        Kinematische Viskositäten.
    sigma : float
        Oberflächenspannung (0 = ohne).
    gravity : (gx, gy)
        Schwerebeschleunigung. Sie wird als Volumenkraft der Simulation gesetzt.
    c_alpha : float
        Kompressionsfaktor der Grenzfläche (OpenFOAM: cAlpha, typisch 1).
    cfl_alpha : float
        Maximale Courant-Zahl für den alpha-Transport.
    alpha_inflow : float
        alpha an offenen Rändern bei Einströmung (Standard 0 = Luft).
    """

    kind = "multiphase"

    def __init__(self, alpha0, rho1: float = 1000.0, rho2: float = 1.0, nu1: float = 1e-6,
                 nu2: float = 1.48e-5, sigma: float = 0.07, gravity=(0.0, -9.81),
                 c_alpha: float = 1.0, cfl_alpha: float = 0.25, alpha_inflow: float = 0.0):
        self.alpha0 = np.asarray(alpha0, dtype=float)
        self.rho1, self.rho2 = float(rho1), float(rho2)
        self.mu1, self.mu2 = float(nu1 * rho1), float(nu2 * rho2)
        self.sigma = float(sigma)
        self.gravity = (float(gravity[0]), float(gravity[1]))
        self.c_alpha = float(c_alpha)
        self.cfl = float(cfl_alpha)
        self.alpha_inflow = float(alpha_inflow)
        self.title = (f"VOF zweiphasig: rho1 = {rho1:g}, rho2 = {rho2:g}, sigma = {sigma:g}, "
                      f"g = {self.gravity}, c_alpha = {c_alpha:g}")

    def attach(self, sim) -> None:
        super().attach(sim)
        g = sim.grid
        if self.alpha0.shape != (g.nx, g.ny):
            raise ValueError(f"alpha0 muss die Form {(g.nx, g.ny)} haben.")
        self.alpha = np.zeros((g.nx + 2, g.ny + 2))
        self.alpha[1:-1, 1:-1] = np.where(g.fluid_interior, np.clip(self.alpha0, 0, 1), 0.0)
        self._tmp = np.zeros_like(self.alpha)
        sim.variable_density = True
        sim.variable_viscosity = True
        sim.body_force = (sim.body_force[0] + self.gravity[0], sim.body_force[1] + self.gravity[1])
        self._update_properties()
        self.volume0 = self.volume()

    # ------------------------------------------------------------------ #
    def _fill_ghosts(self) -> None:
        a = self.alpha
        sim = self.sim
        g = sim.grid
        nx = g.nx
        if sim.periodic:
            a[0, :] = a[nx, :]
            a[nx + 1, :] = a[1, :]
        else:
            a[0, :] = a[1, :]
            a[-1, :] = a[-2, :]
        a[:, 0] = a[:, 1]
        a[:, -1] = a[:, -2]
        solid = g.interior_solid_padded()
        if solid.any():
            # Festkörperzellen: Mittel der Fluidnachbarn (für Gradienten und Flächendichten)
            fl = g.fluid.astype(float)
            af = np.where(g.fluid, a, 0.0)
            s = np.zeros_like(a)
            c = np.zeros_like(a)
            s[1:-1, 1:-1] = af[2:, 1:-1] + af[:-2, 1:-1] + af[1:-1, 2:] + af[1:-1, :-2]
            c[1:-1, 1:-1] = fl[2:, 1:-1] + fl[:-2, 1:-1] + fl[1:-1, 2:] + fl[1:-1, :-2]
            a[solid] = np.where(c[solid] > 0, s[solid] / np.maximum(c[solid], 1), 0.0)

    def _update_properties(self) -> None:
        self._fill_ghosts()
        a = self.alpha
        sim = self.sim
        sim.rho[:] = a * self.rho1 + (1 - a) * self.rho2
        sim.nu_eff[:] = a * self.mu1 + (1 - a) * self.mu2  # dynamische Viskosität

    def volume(self) -> float:
        g = self.sim.grid
        return float(self.alpha[1:-1, 1:-1][g.fluid_interior].sum() * g.dx * g.dy)

    # ------------------------------------------------------------------ #
    def momentum_source(self, sim, hu, hv) -> None:
        if self.sigma <= 0:
            return
        g = sim.grid
        # leicht geglättetes alpha für die Krümmung
        a = self.alpha
        As = a.copy()
        As[1:-1, 1:-1] = (4 * a[1:-1, 1:-1] + a[2:, 1:-1] + a[:-2, 1:-1] + a[1:-1, 2:]
                          + a[1:-1, :-2]) / 8.0
        # Normalen in den Knoten
        gx = 0.5 * ((As[1:, 1:] - As[:-1, 1:]) + (As[1:, :-1] - As[:-1, :-1])) / g.dx
        gy = 0.5 * ((As[1:, 1:] - As[1:, :-1]) + (As[:-1, 1:] - As[:-1, :-1])) / g.dy
        gm = np.sqrt(gx**2 + gy**2) + 1e-8 / g.dx
        nxn, nyn = gx / gm, gy / gm  # (nx+1, ny+1)
        # Krümmung in den Zellen: kappa = -div(n)
        kappa = np.zeros_like(a)
        kappa[1:-1, 1:-1] = -(0.5 * ((nxn[1:, 1:] - nxn[:-1, 1:]) + (nxn[1:, :-1] - nxn[:-1, :-1])) / g.dx
                              + 0.5 * ((nyn[1:, 1:] - nyn[1:, :-1]) + (nyn[:-1, 1:] - nyn[:-1, :-1])) / g.dy)
        rho = sim.rho
        fxs = self.sigma * 0.5 * (kappa[:-1, :] + kappa[1:, :]) * (a[1:, :] - a[:-1, :]) / g.dx
        fys = self.sigma * 0.5 * (kappa[:, :-1] + kappa[:, 1:]) * (a[:, 1:] - a[:, :-1]) / g.dy
        rx = 0.5 * (rho[:-1, :] + rho[1:, :])
        ry = 0.5 * (rho[:, :-1] + rho[:, 1:])
        hu[sim.u_fluid] += (fxs / rx)[sim.u_fluid]
        hv[sim.v_fluid] += (fys / ry)[sim.v_fluid]

    def post_step(self, sim, dt) -> None:
        g = sim.grid
        # Unterteilung, falls die alpha-Courant-Zahl zu groß ist
        umax = max(np.abs(sim.u).max() / g.dx, np.abs(sim.v).max() / g.dy, 1e-12)
        nsub = max(1, int(np.ceil(dt * umax / self.cfl)))
        h = dt / nsub
        for _ in range(nsub):
            vof_advect(self.alpha, sim.u, sim.v, g.fluid, g.nx, g.ny, g.dx, g.dy, h,
                       self.c_alpha, sim.periodic, self.alpha_inflow, self._tmp)
            self.alpha[:] = self._tmp
            self._fill_ghosts()
        self._update_properties()

    def dt_limit(self, sim):
        g = sim.grid
        lim = None
        if self.sigma > 0:
            # Kapillare Zeitschrittbedingung (Brackbill)
            h = min(g.dx, g.dy)
            lim = np.sqrt(0.5 * (self.rho1 + self.rho2) * h**3 / (2 * np.pi * self.sigma))
        # Schwerewellen/Beschleunigung aus dem Stillstand
        gmag = np.hypot(*self.gravity)
        if gmag > 0:
            lg = np.sqrt(min(g.dx, g.dy) / gmag)
            lim = lg if lim is None else min(lim, lg)
        return lim

    def monitors(self, sim) -> dict:
        g = sim.grid
        a = self.alpha[1:-1, 1:-1]
        out = {"volumen_phase1": self.volume(),
               "volumenfehler_rel": self.volume() / max(self.volume0, 1e-300) - 1.0}
        # Front: größtes x mit alpha > 0.5 in der untersten Zellreihe
        row = np.where(a[:, 0] > 0.5)[0]
        out["front_x"] = float(g.xc[row.max()] + 0.5 * g.dx) if row.size else 0.0
        col = np.where(a[0, :] > 0.5)[0]
        out["hoehe_links"] = float(g.yc[col.max()] + 0.5 * g.dy) if col.size else 0.0
        return out

    def fields(self, sim) -> dict:
        solid = sim.grid.solid
        return {"alpha": ("Volumenanteil α (Phase 1)",
                          np.where(solid, np.nan, self.alpha[1:-1, 1:-1])),
                "rho": ("Dichte ρ", np.where(solid, np.nan, sim.rho[1:-1, 1:-1]))}
