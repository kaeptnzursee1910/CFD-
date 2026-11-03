"""Turbulenzmodelle (OpenFOAM-Analogie: ``momentumTransport``-Bibliothek).

* :class:`Smagorinsky`: LES-Feinstrukturmodell, ``nu_t = (C_s Δ)^2 |S|``, mit
  Mischungsweg-Begrenzung ``l = min(C_s Δ, κ d)`` in Wandnähe.
* :class:`KOmegaSST`: RANS-Zweigleichungsmodell nach Menter, Kuntz & Langtry (2003),
  ``kOmegaSST`` in OpenFOAM, als Low-Re-Formulierung bis an die Wand (y+ ≲ 1).

Hinweis: Das Programm ist zweidimensional. RANS-Rechnungen (k-ω SST) sind in 2D
sinnvoll, LES ist es physikalisch nur eingeschränkt, weil echte Turbulenz
dreidimensional ist. Das Smagorinsky-Modell wirkt in 2D vor allem als
physikalisch motivierte, lokale Zusatzdämpfung.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .boundary import SIDES
from .models import Model, ScalarBC, ScalarField

KAPPA = 0.41


# ---------------------------------------------------------------------- #
# Hilfsfunktionen
# ---------------------------------------------------------------------- #
def wall_mask_padded(sim) -> np.ndarray:
    """Maske (nx+2, ny+2) aller Zellen, die als Wand zählen (haftende Ränder und Hindernisse)."""
    g = sim.grid
    wall = g.interior_solid_padded()
    if sim.bcs["left"].kind == "noslip":
        wall[0, :] = True
    if sim.bcs["right"].kind == "noslip":
        wall[-1, :] = True
    if sim.bcs["bottom"].kind == "noslip":
        wall[:, 0] = True
    if sim.bcs["top"].kind == "noslip":
        wall[:, -1] = True
    return wall


def wall_distance(sim) -> np.ndarray:
    """Abstand der Zellmittelpunkte zur nächsten Wand, Form (nx+2, ny+2).

    Euklidische Distanztransformation auf dem Gitter (exakt für achsenparallele
    Wände), abzüglich einer halben Zellweite, weil die Wand auf der Zellfläche liegt.
    Bei Periodizität in x wird das Feld vor der Transformation dreifach aneinandergesetzt.
    """
    g = sim.grid
    wall = wall_mask_padded(sim)
    if not wall.any():
        return np.full(wall.shape, 1e6)
    if sim.periodic:
        core = wall[1:-1, :]
        tiled = np.concatenate([core, core, core], axis=0)
        d = ndimage.distance_transform_edt(~tiled, sampling=(g.dx, g.dy))
        n = core.shape[0]
        d_core = d[n:2 * n, :]
        dist = np.empty(wall.shape)
        dist[1:-1, :] = d_core
        dist[0, :] = d_core[-1, :]
        dist[-1, :] = d_core[0, :]
    else:
        dist = ndimage.distance_transform_edt(~wall, sampling=(g.dx, g.dy))
    # Distanz Zellmitte -> Wandfläche: halbe Zellweite in Wandnormalenrichtung abziehen
    half = 0.5 * min(g.dx, g.dy)
    dist = np.maximum(dist - half, 0.5 * min(g.dx, g.dy))
    dist[~g.fluid] = 0.0
    return dist


def wall_adjacent_cells(sim) -> tuple[np.ndarray, np.ndarray]:
    """Fluidzellen direkt an einer Wand und ihr Wandabstand (halbe Zellweite)."""
    g = sim.grid
    wall = wall_mask_padded(sim)
    fl = g.fluid
    adj = np.zeros_like(fl)
    dist = np.full(fl.shape, np.inf)
    inner = (slice(1, -1), slice(1, -1))
    for sl, h in (((slice(2, None), slice(1, -1)), g.dx), ((slice(0, -2), slice(1, -1)), g.dx),
                  ((slice(1, -1), slice(2, None)), g.dy), ((slice(1, -1), slice(0, -2)), g.dy)):
        nb = wall[sl] & fl[inner]
        adj[inner] |= nb
        dist[inner] = np.where(nb, np.minimum(dist[inner], 0.5 * h), dist[inner])
    return adj, dist


def strain_rate_squared(sim) -> np.ndarray:
    """``S^2 = 2 S_ij S_ij`` in den Zellmittelpunkten, Form (nx, ny)."""
    g = sim.grid
    u, v = sim.u, sim.v
    dudx = (u[1:, 1:-1] - u[:-1, 1:-1]) / g.dx
    dvdy = (v[1:-1, 1:] - v[1:-1, :-1]) / g.dy
    dudy = (u[:, 1:] - u[:, :-1]) / g.dy
    dvdx = (v[1:, :] - v[:-1, :]) / g.dx
    shear = dudy + dvdx  # in den Knoten, (nx+1, ny+1)
    sh2 = 0.25 * (shear[:-1, :-1] ** 2 + shear[1:, :-1] ** 2 + shear[:-1, 1:] ** 2
                  + shear[1:, 1:] ** 2)
    return 2.0 * (dudx**2 + dvdy**2) + sh2


def _grad(phi: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Zentrale Gradienten eines Zellfeldes mit Geisterschicht, Ergebnis (nx, ny)."""
    gx = (phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2 * dx)
    gy = (phi[1:-1, 2:] - phi[1:-1, :-2]) / (2 * dy)
    return gx, gy


def _set_nut_ghosts(sim, nut: np.ndarray) -> None:
    """Geisterzellen von nu_t: 0 an Wänden, sonst Kopie der Innenzelle."""
    g = sim.grid
    nx, ny = g.nx, g.ny
    nut[~g.fluid] = 0.0
    if sim.periodic:
        nut[0, :] = nut[nx, :]
        nut[nx + 1, :] = nut[1, :]
    else:
        nut[0, :] = 0.0 if sim.bcs["left"].kind == "noslip" else nut[1, :]
        nut[nx + 1, :] = 0.0 if sim.bcs["right"].kind == "noslip" else nut[nx, :]
    nut[:, 0] = 0.0 if sim.bcs["bottom"].kind == "noslip" else nut[:, 1]
    nut[:, ny + 1] = 0.0 if sim.bcs["top"].kind == "noslip" else nut[:, ny]
    # Festkörper im Inneren bleiben 0
    nut[g.interior_solid_padded()] = 0.0


# ---------------------------------------------------------------------- #
# Smagorinsky (LES)
# ---------------------------------------------------------------------- #
class Smagorinsky(Model):
    kind = "turbulence"

    def __init__(self, cs: float = 0.17, wall_damping: bool = True):
        self.cs = float(cs)
        self.wall_damping = wall_damping
        self.title = f"LES Smagorinsky (C_s = {cs:g}" + (", Wanddämpfung" if wall_damping else "") + ")"

    def attach(self, sim) -> None:
        super().attach(sim)
        sim.variable_viscosity = True
        g = sim.grid
        delta = np.sqrt(g.dx * g.dy)
        l = np.full((g.nx, g.ny), self.cs * delta)
        if self.wall_damping:
            d = wall_distance(sim)[1:-1, 1:-1]
            l = np.minimum(l, KAPPA * d)
        self.l2 = l**2
        self.pre_momentum(sim)

    def pre_momentum(self, sim) -> None:
        S = np.sqrt(strain_rate_squared(sim))
        sim.nu_t[1:-1, 1:-1] = np.where(sim.grid.fluid_interior, self.l2 * S, 0.0)
        _set_nut_ghosts(sim, sim.nu_t)

    def monitors(self, sim) -> dict:
        return {"nu_t_max": float(sim.nu_t.max())}

    def fields(self, sim) -> dict:
        return {"nu_t": ("Turbulente Viskosität ν_t",
                         np.where(sim.grid.solid, np.nan, sim.nu_t[1:-1, 1:-1]))}


# ---------------------------------------------------------------------- #
# k-omega SST (Menter 2003)
# ---------------------------------------------------------------------- #
class KOmegaSST(Model):
    """k-ω-SST-Modell (Menter, Kuntz & Langtry 2003), Low-Re-Formulierung.

    Randbedingungen: k = 0 an Wänden; ω wird in wandnahen Zellen auf den
    viskosen Wert ``6 ν / (β1 y^2)`` gesetzt (wie ``omegaWallFunction`` bei
    y+ < 1). Am Einlass werden k und ω aus Turbulenzgrad und
    Viskositätsverhältnis bestimmt; sonst gilt Gradient null.

    Parameters
    ----------
    intensity : float
        Turbulenzgrad I am Einlass und für den Anfangszustand: ``k = 1.5 (I U)^2``.
    viscosity_ratio : float
        ``nu_t / nu`` am Einlass und für den Anfangszustand: ``ω = k / (ν * ratio)``.
    U_ref : float
        Bezugsgeschwindigkeit für k.
    """

    kind = "turbulence"
    a1 = 0.31
    beta_star = 0.09
    sk1, so1, beta1, gamma1 = 0.85, 0.5, 0.075, 5.0 / 9.0
    sk2, so2, beta2, gamma2 = 1.0, 0.856, 0.0828, 0.44

    def __init__(self, intensity: float = 0.05, viscosity_ratio: float = 10.0, U_ref: float = 1.0,
                 limiter: str = "upwind"):
        self.I = float(intensity)
        self.ratio = float(viscosity_ratio)
        self.U_ref = float(U_ref)
        self.limiter = limiter
        self.title = (f"RANS k-ω SST (Menter 2003), I = {intensity:g}, "
                      f"ν_t/ν = {viscosity_ratio:g}")

    def attach(self, sim) -> None:
        super().attach(sim)
        sim.variable_viscosity = True
        nu = sim.nu
        k0 = 1.5 * (self.I * self.U_ref) ** 2
        w0 = k0 / (nu * self.ratio)
        self.k_in, self.w_in = k0, w0

        kb, wb = {}, {}
        for side in SIDES:
            kind = sim.bcs[side].kind
            if kind == "inflow":
                kb[side] = ScalarBC("fixed", k0)
                wb[side] = ScalarBC("fixed", w0)
            elif kind == "noslip":
                kb[side] = ScalarBC("fixed", 0.0)
                wb[side] = ScalarBC("zerogradient")
            else:
                kb[side] = ScalarBC("zerogradient")
                wb[side] = ScalarBC("zerogradient")
        self.k = ScalarField("k", nu, kb, ScalarBC("fixed", 0.0), self.limiter, k0,
                             source=self._k_source, lower=1e-14, label="Turbulente kin. Energie k")
        self.w = ScalarField("omega", nu, wb, ScalarBC("zerogradient"), self.limiter, w0,
                             source=self._w_source, lower=1e-8, label="Spez. Dissipationsrate ω")
        self.k.attach(sim)
        self.w.attach(sim)
        sim.scalars["k"] = self.k
        sim.scalars["omega"] = self.w

        self.d = wall_distance(sim)[1:-1, 1:-1]
        self.d = np.maximum(self.d, 1e-12)
        adj, dist = wall_adjacent_cells(sim)
        self.wall_adj = adj[1:-1, 1:-1]
        self.w_wall = np.where(self.wall_adj, 6.0 * nu / (self.beta1 * np.where(
            np.isfinite(dist[1:-1, 1:-1]), dist[1:-1, 1:-1], 1.0) ** 2), 0.0)
        self._apply_wall_omega()
        self.F1 = np.ones((sim.grid.nx, sim.grid.ny))
        self.pre_momentum(sim)

    # ------------------------------------------------------------------ #
    def _apply_wall_omega(self) -> None:
        w = self.w.phi[1:-1, 1:-1]
        w[self.wall_adj] = self.w_wall[self.wall_adj]
        self.w.fill_ghosts()

    def _blending(self, sim):
        g = sim.grid
        nu = sim.nu
        k = np.maximum(self.k.phi[1:-1, 1:-1], 0.0)
        w = np.maximum(self.w.phi[1:-1, 1:-1], 1e-12)
        d = self.d
        dkx, dky = _grad(self.k.phi, g.dx, g.dy)
        dwx, dwy = _grad(self.w.phi, g.dx, g.dy)
        cross = dkx * dwx + dky * dwy
        CD = np.maximum(2.0 * self.so2 / w * cross, 1e-10)
        sk = np.sqrt(k)
        arg1 = np.minimum(np.maximum(sk / (self.beta_star * w * d), 500 * nu / (d**2 * w)),
                          4 * self.so2 * k / (CD * d**2))
        F1 = np.tanh(arg1**4)
        arg2 = np.maximum(2 * sk / (self.beta_star * w * d), 500 * nu / (d**2 * w))
        F2 = np.tanh(arg2**2)
        return F1, F2, cross

    def pre_momentum(self, sim) -> None:
        k = np.maximum(self.k.phi[1:-1, 1:-1], 0.0)
        w = np.maximum(self.w.phi[1:-1, 1:-1], 1e-12)
        S2 = strain_rate_squared(sim)
        self.S2 = S2
        F1, F2, cross = self._blending(sim)
        self.F1, self.F2, self.cross = F1, F2, cross
        nut = self.a1 * k / np.maximum(self.a1 * w, np.sqrt(S2) * F2)
        sim.nu_t[1:-1, 1:-1] = np.where(sim.grid.fluid_interior, nut, 0.0)
        _set_nut_ghosts(sim, sim.nu_t)

        # Blendete Diffusivitäten
        sk = F1 * self.sk1 + (1 - F1) * self.sk2
        so = F1 * self.so1 + (1 - F1) * self.so2
        self.k.diff[1:-1, 1:-1] = sim.nu + sk * nut
        self.w.diff[1:-1, 1:-1] = sim.nu + so * nut
        for f in (self.k, self.w):
            f.diff[0, :] = f.diff[1, :]
            f.diff[-1, :] = f.diff[-2, :]
            f.diff[:, 0] = f.diff[:, 1]
            f.diff[:, -1] = f.diff[:, -2]

    # ------------------------------------------------------------------ #
    def _k_source(self, sim, field):
        k = np.maximum(self.k.phi[1:-1, 1:-1], 0.0)
        w = np.maximum(self.w.phi[1:-1, 1:-1], 1e-12)
        nut = sim.nu_t[1:-1, 1:-1]
        Pk = np.minimum(nut * self.S2, 10.0 * self.beta_star * k * w)
        self._Pk = Pk
        return Pk, self.beta_star * w

    def _w_source(self, sim, field):
        F1 = self.F1
        w = np.maximum(self.w.phi[1:-1, 1:-1], 1e-12)
        gamma = F1 * self.gamma1 + (1 - F1) * self.gamma2
        beta = F1 * self.beta1 + (1 - F1) * self.beta2
        nut = np.maximum(sim.nu_t[1:-1, 1:-1], 1e-14)
        Pw = gamma * self._Pk / nut
        Pw = np.where(sim.nu_t[1:-1, 1:-1] > 1e-14, Pw, gamma * self.S2)
        CDw = 2.0 * (1 - F1) * self.so2 / w * self.cross
        s_exp = Pw + np.maximum(CDw, 0.0)
        s_imp = beta * w + np.maximum(-CDw, 0.0) / w
        return s_exp, s_imp

    def post_step(self, sim, dt) -> None:
        # Quellen mit dem aktuellen Geschwindigkeitsfeld neu auswerten
        self.S2 = strain_rate_squared(sim)
        self.k.step(dt)
        self.w.step(dt)
        self._apply_wall_omega()

    def dt_limit(self, sim):
        return min(self.k.dt_limit(), self.w.dt_limit())

    def monitors(self, sim) -> dict:
        return {"k_mittel": float(np.nanmean(self.k.cell())),
                "nu_t_max": float(sim.nu_t.max())}

    def fields(self, sim) -> dict:
        solid = sim.grid.solid
        return {"k": (self.k.label, self.k.cell()),
                "omega": ("log10 ω (spez. Dissipationsrate)", np.log10(np.maximum(self.w.cell(), 1e-30))),
                "nu_t": ("Turbulente Viskosität ν_t",
                         np.where(solid, np.nan, sim.nu_t[1:-1, 1:-1]))}


TURBULENCE_MODELS = {
    "laminar": None,
    "kOmegaSST": KOmegaSST,
    "Smagorinsky": Smagorinsky,
}
