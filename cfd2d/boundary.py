"""Randbedingungen am Gebietsrand.

Jede der vier Seiten (``left``, ``right``, ``bottom``, ``top``) bekommt eine
:class:`BoundaryCondition`. Mögliche Arten:

``noslip``
    Haftende Wand. ``velocity`` ist die tangentiale Wandgeschwindigkeit,
    zum Beispiel der bewegte Deckel der Cavity.
``slip``
    Reibungsfreie Wand: Normalgeschwindigkeit 0, kein Gradient tangential.
``inflow``
    Einströmrand. ``velocity`` ist die mittlere Geschwindigkeit ins Gebiet
    hinein, ``profile`` ist ``"uniform"`` (Blockprofil) oder ``"parabolic"``.
``outflow``
    Ausströmrand: kein Gradient der Geschwindigkeit, Druck p = 0
    (Dirichlet-Bedingung für die Druckgleichung).
``periodic``
    Periodischer Rand (OpenFOAM: ``cyclic``). Nur paarweise links und rechts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid

SIDES = ("left", "right", "bottom", "top")
SIDE_NAMES_DE = {"left": "links", "right": "rechts", "bottom": "unten", "top": "oben"}
KINDS = ("noslip", "slip", "inflow", "outflow", "periodic")
KIND_NAMES_DE = {
    "noslip": "Haftende Wand",
    "slip": "Reibungsfreie Wand",
    "inflow": "Einströmung",
    "outflow": "Ausströmung",
    "periodic": "Periodisch",
}


@dataclass
class BoundaryCondition:
    kind: str = "noslip"
    velocity: float = 0.0
    profile: str = "uniform"

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unbekannte Randbedingung {self.kind!r}, erlaubt: {KINDS}")
        if self.profile not in ("uniform", "parabolic"):
            raise ValueError("Einströmprofil muss 'uniform' oder 'parabolic' sein.")

    def describe(self) -> str:
        s = KIND_NAMES_DE[self.kind]
        if self.kind == "noslip" and self.velocity != 0.0:
            s += f" (bewegt, U={self.velocity:g})"
        if self.kind == "inflow":
            prof = "parabolisch" if self.profile == "parabolic" else "Blockprofil"
            s += f" ({prof}, U_mittel={self.velocity:g})"
        return s


def _profile_along(coords: np.ndarray, open_mask: np.ndarray, h: float,
                   mean_velocity: float, profile: str) -> np.ndarray:
    """Einströmprofil entlang einer Seite.

    Für jeden zusammenhängenden offenen Abschnitt (die angrenzende Zelle im
    Inneren ist Fluid) wird ein eigenes Profil erzeugt, zum Beispiel nur über
    dem Einlass oberhalb einer Rückwärtsstufe.
    """
    values = np.zeros_like(coords)
    n = len(open_mask)
    j = 0
    while j < n:
        if not open_mask[j]:
            j += 1
            continue
        k = j
        while k + 1 < n and open_mask[k + 1]:
            k += 1
        a = coords[j] - 0.5 * h
        b = coords[k] + 0.5 * h
        s = coords[j:k + 1]
        if profile == "parabolic":
            values[j:k + 1] = 6.0 * mean_velocity * (s - a) * (b - s) / (b - a) ** 2
        else:
            values[j:k + 1] = mean_velocity
        j = k + 1
    return values


class DomainBoundary:
    """Setzt die Randbedingungen des Gebiets auf u und v (inklusive Geisterzellen)."""

    def __init__(self, grid: Grid, bcs: dict[str, BoundaryCondition]):
        missing = [s for s in SIDES if s not in bcs]
        if missing:
            raise ValueError(f"Randbedingungen fehlen für: {missing}")
        self.grid = grid
        self.bcs = bcs
        pl = bcs["left"].kind == "periodic"
        pr = bcs["right"].kind == "periodic"
        if pl != pr:
            raise ValueError("Periodische Ränder müssen links UND rechts gesetzt werden.")
        if bcs["bottom"].kind == "periodic" or bcs["top"].kind == "periodic":
            raise ValueError("Periodische Ränder werden nur in x-Richtung (links/rechts) unterstützt.")
        self.periodic = pl
        if self.periodic:
            grid.set_periodic_x()
        fl = grid.fluid
        nx, ny = grid.nx, grid.ny

        # Offene Randabschnitte: Ist die angrenzende Innenzelle Fluid?
        self.open = {
            "left": fl[1, 1:ny + 1].copy(),
            "right": fl[nx, 1:ny + 1].copy(),
            "bottom": fl[1:nx + 1, 1].copy(),
            "top": fl[1:nx + 1, ny].copy(),
        }

        # Einströmprofile vorab berechnen (Normalgeschwindigkeit ins Gebiet hinein)
        self.inflow_values = {}
        for side, bc in bcs.items():
            if bc.kind != "inflow":
                continue
            if side in ("left", "right"):
                vals = _profile_along(grid.yc, self.open[side], grid.dy, bc.velocity, bc.profile)
            else:
                vals = _profile_along(grid.xc, self.open[side], grid.dx, bc.velocity, bc.profile)
            sign = 1.0 if side in ("left", "bottom") else -1.0
            self.inflow_values[side] = sign * vals

    @property
    def has_outflow(self) -> bool:
        return any(bc.kind == "outflow" for bc in self.bcs.values())

    def outflow_mask(self, side: str) -> np.ndarray:
        """Flächen am Rand mit Druck-Dirichlet-Bedingung (Ausströmung und offen)."""
        if self.bcs[side].kind != "outflow":
            return np.zeros_like(self.open[side])
        return self.open[side]

    def apply(self, u: np.ndarray, v: np.ndarray) -> None:
        """Setzt alle Randwerte auf u und v."""
        g = self.grid
        nx, ny = g.nx, g.ny

        if self.periodic:
            u[0, :] = u[nx, :]
            v[0, :] = v[nx, :]
            v[nx + 1, :] = v[1, :]

        # ---------- links (x = 0): Normalkomponente u[0, j], tangential v[0, :] ----------
        bc = self.bcs["left"]
        op = self.open["left"]
        if bc.kind == "noslip":
            u[0, 1:ny + 1] = 0.0
            v[0, :] = 2.0 * bc.velocity - v[1, :]
        elif bc.kind == "slip":
            u[0, 1:ny + 1] = 0.0
            v[0, :] = v[1, :]
        elif bc.kind == "inflow":
            u[0, 1:ny + 1] = self.inflow_values["left"]
            v[0, :] = -v[1, :]
        elif bc.kind == "outflow":
            # u[0, j] wird von der Druckkorrektur bestimmt
            u[0, 1:ny + 1] = np.where(op, u[0, 1:ny + 1], 0.0)
            v[0, :] = v[1, :]

        # ---------- rechts (x = lx): u[nx, j], tangential v[nx+1, :] ----------
        bc = self.bcs["right"]
        op = self.open["right"]
        if bc.kind == "noslip":
            u[nx, 1:ny + 1] = 0.0
            v[nx + 1, :] = 2.0 * bc.velocity - v[nx, :]
        elif bc.kind == "slip":
            u[nx, 1:ny + 1] = 0.0
            v[nx + 1, :] = v[nx, :]
        elif bc.kind == "inflow":
            u[nx, 1:ny + 1] = self.inflow_values["right"]
            v[nx + 1, :] = -v[nx, :]
        elif bc.kind == "outflow":
            u[nx, 1:ny + 1] = np.where(op, u[nx, 1:ny + 1], 0.0)
            v[nx + 1, :] = v[nx, :]

        # ---------- unten (y = 0): v[i, 0], tangential u[:, 0] ----------
        bc = self.bcs["bottom"]
        op = self.open["bottom"]
        if bc.kind == "noslip":
            v[1:nx + 1, 0] = 0.0
            u[:, 0] = 2.0 * bc.velocity - u[:, 1]
        elif bc.kind == "slip":
            v[1:nx + 1, 0] = 0.0
            u[:, 0] = u[:, 1]
        elif bc.kind == "inflow":
            v[1:nx + 1, 0] = self.inflow_values["bottom"]
            u[:, 0] = -u[:, 1]
        elif bc.kind == "outflow":
            v[1:nx + 1, 0] = np.where(op, v[1:nx + 1, 0], 0.0)
            u[:, 0] = u[:, 1]

        # ---------- oben (y = ly): v[i, ny], tangential u[:, ny+1] ----------
        bc = self.bcs["top"]
        op = self.open["top"]
        if bc.kind == "noslip":
            v[1:nx + 1, ny] = 0.0
            u[:, ny + 1] = 2.0 * bc.velocity - u[:, ny]
        elif bc.kind == "slip":
            v[1:nx + 1, ny] = 0.0
            u[:, ny + 1] = u[:, ny]
        elif bc.kind == "inflow":
            v[1:nx + 1, ny] = self.inflow_values["top"]
            u[:, ny + 1] = -u[:, ny]
        elif bc.kind == "outflow":
            v[1:nx + 1, ny] = np.where(op, v[1:nx + 1, ny], 0.0)
            u[:, ny + 1] = u[:, ny]

        # Randflächen, hinter denen ein Hindernis liegt, sind geschlossen
        if not self.periodic:
            u[0, 1:ny + 1][~self.open["left"]] = 0.0
            u[nx, 1:ny + 1][~self.open["right"]] = 0.0
        v[1:nx + 1, 0][~self.open["bottom"]] = 0.0
        v[1:nx + 1, ny][~self.open["top"]] = 0.0

    def extrapolate_outflow_predictor(self, F: np.ndarray, G: np.ndarray) -> None:
        """Setzt die vorläufigen Geschwindigkeiten an Ausströmrändern (kein Gradient)."""
        g = self.grid
        nx, ny = g.nx, g.ny
        if self.periodic:
            F[0, :] = F[nx, :]
        if self.bcs["left"].kind == "outflow":
            m = self.open["left"]
            F[0, 1:ny + 1][m] = F[1, 1:ny + 1][m]
        if self.bcs["right"].kind == "outflow":
            m = self.open["right"]
            F[nx, 1:ny + 1][m] = F[nx - 1, 1:ny + 1][m]
        if self.bcs["bottom"].kind == "outflow":
            m = self.open["bottom"]
            G[1:nx + 1, 0][m] = G[1:nx + 1, 1][m]
        if self.bcs["top"].kind == "outflow":
            m = self.open["top"]
            G[1:nx + 1, ny][m] = G[1:nx + 1, ny - 1][m]

    def describe(self) -> str:
        return "\n".join(f"  {SIDE_NAMES_DE[s]:>6}: {self.bcs[s].describe()}" for s in SIDES)
