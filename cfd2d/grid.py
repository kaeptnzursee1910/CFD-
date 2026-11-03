"""Versetztes (staggered) MAC-Gitter für 2D-Strömungen.

Indexkonvention (wie bei Griebel, Dornseifer & Neunhoeffer,
"Numerische Simulation in der Strömungsmechanik"):

* Zellen werden mit ``(i, j)`` indiziert, ``i`` in x-Richtung, ``j`` in y-Richtung.
  Inklusive einer Geisterzellenschicht laufen sie von ``0 .. nx+1`` bzw. ``0 .. ny+1``;
  die physikalischen Zellen sind ``1 .. nx`` und ``1 .. ny``.
* Druck ``p`` liegt im Zellmittelpunkt, Form ``(nx+2, ny+2)``.
* ``u[i, j]`` liegt auf der rechten Seitenfläche der Zelle ``(i, j)``
  (zwischen Zelle ``i`` und ``i+1``), Form ``(nx+1, ny+2)``.
* ``v[i, j]`` liegt auf der oberen Seitenfläche der Zelle ``(i, j)``
  (zwischen Zelle ``j`` und ``j+1``), Form ``(nx+2, ny+1)``.

Hindernisse werden über das boolesche Feld ``fluid`` beschrieben
(``True`` = Fluidzelle, ``False`` = Festkörper). Die Geisterzellen am Rand
sind immer ``False``; die Randbedingungen des Gebiets werden separat gesetzt.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


class Grid:
    """Kartesisches, äquidistantes MAC-Gitter mit Hindernismaske."""

    def __init__(self, nx: int, ny: int, lx: float, ly: float):
        if nx < 4 or ny < 4:
            raise ValueError("Das Gitter braucht mindestens 4 Zellen pro Richtung.")
        if lx <= 0 or ly <= 0:
            raise ValueError("Gebietslängen müssen positiv sein.")
        self.nx = int(nx)
        self.ny = int(ny)
        self.lx = float(lx)
        self.ly = float(ly)
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny

        self.fluid = np.zeros((self.nx + 2, self.ny + 2), dtype=np.bool_)
        self.fluid[1:-1, 1:-1] = True
        self.periodic_x = False

    # ------------------------------------------------------------------ #
    # Koordinaten
    # ------------------------------------------------------------------ #
    @property
    def xc(self) -> np.ndarray:
        """x-Koordinaten der Zellmittelpunkte (Länge nx)."""
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def yc(self) -> np.ndarray:
        """y-Koordinaten der Zellmittelpunkte (Länge ny)."""
        return (np.arange(self.ny) + 0.5) * self.dy

    @property
    def xn(self) -> np.ndarray:
        """x-Koordinaten der Gitterknoten (Länge nx+1)."""
        return np.arange(self.nx + 1) * self.dx

    @property
    def yn(self) -> np.ndarray:
        """y-Koordinaten der Gitterknoten (Länge ny+1)."""
        return np.arange(self.ny + 1) * self.dy

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Meshgrid der Zellmittelpunkte, Form (nx, ny), Indizierung 'ij'."""
        return np.meshgrid(self.xc, self.yc, indexing="ij")

    # ------------------------------------------------------------------ #
    # Hindernisse
    # ------------------------------------------------------------------ #
    def interior_solid_padded(self) -> np.ndarray:
        """Festkörper-Maske mit Geisterschicht, Gebietsrand (Geisterzellen) ausgenommen."""
        s = ~self.fluid.copy()
        s[0, :] = s[-1, :] = s[:, 0] = s[:, -1] = False
        return s

    @property
    def solid(self) -> np.ndarray:
        """Festkörperzellen im Inneren des Gebiets, Form (nx, ny)."""
        return ~self.fluid[1:-1, 1:-1]

    @property
    def fluid_interior(self) -> np.ndarray:
        """Fluidzellen im Inneren des Gebiets, Form (nx, ny)."""
        return self.fluid[1:-1, 1:-1]

    def add_obstacle(self, mask: np.ndarray) -> None:
        """Markiert alle Zellen in ``mask`` (Form (nx, ny)) als Festkörper."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.nx, self.ny):
            raise ValueError(
                f"Hindernismaske hat Form {mask.shape}, erwartet {(self.nx, self.ny)}"
            )
        self.fluid[1:-1, 1:-1] &= ~mask

    def finalize(self) -> int:
        """Bereinigt die Geometrie.

        Fluidzellen, die vollständig von Festkörper umschlossen sind (also keine
        Verbindung zum Gebietsrand haben), werden zu Festkörper. Dort kann sich
        ohnehin keine Strömung ausbilden, und das Drucksystem bliebe singulär.

        Returns
        -------
        int
            Anzahl der entfernten Fluidzellen.
        """
        interior = self.fluid[1:-1, 1:-1]
        labels, n = ndimage.label(interior)
        if n <= 1:
            return 0
        border = np.zeros_like(interior)
        border[0, :] = border[-1, :] = True
        border[:, 0] = border[:, -1] = True
        keep = np.unique(labels[border & interior])
        keep = keep[keep > 0]
        enclosed = interior & ~np.isin(labels, keep)
        removed = int(enclosed.sum())
        if removed:
            self.fluid[1:-1, 1:-1] &= ~enclosed
        return removed

    def set_periodic_x(self) -> None:
        """Periodizität in x: Die Geisterspalten spiegeln die gegenüberliegenden Zellen."""
        self.periodic_x = True
        self.fluid[0, :] = self.fluid[self.nx, :]
        self.fluid[self.nx + 1, :] = self.fluid[1, :]
        self.fluid[:, 0] = False
        self.fluid[:, -1] = False

    # ------------------------------------------------------------------ #
    # Flächenklassifikation
    # ------------------------------------------------------------------ #
    def u_fluid_faces(self) -> np.ndarray:
        """u-Flächen, die zwischen zwei Fluidzellen liegen, Form (nx+1, ny+2)."""
        m = np.zeros((self.nx + 1, self.ny + 2), dtype=bool)
        m[:, :] = self.fluid[:-1, :] & self.fluid[1:, :]
        return m

    def v_fluid_faces(self) -> np.ndarray:
        """v-Flächen, die zwischen zwei Fluidzellen liegen, Form (nx+2, ny+1)."""
        m = np.zeros((self.nx + 2, self.ny + 1), dtype=bool)
        m[:, :] = self.fluid[:, :-1] & self.fluid[:, 1:]
        return m

    @property
    def n_fluid(self) -> int:
        return int(self.fluid.sum())

    def __repr__(self) -> str:  # pragma: no cover - nur Darstellung
        return (
            f"Grid(nx={self.nx}, ny={self.ny}, lx={self.lx}, ly={self.ly}, "
            f"dx={self.dx:.4g}, dy={self.dy:.4g}, Fluidzellen={self.n_fluid})"
        )
