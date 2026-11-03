"""Löser für die Druck-Poisson-Gleichung der Projektionsmethode.

Diskretisiert wird ``Lap(p) = div(F, G) / dt`` mit dem 5-Punkte-Stern über
alle Fluidzellen. Randbedingungen:

* Wand, Hindernis, Einströmung: homogene Neumann-Bedingung. Die Fläche hat
  eine vorgegebene Geschwindigkeit und erhält keine Druckkorrektur.
* Ausströmung: Dirichlet-Bedingung p = 0 auf der Randfläche, also
  Geisterwert = -p_innen.

Zusammenhangsgebiete ohne Dirichlet-Rand (z. B. die geschlossene Cavity)
sind nur bis auf eine Konstante bestimmt. Dort wird eine Zelle durch einen
zusätzlichen Diagonalterm festgehalten. Weil die rechte Seite kompatibel
ist, bleibt die Lösung exakt.

Wir lösen das (positiv definite) System ``A p = -rhs`` mit ``A = -Lap``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage

from .boundary import DomainBoundary
from .grid import Grid


class PressureSolver:
    METHODS = ("direct", "cg")

    def __init__(self, grid: Grid, boundary: DomainBoundary, method: str = "direct",
                 tol: float = 1e-10):
        if method not in self.METHODS:
            raise ValueError(f"Unbekannter Drucklöser {method!r}, erlaubt: {self.METHODS}")
        self.grid = grid
        self.method = method
        self.tol = tol
        nx, ny = grid.nx, grid.ny
        dx2 = 1.0 / grid.dx**2
        dy2 = 1.0 / grid.dy**2

        fluid = grid.fluid_interior  # (nx, ny)
        self.fluid = fluid
        n = int(fluid.sum())
        if n == 0:
            raise ValueError("Das Gebiet enthält keine Fluidzellen.")
        idx = -np.ones((nx, ny), dtype=np.int64)
        idx[fluid] = np.arange(n)
        self.idx = idx
        self.n = n

        diag = np.zeros((nx, ny))
        rows, cols, vals = [], [], []

        # Kopplungen in x-Richtung: Zelle (i, j) mit (i+1, j)
        both_x = fluid[:-1, :] & fluid[1:, :]
        a = idx[:-1, :][both_x]
        b = idx[1:, :][both_x]
        rows += [a, b]
        cols += [b, a]
        vals += [np.full(a.size, -dx2), np.full(a.size, -dx2)]
        diag[:-1, :][both_x] += dx2
        diag[1:, :][both_x] += dx2

        # Kopplungen in y-Richtung: Zelle (i, j) mit (i, j+1)
        both_y = fluid[:, :-1] & fluid[:, 1:]
        a = idx[:, :-1][both_y]
        b = idx[:, 1:][both_y]
        rows += [a, b]
        cols += [b, a]
        vals += [np.full(a.size, -dy2), np.full(a.size, -dy2)]
        diag[:, :-1][both_y] += dy2
        diag[:, 1:][both_y] += dy2

        # Dirichlet-Bedingung an Ausströmrändern: Geisterwert = -p, also +2/h^2 auf der Diagonale
        self.dirichlet = {s: boundary.outflow_mask(s) for s in ("left", "right", "bottom", "top")}
        diag[0, :][self.dirichlet["left"]] += 2 * dx2
        diag[-1, :][self.dirichlet["right"]] += 2 * dx2
        diag[:, 0][self.dirichlet["bottom"]] += 2 * dy2
        diag[:, -1][self.dirichlet["top"]] += 2 * dy2

        has_dir = np.zeros((nx, ny), dtype=bool)
        has_dir[0, :] |= self.dirichlet["left"]
        has_dir[-1, :] |= self.dirichlet["right"]
        has_dir[:, 0] |= self.dirichlet["bottom"]
        has_dir[:, -1] |= self.dirichlet["top"]

        # Zusammenhangsgebiete ohne Dirichlet-Rand: je eine Zelle festhalten
        labels, ncomp = ndimage.label(fluid)
        self.labels = labels
        self.neumann_components = []
        for c in range(1, ncomp + 1):
            comp = labels == c
            if not np.any(has_dir & comp):
                first = np.argwhere(comp)[0]
                diag[first[0], first[1]] += dx2
                self.neumann_components.append(c)

        d = diag[fluid]
        all_rows = np.concatenate(rows + [np.arange(n)])
        all_cols = np.concatenate(cols + [np.arange(n)])
        all_vals = np.concatenate(vals + [d])
        A = sp.csc_matrix((all_vals, (all_rows, all_cols)), shape=(n, n))
        self.A = A

        if method == "direct":
            self._lu = spla.splu(A, permc_spec="COLAMD")
        else:
            self._precond = sp.diags(1.0 / d)
        self._x0 = np.zeros(n)
        self.last_iterations = 0

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Löst ``Lap(p) = rhs``.

        Parameters
        ----------
        rhs : ndarray, Form (nx, ny)

        Returns
        -------
        ndarray, Form (nx, ny), mit 0 in Festkörperzellen
        """
        b = rhs[self.fluid].astype(float)
        # Kompatibilität in reinen Neumann-Gebieten erzwingen (Rundungsfehler entfernen)
        if self.neumann_components:
            lab = self.labels[self.fluid]
            for c in self.neumann_components:
                m = lab == c
                b[m] -= b[m].mean()
        b = -b
        if self.method == "direct":
            x = self._lu.solve(b)
            self.last_iterations = 1
        else:
            it = [0]

            def cb(_):
                it[0] += 1

            x, info = spla.cg(self.A, b, x0=self._x0, rtol=self.tol, maxiter=5000,
                              M=self._precond, callback=cb)
            self.last_iterations = it[0]
            if info > 0:
                import warnings
                warnings.warn(f"CG-Drucklöser nicht konvergiert ({info} Iterationen)")
            self._x0 = x
        p = np.zeros(self.fluid.shape)
        p[self.fluid] = x
        # In reinen Neumann-Gebieten den Mittelwert abziehen (Druck nur bis auf Konstante bestimmt)
        if self.neumann_components:
            for c in self.neumann_components:
                m = self.labels == c
                p[m] -= p[m].mean()
        return p
