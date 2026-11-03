"""Rechenintensive Kernel, mit Numba JIT-kompiliert.

Ist Numba nicht installiert, laufen die Funktionen als reines Python.
Das ist korrekt, aber sehr langsam.
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - nur ohne Numba
    HAVE_NUMBA = False
    warnings.warn("Numba ist nicht installiert: Die Simulation läuft sehr langsam. "
                  "Installation: pip install numba", RuntimeWarning, stacklevel=2)

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def deco(f):
            return f

        return deco


@njit(cache=True)
def momentum_tendencies(u, v, fluid, nx, ny, dx, dy, nu, gamma, hu, hv):
    """Explizite Terme der Impulsgleichungen (Konvektion und Diffusion).

    Berechnet ``hu = nu*Lap(u) - d(uu)/dx - d(uv)/dy`` auf allen u-Flächen
    zwischen zwei Fluidzellen und ``hv`` entsprechend für v.

    Die Konvektion wird als Mischung aus zentralen Differenzen und dem
    Donor-Cell-Schema (Upwind) diskretisiert. Der Mischungsfaktor
    ``gamma`` liegt in [0, 1]: 0 heißt rein zentral (2. Ordnung),
    1 heißt rein Upwind (1. Ordnung, stark dämpfend).
    """
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    idx = 1.0 / dx
    idy = 1.0 / dy

    for i in range(nx + 1):
        for j in range(ny + 2):
            hu[i, j] = 0.0
    for i in range(nx + 2):
        for j in range(ny + 1):
            hv[i, j] = 0.0

    # ---------------- u-Impuls ----------------
    for i in range(1, nx):
        for j in range(1, ny + 1):
            if not (fluid[i, j] and fluid[i + 1, j]):
                continue
            uc = u[i, j]
            ue = u[i + 1, j]
            uw = u[i - 1, j]
            un = u[i, j + 1]
            us = u[i, j - 1]

            lap = (ue - 2.0 * uc + uw) * idx2 + (un - 2.0 * uc + us) * idy2

            # d(u^2)/dx
            a_e = 0.5 * (uc + ue)
            a_w = 0.5 * (uw + uc)
            du2dx = idx * (a_e * a_e - a_w * a_w) + gamma * idx * (
                abs(a_e) * 0.5 * (uc - ue) - abs(a_w) * 0.5 * (uw - uc))

            # d(uv)/dy
            v_n = 0.5 * (v[i, j] + v[i + 1, j])
            v_s = 0.5 * (v[i, j - 1] + v[i + 1, j - 1])
            duvdy = idy * (v_n * 0.5 * (uc + un) - v_s * 0.5 * (us + uc)) + gamma * idy * (
                abs(v_n) * 0.5 * (uc - un) - abs(v_s) * 0.5 * (us - uc))

            hu[i, j] = nu * lap - du2dx - duvdy

    # ---------------- v-Impuls ----------------
    for i in range(1, nx + 1):
        for j in range(1, ny):
            if not (fluid[i, j] and fluid[i, j + 1]):
                continue
            vc = v[i, j]
            ve = v[i + 1, j]
            vw = v[i - 1, j]
            vn = v[i, j + 1]
            vs = v[i, j - 1]

            lap = (ve - 2.0 * vc + vw) * idx2 + (vn - 2.0 * vc + vs) * idy2

            # d(v^2)/dy
            b_n = 0.5 * (vc + vn)
            b_s = 0.5 * (vs + vc)
            dv2dy = idy * (b_n * b_n - b_s * b_s) + gamma * idy * (
                abs(b_n) * 0.5 * (vc - vn) - abs(b_s) * 0.5 * (vs - vc))

            # d(uv)/dx
            u_e = 0.5 * (u[i, j] + u[i, j + 1])
            u_w = 0.5 * (u[i - 1, j] + u[i - 1, j + 1])
            duvdx = idx * (u_e * 0.5 * (vc + ve) - u_w * 0.5 * (vw + vc)) + gamma * idx * (
                abs(u_e) * 0.5 * (vc - ve) - abs(u_w) * 0.5 * (vw - vc))

            hv[i, j] = nu * lap - duvdx - dv2dy


@njit(cache=True)
def obstacle_bc(u, v, fluid, nx, ny):
    """Haftbedingung an inneren Hindernissen (Treppenstufen-Rand).

    * Normalkomponente: Eine Fläche zwischen einer Fluid- und einer
      Festkörperzelle bekommt die Geschwindigkeit 0.
    * Tangentialkomponente: Eine Fläche im Festkörper, die an eine
      Fluidfläche grenzt, bekommt den gespiegelten Wert, damit die
      Geschwindigkeit an der Wand im Mittel 0 ist. Grenzen zwei
      Fluidflächen an (dünne Hindernisse), wird gemittelt.
    """
    for i in range(1, nx):
        for j in range(1, ny + 1):
            fl = fluid[i, j]
            fr = fluid[i + 1, j]
            if fl and fr:
                continue
            if fl or fr:
                u[i, j] = 0.0
                continue
            s = 0.0
            c = 0
            if fluid[i, j + 1] and fluid[i + 1, j + 1]:
                s += u[i, j + 1]
                c += 1
            if fluid[i, j - 1] and fluid[i + 1, j - 1]:
                s += u[i, j - 1]
                c += 1
            u[i, j] = -s / c if c > 0 else 0.0

    for i in range(1, nx + 1):
        for j in range(1, ny):
            fb = fluid[i, j]
            ft = fluid[i, j + 1]
            if fb and ft:
                continue
            if fb or ft:
                v[i, j] = 0.0
                continue
            s = 0.0
            c = 0
            if fluid[i + 1, j] and fluid[i + 1, j + 1]:
                s += v[i + 1, j]
                c += 1
            if fluid[i - 1, j] and fluid[i - 1, j + 1]:
                s += v[i - 1, j]
                c += 1
            v[i, j] = -s / c if c > 0 else 0.0


def warmup() -> None:
    """Kompiliert die Kernel einmal vorab (dauert beim ersten Mal einige Sekunden)."""
    nx = ny = 4
    u = np.zeros((nx + 1, ny + 2))
    v = np.zeros((nx + 2, ny + 1))
    fl = np.zeros((nx + 2, ny + 2), dtype=np.bool_)
    fl[1:-1, 1:-1] = True
    hu = np.zeros_like(u)
    hv = np.zeros_like(v)
    momentum_tendencies(u, v, fl, nx, ny, 0.1, 0.1, 0.01, 0.5, hu, hv)
    obstacle_bc(u, v, fl, nx, ny)
