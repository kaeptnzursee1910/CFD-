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


# Begrenzer (Limiter) für TVD-Schemata, Nummerierung wie in scalar_transport
LIMITERS = {"upwind": 0, "vanleer": 1, "minmod": 2, "superbee": 3, "linear": 4, "mc": 5}
LIMITER_NAMES_DE = {
    "upwind": "Upwind (1. Ordnung, beschränkt)",
    "vanleer": "van Leer (TVD, 2. Ordnung)",
    "minmod": "Minmod (TVD, sehr robust)",
    "superbee": "Superbee (TVD, kantenscharf)",
    "mc": "MC / monotonized central (TVD)",
    "linear": "linear / zentral (2. Ordnung, unbeschränkt)",
}


@njit(cache=True)
def momentum_tendencies(u, v, fluid, nx, ny, dx, dy, nu, variable, gamma, periodic, hu, hv):
    """Explizite Terme der Impulsgleichungen (Konvektion und Diffusion).

    Berechnet auf allen Fluidflächen ``hu = Diffusion(u) - d(uu)/dx - d(uv)/dy``
    und ``hv`` entsprechend für v.

    Diffusion
        ``variable = False``: ``nu * Lap(u)`` mit konstanter Viskosität ``nu[0, 0]``.
        ``variable = True``: Spannungsform ``div(nu_eff (grad u + grad u^T))`` mit
        dem Zellfeld ``nu`` (z. B. molekulare plus turbulente Viskosität).

    Konvektion
        Mischung aus zentralen Differenzen und Donor-Cell-Upwind,
        ``gamma`` in [0, 1]: 0 = zentral, 1 = Upwind.

    periodic
        Periodizität in x: u-Fläche ``nx`` ist identisch mit Fläche 0, die
        Geisterspalten 0 und nx+1 entsprechen den Zellen nx und 1.
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

    imax = nx if periodic else nx - 1
    nu0 = nu[1, 1]

    # ---------------- u-Impuls ----------------
    for i in range(1, imax + 1):
        ip = i + 1 if i < nx else 1  # Index der östlichen u-Fläche (periodisch umgebrochen)
        for j in range(1, ny + 1):
            if not (fluid[i, j] and fluid[i + 1, j]):
                continue
            uc = u[i, j]
            ue = u[ip, j]
            uw = u[i - 1, j]
            un = u[i, j + 1]
            us = u[i, j - 1]

            if variable:
                nu_e = nu[i + 1, j]
                nu_w = nu[i, j]
                nu_n = 0.25 * (nu[i, j] + nu[i + 1, j] + nu[i, j + 1] + nu[i + 1, j + 1])
                nu_s = 0.25 * (nu[i, j] + nu[i + 1, j] + nu[i, j - 1] + nu[i + 1, j - 1])
                diff = (2.0 * nu_e * (ue - uc) - 2.0 * nu_w * (uc - uw)) * idx2 + idy * (
                    nu_n * ((un - uc) * idy + (v[i + 1, j] - v[i, j]) * idx)
                    - nu_s * ((uc - us) * idy + (v[i + 1, j - 1] - v[i, j - 1]) * idx))
            else:
                diff = nu0 * ((ue - 2.0 * uc + uw) * idx2 + (un - 2.0 * uc + us) * idy2)

            a_e = 0.5 * (uc + ue)
            a_w = 0.5 * (uw + uc)
            du2dx = idx * (a_e * a_e - a_w * a_w) + gamma * idx * (
                abs(a_e) * 0.5 * (uc - ue) - abs(a_w) * 0.5 * (uw - uc))

            v_n = 0.5 * (v[i, j] + v[i + 1, j])
            v_s = 0.5 * (v[i, j - 1] + v[i + 1, j - 1])
            duvdy = idy * (v_n * 0.5 * (uc + un) - v_s * 0.5 * (us + uc)) + gamma * idy * (
                abs(v_n) * 0.5 * (uc - un) - abs(v_s) * 0.5 * (us - uc))

            hu[i, j] = diff - du2dx - duvdy

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

            if variable:
                nu_n = nu[i, j + 1]
                nu_s = nu[i, j]
                nu_e = 0.25 * (nu[i, j] + nu[i + 1, j] + nu[i, j + 1] + nu[i + 1, j + 1])
                nu_w = 0.25 * (nu[i, j] + nu[i - 1, j] + nu[i, j + 1] + nu[i - 1, j + 1])
                diff = (2.0 * nu_n * (vn - vc) - 2.0 * nu_s * (vc - vs)) * idy2 + idx * (
                    nu_e * ((ve - vc) * idx + (u[i, j + 1] - u[i, j]) * idy)
                    - nu_w * ((vc - vw) * idx + (u[i - 1, j + 1] - u[i - 1, j]) * idy))
            else:
                diff = nu0 * ((ve - 2.0 * vc + vw) * idx2 + (vn - 2.0 * vc + vs) * idy2)

            b_n = 0.5 * (vc + vn)
            b_s = 0.5 * (vs + vc)
            dv2dy = idy * (b_n * b_n - b_s * b_s) + gamma * idy * (
                abs(b_n) * 0.5 * (vc - vn) - abs(b_s) * 0.5 * (vs - vc))

            u_e = 0.5 * (u[i, j] + u[i, j + 1])
            u_w = 0.5 * (u[i - 1, j] + u[i - 1, j + 1])
            duvdx = idx * (u_e * 0.5 * (vc + ve) - u_w * 0.5 * (vw + vc)) + gamma * idx * (
                abs(u_e) * 0.5 * (vc - ve) - abs(u_w) * 0.5 * (vw - vc))

            hv[i, j] = diff - duvdx - dv2dy


@njit(cache=True)
def obstacle_bc(u, v, fluid, nx, ny, periodic):
    """Haftbedingung an inneren Hindernissen (Treppenstufen-Rand).

    * Normalkomponente: Eine Fläche zwischen einer Fluid- und einer
      Festkörperzelle bekommt die Geschwindigkeit 0.
    * Tangentialkomponente: Eine Fläche im Festkörper, die an eine
      Fluidfläche grenzt, bekommt den gespiegelten Wert, damit die
      Geschwindigkeit an der Wand im Mittel 0 ist. Grenzen zwei
      Fluidflächen an (dünne Hindernisse), wird gemittelt.
    """
    imax = nx if periodic else nx - 1
    for i in range(1, imax + 1):
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
    if periodic:
        for j in range(ny + 2):
            u[0, j] = u[nx, j]

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


# ---------------------------------------------------------------------- #
# Skalartransport (Temperatur, Konzentration, k, omega, Volumenanteil ...)
# ---------------------------------------------------------------------- #
@njit(cache=True)
def _limiter(r, kind):
    if kind == 0:
        return 0.0
    if kind == 1:  # van Leer
        if r <= 0.0:
            return 0.0
        return 2.0 * r / (1.0 + r)
    if kind == 2:  # minmod
        if r <= 0.0:
            return 0.0
        return min(r, 1.0)
    if kind == 3:  # superbee
        if r <= 0.0:
            return 0.0
        return max(min(2.0 * r, 1.0), min(r, 2.0))
    if kind == 5:  # MC
        if r <= 0.0:
            return 0.0
        return min(2.0 * r, min(0.5 * (1.0 + r), 2.0))
    return 1.0  # linear


@njit(cache=True)
def _face_value(phi_u, phi_c, phi_d, has_u, kind):
    """TVD-Flächenwert zwischen Zelle C (stromauf) und D (stromab), U = weiter stromauf."""
    if kind == 0:
        return phi_c
    if kind == 4:
        return 0.5 * (phi_c + phi_d)
    if not has_u:
        return phi_c
    den = phi_d - phi_c
    if abs(den) < 1e-300:
        return phi_c
    r = (phi_c - phi_u) / den
    return phi_c + 0.5 * _limiter(r, kind) * den


@njit(cache=True)
def scalar_transport(phi, u, v, fluid, nx, ny, dx, dy, diff, limiter, periodic,
                     bt_l, bv_l, bt_r, bv_r, bt_b, bv_b, bt_t, bv_t, obs_t, obs_v, out):
    """Konvektions- und Diffusionsterm einer skalaren Transportgleichung.

    Berechnet für alle Fluidzellen ``out = -div(u phi) + div(diff grad phi)``
    in konservativer Finite-Volumen-Form.

    Randbedingungen je Randfläche (Typ ``bt_*``, Wert ``bv_*``):
        0 = Gradient null, 1 = fester Wert, 2 = fester Fluss (in das Gebiet hinein)
    Konvektiv wird an Rändern mit Einströmung der Randwert (bei festem Wert)
    verwendet, sonst der Zellwert (Upwind). Hindernisse: ``obs_t`` 0 = adiabat,
    1 = fester Wert ``obs_v``.

    ``diff`` ist ein Zellfeld der Diffusivität (Form wie phi); an inneren
    Flächen wird es harmonisch gemittelt.
    """
    for i in range(nx + 2):
        for j in range(ny + 2):
            out[i, j] = 0.0

    # ---------------- x-Flächen ----------------
    for i in range(0, nx + 1):
        for j in range(1, ny + 1):
            if i == 0 or (i == nx and not periodic):
                # Gebietsrand
                if i == 0:
                    if periodic:
                        continue
                    P = 1
                    sgn = 1.0  # Fluss in +x führt in das Gebiet
                    t = bt_l[j - 1]
                    val = bv_l[j - 1]
                else:
                    P = nx
                    sgn = -1.0
                    t = bt_r[j - 1]
                    val = bv_r[j - 1]
                if not fluid[P, j]:
                    continue
                un = sgn * u[i, j]  # Geschwindigkeit in das Gebiet hinein
                phb = val if (t == 1 and un > 0.0) else phi[P, j]
                flux_in = un * phb
                if t == 1:
                    flux_in += diff[P, j] * (val - phi[P, j]) / (0.5 * dx)
                elif t == 2:
                    flux_in += val
                out[P, j] += flux_in / dx
                continue

            L = i
            R = i + 1 if i < nx else 1
            fL = fluid[L, j]
            fR = fluid[R, j]
            if fL and fR:
                uf = u[i, j]
                if uf >= 0.0:
                    C = L
                    D = R
                    Ui = L - 1
                    if Ui < 1:
                        Ui = nx if periodic else -1
                else:
                    C = R
                    D = L
                    Ui = R + 1
                    if Ui > nx:
                        Ui = 1 if periodic else -1
                has_u = Ui >= 1 and fluid[Ui, j]
                phu = phi[Ui, j] if has_u else 0.0
                phf = _face_value(phu, phi[C, j], phi[D, j], has_u, limiter)
                dl = diff[L, j]
                dr = diff[R, j]
                df = 2.0 * dl * dr / (dl + dr) if dl + dr > 0.0 else 0.0
                F = uf * phf - df * (phi[R, j] - phi[L, j]) / dx
                out[L, j] -= F / dx
                out[R, j] += F / dx
            elif (fL or fR) and obs_t == 1:
                P = L if fL else R
                out[P, j] += diff[P, j] * (obs_v - phi[P, j]) / (0.5 * dx) / dx

    # ---------------- y-Flächen ----------------
    for i in range(1, nx + 1):
        for j in range(0, ny + 1):
            if j == 0 or j == ny:
                if j == 0:
                    P = 1
                    sgn = 1.0
                    t = bt_b[i - 1]
                    val = bv_b[i - 1]
                else:
                    P = ny
                    sgn = -1.0
                    t = bt_t[i - 1]
                    val = bv_t[i - 1]
                if not fluid[i, P]:
                    continue
                vn = sgn * v[i, j]
                phb = val if (t == 1 and vn > 0.0) else phi[i, P]
                flux_in = vn * phb
                if t == 1:
                    flux_in += diff[i, P] * (val - phi[i, P]) / (0.5 * dy)
                elif t == 2:
                    flux_in += val
                out[i, P] += flux_in / dy
                continue

            B = j
            T = j + 1
            fB = fluid[i, B]
            fT = fluid[i, T]
            if fB and fT:
                vf = v[i, j]
                if vf >= 0.0:
                    C = B
                    D = T
                    Uj = B - 1
                else:
                    C = T
                    D = B
                    Uj = T + 1
                has_u = 1 <= Uj <= ny and fluid[i, Uj]
                phu = phi[i, Uj] if has_u else 0.0
                phf = _face_value(phu, phi[i, C], phi[i, D], has_u, limiter)
                db = diff[i, B]
                dt_ = diff[i, T]
                df = 2.0 * db * dt_ / (db + dt_) if db + dt_ > 0.0 else 0.0
                F = vf * phf - df * (phi[i, T] - phi[i, B]) / dy
                out[i, B] -= F / dy
                out[i, T] += F / dy
            elif (fB or fT) and obs_t == 1:
                P = B if fB else T
                out[i, P] += diff[i, P] * (obs_v - phi[i, P]) / (0.5 * dy) / dy

    for i in range(nx + 2):
        for j in range(ny + 2):
            if not fluid[i, j] or i == 0 or i == nx + 1 or j == 0 or j == ny + 1:
                out[i, j] = 0.0


def warmup() -> None:
    """Kompiliert die Kernel einmal vorab (dauert beim ersten Mal einige Sekunden)."""
    nx = ny = 4
    u = np.zeros((nx + 1, ny + 2))
    v = np.zeros((nx + 2, ny + 1))
    fl = np.zeros((nx + 2, ny + 2), dtype=np.bool_)
    fl[1:-1, 1:-1] = True
    hu = np.zeros_like(u)
    hv = np.zeros_like(v)
    nu = np.full((nx + 2, ny + 2), 0.01)
    momentum_tendencies(u, v, fl, nx, ny, 0.1, 0.1, nu, False, 0.5, False, hu, hv)
    obstacle_bc(u, v, fl, nx, ny, False)
    phi = np.zeros((nx + 2, ny + 2))
    out = np.zeros_like(phi)
    ti = np.zeros(ny, dtype=np.int64)
    tv = np.zeros(ny)
    scalar_transport(phi, u, v, fl, nx, ny, 0.1, 0.1, nu, 1, False,
                     ti, tv, ti, tv, ti, tv, ti, tv, 0, 0.0, out)
