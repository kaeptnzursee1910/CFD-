"""Referenzdaten und Kennzahlen zur Validierung."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------- #
# Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387-411
# Tabelle I: u entlang der vertikalen Mittellinie x = 0.5
# Tabelle II: v entlang der horizontalen Mittellinie y = 0.5
# Hinweis: Der Tabellenwert v = -0.23827 bei x = 0.9063 (Re = 400) passt nicht zu seinen
# Nachbarpunkten (-0.44993 bei 0.8594, -0.22847 bei 0.9453) und ist vermutlich ein
# Übertragungsfehler. Er dominiert die maximale v-Abweichung bei Re = 400.
# ---------------------------------------------------------------------- #
GHIA_Y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
                   0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
GHIA_U = {
    100: [1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581,
          -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.00000],
    400: [1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135, -0.11477,
          -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.00000],
    1000: [1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702, -0.06080,
           -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.00000],
}
GHIA_X = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000,
                   0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
GHIA_V = {
    100: [0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533,
          0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.00000],
    400: [0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, -0.38598,
          0.05186, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 0.19713, 0.18360, 0.00000],
    1000: [0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, -0.31966,
           0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012, 0.27485, 0.00000],
}


def ghia_reference(re: float) -> tuple[int, dict] | None:
    """Liefert die Ghia-Daten, falls Re (±5 %) einem tabellierten Wert entspricht."""
    for r in GHIA_U:
        if abs(re - r) / r < 0.05:
            return r, {"y": GHIA_Y, "u": np.array(GHIA_U[r]),
                       "x": GHIA_X, "v": np.array(GHIA_V[r])}
    return None


def cavity_centerline_errors(sim, re: float) -> dict | None:
    """Maximale und RMS-Abweichung der Mittellinienprofile von Ghia et al."""
    ref = ghia_reference(re)
    if ref is None:
        return None
    r, d = ref
    u = sim.sample_u(np.full_like(d["y"], 0.5), d["y"])
    v = sim.sample_v(d["x"], np.full_like(d["x"], 0.5))
    eu = u - d["u"]
    ev = v - d["v"]
    return {"re_ref": r,
            "u_max_fehler": float(np.abs(eu).max()), "u_rms_fehler": float(np.sqrt(np.mean(eu**2))),
            "v_max_fehler": float(np.abs(ev).max()), "v_rms_fehler": float(np.sqrt(np.mean(ev**2)))}


# ---------------------------------------------------------------------- #
# Poiseuille
# ---------------------------------------------------------------------- #
def poiseuille_profile(y: np.ndarray, H: float = 1.0, U_mean: float = 1.0) -> np.ndarray:
    """Analytisches Geschwindigkeitsprofil der ausgebildeten Kanalströmung."""
    return 6.0 * U_mean * y * (H - y) / H**2


def poiseuille_pressure_gradient(nu: float, H: float = 1.0, U_mean: float = 1.0) -> float:
    return -12.0 * nu * U_mean / H**2


# ---------------------------------------------------------------------- #
# Kraftbeiwerte und Strouhal-Zahl
# ---------------------------------------------------------------------- #
def force_coefficients(fx, fy, U: float, L: float, rho: float = 1.0):
    q = 0.5 * rho * U**2 * L
    return np.asarray(fx) / q, np.asarray(fy) / q


def dominant_frequency(t: np.ndarray, signal: np.ndarray, t_start: float | None = None):
    """Dominante Frequenz eines Signals per FFT (mit Parabelinterpolation des Maximums).

    Das Signal wird vorher auf ein äquidistantes Zeitraster interpoliert, weil
    der Zeitschritt adaptiv ist.

    Returns
    -------
    (frequenz, amplitude, freqs, spektrum) oder None, wenn das Signal zu kurz ist
    """
    t = np.asarray(t)
    s = np.asarray(signal)
    if t_start is not None:
        m = t >= t_start
        t, s = t[m], s[m]
    if len(t) < 64 or t[-1] - t[0] <= 0:
        return None
    n = int(2 ** np.ceil(np.log2(len(t))))
    tu = np.linspace(t[0], t[-1], n)
    su = np.interp(tu, t, s)
    su = su - su.mean()
    if np.allclose(su, 0):
        return None
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(su * win))
    freqs = np.fft.rfftfreq(n, tu[1] - tu[0])
    k = int(np.argmax(spec[1:]) + 1)
    f = freqs[k]
    if 1 <= k < len(spec) - 1:
        a, b, c = np.log(spec[k - 1:k + 2] + 1e-300)
        denom = a - 2 * b + c
        if denom != 0:
            f = freqs[k] + 0.5 * (a - c) / denom * (freqs[1] - freqs[0])
    return float(f), float(spec[k]), freqs, spec


def strouhal_number(t, cl, D: float, U: float, t_start: float | None = None) -> float | None:
    """Strouhal-Zahl St = f D / U aus dem Auftriebssignal."""
    res = dominant_frequency(t, cl, t_start)
    if res is None:
        return None
    f, amp, *_ = res
    # nur sinnvoll, wenn es tatsächlich eine Schwingung gibt
    if np.ptp(np.asarray(cl)[np.asarray(t) >= (t_start or 0)]) < 1e-3:
        return None
    return f * D / U


# ---------------------------------------------------------------------- #
# Rückwärtsgewandte Stufe
# ---------------------------------------------------------------------- #
def reattachment_length(sim, x_step: float, wall: str = "bottom") -> float | None:
    """Wiederanlegelänge hinter der Stufe, gemessen ab der Stufenkante.

    Gesucht wird der erste Vorzeichenwechsel der wandnahen Geschwindigkeit u
    von negativ (Rückströmung) nach positiv, linear interpoliert.
    """
    g = sim.grid
    j = 1 if wall == "bottom" else g.ny
    x = np.arange(g.nx + 1) * g.dx
    u = sim.u[:, j]
    m = x > x_step + 0.5 * g.dx
    xs, us = x[m], u[m]
    neg = np.where(us < 0)[0]
    if len(neg) == 0:
        return None
    k0 = neg[0]
    for k in range(k0, len(us) - 1):
        if us[k] < 0 <= us[k + 1]:
            xr = xs[k] - us[k] * (xs[k + 1] - xs[k]) / (us[k + 1] - us[k])
            return float(xr - x_step)
    return None


def separation_bubbles(sim, wall: str = "top", x_min: float = 0.0) -> list[tuple[float, float]]:
    """Alle Rückströmgebiete (Ablösung, Wiederanlegung) entlang einer Wand."""
    g = sim.grid
    j = 1 if wall == "bottom" else g.ny
    x = np.arange(g.nx + 1) * g.dx
    u = sim.u[:, j]
    out = []
    inside = False
    xs = None
    for k in range(1, g.nx):
        if x[k] < x_min or not (g.fluid[k, j] and g.fluid[k + 1, j]):
            continue
        if not inside and u[k] < 0:
            inside, xs = True, x[k]
        elif inside and u[k] >= 0:
            inside = False
            out.append((float(xs), float(x[k])))
    if inside:
        out.append((float(xs), float(x[g.nx - 1])))
    return out
