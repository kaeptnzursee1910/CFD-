"""Erzeugung von Hindernismasken (Treppenstufen-Approximation).

Alle Funktionen liefern eine boolesche Maske der Form ``(nx, ny)``.
``True`` heißt, dass der Zellmittelpunkt im Festkörper liegt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

from .grid import Grid


def circle(grid: Grid, xc: float, yc: float, radius: float) -> np.ndarray:
    """Kreisförmiges Hindernis (z. B. Zylinder)."""
    if radius <= 0:
        raise ValueError("Radius muss positiv sein.")
    X, Y = grid.cell_centers()
    return (X - xc) ** 2 + (Y - yc) ** 2 <= radius**2


def ellipse(grid: Grid, xc: float, yc: float, a: float, b: float,
            angle_deg: float = 0.0) -> np.ndarray:
    """Elliptisches Hindernis mit Halbachsen ``a`` (x) und ``b`` (y), gedreht um ``angle_deg``."""
    X, Y = grid.cell_centers()
    phi = np.deg2rad(angle_deg)
    xr = (X - xc) * np.cos(phi) + (Y - yc) * np.sin(phi)
    yr = -(X - xc) * np.sin(phi) + (Y - yc) * np.cos(phi)
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0


def rectangle(grid: Grid, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Achsenparalleles Rechteck mit den Ecken (x0, y0) und (x1, y1)."""
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    X, Y = grid.cell_centers()
    return (X >= xa) & (X <= xb) & (Y >= ya) & (Y <= yb)


def polygon(grid: Grid, points) -> np.ndarray:
    """Beliebiges geschlossenes Polygon, gegeben als Liste von (x, y)-Punkten."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("Ein Polygon braucht mindestens 3 Punkte (x, y).")
    X, Y = grid.cell_centers()
    path = MplPath(pts)
    inside = path.contains_points(np.column_stack([X.ravel(), Y.ravel()]))
    return inside.reshape(X.shape)


def naca4_coordinates(code: str, chord: float = 1.0, n: int = 200,
                      closed_te: bool = True) -> np.ndarray:
    """Umrisspunkte eines NACA-4-Ziffern-Profils (Vorderkante im Ursprung).

    Parameters
    ----------
    code : str
        Vierstelliger NACA-Code, z. B. ``"0012"`` oder ``"2412"``.
    chord : float
        Profilsehne.
    n : int
        Anzahl Punkte pro Seite.
    closed_te : bool
        Geschlossene Hinterkante (Koeffizient -0.1036 statt -0.1015).
    """
    code = str(code).strip()
    if len(code) != 4 or not code.isdigit():
        raise ValueError("NACA-Code muss aus genau 4 Ziffern bestehen, z. B. '0012'.")
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    if t <= 0:
        raise ValueError("Die Profildicke muss größer als 0 sein.")

    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))  # Kosinus-Verteilung: feiner an den Kanten
    a4 = -0.1036 if closed_te else -0.1015
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 + a4 * x**4)

    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m > 0 and p > 0:
        front = x < p
        yc[front] = m / p**2 * (2 * p * x[front] - x[front] ** 2)
        dyc[front] = 2 * m / p**2 * (p - x[front])
        back = ~front
        yc[back] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[back] - x[back] ** 2)
        dyc[back] = 2 * m / (1 - p) ** 2 * (p - x[back])
    theta = np.arctan(dyc)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    xs = np.concatenate([xu[::-1], xl[1:]]) * chord
    ys = np.concatenate([yu[::-1], yl[1:]]) * chord
    return np.column_stack([xs, ys])


def naca4(grid: Grid, code: str, x_le: float, y_le: float, chord: float,
          angle_of_attack_deg: float = 0.0) -> np.ndarray:
    """NACA-4-Ziffern-Profil mit Vorderkante in (x_le, y_le).

    Ein positiver Anstellwinkel hebt die Nase an, das Profil wird also im
    Uhrzeigersinn um die Vorderkante gedreht.
    """
    pts = naca4_coordinates(code, chord)
    a = np.deg2rad(angle_of_attack_deg)
    xr = pts[:, 0] * np.cos(a) + pts[:, 1] * np.sin(a)
    yr = -pts[:, 0] * np.sin(a) + pts[:, 1] * np.cos(a)
    return polygon(grid, np.column_stack([xr + x_le, yr + y_le]))


def from_image(grid: Grid, path: str | Path, x0: float = 0.0, y0: float = 0.0,
               width: float | None = None, height: float | None = None,
               threshold: float = 0.5, invert: bool = False) -> np.ndarray:
    """Hindernis aus einer Bilddatei (PNG, JPG, BMP …).

    Dunkle Pixel (Helligkeit < ``threshold``) gelten als Festkörper.
    Mit ``invert=True`` ist es umgekehrt. Bei transparenten PNGs zählen
    deckende Pixel als Festkörper.
    Das Bild wird in das Rechteck ``[x0, x0+width] x [y0, y0+height]`` gelegt.
    Fehlt ``width``, wird das ganze Gebiet verwendet. Fehlt ``height``,
    bleibt das Seitenverhältnis erhalten.
    """
    from PIL import Image

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bilddatei nicht gefunden: {path}")
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = np.asarray(img.convert("RGBA"), dtype=float) / 255.0
        gray = rgba[..., :3].mean(axis=2)
        alpha = rgba[..., 3]
        solid_px = (alpha > 0.5) & (gray < threshold) if alpha.min() > 0.5 else alpha > 0.5
    else:
        gray = np.asarray(img.convert("L"), dtype=float) / 255.0
        solid_px = gray < threshold
    if invert:
        solid_px = ~solid_px

    h_px, w_px = solid_px.shape
    if width is None:
        x0, y0, width, height = 0.0, 0.0, grid.lx, grid.ly
    if height is None:
        height = width * h_px / w_px

    X, Y = grid.cell_centers()
    sx = (X - x0) / width
    sy = (Y - y0) / height
    inside = (sx >= 0) & (sx < 1) & (sy >= 0) & (sy < 1)
    # Bildzeile 0 ist oben, also wird die y-Richtung gespiegelt
    col = np.clip((sx * w_px).astype(int), 0, w_px - 1)
    row = np.clip(((1.0 - sy) * h_px).astype(int), 0, h_px - 1)
    mask = np.zeros_like(X, dtype=bool)
    mask[inside] = solid_px[row[inside], col[inside]]
    return mask


def build_obstacle(grid: Grid, spec: dict) -> np.ndarray:
    """Erzeugt eine Maske aus einer Beschreibung als Dictionary.

    Unterstützte Typen (Schlüssel ``"typ"``):

    * ``kreis``:     x, y, r
    * ``ellipse``:   x, y, a, b, [winkel]
    * ``rechteck``:  x0, y0, x1, y1
    * ``naca``:      code, x, y, sehne, [winkel]
    * ``polygon``:   punkte = [(x, y), ...]
    * ``bild``:      pfad, [x, y, breite, hoehe, schwelle, invertieren]
    """
    typ = spec.get("typ", "").lower()
    if typ == "kreis":
        return circle(grid, spec["x"], spec["y"], spec["r"])
    if typ == "ellipse":
        return ellipse(grid, spec["x"], spec["y"], spec["a"], spec["b"],
                       spec.get("winkel", 0.0))
    if typ == "rechteck":
        return rectangle(grid, spec["x0"], spec["y0"], spec["x1"], spec["y1"])
    if typ == "naca":
        return naca4(grid, spec.get("code", "0012"), spec["x"], spec["y"],
                     spec["sehne"], spec.get("winkel", 0.0))
    if typ == "polygon":
        return polygon(grid, spec["punkte"])
    if typ == "bild":
        return from_image(grid, spec["pfad"], spec.get("x", 0.0), spec.get("y", 0.0),
                          spec.get("breite"), spec.get("hoehe"),
                          spec.get("schwelle", 0.5), spec.get("invertieren", False))
    raise ValueError(f"Unbekannter Hindernistyp: {typ!r}")


def describe_obstacle(spec: dict) -> str:
    """Kurze, lesbare Beschreibung eines Hindernisses."""
    typ = spec.get("typ", "?")
    if typ == "kreis":
        return f"Kreis: Mittelpunkt ({spec['x']}, {spec['y']}), r={spec['r']}"
    if typ == "ellipse":
        return (f"Ellipse: Mittelpunkt ({spec['x']}, {spec['y']}), a={spec['a']}, "
                f"b={spec['b']}, Winkel={spec.get('winkel', 0)}°")
    if typ == "rechteck":
        return f"Rechteck: ({spec['x0']}, {spec['y0']}) bis ({spec['x1']}, {spec['y1']})"
    if typ == "naca":
        return (f"NACA {spec.get('code', '0012')}: Vorderkante ({spec['x']}, {spec['y']}), "
                f"Sehne={spec['sehne']}, Anstellwinkel={spec.get('winkel', 0)}°")
    if typ == "polygon":
        return f"Polygon mit {len(spec['punkte'])} Punkten"
    if typ == "bild":
        return f"Bild: {spec['pfad']}"
    return str(spec)
