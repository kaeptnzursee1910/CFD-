"""Interaktive Bedienung über die Konsole (Eingabe von Parametern)."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from . import cases, geometry
from .boundary import KIND_NAMES_DE, SIDE_NAMES_DE, SIDES

BANNER = r"""
 ╔══════════════════════════════════════════════════════════════════╗
 ║          2D-CFD: Strömung, Wärme, Turbulenz, Zweiphasen          ║
 ║   Navier-Stokes · Finite Volumen · MAC-Gitter · Projektion       ║
 ╚══════════════════════════════════════════════════════════════════╝
   Eingabe leer lassen = Standardwert [in Klammern] übernehmen
   Strg+C während der Simulation = Abbruch mit Auswertung
"""


class Abort(Exception):
    """Der Nutzer hat 'q' eingegeben."""


# ---------------------------------------------------------------------- #
# Eingabehilfen
# ---------------------------------------------------------------------- #
def _raw(prompt: str) -> str:
    s = input(prompt).strip()
    if s.lower() in ("q", "quit", "exit", "beenden"):
        raise Abort()
    return s


def _to_float(s: str) -> float:
    return float(s.replace(",", "."))


def ask_float(prompt: str, default: float | None, lo: float | None = None,
              hi: float | None = None, allow_none: bool = False) -> float | None:
    dtxt = "" if default is None else f" [{default:g}]"
    while True:
        s = _raw(f"{prompt}{dtxt}: ")
        if s == "":
            if default is None and not allow_none:
                print("  Bitte einen Wert eingeben.")
                continue
            return default
        if allow_none and s.lower() in ("-", "keine", "none", "aus"):
            return None
        try:
            x = _to_float(s)
        except ValueError:
            print("  Ungültige Zahl, bitte erneut eingeben.")
            continue
        if lo is not None and x < lo:
            print(f"  Wert muss mindestens {lo:g} sein.")
            continue
        if hi is not None and x > hi:
            print(f"  Wert darf höchstens {hi:g} sein.")
            continue
        return x


def ask_int(prompt: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
    while True:
        x = ask_float(prompt, default, lo, hi)
        if float(x).is_integer():
            return int(x)
        print("  Bitte eine ganze Zahl eingeben.")


def ask_bool(prompt: str, default: bool) -> bool:
    d = "J/n" if default else "j/N"
    while True:
        s = _raw(f"{prompt} [{d}]: ").lower()
        if s == "":
            return default
        if s in ("j", "ja", "y", "yes", "1"):
            return True
        if s in ("n", "nein", "no", "0"):
            return False
        print("  Bitte j oder n eingeben.")


def ask_choice(prompt: str, options: list[tuple[str, str]], default: int = 1) -> str:
    """Auswahl aus nummerierten Optionen. Liefert den Schlüssel der gewählten Option."""
    print(prompt)
    for k, (_, label) in enumerate(options, 1):
        print(f"  {k}) {label}")
    while True:
        n = ask_int("Auswahl", default, 1, len(options))
        return options[n - 1][0]


def ask_str(prompt: str, default: str | None = None) -> str:
    dtxt = f" [{default}]" if default else ""
    while True:
        s = _raw(f"{prompt}{dtxt}: ")
        if s == "" and default is not None:
            return default
        if s:
            return s
        print("  Bitte einen Wert eingeben.")


def ask_points(prompt: str, min_points: int = 3) -> list[tuple[float, float]]:
    print(f"{prompt} (Format: x1 y1; x2 y2; x3 y3 ...)")
    while True:
        s = _raw("Punkte: ")
        try:
            pts = []
            for part in s.split(";"):
                part = part.strip()
                if not part:
                    continue
                xy = part.replace(",", " ").split() if part.count(",") == 1 and " " not in part \
                    else part.split()
                pts.append((_to_float(xy[0]), _to_float(xy[1])))
            if len(pts) >= min_points:
                return pts
            print(f"  Mindestens {min_points} Punkt(e) nötig.")
        except (ValueError, IndexError):
            print("  Ungültiges Format. Beispiel: 1 0.4; 1.5 0.5; 1 0.6")




def ask_obstacle(lx: float, ly: float) -> dict | None:
    typ = ask_choice("\nHindernis hinzufügen:", [
        ("kreis", "Kreis / Zylinder"),
        ("rechteck", "Rechteck"),
        ("ellipse", "Ellipse"),
        ("naca", "NACA-4-Ziffern-Profil (Tragflügel)"),
        ("polygon", "Polygon (eigene Eckpunkte)"),
        ("bild", "aus Bilddatei (PNG/JPG, dunkle Pixel = Festkörper)"),
        ("fertig", "fertig, keine weiteren Hindernisse"),
    ], default=7)
    if typ == "fertig":
        return None
    if typ == "kreis":
        return {"typ": "kreis", "x": ask_float("  Mittelpunkt x", lx / 4),
                "y": ask_float("  Mittelpunkt y", ly / 2),
                "r": ask_float("  Radius r", ly / 10, lo=1e-9)}
    if typ == "rechteck":
        x0 = ask_float("  linke untere Ecke x0", lx / 4 - ly / 10)
        y0 = ask_float("  linke untere Ecke y0", 0.4 * ly)
        x1 = ask_float("  rechte obere Ecke x1", lx / 4 + ly / 10)
        y1 = ask_float("  rechte obere Ecke y1", 0.6 * ly)
        return {"typ": "rechteck", "x0": x0, "y0": y0, "x1": x1, "y1": y1}
    if typ == "ellipse":
        return {"typ": "ellipse", "x": ask_float("  Mittelpunkt x", lx / 4),
                "y": ask_float("  Mittelpunkt y", ly / 2),
                "a": ask_float("  Halbachse a (x)", ly / 8, lo=1e-9),
                "b": ask_float("  Halbachse b (y)", ly / 16, lo=1e-9),
                "winkel": ask_float("  Drehwinkel in Grad", 0.0)}
    if typ == "naca":
        while True:
            code = ask_str("  NACA-Code (4 Ziffern)", "0012")
            if len(code) == 4 and code.isdigit() and int(code[2:]) > 0:
                break
            print("  Ungültiger Code, Beispiel: 0012, 2412, 4415")
        return {"typ": "naca", "code": code,
                "x": ask_float("  Vorderkante x", lx / 4),
                "y": ask_float("  Vorderkante y", ly / 2),
                "sehne": ask_float("  Sehnenlänge", min(lx / 4, ly / 2), lo=1e-9),
                "winkel": ask_float("  Anstellwinkel in Grad", 5.0, -90, 90)}
    if typ == "polygon":
        return {"typ": "polygon", "punkte": ask_points("  Eckpunkte des Polygons")}
    if typ == "bild":
        while True:
            pfad = ask_str("  Pfad zur Bilddatei")
            if Path(pfad).expanduser().exists():
                break
            print("  Datei nicht gefunden.")
        spec = {"typ": "bild", "pfad": str(Path(pfad).expanduser())}
        if ask_bool("  Bild über das ganze Gebiet strecken?", False):
            return spec
        spec["x"] = ask_float("  linke untere Ecke x", lx / 4 - ly / 4)
        spec["y"] = ask_float("  linke untere Ecke y", ly / 4)
        spec["breite"] = ask_float("  Breite", ly / 2, lo=1e-9)
        spec["schwelle"] = ask_float("  Helligkeitsschwelle (0..1)", 0.5, 0, 1)
        spec["invertieren"] = ask_bool("  Invertieren (helle Pixel = Festkörper)?", False)
        return spec
    return None


def ask_boundaries(U: float) -> dict:
    """Randbedingungen aller vier Seiten (als speicherbare Dictionaries)."""
    print("\nRandbedingungen festlegen:")
    opts = [(k, KIND_NAMES_DE[k]) for k in ("noslip", "slip", "inflow", "outflow", "periodic")]
    defaults = {"left": 3, "right": 4, "bottom": 2, "top": 2}
    bcs = {}
    for side in SIDES:
        if side == "right" and bcs["left"]["kind"] == "periodic":
            bcs["right"] = {"kind": "periodic"}
            print(" Rand rechts: periodisch (wie links)")
            continue
        o = opts if side == "left" else [x for x in opts if x[0] != "periodic"]
        kind = ask_choice(f" Rand {SIDE_NAMES_DE[side]}:", o, defaults[side])
        if kind == "inflow":
            vel = ask_float("   mittlere Einströmgeschwindigkeit", U, lo=0)
            prof = ask_choice("   Profil:", [("uniform", "Blockprofil"),
                                             ("parabolic", "parabolisch")], 1)
            bcs[side] = {"kind": "inflow", "velocity": vel, "profile": prof}
        elif kind == "noslip":
            vel = ask_float("   tangentiale Wandgeschwindigkeit", 0.0)
            bcs[side] = {"kind": "noslip", "velocity": vel}
        else:
            bcs[side] = {"kind": kind}
    kinds = [b["kind"] for b in bcs.values()]
    if "inflow" in kinds and "outflow" not in kinds:
        print("  Warnung: Es gibt eine Einströmung, aber keine Ausströmung. Die Masse kann nicht"
              " abfließen, die Lösung wird unphysikalisch.")
    return bcs


# ---------------------------------------------------------------------- #
# Fallspezifische Abfragen: liefern eine Case-Spezifikation (JSON-fähig)
# ---------------------------------------------------------------------- #
def _steady_tol(default: float) -> float | None:
    tol = ask_float("Abbruch bei stationärer Lösung: Residuum-Grenze (- = aus)", default,
                    lo=0.0, allow_none=True)
    return tol if tol else None


def setup_cavity() -> dict:
    print("\n--- Lid-Driven Cavity ---")
    print("Quadratische Box (1 x 1), Deckel bewegt sich mit U = 1 nach rechts.")
    print("Ghia-Referenzdaten gibt es für Re = 100, 400 und 1000.")
    re = ask_float("Reynolds-Zahl Re", 100, lo=1)
    n = ask_int("Zellen pro Richtung", 64 if re <= 400 else 128, 8, 1024)
    t_end = ask_float("Maximale Simulationszeit t_end", 50.0 if re <= 400 else 100.0, lo=0)
    tol = _steady_tol(1e-5)
    return {"fall": "cavity", "parameter": {"re": re, "n": n, "t_end": t_end, "steady_tol": tol}}


def setup_cylinder() -> dict:
    print("\n--- Zylinderumströmung ---")
    print("Kanal 2.2 x 0.41, Zylinder mit D = 0.1 bei (0.2, 0.2), Benchmark nach Schäfer & Turek.")
    print("Re = 20: stationär (Test 2D-1), Re = 100: Kármánsche Wirbelstraße (Test 2D-2).")
    re = ask_float("Reynolds-Zahl Re = U_mittel * D / nu", 100, lo=1)
    res = ask_choice("Gitterauflösung:", [
        ("grob", "grob  (220 x 41, D ≈ 10 Zellen), schnell"),
        ("mittel", "mittel (440 x 82, D ≈ 20 Zellen)"),
        ("fein", "fein  (660 x 123, D ≈ 30 Zellen), langsam"),
        ("eigen", "eigene Werte"),
    ], default=2)
    sizes = {"grob": (220, 41), "mittel": (440, 82), "fein": (660, 123)}
    if res == "eigen":
        ny = ask_int("Zellen in y-Richtung (Höhe 0.41)", 82, 10, 2000)
        nx = ask_int("Zellen in x-Richtung (Länge 2.2)", int(round(ny * 2.2 / 0.41)), 10, 10000)
    else:
        nx, ny = sizes[res]
    prof = ask_choice("Einströmprofil:", [("parabolic", "parabolisch (Benchmark)"),
                                          ("uniform", "Blockprofil")], 1)
    t_end = ask_float("Simulationszeit t_end", 8.0 if re >= 50 else 4.0, lo=0)
    return {"fall": "cylinder", "parameter": {"re": re, "nx": nx, "ny": ny, "t_end": t_end,
                                              "profile": prof}}


def setup_channel() -> dict:
    print("\n--- Kanalströmung (Poiseuille) ---")
    print("Kanal der Höhe H = 1, mittlere Geschwindigkeit U = 1. Vergleich mit dem")
    print("analytischen Profil u(y) = 6 U y (H - y) / H² und dp/dx = -12 nu U / H².")
    re = ask_float("Reynolds-Zahl Re = U * H / nu", 20, lo=0.1)
    length = ask_float("Kanallänge L / H", max(6.0, round(0.1 * re + 3)), lo=1)
    ny = ask_int("Zellen über die Kanalhöhe", 32, 6, 1000)
    prof = ask_choice("Einströmprofil:", [("uniform", "Blockprofil (Einlaufströmung)"),
                                          ("parabolic", "parabolisch (bereits ausgebildet)")], 1)
    t_end = ask_float("Maximale Simulationszeit t_end", 40.0, lo=0)
    tol = _steady_tol(1e-5)
    return {"fall": "channel", "parameter": {"re": re, "length": length, "ny": ny,
                                             "profile": prof, "t_end": t_end, "steady_tol": tol}}


def setup_step() -> dict:
    print("\n--- Rückwärtsgewandte Stufe ---")
    print("Kanalhöhe H = 1, Stufenhöhe h = 0.5, parabolische Einströmung (U = 1) über der Stufe.")
    print("Re = U * H / nu (wie Gartling 1990). Ab Re ≈ 400 bildet sich oben eine zweite Ablöseblase.")
    re = ask_float("Reynolds-Zahl Re", 400, lo=1)
    length = ask_float("Kanallänge L / H", 15.0 if re <= 500 else 30.0, lo=3)
    ny = ask_int("Zellen über die Kanalhöhe", 40, 8, 1000)
    inlet = ask_float("Länge des Einlaufkanals vor der Stufe", 1.0, lo=0, hi=length / 2)
    t_end = ask_float("Maximale Simulationszeit t_end", 100.0, lo=0)
    tol = _steady_tol(1e-5)
    return {"fall": "step", "parameter": {"re": re, "length": length, "ny": ny,
                                          "inlet_length": inlet, "t_end": t_end,
                                          "steady_tol": tol}}


def setup_custom() -> dict:
    print("\n--- Eigene Hindernisse ---")
    print("Rechteckiges Gebiet, Standard: Einströmung links, Ausströmung rechts.")
    lx = ask_float("Gebietslänge lx", 4.0, lo=1e-6)
    ly = ask_float("Gebietshöhe ly", 1.0, lo=1e-6)
    ny = ask_int("Zellen in y-Richtung", 80, 8, 4000)
    nx = ask_int("Zellen in x-Richtung", int(round(ny * lx / ly)), 8, 20000)
    U = ask_float("Einströmgeschwindigkeit U", 1.0, lo=1e-9)
    p = {"lx": lx, "ly": ly, "nx": nx, "ny": ny, "U": U}
    if ask_bool("Alle Randbedingungen einzeln festlegen?", False):
        p["bcs"] = ask_boundaries(U)
        if p["bcs"]["left"]["kind"] == "periodic" and ask_bool(
                "Volumenstrom regeln (periodisch, wie meanVelocityForce)?", True):
            p["flow_rate"] = ask_float("  mittlere Geschwindigkeit", U, lo=1e-9)
    else:
        p["inflow_profile"] = ask_choice("Einströmprofil:", [("uniform", "Blockprofil"),
                                                             ("parabolic", "parabolisch")], 1)
        p["walls"] = ask_choice("Wände oben und unten:", [("slip", "reibungsfrei (freie Anströmung)"),
                                                         ("noslip", "haftend (Kanal)")], 1)
    obstacles = []
    while True:
        spec = ask_obstacle(lx, ly)
        if spec is None:
            break
        obstacles.append(spec)
        print(f"  ✓ {geometry.describe_obstacle(spec)}")
    if not obstacles:
        print("  Keine Hindernisse, es wird die leere Strömung gerechnet.")
    p["obstacles"] = obstacles
    p["L_ref"] = ask_float("Charakteristische Länge L_ref für Re und Beiwerte (- = automatisch)",
                           None, lo=1e-9, allow_none=True)
    p["re"] = ask_float("Reynolds-Zahl Re = U * L_ref / nu", 100, lo=0.01)
    p["t_end"] = ask_float("Simulationszeit t_end", 10.0, lo=0)
    p["steady_tol"] = (_steady_tol(1e-5) if ask_bool("Bei stationärer Lösung vorzeitig abbrechen?",
                                                     False) else None)
    return {"fall": "custom", "parameter": p}


def _limiter_choice(default: int = 2) -> str:
    from .kernels import LIMITER_NAMES_DE

    order = ["upwind", "vanleer", "minmod", "superbee", "mc", "linear"]
    return ask_choice("Konvektionsschema für Skalare (fvSchemes):",
                      [(k, LIMITER_NAMES_DE[k]) for k in order], default)


def setup_heated_cavity() -> dict:
    print("\n--- Beheizte Cavity (natürliche Konvektion) ---")
    print("Box 1 x 1, links warm (T = +0.5), rechts kalt (T = -0.5), oben und unten adiabat.")
    print("Referenz: de Vahl Davis (1983) für Ra = 1e3, 1e4, 1e5, 1e6 (Pr = 0.71).")
    ra = ask_float("Rayleigh-Zahl Ra", 1e4, lo=1)
    pr = ask_float("Prandtl-Zahl Pr", 0.71, lo=1e-3)
    n_def = 32 if ra <= 1e3 else 48 if ra <= 1e4 else 80 if ra <= 1e5 else 128
    n = ask_int("Zellen pro Richtung", n_def, 8, 1024)
    t_end = ask_float("Maximale Simulationszeit t_end (Freifall-Einheiten)", 500.0, lo=0)
    tol = _steady_tol(1e-6)
    lim = _limiter_choice()
    return {"fall": "heated_cavity", "parameter": {"ra": ra, "pr": pr, "n": n, "t_end": t_end,
                                                   "steady_tol": tol, "limiter": lim}}


def setup_rayleigh_benard() -> dict:
    print("\n--- Rayleigh-Bénard-Konvektion ---")
    print("Fluidschicht der Höhe 1, unten warm, oben kalt, periodisch in x.")
    print("Kritische Rayleigh-Zahl: 1708. Darunter Wärmeleitung (Nu = 1), darüber Konvektionsrollen.")
    ra = ask_float("Rayleigh-Zahl Ra", 1e4, lo=1)
    pr = ask_float("Prandtl-Zahl Pr", 0.71, lo=1e-3)
    aspect = ask_float("Seitenverhältnis L/H (4.032 = zwei kritische Wellenlängen)", 4.032, lo=0.5)
    ny = ask_int("Zellen über die Höhe", 24, 6, 500)
    t_end = ask_float("Maximale Simulationszeit t_end", 300.0, lo=0)
    tol = _steady_tol(1e-6)
    return {"fall": "rayleigh_benard", "parameter": {"ra": ra, "pr": pr, "aspect": aspect,
                                                     "ny": ny, "t_end": t_end, "steady_tol": tol}}


def setup_turbulent_channel() -> dict:
    print("\n--- Turbulente Kanalströmung (RANS k-ω SST) ---")
    print("Periodischer Kanal (Halbhöhe h = 1), Volumenstrom geregelt (U_b = 1).")
    print("Vergleich: Wandgesetz (u+ = y+, Log-Gesetz) und Reibungsbeiwert nach Dean.")
    re_tau = ask_float("Reibungs-Reynolds-Zahl Re_tau", 180, lo=50, hi=2000)
    yp = ask_float("y+ der ersten Zelle (≤ 1 empfohlen)", 1.0, lo=0.1, hi=5)
    model = ask_choice("Turbulenzmodell:", [("kOmegaSST", "k-ω SST (Menter)"),
                                            ("Smagorinsky", "Smagorinsky (LES-Modell)")], 1)
    t_end = ask_float("Maximale Simulationszeit t_end", 400.0, lo=0)
    tol = _steady_tol(1e-6)
    return {"fall": "turbulent_channel", "parameter": {"re_tau": re_tau, "y_plus_1": yp,
                                                       "model": model, "t_end": t_end,
                                                       "steady_tol": tol}}


def setup_dam_break() -> dict:
    print("\n--- Dammbruch (Zweiphasen-VOF, Wasser/Luft, SI-Einheiten) ---")
    print("Becken 0.584 m x 0.584 m, Wassersäule 0.146 m x 0.292 m an der linken Wand.")
    n = ask_int("Zellen pro Richtung", 64, 16, 512)
    obstacle = ask_bool("Hindernis am Boden (0.024 m x 0.048 m)?", True)
    open_top = ask_bool("Becken oben offen (Atmosphäre)?", True)
    sigma = ask_float("Oberflächenspannung sigma [N/m] (0 = aus)", 0.07, lo=0)
    t_end = ask_float("Simulationszeit t_end [s]", 1.0, lo=0)
    return {"fall": "dam_break", "parameter": {"n": n, "obstacle": obstacle, "open_top": open_top,
                                               "sigma": sigma, "t_end": t_end}}


SETUPS = {"cavity": setup_cavity, "cylinder": setup_cylinder, "channel": setup_channel,
          "step": setup_step, "custom": setup_custom, "heated_cavity": setup_heated_cavity,
          "rayleigh_benard": setup_rayleigh_benard,
          "turbulent_channel": setup_turbulent_channel, "dam_break": setup_dam_break}

# Fälle, denen man Turbulenz, Temperatur und Skalare hinzufügen kann
FLOW_CASES = ("cavity", "cylinder", "channel", "step", "custom")


# ---------------------------------------------------------------------- #
# Zusatzphysik, Numerik, Function Objects
# ---------------------------------------------------------------------- #
def ask_physics(spec: dict) -> dict:
    if spec["fall"] not in FLOW_CASES:
        return spec
    print("\n--- Zusatzphysik ---")
    turb = ask_choice("Turbulenzmodell (momentumTransport):", [
        ("laminar", "laminar (kein Modell)"),
        ("kOmegaSST", "RANS k-ω SST (Menter 2003)"),
        ("Smagorinsky", "LES Smagorinsky")], 1)
    if turb == "kOmegaSST":
        spec["turbulenz"] = {"modell": turb,
                             "intensity": ask_float("  Turbulenzgrad I am Einlass", 0.05, 0, 1),
                             "viscosity_ratio": ask_float("  Viskositätsverhältnis ν_t/ν am Einlass",
                                                          10.0, lo=1e-3)}
    elif turb == "Smagorinsky":
        spec["turbulenz"] = {"modell": turb, "cs": ask_float("  Smagorinsky-Konstante C_s", 0.17,
                                                             0.01, 0.5)}
    if ask_bool("Temperatur / Wärmeübertragung berechnen?", False):
        pr = ask_float("  Prandtl-Zahl Pr (alpha = nu / Pr)", 0.71, lo=1e-4)
        t = {"alpha": None, "prandtl": pr,
             "T_inflow": ask_float("  Temperatur der Einströmung", 0.0)}
        if spec["fall"] in ("cylinder", "custom", "step"):
            if ask_bool("  Hindernisse mit fester Temperatur (beheizt)?", spec["fall"] != "step"):
                t["T_obstacle"] = ask_float("  Temperatur der Hindernisse", 1.0)
        walls = {}
        for side in SIDES:
            if ask_bool(f"  Wand {SIDE_NAMES_DE[side]} mit fester Temperatur?", False):
                walls[side] = ask_float(f"    Temperatur {SIDE_NAMES_DE[side]}", 1.0)
        t["wall_temperatures"] = walls
        t["buoyancy"] = ask_bool("  Auftrieb (Boussinesq) berücksichtigen?", False)
        if t["buoyancy"]:
            t["gravity"] = [0.0, -ask_float("  beta * g (Auftriebsstärke)", 1.0, lo=0)]
        t["limiter"] = _limiter_choice()
        spec["temperatur"] = t
    if ask_bool("Passiven Skalar (Farbstoff/Konzentration) transportieren?", False):
        sk = {"name": "c", "diffusivity": ask_float("  Diffusionskoeffizient D", 1e-3, lo=0)}
        if ask_bool("  Nur in einem Streifen des Einlasses einströmen lassen?", True):
            s0 = ask_float("    Streifen von (Koordinate entlang des Einlasses)", 0.4)
            s1 = ask_float("    Streifen bis", 0.6)
            sk["inflow_band"] = [min(s0, s1), max(s0, s1)]
        sk["limiter"] = _limiter_choice()
        spec["skalar"] = sk
    return spec


def ask_numerics(spec: dict) -> dict:
    if not ask_bool("\nNumerische Einstellungen anpassen (fvSchemes/fvSolution)?", False):
        return spec
    num = {}
    num["scheme"] = ask_choice("Zeitintegration:", [
        ("ab2", "Adams-Bashforth 2. Ordnung (genauer)"),
        ("euler", "Euler explizit 1. Ordnung (robust)")], 1)
    g = ask_float("Upwind-Anteil gamma der Konvektion, 0 = zentral, 1 = Upwind (- = automatisch)",
                  None, 0, 1, allow_none=True)
    num["gamma"] = "auto" if g is None else g
    num["tau"] = ask_float("Sicherheitsfaktor tau für den Zeitschritt (0..1]", 0.5, 0.01, 1)
    num["pressure_solver"] = ask_choice("Drucklöser:", [
        ("direct", "direkt (LU-Zerlegung, schnell)"),
        ("cg", "iterativ (CG, wenig Speicher)")], 1)
    spec["numerik"] = num
    return spec


def ask_function_objects(spec: dict) -> dict:
    if not ask_bool("\nFunction Objects einrichten (Sonden, Linienproben, VTK, Speicherstände)?",
                    False):
        return spec
    fos = []
    while True:
        typ = ask_choice("\nFunction Object hinzufügen:", [
            ("sonden", "Sonden: Zeitreihen an Punkten (probes)"),
            ("linie", "Linienprobe: Profil entlang einer Strecke (sample)"),
            ("mittelwerte", "Zeitmittel und RMS-Schwankungen (fieldAverage)"),
            ("vtk", "VTK-Export für ParaView (foamToVTK)"),
            ("checkpoint", "Speicherstände für den Neustart"),
            ("fertig", "fertig")], 6)
        if typ == "fertig":
            break
        if typ == "sonden":
            pts = ask_points("  Sondenpunkte", min_points=1)
            fos.append({"typ": "sonden", "punkte": pts,
                        "felder": ask_str("  Felder (durch Komma getrennt)", "u,v,pressure").replace(
                            " ", "").split(","),
                        "alle": ask_int("  alle N Schritte aufzeichnen", 1, 1)})
        elif typ == "linie":
            x0 = ask_float("  Startpunkt x", 0.0)
            y0 = ask_float("  Startpunkt y", 0.0)
            x1 = ask_float("  Endpunkt x", 0.0)
            y1 = ask_float("  Endpunkt y", 1.0)
            fos.append({"typ": "linie", "p0": [x0, y0], "p1": [x1, y1],
                        "n": ask_int("  Anzahl Punkte", 100, 2),
                        "felder": ask_str("  Felder (durch Komma getrennt)", "u,v").replace(
                            " ", "").split(","),
                        "name": ask_str("  Name", f"linie{len(fos) + 1}")})
        elif typ == "mittelwerte":
            fos.append({"typ": "mittelwerte", "t_start": ask_float("  Mittelung ab Zeit", 0.0, lo=0)})
        elif typ == "vtk":
            fos.append({"typ": "vtk", "intervall": ask_float("  Schreibintervall Δt", 0.5, lo=1e-9)})
        elif typ == "checkpoint":
            fos.append({"typ": "checkpoint",
                        "intervall": ask_float("  Speicherintervall Δt", 5.0, lo=1e-9)})
        print(f"  ✓ {typ} hinzugefügt")
    spec["function_objects"] = fos
    return spec


def ask_visualization(can_show: bool, case=None) -> dict:
    from .visualization import FIELDS, available_fields

    print("\n--- Visualisierung ---")
    vis = {}
    if can_show:
        vis["live"] = ask_bool("Live-Visualisierung während der Rechnung?", True)
    else:
        print("Kein Bildschirm verfügbar: Live-Fenster nicht möglich, es werden Diagramme gespeichert.")
        vis["live"] = False
    vis["record"] = ask_bool("Animation als GIF speichern?", not can_show)
    if vis["live"] or vis["record"]:
        fields = available_fields(case.sim) if case is not None else dict(FIELDS)
        default = 1
        for key in ("alpha", "T"):
            if key in fields:
                default = list(fields).index(key) + 1
                break
        vis["field"] = ask_choice("Angezeigtes Feld:", list(fields.items()), default)
        vis["streamlines"] = ask_bool("Stromlinien einblenden (langsamer)?", False)
        vis["every"] = ask_int("Aktualisierung alle N Zeitschritte", 50, 1, 100000)
    return vis


def _resolve_temperature(spec: dict) -> dict:
    """alpha aus der Prandtl-Zahl bestimmen, sobald nu bekannt ist (beim Aufbau des Falls)."""
    t = spec.get("temperatur")
    if t and t.get("alpha") is None:
        tmp = dict(spec)
        tmp.pop("temperatur")
        tmp.pop("skalar", None)
        tmp.pop("turbulenz", None)
        nu = cases.build_case(tmp).sim.nu
        t = dict(t)
        t["alpha"] = nu / t.pop("prandtl", 0.71)
        spec = dict(spec)
        spec["temperatur"] = t
    elif t and "prandtl" in t:
        t = dict(t)
        t.pop("prandtl")
        spec = dict(spec)
        spec["temperatur"] = t
    return spec


def build_from_spec(spec: dict, outdir: Path):
    from .functions import build_function_objects

    spec = _resolve_temperature(spec)
    case = cases.build_case(spec)
    case.sim.function_objects = build_function_objects(spec.get("function_objects"), outdir)
    return case


def run_case(case: cases.Case, vis: dict, outdir: Path, t_end: float | None = None) -> dict:
    from . import kernels, report
    from .visualization import LivePlotter

    print("\n" + case.describe())
    print(f"\nErgebnisse werden gespeichert in: {outdir}")
    print("Kompiliere Numba-Kernel (nur beim ersten Start spürbar) ...")
    kernels.warmup()

    plotter = None
    if vis.get("live") or vis.get("record"):
        plotter = LivePlotter(case, field=vis.get("field", "speed"),
                              streamlines=vis.get("streamlines", False),
                              show=vis.get("live", False), record=vis.get("record", False))
    print("\nSimulation läuft (Strg+C bricht ab) ...")
    try:
        reason = case.sim.run(t_end if t_end is not None else case.t_end,
                              steady_tol=case.steady_tol, callback=plotter,
                              callback_every=vis.get("every", 50))
    except FloatingPointError as e:
        print(f"\nFehler: {e}\nDie Auswertung wird mit dem letzten Zustand erstellt.")
        reason = "instabil"
    if plotter is not None and plotter.closed:
        print("Live-Fenster wurde geschlossen, die Simulation ist beendet.")

    print("\nErstelle Diagramme und Auswertung ...")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = report.create_report(case, outdir, reason)
    from .functions import save_checkpoint

    save_checkpoint(case.sim, outdir / "checkpoint.npz")
    if plotter is not None and vis.get("record"):
        gif = plotter.save_animation(outdir / "animation.gif")
        if gif:
            print(f"Animation gespeichert: {gif}")
    report.print_summary(summary)
    print(f"\nDiagramme und Daten liegen in: {outdir.resolve()}")
    for f in sorted(outdir.iterdir()):
        print(f"  - {f.name}")
    print("Mit 'Rechnung fortsetzen' im Hauptmenü kann diese Rechnung später weiterlaufen.")

    if plotter is not None:
        if plotter.show and not plotter.closed:
            print("\nDas Live-Fenster bleibt offen. Zum Fortfahren das Fenster schließen.")
            plotter.keep_open()
        plotter.close()
    return summary


def _new_outdir(output_root, spec) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = str(Path(output_root) / f"{spec['fall']}_{stamp}")
    return Path(ask_str("Ausgabeordner", default_out)).expanduser()


def interactive_new(key: str, can_show: bool, output_root) -> None:
    spec = SETUPS[key]()
    spec = ask_physics(spec)
    spec = ask_numerics(spec)
    spec = ask_function_objects(spec)
    outdir = _new_outdir(output_root, spec)
    case = build_from_spec(spec, outdir)
    vis = ask_visualization(can_show, case)
    from .visualization import ensure_backend

    ensure_backend(vis.get("live", False))
    if ask_bool("Case-Datei (JSON) zusätzlich unter eigenem Namen speichern?", False):
        p = Path(ask_str("Dateiname", f"{spec['fall']}.json")).expanduser()
        cases.save_case(spec, p)
        print(f"  gespeichert: {p}")
    if not ask_bool("\nSimulation starten?", True):
        return
    run_case(case, vis, outdir)


def interactive_load(can_show: bool, output_root) -> None:
    path = Path(ask_str("Pfad zur Case-Datei (JSON)")).expanduser()
    if path.is_dir():
        path = path / "case.json"
    spec = cases.load_case(path)
    print(f"Geladen: {cases.CASE_TITLES.get(spec['fall'], spec['fall'])}")
    if ask_bool("Parameter vor dem Start anpassen?", False):
        p = spec.setdefault("parameter", {})
        for k, v in list(p.items()):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            p[k] = ask_float(f"  {k}", v)
            if isinstance(v, int) and float(p[k]).is_integer():
                p[k] = int(p[k])
    outdir = _new_outdir(output_root, spec)
    case = build_from_spec(spec, outdir)
    vis = ask_visualization(can_show, case)
    from .visualization import ensure_backend

    ensure_backend(vis.get("live", False))
    if ask_bool("\nSimulation starten?", True):
        run_case(case, vis, outdir)


def interactive_restart(can_show: bool) -> None:
    from .functions import load_checkpoint

    d = Path(ask_str("Ergebnisordner der Rechnung (mit case.json und checkpoint.npz)")).expanduser()
    if not (d / "case.json").exists() or not (d / "checkpoint.npz").exists():
        raise FileNotFoundError(f"In {d} fehlen case.json oder checkpoint.npz.")
    spec = cases.load_case(d / "case.json")
    case = build_from_spec(spec, d)
    load_checkpoint(case.sim, d / "checkpoint.npz")
    print(f"Speicherstand geladen: t = {case.sim.t:.4g}, Schritt {case.sim.n}")
    t_end = ask_float("Weiterrechnen bis Zeit t_end", max(case.t_end, case.sim.t * 2 or 1.0),
                      lo=case.sim.t)
    vis = ask_visualization(can_show, case)
    from .visualization import ensure_backend

    ensure_backend(vis.get("live", False))
    run_case(case, vis, d, t_end=t_end)


def main(output_root: str | Path = "ergebnisse") -> None:
    from .visualization import display_available

    print(BANNER)
    can_show = display_available()
    menu = [
        ("cavity", "Lid-Driven Cavity (Validierung nach Ghia et al.)"),
        ("cylinder", "Zylinderumströmung, Kármánsche Wirbelstraße (Schäfer-Turek)"),
        ("channel", "Kanalströmung, Poiseuille (analytische Lösung)"),
        ("step", "Rückwärtsgewandte Stufe (Ablösung und Wiederanlegung)"),
        ("custom", "Eigene Hindernisse / freie Randbedingungen (Kreis, NACA, Polygon, Bild)"),
        ("heated_cavity", "Beheizte Cavity: natürliche Konvektion (de Vahl Davis)"),
        ("rayleigh_benard", "Rayleigh-Bénard-Konvektion (periodisch)"),
        ("turbulent_channel", "Turbulenter Kanal, RANS k-ω SST (Wandgesetz, Dean)"),
        ("dam_break", "Dammbruch: Zweiphasenströmung Wasser/Luft (VOF)"),
        ("load", "Case aus JSON-Datei laden"),
        ("restart", "Rechnung aus Speicherstand fortsetzen"),
        ("quit", "Beenden"),
    ]
    try:
        while True:
            key = ask_choice("\nHauptmenü: Testfall wählen", menu, 1)
            if key == "quit":
                break
            try:
                if key == "load":
                    interactive_load(can_show, output_root)
                elif key == "restart":
                    interactive_restart(can_show)
                else:
                    interactive_new(key, can_show, output_root)
            except Abort:
                print("  Eingabe abgebrochen, zurück zum Hauptmenü.")
                continue
            except (ValueError, FileNotFoundError, KeyError) as e:
                print(f"\nFehler: {e}")
                continue
            if not ask_bool("\nWeitere Simulation durchführen?", True):
                break
    except (Abort, KeyboardInterrupt, EOFError):
        print()
    print("Auf Wiedersehen!")
