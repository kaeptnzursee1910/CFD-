"""Interaktive Bedienung über die Konsole (Eingabe von Parametern)."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from . import cases, geometry
from .boundary import KIND_NAMES_DE, SIDE_NAMES_DE, SIDES
from .boundary import BoundaryCondition as BC

BANNER = r"""
 ╔══════════════════════════════════════════════════════════════════╗
 ║                 2D-CFD: inkompressible Strömung                  ║
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


def ask_points(prompt: str) -> list[tuple[float, float]]:
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
            if len(pts) >= 3:
                return pts
            print("  Mindestens 3 Punkte nötig.")
        except (ValueError, IndexError):
            print("  Ungültiges Format. Beispiel: 1 0.4; 1.5 0.5; 1 0.6")


# ---------------------------------------------------------------------- #
# Fallspezifische Abfragen
# ---------------------------------------------------------------------- #
def _steady_tol(default: float) -> float | None:
    tol = ask_float("Abbruch bei stationärer Lösung: Residuum-Grenze (- = aus)", default,
                    lo=0.0, allow_none=True)
    return tol if tol else None


def setup_cavity() -> cases.Case:
    print("\n--- Lid-Driven Cavity ---")
    print("Quadratische Box (1 x 1), Deckel bewegt sich mit U = 1 nach rechts.")
    print("Ghia-Referenzdaten gibt es für Re = 100, 400 und 1000.")
    re = ask_float("Reynolds-Zahl Re", 100, lo=1)
    n_default = 64 if re <= 400 else 128
    n = ask_int("Zellen pro Richtung", n_default, 8, 1024)
    t_end = ask_float("Maximale Simulationszeit t_end", 50.0 if re <= 400 else 100.0, lo=0)
    tol = _steady_tol(1e-5)
    num = ask_numerics()
    return cases.lid_driven_cavity(re=re, n=n, t_end=t_end, steady_tol=tol, **num)


def setup_cylinder() -> cases.Case:
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
    num = ask_numerics()
    return cases.cylinder_flow(re=re, nx=nx, ny=ny, t_end=t_end, profile=prof, **num)


def setup_channel() -> cases.Case:
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
    num = ask_numerics()
    return cases.channel_flow(re=re, length=length, ny=ny, profile=prof, t_end=t_end,
                              steady_tol=tol, **num)


def setup_step() -> cases.Case:
    print("\n--- Rückwärtsgewandte Stufe ---")
    print("Kanalhöhe H = 1, Stufenhöhe h = 0.5, parabolische Einströmung (U = 1) über der Stufe.")
    print("Re = U * H / nu (wie Gartling 1990). Ab Re ≈ 400 bildet sich oben eine zweite Ablöseblase.")
    re = ask_float("Reynolds-Zahl Re", 400, lo=1)
    length = ask_float("Kanallänge L / H", 15.0 if re <= 500 else 30.0, lo=3)
    ny = ask_int("Zellen über die Kanalhöhe", 40, 8, 1000)
    inlet = ask_float("Länge des Einlaufkanals vor der Stufe", 1.0, lo=0, hi=length / 2)
    t_end = ask_float("Maximale Simulationszeit t_end", 100.0, lo=0)
    tol = _steady_tol(1e-5)
    num = ask_numerics()
    return cases.backward_facing_step(re=re, length=length, ny=ny, inlet_length=inlet,
                                      t_end=t_end, steady_tol=tol, **num)


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
    print("\nRandbedingungen festlegen:")
    opts = [(k, KIND_NAMES_DE[k]) for k in ("noslip", "slip", "inflow", "outflow")]
    defaults = {"left": 3, "right": 4, "bottom": 2, "top": 2}
    bcs = {}
    for side in SIDES:
        kind = ask_choice(f" Rand {SIDE_NAMES_DE[side]}:", opts, defaults[side])
        if kind == "inflow":
            vel = ask_float("   mittlere Einströmgeschwindigkeit", U, lo=0)
            prof = ask_choice("   Profil:", [("uniform", "Blockprofil"),
                                             ("parabolic", "parabolisch")], 1)
            bcs[side] = BC("inflow", vel, prof)
        elif kind == "noslip":
            vel = ask_float("   tangentiale Wandgeschwindigkeit", 0.0)
            bcs[side] = BC("noslip", vel)
        else:
            bcs[side] = BC(kind)
    if not any(b.kind == "outflow" for b in bcs.values()):
        inflow = any(b.kind == "inflow" for b in bcs.values())
        if inflow:
            print("  Warnung: Es gibt eine Einströmung, aber keine Ausströmung. Die Masse kann nicht"
                  " abfließen, die Lösung wird unphysikalisch.")
    return bcs


def setup_custom() -> cases.Case:
    print("\n--- Eigene Hindernisse ---")
    print("Rechteckiges Gebiet, Standard: Einströmung links, Ausströmung rechts.")
    lx = ask_float("Gebietslänge lx", 4.0, lo=1e-6)
    ly = ask_float("Gebietshöhe ly", 1.0, lo=1e-6)
    ny = ask_int("Zellen in y-Richtung", 80, 8, 4000)
    nx = ask_int("Zellen in x-Richtung", int(round(ny * lx / ly)), 8, 20000)
    U = ask_float("Einströmgeschwindigkeit U", 1.0, lo=1e-9)

    if ask_bool("Alle Randbedingungen einzeln festlegen?", False):
        bcs = ask_boundaries(U)
        prof, walls = "uniform", "slip"
    else:
        bcs = None
        prof = ask_choice("Einströmprofil:", [("uniform", "Blockprofil"),
                                              ("parabolic", "parabolisch")], 1)
        walls = ask_choice("Wände oben und unten:", [("slip", "reibungsfrei (freie Anströmung)"),
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

    L_ref = ask_float("Charakteristische Länge L_ref für Re und Beiwerte (- = automatisch)", None,
                      lo=1e-9, allow_none=True)
    re = ask_float("Reynolds-Zahl Re = U * L_ref / nu", 100, lo=0.01)
    t_end = ask_float("Simulationszeit t_end", 10.0, lo=0)
    tol = _steady_tol(0) if ask_bool("Bei stationärer Lösung vorzeitig abbrechen?", False) else None
    num = ask_numerics()
    case = cases.custom_obstacles(obstacles, lx=lx, ly=ly, nx=nx, ny=ny, re=re, U=U, L_ref=L_ref,
                                  inflow_profile=prof, walls=walls, t_end=t_end, bcs=bcs,
                                  steady_tol=tol, **num)
    print(f"  L_ref = {case.L_ref:.4g},  nu = {case.sim.nu:.4g}")
    return case


SETUPS = {"cavity": setup_cavity, "cylinder": setup_cylinder, "channel": setup_channel,
          "step": setup_step, "custom": setup_custom}


def ask_numerics() -> dict:
    if not ask_bool("\nNumerische Einstellungen anpassen?", False):
        return {}
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
    return num


def ask_visualization(can_show: bool) -> dict:
    from .visualization import FIELDS

    print("\n--- Visualisierung ---")
    vis = {}
    if can_show:
        vis["live"] = ask_bool("Live-Visualisierung während der Rechnung?", True)
    else:
        print("Kein Bildschirm verfügbar: Live-Fenster nicht möglich, es werden Diagramme gespeichert.")
        vis["live"] = False
    vis["record"] = ask_bool("Animation als GIF speichern?", not can_show)
    if vis["live"] or vis["record"]:
        vis["field"] = ask_choice("Angezeigtes Feld:", list(FIELDS.items()), 1)
        vis["streamlines"] = ask_bool("Stromlinien einblenden (langsamer)?", False)
        vis["every"] = ask_int("Aktualisierung alle N Zeitschritte", 50, 1, 100000)
    return vis


def run_case(case: cases.Case, vis: dict, outdir: Path) -> dict:
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
    reason = case.sim.run(case.t_end, steady_tol=case.steady_tol, callback=plotter,
                          callback_every=vis.get("every", 50))
    if plotter is not None and plotter.closed:
        print("Live-Fenster wurde geschlossen, die Simulation ist beendet.")

    print("\nErstelle Diagramme und Auswertung ...")
    summary = report.create_report(case, outdir, reason)
    if plotter is not None and vis.get("record"):
        gif = plotter.save_animation(outdir / "animation.gif")
        if gif:
            print(f"Animation gespeichert: {gif}")
    report.print_summary(summary)
    print(f"\nDiagramme und Daten liegen in: {outdir.resolve()}")
    for f in sorted(outdir.iterdir()):
        print(f"  - {f.name}")

    if plotter is not None:
        if plotter.show and not plotter.closed:
            print("\nDas Live-Fenster bleibt offen. Zum Fortfahren das Fenster schließen.")
            plotter.keep_open()
        plotter.close()
    return summary


def main(output_root: str | Path = "ergebnisse") -> None:
    from .visualization import display_available, ensure_backend

    print(BANNER)
    can_show = display_available()
    menu = [
        ("cavity", "Lid-Driven Cavity (Validierung nach Ghia et al.)"),
        ("cylinder", "Zylinderumströmung, Kármánsche Wirbelstraße (Schäfer-Turek)"),
        ("channel", "Kanalströmung, Poiseuille (analytische Lösung)"),
        ("step", "Rückwärtsgewandte Stufe (Ablösung und Wiederanlegung)"),
        ("custom", "Eigene Hindernisse (Kreis, Rechteck, NACA, Polygon, Bild)"),
        ("quit", "Beenden"),
    ]
    try:
        while True:
            key = ask_choice("\nHauptmenü: Testfall wählen", menu, 1)
            if key == "quit":
                break
            try:
                case = SETUPS[key]()
                vis = ask_visualization(can_show)
                ensure_backend(vis.get("live", False))
                stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                default_out = str(Path(output_root) / f"{key}_Re{case.re:g}_{stamp}")
                outdir = Path(ask_str("Ausgabeordner", default_out)).expanduser()
                if not ask_bool("\nSimulation starten?", True):
                    continue
                run_case(case, vis, outdir)
            except Abort:
                print("  Eingabe abgebrochen, zurück zum Hauptmenü.")
                continue
            except (ValueError, FileNotFoundError, FloatingPointError) as e:
                print(f"\nFehler: {e}")
                continue
            if not ask_bool("\nWeitere Simulation durchführen?", True):
                break
    except (Abort, KeyboardInterrupt, EOFError):
        print()
    print("Auf Wiedersehen!")
