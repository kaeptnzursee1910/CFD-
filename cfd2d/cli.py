"""Kommandozeile.

Ohne Argumente startet das interaktive Menü::

    python -m cfd2d

Mit ``--fall`` läuft die Rechnung direkt, ohne Rückfragen, zum Beispiel für Skripte::

    python -m cfd2d --fall cavity --re 400 --nx 96
    python -m cfd2d --fall cylinder --re 100 --t-ende 10 --live --feld vorticity
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

from .visualization import FIELDS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cfd2d",
        description="2D-CFD für inkompressible Strömungen. Ohne --fall startet das interaktive Menü.")
    p.add_argument("--fall", choices=["cavity", "cylinder", "channel", "step", "heated_cavity",
                                      "rayleigh_benard", "turbulent_channel", "dam_break"],
                   help="Testfall direkt starten (nicht interaktiv)")
    p.add_argument("--case", dest="case_file", help="Case-Datei (JSON) laden und rechnen")
    p.add_argument("--ra", type=float, help="Rayleigh-Zahl (thermische Fälle)")
    p.add_argument("--turbulenz", choices=["laminar", "kOmegaSST", "Smagorinsky"],
                   help="Turbulenzmodell ergänzen")
    p.add_argument("--re", type=float, help="Reynolds-Zahl")
    p.add_argument("--nx", type=int, help="Zellen in x-Richtung (bei der Cavity: pro Richtung)")
    p.add_argument("--ny", type=int, help="Zellen in y-Richtung")
    p.add_argument("--laenge", type=float, help="Kanallänge (Kanal und Stufe)")
    p.add_argument("--t-ende", type=float, dest="t_end", help="Simulationszeit")
    p.add_argument("--schema", choices=["ab2", "euler"], help="Zeitintegration")
    p.add_argument("--gamma", type=float, help="Upwind-Anteil 0..1 (Standard: automatisch)")
    p.add_argument("--tau", type=float, help="Sicherheitsfaktor Zeitschritt")
    p.add_argument("--drucksolver", choices=["direct", "cg"], dest="pressure_solver")
    p.add_argument("--live", action="store_true", help="Live-Visualisierung anzeigen")
    p.add_argument("--gif", action="store_true", help="Animation als GIF speichern")
    p.add_argument("--feld", default="speed",
                   help=f"Feld für die Live-Ansicht ({', '.join(FIELDS)}, T, alpha, k, nu_t, ...)")
    p.add_argument("--stromlinien", action="store_true", help="Stromlinien in der Live-Ansicht")
    p.add_argument("--alle", type=int, default=50, help="Live-Ansicht alle N Schritte aktualisieren")
    p.add_argument("--ausgabe", help="Ausgabeordner")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.case_file:
        from . import cases
        from .interactive import build_from_spec, run_case
        from .visualization import ensure_backend

        ensure_backend(args.live)
        spec = cases.load_case(args.case_file)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.ausgabe or f"ergebnisse/{spec['fall']}_{stamp}")
        case = build_from_spec(spec, out)
        vis = {"live": args.live, "record": args.gif, "field": args.feld,
               "streamlines": args.stromlinien, "every": args.alle}
        run_case(case, vis, out)
        return 0
    if args.fall is None:
        from . import interactive

        interactive.main()
        return 0

    from . import cases
    from .interactive import run_case
    from .visualization import ensure_backend

    ensure_backend(args.live)
    num = {"scheme": args.schema, "gamma": args.gamma, "tau": args.tau,
           "pressure_solver": args.pressure_solver}
    num = {k: v for k, v in num.items() if v is not None}
    kw = {}
    if args.re is not None:
        kw["re"] = args.re
    if args.t_end is not None:
        kw["t_end"] = args.t_end
    if args.ra is not None and args.fall in ("heated_cavity", "rayleigh_benard"):
        kw["ra"] = args.ra
        kw.pop("re", None)
    if args.fall in ("turbulent_channel", "dam_break"):
        kw.pop("re", None)
    if args.fall in ("cavity", "heated_cavity", "dam_break"):
        if args.nx:
            kw["n"] = args.nx
    else:
        if args.nx:
            kw["nx"] = args.nx
        if args.ny:
            kw["ny"] = args.ny
        if args.laenge and args.fall in ("channel", "step"):
            kw["length"] = args.laenge
    spec = {"fall": args.fall, "parameter": kw, "numerik": num}
    if args.turbulenz and args.turbulenz != "laminar":
        spec["turbulenz"] = {"modell": args.turbulenz}
    from .interactive import build_from_spec

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.ausgabe or f"ergebnisse/{args.fall}_{stamp}")
    case = build_from_spec(spec, out)
    vis = {"live": args.live, "record": args.gif, "field": args.feld,
           "streamlines": args.stromlinien, "every": args.alle}
    run_case(case, vis, out)
    return 0
