"""Beispiel: cfd2d als Python-Bibliothek nutzen, ohne interaktives Menü.

Aufruf aus dem Projektordner:

    python beispiele/skript_beispiel.py

Gerechnet wird die Umströmung eines angestellten NACA-2412-Profils mit einem
Zylinder im Nachlauf. Dabei läuft die Live-Ansicht mit, und zum Schluss
werden alle Diagramme gespeichert.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfd2d import cases, report  # noqa: E402
from cfd2d.visualization import LivePlotter, ensure_backend  # noqa: E402

live = ensure_backend(live=True)  # fällt ohne Bildschirm automatisch auf Agg zurück

hindernisse = [
    {"typ": "naca", "code": "2412", "x": 0.8, "y": 0.55, "sehne": 1.0, "winkel": 8.0},
    {"typ": "kreis", "x": 2.6, "y": 0.45, "r": 0.1},
]
case = cases.custom_obstacles(hindernisse, lx=5.0, ly=1.2, nx=400, ny=96, re=500,
                              L_ref=1.0, walls="slip", t_end=6.0)
print(case.describe())

plotter = LivePlotter(case, field="vorticity", show=live, record=not live)
grund = case.sim.run(case.t_end, callback=plotter, callback_every=40)

out = Path("ergebnisse/beispiel_naca")
zusammenfassung = report.create_report(case, out, grund)
if plotter.record:
    plotter.save_animation(out / "animation.gif")
report.print_summary(zusammenfassung)
plotter.keep_open()
