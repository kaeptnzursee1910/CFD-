# cfd2d: 2D-CFD für inkompressible Strömungen in Python

Ein vollständiges 2D-Strömungssimulationsprogramm für die **inkompressiblen
Navier-Stokes-Gleichungen**:

- Finite-Volumen-Diskretisierung auf einem versetzten **MAC-Gitter**
- **Projektionsmethode** nach Chorin
- **Numba**-beschleunigte Kernel
- dünn besetzter Drucklöser über **SciPy**
- **Live-Visualisierung** mit Matplotlib
- **interaktive Bedienung per Eingabe** in der Konsole

| Testfall | Validierung |
|---|---|
| Lid-Driven Cavity | Ghia, Ghia & Shin (1982), Re = 100 / 400 / 1000 |
| Zylinderumströmung | Benchmark Schäfer & Turek (1996), Re = 20 (stationär), Re = 100 (Kármánsche Wirbelstraße) |
| Kanalströmung | analytische Poiseuille-Lösung (Profil und Druckgefälle) |
| Rückwärtsgewandte Stufe | Ablöse- und Wiederanlegelängen, Wandschubspannung |
| Eigene Hindernisse | Kreis, Ellipse, Rechteck, NACA-4-Ziffern-Profil, Polygon, **Bilddatei (PNG/JPG)** |

---

## Installation

```bash
pip install -r requirements.txt
```

Benötigt werden Python ≥ 3.10, NumPy, SciPy, Numba, Matplotlib und Pillow.
Für das Live-Fenster braucht Matplotlib ein GUI-Backend, zum Beispiel Tk
(bei den meisten Python-Installationen dabei) oder `pip install PyQt5`.

## Start

### Interaktiv (Bedienung per Eingabe)

```bash
python -m cfd2d
```

Das Programm führt durch alle Einstellungen. Wer **Enter** drückt, übernimmt
den Standardwert in eckigen Klammern. Dezimalzahlen dürfen mit Komma oder
Punkt eingegeben werden, `q` bricht die aktuelle Eingabe ab.

```
Hauptmenü: Testfall wählen
  1) Lid-Driven Cavity (Validierung nach Ghia et al.)
  2) Zylinderumströmung, Kármánsche Wirbelstraße (Schäfer-Turek)
  3) Kanalströmung, Poiseuille (analytische Lösung)
  4) Rückwärtsgewandte Stufe (Ablösung und Wiederanlegung)
  5) Eigene Hindernisse (Kreis, Rechteck, NACA, Polygon, Bild)
  6) Beenden
Auswahl [1]: 2
Reynolds-Zahl Re = U_mittel * D / nu [100]:
Gitterauflösung:
  1) grob  (220 x 41, D ≈ 10 Zellen), schnell
  2) mittel (440 x 82, D ≈ 20 Zellen)
  ...
Live-Visualisierung während der Rechnung? [J/n]:
Angezeigtes Feld:
  1) Geschwindigkeitsbetrag |u|
  2) Wirbelstärke ω
  ...
```

Ablauf:

1. **Testfall** und physikalische Parameter wählen: Reynolds-Zahl, Gitter, Geometrie, Simulationszeit.
2. Optional die **Numerik** anpassen: Zeitschema, Upwind-Anteil, Zeitschritt-Sicherheitsfaktor, Drucklöser.
3. **Visualisierung** festlegen: Live-Fenster, angezeigtes Feld, Stromlinien, Aktualisierungsrate, GIF-Aufnahme.
4. Die Simulation läuft mit Live-Anzeige. Mit **Strg+C** oder durch Schließen des Fensters lässt sie sich jederzeit beenden; die Auswertung wird trotzdem erstellt.
5. Die **Diagramme und Daten** landen im Ausgabeordner (Standard: `ergebnisse/<fall>_Re<re>_<zeit>/`).

Bei *Eigene Hindernisse* lassen sich beliebig viele Hindernisse
nacheinander hinzufügen. Optional kann man alle vier Randbedingungen frei
wählen: haftende Wand (auch bewegt), reibungsfreie Wand, Einströmung mit
Block- oder Parabelprofil, Ausströmung.

### Direkt über die Kommandozeile (ohne Rückfragen)

```bash
python -m cfd2d --fall cavity   --re 1000 --nx 128
python -m cfd2d --fall cylinder --re 100  --t-ende 10 --live --feld vorticity
python -m cfd2d --fall channel  --re 50   --ny 32 --gif
python -m cfd2d --fall step     --re 400  --laenge 15 --ny 40
python -m cfd2d --help
```

### Als Bibliothek

```python
from cfd2d import cases, report
from cfd2d.visualization import LivePlotter

case = cases.custom_obstacles(
    [{"typ": "naca", "code": "0012", "x": 1.0, "y": 0.5, "sehne": 1.0, "winkel": 10}],
    lx=5, ly=1.5, nx=400, ny=120, re=1000, L_ref=1.0, t_end=8)
live = LivePlotter(case, field="vorticity")
grund = case.sim.run(case.t_end, callback=live, callback_every=50)
report.create_report(case, "ergebnisse/naca0012", grund)
```

Ein vollständiges Beispiel steht in [`beispiele/skript_beispiel.py`](beispiele/skript_beispiel.py).

---

## Ausgabe

| Datei | Inhalt |
|---|---|
| `felder.png` | Geschwindigkeitsbetrag mit Stromlinien, Wirbelstärke, Druck, Stromfunktion |
| `konvergenz.png` | Residuum, max. Divergenz (Massenerhaltung), kinetische Energie über der Zeit |
| `validierung_mittellinien.png` | *Cavity*: u- und v-Profile auf den Mittellinien im Vergleich mit Ghia et al. |
| `kraftbeiwerte.png` | *Zylinder und Hindernisse*: c_D(t), c_L(t), Spektrum des Auftriebs, Strouhal-Zahl, Referenzbereiche |
| `profile.png` | *Kanal*: Geschwindigkeitsprofile entlang des Kanals, analytisches Profil, Druckverlauf |
| `stufe_auswertung.png` | *Stufe*: Wandschubspannung, Wiederanlegepunkt, Geschwindigkeitsprofile |
| `animation.gif` | Animation des gewählten Feldes (optional) |
| `ergebnis.npz` | Rohdaten: u, v, p, Fluidmaske, alle Zeitverläufe |
| `zusammenfassung.json` | Parameter und Kennzahlen (Fehler gegenüber Referenz, c_D, c_L, St, x_r …) |

Das Live-Fenster zeigt das gewählte Feld (optional mit Stromlinien), den
Konvergenzverlauf und bei Hindernissen die Kraftbeiwerte. Ohne Bildschirm
(z. B. auf einem Server) wird das automatisch erkannt; dann entstehen nur
die Diagramme und auf Wunsch ein GIF.

---

## Physik und Numerik

### Gleichungen

Dimensionslos, mit Dichte ρ = 1 und ν = U·L/Re:

$$\frac{\partial \mathbf u}{\partial t} + (\mathbf u\cdot\nabla)\mathbf u = -\nabla p + \nu\,\Delta\mathbf u, \qquad \nabla\cdot\mathbf u = 0$$

### Diskretisierung

- **MAC-Gitter** (versetzt): Der Druck liegt in den Zellmitten, u auf den
  senkrechten und v auf den waagerechten Zellflächen. Dadurch entstehen
  keine Schachbrettmuster im Druck, und die diskrete Divergenzfreiheit gilt
  exakt (bis auf Maschinengenauigkeit, ca. 1e-14).
- **Diffusion**: zentrale Differenzen, 2. Ordnung.
- **Konvektion**: Mischung aus zentralen Differenzen und Donor-Cell-Upwind
  mit dem Faktor γ ∈ [0, 1]. Bei `gamma = auto` wird γ für AB2 aus der
  Zell-Reynolds-Zahl bestimmt (gut aufgelöst: γ klein, nahe 2. Ordnung)
  und für Euler aus der Stabilitätsbedingung nach Hirt.
- **Zeitintegration**: Adams-Bashforth 2. Ordnung (Standard) oder Euler
  explizit, mit adaptivem Zeitschritt aus CFL- und Diffusionsbedingung und
  dem Sicherheitsfaktor τ.

### Projektionsmethode (je Zeitschritt)

1. Randbedingungen setzen (Gebietsrand und Hindernisse)
2. Vorläufige Geschwindigkeit: **F** = **u** + Δt·**H**(**u**), mit Konvektion und Diffusion in **H**
3. Druck-Poisson-Gleichung lösen: Δp = ∇·**F** / Δt
4. Projektion: **u** = **F** − Δt·∇p

Für die Druckgleichung wird die dünn besetzte Matrix einmal aufgestellt.
Danach wird sie entweder **einmal LU-zerlegt**, sodass jeder Zeitschritt
nur noch eine Vorwärts-Rückwärts-Substitution kostet, oder iterativ mit
**CG** (Jacobi-Vorkonditionierung) gelöst. Zusammenhangsgebiete ohne
Druck-Randwert (geschlossene Cavity) werden automatisch erkannt und
regularisiert.

### Randbedingungen

| Art | u_normal | u_tangential | Druck |
|---|---|---|---|
| Haftende Wand (`noslip`) | 0 | U_Wand (über Geisterzelle) | Neumann |
| Reibungsfreie Wand (`slip`) | 0 | ∂u_t/∂n = 0 | Neumann |
| Einströmung (`inflow`) | Block- oder Parabelprofil | 0 | Neumann |
| Ausströmung (`outflow`) | aus der Druckkorrektur | ∂u_t/∂n = 0 | p = 0 (Dirichlet) |

**Hindernisse** werden als Treppenstufen auf dem Gitter abgebildet: Eine
Zelle gilt als Festkörper, wenn ihr Mittelpunkt im Körper liegt. Die
Haftbedingung wird über gespiegelte Geisterwerte erfüllt. Eingeschlossene
Fluidbereiche entfernt das Programm automatisch. Die **Kräfte** ergeben sich
aus Druck und Wandschubspannung über alle Fluid-Festkörper-Flächen.

---

## Validierungsergebnisse

Gerechnet mit den Standardeinstellungen (`gamma = auto`, AB2, τ = 0,5):

| Fall | Gitter | Ergebnis | Referenz |
|---|---|---|---|
| Cavity Re = 100 | 64² | max. Abweichung u: 0,003, v: 0,008 | Ghia et al. |
| Cavity Re = 1000 | 128² | max. Abweichung u: 0,023, v: 0,023 | Ghia et al. |
| Kanal Re = 20 | 20 Zellen/H | Profilfehler 0,4 %, dp/dx = −0,597 | −0,600 (analytisch) |
| Zylinder Re = 20 | 220×41 | c_D = 5,47, c_L = 0,0107 | 5,57–5,59 / 0,0104–0,0110 |
| Zylinder Re = 100 | 440×82 | c_D,max ≈ 3,15–3,25, St ≈ 0,28–0,30 | 3,22–3,24 / 0,295–0,305 |

Hinweise:

- Die Amplitude des Auftriebs beim Zylinder mit Re = 100 (c_L,max ≈ 0,75–0,86
  statt 0,99–1,01) wird mit der Treppenstufen-Geometrie und D ≈ 20 Zellen
  unterschätzt. Feinere Gitter (*fein*: 660×123) und ein kleiner
  Upwind-Anteil (z. B. γ = 0,1) bringen sie näher an die Referenz.
- Bei den Ghia-Daten für Re = 400 passt ein Tabellenwert (v bei x = 0,9063)
  nicht zu seinen Nachbarpunkten. Er dominiert dort die maximale v-Abweichung.

## Tests

```bash
python -m pytest -q
```

Die Tests prüfen unter anderem:

- Divergenzfreiheit und Massenerhaltung (auch mit Hindernissen und Ausströmrand)
- Ghia-Vergleich der Cavity sowie Poiseuille-Profil und Druckgefälle
- c_D/c_L beim Zylinder mit Re = 20 und die Rezirkulation hinter der Stufe
- Geometrien (Kreisfläche, NACA-Dicke und -Fläche, Bildimport)
- Erkennung von Instabilitäten
- die interaktive Eingabe (simulierte Tastatureingaben) und die Kommandozeile

## Projektstruktur

```
cfd2d/
├── __main__.py       Einstieg für "python -m cfd2d"
├── cli.py            Kommandozeilenargumente
├── interactive.py    Bedienung per Eingabe (Menüs, Eingabeprüfung)
├── cases.py          vordefinierte Testfälle
├── grid.py           MAC-Gitter und Hindernismaske
├── geometry.py       Hindernisse: Kreis, Ellipse, Rechteck, NACA, Polygon, Bild
├── boundary.py       Randbedingungen am Gebietsrand
├── kernels.py        Numba-Kernel (Impulsgleichungen, Hindernis-Randbedingungen)
├── pressure.py       Druck-Poisson-Löser (LU oder CG)
├── solver.py         Zeitschleife, Projektionsmethode, Auswertungsgrößen
├── validation.py     Referenzdaten (Ghia, Poiseuille, Schäfer-Turek), St, x_r
├── visualization.py  Live-Visualisierung und GIF-Aufnahme
└── report.py         Diagramme, Kennzahlen, Datenexport
beispiele/skript_beispiel.py
tests/test_cfd2d.py
```

## Literatur

- U. Ghia, K. N. Ghia, C. T. Shin: *High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method*, J. Comput. Phys. 48 (1982) 387–411.
- M. Schäfer, S. Turek et al.: *Benchmark computations of laminar flow around a cylinder*, Notes Numer. Fluid Mech. 52 (1996) 547–566.
- D. K. Gartling: *A test problem for outflow boundary conditions – flow over a backward-facing step*, Int. J. Numer. Meth. Fluids 11 (1990) 953–967.
- M. Griebel, T. Dornseifer, T. Neunhoeffer: *Numerische Simulation in der Strömungsmechanik*, Vieweg (1995).
- A. J. Chorin: *Numerical solution of the Navier-Stokes equations*, Math. Comp. 22 (1968) 745–762.
