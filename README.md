# cfd2d: 2D-CFD in Python, nach dem Vorbild von OpenFOAM

Ein 2D-Strömungsprogramm in Python für
- **inkompressible Strömungen**,
- **Wärmeübertragung und natürliche Konvektion**,
- **Turbulenz (RANS/LES)**,
- **passive Skalare** und
- **Zweiphasenströmungen (VOF)**.

Technische Grundlage:
- Finite-Volumen-Verfahren auf einem versetzten **MAC-Gitter**
- **Projektionsmethode**
- **Numba**-beschleunigte Kernel und dünn besetzte Drucklöser über **SciPy**

Bedienung: **interaktiv per Eingabe** in der Konsole, mit **Live-Visualisierung** und automatisch erzeugten Diagrammen.

Den Funktionsumfang von OpenFOAM erreicht das Programm nicht. Es ist bewusst auf
2D, strukturierte Gitter und explizite Zeitintegration beschränkt (siehe
[Grenzen](#grenzen)). Es bildet aber die wichtigsten Löser und Werkzeuge von
OpenFOAM nach, die sich in 2D sinnvoll umsetzen lassen:

| OpenFOAM | cfd2d |
|---|---|
| `icoFoam`, `pimpleFoam` (laminar) | Grundlöser: Navier-Stokes, Projektionsmethode, AB2 oder Euler |
| `simpleFoam`/`pimpleFoam` + `kOmegaSST` | RANS k-ω SST (Menter 2003), Low-Re bis an die Wand |
| LES `Smagorinsky` | Smagorinsky mit Wandbegrenzung des Mischungswegs |
| `scalarTransportFoam` | passiver Skalar mit TVD-Schemata |
| `buoyantBoussinesqPimpleFoam` | Temperatur mit Boussinesq-Auftrieb |
| `interFoam` | VOF: Wasser/Luft, Interface-Kompression, Oberflächenspannung (CSF) |
| `fvSchemes` | Upwind, van Leer, Minmod, Superbee, MC, linear; Upwind-Anteil γ für den Impuls |
| `fvSolution` | Drucklöser: LU direkt oder CG |
| Randbedingungen `fixedValue`, `zeroGradient`, `fixedGradient`, `noSlip`, `slip`, `cyclic` | haftend (auch bewegt), reibungsfrei, Einströmung (Block/Parabel), Ausströmung (p = 0), periodisch; Skalare: fester Wert, Gradient null, fester Fluss |
| `meanVelocityForce` | Volumenstromregelung in periodischen Kanälen |
| `forceCoeffs` | c_D, c_L, Strouhal-Zahl |
| `probes`, `sample`, `fieldAverage` | Sonden, Linienproben, Zeitmittel und RMS |
| `foamToVTK` | VTK-Export (binär) mit Zeitserie für ParaView |
| Zeitverzeichnisse und Neustart | Speicherstände (bitgenaue Fortsetzung) |
| Case-Verzeichnis (`system/`, `constant/`, `0/`) | Case-Datei `case.json` (speichern, laden, ändern) |
| `blockMesh`/`snappyHexMesh` | kartesisches Gitter mit Hindernissen: Kreis, Ellipse, Rechteck, NACA, Polygon, **Bilddatei** |

---

## Installation

```bash
pip install -r requirements.txt
```

Benötigt werden Python ≥ 3.10, NumPy, SciPy, Numba, Matplotlib und Pillow.
Für das Live-Fenster braucht Matplotlib ein GUI-Backend, zum Beispiel Tk
oder `pip install PyQt5`.

## Start

```bash
python -m cfd2d
```

```
Hauptmenü: Testfall wählen
   1) Lid-Driven Cavity (Validierung nach Ghia et al.)
   2) Zylinderumströmung, Kármánsche Wirbelstraße (Schäfer-Turek)
   3) Kanalströmung, Poiseuille (analytische Lösung)
   4) Rückwärtsgewandte Stufe (Ablösung und Wiederanlegung)
   5) Eigene Hindernisse / freie Randbedingungen (Kreis, NACA, Polygon, Bild)
   6) Beheizte Cavity: natürliche Konvektion (de Vahl Davis)
   7) Rayleigh-Bénard-Konvektion (periodisch)
   8) Turbulenter Kanal, RANS k-ω SST (Wandgesetz, Dean)
   9) Dammbruch: Zweiphasenströmung Wasser/Luft (VOF)
  10) Case aus JSON-Datei laden
  11) Rechnung aus Speicherstand fortsetzen
  12) Beenden
```

Das Programm führt Schritt für Schritt durch die Einstellungen. **Enter**
übernimmt den Standardwert in eckigen Klammern, Dezimalzahlen dürfen mit
Komma oder Punkt eingegeben werden, und `q` führt zurück zum Hauptmenü.

1. **Testfall und Parameter**: Reynolds- oder Rayleigh-Zahl, Gitter, Geometrie, Simulationszeit
2. **Zusatzphysik** (Fälle 1–5):
   - Turbulenzmodell (laminar, k-ω SST, Smagorinsky)
   - Temperatur: Einström-, Hindernis- und Wandtemperaturen, optional Auftrieb
   - passiver Skalar, optional nur in einem Streifen des Einlasses
3. **Numerik** (optional): Zeitschema, Upwind-Anteil, Zeitschritt-Sicherheitsfaktor, Drucklöser
4. **Function Objects** (optional): Sonden, Linienproben, Zeitmittel, VTK-Export, Speicherstände
5. **Visualisierung**: Live-Fenster, Feld (z. B. |u|, ω, p, T, α, k, ν_t), Stromlinien, GIF
6. **Rechnung** mit Live-Anzeige. Strg+C oder das Schließen des Fensters beendet sie; die Auswertung wird trotzdem erstellt.

Jede Rechnung legt im Ausgabeordner automatisch `case.json` und
`checkpoint.npz` ab. Über **„Case laden“** lässt sie sich wiederholen oder
abwandeln, über **„Rechnung fortsetzen“** läuft sie bis zu einer neuen
Endzeit weiter.

### Kommandozeile (ohne Rückfragen)

```bash
python -m cfd2d --fall heated_cavity --ra 1e5 --nx 80
python -m cfd2d --fall cylinder --re 1000 --turbulenz kOmegaSST --live --feld nu_t
python -m cfd2d --fall dam_break --nx 96 --live --feld alpha --gif
python -m cfd2d --case ergebnisse/zylinder/case.json --live
python -m cfd2d --help
```

### Als Bibliothek

```python
from cfd2d import cases, report

spec = {"fall": "cylinder",
        "parameter": {"re": 200, "nx": 220, "ny": 41, "t_end": 8},
        "turbulenz": {"modell": "kOmegaSST"},
        "temperatur": {"alpha": 7e-4, "T_obstacle": 1.0},
        "function_objects": [{"typ": "sonden", "punkte": [[0.6, 0.2]]},
                             {"typ": "vtk", "intervall": 0.5}]}
case = cases.build_case(spec)
grund = case.sim.run(case.t_end)
report.create_report(case, "ergebnisse/zylinder_warm", grund)
```

---

## Testfälle und Validierung

| Fall | Gitter | Ergebnis | Referenz |
|---|---|---|---|
| Cavity Re = 100 | 64² | max. Abweichung u: 0,003, v: 0,008 | Ghia et al. (1982) |
| Cavity Re = 1000 | 128² | max. Abweichung u: 0,023, v: 0,023 | Ghia et al. |
| Kanal Re = 20 | 20 Zellen/H | Profilfehler 0,4 %, dp/dx = −0,597 | −0,600 (analytisch) |
| periodischer Kanal (Volumenstrom geregelt) | 8×20 | Antriebskraft 0,597 | 12ν U/H² = 0,600 |
| Zylinder Re = 20 | 220×41 | c_D = 5,47, c_L = 0,0107 | 5,57–5,59 / 0,0104–0,0110 (Schäfer & Turek) |
| Zylinder Re = 100 | 440×82 | c_D,max ≈ 3,15–3,25, St ≈ 0,28–0,30 | 3,22–3,24 / 0,295–0,305 |
| Beheizte Cavity Ra = 10⁴ | 48² | Nu = 2,255, u_max = 16,17, v_max = 19,61 | 2,243 / 16,18 / 19,62 (de Vahl Davis 1983) |
| Beheizte Cavity Ra = 10⁵ | 80² | Nu = 4,548, u_max = 34,78, v_max = 68,37 | 4,519 / 34,73 / 68,59 |
| Rayleigh-Bénard | 24 Zellen/H | Ra = 1500: Nu = 1 (in Ruhe), Ra = 10⁴: Konvektionsrollen, Nu = 2,70 | Ra_krit = 1708 |
| turbulenter Kanal, k-ω SST, Re_τ ≈ 180 | y⁺₁ ≈ 1 | c_f = 0,00787, Re_τ = 176,9 | Dean-Korrelation: 0,00842 |
| Dammbruch (VOF) | 64²–96² | Volumen erhalten (relativer Fehler ≈ 1e-16), Front unterhalb der Ritter-Schranke | qualitativ |
| Hydrostatik (VOF) | 20² | Scheinströmungen < 1e-6 m/s, p_Boden = ρ g h (±2 %) | analytisch |

Hinweise:

- **Auftrieb beim Zylinder:** Bei Re = 100 wird c_L,max mit der
  Treppenstufen-Geometrie unterschätzt (0,75–0,86 statt etwa 1,0).
- **Ghia-Tabelle:** Bei Re = 400 passt ein Tabellenwert (v bei x = 0,9063)
  nicht zu seinen Nachbarpunkten.
- **k-ω SST bei niedrigem Re_τ:** Das Modell liegt im logarithmischen Bereich
  etwas über dem Log-Gesetz; das ist bei niedrigen Re_τ typisch.

## Ausgabe

Diese Dateien entstehen bei jeder Rechnung:

| Datei | Inhalt |
|---|---|
| `felder.png` | Geschwindigkeit mit Stromlinien, Wirbelstärke, Druck, Stromfunktion |
| `felder_modelle.png` | T, Skalar, k, ω, ν_t, α, ρ (je nach Modell) |
| `konvergenz.png`, `monitore.png` | Residuum, Divergenz, kinetische Energie, Modell-Monitore (Nu, Volumenfehler …) |
| `ergebnis.npz`, `zusammenfassung.json` | Rohdaten bzw. Kennzahlen mit Referenzwerten |
| `case.json`, `checkpoint.npz` | Case-Datei bzw. Speicherstand für den Neustart |

Je nach Fall kommen eigene Auswertungen hinzu:

| Fall | Dateien |
|---|---|
| Cavity | `validierung_mittellinien.png` |
| Zylinder und eigene Hindernisse | `kraftbeiwerte.png` |
| Kanal | `profile.png` |
| Stufe | `stufe_auswertung.png` |
| Beheizte Cavity | `waermeuebergang.png` |
| Rayleigh-Bénard | `rayleigh_benard.png` |
| Turbulenter Kanal | `wandgesetz.png` |
| Dammbruch | `dammbruch.png` |

Die Function Objects schreiben zusätzlich:

| Function Object | Dateien |
|---|---|
| Sonden | `sonden.csv`, `sonden.png` |
| Linienproben | `linie_*.csv`, `linie_*.png` |
| Zeitmittel | `mittelwerte.npz`, `mittelwerte.png` |
| VTK-Export | `vtk/feld_*.vtk` und `vtk/feld.vtk.series` (in ParaView öffnen) |
| Speicherstände | `checkpoint.npz` |
| Live-Ansicht | `animation.gif` |

---

## Numerik

### Grundlöser

Die Gleichungen werden dimensionslos gelöst, beim Dammbruch in SI-Einheiten:

$$\frac{\partial \mathbf u}{\partial t} + (\mathbf u\cdot\nabla)\mathbf u = -\frac{1}{\rho}\nabla p + \frac{1}{\rho}\nabla\cdot\big(\mu_\text{eff}(\nabla\mathbf u + \nabla\mathbf u^T)\big) + \mathbf f, \qquad \nabla\cdot\mathbf u = 0$$

- **MAC-Gitter:** Die diskrete Divergenzfreiheit gilt exakt, in der Praxis bis etwa 1e-14.
- **Konvektion:** zentral mit Upwind-Anteil γ. Bei `gamma = auto` wird γ für
  AB2 aus der Zell-Reynolds-Zahl bestimmt und für Euler nach Hirt.
- **Diffusion:** konstante Viskosität als Laplace-Form, variable
  (turbulent oder zweiphasig) in Spannungsform.
- **Zeitintegration:** Adams-Bashforth 2 oder Euler mit adaptivem Zeitschritt.
  Die Grenzen ergeben sich aus CFL-Zahl, Diffusion, Skalaren, Kapillarkräften
  und Schwerewellen.
- **Projektion:** `div((1/ρ) grad p) = div(u*)/Δt`. Bei konstanter Dichte wird
  die Matrix einmal LU-zerlegt, bei VOF in jedem Schritt neu aufgebaut.
- **Volumenkräfte und Volumenstromregelung** werden explizit (nicht über AB2)
  behandelt. Sonst entsteht mit dem Regler eine ungedämpfte Schwingung mit
  zwei Schritten Periode.

### Modelle (`cfd2d/models.py`, `turbulence.py`, `vof.py`)

Alle Modelle docken über dieselben Hooks an den Löser an:
`pre_momentum`, `momentum_source`, `post_step`, `dt_limit`, `monitors`, `fields`.

- **Skalartransport:** konservative Finite-Volumen-Form mit TVD-Limitern,
  expliziter Euler-Schritt und punktweise impliziten Senken.
- **Boussinesq:** f = −β(T − T_ref) g; für die thermischen Testfälle in
  Freifall-Skalierung.
- **k-ω SST:**
  - Mischfunktionen F1 und F2, Begrenzung der Produktion
  - Wandabstand per Distanztransformation (auch bei periodischen Rändern)
  - k = 0 an Wänden, ω in den wandnahen Zellen = 6ν/(β₁y²)
  - Einlasswerte aus Turbulenzgrad und Viskositätsverhältnis
- **Smagorinsky:** ν_t = l²|S| mit l = min(C_s Δ, κ d).
- **VOF:**
  - van-Leer-Transport mit Kompressionsfluss c_α|u| n α(1−α)
  - Unterteilung des Zeitschritts für die Courant-Zahl von α
  - gemischte Stoffwerte ρ und μ
  - Oberflächenspannung nach dem CSF-Modell

## Grenzen

- **Nur 2D und kartesische, äquidistante Gitter.** Krumme Wände werden als
  Treppenstufen dargestellt. Ein unstrukturiertes 3D-Gitter wie bei OpenFOAM
  gibt es nicht.
- **Explizite Zeitintegration.** Für sehr feine Wandauflösung oder lange
  stationäre RANS-Rechnungen ist der Zeitschritt klein; ein SIMPLE-Löser für
  stationäre Rechnungen ist nicht enthalten.
- **LES in 2D ist physikalisch nur eingeschränkt sinnvoll**, weil echte
  Turbulenz dreidimensional ist.
- **VOF:** Die Beschränktheit von α wird nach dem Transport per Abschneiden
  gesichert, nicht über MULES. Der Volumenfehler wird als Monitor ausgegeben.
- **Nicht enthalten:**
  - kompressible Strömungen
  - Verbrennung
  - Kopplung von Wärmeleitung in Fluid und Festkörper (CHT)
  - bewegte Gitter
  - Parallelisierung mit MPI

## Tests

```bash
python -m pytest -q
```

45 Tests decken diese Bereiche ab:

- **Grundlöser:** Divergenzfreiheit, Massenerhaltung, periodische Ränder und Volumenstromregelung
- **Validierung:** Ghia, Poiseuille, Zylinder, de Vahl Davis, Einsetzen der Rayleigh-Bénard-Konvektion, SST gegen Dean
- **Skalare und VOF:** Beschränktheit und Erhaltung aller TVD-Schemata, Hydrostatik und Volumenerhaltung bei VOF
- **Neustart:** bitgenau, auch mit SST, Skalar und VOF
- **Ausgabe:** Aufbau der VTK-Dateien
- **Bedienung:** die interaktive Eingabe (simulierte Tastatureingaben, inklusive Fortsetzen) und die CLI

## Projektstruktur

```
cfd2d/
├── __main__.py, cli.py   Einstieg, Kommandozeile
├── interactive.py        Bedienung per Eingabe (Menüs, Eingabeprüfung)
├── cases.py              Testfälle, Zusatzphysik, Case-Dateien (build/save/load)
├── grid.py, geometry.py  MAC-Gitter, Hindernisse (Kreis, Ellipse, Rechteck, NACA, Polygon, Bild)
├── boundary.py           Randbedingungen des Gebiets (inkl. periodisch)
├── kernels.py            Numba-Kernel: Impuls, Hindernis-Randbedingungen, TVD-Skalartransport
├── pressure.py           Druck-Poisson-Löser (LU/CG, variabler Koeffizient)
├── solver.py             Zeitschleife, Projektion, Modell-Hooks, Auswertungsgrößen
├── models.py             Skalarfeld, passiver Skalar, Boussinesq-Temperatur
├── turbulence.py         k-ω SST, Smagorinsky, Wandabstand
├── vof.py                Zweiphasen-VOF (interFoam-ähnlich)
├── functions.py          Sonden, Linienproben, Zeitmittel, VTK, Speicherstände
├── validation.py         Referenzdaten (Ghia, Schäfer-Turek, de Vahl Davis, Dean …)
├── visualization.py      Live-Visualisierung und GIF
└── report.py             Diagramme, Kennzahlen, Datenexport
beispiele/skript_beispiel.py
tests/
ROADMAP.md                Etappenplan der Erweiterung
```

## Literatur

- U. Ghia, K. N. Ghia, C. T. Shin, J. Comput. Phys. 48 (1982) 387–411.
- M. Schäfer, S. Turek et al., Notes Numer. Fluid Mech. 52 (1996) 547–566.
- G. de Vahl Davis, Int. J. Numer. Meth. Fluids 3 (1983) 249–264.
- F. R. Menter, M. Kuntz, R. Langtry: *Ten years of industrial experience with the SST turbulence model*, Turbulence, Heat and Mass Transfer 4 (2003).
- R. B. Dean, J. Fluids Eng. 100 (1978) 215–223.
- J. U. Brackbill, D. B. Kothe, C. Zemach, J. Comput. Phys. 100 (1992) 335–354.
- J. C. Martin, W. J. Moyce, Phil. Trans. R. Soc. A 244 (1952) 312–324.
- M. Griebel, T. Dornseifer, T. Neunhoeffer: *Numerische Simulation in der Strömungsmechanik*, Vieweg (1995).
- D. K. Gartling, Int. J. Numer. Meth. Fluids 11 (1990) 953–967.
