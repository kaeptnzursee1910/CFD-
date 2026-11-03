# Roadmap: Erweiterung zu einem OpenFOAM-ähnlichen 2D-Programm

Ziel: Das bestehende 2D-Programm (strukturiertes MAC-Gitter, interaktive Bedienung)
um die wichtigsten Funktionen eines allgemeinen CFD-Pakets erweitern. Als Vorbild
dienen die OpenFOAM-Löser und -Hilfsprogramme, die sich in 2D sinnvoll umsetzen lassen.

Arbeitsweise: Jede Etappe wird einzeln committet und gepusht. Nach einer
Unterbrechung (z. B. Nutzungslimit) mit „weiter“ an der ersten offenen Etappe fortsetzen.

| Etappe | Inhalt | OpenFOAM-Vorbild | Status |
|---|---|---|---|
| 1 | Variable (effektive) Viskosität, periodische Ränder in x, Volumenkraft, Steuerung des Volumenstroms | `cyclic`, `meanVelocityForce` | erledigt |
| 2 | Allgemeiner Skalartransport (TVD-Schemata: Upwind, van Leer, Minmod, Superbee; Randbedingungen fest, Gradient null, Fluss) | `scalarTransportFoam`, `fvSchemes` | erledigt |
| 3 | Temperatur mit Boussinesq-Auftrieb; Testfälle: beheizte Cavity (de Vahl Davis), Rayleigh-Bénard | `buoyantBoussinesqPimpleFoam` | erledigt (Nu bei Ra = 1e4: 2,255 statt 2,243) |
| 4 | Turbulenzmodelle: Smagorinsky (LES), k-ω SST (RANS), Wandabstand; Testfall: turbulenter Kanal | `momentumTransport` | erledigt (Re_τ = 180: c_f 6,5 % unter Dean) |
| 5 | Zweiphasenströmung (VOF, Dichtesprung): Dammbruch | `interFoam` | erledigt |
| 6 | Function Objects: Sonden, Linienproben, VTK-Export für ParaView, Speicherstände und Neustart, Case speichern und laden (JSON) | `probes`, `sample`, `foamToVTK`, Zeitverzeichnisse | erledigt |
| 7 | Integration ins interaktive Menü, Tests, Dokumentation | – | offen |
