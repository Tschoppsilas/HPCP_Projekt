# Notizen — Lattice-Boltzmann-Optimierung (Projekt 04)

Arbeitsnotizen, die später zum Write-up werden. Wird laufend ergänzt.

## 0. Setup

- Einzelarbeit, 1 Optimierungsstrategie gefordert.
- Strategie: Numba (`@njit`, später `@njit(parallel=True)` + `prange`), direkt aufbauend auf
  `01_Python/03_Numba.ipynb` aus dem Kurs.
- Repo-Struktur: `baseline/` bleibt unangetastet (Referenz), in `optimized/lbm_d2q9_numba.py`
  passiert die Optimierung, `bench.py` führt beides aus, validiert und loggt nach
  `results/benchmark_log.csv`.
- Zwei Maschinen im Einsatz: lokaler Laptop (Windows, venv) und FHNW-`pub030`-JupyterHub
  (`calc-g-jhub`, ebenfalls venv). Finale Zahlen dort nochmal sauber nachmessen.
- Kernanzahl: Laptop 22 logische Prozessoren (`$env:NUMBER_OF_PROCESSORS`), Cluster (pub030,
  calc-g-jhub) 32 Kerne (`nproc`).

## 1. Baseline-Profiling (cProfile, cylinder, 400x100, 500 Steps)

```
94079 function calls in 2.696 seconds

tottime  cumtime  wo
 0.837    2.696   run()-Körper selbst (Randbedingungen, feq-Dirichlet-Überschreibung,
                   Kollision f - omega*(f-feq), und die Bounce-back-Schleife:
                   `for i in range(Q): fpost[i][bounce] = f[OPP[i]][bounce]`)
 1.053    1.057   equilibrium()   <- grösste Einzelfunktion, 501x aufgerufen
 0.389    0.409   macroscopic()   501x aufgerufen
 0.127    0.393   stream()        500x aufgerufen (0.263s davon ist np.roll selbst, 9000 Calls)
```

Grobe Aufteilung der ~2.7 s: ~31% run()-Körper (Boundary/Kollision/Bounce-back), ~39%
equilibrium(), ~15% macroscopic(), ~15% stream() (davon ~65% np.roll). Grössenordnungsmässig
konsistent mit der späteren lokalen Baseline-Zeit bei `small` (2000 Steps, ~10.45 s ->
~2.6 s pro 500 Steps).

**Anmerkung:** Das deckt sich nicht ganz mit der "Streaming + Kollision, bandbreitenlimitiert"-
Einordnung aus dem obersten Projekt-Readme — equilibrium() und die Bounce-back-Schleife kosten
mindestens genauso viel wie stream(). Arbeitshypothese: ein grosser Teil davon ist
Python-Loop-/Temporär-Array-Overhead (9 Richtungen, bei jedem Aufruf frische Arrays), nicht reine
Speicherbandbreite. Genau das sollte ein fusionierter Numba-Kernel angehen.

## 2. Baseline-Zeiten, gleicher Case/Size, unterschiedliche Maschinen

| Maschine                  | Case     | Size  | Laufzeit | MLUPS  |
|---------------------------|----------|-------|----------|--------|
| Laptop (Windows)          | cylinder | small | 10.45 s  | 7.656  |
| pub030 (calc-g-jhub)      | cylinder | small | 5.58 s   | 14.344 |

Die Cluster-Baseline ist bei reinem NumPy schon ~1.9x schneller als der Laptop — andere CPU/
NumPy-Build. Wichtig: das heisst, vergleichbar sind Speedup-*Faktoren*, nicht absolute Laufzeiten
über Maschinen hinweg — und die finalen Zahlen fürs Write-up sollten von einer konsistenten
Maschine (pub030) kommen, nicht gemischt.

## 3. Gewählte Strategie

**Numba `@njit(parallel=True)` mit `prange`, am Ende Fusion von Macroscopic + Kollision +
Bounce-back + Streaming zu einem einzigen Pass pro Zelle.**

Warum: baut direkt auf `01_Python/03_Numba.ipynb` auf (njit, `@njit(parallel=True)` + `prange`).
Kein GPU-/Treiber-Setup nötig. Sollte die profilten Hotspots angehen (Python-Loop-Overhead,
temporäre Arrays, die volle Kopie in stream()).

## 4. Implementierungs-Log

### Schritt 1 — `@njit` auf `equilibrium()`, unverändert sonst

Nur `from numba import njit` + `@njit` über `equilibrium()` gesetzt, Funktionskörper unverändert
gelassen (weiterhin ein Python-Loop über Q=9 Richtungen, mit vollen NumPy-Array-Ausdrücken pro
Iteration, z. B. `cu = 3.0 * (C[i,0]*ux + C[i,1]*uy)`).

Ergebnis — Korrektheit: `validate.py` meldet **bit-identische** Ausgabe (`max rel err 0.000e+00`
für ux, uy, rho) auf beiden Maschinen. Macht Sinn: gleiche Operationen, gleiche Reihenfolge, noch
kein Umbau.

Ergebnis — Performance, überraschend und maschinenabhängig:

| Maschine  | Baseline               | Optimiert (njit equilibrium)  | Speedup |
|-----------|-------------------------|--------------------------------|---------|
| Laptop    | 10.45 s / 7.656 MLUPS   | 6.59 s / 12.14 MLUPS           | **1.59x schneller** |
| pub030 (1. Lauf) | 5.58 s / 14.344 MLUPS | 6.54 s / 12.233 MLUPS      | **0.85x (langsamer!)** |
| pub030 (2. Lauf) | Baseline wiederverwendet | 6.37 s / 12.551 MLUPS   | weiterhin langsamer als Baseline |

**Zweiter Cluster-Lauf zur Kontrolle:** 6.37 s statt 6.54 s — leichte Schwankung (~2.6%), aber
konsistent immer noch deutlich über der Baseline-Zeit (5.58 s). Das spricht dagegen, dass
Konkurrenz durch andere Nutzer auf dem geteilten JupyterHub-Node die Hauptursache ist — der
Effekt ist stabil reproduzierbar, keine einmalige Störung.

**Hypothese (Schritt 1):** `@njit` allein ändert nicht, *was* der Code berechnet, nur wie er
kompiliert wird. `equilibrium()` besteht weiterhin aus NumPy-Array-Ausdrücken innerhalb eines
Python-Loops über 9 Richtungen — Numba kompiliert das zwar, aber es bleibt im Kern "auf ganzen
Arrays mit Temporären rechnen", was direkt gegen NumPys eigene SIMD-vektorisierte
Elementweise-Operationen antritt. Auf einer starken Server-CPU mit gut vektorisiertem NumPy-Build
(pub030) gewinnt NumPy diesen Wettkampf offenbar klar. Auf dem Laptop war NumPys Overhead pro
Aufruf/Temporär-Array offenbar proportional grösser, deshalb half das Wegschneiden via Numba dort.

**Konsequenz:** Der eigentliche Numba-Vorteil sollte erst kommen, wenn `equilibrium()` (und der
Rest) zu einem **expliziten Scalar-Loop über die Gitterzellen** umgebaut wird (`for x ... for
y ...`, alle 9 Richtungen pro Zelle mit reinen Zahlen statt Zwischen-Arrays) statt die
NumPy-Array-Ausdrucks-Version einfach zu dekorieren.

### Schritt 2 — Umbau zu explizitem Scalar-Loop + `prange` über die Gitterzellen

`equilibrium()` umgebaut: `nx, ny = rho.shape` am Anfang, danach `for x in prange(nx): for y in
range(ny): ...` mit skalarer Rechnung pro Zelle (kein Array mehr, `usqr` und `cu` jetzt mit
`[x, y]` indiziert statt auf dem ganzen Array). Decorator zu `@njit(parallel=True)` geändert,
`from numba import njit, prange`.

Zwei Anfängerfehler unterwegs (beide behoben): `import prange` statt `from numba import prange`
(gleicher Fehler wie bei `njit` in Schritt 1 — `prange`/`njit` sind keine eigenen Packages,
sondern leben in `numba`), und `usqr` zunächst noch ohne `[x, y]`-Indizierung (rechnete auf dem
ganzen Array statt auf der aktuellen Zelle — hätte den ganzen Sinn des Scalar-Loop-Umbaus
zunichtegemacht).

Ergebnis — Korrektheit: weiterhin `OK` in `validate.py` auf beiden Maschinen.

Ergebnis — Performance, nochmal überraschend, diesmal umgekehrt zu Schritt 1:

| Maschine | Baseline | Schritt 1 (njit, Array-Ausdrücke) | Schritt 2 (prange, Scalar-Loop) |
|----------|----------|-------------------------------------|-----------------------------------|
| Laptop   | 10.45 s  | 6.59 s -> **1.59x schneller**       | 7.57 s / 10.562 MLUPS -> **1.38x schneller** (schlechter als Schritt 1!) |
| Cluster  | 5.58 s   | 6.54 s -> **0.85x langsamer**        | 3.80 s / 21.025 MLUPS -> **1.47x schneller** (viel besser als Schritt 1!) |

**Erste Hypothese (verworfen):** paralleler Dispatch-Overhead pro Aufruf (equilibrium() wird
2000x pro Lauf aufgerufen) trifft auf wenige Kerne beim Laptop und viele Kerne auf dem Cluster.
Kernanzahl nachgemessen: Laptop 22 logische Prozessoren, Cluster 32 Kerne -- das ist kein grosser
Unterschied (Faktor ~1.45), erklärt einen so starken Umschwung nicht überzeugend. Diese Hypothese
wird daher nicht als Haupterklärung übernommen.

**Überarbeitete, ehrlichere Hypothese:** wahrscheinlich nicht (nur) die reine Kernzahl, sondern
*wie* die Kerne beschaffen sind und wie viel Speicherbandbreite dahintersteckt:
- Der Laptop hat vermutlich eine heterogene CPU (schnelle "Performance"- und schwächere
  "Efficiency"-Kerne, typisch bei modernen Intel-Laptop-CPUs) -- `NUMBER_OF_PROCESSORS` zählt
  beide gleich, sie liefern aber nicht gleich viel Durchsatz. Ein Cluster-Knoten hat 32
  gleichwertige Server-Kerne.
- Deutlich weniger Speicherbandbreite auf einem Laptop als auf einem Server-Knoten. Falls LBM
  bandbreitenlimitiert ist (wovon das Projekt-Readme ausgeht), bringt mehr Threads auf dem Laptop
  ab einem Punkt nichts mehr oder schadet sogar (alle Kerne teilen sich dieselbe knappe
  Bandbreite); auf dem Cluster mit mehr Bandbreite können mehr Threads tatsächlich mehr Durchsatz
  liefern.
- Hintergrundrauschen auf dem Laptop (Windows, OneDrive-Sync, Energie-/Thermik-Management), das
  ein dedizierter Linux-Cluster-Knoten nicht hat.

**Wichtig für die Einordnung:** die reine Kernzahl ist nicht die ganze Erklärung, das wird hier
bewusst offen und ehrlich so stehen gelassen statt eine hübschere, aber nicht ganz stimmige
Geschichte zu erzählen. Das bestätigt aber nochmal den bereits im Projekt-Readme genannten
Grundsatz: finale Zahlen gehören auf `pub030` gemessen, nicht auf dem Laptop -- Consumer-Hardware
verhält sich bei Thread-Parallelismus unvorhersehbarer als ein dedizierter Server-Knoten.

### Schritt 3 — (als Nächstes) volle Kernel-Fusion

Plan: `macroscopic()`, die Bounce-back-Schleife und `stream()` ebenfalls zu Scalar-Loops
umbauen, und alles (Macroscopic + Randbedingungen + Kollision + Bounce-back + Streaming) zu
einem einzigen `@njit(parallel=True)`-Kernel pro Zeitschritt fusionieren, statt mehrere separat
parallelisierte Funktionen pro Schritt aufzurufen. Ziel: nur noch ein Parallelisierungs-Overhead
pro Zeitschritt statt mehrerer kleiner.

(Rest ausfüllen, sobald erledigt)

## 5. Benchmark-Resultate

(laufendes Log liegt in `results/benchmark_log.csv` — relevante Zeilen hier reinziehen, sobald der
Kernel weiter ist, für die finale Write-up-Tabelle)

## 6. Korrektheit

- `validate.py`, rtol 1e-6: PASS auf beiden Maschinen für Schritt 1 (bit-identisch, 0.000e+00 max
  rel err) und Schritt 2 (weiterhin OK trotz geänderter Loop-Reihenfolge).

## 7. Reflexion

(am Schluss ausfüllen)
