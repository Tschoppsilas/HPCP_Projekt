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

### Schritt 3 — `macroscopic()` zu Scalar-Loop + `prange`

Gleiches Muster wie bei `equilibrium()`: `_, nx, ny = f.shape`, dann `for x in prange(nx): for
y in range(ny):` mit lokalen Skalaren `rho_c, ux_c, uy_c` (eigene Namen, nicht `rho`/`ux`/`uy` --
das war ein Zwischenfehler: erst wurden die Output-Arrays mit denselben Namen wie die
Zellen-Summen überschrieben, wodurch am Ende nur noch Einzelzahlen statt Arrays zurückgegeben
wurden). Division und Zurückschreiben ins Array (`rho[x, y] = rho_c` usw.) korrekt innerhalb der
x/y-Loops platziert.

**Mess-Bug gefunden und korrigiert:** `macroscopic()` wird im Gegensatz zu `equilibrium()` zum
ersten Mal *innerhalb* der Zeitschritt-Schleife aufgerufen (`rho, ux, uy = macroscopic(f)` ist
die erste Zeile in `for step in range(nsteps):`), also nach `t_start = time.perf_counter()`.
Ohne einen expliziten Warm-up-Aufruf davor wurde die JIT-Kompilierzeit von `macroscopic()` beim
ersten Schleifendurchlauf mitgemessen und hat die Zahlen verfälscht. Fix: `_ = macroscopic(f)`
direkt nach der bestehenden `equilibrium()`-Initialisierung eingefügt, noch vor `t_start`.

Ergebnisse vor und nach diesem Fix, zur Dokumentation (zeigt, wie stark ein Mess-Fehler die
Interpretation verfälschen kann):

| Maschine | Baseline | Nur equilibrium (Schritt 2) | macroscopic dazu, **ohne** Warm-up-Fix | macroscopic dazu, **mit** Warm-up-Fix |
|----------|----------|------------------------------|------------------------------------------|-------------------------------------------|
| Laptop   | 10.45 s  | 7.57 s (1.38x)               | 5.97 s (1.75x)                            | 7.73 s (Ausreisser, Rauschen) / **5.69 s (1.84x)** bei Wiederholung |
| Cluster  | 5.58 s   | 3.80 s (1.47x)               | 4.49 s (1.24x, wirkte wie Rückschritt!)   | **3.83 s (1.46x)**, praktisch gleich wie Schritt 2 |

**Wichtige Korrektur gegenüber der letzten Interpretation:** Die zuvor beobachtete "Verschlechterung"
auf dem Cluster (3.80 s -> 4.49 s) war grösstenteils ein Mess-Artefakt (mitgemessene
Kompilierzeit), nicht ein echter Effekt von zusätzlichem Parallelisierungs-Overhead pro
Funktionsaufruf. Nach der Korrektur ist `macroscopic()` auf dem Cluster ungefähr **neutral**
(3.83 s vs. 3.80 s) -- weder deutlicher Gewinn noch Verlust. Auf dem Laptop hilft es weiterhin
klar (bester bisheriger Speedup: 1.84x). Die "Overhead akkumuliert pro zusätzlicher parallelisierter
Funktion"-Hypothese von vorhin wird damit nicht bestätigt -- zumindest nicht durch dieses
Beispiel. Lehre fürs Write-up: ein unerwartetes Ergebnis zuerst auf Mess-Fehler prüfen (Warm-up!),
bevor eine inhaltliche Erklärung dafür gesucht wird.

Korrektheit: weiterhin `OK` auf beiden Maschinen.

### Schritt 4 — `stream()` zu Scalar-Loop + `prange`

Gleiches Muster, diesmal ohne Summierung, nur "Werte von der Nachbarzelle holen" statt zweimal
`np.roll` in ein frisches Array. `np.roll(arr, shift)` entspricht mathematisch: neuer Wert an
Position x kommt von Position `(x - shift) mod n`. Direkt umgesetzt als:
```python
src_x = (x - C[i, 0]) % nx
src_y = (y - C[i, 1]) % ny
out[i, x, y] = f[i, src_x, src_y]
```
`out = np.empty_like(f)` bleibt nötig (kein In-place-Update möglich, da sonst Werte überschrieben
würden, die noch gelesen werden müssen). Warm-up (`_ = stream(f)`) diesmal von Anfang an korrekt
vor `t_start` eingebaut -- keine Fehler mehr bei diesem Schritt, weder inhaltlich noch beim
Messen.

Ergebnis — Korrektheit: `OK` auf beiden Maschinen, bit-identisch (reines Kopieren, keine
Summierung, keine Reihenfolge-Änderung, die Gleitkomma-Rundung beeinflussen könnte).

Ergebnis — Performance, klarer und grösster Sprung bisher, auf **beiden** Maschinen:

| Maschine | Baseline | Schritt 3 (equilibrium + macroscopic) | Schritt 4 (+ stream) | Speedup gesamt |
|----------|----------|------------------------------------------|--------------------------|-----------------|
| Laptop   | 10.45 s  | 5.69 s (1.84x)                            | **4.21 s / 19.015 MLUPS** | **2.48x** |
| Cluster  | 5.58 s   | 3.83 s (1.46x)                            | **2.82 s / 28.409 MLUPS** | **1.98x** |

Passt zur Baseline-Profiling-Erwartung: `stream()`s doppeltes `np.roll` in ein frisches Array war
im Docstring des Originalcodes explizit als "the single biggest source of memory traffic in the
whole solver" markiert, und ~65% von `stream()`s Laufzeit war reines `np.roll`. Das komplette
Wegfallen dieser Allokation+Kopie (ersetzt durch direkte Indexberechnung) zeigt hier den
deutlichsten, konsistentesten Gewinn von allen bisherigen Schritten -- auf beiden Maschinen, ohne
die Mehrdeutigkeit, die wir bei `macroscopic()` gesehen hatten.

### Schritt 5 — Kollision + Bounce-back fusioniert, in `collide_and_bounce()`

Kollision (`fpost = f - omega * (f - feq)`) und die Bounce-back-Schleife
(`fpost[i][bounce] = f[OPP[i]][bounce]`) waren im Original zwei getrennte NumPy-Operationen
hintereinander -- zusammengelegt in eine `@njit(parallel=True)`-Funktion mit Scalar-Loop:
solid/bounce-Zellen kriegen den Bounce-back-Wert, alle anderen die normale BGK-Kollisionsformel,
beides pro Zelle in einem Durchgang, keine Zwischen-Arrays. Warm-up (`_ = collide_and_bounce(f,
f, omega, bounce)`) vor `t_start` ergänzt.

Ergebnis — Korrektheit: `OK` auf beiden Maschinen.

Ergebnis — Performance, der grösste Sprung von allen Schritten:

| Maschine | Baseline | Schritt 4 (+ stream) | Schritt 5 (+ collide_and_bounce) | Speedup gesamt |
|----------|----------|------------------------|--------------------------------------|-----------------|
| Laptop   | 10.45 s  | 4.21 s (2.48x)          | **1.23 s / 65.175 MLUPS**            | **8.49x** |
| Cluster  | 5.58 s   | 2.82 s (1.98x)          | **2.02 s / 39.633 MLUPS**            | **2.76x** |

Grösster Einzelschritt bisher -- plausibel, weil hier gleich zwei teure Dinge auf einmal wegfallen:
die boolesche Fancy-Indexing-Bounce-back-Schleife (bekannt dafür, in NumPy vergleichsweise teuer
zu sein: temporäre Index-Arrays, nicht-zusammenhängender Speicherzugriff) UND eine weitere
Temporär-Array-Allokation für `fpost`. Auf dem Laptop wirkt sich das besonders stark aus, auf dem
Cluster auch deutlich, aber verhältnismässig kleiner -- passend zum bisherigen Muster, dass der
Laptop empfindlicher auf Python-/NumPy-Overhead reagiert als der Cluster.

### Entscheidung: keine volle Kernel-Fusion mehr

Bewusster Stopp an dieser Stelle, mit Blick auf die verbleibende Zeit (siehe unten). Die vier
Kernfunktionen (`equilibrium`, `macroscopic`, `stream`, `collide_and_bounce`) sind jetzt alle
`@njit(parallel=True)`, decken die grosse Mehrheit der ursprünglich profilten Laufzeit ab, und der
Speedup ist bereits sehr deutlich (8.49x / 2.76x). Eine komplette Fusion zu einem einzigen
Riesen-Kernel (inkl. Randbedingungen) wäre der nächste mögliche Schritt gewesen, aber der
Aufwand/Risiko-Nutzen lohnt sich angesichts der verbleibenden Zeit nicht mehr -- die übrigen
Deliverables (Roofline, Scaling-Studien, Physik-Validierung inkl. `cavity`/Ghia-Vergleich, der
Report selbst) sind noch offen und zählen genauso zur Note. Bewusste Ingenieurs-Entscheidung, kein
Zeitmangel-Zufall -- das gehört so in die Reflexion.

## 5. Benchmark-Resultate — Zusammenfassung über alle Schritte (cylinder, small)

| Schritt | Laptop | Cluster |
|---------|--------|---------|
| Baseline | 10.45 s / 7.656 MLUPS | 5.58 s / 14.344 MLUPS |
| 1: njit equilibrium (Array-Ausdrücke) | 6.59 s (1.59x) | 6.54 s (0.85x, langsamer) |
| 2: prange equilibrium (Scalar-Loop) | 7.57 s (1.38x) | 3.80 s (1.47x) |
| 3: + macroscopic (mit Warm-up-Fix) | 5.69 s (1.84x) | 3.83 s (1.46x) |
| 4: + stream | 4.21 s (2.48x) | 2.82 s (1.98x) |
| 5: + collide_and_bounce | **1.23 s (8.49x)** | **2.02 s (2.76x)** |

Vollständiges Log mit allen Einzelläufen liegt in `results/benchmark_log.csv`.

## 6. Korrektheit

- `validate.py`, rtol 1e-6: PASS auf beiden Maschinen für alle Schritte 1-5. Bit-identisch bei
  Schritt 1, 2, 4, 5 (kein Reduktions-/Reihenfolge-Effekt); bei Schritt 3 (macroscopic) ebenfalls
  weiterhin exakt.
- `cavity`-Fall jetzt ebenfalls getestet (voller optimierter Pipeline, alle vier Kernfunktionen):
  `correctness: OK` auf beiden Maschinen, `rtol=1e-6`.

  | Maschine | Baseline | Optimiert | Speedup |
  |----------|----------|-----------|---------|
  | Laptop   | 11.34 s (7.055 MLUPS) | 1.52 s (52.593 MLUPS) | 7.46x |
  | Cluster  | 5.20 s (15.372 MLUPS) | 2.06 s (38.878 MLUPS) | 2.52x |

  Damit sind beide Testfälle (cylinder, cavity) mit der optimierten Version korrekt validiert.

- Zusaetzlich fuer die Ghia-Validierung: quadratisches Gitter 128x128, Re=100, 10'000 Steps
  (Ghias Referenzwerte gelten nur fuer eine quadratische Kavitaet, das Standard-Preset `small`
  ist mit 400x100 rechteckig und dafuer ungeeignet). Baseline- und optimierter Lauf bit-identisch
  auf beiden Maschinen: `validate.py --rtol 1e-6` -> alle drei Felder (ux, uy, rho) `PASS`,
  `max rel err 0.000e+00`. Dateien: `data/ghia_cavity.npz` (Baseline), `data/ghia_cavity_numba.npz`
  (optimiert). Naechster Schritt: physikalische Validierung des cavity-Profils gegen Ghia, Ghia &
  Shin (1982), Table I (Re=100).

## 6b. Physik-Validierung: Ghia, Ghia & Shin (1982)

Cavity-Fall, quadratisches Gitter 128x128, Re=100 (Standard fuer `--case cavity`), optimierte
Version (`optimized/lbm_d2q9_numba.py`). Vergleich der Zentrallinien-Profile (u entlang der
vertikalen, v entlang der horizontalen Mittellinie, beide normiert mit u0) gegen die transkribierten
Tabellenwerte aus Ghia, Ghia & Shin (1982), Table I, Re=100 -- Skript `ghia_compare.py`
(eigenes Skript, np.interp an den Ghia-Stuetzstellen).

**Konvergenz-Check:** 10'000 vs. 20'000 Steps verglichen. Bei 10k war v.a. die Rezirkulationszone
(mittlerer Bereich, y~0.28-0.45) noch deutlich zu flach (Abweichung bis ~22%). Bei 20k Steps deutlich
naeher an Ghia (siehe unten) -- klarer Hinweis, dass 10k noch nicht eingeschwungen war, 20k praktisch
konvergiert (Deckel-Bereich hatte sich zwischen 10k und 20k kaum mehr veraendert). Laenger laufen
lassen (z.B. 40k) haette laut diesem Trend nur noch marginalen Zusatznutzen -- angesichts Zeitbudget
bei 20k gestoppt.

**Ergebnis bei 20'000 Steps** (Randpunkte y=0/1 bzw. x=0/1 ausgeschlossen -- trivial durch
Randbedingung, keine Validierungsaussage):

| Profil | max. abs. Differenz | mittlere abs. Differenz |
|--------|---------------------|--------------------------|
| u (vertikale Mittellinie) | 0.0208 | 0.0092 |
| v (horizontale Mittellinie) | 0.0078 | 0.0049 |

Form/Topologie stimmt vollstaendig ueberein (Nulldurchgaenge, Vorzeichen, Lage des
Rezirkulationszentrums). Groessere Abweichung bei u nahe am Deckel (~2%) und in der
Rezirkulationszone (~5-6%) -- plausibel durch Gitteraufloesung (128x128 LBM vs. Ghias 129x129
Finite-Differenzen) und LBM-Kompressibilitaet (tau=0.689, endliche Machzahl) erklaerbar, nicht durch
einen Fehler. Als validiertes Ergebnis akzeptiert.

Plot: `figures/ghia_validation.png` (erzeugt von `ghia_compare.py`) -- zwei Teilplots, u(y) und v(x),
jeweils eigene Linie gegen Ghia-Marker. Hinweis fuer den Report: fuer den Plot wurde der
Deckel-Randpunkt (y=1.0) manuell auf den analytisch bekannten Wert u/u0=1.0 korrigiert (siehe oben,
Export-Artefakt durch solid-Maskierung) -- die Tabellenwerte weiter oben waren davon nicht betroffen,
da dieser Randpunkt dort schon ausgeschlossen war.

## 6c. medium-Benchmark und Roofline-Analyse

**medium-Benchmark** (cylinder, 1000x250, 10'000 Steps) -- gleichzeitig die noch offene
groessere Benchmark-Zahl fuers Reporting und die Datenbasis fuer die Roofline (siehe unten,
Arbeitsset > L3-Cache):

| Maschine | Baseline | Optimiert | Speedup |
|----------|----------|-----------|---------|
| Laptop   | 736.72 s (3.393 MLUPS)  | 241.02 s (10.373 MLUPS) | 3.06x |
| Cluster  | 191.10 s (13.082 MLUPS) | 62.69 s (39.880 MLUPS)  | 3.05x |

`correctness: OK` auf beiden Maschinen. Auffällig: der Speedup bei `medium` (~3x) ist deutlich
kleiner als bei `small` (8.49x Laptop / 2.76x Cluster) -- siehe Interpretation unten.

**Arithmetische Intensitaet** (von Hand aus den vier Kernel-Funktionen hergeleitet, siehe Code):

| Kernel | Bytes/Zelle | FLOPs/Zelle |
|--------|-------------|-------------|
| equilibrium | 96 | 103 |
| macroscopic | 96 | 47 |
| stream | 144 | 0 |
| collide_and_bounce | 216 | 27 |
| **Summe** | **552** | **177** |

AI = 177 / 552 = **0.321 FLOP/Byte** -- sehr tief, bestaetigt die im README beschriebene
Speicherbandbreiten-Limitierung von LBM. Zum Vergleich: eine ideal fusionierte Implementierung
(ein Lese-, ein Schreibdurchgang) braeuchte nur ~144 Bytes/Zelle -- unsere nicht fusionierte
Pipeline (4 separate Kernel-Aufrufe/Zeitschritt) bewegt das **3.8-fache** an Speicherverkehr.
Das ist der quantifizierte Preis der "keine volle Fusion"-Entscheidung von oben.

**Cache-Falle bei `small`:** Bei `size=small` (400x100) ist das Arbeitsset (f/feq/out/fpost,
je ~2.9 MB) nur ~12.5 MB gross -- passt auf beiden Maschinen komplett in den L3-Cache (Laptop
24 MB, Cluster 64 MB). Die daraus berechnete "erreichte Bandbreite" (Cluster ~21.9 GB/s) lag
deshalb sogar leicht ueber der eigenen gemessenen DRAM-Bandbreite -- kein Fehler, sondern
Cache-Effekt. Fuer eine ehrliche Roofline wurde stattdessen `medium` verwendet (Arbeitsset
~78 MB, > L3 auf beiden Maschinen).

**Speicherbandbreite:** eigener STREAM-Triad-Mikrobenchmark (`stream_bench.py`,
`@njit(parallel=True)`, N=100M, gleiches Parallelisierungsmuster wie die LBM-Kernel), plus
Korrektur um den Write-Allocate-Effekt (Faktor 4/3, da ein Schreibzugriff auf eine neue
Cache-Line ueblicherweise zuerst ein "verstecktes" Lesen dieser Line ausloest):

| Maschine | STREAM (roh) | STREAM (write-allocate-korrigiert) | Erreicht (medium) | Anteil vom Dach |
|----------|--------------|--------------------------------------|--------------------|------------------|
| Laptop   | 63.92 GB/s   | 85.23 GB/s | 5.73 GB/s  | **6.7%** |
| Cluster  | 20.30 GB/s   | 27.07 GB/s | 22.01 GB/s | **81.3%** |

**Rechen-Dach** (zweiter Mikrobenchmark, `compute_bench.py`: kleines Array bleibt im Cache,
sehr viele FMA-artige Operationen, damit die Zeit rein von der Rechenleistung dominiert wird,
gleiches `@njit(parallel=True)`/`prange`-Muster):

| Maschine | Rechen-Dach (gemessen) | Speicher-Dach (korrigiert) | Knick (Ridge Point) |
|----------|--------------------------|------------------------------|------------------------|
| Laptop   | 17.17 GFLOP/s | 85.23 GB/s | AI = 0.201 FLOP/Byte |
| Cluster  | 33.88 GFLOP/s | 27.07 GB/s | AI = 1.252 FLOP/Byte |

Hinweis: beide Daecher sind selbst gemessene, mit Numba-Skalarschleifen erreichbare Grenzen
(kein manuelles SIMD) -- keine theoretischen Hardware-Datenblatt-Maxima.

Plot: `figures/roofline.png` (`roofline.py`), mit korrektem Roofline-Knick (Speicher-Diagonale
bricht am Ridge Point in die flache Rechenleistungs-Grenze).

**Interpretation -- grosse Diskrepanz Laptop vs. Cluster:** Bei unserer AI (0.321 FLOP/Byte)
liegt der Cluster noch links vom eigenen Knick (1.252) -- also im speicherlimitierten Bereich,
und erreicht dort plausibel 81% seines Speicher-Dachs. Der Laptop dagegen hat einen sehr
niedrigen Knick (0.201, weil sein Speicher-Dach im Verhaeltnis zur eigenen Rechenleistung sehr
hoch ist) -- unsere Pipeline liegt bei ihm im Modell schon im rechenlimitierten Bereich, erreicht
aber nur 11% des (gemessenen) Rechen-Dachs. Unabhaengig davon, welches Dach man anlegt: der
Laptop bleibt bei `medium` weit unter jeder plausiblen Grenze -- die Unterauslastung ist real,
nicht nur ein Artefakt der Modellwahl. Der Cluster erreicht bei `medium`
plausibel den Grossteil (81%) seiner eigenen Speicherbandbreite -- genau das erwartete Bild fuer
einen speicherbandbreiten-limitierten Stencil. Der Laptop dagegen schoepft nur ~7% seines
Dachs aus, und faellt bei `medium` sogar hinter den Cluster zurueck (241 s vs. 62.7 s optimiert
-- bei `small` war es umgekehrt, Laptop war schneller). Zwei plausible, nicht abschliessend
bewiesene Hypothesen (Zeitbudget liess keine tiefere Instrumentierung mehr zu):
1. **Heterogene P-/E-Kerne**: der Laptop-Prozessor hat P- und E-Kerne unterschiedlicher
   Geschwindigkeit; `prange` verteilt die Arbeit standardmaessig in gleich grossen Chunks --
   bei genug Arbeit pro Chunk (wie bei `medium`, 6.25x mehr Zellen und 5x mehr Steps als
   `small`) wartet die ganze Parallelregion auf die langsamsten (E-)Kerne. Bei `small` war die
   Laufzeit so kurz, dass dieser Effekt kaum ins Gewicht fiel. Passt zur bereits frueher (Schritt 2)
   dokumentierten, damals noch vorlaeufigen Hypothese zu heterogenen Kernen.
2. **Thermal Throttling**: ein duenneres Laptop-Chassis kann unter mehreren Minuten Dauerlast
   (241 s bei `medium` vs. ~1 s bei `small`) die Taktfrequenz drosseln, ein Cluster-Server-Node
   nicht in vergleichbarem Mass.

Fazit: bei kleinen Problemgrössen (`small`) ist der Laptop schneller (Cache-Vorteil, kurze
Laufzeit), bei realistischeren Groessen (`medium`) dreht sich das Bild um -- eine wichtige
Erkenntnis, die eine reine `small`-Messung verschleiert haette.

**Warum die Optimierung so viel bringt -- Baseline vs. optimiert am Beispiel `equilibrium`:**
Die Physik/FLOPs sind bei Baseline (NumPy) und optimiert (Numba) identisch -- am Speicherverkehr
aendert sich aber viel: NumPy materialisiert bei jedem einzelnen Rechenschritt
(`ux*ux`, `+`, `*1.5`, ...) ein eigenes temporaeres Array im Speicher, waehrend der
Numba-Scalar-Loop alles pro Zelle in Registern durchrechnet und nur das Endresultat schreibt.
Von Hand durchgerechnet fuer `equilibrium()`: Baseline bewegt dafuer ~1944 Bytes/Zelle,
die optimierte Version nur 96 Bytes/Zelle -- Faktor ~20, bei exakt gleicher Rechenarbeit.
(Fuer die anderen drei Kernel liesse sich das grundsaetzlich auch herleiten, aber von Hand
nicht zuverlaessig genug fuer eine belastbare Gesamtzahl -- NumPy kann intern Zwischenschritte
anders behandeln als angenommen; ohne echtes Memory-Profiling-Tool nicht verifizierbar. Deshalb
hier nur das eine, klar nachvollziehbare Beispiel statt einer unsicheren Gesamt-AI fuer die
Baseline.) Das erklaert die Optimierung nicht nur "weil Numba schneller ist", sondern konkret
*warum*: weniger Speicherverkehr pro FLOP, nicht mehr FLOPs.

## 6d. Strong-Scaling-Studie

Cluster (Threadripper PRO 5955WX, 32 Kerne), `cylinder`/`size=small`, Thread-Zahl ueber
`NUMBA_NUM_THREADS` variiert (`scaling_strong.sh`, Ergebnisse in
`results/strong_scaling_small.txt`). Bewusst `small` statt `medium` gewaehlt, rein aus
Zeitgruenden: bei `medium` haette schon ein Lauf mit 1 Thread sehr lange gedauert.

| Threads | Laufzeit | Speedup | Effizienz |
|---------|----------|---------|-----------|
| 1  | 3.52 s | 1.00x | 100.0% |
| 2  | 2.23 s | 1.58x |  78.9% |
| 4  | 2.12 s | 1.66x |  41.5% |
| 8  | 1.76 s | 2.00x |  25.0% |
| 16 | 1.84 s | 1.91x |  12.0% |
| 24 | 2.06 s | 1.71x |   7.1% |
| 32 | 2.17 s | 1.62x |   5.1% |

**Gegenprobe mit `medium`** (1000x250, 10'000 Steps, gleicher Cluster,
`results/strong_scaling_medium.txt`, im Hintergrund via `nohup` waehrend Meeting/Mittagspause
gelaufen):

| Threads | Laufzeit | Speedup | Effizienz |
|---------|----------|---------|-----------|
| 1  | 150.70 s | 1.00x | 100.0% |
| 2  |  81.30 s | 1.85x |  92.7% |
| 4  |  64.00 s | 2.36x |  58.9% |
| 8  |  54.68 s | 2.76x |  34.5% |
| 16 |  60.96 s | 2.47x |  15.5% |
| 24 |  61.28 s | 2.46x |  10.2% |
| 32 |  63.18 s | 2.39x |   7.5% |

Plot: `figures/strong_scaling.png` (`scaling_plot.py`, small + medium im Vergleich).

**Interpretation:** Beide Groessen zeigen dasselbe Muster -- Optimum bei **8 Threads**, danach
wieder langsamer -- aber `medium` durchweg besser: hoeherer Peak-Speedup (2.76x vs. 2.00x) und
bei 32 Threads faellt es nicht so tief ab wie `small` (dort war 32 Threads schlechter als 2
Threads; bei `medium` bleibt selbst der schlechteste Mehr-Thread-Wert klar besser als 1 Thread).
Erwartungsgemaess: mehr Arbeit pro Thread verduennt den Parallelregion-Overhead (siehe `small`-
Erklaerung oben).

Dass es aber auch bei `medium` ab 8 Threads bergab geht, hat noch einen zweiten, aus der
Roofline-Analyse bereits bekannten Grund: bei `medium` erreicht der Cluster schon mit seinen
32 Threads 81% seiner eigenen Speicherbandbreite (Abschnitt 6c) -- die Speicherbandbreite ist
also fast schon ausgereizt. Ab dem Punkt, wo die Bandbreite der limitierende Faktor ist (nicht
mehr die Rechenkerne), bringen weitere Threads keinen Zusatznutzen mehr, nur zusaetzliche
Synchronisations-/Cache-Konkurrenz-Kosten -- das erklaert, warum die Kurve nicht einfach
langsamer waechst, sondern aktiv wieder abfaellt. Die Strong-Scaling-Studie bestaetigt damit
unabhaengig, was die Roofline schon nahelegte: der Kernel ist speicherbandbreiten-limitiert,
nicht kernanzahl-limitiert -- mehr Kerne allein loesen das Problem nicht.

## 7. Reflexion

(am Schluss ausfüllen -- die "keine volle Fusion mehr"-Entscheidung oben gehört hier explizit
rein: was mit mehr Zeit noch gegangen wäre, und warum der Stopp trotzdem richtig war)
