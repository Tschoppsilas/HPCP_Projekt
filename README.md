# HPCP-Projekt

Einzelarbeit, FHNW. Übersicht über das Projekt und Setup-Anleitung.

## Projekt

**Projekt 04 — Lattice Boltzmann:** Beschleunigung eines D2Q9-Lattice-Boltzmann-Fluidlösers mit Numba.

→ [Report lesen](04_LatticeBoltzmann/report.md)
→ [optimierter Code](04_LatticeBoltzmann/optimized/lbm_d2q9_numba.py)

## Setup: virtuelle Umgebung (venv) aus `requirements.txt` erstellen

Alle benötigten Python-Pakete stehen in [`requirements.txt`](requirements.txt). Damit lässt sich auf
jeder Maschine (Laptop wie Cluster) eine saubere, isolierte Umgebung aufsetzen, ohne die
system-weite Python-Installation zu verändern.

**Linux / FHNW-Cluster (`pub030`):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Danach steht die venv mit allen Abhängigkeiten bereit; erkennbar am vorangestellten `(.venv)` im
Terminal-Prompt. Verlassen der Umgebung jederzeit mit:

```bash
deactivate
```
