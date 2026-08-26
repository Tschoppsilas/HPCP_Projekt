#!/bin/bash
# Weak Scaling: Arbeit pro Thread bleibt konstant (Zellen pro Thread ~40'000),
# die Gittergroesse (nx) waechst deshalb proportional zur Thread-Zahl.
# ny und nsteps bleiben fix, damit nur nx die Gesamtarbeit veraendert.
cd optimized
mkdir -p ../results

OUT=../results/weak_scaling.txt
echo "Weak-Scaling: cylinder, ny=100, nsteps=2000, nx = 400 * threads" > "$OUT"

NY=100
NSTEPS=2000

for t in 1 2 4 8 16 24 32; do
    NX=$((400 * t))
    echo "=== threads=$t  nx=$NX ny=$NY ===" | tee -a "$OUT"
    NUMBA_NUM_THREADS=$t python lbm_d2q9_numba.py --case cylinder --nx $NX --ny $NY --nsteps $NSTEPS --quiet | tee -a "$OUT"
done

echo "Fertig. Ergebnisse in $OUT"
