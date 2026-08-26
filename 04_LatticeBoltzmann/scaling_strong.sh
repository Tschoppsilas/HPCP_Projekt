#!/bin/bash
cd optimized
mkdir -p ../results

OUT=../results/strong_scaling_small.txt
echo "Strong-Scaling: cylinder, size=small" > "$OUT"

for t in 1 2 4 8 16 24 32; do
    echo "=== threads=$t ===" | tee -a "$OUT"
    NUMBA_NUM_THREADS=$t python lbm_d2q9_numba.py --case cylinder --size small --quiet | tee -a "$OUT"
done

echo "Fertig. Ergebnisse in $OUT"