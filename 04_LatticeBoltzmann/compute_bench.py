"""Rechenleistungs-Mikrobenchmark (Compute-Dach fuer die Roofline).
Kleines Array (bleibt im Cache) + sehr viele FMA-artige Operationen pro Element,
damit die Zeit von der reinen Rechenleistung dominiert wird, nicht vom Speicherzugriff.
Gleiches Parallelisierungsmuster (@njit(parallel=True)/prange) wie die LBM-Kernel."""
import numpy as np
from numba import njit, prange
import time

N = 10_000
ITERS = 2_000_000

@njit(parallel=True)
def compute_bound(a, iters):
    n = a.shape[0]
    for i in prange(n):
        x = a[i]
        for _ in range(iters):
            x = x * 1.0000001 + 1e-7
        a[i] = x

a = np.random.rand(N)
compute_bound(a, 10)  # Warm-up / JIT

reps = 3
t0 = time.perf_counter()
for _ in range(reps):
    compute_bound(a, ITERS)
t1 = time.perf_counter()
runtime = (t1 - t0) / reps

flops = N * ITERS * 2  # 1 Multiplikation + 1 Addition pro Iteration
gflops = flops / runtime / 1e9
print(f"Laufzeit pro Durchlauf: {runtime*1000:.2f} ms")
print(f"Erreichte Rechenleistung: {gflops:.2f} GFLOP/s")
