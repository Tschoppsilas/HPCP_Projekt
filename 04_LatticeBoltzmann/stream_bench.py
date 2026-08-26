import numpy as np
from numba import njit, prange
import time

N = 100_000_000  # ~800 MB pro Array, damit es klar groesser als jeder Cache ist

@njit(parallel=True)
def triad(a, b, c, scalar):
    for i in prange(a.shape[0]):
        a[i] = b[i] + scalar * c[i]

a = np.empty(N, dtype=np.float64)
b = np.random.rand(N)
c = np.random.rand(N)
scalar = 3.0

triad(a, b, c, scalar)  # Warm-up (JIT-Kompilierung ausschliessen)

reps = 10
t0 = time.perf_counter()
for _ in range(reps):
    triad(a, b, c, scalar)
t1 = time.perf_counter()

runtime = (t1 - t0) / reps
bytes_moved = 3 * N * 8   # STREAM Triad: 2x lesen (b, c) + 1x schreiben (a), float64 = 8 Byte
bandwidth_gbs = bytes_moved / runtime / 1e9

print(f"Laufzeit pro Durchlauf: {runtime*1000:.2f} ms")
print(f"Erreichte Bandbreite: {bandwidth_gbs:.2f} GB/s")