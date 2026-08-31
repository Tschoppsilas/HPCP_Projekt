"""Weak-Scaling-Plot: Laufzeit und Effizienz vs. Thread-Anzahl (Cluster, cylinder).
Zellen pro Thread bleiben konstant (~40'000); nx waechst proportional zur Thread-Zahl."""
import numpy as np
import matplotlib.pyplot as plt

threads = np.array([1, 2, 4, 8, 16, 24, 32])
runtime = np.array([3.53, 4.23, 7.99, 18.63, 74.83, 116.71, 153.07])
mlups = np.array([22.641, 37.858, 40.068, 34.345, 17.106, 16.451, 16.724])

efficiency = runtime[0] / runtime * 100

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

axes[0].axhline(runtime[0], color="gray", linestyle="--", linewidth=1, label="ideal (konstant)")
axes[0].plot(threads, runtime, "o-", color="tab:blue", label="gemessen")
axes[0].set_xlabel("Anzahl Threads (NUMBA_NUM_THREADS)")
axes[0].set_ylabel("Laufzeit [s]")
axes[0].set_title("Weak Scaling: Laufzeit")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.25)
axes[0].set_xticks(threads)

axes[1].axhline(100, color="gray", linestyle="--", linewidth=1, label="ideal (100%)")
axes[1].plot(threads, efficiency, "o-", color="tab:blue", label="gemessen")
axes[1].set_xlabel("Anzahl Threads (NUMBA_NUM_THREADS)")
axes[1].set_ylabel("Weak-Scaling-Effizienz [%]")
axes[1].set_title("Weak Scaling: Effizienz")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.25)
axes[1].set_xticks(threads)
axes[1].set_ylim(0, 110)

fig.suptitle("Weak Scaling -- Cluster (Threadripper PRO 5955WX), cylinder, ~40'000 Zellen/Thread")
fig.tight_layout()
fig.savefig("figures/weak_scaling.png", dpi=150)
print("Plot gespeichert: figures/weak_scaling.png")

for t, r, e, m in zip(threads, runtime, efficiency, mlups):
    print(f"threads={t:>2d}  runtime={r:>7.2f}s  efficiency={e:>5.1f}%  MLUPS={m:>6.2f}")
