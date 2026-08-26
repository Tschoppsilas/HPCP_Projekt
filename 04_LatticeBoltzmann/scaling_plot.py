"""Strong-Scaling-Plot: Speedup und Effizienz vs. Thread-Anzahl (Cluster, cylinder/small)."""
import numpy as np
import matplotlib.pyplot as plt

threads = np.array([1, 2, 4, 8, 16, 24, 32])
runtime = np.array([3.52, 2.23, 2.12, 1.76, 1.84, 2.06, 2.17])

speedup = runtime[0] / runtime
efficiency = speedup / threads * 100

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Speedup vs. Threads
axes[0].plot(threads, speedup, "o-", color="tab:orange", label="gemessen")
axes[0].plot(threads, threads, "--", color="gray", linewidth=1, label="ideal (linear)")
axes[0].set_xlabel("Anzahl Threads (NUMBA_NUM_THREADS)")
axes[0].set_ylabel("Speedup relativ zu 1 Thread")
axes[0].set_title("Strong Scaling: Speedup")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.25)
axes[0].set_xticks(threads)

# Effizienz vs. Threads
axes[1].plot(threads, efficiency, "o-", color="tab:orange")
axes[1].axhline(100, color="gray", linestyle="--", linewidth=1)
axes[1].set_xlabel("Anzahl Threads (NUMBA_NUM_THREADS)")
axes[1].set_ylabel("Parallel-Effizienz [%]")
axes[1].set_title("Strong Scaling: Effizienz")
axes[1].grid(alpha=0.25)
axes[1].set_xticks(threads)
axes[1].set_ylim(0, 110)

fig.suptitle("Strong Scaling -- Cluster (Threadripper PRO 5955WX), cylinder, size=small")
fig.tight_layout()
fig.savefig("figures/strong_scaling.png", dpi=150)
print("Plot gespeichert: figures/strong_scaling.png")

best_i = np.argmax(speedup)
print(f"\nBestes Ergebnis: {threads[best_i]} Threads, Speedup={speedup[best_i]:.2f}x, "
      f"Effizienz={efficiency[best_i]:.1f}%")
print("Danach wird es WIEDER langsamer -- bei 32 Threads sogar schlechter als bei 2 Threads.")
