"""Strong-Scaling-Plot: Speedup und Effizienz vs. Thread-Anzahl (Cluster, cylinder/small)."""
import numpy as np
import matplotlib.pyplot as plt

threads = np.array([1, 2, 4, 8, 16, 24, 32])
datasets = {
    "small (400x100, 2000 Steps)": {
        "runtime": np.array([3.52, 2.23, 2.12, 1.76, 1.84, 2.06, 2.17]),
        "color": "tab:orange",
    },
    "medium (1000x250, 10000 Steps)": {
        "runtime": np.array([150.70, 81.30, 64.00, 54.68, 60.96, 61.28, 63.18]),
        "color": "tab:green",
    },
}

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for name, d in datasets.items():
    runtime = d["runtime"]
    speedup = runtime[0] / runtime
    efficiency = speedup / threads * 100
    axes[0].plot(threads, speedup, "o-", color=d["color"], label=name)
    axes[1].plot(threads, efficiency, "o-", color=d["color"], label=name)
    best_i = np.argmax(speedup)
    print(f"{name}: bestes Ergebnis bei {threads[best_i]} Threads, "
          f"Speedup={speedup[best_i]:.2f}x, Effizienz={efficiency[best_i]:.1f}%")

axes[0].plot(threads, threads, "--", color="gray", linewidth=1, label="ideal (linear)")
axes[0].set_xlabel("Anzahl Threads (NUMBA_NUM_THREADS)")
axes[0].set_ylabel("Speedup relativ zu 1 Thread")
axes[0].set_title("Strong Scaling: Speedup")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.25)
axes[0].set_xticks(threads)

axes[1].axhline(100, color="gray", linestyle="--", linewidth=1)
axes[1].set_xlabel("Anzahl Threads (NUMBA_NUM_THREADS)")
axes[1].set_ylabel("Parallel-Effizienz [%]")
axes[1].set_title("Strong Scaling: Effizienz")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.25)
axes[1].set_xticks(threads)
axes[1].set_ylim(0, 110)

fig.suptitle("Strong Scaling -- Cluster (Threadripper PRO 5955WX), cylinder, small vs. medium")
fig.tight_layout()
fig.savefig("figures/strong_scaling.png", dpi=150)
print("\nPlot gespeichert: figures/strong_scaling.png")
