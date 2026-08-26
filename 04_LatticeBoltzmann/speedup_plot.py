"""Speedup-Verlauf ueber die 5 Optimierungsschritte, Laptop vs. Cluster."""
import numpy as np
import matplotlib.pyplot as plt

steps = ["Baseline", "1: njit\nequilibrium", "2: prange\nequilibrium", "3: +\nmacroscopic",
         "4: +\nstream", "5: +\ncollide_and_bounce"]
x = np.arange(len(steps))

laptop_speedup = [1.00, 1.59, 1.38, 1.84, 2.48, 8.49]
cluster_speedup = [1.00, 0.85, 1.47, 1.46, 1.98, 2.76]

fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.plot(x, laptop_speedup, "o-", color="tab:blue", linewidth=2, markersize=8, label="Laptop")
ax.plot(x, cluster_speedup, "o-", color="tab:orange", linewidth=2, markersize=8, label="Cluster")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (kein Speedup)")

for xi, yi in zip(x, laptop_speedup):
    ax.annotate(f"{yi:.2f}x", (xi, yi), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8, color="tab:blue")
for xi, yi in zip(x, cluster_speedup):
    ax.annotate(f"{yi:.2f}x", (xi, yi), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=8, color="tab:orange")

ax.set_xticks(x)
ax.set_xticklabels(steps, fontsize=8.5)
ax.set_ylabel("Speedup relativ zur Baseline")
ax.set_title("Speedup pro Optimierungsschritt (cylinder, size=small)")
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("figures/speedup_progression.png", dpi=150)
print("Plot gespeichert: figures/speedup_progression.png")
