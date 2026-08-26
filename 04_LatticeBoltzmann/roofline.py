"""
Roofline-Analyse fuer die optimierte LBM-Pipeline (Numba, 4 Kernel: equilibrium,
macroscopic, stream, collide_and_bounce), auf Basis der 'medium'-Benchmarks
(Arbeitsset > L3-Cache auf beiden Maschinen -- 'small' war Cache-verzerrt, siehe Notes.md).
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Arithmetische Intensitaet, von Hand aus dem Code hergeleitet (siehe Notes.md):
#   equilibrium:          96 B/Zelle,  103 FLOPs/Zelle
#   macroscopic:          96 B/Zelle,   47 FLOPs/Zelle
#   stream:               144 B/Zelle,   0 FLOPs/Zelle
#   collide_and_bounce:   216 B/Zelle,  27 FLOPs/Zelle
# Summe pro Zelle und Zeitschritt (alle vier Kernel, nicht fusioniert):
BYTES_PER_CELL = 96 + 96 + 144 + 216   # = 552
FLOPS_PER_CELL = 103 + 47 + 0 + 27      # = 177
AI = FLOPS_PER_CELL / BYTES_PER_CELL    # FLOP/Byte

# ---------------------------------------------------------------------------
# Gemessene Werte: 'medium'-Benchmark (cylinder, 1000x250, 10'000 Steps) --
# das Arbeitsset (~78 MB) ist auf beiden Maschinen groesser als der L3-Cache.
machines = {
    "Laptop (Intel Ultra 7 155H)": {
        "mlups": 10.373,
        "stream_gbs_raw": 63.92,     # eigene STREAM-Triad-Messung
        "color": "tab:blue",
    },
    "Cluster (Threadripper PRO 5955WX)": {
        "mlups": 39.880,
        "stream_gbs_raw": 20.30,
        "color": "tab:orange",
    },
}
WRITE_ALLOCATE_CORRECTION = 4 / 3   # STREAM-Triad-Messung zaehlt nur 3N*8B, real 4N*8B

fig, ax = plt.subplots(figsize=(8, 6))

ai_range = np.logspace(-3, 2, 200)

for name, d in machines.items():
    peak_bw = d["stream_gbs_raw"] * WRITE_ALLOCATE_CORRECTION   # GB/s
    achieved_bw = BYTES_PER_CELL * d["mlups"] / 1000             # GB/s
    achieved_gflops = FLOPS_PER_CELL * d["mlups"] / 1000         # GFLOP/s

    # Memory-Roof: Performance = AI * peak_bandwidth (Speicherbandbreiten-Grenze)
    ax.plot(ai_range, ai_range * peak_bw, "--", color=d["color"], alpha=0.6,
             label=f"{name}: Speicher-Dach ({peak_bw:.1f} GB/s, write-allocate-korrigiert)")

    # gemessener Punkt bei AI der LBM-Pipeline
    ax.plot(AI, achieved_gflops, "o", color=d["color"], markersize=10,
             markeredgecolor="black", zorder=5)
    ax.annotate(f"{achieved_gflops:.2f} GFLOP/s\n({achieved_bw:.1f} GB/s, "
                f"{achieved_bw/peak_bw*100:.0f}% des Dachs)",
                (AI, achieved_gflops), textcoords="offset points", xytext=(12, -4),
                fontsize=8, color=d["color"])

ax.axvline(AI, color="gray", linestyle=":", linewidth=1)
ax.text(AI, ax.get_ylim()[1] if False else 0.005, f"  AI = {AI:.3f} FLOP/Byte\n  (LBM-Pipeline, 4 Kernel)",
        fontsize=8, color="gray", va="bottom")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Arithmetische Intensitaet [FLOP/Byte]")
ax.set_ylabel("Performance [GFLOP/s]")
ax.set_title("Roofline: optimierte D2Q9-LBM-Pipeline (medium, cylinder, Re=200)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig("figures/roofline.png", dpi=150)
print("Plot gespeichert: figures/roofline.png")
print(f"\nAI = {AI:.4f} FLOP/Byte")
for name, d in machines.items():
    peak_bw = d["stream_gbs_raw"] * WRITE_ALLOCATE_CORRECTION
    achieved_bw = BYTES_PER_CELL * d["mlups"] / 1000
    print(f"{name}: achieved {achieved_bw:.2f} GB/s / corrected peak {peak_bw:.2f} GB/s "
          f"= {achieved_bw/peak_bw*100:.1f}%")
