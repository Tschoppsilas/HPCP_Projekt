"""
Roofline-Analyse fuer die optimierte LBM-Pipeline (Numba, 4 Kernel: equilibrium,
macroscopic, stream, collide_and_bounce), auf Basis der 'medium'-Benchmarks
(Arbeitsset > L3-Cache auf beiden Maschinen -- 'small' war Cache-verzerrt, siehe Notes.md).

Speicher- und Rechen-"Daecher" sind selbst gemessene, erreichbare Grenzen fuer
Numba-Skalarschleifen-Code (@njit(parallel=True)/prange, kein manuelles SIMD) --
keine theoretischen Hardware-Maxima aus dem Datenblatt.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Arithmetische Intensitaet, von Hand aus dem Code hergeleitet (siehe Notes.md):
#   equilibrium:          96 B/Zelle,  103 FLOPs/Zelle
#   macroscopic:          96 B/Zelle,   47 FLOPs/Zelle
#   stream:               144 B/Zelle,   0 FLOPs/Zelle
#   collide_and_bounce:   216 B/Zelle,  27 FLOPs/Zelle
BYTES_PER_CELL = 96 + 96 + 144 + 216   # = 552
FLOPS_PER_CELL = 103 + 47 + 0 + 27      # = 177
AI = FLOPS_PER_CELL / BYTES_PER_CELL    # FLOP/Byte

WRITE_ALLOCATE_CORRECTION = 4 / 3   # STREAM-Triad zaehlt nur 3N*8B, real 4N*8B

machines = {
    "Laptop (Intel Ultra 7 155H)": {
        "mlups": 10.373,
        "stream_gbs_raw": 63.92,
        "compute_gflops": 17.17,     # eigener compute_bench.py-Wert
        "color": "tab:blue",
    },
    "Cluster (Threadripper PRO 5955WX)": {
        "mlups": 39.880,
        "stream_gbs_raw": 20.30,
        "compute_gflops": 33.88,
        "color": "tab:orange",
    },
}

fig, ax = plt.subplots(figsize=(8, 6))
ai_range = np.logspace(-2, 2, 300)

y_min, y_max = 1e9, -1e9

for name, d in machines.items():
    peak_bw = d["stream_gbs_raw"] * WRITE_ALLOCATE_CORRECTION   # GB/s
    peak_compute = d["compute_gflops"]                          # GFLOP/s
    ridge_ai = peak_compute / peak_bw

    roof = np.minimum(ai_range * peak_bw, peak_compute)         # klassische Roofline-Form
    ax.plot(ai_range, roof, "-", color=d["color"], linewidth=2, alpha=0.85,
            label=f"{name}\n  Speicher: {peak_bw:.1f} GB/s | Rechnen: {peak_compute:.1f} GFLOP/s "
                  f"| Knick bei AI={ridge_ai:.2f}")

    achieved_bw = BYTES_PER_CELL * d["mlups"] / 1000
    achieved_gflops = FLOPS_PER_CELL * d["mlups"] / 1000
    applicable_roof = min(AI * peak_bw, peak_compute)
    util_pct = achieved_gflops / applicable_roof * 100

    ax.plot(AI, achieved_gflops, "o", color=d["color"], markersize=10,
            markeredgecolor="black", zorder=5)
    ax.annotate(f"{achieved_gflops:.2f} GFLOP/s ({util_pct:.0f}% des Dachs)",
                (AI, achieved_gflops), textcoords="offset points", xytext=(10, -12),
                fontsize=8, color=d["color"])

    y_min = min(y_min, roof.min(), achieved_gflops)
    y_max = max(y_max, roof.max(), achieved_gflops)

ax.axvline(AI, color="gray", linestyle=":", linewidth=1, zorder=1)
# links von der grauen Linie, oben im freien Bereich (Achsen-Koordinaten, unabhaengig von den
# Datenwerten) -- ueberdeckt weder eine Dachlinie noch die Legende
ax.text(0.02, 0.96, f"AI = {AI:.3f} FLOP/Byte\n(unsere LBM-Pipeline)",
        transform=ax.transAxes, fontsize=8, color="dimgray", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.9))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(ai_range[0], ai_range[-1])
ax.set_xlabel("Arithmetische Intensitaet [FLOP/Byte]")
ax.set_ylabel("Performance [GFLOP/s]")
ax.set_title("Roofline: optimierte D2Q9-LBM-Pipeline (medium, cylinder, Re=200)")
ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
ax.grid(True, which="major", alpha=0.25)

fig.tight_layout()
fig.savefig("figures/roofline.png", dpi=150)
print("Plot gespeichert: figures/roofline.png")

print(f"\nAI = {AI:.4f} FLOP/Byte")
for name, d in machines.items():
    peak_bw = d["stream_gbs_raw"] * WRITE_ALLOCATE_CORRECTION
    peak_compute = d["compute_gflops"]
    ridge_ai = peak_compute / peak_bw
    achieved_gflops = FLOPS_PER_CELL * d["mlups"] / 1000
    applicable_roof = min(AI * peak_bw, peak_compute)
    regime = "speicherlimitiert" if AI < ridge_ai else "rechenlimitiert (im Modell)"
    print(f"{name}: Knick bei AI={ridge_ai:.3f} -> unser Kernel liegt im Bereich '{regime}'; "
          f"erreicht {achieved_gflops:.2f} / Dach {applicable_roof:.2f} GFLOP/s "
          f"= {achieved_gflops/applicable_roof*100:.1f}%")
