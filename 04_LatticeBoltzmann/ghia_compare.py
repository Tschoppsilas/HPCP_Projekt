import numpy as np
import json
import matplotlib.pyplot as plt

#Daten laden
data = np.load("data/ghia_cavity_numba_20k.npz")
ux = data["ux"]
uy = data["uy"]

with open("data/ghia_cavity_numba_20k.json") as f:
    meta = json.load(f)

u0 = meta["u0"]
nx = meta["nx"]
ny = meta["ny"]

#Profile extrahieren
u_profile = ux[nx // 2, :]   #u entlang der vertikalen Mittellinie
v_profile = uy[:, ny // 2]   #v entlang der horizontalen Mittellinie

#Normieren
u_norm = u_profile / u0
v_norm = v_profile / u0

#Koordinaten bauen: 0 = eine Wand, 1 = die andere (Deckel)
y = np.linspace(0, 1, ny)
x = np.linspace(0, 1, nx)

#Erstmal nur ausdrucken, um zu prüfen ob es plausibel aussieht
print("u-Profil (erste 10 Werte, von y=0 aufwärts):")
for yi, ui in zip(y[:10], u_norm[:10]):
    print(f"  y={yi:.4f}  u={ui:.5f}")

print("\nu-Profil (letzte 10 Werte, Richtung Deckel):")
for yi, ui in zip(y[-10:], u_norm[-10:]):
    print(f"  y={yi:.4f}  u={ui:.5f}")

# Ghia, Ghia & Shin (1982), Table I, Re=100 -- u entlang der vertikalen Mittellinie
ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                    0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                    0.9688, 0.9766, 1.0000])
ghia_u = np.array([ 0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                    -0.15662, -0.21090, -0.20581, -0.13641,  0.00332,  0.23151,
                     0.68717,  0.73722,  0.78871,  0.84123,  1.00000])

#Ghia-Vergleichsblock
mask = ghia_y < 1.0
u_at_ghia = np.interp(ghia_y[mask], y, u_norm)

print("\nVergleich mit Ghia (y, dein u, Ghia u, Differenz):")
for yi, ui, gi in zip(ghia_y[mask], u_at_ghia, ghia_u[mask]):
    print(f"  y={yi:.4f}  du={ui:+.5f}  ghia={gi:+.5f}  diff={ui-gi:+.5f}")

# Ghia, Ghia & Shin (1982), Table I, Re=100 -- v entlang der horizontalen Mittellinie
ghia_x = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
                    0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
                    0.9609, 0.9688, 1.0000])
ghia_v = np.array([ 0.00000,  0.09233,  0.10091,  0.10890,  0.12317,  0.16077,
                     0.17507,  0.17527,  0.05454, -0.24533, -0.22445, -0.16914,
                    -0.10313, -0.08864, -0.07391, -0.05906,  0.00000])

mask_v = (ghia_x > 0.0) & (ghia_x < 1.0)   # beide Wände hier (x=0 und x=1) trivial -- ausschliessen
v_at_ghia = np.interp(ghia_x[mask_v], x, v_norm)

print("\nVergleich mit Ghia, v-Profil (x, dein v, Ghia v, Differenz):")
for xi, vi, gi in zip(ghia_x[mask_v], v_at_ghia, ghia_v[mask_v]):
    print(f"  x={xi:.4f}  dv={vi:+.5f}  ghia={gi:+.5f}  diff={vi-gi:+.5f}")



#Plot: Vergleich der Zentrallinien-Profile gegen Ghia, Ghia & Shin (1982)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Deckel-Randpunkt (y=1.0) fuer den Plot korrigieren: der Baseline-Export setzt ihn wegen
# solid-Maskierung faelschlich auf 0, physikalisch ist er analytisch bekannt: u/u0 = 1.0
u_norm_plot = u_norm.copy()
u_norm_plot[-1] = 1.0

# u entlang der vertikalen Mittellinie (u auf x-Achse, y auf y-Achse -- klassische Darstellung)
axes[0].plot(u_norm_plot, y, "-", color="tab:blue", label="Numba (128x128, Re=100, 20k Steps)")
axes[0].plot(ghia_u, ghia_y, "o", markerfacecolor="none", color="black", label="Ghia et al. (1982)")
axes[0].set_xlabel("u / u0")
axes[0].set_ylabel("y")
axes[0].set_title("u entlang der vertikalen Mittellinie")
axes[0].legend()
axes[0].grid(alpha=0.3)

# v entlang der horizontalen Mittellinie
axes[1].plot(x, v_norm, "-", color="tab:blue", label="Numba (128x128, Re=100, 20k Steps)")
axes[1].plot(ghia_x, ghia_v, "o", markerfacecolor="none", color="black", label="Ghia et al. (1982)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("v / u0")
axes[1].set_title("v entlang der horizontalen Mittellinie")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle("Lid-driven cavity, Re=100 -- Validierung gegen Ghia, Ghia & Shin (1982)")
fig.tight_layout()
fig.savefig("figures/ghia_validation.png", dpi=150)
print("\nPlot gespeichert: figures/ghia_validation.png")
