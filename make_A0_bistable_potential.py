"""
make_A0_bistable_potential.py
Generator for Figure 1 (fig:bistable): double-well potential
U(V) = -int f(V) dV induced by the cubic ionic current of Eq. (4).

Labels are placed in empty regions (no crossing arrows): the descriptive
state labels sit above each well / above the central peak, and the
equilibrium names V_-, V_0, V_+ sit at the points.

Output: fig1_double-well.png (project root).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 17, "axes.titlesize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

a, Vm, V0, Vp = 1.2, -1.5, 0.0, 1.5
def U(V):                       # U(V) = a (V^4/4 - 2.25/2 V^2), S1=S3=0, S2=-2.25
    return a * (V**4 / 4.0 - 2.25 / 2.0 * V**2)

Um = U(Vm)                       # = -81a/64 ~ -1.52

V = np.linspace(-2.2, 2.2, 700)
fig, ax = plt.subplots(figsize=(8.5, 5.0))
ax.plot(V, U(V), color="#1f4e79", lw=2.6, zorder=2)

# --- equilibrium markers ---
ax.plot(Vm, Um, "o", ms=11, color="#2166ac", zorder=5)
ax.plot(Vp, Um, "o", ms=11, color="#b2182b", zorder=5)
ax.plot(V0, U(V0), "o", ms=11, mfc="white", mec="black", mew=2, zorder=5)

# --- equilibrium names, at the points ---
ax.text(Vm, Um - 0.16, r"$V_-$", ha="center", va="top",   fontsize=17, color="#2166ac")
ax.text(Vp, Um - 0.16, r"$V_+$", ha="center", va="top",   fontsize=17, color="#b2182b")
ax.text(V0 + 0.16, U(V0) + 0.06, r"$V_0$", ha="left", va="bottom", fontsize=17)

# --- descriptive state labels, in empty space (no arrows) ---
ax.text(Vm, -0.45, "stable\n(hyperpolarized)", ha="center", va="center",
        fontsize=11.5, color="#2166ac")
ax.text(Vp, -0.45, "stable\n(depolarized)",    ha="center", va="center",
        fontsize=11.5, color="#b2182b")
ax.text(V0, 0.33, "unstable", ha="center", va="center", fontsize=11.5, color="black")

ax.set_xlim(-2.4, 2.4)
ax.set_ylim(-1.85, 0.75)
ax.set_xlabel(r"Membrane potential $V$")
ax.set_ylabel(r"$U(V)=-\int f(V)\,\mathrm{d}V$")
ax.set_title(r"Double-well potential ($a=1.2,\ V_\pm=\pm1.5,\ V_0=0$)")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig1_double-well.png")
print("fig1_double-well.png written")
