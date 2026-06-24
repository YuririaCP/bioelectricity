"""
make_C3_C4_energy_dynamics.py
Yuriria Cortes-Poza
=============================
Generator for Figure 3 of the manuscript (fig:C3C4):

  Experiments 3-4 -- Energy dynamics and gap-junction adaptation.

  Left : Lyapunov energy E(V) = sum_i U(V_i) + (1/2) sum_ij G_ij (V_i-V_j)^2
         as a function of time (post-lesion), averaged over seeds (+/-1 s.d.),
         for three model variants. All descend monotonically (Theorem 1).
  Right: Gap-junction spatial heterogeneity sigma_G = std(G_ij) over edges.
         Only the full adaptive model develops sigma_G > 0; the fixed-G
         variant keeps sigma_G == 0; the R_i-knockout is intermediate.

Three variants:
  - full        : default couplings
  - R-knockout  : a_RW = a_RV = a_PR = a_RD = 0
  - fixed-G     : alpha_G = beta_G = gamma_G = lambda_G = 0  (G frozen)

Protocol (matches Experiment 2 setup): a half-depolarized / half-
hyperpolarized pattern is equilibrated, a central 50% lesion resets V to
the unstable point V_0 and injects a wound signal, and the tissue is then
integrated.  Lesion onset is t = 0 (dashed vertical line).

Parameters: N = 20, a = 1.2, G_init = 0.4, 6 seeds.

Output: fig3_energy-dynamics.png (project root) and figures/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bioelectric_model import ModelParameters, BioelectricNetwork
from attractor_analysis import lyapunov_energy

# -- large fonts (JMB revision) -------------------------------------------
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 17,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "lines.linewidth": 2.2, "savefig.dpi": 200, "savefig.bbox": "tight",
})

# -- experiment settings --------------------------------------------------
N           = 20
A_BIST      = 1.2
G_INIT      = 0.4
T_EQ        = 35.0     # equilibration of the half-half pattern
T_PRE       = 5.0      # recorded pre-lesion baseline
T_REGEN     = 60.0     # recorded post-lesion regeneration
REC_EVERY   = 50       # record every 50 steps (dt=0.01 -> every 0.5 t.u.)
WOUND_AMP   = 1.5
N_SEEDS     = 6

VARIANTS = {
    "Full model":      dict(),
    r"$R_i$ knockout": dict(a_RW=0.0, a_RV=0.0, a_PR=0.0, a_RD=0.0),
    "Fixed $G$":       dict(alpha_G=0.0, beta_G=0.0, gamma_G=0.0, lambda_G=0.0),
}
COLORS = {"Full model": "#b2182b", r"$R_i$ knockout": "#ef8a62",
          "Fixed $G$": "#2166ac"}


def run_one(overrides, seed):
    """One realization: returns (times_rel, E_series, sigmaG_series)."""
    p = ModelParameters(N=N, dim=1, dt=0.01, a_bist=A_BIST, G_init=G_INIT,
                        **overrides)
    net = BioelectricNetwork(p, seed=seed, polarity_gradient=True)

    # Half-depolarized / half-hyperpolarized initial pattern + small noise
    half = N // 2
    net.X[:half, net.IDX_V] = p.V_plus
    net.X[half:, net.IDX_V] = p.V_minus
    net.X[:, net.IDX_V] += 0.05 * net.rng.standard_normal(N)

    # Equilibrate (not recorded)
    net.run(T_EQ, record_every=10_000, verbose=False)

    # Edge mask from the structural graph (fixed throughout)
    mask = net.G > 0

    # Recorded pre-lesion baseline
    h_pre = net.run(T_PRE, record_every=REC_EVERY, verbose=False)
    t_lesion = net.t

    # ---- Lesion: central 50%, reset V -> V_0, inject wound signal ----
    q = N // 4
    lesion = np.arange(q, N - q)            # central 50% of the cells
    net.X[lesion, net.IDX_V] = p.V_0
    net.X[lesion, net.IDX_W] = WOUND_AMP

    # Recorded post-lesion regeneration
    h_post = net.run(T_REGEN, record_every=REC_EVERY, verbose=False)

    times = np.concatenate([h_pre["times"], h_post["times"]]) - t_lesion
    Vs    = list(h_pre["V"]) + list(h_post["V"])
    Gs    = list(h_pre["G"]) + list(h_post["G"])

    E      = np.array([lyapunov_energy(V, G, p) for V, G in zip(Vs, Gs)])
    sigmaG = np.array([G[mask].std() for G in Gs])
    return times, E, sigmaG


def collect(overrides):
    """Average over seeds; returns times, (E_mean,E_std), (sg_mean,sg_std)."""
    T, E_all, S_all = None, [], []
    for s in range(N_SEEDS):
        t, E, sg = run_one(overrides, seed=s)
        T = t if T is None else T
        E_all.append(E); S_all.append(sg)
    L = min(min(len(e) for e in E_all), len(T))
    T = T[:L]
    E_all = np.array([e[:L] for e in E_all])
    S_all = np.array([s[:L] for s in S_all])
    return T, (E_all.mean(0), E_all.std(0)), (S_all.mean(0), S_all.std(0))


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    fig, (axE, axS) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for name, ov in VARIANTS.items():
        print(f"  [{name}] running {N_SEEDS} seeds ...")
        T, (Em, Es), (Sm, Ss) = collect(ov)
        c = COLORS[name]
        axE.plot(T, Em, color=c, label=name)
        axE.fill_between(T, Em - Es, Em + Es, color=c, alpha=0.20, lw=0)
        axS.plot(T, Sm, color=c, label=name)
        axS.fill_between(T, Sm - Ss, Sm + Ss, color=c, alpha=0.20, lw=0)

    for ax in (axE, axS):
        ax.axvline(0.0, ls="--", color="0.4", lw=1.6)
        ax.set_xlabel(r"Time after lesion $t$")
        ax.grid(alpha=0.25)

    axE.set_ylabel("")
    axE.set_title(r"Lyapunov energy $E(\mathbf{V})$  ($\dot E \leq 0$)")
    axE.legend(frameon=False, loc="upper right")

    axS.set_ylabel("")
    axS.set_title(r"Gap-junction heterogeneity  $\sigma_G=\mathrm{std}(G_{ij})$")
    axS.legend(frameon=False, loc="lower right")

    fig.suptitle("Experiments 3--4: energy dynamics and gap-junction adaptation",
                 fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for d in (save_dir, os.path.join(save_dir, "figures")):
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "fig3_energy-dynamics.png"))
    print("  wrote fig3_energy-dynamics.png (root + figures/)")


if __name__ == "__main__":
    print("Generating Figure 3 (C3_C4_energy_dynamics) ...")
    main()
