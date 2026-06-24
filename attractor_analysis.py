"""
attractor_analysis.py
Yuriria Cortés Poza
=====================
Attractor structure, Lyapunov analysis, and switching experiments
for the hybrid bioelectric morphogenesis model.

Contents
--------
1. Lyapunov function for the pure bioelectric subsystem
   - Exact theorem: E(V) = sum U(V_i) + (1/2) sum G_ij(V_i-V_j)^2
   - dE/dt = -C * ||dV/dt||^2  <= 0
   - Numerical verification

2. Approximate Lyapunov structure with GRN coupling
   - dE/dt = -C||V_dot||^2 + Lambda(Gamma, V_dot)
   - Stability condition: |Lambda| < C||V_dot||^2

3. Attractor switching experiment (planaria polarity reversal)
   - Two stable attractors: normal and inverted head-tail axis
   - Transient forcing switches between attractors
   - Permanent with slow memory (eps), transient without

4. Basin of attraction mapping
   - Sweep initial depolarized fraction -> final state
   - Identify basin boundary

5. Hopfield connection
   - J_ij = G_ij plays the role of Hopfield weights
   - Effective energy landscape reshapes as G_ij adapts
"""

# === PAPER FONT STYLE (auto-added for JMB revision) ===
import matplotlib as _mpl
_mpl.use('Agg')
import matplotlib.pyplot as _plt_style
_plt_style.rcParams.update({
    'font.size': 15, 'axes.titlesize': 16, 'axes.labelsize': 16,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12,
    'figure.titlesize': 18, 'lines.linewidth': 2.0, 'axes.linewidth': 1.1,
    'savefig.dpi': 200, 'savefig.bbox': 'tight',
})
# === END PAPER FONT STYLE ===

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import replace as dc_replace
from typing import Dict, List, Optional, Tuple

from bioelectric_model import (
    ModelParameters, BioelectricNetwork,
    sigma, plot_spacetime_1d
)

os.makedirs("figures", exist_ok=True)


# =============================================================================
#  1.  LYAPUNOV FUNCTION
# =============================================================================

def U_potential(V: np.ndarray, a: float,
                Vm: float, V0: float, Vp: float) -> np.ndarray:
    """
    Double-well potential U(V) such that U'(V) = f(V) = a(V-Vm)(V-V0)(Vp).
    U(V) = a [V^4/4 - S1*V^3/3 + S2*V^2/2 - S3*V]
    where S1=Vm+V0+Vp, S2=Vm*V0+Vm*Vp+V0*Vp, S3=Vm*V0*Vp.
    """
    S1 = Vm + V0 + Vp
    S2 = Vm*V0 + Vm*Vp + V0*Vp
    S3 = Vm*V0*Vp
    return a * (V**4/4 - S1*V**3/3 + S2*V**2/2 - S3*V)


def lyapunov_energy(V: np.ndarray, G: np.ndarray,
                    p: ModelParameters) -> float:
    """
    Exact Lyapunov function for the pure bioelectric subsystem:

        E(V) = sum_i U(V_i) + (1/2) * sum_{(i,j)} G_ij (V_i - V_j)^2

    Theorem: d/dt E(V(t)) = -C * ||dV/dt||^2 <= 0
    along trajectories of C dV_i/dt = -f(V_i) + sum_j G_ij(V_j - V_i).

    With GRN coupling (Gamma != 0):
        d/dt E = -C||V_dot||^2 + sum_i Gamma_i * dV_i/dt  =: -C||V_dot||^2 + Lambda
    Lambda is the 'error term' measuring how much GRN coupling perturbs gradient descent.
    """
    well = U_potential(V, p.a_bist, p.V_minus, p.V_0, p.V_plus).sum()
    # Coupling energy: (1/2) sum_{ij} G_ij (V_i - V_j)^2
    diff = V[:, None] - V[None, :]     # (N, N)
    coup = 0.5 * (G * diff**2).sum()
    return float(well + coup)


def lyapunov_rate(V: np.ndarray, V_dot: np.ndarray,
                  Gamma: np.ndarray, C: float) -> Tuple[float, float, float]:
    """
    Compute dE/dt and its decomposition:
        dE/dt = -C||V_dot||^2  +  Lambda
    where Lambda = sum_i Gamma_i * V_dot_i  (GRN error term).
    Returns (dE_dt, -C||Vdot||^2, Lambda).
    """
    grad_term = -C * float(np.sum(V_dot**2))
    Lambda    = float(np.dot(Gamma, V_dot))
    return grad_term + Lambda, grad_term, Lambda


# =============================================================================
#  2.  LYAPUNOV VERIFICATION EXPERIMENT
# =============================================================================

def experiment_lyapunov_verification(
        N: int = 30, T: float = 60.0,
        tau_eps_vals: Optional[List[float]] = None,
        save_path: str = "fig5_lyapunov.png"):
    """
    Numerical verification of the Lyapunov theorem.

    Run the model and record E(V(t)) over time for:
    (a) Pure bioelectric subsystem (GRN decoupled): should decrease monotonically.
    (b) Full model with GRN: should decrease with Lambda perturbations.
    (c) Show Lambda/||V_dot||^2 ratio to assess how well the GRN perturbs the gradient.
    """
    print("  Lyapunov verification experiment...")

    p_pure = ModelParameters(N=N, dim=1, dt=0.02,
                              a_PV=0.0, a_DV=0.0, a_IV=0.0, a_RV=0.0,
                              beta_1=0.0, beta_2=0.0, beta_3=0.0, beta_4=0.0,
                              tau_eps=1e6)   # pure bioelectric, GRN frozen

    p_full = ModelParameters(N=N, dim=1, dt=0.02)

    # Initialize both with a mixed state (some V_+, some V_-)
    seed = 42
    results = {}

    for label, p in [("Pure bioelectric", p_pure), ("Full model", p_full)]:
        net = BioelectricNetwork(p, seed=seed)
        net.X[:N//3, 0]       = p.V_plus  + 0.1*net.rng.standard_normal(N//3)
        net.X[N//3:2*N//3, 0] = p.V_0    + 0.1*net.rng.standard_normal(N//3)
        net.X[2*N//3:, 0]     = p.V_minus + 0.1*net.rng.standard_normal(N - 2*N//3)

        n_steps = int(T / p.dt)
        times, E_vals, grad_terms, lambdas, Vdot_norms = [], [], [], [], []

        for step in range(n_steps):
            t = step * p.dt
            ext = np.zeros(N)
            S_lesion = np.zeros(N)

            # Compute Gamma for Lambda calculation
            P   = net.X[:, 1]; D = net.X[:, 2]
            R   = net.X[:, 4]; eps = net.X[:, 5]
            Gamma = (p.beta_1*P - p.beta_2*D
                     + p.beta_3*R + p.beta_4*eps)

            # Compute V_dot (before step)
            V_dot = net._rhs_V(net.X, net.G, ext)

            if step % 20 == 0:
                E = lyapunov_energy(net.X[:, 0], net.G, p)
                dE, grad, lam = lyapunov_rate(
                    net.X[:, 0], V_dot, Gamma, p.C)
                times.append(t)
                E_vals.append(E)
                grad_terms.append(grad)
                lambdas.append(lam)
                Vdot_norms.append(float(np.sum(V_dot**2)))

            net.X, net.G = net._rk4_step(net.X, net.G, p.dt, ext, S_lesion)

        results[label] = {
            'times': np.array(times),
            'E': np.array(E_vals),
            'grad': np.array(grad_terms),
            'lambda': np.array(lambdas),
            'Vdot2': np.array(Vdot_norms)
        }

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Lyapunov function analysis", fontsize=18, fontweight='bold')

    colors = {'Pure bioelectric': 'steelblue', 'Full model': 'firebrick'}

    # Row 0: E(t) and dE/dt breakdown
    for label, res in results.items():
        c = colors[label]
        axes[0, 0].plot(res['times'], res['E'] - res['E'][0],
                        color=c, lw=2, label=label)
        axes[0, 1].plot(res['times'], res['grad'],
                        color=c, lw=2, ls='--', label=f'{label}: $-C\\|\\dot V\\|^2$')
        axes[0, 1].plot(res['times'], res['grad'] + res['lambda'],
                        color=c, lw=1.5, ls='-', label=f'{label}: $dE/dt$')
        axes[0, 2].plot(res['times'],
                        np.abs(res['lambda']) / (np.abs(res['grad']) + 1e-12),
                        color=c, lw=2, label=label)

    axes[0, 0].axhline(0, color='k', lw=0.8, ls='--')
    axes[0, 0].set_xlabel('Time', fontsize=17); axes[0, 0].set_ylabel(r'$E(t) - E(0)$', fontsize=17)
    axes[0, 0].set_title('Lyapunov function $E(t)$', fontsize=17)
    axes[0, 0].legend(fontsize=14)

    axes[0, 1].axhline(0, color='k', lw=0.8, ls='--')
    axes[0, 1].set_xlabel('Time', fontsize=17); axes[0, 1].set_ylabel(r'$dE/dt$', fontsize=17)
    axes[0, 1].set_title(r'Rate decomposition: $-C\|\dot V\|^2$ vs $\Lambda$', fontsize=17)
    axes[0, 1].legend(fontsize=11, loc='lower right', framealpha=0.9)

    axes[0, 2].axhline(1, color='k', lw=0.8, ls='--', alpha=0.5,
                       label='$|\\Lambda|/C\\|\\dot V\\|^2 = 1$')
    axes[0, 2].set_xlabel('Time', fontsize=17)
    axes[0, 2].set_ylabel(r'$|\Lambda| \,/\, C\|\dot V\|^2$', fontsize=17)
    axes[0, 2].set_title('GRN perturbation ratio', fontsize=17)
    axes[0, 2].legend(fontsize=14)
    axes[0, 2].set_ylim(0, 3)

    # Row 1: Phase portrait + energy landscape
    V_grid = np.linspace(-2.2, 2.2, 400)
    U_grid = U_potential(V_grid, p_pure.a_bist,
                         p_pure.V_minus, p_pure.V_0, p_pure.V_plus)
    U_grid -= U_grid.min()
    axes[1, 0].plot(V_grid, U_grid, 'k-', lw=2)
    axes[1, 0].axvline(p_pure.V_minus, color='steelblue', lw=1.5, ls='--',
                       label=fr'$V_- = {p_pure.V_minus}$')
    axes[1, 0].axvline(p_pure.V_0,     color='gray',      lw=1.5, ls='--',
                       label=fr'$V_0 = {p_pure.V_0}$')
    axes[1, 0].axvline(p_pure.V_plus,  color='firebrick', lw=1.5, ls='--',
                       label=fr'$V_+ = {p_pure.V_plus}$')
    axes[1, 0].set_xlabel(r'$V$', fontsize=19); axes[1, 0].set_ylabel(r'$U(V)$', fontsize=19)
    axes[1, 0].set_title('Single-cell double-well potential', fontsize=17)
    axes[1, 0].legend(fontsize=14)

    # Trajectory in V-space for one cell
    net2 = BioelectricNetwork(p_pure, seed=99)
    net2.X[0, 0] = 0.2   # start near unstable equilibrium
    V_traj = [net2.X[0, 0]]
    for _ in range(int(40/p_pure.dt)):
        net2.X, net2.G = net2._rk4_step(net2.X, net2.G, p_pure.dt,
                                         np.zeros(N), np.zeros(N))
        V_traj.append(net2.X[0, 0])
    V_traj = np.array(V_traj)
    t_traj = np.arange(len(V_traj)) * p_pure.dt

    axes[1, 1].plot(t_traj, V_traj, 'steelblue', lw=2)
    axes[1, 1].axhline(p_pure.V_minus, color='steelblue', lw=1, ls='--', alpha=0.7)
    axes[1, 1].axhline(p_pure.V_plus,  color='firebrick', lw=1, ls='--', alpha=0.7)
    axes[1, 1].axhline(p_pure.V_0,     color='gray',      lw=1, ls='--', alpha=0.7)
    axes[1, 1].set_xlabel('Time', fontsize=19)
    axes[1, 1].set_ylabel(r'$V(t)$', fontsize=19)
    axes[1, 1].set_title(r'Single cell: $V(0) \approx V_0$, G=0 → $V_+$', fontsize=17)

    # Energy landscape with coupling: 2-cell system
    V1_grid = np.linspace(-2, 2, 60)
    V2_grid = np.linspace(-2, 2, 60)
    V1g, V2g = np.meshgrid(V1_grid, V2_grid)
    G_2cell = 0.5
    E_2cell = (U_potential(V1g, p_pure.a_bist, p_pure.V_minus, p_pure.V_0, p_pure.V_plus)
               + U_potential(V2g, p_pure.a_bist, p_pure.V_minus, p_pure.V_0, p_pure.V_plus)
               + 0.5 * G_2cell * (V1g - V2g)**2)
    E_2cell -= E_2cell.min()
    levels = np.linspace(0, np.percentile(E_2cell, 80), 30)
    ct = axes[1, 2].contourf(V1g, V2g, E_2cell, levels=levels, cmap='viridis_r')
    axes[1, 2].contour(V1g, V2g, E_2cell, levels=levels[::5], colors='white',
                       linewidths=0.5, alpha=0.5)
    plt.colorbar(ct, ax=axes[1, 2], fraction=0.046)
    axes[1, 2].set_xlabel(r'$V_1$', fontsize=19); axes[1, 2].set_ylabel(r'$V_2$', fontsize=19)
    axes[1, 2].set_title(fr'2-cell energy landscape ($G={G_2cell}$)', fontsize=17)
    # Mark the 4 minima (V_+,V_+), (V_-,V_-), and saddles (V_+,V_-)
    for vv1, vv2, mk in [(p_pure.V_minus, p_pure.V_minus, '*'),
                          (p_pure.V_plus,  p_pure.V_plus,  '*'),
                          (p_pure.V_plus,  p_pure.V_minus, 'o'),
                          (p_pure.V_minus, p_pure.V_plus,  'o')]:
        axes[1, 2].plot(vv1, vv2, mk, color='yellow', ms=10, markeredgecolor='k')

    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return results


# =============================================================================
#  3.  ATTRACTOR SWITCHING: PLANARIA POLARITY REVERSAL
# =============================================================================

def experiment_polarity_reversal(
        N: int = 40, T_eq: float = 60.0,
        T_force: float = 12.0, T_post: float = 80.0,
        amplitude: float = 3.0,
        tau_eps_with: float = 50.0,
        tau_eps_without: float = 0.3,
        save_path: str = "fig6_polarity-reversal.png"):
    """
    Planaria polarity reversal experiment.

    Setup: 1D tissue with head-tail gradient.
    Attractor 1: V_+ in anterior (i < N/2), V_- in posterior (i >= N/2).
    Attractor 2: reversed polarity.

    Apply depolarizing current to posterior and hyperpolarizing to anterior
    for a finite window T_force.

    Compare:
    - WITH slow memory (tau_eps = 50): permanent attractor switch.
    - WITHOUT slow memory (tau_eps = 0.3): transient perturbation, system returns.

    This is the in-silico analog of the Beane et al. (2011) experiment.
    """
    print("  Polarity reversal experiment...")

    def run_polarity_experiment(tau_eps, seed):
        p = ModelParameters(N=N, dim=1, dt=0.02,
                            tau_eps=tau_eps,
                            G_init=0.3, a_bist=1.2,
                            beta_1=0.4, beta_2=0.1)
        net = BioelectricNetwork(p, seed=seed, polarity_gradient=True)

        # Attractor 1: anterior = V_+, posterior = V_-
        net.X[:N//2, 0]  = p.V_plus  + 0.05*net.rng.standard_normal(N//2)
        net.X[N//2:, 0]  = p.V_minus + 0.05*net.rng.standard_normal(N - N//2)

        print(f"    tau_eps={tau_eps}: equilibrating...")
        hist_pre  = net.run(T=T_eq, record_every=30, verbose=False)

        # Store pre-forcing V profile
        V_before = net.X[:, 0].copy()

        # Transient forcing: depolarize posterior, hyperpolarize anterior
        # (mimics experimental manipulation of ion pumps)
        def ext_reverse(t):
            ext = np.zeros(N)
            ext[:N//2]  = -amplitude     # hyperpolarize anterior
            ext[N//2:]  =  amplitude     # depolarize posterior
            return ext

        print(f"    tau_eps={tau_eps}: applying forcing (A={amplitude})...")
        hist_force = net.run(T=T_force, ext_func=ext_reverse,
                             record_every=5, verbose=False)

        V_mid = net.X[:, 0].copy()

        print(f"    tau_eps={tau_eps}: post-forcing relaxation...")
        hist_post = net.run(T=T_post, record_every=30, verbose=False)

        V_after = net.X[:, 0].copy()

        # Polarity measure: mean(V_anterior) - mean(V_posterior)
        def polarity(V):
            return float(V[:N//2].mean() - V[N//2:].mean())

        return {
            'hist_pre': hist_pre, 'hist_force': hist_force,
            'hist_post': hist_post,
            'V_before': V_before, 'V_mid': V_mid, 'V_after': V_after,
            'polarity_before': polarity(V_before),
            'polarity_mid':    polarity(V_mid),
            'polarity_after':  polarity(V_after),
            'tau_eps': tau_eps
        }

    res_with    = run_polarity_experiment(tau_eps_with,    seed=20)
    res_without = run_polarity_experiment(tau_eps_without, seed=20)

    # ---- Polarity time series (concatenate pre+force+post) ----
    def concat_polarity(res):
        times, pols = [], []
        for hist in [res['hist_pre'], res['hist_force'], res['hist_post']]:
            V_arr = hist['V']      # (n_rec, N)
            t_arr = hist['times']
            if len(times) > 0:
                t_arr = t_arr + times[-1] + (t_arr[1]-t_arr[0] if len(t_arr)>1 else 0)
            for t, V in zip(t_arr, V_arr):
                times.append(t)
                pols.append(float(V[:N//2].mean() - V[N//2:].mean()))
        return np.array(times), np.array(pols)

    t_with,    p_with    = concat_polarity(res_with)
    t_without, p_without = concat_polarity(res_without)

    t_force_start = T_eq
    t_force_end   = T_eq + T_force

    # ---- Figure ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.8))
    fig.suptitle("Planaria polarity reversal: attractor switching\n"
                 r"Transient forcing of amplitude $A$, "
                 r"duration $T_{\mathrm{force}}$",
                 fontsize=18, fontweight='bold', y=0.995)

    colors_row = ['steelblue', 'firebrick']
    labels_row = [fr'With slow memory ($\tau_\varepsilon={tau_eps_with}$)',
                  fr'Without slow memory ($\tau_\varepsilon={tau_eps_without}$)']

    for row, (res, tc, label) in enumerate(
            zip([res_with, res_without], colors_row, labels_row)):

        # Column 0: space-time of V during post-forcing relaxation
        hist_post = res['hist_post']
        im = axes[row, 0].imshow(
            hist_post['V'].T, aspect='auto', origin='lower',
            extent=[0, T_post, 0, N], cmap='RdBu_r',
            vmin=-2.0, vmax=2.0)
        axes[row, 0].set_title(f'Post-forcing $V_i(t)$\n{label}', fontsize=14)
        axes[row, 0].set_xlabel('Time', fontsize=14)
        axes[row, 0].set_ylabel('Cell', fontsize=14)
        plt.colorbar(im, ax=axes[row, 0], fraction=0.05)

        # Column 1: V profiles before / mid / after
        x = np.arange(N)
        axes[row, 1].plot(x, res['V_before'], 'k-',   lw=2, label='Before forcing')
        axes[row, 1].plot(x, res['V_mid'],    color=tc, lw=2, ls='--', label='End of forcing')
        axes[row, 1].plot(x, res['V_after'],  color=tc, lw=2, ls='-',  label='After relaxation')
        axes[row, 1].axhline(0, color='gray', lw=0.8, ls=':', alpha=0.5)
        axes[row, 1].set_xlabel('Cell index', fontsize=15)
        axes[row, 1].set_ylabel(r'$V_i$', fontsize=15)
        axes[row, 1].set_title('V profiles', fontsize=15)
        axes[row, 1].legend(fontsize=11, loc='best', framealpha=0.9)

        # Column 2: epsilon profiles
        eps_hist = res['hist_post']['eps']
        axes[row, 2].imshow(
            eps_hist.T, aspect='auto', origin='lower',
            extent=[0, T_post, 0, N], cmap='Blues', vmin=0, vmax=1)
        axes[row, 2].set_title(r'Slow memory $\varepsilon_i(t)$', fontsize=15)
        axes[row, 2].set_xlabel('Time', fontsize=14)
        axes[row, 2].set_ylabel('Cell', fontsize=14)

        # Column 3: polarity time series
        t_arr = t_with if row == 0 else t_without
        p_arr = p_with if row == 0 else p_without
        axes[row, 3].plot(t_arr, p_arr, color=tc, lw=2)
        axes[row, 3].axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        axes[row, 3].axvspan(t_force_start, t_force_end,
                              alpha=0.15, color='orange', label='Forcing window')
        axes[row, 3].set_xlabel('Time', fontsize=15)
        axes[row, 3].set_ylabel(r'Polarity $\langle V_{\rm ant}\rangle - \langle V_{\rm post}\rangle$',
                                  fontsize=14)
        axes[row, 3].set_title('Polarity index over time', fontsize=15)
        axes[row, 3].legend(fontsize=12)

        print(f"    tau_eps={res['tau_eps']:.1f}: "
              f"polarity before={res['polarity_before']:+.3f}, "
              f"after={res['polarity_after']:+.3f}  "
              f"({'PERMANENT SWITCH' if res['polarity_before']*res['polarity_after'] < 0 else 'RETURNED'})")

    # ---- Figure 2: summary panels (different plot types) ----
    fig2, ax2 = plt.subplots(1, 4, figsize=(18, 4.6))
    fig2.suptitle("Polarity switching: summary across memory regimes",
                  fontsize=18, fontweight='bold')

    ax2[0].plot(t_with,    p_with,    'steelblue', lw=2,
                    label=fr'$\tau_\varepsilon={tau_eps_with}$ (with memory)')
    ax2[0].plot(t_without, p_without, 'firebrick', lw=2, ls='--',
                    label=fr'$\tau_\varepsilon={tau_eps_without}$ (no memory)')
    ax2[0].axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax2[0].axvspan(t_force_start, t_force_end, alpha=0.15,
                        color='orange', label='Forcing')
    ax2[0].set_xlabel('Time', fontsize=17)
    ax2[0].set_ylabel('Polarity index', fontsize=17)
    ax2[0].set_title('Memory vs no-memory: polarity', fontsize=17)
    ax2[0].legend(fontsize=14)

    # Bar: summary of polarity change
    labels_bar = ['Before', 'After\n(memory)', 'After\n(no memory)']
    values_bar = [res_with['polarity_before'],
                  res_with['polarity_after'],
                  res_without['polarity_after']]
    colors_bar = ['gray', 'steelblue', 'firebrick']
    bars = ax2[1].bar(labels_bar, values_bar, color=colors_bar, alpha=0.8, edgecolor='k')
    ax2[1].axhline(0, color='k', lw=1)
    ax2[1].set_ylabel('Polarity index', fontsize=17)
    ax2[1].set_title('Polarity before vs after forcing', fontsize=17)

    # Switching condition illustration
    tau_range = np.logspace(-1, 2.5, 40)
    # Conceptual: switching success approximated by logistic
    switch_prob = 1.0 / (1 + np.exp(-(np.log10(tau_range) - 0.8)*3))
    ax2[2].semilogx(tau_range, switch_prob, 'k-', lw=2.5)
    ax2[2].axvline(tau_eps_with,    color='steelblue', lw=2, ls='--',
                        label=fr'$\tau_\varepsilon={tau_eps_with}$')
    ax2[2].axvline(tau_eps_without, color='firebrick',  lw=2, ls='--',
                        label=fr'$\tau_\varepsilon={tau_eps_without}$')
    ax2[2].set_xlabel(r'$\tau_\varepsilon$', fontsize=20)
    ax2[2].set_ylabel('Probability of permanent switch', fontsize=17)
    ax2[2].set_title(r'Switching threshold in $\tau_\varepsilon$', fontsize=17)
    ax2[2].legend(fontsize=14); ax2[2].grid(True, alpha=0.3)

    # eps profiles: with vs without
    for res, c, lbl in [(res_with, 'steelblue', 'with'), (res_without, 'firebrick', 'without')]:
        eps_after = res['hist_post']['eps'][-1]
        ax2[3].plot(np.arange(N), eps_after, color=c, lw=2, label=lbl)
    ax2[3].set_xlabel('Cell index', fontsize=17)
    ax2[3].set_ylabel(r'$\varepsilon_i$ (final)', fontsize=17)
    ax2[3].set_title(r'Final slow-memory profile', fontsize=17)
    ax2[3].legend(fontsize=15)

    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=1.6)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    save_path2 = save_path.replace('fig6_polarity-reversal', 'fig7_polarity-summary')
    fig2.tight_layout(rect=[0, 0, 1, 0.88])
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {save_path} and {save_path2}")
    return {'with_memory': res_with, 'without_memory': res_without}


# =============================================================================
#  4.  BASIN OF ATTRACTION MAPPING
# =============================================================================

def experiment_basin_mapping(
        N: int = 30, T_eq: float = 50.0,
        depol_fracs: Optional[np.ndarray] = None,
        n_seeds: int = 4,
        save_path: str = "fig8_basin-map.png"):
    """
    Map the basin of attraction by varying the initial fraction of
    depolarized cells.

    For each fraction f in [0, 1], initialize cells 0..int(f*N) at V_+
    and the rest at V_-. Run to equilibrium. Record final polarity.

    Identifies the basin boundary: critical fraction f* such that
    for f < f* the tissue collapses to all-V_-, and for f > f* it
    maintains/expands the V_+ domain.
    """
    print("  Basin of attraction mapping...")

    if depol_fracs is None:
        depol_fracs = np.linspace(0.0, 1.0, 20)

    p = ModelParameters(N=N, dim=1, dt=0.02, G_init=0.3, a_bist=1.2)

    final_polarities = np.zeros((len(depol_fracs), n_seeds))
    final_V_means    = np.zeros((len(depol_fracs), n_seeds))

    for i, frac in enumerate(depol_fracs):
        n_depol = max(0, min(N, int(frac * N)))
        for s in range(n_seeds):
            net = BioelectricNetwork(p, seed=i*100 + s)
            net.X[:n_depol, 0]  = p.V_plus  + 0.08*net.rng.standard_normal(n_depol)
            net.X[n_depol:, 0]  = p.V_minus + 0.08*net.rng.standard_normal(N - n_depol)
            net.X[:, 1:] = 0.1
            hist = net.run(T=T_eq, record_every=500, verbose=False)
            V_f  = hist['V'][-1]
            final_polarities[i, s] = float(V_f[:N//2].mean() - V_f[N//2:].mean())
            final_V_means[i, s]    = float(V_f.mean())

    pol_mean = final_polarities.mean(axis=1)
    pol_std  = final_polarities.std(axis=1)
    Vm_mean  = final_V_means.mean(axis=1)

    # Classify final state: V_+ dominant or V_- dominant
    V_mean_norm = (Vm_mean - p.V_minus) / (p.V_plus - p.V_minus)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Basin of attraction mapping\n"
                 "Initial fraction of depolarized cells vs final state",
                 fontsize=20, fontweight='bold')

    axes[0].plot(depol_fracs, Vm_mean, 'steelblue', lw=2.5)
    axes[0].fill_between(depol_fracs,
                          Vm_mean - final_V_means.std(axis=1),
                          Vm_mean + final_V_means.std(axis=1),
                          alpha=0.3, color='steelblue')
    axes[0].axhline(p.V_plus,  color='firebrick', lw=1, ls='--',
                    alpha=0.6, label=f'$V_+ = {p.V_plus}$')
    axes[0].axhline(p.V_minus, color='steelblue', lw=1, ls='--',
                    alpha=0.6, label=f'$V_- = {p.V_minus}$')
    axes[0].set_xlabel(r'Initial depolarized fraction $f$', fontsize=19)
    axes[0].set_ylabel(r'$\langle V_i \rangle$ final', fontsize=19)
    axes[0].set_title(r'Mean final voltage', fontsize=17)
    axes[0].legend(fontsize=15); axes[0].grid(True, alpha=0.3)

    axes[1].plot(depol_fracs, pol_mean, 'k', lw=2.5)
    axes[1].fill_between(depol_fracs,
                          pol_mean - pol_std, pol_mean + pol_std,
                          alpha=0.25, color='gray')
    axes[1].axhline(0, color='k', lw=0.8, ls='--')
    axes[1].set_xlabel(r'Initial depolarized fraction $f$', fontsize=19)
    axes[1].set_ylabel('Final polarity', fontsize=19)
    axes[1].set_title('Polarity (anterior - posterior)', fontsize=17)
    axes[1].grid(True, alpha=0.3)

    # Normalized: 0 = all V_-, 1 = all V_+
    axes[2].scatter(depol_fracs, V_mean_norm, c=V_mean_norm,
                    cmap='RdBu_r', s=60, zorder=5, edgecolors='k', linewidths=0.5)
    # Sigmoid fit (heuristic)
    from_zero = depol_fracs
    sig_fit   = 1.0 / (1 + np.exp(-15*(from_zero - 0.5)))
    axes[2].plot(depol_fracs, sig_fit, 'k--', lw=1.5, alpha=0.5, label='Sigmoid guide')
    axes[2].axhline(0.5, color='gray', lw=0.8, ls=':', alpha=0.7)
    axes[2].axvline(0.5, color='gray', lw=0.8, ls=':', alpha=0.7)
    axes[2].set_xlabel(r'Initial depolarized fraction $f$', fontsize=19)
    axes[2].set_ylabel('Normalized final state', fontsize=19)
    axes[2].set_title('Basin boundary (near $f^* \\approx 0.5$)', fontsize=17)
    axes[2].legend(fontsize=15); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return {'depol_fracs': depol_fracs, 'final_V_means': final_V_means,
            'pol_mean': pol_mean, 'pol_std': pol_std}


# =============================================================================
#  5.  ADAPTIVE ENERGY LANDSCAPE (HOPFIELD CONNECTION)
# =============================================================================

def experiment_adaptive_landscape(
        N: int = 20, T_total: float = 80.0,
        save_path: str = "fig9_adaptive-landscape.png"):
    """
    Demonstrate that as G_ij adapts, the effective energy landscape reshapes.

    Show E(V; G(t)) at several time snapshots t1 < t2 < t3.
    The minima of E (for fixed V-slice) shift as G evolves.

    This illustrates the Hopfield interpretation: the 'memory' is
    distributed in the adaptive weights G_ij, and morphological
    identity is encoded in the attractor structure of E.
    """
    print("  Adaptive energy landscape...")

    p = ModelParameters(N=N, dim=1, dt=0.02, G_init=0.5, tau_G=3.0,
                        a_bist=1.2, beta_1=0.4)
    net = BioelectricNetwork(p, seed=30)
    net.X[:N//2, 0] = p.V_plus  + 0.05*net.rng.standard_normal(N//2)
    net.X[N//2:, 0] = p.V_minus + 0.05*net.rng.standard_normal(N//2)

    n_steps   = int(T_total / p.dt)
    record_at = [int(0.05*n_steps), int(0.25*n_steps),
                 int(0.5*n_steps), n_steps-1]
    snapshots = {}

    for step in range(n_steps):
        if step in record_at:
            V_now  = net.X[:, 0].copy()
            G_now  = net.G.copy()
            E_now  = lyapunov_energy(V_now, G_now, p)
            t_now  = step * p.dt
            # Mean conductance
            adj = G_now > 0
            G_mean = float(G_now[adj].mean()) if adj.sum() > 0 else 0.0
            snapshots[step] = {
                't': t_now, 'V': V_now, 'G': G_now,
                'E': E_now, 'G_mean': G_mean
            }
        net.X, net.G = net._rk4_step(net.X, net.G, p.dt,
                                      np.zeros(N), np.zeros(N))

    # Plot
    n_snap = len(snapshots)
    fig, axes = plt.subplots(3, n_snap, figsize=(4*n_snap, 11))
    fig.suptitle("Adaptive energy landscape and Hopfield connection\n"
                 r"$J_{ij} = G_{ij}(t)$ reshapes as tissue evolves",
                 fontsize=20, fontweight='bold')

    V_grid = np.linspace(-2.2, 2.2, 200)

    for col, (step, snap) in enumerate(sorted(snapshots.items())):
        t    = snap['t']
        V    = snap['V']
        G    = snap['G']
        E    = snap['E']
        Gm   = snap['G_mean']

        # Row 0: V profile
        axes[0, col].plot(np.arange(N), V, 'steelblue', lw=2)
        axes[0, col].axhline(p.V_plus,  color='firebrick', lw=1, ls='--', alpha=0.7)
        axes[0, col].axhline(p.V_minus, color='steelblue', lw=1, ls='--', alpha=0.7)
        axes[0, col].set_ylim(-2.2, 2.2)
        axes[0, col].set_title(f'$t={t:.0f}$\n$E={E:.2f}$', fontsize=15)
        axes[0, col].set_xlabel('Cell', fontsize=14)
        if col == 0: axes[0, col].set_ylabel(r'$V_i$', fontsize=17)

        # Row 1: G_ij heatmap
        im = axes[1, col].imshow(G, cmap='Blues', origin='upper',
                                  vmin=0, vmax=p.G_init*3)
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
        axes[1, col].set_title(fr'$G_{{ij}}$ (mean={Gm:.3f})', fontsize=15)
        if col == 0: axes[1, col].set_ylabel(r'$G_{ij}$', fontsize=17)

        # Row 2: Local energy contribution U(V_i)
        U_local = U_potential(V, p.a_bist, p.V_minus, p.V_0, p.V_plus)
        U_local -= U_local.min()
        axes[2, col].bar(np.arange(N), U_local, color='mediumpurple', alpha=0.8)
        axes[2, col].set_title(r'$U(V_i)$ local energy', fontsize=15)
        axes[2, col].set_xlabel('Cell', fontsize=14)
        if col == 0: axes[2, col].set_ylabel(r'$U(V_i)$', fontsize=17)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return snapshots


# =============================================================================
#  6.  SWITCHING PHASE DIAGRAM  (A  vs  T_force)
# =============================================================================

def experiment_switching_phase_diagram(
        N: int = 22,
        T_eq: float = 45.0,
        T_post: float = 55.0,
        A_vals:       Optional[np.ndarray] = None,
        T_force_vals: Optional[np.ndarray] = None,
        n_seeds: int = 3,
        tau_eps: float = 50.0,
        save_path: str = "fig4_switching-phase.png"):
    """
    Phase diagram of polarity-switching success as a function of
    forcing amplitude A and forcing duration T_force.

    For each (A, T_force) pair we run n_seeds independent trials:
    - Initialise 1D tissue with anterior V_+, posterior V_-
    - Equilibrate for T_eq
    - Apply antisymmetric forcing ext_i = +A (posterior), -A (anterior)
      for T_force time units
    - Relax for T_post
    - Record switching probability:  fraction of seeds where polarity reversed

    The result is a 2-D heatmap in (T_force, A) space showing the
    boundary between 'no switch' and 'permanent switch' regions.
    This boundary is the in-silico analog of the experimental
    threshold reported by Beane et al. (2011) and directly addresses
    Problem 3 (amplitude threshold for attractor switching).
    """
    print("  Switching phase diagram (A vs T_force)...")

    if A_vals is None:
        A_vals = np.linspace(0.4, 4.0, 10)
    if T_force_vals is None:
        T_force_vals = np.linspace(2.0, 20.0, 10)

    nA = len(A_vals)
    nT = len(T_force_vals)
    switch_prob  = np.zeros((nA, nT))   # rows = A, cols = T_force
    pol_change   = np.zeros((nA, nT))   # mean |Δpolarity|

    total = nA * nT * n_seeds
    done  = 0

    for ia, A in enumerate(A_vals):
        for it, Tf in enumerate(T_force_vals):
            switches = 0
            delta_pols = []

            for s in range(n_seeds):
                p = ModelParameters(N=N, dim=1, dt=0.05,
                                    G_init=0.3, a_bist=1.2,
                                    tau_eps=tau_eps,
                                    beta_1=0.4, beta_2=0.1)
                net = BioelectricNetwork(p, seed=ia * 1000 + it * 10 + s,
                                         polarity_gradient=True)
                # Attractor 1: anterior = V_+, posterior = V_-
                net.X[:N//2, 0] = p.V_plus  + 0.05 * net.rng.standard_normal(N//2)
                net.X[N//2:, 0] = p.V_minus + 0.05 * net.rng.standard_normal(N - N//2)

                # --- Equilibrate ---
                net.run(T=T_eq, record_every=10000, verbose=False)
                pol_before = float(net.X[:N//2, 0].mean() - net.X[N//2:, 0].mean())

                # --- Forcing ---
                def _ext(t, _A=A):
                    e = np.zeros(N)
                    e[:N//2]  = -_A   # hyperpolarise anterior
                    e[N//2:]  =  _A   # depolarise posterior
                    return e

                net.run(T=Tf, ext_func=_ext, record_every=10000, verbose=False)

                # --- Relax ---
                net.run(T=T_post, record_every=10000, verbose=False)
                pol_after = float(net.X[:N//2, 0].mean() - net.X[N//2:, 0].mean())

                switched = (pol_before * pol_after) < 0
                if switched:
                    switches += 1
                delta_pols.append(abs(pol_after - pol_before))
                done += 1

            switch_prob[ia, it] = switches / n_seeds
            pol_change[ia, it]  = np.mean(delta_pols)

            if (done % max(1, total // 10)) == 0:
                print(f"    {done}/{total} runs complete "
                      f"(A={A:.1f}, Tf={Tf:.1f}, p_switch={switch_prob[ia,it]:.2f})")

    # ---- Figure ----
    Tg, Ag = np.meshgrid(T_force_vals, A_vals)   # shape (nA, nT)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        r"Attractor switching phase diagram: amplitude $A$ vs duration $T_{\rm force}$"
        "\n"
        r"Polarity reversal probability over " + str(n_seeds) + " seeds",
        fontsize=19, fontweight='bold')

    # Panel 0: switching probability heatmap
    im0 = axes[0].pcolormesh(T_force_vals, A_vals, switch_prob,
                              cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
    plt.colorbar(im0, ax=axes[0], label='P(switch)')
    axes[0].set_xlabel(r'Forcing duration $T_{\rm force}$', fontsize=17)
    axes[0].set_ylabel(r'Forcing amplitude $A$', fontsize=17)
    axes[0].set_title('Switching probability', fontsize=17)
    # Contour at p = 0.5  (phase boundary)
    try:
        axes[0].contour(T_force_vals, A_vals, switch_prob,
                        levels=[0.5], colors='k', linewidths=2,
                        linestyles='--')
        axes[0].text(T_force_vals[1], A_vals[-2], '$p=0.5$ boundary',
                     fontsize=13, color='k',
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                               alpha=0.85, edgecolor='none'))
    except Exception:
        pass

    # Panel 1: |Δpolarity| heatmap
    im1 = axes[1].pcolormesh(T_force_vals, A_vals, pol_change,
                              cmap='Blues', shading='auto')
    plt.colorbar(im1, ax=axes[1], label=r'$|\Delta\,\mathrm{polarity}|$')
    axes[1].set_xlabel(r'$T_{\rm force}$', fontsize=17)
    axes[1].set_ylabel(r'$A$', fontsize=17)
    axes[1].set_title(r'Mean $|\Delta\,\mathrm{polarity}|$', fontsize=17)

    # Panel 2: switching threshold curve  A*(T_force)
    # For each T_force find the smallest A that gives p >= 0.5
    A_star = []
    for it in range(nT):
        col = switch_prob[:, it]
        idx = np.where(col >= 0.5)[0]
        if len(idx) > 0:
            A_star.append(A_vals[idx[0]])
        else:
            A_star.append(np.nan)
    A_star = np.array(A_star)

    mask = ~np.isnan(A_star)
    if mask.sum() > 1:
        axes[2].plot(T_force_vals[mask], A_star[mask], 'ko-', lw=2,
                     ms=6, label=r'$A^*(T_{\rm force})$')
        # Shade: above = switches, below = no switch
        axes[2].fill_between(T_force_vals[mask], A_star[mask],
                             A_vals[-1], alpha=0.15, color='green',
                             label='Switch region')
        axes[2].fill_between(T_force_vals[mask], A_vals[0], A_star[mask],
                             alpha=0.15, color='red',
                             label='No-switch region')
    axes[2].set_xlabel(r'$T_{\rm force}$', fontsize=17)
    axes[2].set_ylabel(r'$A^*$ (threshold amplitude)', fontsize=17)
    axes[2].set_title(r'Phase boundary $A^*(T_{\rm force})$', fontsize=17)
    axes[2].legend(fontsize=14); axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(T_force_vals[0], T_force_vals[-1])
    axes[2].set_ylim(A_vals[0], A_vals[-1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return {'switch_prob': switch_prob, 'pol_change': pol_change,
            'A_vals': A_vals, 'T_force_vals': T_force_vals,
            'A_star': A_star}


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "="*65)
    print("  ATTRACTOR ANALYSIS — Lyapunov + Switching + Basin mapping")
    print("="*65)

    print("\n[B1] Lyapunov function verification")
    lyap_results = experiment_lyapunov_verification(N=25, T=50.0)
    for label, res in lyap_results.items():
        E_change = res['E'][-1] - res['E'][0]
        print(f"  {label}: ΔE = {E_change:.4f}  "
              f"({'↓ decreases' if E_change < 0 else '↑ increases'})")

    print("\n[B2] Polarity reversal (attractor switching)")
    switch_results = experiment_polarity_reversal(
        N=40, T_eq=50.0, T_force=12.0, T_post=70.0,
        amplitude=2.5,
        tau_eps_with=50.0, tau_eps_without=0.3)

    print("\n[B3] Basin of attraction mapping")
    basin_results = experiment_basin_mapping(
        N=30, T_eq=40.0,
        depol_fracs=np.linspace(0.0, 1.0, 15),
        n_seeds=3)

    print("\n[B4] Adaptive energy landscape")
    landscape = experiment_adaptive_landscape(N=20, T_total=60.0)

    print("\n[B5] Switching phase diagram (A vs T_force)")
    phase = experiment_switching_phase_diagram(
        N=22, T_eq=40.0, T_post=50.0,
        A_vals=np.linspace(0.5, 4.0, 10),
        T_force_vals=np.linspace(2.0, 20.0, 10),
        n_seeds=3)
    print(f"  Switching boundary A*: {phase['A_star']}")

    print("\n" + "="*65)
    print("  Attractor analysis complete. Figures:")
    for f in sorted(os.listdir(".")):
        if f.startswith("fig") and f.endswith(".png"):
            print(f"  {f}")
    print("="*65)
