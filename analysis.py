"""
analysis.py
Yuriria Cortes Poza
===========
Analytical tools and parametric sweeps for the hybrid bioelectric model.

This module contains:

1. Single-cell bistability analysis
   - Phase portrait of f(V) = a(V-V_-)(V-V_0)(V-V_+)
   - Potential landscape U(V) = integral of f
   - Proposition 1: bistability conditions

2. Network linear stability analysis
   - Prove homogeneous state is LOCALLY STABLE for any G >= 0
   - Identify the correct mechanism: bistable domain formation
     (NOT classical Turing instability)
   - Critical coupling for front propagation

3. Parametric sweeps
   - Phase diagram (a_bist, G_init): pattern formation score
   - Regenerative threshold map (G_init, lesion_fraction)

4. Front propagation analysis
   - Front velocity vs G
   - Maxwell point (stationary front condition)
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
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
# scipy not available: implement minimal required functions with numpy
def quad(f, a, b, **kw):
    """Simple Simpson integration over [a,b]."""
    x = np.linspace(a, b, 2001)
    y = np.vectorize(f)(x)
    return np.trapz(y, x), 0.0

def brentq(f, a, b, xtol=1e-8, **kw):
    """Bisection method fallback."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    for _ in range(100):
        mid = 0.5 * (a + b)
        fm  = f(mid)
        if abs(b - a) < xtol or fm == 0:
            return mid
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)
from dataclasses import replace as dc_replace

from bioelectric_model import (
    ModelParameters, BioelectricNetwork,
    plot_spacetime_1d, sigma
)

os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SINGLE-CELL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def f_bistable(V, a, Vm, V0, Vp):
    """Bistable ionic current f(V) = a*(V-V_-)(V-V_0)(V-V_+)."""
    return a * (V - Vm) * (V - V0) * (V - Vp)


def potential(V, a, Vm, V0, Vp):
    """
    Double-well potential U(V) such that f(V) = -dU/dV.
    U(V) = -∫f(V)dV  (quartic polynomial).
    """
    # Antiderivative of -f(V):
    # -f = -a(V-Vm)(V-V0)(V-Vp)
    # = -a [V^3 - (Vm+V0+Vp)V^2 + (VmV0+VmVp+V0Vp)V - VmV0Vp]
    S1 = Vm + V0 + Vp
    S2 = Vm*V0 + Vm*Vp + V0*Vp
    S3 = Vm*V0*Vp
    return -a * (V**4/4 - S1*V**3/3 + S2*V**2/2 - S3*V)


def maxwell_point(a, Vm, Vp):
    """
    Find V0* such that the potential wells are equal in depth:
      U(Vm) == U(Vp)   =>   ∫_{Vm}^{Vp} f(V) dV = 0
    This is the Maxwell condition (stationary front).
    The integral ∫f dV depends on V0; solve for V0 ∈ (Vm, Vp).
    Returns V0_maxwell.
    """
    def area(v0):
        val, _ = quad(lambda v: f_bistable(v, a, Vm, v0, Vp), Vm, Vp)
        return val

    # area is zero at V0_maxwell; it's positive when V0 > midpoint
    # and negative when V0 < midpoint (for the chosen sign of a).
    mid = 0.5*(Vm + Vp)
    # Check signs
    try:
        return brentq(area, Vm + 1e-4, Vp - 1e-4, xtol=1e-8)
    except ValueError:
        return mid   # symmetric by default


def f_prime_at(V_eq, a, Vm, V0, Vp):
    """
    Derivative f'(V) = a[(V-V0)(V-Vp) + (V-Vm)(V-Vp) + (V-Vm)(V-V0)]
    evaluated at an equilibrium point.
    Stability of C dV/dt = -f(V):  stable if f'(V_eq) > 0.
    """
    return a * ((V_eq - V0)*(V_eq - Vp)
                + (V_eq - Vm)*(V_eq - Vp)
                + (V_eq - Vm)*(V_eq - V0))


def plot_single_cell_analysis(params_list, save_path="figures/A1_single_cell.png"):
    """
    Figure A1: Phase portrait, potential landscape, and stability diagram
    for various bistability parameter combinations.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Single-cell bistability analysis", fontsize=22,
                 fontweight='bold')
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(params_list)))

    ax_f   = axes[0, 0]   # f(V) curves
    ax_U   = axes[0, 1]   # potential U(V)
    ax_eig = axes[0, 2]   # f'(V_eq) at fixed points
    ax_fp  = axes[1, 0]   # fixed-point locations vs V0
    ax_bar = axes[1, 1]   # barrier height vs V0
    ax_mx  = axes[1, 2]   # Maxwell point vs a_bist

    V_plot = np.linspace(-2.5, 2.5, 600)

    for c, (label, a, Vm, V0, Vp) in zip(colors, params_list):
        f_vals = f_bistable(V_plot, a, Vm, V0, Vp)
        U_vals = potential(V_plot, a, Vm, V0, Vp)
        U_vals -= U_vals.min()      # shift minimum to 0

        ax_f.plot(V_plot, f_vals, color=c, lw=1.5, label=label)
        ax_U.plot(V_plot, U_vals, color=c, lw=1.5, label=label)

        # fixed-point stability markers
        for V_eq, marker in [(Vm, 'o'), (V0, 's'), (Vp, '^')]:
            fp = f_prime_at(V_eq, a, Vm, V0, Vp)
            stable = fp > 0
            ax_eig.scatter(V_eq, fp, color=c,
                           marker=marker, s=80, zorder=5,
                           edgecolors='k' if stable else 'none',
                           linewidths=1.0)

    # Annotations
    ax_f.axhline(0, color='k', lw=0.8, ls='--')
    ax_f.set_xlabel(r'$V$', fontsize=19); ax_f.set_ylabel(r'$f(V)$', fontsize=19)
    ax_f.set_title(r'Bistable current $f(V)$', fontsize=17)
    ax_f.legend(fontsize=12, ncol=2)

    ax_U.set_xlabel(r'$V$', fontsize=19); ax_U.set_ylabel(r'$U(V)$', fontsize=19)
    ax_U.set_title(r'Double-well potential $U(V) = -\int f\,dV$', fontsize=17)
    ax_U.legend(fontsize=12, ncol=2)

    ax_eig.axhline(0, color='k', lw=0.8, ls='--')
    ax_eig.set_xlabel(r'Equilibrium $V^*$', fontsize=19)
    ax_eig.set_ylabel(r"$f'(V^*)$", fontsize=19)
    ax_eig.set_title(r"$f'(V^*)$: stability indicator", fontsize=17)
    legend_elements = [Line2D([0],[0], marker='o', color='gray',
                               label='$V_\pm$ (stable)', markerfacecolor='gray',
                               markeredgecolor='k'),
                       Line2D([0],[0], marker='s', color='gray',
                               label='$V_0$ (unstable)', markerfacecolor='none',
                               markeredgecolor='gray')]
    ax_eig.legend(handles=legend_elements, fontsize=14)

    # Fixed-point location vs V0 sweep
    a0   = 1.0; Vm0 = -1.5; Vp0 = 1.5
    V0s  = np.linspace(-1.4, 1.4, 60)
    for V0_val in V0s:
        for V_eq, col in [(Vm0, 'steelblue'), (V0_val, 'crimson'),
                          (Vp0, 'forestgreen')]:
            fp = f_prime_at(V_eq, a0, Vm0, V0_val, Vp0)
            ax_fp.scatter(V0_val, V_eq, color=col, s=10, alpha=0.7)
    ax_fp.set_xlabel(r'$V_0$ (unstable threshold)', fontsize=17)
    ax_fp.set_ylabel(r'Equilibrium positions', fontsize=17)
    ax_fp.set_title(r'Fixed points vs $V_0$', fontsize=17)
    ax_fp.legend([Line2D([0],[0],color='steelblue',  marker='o', lw=0, ms=5),
                  Line2D([0],[0],color='crimson',    marker='o', lw=0, ms=5),
                  Line2D([0],[0],color='forestgreen',marker='o', lw=0, ms=5)],
                 [r'$V_-$', r'$V_0$', r'$V_+$'], fontsize=14)

    # Barrier height vs V0
    barriers = []
    for V0_val in V0s:
        U_at_max = potential(V0_val, a0, Vm0, V0_val, Vp0)
        U_at_min_m = potential(Vm0, a0, Vm0, V0_val, Vp0)
        U_at_min_p = potential(Vp0, a0, Vm0, V0_val, Vp0)
        barrier_L = U_at_max - U_at_min_m
        barrier_R = U_at_max - U_at_min_p
        barriers.append((barrier_L, barrier_R))
    barriers = np.array(barriers)
    ax_bar.plot(V0s, barriers[:,0], 'steelblue', lw=2, label=r'$\Delta U_-$ (from $V_-$)')
    ax_bar.plot(V0s, barriers[:,1], 'forestgreen', lw=2, label=r'$\Delta U_+$ (from $V_+$)')
    ax_bar.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.6, label='symmetric')
    ax_bar.set_xlabel(r'$V_0$', fontsize=17)
    ax_bar.set_ylabel('Barrier height', fontsize=17)
    ax_bar.set_title('Potential barriers vs $V_0$', fontsize=17)
    ax_bar.legend(fontsize=14)

    # Maxwell point vs a
    a_vals  = np.linspace(0.3, 3.0, 50)
    mx_pts  = [maxwell_point(aa, Vm0, Vp0) for aa in a_vals]
    ax_mx.plot(a_vals, mx_pts, 'k-', lw=2)
    ax_mx.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
    ax_mx.set_xlabel(r'Bistability strength $a$', fontsize=17)
    ax_mx.set_ylabel(r'Maxwell point $V_0^*$', fontsize=17)
    ax_mx.set_title(r'Maxwell (stationary front) $V_0^*$ vs $a$', fontsize=17)
    ax_mx.set_ylim(-1.0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  NETWORK LINEAR STABILITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def network_stability_analysis(p: ModelParameters,
                                save_path="figures/A2_network_stability.png"):
    """
    Theorem (network stability of homogeneous state):

    Consider the reduced 2-variable system (V, P) with uniform coupling G:

        C dV_i/dt = -f(V_i) + G*(L*V)_i + beta_1*P_i
        dP_i/dt   = -rho_P*P_i + a_PV*sigma(V_i)

    Linearized around homogeneous equilibrium (V*, P*).
    Local Jacobian J_0, network Jacobian J(mu_k) = J_0 + diag(G*mu_k/C, 0).

    Key result:
        det(J(mu_k)) = det(J_0) + rho_P * G * |mu_k| / C >= det(J_0)

    => Spatial modes are MORE stable than the homogeneous mode under gap-junction coupling.
    => Classical Turing instability CANNOT occur in the (V,P) subsystem
       with gap junctions on V only.

    The correct patterning mechanism is bistable domain formation:
    finite-amplitude perturbations, not infinitesimal ones, can create patterns.
    """
    print("\n  Network linear stability analysis...")

    V_eq  = p.V_minus   # homogeneous hyperpolarized state
    a     = p.a_bist; Vm = p.V_minus; V0 = p.V_0; Vp = p.V_plus
    C     = p.C

    # f'(V_-): restoring force coefficient at hyperpolarized equilibrium
    lam0 = f_prime_at(V_eq, a, Vm, V0, Vp)   # > 0: stable
    rho_P = p.rho_P
    beta1 = p.beta_1
    sig_prime = 0.25   # sigma'(V_-) ≈ sigma'(V_-, k=5) ~ k*sigma*(1-sigma)
    #  More precisely: sigma = 1/(1+exp(-k(V-x0))), sigma' = k*sigma*(1-sigma)
    #  At V_- = -1.5 with x0=0.5, k=5: sigma(-1.5) ≈ 0.018, sigma' ≈ 5*0.018*0.982 ≈ 0.088
    sig_val = 1.0/(1 + np.exp(-5*(V_eq - 0.5)))
    sig_prime_exact = 5.0 * sig_val * (1 - sig_val)

    # Local Jacobian (V,P) at (V_-, P*)
    # J_0 = [[-lam0/C,  beta1/C ],
    #         [aPV*s',  -rhoP   ]]
    a_PV = p.a_PV
    J00 = -lam0 / C
    J01 =  beta1 / C
    J10 =  a_PV * sig_prime_exact
    J11 = -rho_P

    det_J0 = J00*J11 - J01*J10
    tr_J0  = J00 + J11

    print(f"    V_eq = {V_eq:.3f}")
    print(f"    f'(V_-) = lambda_0 = {lam0:.4f}")
    print(f"    sigma'(V_-) = {sig_prime_exact:.5f}")
    print(f"    J_0 = [[{J00:.4f}, {J01:.4f}], [{J10:.4f}, {J11:.4f}]]")
    print(f"    tr(J_0) = {tr_J0:.4f}  (< 0 for stability: {'OK' if tr_J0 < 0 else 'FAIL'})")
    print(f"    det(J_0) = {det_J0:.4f}  (> 0 for stability: {'OK' if det_J0 > 0 else 'FAIL'})")
    if det_J0 > 0 and tr_J0 < 0:
        print("    => Homogeneous state is LOCALLY STABLE")
    else:
        print("    => Homogeneous state may be UNSTABLE")

    # Key formula:  det(J(mu_k)) = det(J_0) + rho_P * G * |mu_k| / C
    # Show this for a range of G and |mu|
    G_vals  = np.linspace(0, 5, 100)
    mu_vals = np.array([0.5, 1.0, 2.0, 4.0])   # representative |mu_k|

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Network linear stability: spatial modes are never destabilized\n"
                 "by gap-junction coupling alone (2-variable $(V,P)$ subsystem)",
                 fontsize=19, fontweight='bold')

    ax1, ax2, ax3 = axes

    # Panel 1: det(J(mu_k)) vs G for several |mu_k|
    for mu_abs in mu_vals:
        det_k = det_J0 + rho_P * G_vals * mu_abs / C
        ax1.plot(G_vals, det_k, lw=2, label=fr'$|\mu_k|={mu_abs}$')
    ax1.axhline(0, color='k', lw=0.8, ls='--')
    ax1.set_xlabel(r'Coupling $G$', fontsize=19)
    ax1.set_ylabel(r'$\det(J(\mu_k))$', fontsize=19)
    ax1.set_title(r'$\det(J(\mu_k)) = \det(J_0) + \rho_P G |\mu_k|/C$', fontsize=17)
    ax1.legend(fontsize=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=min(det_J0 - 0.1, -0.2))

    # Panel 2: tr(J(mu_k)) vs G
    for mu_abs in mu_vals:
        tr_k = tr_J0 - G_vals * mu_abs / C    # coupling reduces tr too
        ax2.plot(G_vals, tr_k, lw=2, label=fr'$|\mu_k|={mu_abs}$')
    ax2.axhline(0, color='k', lw=0.8, ls='--')
    ax2.set_xlabel(r'Coupling $G$', fontsize=19)
    ax2.set_ylabel(r'$\mathrm{tr}(J(\mu_k))$', fontsize=19)
    ax2.set_title(r'$\mathrm{tr}(J(\mu_k))$ vs coupling', fontsize=17)
    ax2.legend(fontsize=15)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Eigenvalues of J_0 as function of bistability strength a
    a_range = np.linspace(0.2, 3.0, 200)
    ev_real_max = []
    for aa in a_range:
        lam = f_prime_at(V_eq, aa, Vm, V0, Vp)
        J00_ = -lam / C; J01_ = beta1/C; J10_ = a_PV*sig_prime_exact; J11_ = rho_P * (-1)
        tr_  = J00_ + J11_
        det_ = J00_*J11_ - J01_*J10_
        disc = tr_**2 - 4*det_
        if disc >= 0:
            ev_max = 0.5*(tr_ + np.sqrt(disc))
        else:
            ev_max = tr_ / 2   # real part of complex eigenvalue
        ev_real_max.append(ev_max)
    ax3.plot(a_range, ev_real_max, 'steelblue', lw=2)
    ax3.axhline(0, color='k', lw=0.8, ls='--')
    ax3.fill_between(a_range,
                     np.minimum(ev_real_max, 0),
                     0, alpha=0.2, color='green', label='stable')
    ax3.fill_between(a_range,
                     0,
                     np.maximum(ev_real_max, 0),
                     alpha=0.2, color='red', label='unstable')
    ax3.set_xlabel(r'Bistability strength $a$', fontsize=19)
    ax3.set_ylabel(r'$\max \mathrm{Re}(\lambda)$ of $J_0$', fontsize=19)
    ax3.set_title(r'Local stability vs $a$ (G=0 mode)', fontsize=17)
    ax3.legend(fontsize=15)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    return {
        'lam0': lam0, 'det_J0': det_J0, 'tr_J0': tr_J0,
        'sig_prime': sig_prime_exact,
        'theorem': 'Spatial modes are NEVER destabilized by gap-junction coupling on V alone'
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FRONT PROPAGATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def front_propagation_analysis(N: int = 80, T: float = 60.0,
                                G_values=None, V0_values=None,
                                save_path="figures/A3_front_propagation.png"):
    """
    Bistable front propagation in 1D.

    Initialize tissue as half V_+ (i < N/2) and half V_- (i >= N/2).
    Track front position x_f(t) = mean position where V_i crosses 0.
    Front velocity c = dx_f/dt.

    Maxwell condition: c=0 when the potential wells have equal depth,
    i.e., when V0 = V0_maxwell.

    Shows: the gap-junction coupling G controls front speed but not
    whether patterns can exist (they exist for any G > 0 given suitable IC).
    The threshold is on V0 (symmetry of the double well), not on G.
    """
    if G_values  is None: G_values  = [0.1, 0.3, 0.5, 1.0, 2.0]
    if V0_values is None: V0_values = [-0.8, -0.4, 0.0, 0.4, 0.8]

    p0 = ModelParameters(N=N, dim=1, dt=0.02, tau_eps=1000.0,
                         a_PV=0.0, a_DV=0.0, a_IV=0.0, a_RV=0.0,
                         beta_1=0.0, beta_2=0.0, beta_3=0.0, beta_4=0.0)
    # Pure bioelectric subsystem: no GRN coupling

    print("\n  Front propagation analysis...")

    # --- Sweep over G ---
    velocities_G = []
    for G in G_values:
        p = dc_replace(p0, G_init=G)
        net = BioelectricNetwork(p, seed=10)
        net.X[:N//2, 0] = p.V_plus  + 0.02*net.rng.standard_normal(N//2)
        net.X[N//2:, 0] = p.V_minus + 0.02*net.rng.standard_normal(N - N//2)
        net.X[:, 1:] = 0.0

        front_positions = []
        def record_front():
            V = net.X[:, 0]
            crossings = np.where(np.diff(np.sign(V)))[0]
            return crossings[0] if len(crossings) > 0 else N//2

        n_steps = int(T / p.dt)
        for step in range(n_steps):
            net.X, net.G = net._rk4_step(net.X, net.G, p.dt,
                                          np.zeros(N), np.zeros(N))
            if step % 50 == 0:
                front_positions.append(record_front())

        front_positions = np.array(front_positions)
        if len(front_positions) > 10:
            # Velocity from linear fit on second half
            mid = len(front_positions)//2
            ts  = np.arange(len(front_positions)) * 50 * p.dt
            vel = np.polyfit(ts[mid:], front_positions[mid:], 1)[0]
        else:
            vel = 0.0
        velocities_G.append(vel)
        print(f"    G = {G:.2f}  => front velocity c = {vel:.4f}")

    # --- Sweep over V0 (asymmetry) ---
    velocities_V0 = []
    for V0 in V0_values:
        p = dc_replace(p0, G_init=0.5, V_0=V0)
        net = BioelectricNetwork(p, seed=11)
        net.X[:N//2, 0] = p.V_plus  + 0.02*net.rng.standard_normal(N//2)
        net.X[N//2:, 0] = p.V_minus + 0.02*net.rng.standard_normal(N - N//2)
        net.X[:, 1:] = 0.0

        front_positions = []
        n_steps = int(T / p.dt)
        for step in range(n_steps):
            net.X, net.G = net._rk4_step(net.X, net.G, p.dt,
                                          np.zeros(N), np.zeros(N))
            if step % 50 == 0:
                V = net.X[:, 0]
                crossings = np.where(np.diff(np.sign(V)))[0]
                fp = crossings[0] if len(crossings) > 0 else N//2
                front_positions.append(fp)

        front_positions = np.array(front_positions)
        if len(front_positions) > 10:
            mid = len(front_positions)//2
            ts  = np.arange(len(front_positions)) * 50 * p.dt
            vel = np.polyfit(ts[mid:], front_positions[mid:], 1)[0]
        else:
            vel = 0.0
        velocities_V0.append(vel)
        print(f"    V0 = {V0:.2f}  => front velocity c = {vel:.4f}")

    # Maxwell point (theoretical)
    V0_mx = maxwell_point(p0.a_bist, p0.V_minus, p0.V_plus)
    print(f"    Theoretical Maxwell point: V0* = {V0_mx:.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Bistable front propagation analysis\n"
                 "(pure bioelectric subsystem, no GRN)",
                 fontsize=20, fontweight='bold')

    ax1, ax2, ax3 = axes

    ax1.plot(G_values, velocities_G, 'o-', color='steelblue', lw=2, ms=7)
    ax1.axhline(0, color='k', lw=0.8, ls='--')
    ax1.set_xlabel(r'Coupling $G$', fontsize=19)
    ax1.set_ylabel(r'Front velocity $c$', fontsize=19)
    ax1.set_title(r'$c$ vs $G$ ($V_0 = 0$, symmetric well)', fontsize=17)
    ax1.grid(True, alpha=0.3)

    ax2.plot(V0_values, velocities_V0, 'o-', color='firebrick', lw=2, ms=7)
    ax2.axhline(0, color='k', lw=0.8, ls='--')
    ax2.axvline(V0_mx, color='green', lw=1.5, ls=':', label=f'$V_0^* = {V0_mx:.2f}$')
    ax2.set_xlabel(r'$V_0$ (unstable threshold)', fontsize=19)
    ax2.set_ylabel(r'Front velocity $c$', fontsize=19)
    ax2.set_title(r'$c$ vs $V_0$ ($G=0.5$): Maxwell point', fontsize=17)
    ax2.legend(fontsize=15)
    ax2.grid(True, alpha=0.3)

    # Space-time of a representative run (V0=0.4, G=0.5)
    p_demo = dc_replace(p0, G_init=0.5, V_0=0.4)
    net_d  = BioelectricNetwork(p_demo, seed=12)
    net_d.X[:N//2, 0] = p_demo.V_plus  + 0.02*net_d.rng.standard_normal(N//2)
    net_d.X[N//2:, 0] = p_demo.V_minus + 0.02*net_d.rng.standard_normal(N - N//2)
    net_d.X[:, 1:] = 0.0
    hist_d = net_d.run(T=T, record_every=25, verbose=False)
    im = ax3.imshow(hist_d['V'].T, aspect='auto', origin='lower',
                    extent=[0, T, 0, N], cmap='RdBu_r',
                    vmin=p_demo.V_minus-0.2, vmax=p_demo.V_plus+0.2)
    ax3.set_xlabel('Time', fontsize=19); ax3.set_ylabel('Cell', fontsize=19)
    ax3.set_title(r'Space-time $V_i(t)$: expanding $V_+$ domain ($V_0=0.4$)',
                  fontsize=17)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    return {
        'G_values': G_values, 'velocities_G': velocities_G,
        'V0_values': V0_values, 'velocities_V0': velocities_V0,
        'V0_maxwell': V0_mx
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PARAMETRIC SWEEP: (a_bist, G_init) PHASE DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────

def pattern_formation_score(history, threshold=0.15):
    """
    Spatial heterogeneity of V at final time:
    score = std(V_final) / (V_+ - V_-) normalized.
    High score => spatially heterogeneous (patterned) state.
    """
    V_final = history['V'][-1]
    return float(np.std(V_final))


def parametric_sweep_phase_diagram(
        N: int = 25,
        T_eq: float = 60.0,
        a_vals=None, G_vals=None,
        n_seeds: int = 3,
        save_path="figures/A4_phase_diagram.png"):
    """
    2D parametric sweep over (a_bist, G_init).
    For each parameter combination, initialize with a small localized
    depolarized perturbation and measure whether a stable heterogeneous
    pattern emerges.

    Pattern formation score: std(V_final) normalized by (V_+ - V_-).
    """
    if a_vals is None:
        a_vals = np.linspace(0.3, 2.5, 10)
    if G_vals is None:
        G_vals = np.linspace(0.05, 3.0, 10)

    print(f"\n  Phase diagram sweep: {len(a_vals)} x {len(G_vals)} x {n_seeds} runs...")

    score_map = np.zeros((len(a_vals), len(G_vals)))

    p_base = ModelParameters(N=N, dim=1, dt=0.02)

    for i, aa in enumerate(a_vals):
        for j, G in enumerate(G_vals):
            scores = []
            for seed in range(n_seeds):
                p = dc_replace(p_base, a_bist=aa, G_init=G)
                net = BioelectricNetwork(p, seed=seed*100+i*10+j)
                # Initialize with small localized depolarized seed
                # (models realistic tissue with small heterogeneity)
                seed_size = max(1, N//8)
                net.X[N//2:N//2+seed_size, 0] = p.V_plus * 0.7
                hist = net.run(T=T_eq, record_every=200, verbose=False)
                scores.append(pattern_formation_score(hist))
            score_map[i, j] = np.mean(scores)
            print(f"    a={aa:.2f}, G={G:.2f}  => score={score_map[i,j]:.3f}")

    # Normalize
    score_map_norm = score_map / (score_map.max() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"Phase diagram: spatial patterning score"
                 "\n(1D, pure bistable + GRN, initialized with small perturbation)",
                 fontsize=19, fontweight='bold')

    im1 = axes[0].imshow(score_map_norm, origin='lower', aspect='auto',
                          extent=[G_vals[0], G_vals[-1],
                                  a_vals[0], a_vals[-1]],
                          cmap='plasma', vmin=0, vmax=1)
    axes[0].set_xlabel(r'$G_{\mathrm{init}}$', fontsize=20)
    axes[0].set_ylabel(r'Bistability strength $a$', fontsize=20)
    axes[0].set_title(r'Normalized $\mathrm{std}(V_{\mathrm{final}})$', fontsize=17)
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Contour at 0.3 (patterning boundary)
    G_grid, a_grid = np.meshgrid(G_vals, a_vals)
    cs = axes[0].contour(G_grid, a_grid, score_map_norm,
                         levels=[0.30], colors='white', linewidths=2)
    axes[0].clabel(cs, fmt='0.30', fontsize=15)

    # Cross-sections
    mid_a_idx = len(a_vals) // 2
    mid_G_idx = len(G_vals) // 2
    axes[1].plot(G_vals, score_map_norm[mid_a_idx, :], 'steelblue',
                 lw=2, label=fr'$a = {a_vals[mid_a_idx]:.2f}$ (mid)')
    axes[1].plot(G_vals, score_map_norm[-1, :], 'steelblue',
                 lw=2, ls='--', label=fr'$a = {a_vals[-1]:.2f}$ (max)')
    axes[1].plot(G_vals, score_map_norm[0, :], 'steelblue',
                 lw=2, ls=':', label=fr'$a = {a_vals[0]:.2f}$ (min)')
    axes[1].axhline(0.30, color='gray', lw=1, ls='--', alpha=0.7)
    axes[1].set_xlabel(r'$G_{\mathrm{init}}$', fontsize=20)
    axes[1].set_ylabel('Patterning score', fontsize=19)
    axes[1].set_title('Score vs $G$ for fixed $a$ values', fontsize=17)
    axes[1].legend(fontsize=14); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    return {'a_vals': a_vals, 'G_vals': G_vals,
            'score_map': score_map, 'score_map_norm': score_map_norm}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  REGENERATIVE THRESHOLD MAP: (G_init, lesion_fraction)
# ─────────────────────────────────────────────────────────────────────────────

def regenerative_threshold_map(
        N: int = 30, T_eq: float = 40.0, T_regen: float = 60.0,
        G_vals=None, lesion_fracs=None,
        n_seeds: int = 3,
        save_path="figures/A5_regen_threshold.png"):
    """
    2D sweep over (G_init, lesion_fraction).
    For each combination, equilibrate tissue, apply central lesion,
    measure regeneration score at final time.

    Maps the critical threshold curve G_c(lesion_fraction).
    """
    if G_vals       is None: G_vals       = np.linspace(0.05, 3.0, 10)
    if lesion_fracs is None: lesion_fracs = np.linspace(0.05, 0.60, 10)

    print(f"\n  Regenerative threshold sweep: "
          f"{len(G_vals)} x {len(lesion_fracs)} x {n_seeds} runs...")

    regen_map = np.zeros((len(G_vals), len(lesion_fracs)))
    p_base    = ModelParameters(N=N, dim=1, dt=0.02)

    for i, G in enumerate(G_vals):
        for j, lf in enumerate(lesion_fracs):
            scores = []
            for seed in range(n_seeds):
                p = dc_replace(p_base, G_init=G)
                net = BioelectricNetwork(p, seed=seed*77+i*7+j,
                                         polarity_gradient=True)
                # Seed depolarized domain for pattern
                net.X[:N//2, 0] = p.V_plus * 0.8
                net.run(T=T_eq, record_every=500, verbose=False)

                # Lesion
                c = N//2; w = max(1, int(lf*N))
                net.apply_lesion(np.arange(max(0,c-w//2), min(N,c+w//2)))

                hist = net.run(T=T_regen, record_every=500,
                               apply_birth=True, verbose=False)
                scores.append(hist['regen_score'][-1])
            regen_map[i, j] = np.mean(scores)
            print(f"    G={G:.2f}, lf={lf:.2f}  => regen={regen_map[i,j]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Regenerative threshold map\n"
                 r"$\rho(\infty)$ as function of $(G_{\mathrm{init}},\; \ell/N)$",
                 fontsize=19, fontweight='bold')

    G_grid, lf_grid = np.meshgrid(lesion_fracs, G_vals)
    im = axes[0].imshow(regen_map, origin='lower', aspect='auto',
                         extent=[lesion_fracs[0], lesion_fracs[-1],
                                 G_vals[0], G_vals[-1]],
                         cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_xlabel(r'Lesion fraction $\ell/N$', fontsize=20)
    axes[0].set_ylabel(r'$G_{\mathrm{init}}$', fontsize=20)
    axes[0].set_title(r'Final regeneration score $\rho(\infty)$', fontsize=17)
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # Threshold contour at rho = 0.75
    cs = axes[0].contour(lf_grid, G_grid, regen_map,
                         levels=[0.75], colors='k', linewidths=2)
    axes[0].clabel(cs, fmt='0.75', fontsize=15)

    # Cross-sections: regen score vs lesion fraction for several G
    for idx in [0, len(G_vals)//3, len(G_vals)//2, -1]:
        axes[1].plot(lesion_fracs, regen_map[idx, :], 'o-',
                     lw=1.8, ms=5,
                     label=fr'$G={G_vals[idx]:.2f}$')
    axes[1].axhline(0.75, color='gray', lw=1, ls='--',
                    alpha=0.8, label='threshold $\\rho=0.75$')
    axes[1].set_xlabel(r'Lesion fraction $\ell/N$', fontsize=20)
    axes[1].set_ylabel(r'$\rho(\infty)$', fontsize=19)
    axes[1].set_title('Regen. score vs lesion for several $G$', fontsize=17)
    axes[1].legend(fontsize=14); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    return {'G_vals': G_vals, 'lesion_fracs': lesion_fracs,
            'regen_map': regen_map}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*65)
    print("  BIOELECTRIC MODEL — ANALYTICAL & PARAMETRIC ANALYSIS")
    print("="*65)

    # Default parameters for analysis
    p = ModelParameters()

    # ── 1. Single-cell analysis ───────────────────────────────────────
    print("\n[1] Single-cell bistability analysis")
    params_list = [
        (r'$a=1.0, V_0=0$',      1.0, -1.5,  0.0,  1.5),
        (r'$a=1.0, V_0=+0.5$',   1.0, -1.5,  0.5,  1.5),
        (r'$a=1.0, V_0=-0.5$',   1.0, -1.5, -0.5,  1.5),
        (r'$a=0.5$',              0.5, -1.5,  0.0,  1.5),
        (r'$a=2.0$',              2.0, -1.5,  0.0,  1.5),
        (r'$V_\pm=\pm 1.0$',     1.0, -1.0,  0.0,  1.0),
    ]
    plot_single_cell_analysis(params_list)

    # ── 2. Network stability ──────────────────────────────────────────
    print("\n[2] Network linear stability analysis")
    result = network_stability_analysis(p)
    print("\n  KEY RESULT:")
    print(" ", result['theorem'])
    print(f"   det(J_0) = {result['det_J0']:.4f}, "
          f"tr(J_0) = {result['tr_J0']:.4f}")
    print("   => Pattern formation requires FINITE-AMPLITUDE perturbations")
    print("      (bistable domain formation, not infinitesimal Turing instability)")

    # ── 3. Front propagation ──────────────────────────────────────────
    print("\n[3] Front propagation analysis")
    front_results = front_propagation_analysis(
        N=60, T=50.0,
        G_values=[0.1, 0.3, 0.5, 1.0, 2.0],
        V0_values=[-0.8, -0.4, 0.0, 0.4, 0.8])

    # ── 4. Phase diagram (a_bist, G_init) ────────────────────────────
    print("\n[4] Phase diagram: pattern formation score")
    phase = parametric_sweep_phase_diagram(
        N=25, T_eq=50.0,
        a_vals=np.linspace(0.3, 2.5, 8),
        G_vals=np.linspace(0.05, 3.0, 8),
        n_seeds=3)

    # ── 5. Regenerative threshold map ────────────────────────────────
    print("\n[5] Regenerative threshold map")
    regen = regenerative_threshold_map(
        N=30, T_eq=35.0, T_regen=50.0,
        G_vals=np.linspace(0.05, 3.0, 8),
        lesion_fracs=np.linspace(0.05, 0.55, 8),
        n_seeds=3)

    print("\n" + "="*65)
    print("  Analysis complete. All figures saved to ./figures/")
    print("="*65)
    print("\nFigures:")
    for f in sorted(os.listdir("figures")):
        print(f"  {f}")
