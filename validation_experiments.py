"""
validation_experiments.py
=========================
Four validation experiments that convert open problems into
directly evidenced results:

  C1 — Nucleation threshold: probability of pattern formation vs
       initial noise amplitude delta.  Demonstrates the finite-amplitude
       threshold predicted by the Lyapunov / double-well analysis.

  C2 — Regeneration vs lesion size: success probability and mean
       regeneration time as a function of |D|/N.  Provides numerical
       evidence for Problem 4 (regenerative threshold).

  C3 — R_i module knockout: full model vs R_i-disabled model for
       increasing lesion sizes.  Validates the minimality claim in
       section 6.4.

  C4 — Fixed vs adaptive conductance: regeneration outcomes with
       static G_ij compared to adaptive G_ij.  Tests necessity of
       dynamic coupling (Problem 5).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from bioelectric_model import ModelParameters, BioelectricNetwork

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mixed_init(net: BioelectricNetwork, frac_plus: float = 0.5,
                noise: float = 0.08) -> None:
    """Initialise tissue with frac_plus fraction at V_+ (anterior)."""
    p  = net.p
    N  = p.N
    k  = int(frac_plus * N)
    net.X[:k,  0] = p.V_plus  + noise * net.rng.standard_normal(k)
    net.X[k:,  0] = p.V_minus + noise * net.rng.standard_normal(N - k)
    net.X[:, 1:]  = 0.05


def _pattern_formed(V: np.ndarray, V0: float, V_plus: float,
                    threshold_frac: float = 0.12) -> bool:
    """True if ≥ threshold_frac cells are in the depolarised state."""
    thresh = 0.5 * (V0 + V_plus)
    return float(np.mean(V > thresh)) >= threshold_frac


def _regen_success(V_final: np.ndarray, V_ref: np.ndarray,
                   V_plus: float, V0: float,
                   tol: float = 0.6) -> bool:
    """
    True if cells that were depolarised in the reference state are
    mostly recovered: fraction of reference-depolarised cells that
    ended depolarised >= tol.
    """
    thresh = 0.5 * (V0 + V_plus)
    ref_depol = V_ref > thresh
    if ref_depol.sum() == 0:
        return True
    recovered = ((V_final > thresh) & ref_depol).sum() / ref_depol.sum()
    return float(recovered) >= tol


def _regen_time(V_hist: np.ndarray, t_hist: np.ndarray,
                V_ref: np.ndarray, V_plus: float, V0: float,
                tol: float = 0.6) -> float:
    """First time the pattern is considered recovered; nan if never."""
    thresh = 0.5 * (V0 + V_plus)
    ref_depol = V_ref > thresh
    if ref_depol.sum() == 0:
        return 0.0
    for t, V in zip(t_hist, V_hist):
        rec = ((V > thresh) & ref_depol).sum() / ref_depol.sum()
        if rec >= tol:
            return float(t)
    return float('nan')


# ---------------------------------------------------------------------------
# C1 — Nucleation threshold
# ---------------------------------------------------------------------------

def experiment_nucleation_threshold(
        N: int = 25,
        T: float = 70.0,
        delta_vals: Optional[np.ndarray] = None,
        n_seeds: int = 20,
        save_path: str = "figures/C1_nucleation.png") -> dict:
    """
    Sweep initial noise amplitude delta.  All cells start at V_- + delta*xi,
    xi ~ N(0,1).  Record:
      - p_pattern: fraction of runs that form a heterogeneous domain
      - t_domain:  mean time to domain formation (over successful runs)
      - frac_plus: mean fraction of cells at V_+ in successful runs
    """
    print("  [C1] Nucleation threshold ...")

    if delta_vals is None:
        delta_vals = np.concatenate([
            np.linspace(0.02, 0.20, 6),
            np.linspace(0.25, 1.20, 8)
        ])

    p_base = ModelParameters(N=N, dim=1, dt=0.05, a_bist=1.2,
                             G_init=0.3, tau_eps=50.0,
                             beta_1=0.40, beta_2=0.10)

    p_pattern = np.zeros(len(delta_vals))
    frac_plus  = np.zeros(len(delta_vals))
    t_domain   = np.full(len(delta_vals), np.nan)

    n_steps  = int(T / p_base.dt)
    check_interval = max(1, n_steps // 30)   # check for pattern every ~30 times
    thresh   = 0.5 * (p_base.V_0 + p_base.V_plus)

    for i, delta in enumerate(delta_vals):
        successes = 0
        fracs, times = [], []

        for s in range(n_seeds):
            net = BioelectricNetwork(p_base, seed=i * 100 + s)
            # All cells start near V_-
            net.X[:, 0] = p_base.V_minus + delta * net.rng.standard_normal(N)
            net.X[:, 1:] = 0.05

            formed_at = None
            for step in range(n_steps):
                net.X, net.G = net._rk4_step(
                    net.X, net.G, p_base.dt,
                    np.zeros(N), np.zeros(N))
                if formed_at is None and step % check_interval == 0:
                    if _pattern_formed(net.X[:, 0], p_base.V_0, p_base.V_plus):
                        formed_at = step * p_base.dt

            V_f = net.X[:, 0]
            formed = _pattern_formed(V_f, p_base.V_0, p_base.V_plus)
            if formed:
                successes += 1
                fracs.append(float(np.mean(V_f > thresh)))
                if formed_at is not None:
                    times.append(formed_at)

        p_pattern[i] = successes / n_seeds
        if fracs:
            frac_plus[i] = float(np.mean(fracs))
        if times:
            t_domain[i] = float(np.mean(times))

        print(f"    delta={delta:.3f}: p_pattern={p_pattern[i]:.2f}, "
              f"frac_plus={frac_plus[i]:.2f}, "
              f"t_domain={t_domain[i]:.1f}")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "C1 — Nucleation threshold\n"
        r"Pattern formation probability vs initial noise amplitude $\delta$",
        fontsize=11, fontweight='bold')

    # Panel 0: nucleation probability
    axes[0].plot(delta_vals, p_pattern, 'ko-', lw=2, ms=6)
    axes[0].axhline(0.5, color='gray', lw=1, ls='--', alpha=0.7)
    # Shade the barrier region
    axes[0].fill_between(delta_vals, 0, p_pattern, alpha=0.15, color='steelblue')
    axes[0].set_xlabel(r'Noise amplitude $\delta$', fontsize=11)
    axes[0].set_ylabel('P(pattern formation)', fontsize=11)
    axes[0].set_title('Nucleation probability', fontsize=10)
    axes[0].set_ylim(-0.05, 1.05); axes[0].grid(True, alpha=0.3)

    # Estimate threshold
    cross = np.where(p_pattern >= 0.5)[0]
    if len(cross):
        delta_star = delta_vals[cross[0]]
        axes[0].axvline(delta_star, color='firebrick', lw=2, ls='--',
                        label=fr'$\delta^* \approx {delta_star:.2f}$')
        axes[0].legend(fontsize=9)

    # Panel 1: fraction of cells at V_+
    mask = frac_plus > 0
    axes[1].plot(delta_vals[mask], frac_plus[mask], 's-',
                 color='firebrick', lw=2, ms=6)
    axes[1].set_xlabel(r'$\delta$', fontsize=11)
    axes[1].set_ylabel(r'Mean $\langle V_i > V^*\rangle$ (successful runs)', fontsize=10)
    axes[1].set_title(r'Domain size (fraction at $V_+$)', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Panel 2: time to domain formation
    mask2 = ~np.isnan(t_domain)
    if mask2.sum() > 1:
        axes[2].plot(delta_vals[mask2], t_domain[mask2], 'd-',
                     color='darkorange', lw=2, ms=6)
        axes[2].set_xlabel(r'$\delta$', fontsize=11)
        axes[2].set_ylabel('Mean time to pattern (successful runs)', fontsize=10)
        axes[2].set_title('Domain formation time', fontsize=10)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'Insufficient data\n(increase T or delta)',
                     ha='center', va='center', transform=axes[2].transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    return {'delta_vals': delta_vals, 'p_pattern': p_pattern,
            'frac_plus': frac_plus, 't_domain': t_domain}


# ---------------------------------------------------------------------------
# C2 — Regeneration vs lesion size
# ---------------------------------------------------------------------------

def experiment_regen_vs_lesion(
        N: int = 30,
        T_eq: float = 40.0,
        T_regen: float = 80.0,
        lesion_fracs: Optional[np.ndarray] = None,
        n_seeds: int = 10,
        params_override: Optional[dict] = None,
        label: str = "Full model",
        save_path: Optional[str] = "figures/C2_regen_lesion.png") -> dict:
    """
    Sweep central lesion fraction |D|/N.  For each value:
      - Equilibrate tissue (mixed half-half initialisation)
      - Remove central lesion cells
      - Monitor for T_regen
    Record: p_success, mean regen time, mean final pattern fidelity.
    """
    print(f"  [C2/C3/C4] Regen vs lesion ({label}) ...")

    if lesion_fracs is None:
        lesion_fracs = np.linspace(0.05, 0.65, 10)

    base_kw = dict(N=N, dim=1, dt=0.05, a_bist=1.2,
                   G_init=0.4, tau_eps=50.0,
                   beta_1=0.40, beta_2=0.10)
    if params_override:
        base_kw.update(params_override)

    p_success   = np.zeros(len(lesion_fracs))
    mean_time   = np.full(len(lesion_fracs), np.nan)
    mean_fid    = np.zeros(len(lesion_fracs))

    for i, lfrac in enumerate(lesion_fracs):
        n_lesion = max(1, int(lfrac * N))
        successes, times, fids = [], [], []

        for s in range(n_seeds):
            p = ModelParameters(**base_kw)
            net = BioelectricNetwork(p, seed=i * 200 + s)
            _mixed_init(net, frac_plus=0.5)

            # Equilibrate
            net.run(T=T_eq, record_every=99999, verbose=False)
            V_ref = net.X[:, 0].copy()

            # Apply lesion (central region)
            start = (N - n_lesion) // 2
            lesion_idx = list(range(start, start + n_lesion))
            net.apply_lesion(lesion_idx, wound_amplitude=1.5)

            # Regenerate
            hist = net.run(T=T_regen, apply_birth=True,
                           record_every=50, verbose=False)

            V_hist = hist['V']
            t_hist = hist['times']
            V_final = V_hist[-1]

            ok = _regen_success(V_final, V_ref,
                                p.V_plus, p.V_0, tol=0.55)
            successes.append(int(ok))
            if ok:
                rt = _regen_time(V_hist, t_hist, V_ref,
                                 p.V_plus, p.V_0, tol=0.55)
                if not np.isnan(rt):
                    times.append(rt)

            # Pattern fidelity: fraction of originally depolarised cells
            # that are again depolarised
            thresh = 0.5 * (p.V_0 + p.V_plus)
            ref_d = V_ref > thresh
            if ref_d.sum() > 0:
                fid = float(((V_final > thresh) & ref_d).sum() / ref_d.sum())
            else:
                fid = 1.0
            fids.append(fid)

        p_success[i] = np.mean(successes)
        if times:
            mean_time[i] = np.mean(times)
        mean_fid[i] = np.mean(fids)

        print(f"    lesion={lfrac:.2f}: p_ok={p_success[i]:.2f}, "
              f"t_regen={mean_time[i]:.1f}, fidelity={mean_fid[i]:.2f}")

    result = {'lesion_fracs': lesion_fracs,
              'p_success':    p_success,
              'mean_time':    mean_time,
              'mean_fid':     mean_fid,
              'label':        label}

    if save_path is not None:
        _plot_regen_single(result, save_path)

    return result


def _plot_regen_single(res: dict, save_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"C2 — Regeneration vs lesion size\n{res['label']}",
        fontsize=11, fontweight='bold')

    lf = res['lesion_fracs']

    axes[0].plot(lf, res['p_success'], 'ko-', lw=2, ms=6)
    axes[0].axhline(0.5, color='gray', lw=1, ls='--', alpha=0.6)
    axes[0].fill_between(lf, 0, res['p_success'], alpha=0.15, color='steelblue')
    axes[0].set_xlabel(r'Lesion fraction $|\mathcal{D}|/N$', fontsize=11)
    axes[0].set_ylabel('P(regeneration success)', fontsize=11)
    axes[0].set_title('Regeneration probability', fontsize=10)
    axes[0].set_ylim(-0.05, 1.05); axes[0].grid(True, alpha=0.3)

    mask = ~np.isnan(res['mean_time'])
    if mask.sum() > 1:
        axes[1].plot(lf[mask], res['mean_time'][mask], 's-',
                     color='darkorange', lw=2, ms=6)
    axes[1].set_xlabel(r'Lesion fraction', fontsize=11)
    axes[1].set_ylabel('Mean regeneration time', fontsize=11)
    axes[1].set_title('Time to regeneration', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(lf, res['mean_fid'], 'd-', color='firebrick', lw=2, ms=6)
    axes[2].set_xlabel(r'Lesion fraction', fontsize=11)
    axes[2].set_ylabel('Mean pattern fidelity', fontsize=11)
    axes[2].set_title('Pattern fidelity (recovered / original)', fontsize=10)
    axes[2].set_ylim(-0.05, 1.05); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison(res_list: list, title: str, save_path: str) -> None:
    """Overlay multiple experiment_regen_vs_lesion results."""
    colors = ['steelblue', 'firebrick', 'darkorange', 'seagreen']
    styles = ['-o', '--s', '-.d', ':^']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=11, fontweight='bold')

    for res, c, st in zip(res_list, colors, styles):
        lf = res['lesion_fracs']
        lbl = res['label']
        axes[0].plot(lf, res['p_success'], st, color=c, lw=2, ms=6, label=lbl)
        mask = ~np.isnan(res['mean_time'])
        if mask.sum() > 1:
            axes[1].plot(lf[mask], res['mean_time'][mask],
                         st, color=c, lw=2, ms=6, label=lbl)
        axes[2].plot(lf, res['mean_fid'], st, color=c, lw=2, ms=6, label=lbl)

    for ax, ylabel, title_ax in zip(
            axes,
            ['P(regeneration success)', 'Mean regen time', 'Pattern fidelity'],
            ['Regeneration probability', 'Time to regeneration',
             'Pattern fidelity']):
        ax.set_xlabel(r'Lesion fraction $|\mathcal{D}|/N$', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title_ax, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    axes[0].axhline(0.5, color='gray', lw=1, ls='--', alpha=0.5)
    axes[0].set_ylim(-0.05, 1.05)
    axes[2].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "="*65)
    print("  VALIDATION EXPERIMENTS  C1 – C4")
    print("="*65)

    lesion_fracs = np.linspace(0.05, 0.65, 9)

    # ---- C1: Nucleation threshold ----
    print("\n[C1] Nucleation threshold")
    c1 = experiment_nucleation_threshold(
        N=25, T=70.0,
        delta_vals=np.concatenate([
            np.linspace(0.02, 0.20, 6),
            np.linspace(0.25, 1.20, 7)
        ]),
        n_seeds=20)

    # ---- C2: Full model regen vs lesion ----
    print("\n[C2] Regeneration vs lesion (full model)")
    c2_full = experiment_regen_vs_lesion(
        N=30, T_eq=40.0, T_regen=80.0,
        lesion_fracs=lesion_fracs,
        n_seeds=10, label="Full model",
        save_path="figures/C2_regen_full.png")

    # ---- C3: R_i knockout ----
    print("\n[C3] Regeneration vs lesion (R_i knockout)")
    # Disable R module: set all R-related couplings to zero
    c3_ko = experiment_regen_vs_lesion(
        N=30, T_eq=40.0, T_regen=80.0,
        lesion_fracs=lesion_fracs,
        n_seeds=10,
        params_override={'a_RW': 0.0, 'a_RV': 0.0, 'a_PR': 0.0,
                         'a_RD': 0.0},
        label=r"$R_i$ knockout",
        save_path=None)

    plot_comparison(
        [c2_full, c3_ko],
        "C3 — Full model vs $R_i$ knockout\nRegeneration across lesion sizes",
        "figures/C3_Ri_knockout.png")

    # ---- C4: Fixed vs adaptive conductance ----
    print("\n[C4] Regeneration vs lesion (fixed G)")
    c4_fixed = experiment_regen_vs_lesion(
        N=30, T_eq=40.0, T_regen=80.0,
        lesion_fracs=lesion_fracs,
        n_seeds=10,
        # Freeze G: no Hebbian terms, no decay → dG/dt ≈ 0
        params_override={'alpha_G': 0.0, 'beta_G': 0.0,
                         'gamma_G': 0.0, 'lambda_G': 0.0},
        label="Fixed $G_{ij}$",
        save_path=None)

    plot_comparison(
        [c2_full, c4_fixed],
        "C4 — Adaptive vs fixed $G_{ij}$\nRegeneration across lesion sizes",
        "figures/C4_fixed_G.png")

    # ---- Summary ----
    print("\n" + "="*65)
    print("  Validation experiments complete. Figures generated:")
    for f in sorted(os.listdir("figures")):
        if f.startswith("C"):
            print(f"  figures/{f}")
    print("="*65)

    # ---- Print key numerical results for LaTeX ----
    print("\n--- Key numbers for text ---")
    # C1: threshold
    cross = np.where(c1['p_pattern'] >= 0.5)[0]
    if len(cross):
        print(f"C1: delta* = {c1['delta_vals'][cross[0]]:.2f}  "
              f"(p_pattern={c1['p_pattern'][cross[0]]:.2f})")

    # C2: critical lesion fraction
    fail = np.where(c2_full['p_success'] < 0.5)[0]
    if len(fail):
        print(f"C2: critical lesion fraction = "
              f"{c2_full['lesion_fracs'][fail[0]]:.2f}")

    # C3: ratio at largest common lesion where full works but KO fails
    for lf, pf, pk in zip(lesion_fracs,
                           c2_full['p_success'], c3_ko['p_success']):
        if pf > 0.5 and pk < 0.5:
            print(f"C3: at lesion={lf:.2f}, full p={pf:.2f}, KO p={pk:.2f}")

    # C3: time ratio (first lesion where both succeed)
    for lf, tf, tk in zip(lesion_fracs,
                           c2_full['mean_time'], c3_ko['mean_time']):
        if not np.isnan(tf) and not np.isnan(tk) and tk > 0:
            print(f"C3: regen time ratio (KO/full) at lesion={lf:.2f}: "
                  f"{tk/tf:.1f}x")
            break

    # C4: adaptive advantage
    for lf, pa, pf in zip(lesion_fracs,
                           c2_full['p_success'], c4_fixed['p_success']):
        if pa > 0.5 and pf < 0.5:
            print(f"C4: adaptive wins over fixed at lesion={lf:.2f} "
                  f"(adaptive p={pa:.2f}, fixed p={pf:.2f})")
            break
