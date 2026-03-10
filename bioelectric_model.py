"""
bioelectric_model.py
Yuriria Cortés Poza
====================
Hybrid mathematical framework for morphogenesis and regeneration.
Stages I + II: Bistable bioelectric network + Synthetic gene regulatory network.

State variables per cell i (continuous):
  V_i   : membrane potential (bistable cubic dynamics)
  P_i   : proliferative module  (GRN)
  D_i   : differentiation module (GRN)
  I_i   : positional/identity module (GRN)
  R_i   : regenerative module (GRN)
  eps_i : slow tissue-memory variable
  W_i   : wound signal (propagating field)

Edge variables:
  G_ij  : adaptive gap-junction conductance (symmetric)

Discrete variables:
  s_i   : morphological state in {0, 1, 2, 3, 4}
  n_i   : occupancy in {0, 1}

Numerical scheme: 4th-order Runge–Kutta for continuous variables,
                  threshold rule for discrete state update.

Reference: "A Hybrid Mathematical Framework for Morphogenesis and Regeneration
            Based on Bistable Bioelectric Networks, Synthetic Gene Regulation,
            and Emergent Tissue Memory"

Authors: [to be filled]
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# =============================================================================
#  PARAMETERS
# =============================================================================

@dataclass
class ModelParameters:
    """
    All model parameters with physically and biologically motivated defaults.

    Time unit: dimensionless (1 unit ~ fast membrane relaxation scale).
    Voltage unit: normalized so that V_minus = -1.5, V_plus = +1.5.
    """

    # --- Network geometry ---
    N: int   = 50      # number of cells (1D) or cells per side (2D)
    dim: int = 1       # spatial dimension: 1 or 2

    # --- Bioelectric layer: bistable cubic ---
    # f(V) = a_bist * (V - V_minus)(V - V_0)(V - V_plus)
    # Stable equilibria: V_minus (hyperpolarized), V_plus (depolarized)
    # Unstable equilibrium: V_0
    C:       float = 1.0    # effective membrane capacitance
    a_bist:  float = 1.0    # bistability strength coefficient
    V_minus: float = -1.5   # stable hyperpolarized state
    V_0:     float =  0.0   # unstable middle equilibrium
    V_plus:  float =  1.5   # stable depolarized state

    # GRN -> V feedback (Gamma_i term)
    # Gamma_i = beta_1*P_i - beta_2*D_i + beta_3*R_i + beta_4*eps_i
    beta_1: float = 0.30    # P depolarizes
    beta_2: float = 0.25    # D hyperpolarizes
    beta_3: float = 0.20    # R depolarizes
    beta_4: float = 0.10    # eps stabilizes

    # --- Adaptive gap junctions ---
    # tau_G * dG_ij/dt = F_G(V_i,V_j,R_i,R_j,eps_i,eps_j) - lambda_G * G_ij
    G_init:   float = 0.50   # initial conductance on all edges
    tau_G:    float = 5.0    # conductance adaptation time scale
    lambda_G: float = 0.10   # linear decay of conductance
    alpha_G:  float = 0.30   # voltage-driven coupling (sigma(V_i)*sigma(V_j))
    beta_G:   float = 0.20   # regenerative-module coupling
    gamma_G:  float = 0.15   # slow-memory coupling

    # --- Synthetic GRN ---
    # P: proliferative module
    rho_P: float = 0.50      # logistic growth rate
    a_PD:  float = 0.30      # D inhibits P
    a_PV:  float = 0.40      # V (depolarized) activates P
    a_PR:  float = 0.20      # R activates P
    a_Pe:  float = 0.10      # eps inhibits P (commitment reduces proliferation)
    a_PW:  float = 0.30      # wound activates P

    # D: differentiation module
    rho_D: float = 0.30
    a_DP:  float = 0.25      # P activates D
    a_DI:  float = 0.20      # I activates D
    a_DV:  float = 0.30      # V modulates D
    a_De:  float = 0.15      # eps promotes differentiation

    # I: positional/identity module
    rho_I: float = 0.20
    a_IV:  float = 0.40      # V (hyperpolarized) activates I
    a_Ie:  float = 0.20      # eps activates I
    a_IN:  float = 0.30      # neighbor morphology -> I (positional context)

    # R: regenerative module
    rho_R: float = 0.40
    a_RW:  float = 0.50      # wound strongly activates R
    a_RV:  float = 0.30      # V activates R
    a_RD:  float = 0.20      # D inhibits R

    # --- Slow tissue memory ---
    # tau_eps * deps/dt = -lambda_eps*eps + b_eV*sigma(V) + b_eI*I + b_eD*D
    tau_eps:    float = 50.0   # slow time scale (>> GRN scale)
    lambda_eps: float = 0.05
    b_eV:       float = 0.30
    b_eI:       float = 0.20
    b_eD:       float = 0.15

    # --- Wound signal ---
    # tau_W * dW/dt = -lambda_W*W + D_W * laplacian(W) + S_lesion
    tau_W:    float = 2.0
    lambda_W: float = 0.50
    D_W:      float = 1.0    # propagation rate on cellular graph

    # --- Morphological state thresholds ---
    theta_P:   float = 0.45
    theta_D:   float = 0.45
    theta_I:   float = 0.45
    theta_eps: float = 0.40

    # --- Numerical ---
    dt:       float = 0.01   # integration time step
    dt_morph: float = 0.10   # morphological state update interval


# =============================================================================
#  ACTIVATION FUNCTIONS
# =============================================================================

def sigma(x: np.ndarray, x0: float = 0.0, k: float = 5.0) -> np.ndarray:
    """
    Smooth sigmoid: sigma(x; x0, k) = 1 / (1 + exp(-k*(x - x0))).
    Clipped for numerical stability.
    """
    z = np.clip(-k * (x - x0), -500, 500)
    return 1.0 / (1.0 + np.exp(z))


def phi_embed(s: np.ndarray) -> np.ndarray:
    """
    Embedding of discrete morphological state s in {0,...,4} into [0,1].
    Used for neighbor positional context in I dynamics.
    """
    return s.astype(float) / 4.0


# =============================================================================
#  MAIN NETWORK CLASS
# =============================================================================

class BioelectricNetwork:
    """
    Hybrid bioelectric–regulatory network on a 1D or 2D lattice.

    Continuous state matrix X: shape (N_cells, 7)
      columns: [V, P, D, I, R, eps, W]

    Edge conductance matrix G: shape (N_cells, N_cells), symmetric, sparse.

    Discrete states:
      s: shape (N_cells,), values in {0,1,2,3,4}
         0 = empty / dead
         1 = proliferative
         2 = early differentiated
         3 = positionally committed, low memory
         4 = fully committed, high memory
      n: shape (N_cells,), occupancy in {0, 1}
    """

    # Column indices in state matrix X
    IDX_V   = 0
    IDX_P   = 1
    IDX_D   = 2
    IDX_I   = 3
    IDX_R   = 4
    IDX_EPS = 5
    IDX_W   = 6
    N_VARS  = 7

    def __init__(self, params: ModelParameters, seed: int = 42,
                 polarity_gradient: bool = False):
        """
        Parameters
        ----------
        params            : ModelParameters
        seed              : random seed
        polarity_gradient : if True, initialize a head-tail positional gradient
                            along axis 0 (planaria-inspired)
        """
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.polarity_gradient = polarity_gradient
        self._build_geometry()
        self._initialize_state()

    # ------------------------------------------------------------------
    #  GEOMETRY
    # ------------------------------------------------------------------

    def _build_geometry(self):
        p = self.p
        if p.dim == 1:
            self.N_cells = p.N
            self.shape   = (p.N,)
            self.neighbors = [[] for _ in range(self.N_cells)]
            for i in range(self.N_cells):
                if i > 0:
                    self.neighbors[i].append(i - 1)
                if i < self.N_cells - 1:
                    self.neighbors[i].append(i + 1)
        else:
            self.N_cells = p.N * p.N
            self.shape   = (p.N, p.N)
            self.neighbors = [[] for _ in range(self.N_cells)]
            for row in range(p.N):
                for col in range(p.N):
                    idx = row * p.N + col
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        r2, c2 = row+dr, col+dc
                        if 0 <= r2 < p.N and 0 <= c2 < p.N:
                            self.neighbors[idx].append(r2 * p.N + c2)

        # Conductance matrix (dense; can be made sparse for large N)
        self.G = np.zeros((self.N_cells, self.N_cells))
        for i in range(self.N_cells):
            for j in self.neighbors[i]:
                self.G[i, j] = p.G_init

        # Degree vector (number of neighbors per cell)
        self.degree = np.array([len(nb) for nb in self.neighbors], dtype=float)

    # ------------------------------------------------------------------
    #  INITIALIZATION
    # ------------------------------------------------------------------

    def _initialize_state(self):
        p   = self.p
        N   = self.N_cells
        rng = self.rng

        self.X = np.zeros((N, self.N_VARS))

        # Voltage: near hyperpolarized state + small noise
        self.X[:, self.IDX_V] = p.V_minus + 0.05 * rng.standard_normal(N)

        # GRN variables: small positive values + noise
        self.X[:, self.IDX_P] = np.clip(0.10 + 0.02 * rng.standard_normal(N), 0, 1)
        self.X[:, self.IDX_D] = np.clip(0.10 + 0.02 * rng.standard_normal(N), 0, 1)
        self.X[:, self.IDX_R] = 0.0

        # Positional module: optional axial gradient (planaria polarity)
        if self.polarity_gradient:
            if p.dim == 1:
                gradient = np.linspace(0.05, 0.50, N)
            else:
                row_idx  = np.arange(N) // p.N
                gradient = np.linspace(0.05, 0.50, p.N)[row_idx]
            self.X[:, self.IDX_I] = np.clip(
                gradient + 0.02 * rng.standard_normal(N), 0, 1)
        else:
            self.X[:, self.IDX_I] = np.clip(0.10 + 0.02 * rng.standard_normal(N), 0, 1)

        self.X[:, self.IDX_EPS] = np.clip(0.10 + 0.02 * rng.standard_normal(N), 0, 1)
        self.X[:, self.IDX_W]   = 0.0

        # Discrete states
        self.s = np.ones(N, dtype=int)   # all undifferentiated at t=0
        self.n = np.ones(N, dtype=int)   # all occupied
        self.t = 0.0

    # ------------------------------------------------------------------
    #  BISTABLE IONIC CURRENT
    # ------------------------------------------------------------------

    def _f_bistable(self, V: np.ndarray) -> np.ndarray:
        """
        f(V) = a_bist * (V - V_minus)(V - V_0)(V - V_plus)

        Zero crossings at V_minus (stable), V_0 (unstable), V_plus (stable).
        """
        p = self.p
        return p.a_bist * (V - p.V_minus) * (V - p.V_0) * (V - p.V_plus)

    # ------------------------------------------------------------------
    #  RIGHT-HAND SIDES
    # ------------------------------------------------------------------

    def _rhs_V(self, X: np.ndarray, G: np.ndarray,
               ext: np.ndarray) -> np.ndarray:
        """
        C * dV_i/dt = -f(V_i)
                     + sum_j G_ij (V_j - V_i)     [gap-junction current]
                     + Gamma_i(g_i, eps_i)          [GRN + memory feedback]
                     + I^ext_i                       [external perturbation]
                     + W_i                           [wound depolarization]

        Gamma_i = beta_1*P_i - beta_2*D_i + beta_3*R_i + beta_4*eps_i
        """
        p   = self.p
        V   = X[:, self.IDX_V]
        P   = X[:, self.IDX_P]
        D   = X[:, self.IDX_D]
        R   = X[:, self.IDX_R]
        eps = X[:, self.IDX_EPS]
        W   = X[:, self.IDX_W]

        f_V = self._f_bistable(V)

        # Gap-junction current: (G @ V)_i - (sum_j G_ij) * V_i
        GJ = G @ V - G.sum(axis=1) * V

        Gamma = p.beta_1 * P - p.beta_2 * D + p.beta_3 * R + p.beta_4 * eps

        return (1.0 / p.C) * (-f_V + GJ + Gamma + ext + W)

    def _rhs_G(self, X: np.ndarray, G: np.ndarray) -> np.ndarray:
        """
        tau_G * dG_ij/dt = F_G(i,j) - lambda_G * G_ij

        F_G = alpha_G * sigma(V_i) * sigma(V_j)
            + beta_G  * (R_i + R_j) / 2
            + gamma_G * (eps_i + eps_j) / 2

        Only edges that exist in the graph are updated (mask by G > 0).
        """
        p   = self.p
        V   = X[:, self.IDX_V]
        R   = X[:, self.IDX_R]
        eps = X[:, self.IDX_EPS]

        sV  = sigma(V, x0=0.0, k=3.0)

        F_G = (p.alpha_G * np.outer(sV, sV)
               + p.beta_G  * 0.5 * (R[:, None]   + R[None, :])
               + p.gamma_G * 0.5 * (eps[:, None]  + eps[None, :]))

        adj = G > 0
        dG  = np.where(adj, (F_G - p.lambda_G * G) / p.tau_G, 0.0)
        return dG

    def _rhs_GRN(self, X: np.ndarray) -> np.ndarray:
        """
        Returns dGRN/dt as array of shape (N, 4) = [dP, dD, dI, dR].

        dP_i/dt = rho_P * P_i*(1-P_i) - a_PD*D_i + a_PV*sigma_P(V_i)
                  + a_PR*R_i - a_Pe*eps_i + a_PW*W_i

        dD_i/dt = -rho_D*D_i + a_DP*P_i + a_DI*I_i
                  + a_DV*sigma_D(V_i) + a_De*eps_i

        dI_i/dt = -rho_I*I_i + a_IV*sigma_I(V_i) + a_Ie*eps_i
                  + a_IN * mean_{j in N(i)} phi(s_j)

        dR_i/dt = -rho_R*R_i + a_RW*W_i + a_RV*sigma_R(V_i) - a_RD*D_i
        """
        p   = self.p
        V   = X[:, self.IDX_V]
        P   = X[:, self.IDX_P]
        D   = X[:, self.IDX_D]
        I   = X[:, self.IDX_I]
        R   = X[:, self.IDX_R]
        eps = X[:, self.IDX_EPS]
        W   = X[:, self.IDX_W]

        # Sigmoidal responses at different voltage thresholds
        sP = sigma(V, x0= 0.5, k=5.0)   # depolarized -> activates P
        sD = sigma(V, x0= 0.0, k=5.0)   # intermediate -> modulates D
        sI = sigma(V, x0=-0.5, k=5.0)   # slightly hyperpolarized -> activates I
        sR = sigma(V, x0= 0.5, k=5.0)   # depolarized -> activates R

        # Neighbor positional embedding for I
        phi_s = phi_embed(self.s)
        nb_phi = np.zeros(self.N_cells)
        for i in range(self.N_cells):
            nb = self.neighbors[i]
            if nb:
                nb_phi[i] = np.mean(phi_s[nb])

        dP = (p.rho_P * P * (1.0 - P)
              - p.a_PD * D
              + p.a_PV * sP
              + p.a_PR * R
              - p.a_Pe * eps
              + p.a_PW * W)

        dD = (-p.rho_D * D
              + p.a_DP * P
              + p.a_DI * I
              + p.a_DV * sD
              + p.a_De * eps)

        dI = (-p.rho_I * I
              + p.a_IV * sI
              + p.a_Ie * eps
              + p.a_IN * nb_phi)

        dR = (-p.rho_R * R
              + p.a_RW * W
              + p.a_RV * sR
              - p.a_RD * D)

        return np.stack([dP, dD, dI, dR], axis=1)

    def _rhs_eps(self, X: np.ndarray) -> np.ndarray:
        """
        tau_eps * deps_i/dt = -lambda_eps*eps_i
                              + b_eV*sigma(V_i) + b_eI*I_i + b_eD*D_i
        """
        p   = self.p
        V   = X[:, self.IDX_V]
        I   = X[:, self.IDX_I]
        D   = X[:, self.IDX_D]
        eps = X[:, self.IDX_EPS]
        sV  = sigma(V, x0=0.0, k=3.0)
        return (-p.lambda_eps * eps + p.b_eV * sV
                + p.b_eI * I + p.b_eD * D) / p.tau_eps

    def _rhs_W(self, X: np.ndarray, G: np.ndarray,
               S_lesion: np.ndarray) -> np.ndarray:
        """
        tau_W * dW_i/dt = -lambda_W*W_i
                         + D_W * sum_j A_ij * (W_j - W_i)
                         + S^lesion_i(t)

        A_ij = 1 if (i,j) is an edge, unweighted for wound propagation.
        S_lesion: external source at damaged boundary cells.
        """
        p   = self.p
        W   = X[:, self.IDX_W]
        adj = (G > 0).astype(float)
        diff_W = adj @ W - adj.sum(axis=1) * W
        return (-p.lambda_W * W + p.D_W * diff_W + S_lesion) / p.tau_W

    # ------------------------------------------------------------------
    #  FULL RIGHT-HAND SIDE
    # ------------------------------------------------------------------

    def _full_rhs(self, X: np.ndarray, G: np.ndarray,
                  ext: np.ndarray, S_lesion: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (dX, dG) for the full coupled system."""
        dX = np.zeros_like(X)
        dX[:, self.IDX_V]   = self._rhs_V(X, G, ext)
        grn                  = self._rhs_GRN(X)
        dX[:, self.IDX_P]   = grn[:, 0]
        dX[:, self.IDX_D]   = grn[:, 1]
        dX[:, self.IDX_I]   = grn[:, 2]
        dX[:, self.IDX_R]   = grn[:, 3]
        dX[:, self.IDX_EPS] = self._rhs_eps(X)
        dX[:, self.IDX_W]   = self._rhs_W(X, G, S_lesion)
        dG                   = self._rhs_G(X, G)
        return dX, dG

    # ------------------------------------------------------------------
    #  RK4 INTEGRATION STEP
    # ------------------------------------------------------------------

    def _rk4_step(self, X: np.ndarray, G: np.ndarray, dt: float,
                  ext: np.ndarray, S_lesion: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """One RK4 step for (X, G)."""
        k1X, k1G = self._full_rhs(X,                  G,                  ext, S_lesion)
        k2X, k2G = self._full_rhs(X + 0.5*dt*k1X,    G + 0.5*dt*k1G,    ext, S_lesion)
        k3X, k3G = self._full_rhs(X + 0.5*dt*k2X,    G + 0.5*dt*k2G,    ext, S_lesion)
        k4X, k4G = self._full_rhs(X +     dt*k3X,    G +     dt*k3G,    ext, S_lesion)

        X_new = X + (dt / 6.0) * (k1X + 2*k2X + 2*k3X + k4X)
        G_new = G + (dt / 6.0) * (k1G + 2*k2G + 2*k3G + k4G)

        # Enforce bounds
        X_new[:, self.IDX_P:self.IDX_R+1] = np.clip(
            X_new[:, self.IDX_P:self.IDX_R+1], 0.0, 1.0)
        X_new[:, self.IDX_EPS] = np.clip(X_new[:, self.IDX_EPS], 0.0, 1.0)
        X_new[:, self.IDX_W]   = np.maximum(X_new[:, self.IDX_W],  0.0)
        G_new                   = np.maximum(G_new, 0.0)
        return X_new, G_new

    # ------------------------------------------------------------------
    #  MORPHOLOGICAL STATE UPDATE
    # ------------------------------------------------------------------

    def _update_morphological_state(self):
        """
        Classify each occupied cell into s_i in {1,2,3,4} using thresholds.

        s = 0 : empty (n_i = 0)
        s = 1 : proliferative    (P > theta_P, D <= theta_D)
        s = 2 : early diff.      (D > theta_D, I <= theta_I)
        s = 3 : positionally committed, low memory
                                 (D > theta_D, I > theta_I, eps <= theta_eps)
        s = 4 : fully committed  (D > theta_D, I > theta_I, eps >  theta_eps)
        """
        p   = self.p
        P   = self.X[:, self.IDX_P]
        D   = self.X[:, self.IDX_D]
        I   = self.X[:, self.IDX_I]
        eps = self.X[:, self.IDX_EPS]

        s          = np.ones(self.N_cells, dtype=int)
        s[self.n == 0] = 0
        occ = self.n == 1

        s[occ & (P > p.theta_P) & (D <= p.theta_D)]                      = 1
        s[occ & (D > p.theta_D) & (I <= p.theta_I)]                      = 2
        s[occ & (D > p.theta_D) & (I > p.theta_I) & (eps <= p.theta_eps)] = 3
        s[occ & (D > p.theta_D) & (I > p.theta_I) & (eps >  p.theta_eps)] = 4
        self.s = s

    # ------------------------------------------------------------------
    #  PERTURBATION METHODS
    # ------------------------------------------------------------------

    def apply_lesion(self, cell_indices: np.ndarray, wound_amplitude: float = 1.5):
        """
        Remove cells: set n_i = 0, zero their state.
        Activate wound signal at boundary cells adjacent to the lesion.
        """
        self.n[cell_indices] = 0
        self.X[cell_indices, :] = 0.0
        self.s[cell_indices]    = 0

        boundary = set()
        for i in cell_indices:
            for j in self.neighbors[i]:
                if self.n[j] == 1:
                    boundary.add(j)

        for j in boundary:
            self.X[j, self.IDX_W] += wound_amplitude

    def apply_reoccupation(self, n_neighbors_required: int = 1):
        """
        Simple birth rule: empty sites adjacent to enough proliferating cells
        may be reoccupied.  New cells inherit the mean state of their neighbors.
        """
        p       = self.p
        empty   = np.where(self.n == 0)[0]
        for i in empty:
            nb  = self.neighbors[i]
            if not nb:
                continue
            prolif_nb = [j for j in nb
                         if self.n[j] == 1 and self.X[j, self.IDX_P] > p.theta_P]
            if len(prolif_nb) >= n_neighbors_required:
                self.n[i] = 1
                self.s[i] = 1
                # Initialize from neighbor average
                self.X[i, :] = self.X[np.array(prolif_nb), :].mean(axis=0)
                self.X[i, self.IDX_W] = 0.0

    def set_external_current(self, cell_indices: np.ndarray,
                              amplitude: float) -> np.ndarray:
        """Return an external current vector."""
        ext                 = np.zeros(self.N_cells)
        ext[cell_indices]   = amplitude
        return ext

    # ------------------------------------------------------------------
    #  DIAGNOSTICS
    # ------------------------------------------------------------------

    def regeneration_score(self) -> float:
        """
        Fraction of originally occupied sites that are currently occupied.
        Simple criterion for regenerative success.
        """
        return float(self.n.sum()) / self.N_cells

    def mean_conductance(self) -> float:
        """Mean gap-junction conductance over active edges."""
        adj = self.G > 0
        if adj.sum() == 0:
            return 0.0
        return float(self.G[adj].mean())

    # ------------------------------------------------------------------
    #  MAIN SIMULATION LOOP
    # ------------------------------------------------------------------

    def run(self,
            T: float,
            ext_func:      Optional[Callable] = None,
            apply_birth:   bool = False,
            record_every:  int  = 20,
            verbose:       bool = True) -> Dict:
        """
        Integrate the model for time T using RK4.

        Parameters
        ----------
        T            : total simulation time
        ext_func     : callable(t) -> np.ndarray(N_cells), external currents
        apply_birth  : if True, run reoccupation rule every 10 steps
        record_every : save snapshot every this many steps
        verbose      : print progress

        Returns
        -------
        history : dict with keys:
          'times', 'V', 'P', 'D', 'I', 'R', 'eps', 'W', 'G', 's', 'n',
          'regen_score', 'mean_G'
        """
        p        = self.p
        dt       = p.dt
        n_steps  = int(round(T / dt))
        m_int    = max(1, int(p.dt_morph / dt))

        history = {k: [] for k in
                   ['times','V','P','D','I','R','eps','W','G','s','n',
                    'regen_score','mean_G']}

        S_lesion = np.zeros(self.N_cells)

        for step in range(n_steps):
            t   = self.t
            ext = (ext_func(t) if ext_func is not None
                   else np.zeros(self.N_cells))

            self.X, self.G = self._rk4_step(
                self.X, self.G, dt, ext, S_lesion)

            # Zero dead cells
            self.X[self.n == 0, :] = 0.0

            # Morphological update
            if step % m_int == 0:
                self._update_morphological_state()

            # Reoccupation
            if apply_birth and step % 10 == 0:
                self.apply_reoccupation()

            self.t += dt

            if step % record_every == 0:
                history['times'].append(t)
                history['V'].append(self.X[:, self.IDX_V].copy())
                history['P'].append(self.X[:, self.IDX_P].copy())
                history['D'].append(self.X[:, self.IDX_D].copy())
                history['I'].append(self.X[:, self.IDX_I].copy())
                history['R'].append(self.X[:, self.IDX_R].copy())
                history['eps'].append(self.X[:, self.IDX_EPS].copy())
                history['W'].append(self.X[:, self.IDX_W].copy())
                history['G'].append(self.G.copy())
                history['s'].append(self.s.copy())
                history['n'].append(self.n.copy())
                history['regen_score'].append(self.regeneration_score())
                history['mean_G'].append(self.mean_conductance())

            if verbose and n_steps >= 10 and step % (n_steps // 10) == 0:
                print(f"  t={t:7.2f}/{T:.1f}  "
                      f"|V|_mean={self.X[:,0].mean():+.3f}  "
                      f"<P>={self.X[:,1].mean():.3f}  "
                      f"<D>={self.X[:,2].mean():.3f}  "
                      f"occ={self.n.mean():.2f}")

        for k in ['V','P','D','I','R','eps','W','s','n']:
            history[k] = np.array(history[k])
        history['G']           = np.array(history['G'])
        history['times']       = np.array(history['times'])
        history['regen_score'] = np.array(history['regen_score'])
        history['mean_G']      = np.array(history['mean_G'])
        return history


# =============================================================================
#  VISUALIZATION
# =============================================================================

_VAR_INFO = [
    ('V',   r'Potential $V_i$',        'RdBu_r',   None,   None),
    ('P',   r'Proliferative $P_i$',    'Greens',   0.0,    1.0),
    ('D',   r'Differentiation $D_i$',  'Oranges',  0.0,    1.0),
    ('I',   r'Positional $I_i$',       'Purples',  0.0,    1.0),
    ('R',   r'Regenerative $R_i$',     'Reds',     0.0,    1.0),
    ('eps', r'Memory $\varepsilon_i$', 'Blues',    0.0,    1.0),
    ('W',   r'Wound $W_i$',            'YlOrBr',   None,   None),
    ('s',   r'State $s_i$',            'tab10',    0,      4),
    ('n',   r'Occupancy $n_i$',        'Greys',    0,      1),
]


def plot_spacetime_1d(history: Dict, title: str = "",
                      save_path: Optional[str] = None) -> plt.Figure:
    """Space–time heatmaps for all state variables (1D model)."""
    times = history['times']
    fig, axes = plt.subplots(3, 3, figsize=(17, 11))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    for ax, (key, label, cmap, vmin, vmax) in zip(axes.flat, _VAR_INFO):
        data = history[key].T        # (N_cells, n_times)
        vm   = data.min() if vmin is None else vmin
        vM   = data.max() if vmax is None else vmax
        im   = ax.imshow(data, aspect='auto', origin='lower',
                         extent=[times[0], times[-1], 0, data.shape[0]],
                         cmap=cmap, vmin=vm, vmax=vM)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('Cell', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_snapshot_2d(history: Dict, time_idx: int = -1,
                     title: str = "", N: int = 20,
                     save_path: Optional[str] = None) -> plt.Figure:
    """2D spatial snapshot at a given recorded time index."""
    t_val = history['times'][time_idx]
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle(f"{title}   (t = {t_val:.1f})", fontsize=12, fontweight='bold')

    for ax, (key, label, cmap, vmin, vmax) in zip(axes.flat, _VAR_INFO):
        data = history[key][time_idx].reshape(N, N)
        vm   = data.min() if vmin is None else vmin
        vM   = data.max() if vmax is None else vmax
        im   = ax.imshow(data, cmap=cmap, vmin=vm, vmax=vM, origin='lower',
                         interpolation='nearest')
        ax.set_title(label, fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_voltage_timeseries(history: Dict, cell_indices: List[int],
                            title: str = "",
                            save_path: Optional[str] = None) -> plt.Figure:
    """Voltage time series for selected cells."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for idx in cell_indices:
        ax.plot(history['times'], history['V'][:, idx],
                label=f'Cell {idx}', linewidth=1.5)
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel(r'$V_i(t)$', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, ncol=min(4, len(cell_indices)))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_regen_score(histories: Dict[str, Dict], title: str = "",
                     save_path: Optional[str] = None) -> plt.Figure:
    """Regeneration score over time for multiple conditions."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, hist in histories.items():
        ax.plot(hist['times'], hist['regen_score'], label=label, lw=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Regeneration score', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# =============================================================================
#  EXPERIMENTS
# =============================================================================

def experiment_spontaneous_patterning(
        params: Optional[ModelParameters] = None,
        dim: int = 1, T: float = 120.0,
        save_dir: str = ".") -> Dict:
    """
    Experiment 1 — Spontaneous pattern formation.

    From near-homogeneous initial conditions, determine whether the coupled
    bistable bioelectric + GRN system spontaneously breaks spatial symmetry
    and generates stable heterogeneous patterns.
    """
    if params is None:
        params = ModelParameters(N=50 if dim == 1 else 20, dim=dim)

    print(f"\n{'='*65}")
    print(f"Experiment 1: Spontaneous patterning ({dim}D, N={params.N})")
    print('='*65)

    net     = BioelectricNetwork(params, seed=0)
    history = net.run(T=T, record_every=20, verbose=True)

    tag = f"exp1_patterning_{dim}d"
    if dim == 1:
        plot_spacetime_1d(history,
                          title="Exp 1: Spontaneous pattern formation (1D)",
                          save_path=f"{save_dir}/{tag}.png")
        plot_voltage_timeseries(
            history,
            cell_indices=list(range(0, params.N, params.N // 8)),
            title="Exp 1: Voltage traces — selected cells",
            save_path=f"{save_dir}/{tag}_V_traces.png")
    else:
        for t_idx, t_tag in [(len(history['times'])//4, 'early'),
                              (-1, 'final')]:
            plot_snapshot_2d(history, time_idx=t_idx,
                             title=f"Exp 1: Patterning ({t_tag})",
                             N=params.N,
                             save_path=f"{save_dir}/{tag}_{t_tag}.png")
    plt.close('all')
    return history


def experiment_lesion_regeneration(
        params: Optional[ModelParameters] = None,
        dim: int = 1, T_pre: float = 80.0, T_post: float = 120.0,
        lesion_fraction: float = 0.25,
        save_dir: str = ".") -> Dict:
    """
    Experiment 2 — Lesion and regeneration.

    Equilibrate tissue, then remove a central region, then observe
    recovery dynamics.  Records regeneration score over time.
    """
    if params is None:
        params = ModelParameters(N=50 if dim == 1 else 20, dim=dim)

    print(f"\n{'='*65}")
    print(f"Experiment 2: Lesion & regeneration ({dim}D, "
          f"lesion={100*lesion_fraction:.0f}%)")
    print('='*65)

    net = BioelectricNetwork(params, seed=1, polarity_gradient=True)

    print("  Phase 1: Equilibration...")
    hist_pre = net.run(T=T_pre, record_every=20, verbose=True)

    # Define lesion region
    N = net.N_cells
    if dim == 1:
        c     = N // 2
        w     = max(1, int(lesion_fraction * N))
        cells = np.arange(max(0, c - w//2), min(N, c + w//2))
    else:
        n_side = params.N
        c, w   = n_side // 2, max(1, int(lesion_fraction * n_side / 2))
        rs, cs = np.meshgrid(np.arange(c-w, c+w), np.arange(c-w, c+w))
        cells  = (rs.flatten() * n_side + cs.flatten())
        cells  = cells[(cells >= 0) & (cells < N)]

    net.apply_lesion(cells)
    print(f"  Lesion: {len(cells)} cells removed "
          f"({100*len(cells)/N:.1f}% of tissue)")

    print("  Phase 2: Regeneration...")
    hist_post = net.run(T=T_post, record_every=20,
                        apply_birth=True, verbose=True)

    tag = f"exp2_lesion_{dim}d"
    if dim == 1:
        plot_spacetime_1d(hist_pre,
                          title="Exp 2: Pre-lesion (1D)",
                          save_path=f"{save_dir}/{tag}_pre.png")
        plot_spacetime_1d(hist_post,
                          title="Exp 2: Post-lesion regeneration (1D)",
                          save_path=f"{save_dir}/{tag}_post.png")
    else:
        for t_idx, t_tag in [(0,'lesion'),(len(hist_post['times'])//2,'mid'),
                              (-1,'final')]:
            plot_snapshot_2d(hist_post, time_idx=t_idx,
                             title=f"Exp 2: {t_tag}", N=params.N,
                             save_path=f"{save_dir}/{tag}_{t_tag}.png")

    plot_regen_score({'post-lesion': hist_post},
                     title="Exp 2: Regeneration score over time",
                     save_path=f"{save_dir}/{tag}_score.png")
    plt.close('all')
    return {'pre': hist_pre, 'post': hist_post}


def experiment_transient_forcing(
        params: Optional[ModelParameters] = None,
        dim: int = 1, T_pre: float = 70.0,
        T_force: float = 10.0, T_post: float = 80.0,
        amplitude: float = 2.5, target_fraction: float = 0.25,
        save_dir: str = ".") -> Dict:
    """
    Experiment 3 — Transient electrical perturbation / attractor switching.

    Apply a localized depolarizing current for a finite time window.
    Tests whether the tissue permanently transitions to a different morphology.
    Planaria-inspired: comparable to bioelectric polarity reversal experiments.
    """
    if params is None:
        params = ModelParameters(N=50 if dim == 1 else 20, dim=dim)

    print(f"\n{'='*65}")
    print(f"Experiment 3: Transient forcing ({dim}D, A={amplitude})")
    print('='*65)

    net = BioelectricNetwork(params, seed=2, polarity_gradient=True)

    print("  Phase 1: Equilibration...")
    hist_pre = net.run(T=T_pre, record_every=20, verbose=True)

    # Target: anterior quarter of tissue
    N      = net.N_cells
    w      = max(1, int(target_fraction * N))
    target = np.arange(0, w)

    def ext_on(t):
        ext          = np.zeros(N)
        ext[target]  = amplitude
        return ext

    print(f"  Phase 2: Forcing {len(target)} cells "
          f"for t in [0, {T_force}]...")
    hist_force = net.run(T=T_force, ext_func=ext_on,
                         record_every=5, verbose=True)

    print("  Phase 3: Free relaxation after forcing...")
    hist_post = net.run(T=T_post, record_every=20, verbose=True)

    tag = f"exp3_forcing_{dim}d"
    if dim == 1:
        for hist, lbl in [(hist_pre,'pre'),(hist_force,'forcing'),(hist_post,'post')]:
            plot_spacetime_1d(hist,
                              title=f"Exp 3: {lbl} ({dim}D)",
                              save_path=f"{save_dir}/{tag}_{lbl}.png")

        # Compare V at end of pre vs end of post
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, hist, lbl in zip(axes,
                                  [hist_pre, hist_post],
                                  ['Pre-forcing', 'Post-forcing (relaxed)']):
            ax.plot(hist['V'][-1], color='steelblue', lw=1.5)
            ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
            ax.set_title(lbl, fontsize=10)
            ax.set_xlabel('Cell index')
            ax.set_ylabel(r'$V_i$')
            ax.set_ylim(params.V_minus - 0.3, params.V_plus + 0.3)
        fig.suptitle("Exp 3: Voltage profile before and after transient forcing",
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{tag}_compare.png", dpi=150, bbox_inches='tight')
    else:
        for t_idx, t_tag in [(-1,'pre'),(-1,'post')]:
            plot_snapshot_2d(hist_post if t_tag == 'post' else hist_pre,
                             time_idx=t_idx, title=f"Exp 3: {t_tag}",
                             N=params.N,
                             save_path=f"{save_dir}/{tag}_{t_tag}.png")
    plt.close('all')
    return {'pre': hist_pre, 'force': hist_force, 'post': hist_post}


def experiment_gap_junction_sweep(
        params: Optional[ModelParameters] = None,
        dim: int = 1, T_eq: float = 60.0, T_regen: float = 80.0,
        G_values: Optional[List[float]] = None,
        lesion_fraction: float = 0.20,
        save_dir: str = ".") -> Dict:
    """
    Experiment 4 — Gap-junction conductance sweep.

    For each baseline conductance G_init, equilibrate the tissue, apply a
    lesion, and measure regeneration success.  This maps the regenerative
    threshold as a function of intercellular coupling strength.
    """
    if params is None:
        params = ModelParameters(N=50 if dim == 1 else 20, dim=dim)
    if G_values is None:
        G_values = [0.05, 0.20, 0.50, 1.00, 2.00, 4.00]

    print(f"\n{'='*65}")
    print(f"Experiment 4: Gap-junction sweep ({dim}D)")
    print('='*65)

    results      = {}
    final_scores = []

    for G_val in G_values:
        from dataclasses import replace
        p_c = ModelParameters(**{**vars(params), 'G_init': G_val})
        net = BioelectricNetwork(p_c, seed=3)

        net.run(T=T_eq, record_every=200, verbose=False)

        N  = net.N_cells
        c  = N // 2
        w  = max(1, int(lesion_fraction * N))
        net.apply_lesion(np.arange(max(0, c-w//2), min(N, c+w//2)))

        hist         = net.run(T=T_regen, record_every=20,
                               apply_birth=True, verbose=False)
        final_scores.append(hist['regen_score'][-1])
        results[G_val] = hist

        print(f"  G_init = {G_val:.2f}  =>  "
              f"final regen score = {hist['regen_score'][-1]:.3f}  "
              f"<V>_final = {hist['V'][-1].mean():+.3f}")

    # Summary plot: regen score vs G
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(G_values, final_scores, 'o-', color='steelblue', lw=2, ms=7)
    axes[0].set_xlabel(r'$G_{\mathrm{init}}$', fontsize=12)
    axes[0].set_ylabel('Final regeneration score', fontsize=11)
    axes[0].set_title('Regenerative threshold vs. coupling', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Space-time V for extreme cases
    if dim == 1:
        for ax, G_val in zip(
                [axes[1]],
                [G_values[len(G_values)//2]]):
            data = results[G_val]['V'].T
            im   = ax.imshow(data, aspect='auto', origin='lower',
                             extent=[0, T_regen, 0, params.N],
                             cmap='RdBu_r')
            ax.set_title(f'$V_i(t)$ at $G_{{init}}={G_val}$', fontsize=10)
            ax.set_xlabel('Time'); ax.set_ylabel('Cell')
            plt.colorbar(im, ax=ax, fraction=0.03)

    plt.suptitle("Experiment 4: Gap-junction modulation", fontsize=11,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/exp4_gap_sweep_{dim}d.png",
                dpi=150, bbox_inches='tight')
    plt.close('all')
    return {'results': results, 'G_values': G_values,
            'final_scores': final_scores}


def experiment_memory_knockout(
        params: Optional[ModelParameters] = None,
        dim: int = 1, T_eq: float = 70.0, T_regen: float = 100.0,
        lesion_fraction: float = 0.25,
        save_dir: str = ".") -> Dict:
    """
    Experiment 5 — Slow-memory knockout.

    Compare regeneration with and without the slow tissue-memory layer eps.
    Tests Problem 5: which regenerative behaviors require eps?
    """
    if params is None:
        params = ModelParameters(N=50 if dim == 1 else 20, dim=dim)

    print(f"\n{'='*65}")
    print(f"Experiment 5: Memory knockout ({dim}D)")
    print('='*65)

    results = {}
    for label, tau_eps in [('with memory', 50.0), ('no memory (fast eps)', 0.5)]:
        from dataclasses import replace
        p_c = ModelParameters(**{**vars(params), 'tau_eps': tau_eps})
        net = BioelectricNetwork(p_c, seed=4, polarity_gradient=True)

        net.run(T=T_eq, record_every=200, verbose=False)

        N  = net.N_cells
        c  = N // 2
        w  = max(1, int(lesion_fraction * N))
        net.apply_lesion(np.arange(max(0, c-w//2), min(N, c+w//2)))

        hist         = net.run(T=T_regen, record_every=20,
                               apply_birth=True, verbose=False)
        results[label] = hist
        print(f"  {label:30s}  final regen = {hist['regen_score'][-1]:.3f}")

    plot_regen_score(results,
                     title="Exp 5: Regeneration score — memory vs. no memory",
                     save_path=f"{save_dir}/exp5_memory_knockout_{dim}d.png")
    plt.close('all')
    return results


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":

    SAVE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\n" + "="*70)
    print("  BIOELECTRIC MORPHOGENESIS MODEL — Stage I+II")
    print("  1D and 2D experiments")
    print("="*70)

    # ------------------------------------------------------------------ 1D
    p1 = ModelParameters(N=50, dim=1, dt=0.01)

    experiment_spontaneous_patterning(p1, dim=1, T=120.0,   save_dir=SAVE_DIR)
    experiment_lesion_regeneration   (p1, dim=1, T_pre=80.0, T_post=120.0,
                                      lesion_fraction=0.25, save_dir=SAVE_DIR)
    experiment_transient_forcing     (p1, dim=1, T_pre=70.0,
                                      T_force=10.0, T_post=80.0,
                                      amplitude=2.5, save_dir=SAVE_DIR)
    experiment_gap_junction_sweep    (p1, dim=1, T_eq=60.0, T_regen=80.0,
                                      G_values=[0.05,0.2,0.5,1.0,2.0,4.0],
                                      save_dir=SAVE_DIR)
    experiment_memory_knockout       (p1, dim=1, T_eq=70.0, T_regen=100.0,
                                      save_dir=SAVE_DIR)

    # ------------------------------------------------------------------ 2D
    p2 = ModelParameters(N=20, dim=2, dt=0.02)

    experiment_spontaneous_patterning(p2, dim=2, T=80.0,    save_dir=SAVE_DIR)
    experiment_lesion_regeneration   (p2, dim=2, T_pre=60.0, T_post=80.0,
                                      lesion_fraction=0.20, save_dir=SAVE_DIR)
    experiment_transient_forcing     (p2, dim=2, T_pre=60.0,
                                      T_force=8.0, T_post=70.0,
                                      amplitude=2.0, save_dir=SAVE_DIR)

    print(f"\nAll experiments complete. Figures saved to: {SAVE_DIR}")
