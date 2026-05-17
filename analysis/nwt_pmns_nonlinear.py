#!/usr/bin/env python3
"""
NWT PMNS Matrix: Nonlinear (GPE) Corrections to Mode Mixing

The linear Helmholtz solver gives n_θ=1 modes that are nearly pure
(n_s, n_θ) states.  The EH/GPE nonlinearity g|ψ|²ψ couples these
modes, breaking the degeneracies and creating the mixing that becomes
the PMNS matrix.

Physical picture:
  - Linear: H₀|ψₙ⟩ = λₙ|ψₙ⟩  (nearly diagonal in (n_s, n_θ) basis)
  - Nonlinear: H₀ + g|ψ|² → effective coupling between modes
  - The nonlinear self-energy depends on the mode's density profile
  - Different (n_s, n_θ) modes have different density profiles
  - → different self-energy shifts → degeneracy broken
  - → mode mixing → PMNS matrix

The coupling g is related to the fine structure constant:
  g ~ α × geometric_factor  (from EH: g = κ_EH × E²)
  From checkpoint 2026-03-16: GPE coupling is order-1 non-perturbative

We scan g to find where the mixing angles match experiment, then check
if that g value is consistent with the known physics.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import time

KAPPA = np.pi**2
BETA_E = np.sqrt(5.0 / 4.0)
ALPHA = 1.0 / 137.036


class TorusKnot:
    def __init__(self, pc, qc, kappa=KAPPA, beta=BETA_E):
        self.pc, self.qc = pc, qc
        self.kappa, self.beta = kappa, beta
        self.R, self.r = beta, beta / kappa
        self._build_arc_table()

    def _build_arc_table(self, N=2000):
        t = np.linspace(0, 2 * np.pi, N + 1)
        ds = np.array([self._ds_dt(ti) for ti in t])
        self._arc_cum = np.cumsum(0.5 * (ds[:-1] + ds[1:]) * np.diff(t))
        self._arc_cum = np.insert(self._arc_cum, 0, 0.0)
        self.L_total = self._arc_cum[-1]
        self._t_table = t

    def _ds_dt(self, t):
        theta = self.qc * t
        return np.sqrt(
            self.pc**2 * (self.kappa + np.cos(theta))**2 + self.qc**2
        ) * self.r

    def t_from_s(self, s_norm):
        return np.interp(s_norm * self.L_total, self._arc_cum, self._t_table)

    def curvature_normal(self, t):
        c, s = np.cos(self.qc * t), np.sin(self.qc * t)
        kc = self.kappa + c
        num2 = (self.pc**4 * kc**2 + self.qc**4
                + 2 * self.pc**2 * self.qc**2 * c * kc
                + self.pc**2 * self.qc**2 * s**2)
        denom = self.pc**2 * kc**2 + self.qc**2
        return np.sqrt(max(num2, 0)) / max(denom**1.5, 1e-30) / self.r


class KnotSolver:
    def __init__(self, knot, Ns=256, Nth=32):
        self.knot, self.Ns, self.Nth = knot, Ns, Nth
        self.ds = 1.0 / Ns
        self.dth = 2 * np.pi / Nth
        self.rL = knot.r / knot.L_total
        self._precompute()

    def _precompute(self):
        Ns, Nth, r = self.Ns, self.Nth, self.knot.r
        self.t_grid = np.array([
            self.knot.t_from_s(s)
            for s in np.linspace(0, 1, Ns, endpoint=False)
        ])
        self.th_grid = np.linspace(0, 2 * np.pi, Nth, endpoint=False)
        self.kappa_n = np.array([self.knot.curvature_normal(t) for t in self.t_grid])
        self.h_s_norm = np.zeros((Ns, Nth))
        for i in range(Ns):
            for j in range(Nth):
                v = r * self.kappa_n[i] * np.cos(self.th_grid[j])
                self.h_s_norm[i, j] = max(1.0 - v, 0.05)

    def build_hamiltonian(self):
        Ns, Nth = self.Ns, self.Nth
        N = Ns * Nth
        ds, dth, rL2 = self.ds, self.dth, self.rL**2
        rows, cols, vals = [], [], []
        def idx(i, j): return (i % Ns) * Nth + (j % Nth)
        for i in range(Ns):
            for j in range(Nth):
                n = idx(i, j)
                hs2_p = 0.5 * (self.h_s_norm[i, j]**2 + self.h_s_norm[(i+1)%Ns, j]**2)
                hs2_m = 0.5 * (self.h_s_norm[i, j]**2 + self.h_s_norm[(i-1)%Ns, j]**2)
                coeff_s = rL2 / ds**2
                coeff_th = 1.0 / dth**2
                rk = self.knot.r * self.kappa_n[i]
                total = coeff_s * (1/hs2_p + 1/hs2_m) + 2*coeff_th - rk**2/4
                rows.append(n); cols.append(n); vals.append(total)
                rows.append(n); cols.append(idx(i+1, j)); vals.append(-coeff_s/hs2_p)
                rows.append(n); cols.append(idx(i-1, j)); vals.append(-coeff_s/hs2_m)
                rows.append(n); cols.append(idx(i, j+1)); vals.append(-coeff_th)
                rows.append(n); cols.append(idx(i, j-1)); vals.append(-coeff_th)
        return csr_matrix((vals, (rows, cols)), shape=(N, N))

    def solve(self, n_modes=80):
        print(f"    Solving {self.Ns}×{self.Nth} = {self.Ns*self.Nth} DOF...", end='')
        t0 = time.time()
        H = self.build_hamiltonian()
        ev, evec = eigsh(H, k=n_modes, sigma=-0.1, which='LM',
                         maxiter=10000, tol=1e-8)
        order = np.argsort(ev)
        print(f" {time.time()-t0:.1f}s")
        return ev[order], evec[:, order]

    def decompose(self, eigvec):
        psi = eigvec.reshape(self.Ns, self.Nth)
        psi_fft = np.fft.fft2(psi)
        power = np.abs(psi_fft)**2
        power[0, 0] = 0
        idx = np.argmax(power)
        i, j = np.unravel_index(idx, power.shape)
        ns = i if i <= self.Ns//2 else i - self.Ns
        nth = j if j <= self.Nth//2 else j - self.Nth
        purity = power[i, j] / np.sum(power)
        return ns, nth, purity


def find_nth1_modes(solver, eigenvalues, eigenvectors, max_modes=6):
    modes = []
    seen_ns = set()
    for n in range(len(eigenvalues)):
        ns, nth, pur = solver.decompose(eigenvectors[:, n])
        if nth == 1 and ns not in seen_ns:
            modes.append({'idx': n, 'ns': ns, 'lam': eigenvalues[n], 'pur': pur})
            seen_ns.add(ns)
        if len(modes) >= max_modes:
            break
    return modes


def compute_nonlinear_coupling(solver, eigvecs, mode_indices):
    """Compute the nonlinear coupling matrix V_mn = ∫|ψ_m|²|ψ_n|² ds dθ.

    Also compute the four-wave mixing matrix:
      W_mnpq = ∫ ψ_m* ψ_n* ψ_p ψ_q ds dθ

    For the GPE nonlinearity g|ψ|²ψ, the first-order correction to the
    eigenvalue of mode m is:
      δλ_m = g × ∫|ψ_m|⁴ ds dθ  (self-interaction)

    The coupling between modes m and n is:
      V_mn = g × ∫|ψ_m|² |ψ_n|² ds dθ  (cross-density)

    And the off-diagonal four-wave mixing that actually rotates eigenstates:
      W_mmnn = g × ∫ ψ_m* ψ_m* ψ_n ψ_n ds dθ  (pair scattering)
    """
    N = len(mode_indices)
    Ns, Nth = solver.Ns, solver.Nth
    dA = solver.ds * solver.dth  # area element

    # Self-interaction: δλ_m = g × ∫|ψ_m|⁴ dA
    self_energy = np.zeros(N)
    for a, idx_a in enumerate(mode_indices):
        psi_a = eigvecs[:, idx_a]
        self_energy[a] = np.sum(np.abs(psi_a)**4) * dA

    # Cross-density: V_mn = g × ∫|ψ_m|² |ψ_n|² dA
    V_cross = np.zeros((N, N))
    for a in range(N):
        psi_a = eigvecs[:, mode_indices[a]]
        rho_a = np.abs(psi_a)**2
        for b in range(N):
            psi_b = eigvecs[:, mode_indices[b]]
            rho_b = np.abs(psi_b)**2
            V_cross[a, b] = np.sum(rho_a * rho_b) * dA

    # Four-wave mixing: W_mn = ∫ ψ_m* ψ_m* ψ_n ψ_n dA
    # This is the term that actually mixes modes (off-diagonal in mode space)
    W_mix = np.zeros((N, N), dtype=complex)
    for a in range(N):
        psi_a = eigvecs[:, mode_indices[a]]
        for b in range(N):
            psi_b = eigvecs[:, mode_indices[b]]
            W_mix[a, b] = np.sum(psi_a.conj()**2 * psi_b**2) * dA

    return self_energy, V_cross, W_mix


def build_effective_hamiltonian(eigenvalues, mode_indices, self_energy,
                                 V_cross, W_mix, g):
    """Build the effective Hamiltonian including GPE nonlinearity.

    H_eff = diag(λ_n) + g × (2V_cross - diag(V_cross_nn)) + g × W_mix

    The factor of 2 in V_cross comes from the Hartree term in the GPE:
      g(2∫|ψ_m|²|ψ_n|² - |ψ_m|⁴)
    which accounts for exchange.

    The W_mix term is the Fock (exchange) term that creates off-diagonal
    couplings between modes.
    """
    N = len(mode_indices)
    H_eff = np.zeros((N, N), dtype=complex)

    # Diagonal: linear eigenvalue + self-energy
    for a in range(N):
        H_eff[a, a] = eigenvalues[mode_indices[a]] + g * self_energy[a]

    # Cross-density (Hartree): shifts diagonal + adds off-diagonal
    H_eff += g * 2 * V_cross

    # Four-wave mixing (Fock): off-diagonal mode coupling
    H_eff += g * W_mix

    return H_eff


def extract_pmns(U_sq):
    """Extract mixing angles from |U|² matrix."""
    sin2_13 = np.clip(U_sq[0, 2], 0, 1)
    cos2_13 = 1.0 - sin2_13
    if cos2_13 > 1e-10:
        sin2_23 = np.clip(U_sq[1, 2] / cos2_13, 0, 1)
        sin2_12 = np.clip(U_sq[0, 1] / cos2_13, 0, 1)
    else:
        sin2_23 = sin2_12 = 0.5
    th12 = np.arcsin(np.sqrt(sin2_12)) * 180 / np.pi
    th23 = np.arcsin(np.sqrt(sin2_23)) * 180 / np.pi
    th13 = np.arcsin(np.sqrt(sin2_13)) * 180 / np.pi
    return th12, th23, th13


def main():
    print("=" * 100)
    print("NWT PMNS MATRIX: NONLINEAR (GPE) MODE MIXING")
    print("=" * 100)

    # ── Solve on trefoil ────────────────────────────────────────────
    print("\n  Trefoil T(2,3) eigenstates:")
    knot = TorusKnot(2, 3)
    solver = KnotSolver(knot, Ns=192, Nth=32)
    ev, evec = solver.solve(n_modes=120)

    # Find n_θ=1 modes (the lepton/neutrino sector)
    modes = find_nth1_modes(solver, ev, evec, max_modes=6)
    print(f"\n  n_θ=1 modes:")
    for rank, m in enumerate(modes):
        print(f"    #{rank}: n_s={m['ns']:+d}, λ={m['lam']:.6f}, pur={m['pur']:.3f}")

    if len(modes) < 3:
        print("  ERROR: fewer than 3 modes found")
        return

    # Use first 3 as our flavor/mass mixing sector
    mode_indices = [modes[i]['idx'] for i in range(3)]

    # ── Compute nonlinear coupling matrices ─────────────────────────
    print("\n  Computing nonlinear coupling matrices...")
    self_energy, V_cross, W_mix = compute_nonlinear_coupling(
        solver, evec, mode_indices)

    print(f"\n  Self-energy (∫|ψ|⁴ dA):")
    for i in range(3):
        print(f"    Mode n_s={modes[i]['ns']:+d}: {self_energy[i]:.6e}")

    print(f"\n  Cross-density V_mn:")
    for i in range(3):
        print(f"    ", end='')
        for j in range(3):
            print(f"  {V_cross[i,j]:.4e}", end='')
        print()

    print(f"\n  Four-wave mixing |W_mn|:")
    for i in range(3):
        print(f"    ", end='')
        for j in range(3):
            print(f"  {abs(W_mix[i,j]):.4e}", end='')
        print()

    # Key diagnostic: ratio of off-diagonal to diagonal
    off_diag_V = (V_cross[0,1] + V_cross[0,2] + V_cross[1,2]) / 3
    diag_V = (V_cross[0,0] + V_cross[1,1] + V_cross[2,2]) / 3
    print(f"\n  V off-diag/diag ratio: {off_diag_V/diag_V:.4f}")

    off_diag_W = (abs(W_mix[0,1]) + abs(W_mix[0,2]) + abs(W_mix[1,2])) / 3
    diag_W = (abs(W_mix[0,0]) + abs(W_mix[1,1]) + abs(W_mix[2,2])) / 3
    print(f"  W off-diag/diag ratio: {off_diag_W/diag_W:.4f}")

    # ── Scan over g values ──────────────────────────────────────────
    print("\n\n" + "━" * 100)
    print("  COUPLING CONSTANT SCAN")
    print("━" * 100)

    # Linear eigenvalue splittings
    dlam_01 = ev[mode_indices[1]] - ev[mode_indices[0]]
    dlam_02 = ev[mode_indices[2]] - ev[mode_indices[0]]
    print(f"\n  Linear splittings: Δλ₁₀={dlam_01:.6f}, Δλ₂₀={dlam_02:.6f}")

    # The mixing angles depend on the ratio g × V_offdiag / Δλ
    # Need g × V ~ Δλ for significant mixing
    g_scale = dlam_01 / max(off_diag_V, 1e-30)
    print(f"  Estimated g for order-1 mixing: {g_scale:.4f}")

    # Scan
    g_values = np.array([0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
                          10.0, 20.0, 50.0, 100.0, 200.0, 500.0])
    # Add values around the scale where mixing should appear
    g_values = np.sort(np.unique(np.concatenate([
        g_values,
        g_scale * np.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    ])))

    print(f"\n  {'g':>10s} {'θ₁₂':>8s} {'θ₂₃':>8s} {'θ₁₃':>8s} "
          f"{'Δm²₂₁':>10s} {'Δm²₃₂':>10s} {'ratio':>8s}")
    print(f"  {'─'*65}")

    best_score = 1e10
    best_g = 0
    best_angles = (0, 0, 0)

    for g in g_values:
        H_eff = build_effective_hamiltonian(ev, mode_indices, self_energy,
                                             V_cross, W_mix, g)
        # Diagonalize
        evals_nl, evecs_nl = np.linalg.eigh(H_eff)

        # The PMNS matrix: rows = flavor (old basis), columns = mass (new basis)
        # U = evecs_nl (columns are new eigenvectors in old basis)
        U = evecs_nl
        U_sq = np.abs(U)**2

        # Normalize rows (should already be, but ensure)
        for alpha in range(3):
            rs = np.sum(U_sq[alpha])
            if rs > 1e-30:
                U_sq[alpha] /= rs

        th12, th23, th13 = extract_pmns(U_sq)

        # Mass splittings (in eigenvalue units)
        dm21 = evals_nl[1] - evals_nl[0]
        dm32 = evals_nl[2] - evals_nl[1]
        ratio = dm32 / dm21 if abs(dm21) > 1e-15 else float('inf')

        # Score: weighted distance from experimental values
        score = ((th12 - 33.41)/33.41)**2 + ((th23 - 49.0)/49.0)**2 + ((th13 - 8.54)/8.54)**2

        if score < best_score:
            best_score = score
            best_g = g
            best_angles = (th12, th23, th13)
            best_U_sq = U_sq.copy()
            best_dm = (dm21, dm32, ratio)

        ratio_str = f"{ratio:.2f}" if np.isfinite(ratio) and abs(ratio) < 1e6 else "∞"
        print(f"  {g:10.3f} {th12:8.2f} {th23:8.2f} {th13:8.2f} "
              f"{dm21:10.4e} {dm32:10.4e} {ratio_str:>8s}")

    # ── Best result ─────────────────────────────────────────────────
    print(f"\n\n" + "━" * 100)
    print(f"  BEST FIT: g = {best_g:.4f}")
    print(f"━" * 100)

    print(f"\n  |U|² matrix:")
    labels = ['ν_e', 'ν_μ', 'ν_τ']
    print(f"  {'':>6s}    ν₁        ν₂        ν₃")
    for alpha in range(3):
        print(f"  {labels[alpha]:>6s}", end='')
        for i in range(3):
            print(f" {best_U_sq[alpha, i]:9.6f}", end='')
        print()

    th12, th23, th13 = best_angles
    print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  MIXING ANGLES (nonlinear GPE correction)                   │")
    print(f"  ├──────────┬──────────────┬──────────────┬────────────────────┤")
    print(f"  │          │  NWT (pred)  │  PDG (exp)   │    Error           │")
    print(f"  ├──────────┼──────────────┼──────────────┼────────────────────┤")
    print(f"  │  θ₁₂     │  {th12:8.2f}°   │   33.41°     │  {abs(th12-33.41)/33.41*100:6.1f}%          │")
    print(f"  │  θ₂₃     │  {th23:8.2f}°   │   49.0°      │  {abs(th23-49.0)/49.0*100:6.1f}%          │")
    print(f"  │  θ₁₃     │  {th13:8.2f}°   │    8.54°     │  {abs(th13-8.54)/8.54*100:6.1f}%          │")
    print(f"  └──────────┴──────────────┴──────────────┴────────────────────┘")

    dm21, dm32, ratio = best_dm
    print(f"\n  Mass splittings: Δm²₂₁ = {dm21:.4e}, Δm²₃₂ = {dm32:.4e}")
    print(f"  Ratio Δm²₃₂/Δm²₂₁ = {ratio:.2f}  (exp: ~33)")

    # ── Physical interpretation of g ────────────────────────────────
    print(f"\n\n" + "━" * 100)
    print(f"  PHYSICAL INTERPRETATION OF g = {best_g:.4f}")
    print(f"━" * 100)

    print(f"""
  The GPE coupling g appears in: iℏ∂ψ/∂t = (-ℏ²/2m*)∇²ψ + g|ψ|²ψ

  In NWT, g is related to the Euler-Heisenberg nonlinearity:
    g = κ_EH × E² × (geometric factor)

  Known values:
    α = 1/137.036 = {ALPHA:.6f}
    1/κ = r/R = 1/π² = {1/KAPPA:.6f}
    α/κ = {ALPHA*KAPPA:.6f}

  Best-fit g = {best_g:.4f}

  Candidate identifications:
    g = α           → {ALPHA:.6f}
    g = 1/κ         → {1/KAPPA:.6f}
    g = α × κ       → {ALPHA*KAPPA:.4f}
    g = α²          → {ALPHA**2:.6e}
    g = 2πα         → {2*np.pi*ALPHA:.6f}
    g = √α          → {np.sqrt(ALPHA):.6f}
    g = π²α         → {np.pi**2*ALPHA:.6f}
    g = 4πα         → {4*np.pi*ALPHA:.6f}
""")

    # ── Fine scan around best g ─────────────────────────────────────
    print("\n  Fine scan around best g:")
    g_fine = np.linspace(max(0.01, best_g * 0.5), best_g * 2.0, 50)
    print(f"  {'g':>10s} {'θ₁₂':>8s} {'θ₂₃':>8s} {'θ₁₃':>8s} {'score':>10s}")
    print(f"  {'─'*45}")

    best_score_fine = 1e10
    best_g_fine = best_g

    for g in g_fine:
        H_eff = build_effective_hamiltonian(ev, mode_indices, self_energy,
                                             V_cross, W_mix, g)
        evals_nl, evecs_nl = np.linalg.eigh(H_eff)
        U_sq = np.abs(evecs_nl)**2
        for alpha in range(3):
            rs = np.sum(U_sq[alpha])
            if rs > 1e-30: U_sq[alpha] /= rs
        th12, th23, th13 = extract_pmns(U_sq)
        score = ((th12 - 33.41)/33.41)**2 + ((th23 - 49.0)/49.0)**2 + ((th13 - 8.54)/8.54)**2

        if score < best_score_fine:
            best_score_fine = score
            best_g_fine = g
            best_angles_fine = (th12, th23, th13)

        print(f"  {g:10.4f} {th12:8.2f} {th23:8.2f} {th13:8.2f} {score:10.4f}")

    th12, th23, th13 = best_angles_fine
    print(f"\n  Fine-tuned best: g = {best_g_fine:.6f}")
    print(f"    θ₁₂ = {th12:.2f}°, θ₂₃ = {th23:.2f}°, θ₁₃ = {th13:.2f}°")


if __name__ == '__main__':
    main()
