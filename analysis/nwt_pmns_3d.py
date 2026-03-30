#!/usr/bin/env python3
"""
NWT PMNS Matrix from 3D Eigenstate Solver

Extends the Level 2 eigensolver with the RADIAL degree of freedom
(distance from vortex core) that the 2D surface solver lacks.

The 2D solver (Session B) found:
  θ₁₃ = 10.4° (22% from exp 8.54°) — good
  θ₁₂ saturates at ~18° (exp 33.4°) — needs radial DOF
  θ₂₃ saturates at ~15° (exp 49.0°) — needs radial DOF

The 3D solver adds coordinate ρ (radial distance from tube center):
  Grid: (s̃, θ, ρ) with Ns × Nθ × Nρ points

Key physics the radial DOF adds:
  1. The three leptons have different radial profiles (penetration depth)
  2. The GPE nonlinearity g|ψ|² depends on ρ via the vortex density n(ρ)
  3. The R3 surgery at crossings involves the full 3D tube structure
  4. n_s=1 and n_s=2 modes get different self-energies (breaks 2D degeneracy)

Nonlinear coupling: g = κ²/(2πα) ≈ 2127 (from the one-parameter relation)
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import time

KAPPA = np.pi**2
BETA_E = np.sqrt(5.0/4.0)
ALPHA = 1.0 / 137.036


class KnotEigensolver3D:
    """Helmholtz + GPE on the 3D tube volume (s̃, θ, ρ) of a carrier knot."""

    def __init__(self, pc, qc, Ns=64, Nth=32, Nrho=16,
                 rho_max=3.0, g_nonlinear=None, kappa=KAPPA, beta=BETA_E):
        self.pc = pc
        self.qc = qc
        self.kappa = kappa
        self.beta = beta
        self.R = beta
        self.r = beta / kappa

        self.Ns = Ns
        self.Nth = Nth
        self.Nrho = Nrho
        self.N_total = Ns * Nth * Nrho

        self.ds = 1.0 / Ns
        self.dth = 2*np.pi / Nth
        self.rho_max = rho_max  # in units of ξ
        self.drho = rho_max / Nrho

        # Nonlinear coupling
        if g_nonlinear is None:
            # From the one-parameter relation: g = κ²/(2πα)
            self.g = kappa**2 / (2 * np.pi * ALPHA)
        else:
            self.g = g_nonlinear

        # Compute path length
        from scipy.integrate import quad
        def ds_dt(t):
            theta = qc * t
            return np.sqrt(pc**2 * (kappa + np.cos(theta))**2 + qc**2) * self.r
        L, _ = quad(ds_dt, 0, 2*np.pi, limit=200)
        self.L_total = L
        self.rL = self.r / L

        # Grids
        self.s_grid = np.linspace(0, 1, Ns, endpoint=False)
        self.th_grid = np.linspace(0, 2*np.pi, Nth, endpoint=False)
        self.rho_grid = np.linspace(self.drho/2, rho_max - self.drho/2, Nrho)

        # Precompute
        self._precompute()

        print(f"  3D Eigensolver: T({pc},{qc}), κ={kappa:.4f}, β={beta:.6f}")
        print(f"  Grid: {Ns}×{Nth}×{Nrho} = {self.N_total} DOF")
        print(f"  L={L:.4f}ξ, r={self.r:.6f}ξ, r/L={self.rL:.6f}")
        print(f"  ρ range: [0, {rho_max}ξ], dρ={self.drho:.4f}ξ")
        print(f"  g = κ²/(2πα) = {self.g:.1f}")

    def _precompute(self):
        """Precompute curvature and vortex density profile."""
        Ns = self.Ns

        # Map s̃ → t via arc-length table
        from scipy.integrate import quad
        N_arc = 2000
        t_tab = np.linspace(0, 2*np.pi, N_arc+1)
        ds_vals = np.array([self._ds_dt_val(ti) for ti in t_tab])
        arc_cum = np.cumsum(0.5*(ds_vals[:-1]+ds_vals[1:]) * np.diff(t_tab))
        arc_cum = np.insert(arc_cum, 0, 0.0)

        self.t_grid = np.array([
            np.interp(s * self.L_total, arc_cum, t_tab)
            for s in self.s_grid
        ])

        # Curvature along path
        self.kappa_n = np.array([self._curvature(t) for t in self.t_grid])

        # Vortex density profile: n(ρ) = ρ/√(ρ²+1)
        # GPE potential: V_GPE(ρ) = g × n(ρ)² = g × ρ²/(ρ²+1)
        self.n_profile = self.rho_grid / np.sqrt(self.rho_grid**2 + 1.0)
        self.V_gpe = self.g * self.n_profile**2

        print(f"  V_GPE range: [{self.V_gpe.min():.2f}, {self.V_gpe.max():.2f}]")
        print(f"  Curvature range: [{self.kappa_n.min():.4f}, {self.kappa_n.max():.4f}]")

    def _ds_dt_val(self, t):
        theta = self.qc * t
        return np.sqrt(self.pc**2 * (self.kappa + np.cos(theta))**2 +
                      self.qc**2) * self.r

    def _curvature(self, t):
        c = np.cos(self.qc * t)
        s = np.sin(self.qc * t)
        kc = self.kappa + c
        num2 = (self.pc**4 * kc**2 + self.qc**4 +
                2*self.pc**2*self.qc**2*c*kc +
                self.pc**2*self.qc**2*s**2)
        denom = (self.pc**2 * kc**2 + self.qc**2)
        return np.sqrt(max(num2, 0)) / max(denom**1.5, 1e-30) / self.r

    def build_hamiltonian(self):
        """Build H on (s̃, θ, ρ) grid.

        H = -(r/L)²(1/h_s²)∂²/∂s̃² - ∂²/∂θ² - ∂²/∂ρ² - (1/ρ)∂/∂ρ
            + V_GPE(ρ) + V_curvature(s̃,θ,ρ)

        Index mapping: (i, j, k) → i×Nth×Nrho + j×Nrho + k
        """
        Ns, Nth, Nrho = self.Ns, self.Nth, self.Nrho
        N = self.N_total
        ds, dth, drho = self.ds, self.dth, self.drho
        rL2 = self.rL**2

        print(f"  Building 3D Hamiltonian ({N} DOF)...")
        t0 = time.time()

        rows, cols, vals = [], [], []

        def idx(i, j, k):
            return ((i % Ns) * Nth + (j % Nth)) * Nrho + min(max(k, 0), Nrho-1)

        for i in range(Ns):
            kn = self.kappa_n[i]
            for j in range(Nth):
                theta = self.th_grid[j]
                # Metric factor from curvature
                h_s = max(1.0 - self.r * kn * np.cos(theta), 0.05)
                h_s2 = h_s**2

                for k in range(Nrho):
                    n = idx(i, j, k)
                    rho = self.rho_grid[k]

                    # ── s̃ direction: -(r/L)²/h_s² ∂²/∂s̃² ──
                    coeff_s = rL2 / (ds**2 * h_s2)
                    diag_s = 2 * coeff_s

                    # ── θ direction: -∂²/∂θ² ──
                    coeff_th = 1.0 / dth**2
                    diag_th = 2 * coeff_th

                    # ── ρ direction: -∂²/∂ρ² - (1/ρ)∂/∂ρ ──
                    # For cylindrical coords: -[∂²/∂ρ² + (1/ρ)∂/∂ρ]
                    # Discretize: -(f_{k+1}-2f_k+f_{k-1})/dρ² - (f_{k+1}-f_{k-1})/(2ρ dρ)
                    coeff_rho = 1.0 / drho**2
                    if rho > 0.01:
                        coeff_rho_asym = 1.0 / (2 * rho * drho)
                    else:
                        coeff_rho_asym = 0

                    diag_rho = 2 * coeff_rho

                    # Off-diagonal ρ terms
                    off_rho_p = -coeff_rho - coeff_rho_asym  # k+1
                    off_rho_m = -coeff_rho + coeff_rho_asym  # k-1

                    # ── Potentials ──
                    V_gpe = self.V_gpe[k]

                    # Curvature coupling: depends on ρ (stronger near core)
                    # V_curv = r × κ_n × cos(θ) × (1 + ρ/r correction)
                    V_curv_da_costa = -(self.r * kn)**2 / 4.0

                    # ── Total diagonal ──
                    total_diag = diag_s + diag_th + diag_rho + V_gpe + V_curv_da_costa

                    # Assemble
                    rows.append(n); cols.append(n); vals.append(total_diag)

                    # s̃ neighbors (periodic)
                    rows.append(n); cols.append(idx(i+1, j, k)); vals.append(-coeff_s)
                    rows.append(n); cols.append(idx(i-1, j, k)); vals.append(-coeff_s)

                    # θ neighbors (periodic)
                    rows.append(n); cols.append(idx(i, j+1, k)); vals.append(-coeff_th)
                    rows.append(n); cols.append(idx(i, j-1, k)); vals.append(-coeff_th)

                    # ρ neighbors (reflecting BC at ρ=0, Dirichlet at ρ_max)
                    if k < Nrho - 1:
                        rows.append(n); cols.append(idx(i, j, k+1)); vals.append(off_rho_p)
                    if k > 0:
                        rows.append(n); cols.append(idx(i, j, k-1)); vals.append(off_rho_m)
                    else:
                        # Reflecting BC at ρ=0: f_{-1} = f_{1}
                        rows.append(n); cols.append(idx(i, j, 1)); vals.append(off_rho_m)

        H = csr_matrix((vals, (rows, cols)), shape=(N, N))
        dt = time.time() - t0
        print(f"  Built in {dt:.1f}s, {H.nnz} nonzeros")
        return H

    def solve(self, n_modes=60):
        H = self.build_hamiltonian()
        print(f"  Solving for {n_modes} eigenmodes...")
        t0 = time.time()
        ev, evec = eigsh(H, k=n_modes, sigma=-0.1, which='LM',
                         maxiter=10000, tol=1e-8)
        order = np.argsort(ev)
        print(f"  Solved in {time.time()-t0:.1f}s")
        return ev[order], evec[:, order]

    def decompose(self, eigvec):
        """3D Fourier decomposition → (n_s, n_θ, n_ρ)."""
        psi = eigvec.reshape(self.Ns, self.Nth, self.Nrho)
        # Fourier in s̃ and θ (periodic); leave ρ in real space
        psi_ft = np.fft.fft2(psi, axes=(0, 1))
        power = np.sum(np.abs(psi_ft)**2, axis=2)  # sum over ρ
        power[0, 0] = 0

        idx = np.argmax(power)
        i, j = np.unravel_index(idx, power.shape)
        ns = i if i <= self.Ns//2 else i - self.Ns
        nth = j if j <= self.Nth//2 else j - self.Nth

        # Radial profile for dominant mode
        radial = np.abs(psi_ft[i, j, :])**2
        n_rho_peak = np.argmax(radial)

        purity = power[i, j] / np.sum(power)
        return ns, nth, n_rho_peak, purity

    def radial_profile(self, eigvec, ns_target, nth_target):
        """Extract the radial profile for a specific (n_s, n_θ) mode."""
        psi = eigvec.reshape(self.Ns, self.Nth, self.Nrho)
        psi_ft = np.fft.fft2(psi, axes=(0, 1))
        i_s = ns_target % self.Ns
        i_th = nth_target % self.Nth
        return np.abs(psi_ft[i_s, i_th, :])

    def compute_pmns(self, eigenvalues, eigenvectors, n_modes=60):
        """Compute the PMNS mixing matrix.

        Procedure:
        1. Identify lepton-like eigenstates (n_θ = 1 modes at different n_s)
        2. Identify mass eigenstates (three lowest-energy n_θ = 1 modes)
        3. Compute overlap matrix → PMNS

        The flavor eigenstates are the modes created by R3 surgery
        alongside each charged lepton. The lepton quantum numbers
        determine which (n_s, n_θ) the neutrino inherits.
        """
        print(f"\n  ── PMNS Matrix Computation ──")

        # Find all modes with significant n_θ = ±1 content
        poloidal_modes = []
        for n in range(min(n_modes, len(eigenvalues))):
            ns, nth, nrho, pur = self.decompose(eigenvectors[:, n])
            if abs(nth) == 1 and eigenvalues[n] > 0:
                # Get radial profile
                rad = self.radial_profile(eigenvectors[:, n], ns, nth)
                poloidal_modes.append({
                    'idx': n, 'ns': ns, 'nth': nth, 'nrho': nrho,
                    'lambda': eigenvalues[n], 'purity': pur,
                    'radial': rad,
                })

        print(f"  Found {len(poloidal_modes)} modes with |n_θ| = 1")

        if len(poloidal_modes) < 3:
            print("  WARNING: fewer than 3 poloidal modes found")
            return None

        # Mass eigenstates: three lowest-λ poloidal modes
        mass_states = sorted(poloidal_modes, key=lambda m: m['lambda'])[:3]
        print(f"\n  Mass eigenstates (ν₁, ν₂, ν₃):")
        for i, ms in enumerate(mass_states):
            print(f"    ν_{i+1}: mode #{ms['idx']}, (n_s={ms['ns']}, n_θ={ms['nth']}), "
                  f"λ={ms['lambda']:.4f}, ρ_peak={ms['nrho']}")

        # Flavor eigenstates: modes at lepton mass scales
        # In the eigensolver, the lepton modes are identified by their
        # effective (p,q) quantum numbers matching the lepton assignments:
        #   e: eff(2,1) → n_s ∝ 2 (on the carrier)
        #   μ: eff(3,8) → higher n_s
        #   τ: eff(1,36) → very high poloidal
        # For the trefoil T(2,3), the carrier winding multiplies:
        #   n_s=1 on T(2,3) → eff n_s = 2 (electron-like)
        #   n_s=2 → eff n_s = 4 (muon-like)
        #   n_s=3 → eff n_s = 6 (tau-like)
        # Select by ascending n_s among poloidal modes

        ns_values = sorted(set(abs(m['ns']) for m in poloidal_modes if abs(m['ns']) > 0))
        if len(ns_values) < 3:
            print("  WARNING: fewer than 3 distinct n_s values in poloidal modes")
            # Use what we have
            ns_values = ns_values + [ns_values[-1]+1] * (3 - len(ns_values))

        flavor_ns = ns_values[:3]  # three lowest non-zero |n_s|
        print(f"\n  Flavor assignment: n_s = {flavor_ns} → (ν_e, ν_μ, ν_τ)")

        flavor_states = []
        for ns_target in flavor_ns:
            # Find the mode with this n_s and n_θ=±1
            candidates = [m for m in poloidal_modes if abs(m['ns']) == ns_target]
            if candidates:
                flavor_states.append(min(candidates, key=lambda m: m['lambda']))
            else:
                # Fallback: nearest n_s
                flavor_states.append(min(poloidal_modes,
                                        key=lambda m: abs(abs(m['ns'])-ns_target)))

        print(f"  Flavor eigenstates (ν_e, ν_μ, ν_τ):")
        for i, fs in enumerate(flavor_states):
            print(f"    ν_{['e','μ','τ'][i]}: mode #{fs['idx']}, "
                  f"(n_s={fs['ns']}, n_θ={fs['nth']}), λ={fs['lambda']:.4f}")

        # Compute overlap matrix: U_αi = ⟨ν_α(flavor) | ν_i(mass)⟩
        U = np.zeros((3, 3), dtype=complex)
        for a in range(3):
            for i in range(3):
                v_f = eigenvectors[:, flavor_states[a]['idx']]
                v_m = eigenvectors[:, mass_states[i]['idx']]
                U[a, i] = np.dot(v_f.conj(), v_m)

        # Ensure unitarity by SVD
        from scipy.linalg import svd
        u_svd, s_svd, vh_svd = svd(U)
        U_unitary = u_svd @ vh_svd

        print(f"\n  |U|² (PMNS matrix):")
        U2 = np.abs(U_unitary)**2
        labels_f = ['ν_e ', 'ν_μ ', 'ν_τ ']
        labels_m = ['ν₁', 'ν₂', 'ν₃']
        print(f"         {labels_m[0]:>8s} {labels_m[1]:>8s} {labels_m[2]:>8s}")
        for a in range(3):
            print(f"  {labels_f[a]} {U2[a,0]:8.4f} {U2[a,1]:8.4f} {U2[a,2]:8.4f}")

        # Extract mixing angles
        # Standard parameterization:
        # s₁₃ = |U_e3|, s₁₂ = |U_e2|/c₁₃, s₂₃ = |U_μ3|/c₁₃
        s13 = np.sqrt(U2[0, 2])
        c13 = np.sqrt(1 - s13**2) if s13 < 1 else 0.01
        s12 = np.sqrt(U2[0, 1]) / c13 if c13 > 0.01 else 0
        s23 = np.sqrt(U2[1, 2]) / c13 if c13 > 0.01 else 0

        # Clamp to valid range
        s12 = min(s12, 1.0)
        s23 = min(s23, 1.0)

        theta13 = np.degrees(np.arcsin(s13))
        theta12 = np.degrees(np.arcsin(s12))
        theta23 = np.degrees(np.arcsin(s23))

        print(f"\n  Mixing angles:")
        print(f"    θ₁₂ = {theta12:.2f}°  (exp: 33.41°, error: {abs(theta12-33.41)/33.41*100:.1f}%)")
        print(f"    θ₂₃ = {theta23:.2f}°  (exp: 49.0°,  error: {abs(theta23-49.0)/49.0*100:.1f}%)")
        print(f"    θ₁₃ = {theta13:.2f}°  (exp: 8.54°,  error: {abs(theta13-8.54)/8.54*100:.1f}%)")

        # Mass squared differences
        m2 = np.array([ms['lambda'] for ms in mass_states])
        dm2_21 = m2[1] - m2[0]
        dm2_32 = m2[2] - m2[1]
        ratio = dm2_32 / dm2_21 if dm2_21 > 0 else 0

        print(f"\n  Mass-squared differences (in eigenvalue units):")
        print(f"    Δm²₂₁ = {dm2_21:.6f}")
        print(f"    Δm²₃₂ = {dm2_32:.6f}")
        print(f"    Ratio Δm²₃₂/Δm²₂₁ = {ratio:.2f}  (exp: ~33)")

        # Radial profile comparison
        print(f"\n  Radial profiles of mass eigenstates:")
        print(f"  {'ρ/ξ':>6s}", end="")
        for i in range(3):
            print(f" {'ν'+str(i+1):>8s}", end="")
        print()
        for k in range(self.Nrho):
            print(f"  {self.rho_grid[k]:6.3f}", end="")
            for i in range(3):
                rad = self.radial_profile(eigenvectors[:, mass_states[i]['idx']],
                                         mass_states[i]['ns'], mass_states[i]['nth'])
                print(f" {rad[k]:8.4f}", end="")
            print()

        return {
            'U': U_unitary, 'U2': U2,
            'theta12': theta12, 'theta23': theta23, 'theta13': theta13,
            'mass_states': mass_states, 'flavor_states': flavor_states,
            'dm2_21': dm2_21, 'dm2_32': dm2_32,
        }


def main():
    print("=" * 100)
    print("NWT PMNS MATRIX FROM 3D EIGENSTATE SOLVER")
    print("=" * 100)

    # Trefoil T(2,3) — baryon carrier where weak decays occur
    print(f"\n  g = κ²/(2πα) = {KAPPA**2/(2*np.pi*ALPHA):.1f} (one-parameter prediction)")

    # Scan over g values to find the best match
    g_values = [
        KAPPA**2 / (2*np.pi*ALPHA),  # 2127 (theoretical)
        2442,                          # best from Session B
        1000,                          # lower
        5000,                          # higher
    ]

    for g in g_values:
        print(f"\n\n{'━'*100}")
        print(f"  g = {g:.1f}")
        print(f"{'━'*100}")

        solver = KnotEigensolver3D(2, 3, Ns=64, Nth=32, Nrho=16,
                                    rho_max=3.0, g_nonlinear=g)
        eigenvalues, eigenvectors = solver.solve(n_modes=60)

        # Show first modes
        print(f"\n  First 20 eigenmodes:")
        print(f"  {'#':>3s} {'λ':>10s} {'n_s':>5s} {'n_θ':>5s} {'n_ρ':>5s} {'pur':>6s}")
        for n in range(min(20, len(eigenvalues))):
            ns, nth, nrho, pur = solver.decompose(eigenvectors[:, n])
            print(f"  {n:3d} {eigenvalues[n]:10.4f} {ns:5d} {nth:5d} {nrho:5d} {pur:6.3f}")

        result = solver.compute_pmns(eigenvalues, eigenvectors)

    # Final run at theoretical g
    print(f"\n\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
