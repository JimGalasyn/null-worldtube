#!/usr/bin/env python3
"""
NWT Level-2 Eigenstate Solver v2: Rescaled coordinates for mode coupling

Key change from v1: the poloidal coordinate is rescaled by κ so that
longitudinal and poloidal modes appear at comparable eigenvalues.

The physical tube has:
  - Longitudinal extent: L ~ 2πR × √(p²+q²/κ²) ~ 7-21 ξ
  - Poloidal circumference: 2πr = 2πR/κ ~ 0.71 ξ

The ratio L/(2πr) ~ 10-30, so poloidal modes are 100-900× higher
in k² than longitudinal modes.  The rescaling puts them on the
same footing, revealing the curvature coupling.

Coordinates: (s̃, θ̃) where
  s̃ = s/L ∈ [0, 1)     normalized arc length
  θ̃ = θ ∈ [0, 2π)      poloidal angle

The Helmholtz equation becomes:
  (1/L²) ∂²ψ/∂s̃² + (1/r²) ∂²ψ/∂θ² + curvature_coupling + k²ψ = 0

We define dimensionless eigenvalue: λ = k² × r²
so that q_mode = 1 gives λ ~ 1 (not ~100).

The curvature coupling term mixes s̃ and θ modes:
  V_curv(s̃, θ) = -κ_n(s̃) r cos(θ) / [1 - κ_n(s̃) r cos(θ)]

This potential is what generates the (p,q) mode mixing that determines
the interaction matrix elements at carrier crossings.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# ── Constants ─────────────────────────────────────────────────────────
KAPPA = np.pi**2
BETA_E = np.sqrt(5.0/4.0)


class TorusKnot:
    """Geometry of T(p,q) on torus with aspect ratio κ = R/r."""

    def __init__(self, pc, qc, kappa=KAPPA, beta=BETA_E):
        self.pc = pc
        self.qc = qc
        self.kappa = kappa
        self.beta = beta
        self.R = beta           # major radius in units of ξ
        self.r = beta / kappa   # tube radius in units of ξ

        # Precompute arc-length table for uniform parameterization
        self._build_arc_table()

    def _build_arc_table(self, N=2000):
        """Build cumulative arc-length table for uniform-s̃ sampling."""
        t = np.linspace(0, 2*np.pi, N+1)
        ds = np.array([self._ds_dt(ti) for ti in t])
        self._arc_cumulative = np.cumsum(0.5*(ds[:-1]+ds[1:]) * np.diff(t))
        self._arc_cumulative = np.insert(self._arc_cumulative, 0, 0.0)
        self.L_total = self._arc_cumulative[-1]
        self._t_table = t

    def _ds_dt(self, t):
        """Arc-length element |dγ/dt| at parameter t."""
        theta = self.qc * t
        return np.sqrt(self.pc**2 * (self.kappa + np.cos(theta))**2 +
                      self.qc**2) * self.r

    def t_from_s(self, s_norm):
        """Map normalized arc length s̃ ∈ [0,1) to parameter t."""
        s_target = s_norm * self.L_total
        return np.interp(s_target, self._arc_cumulative, self._t_table)

    def curvature_normal(self, t):
        """Normal curvature κ_n(t) of the knot centerline.

        For a torus knot T(p,q) on torus (R,r):
        κ_n = component of curvature in the direction toward the torus axis.

        Analytical expression for torus:
          κ_n(θ) = (κ + cos θ) / [(κ + cos θ)² × p² + q²] × (1/r) × (κ + cos θ)

        Simplified for torus knot:
          κ_n(t) ≈ p² (κ + cos(qt))² / [p²(κ+cos(qt))² + q²]^(3/2) × (1/r)

        Actually, the exact curvature of a torus knot is:
          κ(t) = √[p⁴(κ+c)² + q⁴ + 2p²q²c(κ+c) + p²q²s²] / [p²(κ+c)²+q²]^(3/2) × (1/r)
        where c = cos(qt), s = sin(qt).

        The NORMAL curvature (projection onto the outward direction) is what
        couples to the poloidal angle θ in the metric:
          κ_n = (κ + cos(qt)) / [p²(κ+cos(qt))² + q²] × p² / r
        """
        c = np.cos(self.qc * t)
        s = np.sin(self.qc * t)
        kc = self.kappa + c
        denom = (self.pc**2 * kc**2 + self.qc**2)

        # Full 3D curvature
        num2 = (self.pc**4 * kc**2 + self.qc**4 +
                2 * self.pc**2 * self.qc**2 * c * kc +
                self.pc**2 * self.qc**2 * s**2)
        kappa_full = np.sqrt(max(num2, 0)) / max(denom**1.5, 1e-30) / self.r

        return kappa_full

    def metric_along_path(self, t):
        """ds/dt along the knot centerline."""
        return self._ds_dt(t)

    def gpe_index(self, t, theta_tube):
        """GPE refractive index correction: n_GPE = √(1 + ξ²/ρ²).

        ρ is the distance from the torus axis. On the tube surface
        at angle θ_tube from the centerline:
          ρ(t, θ_tube) ≈ R + r cos(q_c t) + r_tube cos(θ_tube)

        For the tube surface, r_tube = r (we're ON the tube).
        The correction is largest at the inner edge (θ_tube = π)
        where ρ is smallest.
        """
        theta_knot = self.qc * t
        rho = self.R + self.r * np.cos(theta_knot)
        # ξ = 1 in our units (R is in units of ξ)
        n_gpe = np.sqrt(1.0 + 1.0 / (rho**2 + 0.01))  # regularized
        return n_gpe


class KnotEigensolverV2:
    """Helmholtz solver on knot tube surface with rescaled coordinates.

    Grid: (s̃, θ) with Ns points along the path and Nth around the tube.

    The eigenvalue equation in rescaled form:
      -[(r/L)² ∂²ψ/∂s̃² + ∂²ψ/∂θ² + V_curv(s̃,θ)ψ] = λ ψ

    where λ = k²r² and V_curv encodes curvature coupling.

    This puts longitudinal mode n and poloidal mode m on comparable
    footing: λ_n ~ (2πnr/L)² and λ_m ~ m².
    """

    def __init__(self, knot, Ns=256, Nth=64):
        self.knot = knot
        self.Ns = Ns
        self.Nth = Nth
        self.ds = 1.0 / Ns       # s̃ step
        self.dth = 2*np.pi / Nth  # θ step

        # Precompute s̃ grid → t grid → geometric quantities
        self.s_grid = np.linspace(0, 1, Ns, endpoint=False)
        self.th_grid = np.linspace(0, 2*np.pi, Nth, endpoint=False)
        self._precompute()

    def _precompute(self):
        """Precompute metric factors and curvature potential on grid."""
        Ns, Nth = self.Ns, self.Nth
        r = self.knot.r
        L = self.knot.L_total

        # Scale factor ratio
        self.rL = r / L  # r/L ~ 0.01-0.05 typically

        # Map s̃ → t
        self.t_grid = np.array([self.knot.t_from_s(s) for s in self.s_grid])

        # Curvature along path
        self.kappa_n = np.array([self.knot.curvature_normal(t) for t in self.t_grid])

        # Metric h_s factor: h_s(s̃, θ) = L × [1 - r κ_n(s̃) cos(θ)]
        # In rescaled units, the s̃-derivative prefactor is:
        #   (r/L)² / h_s_norm² where h_s_norm = 1 - r κ_n cos(θ)
        # This modifies the effective longitudinal wavenumber

        # Curvature potential V(s̃, θ) = r κ_n(s̃) cos(θ)
        # This is the key mode-coupling term
        self.V_curv = np.zeros((Ns, Nth))
        self.h_s_norm = np.zeros((Ns, Nth))
        for i in range(Ns):
            for j in range(Nth):
                v = r * self.kappa_n[i] * np.cos(self.th_grid[j])
                self.V_curv[i, j] = v
                self.h_s_norm[i, j] = max(1.0 - v, 0.05)  # clamp

        # GPE index (optional)
        self.n_gpe = np.zeros((Ns, Nth))
        for i in range(Ns):
            for j in range(Nth):
                self.n_gpe[i, j] = self.knot.gpe_index(self.t_grid[i], self.th_grid[j])

        print(f"  Geometry: L={L:.4f}ξ, r={r:.6f}ξ, r/L={self.rL:.6f}")
        print(f"  Curvature: κ_n min={self.kappa_n.min():.4f}, max={self.kappa_n.max():.4f} (1/ξ)")
        print(f"  V_curv range: [{self.V_curv.min():.6f}, {self.V_curv.max():.6f}]")
        print(f"  GPE n range: [{self.n_gpe.min():.4f}, {self.n_gpe.max():.4f}]")

    def build_hamiltonian(self, include_gpe=True):
        """Build the Hamiltonian H = -∇² + V_curv on the (s̃, θ) grid.

        Returns sparse matrix H such that H ψ = λ ψ where λ = k²r².

        The Laplacian in rescaled coords:
          ∇²ψ = (1/h_s²)(r/L)² ∂²ψ/∂s̃² + (1/r²) ∂²ψ/∂θ² (× r² for rescaling)
               = (r²/L²)(1/h_s²) ∂²ψ/∂s̃² + ∂²ψ/∂θ²

        With curvature correction to the metric:
          h_s(s̃,θ) = 1 - r κ_n(s̃) cos(θ)

        Full operator: -H ψ = λ ψ where
          H = -(rL)² (1/h_s²) ∂²/∂s̃² - ∂²/∂θ² + V_eff(s̃,θ)

        V_eff includes the da Costa potential from the curvature:
          V_daCosta = -(r κ_n)² / 4
        """
        Ns, Nth = self.Ns, self.Nth
        N = Ns * Nth
        ds = self.ds
        dth = self.dth
        rL2 = self.rL**2

        print(f"  Building Hamiltonian: {Ns}×{Nth} = {N} DOF, (r/L)²={rL2:.2e}...")
        t0 = time.time()

        rows = []
        cols = []
        vals = []

        def idx(i, j):
            return (i % Ns) * Nth + (j % Nth)

        for i in range(Ns):
            for j in range(Nth):
                n = idx(i, j)
                hs2 = self.h_s_norm[i, j]**2

                # s̃-direction: -(r/L)² (1/h_s²) ∂²ψ/∂s̃²
                # Using centered differences with metric at half-points
                hs2_p = 0.5 * (self.h_s_norm[i, j]**2 +
                               self.h_s_norm[(i+1)%Ns, j]**2)
                hs2_m = 0.5 * (self.h_s_norm[i, j]**2 +
                               self.h_s_norm[(i-1)%Ns, j]**2)

                coeff_s = rL2 / ds**2
                diag_s = coeff_s * (1.0/hs2_p + 1.0/hs2_m)
                off_sp = -coeff_s / hs2_p
                off_sm = -coeff_s / hs2_m

                # θ-direction: -∂²ψ/∂θ²
                coeff_th = 1.0 / dth**2
                diag_th = 2.0 * coeff_th
                off_thp = -coeff_th
                off_thm = -coeff_th

                # da Costa curvature potential
                rk = self.knot.r * self.kappa_n[i]
                V_dacosta = -rk**2 / 4.0

                # GPE correction (effective potential from index gradient)
                if include_gpe:
                    n_g = self.n_gpe[i, j]
                    V_gpe = (n_g**2 - 1)  # n²-1 acts as a potential
                else:
                    V_gpe = 0

                total_diag = diag_s + diag_th + V_dacosta + V_gpe

                # Assemble
                rows.append(n); cols.append(n); vals.append(total_diag)
                rows.append(n); cols.append(idx(i+1, j)); vals.append(off_sp)
                rows.append(n); cols.append(idx(i-1, j)); vals.append(off_sm)
                rows.append(n); cols.append(idx(i, j+1)); vals.append(off_thp)
                rows.append(n); cols.append(idx(i, j-1)); vals.append(off_thm)

        H = csr_matrix((vals, (rows, cols)), shape=(N, N))
        dt = time.time() - t0
        print(f"  Built in {dt:.2f}s, {H.nnz} nonzeros")
        return H

    def solve(self, n_modes=60, include_gpe=True):
        """Find lowest eigenstates of H ψ = λ ψ.

        Returns (eigenvalues λ, eigenvectors) sorted ascending.
        λ = k²r² (dimensionless).
        """
        H = self.build_hamiltonian(include_gpe=include_gpe)

        print(f"  Solving for {n_modes} eigenmodes...")
        t0 = time.time()
        eigenvalues, eigenvectors = eigsh(H, k=n_modes, sigma=-0.1,
                                          which='LM', maxiter=10000,
                                          tol=1e-8)
        dt = time.time() - t0
        print(f"  Solved in {dt:.2f}s")

        order = np.argsort(eigenvalues)
        return eigenvalues[order], eigenvectors[:, order]

    def decompose_mode(self, eigvec):
        """2D Fourier decomposition → (n_s, n_θ) quantum numbers.

        n_s = number of longitudinal half-waves along the knot path
        n_θ = poloidal winding number

        For the mode to correspond to a (p_mode, q_mode) torus knot mode,
        we need: n_s ∝ p_mode, n_θ = q_mode.
        """
        psi = eigvec.reshape(self.Ns, self.Nth)
        psi_fft = np.fft.fft2(psi)
        power = np.abs(psi_fft)**2
        power[0, 0] = 0  # remove DC

        # Top 3 peaks
        flat = power.flatten()
        top3 = np.argsort(flat)[-3:][::-1]
        peaks = []
        for idx in top3:
            i, j = np.unravel_index(idx, power.shape)
            ns = i if i <= self.Ns//2 else i - self.Ns
            nth = j if j <= self.Nth//2 else j - self.Nth
            peaks.append((ns, nth, flat[idx]))

        total_power = np.sum(power)
        purity = peaks[0][2] / total_power if total_power > 0 else 0

        return peaks[0][0], peaks[0][1], purity, peaks

    def crossing_overlap(self, eigvec_a, eigvec_b, t_crossing):
        """Compute wavefunction overlap at a carrier crossing point.

        For a Hopf link: the crossing is where the two components are closest.
        For a trefoil: the crossings are at specific t values.

        Returns: |⟨ψ_a|ψ_b⟩_crossing|² integrated over θ at fixed s̃_crossing.
        """
        # Map t_crossing to s̃
        s_cross = np.interp(t_crossing,
                           self.knot._t_table,
                           self.knot._arc_cumulative / self.knot.L_total)
        i_cross = int(s_cross * self.Ns) % self.Ns

        # Extract poloidal profiles at the crossing
        psi_a = eigvec_a.reshape(self.Ns, self.Nth)[i_cross, :]
        psi_b = eigvec_b.reshape(self.Ns, self.Nth)[i_cross, :]

        # Overlap integral: ∫ ψ_a* ψ_b dθ
        overlap = np.sum(np.conj(psi_a) * psi_b) * self.dth
        return abs(overlap)**2


def analyze_v2(pc, qc, kappa=KAPPA, beta=BETA_E, n_modes=60, label=None):
    """Run Level-2 v2 eigensolver for carrier (pc, qc)."""
    if label is None:
        label = f"T({pc},{qc})"

    print(f"\n{'━'*100}")
    print(f"  CARRIER: {label}  (p_c={pc}, q_c={qc}), κ={kappa:.4f}, β={beta:.6f}")
    print(f"{'━'*100}")

    knot = TorusKnot(pc, qc, kappa=kappa, beta=beta)

    # Adaptive resolution: more points for more complex carriers
    Ns = max(128, 64 * max(pc, qc))
    Nth = 64  # high poloidal resolution for mode coupling

    solver = KnotEigensolverV2(knot, Ns=Ns, Nth=Nth)
    eigenvalues, eigenvectors = solver.solve(n_modes=n_modes)

    # Convert λ = k²r² to physical k² and then to mass ratio
    r = knot.r
    k2_phys = eigenvalues / r**2  # physical k² in units of 1/ξ²

    # Reference: electron (2,1) mode has k²r² ~ p² × (2πr/L)² + q²
    # For the unknot: k²_e_r² ~ (2πr/L)² × 4 + 1 = 4(r/L)² × 4π² + 1 ≈ 1.001
    # Actually in our rescaled units, the electron eigenvalue is the
    # one with (n_s, n_θ) = (appropriate for (2,1) mode)

    print(f"\n  Eigenvalues λ = k²r² (rescaled):")
    print(f"  {'#':>3s} {'λ':>10s} {'k²(1/ξ²)':>12s} {'n_s':>5s} {'n_θ':>5s} "
          f"{'purity':>8s} {'p_eff':>6s} {'q_eff':>6s} {'p²+q²':>8s}")
    print(f"  {'─'*75}")

    mode_data = []
    for n in range(min(n_modes, len(eigenvalues))):
        lam = eigenvalues[n]
        k2 = k2_phys[n]

        ns, nth, purity, peaks = solver.decompose_mode(eigenvectors[:, n])

        # Map (n_s, n_θ) to effective (p, q):
        # n_s longitudinal nodes on carrier T(pc,qc) → p_eff = n_s * pc (roughly)
        # n_θ poloidal nodes → q_eff = n_θ * qc
        p_eff = abs(ns) * pc
        q_eff = abs(nth) * qc
        pq2 = p_eff**2 + q_eff**2

        mode_data.append({
            'n': n, 'lambda': lam, 'k2': k2,
            'ns': ns, 'nth': nth, 'purity': purity,
            'p_eff': p_eff, 'q_eff': q_eff, 'pq2': pq2,
            'peaks': peaks,
        })

        if lam < -0.5:
            continue  # skip unphysical

        print(f"  {n:3d} {lam:10.4f} {k2:12.4f} {ns:5d} {nth:5d} "
              f"{purity:8.4f} {p_eff:6d} {q_eff:6d} {pq2:8d}")

    # ── Identify key modes ────────────────────────────────────────────
    print(f"\n  KEY MODES (first occurrence of each (n_s, n_θ) pair):")
    seen = set()
    for md in mode_data:
        key = (abs(md['ns']), abs(md['nth']))
        if key in seen or md['lambda'] < -0.5:
            continue
        seen.add(key)
        ns, nth = md['ns'], md['nth']
        print(f"    Mode #{md['n']:3d}: (n_s={ns:+d}, n_θ={nth:+d}) → "
              f"eff({md['p_eff']},{md['q_eff']})  λ={md['lambda']:.4f}  "
              f"purity={md['purity']:.3f}")
        if len(seen) >= 20:
            break

    return eigenvalues, eigenvectors, solver, mode_data


def main():
    print("=" * 100)
    print("NWT LEVEL-2 EIGENSOLVER v2: Rescaled coordinates + poloidal resolution")
    print("=" * 100)

    results = {}

    # 1. Unknot baseline
    print("\n\n" + "=" * 100)
    print("CARRIER 1: Unknot (1,1) — baseline")
    print("=" * 100)
    ev1, vec1, sol1, md1 = analyze_v2(1, 1, label="Unknot (1,1)")
    results['unknot'] = (ev1, md1)

    # 2. Trefoil T(2,3)
    print("\n\n" + "=" * 100)
    print("CARRIER 2: Trefoil T(2,3) — baryon carrier")
    print("=" * 100)
    ev2, vec2, sol2, md2 = analyze_v2(2, 3, label="Trefoil T(2,3)")
    results['trefoil_23'] = (ev2, md2)

    # 3. Trefoil T(3,2)
    print("\n\n" + "=" * 100)
    print("CARRIER 3: Trefoil T(3,2) — alternate baryon carrier")
    print("=" * 100)
    ev3, vec3, sol3, md3 = analyze_v2(3, 2, label="Trefoil T(3,2)")
    results['trefoil_32'] = (ev3, md3)

    # 4. Hopf link component (effectively unknot with doubled path)
    print("\n\n" + "=" * 100)
    print("CARRIER 4: Hopf(2) single component — meson/EWK carrier")
    print("=" * 100)
    ev4, vec4, sol4, md4 = analyze_v2(2, 2, label="Hopf(2)")
    results['hopf2'] = (ev4, md4)

    # ── Mode coupling analysis ────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("MODE COUPLING: Curvature-induced (n_s, n_θ) mixing")
    print("=" * 100)
    print(f"""
  The curvature coupling V_curv = r κ_n(s) cos(θ) mixes
  longitudinal mode n_s with poloidal modes n_θ = ±1.

  Pure modes have definite (n_s, n_θ).
  Mixed modes show multiple peaks in the Fourier spectrum.

  Look for modes where purity < 0.3 — these have significant
  (n_s, n_θ) mixing, which is where the interaction coupling
  constants come from.
""")

    for name, (ev, md) in results.items():
        mixed = [m for m in md if 0 < m['purity'] < 0.30 and m['lambda'] > 0]
        print(f"\n  {name}: {len(mixed)} mixed modes (purity < 0.30)")
        for m in mixed[:8]:
            peaks_str = ', '.join(f"({p[0]:+d},{p[1]:+d}):{p[2]:.0f}"
                                 for p in m['peaks'][:3])
            print(f"    #{m['n']:3d} λ={m['lambda']:8.4f} purity={m['purity']:.3f} "
                  f"peaks=[{peaks_str}]")

    # ── Eigenvalue ratio comparison ───────────────────────────────────
    print("\n\n" + "=" * 100)
    print("EIGENVALUE RATIOS: Rescaled λ = k²r²")
    print("=" * 100)

    print(f"\n  {'#':>3s}", end="")
    for name in results:
        print(f" {name:>14s}", end="")
    print()
    print(f"  {'─'*70}")

    for n in range(20):
        print(f"  {n:3d}", end="")
        for name, (ev, md) in results.items():
            if n < len(ev) and ev[n] > -0.5:
                print(f" {ev[n]:14.4f}", end="")
            else:
                print(f" {'---':>14s}", end="")
        print()

    # ── Crossing overlap for trefoil ──────────────────────────────────
    print("\n\n" + "=" * 100)
    print("CROSSING OVERLAP INTEGRALS (trefoil T(2,3))")
    print("=" * 100)
    print(f"""
  The trefoil T(2,3) has 3 crossings.  The self-crossing points
  occur where the knot path comes closest to itself.

  For T(2,3): crossings at t ≈ 0, 2π/3, 4π/3 (equally spaced
  by the 3-fold symmetry of the trefoil).

  The overlap integral |⟨ψ_m|ψ_n⟩_crossing|² gives the coupling
  matrix element for topology change at that crossing.
""")

    # Crossing at t=0 for trefoil T(2,3)
    t_cross = 0.0
    print(f"  Crossing overlap at t = {t_cross:.2f}:")
    print(f"  {'(m,n)':>10s} {'|⟨m|n⟩|²':>12s}")
    for i in range(min(8, len(ev2))):
        for j in range(i+1, min(8, len(ev2))):
            if ev2[i] < -0.5 or ev2[j] < -0.5:
                continue
            overlap = sol2.crossing_overlap(vec2[:, i], vec2[:, j], t_cross)
            if overlap > 1e-4:
                print(f"  ({i},{j}){'':<5s} {overlap:12.6f}")


if __name__ == '__main__':
    main()
