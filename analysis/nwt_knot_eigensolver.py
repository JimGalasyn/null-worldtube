#!/usr/bin/env python3
"""
NWT Level-2 Eigenstate Solver: EM modes on knotted carrier tubes

Solves the Helmholtz equation ∇²ψ + k²ψ = 0 on the 2D surface of a
torus-knot tube, parameterized by (s, θ) where s is arc length along
the carrier knot and θ is the poloidal angle around the tube cross-section.

The metric on the tube surface of a torus knot T(p_c, q_c) embedded
on a torus with aspect ratio κ = R/r is:

  ds² = h_s² ds̃² + h_θ² dθ²

where:
  h_s(s,θ) = 1 - (r/ρ(s))κ_n(s)cos(θ)   (along-path scale factor)
  h_θ(s,θ) = r                              (poloidal scale factor)
  κ_n(s)   = normal curvature of the knot centerline at arc position s
  ρ(s)     = R + r cos(θ_knot(s))           (distance from torus axis)

The eigenstates are the natural modes of the combined mode+carrier system.

Outputs:
  - Eigenfrequencies (→ mass spectrum)
  - Eigenstates ψ(s, θ) (→ wavefunctions at crossings → couplings)
  - Mode quantum numbers (p_mode, q_mode) from Fourier decomposition
"""

import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Physical constants ────────────────────────────────────────────────
KAPPA_TORUS = np.pi**2          # aspect ratio R/r
BETA_E = np.sqrt(5.0/4.0)      # electron β = √5/2


# ══════════════════════════════════════════════════════════════════════
# TORUS KNOT GEOMETRY
# ══════════════════════════════════════════════════════════════════════

class TorusKnotGeometry:
    """Geometry of a torus knot T(p,q) on a torus with aspect ratio κ."""

    def __init__(self, p_carrier, q_carrier, kappa=KAPPA_TORUS, beta=BETA_E):
        self.pc = p_carrier
        self.qc = q_carrier
        self.kappa = kappa
        self.beta = beta  # β = R/ξ

        # Physical dimensions (in units of ξ)
        self.R = beta          # major radius / ξ
        self.r = beta / kappa  # tube radius / ξ

        # Knot parameters
        self.is_knot = (p_carrier > 1 and q_carrier > 1 and
                        np.gcd(p_carrier, q_carrier) == 1)
        self.is_link = (p_carrier == q_carrier and p_carrier > 1)
        self.n_components = p_carrier if self.is_link else 1

        # Precompute arc length
        self._compute_arc_length()

    def _compute_arc_length(self):
        """Total arc length of the carrier knot (one component if link)."""
        def ds_dt(t):
            theta = self.qc * t
            return np.sqrt(self.pc**2 * (self.kappa + np.cos(theta))**2 +
                          self.qc**2) * self.r
        L, _ = quad(ds_dt, 0, 2*np.pi, limit=200)
        self.L_total = L  # total path length in units of ξ

    def centerline(self, t, component=0):
        """Knot centerline position in Cartesian coords (units of ξ).
        t ∈ [0, 2π) parameterizes the knot.
        For links, component selects which ring (0-indexed).
        """
        phi = self.pc * t + 2*np.pi * component / self.n_components
        theta = self.qc * t
        rho = self.R + self.r * np.cos(theta)  # actually this is wrong
        # The centerline follows the TORUS surface at (R, theta_torus)
        # For T(p,q): φ_torus = p*t, θ_torus = q*t
        x = (self.R + self.r * np.cos(theta)) * np.cos(phi)
        y = (self.R + self.r * np.cos(theta)) * np.sin(phi)
        z = self.r * np.sin(theta)
        return np.array([x, y, z])

    def ds_dt(self, t):
        """Arc length element ds/dt along the knot."""
        theta = self.qc * t
        return np.sqrt(self.pc**2 * (self.kappa + np.cos(theta))**2 +
                      self.qc**2) * self.r

    def curvature(self, t):
        """Curvature of the knot centerline at parameter t.

        For a torus knot T(p,q) on torus (R,r):
        κ(t) = |γ''(t)| / |γ'(t)|²  (Frenet curvature)

        Uses numerical differentiation for generality.
        """
        dt = 1e-6
        g0 = self.centerline(t)
        gp = self.centerline(t + dt)
        gm = self.centerline(t - dt)

        gp1 = (gp - gm) / (2*dt)       # first derivative
        gp2 = (gp - 2*g0 + gm) / dt**2  # second derivative

        cross = np.cross(gp1, gp2)
        kappa_val = np.linalg.norm(cross) / np.linalg.norm(gp1)**3
        return kappa_val

    def metric_hs(self, t, theta_tube):
        """Along-path metric factor h_s at knot parameter t and tube angle θ.

        h_s = |∂r/∂t| evaluated on the tube surface at poloidal angle θ_tube.
        For a tube of radius r_tube << r (the torus minor radius):
          h_s ≈ ds/dt × (1 - r_tube × κ_n(t) × cos(θ_tube))

        where κ_n is the normal curvature projected onto the tube normal.
        For simplicity, use the full torus metric:
          h_s = (κ + cos(q*t + ... )) × ... (complicated for general knots)

        Here we use the exact Lamé coefficient for the torus:
          h_φ = R + r cos(θ) for the toroidal direction
          h_θ = r for the poloidal direction

        On the knot path T(p,q), the arc element mixes both:
          ds² = (p² h_φ² + q² h_θ²) dt²
        """
        theta_knot = self.qc * t  # poloidal position on the centerline
        # On the tube surface at angle θ_tube from the centerline,
        # the effective poloidal angle is θ_knot + correction
        # For a thin tube (r_tube << r), the tube is at θ = θ_knot
        # and the metric varies with θ_tube through the curvature coupling

        # Simplified: use the torus metric at the centerline position
        h_phi = self.R + self.r * np.cos(theta_knot)
        h_theta = self.r

        # Along-path metric: combines toroidal and poloidal components
        ds = np.sqrt(self.pc**2 * h_phi**2 + self.qc**2 * h_theta**2)

        # Curvature correction for tube position (leading order)
        # The tube has radius r_tube; we parameterize points on it by θ_tube
        r_tube = self.r  # in our model, tube radius = torus minor radius
        kappa_n = self.curvature(t)
        correction = 1.0 - r_tube * kappa_n * np.cos(theta_tube)
        # Clamp to avoid negative metric
        correction = max(correction, 0.1)

        return ds * correction

    def metric_htheta(self, t, theta_tube):
        """Poloidal metric factor h_θ (constant = tube radius)."""
        return self.r


# ══════════════════════════════════════════════════════════════════════
# HELMHOLTZ SOLVER ON KNOT TUBE SURFACE
# ══════════════════════════════════════════════════════════════════════

class KnotEigensolver:
    """Solve ∇²ψ + k²ψ = 0 on the tube surface of a carrier knot.

    Discretizes the (s, θ) domain on an Ns × Nθ grid with periodic
    boundary conditions in both directions (the knot closes, and the
    tube cross-section is periodic).

    The Laplacian on the curved surface:
      ∇²ψ = (1/h_s h_θ) [∂/∂s(h_θ/h_s × ∂ψ/∂s) + ∂/∂θ(h_s/h_θ × ∂ψ/∂θ)]
    """

    def __init__(self, geometry, Ns=128, Ntheta=32):
        self.geom = geometry
        self.Ns = Ns
        self.Ntheta = Ntheta

        # Parameter grids
        self.t_grid = np.linspace(0, 2*np.pi, Ns, endpoint=False)
        self.theta_grid = np.linspace(0, 2*np.pi, Ntheta, endpoint=False)
        self.dt = 2*np.pi / Ns
        self.dtheta = 2*np.pi / Ntheta

        # Precompute metric on grid
        self._compute_metric()

    def _compute_metric(self):
        """Compute metric factors on the (t, θ) grid."""
        self.hs = np.zeros((self.Ns, self.Ntheta))
        self.hth = np.zeros((self.Ns, self.Ntheta))

        for i, t in enumerate(self.t_grid):
            for j, th in enumerate(self.theta_grid):
                self.hs[i, j] = self.geom.metric_hs(t, th)
                self.hth[i, j] = self.geom.metric_htheta(t, th)

    def build_laplacian(self):
        """Build the sparse Laplacian matrix for the curved surface.

        Total DOF: N = Ns × Nθ
        Index mapping: (i, j) → i * Nθ + j

        Uses second-order centered differences with periodic BCs.
        """
        Ns = self.Ns
        Nth = self.Ntheta
        N = Ns * Nth
        dt = self.dt
        dth = self.dtheta

        print(f"  Building Laplacian: {Ns}×{Nth} = {N} DOF...")

        # Build as lists of (row, col, value) triplets
        rows = []
        cols = []
        vals = []

        def idx(i, j):
            return (i % Ns) * Nth + (j % Nth)

        for i in range(Ns):
            for j in range(Nth):
                n = idx(i, j)
                hs = self.hs[i, j]
                hth = self.hth[i, j]

                # d/ds terms: (1/hs·hth) × d/ds(hth/hs × dψ/ds)
                # At (i+½, j): hth_p/hs_p; at (i-½, j): hth_m/hs_m
                hs_p = 0.5 * (self.hs[i, j] + self.hs[(i+1)%Ns, j])
                hs_m = 0.5 * (self.hs[i, j] + self.hs[(i-1)%Ns, j])
                hth_p = 0.5 * (self.hth[i, j] + self.hth[(i+1)%Ns, j])
                hth_m = 0.5 * (self.hth[i, j] + self.hth[(i-1)%Ns, j])

                cs_p = hth_p / hs_p  # coefficient at i+½
                cs_m = hth_m / hs_m  # coefficient at i-½

                # d/dθ terms: (1/hs·hth) × d/dθ(hs/hth × dψ/dθ)
                hs_jp = 0.5 * (self.hs[i, j] + self.hs[i, (j+1)%Nth])
                hs_jm = 0.5 * (self.hs[i, j] + self.hs[i, (j-1)%Nth])
                hth_jp = 0.5 * (self.hth[i, j] + self.hth[i, (j+1)%Nth])
                hth_jm = 0.5 * (self.hth[i, j] + self.hth[i, (j-1)%Nth])

                cth_p = hs_jp / hth_jp
                cth_m = hs_jm / hth_jm

                prefactor = 1.0 / (hs * hth)

                # s-direction: d/ds(c_s dψ/ds) ≈ [c_s+ (ψ_{i+1}-ψ_i) - c_s- (ψ_i-ψ_{i-1})] / dt²
                diag_s = -(cs_p + cs_m) / dt**2
                off_s_p = cs_p / dt**2
                off_s_m = cs_m / dt**2

                # θ-direction: d/dθ(c_θ dψ/dθ)
                diag_th = -(cth_p + cth_m) / dth**2
                off_th_p = cth_p / dth**2
                off_th_m = cth_m / dth**2

                # Total diagonal
                total_diag = prefactor * (diag_s + diag_th)

                # Add entries
                rows.append(n); cols.append(n); vals.append(total_diag)
                rows.append(n); cols.append(idx(i+1, j)); vals.append(prefactor * off_s_p)
                rows.append(n); cols.append(idx(i-1, j)); vals.append(prefactor * off_s_m)
                rows.append(n); cols.append(idx(i, j+1)); vals.append(prefactor * off_th_p)
                rows.append(n); cols.append(idx(i, j-1)); vals.append(prefactor * off_th_m)

        from scipy.sparse import csr_matrix
        L = csr_matrix((vals, (rows, cols)), shape=(N, N))
        print(f"  Laplacian: {L.nnz} nonzeros, density = {L.nnz/N**2:.4f}")
        return L

    def solve(self, n_modes=50):
        """Find the lowest n_modes eigenstates of -∇²ψ = k²ψ.

        Returns (eigenvalues, eigenvectors) sorted by eigenvalue.
        Eigenvalues are k² (squared wavenumber).
        """
        L = self.build_laplacian()

        # Solve -L ψ = k² ψ (L is negative semi-definite)
        print(f"  Solving for {n_modes} lowest eigenmodes...")
        # Use shift-invert for smallest eigenvalues of -L
        eigenvalues, eigenvectors = eigsh(-L, k=n_modes, sigma=0.01,
                                          which='LM', maxiter=5000)

        # Sort by eigenvalue
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        return eigenvalues, eigenvectors

    def decompose_mode(self, eigvec):
        """Fourier decompose an eigenstate to find (p_mode, q_mode).

        ψ(s, θ) = Σ A_{pq} exp(i p s/L × 2π + i q θ)

        Returns the dominant (p, q) pair.
        """
        # Reshape to 2D
        psi = eigvec.reshape(self.Ns, self.Ntheta)

        # 2D FFT
        psi_fft = np.fft.fft2(psi)
        power = np.abs(psi_fft)**2

        # Find dominant mode (skip DC)
        power[0, 0] = 0
        idx_flat = np.argmax(power)
        i_max, j_max = np.unravel_index(idx_flat, power.shape)

        # Convert to winding numbers
        # i_max corresponds to p_mode (toroidal windings along knot path)
        # j_max corresponds to q_mode (poloidal windings around tube)
        p_mode = i_max if i_max <= self.Ns//2 else i_max - self.Ns
        q_mode = j_max if j_max <= self.Ntheta//2 else j_max - self.Ntheta

        # Also get the conjugate peak
        total_power = np.sum(power)
        peak_power = power[i_max, j_max]
        purity = peak_power / total_power if total_power > 0 else 0

        return p_mode, q_mode, purity


# ══════════════════════════════════════════════════════════════════════
# MAIN: RUN FOR VARIOUS CARRIER TYPES
# ══════════════════════════════════════════════════════════════════════

def analyze_carrier(pc, qc, kappa=KAPPA_TORUS, beta=BETA_E, n_modes=40, label=None):
    """Solve eigenstates for carrier (pc, qc) and display results."""
    if label is None:
        label = f"T({pc},{qc})"

    print(f"\n{'━'*100}")
    print(f"  CARRIER: {label}  (p_c={pc}, q_c={qc}), κ={kappa:.4f}, β={beta:.6f}")
    print(f"{'━'*100}")

    geom = TorusKnotGeometry(pc, qc, kappa=kappa, beta=beta)
    print(f"  R = {geom.R:.4f} ξ, r = {geom.r:.6f} ξ, L = {geom.L_total:.2f} ξ")

    # Choose resolution based on carrier complexity
    Ns = max(64, 32 * max(pc, qc))
    Nth = 16
    print(f"  Grid: {Ns} × {Nth} = {Ns*Nth} DOF")

    solver = KnotEigensolver(geom, Ns=Ns, Ntheta=Nth)
    eigenvalues, eigenvectors = solver.solve(n_modes=n_modes)

    # Display eigenvalues and mode decomposition
    print(f"\n  {'#':>3s} {'k²':>12s} {'k²/k²_e':>10s} {'p_mode':>7s} {'q_mode':>7s} "
          f"{'purity':>8s} {'eff(p,q)':>10s} {'p²+q²':>8s}")
    print(f"  {'─'*75}")

    # Reference: electron eigenvalue (mode 0 on unknot should give k²_e)
    k2_ref = eigenvalues[0] if eigenvalues[0] > 0.1 else eigenvalues[1]

    for n in range(min(n_modes, len(eigenvalues))):
        k2 = eigenvalues[n]
        if k2 < 0.01:
            continue  # skip near-zero modes

        p_m, q_m, purity = solver.decompose_mode(eigenvectors[:, n])

        # Effective winding
        p_eff = abs(p_m) * pc if not geom.is_link else abs(p_m) * pc
        q_eff = abs(q_m) * qc if not geom.is_link else abs(q_m) * qc
        pq2 = p_eff**2 + q_eff**2

        ratio = k2 / k2_ref if k2_ref > 0 else 0

        print(f"  {n:3d} {k2:12.4f} {ratio:10.4f} {p_m:7d} {q_m:7d} "
              f"{purity:8.4f} ({p_eff},{q_eff}){'':<3s} {pq2:8d}")

    return eigenvalues, eigenvectors, solver


def main():
    print("=" * 100)
    print("NWT LEVEL-2 EIGENSTATE SOLVER: EM modes on knotted carrier tubes")
    print("=" * 100)

    # ── 1. Unknotted carrier (1,1) — should recover standard (p,q) modes
    print("\n\n" + "=" * 100)
    print("TEST 1: Unknotted carrier (1,1) — baseline")
    print("=" * 100)
    ev1, _, _ = analyze_carrier(1, 1, label="Unknot (1,1)")

    # ── 2. Hopf(2) carrier (2,2) — meson carrier
    print("\n\n" + "=" * 100)
    print("TEST 2: Hopf(2) carrier — single component of Hopf link")
    print("  (Each component of Hopf(2) is a (1,1) unknot, so this")
    print("   should look like the unknot but with linking corrections)")
    print("=" * 100)
    ev2, _, _ = analyze_carrier(2, 2, label="Hopf(2) component")

    # ── 3. Trefoil carrier (2,3)
    print("\n\n" + "=" * 100)
    print("TEST 3: Trefoil carrier T(2,3) — baryon carrier")
    print("=" * 100)
    ev3, _, _ = analyze_carrier(2, 3, label="Trefoil T(2,3)")

    # ── 4. Trefoil carrier (3,2)
    print("\n\n" + "=" * 100)
    print("TEST 4: Trefoil carrier T(3,2) — alternate baryon carrier")
    print("=" * 100)
    ev4, _, _ = analyze_carrier(3, 2, label="Trefoil T(3,2)")

    # ── 5. 5₁ knot carrier (2,5)
    print("\n\n" + "=" * 100)
    print("TEST 5: 5₁ knot carrier T(2,5) — pentaquark carrier")
    print("=" * 100)
    ev5, _, _ = analyze_carrier(2, 5, label="5₁ knot T(2,5)")

    # ── Compare eigenvalue spectra
    print("\n\n" + "=" * 100)
    print("EIGENVALUE SPECTRUM COMPARISON")
    print("=" * 100)

    carriers = [
        ("Unknot(1,1)", ev1),
        ("Hopf(2,2)", ev2),
        ("Trefoil(2,3)", ev3),
        ("Trefoil(3,2)", ev4),
        ("5₁(2,5)", ev5),
    ]

    # Show first 10 eigenvalues for each
    print(f"\n  {'#':>3s}", end="")
    for name, _ in carriers:
        print(f" {name:>14s}", end="")
    print()
    print(f"  {'─'*80}")

    for n in range(15):
        print(f"  {n:3d}", end="")
        for name, ev in carriers:
            if n < len(ev) and ev[n] > 0.01:
                print(f" {ev[n]:14.4f}", end="")
            else:
                print(f" {'---':>14s}", end="")
        print()

    # Ratios relative to unknot
    print(f"\n  Eigenvalue ratios (relative to unknot ground state k²₀ = {ev1[1]:.4f}):")
    k2_ref = ev1[1]  # first nonzero mode of unknot
    print(f"\n  {'#':>3s}", end="")
    for name, _ in carriers:
        print(f" {name:>14s}", end="")
    print()
    for n in range(1, 12):
        print(f"  {n:3d}", end="")
        for name, ev in carriers:
            if n < len(ev) and ev[n] > 0.01:
                print(f" {ev[n]/k2_ref:14.4f}", end="")
            else:
                print(f" {'---':>14s}", end="")
        print()


if __name__ == '__main__':
    main()
