#!/usr/bin/env python3
"""
Step 3: Helmholtz eigenvalue on a BPS vortex tube around a trefoil knot.

The trefoil T(2,3) lives on a torus of major radius R and minor
radius a*R.  A vortex tube of healing-length radius ξ wraps around
the trefoil centerline.  The NWT identification is:

    r_tube / R = 1/π²  (Macken impedance aspect ratio)

so in units where ξ = 1 (the BPS profile's natural scale), R = π².

The Helmholtz operator on the tube in (s, ρ) coordinates (averaged
over the azimuthal angle φ for the m=0 sector):

    H = -(1/h̄²) ∂²/∂s² - (1/ρ) ∂/∂ρ (ρ ∂/∂ρ) + V(ρ) + V_curv(s,ρ)

where h̄ is the φ-averaged metric factor and V_curv contains
curvature corrections.

Grid: N_s (FFT, periodic in s) × N_rho (FD, Dirichlet at boundaries).
Sparse matrix, solved with scipy.sparse.linalg.eigsh (shift-invert).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.integrate import cumulative_trapezoid
from pathlib import Path


def load_bps_profile():
    npz_path = Path(__file__).parent / "bps_profile.npz"
    data = np.load(npz_path)
    return data["rho"], data["f"]


# ── Trefoil geometry ──────────────────────────────────────────

def trefoil_centerline(t, R=1.0, a=0.3):
    """T(2,3) torus knot on a torus (R, aR).
    Parametric in t ∈ [0, 2π)."""
    x = (R + a * R * np.cos(3 * t)) * np.cos(2 * t)
    y = (R + a * R * np.cos(3 * t)) * np.sin(2 * t)
    z = a * R * np.sin(3 * t)
    return np.array([x, y, z])


def trefoil_derivatives(t, R=1.0, a=0.3):
    """First, second, third derivatives of the trefoil centerline."""
    c3 = np.cos(3 * t)
    s3 = np.sin(3 * t)
    c2 = np.cos(2 * t)
    s2 = np.sin(2 * t)

    # r'(t)
    dx = -3 * a * R * s3 * c2 - 2 * (R + a * R * c3) * s2
    dy = -3 * a * R * s3 * s2 + 2 * (R + a * R * c3) * c2
    dz = 3 * a * R * c3
    dr = np.array([dx, dy, dz])

    # r''(t)
    ddx = (-9*a*R*c3*c2 + 6*a*R*s3*s2
           + 6*a*R*s3*s2 - 4*(R + a*R*c3)*c2)
    ddy = (-9*a*R*c3*s2 - 6*a*R*s3*c2
           - 6*a*R*s3*c2 - 4*(R + a*R*c3)*s2)
    ddz = -9 * a * R * s3
    ddr = np.array([ddx, ddy, ddz])

    # r'''(t)
    dddx = (27*a*R*s3*c2 + 9*a*R*c3*2*s2
            + 12*a*R*c3*s2 + 6*a*R*s3*2*c2
            + 8*(R + a*R*c3)*s2 + 4*3*a*R*s3*c2)
    # This is getting messy — use numerical differentiation instead
    return dr, ddr


def compute_trefoil_geometry(N_t=4096, R=1.0, a=0.3):
    """Compute arc length, curvature along the trefoil.

    Returns (s_grid, curvature, arc_length_total).
    s_grid is uniformly spaced in arc length.
    """
    t = np.linspace(0, 2 * np.pi, N_t, endpoint=False)
    dt = t[1] - t[0]

    # Centerline position
    r = trefoil_centerline(t, R, a)

    # First derivative (central difference for clean periodicity)
    dr = np.zeros_like(r)
    for i in range(3):
        dr[i] = np.gradient(r[i], dt, edge_order=2)

    # Speed |dr/dt|
    speed = np.sqrt(np.sum(dr**2, axis=0))

    # Arc length
    ds_dt = speed
    s_cumul = np.zeros(N_t)
    s_cumul[1:] = cumulative_trapezoid(ds_dt, t)
    L = s_cumul[-1] + ds_dt[-1] * dt  # close the loop

    # Second derivative
    ddr = np.zeros_like(r)
    for i in range(3):
        ddr[i] = np.gradient(dr[i], dt, edge_order=2)

    # Curvature κ = |r' × r''| / |r'|³
    cross = np.cross(dr.T, ddr.T).T
    cross_mag = np.sqrt(np.sum(cross**2, axis=0))
    curvature = cross_mag / speed**3

    return t, s_cumul, curvature, speed, L


# ── Helmholtz operator ────────────────────────────────────────

def build_trefoil_operator(s_grid, rho_grid, f0_on_rho, curvature_on_s,
                           L_arc):
    """
    Build the 2D Helmholtz operator on (s, ρ) grid for m=0 sector.

    The operator (in the weighted form for L²(ρ ds dρ)):

    A ψ = κ² B ψ

    where:
      A includes: -ρ/h̄² · ∂²/∂s² − ∂/∂ρ(ρ ∂/∂ρ)  + ρ V_eff(s,ρ)
      B = ρ (the radial weight)

    V_eff(s,ρ) = 3f₀²(ρ) - 1 + ρ²κ(s)²/2  (curvature correction)

    s: periodic (FFT-based second derivative)
    ρ: finite difference with Dirichlet BCs
    """
    N_s = len(s_grid)
    N_rho = len(rho_grid)
    N_total = N_s * N_rho
    ds = s_grid[1] - s_grid[0] if N_s > 1 else L_arc
    dr = rho_grid[1] - rho_grid[0]

    V_base = 3.0 * f0_on_rho**2 - 1.0

    # Build sparse operator
    A = lil_matrix((N_total, N_total))
    B_diag = np.zeros(N_total)

    def idx(i_s, i_rho):
        return i_s * N_rho + i_rho

    for i_s in range(N_s):
        kappa_s = curvature_on_s[i_s]

        for i_rho in range(N_rho):
            rho = rho_grid[i_rho]
            I = idx(i_s, i_rho)

            # Weight matrix B = ρ
            B_diag[I] = rho

            # φ-averaged metric factor squared: ⟨h_s²⟩ = 1 + ρ²κ²/2
            h2_avg = 1.0 + 0.5 * rho**2 * kappa_s**2

            # --- s-derivative: -ρ/h̄² · ∂²/∂s² ---
            # Central difference, periodic
            coeff_s = rho / (h2_avg * ds**2)
            i_s_prev = (i_s - 1) % N_s
            i_s_next = (i_s + 1) % N_s
            A[I, idx(i_s_prev, i_rho)] += -coeff_s
            A[I, I] += 2.0 * coeff_s
            A[I, idx(i_s_next, i_rho)] += -coeff_s

            # --- ρ-derivative: -(d/dρ)(ρ d/dρ) in FD form ---
            # This is: -ρ ψ'' - ψ'  (multiply through by nothing;
            # the weighted form has B=ρ on the RHS)
            coeff_rho = rho / dr**2
            if i_rho > 0:
                A[I, idx(i_s, i_rho - 1)] += -coeff_rho
                A[I, idx(i_s, i_rho - 1)] += 1.0 / (2.0 * dr)
            A[I, I] += 2.0 * coeff_rho
            if i_rho < N_rho - 1:
                A[I, idx(i_s, i_rho + 1)] += -coeff_rho
                A[I, idx(i_s, i_rho + 1)] += -1.0 / (2.0 * dr)

            # --- Potential: ρ V_eff(s,ρ) ---
            V_eff = V_base[i_rho] + 0.5 * rho**2 * kappa_s**2
            A[I, I] += rho * V_eff

    return csr_matrix(A), B_diag


def main():
    rho_raw, f_raw = load_bps_profile()

    # Trefoil parameters
    R = np.pi**2       # major radius in units of ξ (Macken: r_tube/R = 1/π²)
    a_values = [0.2, 0.3, 0.4]  # torus aspect ratios to scan

    # Grid parameters
    N_s = 128           # longitudinal (periodic)
    N_rho = 64          # radial (Dirichlet)
    rho_min = 0.05
    rho_max = 8.0

    rho_grid = np.linspace(rho_min, rho_max, N_rho)
    f0 = np.interp(rho_grid, rho_raw, f_raw)

    n_eigs = 20  # number of eigenvalues to find

    print("=" * 72)
    print("STEP 3: HELMHOLTZ EIGENVALUES ON TREFOIL BPS VORTEX TUBE")
    print(f"  R = π² = {R:.4f} (Macken aspect ratio r_tube/R = 1/π²)")
    print(f"  Grid: N_s = {N_s}, N_rho = {N_rho}, DOF = {N_s * N_rho}")
    print("=" * 72)

    # Straight-tube baseline (no curvature)
    print(f"\n{'─'*72}")
    print(f"BASELINE: straight tube (κ(s) = 0)")
    print(f"{'─'*72}")
    zero_curv = np.zeros(N_s)
    s_dummy = np.linspace(0, 2*np.pi*R, N_s, endpoint=False)
    A, B_diag = build_trefoil_operator(s_dummy, rho_grid, f0, zero_curv,
                                        2*np.pi*R)
    B_inv_sqrt = np.sqrt(1.0 / np.maximum(B_diag, 1e-30))
    # Transform to standard eigenvalue: B^{-1/2} A B^{-1/2} v = κ² v
    from scipy.sparse import diags
    B_isq = diags(B_inv_sqrt)
    H_std = B_isq @ A @ B_isq
    evals, evecs = eigsh(H_std, k=n_eigs, which='SM')
    evals = np.sort(evals)

    print(f"  Lowest {n_eigs} eigenvalues (κ²):")
    for k, e in enumerate(evals):
        kappa = np.sqrt(abs(e))
        print(f"    κ²_{k:2d} = {e:10.5f}   κ = {kappa:8.5f}")
    print(f"  Straight-tube radial bound state: κ² ≈ {evals[0]:.4f}")

    # Trefoil scans
    for a in a_values:
        print(f"\n{'─'*72}")
        print(f"TREFOIL: a = {a} (torus minor/major = {a:.1f})")
        print(f"{'─'*72}")

        t_param, s_cumul, curvature, speed, L = compute_trefoil_geometry(
            N_t=8192, R=R, a=a)

        print(f"  Arc length L = {L:.4f} ξ")
        print(f"  Curvature: min = {curvature.min():.5f}, "
              f"max = {curvature.max():.5f}, "
              f"mean = {curvature.mean():.5f}")
        print(f"  ρ_max × κ_max = {rho_max * curvature.max():.3f} "
              f"(should be < 1 for thin-tube validity)")

        # Interpolate curvature to uniform arc-length grid
        s_uniform = np.linspace(0, L, N_s, endpoint=False)
        curv_uniform = np.interp(s_uniform, s_cumul, curvature,
                                  period=L)

        A, B_diag = build_trefoil_operator(s_uniform, rho_grid, f0,
                                            curv_uniform, L)
        B_inv_sqrt = np.sqrt(1.0 / np.maximum(B_diag, 1e-30))
        B_isq = diags(B_inv_sqrt)
        H_std = B_isq @ A @ B_isq

        evals, evecs = eigsh(H_std, k=n_eigs, which='SM')
        evals = np.sort(evals)

        print(f"\n  Lowest {n_eigs} eigenvalues (κ²):")
        for k, e in enumerate(evals):
            kappa = np.sqrt(abs(e))
            tags = []
            if abs(e - np.pi**2) < 1.0:
                tags.append(f"near π²={np.pi**2:.4f}")
            if abs(kappa - np.pi) < 0.3:
                tags.append(f"near π")
            if abs(kappa - 9.844) < 0.5:
                tags.append("★ near κ_SM!")
            if abs(e - 9.844**2) < 5:
                tags.append("★ κ² near κ_SM²!")
            tag = f"  ← {', '.join(tags)}" if tags else ""
            print(f"    κ²_{k:2d} = {e:10.5f}   κ = {kappa:8.5f}{tag}")

        # How much did curvature shift the lowest eigenvalue?
        delta = evals[0] - evals[0]  # placeholder
        print(f"\n  Curvature shift on lowest mode: "
              f"Δκ² = {evals[0] - 0.924:.4f} "
              f"(straight tube: 0.924)")

    # Summary
    print(f"\n{'='*72}")
    print(f"INTERPRETATION")
    print(f"{'='*72}")
    print(f"""
  The straight-tube radial bound state sits at κ² ≈ 0.92 (κ ≈ 0.96).
  The trefoil geometry adds:
    1. Longitudinal quantization: k_s = 2πn/L, small for large L
    2. Curvature correction to the radial potential: O(ρ²κ²)

  If κ_SM = 9.844 does NOT appear as an eigenvalue, it suggests:
    • κ_SM is a COMPOSITE quantity (radial × longitudinal × D_3 phase)
      rather than a single Helmholtz eigenvalue
    • The identification κ_SM² = 1/(√2 α) = 25π√3 + 1)/√2 arises
      from the BPS energy functional, not a spatial eigenvalue
    • Step 3 establishes what the ACTUAL eigenvalue spectrum looks like,
      informing the correct interpretation of κ_SM

  The simulation program continues: the decisive test requires
  the full gravitating abelian Higgs action, not just the linearized
  Helmholtz operator.
""")


if __name__ == "__main__":
    main()
