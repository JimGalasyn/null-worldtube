#!/usr/bin/env python3
"""
CASIMIR ENERGY OF QUANTUM FLUCTUATIONS AROUND KNOTTED VORTICES.

The D_q symmetry of a torus knot T(2,q) creates a periodic curvature
potential V(s) along the vortex:

    V(s) = (ρ²/2) κ²(s)

where κ(s) has period L/q (D_q symmetry).  This potential opens
band gaps in the longitudinal mode spectrum, modifying the vacuum
fluctuation energy.

The Casimir energy difference between a knotted and straight vortex:

    ΔE_Cas = E_Cas(knot) - E_Cas(straight ring, same L)

is FINITE (high modes cancel) and TOPOLOGY-DEPENDENT (through the
D_q symmetry of the curvature).

THE KEY QUESTION: Does ΔE_Cas correlate with the Casimir invariants
C_A(SU(q-1)) that appear in the NWT mass formulas?

If YES: the "Casimir framework" literally IS the Casimir effect
of quantum vacuum fluctuations around torus-knot vortices.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.integrate import cumulative_trapezoid
from pathlib import Path


def load_bps_profile():
    npz_path = Path(__file__).parent.parent / "helmholtz_eigenvalue" / "bps_profile.npz"
    data = np.load(npz_path)
    return data["rho"], data["f"], data["a"]


def knot_curvature_potential(q, N_s, R=None, a_torus=0.3):
    """Compute the curvature-squared potential along T(2,q).

    Returns (V(s), L) where V is the potential and L is arc length.
    """
    if R is None:
        R = np.pi**2
    N_t = max(N_s * 4, 4096)
    t = np.linspace(0, 2 * np.pi, N_t, endpoint=False)
    dt = t[1] - t[0]
    a = a_torus

    x = (R + a*R*np.cos(q*t)) * np.cos(2*t)
    y = (R + a*R*np.cos(q*t)) * np.sin(2*t)
    z = a * R * np.sin(q*t)

    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    dz = np.gradient(z, dt)
    ddx = np.gradient(dx, dt)
    ddy = np.gradient(dy, dt)
    ddz = np.gradient(dz, dt)

    speed = np.sqrt(dx**2 + dy**2 + dz**2)
    cross_mag = np.sqrt((dy*ddz - dz*ddy)**2 +
                        (dz*ddx - dx*ddz)**2 +
                        (dx*ddy - dy*ddx)**2)
    kappa = cross_mag / speed**3

    s = np.zeros(N_t)
    s[1:] = cumulative_trapezoid(speed, t)
    L = s[-1] + speed[-1] * dt

    # Interpolate κ²(s) onto uniform s grid
    s_uniform = np.linspace(0, L, N_s, endpoint=False)
    kappa_uniform = np.interp(s_uniform, s, kappa, period=L)

    # The potential from curvature: V(s) = ⟨ρ²⟩ κ²(s) / 2
    # ⟨ρ²⟩ is the mean-square radial distance weighted by the
    # eigenmode — from the Step 2 breathing mode at κ_rad ≈ 0.96.
    # Use ⟨ρ²⟩ ≈ 2ξ² (characteristic width of the BPS profile).
    rho2_avg = 2.0  # in units of ξ²
    V = 0.5 * rho2_avg * kappa_uniform**2

    return V, s_uniform, L


def build_longitudinal_hamiltonian(V, L, N_s):
    """Build H = -d²/ds² + V(s) on a periodic circle of length L.

    Returns the N_s × N_s Hamiltonian matrix.
    """
    ds = L / N_s

    H = np.zeros((N_s, N_s))
    for i in range(N_s):
        # Kinetic: -d²/ds² (second-order FD, periodic)
        i_prev = (i - 1) % N_s
        i_next = (i + 1) % N_s
        H[i, i_prev] += -1.0 / ds**2
        H[i, i] += 2.0 / ds**2
        H[i, i_next] += -1.0 / ds**2

        # Potential
        H[i, i] += V[i]

    return H


def casimir_energy_difference(eigenvalues_knot, eigenvalues_free,
                               kappa_rad_sq=0.92, N_cutoff=None):
    """Compute the Casimir energy difference between knotted and
    straight vortex.

    ΔE = ½ Σ_n [ω_n(knot) - ω_n(free)]

    where ω_n = √(λ_n + κ²_rad).

    The sum converges because high-n eigenvalues match.
    """
    if N_cutoff is None:
        N_cutoff = len(eigenvalues_knot)
    else:
        N_cutoff = min(N_cutoff, len(eigenvalues_knot),
                       len(eigenvalues_free))

    omega_knot = np.sqrt(np.maximum(eigenvalues_knot[:N_cutoff] +
                                     kappa_rad_sq, 0))
    omega_free = np.sqrt(np.maximum(eigenvalues_free[:N_cutoff] +
                                     kappa_rad_sq, 0))

    delta_E = 0.5 * np.sum(omega_knot - omega_free)
    return delta_E


def main():
    rho_bps, f_bps, a_bps = load_bps_profile()

    R = np.pi**2
    a_torus = 0.3
    N_s = 512
    kappa_rad_sq = 0.92  # from Step 2

    print("=" * 72)
    print("CASIMIR ENERGY OF QUANTUM FLUCTUATIONS AROUND KNOTTED VORTICES")
    print("=" * 72)

    # ── Build the free (straight-ring) spectrum ───────────
    print(f"\n{'─'*72}")
    print(f"  REFERENCE: Straight ring (no curvature modulation)")
    print(f"{'─'*72}")

    # For each knot, we need the free spectrum at the SAME L
    # Free Hamiltonian: H_free = -d²/ds², eigenvalues = (2πn/L)²

    # ── Scan over knot topologies ─────────────────────────
    knots = [
        (3, "trefoil"),
        (5, "cinquefoil"),
        (7, "heptafoil"),
        (9, "nonafoil"),
        (11, "T(2,11)"),
        (13, "T(2,13)"),
    ]

    print(f"\n{'─'*72}")
    print(f"  CASIMIR ENERGY DIFFERENCES ΔE_Cas(T(2,q)) vs CASIMIR INVARIANTS")
    print(f"{'─'*72}")
    print(f"\n  κ²_rad = {kappa_rad_sq} (from Step 2 breathing mode)")
    print(f"  ⟨ρ²⟩ = 2ξ² (BPS profile width)")

    results = []

    for q, label in knots:
        # Build curvature potential
        V, s_grid, L = knot_curvature_potential(q, N_s, R, a_torus)

        # Knot Hamiltonian
        H_knot = build_longitudinal_hamiltonian(V, L, N_s)
        evals_knot = eigh(H_knot, eigvals_only=True)

        # Free Hamiltonian (V=0, same L)
        V_free = np.zeros(N_s)
        H_free = build_longitudinal_hamiltonian(V_free, L, N_s)
        evals_free = eigh(H_free, eigvals_only=True)

        # Casimir energy difference (convergence study)
        dE_values = []
        for N_cut in [50, 100, 200, N_s]:
            dE = casimir_energy_difference(evals_knot, evals_free,
                                            kappa_rad_sq, N_cut)
            dE_values.append((N_cut, dE))

        dE_final = dE_values[-1][1]

        # Topology identifiers
        CA = q - 1  # crossing number → C_A(SU(q-1))... wait
        # Actually for T(2,q), the minimal crossing number is q
        # And C_A(SU(N)) = N for the adjoint rep
        # The gauge group from q crossings is SU(q) or related
        crossings = q
        CA_sq = crossings**2
        q_sq = q**2

        # Mean curvature and potential
        V_mean = np.mean(V)
        V_max = np.max(V)
        kappa_mean = np.sqrt(2 * V_mean / 2.0)  # from V = ρ²κ²/2

        results.append({
            'q': q, 'label': label, 'L': L, 'dE': dE_final,
            'V_mean': V_mean, 'V_max': V_max,
            'crossings': crossings, 'CA_sq': CA_sq, 'q_sq': q_sq,
        })

        print(f"\n  T(2,{q}) — {label}:")
        print(f"    L = {L:.2f}, V_mean = {V_mean:.6f}, V_max = {V_max:.6f}")
        print(f"    Convergence: ", end="")
        for N_cut, dE in dE_values:
            print(f"N={N_cut}: ΔE={dE:.6f}  ", end="")
        print()
        print(f"    ΔE_Cas = {dE_final:.8f}")

    # ── Correlation analysis ──────────────────────────────
    print(f"\n{'='*72}")
    print(f"CORRELATION: ΔE_Cas vs TOPOLOGICAL QUANTITIES")
    print(f"{'='*72}")

    print(f"\n  {'T(2,q)':<12} {'q':>3} {'L':>8} {'ΔE_Cas':>12} "
          f"{'ΔE/L':>12} {'ΔE×L':>12} {'q²':>6} {'C_A=q':>6}")
    print(f"  {'─'*12} {'─'*3} {'─'*8} {'─'*12} {'─'*12} {'─'*12} {'─'*6} {'─'*6}")

    dE_ref = results[0]['dE']  # trefoil as reference

    for r in results:
        dE_per_L = r['dE'] / r['L']
        dE_times_L = r['dE'] * r['L']
        print(f"  T(2,{r['q']}) {r['label']:<8} {r['q']:3d} {r['L']:8.2f} "
              f"{r['dE']:12.8f} {dE_per_L:12.8f} {dE_times_L:12.6f} "
              f"{r['q_sq']:6d} {r['crossings']:6d}")

    # Ratios relative to trefoil
    print(f"\n  RATIOS (relative to trefoil):")
    print(f"\n  {'T(2,q)':<12} {'ΔE/ΔE_tref':>12} {'L/L_tref':>12} "
          f"{'q/3':>8} {'q²/9':>8} {'C_A/3':>8} "
          f"{'sin(2π/q)/sin(2π/3)':>20}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*20}")

    for r in results:
        dE_ratio = r['dE'] / dE_ref if abs(dE_ref) > 1e-15 else 0
        L_ratio = r['L'] / results[0]['L']
        q_ratio = r['q'] / 3
        q2_ratio = r['q_sq'] / 9
        CA_ratio = r['crossings'] / 3
        sin_ratio = np.sin(2*np.pi/r['q']) / np.sin(2*np.pi/3)

        print(f"  T(2,{r['q']}) {r['label']:<8} {dE_ratio:12.6f} "
              f"{L_ratio:12.6f} {q_ratio:8.4f} {q2_ratio:8.4f} "
              f"{CA_ratio:8.4f} {sin_ratio:20.6f}")

    # ── Check specific correlations ───────────────────────
    print(f"\n{'─'*72}")
    print(f"  BEST-FIT SCALING")
    print(f"{'─'*72}")

    # Try ΔE ∝ V_mean × L (= total potential energy)
    print(f"\n  Model: ΔE ∝ V_mean × L (total curvature potential)")
    for r in results:
        pred = r['V_mean'] * r['L']
        ratio = r['dE'] / pred if abs(pred) > 1e-15 else 0
        print(f"    T(2,{r['q']}): ΔE/(V_mean·L) = {ratio:.6f}")

    # Try ΔE ∝ ⟨κ²⟩ × L (mean-square curvature × length)
    print(f"\n  Model: ΔE ∝ ⟨κ²⟩ × L")
    for r in results:
        kappa2_mean = 2 * r['V_mean'] / 2.0  # V = ρ²κ²/2, ρ²=2
        pred = kappa2_mean * r['L']
        ratio = r['dE'] / pred if abs(pred) > 1e-15 else 0
        print(f"    T(2,{r['q']}): ΔE/(⟨κ²⟩·L) = {ratio:.6f}")

    print(f"\n{'='*72}")
    print(f"INTERPRETATION")
    print(f"{'='*72}")

    # Check if ΔE_Cas is topology-independent (∝ L × V_mean)
    ratios = []
    for r in results:
        pred = r['V_mean'] * r['L']
        if abs(pred) > 1e-15:
            ratios.append(r['dE'] / pred)

    if ratios:
        ratio_std = np.std(ratios) / np.mean(ratios)
        if ratio_std < 0.01:
            print(f"""
  ΔE_Cas ∝ V_mean × L with {ratio_std*100:.2f}% scatter.

  The Casimir energy difference is PROPORTIONAL TO THE TOTAL
  CURVATURE POTENTIAL — i.e., it scales with how much the
  knot curves, not with the D_q symmetry class.

  This means the Casimir energy does NOT produce the Casimir
  INVARIANTS C_A(SU(N)).  The band gaps from D_q symmetry
  are present but their contribution to the total energy is
  dominated by the mean curvature, which is smooth and
  topology-blind at this order.

  To see D_q-specific contributions, we would need to go to
  higher order in perturbation theory (the BAND GAP widths
  depend on the Fourier amplitude of the curvature at
  frequency q, which IS D_q-specific).
""")
        else:
            print(f"""
  ΔE_Cas ratio scatter: {ratio_std*100:.1f}% — significant
  topology dependence detected!

  The Casimir energy difference does NOT simply scale with
  V_mean × L.  There is a genuine D_q-dependent contribution
  that may correlate with Casimir invariants.
""")

    # Band gap analysis
    print(f"{'─'*72}")
    print(f"  BAND GAP ANALYSIS")
    print(f"{'─'*72}")
    print(f"\n  The D_q-specific physics lives in the BAND GAPS,")
    print(f"  not the total Casimir energy.  The gap at the q-th")
    print(f"  Brillouin zone boundary is proportional to the q-th")
    print(f"  Fourier component of the curvature potential.\n")

    for q, label in knots[:4]:
        V, s_grid, L = knot_curvature_potential(q, N_s, R, a_torus)
        H = build_longitudinal_hamiltonian(V, L, N_s)
        evals = eigh(H, eigvals_only=True)

        # The q-th band gap: difference between eigenvalues q and q-1
        # (at the q-th Brillouin zone boundary)
        if 2*q < len(evals):
            gap = evals[2*q] - evals[2*q - 1]
            # Free eigenvalues at the same position
            n_bz = q  # mode number at BZ boundary
            free_val = (2 * np.pi * n_bz / L)**2
            print(f"  T(2,{q}): gap at n={q}: Δλ = {gap:.8f}, "
                  f"free λ = {free_val:.4f}, "
                  f"Δλ/free = {gap/free_val:.6f}")

    print(f"\n  If the gap ratios Δλ(q)/Δλ(3) scale like a Casimir")
    print(f"  invariant, that's the connection between the Casimir")
    print(f"  EFFECT and the Casimir INVARIANTS.")


if __name__ == "__main__":
    main()
