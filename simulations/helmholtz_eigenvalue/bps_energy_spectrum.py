#!/usr/bin/env python3
"""
Level 1: BPS vortex energy for arbitrary torus knot T(p,q).

Computes the total BPS energy of a vortex ring on a torus knot
centerline, using the actual BPS profile f₀(ρ), a(ρ) from Step 1.

The energy density in the abelian Higgs model at BPS point:

    ε = ½|Dψ|² + ¼F² + λ/8(|ψ|²-v²)²

In tube coordinates (s, ρ, φ) around the knot centerline, the
energy integrates as:

    E = ∫₀ᴸ ds ∫₀^∞ ρ dρ ∫₀^{2π} dφ  h_s(s,ρ,φ) ε(ρ)

where h_s = 1 - ρ κ(s) cos(φ-φ₀) is the tube metric factor
and κ(s) is the curvature of the centerline.

For a STRAIGHT tube, the φ and s integrations factor out:
    E_straight = L × 2π × ∫ ρ ε(ρ) dρ = L × μ_BPS

where μ_BPS = 2π ∫ ρ ε(ρ) dρ is the BPS line tension.

For a CURVED tube, curvature corrections enter at O(ρ²κ²):
    E_curved = E_straight × [1 + correction terms]

The correction includes:
    - Kelvin-Saffman self-energy: ln(8R_eff/r_tube) factor
    - Curvature energy: ∫ κ²(s) ds × ∫ ρ³ ε(ρ) dρ / (2 ∫ ρ ε(ρ) dρ)
    - Torsion energy: higher order, neglected at leading order

We compute E(p,q) for all NWT particle topologies and compare
mass ratios E(p,q)/E(2,1) to observed m_particle/m_e.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from pathlib import Path
import math


def load_bps_profile():
    npz_path = Path(__file__).parent / "bps_profile.npz"
    data = np.load(npz_path)
    return data["rho"], data["f"], data["a"]


# ── Torus knot geometry ───────────────────────────────────────

def torus_knot_centerline(t, p, q, R=1.0, a=0.3):
    """T(p,q) torus knot on a torus (R, aR).
    Wraps p times around the major circle, q times around the minor.
    Parametric in t ∈ [0, 2π)."""
    x = (R + a * R * np.cos(q * t)) * np.cos(p * t)
    y = (R + a * R * np.cos(q * t)) * np.sin(p * t)
    z = a * R * np.sin(q * t)
    return np.array([x, y, z])


def compute_knot_geometry(p, q, R=1.0, a=0.3, N_t=8192):
    """Compute arc length, curvature, torsion for T(p,q)."""
    t = np.linspace(0, 2 * np.pi, N_t, endpoint=False)
    dt = t[1] - t[0]

    r = torus_knot_centerline(t, p, q, R, a)

    # Derivatives via central difference (handles periodicity well)
    dr = np.zeros_like(r)
    ddr = np.zeros_like(r)
    dddr = np.zeros_like(r)
    for i in range(3):
        dr[i] = np.gradient(r[i], dt, edge_order=2)
        ddr[i] = np.gradient(dr[i], dt, edge_order=2)
        dddr[i] = np.gradient(ddr[i], dt, edge_order=2)

    speed = np.sqrt(np.sum(dr**2, axis=0))

    # Arc length
    s_cumul = np.zeros(N_t)
    s_cumul[1:] = cumulative_trapezoid(speed, t)
    L = s_cumul[-1] + speed[-1] * dt

    # Curvature κ = |r' × r''| / |r'|³
    cross = np.cross(dr.T, ddr.T).T
    cross_mag = np.sqrt(np.sum(cross**2, axis=0))
    curvature = cross_mag / speed**3

    # Torsion τ = (r' × r'') · r''' / |r' × r''|²
    dot_triple = np.sum(cross * dddr, axis=0)
    torsion = dot_triple / (cross_mag**2 + 1e-30)

    return L, curvature, torsion, speed


# ── BPS energy computation ────────────────────────────────────

def compute_bps_line_tension(rho, f, a_gauge):
    """Compute BPS line tension μ = 2π ∫ ρ ε(ρ) dρ
    and the curvature correction moment ∫ ρ³ ε(ρ) dρ."""

    # BPS energy density components (per unit length):
    # At BPS (λ = e² = 1, v = 1):
    #   ε_scalar_kinetic = ½ (f')²
    #   ε_scalar_winding = ½ (1-a)² f² / ρ²
    #   ε_gauge = ½ (a')² / ρ²  [with a' = ρ(1-f²)/2]
    #   ε_potential = (1-f²)² / 8

    dr = rho[1] - rho[0]

    # Numerical derivatives
    f_prime = np.gradient(f, dr)
    a_prime = np.gradient(a_gauge, dr)

    # Energy density (radial, per unit length per 2π)
    eps_sk = 0.5 * f_prime**2
    eps_sw = 0.5 * (1 - a_gauge)**2 * f**2 / (rho**2 + 1e-30)
    eps_g = 0.5 * a_prime**2 / (rho**2 + 1e-30)
    eps_p = (1 - f**2)**2 / 8.0

    eps_total = eps_sk + eps_sw + eps_g + eps_p

    # Line tension: μ = 2π ∫ ρ ε(ρ) dρ
    mu = 2 * np.pi * trapezoid(rho * eps_total, rho)

    # Curvature moment: M₂ = 2π ∫ ρ³ ε(ρ) dρ
    M2 = 2 * np.pi * trapezoid(rho**3 * eps_total, rho)

    # Components for diagnostics
    mu_sk = 2 * np.pi * trapezoid(rho * eps_sk, rho)
    mu_sw = 2 * np.pi * trapezoid(rho * eps_sw, rho)
    mu_g = 2 * np.pi * trapezoid(rho * eps_g, rho)
    mu_p = 2 * np.pi * trapezoid(rho * eps_p, rho)

    return mu, M2, (mu_sk, mu_sw, mu_g, mu_p)


def compute_knot_energy(p, q, R, a, mu, M2, rho_max=15.0):
    """Total BPS energy for T(p,q) on torus (R, aR).

    E = μ L [1 + curvature correction]

    The curvature correction from the metric factor h_s is:
        ΔE/E = ⟨κ²⟩ × M₂ / (2 μ)

    where ⟨κ²⟩ = (1/L) ∫ κ²(s) ds.
    """
    L, curvature, torsion, speed = compute_knot_geometry(p, q, R, a)

    # Mean-square curvature
    kappa2_mean = np.mean(curvature**2)

    # Curvature correction
    curv_correction = kappa2_mean * M2 / (2.0 * mu)

    # Total energy
    E_straight = mu * L
    E_total = E_straight * (1.0 + curv_correction)

    return E_total, L, curv_correction, np.mean(curvature), kappa2_mean


# ── NWT particle catalog ──────────────────────────────────────

# (p, q, name, PDG mass in MeV)
PARTICLES = [
    # Leptons (unknot excitations)
    (2, 1, "e",       0.51100,   1),
    (2, 1, "μ",       105.658,   157),
    (2, 1, "τ",       1776.86,   1900),

    # Light mesons (S² × S² sector)
    (1, 1, "π⁰",      134.977,  1),
    (1, 1, "π±",      139.570,   1),
    (1, 1, "K±",      493.677,   1),

    # Baryons
    (1, 4, "p",       938.272,   1),
    (1, 4, "n",       939.565,   1),

    # Trefoil-based (quarks/strong)
    (2, 3, "J/ψ",     3096.9,   1),

    # Cinquefoil (GUT portal)
    (2, 5, "Pc(4312)", 4311.9,  1),
]


def main():
    rho, f, a_gauge = load_bps_profile()

    print("=" * 72)
    print("LEVEL 1: BPS VORTEX ENERGY SPECTRUM FOR TORUS KNOTS")
    print("=" * 72)

    # Compute line tension
    mu, M2, (mu_sk, mu_sw, mu_g, mu_p) = compute_bps_line_tension(
        rho, f, a_gauge)

    print(f"\n  BPS line tension: μ = {mu:.6f}")
    print(f"    Components: kinetic={mu_sk:.4f}, winding={mu_sw:.4f}, "
          f"gauge={mu_g:.4f}, potential={mu_p:.4f}")
    print(f"    BPS check: kinetic ≈ winding? "
          f"{abs(mu_sk-mu_sw)/mu_sk*100:.2f}% diff")
    print(f"    BPS check: potential ≈ 2×gauge? "
          f"{abs(mu_p-2*mu_g)/(2*mu_g)*100:.2f}% diff")
    print(f"  Curvature moment: M₂ = {M2:.6f}")
    print(f"  Ratio M₂/μ = {M2/mu:.4f} (effective ρ² scale)")

    # Fixed torus parameters
    R = np.pi**2        # Macken: r_tube/R = 1/π²
    a_torus = 0.3       # torus aspect ratio

    print(f"\n  Torus: R = π² = {R:.4f}, a = {a_torus}")
    print(f"  r_tube/R = 1/π² = {1/R:.5f}")

    # ── Scan over all (p,q) topologies ──────────────────────
    print(f"\n{'─'*72}")
    print(f"  ENERGY SPECTRUM: E(p,q) for representative torus knots")
    print(f"{'─'*72}")

    # Broader scan of (p,q) values
    pq_scan = [
        (1, 1, "unknot (Hopf)"),
        (2, 1, "unknot (electron)"),
        (1, 2, "unknot (alt)"),
        (2, 3, "trefoil"),
        (3, 2, "trefoil (alt)"),
        (2, 5, "cinquefoil"),
        (1, 3, "torus link"),
        (1, 4, "proton candidate"),
        (3, 4, "T(3,4)"),
        (2, 7, "heptafoil"),
        (3, 5, "T(3,5)"),
    ]

    results = {}
    print(f"\n  {'T(p,q)':<18} {'L':>8} {'E':>12} {'κ_mean':>8} "
          f"{'curv_corr':>10} {'E/E(2,1)':>10} {'p²+q²':>6}")
    print(f"  {'─'*18} {'─'*8} {'─'*12} {'─'*8} {'─'*10} {'─'*10} {'─'*6}")

    E_electron = None

    for p, q, label in pq_scan:
        try:
            E, L, cc, kappa_m, kappa2_m = compute_knot_energy(
                p, q, R, a_torus, mu, M2)
            results[(p, q)] = E

            if (p, q) == (2, 1):
                E_electron = E

            ratio = E / E_electron if E_electron else 0
            pq2 = p**2 + q**2

            print(f"  T({p},{q}) {label:<12} {L:8.2f} {E:12.4f} "
                  f"{kappa_m:8.5f} {cc:+10.6f} {ratio:10.4f} {pq2:6d}")
        except Exception as ex:
            print(f"  T({p},{q}) {label:<12} FAILED: {ex}")

    # ── Mass ratio predictions ──────────────────────────────
    print(f"\n{'='*72}")
    print(f"MASS RATIO PREDICTIONS vs PDG")
    print(f"{'='*72}")
    print(f"\n  The BPS energy ratio E(p,q)/E(2,1) predicts the mass ratio")
    print(f"  m_particle/m_e IF the particle's topology is T(p,q).")
    print(f"\n  {'Particle':<12} {'T(p,q)':<10} {'E/E_e':>10} "
          f"{'m/m_e PDG':>10} {'Ratio':>8} {'Match':>8}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

    m_e = 0.51100  # MeV

    for p, q, name, m_pdg, m_idx in PARTICLES:
        if (p, q) not in results or E_electron is None:
            continue
        E_ratio = results[(p, q)] / E_electron
        m_ratio_pdg = m_pdg / m_e
        match = E_ratio / m_ratio_pdg if m_ratio_pdg > 0 else 0

        print(f"  {name:<12} T({p},{q})    {E_ratio:10.4f} "
              f"{m_ratio_pdg:10.2f} {match:8.4f} "
              f"{'✓' if abs(match-1) < 0.1 else '✗'}")

    # ── Analysis of what controls the energy ratio ─────────
    print(f"\n{'='*72}")
    print(f"WHAT CONTROLS THE ENERGY RATIOS?")
    print(f"{'='*72}")

    if E_electron and (2, 3) in results:
        E_tref = results[(2, 3)]
        E_elec = E_electron
        L_tref = compute_knot_geometry(2, 3, R, a_torus)[0]
        L_elec = compute_knot_geometry(2, 1, R, a_torus)[0]
        pq2_tref = 13
        pq2_elec = 5

        print(f"""
  E(trefoil)/E(electron) = {E_tref/E_elec:.4f}

  If this were purely arc-length controlled:
    L(2,3)/L(2,1) = {L_tref/L_elec:.4f}

  If this were purely (p²+q²) controlled:
    (p²+q²)_tref / (p²+q²)_elec = {pq2_tref}/{pq2_elec} = {pq2_tref/pq2_elec:.4f}

  Actual ratio: {E_tref/E_elec:.4f}
  Arc-length ratio: {L_tref/L_elec:.4f}

  The energy ratio is dominated by the arc-length ratio L(p,q),
  which scales roughly as √(p²+q²) for thin torus knots.

  The (p²+q²)/5 factor in Paper 6's mass formula arises because:
    L(p,q) ∝ √(p² R² + q² a² R²) = R √(p² + q² a²)
    L(2,1) = R √(4 + a²)    (the electron)
    L(p,q)/L(2,1) ≈ √((p²+q²a²)/(4+a²))

  At a = {a_torus}:
    L(2,3)/L(2,1) = √((4+9×{a_torus}²)/(4+{a_torus}²)) = {np.sqrt((4+9*a_torus**2)/(4+a_torus**2)):.4f}
    vs actual: {L_tref/L_elec:.4f}

  The (p²+q²)/5 normalization in Paper 6 corresponds to the
  a→1 limit: √((p²+q²)/5) for the electron at (2,1).
""")

    # ── The missing physics ───────────────────────────────
    print(f"{'='*72}")
    print(f"WHAT'S MISSING FROM LEVEL 1")
    print(f"{'='*72}")
    print(f"""
  The BPS tube energy gives energy ratios controlled primarily by
  arc length ratios L(p,q)/L(2,1).  These scale as √(p²+q²a²),
  which produces mass ratios of order 1-3, NOT the observed mass
  ratios of order 1-3500 (m_τ/m_e ≈ 3477).

  The missing physics includes:

  1. RADIAL EXCITATION INDEX m: the (2,1) unknot with m=3 (electron)
     vs m=1900 (tau) have hugely different energies.  The m index
     enters via the radial eigenmode energy E_m ∝ m × (radial gap).
     This is the dominant source of the mass hierarchy.

  2. NONLINEAR SELF-INTERACTION: at BPS, the vortex is marginally
     stable; real particles are near-BPS with finite binding energy.
     The self-interaction of the vortex ring (Kelvin-Saffman log
     term, reconnection energy at crossings) adds topology-dependent
     corrections.

  3. GAUGE-FIELD CONTRIBUTIONS: the AB phase and D₃ character
     amplitudes from the seven-step α derivation enter the energy
     functional but are not captured by the tube-adapted BPS energy
     alone.

  CONCLUSION: Level 1 correctly captures the TOPOLOGICAL part of
  the energy (arc length, curvature), but the DYNAMICAL part (radial
  excitations, self-interaction, gauge coupling) requires Level 2
  (full 3D abelian Higgs relaxation) or a more sophisticated
  tube-adapted calculation that includes the m-dependent radial
  eigenmodes from Step 2.
""")


if __name__ == "__main__":
    main()
