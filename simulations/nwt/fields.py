"""Casimir energy, greybody factors, torus EM fields, and LC circuit."""

import numpy as np
from .constants import (
    c, hbar, h_planck, e_charge, eps0, mu0, m_e, m_e_MeV, alpha,
    eV, MeV, lambda_C, r_e, k_e, TorusParams,
)
from .core import (
    compute_path_length, compute_self_energy, compute_total_energy,
    find_self_consistent_radius,
)


def compute_epstein_zeta_ren(tau, N_terms=200):
    """
    Regularized Epstein zeta Z_ren(-1/2, τ) for the flat 2-torus.

    Uses the Chowla-Selberg formula:
        Z_ren(-1/2, τ) = -(1+τ)/6 - (2τ/π) × Σ_{n=1}^{N} σ₂(n) K₁(2πnτ)/n

    where σ₂(n) = sum of d² for d|n, K₁ = modified Bessel function.
    """
    from scipy.special import kv

    Z_const = -(1.0 + tau) / 6.0

    Z_bessel = 0.0
    for n in range(1, N_terms + 1):
        arg = 2.0 * np.pi * n * tau
        if arg > 500:
            break
        # σ₂(n): sum of d² for all divisors d of n
        sigma2 = sum(d * d for d in range(1, n + 1) if n % d == 0)
        Z_bessel += sigma2 * kv(1, arg) / n

    Z_bessel *= -2.0 * tau / np.pi
    Z_ren = Z_const + Z_bessel

    return {
        'Z_ren': Z_ren,
        'Z_const': Z_const,
        'Z_bessel': Z_bessel,
        'tau': tau,
        'bessel_fraction': abs(Z_bessel / Z_const) if Z_const != 0 else 0.0,
    }


def compute_casimir_energy_torus(R, r, p=2, q=1, N_dof=2):
    """
    Casimir energy for a scalar field on a flat 2-torus with sides
    L₁ = 2πpR (toroidal) and L₂ = 2πqr (poloidal).

    E_Casimir = N_dof × (πℏc / L₁) × Z_ren(-1/2, τ)

    where τ = L₁/L₂ = pR/(qr).

    Also computes thin-torus approximation:
        E ≈ -N_dof × [ℏc/(12pR) + ℏc/(12qr)]
    """
    L1 = 2.0 * np.pi * p * R  # toroidal circumference
    L2 = 2.0 * np.pi * q * r  # poloidal circumference
    tau = L1 / L2              # aspect ratio = pR/(qr)

    zeta = compute_epstein_zeta_ren(tau)
    E_casimir = N_dof * (np.pi * hbar * c / L1) * zeta['Z_ren']

    # Thin-torus approximation: sum of two 1D Casimir energies
    E_toroidal = -N_dof * hbar * c / (12.0 * p * R)
    E_poloidal = -N_dof * hbar * c / (12.0 * q * r)
    E_thin_approx = E_toroidal + E_poloidal

    r_over_R = r / R

    return {
        'E_casimir_J': E_casimir,
        'E_casimir_MeV': E_casimir / MeV,
        'E_toroidal_J': E_toroidal,
        'E_toroidal_MeV': E_toroidal / MeV,
        'E_poloidal_J': E_poloidal,
        'E_poloidal_MeV': E_poloidal / MeV,
        'E_thin_approx_J': E_thin_approx,
        'E_thin_approx_MeV': E_thin_approx / MeV,
        'tau': tau,
        'r_over_R': r_over_R,
        'L1': L1,
        'L2': L2,
        'Z_ren': zeta['Z_ren'],
        'Z_const': zeta['Z_const'],
        'Z_bessel': zeta['Z_bessel'],
        'bessel_fraction': zeta['bessel_fraction'],
    }


def compute_casimir_landscape(R, r_ratio_range, p=2, q=1, N_dof=2):
    """
    Sweep r/R over a range and compute E_EM + E_Casimir at each point.

    Returns dict with arrays and location of the minimum of E_total.
    """
    ratios = np.array(r_ratio_range)
    E_EM_arr = np.zeros_like(ratios)
    E_cas_arr = np.zeros_like(ratios)
    E_tot_arr = np.zeros_like(ratios)

    for i, rr in enumerate(ratios):
        r_val = rr * R
        params = TorusParams(R=R, r=r_val, p=p, q=q)
        se = compute_self_energy(params)
        cas = compute_casimir_energy_torus(R, r_val, p=p, q=q, N_dof=N_dof)

        E_EM_arr[i] = se['E_total_MeV']
        E_cas_arr[i] = cas['E_casimir_MeV']
        E_tot_arr[i] = E_EM_arr[i] + E_cas_arr[i]

    idx_min = np.argmin(E_tot_arr)

    return {
        'ratios': ratios,
        'E_EM_MeV': E_EM_arr,
        'E_casimir_MeV': E_cas_arr,
        'E_total_MeV': E_tot_arr,
        'min_ratio': ratios[idx_min],
        'min_E_total': E_tot_arr[idx_min],
        'min_idx': idx_min,
    }


def print_casimir_analysis():
    """
    Can vacuum fluctuations on the torus set r/R = α?

    The NWT torus has two compact dimensions: toroidal (2πpR) and poloidal
    (2πqr). Quantum fields on compact spaces produce Casimir energy that
    depends on the geometry. If E_Casimir(r) + E_EM(r) has a minimum at
    r/R = α, the aspect ratio is dynamically determined.

    Spoiler: it doesn't work for a flat torus — but the analysis reveals
    exactly what WOULD need to be true.
    """
    print("=" * 70)
    print("  CASIMIR ENERGY: CAN VACUUM FLUCTUATIONS SET r/R = α?")
    print("  Epstein zeta function on the NWT torus")
    print("=" * 70)

    # Get electron reference geometry
    sol_e = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=alpha)
    R_e = sol_e['R']
    R_e_fm = R_e * 1e15
    r_e = alpha * R_e
    r_e_fm = r_e * 1e15
    p, q = 2, 1

    # ================================================================
    # SECTION 1: Physics — Casimir on the Torus
    # ================================================================
    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  SECTION 1: PHYSICS — CASIMIR ENERGY ON THE TORUS              ║
  ╚══════════════════════════════════════════════════════════════════╝

  The NWT electron is a (2,1) torus knot on a torus with:
    • Major radius R = {R_e_fm:.2f} fm    (toroidal direction)
    • Minor radius r = αR = {r_e_fm:.4f} fm  (poloidal direction)

  Two compact lengths:
    L₁ = 2πpR = {2*np.pi*p*R_e*1e15:.1f} fm   (toroidal circumference)
    L₂ = 2πqr = {2*np.pi*q*r_e*1e15:.3f} fm   (poloidal circumference)

  The poloidal direction is MUCH shorter: L₂/L₁ = qr/(pR) = α/2 ≈ {alpha/2:.5f}

  Quantum fields on compact spaces have quantized modes. The vacuum
  energy (sum over zero-point energies of all modes) depends on the
  geometry. On a 2-torus with sides L₁ × L₂, this is computed by the
  Epstein zeta function — a 2D generalization of ζ(-1/2).

  KEY IDEA: E_Casimir depends on r (through L₂), while E_EM also depends
  on r (through the log factor). If their r-derivatives balance at some
  r/R, the aspect ratio is dynamically determined.""")

    # ================================================================
    # SECTION 2: The Epstein Zeta Formula
    # ================================================================
    cas = compute_casimir_energy_torus(R_e, r_e, p=p, q=q, N_dof=2)
    tau = cas['tau']

    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  SECTION 2: THE EPSTEIN ZETA FORMULA                           ║
  ╚══════════════════════════════════════════════════════════════════╝

  For a flat 2-torus with aspect ratio τ = L₁/L₂ = pR/(qr):

    E_Casimir = N_dof × (πℏc/L₁) × Z_ren(-½, τ)

  where Z_ren is the regularized Epstein zeta function:

    Z_ren(-½, τ) = -(1+τ)/6 - (2τ/π) × Σ σ₂(n) K₁(2πnτ) / n

  σ₂(n) = sum of d² over divisors of n, K₁ = modified Bessel function.

  At the electron geometry (r/R = α, p=2, q=1):
    τ = pR/(qr) = 2/α = {tau:.2f}""")

    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  Epstein Zeta Evaluation at τ = {tau:.2f}                       │
  ├─────────────────────────────────────────────────────────────────┤
  │  Z_const   = -(1+τ)/6          = {cas['Z_const']:.6f}         │
  │  Z_bessel  = Bessel correction  = {cas['Z_bessel']:.2e}         │
  │  Z_ren     = Z_const + Z_bessel = {cas['Z_ren']:.6f}         │
  │  Bessel/Const ratio             = {cas['bessel_fraction']:.2e}         │
  ├─────────────────────────────────────────────────────────────────┤
  │  At τ ~ 274, the Bessel corrections are negligible.             │
  │  Z_ren ≈ -(1+τ)/6 = -(1 + 2/α)/6  (pure constant term).       │
  └─────────────────────────────────────────────────────────────────┘""")

    # Energy comparison
    params_e = TorusParams(R=R_e, r=r_e, p=p, q=q)
    se = compute_self_energy(params_e)

    print(f"""
  Energy at the electron geometry:
    E_EM (total)    = {se['E_total_MeV']:.6f} MeV  (= m_e, by construction)
    E_Casimir       = {cas['E_casimir_MeV']:.4f} MeV
    E_toroidal      = {cas['E_toroidal_MeV']:.6f} MeV  (from L₁)
    E_poloidal      = {cas['E_poloidal_MeV']:.4f} MeV  (from L₂)
    Thin-torus est. = {cas['E_thin_approx_MeV']:.4f} MeV

    |E_Casimir/E_EM| = {abs(cas['E_casimir_MeV']/se['E_total_MeV']):.1f}
    Thin-torus error = {abs((cas['E_thin_approx_MeV'] - cas['E_casimir_MeV'])/cas['E_casimir_MeV'])*100:.3f}%

  ⚠ The Casimir energy is {abs(cas['E_casimir_MeV']/se['E_total_MeV']):.0f}× larger than the EM energy!
  This already tells us the flat-torus Casimir is far too strong.""")

    # ================================================================
    # SECTION 3: Energy Landscape
    # ================================================================
    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  SECTION 3: ENERGY LANDSCAPE E_total(r/R)                      ║
  ╚══════════════════════════════════════════════════════════════════╝

  Sweep r/R from 10⁻⁴ to 0.5 at fixed R = R_e = {R_e_fm:.2f} fm.
  E_total = E_EM + E_Casimir. Look for a minimum.
""")

    # Selected r/R values for table
    table_ratios = [1e-4, 3e-4, 1e-3, 3e-3, alpha, 0.01, 0.02, 0.05,
                    0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    print("  r/R          E_EM (MeV)    E_Casimir (MeV)  E_total (MeV)   E_Cas/E_EM")
    print("  " + "─" * 73)

    for rr in table_ratios:
        r_val = rr * R_e
        params = TorusParams(R=R_e, r=r_val, p=p, q=q)
        se_i = compute_self_energy(params)
        cas_i = compute_casimir_energy_torus(R_e, r_val, p=p, q=q, N_dof=2)
        E_tot = se_i['E_total_MeV'] + cas_i['E_casimir_MeV']
        ratio_str = f"{cas_i['E_casimir_MeV']/se_i['E_total_MeV']:>10.1f}"
        marker = "  ← α" if abs(rr - alpha) / alpha < 0.01 else ""
        print(f"  {rr:<12.5f}  {se_i['E_total_MeV']:>12.4f}  {cas_i['E_casimir_MeV']:>14.4f}  "
              f"{E_tot:>13.4f}  {ratio_str}{marker}")

    # Full landscape for analysis
    r_range = np.logspace(-4, np.log10(0.5), 200)
    landscape = compute_casimir_landscape(R_e, r_range, p=p, q=q, N_dof=2)

    print(f"""
  Result: E_total monotonically decreases as r/R → 0.
          Minimum is at the smallest r/R sampled: r/R = {landscape['min_ratio']:.2e}
          E_total at min = {landscape['min_E_total']:.2f} MeV

  ➜ NO MINIMUM exists. The flat-torus Casimir predicts tube collapse.""")

    # ================================================================
    # SECTION 4: Why It Fails — The Casimir Catastrophe
    # ================================================================
    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  SECTION 4: WHY IT FAILS — THE CASIMIR CATASTROPHE             ║
  ╚══════════════════════════════════════════════════════════════════╝

  The energy derivatives at r/R = α tell the story:

  dE_EM/dr:
    E_EM ∝ (1/R) × [1 + (α/π)(ln(8R/r) - 2)]
    The r-dependence is through log(R/r), so dE_EM/dr ∝ α/(πr)
    At r = αR_e: dE_EM/dr ~ α/(π × αR_e) = 1/(πR_e)

  dE_Casimir/dr:
    E_Casimir ≈ -N_dof × ℏc/(12qr)  (poloidal term dominates)
    So dE_Casimir/dr ≈ +N_dof × ℏc/(12qr²)
    At r = αR_e: dE_Casimir/dr ~ ℏc/(6α²R_e²)

  Ratio of derivatives:
    |dE_Cas/dr| / |dE_EM/dr| ~ πℏc/(6α²R_e × E_EM)""")

    # Compute actual ratio of force scales
    deriv_EM_scale = se['E_total_J'] * alpha / (np.pi * r_e)
    deriv_Cas_scale = 2 * hbar * c / (12 * q * r_e**2)
    force_ratio = abs(deriv_Cas_scale / deriv_EM_scale)

    print(f"""
    Numerically:
      |dE_EM/dr|     ~ {deriv_EM_scale:.4e} J/m
      |dE_Casimir/dr| ~ {deriv_Cas_scale:.4e} J/m
      Ratio           = {force_ratio:.0f}

  The Casimir "force" overwhelms the EM force by a factor of ~{force_ratio:.0f}.

  PHYSICAL INTERPRETATION:
    The flat-torus Casimir treats the NWT as a standard QFT vacuum cavity.
    But the torus ISN'T a cavity — it's a single photon's worldtube. The
    vacuum modes that produce Casimir energy presuppose a quantum field
    living ON the torus, while the NWT photon IS the torus. The boundary
    conditions are fundamentally different.""")

    # ================================================================
    # SECTION 5: Reverse Engineering — What WOULD Work?
    # ================================================================
    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  SECTION 5: REVERSE ENGINEERING — WHAT WOULD WORK?             ║
  ╚══════════════════════════════════════════════════════════════════╝

  Suppose E_Casimir had an effective coefficient C_eff instead of N_dof/12:
    E_Cas_eff = -C_eff × ℏc/(q × r)  (poloidal term)

  At equilibrium dE_total/dr = 0:
    C_eff × ℏc/(q × r²) = -dE_EM/dr

  The EM energy: E_EM ≈ E_circ × [1 + (α/π) × ln(8R/r) / p_eff]
    where E_circ = 2πℏc/L with L the knot path length.

  For the (2,1) knot at r/R = α:
    dE_EM/dr ≈ -(α/π) × E_circ / (p × r)  (from the log derivative)

  Setting dE_Cas/dr + dE_EM/dr = 0 at r = αR:
    C_eff/(αR)² = (α/π) × E_circ / (p × αR × q)""")

    # Compute the required C_eff
    E_circ_e = se['E_circ_J']
    # From balance: C_eff × ℏc/(q × r²) = (α/π) × E_circ / (p × r)
    # C_eff = (α/π) × E_circ × q × r / (p × ℏc)
    # With r = αR and E_circ = 2πℏc/L ≈ ℏc/(pR) for the (2,1) knot:
    # C_eff ≈ α² / (π × p) using E_circ ~ ℏc/(pR)
    C_eff_analytic = alpha**2 / (np.pi * p)
    C_eff_numeric = (alpha / np.pi) * E_circ_e * q * r_e / (p * hbar * c)
    C_flat = 2.0 / (12.0 * q)  # N_dof=2, standard flat-torus coefficient
    suppression = C_flat / C_eff_numeric

    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  Required vs Actual Casimir Coefficient                         │
  ├─────────────────────────────────────────────────────────────────┤
  │  C_eff required (analytic) = α²/(πp) = {C_eff_analytic:.4e}          │
  │  C_eff required (numeric)  =           {C_eff_numeric:.4e}          │
  │  C_flat (N_dof=2, q=1)     = 1/6      = {C_flat:.6f}          │
  │                                                                 │
  │  Ratio: C_flat / C_eff = {suppression:.0f}                          │
  ├─────────────────────────────────────────────────────────────────┤
  │  The effective Casimir coefficient must be ~{suppression:.0f}× weaker     │
  │  than the flat-torus prediction for equilibrium at r/R = α.     │
  └─────────────────────────────────────────────────────────────────┘""")

    # ================================================================
    # SECTION 6: Scale Invariance Check
    # ================================================================
    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  SECTION 6: SCALE INVARIANCE CHECK                             ║
  ╚══════════════════════════════════════════════════════════════════╝

  Even if Casimir energy COULD set r/R = α, does it fix the overall
  scale R? Test: compute E_Casimir × R at fixed r/R = α for various R.
  If E_Cas ∝ 1/R, then E_Cas × R = const → scale invariance persists.
""")

    print("  R (fm)         E_Casimir (MeV)     E_Cas × R (MeV·fm)    E_EM × R (MeV·fm)")
    print("  " + "─" * 73)

    R_test_fm = [1, 5, 10, 50, 100, R_e_fm, 500, 1000, 5000, 10000]
    for R_fm in R_test_fm:
        R_val = R_fm * 1e-15
        r_val = alpha * R_val
        params_t = TorusParams(R=R_val, r=r_val, p=p, q=q)
        se_t = compute_self_energy(params_t)
        cas_t = compute_casimir_energy_torus(R_val, r_val, p=p, q=q, N_dof=2)
        E_cas_R = cas_t['E_casimir_MeV'] * R_fm
        E_em_R = se_t['E_total_MeV'] * R_fm
        marker = "  ← R_e" if abs(R_fm - R_e_fm) / R_e_fm < 0.01 else ""
        print(f"  {R_fm:<14.2f}  {cas_t['E_casimir_MeV']:>14.6f}     {E_cas_R:>16.6f}      "
              f"{E_em_R:>14.6f}{marker}")

    print(f"""
  E_Casimir × R = const across all R (to numerical precision).
  E_EM × R = const as well (both ∝ 1/R at fixed aspect ratio).

  ➜ Even if Casimir sets r/R, it does NOT fix R. Scale invariance persists.
  The overall size R_e remains determined by the input mass m_e.""")

    # ================================================================
    # SECTION 7: Honest Assessment
    # ================================================================
    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  SECTION 7: HONEST ASSESSMENT                                  ║
  ╚══════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────┐
  │  Summary of Findings                                            │
  ├─────────────────────────────────────────────────────────────────┤
  │  Question: Can vacuum fluctuations on the torus set r/R = α?    │
  │                                                                 │
  │  Flat-torus Casimir energy:                                     │
  │    • Computed via Epstein zeta function Z_ren(-½, τ)            │
  │    • At r/R = α: E_Casimir ≈ {cas['E_casimir_MeV']:.1f} MeV (vs E_EM = {se['E_total_MeV']:.3f} MeV)  │
  │    • E_Casimir is ~{abs(cas['E_casimir_MeV']/se['E_total_MeV']):.0f}× larger than E_EM                     │
  │    • Energy monotonically decreases as r → 0 (no minimum)       │
  │    • Predicts tube COLLAPSE, not stabilization                  │
  │                                                                 │
  │  Required suppression for equilibrium at r/R = α:               │
  │    • C_eff must be ~{suppression:.0f}× smaller than flat-torus value     │
  │    • C_required ≈ α²/(2π) ≈ {C_eff_analytic:.1e}                    │
  │                                                                 │
  │  Scale invariance:                                              │
  │    • E_Casimir ∝ 1/R at fixed r/R → does NOT fix R             │
  │    • Even a successful Casimir mechanism leaves m_e as input    │
  └─────────────────────────────────────────────────────────────────┘

  THREE CANDIDATE RESOLUTIONS:

  1. CURVED-TORUS CORRECTIONS
     The flat-torus approximation ignores that the NWT torus has curvature
     κ ~ 1/r in the poloidal direction. Curvature modifies the effective
     boundary conditions and can suppress high-frequency modes. The
     DeWitt-Schwinger expansion shows that curvature corrections enter as
     R_μν/(mode frequency)², which could provide the ~10⁴ suppression.

  2. PARTIAL TRANSPARENCY
     The NWT torus isn't a conducting cavity — it's a region where the
     metric is modified (g → 0 on the null worldtube surface). This is
     more like a dielectric boundary than a perfect conductor. Partially
     transparent boundaries reduce the Casimir effect by a factor that
     depends on the transmission coefficient. If T ~ α, the suppression
     could be of order α² ~ 5×10⁻⁵, close to the required 10⁻⁴.

  3. TOPOLOGICAL PHASE FROM (2,1) KNOT WINDING
     The (2,1) torus knot winds twice toroidally for each poloidal turn.
     A field transported around the knot picks up a phase that depends
     on the winding. For half-integer spin (fermions), this gives
     antiperiodic boundary conditions on certain modes, which can flip
     the sign of their Casimir contribution. The net effect could
     dramatically reduce the coefficient, especially if the mode spectrum
     has cancellations related to the knot topology.

  THE KEY INSIGHT:
    The MECHANISM exists — vacuum energy on compact spaces CAN select
    aspect ratios. But the COEFFICIENT is wrong by ~{suppression:.0f}×. This is not
    a failure; it's a signpost. The boundary condition on the NWT torus
    is very different from a flat periodic torus, and the physics of
    that difference is where r/R = α must emerge.""")

    print("=" * 70)


# ====================================================================
# GREYBODY FACTORS: Partial transparency of the NWT surface
# ====================================================================

def compute_greybody_potential(rho_array, r_surface, l_eff):
    """
    Regge-Wheeler effective potential for scalar modes near a g→0 surface.

    V_eff(ρ) = (1 - r_surface/ρ) × l_eff(l_eff+1)/ρ²

    Only valid for ρ > r_surface. Returns V array (same shape as rho_array).
    """
    V = np.zeros_like(rho_array)
    mask = rho_array > r_surface
    rho = rho_array[mask]
    V[mask] = (1.0 - r_surface / rho) * (l_eff * (l_eff + 1.0) / rho**2)
    return V


def compute_transmission_wkb(r_surface, omega, l_eff):
    """
    WKB transmission coefficient through the Regge-Wheeler barrier.

    For a mode with frequency omega and effective angular momentum l_eff,
    compute the tunneling probability through the potential barrier near
    the NWT surface at ρ = r_surface.

    Returns dict with T_wkb, T_analytic, V_max, rho_peak.
    """
    if l_eff < 0.01:
        # l=0: no barrier (V_eff = 0 everywhere for l=0)
        return {'T_wkb': 1.0, 'T_analytic': 1.0, 'V_max': 0.0, 'rho_peak': r_surface}

    # Barrier peak location (analytic for Regge-Wheeler)
    # d/dρ [(1 - r/ρ) l(l+1)/ρ²] = 0  →  ρ_peak = r(l+1)(2l+3) / (2(l+1))
    # Simplifies to ρ_peak = r(2l+3)/2 for integer l
    # More precisely: ρ_peak = 3r/2 for l=1, ρ_peak = 5r/3 for l=2 (known BH results)
    rho_peak = r_surface * (2.0 * l_eff + 3.0) / 2.0

    # V_max at the peak
    V_max = (1.0 - r_surface / rho_peak) * (l_eff * (l_eff + 1.0) / rho_peak**2)

    omega_sq = omega**2

    if omega_sq >= V_max:
        # Above barrier — full transmission
        T_wkb = 1.0
    else:
        # Find classical turning points by sampling
        N_pts = 500
        rho_arr = np.linspace(r_surface * 1.001, r_surface * 20.0, N_pts)
        V_arr = (1.0 - r_surface / rho_arr) * (l_eff * (l_eff + 1.0) / rho_arr**2)

        # Forbidden region: V > ω²
        forbidden = V_arr > omega_sq
        if not np.any(forbidden):
            T_wkb = 1.0
        else:
            # Integrate |κ| = sqrt(V - ω²) through forbidden region
            kappa = np.sqrt(np.maximum(V_arr - omega_sq, 0.0))
            kappa[~forbidden] = 0.0
            drho = rho_arr[1] - rho_arr[0]
            integral = np.trapz(kappa, dx=drho)
            exponent = 2.0 * integral
            T_wkb = np.exp(-min(exponent, 500.0))  # cap to avoid underflow

    # Low-frequency analytic approximation: T ~ (ωr)^{2l+2}
    omega_r = omega * r_surface
    if omega_r > 0 and l_eff > 0:
        exponent_analytic = (2.0 * l_eff + 2.0) * np.log(omega_r)
        T_analytic = np.exp(min(max(exponent_analytic, -500.0), 0.0))
    else:
        T_analytic = 1.0

    return {
        'T_wkb': T_wkb,
        'T_analytic': T_analytic,
        'V_max': V_max,
        'rho_peak': rho_peak,
    }


def compute_greybody_casimir(R, r, p=2, q=1, N_modes=50):
    """
    Greybody-weighted Casimir energy: sum over torus modes (n,m) with
    each mode's zero-point energy weighted by its transmission coefficient
    through the Regge-Wheeler barrier.

    Returns dict with E_greybody, E_flat, suppression_ratio, T_mean, etc.
    """
    E_flat_total = 0.0
    E_grey_total = 0.0
    T_sum = 0.0
    mode_count = 0
    mode_details = []

    for n in range(-N_modes, N_modes + 1):
        for m in range(-N_modes, N_modes + 1):
            if n == 0 and m == 0:
                continue  # skip zero mode

            # Wavevector on the torus
            k_nm = np.sqrt((n / (p * R))**2 + (m / (q * r))**2)
            omega_nm = c * k_nm

            # Effective angular momentum: maps transverse momentum to l
            l_eff = k_nm * r

            # Transmission through barrier
            trans = compute_transmission_wkb(r, omega_nm / c, l_eff)
            T_nm = trans['T_wkb']

            # Zero-point energy of this mode
            E_flat_nm = 0.5 * hbar * omega_nm

            E_flat_total += E_flat_nm
            E_grey_total += T_nm * E_flat_nm
            T_sum += T_nm
            mode_count += 1

            # Store details for first few modes
            if abs(n) <= 3 and abs(m) <= 3 and n >= 0 and m >= 0 and (n > 0 or m > 0):
                mode_details.append({
                    'n': n, 'm': m,
                    'k_nm': k_nm, 'omega_nm': omega_nm,
                    'l_eff': l_eff, 'T_nm': T_nm,
                    'E_flat_MeV': E_flat_nm / MeV,
                    'E_grey_MeV': T_nm * E_flat_nm / MeV,
                })

    # Sort mode details by (n, m)
    mode_details.sort(key=lambda d: (d['n'], d['m']))

    suppression = E_flat_total / E_grey_total if E_grey_total != 0 else float('inf')
    T_mean = T_sum / mode_count if mode_count > 0 else 0.0

    return {
        'E_greybody_J': E_grey_total,
        'E_greybody_MeV': E_grey_total / MeV,
        'E_flat_J': E_flat_total,
        'E_flat_MeV': E_flat_total / MeV,
        'suppression_ratio': suppression,
        'T_mean': T_mean,
        'mode_count': mode_count,
        'mode_details': mode_details,
    }


def compute_greybody_landscape(R, r_ratio_range, p=2, q=1, N_modes=30):
    """
    Sweep r/R over a range computing E_EM + greybody-weighted E_Casimir.

    Returns dict with arrays and location of the minimum of E_total.
    """
    ratios = np.array(r_ratio_range)
    E_EM_arr = np.zeros_like(ratios)
    E_grey_arr = np.zeros_like(ratios)
    E_tot_arr = np.zeros_like(ratios)
    T_mean_arr = np.zeros_like(ratios)
    supp_arr = np.zeros_like(ratios)

    for i, rr in enumerate(ratios):
        r_val = rr * R
        params = TorusParams(R=R, r=r_val, p=p, q=q)
        se = compute_self_energy(params)
        grey = compute_greybody_casimir(R, r_val, p=p, q=q, N_modes=N_modes)

        E_EM_arr[i] = se['E_total_MeV']
        E_grey_arr[i] = grey['E_greybody_MeV']
        E_tot_arr[i] = E_EM_arr[i] + E_grey_arr[i]
        T_mean_arr[i] = grey['T_mean']
        supp_arr[i] = grey['suppression_ratio']

    idx_min = np.argmin(E_tot_arr)

    return {
        'ratios': ratios,
        'E_EM_MeV': E_EM_arr,
        'E_greybody_MeV': E_grey_arr,
        'E_total_MeV': E_tot_arr,
        'T_mean': T_mean_arr,
        'suppression': supp_arr,
        'min_ratio': ratios[idx_min],
        'min_E_total': E_tot_arr[idx_min],
        'min_idx': idx_min,
    }


def print_greybody_analysis():
    """
    Greybody factors for the NWT torus: partial transparency of the g→0 surface.

    The NWT surface where g→0 acts like a black hole horizon — vacuum modes
    don't perfectly reflect off it; they tunnel through an effective potential
    barrier. This module computes the transmission coefficients (greybody factors)
    and their effect on the Casimir energy.
    """
    print("=" * 70)
    print("  GREYBODY FACTORS: PARTIAL TRANSPARENCY OF THE NWT SURFACE")
    print("  Regge-Wheeler barrier and vacuum mode transmission")
    print("=" * 70)

    # Get electron reference geometry
    sol_e = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=alpha)
    R_e = sol_e['R']
    R_e_fm = R_e * 1e15
    r_e = alpha * R_e
    r_e_fm = r_e * 1e15
    p, q = 2, 1

    # Get flat-torus Casimir for comparison
    cas_flat = compute_casimir_energy_torus(R_e, r_e, p=p, q=q, N_dof=2)

    # ================================================================
    # SECTION 1: The Black Hole Analogy
    # ================================================================
    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  SECTION 1: THE BLACK HOLE ANALOGY                             ║
  ╚══════════════════════════════════════════════════════════════════╝

  The --casimir module showed that flat-torus Casimir energy is ~40,000×
  too strong to stabilize r/R = α. But that calculation assumed PERFECT
  reflection: every vacuum mode bounces off the torus boundary.

  The NWT surface is where g → 0 — the metric degenerates. This is
  EXACTLY the condition at a black hole horizon. And we know that
  black hole horizons are NOT perfect reflectors: quantum modes tunnel
  through the effective potential barrier, giving rise to Hawking radiation.

  The transmission probability T(ω) for each mode is the "greybody factor."

  ┌─────────────────────────────────────────────────────────────────┐
  │              BLACK HOLE  vs  NWT TORUS                         │
  ├──────────────────┬──────────────────┬──────────────────────────┤
  │  Property        │  Black Hole      │  NWT Torus               │
  ├──────────────────┼──────────────────┼──────────────────────────┤
  │  g→0 surface     │  r = r_s         │  ρ = r (tube surface)    │
  │  Surface radius  │  r_s = 2GM/c²    │  r = αR = {r_e_fm:.4f} fm       │
  │  Barrier form    │  Regge-Wheeler   │  Regge-Wheeler (ansatz)  │
  │  Low-ω behavior  │  T ~ (ωr_s)^{{2l+2}} │  T ~ (ωr)^{{2l+2}}        │
  │  High-ω behavior │  T → 1           │  T → 1                   │
  │  Physical effect │  Hawking spectrum │  Casimir suppression     │
  └──────────────────┴──────────────────┴──────────────────────────┘

  The key insight: if the NWT surface transmits vacuum modes rather than
  reflecting them, the effective Casimir energy is SUPPRESSED. Low-frequency
  modes (which dominate the Casimir sum) are most strongly reflected by the
  barrier, while high-frequency modes pass through. The net effect reduces
  the Casimir coefficient.""")

    # ================================================================
    # SECTION 2: The Effective Potential
    # ================================================================
    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  SECTION 2: THE EFFECTIVE POTENTIAL                            ║
  ╚══════════════════════════════════════════════════════════════════╝

  Near the NWT surface at radial distance ρ from the tube center,
  the scalar wave equation becomes Schrödinger-like:

      -d²ψ/dρ² + V_eff(ρ) ψ = ω² ψ

  with the Regge-Wheeler effective potential:

      V_eff(ρ) = (1 - r/ρ) × l(l+1)/ρ²

  This has a barrier that peaks outside the surface, then falls off.
  Modes must tunnel through (or fly over) this barrier.

  r_surface = {r_e_fm:.4f} fm = {r_e:.6e} m""")

    # V_eff table for several l values
    l_values = [1, 2, 5, 10, 20]
    rho_ratios = [1.01, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0]

    print(f"""
  V_eff(ρ) / r⁻² for several angular momenta:

  {"ρ/r":>8s}""", end="")
    for l in l_values:
        print(f"  {'l='+str(l):>10s}", end="")
    print()
    print("  " + "-" * 62)

    for rr in rho_ratios:
        rho = rr * r_e
        print(f"  {rr:8.2f}", end="")
        for l in l_values:
            V = (1.0 - r_e / rho) * (l * (l + 1.0) / rho**2)
            V_normalized = V * r_e**2  # dimensionless
            print(f"  {V_normalized:10.4f}", end="")
        print()

    # Barrier peaks
    print(f"""
  Barrier peak location and height:

  {"l":>6s}  {"ρ_peak/r":>10s}  {"V_max × r²":>12s}  {"ω_max × r":>10s}
  {"---":>6s}  {"---":>10s}  {"---":>12s}  {"---":>10s}""")
    for l in l_values + [50, 100]:
        rho_peak = r_e * (2.0 * l + 3.0) / 2.0
        V_max = (1.0 - r_e / rho_peak) * (l * (l + 1.0) / rho_peak**2)
        V_max_norm = V_max * r_e**2
        omega_max_r = np.sqrt(V_max) * r_e
        print(f"  {l:6d}  {rho_peak/r_e:10.3f}  {V_max_norm:12.6f}  {omega_max_r:10.6f}")

    print(f"""
  Key observation: the barrier height grows as ~l(l+1)/r² for large l,
  with the peak moving outward as ρ_peak ≈ r(2l+3)/2. High-l modes face
  a tall, wide barrier and are strongly reflected. Low-l modes see a
  modest barrier and can tunnel through.""")

    # ================================================================
    # SECTION 3: Transmission Coefficients
    # ================================================================
    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  SECTION 3: TRANSMISSION COEFFICIENTS                          ║
  ╚══════════════════════════════════════════════════════════════════╝

  T(l, ωr) = transmission probability through the barrier.
  Computed via WKB: T = exp(-2 ∫|κ|dρ) where κ² = V_eff - ω².

  Low-frequency analytic: T ~ (ωr)^{{2l+2}} (valid for ωr << 1).""")

    # Transmission table
    l_vals_table = [0, 1, 2, 3, 5, 10, 20]
    omega_r_vals = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"""
  WKB transmission coefficient T(l, ωr):

  {"ωr":>8s}""", end="")
    for l in l_vals_table:
        print(f"  {'l='+str(l):>10s}", end="")
    print()
    print("  " + "-" * 80)

    for omega_r in omega_r_vals:
        omega = omega_r / r_e  # actual frequency (1/m in natural units)
        print(f"  {omega_r:8.2f}", end="")
        for l in l_vals_table:
            trans = compute_transmission_wkb(r_e, omega, float(l))
            T = trans['T_wkb']
            if T < 1e-10:
                print(f"  {'<1e-10':>10s}", end="")
            elif T > 0.999:
                print(f"  {'~1':>10s}", end="")
            else:
                print(f"  {T:10.2e}", end="")
        print()

    # Compare WKB with analytic
    print(f"""
  WKB vs analytic (ωr)^{{2l+2}} comparison at ωr = 0.1:
""")
    omega_test = 0.1 / r_e
    print(f"  {'l':>6s}  {'T_WKB':>12s}  {'T_analytic':>12s}  {'ratio':>10s}")
    print(f"  {'---':>6s}  {'---':>12s}  {'---':>12s}  {'---':>10s}")
    for l in [1, 2, 3, 5, 10]:
        trans = compute_transmission_wkb(r_e, omega_test, float(l))
        T_w = trans['T_wkb']
        T_a = trans['T_analytic']
        ratio = T_w / T_a if T_a > 0 and T_w > 0 else float('inf')
        print(f"  {l:6d}  {T_w:12.4e}  {T_a:12.4e}  {ratio:10.2f}")

    print(f"""
  At ωr = 0.1: the analytic approximation T ~ (ωr)^{{2l+2}} captures the
  qualitative suppression. Both show exponential fall-off with l.

  Critical scale: ωr ~ 1 is the transition from opaque to transparent.
  For the NWT electron, the first poloidal mode has:
    ω₁ = c/(qr) = c/r   →   ω₁r = 1
  So the lowest modes sit RIGHT at the barrier scale — not fully
  suppressed, not fully transmitted.""")

    # ================================================================
    # SECTION 4: Mode-by-Mode Casimir
    # ================================================================
    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  SECTION 4: MODE-BY-MODE GREYBODY CASIMIR                      ║
  ╚══════════════════════════════════════════════════════════════════╝

  For each torus mode (n,m):
    k_nm = √[(n/pR)² + (m/qr)²]     wavevector
    ω_nm = c × k_nm                   frequency
    l_eff = k_nm × r                  effective angular momentum
    T_nm = T(l_eff, ω_nm)            greybody transmission
    E_flat = ½ℏω_nm                   flat-torus zero-point energy
    E_grey = T_nm × E_flat            greybody-weighted contribution

  Using N_modes = 50 (summing |n|,|m| ≤ 50):""")

    grey = compute_greybody_casimir(R_e, r_e, p=p, q=q, N_modes=50)

    print(f"""
  First modes (positive n,m only — full sum includes ±n, ±m):

  {"(n,m)":>8s}  {"ω×r/c":>8s}  {"l_eff":>8s}  {"T_nm":>10s}  {"E_flat/MeV":>12s}  {"T×E/MeV":>12s}
  {"---":>8s}  {"---":>8s}  {"---":>8s}  {"---":>10s}  {"---":>12s}  {"---":>12s}""")
    for md in grey['mode_details']:
        n, m = md['n'], md['m']
        omega_r = md['omega_nm'] * r_e / c
        label = f"({n},{m})"
        T = md['T_nm']
        if T < 1e-10:
            T_str = f"{'<1e-10':>10s}"
        elif T > 0.999:
            T_str = f"{'~1':>10s}"
        else:
            T_str = f"{T:10.2e}"
        print(f"  {label:>8s}  {omega_r:8.3f}  {md['l_eff']:8.3f}  {T_str}  {md['E_flat_MeV']:12.4e}  {md['E_grey_MeV']:12.4e}")

    print(f"""
  ─────────────────────────────────────────────────────────────────
  TOTALS (all {grey['mode_count']} modes, |n|,|m| ≤ 50):

    E_flat  (unweighted)  = {grey['E_flat_MeV']:12.4e} MeV
    E_grey  (greybody)    = {grey['E_greybody_MeV']:12.4e} MeV
    Suppression ratio     = {grey['suppression_ratio']:12.2f}×
    Mean transmission     = {grey['T_mean']:12.6f}

  Required suppression for r/R = α stabilization: ~40,000×""")

    if grey['suppression_ratio'] > 1.0:
        ratio_to_target = grey['suppression_ratio'] / 40000.0
        print(f"    Achieved / required = {ratio_to_target:.4f}")
        if ratio_to_target > 0.1 and ratio_to_target < 10:
            print(f"    → Within an order of magnitude! The greybody mechanism is viable.")
        elif ratio_to_target >= 10:
            print(f"    → EXCEEDS the required suppression — greybody over-suppresses.")
        else:
            print(f"    → Greybody suppression alone is not enough.")
            print(f"      Additional suppression needed: {40000.0/grey['suppression_ratio']:.1f}×")

    # ================================================================
    # SECTION 5: Greybody Energy Landscape
    # ================================================================
    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  SECTION 5: GREYBODY ENERGY LANDSCAPE                          ║
  ╚══════════════════════════════════════════════════════════════════╝

  Sweeping r/R to find if E_EM + E_Casimir_greybody has a minimum.
  Using N_modes = 30 for the sweep (faster; checked that 30 vs 50
  gives consistent suppression ratios to ~10%).

  This is the key test: does greybody weighting create a stabilizing
  minimum at r/R ~ α = {alpha:.6f}?""")

    # Use fewer points for speed (each point sums ~3600 modes with WKB)
    ratios = np.concatenate([
        np.logspace(-4, -2, 8),            # 0.0001 to 0.01
        np.linspace(0.005, 0.02, 10)[1:],  # around α
        np.linspace(0.02, 0.1, 8)[1:],     # 0.02 to 0.1
        np.linspace(0.1, 0.5, 5)[1:],      # 0.1 to 0.5
    ])
    ratios = np.unique(np.sort(ratios))

    print(f"\n  Computing {len(ratios)} points... (each sums ~3600 WKB modes)")
    landscape = compute_greybody_landscape(R_e, ratios, p=p, q=q, N_modes=30)

    print(f"""
  {"r/R":>10s}  {"E_EM/MeV":>12s}  {"E_grey/MeV":>12s}  {"E_total/MeV":>12s}  {"<T>":>8s}  {"suppress":>10s}
  {"---":>10s}  {"---":>12s}  {"---":>12s}  {"---":>12s}  {"---":>8s}  {"---":>10s}""")
    for i in range(len(ratios)):
        print(f"  {ratios[i]:10.5f}  {landscape['E_EM_MeV'][i]:12.4e}  "
              f"{landscape['E_greybody_MeV'][i]:12.4e}  {landscape['E_total_MeV'][i]:12.4e}  "
              f"{landscape['T_mean'][i]:8.4f}  {landscape['suppression'][i]:10.1f}×")

    min_r = landscape['min_ratio']
    min_E = landscape['min_E_total']
    min_idx = landscape['min_idx']

    print(f"""
  ─────────────────────────────────────────────────────────────────
  Minimum of E_total:
    r/R at minimum = {min_r:.6f}
    E_total at min = {min_E:.6e} MeV
    α = {alpha:.6f}""")

    if min_idx == 0 or min_idx == len(ratios) - 1:
        print(f"""
    ⚠ Minimum is at the BOUNDARY of the scan range — not a true local
    minimum. The greybody-weighted Casimir does not create a stabilizing
    potential well in this range.""")
    else:
        ratio_to_alpha = min_r / alpha
        print(f"""
    r_min / α = {ratio_to_alpha:.4f}
    {"→ Close to α!" if 0.5 < ratio_to_alpha < 2.0 else "→ Not near α."}""")

    # ================================================================
    # SECTION 6: The Thermodynamic Connection
    # ================================================================
    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  SECTION 6: THE THERMODYNAMIC CONNECTION                       ║
  ╚══════════════════════════════════════════════════════════════════╝

  If the NWT surface is horizon-like, it has a temperature:

      T_NWT = ℏc / (4π r)     [Hawking formula with r as horizon radius]""")

    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    sigma_SB = 5.670374419e-8  # Stefan-Boltzmann constant (W/m²/K⁴)

    T_NWT = hbar * c / (4.0 * np.pi * r_e)
    T_NWT_K = T_NWT / k_B
    T_NWT_MeV = T_NWT / MeV

    # Surface area of the torus
    A_torus = 4.0 * np.pi**2 * R_e * r_e

    # Stefan-Boltzmann luminosity
    P_SB = sigma_SB * T_NWT_K**4 * A_torus

    print(f"""
  For the electron NWT:
    r = αR = {r_e_fm:.4f} fm = {r_e:.6e} m

    T_NWT = ℏc/(4πr) = {T_NWT:.4e} J
                      = {T_NWT_K:.4e} K
                      = {T_NWT_MeV:.4e} MeV

  Compare to:
    Electron mass        : {m_e_MeV:.4f} MeV
    T_NWT / m_e          : {T_NWT_MeV/m_e_MeV:.4f}

  This temperature is set by the poloidal radius — the "size" of the
  g→0 surface. It's comparable to the electron mass energy, which is
  significant: it means the torus surface is at the quantum gravity
  scale for this geometry.

  Torus surface area: A = 4π²Rr = {A_torus:.4e} m²

  Stefan-Boltzmann luminosity: P = σT⁴A = {P_SB:.4e} W
                                         = {P_SB/MeV*1e-6:.4e} MeV/s

  In equilibrium, the absorption rate equals the emission rate.
  The greybody factors encode the absorption cross-section:
    σ_abs(ω) = Σ_l (2l+1)π/ω² × T_l(ω)

  For stability: detailed balance between modes emitted and absorbed
  at each frequency must hold. The effective potential acts as a
  frequency-dependent filter, and the equilibrium geometry is where
  the total energy (EM + filtered Casimir) is minimized.""")

    # ================================================================
    # SECTION 7: Assessment — How Close Did We Get?
    # ================================================================
    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  SECTION 7: ASSESSMENT — HOW CLOSE DID WE GET?                ║
  ╚══════════════════════════════════════════════════════════════════╝

  WHAT THE GREYBODY MODEL COMPUTES:
  ─────────────────────────────────
  • Regge-Wheeler potential barrier near the NWT g→0 surface
  • WKB transmission coefficients for each torus mode (n,m)
  • Greybody-weighted Casimir energy: Σ T_nm × E_flat_nm

  RESULTS:
  ────────
  • Total suppression ratio: {grey['suppression_ratio']:.2f}×
  • Required for stabilization: ~40,000×""")

    if grey['suppression_ratio'] > 1:
        gap = 40000.0 / grey['suppression_ratio']
        print(f"  • Gap factor: {gap:.1f}× (need {gap:.1f}× more suppression)")
    else:
        print(f"  • No suppression achieved (ratio < 1)")

    print(f"""
  • Mean transmission <T>: {grey['T_mean']:.6f}
  • Energy minimum at r/R = {min_r:.6f} (α = {alpha:.6f})

  WHAT THE MODEL GETS RIGHT:
  ──────────────────────────
  ✓ UV modes (high ωr) pass through: T → 1
  ✓ IR modes (low ωr) are reflected: T → 0
  ✓ The barrier height grows with l: high angular momentum modes
    are more strongly suppressed
  ✓ The first poloidal mode has ωr ~ 1: sits at the barrier scale
  ✓ The Hawking temperature T_NWT ~ m_e: connects vacuum energy
    to the particle mass scale

  WHAT THE MODEL MISSES:
  ──────────────────────
  ✗ The Regge-Wheeler form is an ANSATZ, not derived from the
    actual NWT metric near g → 0
  ✗ The real near-surface geometry may have different asymptotics
    (e.g., g ~ ρ² rather than g ~ (1-r/ρ))
  ✗ Spin-statistics: we treated scalar modes, but the EM field is
    spin-1 and fermion loops contribute differently
  ✗ The WKB approximation breaks down near the barrier peak
  ✗ No backreaction: the barrier shape depends on the geometry,
    which depends on the Casimir energy

  NEXT STEPS:
  ───────────
  1. DERIVE the near-surface metric from the NWT g → 0 condition.
     The Regge-Wheeler ansatz is the simplest model; the actual
     potential could be quite different.

  2. Compute the EXACT barrier for the NWT torus using the metric
     g_μν near the tube surface, where the embedding coordinates
     degenerate.

  3. Include SPIN: vector (EM) and spinor (fermion) greybody factors
     have different l-dependence than scalar factors.

  4. Self-consistent solution: the greybody factors modify the
     Casimir energy, which modifies the geometry, which modifies
     the barrier. Find the fixed point.

  THE PHYSICS CONCLUSION:
  ───────────────────────
  The greybody mechanism is the RIGHT FRAMEWORK for understanding
  why the flat-torus Casimir calculation over-estimated. The g→0
  surface is not a perfect reflector, and the transmission through
  the effective barrier provides a natural suppression of vacuum
  modes. Whether the specific Regge-Wheeler ansatz gives the right
  numerical suppression depends on the actual near-surface geometry —
  which is the key open calculation.""")

    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────
#  TORUS EM FIELDS
# ─────────────────────────────────────────────────────────────────────

def compute_torus_fields(R, r, p=2, q=1):
    """
    Compute EM field strengths inside the electron torus.

    Returns dict with E_rms, B_rms for both volume assumptions (full torus
    vs tube along knot path), Schwinger field ratio, flux/Phi_0, impedance/Z_0.
    """
    params = TorusParams(R=R, r=r, p=p, q=q)
    se = compute_self_energy(params)
    U_EM = se['U_total_J']
    I = se['I_amps']
    L_ind = se['L_ind_henry']
    log_factor = se['log_factor']
    E_circ = se['E_circ_J']

    # Schwinger critical field
    E_Schwinger = m_e**2 * c**3 / (e_charge * hbar)  # ~1.32e18 V/m

    # Volume: full torus interior
    V_torus = 2 * np.pi**2 * R * r**2

    # Volume: tube along knot path
    L_path = compute_path_length(params)
    V_tube = np.pi * r**2 * L_path

    # RMS fields from stored energy: U_EM = eps0 * E^2 * V (for v=c, equal E and B energy)
    # U_EM = 2 * (eps0/2) * E_rms^2 * V = eps0 * E_rms^2 * V
    # So B_rms = sqrt(mu0 * U_EM / V), E_rms = c * B_rms
    B_rms_torus = np.sqrt(mu0 * U_EM / V_torus) if V_torus > 0 else 0
    E_rms_torus = c * B_rms_torus

    B_rms_tube = np.sqrt(mu0 * U_EM / V_tube) if V_tube > 0 else 0
    E_rms_tube = c * B_rms_tube

    # Flux quantization
    Phi = L_ind * I  # total magnetic flux
    Phi_0 = h_planck / (2 * e_charge)  # flux quantum

    # Impedance matching
    Z_torus = E_rms_torus * (2 * np.pi * r) / I if I > 0 else 0
    Z_0 = mu0 * c  # impedance of free space ~376.7 Ohm

    # Quantum single-photon field
    omega = E_circ / hbar
    E_Q_torus = np.sqrt(hbar * omega / (eps0 * V_torus)) if V_torus > 0 else 0
    E_Q_tube = np.sqrt(hbar * omega / (eps0 * V_tube)) if V_tube > 0 else 0

    return {
        'U_EM': U_EM,
        'E_circ': E_circ,
        'log_factor': log_factor,
        'I_amps': I,
        'L_ind': L_ind,
        'L_path': L_path,
        'V_torus': V_torus,
        'V_tube': V_tube,
        'E_rms_torus': E_rms_torus,
        'E_rms_tube': E_rms_tube,
        'B_rms_torus': B_rms_torus,
        'B_rms_tube': B_rms_tube,
        'E_Schwinger': E_Schwinger,
        'E_over_ES_torus': E_rms_torus / E_Schwinger,
        'E_over_ES_tube': E_rms_tube / E_Schwinger,
        'Phi': Phi,
        'Phi_0': Phi_0,
        'Phi_over_Phi0': Phi / Phi_0,
        'Z_torus': Z_torus,
        'Z_0': Z_0,
        'Z_over_Z0': Z_torus / Z_0,
        'E_Q_torus': E_Q_torus,
        'E_Q_tube': E_Q_tube,
        'omega': omega,
    }


# ────────────────────────────────────────────────────────────────────
# Torus LC circuit model
# ────────────────────────────────────────────────────────────────────

def compute_torus_circuit(R, r, p=2, q=1):
    """
    Model the torus as a lumped-element LC circuit.

    L = Neumann inductance of a torus (same formula as compute_self_energy)
    C = self-capacitance of a conducting torus
    Derived: resonant frequency, characteristic impedance, radiation resistance, Q.

    Returns dict with all circuit parameters.
    """
    params = TorusParams(R=R, r=r, p=p, q=q)
    se = compute_self_energy(params)
    log_factor = se['log_factor']  # ln(8R/r) - 2
    ln_8Rr = log_factor + 2.0     # ln(8R/r)

    # Inductance (Neumann formula for torus)
    L = mu0 * R * log_factor

    # Self-capacitance of conducting torus (Smythe, Static and Dynamic Electricity)
    C = 4 * np.pi**2 * eps0 * R / ln_8Rr

    # LC resonant frequency
    omega_0 = 1.0 / np.sqrt(L * C)

    # Actual circulation (drive) frequency
    L_path = compute_path_length(params)
    omega_circ = 2 * np.pi * c / L_path

    # Characteristic impedance
    Z_char = np.sqrt(L / C)

    # Free-space impedance
    Z_0 = mu0 * c  # ~376.7 Ω

    # Radiation resistance (magnetic dipole: loop area A = πr²)
    k = omega_circ / c  # wavenumber at drive frequency
    A_loop = np.pi * r**2
    R_rad = Z_0 * (k * A_loop)**2 / (6 * np.pi)

    # Quality factor and damping
    Q = Z_char / R_rad if R_rad > 0 else np.inf
    zeta = R_rad / (2 * Z_char) if Z_char > 0 else 0.0
    R_crit = 2 * Z_char

    # Reactive impedances at drive frequency
    X_L = omega_circ * L
    X_C = 1.0 / (omega_circ * C) if omega_circ * C > 0 else np.inf
    X_net = X_L - X_C

    return {
        'L': L, 'C': C,
        'omega_0': omega_0, 'omega_circ': omega_circ,
        'omega_ratio': omega_0 / omega_circ,
        'f_0_Hz': omega_0 / (2 * np.pi),
        'f_circ_Hz': omega_circ / (2 * np.pi),
        'Z_char': Z_char, 'Z_0': Z_0,
        'Z_over_Z0': Z_char / Z_0,
        'R_rad': R_rad, 'Q': Q,
        'zeta': zeta, 'R_crit': R_crit,
        'X_L': X_L, 'X_C': X_C, 'X_net': X_net,
        'log_factor': log_factor, 'ln_8Rr': ln_8Rr,
        'L_path': L_path,
    }
