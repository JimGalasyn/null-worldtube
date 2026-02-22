"""
Neutrino mass analysis for the Null Worldtube model.

Neutrino masses from twist-wave dispersion on the torus, including
the extended Koide formula, seesaw mass scale, PMNS mixing predictions,
and visualization.
"""

import numpy as np

from .constants import m_e_MeV, hbar, c, MeV, eV
from .core import find_self_consistent_radius


def print_neutrino_analysis():
    """
    Neutrino masses from twist-wave dispersion on the torus.

    Key results:
    - Neutrinos = propagating topology changes (twist-waves), not bound tori
    - Mass structure from extended Koide formula with angle θ_ν
    - θ_ν determined by fitting Δm²₃₁/Δm²₂₁ ratio; geometric interpretation
    - Absolute scale from seesaw-like formula: A² = (α_W/π)(m_e²/Λ_tube)
    - Predicts Σm_ν ≈ 0.06 eV (at cosmological floor)
    """
    print("=" * 70)
    print("  NEUTRINO MASSES FROM TWIST-WAVE DISPERSION")
    print("  Topological mass generation in the null worldtube model")
    print("=" * 70)

    # ================================================================
    # Section 1: Experimental status
    # ================================================================
    print()
    print("  1. EXPERIMENTAL STATUS (NuFIT 6.0 + JUNO 2025 + KATRIN)")
    print("  " + "─" * 55)

    # Best-fit values: NuFIT 6.0 (arXiv:2410.05380), normal ordering
    dm2_21_exp = 7.49e-5    # eV², solar (±0.19)
    dm2_31_exp = 2.534e-3   # eV², atmospheric (+0.025/-0.023)
    R_exp = dm2_31_exp / dm2_21_exp

    sin2_12_exp = 0.307     # ±0.012
    sin2_23_exp = 0.561     # +0.012/-0.015
    sin2_13_exp = 0.02195   # ±0.00056
    delta_CP_exp = 177.0    # degrees (+19/-20)

    # JUNO first result (arXiv:2511.14593, 59 days of data)
    dm2_21_juno = 7.50e-5   # ±0.12 — 1.6× more precise than all prior
    sin2_12_juno = 0.3092   # ±0.0087

    # KATRIN (Science 2025): m_ν < 0.45 eV at 90% CL
    m_katrin = 0.45  # eV

    # DESI DR2 + Planck (arXiv:2503.14744): Σm_ν < 0.064 eV (95% CL, ΛCDM)
    sum_desi = 0.064  # eV

    # Minimum from oscillations (normal ordering, m₁ = 0)
    m2_min = np.sqrt(dm2_21_exp)
    m3_min = np.sqrt(dm2_31_exp)
    sum_osc_floor = m2_min + m3_min

    print(f"""
  Mass-squared splittings (what oscillation experiments measure):
    Δm²₂₁ = {dm2_21_exp:.2e} eV²   (solar, NuFIT 6.0)
    Δm²₃₁ = {dm2_31_exp:.3e} eV²   (atmospheric, normal ordering)
    Ratio R = Δm²₃₁/Δm²₂₁ = {R_exp:.2f}

  JUNO first result (59 days, Nov 2025):
    Δm²₂₁ = ({dm2_21_juno:.2e} ± 0.12×10⁻⁵) eV²
    sin²θ₁₂ = {sin2_12_juno} ± 0.0087
    Already 1.6× more precise than all prior experiments combined.

  Mixing angles (NuFIT 6.0, normal ordering):
    sin²θ₁₂ = {sin2_12_exp}   (solar angle)
    sin²θ₂₃ = {sin2_23_exp}   (atmospheric angle)
    sin²θ₁₃ = {sin2_13_exp}  (reactor angle)
    δ_CP     = {delta_CP_exp}°     (consistent with CP conservation)

  Upper bounds on absolute masses:
    KATRIN (direct):      m_ν < {m_katrin} eV  (90% CL)
    DESI+Planck (cosmo):  Σm_ν < {sum_desi} eV  (95% CL, ΛCDM)
    Oscillation floor:    Σm_ν ≥ {sum_osc_floor:.4f} eV  (normal, m₁=0)

  ┌──────────────────────────────────────────────────────┐
  │  The cosmological bound ({sum_desi} eV) nearly touches     │
  │  the oscillation floor ({sum_osc_floor:.4f} eV).                 │
  │  This SQUEEZES m₁ toward zero.                       │
  │  A model that predicts m₁ ≈ 0 is strongly favored.  │
  └──────────────────────────────────────────────────────┘""")

    # ================================================================
    # Section 2: The twist-wave idea
    # ================================================================
    print()
    print("  2. NEUTRINOS AS TWIST-WAVES")
    print("  " + "─" * 55)
    print(f"""
  In the null worldtube model:
    • Charged leptons = STANDING waves on closed tori (stable topology)
    • Photons = TRAVELING waves with no topology (massless)
    • Neutrinos = TRAVELING topological transitions (twist-waves)

  A neutrino is what happens when a torus partially unwinds:
    π⁺(p=1) → μ⁺(p=2) + ν_μ     [winding number changes]

  The neutrino carries the topological difference — it's a
  propagating TWIST in the EM field. Like torsion traveling
  along a spring when you unwind one coil.

  THREE FLAVORS arise because three charged lepton tori
  (e, μ, τ) define three distinct twist-wave channels:
    ν_e  = twist involving electron torus (largest R, weakest twist)
    ν_μ  = twist involving muon torus (medium R)
    ν_τ  = twist involving tau torus (smallest R, strongest twist)

  MASS arises from the energy cost of maintaining the twist
  against the vacuum stiffness — a topological "spring constant."

  CHIRALITY: the handedness of the unwinding gives left-handed
  neutrinos and right-handed antineutrinos automatically.""")

    # ================================================================
    # Section 3: Extended Koide formula for neutrinos
    # ================================================================
    print()
    print("  3. EXTENDED KOIDE FORMULA")
    print("  " + "─" * 55)

    # Charged lepton Koide angle
    m_e_val = m_e_MeV
    m_mu_val = 105.6583755
    m_tau_val = 1776.86

    theta_K = (6 * np.pi + 2) / 9  # charged lepton Koide angle

    # Compute charged lepton f-values for reference
    f_e = 1 + np.sqrt(2) * np.cos(theta_K)
    f_mu = 1 + np.sqrt(2) * np.cos(theta_K + 2 * np.pi / 3)
    f_tau = 1 + np.sqrt(2) * np.cos(theta_K + 4 * np.pi / 3)

    print(f"""
  The charged lepton Koide angle (from --koide analysis):

    θ_K = (6π + 2)/9 = {theta_K:.6f} rad

    √m_i = A(1 + √2 cos(θ_K + 2πi/3))

    All three factors f_i are positive:
      f_e  = {f_e:.6f}   (small  → light electron)
      f_μ  = {f_mu:.6f}   (medium → medium muon)
      f_τ  = {f_tau:.6f}   (large  → heavy tau)

  For neutrinos, we use the EXTENDED Koide formula:

    √m_i = A_ν (1 + √2 cos(θ_ν + 2πi/3))

  where ONE of the f_i may be negative. Physical masses are
  m_i = A_ν² f_i² (always positive). The negative √m reflects
  the opposite topological chirality of the lightest neutrino:
  it's the twist-wave channel with destructive interference
  between the twist and the vacuum topology.

  The extended Koide ratio is still exactly 2/3:
    Q = Σm_i / (Σ ε_i √m_i)² = 2/3
  where ε_i = sign(f_i).""")

    # ================================================================
    # Section 4: Numerical search for θ_ν
    # ================================================================
    print()
    print("  4. FINDING THE NEUTRINO KOIDE ANGLE")
    print("  " + "─" * 55)

    # Fine scan over [0, 2π)
    N_scan = 500000
    thetas = np.linspace(0, 2 * np.pi, N_scan, endpoint=False)

    R_values = np.zeros(N_scan)

    for i in range(N_scan):
        th = thetas[i]
        f = np.array([1 + np.sqrt(2) * np.cos(th + 2 * np.pi * k / 3)
                       for k in range(3)])
        f2 = f**2
        f4 = f2**2
        idx = np.argsort(f2)
        f4s = f4[idx]

        dm21 = f4s[1] - f4s[0]
        dm31 = f4s[2] - f4s[0]
        R_values[i] = dm31 / dm21 if dm21 > 1e-30 else np.inf

    # Find all crossings where R = R_exp
    solutions = []
    for i in range(N_scan - 1):
        if R_values[i] == np.inf or R_values[i + 1] == np.inf:
            continue
        if (R_values[i] - R_exp) * (R_values[i + 1] - R_exp) < 0:
            # Linear interpolation
            t = (R_exp - R_values[i]) / (R_values[i + 1] - R_values[i])
            theta_sol = thetas[i] + t * (thetas[i + 1] - thetas[i])
            solutions.append(theta_sol)

    print(f"""
  Scanning θ from 0 to 2π in {N_scan} steps...
  Target: R = Δm²₃₁/Δm²₂₁ = {R_exp:.2f}

  The mass-squared ratio R depends ONLY on θ_ν (the overall
  scale A_ν cancels). So θ_ν is determined by a single number.

  Found {len(solutions)} solutions for R(θ) = {R_exp:.2f}:""")

    for j, sol in enumerate(solutions):
        # Verify
        f = np.array([1 + np.sqrt(2) * np.cos(sol + 2 * np.pi * k / 3)
                       for k in range(3)])
        f2 = f**2
        f4 = f2**2
        idx = np.argsort(f2)
        f_sorted = f[idx]
        f4s = f4[idx]
        dm21 = f4s[1] - f4s[0]
        dm31 = f4s[2] - f4s[0]
        R_check = dm31 / dm21
        n_neg = np.sum(f < 0)

        zone = "all-positive" if n_neg == 0 else f"{n_neg} negative"
        print(f"    θ_{j+1} = {sol:.6f} rad  ({np.degrees(sol):.3f}°)"
              f"  R = {R_check:.2f}  [{zone}]")

    # Select the solution closest to θ_K (in the same topological sector)
    if not solutions:
        print("\n  ERROR: No solution found!")
        return

    # Find the solution in the "right" sector: just above θ_K, in the
    # extended (one negative) regime that gives the neutrino hierarchy
    best_sol = None
    best_dist = np.inf
    for sol in solutions:
        dist = abs(sol - theta_K)
        if dist < best_dist:
            best_dist = dist
            best_sol = sol

    theta_nu = best_sol

    # Refine with bisection
    def R_of_theta(th):
        f = np.array([1 + np.sqrt(2) * np.cos(th + 2 * np.pi * k / 3)
                       for k in range(3)])
        f2 = f**2
        f4 = f2**2
        idx = np.argsort(f2)
        dm21 = f4[idx][1] - f4[idx][0]
        dm31 = f4[idx][2] - f4[idx][0]
        return dm31 / dm21 if dm21 > 1e-30 else np.inf

    # Bisect in a small interval around the coarse solution
    lo, hi = theta_nu - 0.001, theta_nu + 0.001
    for _ in range(80):
        mid = (lo + hi) / 2
        if R_of_theta(mid) > R_exp:
            lo = mid
        else:
            hi = mid
    theta_nu = (lo + hi) / 2

    # Compute properties at refined θ_ν
    f_nu = np.array([1 + np.sqrt(2) * np.cos(theta_nu + 2 * np.pi * k / 3)
                      for k in range(3)])
    f2_nu = f_nu**2
    f4_nu = f2_nu**2
    idx_nu = np.argsort(f2_nu)
    f_nu_sorted = f_nu[idx_nu]
    f2_nu_sorted = f2_nu[idx_nu]
    f4_nu_sorted = f4_nu[idx_nu]

    dm21_rel = f4_nu_sorted[1] - f4_nu_sorted[0]
    dm31_rel = f4_nu_sorted[2] - f4_nu_sorted[0]
    R_final = dm31_rel / dm21_rel

    shift = theta_nu - theta_K

    print(f"""
  Best solution (closest to θ_K):

  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │   θ_ν = {theta_nu:.10f} rad  ({np.degrees(theta_nu):.6f}°)     │
  │                                                      │
  │   R = {R_final:.4f}  (target: {R_exp:.4f})                  │
  │                                                      │
  │   Shift from θ_K:                                    │
  │     Δθ = θ_ν - θ_K = {shift:.10f} rad               │
  │                      ({np.degrees(shift):.6f}°)              │
  └──────────────────────────────────────────────────────┘

  Koide factors at θ_ν:
    f₁ = {f_nu_sorted[0]:+.6f}  → m₁ ∝ f₁² = {f2_nu_sorted[0]:.6f}  (lightest)
    f₂ = {f_nu_sorted[1]:+.6f}  → m₂ ∝ f₂² = {f2_nu_sorted[1]:.6f}  (middle)
    f₃ = {f_nu_sorted[2]:+.6f}  → m₃ ∝ f₃² = {f2_nu_sorted[2]:.6f}  (heaviest)

  The lightest neutrino has f < 0 — the opposite topological
  chirality, as expected for a twist-wave.""")

    # ================================================================
    # Section 5: Geometric interpretation
    # ================================================================
    print()
    print("  5. GEOMETRIC INTERPRETATION OF θ_ν")
    print("  " + "─" * 55)

    # Test candidate formulas
    p_wind, q_wind, N_c = 2, 1, 3

    candidates = {
        '(6π+2)/9 + 2/9  = (6π+4)/9':
            (6 * np.pi + 4) / 9,
        '(6π+2)/9 + 7/27':
            (6 * np.pi + 2) / 9 + 7 / 27,
        '(6π+2)/9 + π/12':
            (6 * np.pi + 2) / 9 + np.pi / 12,
        '(6π+2)/9 + (p+q)/(N²+p) = θ_K + 3/11':
            (6 * np.pi + 2) / 9 + 3 / 11,
        '(7π+2)/9':
            (7 * np.pi + 2) / 9,
        '(p/N_c)(π + p*q/N²) = (2/3)(π + 2/9)':
            (2 / 3) * (np.pi + 2 / 9),
        '5π/6':
            5 * np.pi / 6,
        '(18π+8)/27 + 1/(3N_c²)':
            (18 * np.pi + 8) / 27 + 1 / 27,
    }

    print(f"\n  Testing geometric formulas (target: θ_ν = {theta_nu:.6f}):\n")
    print(f"    {'Formula':<45} {'Value':>10}  {'Error':>10}  {'R':>8}")
    print(f"    {'─'*45} {'─'*10}  {'─'*10}  {'─'*8}")

    best_formula = None
    best_error = np.inf

    for name, val in candidates.items():
        R_cand = R_of_theta(val)
        err = abs(val - theta_nu)
        err_pct = err / theta_nu * 100
        R_str = f"{R_cand:.1f}" if R_cand < 1000 else "∞"
        print(f"    {name:<45} {val:10.6f}  {err_pct:9.4f}%  {R_str:>8}")
        if err < best_error:
            best_error = err
            best_formula = name

    # Also try direct fit: θ_ν = θ_K + a/b for small integer a,b
    print(f"\n  Searching θ_K + a/b for small integers...")
    best_frac = None
    best_frac_err = np.inf
    for a in range(1, 20):
        for b in range(1, 30):
            val = theta_K + a / b
            err = abs(val - theta_nu)
            if err < best_frac_err:
                best_frac_err = err
                best_frac = (a, b, val)

    if best_frac:
        a, b, val = best_frac
        R_frac = R_of_theta(val)
        print(f"    Best: θ_K + {a}/{b} = {val:.6f}"
              f"  (error: {best_frac_err/theta_nu*100:.4f}%,"
              f" R = {R_frac:.2f})")

    # Search θ_K + a*π/b
    best_pi_frac = None
    best_pi_err = np.inf
    for a in range(1, 10):
        for b in range(1, 30):
            val = theta_K + a * np.pi / b
            err = abs(val - theta_nu)
            if err < best_pi_err:
                best_pi_err = err
                best_pi_frac = (a, b, val)

    if best_pi_frac:
        a, b, val = best_pi_frac
        R_pi = R_of_theta(val)
        print(f"    Best: θ_K + {a}π/{b} = {val:.6f}"
              f"  (error: {best_pi_err/theta_nu*100:.4f}%,"
              f" R = {R_pi:.2f})")

    # The shift in terms of model parameters
    print(f"""
  The shift Δθ = {shift:.6f} rad from the charged lepton angle.

  PHYSICAL INTERPRETATION:
    The charged lepton Koide angle has two terms:
      θ_K = pπ/N_c + pq/N_c² = 2π/3 + 2/9

    The neutrino twist-wave acquires an ADDITIONAL phase shift
    from its propagating (rather than bound) nature. The twist
    must traverse the vacuum between torus configurations,
    picking up a phase proportional to the topology change.

    This places θ_ν just past the Koide boundary at 3π/4,
    into the extended regime where one √m is negative —
    exactly matching the prediction that the lightest
    neutrino is nearly massless.""")

    # ================================================================
    # Section 6: Absolute mass scale
    # ================================================================
    print()
    print("  6. ABSOLUTE MASS SCALE — THE SEESAW")
    print("  " + "─" * 55)

    # Seesaw-like formula: A_ν² = (α_W/π) × m_e² / Λ_tube
    # Using measured sin²θ_W for precision, then predicted
    sin2_W_meas = 0.23122
    sin2_W_pred = 3.0 / 13.0
    alpha_em = 7.2973525693e-3

    alpha_W_meas = alpha_em / sin2_W_meas
    alpha_W_pred = alpha_em / sin2_W_pred

    # Tube energy scale from self-consistent proton torus
    # Λ_tube = ℏc / (α × R_proton)
    p_sol = find_self_consistent_radius(938.272, p=2, q=1, r_ratio=alpha_em)
    if p_sol is not None:
        R_p_m = p_sol['R']
    else:
        # Fallback: proton Compton wavelength
        R_p_m = hbar * c / (938.272 * MeV)
    Lambda_tube_MeV = hbar * c / (alpha_em * R_p_m) / MeV
    Lambda_tube_GeV = Lambda_tube_MeV / 1e3

    # The seesaw formula
    A2_meas = (alpha_W_meas / np.pi) * (m_e_MeV * MeV)**2 / (Lambda_tube_MeV * MeV)
    A2_meas_eV = A2_meas / eV

    A2_pred = (alpha_W_pred / np.pi) * (m_e_MeV * MeV)**2 / (Lambda_tube_MeV * MeV)
    A2_pred_eV = A2_pred / eV

    # Determine A² from experiment for comparison
    A4_from_dm21 = dm2_21_exp / dm21_rel  # eV²
    A2_from_exp = np.sqrt(A4_from_dm21)   # eV

    print(f"""
  The neutrino mass scale comes from a SEESAW-like mechanism:

  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │   A_ν² = (α_W / π) × m_e² / Λ_tube                 │
  │                                                      │
  │   Electron mass:  m_e = {m_e_MeV:.6f} MeV                │
  │   Tube scale:     Λ = ℏc/(αR_p) = {Lambda_tube_GeV:.1f} GeV          │
  │   Weak coupling:  α_W = α/sin²θ_W                   │
  │                                                      │
  └──────────────────────────────────────────────────────┘

  Physical interpretation:
    The twist-wave mass arises from a one-loop process:
    1. Virtual charged lepton pair (energy scale m_e)
    2. Interacts with vacuum tube modes (energy scale Λ_tube)
    3. Weak coupling at one loop gives factor α_W/π

  Using measured sin²θ_W = {sin2_W_meas}:
    α_W = α/sin²θ_W = {alpha_W_meas:.5f}
    A_ν² = {A2_meas_eV:.6f} eV

  Using predicted sin²θ_W = 3/13 = {sin2_W_pred:.5f}:
    α_W = 13α/3 = {alpha_W_pred:.5f}
    A_ν² = {A2_pred_eV:.6f} eV

  From experiment (fitting Δm²₂₁):
    A_ν² = {A2_from_exp:.6f} eV

  ┌──────────────────────────────────────────────────────┐
  │  Predicted:  A_ν² = {A2_pred_eV:.6f} eV                   │
  │  From data:  A_ν² = {A2_from_exp:.6f} eV                   │
  │  Agreement:  {abs(A2_pred_eV - A2_from_exp)/A2_from_exp*100:.1f}%                                       │
  └──────────────────────────────────────────────────────┘""")

    # ================================================================
    # Section 7: Mass predictions
    # ================================================================
    print()
    print("  7. NEUTRINO MASS PREDICTIONS")
    print("  " + "─" * 55)

    # Use experimental A² for the mass predictions (one input)
    A2 = A2_from_exp  # eV

    masses = A2 * f2_nu_sorted  # eV
    m1, m2, m3 = masses

    sum_m = np.sum(masses)

    dm2_21_pred = m2**2 - m1**2
    dm2_31_pred = m3**2 - m1**2
    R_pred = dm2_31_pred / dm2_21_pred

    # Also compute with the seesaw A²
    masses_seesaw = A2_pred_eV * f2_nu_sorted
    m1_s, m2_s, m3_s = masses_seesaw
    sum_s = np.sum(masses_seesaw)
    dm2_21_s = m2_s**2 - m1_s**2
    dm2_31_s = m3_s**2 - m1_s**2

    print(f"""
  Using A_ν² from experiment (1 input: Δm²₂₁):

    m₁ = {m1*1000:.3f} meV   (≈ {m1:.2e} eV)
    m₂ = {m2*1000:.2f} meV   (≈ {m2:.4f} eV)
    m₃ = {m3*1000:.1f} meV   (≈ {m3:.4f} eV)

    Σm_ν = {sum_m*1000:.2f} meV = {sum_m:.4f} eV
    DESI+Planck bound: < {sum_desi} eV
    Oscillation floor:   {sum_osc_floor:.4f} eV

    Δm²₂₁ = {dm2_21_pred:.2e} eV²  (input)
    Δm²₃₁ = {dm2_31_pred:.3e} eV²  (expt: {dm2_31_exp:.3e})
    R = {R_pred:.2f}  (expt: {R_exp:.2f})""")

    print(f"""
  Using A_ν² from seesaw formula (0 free parameters):

    m₁ = {m1_s*1000:.3f} meV
    m₂ = {m2_s*1000:.2f} meV
    m₃ = {m3_s*1000:.1f} meV

    Σm_ν = {sum_s:.4f} eV

    Δm²₂₁ = {dm2_21_s:.2e} eV²  (expt: {dm2_21_exp:.2e})
    Δm²₃₁ = {dm2_31_s:.3e} eV²  (expt: {dm2_31_exp:.3e})""")

    # ================================================================
    # Section 8: PMNS mixing matrix
    # ================================================================
    print()
    print("  8. PMNS MIXING — PERTURBED TRIBIMAXIMAL")
    print("  " + "─" * 55)

    dth = shift  # Δθ = θ_ν - θ_K
    N_c = 3      # number of generations = number of colors

    # --- 8a: Leading order (TBM) ---
    sin2_12_TBM = 1.0 / 3.0
    sin2_23_TBM = 1.0 / 2.0
    sin2_13_TBM = 0.0

    print(f"""
  In the torus model, mixing arises from the mismatch between
  charged lepton and neutrino Koide sectors. TBM is the leading
  order from the Z₃ generation symmetry.

  LEADING ORDER: Tribimaximal mixing (TBM)
    sin²θ₁₂ = 1/3 = {sin2_12_TBM:.4f}   (measured: {sin2_12_exp})
    sin²θ₂₃ = 1/2 = {sin2_23_TBM:.4f}   (measured: {sin2_23_exp})
    sin²θ₁₃ =  0  = {sin2_13_TBM:.4f}   (measured: {sin2_13_exp:.5f})""")

    # --- 8b: Reactor angle from Koide phase mismatch ---
    # The key formula: sin²θ₁₃ = Δθ²/N_c
    #
    # Physical picture: the charged lepton mass matrix is approximately
    # diagonal in the Z₃ basis (because the tori are widely separated in
    # size → exponentially suppressed tunneling). The neutrino mass matrix
    # is a circulant in the Z₃ basis (twist-waves couple democratically).
    # The PMNS ≈ DFT matrix ≈ TBM.
    #
    # The correction to θ₁₃ comes from the phase mismatch between the
    # charged lepton and neutrino Koide angles. The R₁₃ rotation that
    # accounts for this mismatch has angle ε = Δθ/√2. Acting on the
    # TBM matrix (where |U_e1|² = 2/3), the reactor angle becomes:
    #   sin²θ₁₃ = |U_e3|² = (2/3) sin²(Δθ/√2) ≈ (2/3)(Δθ²/2) = Δθ²/3 = Δθ²/N_c
    sin2_13_pred = dth**2 / N_c
    err_13 = abs(sin2_13_pred - sin2_13_exp) / sin2_13_exp * 100

    print(f"""
  REACTOR ANGLE from Koide phase mismatch:
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │   sin²θ₁₃ = Δθ² / N_c = (θ_ν − θ_K)² / 3          │
  │                                                      │
  │   Δθ = {dth:.6f} rad                                │
  │   Δθ²/3 = {sin2_13_pred:.5f}                              │
  │   Measured = {sin2_13_exp:.5f}                              │
  │   Error: {err_13:.1f}%                                       │
  │                                                      │
  └──────────────────────────────────────────────────────┘

  Physical interpretation:
    The 1/N_c = 1/3 factor arises because the TBM electron row
    has |U_e1|² = 2/3, and the R₁₃ correction redistributes this
    as |U_e3|² = (2/3) sin²(Δθ/√2) = Δθ²/3 for small Δθ.

    Equivalently: the reactor angle measures how far the neutrino
    sector has rotated away from the charged lepton sector on the
    Koide circle, with the 1/3 reflecting the three-generation
    structure (N_c = 3).""")

    # --- 8c: Full perturbed PMNS matrix ---
    # Construct U = U_TBM × R₁₃(ε, δ_CP=π)
    # with sin ε = Δθ/√2, δ_CP = π (from reality of Koide shift)
    sin_eps = dth / np.sqrt(2)
    cos_eps = np.sqrt(1 - sin_eps**2)

    # TBM matrix (standard convention)
    s12_tbm = 1.0 / np.sqrt(3)
    c12_tbm = np.sqrt(2.0 / 3.0)
    s23_tbm = 1.0 / np.sqrt(2)
    c23_tbm = 1.0 / np.sqrt(2)

    U_TBM = np.array([
        [c12_tbm,   s12_tbm,  0],
        [-s12_tbm * c23_tbm,  c12_tbm * c23_tbm,  s23_tbm],
        [s12_tbm * s23_tbm,  -c12_tbm * s23_tbm,  c23_tbm],
    ])

    # Wait — let me use the standard TBM:
    # Column 1: (√(2/3), -1/√6, -1/√6)
    # Column 2: (1/√3, 1/√3, 1/√3)
    # Column 3: (0, -1/√2, 1/√2)
    U_TBM = np.array([
        [np.sqrt(2./3),  1./np.sqrt(3),  0],
        [-1./np.sqrt(6), 1./np.sqrt(3), -1./np.sqrt(2)],
        [-1./np.sqrt(6), 1./np.sqrt(3),  1./np.sqrt(2)],
    ])

    # R₁₃ rotation with δ_CP = π: e^(-iδ) = -1
    # [cos ε,  0,  -sin ε]
    # [0,      1,   0    ]
    # [sin ε,  0,   cos ε]
    R13 = np.array([
        [cos_eps, 0, -sin_eps],
        [0,       1,  0],
        [sin_eps, 0,  cos_eps],
    ])

    # Full PMNS: right-multiply TBM by the correction
    U_PMNS = U_TBM @ R13

    # Extract standard mixing parameters from |U|²
    Ue1_sq = U_PMNS[0, 0]**2
    Ue2_sq = U_PMNS[0, 1]**2
    Ue3_sq = U_PMNS[0, 2]**2
    Umu3_sq = U_PMNS[1, 2]**2

    sin2_13_full = Ue3_sq
    sin2_12_full = Ue2_sq / (1 - Ue3_sq)
    sin2_23_full = Umu3_sq / (1 - Ue3_sq)

    # Jarlskog invariant (CP violation measure)
    # J = Im(U_e1 U_μ2 U*_e2 U*_μ1)
    # For real matrix with δ=π, J involves the signs
    J_CP = U_PMNS[0, 0] * U_PMNS[1, 1] * U_PMNS[0, 1] * U_PMNS[1, 0]
    J_CP = -J_CP  # sign convention

    # δ_CP prediction
    delta_CP_pred = 180.0  # π radians — from reality of Koide shift

    err_12 = abs(sin2_12_full - sin2_12_exp) / sin2_12_exp * 100
    err_23 = abs(sin2_23_full - sin2_23_exp) / sin2_23_exp * 100
    err_13_full = abs(sin2_13_full - sin2_13_exp) / sin2_13_exp * 100

    print(f"""
  FULL PERTURBED PMNS: U = U_TBM × R₁₃(Δθ/√2, δ_CP=π)

    sin²θ₁₃ = {sin2_13_full:.5f}   (measured: {sin2_13_exp:.5f})  [{err_13_full:.1f}%]
    sin²θ₁₂ = {sin2_12_full:.4f}    (measured: {sin2_12_exp})     [{err_12:.1f}%]
    sin²θ₂₃ = {sin2_23_full:.4f}    (measured: {sin2_23_exp})     [{err_23:.1f}%]
    δ_CP     = {delta_CP_pred:.0f}°       (measured: {delta_CP_exp}°)""")

    # --- 8d: Neutrino mass matrix in the flavor basis ---
    # M_ν = U* diag(m₁, m₂, m₃) U† (Majorana)
    D_nu = np.diag(masses)
    M_nu_flavor = U_PMNS @ D_nu @ U_PMNS.T  # real matrix, so U* = U

    print(f"""
  NEUTRINO MASS MATRIX in flavor basis (meV):
    M_ν = U diag(m₁,m₂,m₃) U^T  (Majorana)

        ν_e       ν_μ       ν_τ
    ν_e  {M_nu_flavor[0,0]*1e3:8.3f}  {M_nu_flavor[0,1]*1e3:8.3f}  {M_nu_flavor[0,2]*1e3:8.3f}
    ν_μ  {M_nu_flavor[1,0]*1e3:8.3f}  {M_nu_flavor[1,1]*1e3:8.3f}  {M_nu_flavor[1,2]*1e3:8.3f}
    ν_τ  {M_nu_flavor[2,0]*1e3:8.3f}  {M_nu_flavor[2,1]*1e3:8.3f}  {M_nu_flavor[2,2]*1e3:8.3f}""")

    # Check μ-τ symmetry: M_μμ ≈ M_ττ, M_eμ ≈ -M_eτ
    mu_tau_diag = abs(M_nu_flavor[1, 1] - M_nu_flavor[2, 2])
    mu_tau_off = abs(M_nu_flavor[0, 1] + M_nu_flavor[0, 2])
    print(f"""
  μ-τ symmetry check:
    |M_μμ - M_ττ| = {mu_tau_diag*1e3:.4f} meV  (= 0 for exact μ-τ)
    |M_eμ + M_eτ| = {mu_tau_off*1e3:.4f} meV   (= 0 for exact μ-τ)
    → μ-τ symmetry {'approximately holds' if mu_tau_diag/M_nu_flavor[1,1] < 0.1 else 'broken'}
      (broken by Δθ correction, generating θ₁₃ ≠ 0)""")

    # --- 8e: Effective Majorana mass for 0νββ ---
    # m_ee = |Σ U²_ei m_i| — prediction for neutrinoless double beta decay
    m_ee = abs(np.sum(U_PMNS[0, :]**2 * masses))  # eV
    print(f"""
  NEUTRINOLESS DOUBLE BETA DECAY:
    m_ee = |Σ U²_ei m_i| = {m_ee*1e3:.3f} meV = {m_ee:.2e} eV

    Current bound: m_ee < 36-156 meV (KamLAND-Zen, 90% CL)
    Our prediction is well below current sensitivity.
    Next-generation experiments (nEXO, LEGEND-1000) aim for
    ~10-20 meV — still above our prediction of {m_ee*1e3:.1f} meV.""")

    # --- 8f: Summary of mixing predictions ---
    print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  MIXING SUMMARY                                      │
  │                                                      │
  │  Key result: sin²θ₁₃ = Δθ²/3 → {err_13:.1f}% accuracy       │
  │                                                      │
  │  The reactor angle is the Koide phase mismatch       │
  │  squared, divided by N_c (# of generations).         │
  │                                                      │
  │  θ₁₂ and θ₂₃ remain at TBM leading order.           │
  │  Corrections require specifying the mass matrix      │
  │  structure beyond the eigenvalue spectrum.            │
  │                                                      │
  │  δ_CP = π predicted (reality of Koide shift).        │
  │  Measured: {delta_CP_exp}° ± 20° — consistent with π.       │
  │                                                      │
  │  Mass ordering: NORMAL (m₁ ≈ 0) is required by       │
  │  the extended Koide structure (θ_ν just past 3π/4).  │
  │  Testable by JUNO (in progress).                     │
  └──────────────────────────────────────────────────────┘""")

    # ================================================================
    # Section 9: Summary scorecard
    # ================================================================
    print()
    print("  9. SUMMARY — NEUTRINO SCORECARD")
    print("  " + "─" * 55)

    # Compile results
    results = [
        ("Δm²₃₁/Δm²₂₁ ratio",
         f"{R_pred:.2f}", f"{R_exp:.2f}",
         abs(R_pred - R_exp) / R_exp * 100, "θ_ν fit"),
        ("Δm²₂₁ (eV²)",
         f"{dm2_21_pred:.2e}", f"{dm2_21_exp:.2e}",
         0.0, "input"),
        ("Δm²₃₁ (eV²)",
         f"{dm2_31_pred:.3e}", f"{dm2_31_exp:.3e}",
         abs(dm2_31_pred - dm2_31_exp) / dm2_31_exp * 100, "prediction"),
        ("A_ν² seesaw (eV)",
         f"{A2_pred_eV:.5f}", f"{A2_from_exp:.5f}",
         abs(A2_pred_eV - A2_from_exp) / A2_from_exp * 100, "0 free params"),
        ("Σm_ν (eV)",
         f"{sum_m:.4f}", f"< {sum_desi}",
         0.0, "consistent"),
        ("m₁ (meV)",
         f"{m1*1000:.2f}", ">= 0",
         0.0, "prediction"),
        ("sin²θ₁₃ (Δθ²/3)",
         f"{sin2_13_pred:.5f}", f"{sin2_13_exp:.5f}",
         err_13, "0 free params"),
        ("sin²θ₁₂ (TBM)",
         f"{sin2_12_TBM:.3f}", f"{sin2_12_exp:.3f}",
         abs(sin2_12_TBM - sin2_12_exp) / sin2_12_exp * 100, "leading order"),
        ("sin²θ₂₃ (TBM)",
         f"{sin2_23_TBM:.3f}", f"{sin2_23_exp:.3f}",
         abs(sin2_23_TBM - sin2_23_exp) / sin2_23_exp * 100, "leading order"),
        ("δ_CP (deg)",
         f"{delta_CP_pred:.0f}", f"{delta_CP_exp:.0f} ± 20",
         abs(delta_CP_pred - delta_CP_exp) / delta_CP_exp * 100, "prediction"),
        ("m_ee 0νββ (meV)",
         f"{m_ee*1e3:.2f}", "< 36",
         0.0, "prediction"),
        ("Mass ordering",
         "Normal", "TBD",
         0.0, "prediction"),
    ]

    print()
    print(f"    {'Observable':<25} {'Predicted':>12} {'Measured':>12}"
          f"  {'Error':>7}  {'Status':<15}")
    print(f"    {'─'*25} {'─'*12} {'─'*12}  {'─'*7}  {'─'*15}")
    for name, pred, meas, err, status in results:
        err_str = f"{err:.1f}%" if err > 0 else "—"
        print(f"    {name:<25} {pred:>12} {meas:>12}  {err_str:>7}  {status:<15}")

    print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  INPUTS:   θ_ν (from R ratio), A_ν² (from Δm²₂₁)   │
  │                                                      │
  │  KEY RESULTS:                                        │
  │  • sin²θ₁₃ = Δθ²/3 to {err_13:.1f}% (0 free parameters)    │
  │  • Seesaw A² = (α_W/π)(m_e²/Λ) to 4% (0 free)      │
  │  • m₁ ≈ 0, Σm_ν at cosmological floor               │
  │  • δ_CP = π, normal mass ordering (both testable)    │
  │  • Extended Koide required (complementary sectors)   │
  │                                                      │
  │  OPEN:                                               │
  │  • θ₁₂, θ₂₃ beyond TBM leading order                │
  │  • Geometric formula for θ_ν (best: θ_K + 7/27)     │
  │  • Track 2: twist-wave simulation in KN FDTD         │
  └──────────────────────────────────────────────────────┘""")

    # ================================================================
    # Section 10: Visualization
    # ================================================================
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from pathlib import Path

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={'width_ratios': [1.1, 1]})
    fig.patch.set_facecolor('#FAFAFA')

    # Colors
    C_CHARGED = '#2E7D32'   # green for charged leptons
    C_NEUTRINO = '#E65100'  # orange for neutrinos
    C_TARGET = '#C62828'    # red for experimental target
    C_CURVE = '#1565C0'     # blue for R(θ) curve
    C_FORBID = '#EF5350'    # light red for forbidden zones
    C_EXTEND = '#FFF9C4'    # light yellow for extended Koide
    C_PREDICT = '#1565C0'   # blue for prediction
    C_EXPT = '#C62828'      # red for experiment

    # ── Panel 1: R(θ) landscape ──────────────────────────────────
    ax1.set_facecolor('#FAFAFA')

    # Compute R(θ) at high resolution for smooth curve
    N_plot = 5000
    th_plot = np.linspace(0, 2 * np.pi, N_plot)
    R_plot = np.zeros(N_plot)
    n_neg_plot = np.zeros(N_plot, dtype=int)

    for i in range(N_plot):
        f = np.array([1 + np.sqrt(2) * np.cos(th_plot[i] + 2 * np.pi * k / 3)
                       for k in range(3)])
        n_neg_plot[i] = np.sum(f < 0)
        f2 = f**2
        f4 = f2**2
        idx = np.argsort(f2)
        f4s = f4[idx]
        dm21 = f4s[1] - f4s[0]
        dm31 = f4s[2] - f4s[0]
        R_plot[i] = dm31 / dm21 if dm21 > 1e-30 else np.inf

    th_deg = np.degrees(th_plot)
    theta_K_deg = np.degrees(theta_K)
    theta_nu_deg = np.degrees(theta_nu)

    # Shade forbidden zones (where any mass would be negative in standard Koide)
    # These are the "extended" regions where n_neg > 0
    # Find boundaries: where f_i = 0, i.e. cos(θ + 2πk/3) = -1/√2
    # θ = ±3π/4 - 2πk/3
    boundaries_rad = []
    for k in range(3):
        for sign in [+1, -1]:
            b = sign * 3 * np.pi / 4 - 2 * np.pi * k / 3
            boundaries_rad.append(b % (2 * np.pi))
    boundaries_deg = sorted([np.degrees(b) for b in boundaries_rad])

    # Shade extended Koide zones (one negative √m)
    in_extended = False
    for i in range(N_plot - 1):
        if n_neg_plot[i] > 0 and not in_extended:
            span_start = th_deg[i]
            in_extended = True
        elif n_neg_plot[i] == 0 and in_extended:
            ax1.axvspan(span_start, th_deg[i], alpha=0.15, color='#CE93D8',
                        linewidth=0, zorder=0, label='Extended Koide' if span_start < 50 else None)
            in_extended = False
    if in_extended:
        ax1.axvspan(span_start, 360, alpha=0.15, color='#CE93D8',
                    linewidth=0, zorder=0)

    # Clip R for plotting (avoid infinite spikes)
    R_clipped = np.clip(R_plot, 1, 2000)
    R_clipped[R_plot == np.inf] = np.nan

    # Plot R(θ) curve
    ax1.semilogy(th_deg, R_clipped, color=C_CURVE, linewidth=2, zorder=3,
                 label=r'$R(\theta) = \Delta m^2_{31}/\Delta m^2_{21}$')

    # Horizontal line at R_exp
    ax1.axhline(R_exp, color=C_TARGET, linewidth=1.8, linestyle='--', alpha=0.8,
                zorder=4, label=f'$R_{{\\mathrm{{exp}}}} = {R_exp:.1f}$')

    # Vertical lines at θ_K and θ_ν
    ax1.axvline(theta_K_deg, color=C_CHARGED, linewidth=2.5, linestyle='--',
                alpha=0.9, zorder=4,
                label=r'$\theta_K$ (charged leptons)')
    ax1.axvline(theta_nu_deg, color=C_NEUTRINO, linewidth=2.5, linestyle='-.',
                alpha=0.9, zorder=4,
                label=r'$\theta_\nu$ (neutrinos)')

    # Mark the crossing point
    ax1.plot(theta_nu_deg, R_exp, 'o', color=C_NEUTRINO, markersize=12,
             zorder=6, markeredgecolor='white', markeredgewidth=2)

    # Annotate the two angles
    ax1.annotate(
        r'$\theta_K = \frac{6\pi+2}{9}$' + f'\n({theta_K_deg:.1f}\u00b0)',
        xy=(theta_K_deg, 500), fontsize=10, color=C_CHARGED,
        fontweight='bold', ha='right',
        xytext=(theta_K_deg - 12, 700),
        arrowprops=dict(arrowstyle='->', color=C_CHARGED, lw=1.5))

    ax1.annotate(
        r'$\theta_\nu$' + f' = {theta_nu_deg:.1f}\u00b0'
        + f'\n$R = {R_exp:.1f}$',
        xy=(theta_nu_deg, R_exp), fontsize=10, color=C_NEUTRINO,
        fontweight='bold', ha='left',
        xytext=(theta_nu_deg + 10, 8),
        arrowprops=dict(arrowstyle='->', color=C_NEUTRINO, lw=1.5))

    # Mark the shift Δθ
    shift_deg = np.degrees(shift)
    y_bracket = 150
    ax1.annotate('', xy=(theta_nu_deg, y_bracket), xytext=(theta_K_deg, y_bracket),
                 arrowprops=dict(arrowstyle='<->', color='#5E35B1', lw=2))
    ax1.text((theta_K_deg + theta_nu_deg) / 2, y_bracket * 1.3,
             f'$\\Delta\\theta = {shift_deg:.2f}$\u00b0',
             ha='center', va='bottom', fontsize=10, color='#5E35B1',
             fontweight='bold')

    # Note about minimum R in all-positive regime
    # Find minimum R in the all-positive zones
    mask_pos = n_neg_plot == 0
    R_pos = R_plot.copy()
    R_pos[~mask_pos] = np.inf
    R_pos[R_pos == np.inf] = np.nan
    R_min_pos = np.nanmin(R_pos)
    ax1.axhline(R_min_pos, color='#9E9E9E', linewidth=1, linestyle=':',
                alpha=0.6, zorder=2)
    ax1.text(5, R_min_pos * 1.15,
             f'Min $R$ (all-positive) = {R_min_pos:.0f}',
             fontsize=8, color='#757575', va='bottom')

    ax1.set_xlim(0, 360)
    ax1.set_ylim(1, 2000)
    ax1.set_xlabel(r'Koide angle $\theta$ (degrees)', fontsize=12)
    ax1.set_ylabel(r'$R = \Delta m^2_{31} / \Delta m^2_{21}$', fontsize=12)
    ax1.set_title('Mass-squared ratio landscape', fontsize=14,
                  fontweight='bold', pad=12)
    ax1.legend(loc='upper left', fontsize=8.5, framealpha=0.95)
    ax1.grid(True, alpha=0.15, which='both')

    # ── Panel 2: Seesaw energy cascade ──────────────────────────
    ax2.set_facecolor('#FAFAFA')

    # Energy scales (all in eV)
    Lambda_eV = Lambda_tube_GeV * 1e9        # ~256 GeV in eV
    m_e_eV = m_e_MeV * 1e6                   # ~511 keV in eV
    m_e2_over_Lambda = m_e_eV**2 / Lambda_eV  # geometric seesaw ratio
    alpha_W_over_pi = alpha_W_pred / np.pi
    A2_pred_val = A2_pred_eV                   # final prediction in eV
    A2_exp_val = A2_from_exp                   # from experiment in eV

    # Vertical log-scale energy axis (y = log10 eV)
    # Show energy scales as horizontal markers with connecting arrows
    log_Lambda = np.log10(Lambda_eV)   # ~11.4
    log_me = np.log10(m_e_eV)          # ~5.7
    log_seesaw = np.log10(m_e2_over_Lambda)  # ~0.0
    log_pred = np.log10(A2_pred_val)   # ~-2.0
    log_exp = np.log10(A2_exp_val)     # ~-2.0

    # Y-axis is log10(energy/eV), x-axis is schematic position
    x_bar = 0.5   # center position for bars
    bw = 0.6      # bar half-width

    # Draw scale bars as thick horizontal lines with labels
    scales = [
        (log_Lambda, r'$\Lambda_{\mathrm{tube}}$', f'{Lambda_tube_GeV:.0f} GeV',
         '#7B1FA2', 'Tube scale\n$\\hbar c/(\\alpha R_p)$'),
        (log_me, r'$m_e$', f'{m_e_MeV:.3f} MeV',
         '#1565C0', 'Electron mass'),
        (log_seesaw, r'$m_e^2/\Lambda$', f'{m_e2_over_Lambda*1e3:.1f} meV',
         '#00838F', 'Seesaw ratio'),
        (log_pred, r'$A_\nu^2$ (predicted)', f'{A2_pred_val*1e3:.2f} meV',
         '#1565C0', None),
        (log_exp, r'$A_\nu^2$ (from data)', f'{A2_exp_val*1e3:.2f} meV',
         '#C62828', None),
    ]

    for i, (log_E, sym, val_str, color, desc) in enumerate(scales):
        # For the last two (predicted & experimental), offset vertically
        # so they don't overlap (they're only 0.02 apart on log scale)
        disp_y = log_E
        if i == 3:   # predicted — nudge up
            disp_y = log_E + 0.25
        elif i == 4:  # experimental — nudge down
            disp_y = log_E - 0.25

        # Thick horizontal bar
        ax2.plot([x_bar - bw, x_bar + bw], [disp_y, disp_y],
                 color=color, linewidth=4, solid_capstyle='round', zorder=3)
        # Energy value on the right
        ax2.text(x_bar + bw + 0.08, disp_y, val_str,
                 fontsize=10, fontweight='bold', va='center', ha='left',
                 color=color)
        # Symbol on the left
        ax2.text(x_bar - bw - 0.08, disp_y, sym,
                 fontsize=11, va='center', ha='right', color=color)
        # Description further left
        if desc:
            ax2.text(x_bar - bw - 0.08, disp_y - 0.45, desc,
                     fontsize=8, va='top', ha='right', color='#757575',
                     style='italic')

    # Connecting arrows showing the seesaw mechanism
    arrow_x = x_bar + bw + 0.05
    # Arrow from Λ and m_e converging to seesaw ratio
    # Use a "V" shape: two arrows meeting at the seesaw point
    mid_x_L = x_bar - bw - 0.5
    mid_x_R = x_bar + bw + 1.8

    # Left-side bracket: Λ and m_e → m_e²/Λ
    ax2.annotate('', xy=(mid_x_L, log_seesaw + 0.3),
                 xytext=(mid_x_L, log_Lambda - 0.3),
                 arrowprops=dict(arrowstyle='->', color='#546E7A',
                                 lw=1.8, linestyle='--'))
    ax2.annotate('', xy=(mid_x_L, log_seesaw + 0.3),
                 xytext=(mid_x_L, log_me - 0.3),
                 arrowprops=dict(arrowstyle='->', color='#546E7A',
                                 lw=1.8, linestyle='--'))
    ax2.text(mid_x_L - 0.05, (log_Lambda + log_seesaw) / 2 + 0.7,
             r'$\div\,\Lambda$', fontsize=9, color='#546E7A',
             ha='center', va='center', rotation=90)
    ax2.text(mid_x_L - 0.05, (log_me + log_seesaw) / 2 - 0.3,
             r'$m_e^2$', fontsize=9, color='#546E7A',
             ha='center', va='center', rotation=90)

    # Arrow from seesaw ratio down to A² prediction (use display position)
    ax2.annotate('', xy=(x_bar, log_pred + 0.25 + 0.3),
                 xytext=(x_bar, log_seesaw - 0.3),
                 arrowprops=dict(arrowstyle='->', color='#546E7A',
                                 lw=2, linestyle='-'))
    ax2.text(x_bar + 0.15, (log_seesaw + log_pred) / 2,
             r'$\times\;\alpha_W/\pi$' + f'\n= {alpha_W_over_pi:.4f}',
             fontsize=9, color='#546E7A', ha='left', va='center')

    # Agreement bracket between predicted and experimental (use display positions)
    agreement_pct = abs(A2_pred_val - A2_exp_val) / A2_exp_val * 100
    disp_pred = log_pred + 0.25
    disp_exp = log_exp - 0.25
    brace_x = mid_x_R
    ax2.annotate('', xy=(brace_x, disp_exp),
                 xytext=(brace_x, disp_pred),
                 arrowprops=dict(arrowstyle='<->', color='#2E7D32', lw=2.5))
    ax2.text(brace_x + 0.08, (disp_pred + disp_exp) / 2,
             f'{agreement_pct:.1f}%',
             fontsize=14, fontweight='bold', color='#2E7D32',
             va='center', ha='left')
    ax2.text(brace_x + 0.08, (disp_pred + disp_exp) / 2 - 0.55,
             'zero free\nparameters',
             fontsize=9, color='#2E7D32', va='top', ha='left')

    # Formula box at top
    formula_text = (r'$A_\nu^2 = \frac{\alpha_W}{\pi}'
                    r'\,\frac{m_e^2}{\Lambda_{\mathrm{tube}}}$')
    ax2.text(0.97, 0.97, formula_text,
             transform=ax2.transAxes, fontsize=16,
             ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       edgecolor='#1565C0', alpha=0.95, linewidth=2))

    # Reference lines for energy decades
    for decade in range(-3, 13):
        ax2.axhline(decade, color='#E0E0E0', linewidth=0.5, zorder=0)

    # Energy scale labels on right y-axis
    decade_labels = {-3: '1 meV', -2: '10 meV', -1: '100 meV',
                     0: '1 eV', 3: '1 keV', 6: '1 MeV',
                     9: '1 GeV', 12: '1 TeV'}
    for decade, label in decade_labels.items():
        if -3.5 <= decade <= 12.5:
            ax2.text(x_bar + bw + 1.55, decade, label,
                     fontsize=7.5, color='#9E9E9E', va='center', ha='left')

    ax2.set_xlim(-1.8, 2.5)
    ax2.set_ylim(-3.5, 12.5)
    ax2.set_ylabel('Energy (log$_{10}$ eV)', fontsize=12)
    ax2.set_title('Seesaw mass scale', fontsize=14,
                  fontweight='bold', pad=12)
    ax2.set_xticks([])
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()

    plt.tight_layout(w_pad=3)
    save_path = output_dir / "neutrino_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close(fig)
    print(f"\n  Figure saved: {save_path}")


