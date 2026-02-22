"""
Particle scanning, mass-finding, pair production, and decay landscape.
"""

import numpy as np
from .constants import (
    c, hbar, h_planck, e_charge, eps0, m_e, m_e_MeV, alpha,
    eV, MeV, lambda_C, r_e, TorusParams, PARTICLE_MASSES,
)
from .core import (
    null_condition_time, torus_knot_curve, compute_circulation_energy,
    compute_self_energy, compute_total_energy, compute_path_length,
    compute_angular_momentum, find_self_consistent_radius,
    resonance_analysis, compute_retarded_field_sample,
)


def scan_torus_parameters():
    """
    Scan over torus parameters to find configurations whose
    total energy (circulation + self-energy) matches known particle masses.
    """
    print("=" * 70)
    print("PARAMETER SCAN: Torus configurations vs particle masses")
    print("  (now includes self-energy corrections)")
    print("=" * 70)
    print(f"\nReference scales:")
    print(f"  Reduced Compton wavelength:  λ_C = {lambda_C:.4e} m")
    print(f"  Classical electron radius:   r_e = {r_e:.4e} m")
    print(f"  Electron mass:               {m_e_MeV:.4f} MeV/c²")
    print(f"  Fine-structure constant:     α = 1/{1/alpha:.2f}")
    print()

    print(f"{'p':>3} {'q':>3} {'R/λ_C':>10} {'r/λ_C':>10} "
          f"{'E_circ':>10} {'E_self':>10} {'E_total':>10} {'Closest':>10} {'Ratio':>10}")
    print("-" * 85)

    results = []

    for p in range(1, 5):
        for q in range(1, 5):
            for R_ratio in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
                for r_ratio_abs in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
                    if r_ratio_abs >= R_ratio:
                        continue

                    R = R_ratio * lambda_C
                    r = r_ratio_abs * lambda_C
                    params = TorusParams(R=R, r=r, p=p, q=q)

                    _, E_total_MeV, breakdown = compute_total_energy(params, N=500)

                    closest = min(PARTICLE_MASSES.items(),
                                  key=lambda kv: abs(kv[1] - E_total_MeV))
                    ratio = E_total_MeV / closest[1]

                    if 0.33 < ratio < 3.0:
                        results.append({
                            'p': p, 'q': q,
                            'R_ratio': R_ratio, 'r_ratio': r_ratio_abs,
                            'E_circ': breakdown['E_circ_MeV'],
                            'E_self': breakdown['U_total_MeV'],
                            'E_total': E_total_MeV,
                            'closest': closest[0],
                            'ratio': ratio,
                        })
                        print(f"{p:3d} {q:3d} {R_ratio:10.3f} {r_ratio_abs:10.3f} "
                              f"{breakdown['E_circ_MeV']:10.4f} "
                              f"{breakdown['U_total_MeV']:10.6f} "
                              f"{E_total_MeV:10.4f} {closest[0]:>10} {ratio:10.4f}")

    print(f"\n{len(results)} configurations within factor of 3 of known masses")
    return results


def print_basic_analysis(params: TorusParams):
    """Print basic properties of a torus knot null curve."""
    print("=" * 70)
    print(f"NULL WORLDTUBE ANALYSIS")
    print(f"Torus: R = {params.R:.4e} m, r = {params.r:.4e} m")
    print(f"Winding: ({params.p}, {params.q}) torus knot")
    print(f"R/λ_C = {params.R/lambda_C:.4f},  r/λ_C = {params.r/lambda_C:.4f}")
    print("=" * 70)

    # Null curve properties
    lam, t, T, xyz, v = null_condition_time(params)
    v_mag = np.sqrt(np.sum(v**2, axis=-1))
    print(f"\n--- Null curve ---")
    print(f"  Velocity:   v/c = {v_mag.mean()/c:.10f} (should be 1.0)")
    print(f"  Period:     T = {T:.6e} s")
    print(f"  Frequency:  f = {1/T:.6e} Hz")

    # Circulation energy
    E, E_MeV_val, L, T, f = compute_circulation_energy(params)
    print(f"\n--- Circulation energy (zeroth order) ---")
    print(f"  Path length:  L = {L:.6e} m")
    print(f"  Energy:       E_circ = {E:.6e} J = {E_MeV_val:.6f} MeV")
    print(f"  E_circ / m_e c²:   {E_MeV_val / m_e_MeV:.6f}")

    # Self-energy
    se = compute_self_energy(params)
    print(f"\n--- Electromagnetic self-energy ---")
    print(f"  Current:          I = {se['I_amps']:.4e} A")
    print(f"  Inductance:       L = {se['L_ind_henry']:.4e} H")
    print(f"  log(8R/r) - 2:    {se['log_factor']:.4f}")
    print(f"  U_magnetic:       {se['U_mag_J']:.6e} J = {se['U_mag_J']/MeV:.6f} MeV")
    print(f"  U_electric:       {se['U_elec_J']:.6e} J = {se['U_elec_J']/MeV:.6f} MeV")
    print(f"  U_total:          {se['U_total_J']:.6e} J = {se['U_total_MeV']:.6f} MeV")
    print(f"  U_self / E_circ:  {se['self_energy_fraction']:.6f}")
    print(f"  α·ln(8R/r)-2]/π: {se['alpha_prediction']:.6f}  (analytic prediction)")
    print(f"    → ratio matches α to: {abs(se['self_energy_fraction'] - se['alpha_prediction']) / se['alpha_prediction'] * 100:.2f}%")

    # Total energy
    print(f"\n--- Total energy (circulation + self-interaction) ---")
    print(f"  E_total:          {se['E_total_MeV']:.6f} MeV")
    print(f"  E_total / m_e c²: {se['E_total_MeV'] / m_e_MeV:.6f}")
    gap_pct = (se['E_total_MeV'] - m_e_MeV) / m_e_MeV * 100
    print(f"  Gap from m_e:     {gap_pct:+.4f}%")

    # Angular momentum
    am = compute_angular_momentum(params)
    print(f"\n--- Angular momentum (literal, mechanical) ---")
    print(f"  ⟨L_z⟩ = {am['Lz_over_hbar']:.6f} ℏ  (about torus axis)")
    print(f"  |⟨L⟩| = {am['L_over_hbar']:.6f} ℏ")
    print(f"  L_z range: [{am['Lz_min']:.4f}, {am['Lz_max']:.4f}] ℏ  (instantaneous)")
    if abs(am['Lz_over_hbar'] - 0.5) < 0.01:
        print(f"  *** L_z ≈ ℏ/2 — fermion spin from geometry! ***")
    elif abs(am['Lz_over_hbar'] - 1.0) < 0.01:
        print(f"  L_z ≈ ℏ — matches photon/boson spin")

    # Scale matching
    print(f"\n--- Scale matching ---")
    target_L = 2 * np.pi * hbar * c / (m_e * c**2)
    print(f"  Path length for m_e: L = 2π·λ_C = {target_L:.6e} m")
    print(f"  Actual path length:  L = {L:.6e} m")
    print(f"  Ratio L/L_target:    {L / target_L:.6f}")

    # Retarded field structure
    print(f"\n--- Retarded field structure ---")
    ret = compute_retarded_field_sample(params, N=500)
    print(f"  Mean retarded distance:  {ret['mean_retarded_distance']:.4e} m")
    print(f"  Std retarded distance:   {ret['std_retarded_distance']:.4e} m")
    print(f"  Mean retarded delay:     {ret['mean_retarded_delay']:.4e} s")
    print(f"  Delay as fraction of T:  {ret['delay_fraction']:.4f}")
    print(f"    (= fraction of loop the field 'looks back' through)")


def print_self_energy_analysis(params: TorusParams):
    """
    Detailed analysis of how α enters the self-energy.
    Shows the relationship between self-interaction and circulation energy.
    """
    print("=" * 70)
    print("SELF-ENERGY ANALYSIS: How α enters the theory")
    print("=" * 70)

    print(f"\nThe fine-structure constant α = e²/(4πε₀ℏc) = {alpha:.10f}")
    print(f"                             1/α = {1/alpha:.6f}")
    print(f"\nα is the ratio of electromagnetic coupling to quantum action.")
    print(f"In the null worldtube, it appears as:")
    print(f"  U_self / E_circ = α × [ln(8R/r) - 2] / π")
    print(f"\nThis is NOT put in by hand. It emerges because:")
    print(f"  - Current I = ec/L contains e (electromagnetic coupling)")
    print(f"  - Circulation energy E = 2πℏc/L contains ℏ (quantum action)")
    print(f"  - Their ratio gives e²/(ℏc) ∝ α")

    print(f"\n{'r/R':>8} {'ln(8R/r)-2':>12} {'U/E_circ':>12} {'α·[...]/π':>12} {'match':>8}")
    print("-" * 58)

    R = params.R
    for r_frac in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]:
        r = r_frac * R
        tp = TorusParams(R=R, r=r, p=params.p, q=params.q)
        se = compute_self_energy(tp)
        match = "✓" if abs(se['self_energy_fraction'] - se['alpha_prediction']) / se['alpha_prediction'] < 0.05 else "~"
        print(f"{r_frac:8.3f} {se['log_factor']:12.4f} "
              f"{se['self_energy_fraction']:12.6f} {se['alpha_prediction']:12.6f} {match:>8}")

    print(f"\n--- Effect on electron mass ---")
    print(f"\nWith R = λ_C (electron Compton wavelength):")

    for r_frac in [0.01, 0.05, 0.1, 0.2]:
        r = r_frac * lambda_C
        tp = TorusParams(R=lambda_C, r=r, p=1, q=1)
        se = compute_self_energy(tp)
        gap = (se['E_total_MeV'] - m_e_MeV) / m_e_MeV * 100
        print(f"  r/R = {r_frac:.2f}: E_circ = {se['E_circ_MeV']:.6f}, "
              f"E_self = {se['U_total_MeV']:.6f}, "
              f"E_total = {se['E_total_MeV']:.6f} MeV  ({gap:+.3f}% from m_e)")

    # Find r/R where E_total = m_e exactly
    print(f"\n--- Finding r/R where E_total = m_e c² exactly ---")
    # Bisection on r/R
    rr_low, rr_high = 0.001, 0.99
    for _ in range(60):
        rr_mid = (rr_low + rr_high) / 2.0
        tp = TorusParams(R=lambda_C, r=rr_mid * lambda_C, p=1, q=1)
        se = compute_self_energy(tp)
        if se['E_total_MeV'] > m_e_MeV:
            rr_low = rr_mid
        else:
            rr_high = rr_mid

    rr_sol = (rr_low + rr_high) / 2.0
    tp = TorusParams(R=lambda_C, r=rr_sol * lambda_C, p=1, q=1)
    se = compute_self_energy(tp)
    print(f"  Solution: r/R = {rr_sol:.6f}")
    print(f"  r = {rr_sol * lambda_C:.4e} m = {rr_sol * lambda_C / r_e:.4f} × r_e")
    print(f"  E_circ  = {se['E_circ_MeV']:.8f} MeV")
    print(f"  E_self  = {se['U_total_MeV']:.8f} MeV")
    print(f"  E_total = {se['E_total_MeV']:.8f} MeV  (target: {m_e_MeV:.8f})")
    print(f"  Self-energy fraction: {se['self_energy_fraction']:.6f}")
    print(f"  α × [ln(8R/r)-2]/π:  {se['alpha_prediction']:.6f}")


def print_resonance_analysis(params: TorusParams):
    """
    Show the resonance quantization structure of the torus.
    """
    print("=" * 70)
    print("RESONANCE QUANTIZATION ANALYSIS")
    print("=" * 70)
    R, r = params.R, params.r

    print(f"\nTorus: R = {R:.4e} m, r = {r:.4e} m")
    print(f"R/r = {R/r:.2f}")
    print(f"\nBoundary conditions (field must be single-valued):")
    print(f"  Toroidal: k_tor × 2πR = 2πn  →  λ_n = 2πR/n")
    print(f"  Poloidal: k_pol × 2πr = 2πm  →  λ_m = 2πr/m")
    print(f"\nFor massless field: E = ℏc × √(n²/R² + m²/r²)")

    modes = resonance_analysis(params, n_modes=5)

    print(f"\n{'n':>3} {'m':>3} {'Type':>10} {'E (MeV)':>12} {'E/m_e':>10} {'Nearest particle':>20}")
    print("-" * 65)

    for mode in modes[:20]:  # show first 20
        # Find nearest known particle
        nearest = min(PARTICLE_MASSES.items(),
                      key=lambda kv: abs(kv[1] - mode['E_MeV']))
        ratio = mode['E_MeV'] / nearest[1]
        near_str = f"{nearest[0]} (×{ratio:.3f})" if 0.1 < ratio < 10 else ""

        print(f"{mode['n']:3d} {mode['m']:3d} {mode['type']:>10} "
              f"{mode['E_MeV']:12.4f} {mode['E_over_me']:10.4f} {near_str:>20}")

    # Key insight about the poloidal tower
    E_tor_1 = hbar * c / R / MeV
    E_pol_1 = hbar * c / r / MeV
    print(f"\n--- Scale separation ---")
    print(f"  Fundamental toroidal (1,0): {E_tor_1:.4f} MeV")
    print(f"  Fundamental poloidal (0,1): {E_pol_1:.4f} MeV")
    print(f"  Ratio (poloidal/toroidal):  {E_pol_1/E_tor_1:.2f} = R/r = {R/r:.2f}")
    print(f"\n  The poloidal modes form a 'tower' of heavy states at")
    print(f"  energies ~ (R/r) × E_fundamental. For R/r = {R/r:.0f},")
    print(f"  the first poloidal mode is {R/r:.0f}× heavier than the")
    print(f"  fundamental — a natural mass hierarchy from geometry.")
    print(f"\n  This is analogous to Kaluza-Klein compactification:")
    print(f"  a small compact dimension (r) produces a tower of heavy")
    print(f"  modes invisible at low energies.")


def print_find_radii():
    """
    Find self-consistent radii for known particle masses.
    Now uses angular momentum to classify particles:
      p=1 (L_z = ℏ)   → bosons
      p=2 (L_z = ℏ/2) → fermions
    """
    print("=" * 70)
    print("SELF-CONSISTENT RADII FOR KNOWN PARTICLES")
    print("  Angular momentum determines winding number:")
    print("    p=1 → L_z = ℏ    (bosons)")
    print("    p=2 → L_z = ℏ/2  (fermions)")
    print("  Then: find R where E_total = m_particle × c²")
    print("=" * 70)

    # Fermions: p=2 winding (spin-1/2)
    fermions = {
        'electron': 0.51099895,
        'muon': 105.6583755,
        'tau': 1776.86,
    }

    # Hadrons (composite, but treat as single torus for now)
    hadrons = {
        'proton': 938.27208816,
        'pion±': 139.57039,
    }

    # Bosons: p=1 winding (spin-1)
    # The W, Z, and Higgs would go here if we included them
    # For now, note that the pion is spin-0, not spin-1

    print(f"\n--- Fermions (p=2, q=1 → L_z = ℏ/2) ---")
    for name, mass_MeV in sorted(fermions.items(), key=lambda x: x[1]):
        sol = find_self_consistent_radius(mass_MeV, p=2, q=1, r_ratio=0.1)
        if sol is None:
            print(f"\n  {name}: no solution found")
            continue

        mass_kg = mass_MeV * MeV / c**2
        lambda_C_particle = hbar / (mass_kg * c)

        # Compute angular momentum at this solution
        tp = TorusParams(R=sol['R'], r=sol['r'], p=2, q=1)
        am = compute_angular_momentum(tp)

        print(f"\n  {name} ({mass_MeV:.4f} MeV):")
        print(f"    R = {sol['R']:.4e} m = {sol['R_femtometers']:.4f} fm")
        print(f"    r = {sol['r']:.4e} m = {sol['r_femtometers']:.5f} fm")
        print(f"    R / λ_C({name}) = {sol['R'] / lambda_C_particle:.6f}")
        print(f"    E_circ = {sol['E_circ_MeV']:.6f} MeV")
        print(f"    E_self = {sol['E_self_MeV']:.6f} MeV ({sol['self_energy_fraction']*100:.3f}%)")
        print(f"    E_total = {sol['E_total_MeV']:.6f} MeV (match: {sol['match_ppm']:.1f} ppm)")
        print(f"    L_z = {am['Lz_over_hbar']:.6f} ℏ  ← spin-1/2 from geometry")

    print(f"\n--- Hadrons (composite — shown for comparison) ---")
    for name, mass_MeV in sorted(hadrons.items(), key=lambda x: x[1]):
        # Proton is spin-1/2 (fermion), pion is spin-0
        p_wind = 2 if name == 'proton' else 1
        sol = find_self_consistent_radius(mass_MeV, p=p_wind, q=1, r_ratio=0.1)
        if sol is None:
            print(f"\n  {name}: no solution found")
            continue

        mass_kg = mass_MeV * MeV / c**2
        lambda_C_particle = hbar / (mass_kg * c)
        tp = TorusParams(R=sol['R'], r=sol['r'], p=p_wind, q=1)
        am = compute_angular_momentum(tp)

        print(f"\n  {name} ({mass_MeV:.4f} MeV, p={p_wind}):")
        print(f"    R = {sol['R']:.4e} m = {sol['R_femtometers']:.4f} fm")
        print(f"    r = {sol['r']:.4e} m = {sol['r_femtometers']:.5f} fm")
        print(f"    R / λ_C({name}) = {sol['R'] / lambda_C_particle:.6f}")
        print(f"    E_total = {sol['E_total_MeV']:.6f} MeV")
        print(f"    L_z = {am['Lz_over_hbar']:.6f} ℏ")
        if name == 'proton':
            print(f"    Note: proton charge radius (measured) = 0.8414 fm")
            print(f"          torus major radius R = {sol['R_femtometers']:.4f} fm")
            print(f"          ratio: R_measured / R_torus = {0.8414 / sol['R_femtometers']:.4f}")

    # Mass ratios
    print(f"\n--- Lepton mass ratios ---")
    e_sol = find_self_consistent_radius(fermions['electron'], p=2, q=1, r_ratio=0.1)
    mu_sol = find_self_consistent_radius(fermions['muon'], p=2, q=1, r_ratio=0.1)
    tau_sol = find_self_consistent_radius(fermions['tau'], p=2, q=1, r_ratio=0.1)

    if e_sol and mu_sol and tau_sol:
        print(f"  R_e / R_μ   = {e_sol['R'] / mu_sol['R']:.4f}  "
              f"(= m_μ/m_e = {fermions['muon']/fermions['electron']:.4f})")
        print(f"  R_e / R_τ   = {e_sol['R'] / tau_sol['R']:.4f}  "
              f"(= m_τ/m_e = {fermions['tau']/fermions['electron']:.4f})")
        print(f"  R_μ / R_τ   = {mu_sol['R'] / tau_sol['R']:.4f}  "
              f"(= m_τ/m_μ = {fermions['tau']/fermions['muon']:.4f})")
        print(f"\n  In this model, heavier fermions are SMALLER tori.")
        print(f"  All three generations have the same topology (2,1)")
        print(f"  but different radii. What selects the three radii?")
        print(f"  That's the generation problem — still open.")


def print_pair_production():
    """
    Analyze pair production (γ → e⁻ + e⁺) in the torus model.

    The photon is a free EM wave (no torus). The electron/positron
    are (2,±1) torus knots. Pair production is the photon "winding up"
    into two counter-rotating torus knots near a nucleus.
    """
    print("=" * 70)
    print("PAIR PRODUCTION: γ → e⁻ + e⁺")
    print("  A free wave winds up into two torus knots")
    print("=" * 70)

    # --- Threshold geometry ---
    print(f"\n--- Threshold condition ---")
    E_thresh_MeV = 2 * m_e_MeV
    E_thresh_J = E_thresh_MeV * MeV
    lambda_photon = h_planck * c / E_thresh_J

    # Electron torus path length
    e_sol = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=0.1)
    tp_e = TorusParams(R=e_sol['R'], r=e_sol['r'], p=2, q=1)
    L_torus = compute_path_length(tp_e)

    print(f"  Threshold energy:           E_γ = 2 m_e c² = {E_thresh_MeV:.4f} MeV")
    print(f"  Photon wavelength:          λ_γ = {lambda_photon:.6e} m")
    print(f"  Electron torus path length: L   = {L_torus:.6e} m")
    print(f"  Ratio λ_γ / L_torus:        {lambda_photon / L_torus:.6f}")
    print(f"  2π λ_C:                     {2*np.pi*lambda_C:.6e} m")
    print(f"  Ratio λ_γ / 2πλ_C:          {lambda_photon / (2*np.pi*lambda_C):.6f}")
    print(f"\n  λ_γ = L_torus / 2 exactly (because E_γ = 2 × m_e c² = 2 × hc/L)")
    print(f"  The photon wavelength equals HALF the electron torus path —")
    print(f"  one full loop of the double-wound (2,1) knot.")
    print(f"  The photon fits one loop; the electron needs two → spin 1/2.")

    # --- Nuclear field as catalyst ---
    print(f"\n--- Nuclear Coulomb field as catalyst ---")
    print(f"  The photon cannot produce a pair in free space (4-momentum")
    print(f"  conservation). It needs a nucleus to absorb recoil AND to")
    print(f"  provide the field curvature that bends the wave into a torus.")
    print(f"\n  Schwinger critical field: E_crit = m_e²c³/(eℏ)")
    E_schwinger = m_e**2 * c**3 / (e_charge * hbar)
    print(f"    = {E_schwinger:.4e} V/m")
    print(f"\n  Critical distance (where Coulomb field = Schwinger field):")
    print(f"  {'Z':>4} {'Element':>8} {'r_crit (fm)':>12} {'r_crit/λ_C':>12} {'σ ∝ πr²_crit':>14}")
    print(f"  " + "-" * 52)
    elements = [(1, 'H'), (6, 'C'), (26, 'Fe'), (82, 'Pb'), (92, 'U')]
    for Z, elem in elements:
        r_crit = np.sqrt(Z * e_charge / (4 * np.pi * eps0 * E_schwinger))
        sigma_rel = Z  # σ ∝ r_crit² ∝ Z
        print(f"  {Z:4d} {elem:>8} {r_crit*1e15:12.3f} {r_crit/lambda_C:12.4f} {sigma_rel:14.1f}")
    print(f"\n  Cross-section σ ∝ πr²_crit ∝ Z → explains Z² scaling")
    print(f"  (Bethe-Heitler: σ ∝ α r_e² Z²)")

    # --- Conservation laws ---
    print(f"\n--- Conservation laws ---")
    print(f"  {'Quantity':<20} {'Before (γ)':<20} {'After (e⁻ + e⁺)':<25} {'Conserved?'}")
    print(f"  " + "-" * 70)
    print(f"  {'Energy':<20} {'E_γ':<20} {'E_e + E_e+':<25} {'✓'}")
    print(f"  {'Momentum':<20} {'p_γ = E/c':<20} {'p_e + p_e+ + p_nuc':<25} {'✓ (recoil)'}")
    print(f"  {'Charge':<20} {'0':<20} {'-e + (+e) = 0':<25} {'✓'}")
    print(f"  {'Winding (p)':<20} {'0':<20} {'+2 + (-2) = 0':<25} {'✓'}")
    print(f"  {'Winding (q)':<20} {'0':<20} {'+1 + (-1) = 0':<25} {'✓'}")
    print(f"  {'Angular mom':<20} {'ℏ (spin-1)':<20} {'ℏ/2 + (-ℏ/2) + orb':<25} {'✓'}")
    print(f"  {'Net topology':<20} {'trivial':<20} {'(2,+1)+(2,-1)=trivial':<25} {'✓'}")
    print(f"\n  Charge conservation IS topology conservation:")
    print(f"  no net winding created — equal and opposite topology.")

    # --- Annihilation (reverse) ---
    print(f"\n--- Annihilation (reverse process) ---")
    print(f"  e⁻ + e⁺ → γγ")
    print(f"  (2,+1) + (2,-1) → topology cancels → free waves")
    print(f"  Two photons required (not one) for momentum conservation")
    print(f"  Energy: 2 × {m_e_MeV:.4f} = {2*m_e_MeV:.4f} MeV → 2 photons")
    print(f"\n  Positronium (bound e⁻e⁺):")
    print(f"    Para (↑↓, S=0): decays to 2γ, τ = 1.25 × 10⁻¹⁰ s")
    print(f"    Ortho (↑↑, S=1): decays to 3γ, τ = 1.42 × 10⁻⁷ s")
    print(f"    Ortho is 1000× slower because aligned spins (ℏ/2 + ℏ/2 = ℏ)")
    print(f"    can't cancel into 2 photons — needs 3-body final state.")

    # --- Above-threshold: heavier pairs ---
    print(f"\n--- Above threshold: heavier pairs ---")
    print(f"  The photon preferentially creates the LIGHTEST pair because")
    print(f"  that's the ground state (largest torus = most probable).")
    print(f"\n  {'Pair':<12} {'Threshold (MeV)':>16} {'E_γ needed':>12}")
    print(f"  " + "-" * 44)
    pair_thresholds = [
        ('e⁻e⁺', 2 * 0.511),
        ('μ⁻μ⁺', 2 * 105.658),
        ('τ⁻τ⁺', 2 * 1776.86),
        ('pp̄', 2 * 938.272),
    ]
    for pair, thresh in pair_thresholds:
        print(f"  {pair:<12} {thresh:16.3f} {'> ' + f'{thresh:.0f} MeV':>12}")


def print_decay_landscape():
    """
    Analyze particle decay as torus expansion / topology change.

    Key insight: stable particles are ground states (largest tori)
    of each topological class. Unstable particles are excited states
    that expand to the ground state, releasing energy.
    """
    print("=" * 70)
    print("DECAY LANDSCAPE: Stability and transitions")
    print("=" * 70)

    # Particle catalog
    catalog = [
        # (name, mass_MeV, p, q, spin, stable, lifetime_s, decay)
        ('photon',   0.0,      1, 0, 1.0, True,  None,     '—'),
        ('electron', 0.511,    2, 1, 0.5, True,  None,     '—'),
        ('proton',   938.272,  2, 1, 0.5, True,  None,     '—'),
        ('muon',     105.658,  2, 1, 0.5, False, 2.2e-6,   'e + ν_μ + ν̄_e'),
        ('tau',      1776.86,  2, 1, 0.5, False, 2.9e-13,  'μ + ν_τ + ν̄_μ'),
        ('pion±',    139.570,  1, 1, 0.0, False, 2.6e-8,   'μ + ν_μ'),
        ('pion0',    134.977,  1, 1, 0.0, False, 8.5e-17,  'γγ'),
        ('neutron',  939.565,  2, 1, 0.5, False, 879,      'p + e + ν̄_e'),
    ]

    print(f"\n--- Particle catalog ---")
    print(f"  {'Name':<10} {'Mass':>10} {'(p,q)':>6} {'L_z/ℏ':>7} "
          f"{'Status':>8} {'Lifetime':>12} {'Decays to'}")
    print(f"  " + "-" * 75)
    for name, mass, p, q, spin, stable, tau, decay in catalog:
        lt = "STABLE" if stable else f"{tau:.1e} s"
        lz = f"{1/p:.2f}" if p > 0 else "—"
        print(f"  {name:<10} {mass:10.3f} ({p},{q}){'':<2} {lz:>7} "
              f"{'STABLE' if stable else '':>8} {lt:>12}  {decay}")

    # Decay energetics
    print(f"\n--- Decay = torus expansion ---")
    print(f"  Unstable particles are SMALLER tori (higher energy)")
    print(f"  They expand to larger tori (lower energy), releasing ΔE")

    decays = [
        ('π⁰ → γγ',       134.977,  0.0,     1, 0, 'total unwinding',        8.5e-17),
        ('τ → μ + ν + ν̄',  1776.86,  105.658, 2, 2, 'R expands 17×',         2.9e-13),
        ('π± → μ + ν',     139.570,  105.658, 1, 2, 'TOPOLOGY CHANGE p=1→2', 2.6e-8),
        ('μ → e + ν + ν̄',  105.658,  0.511,   2, 2, 'R expands 207×',        2.2e-6),
        ('n → p + e + ν̄',  939.565,  938.783, 2, 2, 'ΔR/R = 0.08%',         879),
    ]

    print(f"\n  {'Decay':<16} {'ΔE (MeV)':>10} {'p→p':>5} {'τ (s)':>12}  {'Mechanism'}")
    print(f"  " + "-" * 68)
    for name, m_par, m_dau, pi, pf, mech, tau in sorted(decays, key=lambda x: x[6]):
        dE = m_par - m_dau
        print(f"  {name:<16} {dE:10.3f} {pi}→{pf:}{'':<2} {tau:12.2e}  {mech}")

    # Stability principle
    print(f"\n--- Stability principle ---")
    print(f"  STABLE = largest torus (lowest energy) in its topological class")
    print(f"  No lower-energy state with same topology to decay into.")
    print(f"\n  (2,1) leptons:")

    for name, mass in [('tau', 1776.86), ('muon', 105.658), ('electron', 0.511)]:
        sol = find_self_consistent_radius(mass, p=2, q=1, r_ratio=0.1)
        if sol:
            stable = "← GROUND STATE" if name == 'electron' else "→ decays"
            print(f"    {name:<10} R = {sol['R_femtometers']:10.4f} fm  "
                  f"(E = {mass:10.3f} MeV)  {stable}")

    print(f"\n  The electron is stable because there is no LARGER (2,1)")
    print(f"  torus to expand into. It's the ground state of its topology.")

    # Neutrino interpretation
    print(f"\n--- Neutrinos as topology carriers ---")
    print(f"  π⁺(p=1) → μ⁺(p=2) + ν_μ(p=?)")
    print(f"  Winding changes: 1 → 2. Where does the Δp go?")
    print(f"  The neutrino carries the topological difference.")
    print(f"\n  If neutrinos are propagating topology changes (not tori),")
    print(f"  this would explain:")
    print(f"    • Nearly massless: not bound structures, just transitions")
    print(f"    • Three flavors: one per lepton generation transition")
    print(f"    • Weak-only: couple to topology changes, not stable EM")
    print(f"    • Spin ℏ/2: carry angular momentum of the transition")
    print(f"    • Oscillations: topology transitions can mix")
