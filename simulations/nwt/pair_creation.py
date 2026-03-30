"""Pair creation, transient impedance, Euler-Heisenberg, Schwinger, and gradient profile."""

import numpy as np
from .constants import (
    c, hbar, h_planck, e_charge, eps0, mu0, m_e, m_e_MeV, alpha,
    eV, MeV, lambda_C, r_e, a_0, k_e, TorusParams,
)
from .core import (
    compute_self_energy, compute_total_energy, find_self_consistent_radius,
    compute_path_length,
)
from .fields import compute_torus_fields, compute_torus_circuit


def compute_field_landscape(r_ratio_array, p=2, q=1):
    """
    Sweep r/R values, computing field quantities at each self-consistent radius.

    Returns dict of arrays, one entry per r/R value that has a valid solution.
    """
    results = {
        'r_ratio': [], 'R_fm': [], 'r_fm': [], 'log_factor': [],
        'E_over_ES_torus': [], 'E_over_ES_tube': [],
        'Phi_over_Phi0': [], 'Z_over_Z0': [],
        'E_Q_over_E_rms_torus': [], 'E_Q_over_E_rms_tube': [],
    }

    for rr in r_ratio_array:
        sol = find_self_consistent_radius(m_e_MeV, p=p, q=q, r_ratio=rr)
        if sol is None:
            continue
        R = sol['R']
        r = rr * R
        fields = compute_torus_fields(R, r, p=p, q=q)

        results['r_ratio'].append(rr)
        results['R_fm'].append(R * 1e15)
        results['r_fm'].append(r * 1e15)
        results['log_factor'].append(fields['log_factor'])
        results['E_over_ES_torus'].append(fields['E_over_ES_torus'])
        results['E_over_ES_tube'].append(fields['E_over_ES_tube'])
        results['Phi_over_Phi0'].append(fields['Phi_over_Phi0'])
        results['Z_over_Z0'].append(fields['Z_over_Z0'])
        results['E_Q_over_E_rms_torus'].append(
            fields['E_Q_torus'] / fields['E_rms_torus'] if fields['E_rms_torus'] > 0 else 0
        )
        results['E_Q_over_E_rms_tube'].append(
            fields['E_Q_tube'] / fields['E_rms_tube'] if fields['E_rms_tube'] > 0 else 0
        )

    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


def print_pair_creation_analysis():
    """
    Pair creation and torus geometry: what fixes r/R = α?

    The NWT model determines R from E_circ + U_EM = m_e c², but the
    aspect ratio r/R is free. This module investigates whether pair
    production physics provides the missing constraint.

    Key insight: E = hν has no amplitude — a photon's field is entirely
    fixed by frequency. Closing it into a torus should fully determine
    the geometry. The Schwinger critical field E_S = m_e²c³/(eℏ) marks
    where the QED vacuum becomes unstable to pair creation, providing
    a natural field-strength threshold.
    """
    print("=" * 70)
    print("PAIR CREATION AND TORUS GEOMETRY")
    print("What determines the torus aspect ratio r/R = α?")
    print("=" * 70)

    # ── Section 1: The Quantization Insight ───────────────────────────
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  SECTION 1: THE QUANTIZATION INSIGHT                               ║
╚══════════════════════════════════════════════════════════════════════╝

  THE PROBLEM: The NWT self-consistency condition

      E_circ + U_EM = m_e c²

  determines R for any given r/R. But it doesn't fix r/R itself.
  The model has one free parameter — the torus aspect ratio.

  THE CLUE FROM PAIR CREATION:

  A photon has E = hν and NOTHING ELSE. No amplitude parameter.
  The electromagnetic field of a single photon is entirely determined
  by its frequency. There is no dial to turn.

  When a photon converts to an electron-positron pair, it must close
  into a toroidal topology (in the NWT picture). Since the field has
  only one parameter (ν), the torus must be FULLY DETERMINED:

      2 unknowns:  R (major radius), r (minor radius)
      2 equations: (1) energy condition, (2) field matching

  CONSTRAINT COUNTING:
    Constraint 1: E_circ + U_EM = m_e c²        → fixes R for given r/R
    Constraint 2: field amplitude = threshold    → fixes r/R

  What is the threshold? The Schwinger critical field — the field
  strength where the QED vacuum spontaneously creates pairs.""")

    # ── Section 2: The Schwinger Field ────────────────────────────────
    print()
    print("=" * 70)
    print("SECTION 2: THE SCHWINGER FIELD AT r/R = α")
    print("=" * 70)

    E_Schwinger = m_e**2 * c**3 / (e_charge * hbar)
    print(f"""
  The Schwinger critical field:

      E_S = m_e² c³ / (e ℏ) = {E_Schwinger:.4e} V/m

  This is the electric field where the QED vacuum becomes unstable:
  virtual e⁺e⁻ pairs gain enough energy over a Compton wavelength
  to become real. It is the natural scale for pair creation.

  Now compute the RMS field inside the electron torus at r/R = α:""")

    # Compute at r/R = alpha, (2,1) torus knot
    sol = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=alpha)
    if sol is None:
        print("  ERROR: Could not find self-consistent radius at r/R = α")
        print("=" * 70)
        return

    R_alpha = sol['R']
    r_alpha = alpha * R_alpha
    fields_alpha = compute_torus_fields(R_alpha, r_alpha, p=2, q=1)

    print(f"""
  Self-consistent electron torus at r/R = α = {alpha:.6f}:
    R = {R_alpha*1e15:.4f} fm       r = {r_alpha*1e15:.6f} fm
    U_EM = {fields_alpha['U_EM']/MeV:.6f} MeV
    log(8R/r) - 2 = {fields_alpha['log_factor']:.4f}

  Field strengths (from U_EM distributed over the torus volume):

    Volume assumption: full torus (V = 2π²Rr²)
      V_torus = {fields_alpha['V_torus']:.4e} m³
      B_rms = {fields_alpha['B_rms_torus']:.4e} T
      E_rms = {fields_alpha['E_rms_torus']:.4e} V/m
      E_rms / E_Schwinger = {fields_alpha['E_over_ES_torus']:.4f}

    Volume assumption: tube along knot path (V = πr² × L_path)
      V_tube = {fields_alpha['V_tube']:.4e} m³
      B_rms = {fields_alpha['B_rms_tube']:.4e} T
      E_rms = {fields_alpha['E_rms_tube']:.4e} V/m
      E_rms / E_Schwinger = {fields_alpha['E_over_ES_tube']:.4f}

  ┌─────────────────────────────────────────────────────────────────┐
  │  RESULT: E_rms / E_Schwinger ≈ {fields_alpha['E_over_ES_torus']:.1f} (torus) or {fields_alpha['E_over_ES_tube']:.1f} (tube)     │
  │                                                                 │
  │  The internal field of the electron torus is within an ORDER    │
  │  OF MAGNITUDE of the Schwinger critical field. This is          │
  │  remarkable — no parameter was tuned to achieve this.           │
  └─────────────────────────────────────────────────────────────────┘""")

    # ── Section 3: Schwinger Field Landscape ──────────────────────────
    print()
    print("=" * 70)
    print("SECTION 3: SCHWINGER FIELD LANDSCAPE — WHERE IS E_rms/E_S = 1?")
    print("=" * 70)
    print("""
  Sweep r/R from 10⁻⁴ to 0.5. At each value, find the self-consistent
  R via E_circ + U_EM = m_e c², then compute E_rms/E_Schwinger.

  The key question: at what r/R does E_rms = E_Schwinger?
  Does this match α = 0.007297...?
""")

    # Build sweep array: dense near alpha, sparse elsewhere
    r_ratios = np.sort(np.unique(np.concatenate([
        np.logspace(-4, -0.3, 30),       # broad sweep 0.0001 to 0.5
        np.linspace(0.001, 0.02, 10),     # dense near alpha
        [alpha],                           # exact alpha
    ])))

    landscape = compute_field_landscape(r_ratios, p=2, q=1)

    # Print table
    print(f"  {'r/R':>10s}  {'R(fm)':>10s}  {'log_factor':>10s}  {'E/E_S(torus)':>12s}  {'E/E_S(tube)':>12s}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}")

    for i in range(len(landscape['r_ratio'])):
        rr = landscape['r_ratio'][i]
        marker = " ◄── α" if abs(rr - alpha) / alpha < 0.01 else ""
        print(f"  {rr:10.6f}  {landscape['R_fm'][i]:10.4f}  "
              f"{landscape['log_factor'][i]:10.4f}  "
              f"{landscape['E_over_ES_torus'][i]:12.4f}  "
              f"{landscape['E_over_ES_tube'][i]:12.4f}{marker}")

    # Interpolate to find where E_rms/E_S = 1
    print(f"\n  Interpolating to find r/R where E_rms/E_S = 1:")
    for label, arr_key in [('torus volume', 'E_over_ES_torus'), ('tube volume', 'E_over_ES_tube')]:
        vals = landscape[arr_key]
        rrs = landscape['r_ratio']
        # E/E_S decreases with increasing r/R (larger volume → lower field)
        # Find crossing point
        found = False
        for j in range(len(vals) - 1):
            if (vals[j] - 1.0) * (vals[j+1] - 1.0) < 0:
                # Linear interpolation
                frac = (1.0 - vals[j]) / (vals[j+1] - vals[j])
                rr_cross = rrs[j] + frac * (rrs[j+1] - rrs[j])
                ratio_to_alpha = rr_cross / alpha
                print(f"    {label:15s}: E/E_S = 1 at r/R = {rr_cross:.6f}  "
                      f"(= {ratio_to_alpha:.2f} × α)")
                found = True
                break
        if not found:
            if len(vals) > 0 and vals[-1] > 1.0:
                print(f"    {label:15s}: E/E_S > 1 for all r/R in range "
                      f"(minimum {vals[-1]:.2f} at r/R = {rrs[-1]:.4f})")
            elif len(vals) > 0 and vals[0] < 1.0:
                print(f"    {label:15s}: E/E_S < 1 for all r/R in range "
                      f"(maximum {vals[0]:.2f} at r/R = {rrs[0]:.4f})")
            else:
                print(f"    {label:15s}: insufficient data for interpolation")

    print(f"\n  α = {alpha:.6f} for comparison")

    # ── Section 4: Quantum vs Classical Field ─────────────────────────
    print()
    print("=" * 70)
    print("SECTION 4: QUANTUM VS CLASSICAL FIELD")
    print("=" * 70)

    E_Q_tor = fields_alpha['E_Q_torus']
    E_Q_tub = fields_alpha['E_Q_tube']
    E_C_tor = fields_alpha['E_rms_torus']
    E_C_tub = fields_alpha['E_rms_tube']
    omega_alpha = fields_alpha['omega']

    print(f"""
  A single photon at the circulation frequency has a quantum field:

      E_Q = √(ℏω / (ε₀V))

  where ω = E_circ/ℏ is the circulation angular frequency.

  At r/R = α:
    ω = {omega_alpha:.4e} rad/s
    ℏω = {hbar * omega_alpha / MeV:.6f} MeV  (= E_circ)

    Torus volume:
      E_Q = {E_Q_tor:.4e} V/m
      E_C (classical, from current) = {E_C_tor:.4e} V/m
      E_Q / E_C = {E_Q_tor/E_C_tor:.4e}

    Tube volume:
      E_Q = {E_Q_tub:.4e} V/m
      E_C (classical, from current) = {E_C_tub:.4e} V/m
      E_Q / E_C = {E_Q_tub/E_C_tub:.4e}

  MATCHING CONDITION E_Q = E_C:
  ─────────────────────────────
  Setting √(ℏω/(ε₀V)) = √(μ₀ U_EM / V) × c requires:

      ℏω = μ₀ c² U_EM / (1) = U_EM    (using μ₀ε₀c² = 1)

  i.e., the single-photon energy must equal the total stored EM energy.
  But U_EM = E_circ × α × log_factor / π, so the condition is:

      1 = α × log_factor / π
      log_factor = π/α ≈ {np.pi/alpha:.1f}
      log(8R/r) - 2 ≈ {np.pi/alpha:.1f}
      R/r ≈ exp({np.pi/alpha:.1f} + 2)/8 ≈ 10^{(np.pi/alpha + 2)/(np.log(10)*8):.0f}

  This gives r/R → 0 (an impossibly thin torus).

  HONEST CONCLUSION: Naive quantum field matching doesn't work.
  ──────────────────
  The electron is NOT a single Fock state |1⟩. It is a coherent,
  self-interacting bound state. The classical current model (which
  correctly predicts α × log_factor / π for the self-energy fraction)
  already incorporates the full field coherently. Trying to match it
  to a single-photon vacuum fluctuation is a category error.

  The Schwinger field comparison (Section 3) is more physical: it asks
  about the absolute field strength, not the photon number.""")

    # ── Section 5: Flux and Impedance ─────────────────────────────────
    print()
    print("=" * 70)
    print("SECTION 5: FLUX QUANTIZATION AND IMPEDANCE MATCHING")
    print("=" * 70)
    print(f"""
  Two other quantization conditions to check:

  FLUX: Φ = L_ind × I through the torus cross-section
        Φ₀ = h/(2e) = {fields_alpha['Phi_0']:.4e} Wb (flux quantum)

  IMPEDANCE: Z_torus = E_rms × (2πr) / I
             Z₀ = μ₀c = {fields_alpha['Z_0']:.2f} Ω (free-space impedance)
""")

    print(f"  {'r/R':>10s}  {'Φ/Φ₀':>12s}  {'Z/Z₀':>12s}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*12}")

    for i in range(len(landscape['r_ratio'])):
        rr = landscape['r_ratio'][i]
        marker = " ◄── α" if abs(rr - alpha) / alpha < 0.01 else ""
        print(f"  {rr:10.6f}  {landscape['Phi_over_Phi0'][i]:12.4f}  "
              f"{landscape['Z_over_Z0'][i]:12.4f}{marker}")

    # Check for integer flux or Z/Z_0 = 1 near alpha
    Phi_at_alpha = fields_alpha['Phi_over_Phi0']
    Z_at_alpha = fields_alpha['Z_over_Z0']
    print(f"""
  At r/R = α:
    Φ/Φ₀ = {Phi_at_alpha:.6f}   (nearest integer: {round(Phi_at_alpha)})
    Z/Z₀ = {Z_at_alpha:.6f}""")

    # Check if Z/Z_0 = 1 anywhere
    z_vals = landscape['Z_over_Z0']
    z_rrs = landscape['r_ratio']
    for j in range(len(z_vals) - 1):
        if (z_vals[j] - 1.0) * (z_vals[j+1] - 1.0) < 0:
            frac = (1.0 - z_vals[j]) / (z_vals[j+1] - z_vals[j])
            rr_cross = z_rrs[j] + frac * (z_rrs[j+1] - z_rrs[j])
            print(f"    Z/Z₀ = 1 at r/R ≈ {rr_cross:.6f} (= {rr_cross/alpha:.2f} × α)")
            break

    # Check if Phi/Phi_0 = integer anywhere near alpha
    for n_target in [1, 2, 3]:
        for j in range(len(landscape['Phi_over_Phi0']) - 1):
            p0 = landscape['Phi_over_Phi0'][j]
            p1 = landscape['Phi_over_Phi0'][j+1]
            if (p0 - n_target) * (p1 - n_target) < 0:
                frac = (n_target - p0) / (p1 - p0)
                rr_cross = z_rrs[j] + frac * (z_rrs[j+1] - z_rrs[j])
                print(f"    Φ/Φ₀ = {n_target} at r/R ≈ {rr_cross:.6f} (= {rr_cross/alpha:.2f} × α)")
                break

    # ── Section 6: The Nuclear Catalyst ───────────────────────────────
    print()
    print("=" * 70)
    print("SECTION 6: THE NUCLEAR CATALYST")
    print("=" * 70)
    print("""
  Pair creation (γ → e⁺e⁻) requires a nucleus. Why?

  The nucleus provides a Coulomb tidal field — a gradient in E that
  varies across the photon's extent. This gradient breaks the symmetry
  between R and r, providing the "shear" that triggers topology change.

  Coulomb tidal field across the torus diameter (2r):

      ΔE = Z k_e e / d² × 2r    (at distance d from nucleus Z)

  At what distance d does the tidal gradient match E_S / r?
  (i.e., the Schwinger field varying over the tube radius)

      d_crit = (2 Z k_e e r² / E_S)^(1/3)
""")

    print(f"  {'Z':>4s}  {'Element':>8s}  {'d_crit(fm)':>12s}  {'d/λ_C':>10s}  {'d/a_0':>12s}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*12}")

    elements = [(1, 'H'), (6, 'C'), (26, 'Fe'), (82, 'Pb')]
    for Z, name in elements:
        # d_crit = (2 Z k_e e r^2 / E_S)^(1/3)
        # Use r at self-consistent alpha solution
        d_crit = (2 * Z * k_e * e_charge * r_alpha**2 / E_Schwinger)**(1.0/3.0)
        print(f"  {Z:4d}  {name:>8s}  {d_crit*1e15:12.4f}  "
              f"{d_crit/lambda_C:10.4f}  {d_crit/a_0:12.4e}")

    print(f"""
  λ_C = {lambda_C*1e15:.4f} fm (reduced Compton wavelength)
  a_0 = {a_0*1e10:.4f} Å  (Bohr radius)

  PHYSICS INTERPRETATION:
  ───────────────────────
  The critical distances are all sub-Compton — well inside the quantum
  regime. The nucleus doesn't DETERMINE α; pair creation happens at
  r/R = α regardless of the nuclear charge. The nucleus merely provides
  the perturbation that TRIGGERS the topology change (photon → torus).

  This is analogous to nucleation in a phase transition: the nucleus
  is the seed crystal, but the crystal structure is determined by the
  material's own energetics, not by the seed.""")

    # ── Section 7: Assessment ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("SECTION 7: ASSESSMENT")
    print("=" * 70)

    print(f"""
  WHAT WORKS:
  ───────────
  1. The Schwinger field scale is the RIGHT BALLPARK.
     At r/R = α, E_rms/E_S ≈ {fields_alpha['E_over_ES_torus']:.1f} (torus) or {fields_alpha['E_over_ES_tube']:.1f} (tube).
     This is within a small integer factor of unity — remarkable given
     that no parameter was tuned and the ratio could range over many
     orders of magnitude across the r/R sweep.

  2. The quantization insight is sound: E = hν having no amplitude
     means the torus geometry must be fully determined by the energy
     condition plus one field-matching condition.

  3. The nuclear catalyst picture is physically sensible: the nucleus
     triggers topology change at sub-Compton distances, consistent
     with pair creation being a short-distance process.

  WHAT DOESN'T WORK:
  ──────────────────
  1. Naive quantum field matching (E_Q = E_C) gives r/R → 0.
     The electron is a coherent bound state, not a Fock state.

  2. Flux quantization gives Φ/Φ₀ = {Phi_at_alpha:.2f} at α — not an integer.
     (Though this may indicate the relevant quantum is Φ₀/n for
     a (2,1) knot, or that the inductance formula needs refinement.)

  3. Impedance matching gives Z/Z₀ = {Z_at_alpha:.2f} at α — not unity.

  THE FACTOR OF ~{fields_alpha['E_over_ES_torus']:.0f} BETWEEN E_rms AND E_S:
  ──────────────────────────────────────────
  Is this a coincidence, or a correctable systematic?

  Possible corrections that could close the gap:
  • Volume geometry: the field is NOT uniformly distributed. Peak
    fields near the knot center-line may be higher than the RMS.
    The ratio of peak-to-RMS depends on the field profile ∝ 1/ρ,
    where ρ is distance from the knot center-line.
  • (2,1) vs (1,1): the trefoil knot has a non-trivial cross-section
    profile; the "effective" volume may be smaller than 2π²Rr².
  • The Schwinger threshold applies to the PEAK field, not RMS.
    If E_peak/E_rms ~ {fields_alpha['E_over_ES_torus']:.0f}, the condition E_peak = E_S gives
    E_rms/E_S = 1/{fields_alpha['E_over_ES_torus']:.0f} — but we have the opposite sign. The torus
    field slightly EXCEEDS E_S, suggesting the electron exists just
    above the pair-creation threshold (self-consistent: it IS a pair).

  NEXT STEPS:
  ───────────
  1. Compute the actual field distribution inside the (2,1) torus
     knot to get peak/RMS ratio and effective volume.
  2. Check whether E_peak = E_S gives r/R = α when the field profile
     is properly accounted for.
  3. Investigate whether the log_factor ln(8R/r) - 2 provides the
     right correction: at r/R = α, log_factor = {fields_alpha['log_factor']:.2f}.
     The ratio E_rms/E_S depends on log_factor through U_EM.
  4. Consider the pair creation RATE (Schwinger formula involves
     exp(-π E_S/E)) rather than just the threshold.""")

    print("=" * 70)


def compute_circuit_landscape(r_ratio_array, p=2, q=1):
    """
    Sweep r/R, computing LC circuit parameters at each self-consistent radius.

    Returns dict of arrays for plotting/tabling.
    """
    results = {
        'r_ratio': [], 'R_fm': [],
        'omega_ratio': [], 'Z_over_Z0': [],
        'Q': [], 'zeta': [],
        'Z_char': [], 'R_rad': [],
        'log_factor': [],
    }

    for rr in r_ratio_array:
        sol = find_self_consistent_radius(m_e_MeV, p=p, q=q, r_ratio=rr)
        if sol is None:
            continue
        R = sol['R']
        r_val = rr * R
        circ = compute_torus_circuit(R, r_val, p=p, q=q)

        results['r_ratio'].append(rr)
        results['R_fm'].append(R * 1e15)
        results['omega_ratio'].append(circ['omega_ratio'])
        results['Z_over_Z0'].append(circ['Z_over_Z0'])
        results['Q'].append(circ['Q'])
        results['zeta'].append(circ['zeta'])
        results['Z_char'].append(circ['Z_char'])
        results['R_rad'].append(circ['R_rad'])
        results['log_factor'].append(circ['log_factor'])

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


def compute_gap_formation(R, r, gap_fraction_array, p=2, q=1):
    """
    Split-ring resonator model of torus formation.

    Gap width d = gap_fraction × 2πR. As the gap closes (d → 0),
    the gap capacitance diverges and the resonant frequency sweeps
    down through the drive frequency.

    Returns dict of arrays.
    """
    params = TorusParams(R=R, r=r, p=p, q=q)
    se = compute_self_energy(params)
    log_factor = se['log_factor']

    L = mu0 * R * log_factor
    L_path = compute_path_length(params)
    omega_circ = 2 * np.pi * c / L_path

    results = {
        'gap_fraction': [], 'd_fm': [],
        'C_gap': [], 'omega_res': [],
        'omega_res_over_omega_circ': [],
    }

    for gf in gap_fraction_array:
        d = gf * 2 * np.pi * R  # gap width
        if d <= 0:
            continue
        # Parallel-plate gap capacitance (area = πr²)
        C_gap = eps0 * np.pi * r**2 / d
        omega_res = 1.0 / np.sqrt(L * C_gap)

        results['gap_fraction'].append(gf)
        results['d_fm'].append(d * 1e15)
        results['C_gap'].append(C_gap)
        results['omega_res'].append(omega_res)
        results['omega_res_over_omega_circ'].append(omega_res / omega_circ)

    for key in results:
        results[key] = np.array(results[key])

    return results


def compute_transient_overshoot(damping_ratio, omega_ratio_initial,
                                omega_ratio_final, sweep_cycles=20,
                                N_steps=10000):
    """
    Numerically integrate driven damped oscillator with swept natural frequency.

      d²q/dt² + 2ζω₀(t)·dq/dt + ω₀(t)²·q = cos(ω_d·t)

    ω₀(t)/ω_d sweeps linearly from omega_ratio_initial to omega_ratio_final
    over sweep_cycles drive periods.

    Uses Euler-Cromer (symplectic) integration.

    Returns (peak_amplitude, steady_state_amplitude, overshoot_ratio, t_array, q_array).
    """
    omega_d = 2 * np.pi  # normalized drive frequency (period = 1)
    T_sweep = sweep_cycles  # total time in drive periods
    dt = T_sweep / N_steps

    t = np.zeros(N_steps + 1)
    q = np.zeros(N_steps + 1)
    v = np.zeros(N_steps + 1)

    # Initial conditions: start from rest
    q[0] = 0.0
    v[0] = 0.0

    for i in range(N_steps):
        t[i + 1] = t[i] + dt
        # Swept natural frequency
        frac = t[i] / T_sweep
        omega_0 = omega_d * (omega_ratio_initial +
                             (omega_ratio_final - omega_ratio_initial) * frac)

        # Driving force
        F = np.cos(omega_d * t[i])

        # Euler-Cromer (symplectic)
        a = F - 2 * damping_ratio * omega_0 * v[i] - omega_0**2 * q[i]
        v[i + 1] = v[i] + a * dt
        q[i + 1] = q[i] + v[i + 1] * dt

    # Steady-state amplitude: last few cycles
    last_cycles = max(1, int(N_steps * 0.1))
    steady_amp = np.max(np.abs(q[-last_cycles:]))

    # Peak amplitude over entire sweep
    peak_amp = np.max(np.abs(q))

    overshoot = peak_amp / steady_amp if steady_amp > 0 else np.inf

    return peak_amp, steady_amp, overshoot, t, q


def print_transient_impedance_analysis():
    """
    Transient impedance analysis: model the electron torus as an LC circuit
    and investigate whether the factor of ~1.4 from pair creation analysis
    arises from a transient impedance effect during topology formation.
    """
    print("=" * 70)
    print("TRANSIENT IMPEDANCE: LC Circuit Model of Pair Creation")
    print("=" * 70)
    print()
    print("The --pair-creation module found E_rms/E_S = 1.41 (tube volume)")
    print("at r/R = α. The factor of ~1.4 begs explanation.")
    print()
    print("Hypothesis: it's a TRANSIENT IMPEDANCE effect during the")
    print("photon → torus topology change. The torus is an LC circuit.")
    print("A forming torus is a split-ring resonator with a closing gap.")
    print("As the gap closes, the resonant frequency sweeps through the")
    print("drive frequency — a transient that can produce field overshoot.")
    print()

    # ────────────────────────────────────────────────────────────────
    # Section 1: The Circuit Picture
    # ────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("SECTION 1: THE CIRCUIT PICTURE")
    print("─" * 70)
    print()
    print("The electron torus as a lumped-element LC circuit:")
    print()
    print("  L = μ₀R[ln(8R/r) - 2]        (Neumann inductance)")
    print("  C = 4π²ε₀R / ln(8R/r)        (self-capacitance)")
    print("  ω₀ = 1/√(LC)                 (LC resonant frequency)")
    print("  Z_char = √(L/C)              (characteristic impedance)")
    print()

    # Compute at r/R = α
    sol_alpha = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=alpha)
    if sol_alpha is None:
        print("ERROR: Could not find self-consistent radius at r/R = α")
        return

    R_a = sol_alpha['R']
    r_a = alpha * R_a
    circ_alpha = compute_torus_circuit(R_a, r_a, p=2, q=1)

    print(f"At r/R = α = {alpha:.6f}:")
    print(f"  R = {R_a*1e15:.4f} fm,  r = {r_a*1e15:.6f} fm")
    print()
    print(f"  Inductance:     L = {circ_alpha['L']:.4e} H")
    print(f"  Capacitance:    C = {circ_alpha['C']:.4e} F")
    print(f"  ln(8R/r):         {circ_alpha['ln_8Rr']:.4f}")
    print(f"  log_factor:       {circ_alpha['log_factor']:.4f}")
    print()
    print(f"  Resonant freq:  ω₀ = {circ_alpha['omega_0']:.4e} rad/s")
    print(f"  Drive freq:     ω_circ = {circ_alpha['omega_circ']:.4e} rad/s")
    print(f"  Frequency ratio:  ω₀/ω_circ = {circ_alpha['omega_ratio']:.4f}")
    print()

    if circ_alpha['omega_ratio'] < 1.0:
        print(f"  → Driven ABOVE resonance (ω_circ > ω₀): INDUCTIVE regime")
    else:
        print(f"  → Driven BELOW resonance (ω_circ < ω₀): CAPACITIVE regime")
    print()

    print(f"  Characteristic impedance:  Z_char = {circ_alpha['Z_char']:.2f} Ω")
    print(f"  Free-space impedance:      Z₀     = {circ_alpha['Z_0']:.2f} Ω")
    print(f"  Impedance ratio:           Z_char/Z₀ = {circ_alpha['Z_over_Z0']:.4f}")
    print()
    print(f"  ★ Z_char/Z₀ = {circ_alpha['Z_over_Z0']:.4f} — within"
          f" {abs(1 - circ_alpha['Z_over_Z0'])*100:.1f}% of unity!")
    print(f"    The electron torus is nearly impedance-matched to the vacuum.")
    print()

    # Analytical verification
    print("  Analytical check (for (2,1) knot where L_path ≈ 4πR):")
    lf = circ_alpha['log_factor']
    ln8 = circ_alpha['ln_8Rr']
    omega_ratio_anal = (1.0 / np.pi) * np.sqrt(ln8 / lf)
    Z_ratio_anal = (1.0 / (2 * np.pi)) * np.sqrt(lf * ln8)
    print(f"    ω₀/ω_circ = (1/π)√(ln(8R/r)/log_factor)")
    print(f"              = (1/π)√({ln8:.2f}/{lf:.2f}) = {omega_ratio_anal:.4f}")
    print(f"    Z_char/Z₀ = (1/2π)√(log_factor × ln(8R/r))")
    print(f"              = (1/2π)√({lf:.2f} × {ln8:.2f}) = {Z_ratio_anal:.4f}")
    print()

    print(f"  Radiation resistance: R_rad = {circ_alpha['R_rad']:.4e} Ω")
    print(f"  Quality factor:      Q = {circ_alpha['Q']:.2f}")
    print(f"  Damping ratio:       ζ = {circ_alpha['zeta']:.4e}")
    print(f"  Critical resistance: R_crit = 2·Z_char = {circ_alpha['R_crit']:.2f} Ω")
    print()
    print(f"  Reactive impedances at ω_circ:")
    print(f"    X_L = ω_circ·L = {circ_alpha['X_L']:.2f} Ω")
    print(f"    X_C = 1/(ω_circ·C) = {circ_alpha['X_C']:.2f} Ω")
    print(f"    X_net = X_L - X_C = {circ_alpha['X_net']:.2f} Ω  (inductive)")
    print()

    # ────────────────────────────────────────────────────────────────
    # Section 2: Circuit Parameter Landscape
    # ────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("SECTION 2: CIRCUIT PARAMETER LANDSCAPE")
    print("─" * 70)
    print()
    print("Sweep r/R from 10⁻⁴ to 0.5, compute LC parameters at each")
    print("self-consistent electron radius:")
    print()

    r_ratios = np.concatenate([
        np.logspace(-4, -2, 15),
        np.linspace(0.015, 0.05, 15),
        np.linspace(0.06, 0.5, 15),
    ])
    landscape = compute_circuit_landscape(r_ratios, p=2, q=1)

    # Print table
    print(f"  {'r/R':>10s}  {'ω₀/ω_circ':>10s}  {'Z_char/Z₀':>10s}"
          f"  {'Q':>12s}  {'ζ':>12s}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}")

    # Select ~15 representative points for the table
    n_pts = len(landscape['r_ratio'])
    if n_pts > 15:
        indices = np.linspace(0, n_pts - 1, 15, dtype=int)
    else:
        indices = range(n_pts)

    for i in indices:
        rr = landscape['r_ratio'][i]
        marker = " ← α" if abs(rr - alpha) / alpha < 0.1 else ""
        print(f"  {rr:10.6f}  {landscape['omega_ratio'][i]:10.4f}"
              f"  {landscape['Z_over_Z0'][i]:10.4f}"
              f"  {landscape['Q'][i]:12.2f}"
              f"  {landscape['zeta'][i]:12.4e}{marker}")
    print()

    # Interpolate crossing points
    rr_arr = landscape['r_ratio']
    or_arr = landscape['omega_ratio']
    zz_arr = landscape['Z_over_Z0']
    ze_arr = landscape['zeta']

    def find_crossing(x, y, target):
        """Find x where y crosses target by linear interpolation."""
        for i in range(len(y) - 1):
            if (y[i] - target) * (y[i+1] - target) <= 0:
                # Linear interpolation
                frac = (target - y[i]) / (y[i+1] - y[i]) if y[i+1] != y[i] else 0.5
                return x[i] + frac * (x[i+1] - x[i])
        return None

    rr_omega_match = find_crossing(rr_arr, or_arr, 1.0)
    rr_Z_match = find_crossing(rr_arr, zz_arr, 1.0)
    rr_zeta_half = find_crossing(rr_arr, ze_arr, 0.5)

    print("  Crossing points:")
    if rr_omega_match is not None:
        print(f"    ω₀ = ω_circ  (resonance match)   at r/R = {rr_omega_match:.6f}"
              f"  (α = {alpha:.6f}, ratio = {rr_omega_match/alpha:.2f})")
    else:
        print(f"    ω₀ = ω_circ  (resonance match)   NOT FOUND in scan range")
        print(f"    (ω₀/ω_circ < 1 throughout — always driven above resonance)")

    if rr_Z_match is not None:
        print(f"    Z_char = Z₀  (impedance match)   at r/R = {rr_Z_match:.6f}"
              f"  (α = {alpha:.6f}, ratio = {rr_Z_match/alpha:.2f})")
    else:
        print(f"    Z_char = Z₀  (impedance match)   NOT FOUND in scan range")

    if rr_zeta_half is not None:
        print(f"    ζ = 0.5      (critical damping)   at r/R = {rr_zeta_half:.6f}"
              f"  (α = {alpha:.6f}, ratio = {rr_zeta_half/alpha:.2f})")
    else:
        print(f"    ζ = 0.5      (critical damping)   NOT FOUND in scan range")
    print()

    # What is ω₀/ω_circ at r/R = α (from the scan)?
    # Find closest point
    idx_alpha = np.argmin(np.abs(rr_arr - alpha))
    print(f"  At the physical point (r/R closest to α = {alpha:.6f}):")
    print(f"    r/R = {rr_arr[idx_alpha]:.6f}")
    print(f"    ω₀/ω_circ = {or_arr[idx_alpha]:.4f}")
    print(f"    Z_char/Z₀  = {zz_arr[idx_alpha]:.4f}")
    print(f"    Q          = {landscape['Q'][idx_alpha]:.2f}")
    print(f"    ζ          = {ze_arr[idx_alpha]:.4e}")
    print()

    # ────────────────────────────────────────────────────────────────
    # Section 3: Split-Ring Formation
    # ────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("SECTION 3: SPLIT-RING FORMATION")
    print("─" * 70)
    print()
    print("Model: forming torus = split-ring resonator with closing gap.")
    print("Gap width d = gap_fraction × 2πR.")
    print("Gap capacitance: C_gap = ε₀πr²/d (parallel plate).")
    print("As gap closes: C_gap → ∞, ω_res → 0.")
    print()

    gap_fracs = np.logspace(-7, np.log10(0.5), 50)
    gap_data = compute_gap_formation(R_a, r_a, gap_fracs, p=2, q=1)

    print(f"  {'gap_frac':>10s}  {'d (fm)':>12s}  {'C_gap (F)':>12s}"
          f"  {'ω_res/ω_circ':>12s}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*12}")

    n_gap = len(gap_data['gap_fraction'])
    if n_gap > 15:
        g_indices = np.linspace(0, n_gap - 1, 15, dtype=int)
    else:
        g_indices = range(n_gap)

    for i in g_indices:
        marker = ""
        if abs(gap_data['omega_res_over_omega_circ'][i] - 1.0) < 0.15:
            marker = " ← ≈ resonance"
        print(f"  {gap_data['gap_fraction'][i]:10.6f}"
              f"  {gap_data['d_fm'][i]:12.4f}"
              f"  {gap_data['C_gap'][i]:12.4e}"
              f"  {gap_data['omega_res_over_omega_circ'][i]:12.4f}{marker}")
    print()

    # Find resonance crossing
    gf_arr = gap_data['gap_fraction']
    or_gap = gap_data['omega_res_over_omega_circ']
    rr_gap_cross = find_crossing(gf_arr, or_gap, 1.0)

    if rr_gap_cross is not None:
        d_cross = rr_gap_cross * 2 * np.pi * R_a
        print(f"  Resonance crossing (ω_res = ω_circ):")
        print(f"    gap_fraction = {rr_gap_cross:.6f}")
        print(f"    gap width d  = {d_cross*1e15:.4f} fm")
        print(f"    d / (2πR)    = {rr_gap_cross:.6f}")
        print(f"    d / r        = {d_cross / r_a:.4f}")
        print()
        print(f"  Physical interpretation: resonance occurs when the gap is")
        print(f"  {rr_gap_cross*100:.3f}% of the circumference, or d/r = {d_cross/r_a:.4f}.")
        if d_cross < r_a:
            print(f"  The gap is SMALLER than the tube radius — the torus is")
            print(f"  nearly closed when resonance sweeps through.")
        else:
            print(f"  The gap is LARGER than the tube radius — resonance occurs")
            print(f"  early in the formation process.")
    else:
        print(f"  Resonance crossing NOT FOUND in gap sweep range.")
        # Check if always above or below
        if len(or_gap) > 0:
            print(f"  ω_res/ω_circ ranges from {or_gap[0]:.4f} to {or_gap[-1]:.4f}")
    print()

    # ────────────────────────────────────────────────────────────────
    # Section 4: Transient Overshoot
    # ────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("SECTION 4: TRANSIENT OVERSHOOT")
    print("─" * 70)
    print()
    print("Numerically integrate the driven damped oscillator with swept")
    print("natural frequency to find the transient overshoot factor:")
    print()
    print("  d²q/dt² + 2ζω₀(t)·dq/dt + ω₀(t)²·q = cos(ω_d·t)")
    print()
    print("ω₀/ω_d sweeps from high (open gap) to the final value (closed torus).")
    print()

    # The physical scenario: gap closing means ω_res sweeps DOWN.
    # Start above drive frequency, end at the final ω₀/ω_circ ratio.
    omega_ratio_final = circ_alpha['omega_ratio']
    omega_ratio_initial = 5.0  # start well above resonance (large gap)

    # Sweep damping ratios
    zeta_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0]

    print(f"  Sweep: ω₀/ω_d from {omega_ratio_initial:.1f} → {omega_ratio_final:.4f}"
          f" over 20 drive periods")
    print()
    print(f"  {'ζ':>8s}  {'Peak amp':>10s}  {'Steady amp':>10s}  {'Overshoot':>10s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")

    overshoot_results = []
    for z in zeta_values:
        peak, steady, overshoot, _, _ = compute_transient_overshoot(
            z, omega_ratio_initial, omega_ratio_final,
            sweep_cycles=20, N_steps=10000
        )
        overshoot_results.append((z, peak, steady, overshoot))
        print(f"  {z:8.3f}  {peak:10.4f}  {steady:10.4f}  {overshoot:10.4f}")
    print()

    # Find ζ where overshoot ≈ 1.4
    zeta_arr = np.array([r[0] for r in overshoot_results])
    over_arr = np.array([r[3] for r in overshoot_results])

    zeta_14 = find_crossing(zeta_arr, over_arr, 1.4)
    if zeta_14 is not None:
        print(f"  Overshoot = 1.4 occurs at ζ ≈ {zeta_14:.4f}")
    else:
        print(f"  Overshoot = 1.4 NOT FOUND in damping range")
        if len(over_arr) > 0:
            print(f"  (overshoot ranges from {over_arr.min():.4f} to {over_arr.max():.4f})")

    # Physical damping ratio
    phys_zeta = circ_alpha['zeta']
    print()
    print(f"  Physical damping ratio (radiation resistance): ζ = {phys_zeta:.4e}")

    # Compute overshoot at physical damping
    peak_phys, steady_phys, over_phys, _, _ = compute_transient_overshoot(
        phys_zeta, omega_ratio_initial, omega_ratio_final,
        sweep_cycles=20, N_steps=10000
    )
    print(f"  Overshoot at physical ζ: {over_phys:.4f}")
    print()

    # Also try a finer sweep around the physical regime
    print("  Fine sweep near very low damping (physical regime):")
    zeta_fine = [1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2]
    print(f"  {'ζ':>10s}  {'Overshoot':>10s}")
    print(f"  {'─'*10}  {'─'*10}")
    for z in zeta_fine:
        _, _, ov, _, _ = compute_transient_overshoot(
            z, omega_ratio_initial, omega_ratio_final,
            sweep_cycles=20, N_steps=10000
        )
        print(f"  {z:10.2e}  {ov:10.4f}")
    print()

    # ────────────────────────────────────────────────────────────────
    # Section 5: The Frequency Response (Bode Plot)
    # ────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("SECTION 5: THE FREQUENCY RESPONSE")
    print("  (Yes, we're doing a Bode plot of the electron)")
    print("─" * 70)
    print()
    print("Transfer function of the LC circuit at the operating point:")
    print()
    print("  H(s) = ω₀²/(s² + 2ζω₀s + ω₀²)")
    print("  |H(jω)| = 1/√[(1 - (ω/ω₀)²)² + (2ζω/ω₀)²]")
    print()

    # Compute at operating point ω = ω_circ
    omega_ratio_op = circ_alpha['omega_circ'] / circ_alpha['omega_0']  # ω/ω₀
    zeta_op = circ_alpha['zeta']

    denom_real = 1 - omega_ratio_op**2
    denom_imag = 2 * zeta_op * omega_ratio_op
    H_mag = 1.0 / np.sqrt(denom_real**2 + denom_imag**2)
    H_phase_deg = -np.degrees(np.arctan2(denom_imag, denom_real))

    print(f"  Operating point: ω/ω₀ = ω_circ/ω₀ = {omega_ratio_op:.4f}")
    print(f"  Damping:         ζ = {zeta_op:.4e}")
    print()
    print(f"  Gain:    |H| = {H_mag:.6f}  ({20*np.log10(H_mag):.2f} dB)")
    print(f"  Phase:   ∠H  = {H_phase_deg:.2f}°")
    print()

    # Gain margin: frequency where phase = -180° (for this simple system,
    # the 2nd-order system reaches -180° only at ω → ∞ for ζ > 0)
    print(f"  For a 2nd-order system:")
    print(f"    Phase = -180° only at ω → ∞ (gain margin is infinite)")
    print(f"    Phase margin = 180° + ∠H(ω_circ) = {180 + H_phase_deg:.2f}°")
    print()

    # Sweep ω/ω₀ for the "Bode plot" table
    omega_sweep = np.array([0.01, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5,
                            2.0, omega_ratio_op, 5.0, 10.0])
    omega_sweep = np.sort(np.unique(omega_sweep))

    print(f"  {'ω/ω₀':>8s}  {'|H| (dB)':>10s}  {'∠H (°)':>10s}  {'Note':>20s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*20}")

    for w in omega_sweep:
        dr = 1 - w**2
        di = 2 * zeta_op * w
        H = 1.0 / np.sqrt(dr**2 + di**2)
        ph = -np.degrees(np.arctan2(di, dr))
        note = ""
        if abs(w - 1.0) < 0.001:
            note = "resonance"
        elif abs(w - omega_ratio_op) / omega_ratio_op < 0.01:
            note = "★ operating point"
        print(f"  {w:8.4f}  {20*np.log10(H):10.2f}  {ph:10.2f}  {note:>20s}")
    print()

    # How gain varies with r/R at the operating point
    print("  Gain at operating point vs r/R:")
    print(f"  {'r/R':>10s}  {'ω/ω₀':>8s}  {'|H| (dB)':>10s}  {'∠H (°)':>10s}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*10}")

    for i in indices:
        rr = landscape['r_ratio'][i]
        or_val = landscape['omega_ratio'][i]
        if or_val > 0:
            w_op = 1.0 / or_val  # ω_circ/ω₀
            dr = 1 - w_op**2
            di = 2 * landscape['zeta'][i] * w_op
            H = 1.0 / np.sqrt(dr**2 + di**2)
            ph = -np.degrees(np.arctan2(di, dr))
            marker = " ← α" if abs(rr - alpha) / alpha < 0.1 else ""
            print(f"  {rr:10.6f}  {w_op:8.4f}  {20*np.log10(H):10.2f}"
                  f"  {ph:10.2f}{marker}")
    print()

    # ────────────────────────────────────────────────────────────────
    # Section 6: Assessment
    # ────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("SECTION 6: ASSESSMENT")
    print("─" * 70)
    print()

    print("  Summary of key results:")
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │ At r/R = α (the physical electron):                │")
    print(f"  │   Z_char/Z₀   = {circ_alpha['Z_over_Z0']:.4f}"
          f"  ({abs(1-circ_alpha['Z_over_Z0'])*100:.1f}% from unity)      │")
    print(f"  │   ω₀/ω_circ   = {circ_alpha['omega_ratio']:.4f}"
          f"  (driven above resonance)     │")
    print(f"  │   Q            = {circ_alpha['Q']:.1f}"
          f"                              │")
    print(f"  │   ζ (physical) = {circ_alpha['zeta']:.2e}"
          f"  (extreme underdamping)  │")
    print(f"  └─────────────────────────────────────────────────────┘")
    print()

    # Does impedance picture explain 1.4?
    print("  Q: Does the impedance picture explain the factor of 1.4?")
    print()
    print(f"  The transient overshoot at the physical damping ratio")
    print(f"  ζ = {phys_zeta:.2e} gives overshoot factor = {over_phys:.4f}.")
    print()
    if abs(over_phys - 1.4) / 1.4 < 0.15:
        print(f"  ★ YES — the transient overshoot is close to 1.4!")
        print(f"    The swept-resonance during gap closure naturally produces")
        print(f"    field amplification near the pair-creation value.")
    elif over_phys > 1.3:
        print(f"  PARTIAL — the overshoot is in the right ballpark ({over_phys:.2f})")
        print(f"  but doesn't precisely match 1.4. The simple swept-frequency")
        print(f"  model captures the qualitative physics but misses quantitative")
        print(f"  details (nonlinear gap geometry, distributed vs lumped circuit).")
    else:
        print(f"  NOT DIRECTLY — at the physical ζ, the transient overshoot")
        print(f"  is {over_phys:.4f}, not 1.4. The extremely low damping means")
        print(f"  the transient rings for many cycles. The simple model may need:")
        print(f"  • Distributed-element corrections (the torus is not lumped)")
        print(f"  • Nonlinear gap geometry (not parallel-plate)")
        print(f"  • Radiation damping during formation (time-dependent losses)")
    print()

    # Does any matching condition select α?
    print("  Q: Does any matching condition select r/R = α?")
    print()

    print(f"  Impedance match (Z_char = Z₀):")
    if rr_Z_match is not None:
        print(f"    Occurs at r/R = {rr_Z_match:.6f} (= {rr_Z_match/alpha:.2f}α)")
        if abs(rr_Z_match - alpha) / alpha < 0.15:
            print(f"    ★ CLOSE TO α — impedance matching may select the geometry!")
        else:
            print(f"    Not at α, but the near-match (Z/Z₀ = {circ_alpha['Z_over_Z0']:.4f})"
                  f" is striking.")
    else:
        closest_Z = zz_arr[np.argmin(np.abs(zz_arr - 1.0))]
        print(f"    Not found. Closest Z/Z₀ = {closest_Z:.4f} "
              f"at r/R = {rr_arr[np.argmin(np.abs(zz_arr - 1.0))]:.6f}")
    print()

    # α as impedance ratio
    print("  Connection: α as impedance ratio")
    R_K = h_planck / e_charge**2  # von Klitzing constant ≈ 25812.8 Ω
    R_Q = R_K / (2 * np.pi)      # quantum of resistance / 2π
    print(f"    von Klitzing constant: R_K = h/e² = {R_K:.2f} Ω")
    print(f"    Z₀ = μ₀c = {circ_alpha['Z_0']:.2f} Ω")
    print(f"    α = Z₀/(2R_K) = {circ_alpha['Z_0']/(2*R_K):.8f}")
    print(f"    Actual α       = {alpha:.8f}")
    print(f"    Match: {abs(circ_alpha['Z_0']/(2*R_K) - alpha)/alpha*1e6:.1f} ppm"
          f" — exact (this IS the definition of α).")
    print()
    print(f"    This is not a coincidence — α = e²/(4πε₀ħc) = Z₀/(2R_K)")
    print(f"    is just the fine-structure constant written in impedance units.")
    print(f"    What IS notable: the torus LC circuit recovers Z_char ≈ Z₀")
    print(f"    at the same r/R = α. The topology builds in the right scale.")
    print()

    # Honest assessment
    print("  WHAT WORKS:")
    print("  ───────────")
    print(f"  1. Z_char/Z₀ ≈ {circ_alpha['Z_over_Z0']:.2f} at r/R = α.")
    print(f"     The electron torus is nearly impedance-matched to free space.")
    print(f"     This is a necessary condition for efficient energy transfer")
    print(f"     during pair creation (vacuum → particle).")
    print()
    print(f"  2. The split-ring model gives a concrete formation mechanism:")
    print(f"     as the torus gap closes, the resonant frequency sweeps")
    print(f"     through the drive frequency, producing a transient.")
    print()
    print(f"  3. The circuit language maps naturally onto pair creation:")
    print(f"     the vacuum photon is the drive signal, the forming torus")
    print(f"     is the resonator, and pair creation is impedance matching.")
    print()

    print("  WHAT DOESN'T WORK (YET):")
    print("  ────────────────────────")
    print(f"  1. The lumped-element model breaks down when the wavelength")
    print(f"     is comparable to the circuit size (it always is here —")
    print(f"     the photon wavelength IS the torus circumference).")
    print()
    print(f"  2. The radiation resistance estimate uses the magnetic dipole")
    print(f"     formula, giving ζ ≈ {phys_zeta:.1e}. This is too small")
    print(f"     to produce significant transient effects. A more realistic")
    print(f"     damping mechanism (radiation during formation, nonlinear")
    print(f"     coupling) could change the picture significantly.")
    print()
    print(f"  3. The factor of 1.4 is not yet explained. The transient")
    print(f"     overshoot depends sensitively on the damping ratio and")
    print(f"     the sweep profile. The simple linear sweep may not capture")
    print(f"     the actual formation dynamics.")
    print()

    print("  NEXT STEPS:")
    print("  ───────────")
    print(f"  1. Distributed-element model: treat the torus as a transmission")
    print(f"     line rather than a lumped LC. This is natural for the torus")
    print(f"     topology and handles the wavelength ≈ size regime.")
    print()
    print(f"  2. Better damping: compute radiation losses during formation")
    print(f"     using the time-dependent multipole moments of the closing")
    print(f"     split-ring configuration.")
    print()
    print(f"  3. Nonlinear gap model: the parallel-plate capacitor is crude.")
    print(f"     The actual gap geometry is toroidal, and the field")
    print(f"     concentration at the gap edges matters.")
    print()
    print(f"  4. Connection to Schwinger rate: the transient field enhancement")
    print(f"     during formation determines the pair-creation probability")
    print(f"     via exp(-π E_S/E). Map the time-dependent fields to the")
    print(f"     Schwinger rate integral.")

    print()
    print("=" * 70)



def schwinger_pair_rate(E_mag):
    """
    Schwinger pair-creation rate (leading n=1 term), vectorized.

    w = (e*E)^2 / (4 pi^3 hbar^2 c) * exp(-pi E_S / |E|)

    Mask at E < 0.1 E_S to avoid exp underflow.
    Returns numpy array of rates (1/m^3/s).
    """
    E_mag = np.asarray(E_mag, dtype=float)
    E_S = m_e**2 * c**3 / (e_charge * hbar)  # 1.32e18 V/m
    rate = np.zeros_like(E_mag)
    mask = E_mag > 0.1 * E_S
    if np.any(mask):
        E_m = E_mag[mask]
        prefactor = (e_charge * E_m)**2 / (4.0 * np.pi**3 * hbar**2 * c)
        exponent = -np.pi * E_S / E_m
        rate[mask] = prefactor * np.exp(exponent)
    return rate


def compute_coulomb_pair_snapshot(Z=92, N=200, L=5.0):
    """
    2D cylindrical grid (rho, z) with Coulomb + quantum photon fields.

    Computes Schwinger pair-creation rate and cylindrically-weighted rate.
    Returns dict with grids, profiles, peak locations, integrated rate.
    """
    E_S = m_e**2 * c**3 / (e_charge * hbar)

    # Nuclear radius (fm): R_nuc ~ 1.2 * A^(1/3), A ~ 2.5*Z for heavy nuclei
    A_nuc = 2.5 * Z  # approximate mass number
    R_nucleus = 1.2e-15 * A_nuc**(1.0/3.0)  # meters
    rho_min = max(R_nucleus, 0.02 * lambda_C)

    # Grid
    rho_arr = np.linspace(rho_min, L * lambda_C, N)
    z_arr = np.linspace(-L * lambda_C, L * lambda_C, N)
    rho_grid, z_grid = np.meshgrid(rho_arr, z_arr, indexing='ij')  # shape (N, N)

    # Distance from origin
    r_dist = np.sqrt(rho_grid**2 + z_grid**2)
    r_dist = np.maximum(r_dist, rho_min)  # floor at rho_min

    # Coulomb field: E = k_e * Z * e / r^2
    E_coulomb = k_e * Z * e_charge / r_dist**2  # |E| in V/m

    # Photon quantum field at Compton scale
    omega_C = m_e * c**2 / hbar
    V_mode = (2.0 * np.pi * lambda_C)**3
    E_photon = np.sqrt(hbar * omega_C / (eps0 * V_mode))

    # Azimuthal average: E_eff^2 = E_Coulomb^2 + E_photon^2 / 2
    # (factor 1/2 from averaging cos^2(phi) for linear polarization)
    E_eff = np.sqrt(E_coulomb**2 + E_photon**2 / 2.0)

    # Schwinger pair rate w(rho, z)
    rate_grid = schwinger_pair_rate(E_eff)

    # Cylindrically-weighted rate: W = w * 2*pi*rho
    cyl_rate_grid = rate_grid * 2.0 * np.pi * rho_grid

    # Radial profile at z=0 (midplane)
    j_mid = N // 2  # z ~ 0 index
    rate_radial = rate_grid[:, j_mid]
    cyl_rate_radial = cyl_rate_grid[:, j_mid]
    E_radial = E_eff[:, j_mid]

    # Pair emergence probability density P(rho) = rho^3 * exp(-pi*rho^2/(Z*alpha*lambda_C^2))
    # The Schwinger exponent exp(-pi*E_S/E) with E = Z*alpha*E_S*(lambda_C/rho)^2
    # gives exp(-pi*rho^2/(Z*alpha*lambda_C^2)). The rho^3 factor encodes:
    #   rho^2 from the spherical shell area (pair emergence surface)
    #   rho   from the pair separation phase space
    # This is the dominant physical factor; the prefactor E^2 ~ rho^-4 is a
    # slowly-varying polynomial that shifts the peak location by < 10%.
    x_arr = rho_arr / lambda_C
    emergence_radial = x_arr**3 * np.exp(-np.pi * x_arr**2 / (Z * alpha))

    # Find peak of emergence probability (radial profile)
    i_peak = np.argmax(emergence_radial)
    rho_peak_grid = rho_arr[i_peak]

    # Analytical peak: rho_peak = lambda_C * sqrt(3*Z*alpha / (2*pi))
    rho_peak_analytical = lambda_C * np.sqrt(3.0 * Z * alpha / (2.0 * np.pi))

    # Axial profile at rho = rho_peak (analytical)
    i_peak_a = np.argmin(np.abs(rho_arr - rho_peak_analytical))
    rate_axial = rate_grid[i_peak_a, :]
    cyl_rate_axial = cyl_rate_grid[i_peak_a, :]

    # Integrated rate: integrate over full (rho, z) grid
    drho = rho_arr[1] - rho_arr[0]
    dz = z_arr[1] - z_arr[0]
    # Integral of w * 2*pi*rho * drho * dz
    total_rate = np.sum(cyl_rate_grid) * drho * dz

    return {
        'Z': Z,
        'N': N,
        'L': L,
        'rho_arr': rho_arr,
        'z_arr': z_arr,
        'rho_min': rho_min,
        'R_nucleus': R_nucleus,
        'E_S': E_S,
        'E_photon': E_photon,
        'E_photon_over_ES': E_photon / E_S,
        'E_coulomb_grid': E_coulomb,
        'E_eff_grid': E_eff,
        'rate_grid': rate_grid,
        'cyl_rate_grid': cyl_rate_grid,
        'E_radial': E_radial,
        'rate_radial': rate_radial,
        'cyl_rate_radial': cyl_rate_radial,
        'emergence_radial': emergence_radial,
        'rate_axial': rate_axial,
        'cyl_rate_axial': cyl_rate_axial,
        'rho_peak_grid': rho_peak_grid,
        'rho_peak_analytical': rho_peak_analytical,
        'total_rate': total_rate,
        'omega_C': omega_C,
        'V_mode': V_mode,
    }


def compute_breit_wheeler_snapshot(N=200, L=10.0):
    """
    Two counter-propagating photons at Compton frequency, linearly polarized.

    At t=0: E fields add (2*E_0*cos(kz)), B fields cancel -> pure E.
    Returns dict with E_max/E_S, peak rate (essentially zero), Schwinger exponent.
    """
    E_S = m_e**2 * c**3 / (e_charge * hbar)
    omega_C = m_e * c**2 / hbar
    k_C = omega_C / c  # = 1/lambda_C
    V_mode = (2.0 * np.pi * lambda_C)**3
    E_0 = np.sqrt(hbar * omega_C / (eps0 * V_mode))

    # At t=0: E fields constructively interfere, B fields cancel
    E_max = 2.0 * E_0
    E_max_over_ES = E_max / E_S

    # Schwinger exponent
    schwinger_exponent = -np.pi * E_S / E_max
    schwinger_log10 = schwinger_exponent / np.log(10)

    # z-grid for spatial profile
    z_arr = np.linspace(-L * lambda_C, L * lambda_C, N)
    E_profile = E_max * np.cos(k_C * z_arr)
    E_mag_profile = np.abs(E_profile)

    # Peak rate (vanishingly small)
    rate_peak = schwinger_pair_rate(np.array([E_max]))[0]

    return {
        'E_0': E_0,
        'E_0_over_ES': E_0 / E_S,
        'E_max': E_max,
        'E_max_over_ES': E_max_over_ES,
        'E_S': E_S,
        'schwinger_exponent': schwinger_exponent,
        'schwinger_log10': schwinger_log10,
        'rate_peak': rate_peak,
        'z_arr': z_arr,
        'E_profile': E_profile,
        'omega_C': omega_C,
        'k_C': k_C,
        'V_mode': V_mode,
    }


def print_euler_heisenberg_analysis():
    """
    Euler-Heisenberg effective Lagrangian: pair creation on a 2D grid.

    Computes where Schwinger pairs preferentially form near a Coulomb source,
    and compares the peak-creation radius to the NWT self-consistent radius.
    """
    print("=" * 70)
    print("  EULER-HEISENBERG EFFECTIVE LAGRANGIAN")
    print("  Pair Creation Grid Computation")
    print("=" * 70)
    print()

    # ==================================================================
    # Section 1: The Euler-Heisenberg Effective Lagrangian
    # ==================================================================
    print("─" * 70)
    print("  1. THE EULER-HEISENBERG EFFECTIVE LAGRANGIAN")
    print("─" * 70)
    print()
    print("  The EH effective Lagrangian adds quantum corrections to Maxwell:")
    print()
    print("    L_EH = L_Maxwell + ξ [(E²−c²B²)² + 7c²(E·B)²]")
    print()
    print("  where ξ = 2α²ε₀²ℏ³ / (45 m_e⁴ c⁵)")
    print()

    # Compute xi
    xi = 2.0 * alpha**2 * eps0**2 * hbar**3 / (45.0 * m_e**4 * c**5)

    # Schwinger critical field
    E_S = m_e**2 * c**3 / (e_charge * hbar)

    # Physical interpretation threshold
    E_threshold_eV = 2.0 * m_e * c**2  # pair creation threshold

    print(f"  Key constants:")
    print()
    print(f"    {'Quantity':30s}  {'Value':>15s}  {'Units'}")
    print(f"    {'─'*30}  {'─'*15}  {'─'*20}")
    print(f"    {'ξ (EH coupling)':30s}  {xi:15.2e}  {'m³/(V²·s)'}")
    print(f"    {'E_Schwinger':30s}  {E_S:15.3e}  {'V/m'}")
    print(f"    {'E_S in terms of m_e':30s}  {'m²c³/(eℏ)':>15s}  {'—'}")
    print(f"    {'Pair threshold':30s}  {E_threshold_eV/eV:15.0f}  {'eV'}")
    print(f"    {'λ_C (reduced Compton)':30s}  {lambda_C:15.3e}  {'m'}")
    print(f"    {'λ_C':30s}  {lambda_C*1e15:15.2f}  {'fm'}")
    print()
    print("  Physical meaning: virtual e⁺e⁻ pairs go real when the field does")
    print(f"  work ≥ 2m_e c² over one Compton wavelength:")
    print()
    print(f"    eE · λ_C ≥ 2 m_e c²   ⟹   E ≥ E_S = {E_S:.3e} V/m")
    print()

    # ==================================================================
    # Section 2: Photon Quantum Field at Compton Scale
    # ==================================================================
    print("─" * 70)
    print("  2. PHOTON QUANTUM FIELD AT COMPTON SCALE")
    print("─" * 70)
    print()

    omega_C = m_e * c**2 / hbar
    V_mode = (2.0 * np.pi * lambda_C)**3
    E_Q = np.sqrt(hbar * omega_C / (eps0 * V_mode))

    print(f"  Single photon in a Compton-scale cavity:")
    print()
    print(f"    E_Q = √(ℏω_C / (ε₀ V))   with V = (2πλ_C)³")
    print()
    print(f"    ω_C     = {omega_C:.4e} rad/s  (Compton frequency)")
    print(f"    V_mode  = {V_mode:.4e} m³")
    print(f"    E_Q     = {E_Q:.4e} V/m")
    print(f"    E_Q/E_S = {E_Q/E_S:.4e}")
    print()
    print(f"  Breit-Wheeler (two photons, constructive):")
    print()
    E_BW = 2.0 * E_Q
    BW_exp = -np.pi * E_S / E_BW
    BW_log10 = BW_exp / np.log(10)
    print(f"    2E_Q/E_S = {E_BW/E_S:.4e}")
    print(f"    Schwinger exponent = exp({BW_exp:.1f}) ~ 10^({BW_log10:.0f})")
    print()
    print("  ★ This is why Breit-Wheeler is perturbative —")
    print("    the Schwinger formula gives effectively zero rate.")
    print("    BW pair creation requires Feynman diagrams, not EH.")
    print()

    # Regime comparison table
    print(f"  {'Regime':25s}  {'E/E_S':>12s}  {'Schwinger exp':>14s}  {'Mechanism'}")
    print(f"  {'─'*25}  {'─'*12}  {'─'*14}  {'─'*15}")
    print(f"  {'Single photon (E_Q)':25s}  {E_Q/E_S:12.4e}  {'—':>14s}  {'—'}")
    print(f"  {'Two photons (2E_Q)':25s}  {E_BW/E_S:12.4e}  {'~10^' + f'{BW_log10:.0f}':>10s}  {'perturbative'}")
    for Z_tab in [1, 26, 82, 92]:
        E_at_lC = k_e * Z_tab * e_charge / lambda_C**2
        ratio = E_at_lC / E_S
        if ratio > 0.1:
            exp_val = -np.pi / ratio
            exp_str = f"exp({exp_val:.1f})"
        else:
            exp_str = "≈ 0"
        print(f"  {'Coulomb Z=' + str(Z_tab) + ' at λ_C':25s}  {ratio:12.4e}  {exp_str:>14s}  "
              f"{'Schwinger' if ratio > 0.1 else '—'}")
    print()

    # ==================================================================
    # Section 3: Coulomb Catalyst — Analytical Profile
    # ==================================================================
    print("─" * 70)
    print("  3. COULOMB CATALYST — ANALYTICAL PROFILE")
    print("─" * 70)
    print()
    print("  Coulomb field: E(r) = k_e Ze / r²")
    print()
    print("  In Schwinger units:  E/E_S = Zα (λ_C/r)²")
    print()
    print("  Critical radius where E = E_S:")
    print("    r_crit = √(Zα) · λ_C")
    print()

    print(f"  {'Z':>5s}  {'Element':>8s}  {'E/E_S at λ_C':>14s}  "
          f"{'r_crit/λ_C':>12s}  {'r_crit (fm)':>12s}  {'Note'}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*14}  {'─'*12}  {'─'*12}  {'─'*20}")
    elements = [(1, 'H'), (6, 'C'), (26, 'Fe'), (82, 'Pb'), (92, 'U')]
    for Z_val, elem in elements:
        E_ratio = Z_val * alpha
        r_crit = np.sqrt(Z_val * alpha) * lambda_C
        note = ""
        if Z_val * alpha > 1.0:
            note = "supercritical"
        elif E_ratio > 0.5:
            note = "strong field"
        print(f"  {Z_val:5d}  {elem:>8s}  {E_ratio:14.4e}  "
              f"{np.sqrt(Z_val * alpha):12.4f}  {r_crit*1e15:12.3f}  {note}")

    print()
    print(f"  Note: Z > 1/α ≈ {1.0/alpha:.0f} would give supercritical Coulomb field")
    print(f"  at r = λ_C. Uranium (Z=92) gives E/E_S = {92*alpha:.4f} at λ_C.")
    print()

    # ==================================================================
    # Section 4: Field Snapshot — 2D Grid
    # ==================================================================
    print("─" * 70)
    print("  4. FIELD SNAPSHOT — 2D CYLINDRICAL GRID")
    print("─" * 70)
    print()

    snap = compute_coulomb_pair_snapshot(Z=92)

    print(f"  Grid parameters (Z = {snap['Z']}, Uranium):")
    print(f"    ρ range:  [{snap['rho_min']/lambda_C:.4f}, {snap['L']:.1f}] λ_C")
    print(f"    z range:  [−{snap['L']:.1f}, {snap['L']:.1f}] λ_C")
    print(f"    Grid:     {snap['N']} × {snap['N']} = {snap['N']**2} points")
    print(f"    R_nucleus = {snap['R_nucleus']*1e15:.1f} fm"
          f" ({snap['R_nucleus']/lambda_C:.4f} λ_C)")
    print()
    print(f"  Photon amplitude:")
    print(f"    E_photon   = {snap['E_photon']:.4e} V/m")
    print(f"    E_Q/E_S    = {snap['E_photon_over_ES']:.4e}")
    print()

    # Radial field profile at z=0
    rho_arr = snap['rho_arr']
    E_radial = snap['E_radial']
    E_S = snap['E_S']

    print(f"  Radial field profile at z = 0:")
    print()
    print(f"    {'ρ/λ_C':>10s}  {'E (V/m)':>14s}  {'E/E_S':>12s}  {'Note'}")
    print(f"    {'─'*10}  {'─'*14}  {'─'*12}  {'─'*20}")

    # Sample at specific radii
    sample_rhos = [0.05, 0.1, 0.2, 0.3, 0.5, 0.57, 0.8, 1.0, 2.0, 3.0]
    for rho_lc in sample_rhos:
        rho_m = rho_lc * lambda_C
        idx = np.argmin(np.abs(rho_arr - rho_m))
        E_val = E_radial[idx]
        ratio = E_val / E_S
        note = ""
        if abs(rho_lc - 0.57) < 0.01:
            note = "← emergence peak"
        elif ratio > 1.0:
            note = "E > E_S"
        elif ratio > 0.5:
            note = "strong field"
        print(f"    {rho_lc:10.2f}  {E_val:14.3e}  {ratio:12.4e}  {note}")
    print()
    print("  Note: photon field is negligible compared to Coulomb at pair-creation radii.")
    print()

    # ==================================================================
    # Section 5: Pair Creation Map
    # ==================================================================
    print("─" * 70)
    print("  5. PAIR CREATION MAP")
    print("─" * 70)
    print()

    rate_radial = snap['rate_radial']
    cyl_rate_radial = snap['cyl_rate_radial']

    print(f"  Schwinger rate at z = 0 (midplane):")
    print()
    print(f"    {'ρ/λ_C':>10s}  {'w (m⁻³s⁻¹)':>14s}  "
          f"{'W=w·2πρ (m⁻²s⁻¹)':>20s}  {'Note'}")
    print(f"    {'─'*10}  {'─'*14}  {'─'*20}  {'─'*20}")

    for rho_lc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.57, 0.7, 0.8, 1.0, 1.5, 2.0]:
        rho_m = rho_lc * lambda_C
        idx = np.argmin(np.abs(rho_arr - rho_m))
        w_val = rate_radial[idx]
        W_val = cyl_rate_radial[idx]
        note = ""
        if abs(rho_lc - 0.57) < 0.01:
            note = "← emergence peak"
        if w_val > 0:
            print(f"    {rho_lc:10.2f}  {w_val:14.4e}  {W_val:20.4e}  {note}")
        else:
            print(f"    {rho_lc:10.2f}  {'0':>14s}  {'0':>20s}  {note}")
    print()
    print("  Note: the raw rate w(ρ) decreases monotonically (stronger field")
    print("  at smaller ρ). The Schwinger formula also becomes unreliable at")
    print("  small ρ where the field gradient violates the LCFA.")
    print()

    # Pair emergence probability — the key quantity
    print("  Pair emergence probability density P(ρ):")
    print()
    print("    P(ρ) ∝ ρ³ × exp(−π ρ²/(Zα λ_C²))")
    print()
    print("    The Schwinger exponent exp(−πE_S/E) is the dominant physical")
    print("    factor. The ρ³ weighting encodes:")
    print("      ρ² — spherical shell area (pair emergence surface)")
    print("      ρ  — pair separation phase space")
    print()

    emergence_radial = snap['emergence_radial']
    rho_peak_g = snap['rho_peak_grid']
    rho_peak_a = snap['rho_peak_analytical']

    print(f"  Peak of pair emergence density:")
    print(f"    Grid peak:        ρ_peak = {rho_peak_g/lambda_C:.4f} λ_C"
          f" = {rho_peak_g*1e15:.2f} fm")
    print(f"    Analytical peak:  ρ_peak = {rho_peak_a/lambda_C:.4f} λ_C"
          f" = {rho_peak_a*1e15:.2f} fm")
    print()
    print(f"  Analytical formula: ρ_peak = λ_C √(3Zα/2π)")
    print(f"    = λ_C √(3 × {92} × {alpha:.6f} / 2π)")
    print(f"    = λ_C × {np.sqrt(3.0*92*alpha/(2.0*np.pi)):.4f}")
    print()

    # Radial profile P(rho) — show how it peaks and decays
    P_max = np.max(emergence_radial) if np.max(emergence_radial) > 0 else 1.0
    print(f"  Radial profile P(ρ) — normalized to peak:")
    print()
    print(f"    {'ρ/λ_C':>10s}  {'P/P_max':>12s}  {'Bar'}")
    print(f"    {'─'*10}  {'─'*12}  {'─'*30}")
    for rho_lc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]:
        rho_m = rho_lc * lambda_C
        idx = np.argmin(np.abs(rho_arr - rho_m))
        P_norm = emergence_radial[idx] / P_max if P_max > 0 else 0
        bar_len = int(P_norm * 30)
        bar = "█" * bar_len
        print(f"    {rho_lc:10.2f}  {P_norm:12.4f}  {bar}")
    print()

    # Axial profile
    z_arr = snap['z_arr']
    cyl_rate_axial = snap['cyl_rate_axial']
    W_ax_max = np.max(cyl_rate_axial) if np.max(cyl_rate_axial) > 0 else 1.0
    print(f"  Axial profile W(z) at ρ = ρ_peak — normalized to peak:")
    print()
    print(f"    {'z/λ_C':>10s}  {'W/W_max':>12s}  {'Bar'}")
    print(f"    {'─'*10}  {'─'*12}  {'─'*30}")
    for z_lc in [-3.0, -2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 3.0]:
        z_m = z_lc * lambda_C
        idx = np.argmin(np.abs(z_arr - z_m))
        W_norm = cyl_rate_axial[idx] / W_ax_max if W_ax_max > 0 else 0
        bar_len = int(W_norm * 30)
        bar = "█" * bar_len
        print(f"    {z_lc:10.1f}  {W_norm:12.4f}  {bar}")
    print()

    print(f"  Total integrated pair rate:")
    print(f"    Γ = ∫ w · 2πρ dρ dz = {snap['total_rate']:.4e} s⁻¹")
    print()

    # ==================================================================
    # Section 6: Geometry Comparison — Grid Peak vs NWT
    # ==================================================================
    print("─" * 70)
    print("  6. GEOMETRY COMPARISON — GRID PEAK vs NWT")
    print("─" * 70)
    print()

    # Get NWT self-consistent radius at pair creation threshold (2 m_e c^2)
    # During pair creation the nascent torus carries the PAIR energy 2m_e c^2,
    # so the relevant NWT radius is for that energy, not single-electron m_e c^2.
    sol = find_self_consistent_radius(2.0 * m_e_MeV, p=1, q=1, r_ratio=alpha)
    if sol is not None:
        R_NWT = sol['R']
        R_NWT_lC = sol['R_over_lambda_C']
    else:
        # Fallback: R ~ hbar c / (2 m_e c^2) = lambda_C / 2
        R_NWT_lC = 0.50
        R_NWT = R_NWT_lC * lambda_C

    agreement = abs(rho_peak_a / lambda_C - R_NWT_lC) / R_NWT_lC * 100

    print(f"  Pair creation peak (Coulomb, Z=92):")
    print(f"    ρ_peak = {rho_peak_a/lambda_C:.4f} λ_C = {rho_peak_a*1e15:.2f} fm")
    print()
    print(f"  NWT self-consistent radius (pair threshold, 2m_e c²):")
    print(f"    R_NWT  = {R_NWT_lC:.4f} λ_C = {R_NWT*1e15:.2f} fm")
    print()
    print(f"  Agreement:  |ρ_peak − R_NWT| / R_NWT = {agreement:.0f}%")
    print()

    # Breit-Wheeler contrast
    bw = compute_breit_wheeler_snapshot()

    print(f"  Breit-Wheeler contrast:")
    print(f"    E_max/E_S      = {bw['E_max_over_ES']:.4e}")
    print(f"    Schwinger exp  = exp({bw['schwinger_exponent']:.1f})"
          f" ~ 10^({bw['schwinger_log10']:.0f})")
    print(f"    Peak rate      = {bw['rate_peak']:.4e} m⁻³s⁻¹")
    print()
    print("  ★ BW creates pairs at the SAME geometry — the torus size is")
    print("    determined by the mass-shell condition (ℏω = 2m_e c²),")
    print("    not the creation mechanism.")
    print()
    print("    Coulomb-Schwinger:  provides the rate (WHERE and HOW FAST)")
    print("    Mass-shell:         provides the geometry (WHAT SIZE)")
    print()

    # Comparison table
    print(f"  {'Quantity':35s}  {'Coulomb (Z=92)':>16s}  {'Breit-Wheeler':>16s}")
    print(f"  {'─'*35}  {'─'*16}  {'─'*16}")
    print(f"  {'E_max/E_S':35s}  {92*alpha:16.4f}  {bw['E_max_over_ES']:16.4e}")
    print(f"  {'Schwinger exponent':35s}  "
          f"{-np.pi/(92*alpha):16.1f}  {bw['schwinger_exponent']:16.1f}")
    print(f"  {'ρ_peak/λ_C':35s}  {rho_peak_a/lambda_C:16.4f}  {'— (uniform)':>16s}")
    print(f"  {'Pair creation mechanism':35s}  {'Schwinger':>16s}  {'perturbative':>16s}")
    print(f"  {'EH Lagrangian applicable?':35s}  {'YES':>16s}  {'NO':>16s}")
    print()

    # ==================================================================
    # Section 7: Assessment
    # ==================================================================
    print("─" * 70)
    print("  7. ASSESSMENT")
    print("─" * 70)
    print()

    print("  What the grid shows:")
    print("  ─────────────────────")
    print("  1. Pair creation concentrated in ring at ρ ~ 0.5-0.6 λ_C")
    print("  2. Schwinger rate extremely sensitive to E/E_S")
    print("     (exponential suppression below E_S)")
    print("  3. Photon field negligible at relevant radii")
    print("     (Coulomb dominates for Z ≳ 80)")
    print()

    print("  What the grid cannot show:")
    print("  ──────────────────────────")
    print("  1. No dynamics — this is a rate, not a trajectory")
    print("  2. No torus formation — WHERE pairs form, not HOW they")
    print("     organize into a torus")
    print("  3. No aspect ratio — the creation ring is fat")
    print("     (thickness ~ λ_C), not a thin torus (r/R = α)")
    print("  4. Schwinger formula assumes uniform field; LCFA")
    print("     (locally constant field approximation) corrections")
    print("     can be O(1) for Coulomb fields")
    print("  5. BW invisible to Schwinger formula — it lives in")
    print("     the perturbative regime (requires Feynman diagrams)")
    print()

    print("  What IS suggestive:")
    print("  ────────────────────")
    print(f"  • ρ_peak ~ R_NWT within {agreement:.0f}% — pair creation occurs")
    print("    at the \"right\" radius for electron formation")
    print("  • Coulomb catalyst provides field strength (E/E_S ~ 0.67)")
    print("    at exactly the right spatial scale (~λ_C)")
    print("  • The pair-creation ring radius depends on Z, α, and λ_C —")
    print("    the same ingredients that determine the NWT electron")
    print()

    # Summary box
    print(f"  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  EULER-HEISENBERG — HEADLINE NUMBERS                       │")
    print(f"  │                                                             │")
    print(f"  │  E_Schwinger        = {E_S:10.3e} V/m                  │")
    print(f"  │  E/E_S at λ_C (Z=92) = {92*alpha:8.4f}                         │")
    print(f"  │  ρ_peak (Coulomb)   = {rho_peak_a/lambda_C:8.4f} λ_C"
          f" (analytical)           │")
    print(f"  │  R_NWT (electron)   = {R_NWT_lC:8.4f} λ_C"
          f" (self-consistent)       │")
    print(f"  │  Agreement          = {agreement:8.0f}%"
          f"                            │")
    print(f"  │  BW: E_max/E_S      = {bw['E_max_over_ES']:8.4e}"
          f"  (perturbative)        │")
    print(f"  │  BW: Schwinger exp  ~ 10^({bw['schwinger_log10']:.0f})"
          f"  (effectively zero)       │")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    print()
    print("=" * 70)


def compute_gradient_profile(Z=92, N=200, L=5.0):
    """
    Field gradient analysis on the Coulomb pair-creation grid.

    Computes numerical and analytical gradients, LCFA validity (Keldysh gamma),
    violence parameter V, energy densities, and pair creation power.
    Returns dict with all grids/profiles, spreading base snapshot data.
    """
    snap = compute_coulomb_pair_snapshot(Z=Z, N=N, L=L)
    E_S = snap['E_S']
    rho_arr = snap['rho_arr']
    z_arr = snap['z_arr']
    E_coulomb = snap['E_coulomb_grid']
    rate_grid = snap['rate_grid']
    rho_peak_a = snap['rho_peak_analytical']

    drho = rho_arr[1] - rho_arr[0]
    dz = z_arr[1] - z_arr[0]

    # Numerical gradient of |E| on the (rho, z) grid
    dE_drho, dE_dz = np.gradient(E_coulomb, drho, dz)
    grad_E_mag = np.sqrt(dE_drho**2 + dE_dz**2)

    # LCFA parameter gamma = m_e c^2 |grad E| / (e E^2)
    # Equivalently gamma = (hbar c |grad E|) / (e E^2) ... but the standard
    # Keldysh-like definition for Schwinger is gamma = m_e c omega_eff / (e E)
    # where omega_eff = c |grad E| / E, giving gamma = m_e c^2 |grad E| / (e E^2)
    gamma_grid = np.zeros_like(E_coulomb)
    mask = E_coulomb > 0.01 * E_S
    gamma_grid[mask] = (m_e * c**2 * grad_E_mag[mask]) / (e_charge * E_coulomb[mask]**2)

    # Violence parameter V = e |dE/dr| lambda_C^2 / (2 m_e c^2)
    # For Coulomb at z=0: V = Z*alpha*(lambda_C/rho)^3
    V_grid = e_charge * grad_E_mag * lambda_C**2 / (2.0 * m_e * c**2)

    # Energy density u = eps0 E^2 / 2
    u_grid = eps0 * E_coulomb**2 / 2.0

    # Pair creation power density P = w * 2 m_e c^2
    power_grid = rate_grid * 2.0 * m_e * c**2

    # Midplane profiles (z=0)
    j_mid = N // 2
    grad_E_radial = grad_E_mag[:, j_mid]
    gamma_radial = gamma_grid[:, j_mid]
    V_radial = V_grid[:, j_mid]
    u_radial = u_grid[:, j_mid]
    power_radial = power_grid[:, j_mid]
    dE_drho_radial = np.abs(dE_drho[:, j_mid])

    # Analytical profiles at z=0 for validation
    # |dE/dr| = 2 k_e Z e / rho^3  (for z=0, r = rho)
    grad_E_analytical = 2.0 * k_e * Z * e_charge / rho_arr**3
    # gamma = 2 / ((E/E_S)(rho/lambda_C))  where E/E_S = Z*alpha*(lambda_C/rho)^2
    gamma_analytical = 2.0 / ((Z * alpha * (lambda_C / rho_arr)**2) * (rho_arr / lambda_C))
    # Simplifies to: gamma = 2 rho^3 / (Z * alpha * lambda_C^3)
    # ... no, let's be explicit:
    # E/E_S = Z*alpha*(lambda_C/rho)^2
    # gamma = 2/((E/E_S)*(rho/lambda_C)) = 2*rho^2 / (Z*alpha*lambda_C^2 * (rho/lambda_C))
    #       = 2*rho / (Z*alpha*lambda_C) ... wait let me recalculate
    # gamma = m_e c^2 |dE/dr| / (e E^2)
    # E = k_e Z e / rho^2,  |dE/dr| = 2 k_e Z e / rho^3
    # gamma = m_e c^2 * 2 k_e Z e / rho^3 / (e * (k_e Z e / rho^2)^2)
    #       = m_e c^2 * 2 k_e Z e * rho^4 / (rho^3 * e * k_e^2 * Z^2 * e^2)
    #       = 2 m_e c^2 * rho / (k_e * Z * e^2)
    # Now k_e e^2 / (m_e c^2) = r_e (classical electron radius)
    # So gamma = 2 rho / (Z * r_e) = 2 rho / (Z * alpha * lambda_C)
    # At rho = lambda_C: gamma = 2/(Z*alpha) = 2/(92*0.00730) = 2.98
    # At rho = 0.57*lambda_C: gamma = 2*0.57/(92*0.00730) = 1.70  YES!
    gamma_analytical = 2.0 * rho_arr / (Z * alpha * lambda_C)
    # V = Z*alpha*(lambda_C/rho)^3
    V_analytical = Z * alpha * (lambda_C / rho_arr)**3

    # Axial profile at rho_peak
    i_peak_a = np.argmin(np.abs(rho_arr - rho_peak_a))
    grad_E_axial = grad_E_mag[i_peak_a, :]
    gamma_axial = gamma_grid[i_peak_a, :]
    V_axial = V_grid[i_peak_a, :]

    # Total integrated pair creation power
    cyl_power_grid = power_grid * 2.0 * np.pi * snap['rho_arr'][:, np.newaxis] * np.ones_like(z_arr)[np.newaxis, :]
    # Actually need rho_grid
    rho_grid = snap['rho_arr'][:, np.newaxis] * np.ones((1, N))
    cyl_power_grid = power_grid * 2.0 * np.pi * rho_grid
    total_power = np.sum(cyl_power_grid) * drho * dz

    # V=1 crossing: Z*alpha*(lambda_C/rho)^3 = 1 => rho = lambda_C * (Z*alpha)^(1/3)
    rho_V1 = lambda_C * (Z * alpha)**(1.0 / 3.0)

    result = dict(snap)  # spread base snapshot
    result.update({
        'grad_E_mag': grad_E_mag,
        'dE_drho': dE_drho,
        'dE_dz': dE_dz,
        'gamma_grid': gamma_grid,
        'V_grid': V_grid,
        'u_grid': u_grid,
        'power_grid': power_grid,
        'grad_E_radial': grad_E_radial,
        'gamma_radial': gamma_radial,
        'V_radial': V_radial,
        'u_radial': u_radial,
        'power_radial': power_radial,
        'dE_drho_radial': dE_drho_radial,
        'grad_E_analytical': grad_E_analytical,
        'gamma_analytical': gamma_analytical,
        'V_analytical': V_analytical,
        'grad_E_axial': grad_E_axial,
        'gamma_axial': gamma_axial,
        'V_axial': V_axial,
        'total_power': total_power,
        'rho_V1': rho_V1,
        'drho': drho,
        'dz': dz,
    })
    return result


def print_gradient_profile_analysis():
    """
    Field gradient analysis: LCFA validity, violence parameter, energy densities.

    Extends the Euler-Heisenberg pair creation analysis by asking HOW VIOLENT
    the process is: computes field gradients, the LCFA validity parameter
    (Keldysh gamma), gradient work on virtual pairs, and energy densities.
    """
    Z = 92
    gp = compute_gradient_profile(Z=Z, N=200, L=5.0)
    E_S = gp['E_S']
    rho_arr = gp['rho_arr']
    rho_peak_a = gp['rho_peak_analytical']

    print("=" * 70)
    print("  FIELD GRADIENT PROFILE")
    print("  LCFA Validity, Violence Parameter, Energy Densities")
    print("=" * 70)
    print()
    print("  The --euler-heisenberg module established WHERE pairs form")
    print("  (ring at rho ~ 0.57 lambda_C). This module asks:")
    print("  HOW VIOLENT is the process?")
    print()

    # ==================================================================
    # Section 1: Coulomb Field Gradient — Analytical
    # ==================================================================
    print("─" * 70)
    print("  1. COULOMB FIELD GRADIENT — ANALYTICAL")
    print("─" * 70)
    print()
    print("  For a Coulomb field E = k_e Z e / r^2:")
    print()
    print("    |dE/dr| = 2E/r = 2 k_e Z e / r^3")
    print()
    print("  Dimensionless gradient (natural units):")
    print()
    print("    lambda_C |dE/dr| / E_S = 2 Z alpha (lambda_C/r)^3")
    print()

    rho_vals_lC = np.array([0.1, 0.2, 0.3, 0.5, 0.57, 0.8, 1.0, 2.0])
    rho_vals = rho_vals_lC * lambda_C

    print(f"    {'rho/lambda_C':>12s}  {'|dE/dr| (V/m^2)':>16s}  {'Dimensionless':>14s}")
    print(f"    {'─'*12}  {'─'*16}  {'─'*14}")
    for x in rho_vals_lC:
        rho = x * lambda_C
        grad = 2.0 * k_e * Z * e_charge / rho**3
        dimless = 2.0 * Z * alpha * (1.0 / x)**3
        marker = "  <-- lambda_C" if abs(x - 1.0) < 0.01 else ""
        marker = "  <-- emergence peak" if abs(x - 0.57) < 0.01 else marker
        print(f"    {x:12.2f}  {grad:16.3e}  {dimless:14.4f}{marker}")
    print()

    grad_at_lC = 2.0 * k_e * Z * e_charge / lambda_C**3
    dimless_at_lC = 2.0 * Z * alpha
    print(f"  At lambda_C: |dE/dr| = {grad_at_lC:.3e} V/m^2")
    print(f"  Dimensionless gradient = 2*Z*alpha = {dimless_at_lC:.4f}")
    print()

    # ==================================================================
    # Section 2: LCFA Validity — The Keldysh Parameter
    # ==================================================================
    print("─" * 70)
    print("  2. LCFA VALIDITY — THE KELDYSH PARAMETER")
    print("─" * 70)
    print()
    print("  The LCFA (locally constant field approximation) treats the field")
    print("  as uniform over the pair's Compton wavelength. Its validity is")
    print("  controlled by the Keldysh-like parameter:")
    print()
    print("    gamma = m_e c^2 |grad E| / (e E^2)")
    print()
    print("  Physical meaning: gamma = (field variation scale) / (pair size)")
    print()
    print("  For Coulomb at z=0:")
    print()
    print("    gamma = 2 rho / (Z alpha lambda_C)")
    print()
    print("  LCFA valid when gamma << 1 (field uniform over pair extent)")
    print()

    print(f"    {'rho/lambda_C':>12s}  {'E/E_S':>10s}  {'gamma':>8s}  {'Status'}")
    print(f"    {'─'*12}  {'─'*10}  {'─'*8}  {'─'*20}")

    for x in rho_vals_lC:
        E_over_ES = Z * alpha * (1.0 / x)**2
        gamma_val = 2.0 * x / (Z * alpha)
        if gamma_val < 0.5:
            status = "LCFA valid"
        elif gamma_val < 1.0:
            status = "borderline"
        elif gamma_val < 2.0:
            status = "MARGINAL"
        else:
            status = "LCFA fails"
        marker = ""
        if abs(x - 0.57) < 0.01:
            marker = " <-- EMERGENCE PEAK"
        elif abs(x - 1.0) < 0.01:
            marker = " <-- lambda_C"
        print(f"    {x:12.2f}  {E_over_ES:10.4f}  {gamma_val:8.3f}  {status}{marker}")
    print()

    gamma_peak = 2.0 * 0.57 / (Z * alpha)
    print(f"  KEY RESULT: At emergence peak (rho = 0.57 lambda_C):")
    print(f"    gamma = {gamma_peak:.3f}")
    print()
    print(f"  gamma ~ 1.7 means the LCFA is MARGINAL at the pair emergence peak.")
    print(f"  The Schwinger rate formula is qualitatively correct (right order of")
    print(f"  magnitude in the exponent) but quantitative corrections are O(1).")
    print()

    # Bar chart: gamma vs rho/lambda_C
    print("  gamma vs rho/lambda_C:")
    print()
    bar_rhos = [0.1, 0.2, 0.3, 0.5, 0.57, 0.8, 1.0, 2.0]
    max_gamma_display = 6.0
    bar_width = 40
    for x in bar_rhos:
        g = 2.0 * x / (Z * alpha)
        bar_len = int(min(g / max_gamma_display, 1.0) * bar_width)
        marker = " *" if abs(x - 0.57) < 0.01 else ""
        lcfa_mark = "|" if bar_len > 0 else ""
        # Mark gamma=1 position
        g1_pos = int((1.0 / max_gamma_display) * bar_width)
        bar = ""
        for i in range(bar_width):
            if i < bar_len:
                bar += "█"
            elif i == g1_pos:
                bar += "┊"
            else:
                bar += " "
        print(f"    {x:4.2f}  {bar}  {g:.2f}{marker}")
    print(f"           {'':>{int((1.0/max_gamma_display)*bar_width)}}┊")
    print(f"           {'':>{int((1.0/max_gamma_display)*bar_width)}}gamma=1")
    print(f"    (* = emergence peak)")
    print()

    # ==================================================================
    # Section 3: Gradient Work — Violence Parameter
    # ==================================================================
    print("─" * 70)
    print("  3. GRADIENT WORK — VIOLENCE PARAMETER")
    print("─" * 70)
    print()
    print("  The gradient does work on a virtual pair separated by lambda_C:")
    print()
    print("    W_grad = e |dE/dr| lambda_C^2")
    print()
    print("  The violence parameter normalizes this to the pair threshold:")
    print()
    print("    V = W_grad / (2 m_e c^2) = e |dE/dr| lambda_C^2 / (2 m_e c^2)")
    print()
    print("  For Coulomb at z=0:")
    print()
    print("    V = Z alpha (lambda_C/rho)^3")
    print()
    print("  V > 1 means the gradient alone rips pairs apart over one lambda_C.")
    print()

    print(f"    {'rho/lambda_C':>12s}  {'W_grad (MeV)':>12s}  {'V':>8s}  {'Interpretation'}")
    print(f"    {'─'*12}  {'─'*12}  {'─'*8}  {'─'*25}")
    for x in rho_vals_lC:
        grad = 2.0 * k_e * Z * e_charge / (x * lambda_C)**3
        W_grad = e_charge * grad * lambda_C**2
        V_val = Z * alpha * (1.0 / x)**3
        W_MeV = W_grad / MeV
        if V_val > 10:
            interp = "extreme tearing"
        elif V_val > 1:
            interp = "gradient rips pairs"
        elif V_val > 0.5:
            interp = "near threshold"
        else:
            interp = "gradient sub-dominant"
        marker = ""
        if abs(x - 0.57) < 0.01:
            marker = " <--"
        print(f"    {x:12.2f}  {W_MeV:12.4f}  {V_val:8.3f}  {interp}{marker}")
    print()

    V_peak = Z * alpha * (1.0 / 0.57)**3
    rho_V1 = gp['rho_V1']
    print(f"  At emergence peak (rho = 0.57 lambda_C): V = {V_peak:.2f}")
    print(f"  V = 1 crossing at rho = {rho_V1/lambda_C:.4f} lambda_C")
    print()
    print(f"  The gradient does not just CREATE pairs — it SEPARATES them.")
    print(f"  Throughout the emergence zone (rho < {rho_V1/lambda_C:.2f} lambda_C),")
    print(f"  the field gradient does more than 2 m_e c^2 of work over one")
    print(f"  Compton wavelength, ensuring prompt pair separation.")
    print()

    # ==================================================================
    # Section 4: 2D Gradient Map — Numerical vs Analytical
    # ==================================================================
    print("─" * 70)
    print("  4. 2D GRADIENT MAP — NUMERICAL VS ANALYTICAL")
    print("─" * 70)
    print()

    # Radial bar chart of |grad E| at z=0 (normalized)
    grad_radial = gp['grad_E_radial']
    grad_analytical = gp['grad_E_analytical']
    rho_arr = gp['rho_arr']

    # Sample at specific rho values for bar chart
    print("  |grad E| at z=0 (radial profile, normalized to value at lambda_C):")
    print()
    grad_at_lC_num = np.interp(lambda_C, rho_arr, grad_radial)
    bar_rhos_full = [0.2, 0.3, 0.4, 0.5, 0.57, 0.7, 0.8, 1.0, 1.5, 2.0]
    max_log = 4.0  # log10 of max normalized gradient
    bar_width = 40
    for x in bar_rhos_full:
        rho = x * lambda_C
        g_num = np.interp(rho, rho_arr, grad_radial) / grad_at_lC_num
        if g_num > 0:
            log_g = np.log10(g_num)
        else:
            log_g = -1
        bar_len = int(max(0, min((log_g + 1) / (max_log + 1), 1.0)) * bar_width)
        marker = " *" if abs(x - 0.57) < 0.01 else ""
        bar = "█" * bar_len
        print(f"    {x:4.2f}  {bar:<{bar_width}s}  {g_num:8.1f}x{marker}")
    print(f"    (* = emergence peak)")
    print()

    # Axial bar chart at rho_peak
    print(f"  |grad E| axial profile at rho = {rho_peak_a/lambda_C:.2f} lambda_C:")
    print()
    z_arr = gp['z_arr']
    grad_axial = gp['grad_E_axial']
    grad_at_z0 = grad_axial[len(z_arr)//2]
    z_samples = [0.0, 0.2, 0.5, 1.0, 2.0]
    for zv in z_samples:
        zv_m = zv * lambda_C
        g = np.interp(zv_m, z_arr, grad_axial) / grad_at_z0 if grad_at_z0 > 0 else 0
        bar_len = int(max(0, g) * bar_width)
        bar = "█" * bar_len
        print(f"    z={zv:4.1f}  {bar:<{bar_width}s}  {g:.3f}")
    print()

    # Numerical vs analytical comparison table
    print("  Numerical vs analytical |grad E| at z=0 (validation):")
    print()
    print(f"    {'rho/lambda_C':>12s}  {'Numerical (V/m^2)':>18s}  {'Analytical (V/m^2)':>18s}  {'Ratio':>8s}")
    print(f"    {'─'*12}  {'─'*18}  {'─'*18}  {'─'*8}")
    check_rhos = [0.2, 0.3, 0.5, 0.57, 0.8, 1.0, 2.0]
    for x in check_rhos:
        rho = x * lambda_C
        g_num = np.interp(rho, rho_arr, grad_radial)
        g_ana = 2.0 * k_e * Z * e_charge / rho**3
        ratio = g_num / g_ana if g_ana > 0 else 0
        print(f"    {x:12.2f}  {g_num:18.3e}  {g_ana:18.3e}  {ratio:8.4f}")
    print()
    print("  Note: Ratio ~ 1.00 validates the numerical gradient computation.")
    print("  Small deviations arise from grid discretization (N=200 points).")
    print()

    # ==================================================================
    # Section 5: Energy Density and Pair Creation Power
    # ==================================================================
    print("─" * 70)
    print("  5. ENERGY DENSITY AND PAIR CREATION POWER")
    print("─" * 70)
    print()
    print("  Energy density:  u = eps_0 E^2 / 2")
    print("  Pair power:      P = w * 2 m_e c^2  (rate * threshold energy)")
    print()

    u_nuclear = 5.0e35  # J/m^3 (approximate nuclear energy density)

    print(f"    {'rho/lambda_C':>12s}  {'u (J/m^3)':>12s}  {'u/u_nuclear':>12s}  {'P (W/m^3)':>12s}")
    print(f"    {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
    for x in rho_vals_lC:
        rho = x * lambda_C
        E = k_e * Z * e_charge / rho**2
        u = eps0 * E**2 / 2.0
        rate = schwinger_pair_rate(np.array([E]))[0]
        P = rate * 2.0 * m_e * c**2
        u_ratio = u / u_nuclear
        marker = ""
        if abs(x - 0.57) < 0.01:
            marker = "  <--"
        print(f"    {x:12.2f}  {u:12.3e}  {u_ratio:12.3e}  {P:12.3e}{marker}")
    print()

    # Values at emergence peak
    rho_peak = 0.57 * lambda_C
    E_peak = k_e * Z * e_charge / rho_peak**2
    u_peak = eps0 * E_peak**2 / 2.0
    print(f"  At emergence peak:")
    print(f"    Energy density u = {u_peak:.3e} J/m^3")
    print(f"    Fraction of nuclear: u/u_nuclear = {u_peak/u_nuclear:.3e}")
    print()

    total_power = gp['total_power']
    print(f"  Total integrated pair creation power: {total_power:.3e} W")
    print(f"  (integrated over full cylindrical volume)")
    print()

    # ==================================================================
    # Section 6: Context — How Extreme Is This?
    # ==================================================================
    print("─" * 70)
    print("  6. CONTEXT — HOW EXTREME IS THIS?")
    print("─" * 70)
    print()

    # Compute reference values
    E_hydrogen = e_charge / (4.0 * np.pi * eps0 * a_0**2)  # E at Bohr radius
    grad_hydrogen = 2.0 * E_hydrogen / a_0

    E_laser = 1.0e14  # 10 PW focused laser, V/m
    scale_laser = 1.0e-6  # focal spot ~1 um
    grad_laser = E_laser / scale_laser

    E_magnetar = 1.0e16  # surface B ~ 10^11 T -> E ~ 10^16 V/m equivalent
    scale_magnetar = 1.0e4  # ~10 km
    grad_magnetar = E_magnetar / scale_magnetar

    E_coulomb_lC = k_e * Z * e_charge / lambda_C**2
    grad_coulomb_lC = 2.0 * k_e * Z * e_charge / lambda_C**3
    gamma_coulomb_lC = 2.0 / (Z * alpha)

    E_coulomb_peak = k_e * Z * e_charge / rho_peak**2
    grad_coulomb_peak = 2.0 * k_e * Z * e_charge / rho_peak**3
    gamma_coulomb_peak = 2.0 * 0.57 / (Z * alpha)

    gamma_hydrogen = m_e * c**2 * grad_hydrogen / (e_charge * E_hydrogen**2)
    gamma_laser = m_e * c**2 * grad_laser / (e_charge * E_laser**2)
    gamma_magnetar = m_e * c**2 * grad_magnetar / (e_charge * E_magnetar**2)

    print(f"    {'Regime':24s}  {'E (V/m)':>12s}  {'|dE/dr| (V/m^2)':>16s}  {'Scale':>10s}  {'gamma':>8s}")
    print(f"    {'─'*24}  {'─'*12}  {'─'*16}  {'─'*10}  {'─'*8}")
    print(f"    {'Atomic (H, Bohr)':24s}  {E_hydrogen:12.2e}  {grad_hydrogen:16.2e}  {'53 pm':>10s}  {gamma_hydrogen:8.1e}")
    print(f"    {'Best laser (10 PW)':24s}  {E_laser:12.2e}  {grad_laser:16.2e}  {'~1 um':>10s}  {gamma_laser:8.1e}")
    print(f"    {'Magnetar surface':24s}  {E_magnetar:12.2e}  {grad_magnetar:16.2e}  {'~10 km':>10s}  {gamma_magnetar:8.1e}")
    print(f"    {'Coulomb Z=92 at lam_C':24s}  {E_coulomb_lC:12.2e}  {grad_coulomb_lC:16.2e}  {'386 fm':>10s}  {gamma_coulomb_lC:8.2f}")
    print(f"    {'Coulomb Z=92 at peak':24s}  {E_coulomb_peak:12.2e}  {grad_coulomb_peak:16.2e}  {'220 fm':>10s}  {gamma_coulomb_peak:8.2f}")
    print()
    print("  The Coulomb near-zone is the most violent EM environment in")
    print("  nature outside a black hole. Key distinctions:")
    print()
    print("  - Lasers: high E but nearly uniform (gamma ~ 10^4)")
    print("  - Magnetars: extreme E but huge scale (gamma ~ 10^8)")
    print("  - Coulomb Z=92: E ~ E_S with sub-femtometer gradients (gamma ~ 2)")
    print()
    print("  Only the Coulomb near-zone combines near-critical field strength")
    print("  with Compton-scale gradients, making gamma ~ O(1).")
    print()

    # ==================================================================
    # Section 7: Assessment
    # ==================================================================
    print("─" * 70)
    print("  7. ASSESSMENT")
    print("─" * 70)
    print()
    print("  1. LCFA MARGINAL (gamma ~ 1.7)")
    print("     The Schwinger formula uses the locally-constant-field approximation.")
    print("     At the pair emergence peak, gamma = 1.7 means the field varies")
    print("     significantly over one Compton wavelength. The Schwinger rate is")
    print("     qualitatively correct (the exponential dominates) but the prefactor")
    print("     receives O(1) corrections. Full WKB or worldline-instanton methods")
    print("     are needed for quantitative rates.")
    print()
    print("  2. PAIR LOCALIZATION ROBUST")
    print("     The ring at rho = 0.57 lambda_C comes from the product of the")
    print("     exponential suppression exp(-pi E_S/E) and the geometric phase")
    print("     space factor rho^3. This is controlled by the EXPONENT, not the")
    print("     LCFA prefactor. The peak location is robust against O(1) prefactor")
    print("     corrections.")
    print()
    print("  3. GRADIENT SEPARATES PAIRS (V > 1)")
    print(f"     Throughout the emergence zone (rho < {rho_V1/lambda_C:.2f} lambda_C),")
    print(f"     the violence parameter V > 1. At the peak, V = {V_peak:.1f}: the gradient")
    print("     does 3.6x the pair threshold energy over one Compton wavelength.")
    print("     Created pairs are immediately torn apart — no recombination.")
    print()
    print("  4. ENERGY DENSITY: ENORMOUS BUT SUB-NUCLEAR")
    print(f"     u ~ {u_peak:.1e} J/m^3 at the peak — far above any terrestrial field.")
    print(f"     But still ~10^(-10) of nuclear energy density ({u_nuclear:.0e} J/m^3).")
    print("     QED pair creation occurs in the electromagnetic sector, well below")
    print("     the QCD scale.")
    print()
    print("  5. NO LABORATORY CAN REPRODUCE")
    print("     Best lasers achieve E ~ 10^14 V/m (10^4 below E_S) with")
    print("     gradients ~ 10^20 V/m^2 (10^10 below Coulomb at lambda_C).")
    print("     The Coulomb near-zone near Z=92 is uniquely extreme: no")
    print("     accessible regime combines comparable E and |grad E|.")
    print()

    # Summary box
    print(f"  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  GRADIENT PROFILE — HEADLINE NUMBERS                       │")
    print(f"  │                                                             │")
    print(f"  │  Keldysh gamma at peak  = {gamma_peak:8.3f}"
          f"   (MARGINAL)             │")
    print(f"  │  Violence V at peak     = {V_peak:8.2f}"
          f"   (gradient rips pairs)   │")
    print(f"  │  V = 1 crossing         = {rho_V1/lambda_C:8.4f} lambda_C"
          f"                 │")
    print(f"  │  |dE/dr| at lambda_C    = {grad_at_lC:.3e} V/m^2"
          f"            │")
    print(f"  │  Energy density at peak = {u_peak:.3e} J/m^3"
          f"            │")
    print(f"  │  Total pair power       = {total_power:.3e} W"
          f"                  │")
    print(f"  │  E/E_S at peak          = {Z*alpha*(1/0.57)**2:8.4f}"
          f"                           │")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    print()
    print("=" * 70)


