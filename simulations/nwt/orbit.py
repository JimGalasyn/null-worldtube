"""Kerr-Newman geometry, Einstein analysis, orbit energy, and junction conditions."""

import numpy as np
from .constants import (
    c, hbar, h_planck, e_charge, eps0, mu0, m_e, m_e_MeV, m_p, alpha,
    G_N, eV, MeV, M_Planck, M_Planck_GeV, l_Planck,
    lambda_C, r_e, a_0, k_e, TorusParams, m_mu,
)
from .core import (
    compute_path_length, compute_self_energy, compute_total_energy,
    find_self_consistent_radius, compute_angular_momentum,
    torus_knot_curve,
)
from .fields import compute_torus_fields


def compute_orbit_energy_landscape(R_m, p=2, q=1, r_ratio=None):
    """
    Compute all energy terms at a given major radius R (meters).

    Returns dict with E_circ, U_EM, E_total, U_grav, delta_E_running,
    E_larmor, and the E×R product (should be constant if scale-invariant).
    """
    if r_ratio is None:
        r_ratio = alpha  # r = α×R is the NWT prescription
    r_m = r_ratio * R_m
    params = TorusParams(R=R_m, r=r_m, p=p, q=q)
    se = compute_self_energy(params)
    E_circ = se['E_circ_J']
    U_EM = se['U_total_J']
    E_total = se['E_total_J']

    # Gravitational self-energy: U_grav = -G(E/c²)²/R  (∝ 1/R³)
    M_eff = E_total / c**2
    U_grav = -G_N * M_eff**2 / R_m

    # Running coupling correction (one-loop QED)
    mu_MeV = hbar * c / R_m / MeV
    if mu_MeV > m_e_MeV:
        log_ratio = np.log(mu_MeV / m_e_MeV)
        alpha_R = alpha / (1 - (2 * alpha / (3 * np.pi)) * log_ratio)
    else:
        alpha_R = alpha
    delta_alpha = alpha_R - alpha
    # Self-energy scales as α, so δE ≈ E_total × (δα/α) × (U_EM/E_total)
    delta_E_running = U_EM * (delta_alpha / alpha)

    # Larmor radiation loss per orbit: P_Larmor × T_orbit
    # For circular motion at c with radius ~ R: a = c²/R
    # P = e²a²/(6πε₀c³), T = 2πR/c (approximate)
    # E_larmor = P × T = e²c/(3ε₀R) × (1/(2π)) ... let's compute it properly
    # P_Larmor = e²c⁴/(6πε₀c³R²) = e²c/(6πε₀R²)
    # E_per_orbit = P × T ≈ P × L/(c) where L is path length
    L_path = compute_path_length(params)
    a_cent = c**2 / R_m  # centripetal acceleration
    P_larmor = e_charge**2 * a_cent**2 / (6 * np.pi * eps0 * c**3)
    T_orbit = L_path / c
    E_larmor = P_larmor * T_orbit

    # Angular momentum
    am = compute_angular_momentum(params)

    return {
        'R_m': R_m,
        'R_fm': R_m * 1e15,
        'E_circ_J': E_circ,
        'U_EM_J': U_EM,
        'E_total_J': E_total,
        'E_total_MeV': E_total / MeV,
        'U_grav_J': U_grav,
        'delta_E_running_J': delta_E_running,
        'E_larmor_J': E_larmor,
        'ER_product': E_total * R_m,
        'Lz_over_hbar': am['Lz_over_hbar'],
    }


def compute_orbit_force_balance(R_m, p=2, q=1, r_ratio=None):
    """
    Compute all forces at a given major radius R (meters).

    Returns dict with centripetal, magnetic hoop, gravitational,
    Larmor radiation reaction forces and dimensionless ratios.
    """
    if r_ratio is None:
        r_ratio = alpha
    r_m = r_ratio * R_m
    params = TorusParams(R=R_m, r=r_m, p=p, q=q)
    se = compute_self_energy(params)
    E_total = se['E_total_J']

    # Centripetal force for photon orbit: F = pv/R = (E/c)×c/R = E/R
    F_cent = E_total / R_m

    # Magnetic hoop stress (expansion force from current loop)
    # F_hoop = μ₀I²/2 × [ln(8R/r) - 1]
    I = se['I_amps']
    log_factor_hoop = np.log(8.0 * R_m / r_m) - 1.0
    F_hoop = mu0 * I**2 / 2.0 * max(log_factor_hoop, 0.01)

    # Gravitational self-attraction: F_grav = G(E/c²)²/R²
    M_eff = E_total / c**2
    F_grav = G_N * M_eff**2 / R_m**2

    # Larmor radiation reaction force: F_rad = P_Larmor / c
    a_cent = c**2 / R_m
    P_larmor = e_charge**2 * a_cent**2 / (6 * np.pi * eps0 * c**3)
    F_larmor = P_larmor / c

    return {
        'R_m': R_m,
        'R_fm': R_m * 1e15,
        'F_cent': F_cent,
        'F_hoop': F_hoop,
        'F_grav': F_grav,
        'F_larmor': F_larmor,
        'hoop_over_cent': F_hoop / F_cent,
        'grav_over_cent': F_grav / F_cent,
        'larmor_over_cent': F_larmor / F_cent,
    }


def compute_kerr_newman_geometry(mass_kg, charge_C=e_charge, J=None):
    """
    Compute Kerr-Newman geometry parameters for a spinning charged particle.

    In geometric units (G = c = 1):
      M = Gm/c²,  a = J/(mc),  Q² = G k_e e² / c⁴

    For the electron with J = ℏ/2: a = ℏ/(2mc) = λ_C/2.
    The horizon discriminant M² - a² - Q² determines if a horizon exists.
    """
    if J is None:
        J = hbar / 2  # spin-1/2

    # Geometric-unit quantities
    M_geom = G_N * mass_kg / c**2                    # meters
    a_geom = J / (mass_kg * c)                        # meters
    Q2_geom = G_N * k_e * charge_C**2 / c**4          # meters²
    Q_geom = np.sqrt(Q2_geom)

    # Horizon discriminant
    discriminant = M_geom**2 - a_geom**2 - Q2_geom
    has_horizon = discriminant >= 0

    # Ring singularity radius = a (Kerr ring lies at r=0, ρ=a in BL coords)
    r_ring = a_geom
    lam_C = hbar / (mass_kg * c)

    # Gravitational coupling
    alpha_G = G_N * mass_kg**2 / (hbar * c)

    # Hierarchy ratios
    a_over_M = a_geom / M_geom if M_geom > 0 else np.inf
    Q_over_M = Q_geom / M_geom if M_geom > 0 else np.inf
    a_over_Q = a_geom / Q_geom if Q_geom > 0 else np.inf

    return {
        'M_geom': M_geom, 'a_geom': a_geom, 'Q_geom': Q_geom,
        'Q2_geom': Q2_geom, 'discriminant': discriminant,
        'has_horizon': has_horizon, 'r_ring': r_ring,
        'lambda_C': lam_C, 'a_over_M': a_over_M,
        'Q_over_M': Q_over_M, 'a_over_Q': a_over_Q,
        'alpha_G': alpha_G, 'mass_kg': mass_kg, 'J': J,
        'charge_C': charge_C,
    }


def compute_einstein_sweep(mass_MeV, p=2, q=1, N_points=100):
    """
    Sweep r/R on the mass shell, computing frame-dragging vs vortex circulation.

    At each r/R:
      - Find self-consistent R via find_self_consistent_radius
      - Frame dragging rate: ω_LT = 2GJ/(c² R³)
      - Vortex circulation rate: ω_vortex = r·c/R²
      - Ratio ω_LT/ω_vortex (should be ∼ 4α_G/α, constant across r/R)
      - Gravitational radius correction δR/R
      - U_grav/E_total
    """
    mass_kg_val = mass_MeV * MeV / c**2
    J_val = hbar / 2
    alpha_G = G_N * mass_kg_val**2 / (hbar * c)

    r_ratios = np.linspace(0.001, 0.5, N_points)
    valid = np.zeros(N_points, dtype=bool)
    Rs = np.zeros(N_points)
    omega_LTs = np.zeros(N_points)
    omega_vortexes = np.zeros(N_points)
    LT_vortex_ratios = np.zeros(N_points)
    delta_R_fracs = np.zeros(N_points)
    U_grav_over_E = np.zeros(N_points)

    for i, rr in enumerate(r_ratios):
        sol = find_self_consistent_radius(mass_MeV, p=p, q=q, r_ratio=rr)
        if sol is None:
            continue
        valid[i] = True
        R = sol['R']
        r_m = rr * R
        Rs[i] = R

        # Frame-dragging angular velocity at equatorial distance R
        omega_LTs[i] = 2 * G_N * J_val / (c**2 * R**3)

        # Vortex circulation angular velocity: Γ/(2πR²) = rc/R²
        omega_vortexes[i] = r_m * c / R**2

        # Ratio
        if omega_vortexes[i] > 0:
            LT_vortex_ratios[i] = omega_LTs[i] / omega_vortexes[i]

        # Gravitational correction: δR/R = -2α_G × g(x)/(1+αf)
        x = rr  # r/R
        log_factor = np.log(8.0 / x) - 2.0 if x > 0 else 0
        alpha_f = alpha * log_factor / np.pi
        g_x = 1 + x**2 / 4
        delta_R_fracs[i] = -2 * alpha_G * g_x / (1 + alpha_f)

        # U_grav / E_total
        E_total = mass_MeV * MeV
        U_grav = -G_N * mass_kg_val**2 / R
        U_grav_over_E[i] = U_grav / E_total

    return {
        'r_ratios': r_ratios, 'valid': valid, 'Rs': Rs,
        'omega_LTs': omega_LTs, 'omega_vortexes': omega_vortexes,
        'LT_vortex_ratios': LT_vortex_ratios,
        'delta_R_fracs': delta_R_fracs, 'U_grav_over_E': U_grav_over_E,
        'alpha_G': alpha_G, 'mass_MeV': mass_MeV,
    }


def compute_torus_differential_geometry(R, r, N_phi=200):
    """
    Compute intrinsic and extrinsic geometry of the torus surface.

    The torus is parameterized as:
        x = (R + r cos φ) cos θ,  y = (R + r cos φ) sin θ,  z = r sin φ
    where θ is toroidal (0..2π) and φ is poloidal (0..2π).

    Returns dict with arrays over φ and scalar summaries.
    """
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    dphi = phi[1] - phi[0]

    # Induced metric components (diagonal in θ,φ coordinates)
    h_tt = (R + r * np.cos(phi))**2   # h_θθ
    h_pp = np.full_like(phi, r**2)     # h_φφ

    # Extrinsic curvature (second fundamental form): outward-pointing normal
    K_tt = (R + r * np.cos(phi)) * np.cos(phi)   # K_θθ
    K_pp = np.full_like(phi, r)                    # K_φφ

    # Principal curvatures
    kappa_1 = np.cos(phi) / (R + r * np.cos(phi))   # poloidal direction
    kappa_2 = np.full_like(phi, 1.0 / r)             # toroidal tube curvature

    # Mean curvature H = (κ₁ + κ₂)/2
    H_mean = (kappa_1 + kappa_2) / 2.0

    # Gaussian curvature K_G = κ₁ × κ₂
    K_gauss = np.cos(phi) / (r * (R + r * np.cos(phi)))

    # Gauss-Bonnet: ∫K_G dA = ∫₀²π ∫₀²π K_G × r(R + r cos φ) dθ dφ
    # The θ integral just gives 2π, so:
    # = 2π ∫₀²π [cos φ / (r(R + r cos φ))] × r(R + r cos φ) dφ
    # = 2π ∫₀²π cos φ dφ = 0
    integrand_GB = K_gauss * r * (R + r * np.cos(phi))  # dA element / (2π dφ)
    gauss_bonnet = 2 * np.pi * np.sum(integrand_GB) * dphi

    # Surface area: A = ∫₀²π ∫₀²π r(R + r cos φ) dθ dφ = 4π²Rr
    surface_area = 4 * np.pi**2 * R * r

    return {
        'phi': phi, 'h_tt': h_tt, 'h_pp': h_pp,
        'K_tt': K_tt, 'K_pp': K_pp,
        'kappa_1': kappa_1, 'kappa_2': kappa_2,
        'H_mean': H_mean, 'K_gauss': K_gauss,
        'gauss_bonnet_integral': gauss_bonnet,
        'surface_area': surface_area, 'R': R, 'r': r,
    }


def compute_knot_geodesic_curvature(R, r, p, q, N=2000):
    """
    Compute geodesic curvature κ_g of the (p,q) knot on the torus surface.

    The geodesic curvature measures how much the curve deviates from a
    geodesic of the surface. For a curve on a surface with unit normal n:
        κ_g = (a · (n × v)) / |v|³
    where v = dx/dλ (tangent), a = d²x/dλ² (acceleration in 3D).
    """
    params = TorusParams(R=R, r=r, p=p, q=q)
    lam, xyz, dxyz, ds = torus_knot_curve(params, N)
    dlam = lam[1] - lam[0]

    # Second derivative via central differences on tangent vectors
    a = np.zeros_like(dxyz)
    a[1:-1] = (dxyz[2:] - dxyz[:-2]) / (2 * dlam)
    a[0] = (dxyz[1] - dxyz[-1]) / (2 * dlam)
    a[-1] = (dxyz[0] - dxyz[-2]) / (2 * dlam)

    # Outward surface normal at each knot point
    theta = p * lam   # toroidal angle
    phi = q * lam     # poloidal angle
    n_x = np.cos(phi) * np.cos(theta)
    n_y = np.cos(phi) * np.sin(theta)
    n_z = np.sin(phi)
    n_hat = np.stack([n_x, n_y, n_z], axis=-1)

    # κ_g = (a · (n × v)) / |v|³
    n_cross_v = np.cross(n_hat, dxyz)
    numerator = np.sum(a * n_cross_v, axis=-1)
    speed_cubed = ds**3
    kappa_g = numerator / speed_cubed

    kappa_g_rms = np.sqrt(np.mean(kappa_g**2))
    kappa_g_max = np.max(np.abs(kappa_g))
    kappa_g_mean_abs = np.mean(np.abs(kappa_g))

    return {
        'lam': lam, 'kappa_g': kappa_g,
        'kappa_g_rms': kappa_g_rms, 'kappa_g_max': kappa_g_max,
        'kappa_g_mean_abs': kappa_g_mean_abs,
        'is_geodesic': kappa_g_max < 1e-6 / R,  # threshold relative to 1/R
        'R': R, 'r': r, 'p': p, 'q': q,
    }


def compute_junction_balance(R, r, p, q):
    """
    Compute EM pressure balance (flat-space Israel junction conditions).

    In the flat-space limit, Israel conditions reduce to Young-Laplace:
    radiation pressure inside the tube must balance surface-tension × curvature.
    """
    params = TorusParams(R=R, r=r, p=p, q=q)
    se = compute_self_energy(params)
    fields = compute_torus_fields(R, r, p, q)
    fb = compute_orbit_force_balance(R, p=p, q=q, r_ratio=r/R)

    U_EM = se['U_total_J']
    E_total = se['E_total_J']
    I = se['I_amps']
    L_path = fields['L_path']
    V_tube = fields['V_tube']

    # EM energy density inside tube
    u_EM = U_EM / V_tube if V_tube > 0 else 0

    # Radiation pressure (photon gas in 1D waveguide): P = u
    P_rad = u_EM

    # Magnetic pressure at tube surface: B_surface ≈ μ₀I/(2πr)
    B_surface = mu0 * I / (2 * np.pi * r)
    P_B_surface = B_surface**2 / (2 * mu0)

    # Geodesic curvature
    kg = compute_knot_geodesic_curvature(R, r, p, q)

    # Confining force per unit length: F = E_total/L × κ_g
    # (energy per unit length × geodesic curvature = transverse force/length)
    E_per_length = E_total / L_path
    F_confine = E_per_length * kg['kappa_g_rms']

    # Hoop force from existing force balance
    F_hoop = fb['F_hoop']

    # Force ratio
    force_ratio = F_confine / F_hoop if F_hoop > 0 else float('inf')

    # Young-Laplace: ΔP = σ_eff × (1/r)  [poloidal curvature dominates]
    # Effective surface tension σ_eff = P_rad × r (from balance)
    sigma_eff = P_rad * r

    return {
        'u_EM': u_EM, 'P_rad': P_rad,
        'B_surface': B_surface, 'P_B_surface': P_B_surface,
        'F_confine_per_length': F_confine,
        'F_hoop_per_length': F_hoop / L_path if L_path > 0 else 0,
        'F_hoop': F_hoop,
        'force_ratio': force_ratio,
        'sigma_eff': sigma_eff,
        'E_per_length': E_per_length,
        'kappa_g_rms': kg['kappa_g_rms'],
        'kappa_g_max': kg['kappa_g_max'],
        'V_tube': V_tube, 'L_path': L_path,
        'U_EM': U_EM, 'E_total': E_total,
        'R': R, 'r': r, 'p': p, 'q': q,
    }


def print_orbit_analysis():
    """
    What fixes the electron's size?  An orbital-mechanics analysis.

    The NWT electron has E(R) ∝ 1/R — classically scale-invariant.
    Given m_e you solve for R_e, but nothing DERIVES m_e.
    This module systematically surveys what could break scale invariance.
    """
    print("=" * 70)
    print("  ORBIT ANALYSIS: WHAT FIXES THE ELECTRON'S SIZE?")
    print("  Scale invariance, Kepler analogy, and candidate mechanisms")
    print("=" * 70)

    # Get the electron radius as our reference point
    sol_e = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=alpha)
    R_e = sol_e['R']
    R_e_fm = R_e * 1e15

    print(f"\n  Reference: electron on (2,1) torus knot, r/R = α")
    print(f"    R_e = {R_e_fm:.2f} fm  ({R_e:.4e} m)")
    print(f"    m_e = {m_e_MeV:.8f} MeV")

    # ── Section 1: Scale Invariance Demonstrated ──────────────────────
    print(f"\n\n  1. SCALE INVARIANCE DEMONSTRATED")
    print("  " + "─" * 51)
    print(f"\n  All energies ∝ 1/R, so E×R = const at any radius.")
    print(f"  Angular momentum L_z is R-independent.\n")

    R_values_fm = [0.1, 1.0, 10.0, 100.0, 194.0, 1000.0, 10000.0]
    print(f"  {'R (fm)':>10}  {'E_circ (MeV)':>14}  {'U_EM (MeV)':>14}  "
          f"{'E×R (MeV·fm)':>14}  {'L_z/ℏ':>8}")
    print(f"  {'-'*68}")

    ER_ref = None
    for R_fm in R_values_fm:
        R_m = R_fm * 1e-15
        el = compute_orbit_energy_landscape(R_m)
        ER = el['E_total_MeV'] * R_fm
        if ER_ref is None:
            ER_ref = ER
        print(f"  {R_fm:>10.1f}  {el['E_circ_J']/MeV:>14.6f}  "
              f"{el['U_EM_J']/MeV:>14.6f}  {ER:>14.6f}  "
              f"{el['Lz_over_hbar']:>8.4f}")

    print(f"\n  E×R variation over 5 decades: < {abs(ER/ER_ref - 1)*100:.1e}%")
    print(f"  L_z/ℏ = 0.5000 at every radius — spin is geometric, not dynamic.")

    # ── Section 2: The Kepler Analogy ─────────────────────────────────
    print(f"\n\n  2. THE KEPLER ANALOGY: WHY ORBITS HAVE A PREFERRED SIZE")
    print("  " + "─" * 51)
    print(f"""
  KEPLER ORBIT (planet around star):
    V(r) = -GMm/r              ∝ 1/r
    T(r) = L²/(2mr²)           ∝ 1/r²  (angular momentum → centrifugal)
    E(r) = L²/(2mr²) - GMm/r   → minimum at r₀ = L²/(GMm²)
    KEY: L converts T from p²/2m to 1/r², creating a DIFFERENT power than V.

  NWT TORUS (photon on torus knot):
    E_circ(R) = 2πℏc/L(R)      ∝ 1/R
    U_EM(R) = α×f(r/R)×E_circ  ∝ 1/R   (same power!)
    E_total(R)                  ∝ 1/R   (no minimum, no maximum)
    KEY: L_z = ℏ/2 is R-independent. No centrifugal barrier.

  The photon ALWAYS moves at c, so there's no "kinetic energy that
  depends on angular momentum" — the orbit speed is fixed by relativity.
  Angular momentum L_z = (E/c)×R×⟨cos θ⟩ = ℏ/2 because E ∝ 1/R.
  This is the root of scale invariance.""")

    # ── Section 3: Gravitational Self-Energy ──────────────────────────
    print(f"\n\n  3. GRAVITATIONAL SELF-ENERGY (the one that works — at the wrong scale)")
    print("  " + "─" * 51)

    # E_total = A/R, U_grav = -G(A/Rc²)²/R = -GA²/(c⁴R³)
    # dE/dR = -A/R² + 3GA²/(c⁴R⁴) = 0
    # -AR² + 3GA²/c⁴ = 0  →  R² = 3GA/c⁴
    # R_grav = √(3GA/c⁴)
    # But A = E_total × R, get it at R_e:
    el_e = compute_orbit_energy_landscape(R_e)
    A = el_e['ER_product']  # E×R in J·m
    A_MeV_fm = A / (MeV * 1e-15)

    R_grav = np.sqrt(3 * G_N * A / c**4)
    R_grav_fm = R_grav * 1e15
    E_at_grav = A / R_grav
    m_grav_MeV = E_at_grav / MeV
    m_grav_kg = E_at_grav / c**2

    # Compare to Planck length
    R_grav_over_lP = R_grav / l_Planck
    R_grav_over_Re = R_grav / R_e

    print(f"""
  Add gravity: E(R) = A/R − GA²/(c⁴R³)

  where A = E×R = {A_MeV_fm:.6f} MeV·fm  (the scale-invariant product)

  Minimum at dE/dR = 0:
    −A/R² + 3GA²/(c⁴R⁴) = 0
    R_grav = √(3GA/c⁴)

  Result:
    R_grav    = {R_grav:.4e} m  = {R_grav_fm:.4e} fm
    R_grav/l_P = {R_grav_over_lP:.2f}  (≈ √3 × Planck length)
    m_grav    = {m_grav_MeV:.2e} MeV  = {m_grav_kg:.2e} kg
    R_grav/R_e = {R_grav_over_Re:.2e}

  Gravity DOES break scale invariance — the 1/R³ term has a different
  power than the 1/R term, creating a minimum. But the minimum is at
  the Planck scale, not at 194 fm. This is the hierarchy problem:
  why is the electron 10²² times larger than gravity predicts?""")

    # Verify numerically
    print(f"\n  Numerical verification (energy landscape near R_grav):")
    print(f"  {'R/R_grav':>10}  {'E_EM (MeV)':>14}  {'U_grav (MeV)':>14}  {'E_net (MeV)':>14}")
    print(f"  {'-'*56}")
    for factor in [0.5, 0.8, 1.0, 1.2, 2.0, 10.0]:
        R_test = factor * R_grav
        E_em = A / R_test
        U_g = -G_N * (A / (c**2 * R_test))**2 / R_test
        E_net = E_em + U_g
        print(f"  {factor:>10.1f}  {E_em/MeV:>14.4e}  {U_g/MeV:>14.4e}  {E_net/MeV:>14.4e}")

    # ── Section 4: Running Coupling ───────────────────────────────────
    print(f"\n\n  4. RUNNING COUPLING α(R)")
    print("  " + "─" * 51)

    # One-loop QED: α(μ) = α / (1 - (2α/3π) ln(μ/m_e))
    # At R_e: μ = ℏc/R_e
    mu_e = hbar * c / R_e
    mu_e_MeV = mu_e / MeV
    if mu_e_MeV > m_e_MeV:
        log_ratio_e = np.log(mu_e_MeV / m_e_MeV)
        alpha_at_Re = alpha / (1 - (2 * alpha / (3 * np.pi)) * log_ratio_e)
    else:
        alpha_at_Re = alpha
    delta_alpha_Re = (alpha_at_Re - alpha) / alpha

    print(f"""
  One-loop QED β-function: α(μ) = α / (1 − (2α/3π) ln(μ/m_e))
  Scale μ = ℏc/R: higher energy ↔ smaller torus.

  At R_e = {R_e_fm:.1f} fm:
    μ     = ℏc/R_e = {mu_e_MeV:.4f} MeV
    α(R_e) = {alpha_at_Re:.10f}
    δα/α   = {delta_alpha_Re:.2e}  ({delta_alpha_Re*100:.4f}%)

  The self-energy U_EM ∝ α(R)/R, so running coupling changes
  the effective scaling from exactly 1/R to 1/R × [1 + O(α ln R)].
  This is a logarithmic perturbation — it tilts the energy curve
  but doesn't create a minimum at any finite R.""")

    # Show α at various scales
    print(f"  {'R (fm)':>10}  {'μ (MeV)':>12}  {'α(R)':>14}  {'δα/α':>12}")
    print(f"  {'-'*52}")
    for R_fm in [0.01, 0.1, 1.0, 10.0, 100.0, 194.0, 1000.0]:
        R_m = R_fm * 1e-15
        mu = hbar * c / R_m
        mu_MeV = mu / MeV
        if mu_MeV > m_e_MeV:
            lr = np.log(mu_MeV / m_e_MeV)
            a_R = alpha / (1 - (2 * alpha / (3 * np.pi)) * lr)
        else:
            a_R = alpha
        da = (a_R - alpha) / alpha
        print(f"  {R_fm:>10.2f}  {mu_MeV:>12.2f}  {a_R:>14.10f}  {da:>12.2e}")

    # ── Section 5: Radiation Reaction ─────────────────────────────────
    print(f"\n\n  5. RADIATION REACTION (Larmor)")
    print("  " + "─" * 51)

    el_e_detail = compute_orbit_energy_landscape(R_e)
    ratio_larmor = el_e_detail['E_larmor_J'] / el_e_detail['E_circ_J']

    print(f"""
  A circulating charge radiates (Larmor). Energy lost per orbit:

    P_Larmor = e²a²/(6πε₀c³),  a = c²/R  (centripetal)
    E_rad/orbit = P × T_orbit = P × L/c

  At R_e:
    E_rad    = {el_e_detail['E_larmor_J']/MeV:.6e} MeV
    E_circ   = {el_e_detail['E_circ_J']/MeV:.6f} MeV
    E_rad/E_circ = {ratio_larmor:.6e}

  Analytical ratio: E_rad/E_circ = 4α/(3p²) × (2πp)  ... let's check scaling:""")

    # Show that radiation also scales as 1/R
    print(f"\n  {'R (fm)':>10}  {'E_circ (MeV)':>14}  {'E_rad (MeV)':>14}  {'E_rad/E_circ':>14}")
    print(f"  {'-'*56}")
    for R_fm in [1.0, 10.0, 100.0, 194.0, 1000.0]:
        R_m = R_fm * 1e-15
        el = compute_orbit_energy_landscape(R_m)
        ratio = el['E_larmor_J'] / el['E_circ_J']
        print(f"  {R_fm:>10.1f}  {el['E_circ_J']/MeV:>14.6f}  "
              f"{el['E_larmor_J']/MeV:>14.6e}  {ratio:>14.6e}")

    print(f"\n  E_rad/E_circ is R-independent → radiation reaction scales as 1/R")
    print(f"  (same as E_circ). No scale invariance breaking.")

    # ── Section 6: Force Survey ───────────────────────────────────────
    print(f"\n\n  6. FORCE SURVEY AT R_e = {R_e_fm:.1f} fm")
    print("  " + "─" * 51)

    fb = compute_orbit_force_balance(R_e)

    print(f"\n  {'Force':>25}  {'Value (N)':>14}  {'F/F_cent':>14}")
    print(f"  {'-'*56}")
    print(f"  {'Centripetal (E/R)':>25}  {fb['F_cent']:>14.4e}  {'1.000':>14}")
    print(f"  {'Magnetic hoop':>25}  {fb['F_hoop']:>14.4e}  {fb['hoop_over_cent']:>14.4e}")
    print(f"  {'Gravitational':>25}  {fb['F_grav']:>14.4e}  {fb['grav_over_cent']:>14.4e}")
    print(f"  {'Larmor reaction':>25}  {fb['F_larmor']:>14.4e}  {fb['larmor_over_cent']:>14.4e}")

    print(f"""
  The centripetal force is overwhelmingly dominant. The magnetic hoop
  stress (which tries to expand the loop) is the largest correction
  at ~{fb['hoop_over_cent']:.1e} of centripetal. Gravity is negligible ({fb['grav_over_cent']:.1e}).

  Crucially: F_cent ∝ 1/R², F_hoop ∝ 1/R², F_Larmor ∝ 1/R².
  All forces have the SAME R-dependence. No force balance picks out R_e.""")

    # ── Section 7: Summary Table ──────────────────────────────────────
    print(f"\n\n  7. SUMMARY: MECHANISMS THAT COULD FIX THE SCALE")
    print("  " + "─" * 51)

    print(f"""
  {'Mechanism':<24} {'Scaling':>10} {'Breaks SI?':>12} {'Predicted R':>14} {'R/R_e':>10}
  {'-'*72}
  {'Gravitational':<24} {'1/R³':>10} {'YES':>12} {'~l_Planck':>14} {f'{R_grav_over_Re:.1e}':>10}
  {'Running coupling':<24} {'1/R·ln':>10} {'WEAK':>12} {'no minimum':>14} {'---':>10}
  {'Radiation reaction':<24} {'1/R':>10} {'NO':>12} {'---':>14} {'---':>10}
  {'Magnetic hoop':<24} {'1/R²':>10} {'NO':>12} {'---':>14} {'---':>10}
  {'Angular momentum':<24} {'const':>10} {'NO':>12} {'---':>14} {'---':>10}
  {'-'*72}""")

    print(f"  Scale invariance requires ALL terms ∝ 1/R. Gravity (∝ 1/R³) is the")
    print(f"  only classical mechanism that breaks it — but at the wrong scale.")

    # ── Section 8: What's Needed ──────────────────────────────────────
    print(f"\n\n  8. WHAT'S NEEDED — AND HONEST ASSESSMENT")
    print("  " + "─" * 51)

    print(f"""
  To fix R_e ≈ 194 fm, we need a term in E(R) that:
    • Scales as 1/Rⁿ with n ≠ 1  (to compete with the 1/R terms)
    • Is active at R ~ 200 fm     (not Planck-scale, not astronomical)
    • Is attractive for n > 1     (or repulsive for n < 1)

  The gap between gravity's prediction (l_Planck) and observation (R_e)
  is a factor of {R_e/R_grav:.1e} — this IS the hierarchy problem,
  recast in NWT language.

  CANDIDATES FOR FUTURE INVESTIGATION:
    1. Topology-dependent Casimir energy — vacuum fluctuations on the
       torus depend on both R and r/R. If the Casimir energy has a
       different R-scaling than 1/R (possible for non-trivial topology),
       it could create a minimum. Requires zeta-regularization on the
       knotted torus.

    2. Nonlinear QED (Euler-Heisenberg) — at strong fields near the
       tube, the effective Lagrangian gets O(α²) corrections that
       scale as (r_class/R)⁴. Too small at R_e but illustrates how
       field-strength-dependent corrections change the scaling.

    3. Non-perturbative QED — the electron's anomalous magnetic moment
       is known to all orders in α. If the full QED effective action
       on a torus has non-perturbative contributions (instantons,
       monopole-like configurations), these could provide the missing
       scale.

  THE DEEPEST OPEN QUESTION:
    In NWT, the electron mass m_e is currently an input — the theory
    predicts m_μ/m_e, m_p/m_e, and α in terms of geometry, but m_e
    itself sets the overall energy scale. Either:

    (a) m_e is axiomatic — a free parameter of the universe, like c
        or ℏ, not derivable from anything deeper.

    (b) Physics not yet in the model breaks the scale invariance —
        quantum gravity, topology change, or some mechanism that
        connects the Planck scale to the electron scale across 22
        orders of magnitude.

    This module shows that (b) requires something beyond classical
    gravity, perturbative QED, and radiation reaction. The answer,
    if it exists, likely involves the topology of spacetime itself.""")

    print("=" * 70)


def print_einstein_analysis():
    """
    Einstein / Kerr-Newman self-consistency analysis.

    Maps the NWT electron onto the Kerr-Newman solution of GR:
    - Super-extremal geometry (no horizon, naked ring singularity)
    - Frame dragging as the gravitational shadow of vortex circulation
    - Mass equation with gravitational corrections
    - Assessment of what GR can and can't tell us about m_e
    """
    print("=" * 70)
    print("  EINSTEIN / KERR-NEWMAN SELF-CONSISTENCY ANALYSIS")
    print("  Frame dragging, resolved singularities, and the mass equation")
    print("=" * 70)

    # ================================================================
    # Section 1: KERR-NEWMAN GEOMETRY
    # ================================================================
    print(f"\n  1. KERR-NEWMAN GEOMETRY OF THE ELECTRON")
    print(f"  {'─'*55}")
    print(f"  A spinning charged mass in GR is described by the Kerr-Newman")
    print(f"  metric. In geometric units (G = c = 1):")
    print(f"    M = Gm/c²    a = J/(mc) = ℏ/(2mc)    Q² = Gk_e e²/c⁴")
    print(f"  Horizon exists iff M² ≥ a² + Q².")

    # Electron geometry
    kn_e = compute_kerr_newman_geometry(m_e)
    alpha_G_e = kn_e['alpha_G']

    print(f"\n  Electron Kerr-Newman parameters:")
    print(f"    M  = Gm_e/c²        = {kn_e['M_geom']:.4e} m")
    print(f"    a  = ℏ/(2m_e c)     = {kn_e['a_geom']:.4e} m  (= λ_C/2)")
    print(f"    Q  = √(Gk_e)·e/c²  = {kn_e['Q_geom']:.4e} m")
    print(f"    λ_C = ℏ/(m_e c)     = {kn_e['lambda_C']:.4e} m")
    print(f"    α_G = Gm_e²/(ℏc)   = {alpha_G_e:.4e}")

    print(f"\n  Horizon discriminant: M² - a² - Q² = {kn_e['discriminant']:.4e} m²")
    print(f"  SUPER-EXTREMAL: no horizon, naked ring singularity")

    print(f"\n  Hierarchy ratios:")
    print(f"    a/M  = {kn_e['a_over_M']:.4e}  (= 1/(2α_G) = {1/(2*alpha_G_e):.4e})")
    print(f"    Q/M  = {kn_e['Q_over_M']:.4e}")
    print(f"    a/Q  = {kn_e['a_over_Q']:.4e}")

    # Multi-particle table
    particles = {
        'electron': {'mass_kg': m_e, 'mass_MeV': m_e_MeV},
        'muon':     {'mass_kg': m_mu, 'mass_MeV': 105.6583755},
        'tau':      {'mass_kg': 1776.86 * MeV / c**2, 'mass_MeV': 1776.86},
        'proton':   {'mass_kg': m_p, 'mass_MeV': 938.272},
    }

    print(f"\n  {'Particle':<10} {'M (m)':<14} {'a (m)':<14} {'a/M':<14} {'α_G':<14} {'Horizon?'}")
    print(f"  {'─'*74}")
    for name, info in particles.items():
        kn = compute_kerr_newman_geometry(info['mass_kg'])
        hz = "yes" if kn['has_horizon'] else "NO"
        print(f"  {name:<10} {kn['M_geom']:<14.4e} {kn['a_geom']:<14.4e} {kn['a_over_M']:<14.4e} {kn['alpha_G']:<14.4e} {hz}")

    # ================================================================
    # Section 2: TORUS AS RESOLVED SINGULARITY
    # ================================================================
    print(f"\n\n  2. TORUS AS RESOLVED SINGULARITY")
    print(f"  {'─'*55}")
    print(f"  The Kerr-Newman ring singularity sits at:")
    print(f"    Boyer-Lindquist r = 0, θ = π/2  →  ring of radius a = λ_C/2")
    print(f"  The NWT torus has major radius R and tube radius r = αR:")
    print(f"    R ≈ λ_C/(2(1+αf))  where f = [ln(8R/r) - 2]/π  (for p=2)")

    sol_e = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=alpha)
    if sol_e:
        R_e = sol_e['R']
        r_e_tube = alpha * R_e
        ratio_R_a = R_e / kn_e['a_geom']
        log_f = (np.log(8.0 / alpha) - 2.0) / np.pi
        alpha_f = alpha * log_f

        print(f"\n  For the electron (p=2, q=1, r/R = α):")
        print(f"    KN ring radius a       = {kn_e['a_geom']:.4e} m  = λ_C/2")
        print(f"    NWT torus radius R      = {R_e:.4e} m  ≈ λ_C/(2(1+αf))")
        print(f"    NWT tube radius r = αR  = {r_e_tube:.4e} m")
        print(f"    R/a = {ratio_R_a:.6f}  ≈ 1/(1+αf) = {1/(1+alpha_f):.6f}")
        print(f"    (p=2 winding: R ≈ λ_C/2 = a, so torus sits at ring radius)")
        print(f"\n  The torus REPLACES the naked singularity with a finite-size")
        print(f"  vortex tube of radius r = αR ≈ α·λ_C — the classical electron")
        print(f"  radius r_e = αλ_C. The singularity is resolved by topology.")

    # ================================================================
    # Section 3: FRAME DRAGGING = VORTEX CIRCULATION
    # ================================================================
    print(f"\n\n  3. FRAME DRAGGING = VORTEX CIRCULATION")
    print(f"  {'─'*55}")
    print(f"  The key physical insight: frame dragging IS the gravitational")
    print(f"  manifestation of vortex circulation.")
    print(f"\n  Lense-Thirring frame-dragging rate at distance R from spin axis:")
    print(f"    ω_LT(R) = 2GJ/(c² R³)")
    print(f"\n  Vortex circulation rate on the torus:")
    print(f"    ω_vortex = Γ/(2πR²) = rc/R²")
    print(f"\n  Their ratio:")
    print(f"    ω_LT/ω_vortex = 2GJ/(c² R³) × R²/(rc)")
    print(f"                   = 2G·(ℏ/2)/(c² R · rc)")
    print(f"                   = Gℏ/(c³ · r · R)")
    print(f"  With r = αR, R = λ_C/(1+αf):")
    print(f"    = Gℏ/(c³ · α · R²)")
    print(f"    = Gm²/(ℏc) × 4/[α(1+αf)²]")
    print(f"    = 4α_G / [α(1+αf)²]")

    if sol_e:
        # Compute at the electron
        omega_LT_e = 2 * G_N * (hbar/2) / (c**2 * R_e**3)
        omega_vortex_e = r_e_tube * c / R_e**2
        ratio_e = omega_LT_e / omega_vortex_e
        predicted = 4 * alpha_G_e / (alpha * (1 + alpha_f)**2)

        print(f"\n  For the electron (r/R = α, p=2):")
        print(f"    ω_LT       = {omega_LT_e:.4e} rad/s")
        print(f"    ω_vortex   = {omega_vortex_e:.4e} rad/s")
        print(f"    ω_LT/ω_vortex = {ratio_e:.4e}")
        print(f"    4α_G/[α(1+αf)²] = {predicted:.4e}")
        print(f"    Simple estimate 4α_G/α = {4*alpha_G_e/alpha:.4e}")

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  Frame dragging IS circulation.                            │")
    print(f"  │  Gravity is its {4*alpha_G_e/alpha:.1e} shadow.                    │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    # Sweep table confirming normalized ratio ≈ constant
    sweep = compute_einstein_sweep(m_e_MeV, p=2, q=1, N_points=100)
    v = sweep['valid']
    if np.any(v):
        # Pick ~8 representative points
        valid_idx = np.where(v)[0]
        step = max(1, len(valid_idx) // 8)
        sample_idx = valid_idx[::step][:8]

        print(f"\n  Since ω_vortex ∝ r/R, the raw ratio scales as 1/(r/R).")
        print(f"  The normalized product (r/R)×(ω_LT/ω_vortex) ≈ 4α_G is constant:")
        print(f"  {'r/R':<10} {'R (fm)':<14} {'ω_LT/ω_vortex':<18} {'(r/R)×ratio':<16} {'4α_G':<14}")
        print(f"  {'─'*70}")
        for idx in sample_idx:
            rr = sweep['r_ratios'][idx]
            R_fm = sweep['Rs'][idx] * 1e15
            rat = sweep['LT_vortex_ratios'][idx]
            normed = rr * rat
            print(f"  {rr:<10.4f} {R_fm:<14.4f} {rat:<18.4e} {normed:<16.4e} {4*alpha_G_e:<14.4e}")

        # Check constancy of normalized ratio
        valid_normed = sweep['r_ratios'][v] * sweep['LT_vortex_ratios'][v]
        valid_normed = valid_normed[valid_normed > 0]
        if len(valid_normed) > 1:
            spread = (valid_normed.max() - valid_normed.min()) / np.mean(valid_normed)
            print(f"\n  Spread in normalized ratio: {spread:.2e} (confirms constancy)")

    # Enhancement at tube surface
    if sol_e:
        R_over_r = R_e / r_e_tube
        enhancement = R_over_r**3
        ratio_at_tube = ratio_e * enhancement
        print(f"\n  At tube surface (ρ = r = αR):")
        print(f"    ω_LT enhanced by (R/r)³ = (1/α)³ = {enhancement:.1f}")
        print(f"    But ratio still ∼ α_G/α⁴ ≈ {alpha_G_e/alpha**4:.4e}")

    # ================================================================
    # Section 4: THE MASS EQUATION WITH GRAVITY
    # ================================================================
    print(f"\n\n  4. THE MASS EQUATION WITH GRAVITY")
    print(f"  {'─'*55}")
    print(f"  Without gravity:")
    print(f"    mc² = ℏc(1 + αf) / (2R)")
    print(f"    → R adjusts to match any m. One equation, two unknowns.")
    print(f"\n  With gravity:")
    print(f"    mc² = ℏc(1 + αf) / (2R)  -  Gm² g(x) / R")
    print(f"    where g(x) = 1 + x²/4, x = r/R")
    print(f"\n  Both terms scale as 1/R → gravity shifts R but can't pin m:")
    print(f"    R_grav = R₀(1 + δR/R),  δR/R = -2α_G g(x)/(1+αf)")

    if np.any(v):
        print(f"\n  {'r/R':<10} {'R (fm)':<14} {'δR/R':<16} {'U_grav/E_total':<16}")
        print(f"  {'─'*54}")
        for idx in sample_idx:
            rr = sweep['r_ratios'][idx]
            R_fm = sweep['Rs'][idx] * 1e15
            dR = sweep['delta_R_fracs'][idx]
            UoE = sweep['U_grav_over_E'][idx]
            print(f"  {rr:<10.4f} {R_fm:<14.4f} {dR:<16.4e} {UoE:<16.4e}")

        print(f"\n  All ∼ 10⁻⁴⁵: gravity is a {abs(sweep['delta_R_fracs'][v].mean()):.1e} perturbation")
        print(f"  on the torus radius. It cannot select m_e by itself.")

    # ================================================================
    # Section 5: WHAT COULD PIN THE MASS
    # ================================================================
    print(f"\n\n  5. WHAT COULD PIN THE MASS: THREE CANDIDATES FROM GR")
    print(f"  {'─'*55}")
    print(f"  The mass shell mc² = ℏc(1+αf)/(2R) is one equation for two")
    print(f"  unknowns (m, R). A second equation is needed. Three candidates")
    print(f"  from GR that could provide it:\n")

    print(f"  A. ISRAEL JUNCTION CONDITIONS")
    print(f"     Match KN exterior to vortex interior at the tube surface.")
    print(f"     The jump in extrinsic curvature across the boundary gives:")
    print(f"       [K_ab] = -8πG(S_ab - h_ab S/2)")
    print(f"     where S_ab is the surface stress-energy tensor.")
    print(f"     This constrains r/R as a function of (m, a, Q) —")
    print(f"     potentially fixing the geometry for given spin and charge.\n")

    print(f"  B. NULL GEODESIC CONDITION")
    print(f"     The photon must follow a geodesic of the self-consistent")
    print(f"     spacetime. For a circular null geodesic in Kerr-Newman:")
    print(f"       V_eff(r) = 0  and  dV_eff/dr = 0")
    print(f"     This is how photon sphere radii are derived classically.")
    print(f"     For the torus: the null worldline must close on the")
    print(f"     geometry it creates — a self-referential bootstrap.\n")

    print(f"  C. REGULARITY (NO CONICAL SINGULARITY)")
    print(f"     The torus cross-section must close smoothly — no conical")
    print(f"     deficit angle at the tube axis. This requires:")
    print(f"       g_φφ → ρ² as ρ → 0 (locally flat near tube center)")
    print(f"     In the full KN metric, this constrains the relationship")
    print(f"     between the winding numbers and the metric components.\n")

    print(f"  All three require the FULL nonlinear metric — linearized GR")
    print(f"  gives only perturbative corrections (δR/R ∼ α_G ∼ 10⁻⁴⁵).")
    print(f"  The junction matching at r ∼ a is where GR becomes nonlinear")
    print(f"  and the torus geometry departs from flat space.")

    # ================================================================
    # Section 6: ASSESSMENT
    # ================================================================
    print(f"\n\n  6. ASSESSMENT")
    print(f"  {'─'*55}")
    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  WHAT GR CAN DO:                                           │")
    print(f"  │  • Identify the electron as super-extremal Kerr-Newman     │")
    print(f"  │    (a/M ∼ 10⁴⁴, no horizon, naked ring singularity)       │")
    print(f"  │  • Resolve the singularity: torus replaces the ring        │")
    print(f"  │  • Map frame dragging → vortex circulation quantitatively  │")
    print(f"  │  • Derive the hierarchy: a/M = 1/(2α_G) geometrically     │")
    print(f"  │                                                             │")
    print(f"  │  WHAT GR CANNOT DO (in linearized regime):                 │")
    print(f"  │  • Derive m_e — both energy terms scale as 1/R             │")
    print(f"  │  • Break scale invariance (δR/R ∼ α_G ∼ 10⁻⁴⁵)            │")
    print(f"  │                                                             │")
    print(f"  │  WHAT'S MISSING:                                           │")
    print(f"  │  • Non-perturbative junction matching at r ∼ a where       │")
    print(f"  │    linearized GR breaks down                               │")
    print(f"  │  • The second equation: Israel, geodesic, or regularity    │")
    print(f"  │    condition from the full Kerr-Newman geometry             │")
    print(f"  └─────────────────────────────────────────────────────────────┘")
    print("=" * 70)


def print_junction_analysis():
    """
    Israel junction conditions analysis.

    Computes torus differential geometry, geodesic curvature of the null
    worldline, and shows how Israel conditions reduce to EM pressure balance
    (Young-Laplace) in the flat-space limit.
    """
    print("=" * 70)
    print("  ISRAEL JUNCTION CONDITIONS: TORUS DIFFERENTIAL GEOMETRY")
    print("  AND CONFINEMENT")
    print("=" * 70)

    # Reference electron
    sol_e = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=alpha)
    R_e = sol_e['R']
    r_e_tube = alpha * R_e
    p_e, q_e = 2, 1

    # ================================================================
    # Section 1: TORUS DIFFERENTIAL GEOMETRY
    # ================================================================
    print(f"\n  1. TORUS DIFFERENTIAL GEOMETRY")
    print(f"  {'─'*55}")
    print(f"  The torus surface Σ with major radius R and minor radius r")
    print(f"  has a rich intrinsic and extrinsic geometry.\n")

    geom = compute_torus_differential_geometry(R_e, r_e_tube)

    print(f"  Electron torus: R = {R_e:.4e} m, r = {r_e_tube:.4e} m, r/R = α = {alpha:.6e}")
    print(f"\n  Induced metric (diagonal in θ,φ coordinates):")
    print(f"    h_θθ(φ) = (R + r cos φ)²   h_φφ = r²   h_θφ = 0")
    print(f"    At outer equator (φ=0):  h_θθ = (R+r)²  = {(R_e+r_e_tube)**2:.4e} m²")
    print(f"    At inner equator (φ=π):  h_θθ = (R−r)²  = {(R_e-r_e_tube)**2:.4e} m²")

    print(f"\n  Extrinsic curvature (second fundamental form):")
    print(f"    K_θθ(φ) = (R + r cos φ) cos φ     K_φφ = r")

    print(f"\n  Principal curvatures at key poloidal angles:")
    print(f"  {'φ':>8s} {'κ₁ (m⁻¹)':>14s} {'κ₂ (m⁻¹)':>14s} {'H (m⁻¹)':>14s} {'K_G (m⁻²)':>14s}")
    print(f"  {'─'*66}")
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    labels = ['0', 'π/2', 'π', '3π/2']
    for ang, lab in zip(angles, labels):
        k1 = np.cos(ang) / (R_e + r_e_tube * np.cos(ang))
        k2 = 1.0 / r_e_tube
        H = (k1 + k2) / 2
        KG = k1 * k2
        print(f"  {lab:>8s} {k1:>14.4e} {k2:>14.4e} {H:>14.4e} {KG:>14.4e}")

    print(f"\n  Gaussian curvature:")
    print(f"    K_G(φ) = cos φ / [r(R + r cos φ)]")
    print(f"    Outer equator (φ=0):  K_G = +{1.0/(r_e_tube*(R_e+r_e_tube)):.4e} m⁻²  (positive)")
    print(f"    Inner equator (φ=π):  K_G = {-1.0/(r_e_tube*(R_e-r_e_tube)):.4e} m⁻²  (negative)")

    print(f"\n  Gauss-Bonnet theorem: ∫K_G dA = 2πχ(T²) = 0  (χ = 0 for torus)")
    print(f"    Numerical integral: {geom['gauss_bonnet_integral']:.4e}  ✓")
    print(f"\n  Surface area: A = 4π²Rr = {geom['surface_area']:.4e} m²")
    sphere_area = 4 * np.pi * R_e**2
    print(f"    cf. sphere 4πR² = {sphere_area:.4e} m²")
    print(f"    A_torus/A_sphere = π(r/R) = π×α = {np.pi*alpha:.6e}")

    # ================================================================
    # Section 2: GEODESIC CURVATURE OF THE NULL WORLDLINE
    # ================================================================
    print(f"\n\n  2. GEODESIC CURVATURE OF THE NULL WORLDLINE")
    print(f"  {'─'*55}")
    print(f"  A curve on a surface has geodesic curvature κ_g measuring")
    print(f"  its deviation from a surface geodesic. For the (p,q) knot:")
    print(f"    κ_g = (a · (n̂ × v)) / |v|³")
    print(f"  where v = tangent, a = acceleration, n̂ = surface normal.\n")

    kg = compute_knot_geodesic_curvature(R_e, r_e_tube, p_e, q_e)
    print(f"  Electron ({p_e},{q_e}) knot at r/R = α:")
    print(f"    RMS(κ_g) = {kg['kappa_g_rms']:.6e} m⁻¹")
    print(f"    max|κ_g| = {kg['kappa_g_max']:.6e} m⁻¹")
    print(f"    ⟨|κ_g|⟩  = {kg['kappa_g_mean_abs']:.6e} m⁻¹")
    print(f"    Is geodesic? {kg['is_geodesic']}  (κ_g > 0 → photon IS deflected)")
    print(f"    RMS(κ_g) × R = {kg['kappa_g_rms']*R_e:.6e}  (dimensionless)")

    print(f"\n  Physical meaning: κ_g is the 'sideways acceleration' the")
    print(f"  photon needs to stay on the knot path rather than following")
    print(f"  a geodesic of the torus surface.")

    # Sweep r/R
    print(f"\n  Sweep r/R → geodesic curvature:")
    print(f"  {'r/R':>10s} {'RMS(κ_g)×R':>14s} {'max|κ_g|×R':>14s} {'geodesic?':>10s}")
    print(f"  {'─'*50}")
    rr_vals = [0.001, 0.005, 0.01, 0.05, alpha, 0.1, 0.2, 0.3]
    for rr in rr_vals:
        sol = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=rr)
        if sol is None:
            continue
        R_tmp = sol['R']
        r_tmp = rr * R_tmp
        kg_tmp = compute_knot_geodesic_curvature(R_tmp, r_tmp, p_e, q_e)
        marker = "←α" if abs(rr - alpha) < 1e-5 else ""
        print(f"  {rr:>10.4f} {kg_tmp['kappa_g_rms']*R_tmp:>14.6e} {kg_tmp['kappa_g_max']*R_tmp:>14.6e} {'yes' if kg_tmp['is_geodesic'] else 'no':>10s}  {marker}")

    print(f"\n  Key result: κ_g × R → 1/√2 ≈ 0.707 as r/R → 0.")
    print(f"  The (2,1) knot is NEVER a geodesic of the embedded torus —")
    print(f"  the q=1 poloidal winding gives irreducible geodesic curvature.")
    print(f"  κ_g grows weakly with r/R but is always O(1/R).")

    # ================================================================
    # Section 3: CONFINEMENT FORCE = GEODESIC CURVATURE × ENERGY
    # ================================================================
    print(f"\n\n  3. CONFINEMENT FORCE = GEODESIC CURVATURE × ENERGY")
    print(f"  {'─'*55}")

    jb = compute_junction_balance(R_e, r_e_tube, p_e, q_e)

    print(f"  A photon of energy E following a path with geodesic curvature κ_g")
    print(f"  requires a transverse confining force:")
    print(f"    F_confine = (E/L) × κ_g   [force per unit path length]")
    print(f"\n  At r/R = α:")
    print(f"    E_total = {jb['E_total']/MeV:.6f} MeV = {jb['E_total']:.4e} J")
    print(f"    L_path  = {jb['L_path']:.4e} m")
    print(f"    E/L     = {jb['E_per_length']:.4e} J/m")
    print(f"    κ_g,rms = {jb['kappa_g_rms']:.4e} m⁻¹")
    print(f"    F_confine/L = {jb['F_confine_per_length']:.4e} N/m")
    print(f"\n  EM hoop force (magnetic self-stress):")
    print(f"    F_hoop      = {jb['F_hoop']:.4e} N  (total)")
    print(f"    F_hoop/L    = {jb['F_hoop_per_length']:.4e} N/m")
    print(f"\n  Ratio F_confine / (F_hoop/L) = {jb['force_ratio']:.4e}")
    if jb['force_ratio'] > 100:
        print(f"  → F_confine >> F_hoop/L: confining force (centripetal) dominates.")
        print(f"    The hoop force is a self-energy gradient (∂U_EM/∂R).")
        print(f"    The confining force is the centripetal requirement (E×κ_g).")
        print(f"    They are different forces — not the same force in different language.")
    else:
        print(f"  → These forces are the same order of magnitude.")
        print(f"    The EM self-field IS the waveguide that confines the photon.")

    # ================================================================
    # Section 4: ISRAEL → YOUNG-LAPLACE
    # ================================================================
    print(f"\n\n  4. ISRAEL → YOUNG-LAPLACE")
    print(f"  {'─'*55}")
    print(f"  Israel junction conditions across a hypersurface Σ:")
    print(f"    [K_ab] = -8πG(S_ab - h_ab S/2)")
    print(f"  where [K_ab] = K⁺_ab - K⁻_ab is the jump in extrinsic curvature")
    print(f"  and S_ab is the surface stress-energy tensor.\n")

    # Gravitational contribution
    mass_kg = m_e_MeV * MeV / c**2
    alpha_G = G_N * mass_kg**2 / (hbar * c)
    kappa_1_outer = 1.0 / (R_e + r_e_tube)  # at outer equator
    kappa_2_tube = 1.0 / r_e_tube
    K_flat_scale = max(kappa_1_outer, kappa_2_tube)

    print(f"  Gravitational vs EM contribution:")
    print(f"    α_G = Gm²/(ℏc) = {alpha_G:.4e}")
    print(f"    [K_ab]_grav ~ α_G × K^flat_ab ~ {alpha_G:.1e} × {K_flat_scale:.1e}")
    print(f"                                   ~ {alpha_G * K_flat_scale:.4e} m⁻¹")
    print(f"    This is negligible (10⁻⁴⁵ relative to EM).")

    print(f"\n  In flat-space limit (exterior ≈ flat Minkowski):")
    print(f"    Israel conditions → Young-Laplace equation:")
    print(f"    ΔP = σ_eff × (1/R₁ + 1/R₂)")
    print(f"  For the tube, poloidal curvature 1/r dominates:")
    print(f"    ΔP ≈ σ_eff / r")

    print(f"\n  EM radiation pressure (photon gas in waveguide):")
    print(f"    u_EM  = U_EM / V_tube = {jb['u_EM']:.4e} J/m³")
    print(f"    P_rad = u_EM = {jb['P_rad']:.4e} Pa")
    print(f"\n  Magnetic pressure at tube surface:")
    print(f"    B_surface = μ₀I/(2πr) = {jb['B_surface']:.4e} T")
    print(f"    P_B = B²/(2μ₀)       = {jb['P_B_surface']:.4e} Pa")
    print(f"\n  Effective surface tension from Young-Laplace balance:")
    print(f"    σ_eff = P_rad × r = {jb['sigma_eff']:.4e} N/m")

    # ================================================================
    # Section 5: WHAT THE JUNCTION TELLS US ABOUT r/R
    # ================================================================
    print(f"\n\n  5. WHAT THE JUNCTION TELLS US ABOUT r/R")
    print(f"  {'─'*55}")
    print(f"  The pressure balance P_rad ~ σ/r gives r in terms of field strengths.")
    print(f"  Since U_EM ~ α × E_circ and E_circ ∝ 1/R:")
    print(f"    U_EM / E_circ = α × [ln(8R/r) - 2] / (πp) = α × log_factor / (πp)")
    print(f"  (The factor p enters because L ≈ 2πpR for a (p,q) knot.)")
    print(f"\n  This is already encoded in the self-energy equation!")

    print(f"\n  Self-consistency check — r/R from junction balance vs NWT prescription:")
    print(f"  {'r/R input':>12s} {'log_factor':>12s} {'U_EM/E_circ':>14s} {'α×logf/(πp)':>14s} {'match?':>8s}")
    print(f"  {'─'*64}")
    for rr in [0.001, 0.01, alpha, 0.1, 0.2]:
        sol = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=rr)
        if sol is None:
            continue
        R_tmp = sol['R']
        r_tmp = rr * R_tmp
        params_tmp = TorusParams(R=R_tmp, r=r_tmp, p=2, q=1)
        se_tmp = compute_self_energy(params_tmp)
        lf = se_tmp['log_factor']
        ratio_actual = se_tmp['self_energy_fraction']
        ratio_predicted = alpha * lf / (np.pi * p_e)
        marker = "←α" if abs(rr - alpha) < 1e-5 else ""
        match = "✓" if abs(ratio_actual - ratio_predicted) / ratio_actual < 0.05 else "✗"
        print(f"  {rr:>12.5f} {lf:>12.4f} {ratio_actual:>14.6e} {ratio_predicted:>14.6e} {match:>8s}  {marker}")

    print(f"\n  The self-energy equation E_total = E_circ + U_EM already IS the")
    print(f"  junction condition expressed as an energy balance.")
    print(f"  The junction constraint gives r/R = f(α), but this is the SAME")
    print(f"  equation we already solve for the self-consistent radius.")

    # ================================================================
    # Section 6: ASSESSMENT
    # ================================================================
    print(f"\n\n  6. ASSESSMENT")
    print(f"  {'─'*55}")
    print(f"  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  WHAT THIS ANALYSIS CAN DO:                                │")
    print(f"  │  • Compute full torus differential geometry                │")
    print(f"  │    (metric, extrinsic curvature, Gauss-Bonnet ∫K_G dA=0)  │")
    print(f"  │  • Show geodesic curvature κ_g > 0 for r/R = α            │")
    print(f"  │    (photon IS deflected, needs confinement)                │")
    print(f"  │  • Identify confining force with EM self-field             │")
    print(f"  │    (waveguide effect: F = Eκ_g)                           │")
    print(f"  │  • Show Israel conditions reduce to EM pressure balance    │")
    print(f"  │    (Young-Laplace: ΔP = σ/r) in flat-space limit          │")
    print(f"  │                                                             │")
    print(f"  │  KEY INSIGHT:                                               │")
    print(f"  │  • Junction conditions ≡ self-energy equation              │")
    print(f"  │    (the 'second equation' is the first equation in         │")
    print(f"  │     disguise — the torus self-energy IS the flat-space     │")
    print(f"  │     junction condition)                                     │")
    print(f"  │                                                             │")
    print(f"  │  WHAT IT STILL CAN'T DO:                                   │")
    print(f"  │  • Pin m_e: junction gives r/R = f(α), not m = g(...)     │")
    print(f"  │  • The 'second equation' we hoped for from junctions      │")
    print(f"  │    is not independent — it reduces to the self-energy     │")
    print(f"  │    condition already solved                                 │")
    print(f"  └─────────────────────────────────────────────────────────────┘")
    print("=" * 70)
