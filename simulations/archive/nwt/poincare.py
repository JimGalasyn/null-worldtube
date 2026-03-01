"""
Poincaré homology sphere proton model.

The Poincaré homology sphere Σ(2,3,5) = S³/2I, where 2I is the binary
icosahedral group (order 120), is a Seifert fibered space with three
exceptional fibers of orders 2, 3, 5 — the only integers satisfying
1/a + 1/b + 1/c > 1 (spherical condition).

The topology forces exactly three classes of short closed geodesics with
energy ratios 5:3:2, determined purely by group theory. Hypothesis: the
proton is three photons trapped in this space, one on each exceptional
fiber.

Single free parameter: the sphere radius R.
"""

import numpy as np
from .constants import c, hbar, e_charge, mu0, alpha, MeV, PARTICLE_MASSES, G_N, M_Planck, l_Planck


# ──────────────────────────────────────────────────────────────────────
# 1. Binary icosahedral group 2I — conjugacy classes
# ──────────────────────────────────────────────────────────────────────

def binary_icosahedral_conjugacy_classes():
    """
    Return the 9 conjugacy classes of the binary icosahedral group 2I.

    Each element q ∈ 2I ⊂ SU(2) ≅ S³ acts on S³ by left multiplication.
    The fixed-point-free action gives S³/2I = Σ(2,3,5).

    A non-identity element q with Re(q) = cos(α) generates a geodesic
    of length L = R × α on Σ(2,3,5) (the shortest geodesic in its
    free homotopy class).

    Returns list of 9 dicts: {name, size, order, alpha, description}.
    """
    pi = np.pi
    classes = [
        {'name': '{1}',   'size': 1,  'order': 1,  'alpha': 0.0,
         'description': 'identity'},
        {'name': 'C_10a', 'size': 12, 'order': 10, 'alpha': pi / 5,
         'description': 'vertices (Z_5 stabilizer)'},
        {'name': 'C_6',   'size': 20, 'order': 6,  'alpha': pi / 3,
         'description': 'faces (Z_3 stabilizer)'},
        {'name': 'C_4',   'size': 30, 'order': 4,  'alpha': pi / 2,
         'description': 'edges (Z_2 stabilizer)'},
        {'name': 'C_5a',  'size': 12, 'order': 5,  'alpha': 2 * pi / 5,
         'description': 'vertex harmonics'},
        {'name': 'C_5b',  'size': 12, 'order': 5,  'alpha': 3 * pi / 5,
         'description': 'vertex harmonics'},
        {'name': 'C_3',   'size': 20, 'order': 3,  'alpha': 2 * pi / 3,
         'description': 'face harmonics'},
        {'name': 'C_10b', 'size': 12, 'order': 10, 'alpha': 4 * pi / 5,
         'description': 'vertex harmonics'},
        {'name': '{-1}',  'size': 1,  'order': 2,  'alpha': pi,
         'description': 'center'},
    ]

    total = sum(cl['size'] for cl in classes)
    assert total == 120, f"|2I| = {total}, expected 120"

    return classes


# ──────────────────────────────────────────────────────────────────────
# 2. Geodesic spectrum
# ──────────────────────────────────────────────────────────────────────

def compute_geodesic_spectrum(R):
    """
    Compute closed geodesic lengths and photon energies on Σ(2,3,5).

    Each non-identity conjugacy class [g] gives a family of closed
    geodesics of length L = R × α, where α = arccos(Re(g)).

    Photon energy: E = 2πℏc / L = 2πℏc / (Rα).

    Returns list of dicts sorted by length (shortest first), excluding
    the identity class.
    """
    classes = binary_icosahedral_conjugacy_classes()
    geodesics = []

    for cl in classes:
        if cl['alpha'] < 1e-12:
            continue  # skip identity
        L = R * cl['alpha']
        E_J = 2 * np.pi * hbar * c / L
        geodesics.append({
            'class': cl['name'],
            'size': cl['size'],
            'order': cl['order'],
            'alpha': cl['alpha'],
            'alpha_over_pi': cl['alpha'] / np.pi,
            'length': L,
            'length_fm': L * 1e15,
            'energy_J': E_J,
            'energy_MeV': E_J / MeV,
            'description': cl['description'],
        })

    geodesics.sort(key=lambda g: g['length'])
    return geodesics


# ──────────────────────────────────────────────────────────────────────
# 3. Spin character (Laplacian eigenvalue multiplicity)
# ──────────────────────────────────────────────────────────────────────

def spin_character(l, alpha):
    """
    Character of the spin-l representation of SU(2) at angle α.

    χ_l(α) = sin((l+1)α) / sin(α)

    Handles limits:
    - α → 0: χ_l(0) = l + 1
    - α → π: χ_l(π) = (-1)^l × (l + 1)
    """
    if abs(alpha) < 1e-12:
        return float(l + 1)
    if abs(alpha - np.pi) < 1e-12:
        return float((-1)**l * (l + 1))

    return np.sin((l + 1) * alpha) / np.sin(alpha)


# ──────────────────────────────────────────────────────────────────────
# 4. Laplacian spectrum via Molien sum
# ──────────────────────────────────────────────────────────────────────

def compute_laplacian_spectrum(R, l_max=30):
    """
    Eigenvalue multiplicities of the scalar Laplacian on Σ(2,3,5).

    The Molien formula for the quotient S³/Γ gives the multiplicity
    of the l-th spherical harmonic:

        m(l) = (1/|Γ|) × Σ_{classes} |class| × χ_l(α)

    where χ_l(α) = sin((l+1)α)/sin(α) is the spin-l character.

    Famous result: m(l) = 0 for l = 1, ..., 11. First nonzero mode
    at l = 12 with multiplicity 13. This is the spectral fingerprint
    of Σ(2,3,5).

    Returns list of dicts for nonzero multiplicities:
    {l, multiplicity, eigenvalue, eigenvalue_ratio}.
    """
    classes = binary_icosahedral_conjugacy_classes()
    group_order = 120
    spectrum = []

    for l in range(l_max + 1):
        # Molien sum with (l+1) factor: eigenspace on S³/Γ has
        # dimension d_l × (l+1), where d_l counts Γ-invariant
        # harmonics in the left representation and (l+1) is the
        # dimension of the untouched right representation.
        m_raw = 0.0
        for cl in classes:
            m_raw += cl['size'] * spin_character(l, cl['alpha'])
        m_raw *= (l + 1) / group_order

        m = int(round(m_raw))
        assert m >= 0, f"Negative multiplicity m({l}) = {m_raw:.6f}"

        if m > 0:
            # Eigenvalue of Laplacian on S³ of radius R: λ_l = l(l+2)/R²
            eigenvalue = l * (l + 2) / R**2
            spectrum.append({
                'l': l,
                'multiplicity': m,
                'eigenvalue': eigenvalue,
                'eigenvalue_R2': l * (l + 2),
                'energy_MeV': hbar * c * np.sqrt(eigenvalue) / MeV,
            })

    # Add eigenvalue ratio to fundamental (l=0 has eigenvalue 0, skip)
    if len(spectrum) > 1:
        e0 = spectrum[0]['eigenvalue_R2']
        for s in spectrum:
            s['eigenvalue_ratio'] = s['eigenvalue_R2'] / e0 if e0 > 0 else 0.0

    return spectrum


# ──────────────────────────────────────────────────────────────────────
# 5. Cone self-energy
# ──────────────────────────────────────────────────────────────────────

def compute_cone_self_energy(L_geodesic, cone_order, R_sphere):
    """
    Self-energy of a photon on an exceptional fiber of cone order α_cone.

    Near an exceptional fiber, the transverse space is a cone of angle
    2π/α_cone. The image charges from the cone give an enhancement
    factor of α_cone in the inductance.

    Parameters:
        L_geodesic: length of the closed geodesic (m)
        cone_order: order of the cone singularity (2, 3, or 5)
        R_sphere: radius of the Poincaré sphere (m)

    Following the torus self-energy pattern from core.py:
        I = ec/L  (time-averaged current)
        R_loop = L/(2π)  (effective loop radius)
        r_tube = R_sphere / (2 × cone_order)  (injectivity radius)
        L_ind = cone_order × μ₀ × R_loop × [ln(8 R_loop / r_tube) − 2]
        U_self = 2 × ½ × L_ind × I²  (factor 2: electric = magnetic at v=c)
    """
    I = e_charge * c / L_geodesic
    R_loop = L_geodesic / (2 * np.pi)
    r_tube = R_sphere / (2 * cone_order)

    log_factor = np.log(8.0 * R_loop / r_tube) - 2.0
    log_factor = max(log_factor, 0.01)

    L_ind = cone_order * mu0 * R_loop * log_factor

    U_mag = 0.5 * L_ind * I**2
    U_total = 2 * U_mag  # electric = magnetic at v = c

    E_circ = 2 * np.pi * hbar * c / L_geodesic

    return {
        'I_amps': I,
        'R_loop': R_loop,
        'r_tube': r_tube,
        'log_factor': log_factor,
        'cone_order': cone_order,
        'L_ind': L_ind,
        'U_mag_J': U_mag,
        'U_total_J': U_total,
        'U_total_MeV': U_total / MeV,
        'E_circ_J': E_circ,
        'E_circ_MeV': E_circ / MeV,
        'self_energy_fraction': U_total / E_circ,
    }


# ──────────────────────────────────────────────────────────────────────
# 6. Three-photon proton
# ──────────────────────────────────────────────────────────────────────

# The three exceptional fibers and their cone orders
PHOTON_FIBERS = [
    {'name': 'gamma_5', 'cone_order': 5, 'alpha': np.pi / 5,
     'energy_coeff': 10, 'description': 'vertices (order 5)'},
    {'name': 'gamma_3', 'cone_order': 3, 'alpha': np.pi / 3,
     'energy_coeff': 6, 'description': 'faces (order 3)'},
    {'name': 'gamma_2', 'cone_order': 2, 'alpha': np.pi / 2,
     'energy_coeff': 4, 'description': 'edges (order 2)'},
]


def compute_poincare_proton(R):
    """
    Total energy of three photons on the Poincaré homology sphere.

    Each photon circulates on one of the three exceptional Seifert fibers:
        γ₅: L = πR/5,  E = 10ℏc/R  (cone order 5, vertices)
        γ₃: L = πR/3,  E = 6ℏc/R   (cone order 3, faces)
        γ₂: L = πR/2,  E = 4ℏc/R   (cone order 2, edges)

    Total circulation: E = (10 + 6 + 4)ℏc/R = 20ℏc/R

    Plus cone self-energy corrections for each fiber.
    """
    photons = []
    total_circ_J = 0.0
    total_self_J = 0.0

    for fiber in PHOTON_FIBERS:
        L = R * fiber['alpha']
        E_circ_J = 2 * np.pi * hbar * c / L

        se = compute_cone_self_energy(L, fiber['cone_order'], R)

        E_total_J = E_circ_J + se['U_total_J']

        photon = {
            'name': fiber['name'],
            'cone_order': fiber['cone_order'],
            'alpha': fiber['alpha'],
            'alpha_over_pi': fiber['alpha'] / np.pi,
            'L': L,
            'L_fm': L * 1e15,
            'E_circ_J': E_circ_J,
            'E_circ_MeV': E_circ_J / MeV,
            'energy_coeff': fiber['energy_coeff'],
            'self_energy': se,
            'E_total_J': E_total_J,
            'E_total_MeV': E_total_J / MeV,
            'description': fiber['description'],
        }
        photons.append(photon)
        total_circ_J += E_circ_J
        total_self_J += se['U_total_J']

    total_J = total_circ_J + total_self_J

    return {
        'R': R,
        'R_fm': R * 1e15,
        'photons': photons,
        'total_circ_J': total_circ_J,
        'total_circ_MeV': total_circ_J / MeV,
        'total_self_J': total_self_J,
        'total_self_MeV': total_self_J / MeV,
        'total_J': total_J,
        'total_MeV': total_J / MeV,
        'self_fraction': total_self_J / total_circ_J if total_circ_J > 0 else 0,
        'energy_sum_coeff': sum(f['energy_coeff'] for f in PHOTON_FIBERS),
    }


# ──────────────────────────────────────────────────────────────────────
# 7. Find radius for target mass
# ──────────────────────────────────────────────────────────────────────

def find_poincare_radius(target_MeV):
    """
    Find sphere radius R where total Poincaré proton energy = target mass.

    Uses geometric bisection (60 iterations) over R ∈ [10⁻¹⁸, 10⁻¹²] m.
    Energy ~ 1/R so it decreases with R.
    """
    target_J = target_MeV * MeV

    R_lo, R_hi = 1e-18, 1e-12
    E_lo = compute_poincare_proton(R_lo)['total_J']
    E_hi = compute_poincare_proton(R_hi)['total_J']

    if not (E_lo > target_J > E_hi):
        return None

    for _ in range(60):
        R_mid = np.sqrt(R_lo * R_hi)
        E_mid = compute_poincare_proton(R_mid)['total_J']
        if E_mid > target_J:
            R_lo = R_mid
        else:
            R_hi = R_mid

    R_sol = np.sqrt(R_lo * R_hi)
    result = compute_poincare_proton(R_sol)
    result['target_MeV'] = target_MeV
    result['match_ppm'] = abs(result['total_MeV'] - target_MeV) / target_MeV * 1e6
    return result


# ──────────────────────────────────────────────────────────────────────
# 8. Energy scan
# ──────────────────────────────────────────────────────────────────────

def scan_poincare_energy(R_min=1e-17, R_max=1e-14, n_points=40):
    """Logarithmic sweep of R, returning Poincaré proton energy at each."""
    R_values = np.geomspace(R_min, R_max, n_points)
    return [compute_poincare_proton(R) for R in R_values]


# ──────────────────────────────────────────────────────────────────────
# 9. Gravitational self-consistency
# ──────────────────────────────────────────────────────────────────────

def compute_gravitational_binding(R, m_MeV):
    """
    Newtonian gravitational self-energy of a mass m confined to radius R.

    Returns U_grav (J, MeV), ratio to rest mass, and the gravitational
    coupling α_G = G m² / (ℏc) = (m/M_Planck)².
    """
    m_kg = m_MeV * MeV / c**2
    U_grav_J = -G_N * m_kg**2 / R
    U_grav_MeV = U_grav_J / MeV
    mc2_J = m_kg * c**2
    ratio = abs(U_grav_J) / mc2_J
    alpha_G = G_N * m_kg**2 / (hbar * c)  # = (m/M_Planck)²

    return {
        'U_grav_J': U_grav_J,
        'U_grav_MeV': U_grav_MeV,
        'ratio_to_mc2': ratio,
        'alpha_G': alpha_G,
        'm_over_M_Planck': m_kg / M_Planck,
    }


def compute_schwarzschild_self_consistency():
    """
    Geon condition: when does r_S = R for E = 20ℏc/R on Σ(2,3,5)?

    The Schwarzschild radius of energy E is r_S = 2GE/c⁴.
    With E = 20ℏc/R:
        r_S = 40 G ℏ / (R c³)
    Self-consistency r_S = R gives:
        R² = 40 G ℏ / c³ = 40 l_P²
        R_geon = √40 × l_P
        E_geon = 20 ℏc / R_geon = 20/√40 × M_Planck c²
    """
    R_geon = np.sqrt(40) * l_Planck
    E_geon_J = 20 * hbar * c / R_geon
    E_geon_MeV = E_geon_J / MeV
    E_geon_M_Pl = E_geon_J / (M_Planck * c**2)

    # Proton scale for comparison
    m_p_MeV = PARTICLE_MASSES['proton']
    R_proton = 20 * hbar * c / (m_p_MeV * MeV)

    return {
        'R_geon': R_geon,
        'R_geon_m': R_geon,
        'E_geon_J': E_geon_J,
        'E_geon_MeV': E_geon_MeV,
        'E_geon_M_Planck': E_geon_M_Pl,
        'sqrt_40': np.sqrt(40),
        'R_geon_over_l_P': R_geon / l_Planck,
        'R_proton_over_R_geon': R_proton / R_geon,
    }


def compute_deficit_angles(R):
    """
    Gravitational deficit angles from photon energy density on Σ(2,3,5).

    Each photon on a geodesic of length L has linear energy density
    μ = E/(L c²). The gravitational deficit angle per unit length:
        δ = 8πGμ/c² = 8πG × (2πℏ)/(L² c³)

    For the order-5 fiber (shortest, strongest gravity):
        L = πR/5, so δ = 8πG × (2πℏ)/((πR/5)² c³) = 400 × 2Gℏ/(R²c³)
                       = 400 l_P² / R² × (2π/R)
    Total deficit around one loop: Δ = δ × L ∝ l_P²/R².
    """
    results = []
    for fiber in PHOTON_FIBERS:
        L = R * fiber['alpha']
        E_J = 2 * np.pi * hbar * c / L
        mu = E_J / (L * c**2)  # linear energy density (kg/m)
        delta_per_length = 8 * np.pi * G_N * mu / c**2  # rad/m
        delta_total = delta_per_length * L  # total deficit around loop (rad)
        results.append({
            'name': fiber['name'],
            'cone_order': fiber['cone_order'],
            'L': L,
            'mu_kg_per_m': mu,
            'delta_per_length': delta_per_length,
            'delta_total_rad': delta_total,
            'topological_deficit_rad': 2 * np.pi * (1 - 1.0 / fiber['cone_order']),
        })

    return results


def compute_multiplicity_difference(l_max=200):
    """
    Compute Δm(l) = m_Σ(l) - (l+1)²/120 for l = 0..l_max.

    m_Σ(l): eigenvalue multiplicity on Σ(2,3,5) from Molien sum.
    (l+1)²/120: S³ multiplicity (l+1)² divided by |2I| = 120.

    Key properties:
    - Δm(l) = 0 for large l (same local geometry → same Weyl law)
    - Oscillates with decaying amplitude
    - Sum of Δm(l) over large range ≈ 0
    """
    classes = binary_icosahedral_conjugacy_classes()
    group_order = 120

    results = []
    running_sum = 0.0

    for l in range(l_max + 1):
        # Molien sum for Σ(2,3,5)
        m_raw = 0.0
        for cl in classes:
            m_raw += cl['size'] * spin_character(l, cl['alpha'])
        m_raw *= (l + 1) / group_order
        m_sigma = int(round(m_raw))

        # S³/120 smooth multiplicity
        m_s3_120 = (l + 1)**2 / group_order

        delta_m = m_sigma - m_s3_120
        running_sum += delta_m

        results.append({
            'l': l,
            'm_sigma': m_sigma,
            'm_s3_120': m_s3_120,
            'delta_m': delta_m,
            'running_sum': running_sum,
        })

    return results


def compute_casimir_energy_difference(R, l_max=200, n_t=2000):
    """
    Casimir energy difference between Σ(2,3,5) and S³/120.

    Uses heat kernel zeta regularization:
        ΔK'(t) = Σ_{l≥1} Δm(l) × exp(-l(l+2)t)
        Δζ(-½) = -(1/(2√π)) × ∫₀^∞ t^{-3/2} ΔK'(t) dt
        ΔE_Cas = (ℏc/(2R)) × Δζ(-½) = c_Cas × ℏc/R

    The integral converges because:
    - UV (t→0): same Weyl law → ΔK'(t) → 0
    - IR (t→∞): spectral gap → exponential decay

    Numerical method: log-transformed integral with trapezoidal rule.
    """
    # Step 1: Compute multiplicity differences
    mult_diff = compute_multiplicity_difference(l_max)

    # Extract arrays for l >= 1 (exclude l=0 zero mode)
    l_vals = np.array([d['l'] for d in mult_diff if d['l'] >= 1])
    delta_m = np.array([d['delta_m'] for d in mult_diff if d['l'] >= 1])
    eigenvalues = l_vals * (l_vals + 2)  # l(l+2) on unit S³

    # Step 2: Log-transformed integral
    # Substitute u = ln(t), dt = e^u du
    # ∫ t^{-3/2} ΔK'(t) dt = ∫ e^{-u/2} ΔK'(e^u) du
    u_min, u_max = -8.0, 1.0
    u_grid = np.linspace(u_min, u_max, n_t)
    du = u_grid[1] - u_grid[0]

    integrand = np.zeros(n_t)
    for i, u in enumerate(u_grid):
        t = np.exp(u)
        # ΔK'(t) = Σ_{l≥1} Δm(l) × exp(-l(l+2)t)
        exponents = -eigenvalues * t
        # Clip to avoid underflow warnings
        exponents = np.clip(exponents, -700, 0)
        delta_K_prime = np.sum(delta_m * np.exp(exponents))
        # Integrand: e^{-u/2} × ΔK'(e^u)
        integrand[i] = np.exp(-u / 2) * delta_K_prime

    # Trapezoidal rule
    integral = np.trapz(integrand, u_grid)

    # Δζ(-½) = -(1/(2√π)) × integral
    delta_zeta = -integral / (2 * np.sqrt(np.pi))

    # Casimir energy: ΔE = (ℏc/(2R)) × Δζ(-½) = c_Cas × ℏc/R
    c_Cas = delta_zeta / 2  # coefficient in front of ℏc/R
    E_Cas_J = c_Cas * hbar * c / R
    E_Cas_MeV = E_Cas_J / MeV

    # Convergence check: compute with half the l_max
    l_half = l_max // 2
    delta_m_half = delta_m[:l_half]
    eig_half = eigenvalues[:l_half]
    integrand_half = np.zeros(n_t)
    for i, u in enumerate(u_grid):
        t = np.exp(u)
        exponents = -eig_half * t
        exponents = np.clip(exponents, -700, 0)
        integrand_half[i] = np.exp(-u / 2) * np.sum(delta_m_half * np.exp(exponents))
    integral_half = np.trapz(integrand_half, u_grid)
    delta_zeta_half = -integral_half / (2 * np.sqrt(np.pi))
    c_Cas_half = delta_zeta_half / 2

    return {
        'c_Cas': c_Cas,
        'delta_zeta_minus_half': delta_zeta,
        'E_Cas_J': E_Cas_J,
        'E_Cas_MeV': E_Cas_MeV,
        'modified_coeff': 20 + c_Cas,
        'l_max': l_max,
        'n_t': n_t,
        'convergence': {
            'c_Cas_full': c_Cas,
            'c_Cas_half_lmax': c_Cas_half,
            'relative_change': abs(c_Cas - c_Cas_half) / abs(c_Cas) if abs(c_Cas) > 1e-30 else 0,
        },
        'integral_raw': integral,
    }


def compute_field_count_contributions(c_Cas_scalar):
    """
    Scale scalar Casimir result for different field types.

    - Scalar (s=0): c_Cas (computed)
    - EM (s=1, 2 polarizations): ~2 × c_Cas
      (note: conformal coupling on S³ gives exact cancellation for massless
       conformally coupled scalar; real EM has 2 physical polarizations)
    - Graviton (s=2): not computed (spin-2 spectral theory on Σ is open)
    """
    return {
        'scalar': {
            'c_Cas': c_Cas_scalar,
            'note': 'Minimally coupled massless scalar',
        },
        'em': {
            'c_Cas': 2 * c_Cas_scalar,
            'note': '2 polarizations; conformal coupling caveat on curved S³',
        },
        'graviton': {
            'c_Cas': None,
            'note': 'Spin-2 spectral theory on Σ(2,3,5) is a research problem',
        },
    }


# ──────────────────────────────────────────────────────────────────────
# 15. Analysis output
# ──────────────────────────────────────────────────────────────────────

def print_poincare_analysis():
    """Full Poincaré homology sphere proton analysis."""

    m_p_MeV = PARTICLE_MASSES['proton']
    r_proton_fm = 0.8414  # charge radius (fm)

    print("=" * 70)
    print("POINCARÉ HOMOLOGY SPHERE PROTON MODEL")
    print("  Σ(2,3,5) = S³/2I — three photons on exceptional Seifert fibers")
    print("  Single parameter: sphere radius R")
    print("=" * 70)

    # ─── 1. Geometry of Σ(2,3,5) ─────────────────────────────────────
    print("\n┌─ 1. GEOMETRY OF Σ(2,3,5) ──────────────────────────────")
    print(f"│  Σ(2,3,5) = S³ / 2I")
    print(f"│  2I = binary icosahedral group, |2I| = 120 = 5!")
    print(f"│")
    print(f"│  Seifert fibration over S² with 3 exceptional fibers:")
    print(f"│    Order 5 — icosahedron vertices  (12 elements)")
    print(f"│    Order 3 — icosahedron faces     (20 elements)")
    print(f"│    Order 2 — icosahedron edges     (30 elements)")
    print(f"│")
    print(f"│  Spherical condition: 1/2 + 1/3 + 1/5 = 31/30 > 1")
    print(f"│  (unique among triples of integers ≥ 2)")
    print(f"│")
    print(f"│  Homology: H₁(Σ) = 0  (homology sphere)")
    print(f"│  π₁(Σ) = 2I ≠ 1    (not simply connected)")
    print(f"│  Fundamental domain: 1/120 of S³")
    print(f"└────────────────────────────────────────────────────────")

    # ─── 2. Conjugacy classes ─────────────────────────────────────────
    print("\n┌─ 2. CONJUGACY CLASSES OF 2I ───────────────────────────")
    classes = binary_icosahedral_conjugacy_classes()
    print(f"│  {'Class':>7}  {'Size':>4}  {'Order':>5}  "
          f"{'α/π':>6}  {'α (rad)':>8}  Description")
    print(f"│  " + "─" * 60)

    for cl in classes:
        a_pi = cl['alpha'] / np.pi if cl['alpha'] > 0 else 0.0
        print(f"│  {cl['name']:>7}  {cl['size']:4d}  {cl['order']:5d}  "
              f"{a_pi:6.3f}  {cl['alpha']:8.5f}  {cl['description']}")

    total = sum(cl['size'] for cl in classes)
    print(f"│  " + "─" * 60)
    print(f"│  Total: {total} elements  ✓")
    print(f"└────────────────────────────────────────────────────────")

    # ─── 3. Geodesic spectrum ─────────────────────────────────────────
    print("\n┌─ 3. GEODESIC SPECTRUM ─────────────────────────────────")
    R_ref = 1e-15  # 1 fm reference
    geodesics = compute_geodesic_spectrum(R_ref)
    print(f"│  Reference radius: R = {R_ref*1e15:.1f} fm")
    print(f"│")
    print(f"│  {'Class':>7}  {'Mult':>4}  {'L/R':>7}  "
          f"{'L (fm)':>8}  {'E (MeV)':>10}  {'E×R/ℏc':>8}")
    print(f"│  " + "─" * 56)

    for g in geodesics:
        E_Rhc = g['energy_MeV'] * MeV * R_ref / (hbar * c)
        print(f"│  {g['class']:>7}  {g['size']:4d}  "
              f"{g['alpha']:7.4f}  {g['length_fm']:8.4f}  "
              f"{g['energy_MeV']:10.2f}  {E_Rhc:8.3f}")

    print(f"│")
    print(f"│  Three shortest (exceptional fibers):")
    print(f"│    γ₅: L = πR/5  →  E = 10ℏc/R  (vertices, order 5)")
    print(f"│    γ₃: L = πR/3  →  E = 6ℏc/R   (faces, order 3)")
    print(f"│    γ₂: L = πR/2  →  E = 4ℏc/R   (edges, order 2)")
    print(f"│")
    print(f"│  Energy ratios: 10 : 6 : 4 = 5 : 3 : 2")
    print(f"│  Pure topology — no free parameters.")
    print(f"└────────────────────────────────────────────────────────")

    # ─── 4. Laplacian spectrum ────────────────────────────────────────
    print("\n┌─ 4. LAPLACIAN SPECTRUM (MOLIEN SUM) ───────────────────")
    spectrum = compute_laplacian_spectrum(R_ref, l_max=30)
    print(f"│  m(l) = (1/120) Σ_classes |class| × sin((l+1)α)/sin(α)")
    print(f"│")
    print(f"│  {'l':>3}  {'m(l)':>5}  {'l(l+2)':>7}  "
          f"{'E (MeV)':>10}  Note")
    print(f"│  " + "─" * 44)

    # Show the gap
    for l in range(13):
        found = [s for s in spectrum if s['l'] == l]
        if found:
            s = found[0]
            note = ""
            if s['l'] == 0:
                note = "← ground state"
            elif s['l'] == 12:
                note = "← FIRST excited (spectral gap!)"
            print(f"│  {s['l']:3d}  {s['multiplicity']:5d}  "
                  f"{s['eigenvalue_R2']:7.0f}  "
                  f"{s['energy_MeV']:10.2f}  {note}")
        else:
            print(f"│  {l:3d}  {'0':>5}  {l*(l+2):7d}  "
                  f"{'—':>10}  (killed by 2I)")

    # Continue to l=30
    print(f"│  ...")
    for s in spectrum:
        if s['l'] > 12 and s['l'] <= 30:
            print(f"│  {s['l']:3d}  {s['multiplicity']:5d}  "
                  f"{s['eigenvalue_R2']:7.0f}  {s['energy_MeV']:10.2f}")

    print(f"│")
    print(f"│  Spectral gap: l = 0 → l = 12")
    print(f"│    Eigenvalue ratio: 168 / 0 = ∞ (from ground state)")
    print(f"│    vs S³ first mode: λ₁₂/λ₁ = 168/3 = 56×")
    print(f"│    The quotient by 2I kills 11 angular momentum sectors")
    print(f"│    First nonzero mode: l = 12, multiplicity 13")
    print(f"└────────────────────────────────────────────────────────")

    # ─── 5. Three-photon proton (raw) ─────────────────────────────────
    print("\n┌─ 5. THREE-PHOTON PROTON (CIRCULATION ONLY) ────────────")
    # Raw prediction: R = 20 ℏc / m_p c²
    R_raw = 20 * hbar * c / (m_p_MeV * MeV)
    print(f"│  E_total = (10 + 6 + 4) ℏc/R = 20 ℏc/R")
    print(f"│")
    print(f"│  At m_p = {m_p_MeV:.4f} MeV:")
    print(f"│    R_raw = 20 × ℏc / m_p = 20 × 197.327 / {m_p_MeV:.3f} fm")
    print(f"│         = {R_raw*1e15:.4f} fm")
    print(f"│")

    res_raw = compute_poincare_proton(R_raw)
    for ph in res_raw['photons']:
        pct = ph['E_circ_MeV'] / res_raw['total_circ_MeV'] * 100
        print(f"│    {ph['name']:>7}: E = {ph['energy_coeff']}ℏc/R = "
              f"{ph['E_circ_MeV']:.4f} MeV  ({pct:.1f}%)")

    print(f"│    {'total':>7}: E = 20ℏc/R  = {res_raw['total_circ_MeV']:.4f} MeV")
    print(f"│")
    print(f"│  Compare: m_p = {m_p_MeV:.4f} MeV")
    naive_gap = (res_raw['total_circ_MeV'] - m_p_MeV) / m_p_MeV * 100
    print(f"│  Gap (circulation only): {naive_gap:+.4f}%")
    print(f"└────────────────────────────────────────────────────────")

    # ─── 6. Cone self-energy corrections ──────────────────────────────
    print("\n┌─ 6. CONE SELF-ENERGY CORRECTIONS ──────────────────────")
    print(f"│  Near exceptional fiber of order n, transverse space")
    print(f"│  is a cone of angle 2π/n. Image charges give:")
    print(f"│    L_ind = n × μ₀ R_loop × [ln(8R_loop/r_tube) − 2]")
    print(f"│    r_tube = R/(2n)  (injectivity radius)")
    print(f"│    U_self = L_ind × I²  (electric = magnetic at v=c)")
    print(f"│")
    print(f"│  At R = {R_raw*1e15:.4f} fm:")
    print(f"│")
    print(f"│  {'Photon':>8}  {'cone':>4}  {'L_ind (H)':>12}  "
          f"{'U_self (MeV)':>12}  {'U/E_circ':>10}  {'enhance':>7}")
    print(f"│  " + "─" * 60)

    for ph in res_raw['photons']:
        se = ph['self_energy']
        enhance = se['self_energy_fraction'] / alpha  # relative to bare α
        print(f"│  {ph['name']:>8}  {ph['cone_order']:4d}  "
              f"{se['L_ind']:12.4e}  {se['U_total_MeV']:12.6f}  "
              f"{se['self_energy_fraction']:10.6f}  {enhance:7.2f}α")

    print(f"│")
    print(f"│  Total self-energy: {res_raw['total_self_MeV']:.6f} MeV")
    print(f"│  Self/circulation:  {res_raw['self_fraction']:.6f}")
    print(f"└────────────────────────────────────────────────────────")

    # ─── 7. Proton mass fit ───────────────────────────────────────────
    print("\n┌─ 7. PROTON MASS FIT ───────────────────────────────────")
    print(f"│  Find R where E_total (circ + self) = m_p = {m_p_MeV:.4f} MeV")
    print(f"│")

    sol = find_poincare_radius(m_p_MeV)
    if sol:
        print(f"│  R = {sol['R_fm']:.6f} fm")
        print(f"│  E_total = {sol['total_MeV']:.4f} MeV")
        print(f"│  Match: {sol['match_ppm']:.1f} ppm")
        print(f"│")

        # Breakdown
        print(f"│  Breakdown:")
        for ph in sol['photons']:
            se = ph['self_energy']
            print(f"│    {ph['name']:>7}: E_circ = {ph['E_circ_MeV']:8.4f} MeV, "
                  f"E_self = {se['U_total_MeV']:8.6f} MeV, "
                  f"E_total = {ph['E_total_MeV']:8.4f} MeV")

        print(f"│    {'sum':>7}: E_circ = {sol['total_circ_MeV']:8.4f} MeV, "
              f"E_self = {sol['total_self_MeV']:8.6f} MeV, "
              f"E_total = {sol['total_MeV']:8.4f} MeV")
        print(f"│")

        # Compare with charge radius
        ratio = sol['R_fm'] / r_proton_fm
        print(f"│  R / r_proton = {sol['R_fm']:.4f} / {r_proton_fm:.4f} = {ratio:.4f}")
        print(f"│  (r_proton = {r_proton_fm} fm, charge radius)")

        # Energy ratios
        print(f"│")
        print(f"│  Energy ratios (from topology):")
        energies = [ph['E_circ_MeV'] for ph in sol['photons']]
        e_min = min(energies)
        print(f"│    γ₅ : γ₃ : γ₂ = "
              f"{energies[0]/e_min:.1f} : {energies[1]/e_min:.1f} : {energies[2]/e_min:.1f}"
              f" = 5 : 3 : 2")
    else:
        print(f"│  Could not bracket m_p in R range [10⁻¹⁸, 10⁻¹²] m")

    print(f"└────────────────────────────────────────────────────────")

    # ─── 8. Physics summary ───────────────────────────────────────────
    print("\n┌─ 8. PHYSICS SUMMARY ───────────────────────────────────")
    print(f"│")
    print(f"│  What (2,3,5) predicts:")
    print(f"│    • Exactly 3 photon species (exceptional fibers)")
    print(f"│    • Energy ratios 5:3:2 (pure group theory)")
    print(f"│    • Sum coefficient 20 (fixes mass scale)")
    print(f"│    • Spectral gap at l=12 (rigid topology)")
    print(f"│")
    print(f"│  Numerology of (2, 3, 5):")
    print(f"│    2 + 3 + 5 = 10  (total circulation: 20 = 2×10)")
    print(f"│    2 × 3 × 5 = 30  (edges of icosahedron)")
    print(f"│    |2I| = 120 = 5! (binary icosahedral group)")
    print(f"│    1/2 + 1/3 + 1/5 = 31/30 > 1  (spherical condition)")
    print(f"│")
    print(f"│  Why Poincaré, not Borromean?")
    print(f"│    Borromean: 3 external torii, M/L ~ 10⁻⁵ (α-suppressed)")
    print(f"│    Poincaré: 3 intrinsic fibers, sharing ALL of the topology")
    print(f"│    The photons don't link — they COHABIT the same space.")
    print(f"│")

    if sol:
        print(f"│  Result:")
        print(f"│    R = {sol['R_fm']:.4f} fm, E = {sol['total_MeV']:.2f} MeV")
        R_check = 20 * hbar * c / (m_p_MeV * MeV)
        print(f"│    R_raw (no self-energy) = {R_check*1e15:.4f} fm")
        correction = (R_check - sol['R']) / R_check * 100
        print(f"│    Self-energy correction to R: {correction:+.4f}%")

    print(f"│")
    print(f"└────────────────────────────────────────────────────────")

    # ─── 9. Gravitational self-consistency ─────────────────────────────
    print("\n┌─ 9. GRAVITATIONAL SELF-CONSISTENCY ────────────────────")
    print(f"│  Can gravity constrain R? Three levels of analysis:")
    print(f"│")

    # Use proton-fit radius if available, otherwise raw
    R_grav = sol['R'] if sol else R_raw

    # Newtonian binding
    grav = compute_gravitational_binding(R_grav, m_p_MeV)
    print(f"│  (a) Newtonian binding energy:")
    print(f"│    α_G = G m_p² / (ℏc) = (m_p/M_Planck)² = {grav['alpha_G']:.4e}")
    print(f"│    m_p / M_Planck = {grav['m_over_M_Planck']:.4e}")
    print(f"│    U_grav = −G m² / R = {grav['U_grav_MeV']:.4e} MeV")
    print(f"│    |U_grav| / m_p c² = {grav['ratio_to_mc2']:.4e}")
    print(f"│    → Negligible: suppressed by α_G ~ 10⁻³⁹")
    print(f"│")

    # Geon condition
    geon = compute_schwarzschild_self_consistency()
    print(f"│  (b) Geon self-consistency (r_S = R):")
    print(f"│    r_S = 2GE/c⁴ = 40 G ℏ / (R c³)  [with E = 20ℏc/R]")
    print(f"│    R² = 40 l_P²")
    print(f"│    R_geon = √40 × l_P = {geon['sqrt_40']:.3f} × {l_Planck:.3e} m")
    print(f"│           = {geon['R_geon']:.4e} m")
    print(f"│    E_geon = 20ℏc/R_geon = {geon['E_geon_M_Planck']:.3f} M_Planck")
    print(f"│           = {geon['E_geon_MeV']:.4e} MeV")
    print(f"│    R_proton / R_geon = {geon['R_proton_over_R_geon']:.4e}")
    print(f"│    → Planck scale, not proton scale")
    print(f"│")

    # Deficit angles
    deficits = compute_deficit_angles(R_grav)
    print(f"│  (c) Gravitational deficit angles:")
    print(f"│    Each photon curves spacetime by δ = 8πGμ/c²")
    print(f"│")
    print(f"│    {'Fiber':>8}  {'δ_grav (rad)':>14}  {'δ_topo (rad)':>13}  {'ratio':>10}")
    print(f"│    " + "─" * 52)
    for d in deficits:
        ratio_d = d['delta_total_rad'] / d['topological_deficit_rad']
        print(f"│    {d['name']:>8}  {d['delta_total_rad']:14.4e}  "
              f"{d['topological_deficit_rad']:13.4f}  {ratio_d:10.2e}")
    print(f"│")
    print(f"│    Gravitational deficits ~ 10⁻³⁹ × topological deficits")
    print(f"│    → Classical gravity cannot constrain R")
    print(f"└────────────────────────────────────────────────────────")

    # ─── 10. Casimir energy on Σ(2,3,5) ───────────────────────────────
    print("\n┌─ 10. CASIMIR ENERGY ON Σ(2,3,5) ──────────────────────")
    print(f"│  Quantum vacuum effect: does topology-dependent Casimir")
    print(f"│  energy introduce a NEW scale?")
    print(f"│")

    # Multiplicity difference table
    mult = compute_multiplicity_difference(l_max=200)
    print(f"│  Multiplicity difference Δm(l) = m_Σ(l) − (l+1)²/120:")
    print(f"│")
    print(f"│  {'l':>4}  {'m_Σ':>5}  {'(l+1)²/120':>10}  {'Δm':>8}  {'Σ Δm':>8}")
    print(f"│  " + "─" * 42)
    for d in mult[:31]:  # l = 0..30
        print(f"│  {d['l']:4d}  {d['m_sigma']:5d}  {d['m_s3_120']:10.3f}  "
              f"{d['delta_m']:8.3f}  {d['running_sum']:8.3f}")
    print(f"│  ...")

    # Show running sum at end
    last = mult[-1]
    print(f"│  {last['l']:4d}  {last['m_sigma']:5d}  {last['m_s3_120']:10.3f}  "
          f"{last['delta_m']:8.3f}  {last['running_sum']:8.3f}")
    print(f"│")
    print(f"│  Running sum Σ Δm → {last['running_sum']:.3f} (Weyl law consistency)")
    print(f"│")

    # Heat kernel and Casimir computation
    print(f"│  Heat kernel zeta regularization:")
    print(f"│    ΔK'(t) = Σ_{{l≥1}} Δm(l) × exp(−l(l+2)t)")
    print(f"│    → 0 at t→0 (same Weyl law) and t→∞ (spectral gap)")
    print(f"│    Δζ(−½) = −(1/(2√π)) ∫ t^{{−3/2}} ΔK'(t) dt")
    print(f"│    ΔE_Cas = (ℏc/(2R)) × Δζ(−½) = c_Cas × ℏc/R")
    print(f"│")

    cas = compute_casimir_energy_difference(R_grav, l_max=200, n_t=2000)
    print(f"│  Numerical result (l_max=200, n_t=2000):")
    print(f"│    Δζ(−½) = {cas['delta_zeta_minus_half']:.6f}")
    print(f"│    c_Cas  = {cas['c_Cas']:.6f}")
    print(f"│    ΔE_Cas = {cas['E_Cas_MeV']:.6f} MeV  (at R = {R_grav*1e15:.4f} fm)")
    print(f"│")

    # Convergence
    conv = cas['convergence']
    print(f"│  Convergence check:")
    print(f"│    l_max=200: c_Cas = {conv['c_Cas_full']:.6f}")
    print(f"│    l_max=100: c_Cas = {conv['c_Cas_half_lmax']:.6f}")
    print(f"│    Relative change: {conv['relative_change']:.2e}")
    print(f"│")

    # Modified energy
    print(f"│  Modified proton energy:")
    print(f"│    E = (20 + c_Cas) × ℏc/R = {cas['modified_coeff']:.6f} × ℏc/R")
    print(f"│    Casimir shift: {cas['c_Cas']/20*100:+.4f}% of circulation")
    print(f"│    Still scales as ℏc/R — no new scale introduced!")
    print(f"│")

    # Field-type scaling
    fields = compute_field_count_contributions(cas['c_Cas'])
    print(f"│  Field-type scaling:")
    print(f"│    Scalar:   c_Cas = {fields['scalar']['c_Cas']:.6f}  ({fields['scalar']['note']})")
    print(f"│    EM:       c_Cas ≈ {fields['em']['c_Cas']:.6f}  ({fields['em']['note']})")
    print(f"│    Graviton: {fields['graviton']['note']}")
    print(f"│")
    print(f"│  c_Cas is a zero-parameter topological invariant of Σ(2,3,5).")
    print(f"└────────────────────────────────────────────────────────")

    # ─── 11. The honest conclusion ─────────────────────────────────────
    print("\n┌─ 11. THE HONEST CONCLUSION ────────────────────────────")
    print(f"│")
    print(f"│  Q: Can gravitational self-consistency constrain R?")
    print(f"│  A: No. But the analysis reveals something deeper.")
    print(f"│")
    print(f"│  Why gravity fails to fix R:")
    print(f"│    • α_G = (m_p/M_Planck)² ≈ {grav['alpha_G']:.1e}")
    print(f"│    • Geon scale: R_geon ≈ 10⁻³⁴ m (Planck, not proton)")
    print(f"│    • Deficit angles: 10⁻³⁹ × topological angles")
    print(f"│    • Casimir shifts coefficient (20 → {cas['modified_coeff']:.4f}),")
    print(f"│      not scaling law (still ℏc/R)")
    print(f"│")
    print(f"│  What DOES constrain structure (topology alone):")
    print(f"│    • Exactly 3 photon species")
    print(f"│    • Energy ratios 5:3:2")
    print(f"│    • Spectral gap at l = 12")
    print(f"│    • Casimir coefficient c_Cas = {cas['c_Cas']:.6f}")
    print(f"│")
    print(f"│  What COULD fix R (open questions):")
    print(f"│    • Confinement mechanism (what holds the photons in?)")
    print(f"│    • Running coupling at strong scale")
    print(f"│    • Quantum gravity (topology change, foam)")
    print(f"│    • Boundary conditions at R (matching to flat space)")
    print(f"│")
    print(f"│  The topology of Σ(2,3,5) is a perfect constraint machine")
    print(f"│  for RATIOS and COUNTS. Absolute scale requires physics")
    print(f"│  beyond the topology itself — and that's honest.")
    print(f"│")
    print(f"└────────────────────────────────────────────────────────")

    # ─── 12. Topological surgery and the hadron landscape ──────────────
    print("\n┌─ 12. TOPOLOGICAL SURGERY AND THE HADRON LANDSCAPE ─────")
    print(f"│")
    print(f"│  The proton is the ONLY stable hadron. Every other hadron")
    print(f"│  decays — with lifetimes spanning 58 orders of magnitude:")
    print(f"│")
    print(f"│    Particle     τ (s)       Decay               Type")
    print(f"│    " + "─" * 56)
    print(f"│    proton       > 10³⁴      (stable)            —")
    print(f"│    neutron      879          p e⁻ ν̄_e           weak")
    print(f"│    Λ            2.6×10⁻¹⁰   p π⁻ / n π⁰        weak")
    print(f"│    π±           2.6×10⁻⁸    μ± ν_μ              weak")
    print(f"│    K±           1.2×10⁻⁸    μ± ν_μ / π π        weak")
    print(f"│    B±           1.6×10⁻¹²   (many channels)     weak")
    print(f"│    π⁰           8.5×10⁻¹⁷   γ γ                 EM")
    print(f"│    ρ            4.5×10⁻²⁴   π π                 strong")
    print(f"│    Δ            5.6×10⁻²⁴   N π                 strong")
    print(f"│")
    print(f"│  KEY OBSERVATION: neutrinos appear if and only if the")
    print(f"│  decay involves a flavor change (weak interaction).")
    print(f"│  Strong decays (ρ → ππ) and EM decays (π⁰ → γγ)")
    print(f"│  never produce neutrinos.")
    print(f"│")
    print(f"│  TOPOLOGICAL INTERPRETATION")
    print(f"│  ─────────────────────────────────────────────────────")
    print(f"│")
    print(f"│  If hadrons are closed topologies and the neutrino is an")
    print(f"│  OPEN topology (a propagating twist-wave — see --neutrino),")
    print(f"│  then three classes of decay correspond to three types of")
    print(f"│  topological transition:")
    print(f"│")
    print(f"│    Strong decay:  closed → closed + closed")
    print(f"│      Topology rearranges among closed manifolds.")
    print(f"│      No cutting required. Fast (10⁻²⁴ to 10⁻²² s).")
    print(f"│      Example: ρ → π π (fiber rearrangement)")
    print(f"│")
    print(f"│    EM decay:      closed → open radiation")
    print(f"│      Topology annihilates to photons (trivial topology).")
    print(f"│      No surgery. Fast (10⁻²⁰ to 10⁻¹⁶ s).")
    print(f"│      Example: π⁰ → γγ (fiber-antiber annihilation)")
    print(f"│")
    print(f"│    Weak decay:    closed → closed + OPEN remnant")
    print(f"│      Topology must be CUT to change flavor.")
    print(f"│      The piece that can't close back up escapes as ν.")
    print(f"│      Slow (10⁻¹² to 10³ s). Always produces neutrinos.")
    print(f"│      Example: n → p e⁻ ν̄_e (one fiber recut → ν)")
    print(f"│")
    print(f"│  THE NEUTRINO AS TOPOLOGICAL COST OF SURGERY")
    print(f"│  ─────────────────────────────────────────────────────")
    print(f"│")
    print(f"│  Neutrino emission is not bookkeeping — it is the")
    print(f"│  topological cost of cutting a closed manifold.")
    print(f"│  This explains:")
    print(f"│")
    print(f"│    • Nearly massless: open topology has no ℏc/R scale")
    print(f"│    • Weakly interacting: open topology doesn't couple")
    print(f"│      easily to closed structures")
    print(f"│    • Always accompanies flavor change: flavor change IS")
    print(f"│      the surgery, the neutrino IS the remnant")
    print(f"│    • Three flavors: three charged lepton tori define")
    print(f"│      three distinct twist-wave channels")
    print(f"│")
    print(f"│  METASTABILITY HIERARCHY")
    print(f"│  ─────────────────────────────────────────────────────")
    print(f"│")
    print(f"│  Decay rate depends on how much surgery is needed:")
    print(f"│")
    print(f"│    Minimal surgery (fiber adjustment, same topology class):")
    print(f"│      n → p e⁻ ν̄_e    τ = 879 s (almost stable)")
    print(f"│      The neutron differs from the proton by a subtle")
    print(f"│      fiber reorientation — one small cut.")
    print(f"│")
    print(f"│    Moderate surgery (topology class change):")
    print(f"│      Λ → p π⁻         τ ~ 10⁻¹⁰ s")
    print(f"│      K → μ ν           τ ~ 10⁻⁸ s")
    print(f"│      Strange quark → lighter quark requires deeper cut.")
    print(f"│")
    print(f"│    Radical surgery (heavy flavor restructuring):")
    print(f"│      B± → many         τ ~ 10⁻¹² s")
    print(f"│      Massive topology change, many channels available.")
    print(f"│")
    print(f"│    No surgery needed (rearrangement only):")
    print(f"│      ρ → ππ, Δ → Nπ    τ ~ 10⁻²⁴ s")
    print(f"│      Topology is already in the right class,")
    print(f"│      just needs to split. No neutrino emitted.")
    print(f"│")
    print(f"│  PROTON STABILITY AS TOPOLOGICAL GROUND STATE")
    print(f"│  ─────────────────────────────────────────────────────")
    print(f"│")
    print(f"│  Σ(2,3,5) is the unique manifold with the minimum-energy")
    print(f"│  closed topology satisfying the spherical condition.")
    print(f"│  (2,3,5) is the ONLY triple of distinct integers ≥ 2 with")
    print(f"│  1/a + 1/b + 1/c > 1. There is nothing lower to decay to.")
    print(f"│")
    print(f"│  The proton's stability is then a THEOREM, not a parameter:")
    print(f"│    • No closed topology has lower energy at fixed R")
    print(f"│    • Surgery to a different topology costs a neutrino")
    print(f"│    • But there is no valid target topology below (2,3,5)")
    print(f"│    • Therefore: proton is absolutely stable")
    print(f"│")
    print(f"│  EXTENSION TO MESONS AND EXOTICS")
    print(f"│  ─────────────────────────────────────────────────────")
    print(f"│")
    print(f"│  The Seifert fiber framework extends beyond baryons:")
    print(f"│")
    print(f"│    Baryons (3 quarks):  3 exceptional fibers")
    print(f"│      Spherical: 1/a + 1/b + 1/c > 1")
    print(f"│      Unique ground state: (2,3,5)")
    print(f"│")
    print(f"│    Mesons (qq̄):  2 exceptional fibers")
    print(f"│      Lens spaces L(p,q), flat Seifert spaces")
    print(f"│      ALL are metastable — no unique ground state")
    print(f"│      Consistent: no stable mesons exist in nature")
    print(f"│")
    print(f"│    Pentaquarks (4q + q̄ or 5q): 4-5 fibers")
    print(f"│      Cannot satisfy spherical condition")
    print(f"│      Require hyperbolic base (1/Σ(1/aᵢ) < 1)")
    print(f"│      Highly metastable — consistent with observation")
    print(f"│")
    print(f"│  The pattern: stability correlates with the rigidity")
    print(f"│  of the topology. Only (2,3,5) is both spherical and")
    print(f"│  unique. Everything else has moduli → metastable.")
    print(f"│")
    print(f"└────────────────────────────────────────────────────────")
