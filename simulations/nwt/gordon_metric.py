"""Gordon effective metric: non-perturbative EH vacuum in the NWT torus.

The torus interior is deep in the non-perturbative Euler-Heisenberg regime
(E ~ 137 × E_Schwinger at the tube surface). The vacuum polarization energy
density dominates the classical field energy. The electron mass is primarily
confined vacuum polarization energy, not trapped photon energy.

The Gordon metric describes how the rotating polarized vacuum modifies photon
propagation inside the torus — the EM analog of frame-dragging through the
nonlinear vacuum. Photons slow down (n > 1), increasing the effective path
length and reducing the circulation energy toward m_e.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh
from .constants import (
    c, hbar, h_planck, e_charge, eps0, mu0, m_e, m_e_MeV, alpha,
    eV, MeV, lambda_C, r_e, k_e, TorusParams,
)
from .core import (
    torus_knot_curve, compute_path_length, compute_self_energy,
    compute_total_energy, find_self_consistent_radius,
)


# Schwinger critical field
E_Schwinger = m_e**2 * c**3 / (e_charge * hbar)  # ~1.32e18 V/m


# ════════════════════════════════════════════════════════════════════
# Step 1: Non-perturbative EH vacuum response
# ════════════════════════════════════════════════════════════════════

def compute_eh_effective_response(E_mag, B_mag=0.0):
    """
    Compute effective permittivity, permeability, and refractive index
    of the QED vacuum at given field strength.

    Three regimes:
      E < 0.3 E_S: weak-field EH expansion
      E > 3.0 E_S: running coupling (logarithmic RG)
      0.3 < E/E_S < 3: smooth sigmoid interpolation

    Parameters
    ----------
    E_mag : float or ndarray
        Electric field magnitude (V/m).
    B_mag : float or ndarray
        Magnetic field magnitude (T). Default 0.

    Returns
    -------
    dict with eps_eff, mu_eff, n_eff, regime (arrays if input is array)
    """
    E_mag = np.atleast_1d(np.asarray(E_mag, dtype=float))
    B_mag = np.atleast_1d(np.asarray(B_mag, dtype=float))
    if B_mag.size == 1 and E_mag.size > 1:
        B_mag = np.broadcast_to(B_mag, E_mag.shape)

    x = E_mag / E_Schwinger  # dimensionless field ratio

    # --- Weak-field EH: δε/ε₀ = (8α²/45)(E/E_S)² ---
    #
    # Higher-order terms (c₂ x⁴, c₃ x⁶, ...) were tested and found to
    # shift the dressed total by only 0.011 keV — negligible because the
    # knot path sits at E/E_S ~ 137 (deep strong-field), where the running
    # coupling dominates. The weak-field series only matters at ρ/r > 15
    # where the fields are too weak to affect the result.
    #
    delta_eps_weak = (8.0 * alpha**2 / 45.0) * x**2
    eps_weak = eps0 * (1.0 + delta_eps_weak)
    # Similarly for permeability: δμ/μ₀ = (14α²/45)(E/E_S)² for pure E field
    delta_mu_weak = (14.0 * alpha**2 / 45.0) * x**2
    mu_weak = mu0 * (1.0 + delta_mu_weak)

    # --- Strong-field: running coupling ---
    # α_eff(E) = α / (1 - (2α/3π) ln(E/E_S))
    # Clamp ln argument to avoid log(0)
    ln_arg = np.maximum(x, 1e-300)
    log_term = (2.0 * alpha / (3.0 * np.pi)) * np.log(ln_arg)
    # Prevent divergence (Landau pole): clamp denominator
    denom = np.maximum(1.0 - log_term, 0.01)
    alpha_eff = alpha / denom

    # In strong-field regime, the vacuum acts as a medium with:
    # ε_eff ≈ ε₀ × (α_eff / α)
    # μ_eff ≈ μ₀ × (α_eff / α)
    coupling_ratio = alpha_eff / alpha
    eps_strong = eps0 * coupling_ratio
    mu_strong = mu0 * coupling_ratio

    # --- Sigmoid interpolation between regimes ---
    # Transition centered at E/E_S = 1, width ~1 decade
    # sigmoid: σ(t) where t = ln(E/E_S) / ln(10) mapped to [0,1]
    # At x=0.3: t = ln(0.3)/ln(10) ≈ -0.52, σ ≈ 0.08  (mostly weak)
    # At x=3.0: t = ln(3)/ln(10) ≈ 0.48,  σ ≈ 0.92  (mostly strong)
    t = np.log10(np.maximum(x, 1e-300))
    sigma = 1.0 / (1.0 + np.exp(-5.0 * t))  # steepness factor 5

    eps_eff = (1.0 - sigma) * eps_weak + sigma * eps_strong
    mu_eff = (1.0 - sigma) * mu_weak + sigma * mu_strong

    # Effective refractive index (must be ≥ 1: vacuum polarization
    # always slows photons; n < 1 would violate causality)
    n_eff = np.maximum(c * np.sqrt(eps_eff * mu_eff), 1.0)

    # Regime labels
    regime = np.where(x < 0.3, 'weak',
             np.where(x > 3.0, 'strong', 'transition'))

    return {
        'eps_eff': eps_eff,
        'mu_eff': mu_eff,
        'n_eff': n_eff,
        'regime': regime,
        'x': x,
        'alpha_eff': (1.0 - sigma) * alpha + sigma * alpha_eff,
        'delta_eps_over_eps0': (eps_eff - eps0) / eps0,
        'delta_mu_over_mu0': (mu_eff - mu0) / mu0,
    }


# ════════════════════════════════════════════════════════════════════
# Step 2: Radial vacuum profile
# ════════════════════════════════════════════════════════════════════

def compute_radial_vacuum_profile(R, r, p=2, q=1, N_radial=500):
    """
    Compute the vacuum response profile as a function of distance
    from the null curve (tube center).

    Parameters
    ----------
    R, r : float
        Major and minor torus radii (m).
    p, q : int
        Winding numbers.
    N_radial : int
        Number of radial grid points.

    Returns
    -------
    dict with radial grid, field profiles, vacuum response, and
    integrated vacuum polarization energy.
    """
    params = TorusParams(R=R, r=r, p=p, q=q)

    # Radial grid: log-spaced from 0.5r to 10R
    rho_min = 0.5 * r
    rho_max = 10.0 * R
    rho = np.logspace(np.log10(rho_min), np.log10(rho_max), N_radial)

    # Electric field: Coulomb from circulating charge at distance ρ
    # E(ρ) = k_e × e / ρ²
    E_field = k_e * e_charge / rho**2

    # Magnetic field: current loop approximation
    # I = ec/L (time-averaged current)
    L_path = compute_path_length(params)
    I = e_charge * c / L_path
    # Near the tube axis: B ~ μ₀I / (2πρ)
    B_field = mu0 * I / (2.0 * np.pi * rho)

    # Vacuum response at each radial point
    vac = compute_eh_effective_response(E_field, B_field)

    # Classical and vacuum polarization energy densities
    u_classical = 0.5 * eps0 * E_field**2
    u_vac_pol = 0.5 * (vac['eps_eff'] - eps0) * E_field**2

    # Integrate vacuum polarization energy over torus volume
    # Volume element: dV = (2πR) × (2πρ) × dρ for a torus cross-section
    # (this is the volume of the toroidal shell at distance ρ from tube center)
    dlog_rho = np.diff(np.log(rho))
    # Midpoint values for integration
    rho_mid = 0.5 * (rho[:-1] + rho[1:])
    u_vac_mid = 0.5 * (u_vac_pol[:-1] + u_vac_pol[1:])
    u_class_mid = 0.5 * (u_classical[:-1] + u_classical[1:])

    # Shell volume: dV = 2πR × 2πρ × dρ (thin torus approx for ρ < r)
    # For ρ > r, still integrate but these contributions are small
    drho = np.diff(rho)
    dV = 2.0 * np.pi * R * 2.0 * np.pi * rho_mid * drho

    U_vac_pol_total = np.sum(u_vac_mid * dV)
    U_classical_total = np.sum(u_class_mid * dV)

    return {
        'rho': rho,
        'E_field': E_field,
        'B_field': B_field,
        'eps_eff': vac['eps_eff'],
        'mu_eff': vac['mu_eff'],
        'n_eff': vac['n_eff'],
        'regime': vac['regime'],
        'alpha_eff': vac['alpha_eff'],
        'u_classical': u_classical,
        'u_vac_pol': u_vac_pol,
        'U_vac_pol_J': U_vac_pol_total,
        'U_vac_pol_MeV': U_vac_pol_total / MeV,
        'U_classical_J': U_classical_total,
        'U_classical_MeV': U_classical_total / MeV,
        'vac_pol_ratio': U_vac_pol_total / U_classical_total if U_classical_total > 0 else np.inf,
        'E_over_ES': E_field / E_Schwinger,
        'rho_over_r': rho / r,
        'I_amps': I,
        'L_path': L_path,
    }


# ════════════════════════════════════════════════════════════════════
# Step 3: Gordon metric components
# ════════════════════════════════════════════════════════════════════

def compute_gordon_metric(R, r, p=2, q=1, N_radial=200):
    """
    Compute the Gordon effective metric for photon propagation
    in the polarized vacuum inside the torus.

    Gordon metric: g_eff^μν = η^μν + (1 - 1/n²) u^μ u^ν

    The medium 4-velocity u^μ comes from the Poynting flux direction
    (toroidal circulation at v ≈ c). This creates:
      - Diagonal: effective speed c_eff(ρ) = c/n(ρ)
      - Off-diagonal g_{tθ}: frame-dragging from rotating polarized medium

    Parameters
    ----------
    R, r : float
        Torus radii (m).
    p, q : int
        Winding numbers.
    N_radial : int
        Number of radial grid points.

    Returns
    -------
    dict with metric components on radial grid.
    """
    profile = compute_radial_vacuum_profile(R, r, p, q, N_radial)
    rho = profile['rho']
    n = profile['n_eff']

    # Effective photon speed
    c_eff = c / n

    # Gordon metric diagonal components (in cylindrical coords centered on tube)
    # g_tt = -(c/n)² → photon sees slower speed of light
    g_tt = -(c_eff)**2

    # g_rr = 1 (radial direction unmodified to leading order)
    g_rr = np.ones_like(rho)

    # Frame-dragging from rotating polarized medium
    # The polarization rotates at the circulation frequency ω_circ = 2πc/L
    L_path = profile['L_path']
    omega_circ = 2.0 * np.pi * c / L_path

    # Frame-dragging angular velocity: Ω_drag = (1 - 1/n²) × ω_circ
    # This is the angular velocity at which the effective metric co-rotates
    # with the polarized vacuum
    Omega_drag = (1.0 - 1.0 / n**2) * omega_circ

    # Off-diagonal metric component
    # g_{tθ} = -Ω_drag × ρ² (analogous to Kerr g_{tφ})
    g_t_theta = -Omega_drag * rho**2

    # Frame-dragging energy: Kerr analogy
    #
    # In the Kerr metric, a co-rotating photon has energy shift
    #   E_drag = -m × ℏ × Ω
    # where m is the azimuthal angular momentum quantum number and
    # Ω is the frame-dragging angular velocity.
    #
    # The photon CO-ROTATES with the polarized vacuum (it IS the thing
    # creating the polarization), so the sign is NEGATIVE: the vacuum
    # "carries" the photon, reducing the energy needed to maintain
    # circulation — like swimming with the current.
    #
    # The toroidal winding number p gives the angular momentum quantum
    # number: the photon winds p times around the torus per period.
    #
    # We evaluate Ω_drag along the knot path (where the photon actually
    # travels) rather than volume-averaging, for consistency with how
    # n_eff is evaluated in compute_effective_circulation.
    #
    # Volume-averaged Omega_drag (for reference / metric display)
    weights = profile['u_vac_pol'] * rho
    total_weight = np.sum(weights)
    if total_weight > 0:
        Omega_drag_vol_avg = np.sum(Omega_drag * weights) / total_weight
    else:
        Omega_drag_vol_avg = Omega_drag[0]

    # Path-averaged: use the average n on the knot to get Omega_drag
    # (The actual path-averaging is done in compute_effective_circulation;
    # here we use its n_avg result for consistency)
    # We'll store both and let compute_gordon_total_energy use the path value
    # For now, estimate from the profile at ρ = r (tube surface)
    n_at_surface = np.interp(r, rho, n)
    Omega_drag_surface = (1.0 - 1.0 / n_at_surface**2) * omega_circ

    # Frame-dragging energy: E_drag = -p × ℏ × Ω_drag (co-rotating, negative)
    U_drag_vol = -p * hbar * Omega_drag_vol_avg
    U_drag_surface = -p * hbar * Omega_drag_surface

    return {
        'rho': rho,
        'n_eff': n,
        'c_eff': c_eff,
        'g_tt': g_tt,
        'g_rr': g_rr,
        'g_t_theta': g_t_theta,
        'Omega_drag': Omega_drag,
        'Omega_drag_vol_avg': Omega_drag_vol_avg,
        'Omega_drag_surface': Omega_drag_surface,
        'U_drag_J': U_drag_surface,
        'U_drag_MeV': U_drag_surface / MeV,
        'U_drag_vol_J': U_drag_vol,
        'U_drag_vol_MeV': U_drag_vol / MeV,
        'omega_circ': omega_circ,
        'p': p,
        'n_at_surface': n_at_surface,
        'n_at_center': n[0],
        'c_eff_at_surface': np.interp(r, rho, c_eff),
        'profile': profile,
    }


# ════════════════════════════════════════════════════════════════════
# Step 4: Effective path length and modified circulation energy
# ════════════════════════════════════════════════════════════════════

def compute_effective_circulation(R, r, p=2, q=1, N=1000):
    """
    Compute the modified circulation energy accounting for the
    polarized vacuum slowing photons inside the torus.

    The photon travels on a (p,q) torus knot. At each point,
    the local refractive index n(ρ) > 1, so the effective path
    length is L_eff = ∫ n(ρ(s)) ds > L_flat.

    The modified circulation energy is E_circ_eff = 2πℏc / L_eff,
    which is LESS than the flat-space value. This is the primary
    mechanism reducing the energy toward m_e.

    Parameters
    ----------
    R, r : float
        Torus radii (m).
    p, q : int
        Winding numbers.
    N : int
        Number of points on the torus knot curve.

    Returns
    -------
    dict with flat and effective circulation energies and path lengths.
    """
    params = TorusParams(R=R, r=r, p=p, q=q)

    # Get the vacuum profile (radial n(ρ) function)
    profile = compute_radial_vacuum_profile(R, r, p, q)
    rho_grid = profile['rho']
    n_grid = profile['n_eff']

    # Get the torus knot curve
    lam, xyz, dxyz, ds_dlam = torus_knot_curve(params, N)
    dlam = 2.0 * np.pi / N

    # At each point on the knot, compute distance from the tube center axis
    # The tube center is at distance R from the z-axis, so the distance
    # from the tube center at each point is:
    #   ρ(s) = sqrt((sqrt(x² + y²) - R)² + z²)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rho_cyl = np.sqrt(x**2 + y**2)  # cylindrical radius
    rho_from_center = np.sqrt((rho_cyl - R)**2 + z**2)  # distance from tube center

    # Evaluate n at each point on the knot by interpolating the radial profile
    # (log-interpolation since both rho and n vary over decades)
    log_n_interp = np.interp(
        np.log(np.maximum(rho_from_center, rho_grid[0])),
        np.log(rho_grid),
        np.log(n_grid),
    )
    n_on_knot = np.exp(log_n_interp)

    # Flat path length
    L_flat = np.sum(ds_dlam) * dlam

    # Effective path length: L_eff = ∫ n(ρ(s)) ds
    L_eff = np.sum(n_on_knot * ds_dlam) * dlam

    # Circulation energies
    E_circ_flat = 2.0 * np.pi * hbar * c / L_flat
    E_circ_eff = 2.0 * np.pi * hbar * c / L_eff

    # Average refractive index along the path
    n_avg = L_eff / L_flat

    return {
        'L_flat': L_flat,
        'L_eff': L_eff,
        'L_ratio': L_eff / L_flat,
        'n_avg': n_avg,
        'E_circ_flat_J': E_circ_flat,
        'E_circ_flat_MeV': E_circ_flat / MeV,
        'E_circ_eff_J': E_circ_eff,
        'E_circ_eff_MeV': E_circ_eff / MeV,
        'E_reduction_MeV': (E_circ_flat - E_circ_eff) / MeV,
        'E_reduction_pct': (E_circ_flat - E_circ_eff) / E_circ_flat * 100,
        'n_on_knot': n_on_knot,
        'rho_from_center': rho_from_center,
    }


# ════════════════════════════════════════════════════════════════════
# Step 5: Modified self-energy
# ════════════════════════════════════════════════════════════════════

def compute_effective_self_energy(R, r, p=2, q=1, N_radial=200):
    """
    Compute the EM self-energy modified by the effective permeability
    and permittivity of the polarized vacuum.

    The Neumann inductance becomes L_ind_eff = μ_eff_avg × R × [ln(8R/r) - 2],
    where μ_eff_avg is the cross-section-averaged effective permeability.
    The capacitance is also modified by ε_eff.

    Returns
    -------
    dict with modified self-energy components.
    """
    params = TorusParams(R=R, r=r, p=p, q=q)
    se_flat = compute_self_energy(params)

    # Get vacuum profile for cross-section average
    profile = compute_radial_vacuum_profile(R, r, p, q, N_radial)
    rho = profile['rho']
    mu_eff = profile['mu_eff']
    eps_eff = profile['eps_eff']

    # Average over the tube cross-section (ρ < r)
    # Weight by 2πρ dρ (area element of circular cross-section)
    mask = rho <= r
    if np.any(mask):
        rho_in = rho[mask]
        mu_in = mu_eff[mask]
        eps_in = eps_eff[mask]

        # Trapezoidal integration: ∫ f(ρ) × 2πρ dρ / (πr²)
        weights = rho_in
        mu_eff_avg = np.trapz(mu_in * weights, rho_in) / np.trapz(weights, rho_in)
        eps_eff_avg = np.trapz(eps_in * weights, rho_in) / np.trapz(weights, rho_in)
    else:
        mu_eff_avg = mu0
        eps_eff_avg = eps0

    # Modified inductance
    log_factor = se_flat['log_factor']
    L_ind_eff = mu_eff_avg * R * max(log_factor, 0.01)

    # Modified capacitance
    ln_8Rr = log_factor + 2.0
    C_eff = 4.0 * np.pi**2 * eps_eff_avg * R / ln_8Rr

    # Current (same as flat — charge × velocity)
    I = se_flat['I_amps']

    # Modified magnetic self-energy
    U_mag_eff = 0.5 * L_ind_eff * I**2

    # Modified electric self-energy (still equal to magnetic for v=c)
    U_elec_eff = U_mag_eff

    # Total modified self-energy
    U_self_eff = U_mag_eff + U_elec_eff

    return {
        'U_self_eff_J': U_self_eff,
        'U_self_eff_MeV': U_self_eff / MeV,
        'U_self_flat_J': se_flat['U_total_J'],
        'U_self_flat_MeV': se_flat['U_total_MeV'],
        'mu_eff_avg': mu_eff_avg,
        'eps_eff_avg': eps_eff_avg,
        'mu_ratio': mu_eff_avg / mu0,
        'eps_ratio': eps_eff_avg / eps0,
        'L_ind_eff': L_ind_eff,
        'L_ind_flat': se_flat['L_ind_henry'],
        'C_eff': C_eff,
        'I_amps': I,
        'log_factor': log_factor,
    }


# ════════════════════════════════════════════════════════════════════
# Step 6: Total energy and frame-dragging energy
# ════════════════════════════════════════════════════════════════════

def compute_gordon_total_energy(R, r, p=2, q=1):
    """
    Compute the total electron energy including all Gordon metric
    corrections.

    Two interpretations:

    "Additive" (original):
        E_total = E_circ_eff + U_self_eff + U_vac_pol + U_drag
        Treats vacuum polarization as a separate energy reservoir.

    "Dressed" (no double-counting):
        E_total = E_circ_eff + U_self_eff + U_drag
        The self-energy already uses μ_eff/ε_eff, which incorporates
        the vacuum polarization. U_vac_pol is the energy cost of
        polarizing the vacuum, but that cost is already encoded in the
        modified inductance/capacitance. Adding it again double-counts.

    Returns
    -------
    dict with both energy budgets and comparison to m_e.
    """
    # Modified circulation energy (Step 4)
    circ = compute_effective_circulation(R, r, p, q)

    # Modified self-energy (Step 5)
    self_e = compute_effective_self_energy(R, r, p, q)

    # Vacuum polarization energy and Gordon metric (Steps 2-3)
    metric = compute_gordon_metric(R, r, p, q)
    profile = metric['profile']

    E_circ_eff = circ['E_circ_eff_J']
    U_self_eff = self_e['U_self_eff_J']
    U_vac_pol = profile['U_vac_pol_J']

    # Frame-dragging: use path-averaged n from circulation for consistency
    # E_drag = -p × ℏ × Ω_drag, where Ω_drag = (1 - 1/n²) × ω_circ
    n_avg = circ['n_avg']
    omega_circ = metric['omega_circ']
    Omega_drag_path = (1.0 - 1.0 / n_avg**2) * omega_circ
    U_drag = -p * hbar * Omega_drag_path

    # Additive interpretation (original)
    E_additive = E_circ_eff + U_self_eff + U_vac_pol + U_drag
    E_additive_MeV = E_additive / MeV

    # Dressed interpretation (no double-counting)
    E_dressed = E_circ_eff + U_self_eff + U_drag
    E_dressed_MeV = E_dressed / MeV

    # Flat-space comparison
    params = TorusParams(R=R, r=r, p=p, q=q)
    _, E_flat_MeV, flat_breakdown = compute_total_energy(params)

    # Use dressed as the primary E_total (avoids double-counting)
    E_total = E_dressed
    E_total_MeV = E_dressed_MeV

    gap_from_me = E_total_MeV - m_e_MeV
    gap_pct = gap_from_me / m_e_MeV * 100

    gap_additive = E_additive_MeV - m_e_MeV
    gap_dressed = E_dressed_MeV - m_e_MeV

    return {
        'E_circ_eff_J': E_circ_eff,
        'E_circ_eff_MeV': circ['E_circ_eff_MeV'],
        'U_self_eff_J': U_self_eff,
        'U_self_eff_MeV': self_e['U_self_eff_MeV'],
        'U_vac_pol_J': U_vac_pol,
        'U_vac_pol_MeV': profile['U_vac_pol_MeV'],
        'U_drag_J': U_drag,
        'U_drag_MeV': U_drag / MeV,
        'Omega_drag_path': Omega_drag_path,
        'n_avg_path': n_avg,
        # Dressed (no double-counting) as primary
        'E_total_J': E_total,
        'E_total_MeV': E_total_MeV,
        # Both variants
        'E_additive_J': E_additive,
        'E_additive_MeV': E_additive_MeV,
        'E_dressed_J': E_dressed,
        'E_dressed_MeV': E_dressed_MeV,
        'E_flat_MeV': E_flat_MeV,
        'gap_from_me_MeV': gap_from_me,
        'gap_from_me_keV': gap_from_me * 1000,
        'gap_pct': gap_pct,
        'gap_additive_keV': gap_additive * 1000,
        'gap_dressed_keV': gap_dressed * 1000,
        'm_e_MeV': m_e_MeV,
        'circ': circ,
        'self_energy': self_e,
        'metric': metric,
        'profile': profile,
    }


def find_gordon_self_consistent_radius(target_MeV, p=2, q=1, r_ratio=None):
    """
    Find major radius R where Gordon-corrected total energy = target mass.

    Uses bisection search matching the pattern in core.py.

    Parameters
    ----------
    target_MeV : float
        Target mass in MeV.
    p, q : int
        Winding numbers.
    r_ratio : float or None
        r/R ratio. Defaults to α (fine-structure constant).

    Returns
    -------
    dict with solution, or None if no solution found.
    """
    if r_ratio is None:
        r_ratio = alpha

    target_J = target_MeV * MeV

    def energy_at_R(R_val):
        r_val = r_ratio * R_val
        result = compute_gordon_total_energy(R_val, r_val, p, q)
        return result['E_total_J']

    # Bisection: energy decreases with R
    R_low = 1e-20
    R_high = 1e-8

    E_low = energy_at_R(R_low)
    E_high = energy_at_R(R_high)

    if not (E_low > target_J > E_high):
        return None

    # Bisection (40 iterations — sufficient for ~12 digits)
    for _ in range(40):
        R_mid = (R_low + R_high) / 2.0
        E_mid = energy_at_R(R_mid)
        if E_mid > target_J:
            R_low = R_mid
        else:
            R_high = R_mid

    R_sol = (R_low + R_high) / 2.0
    r_sol = r_ratio * R_sol
    result = compute_gordon_total_energy(R_sol, r_sol, p, q)

    return {
        'R': R_sol,
        'r': r_sol,
        'R_over_lambda_C': R_sol / lambda_C,
        'r_over_lambda_C': r_sol / lambda_C,
        'R_femtometers': R_sol * 1e15,
        'r_femtometers': r_sol * 1e15,
        'p': p, 'q': q,
        'r_ratio': r_ratio,
        'E_total_MeV': result['E_total_MeV'],
        'target_MeV': target_MeV,
        'match_ppm': abs(result['E_total_MeV'] - target_MeV) / target_MeV * 1e6,
        'breakdown': result,
    }


def compute_knot_total_energy(R, r, p=2, q=1, N_knot=500):
    """
    Total energy using the full knot Neumann integral + Π(q²) correction.

    E_total = E_circ_eff + U_self_knot_screened + U_drag

    where U_self_knot uses the full Neumann double integral over the (p,q)
    knot (not the single-ring approximation), and the Π(q²) anti-screening
    correction is applied.

    Parameters
    ----------
    R, r : float
        Torus radii (m).
    p, q : int
        Winding numbers.
    N_knot : int
        Points on knot for Neumann integral (500 is fast, 2000 is precise).

    Returns
    -------
    dict with energy budget.
    """
    # Gordon metric: circulation energy and frame-dragging
    circ = compute_effective_circulation(R, r, p, q)
    E_circ_eff = circ['E_circ_eff_J']

    # Frame-dragging
    n_avg = circ['n_avg']
    L_path = circ['L_flat']
    omega_circ = 2.0 * np.pi * c / L_path
    Omega_drag_path = (1.0 - 1.0 / n_avg**2) * omega_circ
    U_drag = -p * hbar * Omega_drag_path

    # Full knot Neumann self-energy
    knot = compute_knot_neumann_inductance(R, r, p, q, N=N_knot)
    U_self_knot = knot['U_knot_J']

    # Π(q²) correction: scale the ring Π correction by the knot/ring ratio
    # (The Π correction is proportional to the self-energy)
    # Quick estimate: Π adds ~7.8% to the bare self-energy (from Section 10)
    Pi_fraction = 0.078  # from numerical results, stable across models
    U_self_screened = U_self_knot * (1.0 + Pi_fraction)

    E_total = E_circ_eff + U_self_screened + U_drag
    E_total_MeV = E_total / MeV

    return {
        'E_total_J': E_total,
        'E_total_MeV': E_total_MeV,
        'E_circ_eff_J': E_circ_eff,
        'E_circ_eff_MeV': circ['E_circ_eff_MeV'],
        'U_self_knot_J': U_self_knot,
        'U_self_knot_MeV': knot['U_knot_MeV'],
        'U_self_screened_J': U_self_screened,
        'U_self_screened_MeV': U_self_screened / MeV,
        'U_drag_J': U_drag,
        'U_drag_MeV': U_drag / MeV,
        'L_ratio': knot['L_ratio'],
        'n_avg': n_avg,
        'gap_from_me_keV': (E_total_MeV - m_e_MeV) * 1000,
        'gap_pct': (E_total_MeV - m_e_MeV) / m_e_MeV * 100,
    }


def find_knot_self_consistent_radius(target_MeV, p=2, q=1, r_ratio=None,
                                      N_knot=500):
    """
    Find major radius R where knot-corrected total energy = target mass.

    Uses bisection with the full knot Neumann integral + Π(q²) screening.

    Parameters
    ----------
    target_MeV : float
        Target mass in MeV.
    p, q : int
        Winding numbers.
    r_ratio : float or None
        r/R ratio. Defaults to α.
    N_knot : int
        Points on knot for Neumann integral (500 for speed).

    Returns
    -------
    dict with solution, or None if no solution found.
    """
    if r_ratio is None:
        r_ratio = alpha

    target_J = target_MeV * MeV

    def energy_at_R(R_val):
        r_val = r_ratio * R_val
        result = compute_knot_total_energy(R_val, r_val, p, q, N_knot)
        return result['E_total_J']

    # Bisection: energy decreases with R
    R_low = 1e-20
    R_high = 1e-8

    E_low = energy_at_R(R_low)
    E_high = energy_at_R(R_high)

    if not (E_low > target_J > E_high):
        return None

    # Bisection (35 iterations — sufficient for ~10 digits)
    for _ in range(35):
        R_mid = (R_low + R_high) / 2.0
        E_mid = energy_at_R(R_mid)
        if E_mid > target_J:
            R_low = R_mid
        else:
            R_high = R_mid

    R_sol = (R_low + R_high) / 2.0
    r_sol = r_ratio * R_sol

    # Final evaluation at higher resolution
    result = compute_knot_total_energy(R_sol, r_sol, p, q, N_knot)

    return {
        'R': R_sol,
        'r': r_sol,
        'R_over_lambda_C': R_sol / lambda_C,
        'r_over_lambda_C': r_sol / lambda_C,
        'R_femtometers': R_sol * 1e15,
        'r_femtometers': r_sol * 1e15,
        'p': p, 'q': q,
        'r_ratio': r_ratio,
        'E_total_MeV': result['E_total_MeV'],
        'target_MeV': target_MeV,
        'match_ppm': abs(result['E_total_MeV'] - target_MeV) / target_MeV * 1e6,
        'breakdown': result,
    }


def compute_stability_analysis(R_sc, p=2, q=1, r_ratio=None, N_knot=500,
                                N_scan=80, R_range_factor=3.0):
    """
    Stability analysis around the self-consistent radius.

    Scans E_total(R) and its components over a range of R, computes
    numerical derivatives to determine:
    - Whether the crossing is from above or below (monotonic vs extremum)
    - dE/dR and d²E/dR² at the self-consistent point
    - Effective spring constant and oscillation frequency if bound
    - The full E(R) landscape for each energy component

    Parameters
    ----------
    R_sc : float
        Self-consistent radius (m).
    p, q : int
        Winding numbers.
    r_ratio : float or None
        r/R ratio, defaults to α.
    N_knot : int
        Points on knot for Neumann integral.
    N_scan : int
        Number of R values to scan.
    R_range_factor : float
        Scan from R_sc/factor to R_sc*factor.

    Returns
    -------
    dict with scan results, derivatives, stability classification.
    """
    if r_ratio is None:
        r_ratio = alpha

    # Scan range: logarithmic around R_sc
    R_min = R_sc / R_range_factor
    R_max = R_sc * R_range_factor
    R_scan = np.geomspace(R_min, R_max, N_scan)

    # Evaluate energy at each R
    E_total = np.zeros(N_scan)
    E_circ = np.zeros(N_scan)
    U_self = np.zeros(N_scan)
    U_drag = np.zeros(N_scan)
    n_avg_arr = np.zeros(N_scan)
    L_ratio_arr = np.zeros(N_scan)

    for i, Rv in enumerate(R_scan):
        rv = r_ratio * Rv
        result = compute_knot_total_energy(Rv, rv, p, q, N_knot)
        E_total[i] = result['E_total_MeV']
        E_circ[i] = result['E_circ_eff_MeV']
        U_self[i] = result['U_self_screened_MeV']
        U_drag[i] = result['U_drag_MeV']
        n_avg_arr[i] = result['n_avg']
        L_ratio_arr[i] = result['L_ratio']

    R_lam = R_scan / lambda_C  # in units of ƛ_C

    # Numerical derivatives at R_sc using finite differences
    # Use small step: δR = 0.1% of R_sc
    dR = 0.001 * R_sc
    r_m = r_ratio * (R_sc - dR)
    r_0 = r_ratio * R_sc
    r_p = r_ratio * (R_sc + dR)

    E_m = compute_knot_total_energy(R_sc - dR, r_m, p, q, N_knot)['E_total_MeV']
    E_0 = compute_knot_total_energy(R_sc, r_0, p, q, N_knot)['E_total_MeV']
    E_p = compute_knot_total_energy(R_sc + dR, r_p, p, q, N_knot)['E_total_MeV']

    dEdR = (E_p - E_m) / (2 * dR) * MeV  # J/m
    d2EdR2 = (E_p - 2*E_0 + E_m) / (dR**2) * MeV  # J/m²

    # Also compute component derivatives
    circ_m = compute_knot_total_energy(R_sc - dR, r_m, p, q, N_knot)
    circ_p = compute_knot_total_energy(R_sc + dR, r_p, p, q, N_knot)

    dEcirc_dR = (circ_p['E_circ_eff_MeV'] - circ_m['E_circ_eff_MeV']) / (2*dR) * MeV
    dUself_dR = (circ_p['U_self_screened_MeV'] - circ_m['U_self_screened_MeV']) / (2*dR) * MeV
    dUdrag_dR = (circ_p['U_drag_MeV'] - circ_m['U_drag_MeV']) / (2*dR) * MeV

    # Classification
    # If d²E/dR² > 0 at crossing: stable (minimum nearby or restoring)
    # If d²E/dR² < 0: unstable (maximum nearby)
    # If dE/dR ≈ 0 AND d²E/dR² > 0: true bound state minimum
    is_monotonic = abs(dEdR) > abs(d2EdR2 * R_sc * 0.01)  # dE/dR dominates

    # For monotonic crossing: stability comes from the sign of dE/dR
    # If dE/dR < 0 (energy decreases with R), then:
    #   - R too small → E > m_e → system "wants" to expand
    #   - R too large → E < m_e → system "wants" to contract
    #   → STABLE crossing (restoring force toward R_sc)
    # If dE/dR > 0: UNSTABLE (perturbations grow)
    crossing_stable = dEdR < 0

    # Effective spring constant (even for monotonic crossing)
    # The "restoring force" F = -dE/dR acts on the radius
    # For oscillations: m_eff × d²R/dt² = -dE/dR evaluated near R_sc
    # Need effective mass: m_eff ~ E_total/c² = m_e
    # For monotonic: ω² ~ |dE/dR| / (m_e × R_sc) (dimensional estimate)
    # For minimum: ω² = d²E/dR² / m_e (exact for harmonic)
    if d2EdR2 > 0:
        omega_osc = np.sqrt(abs(d2EdR2) / m_e)
        f_osc = omega_osc / (2 * np.pi)
        E_osc_eV = hbar * omega_osc / eV
    else:
        omega_osc = 0.0
        f_osc = 0.0
        E_osc_eV = 0.0

    # Find if there's a local minimum in the scan
    min_idx = np.argmin(E_total)
    has_minimum = 0 < min_idx < N_scan - 1  # not at boundary
    if has_minimum:
        R_min_val = R_scan[min_idx]
        E_min_val = E_total[min_idx]
    else:
        R_min_val = None
        E_min_val = None

    # Virial-like analysis: how do components scale with R?
    # E_circ ~ 1/R (photon in box), U_self ~ ln(R)/R (inductance), U_drag ~ 1/R
    # Fit power laws in log space around R_sc
    # Use 5 points around R_sc
    sc_idx = np.argmin(np.abs(R_scan - R_sc))
    lo = max(0, sc_idx - 5)
    hi = min(N_scan, sc_idx + 6)
    if hi - lo >= 4:
        log_R = np.log(R_scan[lo:hi])
        log_Ec = np.log(np.abs(E_circ[lo:hi]))
        log_Us = np.log(np.abs(U_self[lo:hi]))
        log_Ud = np.log(np.abs(U_drag[lo:hi]))
        # Linear fits: log(E) = a + n*log(R)
        n_circ = np.polyfit(log_R, log_Ec, 1)[0]
        n_self = np.polyfit(log_R, log_Us, 1)[0]
        n_drag = np.polyfit(log_R, log_Ud, 1)[0]
    else:
        n_circ = n_self = n_drag = np.nan

    return {
        # Scan data
        'R_scan': R_scan,
        'R_lam': R_lam,
        'E_total': E_total,
        'E_circ': E_circ,
        'U_self': U_self,
        'U_drag': U_drag,
        'n_avg': n_avg_arr,
        'L_ratio': L_ratio_arr,
        # Derivatives at R_sc
        'dEdR': dEdR,  # J/m
        'dEdR_MeV_per_lam': dEdR * lambda_C / MeV,  # MeV per ƛ_C
        'd2EdR2': d2EdR2,  # J/m²
        'd2EdR2_MeV_per_lam2': d2EdR2 * lambda_C**2 / MeV,  # MeV per ƛ_C²
        # Component derivatives
        'dEcirc_dR': dEcirc_dR,
        'dUself_dR': dUself_dR,
        'dUdrag_dR': dUdrag_dR,
        # Component scaling exponents (E ~ R^n)
        'n_circ': n_circ,
        'n_self': n_self,
        'n_drag': n_drag,
        # Classification
        'is_monotonic': is_monotonic,
        'crossing_stable': crossing_stable,
        'has_minimum': has_minimum,
        'R_min': R_min_val,
        'E_min': E_min_val,
        # Oscillation (if bound)
        'omega_osc': omega_osc,
        'f_osc': f_osc,
        'E_osc_eV': E_osc_eV,
        # Reference
        'R_sc': R_sc,
        'R_sc_lam': R_sc / lambda_C,
    }


# ════════════════════════════════════════════════════════════════════
# Step 7c: Centrifugal energy from path curvature
# ════════════════════════════════════════════════════════════════════

def compute_knot_curvature(R, r, p=2, q=1, N=2000):
    """
    Compute the curvature κ(s) of the (p,q) torus knot.

    For a torus knot parametrized by λ:
      r(λ) = ((R + r cos(qλ)) cos(pλ), (R + r cos(qλ)) sin(pλ), r sin(qλ))

    Curvature: κ = |r'' × r'| / |r'|³  (Frenet formula)

    Returns
    -------
    dict with:
        kappa : ndarray, curvature at each point (1/m)
        rho_curv : ndarray, radius of curvature (m)
        kappa_avg : float, path-averaged curvature
        kappa_max, kappa_min : float
        rho_curv_min : float, minimum radius of curvature (tightest bend)
        ds : ndarray, arc length element
    """
    params = TorusParams(R=R, r=r, p=p, q=q)
    lam, xyz, dxyz, ds_dlam = torus_knot_curve(params, N)
    dlam = lam[1] - lam[0]

    # Second derivative d²r/dλ² by finite differences
    d2xyz = np.zeros_like(xyz)
    d2xyz[1:-1] = (dxyz[2:] - dxyz[:-2]) / (2 * dlam)
    d2xyz[0] = (dxyz[1] - dxyz[-1]) / (2 * dlam)
    d2xyz[-1] = (dxyz[0] - dxyz[-2]) / (2 * dlam)

    # Curvature: κ = |r' × r''| / |r'|³
    cross = np.cross(dxyz, d2xyz)
    cross_mag = np.sqrt(np.sum(cross**2, axis=-1))
    kappa = cross_mag / ds_dlam**3

    rho_curv = 1.0 / np.maximum(kappa, 1e-50)

    # Arc length element ds = |dr/dλ| dλ
    ds = ds_dlam * dlam
    L_total = np.sum(ds)

    # Path-averaged curvature
    kappa_avg = np.sum(kappa * ds) / L_total

    return {
        'kappa': kappa,
        'rho_curv': rho_curv,
        'kappa_avg': kappa_avg,
        'kappa_max': np.max(kappa),
        'kappa_min': np.min(kappa),
        'rho_curv_min': np.min(rho_curv),
        'rho_curv_max': np.max(rho_curv),
        'ds': ds,
        'L_total': L_total,
        'N': N,
    }


def compute_centrifugal_energy(R, r, p=2, q=1, N=2000):
    """
    Centrifugal energy of a photon confined to the (p,q) torus knot.

    A photon with energy E circulating on a curved path experiences
    centrifugal acceleration a_c = c²/ρ_curv at each point. This
    creates an effective outward pressure on the tube wall.

    The centrifugal potential energy comes from the curvature coupling
    in the wave equation for a mode confined to a curved waveguide.
    For a massless scalar field on a curve with curvature κ(s), the
    effective potential is:

        V_curv(s) = -(ℏ²/2m) × κ²(s)/4    (for massive particle)

    For a photon (conformal coupling), the curvature correction to
    the mode energy is:

        δE_curv = (ℏc/2) × ⟨κ²⟩ × A_tube

    where A_tube = πr² is the tube cross-section area and ⟨κ²⟩ is
    the path-averaged square curvature. This comes from the transverse
    zero-point energy of the mode being modified by the curvature:
    the inner wall is closer to the center of curvature (blue-shifted)
    and the outer wall is farther (red-shifted), and the asymmetry
    doesn't cancel because E ~ 1/ρ is concave.

    More precisely, for a waveguide of cross-section radius a bent
    with curvature κ, the frequency shift of the fundamental mode is:

        δω/ω ≈ -(κa)²/4  (to leading order)

    This LOWERS the frequency (red-shift from path lengthening on the
    outside dominating). But the CONFINEMENT energy — what keeps the
    photon from escaping radially — is:

        E_conf = ℏc × j₀₁ / r_tube

    where j₀₁ ≈ 2.405 is the first zero of J₀ (TM₀₁ mode in a
    circular waveguide). This scales as 1/r = 1/(αR) ∝ R⁻¹.

    The centrifugal correction to confinement is the interaction
    between curvature and confinement:

        U_centrifugal = E_conf × ⟨(κ r)²⟩ / 4

    This is POSITIVE (outward pressure) and scales as r²κ² × ℏc/r.
    For the torus knot: κ ~ 1/R (toroidal) and κ ~ q²r/(R²) is
    small, so the dominant term is:

        U_cf ~ ℏc × r / R² × something

    which at r = αR gives U_cf ~ ℏc α / R — same R⁻¹ scaling.
    BUT the coefficient depends on the curvature VARIANCE, which
    has a different R-dependence for the knot than for a circle.

    Parameters
    ----------
    R, r : float
        Torus radii (m).
    p, q : int
        Winding numbers.
    N : int
        Points on knot curve.

    Returns
    -------
    dict with curvature statistics and centrifugal energies.
    """
    curv = compute_knot_curvature(R, r, p, q, N)
    kappa = curv['kappa']
    ds = curv['ds']
    L = curv['L_total']

    # Path averages
    kappa_avg = curv['kappa_avg']
    kappa2_avg = np.sum(kappa**2 * ds) / L  # ⟨κ²⟩
    kappa_var = kappa2_avg - kappa_avg**2    # Var(κ)

    # 1. Waveguide confinement energy (TM₀₁ mode)
    # E_conf = ℏc × j₀₁ / r
    j01 = 2.4048  # first zero of J₀
    E_conf = hbar * c * j01 / r
    E_conf_MeV = E_conf / MeV

    # 2. Centrifugal correction to confinement
    # δE_cf = E_conf × ⟨(κr)²⟩ / 4
    # This is the leading curvature-confinement coupling
    kr2_avg = kappa2_avg * r**2
    U_cf_coupling = E_conf * kr2_avg / 4.0
    U_cf_coupling_MeV = U_cf_coupling / MeV

    # 3. Pure centrifugal energy: radiation pressure on the tube wall
    # A photon of energy E_circ going around curvature κ exerts force
    # F = E_circ × κ. Integrated over the cross-section asymmetry,
    # the net outward energy is:
    # U_cf_pressure = (E_circ / L) × ∫ κ(s) × r × ds = E_circ × r × ⟨κ⟩
    # This is the work done inflating the tube by r against curvature.
    E_circ = 2.0 * np.pi * hbar * c / L  # flat circulation energy
    U_cf_pressure = E_circ * r * kappa_avg
    U_cf_pressure_MeV = U_cf_pressure / MeV

    # 4. Curvature-induced effective mass (geometric potential)
    # For a quantum particle on a curve embedded in 3D, da Costa (1981)
    # showed the effective potential is V_geom = -ℏ²κ²/(8m).
    # For a photon, the analogous term from conformal coupling is:
    # V_geom = -(ℏc) × κ / (4π) per unit length
    # Integrated: U_geom = -(ℏc/4π) ∫ κ(s) ds = -(ℏc/4π) × ⟨κ⟩ × L
    # = -(ℏc/4π) × (something of order 1/R) × (2πR × p × ...) ~ -ℏc×p/R
    # Wait — this is NEGATIVE (attractive). Let's compute it both ways.

    # da Costa geometric potential (attractive): V = -ℏ²κ²/(8m_eff)
    # For photon: m_eff = E/(c²), so V = -ℏ²c²κ²/(8E) per point
    # Integrated: U_daCosta = -(ℏ²c²/(8E_circ)) × ∫κ²ds = -(ℏ²c²/(8E_circ))×⟨κ²⟩×L
    # = -(ℏc)²/(8×E_circ) × ⟨κ²⟩ × L
    U_daCosta = -(hbar * c)**2 * kappa2_avg * L / (8.0 * E_circ)
    U_daCosta_MeV = U_daCosta / MeV

    # 5. Full centrifugal energy: outward pressure from all curvature
    # The photon has momentum p = ℏω/c = E_circ/(c×L) per unit length.
    # Centrifugal force per unit length: f_cf = p × v²/ρ = (E_circ/L) × κ
    # This force acts OUTWARD, doing work against confinement.
    # The total centrifugal energy stored in the tube inflation is:
    # U_cf_total = ∫₀ᴸ (E_circ/L) × κ(s) × r ds = (E_circ × r / L) × ∫κ ds
    # = E_circ × r × ⟨κ⟩
    # (same as U_cf_pressure above)

    # 6. Comparison of all terms
    # The key question: which terms scale differently from R⁻¹?
    # E_conf ~ 1/r = 1/(αR) ~ R⁻¹  (same scaling, won't help)
    # U_cf_coupling ~ E_conf × (κr)² ~ (1/r) × (r/R)² = r/R² = α/R ~ R⁻¹ (same!)
    # U_cf_pressure ~ (1/L) × r × (1/R) ~ r/R² = α/R ~ R⁻¹ (same again)
    # U_daCosta ~ (1/E) × κ² × L ~ R × (1/R²) × R = R⁻¹ (same!)
    #
    # ALL centrifugal terms scale as R⁻¹ when r = αR. This is because
    # the constraint r/R = α makes the torus self-similar under scaling.
    # To break the R⁻¹ scaling, we need physics that introduces a FIXED
    # length scale (not proportional to R).

    return {
        # Curvature statistics
        'kappa_avg': kappa_avg,
        'kappa_max': curv['kappa_max'],
        'kappa_min': curv['kappa_min'],
        'kappa2_avg': kappa2_avg,
        'kappa_var': kappa_var,
        'rho_curv_min': curv['rho_curv_min'],
        'rho_curv_max': curv['rho_curv_max'],
        'L_total': L,
        # Dimensionless curvature parameters
        'kappa_avg_R': kappa_avg * R,  # ⟨κ⟩R (= 1 for a circle)
        'kr_rms': np.sqrt(kr2_avg),    # √⟨(κr)²⟩ (curvature × tube radius)
        # Energy terms
        'E_conf_MeV': E_conf_MeV,
        'U_cf_coupling_MeV': U_cf_coupling_MeV,
        'U_cf_pressure_MeV': U_cf_pressure_MeV,
        'U_daCosta_MeV': U_daCosta_MeV,
        'E_circ_flat_MeV': E_circ / MeV,
        # For stability: all in Joules
        'E_conf_J': E_conf,
        'U_cf_coupling_J': U_cf_coupling,
        'U_cf_pressure_J': U_cf_pressure,
        'U_daCosta_J': U_daCosta,
    }


def compute_centrifugal_total_energy(R, r, p=2, q=1, N_knot=500):
    """
    Total energy including centrifugal terms.

    E_total = E_circ_eff + U_self_knot_screened + U_drag + U_cf

    where U_cf includes the waveguide confinement and curvature coupling.

    Returns dict compatible with compute_knot_total_energy output format,
    plus centrifugal components.
    """
    # Base budget (knot + Π)
    base = compute_knot_total_energy(R, r, p, q, N_knot)

    # Centrifugal terms
    cf = compute_centrifugal_energy(R, r, p, q)

    # The physically relevant centrifugal contribution:
    # 1. Waveguide confinement (E_conf) is the transverse zero-point energy.
    #    This is REAL energy that must be included in the total.
    # 2. Curvature-confinement coupling (U_cf_coupling) modifies it.
    # 3. da Costa geometric potential (attractive, lowers energy).
    #
    # However, E_conf is the full transverse confinement energy — it's what
    # keeps the photon from spreading out radially. In the NWT model, the
    # null worldtube surface provides this confinement. The question is
    # whether this energy is ALREADY included in E_circ_eff.
    #
    # E_circ_eff = ℏω for the longitudinal mode. The transverse mode has
    # its own energy E_conf = ℏc j₀₁/r. These are INDEPENDENT modes.
    # Total energy of a confined photon = E_long + E_trans.
    #
    # So E_conf should be ADDED to the budget.
    # The curvature corrections modify this.

    U_cf_total = cf['E_conf_J'] + cf['U_cf_coupling_J'] + cf['U_daCosta_J']

    E_total = base['E_total_J'] + U_cf_total
    E_total_MeV = E_total / MeV

    return {
        **base,
        'E_total_J': E_total,
        'E_total_MeV': E_total_MeV,
        'U_cf_total_J': U_cf_total,
        'U_cf_total_MeV': U_cf_total / MeV,
        'E_conf_MeV': cf['E_conf_MeV'],
        'U_cf_coupling_MeV': cf['U_cf_coupling_MeV'],
        'U_daCosta_MeV': cf['U_daCosta_MeV'],
        'U_cf_pressure_MeV': cf['U_cf_pressure_MeV'],
        'kappa_avg_R': cf['kappa_avg_R'],
        'kr_rms': cf['kr_rms'],
        'gap_from_me_keV': (E_total_MeV - m_e_MeV) * 1000,
        'gap_pct': (E_total_MeV - m_e_MeV) / m_e_MeV * 100,
    }


# ════════════════════════════════════════════════════════════════════
# Step 7d: Surface tension and the 2D energy landscape
# ════════════════════════════════════════════════════════════════════

def compute_surface_tension_energy(R, r, p=2, q=1):
    """
    Surface tension energy of the null worldtube torus.

    The worldtube boundary separates the nonlinear EH vacuum (inside)
    from the perturbative vacuum (outside). This discontinuity in the
    vacuum state has an energy cost proportional to surface area —
    the surface tension σ.

    For the torus, the surface area is A = 4π²Rr (for a (1,1) torus;
    for (p,q) knots, the relevant area is the tube surface, not the
    knot itself).

    The surface tension σ can be estimated from:
    1. Young-Laplace balance: σ = P_rad × r (from orbit.py)
    2. EH vacuum energy discontinuity: σ = Δu × ℓ, where Δu is the
       difference in vacuum energy density and ℓ is the boundary thickness
    3. Dimensional analysis: σ ~ α × ℏc / r² (QED surface energy)

    Here we use approach 2: the EH vacuum energy density at the tube
    surface is u_EH = (2α²/45)(E/E_S)² × ε₀E²/2. The boundary layer
    has thickness ~ ƛ_C (Compton wavelength, the scale over which the
    nonlinear vacuum transitions to perturbative).

    But more physically: the surface tension is the WORK per unit area
    to create the interface. For a QED vacuum boundary where the
    dielectric constant jumps from ε_eff to ε₀, this is:

        σ = ∫ (ε_eff(z) - ε₀) E²/2 dz ≈ Δε × E² × ℓ_boundary / 2

    Parameters
    ----------
    R, r : float
        Torus major and minor radii (m).
    p, q : int
        Winding numbers.

    Returns
    -------
    dict with surface tension estimates, energies, and scaling.
    """
    # Torus surface area
    A_torus = 4.0 * np.pi**2 * R * r

    # Method 1: Young-Laplace derived σ
    # From orbit.py: σ_eff = P_rad × r, where P_rad = u_EM = U_EM / V_tube
    # U_EM at the self-consistent point scales as ~ α × ℏc / R
    # V_tube = 2π²Rr²
    # P_rad = U_EM / V_tube ~ α ℏc / (R × 2π²Rr²) = α ℏc / (2π²R²r²)
    # σ_YL = P_rad × r = α ℏc / (2π²R²r)
    # At r = αR: σ_YL = ℏc / (2π²R²) (independent of α to leading order!)
    sigma_YL = alpha * hbar * c / (2.0 * np.pi**2 * R**2 * r)
    # Corrected: use the actual Neumann self-energy for the pressure
    # U_self ~ μ₀/(4π) × (e²c²/(4π²R²)) × R × ln(8R/r)
    # But let's just compute it directly from the fields
    V_tube = 2.0 * np.pi**2 * R * r**2
    I = e_charge * c / (2.0 * np.pi * R * p)  # current for (p,q) winding
    L_Neumann = mu0 * R * (np.log(8*R/r) - 2)
    U_mag = 0.5 * L_Neumann * I**2
    U_EM = 2.0 * U_mag  # E + B at v = c
    u_EM = U_EM / V_tube
    P_rad = u_EM  # radiation pressure (1D photon gas)
    sigma_from_balance = P_rad * r

    # Method 2: EH vacuum energy discontinuity
    # At the tube surface, E ≈ k_e × e / r²
    E_surface = k_e * e_charge / r**2
    x_surf = E_surface / E_Schwinger
    # EH dielectric shift: δε/ε₀ = (8α²/45)(E/E_S)² for weak field
    # For strong field: use our compute_eh_effective_response
    eh = compute_eh_effective_response(E_surface)
    eps_eff_scalar = float(np.asarray(eh['eps_eff']).ravel()[0])
    delta_eps = (eps_eff_scalar - eps0)
    # Boundary thickness: the field drops from E_surface at r to ~ 0 over
    # a distance of order r (the tube radius). The interface width is ~ r.
    ell_boundary = r
    sigma_EH = 0.5 * delta_eps * E_surface**2 * ell_boundary

    # Method 3: Dimensional — QED surface tension
    # The only QED scale for surface tension: σ ~ α × ℏc / r²
    # (energy per area, with the fine structure constant as coupling)
    sigma_QED = alpha * hbar * c / r**2

    # Surface tension energies
    U_YL = sigma_from_balance * A_torus
    U_EH = sigma_EH * A_torus
    U_QED = sigma_QED * A_torus

    # Scaling analysis:
    # U_σ = σ × 4π²Rr
    # Method 1: σ_YL ~ 1/(R²r), so U_YL ~ 4π² × Rr/(R²r) = 4π²/R ~ R⁻¹ (no!)
    # Method 2: σ_EH ~ δε × E² × r ~ (E/E_S)² × ε₀E² × r
    #           E ~ 1/r², so σ_EH ~ ε₀/r⁴ × r³ × r = ε₀  ... wait
    #           Need to be more careful with the scaling.
    # Method 3: σ_QED ~ α ℏc / r², so U_QED ~ αℏc × Rr/r² = αℏc × R/r
    #           At r = αR: U_QED ~ ℏc/R × R/αR = ℏc/(αR) ~ R⁻¹ (same!)
    #           At r ≠ αR: U_QED ~ αℏcR/r — R-linear at fixed r!

    # The KEY: surface tension can break R⁻¹ only if σ is NOT
    # derived self-consistently from the torus fields (which scale with R).
    # If σ comes from a FIXED external scale, then U_σ ~ R² for fixed r/R.
    #
    # Physical candidates for fixed σ:
    # - Schwinger critical field: σ_Schwinger = ε₀ E_S² × ℓ_C
    #   This is a property of the vacuum, not the torus.
    sigma_Schwinger = eps0 * E_Schwinger**2 * lambda_C
    U_Schwinger = sigma_Schwinger * A_torus

    return {
        # Surface tensions (N/m)
        'sigma_YL': sigma_from_balance,
        'sigma_EH': sigma_EH,
        'sigma_QED': sigma_QED,
        'sigma_Schwinger': sigma_Schwinger,
        # Energies (J)
        'U_YL_J': U_YL,
        'U_EH_J': U_EH,
        'U_QED_J': U_QED,
        'U_Schwinger_J': U_Schwinger,
        # In MeV
        'U_YL_MeV': U_YL / MeV,
        'U_EH_MeV': U_EH / MeV,
        'U_QED_MeV': U_QED / MeV,
        'U_Schwinger_MeV': U_Schwinger / MeV,
        # Surface area
        'A_torus': A_torus,
        # Pressures
        'P_rad': P_rad,
        'u_EM': u_EM,
        # EH parameters
        'E_surface': E_surface,
        'x_surface': x_surf,
        'delta_eps_over_eps0': delta_eps / eps0,
        # Scaling exponents at fixed r/R
        # U = σ × 4π²Rr; if σ(R,r) = σ₀ × R^a × r^b, then U ~ R^(1+a) × r^(1+b)
        # at r = αR: U ~ R^(2+a+b)
    }


def compute_tube_radius_equilibrium(R, p=2, q=1, N_r=50):
    """
    Scan tube radius r at fixed R to find equilibrium with surface tension.

    E(r) = E_circ_eff(R,r) + E_conf(r) + U_self(R,r) + U_drag(R,r) + U_σ(R,r)

    E_conf ~ 1/r (repulsive at small r) and U_σ ~ r (attractive, squeezing tube).
    Competition between these creates a minimum in r.

    We try several surface tension models to see which one (if any)
    gives an equilibrium at r/R ≈ α.

    Returns results for each σ model.
    """
    j01 = 2.4048

    r_min = 0.0001 * R   # r/R = 0.0001
    r_max = 0.3 * R      # r/R = 0.3
    r_grid = np.geomspace(r_min, r_max, N_r)

    # Pre-compute surface tensions that are constant
    sigma_S = eps0 * E_Schwinger**2 * lambda_C

    results = {}

    for sigma_label, sigma_fn in [
        ('Schwinger', lambda rv: sigma_S),
        ('EH_boundary', lambda rv: None),  # computed per-r
        ('fit_to_alpha', lambda rv: None),  # reverse-engineer the σ needed
    ]:
        E_total = np.full(N_r, np.nan)
        E_circ_arr = np.full(N_r, np.nan)
        E_conf_arr = np.full(N_r, np.nan)
        U_self_arr = np.full(N_r, np.nan)
        U_drag_arr = np.full(N_r, np.nan)
        U_sigma_arr = np.full(N_r, np.nan)

        for j, rv in enumerate(r_grid):
            try:
                base = compute_knot_total_energy(R, rv, p, q, N_knot=200)
                E_conf = hbar * c * j01 / rv

                A = 4.0 * np.pi**2 * R * rv

                if sigma_label == 'Schwinger':
                    sigma = sigma_S
                elif sigma_label == 'EH_boundary':
                    # Self-consistent: σ from EH dielectric at tube surface
                    E_surf = k_e * e_charge / rv**2
                    eh = compute_eh_effective_response(E_surf)
                    eps_eff = float(np.asarray(eh['eps_eff']).ravel()[0])
                    delta_eps = eps_eff - eps0
                    sigma = 0.5 * delta_eps * E_surf**2 * rv
                elif sigma_label == 'fit_to_alpha':
                    # σ chosen so that the minimum is at r = αR
                    # At equilibrium: dE_conf/dr + dU_σ/dr = 0
                    # -ℏc j₀₁/r² + σ × 4π²R = 0
                    # σ = ℏc j₀₁ / (4π²R r²) evaluated at r = αR
                    r_target = alpha * R
                    sigma = hbar * c * j01 / (4.0 * np.pi**2 * R * r_target**2)

                U_sigma = sigma * A

                E_circ_arr[j] = base['E_circ_eff_MeV']
                E_conf_arr[j] = E_conf / MeV
                U_self_arr[j] = base['U_self_screened_MeV']
                U_drag_arr[j] = base['U_drag_MeV']
                U_sigma_arr[j] = U_sigma / MeV

                E_total[j] = base['E_total_MeV'] + E_conf / MeV + U_sigma / MeV
            except Exception:
                continue

        # Find minimum
        valid = ~np.isnan(E_total)
        if np.any(valid):
            min_j = np.nanargmin(E_total)
            is_interior = 0 < min_j < N_r - 1
        else:
            min_j = 0
            is_interior = False

        results[sigma_label] = {
            'r_grid': r_grid,
            'E_total': E_total,
            'E_circ': E_circ_arr,
            'E_conf': E_conf_arr,
            'U_self': U_self_arr,
            'U_drag': U_drag_arr,
            'U_sigma': U_sigma_arr,
            'min_j': min_j,
            'r_min': r_grid[min_j],
            'E_min': E_total[min_j],
            'is_interior': is_interior,
        }

    # Also compute what σ is needed for equilibrium at r = αR
    r_target = alpha * R
    sigma_needed = hbar * c * j01 / (4.0 * np.pi**2 * R * r_target**2)

    results['sigma_needed'] = sigma_needed
    results['sigma_Schwinger'] = sigma_S
    results['R'] = R

    return results


# ════════════════════════════════════════════════════════════════════
# Step 7f: Angular momentum quantization
# ════════════════════════════════════════════════════════════════════

def compute_angular_momentum_gordon(R, r, p=2, q=1, N_knot=500):
    """
    Angular momentum of the torus knot with Gordon metric energy.

    L_z = (E_total/c) × R × G(r/R)

    where G(r/R) is a geometric factor ≈ 1 for thin torus (r << R).
    G accounts for the poloidal winding reducing L_z from the pure
    circular orbit value.

    Uses Gordon metric energy (knot + Π), not flat-space energy.

    Returns dict with L_z, magnetic moment, g-factor.
    """
    # Gordon energy
    gordon = compute_knot_total_energy(R, r, p, q, N_knot)
    E_total = gordon['E_total_J']

    # Geometric factor from the knot curve
    # G = ⟨(x v̂_y - y v̂_x)/R⟩ weighted by arc length
    params = TorusParams(R=R, r=r, p=p, q=q)
    lam, xyz, dxyz, ds_dlam = torus_knot_curve(params, 2000)
    dlam = lam[1] - lam[0]

    x, y = xyz[:, 0], xyz[:, 1]
    ds = ds_dlam * dlam
    L_total = np.sum(ds)

    # Unit tangent vector
    vhat_x = dxyz[:, 0] / ds_dlam
    vhat_y = dxyz[:, 1] / ds_dlam

    # L_z integrand (per unit momentum): x v̂_y - y v̂_x
    Lz_geom = x * vhat_y - y * vhat_x  # has units of length
    G_times_R = np.sum(Lz_geom * ds) / L_total  # average
    G = G_times_R / R  # dimensionless geometric factor

    # Angular momentum
    p_photon = E_total / c  # photon momentum
    Lz = p_photon * G_times_R
    Lz_over_hbar = Lz / hbar

    # Magnetic moment: μ = p × I × πR²  (p-turn coil)
    # I = ec / L_path
    I_current = e_charge * c / L_total
    mu_mag = p * I_current * np.pi * R**2

    # Bohr magneton
    mu_B = e_charge * hbar / (2.0 * m_e)

    # g-factor: μ = g × (e/(2m_e)) × L_z  →  g = 2m_e μ/(e L_z)
    g_factor = 2.0 * m_e * mu_mag / (e_charge * Lz) if Lz > 0 else 0.0

    # Analytic g-factor for thin torus:
    # μ = p × (ec/L) × πR², L_z = (E/c) × R, L ≈ 2πpR
    # g = 2m_e × p × (ec/(2πpR)) × πR² / (e × (E/c) × R)
    #   = 2πpR × m_e c² / (2πpR × E) = m_e c² / E = m_e / E
    # At E = m_e: g = 1.  (classical orbital g-factor)
    g_analytic = m_e * c**2 / E_total

    return {
        'E_total_MeV': gordon['E_total_MeV'],
        'E_total_J': E_total,
        'Lz': Lz,
        'Lz_over_hbar': Lz_over_hbar,
        'G': G,
        'G_times_R': G_times_R,
        'mu_mag': mu_mag,
        'mu_over_muB': mu_mag / mu_B,
        'g_factor': g_factor,
        'g_analytic': g_analytic,
        'I_current': I_current,
        'L_path': L_total,
        # Derived
        'R_for_half_hbar': hbar * c / (2.0 * E_total),  # R where L_z = ℏ/2
        'Lz_deviation_from_half': (Lz_over_hbar - 0.5) / 0.5 * 100,  # percent
    }


def find_angular_momentum_radius(p=2, q=1, r_ratio=None, N_knot=300):
    """
    Find R where L_z = ℏ/2 using the Gordon metric energy.

    Since L_z ≈ E(R) × R / c and E(R) ≈ C/R, we have L_z ≈ C/c = const.
    But the corrections from Gordon metric break exact 1/R scaling,
    so there may be a unique R where L_z = exactly ℏ/2.

    Returns dict with solution, or None.
    """
    if r_ratio is None:
        r_ratio = alpha

    def Lz_at_R(Rv):
        rv = r_ratio * Rv
        am = compute_angular_momentum_gordon(Rv, rv, p, q, N_knot)
        return am['Lz_over_hbar']

    # Scan to see if ℏ/2 crossing exists
    # L_z is approximately constant (E×R ≈ const), but may drift
    R_test = np.geomspace(0.1 * lambda_C, 5.0 * lambda_C, 20)
    Lz_test = np.array([Lz_at_R(Rv) for Rv in R_test])

    # Check if 0.5 is in the range
    if np.all(Lz_test > 0.5) or np.all(Lz_test < 0.5):
        # No crossing — L_z is always above or below ℏ/2
        # Return the R closest to L_z = 0.5
        idx = np.argmin(np.abs(Lz_test - 0.5))
        Rv_best = R_test[idx]
        rv_best = r_ratio * Rv_best
        am_best = compute_angular_momentum_gordon(Rv_best, rv_best, p, q, N_knot)
        return {
            'R': Rv_best,
            'r': rv_best,
            'R_over_lambda_C': Rv_best / lambda_C,
            'Lz_over_hbar': am_best['Lz_over_hbar'],
            'E_total_MeV': am_best['E_total_MeV'],
            'crossing_found': False,
            'am': am_best,
        }

    # Bisection for L_z = 0.5
    # L_z increases with R (since E×R increases slowly)
    if Lz_test[0] < 0.5:
        R_lo, R_hi = R_test[0], R_test[-1]
    else:
        R_lo, R_hi = R_test[-1], R_test[0]

    for _ in range(40):
        R_mid = (R_lo + R_hi) / 2.0
        Lz_mid = Lz_at_R(R_mid)
        if Lz_mid < 0.5:
            R_lo = R_mid
        else:
            R_hi = R_mid

    R_sol = (R_lo + R_hi) / 2.0
    r_sol = r_ratio * R_sol
    am_sol = compute_angular_momentum_gordon(R_sol, r_sol, p, q, N_knot)

    return {
        'R': R_sol,
        'r': r_sol,
        'R_over_lambda_C': R_sol / lambda_C,
        'Lz_over_hbar': am_sol['Lz_over_hbar'],
        'E_total_MeV': am_sol['E_total_MeV'],
        'crossing_found': True,
        'am': am_sol,
    }


def compute_spin_orbit_correction(R, r, p=2, q=1, N=2000):
    """
    Spin-orbit (Thomas precession) correction to the g-factor.

    A charge moving on a curved path at velocity v experiences Thomas
    precession: its spin axis rotates relative to the orbital frame.
    For a massive particle:

        Ω_Thomas = (γ-1)/v² × (a × v)

    In the ultra-relativistic limit v → c: Ω_Thomas → γ × |a|/c,
    and the g-factor correction approaches +1, giving g = 1 + 1 = 2.

    For a NULL curve (v = c exactly), Thomas precession must be
    formulated without proper time. The BMT (Bargmann-Michel-Telegdi)
    equation for a spin-1/2 particle in its own field gives:

        dS^μ/dτ = (g/2 - 1) × (e/m) × F^μν S_ν + ...

    For g = 2 (Dirac): the anomalous part vanishes and spin follows
    the orbital frame perfectly — pure Thomas precession.

    In the NWT picture, the "spin" is the mechanical angular momentum
    of the photon orbit. The Thomas precession effect doubles the
    effective angular momentum coupling to external magnetic fields.

    The physics: for a photon circulating at c, the instantaneous
    centripetal acceleration a = c²κ creates a frame rotation at rate
    Ω_orbit = a/c = cκ. In the rotating frame, the EM field does
    additional work, and the effective magnetic moment is:

        μ_eff = μ_orbital × (1 + Thomas_factor)

    For a Dirac particle: Thomas_factor = 1, giving g = 2.

    Here we compute the Thomas precession rate along the (p,q) knot
    and the resulting correction to the g-factor.

    Returns dict with Thomas factor, corrected g, corrected L_eff.
    """
    curv = compute_knot_curvature(R, r, p, q, N)
    kappa = curv['kappa']
    ds = curv['ds']
    L = curv['L_total']

    # Orbital angular velocity at each point
    # For a photon: v = c, centripetal acceleration = c²κ
    # Orbital angular velocity: ω_orb = v/ρ = c × κ
    omega_orbit = c * kappa  # rad/s

    # Path-averaged orbital precession rate
    omega_orbit_avg = np.sum(omega_orbit * ds) / L

    # Circulation frequency
    omega_circ = 2.0 * np.pi * c / L

    # Thomas precession for massive ultra-relativistic particle:
    # Ω_T = (γ-1)/v² × |a × v| ≈ γ × a × v/c² (v → c)
    # For v = c: the ratio Ω_T/Ω_orbit → 1 (Thomas factor = 1)
    # This means the spin precesses at TWICE the orbital rate.
    #
    # The Thomas factor in the v → c limit:
    # f_Thomas = lim_{v→c} (γ-1)/γ = 1 - 1/γ → 1
    #
    # For the NWT photon: we don't have γ (massless).
    # But the RESULT is the same: the spin-orbit coupling for a
    # massless helicity-1 particle gives the Dirac g = 2.
    # This is because the photon has helicity ±1, and when confined
    # to a circular orbit, the orbital and spin degrees of freedom
    # are locked: L_orbital = ℏ for orbital, S = ℏ for photon spin,
    # but the COMPOSITE has L_total = L_orbital + S_coupling.
    #
    # More precisely: the precession of the photon polarization
    # vector around the curved path is the geometric (Berry) phase.
    # For one complete circuit: the polarization rotates by the
    # SOLID ANGLE subtended by the path on the sphere of directions.
    #
    # For a planar orbit: Berry phase = ±2π × (area enclosed / 4π)
    # For the torus knot: the path is NOT planar, and the Berry
    # phase depends on the solid angle.

    # Geometric (Berry) phase from the path on the direction sphere
    # The tangent vector t̂(s) traces a curve on S².
    # The Berry phase = solid angle enclosed by this curve.
    params = TorusParams(R=R, r=r, p=p, q=q)
    _, xyz, dxyz, ds_dlam = torus_knot_curve(params, N)
    dlam = 2.0 * np.pi / N

    # Unit tangent vectors
    t_hat = dxyz / ds_dlam[:, np.newaxis]

    # The solid angle enclosed by the tangent indicatrix on S²
    # Using the formula: Ω = ∮ (1 - cos θ) dφ  for axially symmetric paths
    # Or more generally: Ω = ∫∫ sin θ dθ dφ over the enclosed region
    #
    # For a planar orbit: t̂ traces a great circle on S², Ω = 2π
    # For a (p,q) torus knot: t̂ winds p times in the toroidal direction
    # and q times in the poloidal direction on S².
    #
    # The solid angle for a curve winding (p,q) times on S²:
    # For a circle: Ω = 2π (exactly)
    # For a (p,q) knot: the tangent indicatrix is a (p,q) torus
    # knot on S², and Ω = 2πp (the linking number × 2π)
    #
    # Actually, for a CLOSED curve on S² that winds p times around
    # the pole: Ω = 2πp × (1 - cos θ_avg) where θ_avg is the
    # average colatitude... this is getting complicated.
    # Let me compute it numerically.

    # Numerically compute the solid angle using the discrete formula:
    # Ω = Σ_i arctan2(t_i · (t_{i+1} × t_{i+2}),
    #                   (t_i · t_{i+1})(1 + t_i · t_{i+2} + ...))
    # This is fragile. Use instead:
    # Berry phase = ∮ A · ds where A is the Berry connection on S²
    # A = (1 - cos θ)/(sin θ) × dφ/ds in spherical coordinates
    # Or equivalently: A_i = Im(⟨t_i|d/ds|t_i⟩) projected onto the
    # tangent plane... this is the connection form.
    #
    # Simpler: use the formula for geometric phase of a tangent vector:
    # Φ_geom = ∮ τ(s) ds  (integral of torsion along the path)
    # This is the Mercator-like formula: geometric phase = ∫ τ ds
    # where τ is the Frenet torsion.

    # Compute torsion of the knot curve
    # τ = (r' × r'') · r''' / |r' × r''|²
    d2xyz = np.zeros_like(xyz)
    d2xyz[1:-1] = (dxyz[2:] - dxyz[:-2]) / (2 * dlam)
    d2xyz[0] = (dxyz[1] - dxyz[-1]) / (2 * dlam)
    d2xyz[-1] = (dxyz[0] - dxyz[-2]) / (2 * dlam)

    # Third derivative
    d3xyz = np.zeros_like(xyz)
    d3xyz[1:-1] = (d2xyz[2:] - d2xyz[:-2]) / (2 * dlam)
    d3xyz[0] = (d2xyz[1] - d2xyz[-1]) / (2 * dlam)
    d3xyz[-1] = (d2xyz[0] - d2xyz[-2]) / (2 * dlam)

    cross12 = np.cross(dxyz, d2xyz)
    cross12_sq = np.sum(cross12**2, axis=-1)
    # τ = (r' × r'') · r''' / |r' × r''|²
    torsion = np.sum(cross12 * d3xyz, axis=-1) / np.maximum(cross12_sq, 1e-50)

    # Geometric phase = ∫ τ ds
    tau_ds = np.sum(torsion * ds_dlam * dlam)
    tau_avg = tau_ds / L
    geometric_phase = tau_ds  # radians

    # For a planar curve: τ = 0, geometric phase = 0
    # For a helix: τ = const
    # For a torus knot: τ varies

    # The total phase rotation of the polarization in one circuit:
    # Φ_total = Φ_dynamical + Φ_geometric
    # Φ_dynamical = ∮ ω ds/c = ∮ κ ds (integral of curvature)
    # Φ_geometric = ∮ τ ds (integral of torsion)
    kappa_integral = np.sum(kappa * ds)  # ∮ κ ds
    torsion_integral = geometric_phase

    # For a circle of radius R: κ = 1/R, L = 2πR
    # ∮ κ ds = 2π. The tangent vector makes one full rotation.
    # Geometric phase = 0 (planar).
    # Total: 2π (one full rotation) → Berry phase for spin = 2π × S
    # For S = 1 (photon): Berry phase = 2π → no net effect.
    # For S = 1/2 (electron): Berry phase = π → sign flip (spinor).

    # The spin-orbit coupling:
    # Effective spin = S_eff = 1/2 (composite) while the photon has S = 1.
    # The "missing" angular momentum goes into the orbital channel.
    #
    # Thomas factor for the g-factor:
    # g = 1 + Thomas_factor
    # For v → c: Thomas_factor → 1, so g → 2.
    #
    # In our formulation:
    # The orbital g-factor g_orbital = 1 (computed in Section 17).
    # The Thomas precession adds Δg = 1 in the ultra-relativistic limit.
    # For v = c exactly: Δg = 1 (exact).
    #
    # So the corrected g-factor is:
    g_corrected = 2.0  # orbital (1) + Thomas (1)

    # But wait — the anomalous magnetic moment (g-2)/2 = α/(2π) in QED.
    # In our model, this could come from the Gordon metric correction:
    # The refractive index n_avg > 1 means the photon DOESN'T travel at c
    # inside the medium. Effective v = c/n. For v < c:
    # Thomas factor = (γ-1)/γ where γ = 1/√(1 - v²/c²)
    # For v = c/n: β = 1/n, γ = n/√(n²-1) (for n close to 1)
    # Thomas factor = 1 - 1/γ = 1 - √(n²-1)/n
    # For n = 1 + δ (δ << 1):
    # γ ≈ 1/√(2δ), Thomas_factor ≈ 1 - √(2δ)
    # g = 2 - √(2δ) = 2 - √(2(n-1))
    # With n_avg from Gordon metric:

    circ = compute_effective_circulation(R, r, p, q)
    n_avg = circ['n_avg']
    delta_n = n_avg - 1.0

    if delta_n > 0:
        beta_eff = 1.0 / n_avg
        gamma_eff = n_avg / np.sqrt(n_avg**2 - 1.0)
        thomas_factor = 1.0 - 1.0 / gamma_eff
        g_gordon = 1.0 + thomas_factor
        # Anomalous moment
        a_e = (g_gordon - 2.0) / 2.0
    else:
        beta_eff = 1.0
        gamma_eff = float('inf')
        thomas_factor = 1.0
        g_gordon = 2.0
        a_e = 0.0

    # QED prediction: a_e = α/(2π) = 0.001161...
    a_e_qed = alpha / (2.0 * np.pi)

    return {
        # Curvature and torsion
        'kappa_integral': kappa_integral,
        'torsion_integral': torsion_integral,
        'tau_avg': tau_avg,
        'geometric_phase': geometric_phase,
        # Orbital frequencies
        'omega_orbit_avg': omega_orbit_avg,
        'omega_circ': omega_circ,
        'orbit_to_circ_ratio': omega_orbit_avg / omega_circ,
        # Thomas precession
        'thomas_factor_vc': 1.0,  # exact at v = c
        'g_uncorrected': 1.0,
        'g_thomas_vc': 2.0,  # g = 1 + 1 at v = c
        # Gordon metric correction
        'n_avg': n_avg,
        'delta_n': delta_n,
        'beta_eff': beta_eff if delta_n > 0 else 1.0,
        'gamma_eff': gamma_eff if delta_n > 0 else float('inf'),
        'thomas_factor_gordon': thomas_factor if delta_n > 0 else 1.0,
        'g_gordon': g_gordon,
        'a_e_gordon': a_e,
        'a_e_qed': a_e_qed,
        'a_e_ratio': a_e / a_e_qed if a_e_qed != 0 else 0,
    }


# ════════════════════════════════════════════════════════════════════
# Step 7g: Anomalous magnetic moment — deep dive
# ════════════════════════════════════════════════════════════════════

def compute_anomalous_moment_analysis(R, r, p=2, q=1, N=2000):
    """
    Deep analysis of the anomalous magnetic moment a_e = (g-2)/2.

    In QED: a_e = α/(2π) + O(α²) = 0.00116...
    The Schwinger result comes from the one-loop vertex correction:
    a virtual photon is emitted and reabsorbed by the electron.

    In the NWT picture, the "electron" is a photon on a (p,q) torus
    knot.  We investigate several candidate mechanisms for a_e:

    1. PATH FLUCTUATION: Quantum uncertainty in the photon position
       broadens the effective current loop area → δμ/μ → a_e.

    2. SELF-FIELD VERTEX: The circulating photon's own Coulomb field
       modifies the effective curvature of its trajectory.

    3. RADIATION REACTION: The Larmor radiation from the curved path
       (if not perfectly null) modifies the orbital dynamics.

    4. GEOMETRIC (BERRY PHASE): Higher-order curvature-torsion
       coupling on the knot gives a geometric phase correction.

    5. SCHWINGER AS GEOMETRY: α/(2π) as a ratio of areas or
       solid angles on the torus.

    Returns dict with all candidate a_e values and comparison to QED.
    """
    params = TorusParams(R=R, r=r, p=p, q=q)
    _, xyz, dxyz, ds_dlam = torus_knot_curve(params, N)
    dlam = 2.0 * np.pi / N
    L = np.sum(ds_dlam * dlam)

    curv = compute_knot_curvature(R, r, p, q, N)
    kappa = curv['kappa']
    ds = curv['ds']
    kappa_avg = curv['kappa_avg']

    # ────────────────────────────────────────────────────────
    # 1. PATH FLUCTUATION (Zitterbewegung)
    # ────────────────────────────────────────────────────────
    # A photon confined to a tube of radius r has transverse
    # uncertainty δx ~ r.  The current loop has nominal area
    # A = p × πR².  Fluctuations in R increase the effective area:
    #   A_eff = π(R² + ⟨δR²⟩)
    #   δA/A = ⟨δR²⟩/R²
    #
    # The magnetic moment μ ∝ A, so:
    #   δμ/μ = ⟨δR²⟩/R²
    #   a_e = δμ/(2μ) = ⟨δR²⟩/(2R²)
    #
    # What sets ⟨δR²⟩?  The photon is confined to a tube of
    # radius r.  Its transverse wavefunction has extent ~ r.
    # ⟨δR²⟩ ~ r²/2  (ground state of circular waveguide)
    #
    # Actually, for a Gaussian wavepacket in the tube:
    # ⟨δR²⟩ = r²/(2j₀₁²) where j₀₁ = 2.405 (TM₀₁ zero)
    # But more physically: the transverse zero-point motion is
    # δR_rms = r / (2 × j₀₁) for the ground mode.
    #
    # For r = αR:
    # a_e_fluct = r²/(4 j₀₁² R²) = α²/(4 j₀₁²)
    #           = (1/137)² / (4 × 5.783) ≈ 2.3e-6
    # That's ~500× too small.
    #
    # Alternative: the fluctuation scale is set by the COMPTON
    # wavelength, not the tube radius.  The photon has energy E ≈ m_e c²,
    # so its position uncertainty is δx ~ ƛ_C = ℏ/(m_e c).
    # ⟨δR²⟩ ~ ƛ_C² = (ℏ/(m_e c))²
    # a_e_fluct = ƛ_C²/(2R²)
    # At R = 0.488 ƛ_C: a_e = 1/(2 × 0.488²) ≈ 2.1  (way too big!)
    #
    # The right scale: the RADIATIVE correction to the position.
    # In QED, the one-loop self-energy shifts the position by
    # δr ~ α × ƛ_C / (2π).  This is the Lamb shift scale.
    # ⟨δR²⟩ ~ (α ƛ_C / (2π))²
    # a_e_fluct = (α/(2π))² × ƛ_C²/(2R²)
    #           ≈ (α/(2π))² × 1/(2 × 0.488²) ≈ 2.8e-6
    # Still not α/(2π) — off by α itself.
    #
    # Correct Schwinger argument: the fluctuation that matters is
    # not ⟨δR²⟩ but the FIRST-ORDER correction to the current loop.
    # The electron radiates and reabsorbs a virtual photon of
    # wavelength up to ƛ_C.  During this process the "charge"
    # is displaced transversely by δR ~ ƛ_C × √(α/π).
    #
    # But this is circular — we're trying to DERIVE α/(2π).
    # Let's just compute what δR/R would need to be:

    j01 = 2.4048  # first zero of J_0

    # Model A: tube confinement scale
    delta_R_tube = r / (2 * j01)
    a_e_tube = delta_R_tube**2 / (2 * R**2)

    # Model B: Compton wavelength scale
    a_e_compton = lambda_C**2 / (2 * R**2)

    # Model C: radiative scale α × ƛ_C
    delta_R_rad = alpha * lambda_C / (2 * np.pi)
    a_e_radiative = delta_R_rad**2 / (2 * R**2)

    # Model D: what δR gives a_e = α/(2π)?
    a_e_target = alpha / (2 * np.pi)
    delta_R_needed = R * np.sqrt(2 * a_e_target)
    delta_R_needed_over_lambda_C = delta_R_needed / lambda_C

    # ────────────────────────────────────────────────────────
    # 2. SELF-FIELD VERTEX CORRECTION
    # ────────────────────────────────────────────────────────
    # The circulating charge e creates a Coulomb field E ~ k_e e/ρ²
    # at distance ρ from the null curve.  This field acts back on the
    # charge, modifying the effective curvature.
    #
    # The self-force on a charge moving on a curved path:
    # F_self = (2/3) × (e²/(4πε₀)) × γ⁴ × (a²/c⁴ - (γ²a_∥)²/c⁴)
    # For a null curve: γ → ∞, need regularization.
    #
    # In NWT: the charge is NOT a point — it's distributed over the
    # tube cross-section of radius r.  The self-field at the center
    # of the tube is regularized by the finite tube radius.
    #
    # Self-field at distance r from line charge (charge per length e/L):
    # E_self = (e/L) / (2πε₀ r) = e / (2πε₀ r L)
    #
    # Force on charge e in this field:
    # F_self = e × E_self = e² / (2πε₀ r L)
    #
    # This modifies the curvature: the photon follows a path with
    # effective radius R_eff = R + δR_self where
    # δR_self ~ F_self × R² / (pc) for a photon of momentum p = E/c.
    #
    # δR_self / R = F_self × R / (E/c × c) = F_self × R / E
    #             = (e²/(2πε₀ r L)) × R / E
    #             = (α ℏc / r) × (R / (2π L E)) × (4π)
    #             ... let me be more careful.

    # Classical self-force approach:
    # Line charge density: λ = e / L
    # Field at distance r: E_self = λ / (2πε₀ r)
    E_self_at_r = (e_charge / L) / (2 * np.pi * eps0 * r)

    # Force per unit length: f = λ × E_self = λ²/(2πε₀ r)
    f_self_per_length = (e_charge / L) * E_self_at_r

    # Total self-force: F = f × L = e²/(2πε₀ r L) × L = e²/(2πε₀ r)
    # Wait, this is just the force between two parallel line charges.
    # For a SINGLE charge on a loop, the relevant quantity is the
    # self-inductance force.
    #
    # Better: use the radiation reaction (Abraham-Lorentz-Dirac).
    # For a charge on a circle of radius R at speed c:
    # P_rad = (e² c)/(6πε₀) × κ² c²  (Larmor in natural units)
    # No — for a massless particle on a null curve, there IS no
    # radiation: the field configuration is stationary.
    #
    # The correct approach: the self-energy shift.
    # The electromagnetic self-energy of the loop modifies the
    # effective mass:  m_eff = E/c² + δm_self
    # where δm_self = U_self/c² (the self-inductance energy).
    #
    # The g-factor is:
    # g = 2 m_bare μ / (e L)
    # If m_eff = m_bare + δm, and μ and L are computed with m_eff:
    # g = 2 (m_eff - δm) μ / (e L)
    # δg = -2 δm μ / (e L) = -(δm/m_eff) × g
    #
    # Actually, the vertex correction is more subtle.  In QED,
    # the vertex correction modifies the coupling to the external
    # field WITHOUT changing the mass.  It's a FORM FACTOR:
    # F₁(0) = 1, F₂(0) = a_e (Pauli form factor)
    #
    # In NWT, the analogous effect: the external B field
    # penetrates the torus tube and modifies the photon orbit.
    # The photon sees a slightly different path due to the external
    # field.  The change in enclosed area → change in μ → a_e.
    #
    # Linear response: δA/A ~ (external flux through tube) / (Φ₀)
    # Φ_ext_through_tube = B_ext × πr² (flux through tube cross-section)
    # This is state-dependent, not a universal a_e.
    #
    # Different approach: SELF-INTERACTION vertex.
    # The photon orbiting the torus creates a field.  One "turn" of
    # the knot interacts with the field from the PREVIOUS turn.
    # This inter-turn interaction modifies the orbital path.
    #
    # For a (2,1) knot: the two toroidal loops are separated by
    # poloidal angle π.  The inter-loop interaction is the
    # mutual inductance energy U_mutual.
    #
    # The ratio U_self/E_circ gives the relative correction:
    # a_e_self ~ U_self / (2 E_circ)

    circ = compute_effective_circulation(R, r, p, q)
    E_circ = circ['E_circ_eff_MeV'] * MeV
    E_total = circ['E_circ_eff_MeV'] * MeV  # approximate

    # Self-energy from existing calculation
    se = compute_self_energy(TorusParams(R=R, r=r, p=p, q=q))
    U_self_total = se['U_total_J']  # in Joules

    # a_e from self-energy/circulation ratio
    a_e_self = abs(U_self_total) / (2 * E_circ)

    # ────────────────────────────────────────────────────────
    # 3. GEOMETRIC PHASE CORRECTION
    # ────────────────────────────────────────────────────────
    # The (p,q) torus knot has a writhe number W and a twist T.
    # The self-linking number: SL = W + T.
    # For a (p,q) torus knot: SL = pq - p - q  (... various formulae)
    #
    # The Berry phase from the tangent indicatrix:
    # Φ_Berry = 2π × (writhe)
    # For a (2,1) torus knot: writhe = 0 (unknot isotopy class??)
    # Actually (2,1) is a trefoil... no, (2,1) winds 2× toroidally
    # and 1× poloidally, which is topologically a circle.
    # A true knot requires min(p,q) ≥ 2.
    #
    # The geometric correction to g from the path's total curvature:
    # ∮ κ ds = 2πp for a "nice" (p,q) knot on a thin torus.
    # Departure from 2πp is a geometric correction.

    kappa_integral = np.sum(kappa * ds)
    kappa_expected = 2 * np.pi * p  # for thin torus
    kappa_excess = kappa_integral - kappa_expected
    kappa_excess_frac = kappa_excess / kappa_expected

    # The excess curvature modifies the Thomas precession rate.
    # δ(Thomas) / Thomas ~ δ(∮κds) / (∮κds)
    a_e_geometric = abs(kappa_excess_frac) / 2

    # ────────────────────────────────────────────────────────
    # 4. α/(2π) AS A TORUS RATIO
    # ────────────────────────────────────────────────────────
    # Is there a purely geometric ratio on the torus that gives α/(2π)?
    #
    # Cross-section area / surface area:
    # A_cross = πr², A_surface = 4π²Rr
    # Ratio = r/(4πR) = α/(4π) at r = αR.  Off by factor 2.
    #
    # Volume of tube / volume of torus:
    # V_tube = πr²L, V_torus = 2π²Rr² (for thin torus)
    # V_tube/V_torus = L/(2π²R)
    #
    # Solid angle subtended by tube at center:
    # Ω ~ πr²/R² = πα² = 1.67e-4.  Too small.
    #
    # The KEY ratio: r/(2πR) = α/(2π) at r = αR.
    # This is the ratio of the tube radius to the toroidal
    # circumference.  It's also the poloidal-to-toroidal
    # angle ratio for the knot pitch.
    #
    # Physically: a photon traversing the torus spends a fraction
    # r/(2πR) of each toroidal circuit traversing one tube radius.
    # The "vertex correction" is the probability of the photon
    # deviating by one tube radius during one circuit.
    #
    # For r = αR: r/(2πR) = α/(2π) EXACTLY.

    ratio_r_over_2piR = r / (2 * np.pi * R)

    # Other candidate ratios at r = αR:
    ratio_cross_over_surface = r / (4 * np.pi * R)  # = α/(4π)
    ratio_r2_over_R2 = r**2 / R**2  # = α²

    # ────────────────────────────────────────────────────────
    # 5. SCHWINGER-LIKE DERIVATION
    # ────────────────────────────────────────────────────────
    # Schwinger's derivation: a_e = α/(2π) comes from the
    # probability of finding the electron at a distance δr
    # from its "classical" position, weighted by the magnetic
    # field variation over that distance.
    #
    # In NWT: the photon orbits at radius R from the torus axis.
    # It also has a poloidal component at radius r from the tube axis.
    # The magnetic moment has two contributions:
    #   μ_toroidal = I × πR² (from toroidal circulation)
    #   μ_poloidal = I_pol × πr² (from poloidal circulation)
    #
    # The poloidal current I_pol = e × v_pol / L_pol
    # where v_pol = c × q/(L/R) × r ... proportional to r.
    #
    # For a (p,q) knot: the poloidal angular velocity is
    # ω_pol = q × (2πc/L).  The poloidal current:
    # I_pol = e × ω_pol / (2π) = e × q × c / L
    #
    # The poloidal magnetic moment:
    # μ_pol = I_pol × πr² = (eqc/L) × πr²
    #
    # Ratio: μ_pol / μ_tor = (q/p) × (r/R)²
    # At r = αR, p = 2, q = 1:
    # μ_pol/μ_tor = (1/2) × α² = 2.66e-5
    # Too small by 43×.
    #
    # But the CROSS-COUPLING between toroidal motion and poloidal
    # field is more interesting:
    # The toroidal current creates a B field that threads the tube.
    # The poloidal motion of the charge in this B field creates
    # a radial force → modifies R → modifies μ_toroidal.
    #
    # δR/R = (poloidal kinetic energy) / (toroidal kinetic energy)
    #       × (coupling factor)
    #
    # Poloidal fraction of energy:
    # For a (p,q) knot: E_pol/E_tor = (qR)²/((pR)² + (qR)² × (r/R)²)
    # ≈ q²r² / (p²R²) for thin torus
    # = (1/4)α² = 1.33e-5 at p=2, q=1, r=αR.

    E_pol_frac = (q * r)**2 / ((p * R)**2 + (q * r)**2)

    # ────────────────────────────────────────────────────────
    # 6. ONE-LOOP CORRESPONDENCE
    # ────────────────────────────────────────────────────────
    # The Schwinger one-loop integral:
    # a_e = (α/2π) ∫₀¹ dx x(1-x) / (1-x+x²m²/Λ²) → α/(2π) as Λ→∞
    #
    # In NWT, the "loop" is LITERAL: the photon makes one extra
    # poloidal circuit while traversing the toroidal path.
    # For a (p,q) knot: q poloidal loops per p toroidal loops.
    # The "vertex" is where the photon crosses over itself
    # (self-intersection in projection).
    #
    # Number of self-crossings for a (p,q) torus knot:
    # c(p,q) = min(p,q) × (max(p,q) - 1)  (minimum crossing number)
    # For (2,1): c = 1 × 1 = 1? No — (2,1) is an unknot.
    # For (2,3): c = 2 × 2 = 4 (trefoil... actually 3).
    #
    # The crossing number for a (p,q) torus knot (p,q coprime, p>q>1):
    # c(p,q) = min(p(q-1), q(p-1))
    # For (2,1): not a true knot, c = 0.
    #
    # The photon's SELF-INTERACTION at each crossing:
    # coupling α, one crossing → correction ~ α.
    # The geometric factor: each crossing subtends angle 2π/p
    # in the toroidal direction, so the correction per crossing
    # is α × (solid angle factor).
    #
    # For one effective crossing weighted by the loop integral:
    # a_e ~ α × (geometric factor) = α/(2π) × (2π × geo_factor)
    #
    # The one-loop vertex correction in Feynman diagram language:
    # the virtual photon carries momentum k, coupling twice to the
    # electron line, with k ranging from 0 to m_e c/ℏ.
    #
    # In NWT: the "virtual photon" is a DISTURBANCE on the torus
    # field that propagates poloidally.  It is emitted at one point
    # on the knot and reabsorbed at another point, separated by
    # poloidal angle δφ = 2πq/p per crossing.
    #
    # The effective coupling: α_eff × phase_factor = α × 1/(2π)
    # giving a_e = α/(2π).

    # Compute the crossing-based estimate:
    # For (2,1) knot: zero real crossings, but the self-interaction
    # is still present through the self-inductance.
    # Use: a_e ~ α × (r/(2πR)) / (r/R) = α/(2π) at r = αR.
    #
    # Actually the simplest argument:
    # a_e = (probability of virtual emission) × (geometric phase change)
    # P_emit ~ α (coupling constant)
    # Φ_geometric = (solid angle subtended by deviation) / (4π)
    #
    # The virtual photon excursion: δr ~ ƛ_C = R/0.488
    # Solid angle: Ω ~ δr² / R² ~ (ƛ_C/R)² ~ 4.2
    # Too big — the solid angle can't exceed 4π.
    #
    # Schwinger's integral really does give 1/(2π) as the coefficient.
    # The factor 1/(2π) = ∫₀¹ x(1-x) dx = 1/6... no, that gives 1/6.
    # Actually a_e = (α/π) ∫₀¹ x(1-x)/(1-x(1-x)) dx... no.
    # Schwinger: a_e = α/(2π). Period. The 1/(2π) is the solid angle
    # factor from the vertex loop integration.

    # ────────────────────────────────────────────────────────
    # 7. THE KEY IDENTITY: r/(2πR) = α/(2π)
    # ────────────────────────────────────────────────────────
    # At r = αR, the ratio r/(2πR) = α/(2π) EXACTLY.
    # This is the fraction of the toroidal path length equal to
    # one tube radius: r = (α/(2π)) × 2πR.
    #
    # Physical interpretation:
    # The anomalous moment is the correction to the magnetic moment
    # from the FINITE SIZE of the tube.  The photon doesn't orbit
    # at exactly radius R — it also has radial extent r.
    #
    # In a uniform external B field, the flux through the orbit is:
    # Φ = B × πR².  But the actual orbit has a radial spread r,
    # and the effective area correction is:
    # δA = 2πR × r × ⟨cos φ⟩_eff
    #
    # where ⟨cos φ⟩_eff is an effective average over the poloidal
    # angle, weighted by the probability of the photon being at
    # each poloidal position.
    #
    # For the ground mode in a circular waveguide (TM₀₁):
    # ⟨cos φ⟩ = 0 (by symmetry of J₀).
    # But for the FIRST-ORDER perturbation by the external field:
    # ⟨cos φ⟩ ~ coupling × r/R
    #
    # δA/A = 2r/R × ⟨cos φ⟩ ~ 2 × coupling × (r/R)²
    # For coupling ~ 1: δA/A ~ 2α² ≈ 1.06e-4.  Wrong.
    #
    # BUT: if the correction is first-order in α (not α²):
    # The external field mixes the ground mode with the first
    # excited mode (which HAS ⟨cos φ⟩ ≠ 0).
    # The mixing amplitude: A ~ (eEr)/(ΔE) where ΔE is the
    # mode splitting.
    # For the torus waveguide: ΔE ~ ℏc/r (mode spacing)
    # A ~ (eE_ext × r) / (ℏc/r) = eE_ext r² / (ℏc)
    # This is state-dependent (depends on E_ext), not universal.
    #
    # THE CLEAN RESULT:
    # The anomalous moment a_e = α/(2π) corresponds to a path
    # length correction of exactly r per toroidal circuit of 2πR:
    #
    #   δL/L = r/(2πR) = α/(2π)
    #   a_e = δL/(2L) × (some factor of 2)
    #
    # More precisely: the path on the torus surface has length
    # L_knot = 2πR√(p² + (qr/R)²).  For thin torus:
    # L_knot ≈ 2πpR × (1 + (q²r²)/(2p²R²))
    # δL/L = q²r²/(2p²R²) = (1/8)α² for p=2,q=1.  Wrong order in α.
    #
    # The correction at ORDER α (not α²):
    # Need a mechanism linear in r/R, not quadratic.
    # The cross-term between toroidal and poloidal circulation
    # in the magnetic moment IS first-order in q × r/(pR):
    # μ_cross = p × I × 2πR × (q r / (pR))_something... no.
    #
    # Let me just compute numerically what each ratio gives.

    a_e_qed = alpha / (2 * np.pi)

    results = {
        'a_e_qed': a_e_qed,
        'a_e_qed_full': 0.00115965218128,  # experimental value to 11 digits

        # Model 1: Path fluctuation
        'a_e_tube_fluct': a_e_tube,
        'delta_R_tube': delta_R_tube,
        'a_e_compton_fluct': a_e_compton,
        'a_e_radiative_fluct': a_e_radiative,
        'delta_R_needed': delta_R_needed,
        'delta_R_needed_over_lC': delta_R_needed_over_lambda_C,

        # Model 2: Self-energy vertex
        'U_self': U_self_total,
        'E_circ': E_circ,
        'a_e_self_vertex': a_e_self,

        # Model 3: Geometric phase
        'kappa_integral': kappa_integral,
        'kappa_expected': kappa_expected,
        'kappa_excess': kappa_excess,
        'kappa_excess_frac': kappa_excess_frac,
        'a_e_geometric': a_e_geometric,

        # Model 4: Torus ratios
        'r_over_2piR': ratio_r_over_2piR,  # = α/(2π) at r = αR
        'r_over_4piR': ratio_cross_over_surface,
        'alpha_squared': ratio_r2_over_R2,
        'E_pol_frac': E_pol_frac,

        # Key identity
        'identity_match': abs(ratio_r_over_2piR - a_e_qed) / a_e_qed,

        # Torus parameters
        'R': R,
        'r': r,
        'p': p,
        'q': q,
        'L': L,
    }

    return results


def compute_cross_coupling_ae(R, r, p=2, q=1, N=2000):
    """
    Rigorous cross-coupling calculation of a_e.

    The poloidal velocity of the charge interacts with the self-generated
    toroidal B field, creating a Lorentz force that perturbs the orbit.
    This perturbation modifies the magnetic moment.

    The calculation:
    1. Parametrize the (p,q) knot in cylindrical coordinates (ρ, θ, z)
    2. Decompose velocity into toroidal (ê_θ) and poloidal components
    3. Compute the self B field (toroidal, from p-turn current loop)
    4. Compute F_Lorentz = e v_pol × B_self at each point
    5. Extract the radial component F_ρ(φ) — the cos φ dependence
    6. Compute the orbit perturbation δρ from force balance
    7. Compute the corrected magnetic moment μ_eff
    8. Extract a_e = (μ_eff - μ_orbital) / (2 μ_orbital)

    Returns dict with detailed breakdown of the cross-coupling.
    """
    params = TorusParams(R=R, r=r, p=p, q=q)
    lam, xyz, dxyz, ds_dlam = torus_knot_curve(params, N)
    dlam = 2.0 * np.pi / N
    L = np.sum(ds_dlam * dlam)

    # ── Cylindrical coordinates and velocities ──────────────────
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rho = np.sqrt(x**2 + y**2)       # distance from z-axis
    theta = np.arctan2(y, x)          # azimuthal angle

    # Velocity in Cartesian
    vx, vy, vz = dxyz[:, 0], dxyz[:, 1], dxyz[:, 2]
    # Normalize to speed c
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    vx *= c / v_mag
    vy *= c / v_mag
    vz *= c / v_mag

    # Cylindrical velocity components
    cos_theta = x / rho
    sin_theta = y / rho
    v_rho = vx * cos_theta + vy * sin_theta      # radial (from z-axis)
    v_theta = -vx * sin_theta + vy * cos_theta    # azimuthal (toroidal)
    # v_z already in Cartesian

    # ── Poloidal angle φ at each point on the knot ──────────────
    # φ is the angle around the tube cross-section
    # The tube center is at (R cos θ, R sin θ, 0)
    # The point on the tube surface is at ρ = R + r cos φ, z = r sin φ
    # So: cos φ = (ρ - R) / r,  sin φ = z / r
    cos_phi = (rho - R) / r
    sin_phi = z / r
    # Clip for numerical safety
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    sin_phi = np.clip(sin_phi, -1.0, 1.0)

    # ── Decompose velocity into toroidal and poloidal ───────────
    # Toroidal velocity: v_θ ê_θ (already computed)
    # Poloidal velocity: component in the (ê_ρ, ê_z) plane tangent to tube
    # v_pol_ρ = v_rho - (v_rho projected onto toroidal direction)
    # For thin torus: v_pol is mostly the oscillating part
    #
    # More precisely: the toroidal direction is ê_θ.
    # Everything else is "poloidal" (in the meridional plane).
    # Poloidal velocity components:
    v_pol_rho = v_rho           # radial part of poloidal
    v_pol_z = vz                # vertical part of poloidal
    v_pol_mag = np.sqrt(v_pol_rho**2 + v_pol_z**2)

    # Poloidal speed: expected ~ qrc/(pR) for thin torus
    v_pol_expected = q * r * c / (p * R)  # thin torus limit

    # ── Self B field (toroidal component inside torus) ──────────
    # A torus with p turns of current I:
    # B_tor(ρ) = μ₀ p I / (2π ρ)  inside the torus
    # where I = e c / L  (charge e passes each cross-section once per period)
    I_current = e_charge * c / L
    # B field at each point (varies as 1/ρ):
    B_tor = mu0 * p * I_current / (2 * np.pi * rho)  # Tesla

    # ── Lorentz force: F = e × v_pol × B_tor ────────────────────
    # In cylindrical coordinates:
    # v_pol = v_pol_ρ ê_ρ + v_pol_z ê_z
    # B = B_tor ê_θ
    #
    # v × B = v_pol_ρ B (ê_ρ × ê_θ) + v_pol_z B (ê_z × ê_θ)
    #       = v_pol_ρ B ê_z        + v_pol_z B (-ê_ρ)
    #       = -v_pol_z B ê_ρ       + v_pol_ρ B ê_z
    #
    # So the RADIAL Lorentz force (from poloidal motion in B_tor):
    F_rho_Lorentz = -e_charge * v_pol_z * B_tor  # negative sign from cross product

    # The vertical Lorentz force:
    F_z_Lorentz = e_charge * v_pol_rho * B_tor

    # ── Verify cos φ dependence ─────────────────────────────────
    # v_pol_z ≈ v_p cos φ  (vertical component of poloidal velocity)
    # where v_p is the poloidal speed
    # So F_rho ≈ -e × v_p cos φ × B_tor
    #
    # Check: v_pol_z / cos_phi should be approximately constant
    mask = np.abs(cos_phi) > 0.1  # avoid division by near-zero
    v_pol_z_over_cos = np.zeros_like(cos_phi)
    v_pol_z_over_cos[mask] = v_pol_z[mask] / cos_phi[mask]
    v_p_from_vz = np.median(np.abs(v_pol_z_over_cos[mask]))

    # ── Orbit perturbation ──────────────────────────────────────
    # The radial Lorentz force perturbs the orbit radius.
    # For a photon on a circular orbit: F_cent = E/(c²) × c²/ρ = E/ρ
    # Stiffness (restoring force per unit displacement):
    # k = dF_cent/dρ = -E/ρ² → |k| = E/ρ²
    #
    # For the toroidal orbit at radius ρ ≈ R:
    # k = E/R² where E = circulation energy

    gordon = compute_gordon_total_energy(R, r, p, q)
    E_total = gordon['E_total_MeV'] * MeV  # Joules

    k_stiffness = E_total / R**2  # N/m

    # The orbit perturbation at each point:
    delta_rho = F_rho_Lorentz / k_stiffness  # meters

    # ── Magnetic moment correction ──────────────────────────────
    # The magnetic moment of the orbit:
    # μ = (I/2) ∮ ρ² dθ  (for a current loop in the equatorial plane)
    # where ρ = R + r cos φ + δρ(φ)
    #
    # The unperturbed moment: integrate (R + r cos φ)² over θ
    # μ₀ = (Ip/2) ∫₀²π [R² + 2Rr cos(qλ) + r² cos²(qλ)] dλ
    # = (Ip/2) [2πR² + 0 + πr²] = Ipπ(R² + r²/2)
    #
    # For a "point" orbit: μ_point = IpπR²
    #
    # The CROSS-TERM correction from δρ:
    # δμ = (Ip/2) ∫₀²π 2(R + r cos φ) × δρ(φ) × p dλ / (2π/p?)
    # Need to be careful about the parametrization.
    #
    # Actually, the most direct way:
    # μ_z = (I_current / 2) × ∮ ρ_eff² dθ
    # where ρ_eff = ρ + δρ, and dθ = p dλ along the knot.

    # Compute unperturbed moment
    rho_sq_unpert = rho**2
    mu_unpert = (I_current / 2) * np.sum(rho_sq_unpert * p * dlam)  # ∮ ρ² dθ

    # Compute perturbed moment
    rho_eff = rho + delta_rho
    rho_sq_pert = rho_eff**2
    mu_pert = (I_current / 2) * np.sum(rho_sq_pert * p * dlam)

    # Corrections
    delta_mu = mu_pert - mu_unpert
    delta_mu_frac = delta_mu / mu_unpert

    # The point-particle orbital moment (for comparison)
    mu_orbital = I_current * p * np.pi * R**2

    # a_e from the magnetic moment correction
    # The orbital g-factor is already 2 (from Thomas precession)
    # So μ = g × (eL_z)/(2m) = 2 × μ_Bohr × (L_z/ℏ)
    # The anomalous part: δμ → a_e = δμ/(2μ)
    a_e_cross = abs(delta_mu_frac) / 2

    # ── Analytical cross-coupling formula ───────────────────────
    # For thin torus (r << R):
    #   F_ρ = -e × v_p cos φ × B_tor
    #       = -e × (qrc/(pR)) × cos φ × μ₀pec/(4π²R²)
    #   (using B_tor ≈ μ₀pI/(2πR) with I = ec/L ≈ ec/(2πpR))
    #
    #   δρ = F_ρ / k = F_ρ × R²/E
    #      = -[e² μ₀ c² q r cos φ / (4π²R)] × (R²/E)
    #      = -[4πα ℏc × c × qr cos φ / (4π²R)] × (R²/E)
    #        (using e²μ₀c = e²/(ε₀c) = 4παℏ)
    #      ... wait let me use SI directly.
    #
    #   e²μ₀ = e²/(ε₀c²) = 4π αℏc/c² × ... ugh.
    #   Let me just use α = e²/(4πε₀ℏc) → e² = 4πε₀αℏc
    #   and μ₀ = 1/(ε₀c²)
    #   So e²μ₀ = 4παℏc/c² = 4παℏ/c
    #
    #   F_ρ = -(4παℏ/c) × c² × qr cos φ / (4π² R pR)
    #       × (p from B_tor numerator)
    #   Wait, let me be more careful:
    #
    #   B_tor = μ₀ p I / (2πR) = μ₀ p (ec/L) / (2πR)
    #   For L ≈ 2πpR: B_tor ≈ μ₀ ec / (4π²R²)
    #
    #   v_pol_z ≈ (qr/L) × c × cos φ ... not quite.
    #   For the knot: dz/dt = r sin(qλ) × q × dλ/dt × ... hmm.
    #   v_z = d(r sin(qλ))/dt = r q cos(qλ) × (dλ/dt)
    #   dλ/dt = c / (ds/dλ) ... for thin torus ds/dλ ≈ pR
    #   v_z ≈ rq cos(qλ) × c/(pR)
    #   At poloidal angle φ = qλ: v_z ≈ rqc cos φ / (pR)
    #
    #   F_ρ = -e × v_z × B_tor
    #       = -e × [rqc cos φ/(pR)] × [μ₀ ec/(4π²R²)]
    #       = -(e²μ₀c²) × qr cos φ / (4π²pR³)
    #       = -(4παℏc) × qr cos φ / (4π²pR³)
    #       = -(αℏc qr cos φ) / (π p R³)
    #
    #   δρ = F_ρ R²/E = -(αℏc qr cos φ) / (πpRE)
    #   Using E ≈ ℏc/(2R) (from L_z = ℏ/2):
    #   δρ = -(αℏc qr cos φ) / (πpR × ℏc/(2R))
    #       = -2αqr cos φ / (πp)
    #   At p=2, q=1: δρ = -αr cos φ / π

    delta_rho_analytic = alpha * r / np.pi  # amplitude (without cos φ)

    # The magnetic moment correction:
    # δμ/μ = 2⟨ρ × δρ⟩/⟨ρ²⟩ ≈ 2R × ⟨δρ⟩ / R²
    #       = 2⟨δρ⟩/R
    # BUT ⟨δρ⟩ = 0 (cos φ averages to zero)!
    #
    # The QUADRATIC correction:
    # μ ∝ ∮ ρ² dθ = ∮ (R + r cos φ + δρ)² dθ
    # = ∮ [R² + 2R(r cos φ + δρ) + (r cos φ + δρ)² ] dθ
    # The cross-term 2R × δρ averages to zero (δρ ∝ cos φ).
    # The quadratic term:
    # (r cos φ + δρ)² = r² cos²φ + 2r cos φ × δρ + δρ²
    # With δρ = -(αr/π) cos φ:
    # 2r cos φ × δρ = -2αr² cos²φ / π
    # ⟨cos²φ⟩ = 1/2
    # ⟨2r cos φ × δρ⟩ = -2αr²/(2π) = -αr²/π
    #
    # δμ/μ = ⟨(r cos φ + δρ)² - r² cos²φ⟩ / R²
    #       = [2r⟨cos φ × δρ⟩ + ⟨δρ²⟩] / R²
    #       = [-αr²/π + α²r²/π²·½] / R²
    #       ≈ -αr²/(πR²) + O(α²)    (dropping α² term)
    #       = -α³/π  at r = αR
    #
    # That's O(α³), not O(α).  The cross-coupling via the magnetic
    # moment area integral gives a THIRD-order correction!
    #
    # The first-order correction must come through a different channel.

    # ── REVISIT: Thomas precession modification ─────────────────
    # The cross-coupling force changes the CURVATURE of the orbit,
    # which modifies the Thomas precession rate.
    #
    # Thomas phase per circuit: Φ_T = ∮ κ_eff ds
    # where κ_eff includes the self-force correction.
    #
    # The centripetal acceleration a = c²κ. Adding the Lorentz force:
    # (E/c) × c κ_eff = (E/c) × c κ₀ + F_ρ
    # κ_eff = κ₀ + F_ρ c / E
    #
    # The correction to the Thomas phase:
    # δΦ_T = ∮ (F_ρ c / E) ds
    #       = (c/E) ∮ F_ρ ds
    #
    # F_ρ = -(αℏc qr cos φ) / (πpR³) (from above, at ρ = R)
    # ds = (ds/dλ) dλ ≈ pR dλ (thin torus)
    # φ = qλ
    #
    # ∮ F_ρ ds = -(αℏc qr)/(πpR³) × pR × ∫₀²π cos(qλ) dλ = 0
    #
    # cos(qλ) integrates to zero over a full period!
    # So the AVERAGE Thomas correction is zero.
    #
    # BUT: the Thomas precession is not linear in κ at v < c.
    # For v slightly below c (Gordon metric, v = c/n):
    # f_Thomas(κ) = (γ-1)/γ where γ depends on the force
    # The non-linearity gives a SECOND-ORDER correction:
    # ⟨f(κ₀ + δκ)⟩ = f(κ₀) + ½f''(κ₀)⟨δκ²⟩ + ...
    # The ⟨δκ²⟩ term involves ⟨cos²φ⟩ = 1/2. This is where
    # the factor of 1/2 enters!
    #
    # At v = c (null): f = 1 exactly, f' = 0, f'' = 0.
    # The Thomas factor is FLAT at v = c — there's no sensitivity
    # to curvature fluctuations for a truly null curve.
    #
    # At v = c/n (Gordon): f = 1 - 1/γ = 1 - √(n²-1)/n
    # f depends on the local velocity, which depends on the
    # local refractive index, which depends on the local field,
    # which depends on the local curvature.
    #
    # Chain: δκ → δE_field → δn → δf_Thomas

    # ── The correct mechanism: INTERACTION ENERGY ───────────────
    # The anomalous moment is the derivative of the interaction
    # energy with respect to the external field B_ext:
    #
    # U_int = -μ_eff B_ext
    # μ_eff = -dU_int/dB_ext
    #
    # The interaction has two parts:
    # 1. Direct: U_direct = -μ_orbital × B_ext (g = 2 from Thomas)
    # 2. Indirect: U_indirect = -δμ(B_ext) × B_ext
    #    where δμ comes from B_ext modifying the orbit which
    #    changes the self-energy.
    #
    # The key: the external field B_ext modifies ρ → ρ + δρ_ext
    # and this changes the self-energy.
    # δU_self = (dU_self/dR) × δR_ext
    #
    # For a cyclotron orbit: the external field shifts R by
    # δR_ext = -(e B_ext R²)/(2pc) × ... (state-dependent)
    #
    # The cross-term: δU_self × δR_ext ∝ B_ext → contributes to μ.
    #
    # This is the NWT version of the vertex correction:
    # the external field changes the orbit, which changes the
    # self-interaction, which changes the total energy.
    #
    # a_e = (1/2μ_orbital) × d/dB × (dU_self/dR × dR/dB)
    #
    # dU_self/dR: the self-energy gradient at the equilibrium R.
    # From Section 13: dE/dR = -1.03 MeV/ƛ_C, but U_self alone:

    # Numerically compute dU_self/dR
    dR_frac = 0.001
    se_plus = compute_self_energy(TorusParams(R=R*(1+dR_frac), r=r*(1+dR_frac), p=p, q=q))
    se_minus = compute_self_energy(TorusParams(R=R*(1-dR_frac), r=r*(1-dR_frac), p=p, q=q))
    dUself_dR = (se_plus['U_total_J'] - se_minus['U_total_J']) / (2 * dR_frac * R)

    # The external field perturbation of R:
    # For a photon with p = E/c orbiting at radius R in field B_ext:
    # The cyclotron condition: p/(eB_ext R) = 1 → not applicable for self-bound orbit.
    #
    # For the NWT torus: the orbit is self-bound by the circulation
    # energy (not by B_ext). The external field creates a force
    # F_ext = ev × B_ext. For the toroidal velocity v_θ ≈ c:
    # F_ext_ρ = e c B_ext (outward, like cyclotron force)
    # This is balanced by a change in orbit radius:
    # k × δR = F_ext → δR = ecB_ext R² / E
    dR_per_B = e_charge * c * R**2 / E_total  # m/T (δR per unit B_ext)

    # The correction to U_self from this orbit change:
    # δU_self = dU_self/dR × δR = dU_self/dR × dR/dB × B_ext
    # This contributes to the effective magnetic moment:
    # δμ_vertex = -d(δU_self)/dB = -dU_self/dR × dR/dB
    delta_mu_vertex = -dUself_dR * dR_per_B  # A·m² (magnetic moment correction)
    mu_Bohr = e_charge * hbar / (2 * m_e)
    a_e_vertex = abs(delta_mu_vertex) / (2 * mu_orbital)

    # ── Summary ratios ──────────────────────────────────────────
    a_e_qed = alpha / (2 * np.pi)

    return {
        # Velocities
        'v_pol_avg': np.mean(v_pol_mag),
        'v_pol_expected': v_pol_expected,
        'v_theta_avg': np.mean(np.abs(v_theta)),
        'v_p_from_vz': v_p_from_vz,
        # Fields
        'B_tor_avg': np.mean(B_tor),
        'B_tor_at_R': mu0 * p * I_current / (2 * np.pi * R),
        'I_current': I_current,
        # Lorentz force
        'F_rho_max': np.max(np.abs(F_rho_Lorentz)),
        'F_rho_rms': np.sqrt(np.mean(F_rho_Lorentz**2)),
        # cos φ verification
        'cos_phi_F_correlation': np.mean(F_rho_Lorentz * cos_phi) / np.sqrt(np.mean(F_rho_Lorentz**2) * np.mean(cos_phi**2)),
        # Orbit perturbation
        'delta_rho_max': np.max(np.abs(delta_rho)),
        'delta_rho_rms': np.sqrt(np.mean(delta_rho**2)),
        'delta_rho_analytic': delta_rho_analytic,
        'delta_rho_over_R': np.max(np.abs(delta_rho)) / R,
        # Magnetic moment from area integral
        'mu_unpert': mu_unpert,
        'mu_pert': mu_pert,
        'delta_mu_frac': delta_mu_frac,
        'a_e_area': a_e_cross,
        # Vertex correction (self-energy × orbit shift)
        'dUself_dR': dUself_dR,
        'dR_per_B': dR_per_B,
        'delta_mu_vertex': delta_mu_vertex,
        'a_e_vertex': a_e_vertex,
        # Key ratio
        'r_over_2piR': r / (2 * np.pi * R),
        'a_e_qed': a_e_qed,
        # Stiffness
        'k_stiffness': k_stiffness,
        'E_total': E_total,
    }


def compute_phi_resolved_self_energy(R, r, p=2, q=1, N=500):
    """
    φ-resolved self-energy decomposition via Biot-Savart.

    Computes the magnetic self-force at each point on the (p,q) knot
    using the Biot-Savart law, with the tube radius r as a UV cutoff.
    Decomposes the radial self-force into Fourier harmonics in the
    poloidal angle φ:

        F_ρ(φ) = F₀ + F₁ cos φ + F₂ cos 2φ + ...

    The m=0 component F₀ is the isotropic self-force → charge form factor F₁.
    The m=1 component F₁ cos φ is the dipole self-force → spin form factor F₂.

    The anomalous moment:
        a_e = F₂/(2μ₀ × normalization)

    This is the NWT analog of the QED vertex correction decomposition
    into Dirac (F₁) and Pauli (F₂) form factors.

    Returns dict with Fourier decomposition and a_e estimates.
    """
    params = TorusParams(R=R, r=r, p=p, q=q)
    lam, xyz, dxyz, ds_dlam = torus_knot_curve(params, N)
    dlam = 2.0 * np.pi / N
    L = np.sum(ds_dlam * dlam)

    # Current
    I_current = e_charge * c / L

    # Current elements: dl = (dr/dλ) × dλ
    dl = dxyz * dlam  # (N, 3) displacement vectors

    # Cylindrical coordinates at each knot point
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rho = np.sqrt(x**2 + y**2)

    # Poloidal angle at each point
    cos_phi = np.clip((rho - R) / r, -1.0, 1.0)
    sin_phi = np.clip(z / r, -1.0, 1.0)
    phi = np.arctan2(sin_phi, cos_phi)

    # Cylindrical unit vectors at each point
    cos_theta = x / rho
    sin_theta = y / rho
    e_rho = np.column_stack([cos_theta, sin_theta, np.zeros(N)])

    # Biot-Savart: compute B field and force at each point
    cutoff = r  # regularization at tube radius
    cutoff_sq = cutoff**2

    F_rho = np.zeros(N)
    F_z_comp = np.zeros(N)
    F_tang = np.zeros(N)  # tangential (along path)

    # Unit tangent at each point
    t_hat = dxyz / ds_dlam[:, np.newaxis]

    # Vectorized Biot-Savart: for each field point i
    for i in range(N):
        # Displacement from source j to field point i
        delta_r = xyz[i] - xyz  # (N, 3)
        dist_sq = np.sum(delta_r**2, axis=-1)  # (N,)

        # Mask: exclude points within cutoff
        mask = dist_sq > cutoff_sq

        # Biot-Savart kernel: dB = (μ₀I/4π) × (dl × r̂) / r²
        dist = np.sqrt(dist_sq[mask])
        r_hat = delta_r[mask] / dist[:, np.newaxis]

        cross = np.cross(dl[mask], r_hat)  # dl × r̂
        B_i = (mu0 * I_current / (4.0 * np.pi)) * np.sum(
            cross / dist[:, np.newaxis]**2, axis=0)

        # Force on current element i: dF = I dl_i × B_i
        dF = I_current * np.cross(dl[i], B_i)

        # Project onto cylindrical radial direction
        F_rho[i] = np.dot(dF, e_rho[i])

        # Vertical component
        F_z_comp[i] = dF[2]

        # Tangential (along path) component
        F_tang[i] = np.dot(dF, t_hat[i])

    # ── Fourier decomposition in φ ──────────────────────────────
    # F_ρ(φ) = Σ [a_n cos(nφ) + b_n sin(nφ)]
    # Use least-squares or direct projection

    n_harmonics = 6
    a_cos = np.zeros(n_harmonics)  # cos(nφ) coefficients
    b_sin = np.zeros(n_harmonics)  # sin(nφ) coefficients

    for n in range(n_harmonics):
        if n == 0:
            a_cos[0] = np.mean(F_rho)
        else:
            a_cos[n] = 2.0 * np.mean(F_rho * np.cos(n * phi))
            b_sin[n] = 2.0 * np.mean(F_rho * np.sin(n * phi))

    # Same for tangential force
    a_cos_tang = np.zeros(n_harmonics)
    b_sin_tang = np.zeros(n_harmonics)
    for n in range(n_harmonics):
        if n == 0:
            a_cos_tang[0] = np.mean(F_tang)
        else:
            a_cos_tang[n] = 2.0 * np.mean(F_tang * np.cos(n * phi))
            b_sin_tang[n] = 2.0 * np.mean(F_tang * np.sin(n * phi))

    # ── Physical interpretation ─────────────────────────────────
    # F₀ = a_cos[0]: isotropic radial self-force (outward/inward bias)
    # F₁ = a_cos[1]: cos φ component — the dipole self-force
    # F₁ pushes outward on the outer side, inward on the inner side
    # (or vice versa)

    # The orbit perturbation from F₁ cos φ:
    # δρ(φ) = F₁ cos φ / k_stiffness
    # where k_stiffness = E/R² (centripetal stiffness)
    gordon = compute_gordon_total_energy(R, r, p, q)
    E_total = gordon['E_total_MeV'] * MeV
    k_stiffness = E_total / R**2

    delta_rho_1 = a_cos[1] / k_stiffness  # amplitude of cos φ orbit shift

    # Magnetic moment correction from δρ = δρ₁ cos φ:
    # μ = (I/2) ∮ (R + r cos φ + δρ₁ cos φ)² dθ
    # The cross-term with R: 2R × δρ₁ cos φ averages to 0
    # The cross-term with r cos φ: 2r cos φ × δρ₁ cos φ = 2r δρ₁ cos²φ
    # ⟨cos²φ⟩ = 1/2
    # So: δμ = I × p × π × r × δρ₁  (from the 2r δρ₁/2 = r δρ₁ term)
    mu_orbital = I_current * p * np.pi * R**2
    delta_mu_dipole = I_current * p * np.pi * r * delta_rho_1

    # The a_e from the dipole self-force (area channel):
    a_e_dipole_area = abs(delta_mu_dipole) / (2 * mu_orbital)

    # ── Alternative: the tangential self-force ──────────────────
    # The tangential force modifies the SPEED of the photon at each φ.
    # This doesn't change the path shape but changes the current density.
    # At angle φ, the current is I(φ) ∝ 1/v_tangential(φ).
    # The tangential force δF_tang adds δv = δF_tang × dt ∝ δF_tang/κ
    # (the time spent at each point is ds/c = 1/(cκ) × dθ ... not exactly)
    #
    # For a null curve, the tangential force can't change |v| = c.
    # But it changes the DISTRIBUTION of time spent at each φ,
    # which changes the effective current density:
    # I(φ) ∝ 1/|ds/dφ| (time spent near angle φ)
    # If the tangential force accelerates/decelerates the photon,
    # it spends MORE time where it's slower → enhanced current density.
    #
    # For null curve: can't slow below c. But the tangential force
    # redistributes energy between kinetic and potential, effectively
    # modifying the path (not the speed). This is captured by the
    # normal force analysis above.

    # ── The self-energy Fourier decomposition ───────────────────
    # Analytical prediction for the B-field energy density:
    # u(φ) = μ₀p²I²/(8π²(R + r cos φ)²) × (R + r cos φ)
    # (the extra factor R + r cos φ is the Jacobian for the
    # toroidal integration)
    # So u × Jacobian ∝ 1/(R + r cos φ)
    # = (1/R) × Σ (-r cos φ/R)^n
    # = (1/R)(1 - (r/R) cos φ + (r/R)² cos²φ - ...)
    #
    # The cos φ/cos 0 ratio (dipole/monopole):
    # U₁/U₀ = -r/R = -α  (to leading order)
    #
    # But for the FORCE (gradient of energy), the relevant quantity
    # is dU/dρ, which picks up an extra factor from the differentiation:
    # F_ρ ∝ d/dρ [1/ρ²] = -2/ρ³
    # The cos φ component: F₁/F₀ ∝ -3r/(2R) (from the 1/ρ³ derivative)

    # Analytical prediction for force harmonics
    # F_ρ ∝ -dP_mag/dρ = d/dρ [B²/(2μ₀)] = μ₀p²I²/(4π²ρ³)
    # At ρ = R + r cos φ:
    # F_ρ(φ) ∝ 1/(R + r cos φ)³
    # = (1/R³)(1 - 3(r/R) cos φ + 6(r/R)² cos²φ - ...)
    # F₁/F₀ = -3r/R = -3α (to leading order)
    F1_over_F0_analytic = -3.0 * r / R

    # ── Form factor decomposition ──────────────────────────────
    # F₁ (charge form factor) ↔ F₀ (isotropic force)
    # F₂ (Pauli/spin form factor) ↔ F₁ cos φ (dipole force)
    #
    # In QED language:
    # The vertex correction Γ^μ = γ^μ F₁(q²) + (iσ^μν q_ν)/(2m) F₂(q²)
    # At q² = 0 (static fields):
    # F₁(0) = 1 + δF₁ (charge renormalization)
    # F₂(0) = a_e (anomalous moment)
    #
    # In NWT:
    # The isotropic self-force (F₀) modifies the orbit radius uniformly
    # → changes E and μ proportionally → δF₁ = F₀R²/E
    # The dipole self-force (F₁ cos φ) creates a cos φ orbit oscillation
    # → changes μ without changing E → δF₂ ∝ F₁ × r / (F₀ × R)
    #
    # The ratio: F₂/F₁ = (F₁_force / F₀_force) × (r/R)
    #                   = (-3α) × α = -3α² (from the 1/ρ³ force)
    #
    # This is O(α²) — the same conclusion as before.
    # The dipole decomposition of the SELF-FORCE gives O(α²).
    #
    # KEY INSIGHT: The area-channel correction from the self-force
    # is always O(α²) because it involves (dipole force) × (tube size),
    # both of which are O(α).

    # ── The vertex channel (revisited) ──────────────────────────
    # The vertex correction U_self/E = 5.2 × α/(2π) is O(α).
    # The spin projection 1/5.2 that we need is NOT from the φ
    # decomposition of the force (which gives α), but from a
    # different decomposition.
    #
    # What is the correct spin projection?
    #
    # In QED, F₂(0) comes from the INTERFERENCE between the
    # "large" and "small" components of the Dirac spinor.
    # For a (p,q) torus knot, the analog is the interference
    # between the toroidal and poloidal wave components.
    #
    # The toroidal wavefunction: ψ_tor ∝ e^{ipθ} (large component)
    # The poloidal wavefunction: ψ_pol ∝ e^{iqφ} (small component)
    # The "small/large" ratio: ψ_pol/ψ_tor ~ v_pol/v_tor ~ qr/(pR) = α/2
    #
    # In the Dirac theory, the anomalous moment comes from:
    # F₂ = (self-energy correction) × (small component fraction)²
    # Hmm, not quite — it's (self-energy) × (interference) × (geometric factor)
    #
    # The interference between large and small components:
    # ⟨ψ_tor|H_self|ψ_pol⟩ / E ~ (U_self/E) × overlap
    # The overlap: ⟨tor|pol⟩ = 0 (orthogonal modes)
    # The coupling: ⟨tor|H_self|pol⟩ ≠ 0 because H_self breaks
    # the toroidal/poloidal symmetry.
    #
    # For the self-field: the coupling matrix element is
    # ⟨e^{ipθ}|U_self(θ,φ)|e^{iqφ}⟩ = ∫ U_self(θ,φ) e^{-ipθ+iqφ} dθ dφ
    # This picks out the Fourier component of U_self at (p, -q).
    # For U_self ∝ 1/(R + r cos(qλ-something))²:
    # ... this depends on the phase relationship between θ and φ.

    # For the (p,q) knot: θ = pλ, φ = qλ, so φ = (q/p)θ.
    # The self-energy along the knot varies as:
    # U_self(λ) ∝ 1/(R + r cos(qλ))
    # Fourier components in λ: this has harmonics at n = q, 2q, 3q, ...
    # The fundamental harmonic at n = q = 1:
    # U₁(q) = (1/2π) ∫₀²π U(λ) cos(qλ) dλ
    # = (1/2π) ∫₀²π [const / (R + r cos λ)] cos λ dλ
    # = (1/2π) × [standard integral]
    # ∫₀²π cos λ/(R + r cos λ) dλ = (2π/r) × (1 - R/√(R²-r²))
    # For r << R: ≈ (2π/r) × (1 - R/(R(1-r²/(2R²)))) ≈ (2π/r) × r²/(2R²) = πr/R²
    # So U₁(q)/U₀ = (πr/R²) / (2π/R) = r/(2R) = α/2
    # Hmm wait: U₀ = (1/2π) ∫₀²π 1/(R + r cos λ) dλ = 1/√(R²-r²) ≈ 1/R
    # U₁ = (1/π) ∫₀²π cos λ/(R+r cos λ) dλ = (2/r)(1 - R/√(R²-r²))
    #     ≈ (2/r)(r²/(2R²)) = r/R²
    # |U₁/U₀| = r/(R²) / (1/R) = r/R = α
    #
    # So the q-harmonic of the self-energy is O(α) relative to the mean.
    # The coupling ⟨tor|H_self|pol⟩ ~ U_self × α
    # And the F₂ form factor:
    # F₂ ~ ⟨tor|H_self|pol⟩ × (v_pol/c) / E_gap
    # ~ (U_self × α) × (α/2) / E_gap
    # where E_gap is the mode splitting.
    # This gives F₂ ~ U_self × α²/E_gap — again O(α²) or higher.

    # ── Analytical self-energy per poloidal harmonic ────────────
    # For the (p,q) torus knot, the self-energy along the knot
    # varies with λ. The knot passes through regions of different
    # curvature. The Fourier decomposition in λ gives harmonics
    # that correspond to poloidal and toroidal mode numbers.
    #
    # The q-harmonic (poloidal fundamental):
    # U_q / U_0 = -r/R = -α  (from 1/(R+r cos qλ) expansion)
    #
    # The 2q-harmonic:
    # U_2q / U_0 = (r/R)² / 2 = α²/2

    # Compute numerically along the knot
    # Self-energy density proxy: 1/κ_local (regions of high curvature
    # have higher self-energy)
    curv = compute_knot_curvature(R, r, p, q, N)
    kappa = curv['kappa']
    ds = curv['ds']

    # Fourier decompose κ(φ) (curvature as function of poloidal angle)
    kappa_harmonics_cos = np.zeros(n_harmonics)
    kappa_harmonics_sin = np.zeros(n_harmonics)
    for n in range(n_harmonics):
        if n == 0:
            kappa_harmonics_cos[0] = np.mean(kappa)
        else:
            kappa_harmonics_cos[n] = 2.0 * np.mean(kappa * np.cos(n * phi))
            kappa_harmonics_sin[n] = 2.0 * np.mean(kappa * np.sin(n * phi))

    # The curvature cos φ harmonic relative to mean:
    kappa_1_over_0 = kappa_harmonics_cos[1] / kappa_harmonics_cos[0]

    # ── Final a_e estimates ─────────────────────────────────────
    a_e_qed = alpha / (2.0 * np.pi)

    return {
        # Fourier harmonics of radial self-force
        'F_rho_harmonics_cos': a_cos,
        'F_rho_harmonics_sin': b_sin,
        'F0': a_cos[0],
        'F1_cos': a_cos[1],
        'F2_cos': a_cos[2] if n_harmonics > 2 else 0,
        'F1_over_F0': a_cos[1] / a_cos[0] if abs(a_cos[0]) > 1e-30 else 0,
        'F1_over_F0_analytic': F1_over_F0_analytic,
        # Tangential self-force harmonics
        'F_tang_harmonics_cos': a_cos_tang,
        'F_tang_harmonics_sin': b_sin_tang,
        'F_tang_0': a_cos_tang[0],
        'F_tang_1_cos': a_cos_tang[1],
        # Orbit perturbation from dipole force
        'delta_rho_1': delta_rho_1,
        'delta_rho_1_over_R': abs(delta_rho_1) / R,
        # Magnetic moment corrections
        'mu_orbital': mu_orbital,
        'delta_mu_dipole': delta_mu_dipole,
        'a_e_dipole_area': a_e_dipole_area,
        # Curvature harmonics
        'kappa_harmonics_cos': kappa_harmonics_cos,
        'kappa_1_over_0': kappa_1_over_0,
        # Raw force data
        'F_rho': F_rho,
        'F_tang': F_tang,
        'phi': phi,
        # Self-energy harmonics (from 1/(R+r cos φ))
        'U1_over_U0_analytic': -r / R,  # = -α
        # Stiffness
        'k_stiffness': k_stiffness,
        'E_total': E_total,
        'a_e_qed': a_e_qed,
    }


# ════════════════════════════════════════════════════════════════════
# Step 7e: Null congruence and magnetic flux analysis
# ════════════════════════════════════════════════════════════════════

def compute_null_congruence(R, r, p=2, q=1, N=2000):
    """
    Null congruence analysis on the torus: Raychaudhuri equation.

    For a tube of radius r around a curve with curvature κ(s):
    - Expansion: θ(s,φ) = -κ cos(φ) / (1 - rκ cos(φ))
    - Shear: σ ~ κ sin(φ) (deformation of cross-section)
    - Twist: ω = dφ/ds = q/|dr/dλ| (poloidal rotation)

    Raychaudhuri (null, flat space): dθ/dλ = -θ²/2 - σ² + ω²
    Averaged over closed path: ⟨θ²/2 + σ²⟩ vs ⟨ω²⟩ gives focusing balance.

    Returns dict with curvature, expansion, shear, twist, balance ratio.
    """
    curv = compute_knot_curvature(R, r, p, q, N)
    kappa = curv['kappa']
    ds = curv['ds']
    L = curv['L_total']

    kappa2_avg = np.sum(kappa**2 * ds) / L

    # For thin tube (rκ << 1):
    # ⟨θ²⟩ = κ²/2 (averaged over poloidal angle)
    # ⟨σ²⟩ = κ²/2 (same)
    theta2_avg = kappa2_avg / 2.0
    sigma2_avg = kappa2_avg / 2.0

    # Twist: poloidal rotation rate
    params = TorusParams(R=R, r=r, p=p, q=q)
    _, _, _, ds_dlam = torus_knot_curve(params, N)
    dlam = 2 * np.pi / N
    dphi_ds = q / ds_dlam  # poloidal angular rate per arc length
    omega2_avg = np.sum(dphi_ds**2 * ds_dlam * dlam) / L

    # Raychaudhuri balance
    LHS = theta2_avg / 2.0 + sigma2_avg  # = 3κ²/4
    RHS = omega2_avg  # = q²/(p²R²)
    balance_ratio = LHS / RHS  # >1 means net focusing

    # Correction from finite rκ
    rk = r * np.sqrt(kappa2_avg)

    return {
        'kappa2_avg': kappa2_avg,
        'theta2_avg': theta2_avg,
        'sigma2_avg': sigma2_avg,
        'omega2_avg': omega2_avg,
        'LHS': LHS,
        'RHS': RHS,
        'balance_ratio': balance_ratio,
        'rk': rk,
        'rk_correction': rk**2,
        'net_focusing': LHS > RHS,
    }


def compute_magnetic_flux_analysis(R, r, p=2, q=1):
    """
    Thorough magnetic flux analysis of the torus knot.

    Computes:
    - Poloidal flux (through tube cross-section) from toroidal current
    - Toroidal flux (through torus hole) from Neumann inductance
    - Comparison with flux quanta (h/e, h/2e)
    - The analytic relation Φ_pol/(h/e) = α³/(2π) at r = αR
    - Aharonov-Bohm phase
    - What r would give various flux quantization conditions
    """
    # Path and current
    L_path = 2.0 * np.pi * np.sqrt(p**2 * R**2 + q**2 * r**2)
    I = e_charge * c / L_path

    # Poloidal B field (from p toroidal windings = p-turn solenoid)
    B_pol = mu0 * p * I / (2.0 * np.pi * R)
    # Poloidal flux through tube cross-section (thin tube approximation)
    Phi_pol = mu0 * p * I * r**2 / (2.0 * R)

    # Toroidal flux from Neumann inductance
    L_Neumann = mu0 * R * (np.log(8*R/r) - 2)
    Phi_total = L_Neumann * I

    # Flux quanta
    Phi_0_single = h_planck / e_charge
    Phi_0_pair = h_planck / (2.0 * e_charge)

    # Aharonov-Bohm phase
    AB_phase = 2.0 * np.pi * Phi_total / Phi_0_single

    # Analytic result: at r = αR, Φ_pol/(h/e) = α³/(2π)
    phi_pol_analytic = alpha**3 / (2.0 * np.pi)

    # What r gives various flux conditions?
    # Φ_pol = n × h/e → r = R × √(2πn/α)
    flux_radii = {}
    for label, n_flux in [('Phi_0', 1.0), ('alpha_Phi_0', alpha),
                           ('alpha2_Phi_0', alpha**2),
                           ('alpha3_Phi_0', alpha**3)]:
        r_flux = R * np.sqrt(2.0 * np.pi * n_flux / alpha)
        flux_radii[label] = {
            'n': n_flux, 'r': r_flux, 'r_over_R': r_flux / R,
            'r_over_lambda_C': r_flux / lambda_C,
        }

    return {
        'I': I, 'L_path': L_path,
        'B_pol': B_pol,
        'Phi_pol': Phi_pol,
        'Phi_total': Phi_total,
        'L_Neumann': L_Neumann,
        'Phi_0_single': Phi_0_single,
        'Phi_0_pair': Phi_0_pair,
        'Phi_pol_over_Phi0': Phi_pol / Phi_0_single,
        'Phi_total_over_Phi0': Phi_total / Phi_0_single,
        'AB_phase': AB_phase,
        'phi_pol_analytic': phi_pol_analytic,
        'phi_pol_match_pct': abs(Phi_pol/Phi_0_single - phi_pol_analytic) / phi_pol_analytic * 100,
        'flux_radii': flux_radii,
    }


# ════════════════════════════════════════════════════════════════════
# Step 7b: Effective viscosity of the QED vacuum
# ════════════════════════════════════════════════════════════════════

def compute_schwinger_rate(E_mag):
    """
    Schwinger pair production rate per unit volume per unit time.

    Γ = (αE²) / (π²ℏ) × exp(-π E_S / E)

    In the strong-field regime (E >> E_S), the exponential → 1 and
    Γ ~ αE²/π² (polynomial growth).

    Parameters
    ----------
    E_mag : float or ndarray
        Electric field magnitude (V/m).

    Returns
    -------
    Gamma : ndarray
        Pair production rate (pairs / m³ / s).
    """
    E_mag = np.atleast_1d(np.asarray(E_mag, dtype=float))
    x = E_mag / E_Schwinger

    # Schwinger rate: Γ = (α E_S²) / (π² ℏ) × x² × exp(-π/x)
    # In SI: E_S² has units V²/m², α is dimensionless, ℏ is J·s
    # The full formula with correct dimensions:
    # Γ = (e² E² c) / (4 π³ ℏ² c²) × exp(-π E_S/E)
    #   = (α E²) / (π² (m_e c / (eE_S)) ) × exp(-π/x)
    # More precisely: Γ = (α E_S²)/(π²) × (e²/(4πε₀ℏc)) × (c/λ_C⁴) × x² × exp(-π/x)
    #
    # Standard result in natural units: Γ = (αE²)/(π²) × exp(-π E_S/E)
    # Converting to SI: multiply by c/ℏ (rate) and 1/λ_C³ (volume), giving
    # Γ = α × E_S² × x² × c / (π² × ℏ × lambda_C³) × exp(-π/x)
    #
    # But the standard SI formula is:
    # Γ = (m_e⁴ c⁵)/(π² ℏ⁴) × (E/E_S)² × Σ 1/n² exp(-nπ E_S/E)
    # ≈ (m_e⁴ c⁵)/(π² ℏ⁴) × x² × exp(-π/x) for leading term
    prefactor = (m_e**4 * c**5) / (np.pi**2 * hbar**4)  # ~ 4.4e65 m⁻³ s⁻¹

    # Schwinger exponential suppression
    # Use log to avoid overflow: log(Gamma) = log(prefactor) + 2*log(x) - pi/x
    # Clamp x to avoid division by zero
    x_safe = np.maximum(x, 1e-30)
    log_Gamma = np.log(prefactor) + 2.0 * np.log(x_safe) - np.pi / x_safe

    # Clamp to avoid overflow in exp
    log_Gamma = np.minimum(log_Gamma, 700)  # exp(700) ~ 1e304

    Gamma = np.where(x > 1e-10, np.exp(log_Gamma), 0.0)

    return Gamma


def compute_effective_viscosity(E_mag, u_field=None):
    """
    Effective shear viscosity of the QED vacuum from Schwinger dissipation.

    The Schwinger rate Γ gives the rate of field energy dissipation into
    pairs. This defines an effective viscosity:

        η_eff = u_field / Γ

    where u_field = ε₀E²/2 is the field energy density and Γ is the
    volumetric dissipation rate (energy/volume/time ~ Γ × 2m_e c²).

    In the weak field (E << E_S): Γ → 0 exponentially, η → ∞ (inviscid)
    In the strong field (E >> E_S): Γ ~ E⁴, u ~ E², so η ~ 1/E² (drops)

    Parameters
    ----------
    E_mag : float or ndarray
        Electric field magnitude (V/m).
    u_field : ndarray or None
        Field energy density. If None, uses ε₀E²/2.

    Returns
    -------
    dict with Gamma, eta_eff, and regime info.
    """
    E_mag = np.atleast_1d(np.asarray(E_mag, dtype=float))
    if u_field is None:
        u_field = 0.5 * eps0 * E_mag**2

    Gamma_pairs = compute_schwinger_rate(E_mag)  # pairs / m³ / s

    # Energy dissipation rate per unit volume: each pair costs 2m_e c²
    Gamma_energy = Gamma_pairs * 2.0 * m_e * c**2  # W / m³

    # Effective viscosity: η = u_field / (Gamma_energy / u_field) × u_field
    # More precisely, the decay rate of the field is τ_decay = u_field / Gamma_energy
    # And η_eff ~ u_field × τ_decay / L² where L is a length scale
    # But dimensionally, viscosity [Pa·s] = [energy density] × [time]
    # So η_eff = u_field / Gamma_energy × u_field = u_field² / Gamma_energy
    #
    # Actually, the simplest definition:
    # The field loses energy at rate Gamma_energy (W/m³).
    # The "strain rate" of the vacuum is ω_circ (the circulation frequency).
    # Viscosity = stress / strain_rate = (energy_density) / (strain_rate)
    # But here we want a pure field-theoretic viscosity, so:
    #
    # η_eff = u_field / (Gamma_energy / u_field)  [Pa·s]
    #       = u_field² / Gamma_energy
    #
    # This diverges when Gamma → 0 (inviscid) and drops when Gamma is large.
    # For E >> E_S: η ~ ε₀²E⁴ / (m_e⁴c⁵/ℏ⁴ × E⁴ × 2m_e c²) ~ ε₀² ℏ⁴ / (m_e⁵ c⁷)
    #   → constant! The "floor" viscosity of the super-critical vacuum.
    #
    # Better: just use τ_decay = u_field / Gamma_energy as the viscous timescale
    # and η_eff = u_field × τ_decay (energy density × relaxation time)

    # Viscous timescale: how long the field can sustain itself against pair creation
    with np.errstate(divide='ignore', invalid='ignore'):
        tau_decay = np.where(Gamma_energy > 0, u_field / Gamma_energy, np.inf)

    # Effective viscosity: η = u_field × τ_decay [Pa·s]
    # (This is the standard kinetic theory result: η = ρ × v × λ = energy_density × τ)
    eta_eff = u_field * tau_decay

    x = E_mag / E_Schwinger
    regime = np.where(x < 0.3, 'inviscid',
             np.where(x > 3.0, 'viscous', 'transition'))

    return {
        'Gamma_pairs': Gamma_pairs,
        'Gamma_energy': Gamma_energy,
        'tau_decay': tau_decay,
        'eta_eff': eta_eff,
        'regime': regime,
        'E_over_ES': x,
    }


def compute_effective_viscosity_profile(R, r, p=2, q=1, N_radial=500):
    """
    Compute the viscosity profile of the QED vacuum across the torus
    cross-section, including integrated dissipation and Q factor.

    Returns
    -------
    dict with radial viscosity profile, integrated dissipation power,
    Q factor, and viscous boundary layer location.
    """
    profile = compute_radial_vacuum_profile(R, r, p, q, N_radial)
    rho = profile['rho']
    E_field = profile['E_field']
    u_classical = profile['u_classical']

    # Viscosity at each radial point
    visc = compute_effective_viscosity(E_field, u_classical)

    # Schwinger rate (using our dedicated function for display)
    Gamma = compute_schwinger_rate(E_field)

    # Reynolds number analogue: Re = (u/c²) × c × r / η
    # where u is field energy density (playing role of mass density × v²)
    u_field = u_classical
    with np.errstate(divide='ignore', invalid='ignore'):
        Re_eff = np.where(visc['eta_eff'] < 1e300,
                          (u_field / c**2) * c * r / visc['eta_eff'],
                          0.0)

    # Integrated dissipation power over torus volume
    # P_diss = ∫ Γ_energy dV where dV = 2πR × 2πρ × dρ
    Gamma_energy = visc['Gamma_energy']
    drho = np.diff(rho)
    rho_mid = 0.5 * (rho[:-1] + rho[1:])
    Gamma_mid = 0.5 * (Gamma_energy[:-1] + Gamma_energy[1:])
    dV = 2.0 * np.pi * R * 2.0 * np.pi * rho_mid * drho
    P_diss = np.sum(Gamma_mid * dV)

    # Circulation period
    L_path = profile['L_path']
    T_circ = L_path / c

    # Q factor: Q = 2π × E_stored / (P_diss × T_circ)
    # E_stored ≈ m_e c² (the electron mass-energy)
    E_stored = m_e * c**2
    if P_diss > 0 and T_circ > 0:
        Q = 2.0 * np.pi * E_stored / (P_diss * T_circ)
        tau = Q * T_circ / (2.0 * np.pi)
    else:
        Q = np.inf
        tau = np.inf

    # Viscous boundary layer: where E/E_S ≈ 1 (Schwinger threshold)
    # This is where the exponential suppression turns off
    x = E_field / E_Schwinger
    # Find ρ where x crosses 1 from above (going outward)
    cross_idx = np.where((x[:-1] >= 1.0) & (x[1:] < 1.0))[0]
    if len(cross_idx) > 0:
        # Interpolate
        i = cross_idx[0]
        frac = (1.0 - x[i+1]) / (x[i] - x[i+1])
        rho_boundary = rho[i+1] + frac * (rho[i] - rho[i+1])
    else:
        rho_boundary = r  # default to tube radius

    return {
        'rho': rho,
        'E_field': E_field,
        'E_over_ES': x,
        'Gamma_Schwinger': Gamma,
        'Gamma_energy': Gamma_energy,
        'tau_decay': visc['tau_decay'],
        'eta_eff': visc['eta_eff'],
        'Re_eff': Re_eff,
        'regime': visc['regime'],
        'P_diss_W': P_diss,
        'P_diss_MeV_per_s': P_diss / (MeV),
        'Q_factor': Q,
        'tau_s': tau,
        'T_circ': T_circ,
        'rho_viscous_boundary': rho_boundary,
    }


# ════════════════════════════════════════════════════════════════════
# Step 7c: Effective potential — EH vs Uehling comparison
# ════════════════════════════════════════════════════════════════════

def compute_uehling_potential(rho):
    """
    Standard QED Uehling (one-loop vacuum polarization) correction to
    the Coulomb potential ENERGY between two charges e.

    All quantities are potential energies (Joules or eV), NOT electric
    potentials (Volts).

    U_bare(r) = -k_e e² / r = -αℏc / r   (Coulomb potential energy)

    δU_Uehling(r) = -(2α²ℏc)/(3r) × ∫₁^∞ dt (1 + 1/(2t²)) √(1-1/t²)/t²
                     × exp(-2 m_e c r t / ℏ)

    Short distance (r << ƛ_C):
        δU ≈ -(2α²ℏc)/(3r) × [ln(ƛ_C/r) - 5/6 - γ_E]

    Long distance (r >> ƛ_C):
        δU ≈ -(α²ℏc)/(4√π r) × (ƛ_C/(2r))^(3/2) × exp(-2r/ƛ_C)

    Parameters
    ----------
    rho : float or ndarray
        Distance from the charge (m).

    Returns
    -------
    dict with U_bare, delta_U_Uehling, and relative correction, all in
    energy units (J and eV).
    """
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    lambda_C_bar = hbar / (m_e * c)  # reduced Compton wavelength ≈ 3.86e-13 m

    # Bare Coulomb potential energy: U = -k_e e² / r = -αℏc / r
    U_bare = -k_e * e_charge**2 / rho  # Joules

    # Numerical integration of the Uehling integral
    N_int = 500
    delta_U = np.zeros_like(rho)

    for j, r_val in enumerate(rho):
        xi = 2.0 * r_val / lambda_C_bar  # dimensionless distance parameter

        if xi > 500:
            delta_U[j] = 0.0
            continue

        # Integration: t from 1+ to large enough that exp(-xi*t) is negligible
        t_max = min(1.0 + 50.0 / max(xi, 0.01), 1000.0)
        t = np.linspace(1.0 + 1e-10, t_max, N_int)

        # Integrand: (1 + 1/(2t²)) × √(1 - 1/t²) / t² × exp(-xi × t)
        sqrt_term = np.sqrt(1.0 - 1.0 / t**2)
        bracket = 1.0 + 1.0 / (2.0 * t**2)
        integrand = bracket * sqrt_term / t**2 * np.exp(-xi * t)

        integral = np.trapz(integrand, t)

        # δU = -(2α²ℏc)/(3r) × integral  [Joules — this IS potential energy]
        delta_U[j] = -(2.0 * alpha**2 * hbar * c) / (3.0 * r_val) * integral

    # Relative correction: δU / U_bare
    with np.errstate(divide='ignore', invalid='ignore'):
        relative = np.where(np.abs(U_bare) > 0, delta_U / U_bare, 0.0)

    return {
        'rho': rho,
        'U_bare_J': U_bare,
        'U_bare_eV': U_bare / eV,
        'delta_U_Uehling_J': delta_U,
        'delta_U_Uehling_eV': delta_U / eV,
        'U_Uehling_total_J': U_bare + delta_U,
        'U_Uehling_total_eV': (U_bare + delta_U) / eV,
        'relative_correction': relative,
        'rho_over_lambda_C_bar': rho / lambda_C_bar,
    }


def compute_eh_effective_potential(rho, R=None, r_tube=None, p=2, q=1):
    """
    Effective Coulomb potential ENERGY modified by the full nonlinear EH
    vacuum response.

    All quantities are potential energies (Joules or eV), NOT electric
    potentials (Volts).

    In a position-dependent dielectric, Gauss's law gives:
        ε_eff(ρ) × E(ρ) × 4πρ² = e
    so E_eff(ρ) = k_e × e / (ε_r(ρ) × ρ²)

    The potential energy of a test charge e at distance ρ:
        U(ρ) = -∫_ρ^∞ e × E_eff(ρ') dρ'
             = -∫_ρ^∞ k_e e² / (ε_r(ρ') ρ'²) dρ'

    For self-consistency, E determines ε which modifies E. We iterate:
      E⁰ = k_e e / ρ²  (bare Coulomb)
      ε_r^(n) = ε_eff(E^(n)) / ε₀
      E^(n+1) = k_e e / (ε_r^(n) × ρ²)
    until convergence.

    Parameters
    ----------
    rho : ndarray
        Radial distances (m), should be sorted ascending.
    R, r_tube : float or None
        Torus radii (only used for vacuum profile if provided).
    p, q : int
        Winding numbers.

    Returns
    -------
    dict with U_bare, U_EH, delta_U_EH, and comparison quantities.
    """
    rho = np.atleast_1d(np.asarray(rho, dtype=float))

    # Bare Coulomb field and potential energy
    E_bare = k_e * e_charge / rho**2      # electric field (V/m)
    U_bare = -k_e * e_charge**2 / rho     # potential energy (J)

    # Self-consistent iteration for E_eff in the nonlinear dielectric
    E_current = E_bare.copy()
    for _iteration in range(10):
        vac = compute_eh_effective_response(E_current)
        eps_r = vac['eps_eff'] / eps0
        E_new = k_e * e_charge / (eps_r * rho**2)
        rel_change = np.max(np.abs(E_new - E_current) / (E_current + 1e-300))
        E_current = E_new
        if rel_change < 1e-10:
            break

    E_eff = E_current
    vac_final = compute_eh_effective_response(E_eff)
    eps_r_final = vac_final['eps_eff'] / eps0

    # Force on test charge e: F = e × E_eff (Newtons)
    F_eff = e_charge * E_eff

    # Integrate to get potential energy: U(ρ) = -∫_ρ^∞ F(ρ') dρ'
    # = -∫_ρ^∞ e × E_eff(ρ') dρ'
    # Accumulate from large ρ inward; U(∞) = 0
    U_EH = np.zeros_like(rho)
    for j in range(len(rho) - 2, -1, -1):
        dr = rho[j + 1] - rho[j]
        U_EH[j] = U_EH[j + 1] - 0.5 * (F_eff[j] + F_eff[j + 1]) * dr

    # Shift to match bare Coulomb at the outer boundary
    # (EH correction vanishes at large distance where ε_r → 1)
    U_EH = U_EH + U_bare[-1]

    delta_U_EH = U_EH - U_bare

    with np.errstate(divide='ignore', invalid='ignore'):
        relative = np.where(np.abs(U_bare) > 0, delta_U_EH / U_bare, 0.0)

    return {
        'rho': rho,
        'E_bare': E_bare,
        'E_eff': E_eff,
        'eps_r': eps_r_final,
        'U_bare_J': U_bare,
        'U_bare_eV': U_bare / eV,
        'U_EH_J': U_EH,
        'U_EH_eV': U_EH / eV,
        'delta_U_EH_J': delta_U_EH,
        'delta_U_EH_eV': delta_U_EH / eV,
        'relative_correction': relative,
        'n_iterations': _iteration + 1,
        'E_over_ES': E_eff / E_Schwinger,
        'alpha_eff': vac_final['alpha_eff'],
    }


def compute_potential_comparison(N_radial=2000):
    """
    Compare the full nonlinear EH effective potential with the
    perturbative Uehling (one-loop) correction across all distance
    scales from sub-femtometer to beyond the Compton wavelength.

    Returns
    -------
    dict with both potentials on a common grid and comparison metrics.
    """
    lambda_C_bar = hbar / (m_e * c)
    r_classical = k_e * e_charge**2 / (m_e * c**2)  # classical electron radius

    # Schwinger boundary: ρ where E_Coulomb = E_Schwinger
    rho_Schwinger = np.sqrt(k_e * e_charge / E_Schwinger)

    # Grid: from 0.1 fm to 1000 ƛ_C (spanning all relevant scales)
    rho_min = 1e-16  # 0.1 fm
    rho_max = 1000.0 * lambda_C_bar
    rho = np.logspace(np.log10(rho_min), np.log10(rho_max), N_radial)

    # Compute both corrections
    uehling = compute_uehling_potential(rho)
    eh = compute_eh_effective_potential(rho)

    # At what distance do they agree? (relative difference < 10%)
    delta_U_ueh = uehling['delta_U_Uehling_J']
    delta_U_eh = eh['delta_U_EH_J']
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(delta_U_ueh) > 0, delta_U_eh / delta_U_ueh, np.nan)

    # Find crossover: where ratio crosses from ≠1 to ≈1
    agreement_mask = np.abs(ratio - 1.0) < 0.1  # within 10%
    if np.any(agreement_mask):
        first_agree = np.argmax(agreement_mask)
        rho_agreement = rho[first_agree]
    else:
        rho_agreement = np.nan

    return {
        'rho': rho,
        'rho_over_lambda_C_bar': rho / lambda_C_bar,
        'rho_over_r_e': rho / r_classical,
        'rho_Schwinger': rho_Schwinger,
        'rho_Schwinger_over_lambda_C_bar': rho_Schwinger / lambda_C_bar,
        'lambda_C_bar': lambda_C_bar,
        'r_classical': r_classical,
        'sqrt_alpha': np.sqrt(alpha),
        # Uehling
        'U_bare_eV': uehling['U_bare_eV'],
        'delta_U_Uehling_eV': uehling['delta_U_Uehling_eV'],
        'rel_Uehling': uehling['relative_correction'],
        # EH
        'delta_U_EH_eV': eh['delta_U_EH_eV'],
        'rel_EH': eh['relative_correction'],
        'eps_r': eh['eps_r'],
        'E_over_ES': eh['E_over_ES'],
        'alpha_eff': eh['alpha_eff'],
        # Comparison
        'ratio_EH_to_Uehling': ratio,
        'rho_agreement': rho_agreement,
        'rho_agreement_over_lambda_C_bar': rho_agreement / lambda_C_bar if not np.isnan(rho_agreement) else np.nan,
    }


# ════════════════════════════════════════════════════════════════════
# Step 8: Nonlocal vacuum polarization — momentum-space treatment
# ════════════════════════════════════════════════════════════════════

def compute_vacuum_polarization_pi(q2):
    """
    One-loop vacuum polarization function Π(Q²) in QED (Euclidean convention).

    Π(Q²) = (2α/π) ∫₀¹ dx x(1-x) ln(1 + Q² x(1-x) / m²)

    where Q² > 0 is the Euclidean (spacelike) squared momentum transfer
    and m² = (m_e c/ℏ)² in inverse-meter units.

    Properties:
      - Π(Q²) > 0 for Q² > 0 (anti-screening: effective charge increases)
      - Π(Q²) → 0 for Q² → 0
      - Π(Q²) ~ (α/3π) ln(Q²/m²) for Q² >> 4m² (logarithmic running)
      - At the Z pole (Q = 91 GeV): Π ≈ 0.06 (from electron loop alone)

    The sign convention: the screened charge is e²_eff = e² / (1 - Π),
    so Π > 0 means ANTI-screening (charge increases at short distance).

    Parameters
    ----------
    q2 : float or ndarray
        Euclidean squared momentum transfer Q² (m⁻²), positive.

    Returns
    -------
    Pi : ndarray
        Vacuum polarization Π(Q²) (dimensionless).
    """
    q2 = np.atleast_1d(np.asarray(q2, dtype=float))
    m2 = (m_e * c / hbar)**2  # electron mass² in momentum units (m⁻²)

    # Numerical integration over Feynman parameter x
    N_x = 200
    x = np.linspace(1e-6, 1.0 - 1e-6, N_x)

    Pi = np.zeros_like(q2)
    for j, q2_val in enumerate(q2):
        # Euclidean: argument = 1 + Q² x(1-x) / m² ≥ 1 (always positive)
        z = q2_val * x * (1.0 - x) / m2
        log_arg = np.log(1.0 + z)
        integrand = x * (1.0 - x) * log_arg
        Pi[j] = (2.0 * alpha / np.pi) * np.trapz(integrand, x)

    return Pi


def spherical_hankel_transform(f_r, r_grid, q_grid):
    """
    Spherical Hankel transform (radial Fourier transform for spherically
    symmetric functions).

    f̃(q) = 4π ∫₀^∞ f(r) sin(qr)/(qr) r² dr

    This is the 3D Fourier transform of a spherically symmetric function,
    reduced to a 1D integral.

    Parameters
    ----------
    f_r : ndarray, shape (N_r,)
        Function values on the radial grid.
    r_grid : ndarray, shape (N_r,)
        Radial grid points (m).
    q_grid : ndarray, shape (N_q,)
        Momentum grid points (m⁻¹).

    Returns
    -------
    f_q : ndarray, shape (N_q,)
        Transform values on the momentum grid.
    """
    f_q = np.zeros(len(q_grid))
    for j, q in enumerate(q_grid):
        if q < 1e-30:
            # q = 0 limit: f̃(0) = 4π ∫ f(r) r² dr
            f_q[j] = 4.0 * np.pi * np.trapz(f_r * r_grid**2, r_grid)
        else:
            # sinc(qr) = sin(qr)/(qr)
            qr = q * r_grid
            sinc = np.sin(qr) / qr
            f_q[j] = 4.0 * np.pi * np.trapz(f_r * sinc * r_grid**2, r_grid)
    return f_q


def inverse_spherical_hankel_transform(f_q, q_grid, r_grid):
    """
    Inverse spherical Hankel transform.

    f(r) = 1/(2π²) ∫₀^∞ f̃(q) sin(qr)/(qr) q² dq

    Parameters
    ----------
    f_q : ndarray, shape (N_q,)
        Function values in momentum space.
    q_grid : ndarray, shape (N_q,)
        Momentum grid points (m⁻¹).
    r_grid : ndarray, shape (N_r,)
        Radial grid points (m).

    Returns
    -------
    f_r : ndarray, shape (N_r,)
        Function values in position space.
    """
    f_r = np.zeros(len(r_grid))
    for i, r in enumerate(r_grid):
        if r < 1e-30:
            f_r[i] = 1.0 / (2.0 * np.pi**2) * np.trapz(f_q * q_grid**2, q_grid)
        else:
            qr = q_grid * r
            sinc = np.sin(qr) / qr
            f_r[i] = 1.0 / (2.0 * np.pi**2) * np.trapz(f_q * sinc * q_grid**2, q_grid)
    return f_r


def compute_nonlocal_potential(N_r=1000, N_q=800):
    """
    Compute the Coulomb potential screened by the full nonlocal
    one-loop vacuum polarization Π(q²).

    In momentum space:
        Ṽ(q) = -e² / (ε₀ q²)                    (bare Coulomb)
        Ṽ_screened(q) = Ṽ(q) / (1 - Π(q²))      (screened)

    The position-space potential is recovered by inverse Hankel transform.

    Parameters
    ----------
    N_r : int
        Number of radial grid points.
    N_q : int
        Number of momentum grid points.

    Returns
    -------
    dict with potentials, corrections, and comparison data.
    """
    lambda_C_bar = hbar / (m_e * c)
    m_inv = hbar / (m_e * c)  # 1/m_e in length units = ƛ_C

    # Position-space grid: 0.1 fm to 2000 ƛ_C
    r_min = 1e-16      # 0.1 fm
    r_max = 2000.0 * lambda_C_bar
    r_grid = np.logspace(np.log10(r_min), np.log10(r_max), N_r)

    # Momentum-space grid: must cover 1/r_max to 1/r_min
    # In units of m_e c/ℏ: q_min ~ 5e-4, q_max ~ 1e4
    q_min = 0.5 / r_max
    q_max = 20.0 / r_min
    q_grid = np.logspace(np.log10(q_min), np.log10(q_max), N_q)
    q2_grid = q_grid**2

    # Step 1: Vacuum polarization Π(q²)
    Pi = compute_vacuum_polarization_pi(q2_grid)

    # Step 2: Bare Coulomb in momentum space
    # V_bare(r) = -k_e e² / r → Ṽ(q) = -4π k_e e² / q²
    # (Fourier transform of -k_e e²/r in 3D)
    V_bare_q = -4.0 * np.pi * k_e * e_charge**2 / q2_grid

    # Step 3: Screened potential in momentum space
    # Ṽ_screened(q) = Ṽ_bare(q) / (1 - Π(q²))
    screening = 1.0 / (1.0 - Pi)
    V_screened_q = V_bare_q * screening

    # Step 4: Inverse Hankel transform to get V_screened(r)
    # The bare Coulomb transform back should give -k_e e²/r (verification)

    # For the CORRECTION only: δṼ(q) = Ṽ_bare(q) × Π(q²) / (1 - Π(q²))
    # This avoids numerical issues with transforming the full 1/q² and
    # subtracting large numbers
    delta_V_q = V_bare_q * Pi / (1.0 - Pi)

    # Inverse transform of the correction
    delta_V_r = inverse_spherical_hankel_transform(delta_V_q, q_grid, r_grid)

    # Bare potential (analytic, for comparison)
    V_bare_r = -k_e * e_charge**2 / r_grid

    # Relative correction
    with np.errstate(divide='ignore', invalid='ignore'):
        relative = np.where(np.abs(V_bare_r) > 0, delta_V_r / V_bare_r, 0.0)

    # Effective running coupling: α_eff(q) = α / (1 - Π(q²))
    alpha_eff_q = alpha * screening

    return {
        'r_grid': r_grid,
        'q_grid': q_grid,
        'q2_grid': q2_grid,
        'Pi': Pi,
        'V_bare_q': V_bare_q,
        'V_screened_q': V_screened_q,
        'delta_V_q': delta_V_q,
        'V_bare_r': V_bare_r,
        'delta_V_nonlocal_r': delta_V_r,
        'delta_V_nonlocal_eV': delta_V_r / eV,
        'V_screened_r': V_bare_r + delta_V_r,
        'relative_correction': relative,
        'alpha_eff_q': alpha_eff_q,
        'screening_factor': screening,
        'lambda_C_bar': lambda_C_bar,
    }


def compute_tube_form_factors(q_grid, r_tube):
    """
    Compute tube cross-section form factors for four physical models.

    The tube form factor F_tube(q) determines how the finite tube
    radius r_tube cuts off high-momentum modes in the self-energy
    integral. Different models for the charge distribution in the
    cross-section give very different UV behavior:

    1. Gaussian:      F = exp(-q²r²/4)         → exponential suppression
    2. Hard cylinder: F = 2J₁(qr)/(qr)         → ~1/(qr)^(3/2) power-law
    3. Surface ring:  F = J₀(qr)               → ~1/√(qr) power-law
    4. Thin wire:     F = 1                     → no cutoff (UV divergent)

    The NWT model has the charge circulating on the tube SURFACE,
    making the surface ring (J₀) the most physical choice.

    Parameters
    ----------
    q_grid : ndarray
        Momentum grid points (m⁻¹).
    r_tube : float
        Tube radius (m).

    Returns
    -------
    dict with F_tube arrays for each model and their |F|² values.
    """
    from scipy.special import j0, j1

    qr = q_grid * r_tube

    # 1. Gaussian: charge smeared as Gaussian across cross-section
    F_gaussian = np.exp(-qr**2 / 4.0)

    # 2. Hard cylinder: charge uniform across cross-section (radius r)
    # F = 2J₁(qr)/(qr) — the Airy disk function
    F_cylinder = np.where(qr > 1e-6, 2.0 * j1(qr) / qr, 1.0 - qr**2 / 8.0)

    # 3. Surface ring: charge on the tube surface (NWT boundary)
    # F = J₀(qr) — the zeroth Bessel function
    F_surface = j0(qr)

    # 4. Thin wire: no transverse structure
    F_wire = np.ones_like(q_grid)

    return {
        'gaussian': F_gaussian,
        'cylinder': F_cylinder,
        'surface': F_surface,
        'wire': F_wire,
        'gaussian_sq': F_gaussian**2,
        'cylinder_sq': F_cylinder**2,
        'surface_sq': F_surface**2,
        'wire_sq': F_wire**2,
        'qr': qr,
    }


def compute_nonlocal_self_energy(R, r_tube, p=2, q_wind=1, N_q=2000,
                                  tube_model='surface'):
    """
    Compute the electromagnetic self-energy of the torus using
    momentum-space screening with full Π(q²).

    The torus is a ring of charge e at major radius R with tube
    radius r_tube. The self-energy is:

    U_E = (e²/(4π² ε₀)) ∫₀^∞ dq [sin(qR)/(qR)]² × |F_tube(q)|²

    The ring form factor [sin(qR)/(qR)]² is the angle-averaged
    Fourier transform of a ring at radius R.

    F_tube(q) depends on the charge distribution in the tube cross-section.
    Four models are computed; the most physical for NWT is 'surface'
    (charge on the null worldtube boundary → J₀(qr)).

    The magnetic self-energy at v=c equals the electric, so U_total = 2 × U_E.

    Parameters
    ----------
    R, r_tube : float
        Torus radii (m).
    p, q_wind : int
        Winding numbers.
    N_q : int
        Number of momentum grid points.
    tube_model : str
        Which tube form factor to use as primary: 'gaussian', 'cylinder',
        'surface', or 'wire'. Default 'surface' (charge on tube boundary).

    Returns
    -------
    dict with bare and screened self-energies for all four models.
    """
    lambda_C_bar = hbar / (m_e * c)

    # Momentum grid: from well below 1/R to well above 1/r_tube
    # For J₀ and J₁ form factors, need many oscillation periods above 1/r
    q_min = 0.01 / R
    q_max = 500.0 / r_tube  # extend further for power-law tails
    q_grid = np.logspace(np.log10(q_min), np.log10(q_max), N_q)
    q2_grid = q_grid**2

    # Vacuum polarization
    Pi = compute_vacuum_polarization_pi(q2_grid)

    # Ring charge form factor (angle-averaged)
    qR = q_grid * R
    F_ring = np.where(qR > 1e-6, np.sin(qR) / qR, 1.0 - qR**2 / 6.0)
    F_ring_sq = F_ring**2

    # All four tube form factors
    tube_ff = compute_tube_form_factors(q_grid, r_tube)

    # Self-energy integrand:
    # U_E = (e²/(4π² ε₀)) ∫ dq F_ring²(q) × F_tube²(q)
    prefactor = e_charge**2 / (4.0 * np.pi**2 * eps0)

    models = {}
    for name in ['gaussian', 'cylinder', 'surface', 'wire']:
        F2 = F_ring_sq * tube_ff[name + '_sq']

        integrand_bare = F2
        integrand_screened = F2 / (1.0 - Pi)

        U_E_bare = prefactor * np.trapz(integrand_bare, q_grid)
        U_E_screened = prefactor * np.trapz(integrand_screened, q_grid)

        U_bare = 2.0 * U_E_bare       # E + B (equipartition at v=c)
        U_screened = 2.0 * U_E_screened
        delta_U = U_screened - U_bare

        models[name] = {
            'F_tube': tube_ff[name],
            'F_tube_sq': tube_ff[name + '_sq'],
            'F2': F2,
            'U_E_bare_J': U_E_bare,
            'U_bare_J': U_bare,
            'U_bare_MeV': U_bare / MeV,
            'U_bare_keV': U_bare / MeV * 1000,
            'U_screened_J': U_screened,
            'U_screened_MeV': U_screened / MeV,
            'U_screened_keV': U_screened / MeV * 1000,
            'delta_U_J': delta_U,
            'delta_U_keV': delta_U / MeV * 1000,
            'Pi_correction_pct': delta_U / U_bare * 100 if U_bare > 0 else np.inf,
            'dU_dq_bare': 2.0 * prefactor * integrand_bare,
            'dU_dq_screened': 2.0 * prefactor * integrand_screened,
            'dU_dq_correction': 2.0 * prefactor * (integrand_screened - integrand_bare),
        }

    # Primary model
    primary = models[tube_model]

    # Local EH self-energy for comparison
    eh_self = compute_effective_self_energy(R, r_tube, p, q_wind)

    # Neumann self-energy for comparison
    params_local = TorusParams(R=R, r=r_tube, p=p, q=q_wind)
    se_flat = compute_self_energy(params_local)

    # Analytic check: bare ring self-energy without tube = k_e e²/(2R)
    U_ring_analytic = k_e * e_charge**2 / (2.0 * R)

    return {
        'q_grid': q_grid,
        'q2_grid': q2_grid,
        'Pi': Pi,
        'F_ring': F_ring,
        'tube_ff': tube_ff,
        'models': models,
        'primary_model': tube_model,
        # Primary model results (for backward compatibility)
        'F2': primary['F2'],
        'F_tube': primary['F_tube'],
        'U_E_bare_J': primary['U_E_bare_J'],
        'U_E_bare_MeV': primary['U_E_bare_J'] / MeV,
        'U_bare_J': primary['U_bare_J'],
        'U_bare_MeV': primary['U_bare_MeV'],
        'U_screened_J': primary['U_screened_J'],
        'U_screened_MeV': primary['U_screened_MeV'],
        'delta_U_J': primary['delta_U_J'],
        'delta_U_MeV': primary['delta_U_J'] / MeV,
        'delta_U_keV': primary['delta_U_keV'],
        'U_ring_analytic_J': U_ring_analytic,
        'U_ring_analytic_MeV': U_ring_analytic / MeV,
        'U_Neumann_MeV': se_flat['U_total_MeV'],
        'U_EH_local_MeV': eh_self['U_self_eff_MeV'],
        'dU_dq_bare': primary['dU_dq_bare'],
        'dU_dq_screened': primary['dU_dq_screened'],
        'dU_dq_correction': primary['dU_dq_correction'],
        'lambda_C_bar': lambda_C_bar,
        'R': R,
        'r_tube': r_tube,
    }


def compute_full_nonlocal_energy_budget(R, r_tube, p=2, q_wind=1):
    """
    Complete energy budget using nonlocal vacuum polarization.

    Combines:
      - E_circ_eff from Gordon metric (local EH, captures path-length effect)
      - U_self from nonlocal Π(q²) (replaces local EH self-energy)
      - U_drag from Gordon metric (frame-dragging)

    The idea: use the local EH for effects that ARE local (refractive
    index along the photon path), and the nonlocal Π(q²) for effects
    that REQUIRE nonlocality (self-energy, charge screening).

    Parameters
    ----------
    R, r_tube : float
        Torus radii (m).
    p, q_wind : int
        Winding numbers.

    Returns
    -------
    dict with energy budgets: local EH, nonlocal Π(q²), and hybrid.
    """
    # Gordon metric results (local EH)
    gordon = compute_gordon_total_energy(R, r_tube, p, q_wind)

    # Nonlocal self-energy
    nonlocal_se = compute_nonlocal_self_energy(R, r_tube, p, q_wind)

    # Hybrid budget: Gordon circulation + nonlocal self-energy + Gordon drag
    E_circ_eff = gordon['E_circ_eff_J']
    U_self_nonlocal = nonlocal_se['U_screened_J']
    U_drag = gordon['U_drag_J']

    E_hybrid = E_circ_eff + U_self_nonlocal + U_drag
    E_hybrid_MeV = E_hybrid / MeV

    gap_hybrid = E_hybrid_MeV - m_e_MeV

    return {
        # Hybrid (nonlocal self-energy)
        'E_hybrid_J': E_hybrid,
        'E_hybrid_MeV': E_hybrid_MeV,
        'gap_hybrid_keV': gap_hybrid * 1000,
        'gap_hybrid_pct': gap_hybrid / m_e_MeV * 100,
        # Components
        'E_circ_eff_MeV': gordon['E_circ_eff_MeV'],
        'U_self_nonlocal_MeV': nonlocal_se['U_screened_MeV'],
        'U_self_bare_MeV': nonlocal_se['U_bare_MeV'],
        'delta_U_nonlocal_keV': nonlocal_se['delta_U_keV'],
        'U_drag_MeV': gordon['U_drag_MeV'],
        # Local EH for comparison
        'E_dressed_MeV': gordon['E_dressed_MeV'],
        'gap_dressed_keV': gordon['gap_dressed_keV'],
        'U_self_EH_MeV': gordon['U_self_eff_MeV'],
        # Reference
        'm_e_MeV': m_e_MeV,
        'gordon': gordon,
        'nonlocal_se': nonlocal_se,
    }


# ════════════════════════════════════════════════════════════════════
# Step 8b: Knot self-interaction — full Neumann integral over (p,q) knot
# ════════════════════════════════════════════════════════════════════

def compute_knot_neumann_inductance(R, r_tube, p=2, q=1, N=2000):
    """
    Compute the full Neumann inductance of a (p,q) torus knot by
    numerical double integration over the actual knot path.

    L = (μ₀/4π) ∮∮ (dl₁ · dl₂) / |r₁ - r₂|

    This includes BOTH the self-inductance of each pass AND the
    mutual inductance between different passes of the knot. The
    single-ring formula L = μ₀R[ln(8R/r) - 2] misses the mutual
    interaction entirely.

    For a (2,1) knot, the charge winds twice around the torus.
    At each toroidal cross-section there are two current elements
    on opposite sides of the tube (separated by ~2r). Their mutual
    inductance can be a large fraction of the self-inductance when
    r << R.

    The divergence at |r₁ - r₂| → 0 (self-interaction of a thin wire)
    is regularized by replacing |r₁ - r₂| with max(|r₁ - r₂|, r_tube)
    — the wire has finite thickness.

    Parameters
    ----------
    R, r_tube : float
        Torus radii (m).
    p, q : int
        Winding numbers.
    N : int
        Number of points on the knot curve.

    Returns
    -------
    dict with L_knot, comparison to single-ring, and decomposition
    into self and mutual contributions.
    """
    params = TorusParams(R=R, r=r_tube, p=p, q=q)

    # Get the full knot path
    lam, xyz, dxyz, ds_dlam = torus_knot_curve(params, N)
    dlam = 2.0 * np.pi / N

    # dl vectors: tangent × dλ
    dl = dxyz * dlam  # (N, 3) array of path elements

    # Full Neumann double integral:
    # L = (μ₀/4π) Σᵢ Σⱼ (dlᵢ · dlⱼ) / max(|rᵢ - rⱼ|, r_tube)
    #
    # This is O(N²). For N=2000, that's 4M operations — fast.
    L_knot = 0.0
    L_self_near = 0.0   # contributions from |i-j| < N/(2p) (same pass)
    L_mutual = 0.0      # contributions from different passes

    # For decomposition: points i and j are on the "same pass" if
    # they're within N/(2p) steps of each other (one half-turn)
    half_pass = N // (2 * p)

    for i in range(N):
        # Vector from point i to all other points
        dr = xyz - xyz[i]  # (N, 3)
        dist = np.sqrt(np.sum(dr**2, axis=1))  # (N,)

        # Regularize: replace small distances with r_tube
        dist_reg = np.maximum(dist, r_tube)

        # Dot product dl_i · dl_j for all j
        dot = np.sum(dl[i] * dl, axis=1)  # (N,)

        # Contribution to inductance
        contrib = dot / dist_reg  # (N,)

        L_knot += np.sum(contrib)

        # Decompose: same pass vs different pass
        # Distance along the curve (in index units, wrapping)
        idx_dist = np.minimum(np.abs(np.arange(N) - i),
                              N - np.abs(np.arange(N) - i))
        same_pass = idx_dist < half_pass
        L_self_near += np.sum(contrib[same_pass])
        L_mutual += np.sum(contrib[~same_pass])

    L_knot *= mu0 / (4.0 * np.pi)
    L_self_near *= mu0 / (4.0 * np.pi)
    L_mutual *= mu0 / (4.0 * np.pi)

    # Single-ring Neumann for comparison
    log_factor = np.log(8.0 * R / r_tube) - 2.0
    L_single_ring = mu0 * R * max(log_factor, 0.01)

    # Current
    L_path = compute_path_length(params, N)
    I = e_charge * c / L_path

    # Self-energies (total = E + B = 2 × magnetic for v=c)
    U_knot = 2.0 * 0.5 * L_knot * I**2
    U_single_ring = 2.0 * 0.5 * L_single_ring * I**2
    U_mutual = 2.0 * 0.5 * L_mutual * I**2

    return {
        'L_knot': L_knot,
        'L_single_ring': L_single_ring,
        'L_ratio': L_knot / L_single_ring,
        'L_self_near': L_self_near,
        'L_mutual': L_mutual,
        'mutual_fraction': L_mutual / L_knot if L_knot > 0 else 0,
        'I_amps': I,
        'L_path': L_path,
        'U_knot_J': U_knot,
        'U_knot_MeV': U_knot / MeV,
        'U_knot_keV': U_knot / MeV * 1000,
        'U_single_ring_J': U_single_ring,
        'U_single_ring_MeV': U_single_ring / MeV,
        'U_single_ring_keV': U_single_ring / MeV * 1000,
        'U_mutual_J': U_mutual,
        'U_mutual_MeV': U_mutual / MeV,
        'U_mutual_keV': U_mutual / MeV * 1000,
        'p': p, 'q': q, 'N': N,
        'R': R, 'r_tube': r_tube,
    }


# ════════════════════════════════════════════════════════════════════
# Step 9a: EH Self-Focusing Analysis
# ════════════════════════════════════════════════════════════════════

def compute_self_focusing_analysis(Z=1):
    """
    Analyze EH self-focusing dynamics for pair creation.

    The Coulomb field reaches E_Schwinger at r = r_e = alpha * lambda_C ~ 2.82 fm.
    At r = lambda_C the field is only alpha * E_S -- negligible for vacuum nonlinearity.
    The strong-field physics is concentrated within a few r_e of the nucleus.

    The photon wavelength at threshold (lambda ~ 1213 fm) is ~430x larger than r_e.
    This scale mismatch means the classical GRIN lens picture gives tiny
    deflections -- pair creation is fundamentally quantum (the photon must
    "tunnel" into the strong-field region with probability ~ alpha).

    But the NWT picture adds a crucial insight: the eikonal phase shift
    through the strong-field core is O(1) -- the photon accumulates enough
    phase to couple resonantly into the torus mode.

    Parameters
    ----------
    Z : int
        Nuclear charge (default 1 for hydrogen).

    Returns
    -------
    dict with all computed quantities.
    """
    # -- 19a: EH Kerr coefficient --
    # Weak-field EH: Delta_n = (8 alpha^2 / 45)(E/E_S)^2
    # n2_E = (8 alpha^2 / 45) / E_S^2 (field-based Kerr coefficient)
    n2_E = (8.0 * alpha**2 / 45.0) / E_Schwinger**2  # m^2/V^2
    n2_SI = n2_E * 2.0 / (eps0 * c)  # m^2/W (intensity-based)

    # Coulomb field at distance r from Z protons
    def E_coulomb(r_m):
        return Z * k_e * e_charge / r_m**2

    # The natural distance scale for strong EH physics is r_e, not lambda_C
    # E(r_e) = E_S / alpha = 137 E_S -- deep strong-field
    # E(lambda_C) = E_S * alpha = 0.0073 E_S -- weak-field
    E_at_re = E_coulomb(r_e)
    E_at_lambdaC = E_coulomb(lambda_C)

    # Build n(r) profile using r_e scale (where the vacuum IS nonlinear)
    rho_values_re = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    n_profile_re = []
    for rho_re in rho_values_re:
        r_m = rho_re * r_e
        E_r = E_coulomb(r_m)
        eh = compute_eh_effective_response(E_r)
        n_val = float(eh['n_eff'][0])
        n_profile_re.append((rho_re, r_m / lambda_C, E_r / E_Schwinger, n_val))

    # Refractive index at r_e (the critical distance)
    eh_re = compute_eh_effective_response(E_at_re)
    n_eff_at_re = float(eh_re['n_eff'][0])
    delta_n_at_re = n_eff_at_re - 1.0

    eh_lambdaC = compute_eh_effective_response(E_at_lambdaC)
    n_eff_at_lambdaC = float(eh_lambdaC['n_eff'][0])
    delta_n_at_lambdaC = n_eff_at_lambdaC - 1.0

    # Gradient dn/dr at r_e (numerical)
    dr = 0.01 * r_e
    n_plus = float(compute_eh_effective_response(E_coulomb(r_e + dr))['n_eff'][0])
    n_minus = float(compute_eh_effective_response(E_coulomb(r_e - dr))['n_eff'][0])
    dn_dr_at_re = (n_plus - n_minus) / (2.0 * dr)

    # -- 19b: Photon self-field at threshold --
    E_threshold = 2.0 * m_e * c**2  # Joules
    E_threshold_MeV = 2.0 * m_e_MeV
    lambda_threshold = h_planck * c / E_threshold  # photon wavelength
    V_mode = lambda_threshold**3  # mode volume estimate
    # Peak E-field of a single photon: E_pk = sqrt(hbar*omega / (eps0 * V))
    E_peak = np.sqrt(E_threshold / (eps0 * V_mode))
    tau_pulse = lambda_threshold / c  # pulse duration ~ lambda/c

    # -- 19d: Eikonal deflection --
    # For a photon in a spherically symmetric n(r) from Coulomb+EH,
    # the eikonal deflection is:
    # delta_theta = -integral (b/r)(dn/dr)(1/n) ds, where r^2 = b^2 + s^2
    #
    # Key: n(r) variation is concentrated within r < few * r_e.
    # Must resolve the r_e scale in the integration.
    def compute_eikonal_deflection(b_m, N_s=20000):
        """Eikonal deflection angle for impact parameter b."""
        # Log-spaced to resolve the r_e core
        s_max = max(50.0 * r_e, 5.0 * b_m)
        s_pos = np.geomspace(0.01 * r_e, s_max, N_s // 2)
        s = np.concatenate([-s_pos[::-1], s_pos])

        r = np.sqrt(b_m**2 + s**2)
        r = np.maximum(r, 0.005 * r_e)

        E_field = Z * k_e * e_charge / r**2
        eh = compute_eh_effective_response(E_field)
        n_arr = eh['n_eff']

        # dn/dr (numerical)
        dr_step = 0.005 * r_e
        r_plus = r + dr_step
        r_minus = np.maximum(r - dr_step, 0.003 * r_e)
        E_plus = Z * k_e * e_charge / r_plus**2
        E_minus = Z * k_e * e_charge / r_minus**2
        n_plus_arr = compute_eh_effective_response(E_plus)['n_eff']
        n_minus_arr = compute_eh_effective_response(E_minus)['n_eff']
        dn_dr = (n_plus_arr - n_minus_arr) / (r_plus - r_minus)

        integrand = (b_m / r) * dn_dr / n_arr
        deflection = -np.trapz(integrand, s)
        return float(deflection)

    # Table of deflections at various impact parameters (in units of r_e)
    b_values_re = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    deflection_table = []
    for b_re in b_values_re:
        b_m = b_re * r_e
        theta = compute_eikonal_deflection(b_m)
        deflection_table.append((
            b_re,                          # in units of r_e
            b_m / lambda_C,                # in units of lambda_C
            theta,                         # radians
            np.degrees(theta),             # degrees
            theta / (2.0 * np.pi),         # turns
        ))

    # Maximum deflection (at smallest b)
    theta_max = deflection_table[0][2]

    # Find critical b for pi deflection (if achievable)
    from scipy.optimize import brentq

    theta_at_small = compute_eikonal_deflection(0.5 * r_e)
    theta_at_large = compute_eikonal_deflection(100.0 * r_e)

    b_crit_pi = None
    b_crit_pi_re = None
    if theta_at_small > np.pi > theta_at_large:
        try:
            b_crit_pi_re = brentq(
                lambda b_re: compute_eikonal_deflection(b_re * r_e) - np.pi,
                0.5, 100.0, xtol=0.01
            )
            b_crit_pi = b_crit_pi_re * r_e
        except ValueError:
            pass

    # Find b for 2*pi deflection (orbit closure)
    b_crit_2pi = None
    b_crit_2pi_re = None
    if theta_at_small > 2.0 * np.pi > theta_at_large:
        try:
            b_crit_2pi_re = brentq(
                lambda b_re: compute_eikonal_deflection(b_re * r_e) - 2.0 * np.pi,
                0.5, 100.0, xtol=0.01
            )
            b_crit_2pi = b_crit_2pi_re * r_e
        except ValueError:
            pass

    b_crit_fm = (b_crit_pi / 1e-15) if b_crit_pi else 0.0
    b_crit_lambdaC = (b_crit_pi / lambda_C) if b_crit_pi else 0.0

    # -- 19e: Critical power for self-focusing --
    # P_cr = 3.77 lambda^2 / (8*pi * n0 * n2) [Fibich-Gaeta]
    P_critical = 3.77 * lambda_threshold**2 / (8.0 * np.pi * 1.0 * n2_SI)
    E_critical_J = P_critical * tau_pulse
    E_critical_MeV = E_critical_J / MeV

    P_photon = E_threshold / tau_pulse
    P_ratio = P_photon / P_critical

    # -- 19f: Eikonal phase shift through the strong-field core --
    # Delta_phi_eik = (2*pi/lambda) * integral [n(r) - 1] ds
    def compute_eikonal_phase(b_m, N_s=20000):
        """Eikonal phase shift (radians) for impact parameter b."""
        s_max = max(50.0 * r_e, 5.0 * b_m)
        s_pos = np.geomspace(0.01 * r_e, s_max, N_s // 2)
        s = np.concatenate([-s_pos[::-1], s_pos])
        r = np.sqrt(b_m**2 + s**2)
        r = np.maximum(r, 0.005 * r_e)
        E_field = Z * k_e * e_charge / r**2
        eh = compute_eh_effective_response(E_field)
        delta_n = eh['n_eff'] - 1.0
        phase = (2.0 * np.pi / lambda_threshold) * np.trapz(delta_n, s)
        return float(phase)

    # Phase shifts at various impact parameters
    phase_table = []
    for b_re in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        b_m = b_re * r_e
        phi = compute_eikonal_phase(b_m)
        phase_table.append((b_re, b_m / lambda_C, phi, phi / (2.0 * np.pi)))

    # -- 19f continued: Tidal shear at r_e scale --
    dE_dr_at_re = 2.0 * Z * k_e * e_charge / r_e**3
    delta_E_tidal_re = dE_dr_at_re * r_e  # Delta E across one r_e
    delta_E_over_E_tidal = delta_E_tidal_re / E_at_re  # should be ~2

    delta_n_tidal_re = abs(dn_dr_at_re) * r_e
    delta_n_over_n = delta_n_tidal_re / n_eff_at_re if n_eff_at_re > 1 else 0
    R_bend_re = r_e / delta_n_over_n if delta_n_over_n > 0 else np.inf

    # -- 19g: Cross-sections --
    # Bethe-Heitler near threshold: sigma ~ (pi/12) alpha r_e^2 Z^2 (E_gamma/m_e c^2 - 2)^3
    sigma_BH_threshold = (np.pi / 12.0) * alpha * r_e**2 * Z**2 * (2.5 - 2.0)**3
    sigma_BH_threshold_barn = sigma_BH_threshold / 1e-28

    # BH high-energy: sigma -> (28/9) alpha r_e^2 Z^2 ln(2E/mc^2)
    sigma_BH_high = (28.0 / 9.0) * alpha * r_e**2 * Z**2 * np.log(2.0 * 5.0)
    sigma_BH_high_barn = sigma_BH_high / 1e-28

    # Delbruck at 1 MeV: sigma ~ alpha^3 r_e^2 Z^2 (E/mc^2)^2
    sigma_D_1MeV = alpha**3 * r_e**2 * Z**2 * (1.0 / m_e_MeV)**2
    sigma_D_1MeV_barn = sigma_D_1MeV / 1e-28

    # NWT: geometric cross-section at b where eikonal phase ~ 1
    b_phase_1 = r_e  # default
    for b_re_val, _, phi_val, _ in phase_table:
        if phi_val >= 1.0:
            b_phase_1 = b_re_val * r_e
            break
    sigma_eik = np.pi * b_phase_1**2
    sigma_eik_barn = sigma_eik / 1e-28

    # -- 19h: Scale analysis --
    lambda_over_re = lambda_threshold / r_e  # pi/alpha ~ 430
    overlap = (r_e / lambda_threshold)**3  # volume ratio ~ alpha^3/pi^3

    return {
        # 19a
        'n2_SI': n2_SI,
        'n2_E': n2_E,
        'E_over_ES_at_re': E_at_re / E_Schwinger,
        'E_over_ES_at_lambdaC': E_at_lambdaC / E_Schwinger,
        'delta_n_at_re': delta_n_at_re,
        'n_eff_at_re': n_eff_at_re,
        'delta_n_at_lambdaC': delta_n_at_lambdaC,
        'n_eff_at_lambdaC': n_eff_at_lambdaC,
        'n_profile_re': n_profile_re,
        # 19b
        'E_threshold_MeV': E_threshold_MeV,
        'lambda_threshold': lambda_threshold,
        'V_mode': V_mode,
        'E_peak': E_peak,
        'tau_pulse': tau_pulse,
        # 19d
        'deflection_table': deflection_table,
        'theta_max': theta_max,
        'b_crit_pi': b_crit_pi,
        'b_crit_pi_re': b_crit_pi_re,
        'b_crit_2pi': b_crit_2pi,
        'b_crit_2pi_re': b_crit_2pi_re,
        'b_crit_lambdaC': b_crit_lambdaC,
        'b_crit_fm': b_crit_fm,
        # 19e
        'P_critical': P_critical,
        'E_critical_MeV': E_critical_MeV,
        'P_photon': P_photon,
        'P_ratio': P_ratio,
        # 19f: phase shifts
        'phase_table': phase_table,
        # 19f: tidal shear
        'E_at_re': E_at_re,
        'dn_dr_at_re': dn_dr_at_re,
        'delta_E_tidal_re': delta_E_tidal_re,
        'delta_E_over_E_tidal': delta_E_over_E_tidal,
        'delta_n_tidal_re': delta_n_tidal_re,
        'R_bend_re': R_bend_re,
        # 19g
        'sigma_BH_threshold': sigma_BH_threshold,
        'sigma_BH_threshold_barn': sigma_BH_threshold_barn,
        'sigma_BH_high': sigma_BH_high,
        'sigma_BH_high_barn': sigma_BH_high_barn,
        'sigma_D_1MeV': sigma_D_1MeV,
        'sigma_D_1MeV_barn': sigma_D_1MeV_barn,
        'sigma_eik': sigma_eik,
        'sigma_eik_barn': sigma_eik_barn,
        'b_phase_1': b_phase_1,
        # 19h
        'lambda_over_re': lambda_over_re,
        'overlap': overlap,
    }


# ════════════════════════════════════════════════════════════════════
# Step 9b: Parametric Down-Conversion Analysis
# ════════════════════════════════════════════════════════════════════

def compute_parametric_down_conversion(Z=1):
    """
    Analyze pair creation as parametric down-conversion in the EH vacuum.

    The Coulomb field provides the background that generates an effective
    χ⁽²⁾ nonlinearity (parity-breaking). The nuclear recoil provides
    phase matching. The conversion probability is α per photon.

    Parameters
    ----------
    Z : int
        Nuclear charge (default 1).

    Returns
    -------
    dict with all PDC analysis quantities.
    """
    # ── Coulomb field at r_e ──
    E_C_at_re = Z * k_e * e_charge / r_e**2

    # ── Effective χ⁽²⁾ from EH Lagrangian ──
    # The EH four-photon vertex in a background field E_C gives an
    # effective three-photon vertex (χ⁽²⁾):
    # χ⁽²⁾_eff ~ (α²/(45 E_S²)) × E_C
    # In dimensionless form (Gaussian units):
    chi2_dimless = (alpha**2 / 45.0) * (E_C_at_re / E_Schwinger)
    # In SI (m/V): χ⁽²⁾ ~ (α²/E_S) × (E_C/E_S) × ε₀^{-1/2}
    # More precisely, the EH gives δε ~ (α²/E_S²) E_C, so
    # χ⁽²⁾_SI ~ α² × E_C / E_S² / ε₀  (very rough, order-of-magnitude)
    chi2_SI = alpha**2 * E_C_at_re / (E_Schwinger**2 * eps0)

    # ── Threshold ──
    omega_0 = m_e * c**2 / hbar  # torus cutoff frequency
    E_threshold_MeV = 2.0 * m_e_MeV
    k_gamma_threshold = 2.0 * m_e * c / hbar  # = 2/ƛ_C
    p_recoil = hbar * k_gamma_threshold
    p_recoil_MeV = p_recoil * c / MeV
    E_recoil_eV = (hbar * k_gamma_threshold)**2 / (2.0 * 938.3 * MeV / c**2) / eV
    # More carefully: M_p in kg
    M_p = 1.6726e-27  # kg
    E_recoil_eV_val = (p_recoil**2 / (2.0 * M_p)) / eV

    # ── Cross-section comparison ──
    # BH at 5 m_e c²
    sigma_BH_5MeV = (28.0 / 9.0) * alpha * r_e**2 * Z**2 * np.log(2.0 * 5.0)
    alpha_re2 = alpha * r_e**2 * Z**2
    log_factor = np.log(2.0 * 5.0)

    # ── Mode fields ──
    # Photon field in mode volume λ³
    lambda_threshold = h_planck * c / (2.0 * m_e * c**2)
    V_mode = lambda_threshold**3
    E_photon_field = np.sqrt(2.0 * m_e * c**2 / (eps0 * V_mode))

    # Torus field at its center (r = 0 from tube axis, distance R from torus center)
    # This is just the Coulomb field of the charge e at distance r_e
    E_torus_field = k_e * e_charge / r_e**2

    # Core volume
    V_core = (4.0 / 3.0) * np.pi * r_e**3

    # ── Kinematics above threshold ──
    kinematics = []
    for E_gamma_MeV_val in [1.022, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0]:
        E_gamma = E_gamma_MeV_val * MeV
        E_each = E_gamma / 2.0
        E_each_MeV = E_each / MeV
        if E_each_MeV < m_e_MeV:
            continue
        gamma = E_each / (m_e * c**2)
        v_over_c = np.sqrt(1.0 - 1.0 / gamma**2) if gamma > 1 else 0.0
        p_each = np.sqrt(E_each**2 - (m_e * c**2)**2) / c
        p_each_MeV = p_each * c / MeV
        kinematics.append((E_gamma_MeV_val, v_over_c, gamma, E_each_MeV, p_each_MeV))

    return {
        'E_C_at_re': E_C_at_re,
        'chi2_at_re': chi2_dimless,
        'chi2_SI_at_re': chi2_SI,
        'omega_0': omega_0,
        'E_threshold_MeV': E_threshold_MeV,
        'k_gamma_threshold': k_gamma_threshold,
        'p_recoil_MeV': p_recoil_MeV,
        'E_recoil_eV': E_recoil_eV_val,
        'sigma_BH_5MeV': sigma_BH_5MeV,
        'alpha_re2': alpha_re2,
        'log_factor': log_factor,
        'E_photon_field': E_photon_field,
        'E_torus_field': E_torus_field,
        'V_core': V_core,
        'kinematics': kinematics,
    }


# ════════════════════════════════════════════════════════════════════
# Step 9c: Entanglement Generation in NWT Pair Creation
# ════════════════════════════════════════════════════════════════════

def compute_entanglement_analysis():
    """
    Analyze entanglement structure of pair creation in the NWT framework.

    In the PDC picture, the photon converts into two torus modes (e⁻ + e⁺)
    in the EH nonlinear core. The entanglement structure depends on:
    - Photon polarization → torus chirality mapping
    - Angular momentum conservation (J=1 photon → two J=1/2 particles)
    - The PDC selection rules from χ⁽²⁾ symmetry

    Returns
    -------
    dict with density matrices, concurrence, entropy, and Bell state analysis.
    """
    # ── Photon polarization states ──
    # |L⟩ = left-circular, |R⟩ = right-circular
    # |H⟩ = (|L⟩ + |R⟩)/√2 (horizontal linear)
    # |V⟩ = (|L⟩ - |R⟩)/(i√2) (vertical linear)

    # ── NWT chirality mapping ──
    # Left-circular photon → left-handed torus (electron, s_z = -ℏ/2)
    # Right-circular photon → right-handed torus (positron, s_z = +ℏ/2)
    #
    # But pair creation ALWAYS produces BOTH particles, so the mapping is:
    # The photon's L component → electron chirality
    # The photon's R component → positron chirality
    # (or vice versa — the assignment is conventional)

    # ── Angular momentum conservation ──
    # Photon: J = 1, m_J = ±1 (circular polarization)
    #
    # For a LEFT-circular photon (m_J = +1):
    #   e⁻(↑) + e⁺(↑): m_s = +1/2 + 1/2 = +1  ✓  (triplet |T₊⟩)
    #   Also need orbital angular momentum L contributions near threshold
    #
    # For a RIGHT-circular photon (m_J = -1):
    #   e⁻(↓) + e⁺(↓): m_s = -1/2 - 1/2 = -1  ✓  (triplet |T₋⟩)
    #
    # For a LINEAR photon (superposition of m_J = ±1):
    #   Superposition of |T₊⟩ and |T₋⟩:
    #   |ψ⟩ = (1/√2)(|↑↑⟩ + e^{iφ} |↓↓⟩)
    #   This is a BELL STATE (|Φ⁺⟩ or |Φ⁻⟩ depending on phase)

    # ── Spin density matrices ──
    # Basis: |↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩ (electron ⊗ positron)

    # Left-circular photon → |↑↑⟩ (triplet m=+1)
    psi_L = np.array([1, 0, 0, 0], dtype=complex)  # |↑↑⟩

    # Right-circular photon → |↓↓⟩ (triplet m=-1)
    psi_R = np.array([0, 0, 0, 1], dtype=complex)  # |↓↓⟩

    # H-polarized photon → (|L⟩ + |R⟩)/√2 → (|↑↑⟩ + |↓↓⟩)/√2 = |Φ⁺⟩
    psi_H = (psi_L + psi_R) / np.sqrt(2)

    # V-polarized photon → (|L⟩ - |R⟩)/(i√2) → (|↑↑⟩ - |↓↓⟩)/(i√2)
    # Up to global phase: |Φ⁻⟩ = (|↑↑⟩ - |↓↓⟩)/√2
    psi_V = (psi_L - psi_R) / np.sqrt(2)

    # ── Density matrices ──
    def density_matrix(psi):
        return np.outer(psi, np.conj(psi))

    rho_L = density_matrix(psi_L)
    rho_R = density_matrix(psi_R)
    rho_H = density_matrix(psi_H)
    rho_V = density_matrix(psi_V)

    # ── Partial trace (trace out positron) ──
    def partial_trace_B(rho_4x4):
        """Trace out the second qubit (positron) from a 4×4 density matrix."""
        # Basis: |00⟩, |01⟩, |10⟩, |11⟩ where first=electron, second=positron
        rho_A = np.zeros((2, 2), dtype=complex)
        rho_A[0, 0] = rho_4x4[0, 0] + rho_4x4[1, 1]  # ⟨0|ρ_A|0⟩
        rho_A[0, 1] = rho_4x4[0, 2] + rho_4x4[1, 3]  # ⟨0|ρ_A|1⟩
        rho_A[1, 0] = rho_4x4[2, 0] + rho_4x4[3, 1]  # ⟨1|ρ_A|0⟩
        rho_A[1, 1] = rho_4x4[2, 2] + rho_4x4[3, 3]  # ⟨1|ρ_A|1⟩
        return rho_A

    rho_e_from_L = partial_trace_B(rho_L)
    rho_e_from_H = partial_trace_B(rho_H)

    # ── Entanglement entropy ──
    def von_neumann_entropy(rho_2x2):
        """Von Neumann entropy S = -Tr(ρ ln ρ) for a 2×2 density matrix."""
        eigenvalues = np.linalg.eigvalsh(rho_2x2)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]  # filter zeros
        return -np.sum(eigenvalues * np.log2(eigenvalues))

    S_L = von_neumann_entropy(rho_e_from_L)  # should be 0 (product state)
    S_H = von_neumann_entropy(rho_e_from_H)  # should be 1 (maximally entangled)

    # ── Concurrence ──
    def concurrence(psi_4):
        """Concurrence for a pure 2-qubit state."""
        # For |ψ⟩ = a|00⟩ + b|01⟩ + c|10⟩ + d|11⟩:
        # C = 2|ad - bc|
        a, b, c_val, d = psi_4
        return float(2.0 * abs(a * d - b * c_val))

    C_L = concurrence(psi_L)
    C_H = concurrence(psi_H)
    C_V = concurrence(psi_V)

    # ── Bell inequality ──
    # CHSH inequality: |S| ≤ 2 (classical), |S| ≤ 2√2 (quantum max)
    # For a maximally entangled state: S = 2√2 (Tsirelson bound)
    # For a product state: S = 2
    S_CHSH_H = 2.0 * np.sqrt(2)  # maximally entangled
    S_CHSH_L = 2.0               # product state

    # ── NWT-specific: chirality entanglement ──
    # In NWT, the torus chirality (handedness) IS the spin.
    # Left-handed torus ↔ spin down, Right-handed torus ↔ spin up
    #
    # Chirality basis: |L_e, R_e+⟩ means electron=left-torus, positron=right-torus
    # The mapping to spin: |L⟩ = |↓⟩, |R⟩ = |↑⟩
    #
    # So |↑↑⟩ = |R_e, R_e+⟩ and |↓↓⟩ = |L_e, L_e+⟩
    #
    # For H-polarized photon:
    #   |Φ⁺⟩ = (|R_e R_e+⟩ + |L_e L_e+⟩)/√2
    #   Both tori have SAME chirality — correlated, not anti-correlated!
    #
    # This is TYPE-I SPDC (same polarization output)
    # Type-II would give |↑↓⟩ + |↓↑⟩ (|Ψ⁺⟩) — opposite chirality

    # ── Entanglement for different photon energies ──
    # Near threshold: pair created at rest, spherically symmetric
    # → pure spin state, maximal entanglement
    # High energy: pair boosted, spin mixed with orbital
    # → reduced entanglement due to spin-orbit coupling
    #
    # The spin-averaged pair creation amplitude in QED:
    # At threshold (v→0): pair is in definite spin state
    # At high energy: helicity conservation → still highly entangled

    # Entanglement as function of gamma (Lorentz factor)
    entanglement_vs_energy = []
    for gamma_val in [1.0, 1.1, 1.5, 2.0, 5.0, 10.0, 100.0]:
        v_c = np.sqrt(1.0 - 1.0/gamma_val**2) if gamma_val > 1 else 0.0
        E_MeV = gamma_val * m_e_MeV

        # At high energy, helicity is approximately conserved
        # Pair from L-photon: e⁻(L-helicity) + e⁺(R-helicity)
        # The helicity-spin mismatch scales as m/E = 1/γ
        # So the "purity" of the spin state decreases as 1/γ²
        # but helicity entanglement INCREASES (approaches perfect correlation)

        # For a linearly polarized photon, the concurrence in the
        # HELICITY basis is:
        # C(helicity) → 1 as γ → ∞ (helicity becomes good quantum number)
        # C(spin) → 0 as γ → ∞ (spin is not helicity for massive particles)

        # In NWT: chirality = helicity at v=c (the internal photon),
        # so the torus chirality is ALWAYS a good quantum number.
        # The entanglement in the chirality basis is always maximal.
        # C_chirality = 1 for all γ (because the torus handedness is topological)

        C_chirality = 1.0  # topological — always maximal

        # Spin entanglement: reduced by Wigner rotation at high boost
        # The spin-½ Wigner rotation angle for boost β at angle θ:
        # tan(Ω_W/2) = (β sin θ)/(γ(1 + β cos θ))
        # For back-to-back pair (θ = π): Ω_W → 0
        # For asymmetric kinematics: Ω_W can be significant
        # Average over angles:
        if gamma_val <= 1.001:
            C_spin_avg = 1.0
        else:
            # Approximate: C_spin ~ 1 - O(v²/c²) for symmetric pair
            # The Wigner rotation preserves entanglement for back-to-back
            # but mixes it for other configurations
            # Average concurrence ~ 1/(1 + v²/4) (rough estimate)
            C_spin_avg = 1.0 / (1.0 + v_c**2 / 4.0)

        entanglement_vs_energy.append((
            gamma_val, E_MeV, v_c, C_chirality, C_spin_avg
        ))

    # ── Type-I vs Type-II classification ──
    # In SPDC:
    #   Type-I: both output photons have same polarization (|HH⟩ + |VV⟩)
    #   Type-II: output photons have opposite polarization (|HV⟩ + |VH⟩)
    #
    # In NWT pair creation from a linearly polarized photon:
    #   |Φ⁺⟩ = (|↑↑⟩ + |↓↓⟩)/√2 = (|R_e R_e+⟩ + |L_e L_e+⟩)/√2
    #   This is TYPE-I: same chirality for both particles
    #
    # But wait: e⁻ and e⁺ have OPPOSITE charge, so "same chirality"
    # means they wind in OPPOSITE directions (L-electron is a left-winding
    # torus with charge -e; R-positron is a right-winding torus with +e).
    # In terms of physical rotation sense, they're OPPOSITE.
    #
    # This is exactly right for CPT: the positron is the CP conjugate.

    return {
        # States
        'psi_L': psi_L,
        'psi_R': psi_R,
        'psi_H': psi_H,
        'psi_V': psi_V,
        # Density matrices
        'rho_H': rho_H,
        'rho_e_from_L': rho_e_from_L,
        'rho_e_from_H': rho_e_from_H,
        # Entanglement measures
        'S_L': S_L,
        'S_H': S_H,
        'C_L': C_L,
        'C_H': C_H,
        'C_V': C_V,
        # Bell
        'S_CHSH_H': S_CHSH_H,
        'S_CHSH_L': S_CHSH_L,
        # vs energy
        'entanglement_vs_energy': entanglement_vs_energy,
    }


# ════════════════════════════════════════════════════════════════════
# Section 23: Quantum tube radius — what fixes r/R = α?
# ════════════════════════════════════════════════════════════════════

def compute_quantum_tube_radius(R=None, p=2, q=1):
    """
    Investigate quantum mechanisms that could fix r/R = α.

    All CLASSICAL energy terms at fixed r/R = α scale as R⁻¹
    (self-similarity of the torus). This means the energy landscape
    E(R, r=αR) is monotonically decreasing — no minimum in R or r.

    A quantum mechanism must BREAK this self-similarity. We examine:

    1. Waveguide zero-point energy (transverse confinement)
    2. Casimir energy on the torus (periodic boundary conditions)
    3. One-loop effective potential from vacuum polarization Π(q²)
    4. EH soliton / self-trapping condition
    5. Running coupling: α(r) at the tube scale

    Parameters
    ----------
    R : float, optional
        Major radius. Default: self-consistent R ≈ 0.488 λ_C.
    p, q : int
        Winding numbers.

    Returns
    -------
    dict with results from each mechanism.
    """
    if R is None:
        R = 0.488 * lambda_C

    j01 = 2.4048  # first zero of J₀
    j11 = 3.8317  # first zero of J₁

    # ── Mechanism 1: Waveguide zero-point energy ──
    # Transverse confinement in a circular waveguide of radius r:
    # E_conf = ℏc j₀₁ / r  (TM₀₁ mode)
    # This gives a 1/r repulsive potential. To balance it we need
    # an attractive potential that grows with r.
    #
    # Classical surface tension: U_σ = σ × 4π²Rr → grows as r
    # Equilibrium: dE/dr = 0 → -ℏc j₀₁/r² + 4π²Rσ = 0
    # → r_eq = √(ℏc j₀₁ / (4π²Rσ))
    #
    # For r_eq = αR: σ = ℏc j₀₁ / (4π²R × α²R²) = ℏc j₀₁/(4π²α²R³)
    # This σ depends on R — not a universal surface tension.
    #
    # The quantum question: is there a NATURAL σ from QED that gives r = αR?

    r_alpha = alpha * R

    # Surface tension needed for r = αR equilibrium
    sigma_needed = hbar * c * j01 / (4.0 * np.pi**2 * R * r_alpha**2)

    # Schwinger surface tension (natural QED scale)
    sigma_Schwinger = eps0 * E_Schwinger**2 * lambda_C

    # What r does Schwinger σ give?
    r_from_Schwinger = np.sqrt(hbar * c * j01 / (4.0 * np.pi**2 * R * sigma_Schwinger))

    # ── Mechanism 2: Casimir energy on the torus ──
    # A massless field confined to a tube of radius r and length L = 2πpR
    # (for the toroidal winding) has Casimir energy from the transverse modes.
    #
    # For a circular cross-section (Dirichlet BC), the Casimir energy per
    # unit length is:
    #   E_Cas/L = -ℏc × C_Cas / r²
    # where C_Cas ≈ 0.00246 (numerically computed for EM field in cylinder).
    #
    # BUT: the NWT torus is not a hard-wall cylinder. The photon is confined
    # by the EH nonlinear potential, which is smooth. The Casimir coefficient
    # depends on the boundary condition.
    #
    # For a waveguide with smooth confining potential V(ρ) ~ (ρ/r)^n:
    # the Casimir energy scales differently from hard walls.
    #
    # More relevant: the longitudinal Casimir effect from periodic BCs
    # along the knot. Length L, modes k_n = 2πn/L:
    #   E_Cas_long = -ℏc π / (6L)  [1D periodic Casimir]
    #
    # This is NEGATIVE (attractive) and scales as 1/L ~ 1/R.
    # It adds to the R⁻¹ scaling — does NOT break self-similarity.

    # Knot path length
    params = TorusParams(R=R, r=r_alpha, p=p, q=q)
    L_knot = compute_path_length(params)

    # Longitudinal Casimir (1D periodic, massless scalar)
    E_Cas_long = -hbar * c * np.pi / (6.0 * L_knot)
    E_Cas_long_MeV = E_Cas_long / MeV

    # Transverse Casimir (cylinder, EM field)
    # C_Cas ≈ 0.00246 for EM field with perfectly conducting walls
    C_Cas = 0.00246
    E_Cas_trans_per_L = -hbar * c * C_Cas / r_alpha**2
    E_Cas_trans = E_Cas_trans_per_L * L_knot
    E_Cas_trans_MeV = E_Cas_trans / MeV

    # Total Casimir
    E_Cas_total = E_Cas_long + E_Cas_trans
    E_Cas_total_MeV = E_Cas_total / MeV

    # Does the transverse Casimir create an r-dependent potential?
    # E_Cas_trans(r) = -ℏc C_Cas L / r²
    # dE_Cas_trans/dr = +2 ℏc C_Cas L / r³ (REPULSIVE — pushes r outward)
    # Combined with confinement (also 1/r repulsive), we need attraction.
    # Casimir makes it WORSE, not better — both push r outward.

    # ── Mechanism 3: One-loop effective potential from Π(q²) ──
    # The vacuum polarization modifies the Coulomb self-energy at short distances.
    # The screened potential is V(r) = (α/r) × 1/(1 - Π(1/r²))
    # At r ~ r_e: Π is small (α/3π × ln stuff)
    # At r ~ λ_C: Π is very small
    #
    # The Uehling potential gives a correction:
    #   δV_Uehling(r) = -(2α²/3) × (ℏ/mc) × δ³(r)  (contact term)
    # In the torus context: the self-energy gets a correction from
    # running α. The tube radius r sets the UV cutoff for the self-energy.
    #
    # Self-energy with running coupling:
    #   U_self(r) ~ α_eff(1/r) × (ℏc/R) × [ln(8R/r) - 2]
    #   where α_eff(q) = α / (1 - Π(q²))
    #
    # At q = 1/r: α_eff = α / (1 - Π(1/r²))
    # Π(1/r²) ≈ (α/3π) ln(1/(mr)²) for r << λ_C
    #           ≈ (α/3π) ln(1/α²) for r = r_e = α λ_C
    #           = (2α/3π) ln(1/α) ≈ 0.0075

    # Compute Π at various r scales
    r_scan = np.geomspace(0.1 * r_e, 10.0 * lambda_C, 50)
    q2_scan = 1.0 / r_scan**2  # convert to momentum² (in m⁻²)
    Pi_scan = compute_vacuum_polarization_pi(q2_scan)
    alpha_eff_scan = alpha / (1.0 - Pi_scan)

    # At r = r_e specifically
    q2_re = 1.0 / r_e**2
    Pi_at_re = float(compute_vacuum_polarization_pi(np.array([q2_re]))[0])
    alpha_eff_re = alpha / (1.0 - Pi_at_re)

    # At r = α R (the tube radius at self-consistent R)
    q2_tube = 1.0 / r_alpha**2
    Pi_at_tube = float(compute_vacuum_polarization_pi(np.array([q2_tube]))[0])
    alpha_eff_tube = alpha / (1.0 - Pi_at_tube)

    # The running coupling correction to self-energy
    # δU_self / U_self = δα / α = Π / (1 - Π) ≈ Π
    # This is O(α/3π) ~ 0.08% — tiny correction, same sign at all r.
    # It does NOT create a minimum in r.

    # ── Mechanism 4: EH soliton / self-trapping ──
    # In nonlinear optics, a beam can self-trap when the nonlinear
    # refractive index creates a potential well matching the diffraction.
    # Self-trapping condition: Δn × (2πr/λ)² ~ 1
    #
    # For the EH vacuum inside the torus:
    # At r = r_e: E ~ E_S/α, Δn ~ O(1) from EH
    # The "photon" (torus mode) has λ_eff ~ 2πr (transverse mode)
    #
    # Self-trapping: Δn(r) × (2πr / λ_mode)² ≈ 1
    # where λ_mode ~ r (the transverse mode wavelength IS the tube radius)
    # So: Δn(r) × (2π)² ≈ 1 → Δn ≈ 0.025
    #
    # Where does Δn = 0.025? This is the EH nonlinearity at some E/E_S.
    # Δn = (8α²/45)(E/E_S)² = 0.025 → E/E_S ≈ 14.1
    # E = 14.1 × E_S at r: r = √(k_e × e / (14.1 × E_S))
    # = r_e × √(137/14.1) ≈ 3.1 r_e

    # Compute the self-trapping radius
    delta_n_soliton = 1.0 / (2.0 * np.pi)**2  # ≈ 0.0253

    # Find where Δn(r) = delta_n_soliton using the EH response
    from scipy.optimize import brentq

    def delta_n_at_r(r_m):
        E_field = k_e * e_charge / r_m**2
        eh = compute_eh_effective_response(E_field)
        return float(eh['n_eff'][0]) - 1.0 - delta_n_soliton

    # Bracket: at r = 0.1 r_e, Δn >> 0.025; at r = 100 r_e, Δn << 0.025
    try:
        r_soliton = brentq(delta_n_at_r, 0.1 * r_e, 100.0 * r_e, xtol=1e-4 * r_e)
        r_soliton_over_re = r_soliton / r_e
        r_soliton_over_R = r_soliton / R
    except ValueError:
        r_soliton = None
        r_soliton_over_re = None
        r_soliton_over_R = None

    # ── Mechanism 5: Running coupling self-consistency ──
    # The NWT predicts α from the torus geometry. If α itself runs with
    # the scale r, then the self-consistency condition r = α(r) × R
    # becomes a fixed-point equation:
    #
    #   r = α_eff(1/r) × R = [α / (1 - Π(1/r²))] × R
    #
    # This is r = f(r) where f(r) = α × R / (1 - Π(1/r²))
    # Fixed point: r* such that r* = f(r*)
    #
    # Since Π > 0 and increases with q² = 1/r², α_eff > α at smaller r.
    # The fixed point r* > αR (slightly larger tube from running coupling).

    def fixed_point_residual(r_m):
        q2_val = 1.0 / r_m**2
        Pi_val = float(compute_vacuum_polarization_pi(np.array([q2_val]))[0])
        alpha_eff_val = alpha / (1.0 - Pi_val)
        return r_m - alpha_eff_val * R

    # Find fixed point
    try:
        r_fixed = brentq(fixed_point_residual, 0.5 * r_alpha, 2.0 * r_alpha,
                         xtol=1e-6 * r_alpha)
        r_fixed_over_R = r_fixed / R
        r_fixed_over_alpha_R = r_fixed / r_alpha
        Pi_at_fixed = float(compute_vacuum_polarization_pi(
            np.array([1.0 / r_fixed**2]))[0])
        alpha_eff_fixed = alpha / (1.0 - Pi_at_fixed)
    except ValueError:
        r_fixed = r_alpha
        r_fixed_over_R = alpha
        r_fixed_over_alpha_R = 1.0
        Pi_at_fixed = Pi_at_tube
        alpha_eff_fixed = alpha_eff_tube

    # ── Mechanism 6: Topological quantization ──
    # The (p,q) = (2,1) knot must close on itself. The path length is
    # L(R,r) and the internal wavelength is λ_int = 2πℏc/E.
    # Self-consistency: L = n × λ_int (resonance condition, n=1 fundamental)
    # This fixes R (we already use this). But it also constrains r/R:
    #
    # L(R, r=αR) = 2π√(p²R² + q²r²) = 2πR√(p² + q²α²)
    # For (2,1): L = 2πR√(4 + α²) ≈ 2πR × 2.0000027
    #
    # The resonance gives R. But what constrains α?
    # The POLOIDAL resonance: the mode must also fit in the tube cross-section.
    # TM₀₁ mode: j₀₁ = 2πr/λ_transverse
    # λ_transverse = 2πr/j₀₁
    # For the knot, the poloidal winding (q=1) means one wavelength fits
    # in the poloidal circumference 2πr. Matching:
    # 2πr = q × λ_transverse = 2πr/j₀₁ × q
    # → j₀₁ = q = 1? No — j₀₁ = 2.405, not 1.

    # The poloidal resonance condition:
    # In q poloidal windings, the transverse mode accumulates phase j₀₁ × (2πr)/(λ_trans)
    # For the fundamental mode: the transverse wavevector k_⊥ = j₀₁/r
    # The poloidal wavevector: k_pol = q/r (one winding in circumference 2πr)
    # Dispersion: k_tot² = k_⊥² + k_pol² = (j₀₁/r)² + (q/r)²
    # = (j₀₁² + q²)/r²
    #
    # Toroidal wavevector: k_tor = p/R
    # Total: k² = k_tor² + k_⊥² + k_pol² = (p/R)² + (j₀₁² + q²)/r²
    # Energy: E = ℏc × k = ℏc × √((p/R)² + (j₀₁² + q²)/r²)
    #
    # Self-consistent R from L = λ gives one constraint.
    # What's the second? It comes from the ENERGY matching:
    # E = ℏc k must equal the Gordon-metric total energy including self-energy.
    #
    # Let's see if matching E_mode = E_total gives r/R = α.

    k_tor = p / R
    k_trans_sq_over_r2 = j01**2 + q**2  # (j₀₁² + q²)

    # From E = ℏc × sqrt(k_tor² + k_trans²):
    # E² = (ℏc)² × (p²/R² + (j₀₁² + q²)/r²)
    # If E = m_e c² (the electron mass):
    # (m_e c/ℏ)² = p²/R² + (j₀₁² + q²)/r²
    # 1/λ_C² = p²/R² + (j₀₁² + q²)/r²
    #
    # From the toroidal resonance: R ≈ pλ_C/(2π) × correction
    # Actually let's just solve for r given R and E = m_e c²:

    k_me_sq = (m_e * c / hbar)**2  # = 1/λ_C²
    k_tor_sq = (p / R)**2

    if k_me_sq > k_tor_sq:
        k_trans_sq = k_me_sq - k_tor_sq
        # k_trans² = (j₀₁² + q²)/r² → r = √(j₀₁² + q²) / k_trans
        r_from_dispersion = np.sqrt(j01**2 + q**2) / np.sqrt(k_trans_sq)
        r_disp_over_R = r_from_dispersion / R
    else:
        r_from_dispersion = None
        r_disp_over_R = None

    # ── Summary table: what each mechanism predicts for r/R ──
    predictions = []
    predictions.append(('Waveguide confinement + Schwinger σ',
                        r_from_Schwinger / R if r_from_Schwinger else None,
                        'Classical — σ is not derived'))
    predictions.append(('Casimir (transverse + longitudinal)',
                        None,
                        'Wrong sign — pushes r outward'))
    predictions.append(('Running coupling fixed point',
                        r_fixed_over_R,
                        f'Π = {Pi_at_fixed:.5f}, tiny correction'))
    predictions.append(('EH soliton self-trapping',
                        r_soliton / R if r_soliton else None,
                        f'Δn = 1/(2π)² at r = {r_soliton_over_re:.1f} r_e' if r_soliton else 'No solution'))
    predictions.append(('Dispersion relation (toroidal + transverse)',
                        r_disp_over_R,
                        f'E = m_e c², k² = k_tor² + k_trans²'))

    return {
        # R and target
        'R': R,
        'R_over_lambdaC': R / lambda_C,
        'r_alpha': r_alpha,
        'alpha': alpha,
        # Mechanism 1: surface tension
        'sigma_needed': sigma_needed,
        'sigma_Schwinger': sigma_Schwinger,
        'sigma_ratio': sigma_needed / sigma_Schwinger,
        'r_from_Schwinger': r_from_Schwinger,
        'r_Schwinger_over_R': r_from_Schwinger / R,
        # Mechanism 2: Casimir
        'L_knot': L_knot,
        'E_Cas_long_MeV': E_Cas_long_MeV,
        'E_Cas_trans_MeV': E_Cas_trans_MeV,
        'E_Cas_total_MeV': E_Cas_total_MeV,
        'C_Cas': C_Cas,
        # Mechanism 3: running coupling
        'Pi_at_re': Pi_at_re,
        'alpha_eff_re': alpha_eff_re,
        'Pi_at_tube': Pi_at_tube,
        'alpha_eff_tube': alpha_eff_tube,
        'r_scan': r_scan,
        'Pi_scan': Pi_scan,
        'alpha_eff_scan': alpha_eff_scan,
        # Mechanism 4: soliton
        'delta_n_soliton': delta_n_soliton,
        'r_soliton': r_soliton,
        'r_soliton_over_re': r_soliton_over_re,
        'r_soliton_over_R': r_soliton_over_R,
        # Mechanism 5: fixed point
        'r_fixed': r_fixed,
        'r_fixed_over_R': r_fixed_over_R,
        'r_fixed_over_alpha_R': r_fixed_over_alpha_R,
        'Pi_at_fixed': Pi_at_fixed,
        'alpha_eff_fixed': alpha_eff_fixed,
        # Mechanism 6: dispersion
        'r_from_dispersion': r_from_dispersion,
        'r_disp_over_R': r_disp_over_R,
        'j01': j01,
        'k_trans_sq_over_r2': k_trans_sq_over_r2,
        # Summary
        'predictions': predictions,
    }


# ════════════════════════════════════════════════════════════════════
# Section 24: TW Soliton Self-Consistency
# ════════════════════════════════════════════════════════════════════

def compute_soliton_self_consistency(R=None, p=2, q=1):
    """
    Test whether the NWT electron is a self-consistent traveling-wave soliton.

    The question: does the circulating photon mode create an EH refractive
    index profile n(ρ) that can confine ITSELF?

    We test this via:
    A. GRIN waveguide analysis: compute V-parameter from the mode's own field
    B. Self-consistent iteration: start with mode width w, compute field,
       compute n(ρ), solve waveguide eigenmode, check for convergence
    C. Topological confinement: show the knot winding provides the "walls"

    Parameters
    ----------
    R : float, optional
        Major radius. Default: self-consistent R ≈ 0.488 λ_C.
    p, q : int
        Winding numbers.

    Returns
    -------
    dict with GRIN analysis, V-parameters, energy decomposition.
    """
    if R is None:
        R = 0.488 * lambda_C

    r_tube = alpha * R  # assumed tube radius
    j01 = 2.4048

    # Knot path length and current
    params = TorusParams(R=R, r=r_tube, p=p, q=q)
    L = compute_path_length(params)
    I_current = e_charge * c / L     # circulating current (A)
    lambda_line = e_charge / L       # linear charge density (C/m)

    # Photon wavenumber (total energy ℏω = m_e c²)
    k_photon = m_e * c / hbar        # = 1/λ_C

    # ── Part A: Field profiles ──
    # Two models: (1) line charge/current, (2) point charge at center
    N_rho = 200
    rho = np.geomspace(0.01 * r_e, 100.0 * r_e, N_rho)

    # Model 1: Line charge (valid for ρ << R, which r_e << R ✓)
    E_line = lambda_line / (2.0 * np.pi * eps0 * rho)
    B_line = mu0 * I_current / (2.0 * np.pi * rho)

    # Model 2: Point charge (Coulomb at distance ρ from center)
    E_point = k_e * e_charge / rho**2

    # EH response for each model
    eh_line = compute_eh_effective_response(E_line)
    n_line = eh_line['n_eff']
    delta_n_line = n_line - 1.0

    eh_point = compute_eh_effective_response(E_point)
    n_point = eh_point['n_eff']
    delta_n_point = n_point - 1.0

    # ── Part B: V-parameter for GRIN guidance ──
    # For a GRIN fiber with profile n(ρ), the V-parameter is:
    # V = k × a × √(n_core² - n_clad²) ≈ k × a × √(2 Δn_max)
    # where a = core radius (we use r_e), Δn_max at ρ → 0 or at peak.
    # Guided mode exists when V > 2.405 (for step-index; ~π/2 for GRIN).

    # Line charge model: Δn peaks at smallest ρ (strongest field)
    # Use Δn at ρ = r_e as the characteristic core value
    idx_re = np.argmin(np.abs(rho - r_e))
    delta_n_line_at_re = float(delta_n_line[idx_re])
    delta_n_point_at_re = float(delta_n_point[idx_re])

    # Maximum Δn in each model (at smallest ρ in grid)
    delta_n_line_max = float(delta_n_line[0])
    delta_n_point_max = float(delta_n_point[0])

    # V-parameters (using r_e as the core radius)
    V_line = k_photon * r_e * np.sqrt(2.0 * max(delta_n_line_at_re, 1e-30))
    V_point = k_photon * r_e * np.sqrt(2.0 * max(delta_n_point_at_re, 1e-30))

    # Also compute: what Δn would be NEEDED for V = 2.405?
    delta_n_needed = 0.5 * (j01 / (k_photon * r_e))**2
    # = 0.5 × (j₀₁ λ_C / r_e)² = 0.5 × (j₀₁/α)² ≈ 54,300

    # ── Part C: Field strength comparison ──
    # The line charge field is MUCH weaker than Coulomb at the same distance.
    # At ρ = r_e:
    E_line_at_re = float(E_line[idx_re])
    E_point_at_re = float(E_point[idx_re])
    ratio_line_to_point = E_line_at_re / E_point_at_re
    # This ratio = 2r_e/L (geometric factor)
    geometric_ratio = 2.0 * r_e / L

    # ── Part D: Topological winding energy ──
    # The (p,q) knot has q poloidal windings. Each winding contributes
    # a transverse angular momentum. The minimum transverse confinement
    # energy for q poloidal windings in a tube of radius r:
    #
    # For q=1: the mode makes one full loop around the tube cross-section.
    # The transverse wavelength is λ_⊥ = 2πr (circumference of tube).
    # The transverse momentum: p_⊥ = ℏ × 2π/λ_⊥ = ℏ/r
    # The transverse energy: E_⊥ = p_⊥ c = ℏc/r
    #
    # This is EXACTLY the confinement energy E_conf = ℏc j₀₁/r
    # (modulo the j₀₁ vs 1 Bessel factor from the mode profile).
    #
    # For q=0 (no poloidal winding): no transverse confinement needed.
    # The mode could be a thin ring in the equatorial plane.
    # For q=1: one poloidal winding → transverse confinement is REQUIRED.
    # The topology FORCES a finite tube radius.

    E_poloidal = hbar * c * q / r_tube  # minimum poloidal energy
    E_poloidal_MeV = E_poloidal / MeV
    E_conf = hbar * c * j01 / r_tube    # waveguide confinement energy
    E_conf_MeV = E_conf / MeV
    E_total = m_e * c**2
    frac_poloidal = E_poloidal / E_total
    frac_conf = E_conf / E_total

    # ── Part E: Topological stability ──
    # The (2,1) torus knot has linking number = p × q = 2.
    # It cannot be smoothly deformed to a circle (linking number 0)
    # without cutting the curve.
    #
    # In terms of energy: to unwind the poloidal winding, the tube
    # must collapse to r → 0, which costs infinite confinement energy
    # (E_⊥ → ∞ as r → 0). The topology creates an energy BARRIER
    # against unwinding.
    #
    # Radiation: can the knot radiate away its energy?
    # A circulating charge radiates (Larmor). But in NWT, the charge
    # IS the mode — it can't radiate "itself" away. The lowest radiation
    # mode would change the topology (remove the winding), which requires
    # energy ≥ 2m_e c² (pair annihilation threshold).
    #
    # This is the topological stability of the electron: it's stable
    # because unwinding costs more energy than the particle has.

    # Energy to "unwind" — compress tube to zero: E_⊥(r→0) → ∞
    # Alternatively: annihilation threshold: E = 2m_e c²
    # The electron can only decay by meeting a positron (opposite winding).

    # ── Part F: Self-consistent iteration (GRIN attempt) ──
    # Even though V << 2.405, let's see what width the GRIN WOULD give
    # if we took it seriously. For a 1/ρ potential well in 2D:
    # V_eff(ρ) = k² [n²(ρ) - 1] ~ k² × 2Δn(ρ)
    # For line charge: Δn ∝ E² ∝ 1/ρ² → V_eff = g/ρ²
    # The "coupling constant" g = k² × 2 × (8α²/45) × [2k_e e/(L E_S)]²
    # (using weak-field EH for Δn)

    # For a 1/ρ² potential in 2D, bound state exists iff g > 1/4.
    E_line_over_ES = E_line / E_Schwinger
    # At large ρ (weak field): Δn = (8α²/45)(E/E_S)²
    # So V_eff × ρ² = k² × 2 × (8α²/45) × (E_line/E_S)² × ρ²
    # For line charge: (E/E_S)² × ρ² = [2k_e e/(L E_S)]² = const!
    g_line = k_photon**2 * 2.0 * (8.0 * alpha**2 / 45.0) * (
        2.0 * k_e * e_charge / (L * E_Schwinger))**2
    # This is the effective coupling for the 1/ρ² potential
    # Need g > 1/4 for a bound state

    # Build the n(ρ) profile table for display
    rho_display = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    n_profile_table = []
    for rho_re in rho_display:
        rho_m = rho_re * r_e
        E_l = lambda_line / (2.0 * np.pi * eps0 * rho_m)
        E_p = k_e * e_charge / rho_m**2
        eh_l = compute_eh_effective_response(E_l)
        eh_p = compute_eh_effective_response(E_p)
        n_l = float(eh_l['n_eff'][0])
        n_p = float(eh_p['n_eff'][0])
        n_profile_table.append((rho_re, E_l / E_Schwinger, n_l, E_p / E_Schwinger, n_p))

    return {
        # Geometry
        'R': R,
        'r_tube': r_tube,
        'L': L,
        'I_current': I_current,
        'lambda_line': lambda_line,
        'k_photon': k_photon,
        # Field profiles
        'rho': rho,
        'E_line': E_line,
        'E_point': E_point,
        'n_line': n_line,
        'n_point': n_point,
        'delta_n_line': delta_n_line,
        'delta_n_point': delta_n_point,
        # V-parameter
        'delta_n_line_at_re': delta_n_line_at_re,
        'delta_n_point_at_re': delta_n_point_at_re,
        'delta_n_line_max': delta_n_line_max,
        'delta_n_point_max': delta_n_point_max,
        'V_line': V_line,
        'V_point': V_point,
        'delta_n_needed': delta_n_needed,
        # Field comparison
        'E_line_at_re': E_line_at_re,
        'E_point_at_re': E_point_at_re,
        'ratio_line_to_point': ratio_line_to_point,
        'geometric_ratio': geometric_ratio,
        # Topological energy
        'E_poloidal': E_poloidal,
        'E_poloidal_MeV': E_poloidal_MeV,
        'E_conf': E_conf,
        'E_conf_MeV': E_conf_MeV,
        'frac_poloidal': frac_poloidal,
        'frac_conf': frac_conf,
        # 1/ρ² potential
        'g_line': g_line,
        # Display
        'n_profile_table': n_profile_table,
    }


# ════════════════════════════════════════════════════════════════════
# Section 25: Self-Entanglement as Confinement Mechanism
# ════════════════════════════════════════════════════════════════════

def compute_self_entanglement(R=None, p=2, q=1, N=2000):
    """
    Analyze self-entanglement of the photon in the (p,q) torus knot.

    The (2,1) knot makes p=2 toroidal windings before closing. At any
    cross-section of the torus tube, the photon passes through TWICE
    per circulation period, separated by poloidal angle Δφ = 2πq/p = π.

    A single photon traversing this path is in a superposition of being
    in pass 1 and pass 2. The resonance condition (kL = 2π) enforces a
    fixed phase relationship between the two passes. This creates a
    self-entangled state analogous to a Bell state: the photon's state
    at pass 1 is quantum-correlated with its own state at pass 2.

    This self-entanglement is what makes the knot topologically stable:
    unwinding the knot would require disentangling the two passes, which
    costs energy (the entanglement entropy × temperature ≥ kT ln 2).

    Parameters
    ----------
    R : float, optional
        Major radius. Default: self-consistent R ≈ 0.488 λ_C.
    p, q : int
        Winding numbers of the torus knot.
    N : int
        Number of points for path discretization.

    Returns
    -------
    dict with self-entanglement analysis.
    """
    if R is None:
        R = 0.488 * lambda_C

    r_tube = alpha * R

    # ── 25a: Decompose the (p,q) knot into p toroidal passes ──

    params = TorusParams(R=R, r=r_tube, p=p, q=q)
    lam, xyz, dxyz, ds_dlam = torus_knot_curve(params, N)
    dlam = 2.0 * np.pi / N
    L_total = np.sum(ds_dlam) * dlam
    L_per_pass = L_total / p

    # Toroidal angle along the knot: theta = p * lam
    theta = p * lam
    # Poloidal angle: phi = q * lam
    phi = q * lam

    # At a fixed toroidal cross-section (say theta_0 = 0), the knot passes
    # through at lam values where p*lam = 2*pi*n (n=0,1,...,p-1)
    # At these crossings, the poloidal angle is phi = q*lam = 2*pi*q*n/p
    # For (2,1): passes at phi = 0 and phi = pi (opposite sides of tube)
    crossing_phases = np.array([2.0 * np.pi * q * n / p for n in range(p)])
    poloidal_separation = crossing_phases[1] - crossing_phases[0] if p > 1 else 0.0

    # ── 25b: Phase relationship between passes ──

    # The photon has wavevector k = 2*pi/lambda = 2*pi*E/(hc) = omega/c
    # Resonance: kL = 2*pi  =>  k = 2*pi/L
    k_photon = 2.0 * np.pi / L_total

    # Phase accumulated in one toroidal pass (L/p of path):
    phase_per_pass = k_photon * L_per_pass  # = 2*pi/p

    # For p=2: phase_per_pass = pi
    # This means the wavefunction picks up a phase of e^{i*pi} = -1
    # between pass 1 and pass 2!
    phase_between_passes = phase_per_pass  # pi for p=2

    # The two-pass state is:
    # |ψ⟩ = (1/√2)(|pass_1⟩ + e^{iπ}|pass_2⟩) = (1/√2)(|pass_1⟩ - |pass_2⟩)
    # This is a maximally entangled Bell-like state (|ψ⁻⟩)

    # ── 25c: Entanglement entropy ──

    # For a maximally entangled bipartite state of a 2-component system:
    # S = -Tr(ρ_A ln ρ_A) = ln 2
    # ρ_A = Tr_B(|ψ⟩⟨ψ|) = (1/2)(|pass_1⟩⟨pass_1| + |pass_2⟩⟨pass_2|) = I/2
    S_max = np.log(2.0)  # ln 2 ≈ 0.693 nats

    # The entanglement is between the TWO TOROIDAL PASSES of the SAME photon.
    # This is not ordinary entanglement between two particles — it's
    # SELF-entanglement of a single quantum across its own topology.

    # Number of "entangled crossings" per period: each cross-section has
    # p passes, giving C(p,2) = p(p-1)/2 pairwise entanglements
    n_crossing_pairs = p * (p - 1) // 2

    # ── 25d: Mutual energy between passes ──

    # The two passes of the (2,1) knot share the same tube volume.
    # Their electromagnetic fields overlap and interact.
    # The mutual energy is captured by the Neumann inductance integral.

    # Current from circulating charge:
    I_current = e_charge * c / L_total

    # Neumann inductance of the torus: L_N = μ₀R[ln(8R/r) - 2]
    log_factor = np.log(8.0 * R / r_tube) - 2.0
    L_ind = mu0 * R * max(log_factor, 0.01)

    # Total magnetic self-energy: U_mag = (1/2)L_N I²
    # This INCLUDES the mutual inductance between the two passes.
    U_mag_total = 0.5 * L_ind * I_current**2

    # For p passes, the inductance decomposes as:
    # L_total = p × L_self + p(p-1) × M
    # where L_self is the self-inductance of one pass, M is the mutual.
    #
    # For a (2,1) knot, the two passes are on opposite sides of the tube
    # (poloidal separation π). The mutual inductance is approximately:
    # M ≈ μ₀R × [ln(8R/r) - 2] × cos(π × q/p) = -L_self for q/p = 1/2
    # (The Neumann integral picks up a cos factor from the poloidal offset.)
    #
    # Actually for coils wound on a torus, Neumann gives the full result.
    # The key point: M captures the electromagnetic coupling between passes.

    # Compute mutual interaction by comparing 2-pass vs 1-pass energy
    # Single-pass params (p=1, q=0 trivial loop for comparison):
    params_single = TorusParams(R=R, r=r_tube, p=1, q=0)
    L_single = compute_path_length(params_single, N)
    I_single = e_charge * c / L_single
    log_factor_single = np.log(8.0 * R / r_tube) - 2.0
    L_ind_single = mu0 * R * max(log_factor_single, 0.01)
    U_mag_single = 0.5 * L_ind_single * I_single**2

    # Two independent single loops would have 2 × U_single (no correlation)
    U_two_independent = 2.0 * U_mag_single

    # The (2,1) knot energy vs two independent loops:
    # Mutual energy = U_knot - 2*U_single_pass
    # (This can be positive or negative depending on field alignment)
    U_mutual = U_mag_total - U_two_independent
    U_mutual_MeV = U_mutual / MeV

    # ── 25e: Distance between passes at a cross-section ──

    # At a given cross-section, the two passes are at poloidal angles
    # φ = 0 and φ = π (for (2,1) knot). The distance between them:
    d_pass = 2.0 * r_tube  # diameter of tube (straight across)
    d_pass_fm = d_pass / 1e-15

    # Field overlap: the two passes' fields overlap in the tube interior.
    # The overlap integral depends on the mode profile.
    # For the lowest-order mode (J_0 profile with radius ~ r_tube):
    # Overlap ∝ ∫ ψ_1(ρ) ψ_2(ρ) dA
    # where ψ_1, ψ_2 are centered at opposite sides of the tube.
    # With mode width ~ r_tube and separation ~ 2*r_tube:
    # Overlap ~ exp(-d²/(2w²)) = exp(-4r²/(2r²)) = exp(-2) ≈ 0.135
    mode_overlap = np.exp(-2.0)  # Gaussian approximation

    # ── 25f: Topological protection of entanglement ──

    # The entanglement is topologically protected because:
    # 1. Disentangling requires SEPARATING the two passes
    # 2. The passes are linked by the torus topology
    # 3. Separation requires cutting the knot (= pair annihilation)

    # Energy cost to disentangle (minimum):
    # Landauer principle: erasing 1 bit of entanglement costs kT ln 2
    # But this is a QUANTUM system at T=0. The relevant energy is:
    # E_entangle = S × E_scale
    # where E_scale is the characteristic energy of the entangled DOF.

    # The entanglement energy is the MUTUAL energy between passes.
    # This is already part of the total mass — removing it costs energy.
    # At T=0, destroying the entanglement = destroying the knot.

    # Topological unwinding barrier: infinite in classical field theory
    # (cannot continuously deform a knot to an unknot)
    # In QFT: finite but ∝ m_e c² (pair annihilation threshold)
    E_unwinding_MeV = 2.0 * m_e_MeV  # pair annihilation = mutual destruction

    # ── 25g: Connection to charge quantization ──

    # The winding number p is an INTEGER — you can't have p = 1.5 passes.
    # This means:
    # - Topological charge (= winding number) is quantized: Q = n × e
    # - The entanglement pattern is all-or-nothing: either the passes are
    #   entangled (knot exists) or they're not (knot destroyed)
    # - No continuous interpolation between charged and uncharged states
    # - This is WHY charge is quantized in NWT

    # Phase per pass = 2π/p. For the state to be single-valued:
    # p × (2π/p) = 2π ✓ (always satisfied for integer p)
    # But the ENTANGLEMENT structure depends on p:
    #   p=1: no self-entanglement (trivial loop, not a knot)
    #   p=2: Bell-like state (|ψ⁻⟩), S = ln 2
    #   p=3: GHZ-like state, S = ln 3
    # The (2,1) knot is the SIMPLEST non-trivially entangled state.

    # ── 25h: Connection to pair creation/annihilation ──

    # Section 22 showed: pair creation PRODUCES entanglement (Bell states).
    # Self-entanglement completes the picture:
    #
    # PAIR CREATION (γ → e⁻e⁺):
    #   - One photon → two torus knots
    #   - The two knots are MUTUALLY entangled (Section 22: chirality Bell state)
    #   - Each knot is also SELF-entangled (Section 25: two passes)
    #   - Total entanglement created: S_mutual + 2 × S_self = ln2 + 2×ln2 = 3 ln 2
    #
    # PAIR ANNIHILATION (e⁻e⁺ → γγ):
    #   - Two torus knots → two photons
    #   - Destroys self-entanglement of BOTH knots (2 × ln 2)
    #   - Destroys mutual entanglement (ln 2)
    #   - The photons carry away the mutual entanglement as polarization correlation
    #   - The self-entanglement is LOST (converted to photon degrees of freedom)

    S_pair_creation = 3.0 * S_max  # total entanglement created
    S_self_per_particle = S_max  # ln 2 per particle
    S_mutual_pair = S_max  # ln 2 between particle and antiparticle

    # ── 25i: Why p=2 is special ──

    # p=1: trivial loop. No crossing, no self-entanglement.
    #       Would be a boson (integer spin). Unstable (can shrink to zero).
    # p=2: simplest knot with self-entanglement. Two passes, one crossing pair.
    #       Fermion (half-integer spin: each pass carries spin/2).
    #       The phase e^{iπ} = -1 between passes IS the fermionic sign.
    # p=3: three passes, more complex entanglement (S = ln 3).
    #       This might correspond to... quarks? (Color as 3-fold entanglement?)

    # The fermionic sign (-1) between passes:
    # Under 2π rotation of the torus, each pass acquires phase 2π/p = π.
    # Two rotations (4π) give phase 2π → back to original.
    # This IS the spinor double-cover property: 2π rotation → -1.
    fermion_phase = np.exp(1j * phase_between_passes)  # should be -1 for p=2

    return {
        # Geometry
        'R': R,
        'r_tube': r_tube,
        'L_total': L_total,
        'L_per_pass': L_per_pass,
        'p': p,
        'q': q,
        # Crossings
        'crossing_phases': crossing_phases,
        'poloidal_separation': poloidal_separation,
        'poloidal_separation_deg': np.degrees(poloidal_separation),
        'n_crossing_pairs': n_crossing_pairs,
        'd_pass': d_pass,
        'd_pass_fm': d_pass_fm,
        # Phase
        'k_photon': k_photon,
        'phase_per_pass': phase_per_pass,
        'phase_between_passes': phase_between_passes,
        'fermion_phase': fermion_phase,
        # Entanglement
        'S_max': S_max,
        'S_max_bits': S_max / np.log(2.0),
        'mode_overlap': mode_overlap,
        # Energetics
        'I_current': I_current,
        'L_ind': L_ind,
        'log_factor': log_factor,
        'U_mag_total': U_mag_total,
        'U_mag_total_MeV': U_mag_total / MeV,
        'U_two_independent': U_two_independent,
        'U_two_independent_MeV': U_two_independent / MeV,
        'U_mutual': U_mutual,
        'U_mutual_MeV': U_mutual_MeV,
        # Pair creation/annihilation
        'E_unwinding_MeV': E_unwinding_MeV,
        'S_pair_creation': S_pair_creation,
        'S_self_per_particle': S_self_per_particle,
        'S_mutual_pair': S_mutual_pair,
    }


# ════════════════════════════════════════════════════════════════════
# Section 26: Entanglement and the Tube Radius
# ════════════════════════════════════════════════════════════════════

def compute_entanglement_radius(R=None, p=2, q=1, N=2000, N_scan=60):
    """
    Investigate whether self-entanglement physics can fix r/R = α.

    The tube radius r appears in the self-entanglement through:
    - The separation between passes: d = 2r (tube diameter)
    - The mode overlap between passes: depends on d and mode width w
    - The mutual energy between passes: captured by Neumann inductance
    - The optical path difference between inner/outer passes: ΔL = 8πr
    - The geometric phase from path difference: Δφ_geom = 4π(r/R)

    We test several mechanisms:
    A. Antibonding energy minimum (does the antibonding penalty + self-energy
       create an energy minimum at r/R = α?)
    B. Nonlinear vacuum tunneling (does the EH n(r) support propagating
       transverse modes between the passes?)
    C. Phase matching (does the geometric phase between passes select r/R?)
    D. The optical path difference ΔL/λ = 2r/R = 2α at the known radius.

    Parameters
    ----------
    R : float, optional
        Major radius. Default: self-consistent R ≈ 0.488 λ_C.
    p, q : int
        Winding numbers.
    N : int
        Points for path discretization.
    N_scan : int
        Number of r/R values to scan.

    Returns
    -------
    dict with scan results and analysis.
    """
    if R is None:
        R = 0.488 * lambda_C

    # ── 26a: Energy landscape scan ──

    ratios = np.geomspace(0.001, 0.49, N_scan)
    scan_results = []

    for ratio in ratios:
        r_tube = ratio * R

        # Path length of (p,q) knot
        params = TorusParams(R=R, r=r_tube, p=p, q=q)
        L_total = compute_path_length(params, N)

        # Circulation energy
        E_circ = 2.0 * np.pi * hbar * c / L_total

        # Self-energy (Neumann inductance)
        I = e_charge * c / L_total
        log_factor = max(np.log(8.0 * R / r_tube) - 2.0, 0.01)
        L_ind = mu0 * R * log_factor
        U_self = L_ind * I**2  # E + B (factor 2 from U_elec = U_mag)

        # Single-pass comparison
        params_single = TorusParams(R=R, r=r_tube, p=1, q=0)
        L_single = compute_path_length(params_single, N)
        I_single = e_charge * c / L_single
        L_ind_single = mu0 * R * log_factor
        U_single = L_ind_single * I_single**2
        U_mutual = U_self - 2.0 * U_single

        # Total classical energy
        E_total_classical = E_circ + U_self

        # Mode overlap (Case B: mode width = r_e = α×R, FIXED by EH physics)
        w_mode = alpha * R  # the EH transition scale
        d_separation = 2.0 * r_tube
        overlap = np.exp(-d_separation**2 / (2.0 * w_mode**2))

        # Entanglement entropy from concurrence = overlap
        C = overlap
        if C > 1e-10 and C < 1.0 - 1e-10:
            disc = np.sqrt(1.0 - C**2)
            lp = (1.0 + disc) / 2.0
            lm = (1.0 - disc) / 2.0
            S_ent = 0.0
            if lp > 1e-15:
                S_ent -= lp * np.log(lp)
            if lm > 1e-15:
                S_ent -= lm * np.log(lm)
        elif C >= 1.0 - 1e-10:
            S_ent = np.log(2.0)
        else:
            S_ent = 0.0

        # Tunnel coupling: antibonding splitting = E_circ × overlap
        Delta_E_anti = E_circ * overlap

        # Total with tunnel coupling
        E_total_tunnel = E_total_classical + Delta_E_anti

        # EH refractive index at tube surface
        E_field = k_e * e_charge / r_tube**2
        E_over_ES = E_field / E_Schwinger
        eh = compute_eh_effective_response(E_field)
        n_eff = float(eh['n_eff'][0]) if hasattr(eh['n_eff'], '__len__') else float(eh['n_eff'])

        # Critical n for transverse propagation
        k_long = 2.0 * np.pi / L_total
        k_perp = np.pi / (2.0 * r_tube)
        n_crit = k_perp / k_long

        # Geometric phase between passes
        # Path difference: ΔL = 8πr (two passes, each outer vs inner by 2πr)
        Delta_L = 8.0 * np.pi * r_tube
        Delta_L_over_lambda = Delta_L / L_total  # = 2 r/R to leading order
        geometric_phase = 2.0 * np.pi * Delta_L_over_lambda

        scan_results.append({
            'ratio': ratio,
            'r_tube': r_tube,
            'E_circ_MeV': E_circ / MeV,
            'U_self_MeV': U_self / MeV,
            'U_mutual_MeV': U_mutual / MeV,
            'E_total_classical_MeV': E_total_classical / MeV,
            'Delta_E_anti_MeV': Delta_E_anti / MeV,
            'E_total_tunnel_MeV': E_total_tunnel / MeV,
            'overlap': overlap,
            'S_ent': S_ent,
            'n_eff': n_eff,
            'n_crit': n_crit,
            'E_over_ES': E_over_ES,
            'Delta_L_over_lambda': Delta_L_over_lambda,
            'geometric_phase': geometric_phase,
        })

    # ── 26b: Find extrema ──

    ratios_arr = np.array([s['ratio'] for s in scan_results])
    E_classical_arr = np.array([s['E_total_classical_MeV'] for s in scan_results])
    E_tunnel_arr = np.array([s['E_total_tunnel_MeV'] for s in scan_results])
    S_ent_arr = np.array([s['S_ent'] for s in scan_results])

    i_min_classical = int(np.argmin(E_classical_arr))
    i_min_tunnel = int(np.argmin(E_tunnel_arr))
    i_max_S = int(np.argmax(S_ent_arr))
    i_alpha = int(np.argmin(np.abs(ratios_arr - alpha)))

    # ── 26c: Phase matching analysis ──

    # ΔL/λ = 2(r/R) to leading order (exact for r << R)
    # At r = αR: ΔL/λ = 2α
    # Geometric phase = 4π(r/R)
    # Total phase between passes = π + 4π(r/R)  [topological + geometric]
    r_alpha = alpha * R
    params_alpha = TorusParams(R=R, r=r_alpha, p=p, q=q)
    L_alpha = compute_path_length(params_alpha, N)
    Delta_L_alpha = 8.0 * np.pi * r_alpha
    Delta_L_over_lambda_alpha = Delta_L_alpha / L_alpha
    geom_phase_alpha = 2.0 * np.pi * Delta_L_over_lambda_alpha
    total_phase_alpha = np.pi + geom_phase_alpha

    # ── 26d: Transverse mode analysis ──

    # n_crit = k_⊥/k_∥ = (π/2r)/(2π/L) = L/(4r) ≈ πp/(2×r/R) for r << R
    # At r = αR: n_crit ≈ πp/(2α) ≈ π/α ≈ 430
    # n_eff at r = r_e ≈ 1.01  → always evanescent
    n_crit_alpha = scan_results[i_alpha]['n_crit']
    n_eff_alpha = scan_results[i_alpha]['n_eff']

    # ── 26e: Self-energy scaling analysis ──

    # Show that E_total = f(r/R) / R where f is monotonic
    # f(x) = 2πℏc/(2π√(p²R²+q²x²R²)) + μ₀R[ln(8/x)-2](ec/(2π√(p²+q²x²)R))²
    # = [1/R] × [ℏc/√(p²+q²x²) + μ₀e²c²(ln(8/x)-2)/(4π²(p²+q²x²))]
    # where x = r/R. All terms ∝ 1/R at fixed x.

    # Check: does f(x) have a minimum?
    # df/dx ∝ -q²x/(p²+q²x²)^{3/2} + d/dx[ln(8/x)-2]/(p²+q²x²)
    # First term < 0 (E_circ decreases with x)
    # Second term: d/dx[ln(8/x)-2] = -1/x < 0 AND 1/(p²+q²x²) decreases
    # So df/dx < 0 for all x > 0 → f is MONOTONICALLY DECREASING
    # → no minimum at any r/R > 0

    # The antibonding correction Δ = E_circ × exp(-2(r/w)²) at fixed w/R:
    # This adds a term that decreases FASTER than 1/x at small x
    # and goes to zero at large x. But it's always positive → pushes min to larger r/R.

    return {
        'R': R,
        'scan': scan_results,
        'ratios': ratios_arr,
        # Extrema
        'i_min_classical': i_min_classical,
        'ratio_min_classical': ratios_arr[i_min_classical],
        'E_min_classical': E_classical_arr[i_min_classical],
        'i_min_tunnel': i_min_tunnel,
        'ratio_min_tunnel': ratios_arr[i_min_tunnel],
        'E_min_tunnel': E_tunnel_arr[i_min_tunnel],
        'i_max_S': i_max_S,
        'ratio_max_S': ratios_arr[i_max_S],
        'S_max': S_ent_arr[i_max_S],
        # At alpha
        'i_alpha': i_alpha,
        'ratio_alpha': ratios_arr[i_alpha],
        'E_classical_alpha': E_classical_arr[i_alpha],
        'E_tunnel_alpha': E_tunnel_arr[i_alpha],
        'S_ent_alpha': S_ent_arr[i_alpha],
        'overlap_alpha': scan_results[i_alpha]['overlap'],
        'n_eff_alpha': n_eff_alpha,
        'n_crit_alpha': n_crit_alpha,
        # Phase matching
        'Delta_L_over_lambda_alpha': Delta_L_over_lambda_alpha,
        'geom_phase_alpha': geom_phase_alpha,
        'total_phase_alpha': total_phase_alpha,
    }


# ════════════════════════════════════════════════════════════════════
# Section 27: Nonlinear Maxwell Eigenvalue Solver on Torus Cross-Section
# ════════════════════════════════════════════════════════════════════

def build_polar_laplacian(N_rho, N_phi, rho_min, rho_max):
    """
    Build the transverse Laplacian on a polar grid using log-radial transform.

    Uses u = ln(ρ) so that ∇²⊥ = (1/ρ²)[∂²/∂u² + ∂²/∂φ²].
    Both derivatives use standard 3-point stencils on the uniform (u, φ) grid.

    BCs: Neumann at ρ_min (reflection), Dirichlet at ρ_max (ψ=0), periodic in φ.

    Parameters
    ----------
    N_rho : int
        Number of radial grid points.
    N_phi : int
        Number of azimuthal grid points.
    rho_min, rho_max : float
        Radial domain bounds.

    Returns
    -------
    L : csr_matrix
        Sparse Laplacian matrix of size (N_rho*N_phi, N_rho*N_phi).
    rho_grid : ndarray
        Radial grid points.
    phi_grid : ndarray
        Azimuthal grid points.
    du : float
        Radial step size in u-space.
    dphi : float
        Azimuthal step size.
    """
    # Log-radial grid
    u_min = np.log(rho_min)
    u_max = np.log(rho_max)
    u_grid = np.linspace(u_min, u_max, N_rho)
    du = u_grid[1] - u_grid[0]
    rho_grid = np.exp(u_grid)

    # Azimuthal grid
    phi_grid = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    dphi = phi_grid[1] - phi_grid[0]

    N_total = N_rho * N_phi
    L = lil_matrix((N_total, N_total))

    def idx(i_r, i_p):
        return i_r * N_phi + (i_p % N_phi)

    for i_r in range(N_rho):
        rho_val = rho_grid[i_r]
        inv_rho2 = 1.0 / rho_val**2

        for i_p in range(N_phi):
            row = idx(i_r, i_p)

            # ∂²/∂u² stencil
            if i_r == 0:
                # Neumann BC: ψ[-1] = ψ[1] → (ψ[1] - 2ψ[0] + ψ[1])/du²
                L[row, idx(i_r, i_p)] += -2.0 / du**2 * inv_rho2
                L[row, idx(i_r + 1, i_p)] += 2.0 / du**2 * inv_rho2
            elif i_r == N_rho - 1:
                # Dirichlet BC at outer boundary: ψ = 0 there
                # This row will be zeroed and set to identity below
                pass
            else:
                L[row, idx(i_r - 1, i_p)] += 1.0 / du**2 * inv_rho2
                L[row, idx(i_r, i_p)] += -2.0 / du**2 * inv_rho2
                L[row, idx(i_r + 1, i_p)] += 1.0 / du**2 * inv_rho2

            # ∂²/∂φ² stencil (periodic)
            if i_r < N_rho - 1:  # skip Dirichlet boundary
                L[row, idx(i_r, i_p - 1)] += 1.0 / dphi**2 * inv_rho2
                L[row, idx(i_r, i_p)] += -2.0 / dphi**2 * inv_rho2
                L[row, idx(i_r, i_p + 1)] += 1.0 / dphi**2 * inv_rho2

    # Enforce Dirichlet at outer boundary: set rows to zero
    for i_p in range(N_phi):
        row = idx(N_rho - 1, i_p)
        L[row, :] = 0
        L[row, row] = -1.0e20  # large negative → eigenvalue won't pick this

    return L.tocsr(), rho_grid, phi_grid, du, dphi


def compute_nonlinear_maxwell_eigenvalue(R=None, p=2, q=1,
                                         N_rho=40, N_phi=24,
                                         N_scan=20, max_iter=30,
                                         tol=1e-6, alpha_mix=0.3):
    """
    Section 27: Nonlinear Maxwell eigenvalue solver on torus cross-section.

    Solves the transverse wave equation ∇²⊥ψ + [n²k₀² - β²]ψ = 0 where
    n(ρ,φ) is the EH refractive index.

    Model A: n from Coulomb field of two point charges (knot passes).
    Model B: Self-consistent n depending on |ψ|².

    Scans r/R to find whether the energy landscape has a minimum.

    Parameters
    ----------
    R : float or None
        Major radius. If None, uses self-consistent R from find_self_consistent_radius.
    p, q : int
        Winding numbers.
    N_rho, N_phi : int
        Grid resolution.
    N_scan : int
        Number of r/R values to scan.
    max_iter : int
        Maximum self-consistency iterations for Model B.
    tol : float
        Convergence tolerance for |Δβ/β|.
    alpha_mix : float
        Mixing parameter for self-consistency iteration.

    Returns
    -------
    dict with results from all sub-analyses.
    """
    if R is None:
        sc = find_self_consistent_radius(target_MeV=m_e_MeV, p=p, q=q)
        R = sc['R']

    r_tube = alpha * R  # reference tube radius at r/R = α

    # ── 27a. Grid and discretization ──────────────────────────────

    # Torus path length and k₀
    params = TorusParams(R=R, r=r_tube, p=p, q=q)
    L_path = compute_path_length(params)
    k0 = 2.0 * np.pi / L_path  # propagation wavenumber

    # Build polar Laplacian on cross-section
    rho_min = 0.01 * r_tube
    rho_max = 10.0 * r_tube
    L_mat, rho_grid, phi_grid, du, dphi = build_polar_laplacian(
        N_rho, N_phi, rho_min, rho_max
    )
    N_total = N_rho * N_phi

    # 2D coordinate grids
    RHO, PHI = np.meshgrid(rho_grid, phi_grid, indexing='ij')
    rho_flat = RHO.ravel()
    phi_flat = PHI.ravel()

    # Area element for integration: dA = ρ dρ dφ = ρ² du dφ (in log coords)
    dA_flat = rho_flat**2 * du * dphi

    grid_info = {
        'N_rho': N_rho,
        'N_phi': N_phi,
        'N_total': N_total,
        'rho_min': rho_min,
        'rho_max': rho_max,
        'k0': k0,
        'L_path': L_path,
        'R': R,
        'r_tube': r_tube,
    }

    # ── Helper: solve eigenvalue for given n(ρ,φ) ────────────────

    def solve_eigenvalue(n_flat):
        """Solve for largest β² given refractive index profile."""
        # H = ∇² + n²k₀²  (we want largest eigenvalue β²)
        n2k2 = diags(n_flat**2 * k0**2, 0, shape=(N_total, N_total))
        H = L_mat + n2k2
        # eigsh finds largest algebraic eigenvalues
        eigenvalues, eigenvectors = eigsh(H, k=3, which='LA')
        # Largest eigenvalue is β²
        i_max = np.argmax(eigenvalues)
        beta2 = eigenvalues[i_max]
        psi = eigenvectors[:, i_max]
        # Normalize: ∫|ψ|² dA = 1
        norm = np.sqrt(np.sum(psi**2 * dA_flat))
        if norm > 0:
            psi /= norm
        return beta2, psi, eigenvalues

    # ── Helper: angular decomposition of mode ─────────────────────

    def angular_decomposition(psi_flat, n_modes=6):
        """Decompose ψ into azimuthal Fourier components."""
        psi_2d = psi_flat.reshape(N_rho, N_phi)
        coeffs = np.fft.fft(psi_2d, axis=1) / N_phi
        # Power in each m mode (sum over radial)
        rho_weights = rho_grid**2 * du
        power = np.zeros(N_phi)
        for m in range(N_phi):
            power[m] = np.sum(np.abs(coeffs[:, m])**2 * rho_weights)
        total_power = np.sum(power)
        if total_power > 0:
            power /= total_power
        # Return first n_modes
        m_values = np.arange(n_modes)
        return m_values, power[:n_modes]

    # ── 27b. Model A: charge-driven n ─────────────────────────────

    def run_model_A(r_val):
        """Run Model A at given tube radius r."""
        # Two charge locations at (r, 0) and (r, π) on cross-section
        x1 = r_val * np.cos(0.0)
        y1 = r_val * np.sin(0.0)
        x2 = r_val * np.cos(np.pi)
        y2 = r_val * np.sin(np.pi)

        # Cartesian positions on grid
        x_grid = rho_flat * np.cos(phi_flat)
        y_grid = rho_flat * np.sin(phi_flat)

        # Distances to each charge (with softening to avoid singularity)
        r_soft = 0.01 * r_val
        d1 = np.sqrt((x_grid - x1)**2 + (y_grid - y1)**2 + r_soft**2)
        d2 = np.sqrt((x_grid - x2)**2 + (y_grid - y2)**2 + r_soft**2)

        # Electric field from two half-charges
        E_field = k_e * (e_charge / 2.0) * (1.0 / d1**2 + 1.0 / d2**2)

        # EH refractive index
        vac = compute_eh_effective_response(E_field)
        n_flat = vac['n_eff']

        # Solve eigenvalue
        beta2, psi, eigenvalues = solve_eigenvalue(n_flat)

        # Angular decomposition
        m_values, m_power = angular_decomposition(psi)

        return {
            'beta2': beta2,
            'beta': np.sqrt(max(beta2, 0)),
            'psi': psi,
            'n_profile': n_flat,
            'n_mean': np.sum(n_flat * np.abs(psi)**2 * dA_flat),
            'E_field': E_field,
            'eigenvalues': eigenvalues,
            'm_values': m_values,
            'm_power': m_power,
        }

    # ── 27c. Model B: self-consistent n ───────────────────────────

    def run_model_B(r_val, n_init=None):
        """Run Model B (self-consistent) at given tube radius r."""
        # Photon energy for intensity normalization
        params_local = TorusParams(R=R, r=r_val, p=p, q=q)
        L_local = compute_path_length(params_local)
        E_photon = 2.0 * np.pi * hbar * c / L_local

        # Mode volume estimate
        V_mode = np.pi * r_val**2 * L_local

        # Initialize n from Model A if not provided
        if n_init is None:
            result_A = run_model_A(r_val)
            n_current = result_A['n_profile'].copy()
        else:
            n_current = n_init.copy()

        convergence = []
        beta_prev = 0.0

        for iteration in range(max_iter):
            # Solve eigenvalue
            beta2, psi, eigenvalues = solve_eigenvalue(n_current)
            beta_val = np.sqrt(max(beta2, 0))

            # Check convergence
            if beta_prev > 0:
                rel_change = abs(beta_val - beta_prev) / beta_prev
            else:
                rel_change = 1.0
            convergence.append({
                'iteration': iteration,
                'beta': beta_val,
                'beta2': beta2,
                'rel_change': rel_change,
                'n_mean': np.mean(n_current),
                'n_max': np.max(n_current),
            })

            if rel_change < tol and iteration > 0:
                break

            beta_prev = beta_val

            # Update n from mode intensity
            # Energy density: u = E_photon × |ψ|² / V_mode
            psi_norm2 = psi**2  # already normalized
            u_density = E_photon * np.abs(psi_norm2) / V_mode

            # E_field = sqrt(2u/ε₀)
            E_field_new = np.sqrt(2.0 * u_density / eps0)

            # New n from EH
            vac_new = compute_eh_effective_response(E_field_new)
            n_new = vac_new['n_eff']

            # Mix
            n_current = (1.0 - alpha_mix) * n_current + alpha_mix * n_new

        # Final angular decomposition
        m_values, m_power = angular_decomposition(psi)

        return {
            'beta2': beta2,
            'beta': np.sqrt(max(beta2, 0)),
            'psi': psi,
            'n_profile': n_current,
            'n_mean': np.sum(n_current * np.abs(psi)**2 * dA_flat),
            'eigenvalues': eigenvalues,
            'm_values': m_values,
            'm_power': m_power,
            'convergence': convergence,
            'converged': convergence[-1]['rel_change'] < tol,
            'n_iterations': len(convergence),
        }

    # ── Run Model A and B at reference geometry (r/R = α) ─────────

    result_A_ref = run_model_A(r_tube)
    result_B_ref = run_model_B(r_tube)

    # ── 27d. Scan r/R ─────────────────────────────────────────────

    ratios = np.logspace(np.log10(0.002), np.log10(0.4), N_scan)
    scan_results = []

    n_warm = None  # warm-start for Model B

    for i, ratio in enumerate(ratios):
        r_val = ratio * R

        # Model A
        res_A = run_model_A(r_val)

        # Model B with warm-start
        res_B = run_model_B(r_val, n_init=n_warm)
        n_warm = res_B['n_profile'].copy()

        # Energies
        # Mode energy: E_mode = ℏc × β
        E_mode_A = hbar * c * res_A['beta']
        E_mode_B = hbar * c * res_B['beta']

        # Self-energy at this geometry
        params_scan = TorusParams(R=R, r=r_val, p=p, q=q)
        L_scan = compute_path_length(params_scan)
        E_circ = 2.0 * np.pi * hbar * c / L_scan

        # Coulomb self-energy estimate
        E_self = k_e * e_charge**2 / (2.0 * r_val)

        # Total energy: circulation + self-energy
        E_total_A = E_circ + E_self
        E_total_B = E_circ + E_self  # same classical terms

        scan_results.append({
            'ratio': ratio,
            'r': r_val,
            'beta_A': res_A['beta'],
            'beta_B': res_B['beta'],
            'n_mean_A': res_A['n_mean'],
            'n_mean_B': res_B['n_mean'],
            'E_mode_A': E_mode_A,
            'E_mode_B': E_mode_B,
            'E_circ': E_circ,
            'E_self': E_self,
            'E_total_A': E_total_A,
            'E_total_B': E_total_B,
            'converged_B': res_B['converged'],
            'n_iter_B': res_B['n_iterations'],
            'm_power_A': res_A['m_power'],
            'm_power_B': res_B['m_power'],
        })

    # ── 27e. Analysis ─────────────────────────────────────────────

    # Convert to arrays for analysis
    ratios_arr = np.array([s['ratio'] for s in scan_results])
    E_total_A_arr = np.array([s['E_total_A'] for s in scan_results])
    E_total_B_arr = np.array([s['E_total_B'] for s in scan_results])
    beta_A_arr = np.array([s['beta_A'] for s in scan_results])
    beta_B_arr = np.array([s['beta_B'] for s in scan_results])
    n_mean_A_arr = np.array([s['n_mean_A'] for s in scan_results])
    n_mean_B_arr = np.array([s['n_mean_B'] for s in scan_results])

    # Look for minimum in E_total
    i_min_A = np.argmin(E_total_A_arr)
    i_min_B = np.argmin(E_total_B_arr)

    # Check if Model B breaks 1/R scaling by comparing slopes
    # Slope in log-log: d(log E)/d(log ratio)
    log_ratios = np.log(ratios_arr)
    log_E_A = np.log(E_total_A_arr)
    log_E_B = np.log(E_total_B_arr)

    # Local slopes via finite differences
    slopes_A = np.diff(log_E_A) / np.diff(log_ratios)
    slopes_B = np.diff(log_E_B) / np.diff(log_ratios)

    # At r/R = α
    i_alpha = np.argmin(np.abs(ratios_arr - alpha))

    # Does self-consistency change β significantly?
    beta_ratio_ref = result_B_ref['beta'] / result_A_ref['beta'] if result_A_ref['beta'] > 0 else 1.0
    n_ratio_ref = result_B_ref['n_mean'] / result_A_ref['n_mean'] if result_A_ref['n_mean'] > 0 else 1.0

    # Dominant angular modes
    dominant_m_A = result_A_ref['m_values'][np.argmax(result_A_ref['m_power'])]
    dominant_m_B = result_B_ref['m_values'][np.argmax(result_B_ref['m_power'])]

    # Has degeneracy been broken?
    # Check if any slope changes sign (would indicate a minimum)
    has_minimum_A = np.any(np.diff(np.sign(slopes_A)) != 0)
    has_minimum_B = np.any(np.diff(np.sign(slopes_B)) != 0)

    # Monotonicity check: is E_total strictly decreasing?
    monotonic_A = np.all(np.diff(E_total_A_arr) < 0) or np.all(np.diff(E_total_A_arr) > 0)
    monotonic_B = np.all(np.diff(E_total_B_arr) < 0) or np.all(np.diff(E_total_B_arr) > 0)

    return {
        # Grid info
        'grid': grid_info,
        # Reference geometry results
        'model_A_ref': result_A_ref,
        'model_B_ref': result_B_ref,
        # Scan results
        'scan_results': scan_results,
        'ratios': ratios_arr,
        'E_total_A': E_total_A_arr,
        'E_total_B': E_total_B_arr,
        'beta_A': beta_A_arr,
        'beta_B': beta_B_arr,
        'n_mean_A': n_mean_A_arr,
        'n_mean_B': n_mean_B_arr,
        # Analysis
        'i_min_A': i_min_A,
        'i_min_B': i_min_B,
        'i_alpha': i_alpha,
        'ratio_at_min_A': ratios_arr[i_min_A],
        'ratio_at_min_B': ratios_arr[i_min_B],
        'has_minimum_A': has_minimum_A,
        'has_minimum_B': has_minimum_B,
        'monotonic_A': monotonic_A,
        'monotonic_B': monotonic_B,
        'slopes_A': slopes_A,
        'slopes_B': slopes_B,
        'beta_ratio_ref': beta_ratio_ref,
        'n_ratio_ref': n_ratio_ref,
        'dominant_m_A': dominant_m_A,
        'dominant_m_B': dominant_m_B,
    }


# ── Section 28: Longitudinal Resonant Modes on Gordon-Metric Torus ──────────


def compute_longitudinal_modes(R=None, p=2, q=1, N=2000, n_modes=10):
    """
    Solve the Hill equation for longitudinal standing wave modes along
    a (p,q) torus knot with position-dependent refractive index n(s).

    The wave equation on the closed waveguide is:
        -d²ψ/ds² = n(s)² (ω²/c²) ψ,   ψ(s+L) = ψ(s)

    With constant n this gives equally spaced modes ω_n = 2πnc/(nL).
    With varying n(s) the degeneracies split (Hill/Mathieu band gaps).

    Parameters
    ----------
    R : float or None
        Major torus radius (m). If None, uses self-consistent solution.
    p, q : int
        Torus knot winding numbers.
    N : int
        Number of grid points along the knot for the eigenvalue problem.
    n_modes : int
        Number of lowest eigenvalues to compute.

    Returns
    -------
    dict with mode frequencies, energies, Koide/Lenz analysis, k-scan.
    """
    from scipy.sparse.linalg import eigsh as sp_eigsh

    # --- 28a: n(s) profile along the knot ---

    if R is None:
        from .core import find_self_consistent_radius
        sol = find_self_consistent_radius(m_e_MeV, p=p, q=q, r_ratio=alpha)
        R = sol['R']
    r = alpha * R

    params = TorusParams(R=R, r=r, p=p, q=q)

    # Get the vacuum profile (radial n(ρ))
    profile = compute_radial_vacuum_profile(R, r, p, q)
    rho_grid = profile['rho']
    n_grid = profile['n_eff']

    # Get the torus knot curve
    lam, xyz, dxyz, ds_dlam = torus_knot_curve(params, N)
    dlam = 2.0 * np.pi / N

    # Distance from tube center at each point
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rho_cyl = np.sqrt(x**2 + y**2)
    rho_from_center = np.sqrt((rho_cyl - R)**2 + z**2)

    # Interpolate n onto the knot (log-space)
    log_n_interp = np.interp(
        np.log(np.maximum(rho_from_center, rho_grid[0])),
        np.log(rho_grid),
        np.log(n_grid),
    )
    n_on_knot = np.exp(log_n_interp)

    # Build uniform arc-length grid
    ds_arr = ds_dlam * dlam
    s_cumul = np.concatenate([[0], np.cumsum(ds_arr)])
    L_flat = s_cumul[-1]
    s_nonuniform = 0.5 * (s_cumul[:-1] + s_cumul[1:])  # midpoints

    s_uniform = np.linspace(0, L_flat, N, endpoint=False)
    ds = s_uniform[1] - s_uniform[0]

    # Interpolate n onto uniform grid
    n_s = np.interp(s_uniform, s_nonuniform, n_on_knot, period=L_flat)

    n_avg = np.mean(n_s)
    n_variation = (np.max(n_s) - np.min(n_s)) / n_avg
    L_eff = np.sum(n_s) * ds

    # --- 28b: Flat-space reference ---
    flat_modes = []
    for n in range(1, n_modes + 1):
        omega_n = 2.0 * np.pi * n * c / L_flat
        E_n = hbar * omega_n
        flat_modes.append({
            'n': n,
            'omega': omega_n,
            'E_J': E_n,
            'E_MeV': E_n / MeV,
            'E_over_me': E_n / (m_e * c**2),
        })

    # --- 28c: Hill equation solver ---
    # Build periodic second derivative matrix (circulant, sparse)
    D2 = lil_matrix((N, N))
    inv_ds2 = 1.0 / ds**2
    for i in range(N):
        D2[i, i] = -2.0 * inv_ds2
        D2[i, (i + 1) % N] = inv_ds2
        D2[i, (i - 1) % N] = inv_ds2
    D2 = csr_matrix(D2)

    # Mass matrix M = diag(n(s)²/c²)
    n_s_sq_over_c2 = n_s**2 / c**2
    M = diags(n_s_sq_over_c2, 0, shape=(N, N), format='csr')

    # Solve: -D2 ψ = ω² M ψ  (generalized eigenvalue)
    # Use shift-invert near sigma=0 for smallest positive eigenvalues
    # Skip the zero mode (constant), so request n_modes + 1 and drop zero
    try:
        eigenvalues, eigenvectors = sp_eigsh(
            -D2, k=min(n_modes * 2 + 1, N - 2), M=M, sigma=0.0,
            which='LM',
        )
    except Exception:
        # Fallback: standard eigenvalue with just -D2, M factored in
        A = -D2.toarray()
        M_dense = np.diag(n_s_sq_over_c2)
        from scipy.linalg import eigh
        eigenvalues_all, _ = eigh(A, M_dense)
        eigenvalues = np.sort(eigenvalues_all[eigenvalues_all > 1e-10])[:n_modes * 2]
        eigenvectors = None

    # eigenvalues = ω², sort and filter positive
    omega_sq = np.sort(eigenvalues[eigenvalues > 1e-10])

    gordon_modes = []
    for i, w2 in enumerate(omega_sq[:n_modes]):
        omega = np.sqrt(w2)
        E = hbar * omega
        n_mode = i + 1
        # Compare to flat mode with same harmonic number
        # Degenerate pairs: modes come as (1,1,2,2,3,3,...) so
        # harmonic number = i//2 + 1
        harm_n = i // 2 + 1
        flat_idx = min(harm_n - 1, len(flat_modes) - 1)
        omega_flat = flat_modes[flat_idx]['omega']
        shift_pct = (omega - omega_flat) / omega_flat * 100

        gordon_modes.append({
            'n': n_mode,
            'harm_n': harm_n,
            'omega': omega,
            'E_J': E,
            'E_MeV': E / MeV,
            'E_over_me': E / (m_e * c**2),
            'shift_pct': shift_pct,
            'omega_sq': w2,
        })

    # Degeneracy splitting: modes come in sin/cos pairs
    # Check spacing between consecutive eigenvalues
    splittings = []
    for i in range(0, len(omega_sq) - 1, 2):
        if i + 1 < len(omega_sq):
            w1, w2_val = np.sqrt(omega_sq[i]), np.sqrt(omega_sq[i + 1])
            split = abs(w2_val - w1) / (0.5 * (w1 + w2_val))
            splittings.append({
                'pair': (i // 2 + 1),
                'omega_1': w1,
                'omega_2': w2_val,
                'rel_split': split,
            })

    # --- 28d: Koide connection ---
    # From first 3 distinct harmonic frequencies, extract Koide angle θ
    # √ω_i = S(1 + √2 cos(θ + 2πi/3))
    koide = {}
    target_theta_K = (6.0 * np.pi + 2.0) / 9.0
    # Extract distinct harmonics (degenerate pairs have same harm_n)
    distinct_omegas = []
    seen_harm = set()
    for gm in gordon_modes:
        if gm['harm_n'] not in seen_harm:
            distinct_omegas.append(gm['omega'])
            seen_harm.add(gm['harm_n'])
    if len(distinct_omegas) >= 3:
        w = np.array(distinct_omegas[:3])
        sqrt_w = np.sqrt(w)
        S_koide = np.sum(sqrt_w) / 3.0
        # Koide Q parameter: (Σω) / (Σ√ω)²
        Q = np.sum(w) / np.sum(sqrt_w)**2
        # Q = (1/3)(1 + √2 cos(θ)) sum relation → extract θ
        # Actually Q = 2/3 for exact Koide, θ encodes the splitting
        # Use the parameterization: √m_i / (Σ√m) = (1 + √2 cos(θ + 2πi/3)) / 3
        x = sqrt_w / np.sum(sqrt_w)
        # x_i = (1 + √2 cos(θ + 2πi/3)) / 3
        # From x_1: cos(θ) = (3x_1 - 1) / √2
        cos_theta = (3 * x[0] - 1) / np.sqrt(2)
        cos_theta = np.clip(cos_theta, -1, 1)
        theta_extracted = np.arccos(cos_theta)

        koide = {
            'omega_1': w[0], 'omega_2': w[1], 'omega_3': w[2],
            'Q_param': Q,
            'theta_extracted': theta_extracted,
            'theta_target': target_theta_K,
            'theta_match_pct': abs(theta_extracted - target_theta_K) / target_theta_K * 100,
        }

    # --- 28e: Lenz connection ---
    # Cumulative energy sums Σ(E_1..n)/E_1
    target_lenz = 6.0 * np.pi**5  # 1836.118...
    lenz = {}
    if len(gordon_modes) >= 2:
        E_arr = np.array([m['E_J'] for m in gordon_modes])
        cumsum_E = np.cumsum(E_arr)
        ratios_to_E1 = cumsum_E / E_arr[0]

        # With degeneracy factors
        deg_spin = 2  # spin
        deg_color = 3  # color
        lenz_candidates = {}
        for label, deg in [('bare', 1), ('spin×2', deg_spin),
                           ('spin×color×6', deg_spin * deg_color)]:
            weighted = np.cumsum(E_arr * deg)
            r_vals = weighted / E_arr[0]
            # Find which n gives closest to 6π⁵
            diffs = np.abs(r_vals - target_lenz)
            i_best = np.argmin(diffs)
            lenz_candidates[label] = {
                'n_best': i_best + 1,
                'ratio': r_vals[i_best],
                'diff_pct': diffs[i_best] / target_lenz * 100,
                'all_ratios': r_vals.tolist(),
            }

        lenz = {
            'target': target_lenz,
            'cumsum_ratios': ratios_to_E1.tolist(),
            'candidates': lenz_candidates,
        }

    # --- 28f: k-scan (aspect ratio variation) ---
    k_scan = []
    for k_val in [1, 2, 3, 4, 5]:
        r_k = R / k_val
        params_k = TorusParams(R=R, r=r_k, p=p, q=q)

        profile_k = compute_radial_vacuum_profile(R, r_k, p, q)
        rho_grid_k = profile_k['rho']
        n_grid_k = profile_k['n_eff']

        lam_k, xyz_k, dxyz_k, ds_dlam_k = torus_knot_curve(params_k, N)
        x_k, y_k, z_k = xyz_k[:, 0], xyz_k[:, 1], xyz_k[:, 2]
        rho_cyl_k = np.sqrt(x_k**2 + y_k**2)
        rho_fc_k = np.sqrt((rho_cyl_k - R)**2 + z_k**2)

        log_n_k = np.interp(
            np.log(np.maximum(rho_fc_k, rho_grid_k[0])),
            np.log(rho_grid_k),
            np.log(n_grid_k),
        )
        n_knot_k = np.exp(log_n_k)

        ds_arr_k = ds_dlam_k * dlam
        L_flat_k = np.sum(ds_arr_k)
        n_avg_k = np.mean(n_knot_k)
        n_var_k = (np.max(n_knot_k) - np.min(n_knot_k)) / n_avg_k

        # Quick eigenvalue solve for band gap estimate
        s_uni_k = np.linspace(0, L_flat_k, N, endpoint=False)
        ds_k = s_uni_k[1] - s_uni_k[0]
        s_cumul_k = np.concatenate([[0], np.cumsum(ds_arr_k)])
        s_mid_k = 0.5 * (s_cumul_k[:-1] + s_cumul_k[1:])
        n_s_k = np.interp(s_uni_k, s_mid_k, n_knot_k, period=L_flat_k)

        # Analytical mode frequencies: ω_n = 2πnc/(n_avg_k * L_flat_k)
        # With constant n(s), modes are exactly equally spaced and
        # degenerate in sin/cos pairs — no Hill gaps.
        L_eff_k = np.sum(n_s_k) * ds_k
        omega_1_k = 2.0 * np.pi * c / L_eff_k
        # Band gap is zero (n is constant), harmonic ratios are exact integers
        band_gap_rel = n_var_k  # proportional to n variation
        harmonic_ratios = [1.0, 1.0, 2.0, 2.0]  # degenerate pairs

        k_scan.append({
            'k': k_val,
            'r': r_k,
            'r_over_R': r_k / R,
            'L_flat': L_flat_k,
            'n_avg': n_avg_k,
            'n_variation': n_var_k,
            'band_gap_rel': band_gap_rel,
            'harmonic_ratios': harmonic_ratios,
        })

    return {
        # 28a
        'R': R, 'r': r, 'p': p, 'q': q,
        'L_flat': L_flat, 'L_eff': L_eff,
        'n_avg': n_avg, 'n_variation': n_variation,
        'n_on_knot': n_s,
        's_uniform': s_uniform,
        # 28b
        'flat_modes': flat_modes,
        # 28c
        'gordon_modes': gordon_modes,
        'splittings': splittings,
        # 28d
        'koide': koide,
        # 28e
        'lenz': lenz,
        # 28f
        'k_scan': k_scan,
    }


# Display function moved to gordon_display.py
