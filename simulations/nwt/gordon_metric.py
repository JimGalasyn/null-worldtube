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
    One-loop vacuum polarization function Π(q²) in QED.

    Π(q²) = -(2α/3π) ∫₀¹ dx x(1-x) ln(1 - q² x(1-x) / m_e²c²)

    For spacelike momentum transfer (q² > 0 in our convention,
    corresponding to static screening):
      - Π(q²) > 0 for q² > 0 (anti-screening: effective charge increases)
      - Π(q²) → 0 for q² → 0
      - Π(q²) ~ (α/3π) ln(q²/m²) for q² >> 4m² (logarithmic running)

    The sign convention: the screened charge is e²_eff = e² / (1 - Π),
    so Π > 0 means ANTI-screening (charge increases at short distance).

    Parameters
    ----------
    q2 : float or ndarray
        Squared momentum transfer (m⁻²), spacelike positive.

    Returns
    -------
    Pi : ndarray
        Vacuum polarization Π(q²) (dimensionless).
    """
    q2 = np.atleast_1d(np.asarray(q2, dtype=float))
    m2 = (m_e * c / hbar)**2  # electron mass² in momentum units (m⁻²)

    # Numerical integration over Feynman parameter x
    N_x = 200
    x = np.linspace(1e-6, 1.0 - 1e-6, N_x)
    dx = x[1] - x[0]

    Pi = np.zeros_like(q2)
    for j, q2_val in enumerate(q2):
        # Argument of the log: 1 - q² x(1-x) / m²
        z = q2_val * x * (1.0 - x) / m2
        # For spacelike q² > 0: z > 0, so argument = 1 - z
        # When z > 1, we're above threshold — add iε prescription
        # For static (spacelike) case, z can exceed 1 for large q²
        arg = 1.0 - z
        # Use log|arg| for the real part (imaginary part = pair production)
        log_arg = np.log(np.abs(np.maximum(arg, 1e-300)))
        integrand = x * (1.0 - x) * log_arg
        Pi[j] = -(2.0 * alpha / (3.0 * np.pi)) * np.trapz(integrand, x)

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


def compute_nonlocal_self_energy(R, r_tube, p=2, q_wind=1, N_q=2000):
    """
    Compute the electromagnetic self-energy of the torus using
    momentum-space screening with full Π(q²).

    The torus is a ring of charge e at major radius R with tube
    radius r_tube. The self-energy is:

    U_E = (e²/(4π² ε₀)) ∫₀^∞ dq [sin(qR)/(qR)]² × exp(-q²r²/2)

    The charge form factor [sin(qR)/(qR)]² is the angle-averaged
    Fourier transform of a ring at radius R, and exp(-q²r²/2) is the
    Gaussian tube smearing.

    Without the tube: U_E = k_e e²/(2R)  (electrostatic self-energy of a ring)
    With the tube: adds the ln(8R/r) factor.

    The magnetic self-energy at v=c equals the electric, so U_total = 2 × U_E.

    With vacuum polarization:
    U_screened = (e²/(4π² ε₀)) ∫₀^∞ dq [sin(qR)/(qR)]² × exp(-q²r²/2)/(1-Π(q²))

    The anti-screening (Π > 0) INCREASES the self-energy.

    Parameters
    ----------
    R, r_tube : float
        Torus radii (m).
    p, q_wind : int
        Winding numbers.
    N_q : int
        Number of momentum grid points.

    Returns
    -------
    dict with bare and screened self-energies.
    """
    lambda_C_bar = hbar / (m_e * c)

    # Momentum grid: from well below 1/R to well above 1/r_tube
    q_min = 0.01 / R
    q_max = 200.0 / r_tube
    q_grid = np.logspace(np.log10(q_min), np.log10(q_max), N_q)
    q2_grid = q_grid**2

    # Vacuum polarization
    Pi = compute_vacuum_polarization_pi(q2_grid)

    # Ring charge form factor (angle-averaged)
    qR = q_grid * R
    F_ring = np.where(qR > 1e-6, np.sin(qR) / qR, 1.0 - qR**2 / 6.0)

    # Tube smearing: Gaussian profile with RMS width r_tube
    qr = q_grid * r_tube
    F_tube = np.exp(-qr**2 / 2.0)

    # Combined form factor squared
    F2 = F_ring**2 * F_tube**2

    # Self-energy integrand:
    # U_E = (e²/(4π² ε₀)) ∫ dq F²(q)
    prefactor = e_charge**2 / (4.0 * np.pi**2 * eps0)

    integrand_bare = F2
    integrand_screened = F2 / (1.0 - Pi)

    U_E_bare = prefactor * np.trapz(integrand_bare, q_grid)
    U_E_screened = prefactor * np.trapz(integrand_screened, q_grid)

    # Magnetic self-energy equals electric at v = c (equipartition)
    # Total self-energy = 2 × U_E
    U_bare = 2.0 * U_E_bare
    U_screened = 2.0 * U_E_screened
    delta_U = U_screened - U_bare

    # Local EH self-energy for comparison
    eh_self = compute_effective_self_energy(R, r_tube, p, q_wind)

    # Neumann self-energy for comparison
    params_local = TorusParams(R=R, r=r_tube, p=p, q=q_wind)
    se_flat = compute_self_energy(params_local)

    # Spectral density: dU/dq (for analysis of where the correction lives)
    dU_dq_bare = 2.0 * prefactor * integrand_bare
    dU_dq_screened = 2.0 * prefactor * integrand_screened
    dU_dq_correction = dU_dq_screened - dU_dq_bare

    # Analytic check: bare ring self-energy without tube = k_e e²/(2R)
    U_ring_analytic = k_e * e_charge**2 / (2.0 * R)

    return {
        'q_grid': q_grid,
        'q2_grid': q2_grid,
        'Pi': Pi,
        'F2': F2,
        'F_ring': F_ring,
        'F_tube': F_tube,
        'U_E_bare_J': U_E_bare,
        'U_E_bare_MeV': U_E_bare / MeV,
        'U_bare_J': U_bare,
        'U_bare_MeV': U_bare / MeV,
        'U_screened_J': U_screened,
        'U_screened_MeV': U_screened / MeV,
        'delta_U_J': delta_U,
        'delta_U_MeV': delta_U / MeV,
        'delta_U_keV': delta_U / MeV * 1000,
        'U_ring_analytic_J': U_ring_analytic,
        'U_ring_analytic_MeV': U_ring_analytic / MeV,
        'U_Neumann_MeV': se_flat['U_total_MeV'],
        'U_EH_local_MeV': eh_self['U_self_eff_MeV'],
        'dU_dq_bare': dU_dq_bare,
        'dU_dq_screened': dU_dq_screened,
        'dU_dq_correction': dU_dq_correction,
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
# Step 9: Display
# ════════════════════════════════════════════════════════════════════

def print_gordon_metric_analysis():
    """
    Full Gordon metric analysis: vacuum profile, metric components,
    energy budget, and comparison of flat vs Gordon vs m_e.
    """
    print("=" * 70)
    print("  GORDON EFFECTIVE METRIC: NON-PERTURBATIVE EH VACUUM")
    print("  Confined vacuum polarization energy in the NWT torus")
    print("=" * 70)

    # Reference geometry: (2,1) torus knot with r/R = α
    p, q = 2, 1
    sol_flat = find_self_consistent_radius(m_e_MeV, p=p, q=q, r_ratio=alpha)
    R = sol_flat['R']
    r = alpha * R

    print(f"""
  Reference geometry (flat-space solution):
    (p, q) = ({p}, {q})       r/R = α = {alpha:.6f}
    R = {R:.4e} m = {R/lambda_C:.6f} λ_C = {R*1e15:.2f} fm
    r = {r:.4e} m = {r/lambda_C:.6f} λ_C = {r*1e15:.4f} fm
    E_flat = {sol_flat['E_total_MeV']:.6f} MeV   (m_e = {m_e_MeV:.6f} MeV)
""")

    # ════════════════════════════════════════════════════════════════
    # SECTION 1: Vacuum Profile
    # ════════════════════════════════════════════════════════════════
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  SECTION 1: RADIAL VACUUM PROFILE                              ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")

    profile = compute_radial_vacuum_profile(R, r, p, q)

    print(f"""
  Schwinger critical field: E_S = {E_Schwinger:.3e} V/m

  Vacuum polarization energy: {profile['U_vac_pol_MeV']:.6f} MeV
  Classical field energy:     {profile['U_classical_MeV']:.6f} MeV
  Ratio (vac pol / classical): {profile['vac_pol_ratio']:.2f}×
""")

    print(f"  {'ρ/r':>8s}  {'E/E_S':>10s}  {'ε_eff/ε₀':>10s}  {'μ_eff/μ₀':>10s}"
          f"  {'n_eff':>10s}  {'regime':>10s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    rho = profile['rho']
    # Sample points
    indices = np.linspace(0, len(rho) - 1, 15, dtype=int)
    for i in indices:
        regime_str = profile['regime'][i]
        print(f"  {rho[i]/r:8.3f}  {profile['E_over_ES'][i]:10.4f}"
              f"  {profile['eps_eff'][i]/eps0:10.6f}"
              f"  {profile['mu_eff'][i]/mu0:10.6f}"
              f"  {profile['n_eff'][i]:10.6f}  {regime_str:>10s}")

    # ════════════════════════════════════════════════════════════════
    # SECTION 2: Gordon Metric
    # ════════════════════════════════════════════════════════════════
    print()
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  SECTION 2: GORDON METRIC COMPONENTS                           ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")

    metric = compute_gordon_metric(R, r, p, q)

    print(f"""
  Gordon metric: g_eff^μν = η^μν + (1 - 1/n²) u^μ u^ν

  At tube surface (ρ = r):
    n_eff = {metric['n_at_surface']:.6f}
    c_eff = c/{metric['n_at_surface']:.4f} = {metric['c_eff_at_surface']:.4e} m/s
    c_eff/c = {metric['c_eff_at_surface']/c:.6f}

  At tube center (ρ = 0.5r):
    n_eff = {metric['n_at_center']:.6f}

  Frame-dragging (Kerr analogy):
    ω_circ = {metric['omega_circ']:.4e} rad/s
    Ω_drag (at surface)   = {metric['Omega_drag_surface']:.4e} rad/s
    Ω_drag (vol-averaged) = {metric['Omega_drag_vol_avg']:.4e} rad/s

    E_drag = -p × ℏ × Ω_drag  (co-rotating → NEGATIVE)
    Photon co-rotates with polarized vacuum (p = {p}):
      E_drag (surface)    = {metric['U_drag_MeV']*1000:+.4f} keV
      E_drag (vol-avg)    = {metric['U_drag_vol_MeV']*1000:+.4f} keV
""")

    print(f"  {'ρ/r':>8s}  {'n_eff':>10s}  {'c_eff/c':>10s}"
          f"  {'Ω_drag':>12s}  {'g_tθ':>12s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}")

    rho_m = metric['rho']
    indices_m = np.linspace(0, len(rho_m) - 1, 12, dtype=int)
    for i in indices_m:
        print(f"  {rho_m[i]/r:8.3f}  {metric['n_eff'][i]:10.6f}"
              f"  {metric['c_eff'][i]/c:10.6f}"
              f"  {metric['Omega_drag'][i]:12.4e}"
              f"  {metric['g_t_theta'][i]:12.4e}")

    # ════════════════════════════════════════════════════════════════
    # SECTION 3: Modified Circulation Energy
    # ════════════════════════════════════════════════════════════════
    print()
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  SECTION 3: EFFECTIVE CIRCULATION ENERGY                       ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")

    circ = compute_effective_circulation(R, r, p, q)

    print(f"""
  Flat path length:     L_flat = {circ['L_flat']:.6e} m
  Effective path length: L_eff = {circ['L_eff']:.6e} m
  Ratio L_eff/L_flat = {circ['L_ratio']:.6f}  (average n = {circ['n_avg']:.6f})

  Flat circulation energy:     E_circ_flat = {circ['E_circ_flat_MeV']:.6f} MeV
  Effective circulation energy: E_circ_eff = {circ['E_circ_eff_MeV']:.6f} MeV
  Reduction: {circ['E_reduction_MeV']*1000:.4f} keV ({circ['E_reduction_pct']:.4f}%)

  Mechanism: n > 1 inside torus → photon slows → longer effective path
             → lower resonant frequency → lower energy
""")

    # ════════════════════════════════════════════════════════════════
    # SECTION 4: Modified Self-Energy
    # ════════════════════════════════════════════════════════════════
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  SECTION 4: MODIFIED SELF-ENERGY                               ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")

    self_e = compute_effective_self_energy(R, r, p, q)

    print(f"""
  Cross-section averaged vacuum response:
    ε_eff_avg / ε₀ = {self_e['eps_ratio']:.6f}
    μ_eff_avg / μ₀ = {self_e['mu_ratio']:.6f}

  Flat-space self-energy:    U_self_flat = {self_e['U_self_flat_MeV']:.6f} MeV
  Gordon self-energy:        U_self_eff  = {self_e['U_self_eff_MeV']:.6f} MeV

  Inductance: L_flat = {self_e['L_ind_flat']:.4e} H → L_eff = {self_e['L_ind_eff']:.4e} H
""")

    # ════════════════════════════════════════════════════════════════
    # SECTION 5: Total Energy Budget
    # ════════════════════════════════════════════════════════════════
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  SECTION 5: TOTAL ENERGY BUDGET — TWO INTERPRETATIONS          ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")

    result = compute_gordon_total_energy(R, r, p, q)

    print(f"""
  The self-energy question: does U_vac_pol double-count U_self_eff?

  U_self_eff already uses μ_eff, ε_eff (the polarized vacuum response).
  The inductance L_eff = μ_eff × R × [ln(8R/r) - 2] already encodes
  the energy cost of polarizing the vacuum. Adding U_vac_pol = ∫(ε_eff - ε₀)E²/2 dV
  on top may count the same energy twice.

  ┌───────────────────────────────────────────────────────────────────┐
  │  INTERPRETATION A: "Additive" (U_vac_pol is independent)         │
  │                                                                   │
  │  E_circ_eff (modified circulation)    {result['E_circ_eff_MeV']:12.6f} MeV        │
  │  U_self_eff (modified self-energy)    {result['U_self_eff_MeV']:12.6f} MeV        │
  │  U_vac_pol  (vacuum polarization)     {result['U_vac_pol_MeV']:12.6f} MeV        │
  │  U_drag     (frame-dragging)          {result['U_drag_MeV']:12.6f} MeV        │
  │  ───────────────────────────────────────────────────────────────  │
  │  E_total (additive)                   {result['E_additive_MeV']:12.6f} MeV        │
  └───────────────────────────────────────────────────────────────────┘

  ┌───────────────────────────────────────────────────────────────────┐
  │  INTERPRETATION B: "Dressed" (U_self_eff already includes it)    │
  │                                                                   │
  │  E_circ_eff (modified circulation)    {result['E_circ_eff_MeV']:12.6f} MeV        │
  │  U_self_eff (dressed self-energy)     {result['U_self_eff_MeV']:12.6f} MeV        │
  │  U_drag     (frame-dragging)          {result['U_drag_MeV']:12.6f} MeV        │
  │  ───────────────────────────────────────────────────────────────  │
  │  E_total (dressed)                    {result['E_dressed_MeV']:12.6f} MeV        │
  └───────────────────────────────────────────────────────────────────┘
""")

    # ════════════════════════════════════════════════════════════════
    # SECTION 6: Comparison
    # ════════════════════════════════════════════════════════════════
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  SECTION 6: COMPARISON — FLAT vs GORDON vs m_e                 ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")

    E_flat = result['E_flat_MeV']
    E_additive = result['E_additive_MeV']
    E_dressed = result['E_dressed_MeV']
    gap_flat = (E_flat - m_e_MeV) * 1000  # keV
    gap_additive = result['gap_additive_keV']
    gap_dressed = result['gap_dressed_keV']

    print(f"""
  At flat-space self-consistent R = {R/lambda_C:.6f} λ_C:

  E_flat     = {E_flat:.6f} MeV    gap = {gap_flat:+.4f} keV  ({(E_flat - m_e_MeV)/m_e_MeV*100:+.4f}%)
  E_additive = {E_additive:.6f} MeV    gap = {gap_additive:+.4f} keV  ({(E_additive - m_e_MeV)/m_e_MeV*100:+.4f}%)
  E_dressed  = {E_dressed:.6f} MeV    gap = {gap_dressed:+.4f} keV  ({(E_dressed - m_e_MeV)/m_e_MeV*100:+.4f}%)
  m_e        = {m_e_MeV:.6f} MeV    (target)

  Energy fractions (dressed):
    Circulation:    {result['E_circ_eff_MeV']/E_dressed*100:6.2f}%
    Self-energy:    {result['U_self_eff_MeV']/E_dressed*100:6.2f}%
    Frame-dragging: {result['U_drag_MeV']/E_dressed*100:6.2f}%
""")

    # ════════════════════════════════════════════════════════════════
    # SECTION 7: Self-consistent Gordon radius (dressed)
    # ════════════════════════════════════════════════════════════════
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  SECTION 7: SELF-CONSISTENT GORDON RADIUS                      ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")
    print()
    print("  Using dressed interpretation (E = E_circ_eff + U_self_eff + U_drag)")
    print("  Finding R where E_total(dressed) = m_e ...")

    sol = find_gordon_self_consistent_radius(m_e_MeV, p=p, q=q, r_ratio=alpha)
    if sol is not None:
        bd = sol['breakdown']
        print(f"""
  Solution found:
    R = {sol['R']:.6e} m = {sol['R_over_lambda_C']:.6f} λ_C = {sol['R_femtometers']:.2f} fm
    r = {sol['r']:.6e} m = {sol['r_over_lambda_C']:.6f} λ_C = {sol['r_femtometers']:.4f} fm
    E_total = {sol['E_total_MeV']:.8f} MeV  (target: {m_e_MeV:.8f})
    Match: {sol['match_ppm']:.1f} ppm

  Flat-space R/λ_C = {sol_flat['R_over_lambda_C']:.6f}
  Dressed R/λ_C    = {sol['R_over_lambda_C']:.6f}
  Shift: {(sol['R_over_lambda_C'] - sol_flat['R_over_lambda_C'])/sol_flat['R_over_lambda_C']*100:.4f}%

  Energy budget at self-consistent R:
    E_circ_eff = {bd['E_circ_eff_MeV']:.6f} MeV  ({bd['E_circ_eff_MeV']/m_e_MeV*100:.2f}% of m_e)
    U_self_eff = {bd['U_self_eff_MeV']:.6f} MeV  ({bd['U_self_eff_MeV']/m_e_MeV*100:.2f}% of m_e)
    U_drag     = {bd['U_drag_MeV']:.6f} MeV  ({bd['U_drag_MeV']/m_e_MeV*100:.2f}% of m_e)

  (For reference, U_vac_pol at this R = {bd['U_vac_pol_MeV']:.6f} MeV — NOT included)
""")
    else:
        print("  No solution found in search range.")

    # ════════════════════════════════════════════════════════════════
    # Summary box
    # ════════════════════════════════════════════════════════════════
    print("  ┌──────────────────────────────────────────────────────────────────┐")
    print("  │  SUMMARY                                                        │")
    print("  │                                                                 │")
    print(f"  │  Flat-space:  E = {E_flat:.6f} MeV  gap = {gap_flat:+8.4f} keV          │")
    print(f"  │  Additive:   E = {E_additive:.6f} MeV  gap = {gap_additive:+8.4f} keV          │")
    print(f"  │  Dressed:    E = {E_dressed:.6f} MeV  gap = {gap_dressed:+8.4f} keV          │")
    print(f"  │  m_e:        E = {m_e_MeV:.6f} MeV  (target)                     │")
    print(f"  │                                                                 │")
    direction = "below" if gap_dressed < 0 else "above"
    print(f"  │  Dressed E is {abs(gap_dressed):.4f} keV {direction} m_e"
          f" ({gap_dressed/m_e_MeV/10:+.4f}%)             │")
    print(f"  │  Gordon corrections shift E by {gap_dressed - gap_flat:+.4f} keV"
          f" from flat-space    │")
    print(f"  │                                                                 │")
    if sol is not None:
        print(f"  │  Self-consistent dressed R/λ_C = {sol['R_over_lambda_C']:.6f}"
              f"                    │")
    print("  └──────────────────────────────────────────────────────────────────┘")

    # ════════════════════════════════════════════════════════════════
    # SECTION 8: Effective Viscosity Profile
    # ════════════════════════════════════════════════════════════════
    print()
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  SECTION 8: EFFECTIVE VISCOSITY OF THE QED VACUUM              ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")

    visc = compute_effective_viscosity_profile(R, r, p, q)
    print(f"""
  The imaginary part of the Euler-Heisenberg effective action gives the
  Schwinger pair production rate Γ, which acts as dissipation. This defines
  an effective shear viscosity: η_eff ~ u_field / Γ, where u_field is the
  field energy density and Γ is the decay rate of the field.

  Physical picture: the vacuum is a non-Newtonian fluid.
    - Far from the torus (E << E_S): η → ∞ (inviscid, exponentially suppressed)
    - At tube surface (E ~ E_S): finite viscosity, pair production active
    - Inside tube (E >> E_S): viscosity drops as Γ ~ E⁴ (superfluid-like runaway)
  This is a "shear-thinning" vacuum: stronger fields → lower viscosity.

  Reynolds number Re = ρ v L / η analogues:
    ρ → u_field/c², v → c, L → r (tube radius)

  Schwinger critical field: E_S = {E_Schwinger:.3e} V/m
  Schwinger rate prefactor: (αE²)/(π²) × exp(-πE_S/E)
""")

    print(f"  {'ρ/r':>8s}  {'E/E_S':>10s}  {'Γ (s⁻¹m⁻³)':>14s}  {'η_eff (Pa·s)':>14s}"
          f"  {'Re_eff':>10s}  {'regime':>10s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*14}  {'─'*14}  {'─'*10}  {'─'*10}")

    rho_v = visc['rho']
    indices_v = np.linspace(0, len(rho_v) - 1, 15, dtype=int)
    for i in indices_v:
        Gamma_i = visc['Gamma_Schwinger'][i]
        eta_i = visc['eta_eff'][i]
        Re_i = visc['Re_eff'][i]
        regime_i = visc['regime'][i]
        G_str = f"{Gamma_i:14.4e}" if Gamma_i > 1e-300 else "          ~0  "
        eta_str = f"{eta_i:14.4e}" if eta_i < 1e300 else "          ~∞  "
        Re_str = f"{Re_i:10.4e}" if Re_i > 1e-300 else "       ~0 "
        print(f"  {rho_v[i]/r:8.3f}  {visc['E_over_ES'][i]:10.4f}"
              f"  {G_str}  {eta_str}  {Re_str}  {regime_i:>10s}")

    print(f"""
  Integrated quantities:
    Total dissipation power:  P_diss = {visc['P_diss_W']:.4e} W = {visc['P_diss_MeV_per_s']:.4e} MeV/s
    Resonator Q factor:       Q = E_stored / (P_diss × T_circ)
      E_stored (dressed):     {E_dressed:.6f} MeV
      T_circ = 2π/ω_circ:    {visc['T_circ']:.4e} s
      Q = {visc['Q_factor']:.4e}
      Lifetime τ = Q × T_circ: {visc['tau_s']:.4e} s

    Viscous boundary layer (where η drops from ∞ to finite):
      ρ_visc / r ≈ {visc['rho_viscous_boundary']/r:.4f}  (E/E_S ≈ 1 at this radius)
      → The torus has a thin viscous core surrounded by inviscid vacuum
      → Exactly like a real vortex: viscous core + inviscid exterior
""")

    # ════════════════════════════════════════════════════════════════
    # SECTION 9: Effective Potential — EH vs Uehling
    # ════════════════════════════════════════════════════════════════
    print()
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  SECTION 9: EFFECTIVE POTENTIAL — NONLINEAR EH vs UEHLING      ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")

    pot = compute_potential_comparison()
    rho_p = pot['rho']
    lambda_C_bar = pot['lambda_C_bar']
    r_classical = pot['r_classical']

    print(f"""
  The Uehling potential is the standard one-loop (perturbative) QED vacuum
  polarization correction to Coulomb. It is the LINEARIZED EH response.

  The full nonlinear EH effective potential self-consistently accounts for
  the position-dependent dielectric: ε_eff(E(ρ)) × E(ρ) × 4πρ² = e.

  If the nonlinear EH reproduces known QED at long distance and departs
  at short distance, the NWT vacuum structure is consistent with QED
  but predicts NEW short-distance physics.

  Key scales:
    r_e (classical electron radius) = {r_classical:.4e} m = {r_classical*1e15:.4f} fm
    ρ_Schwinger (E = E_S boundary)  = {pot['rho_Schwinger']:.4e} m = {pot['rho_Schwinger']*1e15:.2f} fm
    ƛ_C (reduced Compton wavelength) = {lambda_C_bar:.4e} m = {lambda_C_bar*1e15:.2f} fm
    ρ_Schwinger / ƛ_C = √α = {pot['sqrt_alpha']:.6f}  (geometric mean of r_e and ƛ_C)
""")

    # Table header
    print(f"  {'ρ (fm)':>10s}  {'ρ/ƛ_C':>8s}  {'E/E_S':>8s}  {'ε_r':>8s}"
          f"  {'δU_EH (eV)':>12s}  {'δU_Uehl (eV)':>12s}  {'ratio':>8s}  {'α_eff':>10s}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}"
          f"  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*10}")

    # Sample ~20 representative points spanning all regimes
    indices_p = np.unique(np.logspace(
        0, np.log10(len(rho_p) - 1), 22, dtype=int
    ))
    for i in indices_p:
        rho_fm = rho_p[i] * 1e15
        rho_lc = rho_p[i] / lambda_C_bar
        x_es = pot['E_over_ES'][i]
        eps_r = pot['eps_r'][i]
        dU_eh = pot['delta_U_EH_eV'][i]
        dU_u = pot['delta_U_Uehling_eV'][i]
        rat = pot['ratio_EH_to_Uehling'][i]
        a_eff = pot['alpha_eff'][i]

        # Format with appropriate precision
        dU_eh_str = f"{dU_eh:12.4e}" if abs(dU_eh) > 1e-30 else "        ~0  "
        dU_u_str = f"{dU_u:12.4e}" if abs(dU_u) > 1e-30 else "        ~0  "
        rat_str = f"{rat:8.4f}" if not np.isnan(rat) and abs(rat) < 1e6 else "     ---"

        print(f"  {rho_fm:10.4f}  {rho_lc:8.4f}  {x_es:8.3f}  {eps_r:8.4f}"
              f"  {dU_eh_str}  {dU_u_str}  {rat_str}  {a_eff:10.6f}")

    # Agreement analysis
    rho_agree = pot['rho_agreement']
    if not np.isnan(rho_agree):
        print(f"""
  Agreement between EH and Uehling (within 10%):
    Begins at ρ ≈ {rho_agree:.4e} m = {rho_agree*1e15:.2f} fm
    ρ/ƛ_C ≈ {rho_agree/lambda_C_bar:.4f}
""")
    else:
        print("""
  No region found where EH and Uehling agree to within 10%.
  (May need finer grid or wider range.)
""")

    # Summary of regimes
    # Find where EH correction is strongest
    abs_dU = np.abs(pot['delta_U_EH_eV'])
    max_idx = np.argmax(abs_dU)
    rel_EH = pot['rel_EH']

    print(f"  Correction regimes:")
    print(f"    ρ < r_e ({r_classical*1e15:.2f} fm):  STRONG nonlinear — ε_r >> 1, α_eff >> α")
    print(f"    r_e < ρ < ρ_S ({pot['rho_Schwinger']*1e15:.1f} fm): nonlinear transition")
    print(f"    ρ_S < ρ < ƛ_C ({lambda_C_bar*1e15:.0f} fm):  perturbative — EH ≈ Uehling")
    print(f"    ρ > ƛ_C:  both exponentially suppressed")
    print()
    print(f"  Peak |δU_EH| = {abs_dU[max_idx]:.4e} eV at ρ = {rho_p[max_idx]*1e15:.4f} fm")
    print(f"  Max relative correction |δU/U| = {np.max(np.abs(rel_EH)):.6f}")
    print(f"    = {np.max(np.abs(rel_EH))/alpha*100:.2f}% of α — compare to Uehling ~ α/3π ≈ {alpha/(3*np.pi):.6f}")
    print()

    # ════════════════════════════════════════════════════════════════
    # SECTION 10: Nonlocal Vacuum Polarization — Momentum Space
    # ════════════════════════════════════════════════════════════════
    print()
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  SECTION 10: NONLOCAL Π(q²) — MOMENTUM-SPACE TREATMENT        ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")

    print(f"""
  The local EH dielectric ε(E(ρ)) always SCREENS (δU > 0 at short range).
  The full QED vacuum polarization Π(q²) ANTI-SCREENS at high q:
    e²_eff(q) = e² / (1 - Π(q²)),  Π > 0 for spacelike q²

  This anti-screening INCREASES the self-energy at short distance.

  Strategy: compute the torus self-energy in momentum space using
  the full one-loop Π(q²), with the torus form factor suppressing
  UV modes above 1/r_tube.
""")

    # Nonlocal self-energy at the reference geometry
    print("  Computing nonlocal self-energy at reference geometry...")
    nonlocal_se = compute_nonlocal_self_energy(R, r, p, q)

    q_g = nonlocal_se['q_grid']
    Pi_g = nonlocal_se['Pi']
    lambda_C_bar_nl = nonlocal_se['lambda_C_bar']

    print(f"""
  Torus self-energy in momentum space:
    U_E = (e²/4π²ε₀) ∫ dq [sin(qR)/(qR)]² × exp(-q²r²/2)
    U_total = 2 × U_E  (electric + magnetic at v = c)

  Analytic check: ring without tube = k_e e²/(2R) = {nonlocal_se['U_ring_analytic_MeV']:.6f} MeV
  Numerical bare (with tube cutoff)  = {nonlocal_se['U_E_bare_MeV']:.6f} MeV (electric)
  Neumann inductance (position-space) = {nonlocal_se['U_Neumann_MeV']:.6f} MeV

  Vacuum polarization Π(q²) profile:
""")
    print(f"  {'q × ƛ_C':>10s}  {'Π(q²)':>12s}  {'1/(1-Π)':>10s}"
          f"  {'|F_ring|²':>10s}  {'|F_tube|²':>10s}  {'|F|²':>10s}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    indices_nl = np.unique(np.logspace(0, np.log10(len(q_g) - 1), 20, dtype=int))
    for i in indices_nl:
        q_lc = q_g[i] * lambda_C_bar_nl
        screening = 1.0 / (1.0 - Pi_g[i])
        F_r2 = nonlocal_se['F_ring'][i]**2
        F_t2 = nonlocal_se['F_tube'][i]**2
        print(f"  {q_lc:10.4f}  {Pi_g[i]:12.6e}  {screening:10.6f}"
              f"  {F_r2:10.4e}  {F_t2:10.4e}  {nonlocal_se['F2'][i]:10.4e}")

    print(f"""
  Self-energy results (total = electric + magnetic):
    U_bare (ring + tube, no Π)   = {nonlocal_se['U_bare_MeV']:.6f} MeV  = {nonlocal_se['U_bare_MeV']*1000:.4f} keV
    U_screened (with Π(q²))      = {nonlocal_se['U_screened_MeV']:.6f} MeV  = {nonlocal_se['U_screened_MeV']*1000:.4f} keV
    δU = U_screened - U_bare     = {nonlocal_se['delta_U_keV']:+.6f} keV
    Neumann (position-space)     = {nonlocal_se['U_Neumann_MeV']:.6f} MeV  = {nonlocal_se['U_Neumann_MeV']*1000:.4f} keV
    EH local (from Section 4)    = {nonlocal_se['U_EH_local_MeV']:.6f} MeV  = {nonlocal_se['U_EH_local_MeV']*1000:.4f} keV

  The Π(q²) correction matters at q × ƛ_C ≳ 1 (momentum ~ m_e c).
  But the ring form factor [sin(qR)/(qR)]² suppresses modes above
  q ~ 1/R, i.e., q × ƛ_C ~ ƛ_C/R = {lambda_C_bar_nl/R:.4f}.
  At that scale, Π ~ {Pi_g[np.searchsorted(q_g, 1.0/R)]:.4e} — negligible!

  The anti-screening lives at q × ƛ_C ~ 1-10, where |F_ring|² ~ {nonlocal_se['F_ring'][np.searchsorted(q_g * lambda_C_bar_nl, 1.0)]**2:.4e}.
  The tube cutoff exp(-q²r²/2) kills modes above q ~ 1/r (q×ƛ_C ~ {lambda_C_bar_nl/r:.1f}).
""")

    # Full energy budget comparison
    print("  Computing full nonlocal energy budget...")
    budget = compute_full_nonlocal_energy_budget(R, r, p, q)

    print(f"""
  ┌───────────────────────────────────────────────────────────────────┐
  │  ENERGY BUDGET COMPARISON                                        │
  │                                                                   │
  │  A) Local EH dressed (from Section 5):                           │
  │     E_circ_eff      = {budget['E_circ_eff_MeV']:12.6f} MeV                      │
  │     U_self (EH)     = {budget['U_self_EH_MeV']:12.6f} MeV                      │
  │     U_drag          = {budget['U_drag_MeV']:12.6f} MeV                      │
  │     ─────────────────────────────────────                        │
  │     E_dressed       = {budget['E_dressed_MeV']:12.6f} MeV  gap = {budget['gap_dressed_keV']:+.4f} keV  │
  │                                                                   │
  │  B) Hybrid (nonlocal self-energy):                               │
  │     E_circ_eff      = {budget['E_circ_eff_MeV']:12.6f} MeV  (same — local EH)   │
  │     U_self (Π(q²))  = {budget['U_self_nonlocal_MeV']:12.6f} MeV  (momentum-space) │
  │     U_drag          = {budget['U_drag_MeV']:12.6f} MeV  (same — local EH)   │
  │     ─────────────────────────────────────                        │
  │     E_hybrid        = {budget['E_hybrid_MeV']:12.6f} MeV  gap = {budget['gap_hybrid_keV']:+.4f} keV  │
  │                                                                   │
  │  m_e                = {m_e_MeV:12.6f} MeV  (target)               │
  └───────────────────────────────────────────────────────────────────┘

  The nonlocal Π(q²) correction to self-energy: {budget['delta_U_nonlocal_keV']:+.6f} keV
  (Positive = anti-screening adds energy; this should help close the gap)

  Remaining gap: {budget['gap_hybrid_keV']:+.4f} keV ({budget['gap_hybrid_pct']:+.4f}% of m_e)
""")

    # Spectral analysis
    print("  Self-energy spectral density dU/dq:")
    print(f"  {'q × ƛ_C':>8s}  {'dU/dq bare':>14s}  {'dU/dq screened':>14s}"
          f"  {'dU/dq correction':>16s}")
    print(f"  {'─'*8}  {'─'*14}  {'─'*14}  {'─'*16}")

    dU_bare = nonlocal_se['dU_dq_bare']
    dU_scr = nonlocal_se['dU_dq_screened']
    dU_corr = nonlocal_se['dU_dq_correction']
    for i in indices_nl:
        q_lc = q_g[i] * lambda_C_bar_nl
        print(f"  {q_lc:8.4f}  {dU_bare[i]:14.4e}  {dU_scr[i]:14.4e}"
              f"  {dU_corr[i]:16.4e}")

    print()
    print("=" * 70)
