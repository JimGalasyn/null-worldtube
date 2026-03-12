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

    # Nonlocal self-energy at the reference geometry — all four tube models
    print("  Computing nonlocal self-energy (4 tube cross-section models)...")
    nonlocal_se = compute_nonlocal_self_energy(R, r, p, q, tube_model='surface')

    q_g = nonlocal_se['q_grid']
    Pi_g = nonlocal_se['Pi']
    lambda_C_bar_nl = nonlocal_se['lambda_C_bar']
    models = nonlocal_se['models']

    print(f"""
  Torus self-energy in momentum space:
    U_E = (e²/4π²ε₀) ∫ dq [sin(qR)/(qR)]² × |F_tube(q)|²
    U_total = 2 × U_E  (electric + magnetic at v = c)

  The tube form factor F_tube(q) determines the UV behavior:
    Gaussian:      exp(-q²r²/4)     — exponential kill (fuzzy blob)
    Hard cylinder: 2J₁(qr)/(qr)    — power-law ~1/(qr)^3/2 (uniform fill)
    Surface ring:  J₀(qr)           — power-law ~1/√(qr) (charge on tube wall)
    Thin wire:     1                 — no cutoff (UV divergent, shown for reference)

  Analytic check: ring without tube = k_e e²/(2R) = {nonlocal_se['U_ring_analytic_MeV']:.6f} MeV
  Neumann inductance (position-space)              = {nonlocal_se['U_Neumann_MeV']:.6f} MeV
""")

    # ── Form factor comparison table ──
    print(f"  {'q × ƛ_C':>10s}  {'Π(q²)':>10s}  {'1/(1-Π)':>8s}"
          f"  {'Gauss²':>10s}  {'Cylind²':>10s}  {'Surf²':>10s}  {'Wire²':>8s}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")

    tube_ff = nonlocal_se['tube_ff']
    indices_nl = np.unique(np.logspace(0, np.log10(len(q_g) - 1), 22, dtype=int))
    for i in indices_nl:
        q_lc = q_g[i] * lambda_C_bar_nl
        scr = 1.0 / (1.0 - Pi_g[i])
        print(f"  {q_lc:10.4f}  {Pi_g[i]:10.4e}  {scr:8.4f}"
              f"  {tube_ff['gaussian_sq'][i]:10.4e}"
              f"  {tube_ff['cylinder_sq'][i]:10.4e}"
              f"  {tube_ff['surface_sq'][i]:10.4e}"
              f"  {1.0:8.4f}")

    # ── Self-energy comparison across models ──
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  SELF-ENERGY BY TUBE CROSS-SECTION MODEL                               │
  │                                                                         │
  │  Model          U_bare (keV)  U_screened (keV)  δU_Π (keV)  Π corr %  │
  │  ──────────────────────────────────────────────────────────────────     │""")
    for name, label in [('gaussian', 'Gaussian    '),
                         ('cylinder', 'Hard cylndr '),
                         ('surface',  'Surface ring'),
                         ('wire',     'Thin wire   ')]:
        m = models[name]
        u_b = m['U_bare_keV']
        u_s = m['U_screened_keV']
        du = m['delta_U_keV']
        pct = m['Pi_correction_pct']
        marker = ' ◄── NWT' if name == 'surface' else ''
        if name == 'wire' and (u_b > 1e6 or np.isinf(u_b)):
            print(f"  │  {label}     DIVERGENT        DIVERGENT          ---       ---   │")
        else:
            print(f"  │  {label}   {u_b:10.4f}      {u_s:10.4f}     {du:+10.4f}   {pct:6.2f}%{marker}  │")
    print(f"  │                                                                         │")
    print(f"  │  Neumann inductance (position-space):  {nonlocal_se['U_Neumann_MeV']*1000:10.4f} keV              │")
    print(f"  │  EH local (from Section 4):            {nonlocal_se['U_EH_local_MeV']*1000:10.4f} keV              │")
    print(f"  └─────────────────────────────────────────────────────────────────────────┘")

    # Key physics insight
    surf = models['surface']
    gauss = models['gaussian']
    print(f"""
  The surface ring model (J₀) is the most physical for NWT: the charge
  circulates on the null worldtube BOUNDARY, not smeared through the interior.

  Surface ring vs Gaussian:
    U_bare:     {surf['U_bare_keV']:.4f} keV vs {gauss['U_bare_keV']:.4f} keV  ({surf['U_bare_keV']/gauss['U_bare_keV']:.2f}×)
    δU from Π:  {surf['delta_U_keV']:+.4f} keV vs {gauss['delta_U_keV']:+.4f} keV  ({surf['delta_U_keV']/gauss['delta_U_keV']:.2f}× if both nonzero)

  The J₀ form factor decays as 1/√(qr), letting through modes up to
  q × ƛ_C ~ {lambda_C_bar_nl/r:.0f} (vs Gaussian cutoff at q×ƛ_C ~ {np.sqrt(2)*lambda_C_bar_nl/r:.0f}).
  This is exactly where Π(q²) peaks (~18% anti-screening).
""")

    # Full energy budget comparison using the surface model
    print("  Computing full nonlocal energy budget (surface ring model)...")
    budget = compute_full_nonlocal_energy_budget(R, r, p, q)

    print(f"""
  ┌───────────────────────────────────────────────────────────────────┐
  │  ENERGY BUDGET COMPARISON                                        │
  │                                                                   │
  │  A) Local EH dressed (Neumann self-energy):                      │
  │     E_circ_eff      = {budget['E_circ_eff_MeV']:12.6f} MeV                      │
  │     U_self (EH)     = {budget['U_self_EH_MeV']:12.6f} MeV                      │
  │     U_drag          = {budget['U_drag_MeV']:12.6f} MeV                      │
  │     ─────────────────────────────────────                        │
  │     E_dressed       = {budget['E_dressed_MeV']:12.6f} MeV  gap = {budget['gap_dressed_keV']:+.4f} keV  │
  │                                                                   │
  │  B) Hybrid (surface ring + Π(q²)):                               │
  │     E_circ_eff      = {budget['E_circ_eff_MeV']:12.6f} MeV  (local EH)          │
  │     U_self (Π(q²))  = {budget['U_self_nonlocal_MeV']:12.6f} MeV  (surface, screened)│
  │     U_drag          = {budget['U_drag_MeV']:12.6f} MeV  (local EH)          │
  │     ─────────────────────────────────────                        │
  │     E_hybrid        = {budget['E_hybrid_MeV']:12.6f} MeV  gap = {budget['gap_hybrid_keV']:+.4f} keV  │
  │                                                                   │
  │  m_e                = {m_e_MeV:12.6f} MeV  (target)               │
  └───────────────────────────────────────────────────────────────────┘

  Π(q²) correction to surface ring self-energy: {budget['delta_U_nonlocal_keV']:+.6f} keV
  Remaining gap: {budget['gap_hybrid_keV']:+.4f} keV ({budget['gap_hybrid_pct']:+.4f}% of m_e)
""")

    # ════════════════════════════════════════════════════════════════
    # SECTION 11: Knot Self-Interaction — Full Neumann Integral
    # ════════════════════════════════════════════════════════════════
    print()
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  SECTION 11: KNOT SELF-INTERACTION — (p,q) NEUMANN INTEGRAL    ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")

    print(f"""
  The single-ring Neumann formula L = μ₀R[ln(8R/r) - 2] treats the
  knot as a simple circular loop. But the ({p},{q}) torus knot winds
  p = {p} times around the torus. At each cross-section, there are {p}
  current elements on the tube, separated by ~2r sin(π/{p}).

  The full Neumann double integral over the actual knot path:
    L = (μ₀/4π) ∮∮ (dl₁ · dl₂) / max(|r₁ - r₂|, r_tube)

  includes BOTH the self-inductance of each pass AND the mutual
  inductance between passes. For closely-wound turns (r << R),
  L_total ≈ p² × L_single (like an inductor coil).

  Computing full Neumann integral over ({p},{q}) knot...
""")

    knot = compute_knot_neumann_inductance(R, r, p, q)

    print(f"""
  Results:
    L_single_ring  = {knot['L_single_ring']:.6e} H  (formula: μ₀R[ln(8R/r)-2])
    L_knot (full)  = {knot['L_knot']:.6e} H  (numerical double integral)
    L_knot / L_ring = {knot['L_ratio']:.4f}  (expected ≈ p² = {p**2} for tight winding)

  Decomposition:
    L_self (same pass)  = {knot['L_self_near']:.6e} H  ({knot['L_self_near']/knot['L_knot']*100:.1f}%)
    L_mutual (between)  = {knot['L_mutual']:.6e} H  ({knot['mutual_fraction']*100:.1f}%)

  Self-energy (total = E + B = 2 × ½LI²):
    U_single_ring   = {knot['U_single_ring_keV']:.4f} keV
    U_knot (full)   = {knot['U_knot_keV']:.4f} keV  ({knot['L_ratio']:.2f}× single ring)
    U_mutual only   = {knot['U_mutual_keV']:.4f} keV  (energy from inter-pass interaction)
""")

    # Updated energy budget with knot self-interaction
    gordon = budget['gordon']
    E_circ_eff = gordon['E_circ_eff_MeV']
    U_drag = gordon['U_drag_MeV']

    # Replace Neumann self-energy with full knot inductance
    U_knot_MeV = knot['U_knot_MeV']
    E_knot_dressed = E_circ_eff + U_knot_MeV + U_drag
    gap_knot = (E_knot_dressed - m_e_MeV) * 1000

    # And with Π(q²) correction scaled by the knot/ring ratio
    # The Π correction scales with the self-energy, so multiply by L_ratio
    Pi_corr_keV = budget['delta_U_nonlocal_keV'] * knot['L_ratio']
    U_knot_screened_keV = knot['U_knot_keV'] + Pi_corr_keV
    E_knot_screened = E_circ_eff + U_knot_screened_keV / 1000 + U_drag
    gap_knot_screened = (E_knot_screened - m_e_MeV) * 1000

    print(f"""
  ┌───────────────────────────────────────────────────────────────────┐
  │  UPDATED ENERGY BUDGET WITH KNOT SELF-INTERACTION                │
  │                                                                   │
  │  C) Knot inductance (no Π):                                      │
  │     E_circ_eff      = {E_circ_eff:12.6f} MeV                      │
  │     U_self (knot)   = {U_knot_MeV:12.6f} MeV  ({knot['L_ratio']:.2f}× Neumann)    │
  │     U_drag          = {U_drag:12.6f} MeV                      │
  │     ─────────────────────────────────────                        │
  │     E_total         = {E_knot_dressed:12.6f} MeV  gap = {gap_knot:+.4f} keV  │
  │                                                                   │
  │  D) Knot + Π(q²) correction:                                     │
  │     U_self (knot+Π) = {U_knot_screened_keV/1000:12.6f} MeV                      │
  │     E_total         = {E_knot_screened:12.6f} MeV  gap = {gap_knot_screened:+.4f} keV  │
  │                                                                   │
  │  Compare to previous:                                             │
  │     A) Neumann dressed:    gap = {budget['gap_dressed_keV']:+.4f} keV              │
  │     B) Ring + Π(q²):       gap = {budget['gap_hybrid_keV']:+.4f} keV              │
  │     C) Knot inductance:    gap = {gap_knot:+.4f} keV              │
  │     D) Knot + Π(q²):       gap = {gap_knot_screened:+.4f} keV              │
  │  m_e = {m_e_MeV:.6f} MeV                                         │
  └───────────────────────────────────────────────────────────────────┘
""")

    # Also compute for p=1 as sanity check
    print("  Sanity check: p=1 (single ring, should recover Neumann)...")
    knot_p1 = compute_knot_neumann_inductance(R, r, p=1, q=1)
    print(f"    L_knot(1,1) / L_single_ring = {knot_p1['L_ratio']:.4f}  (should be ~1.0)")
    print(f"    U_knot(1,1) = {knot_p1['U_knot_keV']:.4f} keV vs Neumann = {knot_p1['U_single_ring_keV']:.4f} keV")
    print()

    # ════════════════════════════════════════════════════════════════
    # SECTION 12: Self-Consistent Radius with Full Knot Budget
    # ════════════════════════════════════════════════════════════════
    print()
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  SECTION 12: SELF-CONSISTENT RADIUS — KNOT + Π(q²) BUDGET     ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")

    print(f"""
  All previous sections used the flat-space self-consistent radius
  R = {R/lambda_C:.4f} ƛ_C from the single-ring Neumann formula. But the
  full knot Neumann integral changes the self-energy by {knot['L_ratio']:.2f}×.

  Question: what R makes the FULL knot + Π(q²) budget exactly equal m_e?

  E_total(R) = E_circ_eff(R) + U_self_knot(R) × 1.078 + U_drag(R) = m_e

  Searching by bisection (r/R = α)...
""")

    # Current energy at flat-space R for reference
    current = compute_knot_total_energy(R, r, p, q, N_knot=500)
    print(f"  At flat-space R = {R/lambda_C:.4f} ƛ_C:")
    print(f"    E_total = {current['E_total_MeV']:.6f} MeV  (gap = {current['gap_from_me_keV']:+.4f} keV)")
    print()

    # Find self-consistent R
    print("  Running bisection search...")
    sol = find_knot_self_consistent_radius(m_e_MeV, p=p, q=q)

    if sol is not None:
        bd = sol['breakdown']
        print(f"""
  ┌───────────────────────────────────────────────────────────────────┐
  │  SELF-CONSISTENT SOLUTION  (knot + Π budget)                     │
  │                                                                   │
  │  R = {sol['R_over_lambda_C']:.6f} ƛ_C  ({sol['R_femtometers']:.3f} fm)                     │
  │  r = {sol['r_over_lambda_C']:.6f} ƛ_C  ({sol['r_femtometers']:.4f} fm)                    │
  │  r/R = α = {sol['r_ratio']:.6e}                                 │
  │                                                                   │
  │  Energy budget at self-consistent R:                              │
  │     E_circ_eff      = {bd['E_circ_eff_MeV']:12.6f} MeV                      │
  │     U_self (knot+Π) = {bd['U_self_screened_MeV']:12.6f} MeV  ({bd['L_ratio']:.2f}× ring)  │
  │     U_drag          = {bd['U_drag_MeV']:12.6f} MeV                      │
  │     ─────────────────────────────────────                        │
  │     E_total         = {bd['E_total_MeV']:12.6f} MeV                      │
  │     m_e             = {m_e_MeV:12.6f} MeV                      │
  │     Match           = {sol['match_ppm']:.1f} ppm                               │
  │                                                                   │
  │  Shift from flat-space R:                                         │
  │     ΔR = {(sol['R_over_lambda_C'] - R/lambda_C):.6f} ƛ_C  ({(sol['R_over_lambda_C'] - R/lambda_C)/(R/lambda_C)*100:+.4f}%)          │
  │     n_avg = {bd['n_avg']:.6f}  (effective refractive index)         │
  └───────────────────────────────────────────────────────────────────┘
""")

        # Interpretation
        dR_pct = (sol['R_over_lambda_C'] - R/lambda_C) / (R/lambda_C) * 100
        print(f"  The self-consistent R {'increases' if dR_pct > 0 else 'decreases'} by {abs(dR_pct):.2f}%.")
        print(f"  E_circ accounts for {bd['E_circ_eff_MeV']/m_e_MeV*100:.2f}% of m_e.")
        print(f"  U_self accounts for {bd['U_self_screened_MeV']/m_e_MeV*100:.2f}% of m_e.")
        print(f"  U_drag accounts for {bd['U_drag_MeV']/m_e_MeV*100:.4f}% of m_e.")
    else:
        print("  *** No solution found in search range! ***")
        sol = None
    print()

    # ════════════════════════════════════════════════════════════════
    # SECTION 13: Stability Analysis
    # ════════════════════════════════════════════════════════════════
    if sol is not None:
        R_sc = sol['R']

        print()
        print("  ╔══════════════════════════════════════════════════════════════════╗")
        print("  ║  SECTION 13: STABILITY ANALYSIS — IS R = 0.488 ƛ_C A BOUND    ║")
        print("  ║  STATE OR JUST A CROSSING POINT?                                ║")
        print("  ╚══════════════════════════════════════════════════════════════════╝")

        print(f"""
  At R_sc = {R_sc/lambda_C:.4f} ƛ_C, E_total(R_sc) = m_e by construction.
  But is this a STABLE equilibrium?

  Scanning E_total(R) over [{R_sc/lambda_C/3:.3f}, {R_sc/lambda_C*3:.3f}] ƛ_C...
""")

        stab = compute_stability_analysis(R_sc, p=p, q=q, N_knot=300, N_scan=60)

        # Display the E(R) landscape as ASCII
        R_lam = stab['R_lam']
        E_tot = stab['E_total']
        E_crc = stab['E_circ']
        U_slf = stab['U_self']
        U_drg = stab['U_drag']

        # Table of selected values
        indices = np.linspace(0, len(R_lam)-1, 15, dtype=int)
        print("  ┌─────────────────────────────────────────────────────────────────┐")
        print("  │  R/ƛ_C    E_circ    U_self    U_drag    E_total    gap (keV)   │")
        print("  │           (MeV)     (MeV)     (MeV)     (MeV)                  │")
        print("  ├─────────────────────────────────────────────────────────────────┤")
        for i in indices:
            gap = (E_tot[i] - m_e_MeV) * 1000
            marker = " ◄" if abs(R_lam[i] - stab['R_sc_lam']) / stab['R_sc_lam'] < 0.05 else ""
            print(f"  │  {R_lam[i]:.4f}  {E_crc[i]:9.4f}  {U_slf[i]:9.6f}  {U_drg[i]:9.4f}"
                  f"  {E_tot[i]:9.4f}  {gap:+9.2f}{marker:3s}│")
        print("  └─────────────────────────────────────────────────────────────────┘")

        # Derivatives
        print(f"""
  Derivatives at R_sc = {stab['R_sc_lam']:.4f} ƛ_C:

    dE/dR         = {stab['dEdR_MeV_per_lam']:+.4f} MeV/ƛ_C
    d²E/dR²       = {stab['d2EdR2_MeV_per_lam2']:+.4f} MeV/ƛ_C²

    Component slopes (MeV/ƛ_C):
      dE_circ/dR  = {stab['dEcirc_dR']*lambda_C/MeV:+.4f}   (circulation)
      dU_self/dR  = {stab['dUself_dR']*lambda_C/MeV:+.6f}   (self-energy)
      dU_drag/dR  = {stab['dUdrag_dR']*lambda_C/MeV:+.4f}   (frame-drag)

    Power-law scaling (E ~ R^n) near R_sc:
      E_circ ~ R^{{{stab['n_circ']:.3f}}}   (expected -1 for photon-in-box)
      U_self ~ R^{{{stab['n_self']:.3f}}}   (expected ~-1 for inductance)
      U_drag ~ R^{{{stab['n_drag']:.3f}}}   (expected -1 for frame-drag)
""")

        # Stability classification
        if stab['has_minimum']:
            print(f"  ★ E_total(R) has a LOCAL MINIMUM at R = {stab['R_min']/lambda_C:.4f} ƛ_C")
            print(f"    E_min = {stab['E_min']:.6f} MeV")
            depth = (m_e_MeV - stab['E_min']) * 1000
            print(f"    Binding depth below m_e: {depth:.2f} keV")
            if stab['E_osc_eV'] > 0:
                print(f"    Oscillation quantum: ℏω = {stab['E_osc_eV']:.2f} eV")
                print(f"    Oscillation frequency: f = {stab['f_osc']:.3e} Hz")
        else:
            print("  E_total(R) is MONOTONICALLY DECREASING over the scan range.")
            print("  No local minimum found — R_sc is a CROSSING POINT, not a bound minimum.")

        if stab['crossing_stable']:
            print(f"""
  STABLE CROSSING: dE/dR = {stab['dEdR_MeV_per_lam']:+.4f} MeV/ƛ_C < 0
    → R too small ⟹ E > m_e ⟹ system expands toward R_sc
    → R too large ⟹ E < m_e ⟹ system contracts toward R_sc
    This is a SELF-REGULATING equilibrium: perturbations are restored.""")
        else:
            print(f"""
  UNSTABLE CROSSING: dE/dR = {stab['dEdR_MeV_per_lam']:+.4f} MeV/ƛ_C > 0
    → Perturbations in R would grow, not restore.
    → Additional physics needed for stability.""")

        # Virial theorem check
        print(f"""
  ┌───────────────────────────────────────────────────────────────────┐
  │  VIRIAL / FORCE BALANCE AT R_sc                                  │
  │                                                                   │
  │  For a stationary configuration, the "forces" must balance:      │
  │  dE_circ/dR + dU_self/dR + dU_drag/dR = 0  (equilibrium)       │
  │                                                                   │
  │  Actually:                                                        │
  │  dE_circ/dR = {stab['dEcirc_dR']*lambda_C/MeV:+12.4f} MeV/ƛ_C  (inward: shrink R)   │
  │  dU_self/dR = {stab['dUself_dR']*lambda_C/MeV:+12.6f} MeV/ƛ_C  (self-energy)        │
  │  dU_drag/dR = {stab['dUdrag_dR']*lambda_C/MeV:+12.4f} MeV/ƛ_C  (outward: expand R)  │
  │  ─────────────────────────────────────                           │
  │  Total dE/dR = {stab['dEdR_MeV_per_lam']:+12.4f} MeV/ƛ_C                       │
  │                                                                   │
  │  Ratio |dE_circ/dR| : |dU_drag/dR| = {abs(stab['dEcirc_dR']/stab['dUdrag_dR']):.3f}               │
  └───────────────────────────────────────────────────────────────────┘
""")

    # ════════════════════════════════════════════════════════════════
    # SECTION 14: Centrifugal Energy and Waveguide Confinement
    # ════════════════════════════════════════════════════════════════
    print()
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  SECTION 14: CENTRIFUGAL ENERGY & WAVEGUIDE CONFINEMENT        ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")

    R_cf = R_sc if sol is not None else R
    r_cf = alpha * R_cf

    print(f"""
  The torus tube is a curved waveguide. A photon confined to the tube
  has TRANSVERSE zero-point energy from confinement (like a particle
  in a cylindrical box), plus curvature corrections.

  Key energies for a circular waveguide of radius r = {r_cf*1e15:.3f} fm:
""")

    cf = compute_centrifugal_energy(R_cf, r_cf, p, q)

    print(f"""
  ┌───────────────────────────────────────────────────────────────────┐
  │  CURVATURE OF THE ({p},{q}) TORUS KNOT at R = {R_cf/lambda_C:.4f} ƛ_C        │
  │                                                                   │
  │  Path-averaged curvature:  ⟨κ⟩ = {cf['kappa_avg_R']/R_cf:.4e} /m               │
  │  Dimensionless:            ⟨κ⟩R = {cf['kappa_avg_R']:.4f}                     │
  │  Curvature range:          κ_min R = {cf['kappa_avg_R'] - np.sqrt(cf['kr_rms']**2*R_cf**2/r_cf**2 - cf['kappa_avg_R']**2) if cf['kr_rms']**2*R_cf**2/r_cf**2 > cf['kappa_avg_R']**2 else 0:.4f}                     │
  │  Min radius of curvature:  ρ_min = {cf['rho_curv_min']*1e15:.2f} fm             │
  │  Max radius of curvature:  ρ_max = {cf['rho_curv_max']*1e15:.2f} fm             │
  │  √⟨(κr)²⟩ = {cf['kr_rms']:.6f}  (curvature × tube radius)       │
  └───────────────────────────────────────────────────────────────────┘

  ┌───────────────────────────────────────────────────────────────────┐
  │  CENTRIFUGAL ENERGY TERMS                                        │
  │                                                                   │
  │  1. Waveguide confinement (TM₀₁ mode):                          │
  │     E_conf = ℏc × j₀₁ / r  = {cf['E_conf_MeV']:12.6f} MeV              │
  │     (j₀₁ = 2.4048, first zero of J₀)                            │
  │     E_conf / m_e            = {cf['E_conf_MeV']/m_e_MeV:12.2f}                  │
  │                                                                   │
  │  2. Curvature-confinement coupling:                               │
  │     δE = E_conf × ⟨(κr)²⟩/4  = {cf['U_cf_coupling_MeV']:12.6f} MeV              │
  │                                                                   │
  │  3. Centrifugal pressure (radiation on walls):                    │
  │     U_cf = E_circ × r × ⟨κ⟩  = {cf['U_cf_pressure_MeV']:12.6f} MeV              │
  │                                                                   │
  │  4. da Costa geometric potential (attractive):                    │
  │     U_geom = -(ℏc)²⟨κ²⟩L/(8E) = {cf['U_daCosta_MeV']:12.6f} MeV              │
  │                                                                   │
  │  Net centrifugal: E_conf + coupling + da Costa                   │""")

    U_cf_net = cf['E_conf_MeV'] + cf['U_cf_coupling_MeV'] + cf['U_daCosta_MeV']
    print(f"  │     = {U_cf_net:12.6f} MeV  ({U_cf_net/m_e_MeV*100:.1f}% of m_e)              │")
    print(f"  └───────────────────────────────────────────────────────────────────┘")

    # The big question: scaling
    print(f"""
  SCALING ANALYSIS — does centrifugal energy break R⁻¹ universality?

  At r = αR (fixed aspect ratio), all terms scale as:
    E_conf = ℏc j₀₁/(αR) ~ R⁻¹
    U_cf_coupling ~ (ℏc/r)(κr)² ~ (1/R)(r/R)² ~ α²/R ~ R⁻¹
    U_daCosta ~ (ℏc)²κ²L/E ~ (ℏc)/R ~ R⁻¹

  → ALL centrifugal terms scale as R⁻¹ at fixed r/R = α.
  → They cannot create a minimum in E(R).

  This is a consequence of SELF-SIMILARITY: when r/R is fixed,
  the torus looks the same at every scale. The only way to get
  a minimum is to introduce a FIXED length scale.
""")

    # Now the key insight: what if r is NOT slaved to R?
    print(f"  ═══════════════════════════════════════════════════════════════")
    print(f"  BREAKING SELF-SIMILARITY: 2D energy landscape E(R, r)")
    print(f"  ═══════════════════════════════════════════════════════════════")

    print(f"""
  If r is an independent dynamical variable (not r = αR), then:
    E_circ ~ 1/R  (longitudinal, depends mainly on R)
    E_conf ~ 1/r  (transverse, depends mainly on r)
    U_self ~ R × ln(R/r)  ... complex dependence on both
    U_drag ~ 1/R

  The confinement energy E_conf = ℏc j₀₁/r DIVERGES as r → 0,
  while E_circ = 2πℏc/L(R) DIVERGES as R → 0.
  These are INDEPENDENT repulsive walls in the (R, r) plane.

  Scanning the 2D landscape...
""")

    # 2D scan: E(R, r) with both as free variables
    N_R2d = 25
    N_r2d = 25
    R_2d = np.geomspace(0.1 * lambda_C, 2.0 * lambda_C, N_R2d)
    r_2d = np.geomspace(0.001 * lambda_C, 0.05 * lambda_C, N_r2d)

    E_2d = np.zeros((N_R2d, N_r2d))
    E_circ_2d = np.zeros((N_R2d, N_r2d))
    E_conf_2d = np.zeros((N_R2d, N_r2d))
    U_self_2d = np.zeros((N_R2d, N_r2d))
    U_drag_2d = np.zeros((N_R2d, N_r2d))

    print("  Computing 2D energy surface (25×25 grid)...")
    for i, Rv in enumerate(R_2d):
        for j, rv in enumerate(r_2d):
            if rv >= Rv:
                E_2d[i, j] = np.nan
                continue
            try:
                base = compute_knot_total_energy(Rv, rv, p, q, N_knot=200)
                cf2 = compute_centrifugal_energy(Rv, rv, p, q, N=500)
                E_circ_2d[i, j] = base['E_circ_eff_MeV']
                U_self_2d[i, j] = base['U_self_screened_MeV']
                U_drag_2d[i, j] = base['U_drag_MeV']
                E_conf_2d[i, j] = cf2['E_conf_MeV']
                E_2d[i, j] = base['E_total_MeV'] + cf2['E_conf_MeV'] + cf2['U_daCosta_MeV']
            except Exception:
                E_2d[i, j] = np.nan

    # Find minimum
    E_2d_masked = np.where(np.isnan(E_2d), np.inf, E_2d)
    min_idx = np.unravel_index(np.argmin(E_2d_masked), E_2d.shape)
    R_min_2d = R_2d[min_idx[0]]
    r_min_2d = r_2d[min_idx[1]]
    E_min_2d = E_2d[min_idx]

    # Check if minimum is interior (not on boundary)
    is_interior = (0 < min_idx[0] < N_R2d - 1) and (0 < min_idx[1] < N_r2d - 1)

    print(f"""
  ┌───────────────────────────────────────────────────────────────────┐
  │  2D ENERGY LANDSCAPE E(R, r)                                     │
  │                                                                   │
  │  E_total = E_circ_eff + U_self_knot + U_drag + E_conf + U_geom  │
  │                                                                   │
  │  Grid minimum: R = {R_min_2d/lambda_C:.4f} ƛ_C, r = {r_min_2d/lambda_C:.5f} ƛ_C   │
  │                r/R = {r_min_2d/R_min_2d:.6f}  (α = {alpha:.6f})             │
  │                E_min = {E_min_2d:.6f} MeV                      │
  │                Interior minimum: {'YES' if is_interior else 'NO (boundary)'}                    │
  │                                                                   │""")

    if is_interior:
        # Show the energy components at the minimum
        i0, j0 = min_idx
        print(f"  │  At the minimum:                                              │")
        print(f"  │     E_circ   = {E_circ_2d[i0,j0]:10.4f} MeV                         │")
        print(f"  │     E_conf   = {E_conf_2d[i0,j0]:10.4f} MeV                         │")
        print(f"  │     U_self   = {U_self_2d[i0,j0]:10.6f} MeV                         │")
        print(f"  │     U_drag   = {U_drag_2d[i0,j0]:10.4f} MeV                         │")
        print(f"  │     E/m_e    = {E_min_2d/m_e_MeV:10.4f}                              │")
    print(f"  └───────────────────────────────────────────────────────────────────┘")

    # Show slices at fixed R and fixed r
    print(f"\n  Slice at R = {R_min_2d/lambda_C:.4f} ƛ_C (vary r):")
    i_R = min_idx[0]
    print(f"    {'r/ƛ_C':>10s}  {'r/R':>10s}  {'E_circ':>8s}  {'E_conf':>8s}  {'U_self':>10s}  {'E_total':>8s}")
    for j in range(N_r2d):
        if np.isnan(E_2d[i_R, j]):
            continue
        print(f"    {r_2d[j]/lambda_C:10.5f}  {r_2d[j]/R_2d[i_R]:10.6f}"
              f"  {E_circ_2d[i_R,j]:8.4f}  {E_conf_2d[i_R,j]:8.4f}"
              f"  {U_self_2d[i_R,j]:10.6f}  {E_2d[i_R,j]:8.4f}")

    print(f"\n  Slice at r = {r_min_2d/lambda_C:.5f} ƛ_C (vary R):")
    j_r = min_idx[1]
    print(f"    {'R/ƛ_C':>10s}  {'r/R':>10s}  {'E_circ':>8s}  {'E_conf':>8s}  {'U_self':>10s}  {'E_total':>8s}")
    for i in range(N_R2d):
        if np.isnan(E_2d[i, j_r]):
            continue
        print(f"    {R_2d[i]/lambda_C:10.4f}  {r_2d[j_r]/R_2d[i]:10.6f}"
              f"  {E_circ_2d[i,j_r]:8.4f}  {E_conf_2d[i,j_r]:8.4f}"
              f"  {U_self_2d[i,j_r]:10.6f}  {E_2d[i,j_r]:8.4f}")

    print(f"""
  CONCLUSION: At fixed r/R = α, E(R) ~ 1/R everywhere — no minimum.
  The size is SET by E = m_e (mass-shell), not by a potential well.

  But the 2D landscape E(R, r) has independent axes. What's missing
  is a term that grows with r to prevent tube expansion. The natural
  candidate: SURFACE TENSION of the null worldtube boundary.
""")

    # ════════════════════════════════════════════════════════════════
    # SECTION 15: Surface Tension and the Full 2D Landscape
    # ════════════════════════════════════════════════════════════════
    print()
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  SECTION 15: SURFACE TENSION — WHAT CONFINES THE TUBE?         ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")

    st = compute_surface_tension_energy(R_cf, r_cf, p, q)

    print(f"""
  The worldtube boundary separates the nonlinear EH vacuum (inside)
  from the perturbative vacuum (outside). Maintaining this interface
  costs energy — surface tension σ.

  At the tube surface (r = {r_cf*1e15:.3f} fm):
    E_surface = {st['E_surface']:.4e} V/m  ({st['x_surface']:.1f} × E_Schwinger)
    δε/ε₀ = {st['delta_eps_over_eps0']:.4f}  (dielectric shift)

  Four estimates of σ:

  ┌───────────────────────────────────────────────────────────────────┐
  │  Method                    σ (N/m)       U_σ (MeV)  Scaling     │
  │                                                                   │
  │  1. Young-Laplace:     {st['sigma_YL']:12.4e}  {st['U_YL_MeV']:10.4f}   ~R⁻¹     │
  │     σ = P_rad × r (force balance at tube surface)               │
  │                                                                   │
  │  2. EH boundary layer: {st['sigma_EH']:12.4e}  {st['U_EH_MeV']:10.4f}   ~R⁻¹     │
  │     σ = ½ δε E² × r (dielectric mismatch × width)              │
  │                                                                   │
  │  3. QED dimensional:   {st['sigma_QED']:12.4e}  {st['U_QED_MeV']:10.4f}   ~R⁻¹     │
  │     σ = αℏc/r² (fine structure × quantum surface energy)        │
  │                                                                   │
  │  4. Schwinger vacuum:  {st['sigma_Schwinger']:12.4e}  {st['U_Schwinger_MeV']:10.4f}   ~R²      │
  │     σ = ε₀E_S²ƛ_C (vacuum property, independent of torus)      │
  └───────────────────────────────────────────────────────────────────┘

  Methods 1-3 derive σ from the torus fields → σ scales with the torus
  → U_σ ~ R⁻¹ (self-similar, no minimum).

  Method 4 uses a FIXED vacuum property: σ_Schwinger = ε₀ E_S² ƛ_C.
  This is the energy per area to maintain a critical-field boundary
  against the QED vacuum. It does NOT scale with R.

  U_Schwinger = σ_S × 4π²Rr = σ_S × 4π²αR²  at r = αR
  → scales as R²  → GROWS with R → creates confining potential!
""")

    # Tube radius equilibrium scan
    print("  Scanning E(r) at fixed R to find tube equilibrium...")
    print("  E = E_circ_eff + E_conf + U_self + U_drag + U_σ")
    print()

    tube_eq = compute_tube_radius_equilibrium(R_cf, p=p, q=q, N_r=40)

    # Show results for each surface tension model
    for sigma_label, nice_name, explain in [
        ('Schwinger', 'σ = ε₀E_S²ƛ_C (vacuum)',
         'Fixed vacuum property — does NOT depend on torus fields'),
        ('EH_boundary', 'σ = ½δε E² r (EH boundary)',
         'Self-consistent dielectric mismatch at tube surface'),
        ('fit_to_alpha', 'σ tuned for r/R = α',
         'Reverse-engineered: what σ gives equilibrium at r = αR?'),
    ]:
        res = tube_eq[sigma_label]
        r_g = res['r_grid']
        min_j = res['min_j']
        r_eq = res['r_min']
        E_eq = res['E_min']

        print(f"  ── {nice_name} ──")
        print(f"  {explain}")

        if res['is_interior']:
            print(f"  ★ EQUILIBRIUM FOUND at r = {r_eq/lambda_C:.5f} ƛ_C"
                  f"  (r/R = {r_eq/R_cf:.6f})")
            print(f"    E_min = {E_eq:.4f} MeV")
        else:
            if min_j == 0:
                print(f"  No minimum — tube collapses (σ too strong)")
            else:
                print(f"  No minimum — tube expands indefinitely (σ too weak)")

        # Show a few points around the minimum/boundary
        indices = np.linspace(0, len(r_g)-1, 12, dtype=int)
        print(f"    {'r/R':>10s}  {'E_conf':>8s}  {'U_σ':>8s}  {'E_total':>9s}")
        for j in indices:
            if np.isnan(res['E_total'][j]):
                continue
            marker = " ◄" if j == min_j else ""
            print(f"    {r_g[j]/R_cf:10.6f}  {res['E_conf'][j]:8.3f}"
                  f"  {res['U_sigma'][j]:8.4f}  {res['E_total'][j]:9.4f}{marker}")
        print()

    # The key result: what σ is needed?
    sigma_needed = tube_eq['sigma_needed']
    sigma_S = tube_eq['sigma_Schwinger']
    ratio_to_S = sigma_needed / sigma_S

    print(f"""
  ┌───────────────────────────────────────────────────────────────────┐
  │  WHAT SURFACE TENSION REPRODUCES r/R = α?                        │
  │                                                                   │
  │  At equilibrium: dE_conf/dr + dU_σ/dr = 0                       │
  │  → -ℏc j₀₁/r² + σ × 4π²R = 0                                   │
  │  → σ_needed = ℏc j₀₁/(4π²Rr²)                                  │
  │                                                                   │
  │  At r = αR = {r_cf*1e15:.3f} fm:                                       │
  │     σ_needed    = {sigma_needed:.4e} N/m                         │
  │     σ_Schwinger = {sigma_S:.4e} N/m                         │
  │     σ_EH_bdry   = {st['sigma_EH']:.4e} N/m                         │
  │                                                                   │
  │     σ_needed / σ_Schwinger = {ratio_to_S:.1f}                          │
  │     σ_needed / σ_EH       = {sigma_needed/st['sigma_EH']:.2f}                             │
  └───────────────────────────────────────────────────────────────────┘
""")

    # Physical interpretation
    u_needed = sigma_needed / r_cf  # equivalent volume energy density
    u_Schwinger = eps0 * E_Schwinger**2 / 2
    u_inside = st['P_rad']  # radiation pressure inside tube

    print(f"  Physical interpretation of σ_needed:")
    print(f"    Energy density equiv:   {u_needed:.4e} J/m³")
    print(f"    u_Schwinger:            {u_Schwinger:.4e} J/m³")
    print(f"    u_radiation (inside):   {u_inside:.4e} J/m³")
    print(f"    σ_needed/σ_Schwinger:   {ratio_to_S:.1f}")
    print(f"    u_needed/u_radiation:   {u_needed/u_inside:.1f}")
    print()

    # Can the EH boundary tension do it?
    res_EH = tube_eq['EH_boundary']
    if res_EH['is_interior']:
        r_EH = res_EH['r_min']
        print(f"  ★ The EH boundary tension DOES create an equilibrium!")
        print(f"    r_eq/R = {r_EH/R_cf:.6f}  (target: α = {alpha:.6f})")
        print(f"    Match: {abs(r_EH/R_cf - alpha)/alpha*100:.1f}%")
    else:
        print(f"  The EH boundary tension is {'too strong' if res_EH['min_j'] == 0 else 'too weak'}"
              f" for equilibrium at this R.")

    # ════════════════════════════════════════════════════════════════
    # SECTION 16: Minkowski Geometry & Magnetic Flux
    # ════════════════════════════════════════════════════════════════
    print()
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║  SECTION 16: MINKOWSKI NULL SURFACE & MAGNETIC FLUX            ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")

    print(f"""
  Two potentially neglected aspects of the model:
  A) The null worldtube is a NULL SURFACE in Minkowski space — what
     does the Raychaudhuri equation (null geodesic focusing) constrain?
  B) The torus carries MAGNETIC FLUX — does flux quantization set r/R?
""")

    # ─── Part A: Null congruence ───
    print("  ── A) NULL CONGRUENCE — RAYCHAUDHURI EQUATION ──")

    nc = compute_null_congruence(R_cf, r_cf, p, q)

    print(f"""
  The null worldtube is generated by a congruence of null rays.
  The Raychaudhuri equation (flat space, R_μν = 0):

    dθ/dλ = -θ²/2 - σ² + ω²

  where θ = expansion, σ = shear, ω = twist.
  Averaged over one closed orbit: ⟨dθ/dλ⟩ = 0, so:

    ⟨θ²/2 + σ²⟩  vs  ⟨ω²⟩

  For a thin tube (rκ = {nc['rk']:.4f} << 1) around the ({p},{q}) knot:
    ⟨θ²⟩ = ⟨κ²⟩/2 = {nc['theta2_avg']:.4e} /m²
    ⟨σ²⟩ = ⟨κ²⟩/2 = {nc['sigma2_avg']:.4e} /m²
    ⟨ω²⟩ = q²/(p²R²) = {nc['omega2_avg']:.4e} /m²

    LHS (focusing) = {nc['LHS']:.4e} /m²
    RHS (defocusing) = {nc['RHS']:.4e} /m²
    Ratio = {nc['balance_ratio']:.4f}

  The congruence is {'' if nc['net_focusing'] else 'NOT '}FOCUSING (ratio {'>' if nc['net_focusing'] else '<'} 1).
  For (p,q) = ({p},{q}): ratio = 3p²/q² / 4 = {3*p**2/(4*q**2):.1f}

  → Expansion + shear overpower twist by {nc['balance_ratio']:.0f}:1
  → The null congruence WANTS to collapse (net focusing)
  → Something must prevent this — the tube boundary itself.

  CRITICAL: The Raychaudhuri balance depends on p,q (topology)
  but NOT on r/R at leading order (correction ~ (rκ)² = {nc['rk_correction']:.2e}).
  → Minkowski null geometry constrains WINDING NUMBERS, not tube radius.
""")

    # ─── Part B: Magnetic flux ───
    print("  ── B) MAGNETIC FLUX STRUCTURE ──")

    mf = compute_magnetic_flux_analysis(R_cf, r_cf, p, q)

    print(f"""
  The ({p},{q}) knot carries current I = ec/L = {mf['I']:.2f} A.
  Two independent magnetic circuits:

  ┌───────────────────────────────────────────────────────────────────┐
  │  POLOIDAL FLUX (through tube πr², from p toroidal windings)      │
  │                                                                   │
  │  B_pol = μ₀pI/(2πR) = {mf['B_pol']:.4e} T                      │
  │  Φ_pol = μ₀pIr²/(2R) = {mf['Phi_pol']:.4e} Wb                  │
  │  Φ_pol / (h/e)  = {mf['Phi_pol_over_Phi0']:.6e}                 │
  │                                                                   │
  │  ANALYTIC RESULT: at r = αR,                                     │
  │    Φ_pol / (h/e) = α³/(2π) = {mf['phi_pol_analytic']:.6e}  ✓   │
  │    (verified numerically to {mf['phi_pol_match_pct']:.2f}%)                      │
  │                                                                   │
  │  The poloidal flux is α³/(2π) ≈ 6.2 × 10⁻⁸ flux quanta —       │
  │  deeply sub-quantum. Flux quantization cannot fix r at this scale.│
  ├───────────────────────────────────────────────────────────────────┤
  │  TOROIDAL FLUX (through torus hole, from Neumann inductance)     │
  │                                                                   │
  │  Φ_total = LI = {mf['Phi_total']:.4e} Wb                        │
  │  Φ_total / (h/e)  = {mf['Phi_total_over_Phi0']:.6e}             │
  │  Φ_total / (h/2e) = {mf['Phi_total_over_Phi0']*2:.6e}          │
  │                                                                   │
  │  Also sub-quantum but closer: ~0.6% of h/e.                     │
  ├───────────────────────────────────────────────────────────────────┤
  │  AHARONOV-BOHM PHASE                                             │
  │                                                                   │
  │  Δφ_AB = 2π Φ/Φ₀ = {mf['AB_phase']:.4f} rad ({mf['AB_phase']/np.pi:.4f}π)         │
  │  Tiny — no Aharonov-Bohm quantization condition active.          │
  └───────────────────────────────────────────────────────────────────┘

  What tube radius r gives Φ_pol = n × (h/e)?
""")

    for label, nice in [('Phi_0', 'n = 1'),
                         ('alpha_Phi_0', 'n = α'),
                         ('alpha2_Phi_0', 'n = α²'),
                         ('alpha3_Phi_0', 'n = α³')]:
        fr = mf['flux_radii'][label]
        print(f"    {nice:12s}: r/R = {fr['r_over_R']:.4f}"
              f"  r = {fr['r_over_lambda_C']:.4f} ƛ_C"
              f"  {'← MATCH' if abs(fr['r_over_R'] - alpha)/alpha < 0.5 else ''}")

    print(f"""
  None of the simple flux quantization conditions give r/R ≈ α.
  The result Φ_pol/(h/e) = α³/(2π) is EXACT (analytic), showing that
  the poloidal flux is a CONSEQUENCE of r/R = α, not its cause.

  ═══════════════════════════════════════════════════════════════
  SYNTHESIS: WHAT FIXES r/R = α?

  From the analyses in Sections 13-16:

  1. STABILITY (Sec 13): E(R) is monotonically decreasing at fixed
     r/R = α. The crossing E = m_e is stable (self-correcting).
     But there is no minimum — R is set by mass-shell, not a well.

  2. SELF-SIMILARITY (Sec 14): At fixed r/R, ALL energy terms scale
     as R⁻¹. No minimum is possible without a fixed length scale.

  3. SURFACE TENSION (Sec 15): E_conf ~ 1/r (repulsive) vs U_σ ~ r
     (attractive) CAN create a minimum in r, but the required σ is
     ~900× σ_Schwinger. No known QED mechanism provides this.

  4. NULL GEOMETRY (Sec 16A): Raychaudhuri constrains p,q topology
     but not r/R (correction is O(α²) ~ 5×10⁻⁵).

  5. MAGNETIC FLUX (Sec 16B): Φ_pol/(h/e) = α³/(2π) — deeply
     sub-quantum. Flux quantization is not active.

  OPEN QUESTION: r/R = α may be set by:
  • The topology of the null surface (global, not local geometry)
  • Quantization of angular momentum: L = ℏ/2 → R = ƛ_C/2 (97.6% match)
  • The EH vacuum matching condition across the tube boundary
    (Israel junction + dielectric matching simultaneously)
  • A variational principle we haven't yet identified
  ═══════════════════════════════════════════════════════════════
""")

    print("=" * 70)
