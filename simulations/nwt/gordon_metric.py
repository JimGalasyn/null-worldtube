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
# Step 7: Display
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
    print()
    print("=" * 70)
