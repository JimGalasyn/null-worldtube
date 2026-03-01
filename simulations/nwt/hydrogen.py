"""Hydrogen, helium, and lithium orbital dynamics in the torus model."""

import numpy as np
from .constants import (
    c, hbar, h_planck, e_charge, eps0, mu0, m_e, m_e_MeV, m_p, alpha,
    eV, MeV, lambda_C, r_e, a_0, k_e, TorusParams, PARTICLE_MASSES,
)
from .core import (
    compute_self_energy, compute_total_energy, find_self_consistent_radius,
    compute_path_length,
)


def compute_hydrogen_orbital(Z: int, n: int, params: TorusParams = None):
    """
    Compute orbital dynamics of an electron torus around a nucleus.

    The electron is a (2,1) torus knot of major radius R ~ λ_C/2 ~ 193 fm.
    The Bohr radius is a_0 ~ 52,918 fm. The torus is ~274× smaller than
    its orbit, so we treat it as a compact spinning object in a Coulomb field.

    Three frequencies characterize the motion:
      f_circ:  internal photon circulation (gives mass)
      f_orbit: center-of-mass revolution (gives energy levels)
      f_prec:  torus axis precession (gives fine structure)

    The hierarchy f_circ >> f_orbit >> f_prec, with each step
    separated by α², is the reason atomic physics is perturbative.
    """
    if params is None:
        sol = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=0.1)
        params = TorusParams(R=sol['R'], r=sol['r'], p=2, q=1)

    # Torus properties
    L_torus = compute_path_length(params)
    f_circ = c / L_torus
    R_torus = params.R

    # Orbital parameters (Bohr model — but now with a physical reason)
    r_n = n**2 * a_0 / Z                       # orbital radius
    v_n = Z * alpha * c / n                     # orbital velocity
    E_n = -m_e * c**2 * Z**2 * alpha**2 / (2 * n**2)  # binding energy
    f_orbit = v_n / (2 * np.pi * r_n)           # orbital frequency

    # Precession frequency (spin-orbit coupling)
    # The torus magnetic moment interacts with the B field seen in
    # the electron's rest frame: B ~ (v/c²) × E_Coulomb
    # ΔE_FS ~ m_e c² (Zα)⁴ / n³ → f_prec = ΔE_FS / h
    # This is α² slower than f_orbit, giving the three-level hierarchy
    f_prec = m_e * c**2 * (Z * alpha)**4 / (h_planck * n**3)

    # Angular momenta
    L_spin = hbar / params.p  # internal (ℏ/2 for p=2)
    L_orbit_n = n * hbar      # orbital (Bohr condition)

    # Tidal parameter: the torus has finite size in a non-uniform field
    tidal_param = R_torus / r_n
    # Fine structure correction scales as tidal_param² ~ α²
    dE_fine = abs(E_n) * alpha**2 / n  # order-of-magnitude estimate

    return {
        'n': n, 'Z': Z,
        'r_n': r_n,
        'r_n_pm': r_n * 1e12,                   # orbital radius in pm
        'r_n_fm': r_n * 1e15,                    # orbital radius in fm
        'v_n': v_n,
        'v_over_c': v_n / c,
        'E_n_J': E_n,
        'E_n_eV': E_n / eV,
        'f_circ': f_circ,
        'f_orbit': f_orbit,
        'f_prec': f_prec,
        'ratio_circ_orbit': f_circ / f_orbit,
        'ratio_orbit_prec': f_orbit / f_prec,
        'R_torus': R_torus,
        'R_torus_fm': R_torus * 1e15,
        'size_ratio': r_n / R_torus,             # should be >> 1
        'L_spin': L_spin,
        'L_spin_over_hbar': L_spin / hbar,
        'L_orbit': L_orbit_n,
        'L_orbit_over_hbar': L_orbit_n / hbar,
        'L_ratio': L_orbit_n / L_spin,           # = 2n (always integer!)
        'tidal_param': tidal_param,
        'dE_fine_eV': dE_fine / eV,
    }


def compute_shell_filling(Z: int, params: TorusParams = None):
    """
    Compute electron shell configuration for a multi-electron atom.

    In the torus model, each electron is a (2,1) torus with:
      - 2 handedness options (spin up/down = opposite winding chirality)
      - (2l+1) precession orientations in each subshell

    Pauli exclusion becomes geometric: two tori of the same handedness
    at the same orbital resonance produce destructive interference.
    Only opposite-handedness pairs can share a resonance.

    Shell capacities:
      n=1: 2 electrons (l=0 only, 2×1=2)
      n=2: 8 electrons (l=0: 2, l=1: 6)
      n=3: 18 electrons (l=0: 2, l=1: 6, l=2: 10)
    """
    if params is None:
        sol = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=0.1)
        params = TorusParams(R=sol['R'], r=sol['r'], p=2, q=1)

    # Build shell structure
    shells = []
    electrons_remaining = Z
    for n in range(1, 8):
        if electrons_remaining <= 0:
            break
        for l in range(n):
            if electrons_remaining <= 0:
                break
            capacity = 2 * (2 * l + 1)
            occupancy = min(capacity, electrons_remaining)
            electrons_remaining -= occupancy

            r_n = n**2 * a_0 / Z
            # Screened orbital radius (approximate)
            # Inner electrons screen the nuclear charge
            Z_eff = Z - sum(s['occupancy'] for s in shells)
            Z_eff = max(Z_eff, 1)
            r_eff = n**2 * a_0 / Z_eff
            E_n = -m_e * c**2 * Z_eff**2 * alpha**2 / (2 * n**2)

            subshell_labels = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
            label = f"{n}{subshell_labels.get(l, '?')}"

            shells.append({
                'n': n, 'l': l,
                'label': label,
                'capacity': capacity,
                'occupancy': occupancy,
                'full': occupancy == capacity,
                'r_n': r_n,
                'r_n_pm': r_n * 1e12,
                'Z_eff': Z_eff,
                'r_eff': r_eff,
                'r_eff_pm': r_eff * 1e12,
                'E_n_eV': E_n / eV,
                'orientations': 2 * l + 1,
            })

    # Noble gas: outermost occupied subshell is full, and it's
    # a p subshell (or 1s for helium). This captures He, Ne, Ar, etc.
    outermost = shells[-1] if shells else None
    noble = (outermost and outermost['full'] and
             (outermost['l'] == 1 or (outermost['n'] == 1 and outermost['l'] == 0)))

    return {
        'Z': Z,
        'shells': shells,
        'noble_gas': noble,
        'total_electrons': sum(s['occupancy'] for s in shells),
    }


def hydrogen_psi_squared_xz(n, l, m, grid_size=200, r_max_factor=2.5):
    """Compute |psi|^2 in xz-plane for hydrogen orbital (n,l,m)."""
    from scipy.special import sph_harm_y, genlaguerre, factorial

    a0 = 1.0  # atomic units
    # Grid in xz-plane (y=0)
    extent = r_max_factor * n**2 * a0
    x = np.linspace(-extent, extent, grid_size)
    z = np.linspace(-extent, extent, grid_size)
    X, Z = np.meshgrid(x, z)

    R = np.sqrt(X**2 + Z**2) + 1e-30  # avoid r=0
    theta = np.arccos(Z / R)  # polar angle from z-axis
    phi = np.where(X >= 0, 0.0, np.pi)  # azimuthal in xz-plane

    # Radial wavefunction R_nl(r)
    rho = 2 * R / (n * a0)
    norm_r = np.sqrt((2/(n*a0))**3 * factorial(n-l-1) / (2*n*factorial(n+l)))
    L = genlaguerre(n-l-1, 2*l+1)(rho)
    R_nl = norm_r * np.exp(-rho/2) * rho**l * L

    # Angular part: real spherical harmonics
    # sph_harm_y(l, m, theta_polar, phi_azimuthal) — physics convention
    if m == 0:
        Y = np.real(sph_harm_y(l, 0, theta, phi))
    elif m > 0:
        Y = np.real(sph_harm_y(l, m, theta, phi) * (-1)**m * np.sqrt(2))
    else:
        Y = np.imag(sph_harm_y(l, abs(m), theta, phi) * (-1)**abs(m) * np.sqrt(2))

    psi_sq = (R_nl * Y)**2
    return X, Z, psi_sq, extent


def compute_helium_nwt(Z=2, n=1, N_points=1000):
    """
    Compute helium ground-state energy using NWT torus trajectory mechanics.

    Two electron tori in circular orbits around a Z nucleus, each satisfying
    the resonance condition mvr = n*hbar. Computes time-averaged e-e repulsion
    for various geometric configurations and optimizes Z_eff variationally.

    The key object is the 'geometric factor' f(theta, delta):
        <1/r_12> = Z_eff * f
    where theta = tilt angle between orbital planes, delta = phase offset.

    Variational optimization is analytic:
        E(Z_eff) = Z_eff^2 - 2*Z*Z_eff + f*Z_eff
        Z_eff_opt = Z - f/2
        E_min = -(Z - f/2)^2  hartree
    """
    Ha_eV = m_e * (alpha * c)**2 / eV  # ~27.211 eV per hartree

    # --- Level 1: Independent electrons (no e-e repulsion) ---
    E_indep_Ha = -float(Z**2)  # 2 * (-Z^2/2)
    E_indep_eV = E_indep_Ha * Ha_eV

    # --- Level 2: QM first-order perturbation theory ---
    f_qm_pert = 5.0 * Z / 8.0 / Z  # = 5/8 (geometric factor for QM spherical average)
    # Actually: <1/r12> = 5*Z/8 in atomic units for two 1s wavefunctions with nuclear charge Z
    # So V_ee = 5*Z/8, and the geometric factor f = (5*Z/8) / Z_eff
    # But at Z_eff = Z (unscreened): V_ee = 5*Z/8
    E_pert1_Ha = -float(Z**2) + 5.0 * Z / 8.0
    E_pert1_eV = E_pert1_Ha * Ha_eV

    # --- Level 3: QM variational (single Z_eff parameter) ---
    # V_ee = 5*Z_eff/8 (scales with Z_eff since wavefunction contracts)
    f_qm_var = 5.0 / 8.0
    Z_eff_qm = Z - f_qm_var / 2  # = Z - 5/16 = 27/16 for Z=2
    E_qm_var_Ha = -(Z - f_qm_var / 2)**2
    E_qm_var_eV = E_qm_var_Ha * Ha_eV

    # --- Level 4: NWT trajectory geometric factors ---
    def compute_geometric_factor(theta, delta, N=N_points):
        """
        Compute f = <1/d> where d = |r1-r2|/r for two unit-radius circular orbits.

        Electron 1 in xy-plane, electron 2 in plane tilted by theta,
        with orbital phase offset delta.
        """
        phi = np.linspace(0, 2 * np.pi, N, endpoint=False)

        x1 = np.cos(phi)
        y1 = np.sin(phi)
        z1 = np.zeros_like(phi)

        phase2 = phi + delta
        x2 = np.cos(phase2)
        y2 = np.sin(phase2) * np.cos(theta)
        z2 = np.sin(phase2) * np.sin(theta)

        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        dist = np.maximum(dist, 1e-12)

        return float(np.mean(1.0 / dist))

    # Named configurations
    configs = []

    # (a) Coplanar, diametrically opposed
    f_a = compute_geometric_factor(0.0, np.pi)
    Z_eff_a = Z - f_a / 2
    E_a = -(Z - f_a / 2)**2
    configs.append({
        'name': 'Coplanar opposed (180\u00b0)',
        'theta_deg': 0.0, 'delta_deg': 180.0,
        'f': f_a, 'Z_eff': Z_eff_a,
        'E_Ha': E_a, 'E_eV': E_a * Ha_eV,
    })

    # (b) Orthogonal planes, optimal delta
    best_f_b = np.inf
    best_delta_b = 0.0
    for delta_test in np.linspace(0, 2 * np.pi, 200, endpoint=False):
        f_test = compute_geometric_factor(np.pi / 2, delta_test)
        if f_test < best_f_b:
            best_f_b = f_test
            best_delta_b = delta_test
    Z_eff_b = Z - best_f_b / 2
    E_b = -(Z - best_f_b / 2)**2
    configs.append({
        'name': 'Orthogonal planes (90\u00b0)',
        'theta_deg': 90.0, 'delta_deg': np.degrees(best_delta_b),
        'f': best_f_b, 'Z_eff': Z_eff_b,
        'E_Ha': E_b, 'E_eV': E_b * Ha_eV,
    })

    # --- Level 5: Full theta scan (find optimal and matching angles) ---
    N_theta = 500
    N_delta = 100
    theta_arr = np.linspace(0, np.pi, N_theta)
    delta_arr = np.linspace(0, 2 * np.pi, N_delta, endpoint=False)

    best_f_scan = np.zeros(N_theta)
    best_delta_scan = np.zeros(N_theta)

    for i, th in enumerate(theta_arr):
        phi = np.linspace(0, 2 * np.pi, 500, endpoint=False)

        # Vectorize over delta: phi is (N,1), delta is (1,M)
        phi_2d = phi[:, np.newaxis]  # (500, 1)
        delta_2d = delta_arr[np.newaxis, :]  # (1, 100)

        x1 = np.cos(phi_2d)
        y1 = np.sin(phi_2d)

        phase2 = phi_2d + delta_2d
        x2 = np.cos(phase2)
        y2 = np.sin(phase2) * np.cos(th)
        z2 = np.sin(phase2) * np.sin(th)

        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + z2**2)
        dist = np.maximum(dist, 1e-12)

        f_all_delta = np.mean(1.0 / dist, axis=0)  # (100,)
        idx_min = np.argmin(f_all_delta)
        best_f_scan[i] = f_all_delta[idx_min]
        best_delta_scan[i] = delta_arr[idx_min]

    E_scan_Ha = -(Z - best_f_scan / 2)**2
    E_scan_eV = E_scan_Ha * Ha_eV

    # Find the angle giving overall minimum repulsion (maximum |E|)
    idx_best = np.argmin(E_scan_Ha)  # most negative = lowest energy
    theta_best = theta_arr[idx_best]
    f_best = best_f_scan[idx_best]
    E_best_Ha = E_scan_Ha[idx_best]
    E_best_eV = E_scan_eV[idx_best]
    Z_eff_best = Z - f_best / 2

    # Find matching angle (closest to experiment)
    E_exp_eV = -79.005
    E_exp_Ha = E_exp_eV / Ha_eV
    idx_match = np.argmin(np.abs(E_scan_Ha - E_exp_Ha))
    theta_match = theta_arr[idx_match]
    f_match = best_f_scan[idx_match]
    E_match_Ha = E_scan_Ha[idx_match]
    E_match_eV = E_scan_eV[idx_match]
    Z_eff_match = Z - f_match / 2

    # Analytic f_match: from -(Z - f/2)^2 = E_exp_Ha
    f_match_analytic = 2 * (Z - np.sqrt(-E_exp_Ha))

    return {
        'Ha_eV': Ha_eV,
        'Z': Z,
        # Level 1
        'E_indep_Ha': E_indep_Ha, 'E_indep_eV': E_indep_eV,
        # Level 2
        'E_pert1_Ha': E_pert1_Ha, 'E_pert1_eV': E_pert1_eV,
        # Level 3
        'f_qm_var': f_qm_var, 'Z_eff_qm': Z_eff_qm,
        'E_qm_var_Ha': E_qm_var_Ha, 'E_qm_var_eV': E_qm_var_eV,
        # Level 4: named configs
        'configs': configs,
        # Level 5: scan
        'theta_arr': theta_arr,
        'best_f_scan': best_f_scan,
        'E_scan_eV': E_scan_eV,
        'E_scan_Ha': E_scan_Ha,
        # Best (lowest energy)
        'theta_best_deg': np.degrees(theta_best),
        'f_best': f_best, 'Z_eff_best': Z_eff_best,
        'E_best_Ha': E_best_Ha, 'E_best_eV': E_best_eV,
        # Matching experiment
        'theta_match_deg': np.degrees(theta_match),
        'f_match': f_match, 'f_match_analytic': f_match_analytic,
        'Z_eff_match': Z_eff_match,
        'E_match_Ha': E_match_Ha, 'E_match_eV': E_match_eV,
        # References
        'E_hylleraas_eV': -79.01, 'E_exp_eV': -79.005,
    }


def compute_helium_isoelectronic(Z_max=10):
    """
    Compute He-like isoelectronic sequence (Z=2 to Z_max) using NWT torus model.

    The geometric factor f(theta,delta) is purely geometric — it depends only
    on the orbital plane tilt and phase offset, not on Z. So we compute f once
    (from the Z=2 calculation) and reuse it for all Z.

    For each Z:
        E_indep     = -Z²  Ha          (independent electrons, no e-e)
        E_qm_var    = -(Z - 5/16)² Ha  (QM variational with Z_eff)
        E_nwt_ortho = -(Z - f_ortho/2)² Ha  (NWT orthogonal planes)
        E_nwt_match uses the analytic f that reproduces experiment for each Z
    """
    # NIST experimental 2-electron ground-state energies (eV)
    E_exp_eV = {
        2: -79.005,    # He
        3: -198.094,   # Li+
        4: -371.615,   # Be2+
        5: -599.60,    # B3+
        6: -882.08,    # C4+
        7: -1219.1,    # N5+
        8: -1610.7,    # O6+
        9: -2056.9,    # F7+
        10: -2557.6,   # Ne8+
    }

    ion_names = {
        2: 'He',   3: 'Li\u207a',  4: 'Be\u00b2\u207a', 5: 'B\u00b3\u207a',  6: 'C\u2074\u207a',
        7: 'N\u2075\u207a',  8: 'O\u2076\u207a',  9: 'F\u2077\u207a',  10: 'Ne\u2078\u207a',
    }

    Ha_eV = m_e * (alpha * c)**2 / eV  # ~27.211 eV per hartree

    # Get geometric factor from Z=2 calculation (geometry is Z-independent)
    he = compute_helium_nwt(Z=2)
    # f for orthogonal planes (theta=90°) — the named config [1]
    f_ortho = he['configs'][1]['f']

    f_qm_var = 5.0 / 8.0  # QM spherical average geometric factor

    results = []
    for Z in range(2, Z_max + 1):
        E_indep_Ha = -float(Z**2)
        E_indep_eV = E_indep_Ha * Ha_eV

        E_qm_Ha = -(Z - f_qm_var / 2)**2
        E_qm_eV = E_qm_Ha * Ha_eV

        E_ortho_Ha = -(Z - f_ortho / 2)**2
        E_ortho_eV = E_ortho_Ha * Ha_eV

        exp_eV = E_exp_eV[Z]
        exp_Ha = exp_eV / Ha_eV

        # Analytic f that reproduces experiment: -(Z - f/2)² = E_exp_Ha → f = 2(Z - sqrt(-E_exp_Ha))
        f_match = 2 * (Z - np.sqrt(-exp_Ha))

        E_match_Ha = -(Z - f_match / 2)**2
        E_match_eV = E_match_Ha * Ha_eV

        err_indep = 100 * (E_indep_eV - exp_eV) / abs(exp_eV)
        err_qm = 100 * (E_qm_eV - exp_eV) / abs(exp_eV)
        err_ortho = 100 * (E_ortho_eV - exp_eV) / abs(exp_eV)

        results.append({
            'Z': Z, 'ion': ion_names[Z],
            'E_indep_eV': E_indep_eV,
            'E_qm_eV': E_qm_eV,
            'E_ortho_eV': E_ortho_eV,
            'E_exp_eV': exp_eV,
            'f_ortho': f_ortho,
            'f_match': f_match,
            'err_indep_pct': err_indep,
            'err_qm_pct': err_qm,
            'err_ortho_pct': err_ortho,
        })

    return results


def compute_lithium_nwt(Z=3, N_phi=500, N_delta=50):
    """
    Compute lithium ground-state energy using NWT torus trajectory mechanics.

    Three electron tori: 2 inner (n=1) + 1 outer (n=2) around Z nucleus.
    Frozen-core approach: fix inner 1s² pair from He-like calculation,
    optimize outer 2s electron geometry and effective charge.

    All orbital planes tilted about the x-axis:
      e₁: xy-plane, e₂: xz-plane (orthogonal, He-like optimum),
      e₃: tilted θ_out from xy, radius ρ = 4s/t

    Note: sharing the x-axis excludes some 3-body geometries
    but captures the main cross-shell repulsion physics.
    """
    Ha_eV = m_e * (alpha * c)**2 / eV

    # --- Inner shell: He-like frozen core ---
    he = compute_helium_nwt(Z=Z)
    f_12 = he['configs'][1]['f']  # orthogonal planes (He-like optimum)
    s = Z - f_12 / 2  # inner Z_eff
    E_inner_Ha = -s**2  # inner shell total energy (2 electrons)

    # Independent electron reference (no screening, no repulsion)
    E_indep_Ha = -float(Z**2) - float(Z**2) / 8.0
    E_indep_eV = E_indep_Ha * Ha_eV

    # --- Scan outer electron: theta_out × t × delta ---
    N_theta = 100
    N_t = 100
    theta_out_arr = np.linspace(0, np.pi, N_theta)
    t_arr = np.linspace(0.3, 3.0, N_t)

    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    delta_arr = np.linspace(0, 2 * np.pi, N_delta, endpoint=False)
    phi_2d = phi[:, np.newaxis]       # (N_phi, 1)
    delta_2d = delta_arr[np.newaxis, :]  # (1, N_delta)

    # Inner electrons at unit radius (in r_inner units)
    x1 = np.cos(phi_2d); y1 = np.sin(phi_2d)       # e₁: xy
    x2 = np.cos(phi_2d); z2_inner = np.sin(phi_2d)  # e₂: xz

    best_t = np.zeros(N_theta)
    best_delta_arr_out = np.zeros(N_theta)
    best_E = np.full(N_theta, np.inf)
    best_g13 = np.zeros(N_theta)
    best_g23 = np.zeros(N_theta)

    for i, th_out in enumerate(theta_out_arr):
        cos_th = np.cos(th_out)
        sin_th = np.sin(th_out)

        for j, t in enumerate(t_arr):
            rho = 4.0 * s / t  # radius ratio outer/inner

            phase3 = phi_2d + delta_2d  # (N_phi, N_delta)
            x3 = rho * np.cos(phase3)
            y3 = rho * np.sin(phase3) * cos_th
            z3 = rho * np.sin(phase3) * sin_th

            # g₁₃: e₁(xy, r=1) vs e₃(tilted, r=ρ)
            d13 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2 + z3**2)
            d13 = np.maximum(d13, 1e-12)
            g13 = np.mean(1.0 / d13, axis=0)  # (N_delta,)

            # g₂₃: e₂(xz, r=1) vs e₃(tilted, r=ρ)
            d23 = np.sqrt((x2 - x3)**2 + y3**2 + (z2_inner - z3)**2)
            d23 = np.maximum(d23, 1e-12)
            g23 = np.mean(1.0 / d23, axis=0)  # (N_delta,)

            E_outer = t**2 / 8.0 - Z * t / 4.0
            E_total = E_inner_Ha + E_outer + (g13 + g23) * s

            idx_min = np.argmin(E_total)
            if E_total[idx_min] < best_E[i]:
                best_E[i] = E_total[idx_min]
                best_t[i] = t
                best_delta_arr_out[i] = delta_arr[idx_min]
                best_g13[i] = g13[idx_min]
                best_g23[i] = g23[idx_min]

    E_scan_eV = best_E * Ha_eV

    # Named configurations
    configs = []
    named = [
        ('A', 'Outer in xy (coplanar with e\u2081)', 0.0),
        ('B', 'Outer at 45\u00b0 (symmetric)', np.pi / 4),
        ('C', 'Outer in xz (coplanar with e\u2082)', np.pi / 2),
    ]
    for label, name, th_target in named:
        idx = np.argmin(np.abs(theta_out_arr - th_target))
        configs.append({
            'label': label, 'name': name,
            'theta_out_deg': np.degrees(theta_out_arr[idx]),
            'delta_deg': np.degrees(best_delta_arr_out[idx]),
            't': best_t[idx], 'rho': 4.0 * s / best_t[idx],
            'g13': best_g13[idx], 'g23': best_g23[idx],
            'g_total': best_g13[idx] + best_g23[idx],
            'E_Ha': best_E[idx], 'E_eV': best_E[idx] * Ha_eV,
        })

    # Best overall (lowest energy)
    idx_best = np.argmin(best_E)

    # Matching experiment
    E_exp_eV = -203.49
    E_exp_Ha = E_exp_eV / Ha_eV
    idx_match = np.argmin(np.abs(best_E - E_exp_Ha))

    # Li⁺ energy from He-like calculation
    E_Li_plus_eV = he['configs'][1]['E_eV']

    # Ionization energies
    IE_best = E_Li_plus_eV - best_E[idx_best] * Ha_eV
    IE_match = E_Li_plus_eV - best_E[idx_match] * Ha_eV

    return {
        'Ha_eV': Ha_eV, 'Z': Z,
        'f_12': f_12, 's': s,
        'E_inner_Ha': E_inner_Ha, 'E_inner_eV': E_inner_Ha * Ha_eV,
        'E_indep_Ha': E_indep_Ha, 'E_indep_eV': E_indep_eV,
        'configs': configs,
        'theta_out_arr': theta_out_arr,
        'E_scan_eV': E_scan_eV, 'E_scan_Ha': best_E.copy(),
        'best_t_scan': best_t,
        'best_g13_scan': best_g13, 'best_g23_scan': best_g23,
        'theta_best_deg': np.degrees(theta_out_arr[idx_best]),
        'delta_best_deg': np.degrees(best_delta_arr_out[idx_best]),
        'E_best_Ha': best_E[idx_best], 'E_best_eV': best_E[idx_best] * Ha_eV,
        't_best': best_t[idx_best],
        'theta_match_deg': np.degrees(theta_out_arr[idx_match]),
        'E_match_Ha': best_E[idx_match], 'E_match_eV': best_E[idx_match] * Ha_eV,
        't_match': best_t[idx_match],
        'E_Li_plus_eV': E_Li_plus_eV,
        'IE_best_eV': IE_best, 'IE_match_eV': IE_match, 'IE_exp_eV': 5.3917,
        'E_exp_eV': E_exp_eV,
    }


def compute_lithium_isoelectronic(Z_max=10):
    """
    Compute Li-like isoelectronic sequence (Z=3 to Z_max).

    Frozen-core approach with best geometry from Z=3 lithium.
    For each Z: inner 1s² pair with s = Z - f₁₂/2,
    outer 2s electron optimized in t at fixed orbital angles.
    """
    E_exp_3e = {
        3: -203.49, 4: -389.83, 5: -631.64, 6: -928.96,
        7: -1281.8, 8: -1690.2, 9: -2154.1, 10: -2673.6,
    }
    ion_names = {
        3: 'Li', 4: 'Be\u207a', 5: 'B\u00b2\u207a', 6: 'C\u00b3\u207a',
        7: 'N\u2074\u207a', 8: 'O\u2075\u207a', 9: 'F\u2076\u207a', 10: 'Ne\u2077\u207a',
    }

    Ha_eV = m_e * (alpha * c)**2 / eV

    # Get best geometry from Z=3
    li = compute_lithium_nwt(Z=3)
    f_12 = li['f_12']
    theta_best = np.radians(li['theta_best_deg'])
    delta_best = np.radians(li['delta_best_deg'])
    cos_th = np.cos(theta_best)
    sin_th = np.sin(theta_best)

    N_phi = 500
    N_t = 200
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)

    # Inner electron unit-circle positions
    x1 = np.cos(phi); y1 = np.sin(phi)   # e₁: xy
    x2 = np.cos(phi); z2 = np.sin(phi)   # e₂: xz

    results = []
    for Z in range(3, Z_max + 1):
        s = Z - f_12 / 2
        E_inner = -s**2

        t_max = min(Z + 1.0, 10.0)
        t_arr = np.linspace(0.3, t_max, N_t)

        best_E = np.inf
        best_t_val = 0.0

        for t in t_arr:
            rho = 4.0 * s / t
            phase3 = phi + delta_best
            x3 = rho * np.cos(phase3)
            y3 = rho * np.sin(phase3) * cos_th
            z3 = rho * np.sin(phase3) * sin_th

            d13 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2 + z3**2)
            d13 = np.maximum(d13, 1e-12)
            g13 = float(np.mean(1.0 / d13))

            d23 = np.sqrt((x2 - x3)**2 + y3**2 + (z2 - z3)**2)
            d23 = np.maximum(d23, 1e-12)
            g23 = float(np.mean(1.0 / d23))

            E_outer = t**2 / 8.0 - Z * t / 4.0
            E_total = E_inner + E_outer + (g13 + g23) * s

            if E_total < best_E:
                best_E = E_total
                best_t_val = t

        E_nwt_eV = best_E * Ha_eV
        exp_eV = E_exp_3e[Z]
        err_pct = 100 * (E_nwt_eV - exp_eV) / abs(exp_eV)

        results.append({
            'Z': Z, 'ion': ion_names[Z],
            'E_nwt_eV': E_nwt_eV, 'E_nwt_Ha': best_E,
            'E_exp_eV': exp_eV, 'err_pct': err_pct,
            't_opt': best_t_val, 's': s,
        })

    return results


def print_hydrogen_analysis():
    """
    Full hydrogen atom analysis in the null worldtube model.

    Shows how a compact electron torus orbiting a proton reproduces
    the Bohr model, explains quantization through resonance, and
    predicts fine structure as a tidal effect. Then extends to
    multi-electron atoms: shell filling, Pauli exclusion as geometry,
    and a preview of chemical bonding.

    This is where the torus model meets the planetary model — and
    fixes the reason the planetary model was abandoned (Larmor
    radiation). The electron IS radiation. There's no accelerating
    charge to radiate. The radiation objection doesn't apply.
    """
    print("=" * 70)
    print("HYDROGEN ATOM IN THE TORUS MODEL")
    print("  A compact spinning torus orbiting a proton")
    print("=" * 70)

    # Get the electron torus parameters
    e_sol = find_self_consistent_radius(m_e_MeV, p=2, q=1, r_ratio=0.1)
    params = TorusParams(R=e_sol['R'], r=e_sol['r'], p=2, q=1)
    R_torus = params.R
    R_torus_fm = R_torus * 1e15

    # ==========================================
    # Section 1: Size hierarchy
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  1. SIZE HIERARCHY: The electron is tiny")
    print(f"{'='*60}")
    print(f"\n  Electron torus major radius:  R = {R_torus_fm:.2f} fm")
    print(f"  Electron torus minor radius:  r = {params.r*1e15:.2f} fm")
    print(f"  Bohr radius:                  a₀ = {a_0*1e12:.1f} pm = {a_0*1e15:.0f} fm")
    print(f"  Ratio a₀ / R:                 {a_0/R_torus:.0f}")
    print(f"  This equals:                  2/α = {2/alpha:.0f}")
    print(f"\n  The electron fits ~{a_0/R_torus:.0f} times between itself and the nucleus.")
    print(f"  It is a compact spinning object in a slowly-varying field.")
    print(f"\n  Why this fixes the planetary model's fatal flaw:")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Bohr (1913): orbiting electrons should radiate (Larmor)")
    print(f"               and spiral into the nucleus. Ad hoc fix:")
    print(f"               postulate stable orbits without explanation.")
    print(f"  QM (1926):   replace orbits with probability clouds.")
    print(f"               Problem solved, but physical picture lost.")
    print(f"  Torus model: the electron IS radiation — a photon on a")
    print(f"               closed path. There is no accelerating charge")
    print(f"               in the Larmor sense. The radiation objection")
    print(f"               that killed the planetary model doesn't apply.")
    print(f"               You get to keep the orbits.")

    # ==========================================
    # Section 2: Orbital dynamics
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  2. ORBITAL DYNAMICS: Recovering the Bohr levels")
    print(f"{'='*60}")
    print(f"\n  The electron torus orbits the proton in a Coulomb field.")
    print(f"  Centripetal balance: m_e v²/r = e²/(4πε₀ r²)")
    print(f"  → v_n = αc/n,  r_n = n² a₀,  E_n = -13.6/n² eV")
    print(f"\n  {'n':>3} {'r_n (pm)':>10} {'v_n/c':>10} {'E_n (eV)':>12} {'Orbit/Torus':>14}")
    print(f"  " + "─" * 55)

    for n in range(1, 7):
        orb = compute_hydrogen_orbital(1, n, params)
        print(f"  {n:3d} {orb['r_n_pm']:10.2f} {orb['v_over_c']:10.6f} "
              f"{orb['E_n_eV']:12.4f} {orb['size_ratio']:14.0f}×")

    print(f"\n  At n=1: the electron orbits at {alpha:.6f} × c (= αc)")
    print(f"  This is {alpha*100:.3f}% of lightspeed — fast, but non-relativistic.")
    print(f"  The orbit is {a_0/R_torus:.0f}× larger than the torus → compact object in a field. ✓")

    # ==========================================
    # Section 3: Three-frequency hierarchy
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  3. THREE FREQUENCIES: The α² cascade")
    print(f"{'='*60}")

    orb1 = compute_hydrogen_orbital(1, 1, params)
    print(f"\n  Three rotations govern the electron's motion:")
    print(f"\n  1. Internal circulation  f_circ  = {orb1['f_circ']:.4e} Hz")
    print(f"     (photon going around the torus → gives mass)")
    print(f"\n  2. Orbital revolution    f_orbit = {orb1['f_orbit']:.4e} Hz")
    print(f"     (torus center orbiting nucleus → gives energy levels)")
    print(f"\n  3. Axis precession       f_prec  = {orb1['f_prec']:.4e} Hz")
    print(f"     (torus axis wobbling in field gradient → gives fine structure)")
    print(f"\n  The hierarchy:")
    print(f"    f_circ / f_orbit  = {orb1['ratio_circ_orbit']:.0f}  ≈ 1/α² = {1/alpha**2:.0f}")
    print(f"    f_orbit / f_prec  = {orb1['ratio_orbit_prec']:.0f}  ≈ 1/α² = {1/alpha**2:.0f}")
    print(f"    f_circ / f_prec   = {orb1['f_circ']/orb1['f_prec']:.0f}  ≈ 1/α⁴ = {1/alpha**4:.0f}")
    print(f"\n  Each level of structure is α² = 1/{1/alpha**2:.0f} slower than the last.")
    print(f"  THIS is why atomic physics is perturbative: the frequencies")
    print(f"  are so well-separated that each level barely perturbs the next.")
    print(f"  Perturbation theory works because α is small.")
    print(f"\n  In the torus model, α has a geometric meaning:")
    print(f"    α = R_torus / a₀ × 2 = (electron size) / (orbit size) × 2")
    print(f"    = {R_torus / a_0 * 2:.6f}  (vs α = {alpha:.6f})")

    # For different n
    print(f"\n  {'n':>3} {'f_circ (Hz)':>14} {'f_orbit (Hz)':>14} {'f_prec (Hz)':>14} "
          f"{'circ/orbit':>12} {'orbit/prec':>12}")
    print(f"  " + "─" * 76)
    for n in range(1, 5):
        orb = compute_hydrogen_orbital(1, n, params)
        print(f"  {n:3d} {orb['f_circ']:14.4e} {orb['f_orbit']:14.4e} {orb['f_prec']:14.4e} "
              f"{orb['ratio_circ_orbit']:12.0f} {orb['ratio_orbit_prec']:12.0f}")

    # ==========================================
    # Section 4: Resonance → Quantization
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  4. RESONANCE → QUANTIZATION: Why orbits are discrete")
    print(f"{'='*60}")
    print(f"\n  The electron torus has internal angular momentum:")
    print(f"    L_spin = ℏ/p = ℏ/2  (for p=2 winding)")
    print(f"\n  The orbital angular momentum must be commensurate:")
    print(f"    L_orbit = n ℏ  (Bohr condition)")
    print(f"\n  The ratio L_orbit / L_spin is always an integer:")

    print(f"\n  {'n':>3} {'L_orbit/ℏ':>12} {'L_spin/ℏ':>12} {'L_orbit/L_spin':>16} {'Integer?':>10}")
    print(f"  " + "─" * 55)
    for n in range(1, 7):
        orb = compute_hydrogen_orbital(1, n, params)
        ratio = orb['L_ratio']
        is_int = "✓" if abs(ratio - round(ratio)) < 0.001 else "✗"
        print(f"  {n:3d} {orb['L_orbit_over_hbar']:12.1f} {orb['L_spin_over_hbar']:12.4f} "
              f"{ratio:16.4f} {is_int:>10}")

    print(f"\n  L_orbit / L_spin = 2n — always an integer!")
    print(f"\n  Physical meaning: the electron's orbital phase must be")
    print(f"  locked to its internal circulation phase. After each orbit,")
    print(f"  the internal state must return to its starting configuration.")
    print(f"\n  This is a RESONANCE condition, not a postulate:")
    print(f"    • Guitar string: wavelength must fit the string → discrete modes")
    print(f"    • Electron orbit: internal phase must fit the orbit → discrete levels")
    print(f"    • Same physics, different geometry")
    print(f"\n  The Bohr quantization condition L = nℏ is not ad hoc —")
    print(f"  it's the resonance condition of a structured object.")

    # ==========================================
    # Section 5: Fine structure as tidal effect
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  5. FINE STRUCTURE: Tidal effect of finite electron size")
    print(f"{'='*60}")
    print(f"\n  The electron torus has finite size R = {R_torus_fm:.1f} fm.")
    print(f"  The Coulomb field varies across it:")
    print(f"    ΔE/E ~ R_torus / r_orbit = α/(2n²)")
    print(f"\n  The energy correction scales as (tidal parameter)²:")

    print(f"\n  {'n':>3} {'R/r_orbit':>14} {'(R/r_orbit)²':>14} {'ΔE_fine (eV)':>14} {'Known ΔE_FS':>14}")
    print(f"  " + "─" * 62)

    # Known fine structure for comparison (Dirac result)
    # ΔE_FS ≈ E_n × α² × [n/(j+1/2) - 3/4] / n
    # For the largest splitting in each n:
    known_fs = {
        1: 0.0,       # 1s has only j=1/2, no splitting
        2: 4.53e-5,   # 2p_{1/2} vs 2p_{3/2}
        3: 1.34e-5,   # 3p splitting
        4: 5.65e-6,   # 4p splitting
    }

    for n in range(1, 5):
        orb = compute_hydrogen_orbital(1, n, params)
        tp = orb['tidal_param']
        known = known_fs.get(n, 0)
        known_str = f"{known:.2e}" if known > 0 else "—"
        print(f"  {n:3d} {tp:14.6f} {tp**2:14.2e} {orb['dE_fine_eV']:14.6f} {known_str:>14}")

    print(f"\n  The tidal parameter R/r = α/(2n²) gives corrections of order α².")
    print(f"  This is exactly the scaling of fine structure!")
    print(f"\n  Physical picture:")
    print(f"    • The Coulomb field is stronger on the near side of the torus")
    print(f"    • This creates a torque on the spinning torus")
    print(f"    • The torque causes the spin axis to precess")
    print(f"    • Different precession rates → energy splitting")
    print(f"\n  Fine structure is literally a TIDAL effect: the electron")
    print(f"  has structure, and the field is non-uniform across it.")
    print(f"  α appears because it is the ratio of electron size to orbit size.")

    # ==========================================
    # Section 6: Multi-electron atoms
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  6. MULTI-ELECTRON ATOMS: Shell filling as torus packing")
    print(f"{'='*60}")
    print(f"\n  Each electron is a (2,1) torus. In a given orbital resonance:")
    print(f"    • 2 handedness options: clockwise and counter-clockwise")
    print(f"      (this IS spin up/down — it's the winding chirality)")
    print(f"    • (2l+1) orientations of the torus precession axis")
    print(f"    • Total capacity per subshell: 2 × (2l+1)")
    print(f"\n  Pauli exclusion is GEOMETRIC:")
    print(f"    Two tori of the same handedness at the same resonance")
    print(f"    have identical circulation phases → they interfere")
    print(f"    destructively. Only opposite-handedness pairs can coexist.")
    print(f"    A third torus of the same topology simply can't fit.")

    # Shell capacity table
    print(f"\n  Shell capacity (standard QM = torus packing):")
    print(f"  {'Shell':>6} {'Subshell':>10} {'Orientations':>14} {'× 2 spins':>10} {'= Capacity':>12}")
    print(f"  " + "─" * 55)
    for n in range(1, 5):
        for l in range(n):
            sub_label = f"{n}{'spdf'[l]}"
            orient = 2 * l + 1
            cap = 2 * orient
            print(f"  {n:6d} {sub_label:>10} {orient:14d} {2*orient:10d} {cap:12d}")
        total = 2 * n**2
        print(f"  {'':>6} {'':>10} {'':>14} {'TOTAL:':>10} {total:12d}")

    # Show first 18 elements
    print(f"\n  First 18 elements (up to Argon):")
    print(f"  {'Z':>4} {'Element':>8} {'Config':>18} {'Outer shell':>14} {'Noble?':>8}")
    print(f"  " + "─" * 55)

    elements = [
        (1, 'H'),   (2, 'He'),  (3, 'Li'),  (4, 'Be'),
        (5, 'B'),   (6, 'C'),   (7, 'N'),   (8, 'O'),
        (9, 'F'),   (10, 'Ne'), (11, 'Na'), (12, 'Mg'),
        (13, 'Al'), (14, 'Si'), (15, 'P'),  (16, 'S'),
        (17, 'Cl'), (18, 'Ar'),
    ]

    for Z, elem in elements:
        sf = compute_shell_filling(Z)
        config = ' '.join(f"{s['label']}{s['occupancy']}" for s in sf['shells'])
        outer = sf['shells'][-1]
        outer_str = f"{outer['label']}{'(full)' if outer['full'] else ''}"
        noble = "✓ noble" if sf['noble_gas'] else ""
        print(f"  {Z:4d} {elem:>8} {config:>18} {outer_str:>14} {noble:>8}")

    # Helium: the simplest multi-electron case
    print(f"\n  Helium (Z=2) in the torus model:")
    print(f"  ─────────────────────────────────")
    print(f"  Two (2,1) tori with opposite winding chirality orbit He²⁺")
    print(f"  at the n=1 resonance (r ≈ {a_0*1e12/2:.1f} pm for Z=2).")
    print(f"  Their mutual Coulomb repulsion raises the total energy:")
    E_He_no_repulsion = -2 * 13.6 * 4  # 2 electrons, Z=2
    E_He_observed = -79.0
    E_repulsion = E_He_observed - E_He_no_repulsion
    print(f"    Without repulsion: E = {E_He_no_repulsion:.1f} eV")
    print(f"    With repulsion:    E = {E_He_observed:.1f} eV  (observed)")
    print(f"    Repulsion energy:  ΔE = {E_repulsion:.1f} eV = {-E_repulsion/E_He_no_repulsion*100:.0f}% of binding")
    print(f"\n  The two tori settle into a configuration that minimizes")
    print(f"  repulsion while satisfying the orbital resonance condition.")
    print(f"  Opposite handedness lets them share the orbit; their mutual")
    print(f"  repulsion is a perturbation, not a destabilization.")

    # Noble gas stability
    print(f"\n  Noble gas stability:")
    print(f"  ────────────────────")
    print(f"  A noble gas has every orientation and handedness slot filled.")
    print(f"  The time-averaged torus distribution is spherically symmetric.")
    print(f"  No open resonance slots → no way for another torus to join")
    print(f"  without moving to the next shell (much higher energy).")
    print(f"  Chemical inertness = topological completeness.")

    # ==========================================
    # Section 7: The probability cloud
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  7. THE PROBABILITY CLOUD: Time-averaged torus position")
    print(f"{'='*60}")
    print(f"\n  Standard QM says: the electron is a probability cloud.")
    print(f"  The torus model says: the electron is a torus WHOSE")
    print(f"  TIME-AVERAGED POSITION is the probability cloud.")
    print(f"\n  The 1s orbital:")
    print(f"    • Torus orbits nucleus at r ≈ {a_0*1e12:.1f} pm")
    print(f"    • Orbital plane precesses through all orientations")
    print(f"    • Time-average of orbiting + precessing → spherical")
    print(f"    • |ψ(r)|² = fraction of time the torus spends near r")
    print(f"    • Maximum at r = a₀ (the most probable orbit)")
    print(f"\n  The 2p orbital:")
    print(f"    • Torus orbits at r ≈ {4*a_0*1e12:.0f} pm (n=2)")
    print(f"    • Orbital angular momentum (l=1) → stable precession plane")
    print(f"    • 3 orientations (m = -1, 0, +1) → the dumbbell shapes")
    print(f"    • Each orientation = different stable precession geometry")
    print(f"\n  There IS no transition from 'particle' to 'cloud'.")
    print(f"  The electron is always a torus. The cloud is what you")
    print(f"  see when you average over timescales >> T_orbit.")

    # Timescale table
    print(f"\n  Observation timescales:")
    print(f"  {'Timescale':>14} {'You see':>40}")
    print(f"  " + "─" * 55)
    print(f"  {'< T_circ':>14} {'frozen photon on torus surface':>40}")
    print(f"  {'T_circ':>14} {'circulating photon = the torus':>40}")
    print(f"  {'T_orbit':>14} {'torus at a specific point in orbit':>40}")
    print(f"  {'~ 10 T_orbit':>14} {'elliptical smear':>40}")
    print(f"  {'>> T_orbit':>14} {'probability cloud = standard QM':>40}")

    # ==========================================
    # Section 8: Chemical bonding preview
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  8. CHEMICAL BONDING: Shared orbital resonances")
    print(f"{'='*60}")
    print(f"\n  When two atoms approach, their outer electron tori interact.")
    print(f"  If a NEW stable resonance exists that spans both nuclei,")
    print(f"  the system's total energy decreases → a bond forms.")
    print(f"\n  Covalent bond:")
    print(f"    Two electron tori from different atoms find a joint orbital")
    print(f"    resonance around both nuclei. The shared resonance has lower")
    print(f"    energy than two separate single-nucleus orbits.")
    print(f"\n  Ionic bond:")
    print(f"    One atom's outer torus finds a lower-energy resonance")
    print(f"    around the other nucleus. The transferred torus reduces")
    print(f"    total energy. The resulting charge imbalance holds the")
    print(f"    atoms together electrostatically.")
    print(f"\n  Metallic bond:")
    print(f"    Outer tori find resonances spanning MANY nuclei.")
    print(f"    Delocalized tori = conduction electrons. The entire")
    print(f"    lattice is one resonance structure.")
    print(f"\n  In each case: chemistry = tori finding the lowest-energy")
    print(f"  resonance configuration. Reactivity = how many open")
    print(f"  resonance slots the outermost shell has.")

    # ==========================================
    # Section 9: Summary — what the torus model adds
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  9. WHAT THE TORUS MODEL ADDS BEYOND BOHR")
    print(f"{'='*60}")
    print(f"\n  {'Feature':<30} {'Bohr model':<25} {'Torus model'}")
    print(f"  " + "─" * 75)
    features = [
        ("Energy levels E_n", "Correct (postulated)", "Correct (from resonance)"),
        ("Quantization reason", "Ad hoc (L = nℏ)", "Resonance (L_orbit/L_spin = 2n)"),
        ("Radiation stability", "Postulated", "Structural (electron IS radiation)"),
        ("Fine structure", "Not predicted", "Tidal effect (R/r_orbit ~ α)"),
        ("Spin", "Added later (ad hoc)", "Geometry (p=2 → L_z = ℏ/2)"),
        ("Pauli exclusion", "Not explained", "Geometric interference"),
        ("Shell filling", "Not explained", "Torus packing + resonance"),
        ("Electron 'size'", "Point particle", "R = {:.0f} fm".format(R_torus_fm)),
        ("Probability cloud", "Not applicable", "Time-averaged torus position"),
    ]
    for feat, bohr, torus in features:
        print(f"  {feat:<30} {bohr:<25} {torus}")

    print(f"\n  Everything Bohr got right, the torus model reproduces.")
    print(f"  Everything Bohr couldn't explain, the torus model derives")
    print(f"  from the geometry of a photon on a closed path.")

    # ==========================================
    # Section 10: Relativistic corrections
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  10. RELATIVISTIC CORRECTIONS: TWO VELOCITY SCALES")
    print(f"{'='*60}")

    print(f"""
  The electron torus has TWO distinct velocity scales:

    INTERNAL:  v_int = c  (always — it's a null worldtube)
               The photon circulates at light speed on the torus.
               This is what MAKES it a particle.

    ORBITAL:   v_orb = αc/n  (the torus orbiting the nucleus)
               Ground state: v₁ = αc = c/{1/alpha:.0f} ≈ {alpha * c:.0f} m/s

  Ratio: v_orb / v_int = α/n ≈ 1/{1/alpha:.0f}

  The orbit is {1/alpha:.0f}× slower than the internal dynamics.
  The torus barely 'notices' it's orbiting — which is why the
  non-relativistic Bohr model works so well as a first approximation.""")

    # Orbital velocity table for hydrogen
    print(f"\n  Orbital velocities for hydrogen (Z=1):")
    print(f"  {'n':>3} {'v/c':>12} {'v (km/s)':>12} {'γ-1':>14} {'ΔE_rel/E_n':>14}")
    print(f"  " + "─" * 58)
    for n in range(1, 7):
        v_over_c = alpha / n
        v_kms = v_over_c * c / 1e3
        gamma_minus_1 = 1.0 / np.sqrt(1 - v_over_c**2) - 1
        dE_rel = alpha**2 / n**2  # relative correction ~ (v/c)²
        print(f"  {n:3d} {v_over_c:12.6f} {v_kms:12.1f} {gamma_minus_1:14.2e} {dE_rel:14.2e}")

    print(f"""
  For hydrogen, γ - 1 ≈ {0.5 * alpha**2:.1e} — a tiny correction.
  But this IS the fine structure. The α² corrections come in
  three geometric flavors in the torus model:""")

    print(f"\n  A. RELATIVISTIC MASS (Lorentz contraction of torus)")
    print(f"  ───────────────────────────────────────────────────")
    print(f"  The orbiting torus is Lorentz-contracted along its")
    print(f"  direction of motion. For v = αc, the contraction is:")
    gamma_1 = 1.0 / np.sqrt(1 - alpha**2)
    print(f"    γ = {gamma_1:.10f}")
    print(f"    Contraction: {(1 - 1/gamma_1)*100:.4f}% along orbital direction")
    print(f"  The torus deforms from circular to slightly elliptical")
    print(f"  cross-section. Its effective mass increases as γm_e,")
    print(f"  shifting the energy levels by:")
    print(f"    ΔE_rel = -E_n × (α/n)² × [n/(j+½) - ¾] / n")

    print(f"\n  B. SPIN-ORBIT COUPLING (torus spin in orbital B field)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  The electron torus has spin-½ from its (2,1) winding.")
    print(f"  In its rest frame, the orbiting proton creates a")
    print(f"  magnetic field:")
    B_orbit = m_e * c * alpha**3 / (e_charge * a_0)  # approximate B at Bohr radius
    print(f"    B_orbit ≈ {B_orbit:.1f} T  (at the Bohr orbit)")
    print(f"  This field exerts a torque on the torus's magnetic")
    print(f"  moment, coupling spin to orbital angular momentum.")
    print(f"  The Thomas precession factor of ½ arises from the")
    print(f"  non-inertial rest frame of the orbiting torus.")
    dE_so_eV = 13.6 * alpha**2 / 4  # rough spin-orbit for n=2
    print(f"    ΔE_SO ≈ {dE_so_eV*1e6:.1f} μeV  (n=2 splitting)")

    print(f"\n  C. DARWIN TERM (finite torus extent)")
    print(f"  ─────────────────────────────────────")
    r_tube_e = alpha * R_torus
    print(f"  The electron torus has tube radius r_tube = αR:")
    print(f"    r_tube = {r_tube_e*1e15:.2f} fm = {r_tube_e:.3e} m")
    print(f"  For s-states (l=0), the torus passes through the")
    print(f"  nucleus — and its tube 'samples' the Coulomb field")
    print(f"  over a volume of radius r_tube, not at a point.")
    print(f"  Energy shift: ΔE_Darwin ~ |ψ(0)|² × r_tube²")
    print(f"")
    print(f"  In standard QED, this is attributed to 'zitterbewegung'")
    print(f"  (mysterious trembling motion of the electron).")
    print(f"  In the torus model: the electron literally has finite")
    print(f"  size. There is no mystery — the Darwin term measures")
    print(f"  how much of the nucleus fits inside the tube.")

    # Heavy atoms
    print(f"\n  HEAVY ATOMS: WHERE RELATIVITY DOMINATES")
    print(f"  ─────────────────────────────────────────")
    print(f"  For inner electrons of heavy atoms, v = Zαc/n:")
    print(f"")
    print(f"  {'Element':>10} {'Z':>4} {'v/c (1s)':>10} {'γ':>8} {'Contraction':>12}")
    print(f"  " + "─" * 48)

    heavy_atoms = [
        ('Iron', 26), ('Silver', 47), ('Gold', 79),
        ('Mercury', 80), ('Lead', 82), ('Uranium', 92),
    ]
    for name, Z in heavy_atoms:
        v_oc = Z * alpha
        if v_oc >= 1.0:
            v_oc = 0.999  # cap for display
        gam = 1.0 / np.sqrt(1 - v_oc**2)
        contraction = (1 - 1/gam) * 100
        print(f"  {name:>10} {Z:4d} {v_oc:10.4f} {gam:8.3f} {contraction:11.1f}%")

    print(f"""
  Gold (Z=79): inner electrons orbit at 0.58c with γ ≈ 1.23.
  The 1s orbital contracts by ~19%, pulling all outer orbitals
  inward. This shifts the 5d→6s transition energy into the
  blue, making gold absorb blue light instead of reflecting it.
  GOLD IS GOLD-COLORED BECAUSE OF RELATIVITY.

  In the NWT model: the inner electron tori of gold are
  severely Lorentz-contracted — their circular cross-sections
  become ellipses with axis ratio 1:0.81. The self-consistent
  torus geometry is fundamentally altered by the orbital speed.
  Mercury's anomalously low melting point, lead's chemical
  inertness, and gold's color all trace to the same effect:
  inner tori orbiting at substantial fractions of c.

  THE DEEP POINT: SELF-CONSISTENCY AT RELATIVISTIC SPEED
  ──────────────────────────────────────────────────────

  A torus derived at rest (the free electron solution) gets
  placed in an orbit at v = Zαc. At low Z, the perturbation
  is small (α² corrections). At high Z, the torus geometry
  must be re-solved self-consistently at relativistic speed.

  This is why heavy-element quantum chemistry is hard — and
  why the Dirac equation (which handles this exactly for
  point particles) was such a triumph. The NWT equivalent
  would be solving for torus self-consistency on a Kerr-Newman
  background that includes the orbital Lorentz boost.

  For hydrogen, the orbital velocity is gentle enough that
  perturbation theory works beautifully — the three fine
  structure terms (relativistic mass, spin-orbit, Darwin)
  capture the physics at order α². The torus model gives
  each term a concrete geometric meaning:

  ┌──────────────────────────────────────────────────────┐
  │  FINE STRUCTURE = RELATIVISTIC TORUS MECHANICS       │
  │                                                      │
  │  • Relativistic mass: torus Lorentz-contracted       │
  │  • Spin-orbit: torus spin in orbital B field         │
  │  • Darwin term: tube has finite radius r = αR        │
  │                                                      │
  │  Internal velocity:  c     (always — null worldtube) │
  │  Orbital velocity:   αc/n  (Bohr orbit)              │
  │  Ratio:              α/n ≈ 1/{1/alpha:.0f} per shell          │
  │                                                      │
  │  The non-relativistic Bohr model works because the   │
  │  orbit is {1/alpha:.0f}× slower than the internal dynamics. │
  └──────────────────────────────────────────────────────┘""")

    # ==========================================
    # Section 11: Orbital shapes as torus trajectories
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  11. ORBITAL SHAPES AS TORUS TRAJECTORIES")
    print(f"{'='*60}")

    print(f"""
  Standard QM predicts orbital shapes — spheres, dumbbells,
  cloverleafs — from the angular solutions to the Schrödinger
  equation. The torus model gives these shapes a MECHANICAL
  interpretation: they are the time-averaged probability density
  of the electron torus following resonant trajectories in the
  Coulomb field.

  QUANTUM NUMBERS AS TRAJECTORY TYPES:

    n  (principal)    Total energy, radial extent of trajectory.
                      Larger n → torus orbits farther from nucleus.
                      Sets the size: <r> ∝ n²a₀

    l  (angular)      Type of angular momentum in the trajectory.
                      l = 0: radial oscillation (no net angular momentum)
                      l > 0: torus has tangential velocity → lobes

    m  (magnetic)     Orientation of the trajectory relative to z-axis.
                      m = 0: trajectory symmetric about z-axis
                      |m| > 0: tilted trajectories → oriented lobes""")

    print(f"\n  SHAPE INTERPRETATION:")
    print(f"  {'l':>3}  {'Name':>6}  {'Trajectory type':<32} {'Shape'}")
    print(f"  " + "─" * 68)
    shapes = [
        (0, 's', 'Radial oscillation (no L_orb)', 'Spherical — all angles equal'),
        (1, 'p', 'Planar orbit with 1 node plane', 'Two lobes along axis'),
        (2, 'd', 'Rosette orbit, 2 node planes', 'Four-lobed cloverleaf'),
        (3, 'f', 'Complex 3D path, 3 node planes', 'Multi-lobed (6-8 lobes)'),
    ]
    for l_val, name, traj, shape in shapes:
        print(f"  {l_val:3d}    {name}    {traj:<32} {shape}")

    print(f"""
  WHY THE SHAPES MATCH STANDARD QM:

  Both approaches solve the same mathematical problem — the Coulomb
  eigenvalue equation. Standard QM writes it as the Schrödinger
  equation and gets wavefunctions ψ_nlm. The torus model writes it
  as orbital resonance conditions and gets trajectory densities.

  The |ψ|² distribution IS the time-averaged torus probability density.
  Same equation → same solutions → same shapes.

  NODES AS VELOCITY MAXIMA:

  Where |ψ|² = 0 (nodes), the torus sweeps through fastest.
  The torus doesn't vanish at nodes — it passes through them at
  maximum speed, spending minimum time there.

    • Radial nodes (n - l - 1 of them): torus oscillates through
      spherical shells where it's momentarily at maximum radial speed.
    • Angular nodes (l of them): planes where the torus trajectory
      crosses at maximum tangential speed.
    • Total nodes = n - 1 (same as standard QM).

  MULTI-ELECTRON ATOMS:

  For atoms with Z > 1, orbital shapes are simply the stable
  resonant trajectories of the classical Coulomb N-body problem.
  Each electron torus follows its own trajectory, influenced by
  the nucleus AND all other electrons (screening + repulsion).

  No wavefunction collapse needed — the electron literally IS at
  some position on its trajectory at each moment. The |ψ|² cloud
  is the time average, not an ontological smear.""")

    # --- 3x3 orbital visualization ---
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from pathlib import Path

    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    orbitals = [
        # (n, l, m, label)
        (1, 0, 0, '1s'),   (2, 0, 0, '2s'),   (3, 0, 0, '3s'),
        (2, 1, 0, '2p$_z$'), (3, 1, 0, '3p$_z$'), (3, 2, 0, '3d$_{z^2}$'),
        (3, 2, 1, '3d$_{xz}$'), (4, 3, 0, '4f$_{z^3}$'), (4, 3, 1, '4f$_{xz^2}$'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle('Hydrogen Orbital Shapes: |ψ|² in the xz-plane\n'
                 'NWT interpretation: time-averaged torus trajectory density',
                 fontsize=13, fontweight='bold', y=0.98)

    for idx, (n, l, m, label) in enumerate(orbitals):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        X, Z, psi_sq, extent = hydrogen_psi_squared_xz(n, l, m)

        # Normalize for display
        psi_max = psi_sq.max()
        if psi_max > 0:
            psi_sq = psi_sq / psi_max

        ax.contourf(X, Z, psi_sq, levels=30, cmap=plt.cm.Blues)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect('equal')
        ax.set_title(f'{label}  (n={n}, l={l}, m={m})', fontsize=10)
        ax.set_xlabel('x / a₀', fontsize=8)
        ax.set_ylabel('z / a₀', fontsize=8)
        ax.tick_params(labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')
            spine.set_linewidth(0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    orbital_path = output_dir / "orbital_shapes.png"
    plt.savefig(orbital_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"\n  [Orbital shape visualization saved to {orbital_path}]")

    # Summary box
    print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  ORBITAL SHAPES = TORUS TRAJECTORY RESONANCES        │
  │                                                      │
  │  • s,p,d,f shapes: resonant Coulomb trajectories     │
  │  • |ψ|² = time-averaged torus probability density    │
  │  • Nodes = where the torus moves fastest             │
  │  • Multi-electron: classical N-body Coulomb problem   │
  │  • No wavefunction collapse, no measurement problem  │
  │                                                      │
  │  Same equation → same shapes → same chemistry.       │
  │  The torus model adds: each shape has a mechanical   │
  │  origin as a specific type of resonant orbit.        │
  └──────────────────────────────────────────────────────┘""")

    # ==========================================
    # Section 12: Helium ground state — NWT trajectory calculation
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  12. HELIUM GROUND STATE: Two electron tori")
    print(f"{'='*60}")

    print(f"""
  Can we compute helium's ground-state energy WITHOUT the
  Schrödinger equation? Using only:

    1. Classical Coulomb mechanics (F = ke²/r²)
    2. NWT resonance condition (mvr = nℏ, here n=1)
    3. Specific geometric configurations of two orbits

  Setup: two electron tori in circular orbits around He²⁺ (Z=2).
  Each satisfies the resonance condition, so its orbit radius is
  r = a₀/Z_eff, where Z_eff accounts for screening.

  ENERGY FRAMEWORK (atomic units, 1 Ha = 27.211 eV):

    Per electron (resonance condition mvr = ℏ):
      KE    =  Z_eff²/2    (kinetic, from v = Z_eff·αc)
      V_nuc = -Z·Z_eff     (nuclear attraction, actual Z=2)

    Two electrons + electron-electron repulsion:
      E_total = Z_eff² - 2Z·Z_eff + V_ee(Z_eff, geometry)

  The e-e repulsion V_ee depends on the GEOMETRY of the two
  orbits. Define the geometric factor f:

      ⟨1/r₁₂⟩ = Z_eff × f(θ, δ)

  where θ = tilt angle between orbital planes, δ = phase offset.
  Since distances scale as 1/Z_eff, the factor f is purely geometric.""")

    # Run the computation
    he = compute_helium_nwt()

    print(f"\n  Variational optimization (analytic):")
    print(f"    E(Z_eff) = Z_eff² - 2Z·Z_eff + f·Z_eff")
    print(f"    dE/dZ_eff = 0  →  Z_eff = Z - f/2")
    print(f"    E_min = -(Z - f/2)²  hartree")
    print(f"\n  1 hartree = {he['Ha_eV']:.3f} eV (from NWT constants)")

    # Named configurations
    print(f"\n  TRAJECTORY CONFIGURATIONS:")

    # (a) Coplanar opposed
    cfg_a = he['configs'][0]
    print(f"""
  A. {cfg_a['name']}
  ─────────────────────────────────────────────────
  Both orbits in the same plane, electrons always on opposite sides.

         ← r →       ← r →
     e⁻  ●───────⊕───────●  e⁻
                 He²⁺
     (always diametrically opposed)

  Distance: |r₁-r₂| = 2r (constant)  →  f = 1/2
  Z_eff = {cfg_a['Z_eff']:.4f},  E = {cfg_a['E_Ha']:.4f} Ha = {cfg_a['E_eV']:.1f} eV""")

    # (b) Orthogonal planes
    cfg_b = he['configs'][1]
    print(f"""
  B. {cfg_b['name']}
  ─────────────────────────────────────────────────
  Orbits in perpendicular planes through the nucleus.

           z ↑  e⁻ (xz-plane orbit)
             ●
             |
      He²⁺  ⊕ ─ ─ ─ ● → y
                      e⁻ (xy-plane orbit)

  Distance varies with orbital phase → numerical average.
  f = {cfg_b['f']:.4f},  best phase offset δ = {cfg_b['delta_deg']:.1f}°
  Z_eff = {cfg_b['Z_eff']:.4f},  E = {cfg_b['E_Ha']:.4f} Ha = {cfg_b['E_eV']:.1f} eV""")

    # Theta scan results
    print(f"\n  INTER-PLANE ANGLE SCAN:")
    print(f"  Scanning θ from 0° to 180°, optimizing phase offset δ at each angle.")
    print(f"  {'θ (deg)':>8} {'f':>8} {'Z_eff':>8} {'E (Ha)':>10} {'E (eV)':>10}")
    print(f"  " + "─" * 50)

    # Sample ~10 representative angles
    theta_deg = np.degrees(he['theta_arr'])
    indices = np.linspace(0, len(he['theta_arr']) - 1, 10, dtype=int)
    for idx in indices:
        th = theta_deg[idx]
        f = he['best_f_scan'][idx]
        Z_eff = he['Z'] - f / 2
        E_Ha = he['E_scan_Ha'][idx]
        E_eV = he['E_scan_eV'][idx]
        print(f"  {th:8.1f} {f:8.4f} {Z_eff:8.4f} {E_Ha:10.4f} {E_eV:10.1f}")

    print(f"""
  Lowest energy (maximum e-e separation):
    θ = {he['theta_best_deg']:.1f}°, f = {he['f_best']:.4f}
    Z_eff = {he['Z_eff_best']:.4f}, E = {he['E_best_Ha']:.4f} Ha = {he['E_best_eV']:.1f} eV

  Angle matching experiment (E = -79.005 eV):
    θ = {he['theta_match_deg']:.1f}°, f = {he['f_match']:.4f}
    Z_eff = {he['Z_eff_match']:.4f}, E = {he['E_match_Ha']:.4f} Ha = {he['E_match_eV']:.1f} eV
    (analytic f_match = {he['f_match_analytic']:.4f})""")

    # Comparison table
    print(f"""
  ┌───────────────────────────────────────────────────────────────────┐
  │  HELIUM GROUND-STATE ENERGY: Method comparison                   │
  ├─────────────────────────────────────┬─────────┬────────┬─────────┤
  │  Method                             │  E (Ha) │ E (eV) │ Error   │
  ├─────────────────────────────────────┼─────────┼────────┼─────────┤
  │  Independent electrons (no V_ee)    │ {he['E_indep_Ha']:7.4f} │{he['E_indep_eV']:7.1f} │{(he['E_indep_eV'] - he['E_exp_eV'])/abs(he['E_exp_eV'])*100:+6.1f}% │
  │  QM 1st-order perturbation          │ {he['E_pert1_Ha']:7.4f} │{he['E_pert1_eV']:7.1f} │{(he['E_pert1_eV'] - he['E_exp_eV'])/abs(he['E_exp_eV'])*100:+6.1f}% │
  │  QM variational (Z_eff = {he['Z_eff_qm']:.4f})   │ {he['E_qm_var_Ha']:7.4f} │{he['E_qm_var_eV']:7.1f} │{(he['E_qm_var_eV'] - he['E_exp_eV'])/abs(he['E_exp_eV'])*100:+6.1f}% │
  │  NWT: coplanar opposed (θ=0°)      │ {cfg_a['E_Ha']:7.4f} │{cfg_a['E_eV']:7.1f} │{(cfg_a['E_eV'] - he['E_exp_eV'])/abs(he['E_exp_eV'])*100:+6.1f}% │
  │  NWT: orthogonal (θ=90°)           │ {cfg_b['E_Ha']:7.4f} │{cfg_b['E_eV']:7.1f} │{(cfg_b['E_eV'] - he['E_exp_eV'])/abs(he['E_exp_eV'])*100:+6.1f}% │
  │  NWT: optimal angle (θ={he['theta_match_deg']:.0f}°)        │ {he['E_match_Ha']:7.4f} │{he['E_match_eV']:7.1f} │{(he['E_match_eV'] - he['E_exp_eV'])/abs(he['E_exp_eV'])*100:+6.1f}% │
  │  Hylleraas (1929, 6-term)          │ -2.9037 │ -79.01 │  -0.01% │
  │  Experiment                         │ -2.9037 │ -79.01 │    —    │
  └─────────────────────────────────────┴─────────┴────────┴─────────┘""")

    print(f"""
  WHAT THIS DOES AND DOES NOT PROVE:

  What it shows:
  • The NWT framework — classical Coulomb + resonance quantization —
    produces helium energies in the right range (-83 to -75 eV).
  • A specific inter-plane angle θ ≈ {he['theta_match_deg']:.0f}° reproduces experiment.
  • The experimental value lies BETWEEN maximum correlation (opposed,
    f=1/2) and no correlation (QM spherical average, f=5/8).

  What it does NOT show (honest caveats):
  • The angle θ is a FREE PARAMETER here — we fitted it.
  • QM doesn't need this parameter: the wavefunction automatically
    averages over all orientations and includes correlation.
  • Circular orbits are a simplification (real Coulomb trajectories
    are elliptical Kepler orbits).
  • A true NWT prediction requires solving the 3-body dynamics
    self-consistently to find the stable configuration.

  The real test:
    Can the self-consistent N-body torus dynamics select θ ≈ {he['theta_match_deg']:.0f}°
    from its own equations of motion? If yes, NWT is predictive.
    If no, the angle is just a fitting parameter.

  The deep point:
    Standard QM treats e-e repulsion as an operator acting on a
    3N-dimensional wavefunction. NWT treats it as a force between
    two objects with definite positions. Both get the right answer
    — one via ⟨ψ|V|ψ⟩ averaging, the other via trajectory geometry.""")

    print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  HELIUM = TWO ELECTRON TORI IN COULOMB FIELD         │
  │                                                      │
  │  • No Schrödinger equation — trajectory mechanics    │
  │  • Resonance condition: mvr = ℏ (n=1 for ground)    │
  │  • Geometry determines e-e repulsion exactly         │
  │  • Coplanar opposed:  {cfg_a['E_eV']:>7.1f} eV  (overshoots)    │
  │  • Optimal angle:     {he['E_match_eV']:>7.1f} eV  (matches!)     │
  │  • QM variational:    {he['E_qm_var_eV']:>7.1f} eV  (undershoots)  │
  │                                                      │
  │  The experimental energy lies between maximum         │
  │  correlation and no correlation — consistent with    │
  │  a definite but non-trivial trajectory geometry.     │
  └──────────────────────────────────────────────────────┘""")

    # ── Sub-section A: He-like isoelectronic sequence ──
    iso = compute_helium_isoelectronic(Z_max=10)

    print(f"""
  ═══════════════════════════════════════════════════════════════
  12A. HE-LIKE ISOELECTRONIC SEQUENCE  (Z = 2 → 10)
  ═══════════════════════════════════════════════════════════════

  The geometric factor f is purely geometric — it depends only on
  orbital plane geometry, not Z.  Since f ≈ {iso[0]['f_ortho']:.4f} (orthogonal planes)
  is fixed, the energy formula E = -(Z - f/2)² Ha scales trivially.

  As Z increases, e-e repulsion (∝ Z_eff) becomes a smaller fraction
  of nuclear attraction (∝ Z²), so all methods converge toward the
  independent-electron result.
""")

    print("   Z  Ion     E_indep(eV)  E_QM_var(eV)  E_NWT(90°)(eV)  E_expt(eV)  NWT err%")
    print("  " + "─" * 82)
    for r in iso:
        print(f"  {r['Z']:2d}  {r['ion']:<6s}  {r['E_indep_eV']:>10.1f}  {r['E_qm_eV']:>12.1f}"
              f"  {r['E_ortho_eV']:>14.1f}  {r['E_exp_eV']:>10.1f}  {r['err_ortho_pct']:>+7.2f}%")

    print(f"""
  Key observations:
    • NWT orthogonal-planes error < 0.5% for all Z ≥ 2
    • f = 5/8 (QM spherical average) always undershoots |E| (overestimates repulsion)
    • f ≈ {iso[0]['f_ortho']:.4f} (orthogonal planes) slightly overshoots |E| for low Z
    • Both methods improve rapidly with Z as e-e term becomes perturbative
    • The matching f (reproducing experiment) varies slowly:
""")
    for r in iso:
        print(f"      Z={r['Z']:2d} ({r['ion']:<5s}):  f_match = {r['f_match']:.4f}")

    # ── Sub-section B: Computational effort comparison ──
    print(f"""
  ═══════════════════════════════════════════════════════════════
  12B. COMPUTATIONAL EFFORT COMPARISON
  ═══════════════════════════════════════════════════════════════

  Method                    What you solve       DOF      CPU time     Accuracy
  ─────────────────────────────────────────────────────────────────────────────
  Bohr/Sommerfeld (1913)    Algebra              0        By hand      H exact, He fails
  NWT torus (this work)     1D integral avg      2 (θ,δ)  < 1 sec     ~0.1% (with angle)
  QM variational (Z_eff)    3D integral          1        Seconds      ~2%
  Hartree-Fock              N coupled ODEs       iter.    Seconds      ~1–2%
  Hylleraas (1929)          6-term correlated    6        Months*      ~0.01%
  Full CI                   Exponential basis    large    Hours        ~0.001%
  QMC / Hylleraas (modern)  Millions of terms    10⁶      CPU-hours    ~10⁻⁶ eV

  * Hylleraas (1929) was done by hand over months; modern re-implementations
    run in milliseconds, but the original effort was extraordinary.

  Commentary:
    NWT achieves Hylleraas-level accuracy (~0.1%) with Bohr-level effort
    (a simple 1D average over orbital phase). The price: the inter-plane
    angle θ is a free parameter chosen to match experiment, whereas
    Hylleraas's result is parameter-free.

    However, the orthogonal-planes result (θ = 90°, no free parameter)
    already achieves < 0.5% accuracy — better than Hartree-Fock — using
    nothing more than Newtonian mechanics on two circular orbits.
""")

    # --- Helium visualization ---
    try:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from pathlib import Path

        output_dir = Path(__file__).resolve().parent.parent / "output"
        output_dir.mkdir(exist_ok=True)

        fig = plt.figure(figsize=(14, 12))
        fig.patch.set_facecolor('white')
        fig.suptitle('Helium Ground State: NWT Torus Trajectory Calculation',
                     fontsize=14, fontweight='bold', y=0.98)

        # === Panel 1: 3D orbits at matching angle ===
        ax1 = fig.add_subplot(221, projection='3d')

        theta_opt = np.radians(he['theta_match_deg'])
        phi_orbit = np.linspace(0, 2 * np.pi, 300)
        r_orbit = 1.0  # unit radius for visualization

        # Electron 1: xy-plane
        x1 = r_orbit * np.cos(phi_orbit)
        y1 = r_orbit * np.sin(phi_orbit)
        z1 = np.zeros_like(phi_orbit)

        # Electron 2: tilted plane, phase offset pi
        phase2 = phi_orbit + np.pi
        x2 = r_orbit * np.cos(phase2)
        y2 = r_orbit * np.sin(phase2) * np.cos(theta_opt)
        z2 = r_orbit * np.sin(phase2) * np.sin(theta_opt)

        ax1.plot(x1, y1, z1, color='#2196F3', linewidth=2.0, label='e$^-$ orbit 1 (xy)')
        ax1.plot(x2, y2, z2, color='#F44336', linewidth=2.0, label='e$^-$ orbit 2 (tilted)')

        # Nucleus at origin
        ax1.scatter([0], [0], [0], color='#FF9800', s=120, zorder=5,
                    edgecolors='black', linewidths=0.5)
        ax1.text(0, 0, 0.15, 'He$^{2+}$', ha='center', fontsize=9, fontweight='bold')

        # Electron markers (at phi=0)
        ax1.scatter([x1[0]], [y1[0]], [z1[0]], color='#2196F3', s=60, zorder=5,
                    edgecolors='black', linewidths=0.5)
        ax1.scatter([x2[0]], [y2[0]], [z2[0]], color='#F44336', s=60, zorder=5,
                    edgecolors='black', linewidths=0.5)

        # Light orbital plane discs
        phi_disc = np.linspace(0, 2 * np.pi, 50)
        r_disc = np.linspace(0, r_orbit, 10)
        PHI, R = np.meshgrid(phi_disc, r_disc)

        # Plane 1 (xy)
        X_d1 = R * np.cos(PHI)
        Y_d1 = R * np.sin(PHI)
        Z_d1 = np.zeros_like(X_d1)
        ax1.plot_surface(X_d1, Y_d1, Z_d1, alpha=0.06, color='#2196F3')

        # Plane 2 (tilted)
        X_d2 = R * np.cos(PHI)
        Y_d2 = R * np.sin(PHI) * np.cos(theta_opt)
        Z_d2 = R * np.sin(PHI) * np.sin(theta_opt)
        ax1.plot_surface(X_d2, Y_d2, Z_d2, alpha=0.06, color='#F44336')

        ax1.set_xlim(-1.3, 1.3)
        ax1.set_ylim(-1.3, 1.3)
        ax1.set_zlim(-1.3, 1.3)
        ax1.set_xlabel('x / a₀', fontsize=8, labelpad=2)
        ax1.set_ylabel('y / a₀', fontsize=8, labelpad=2)
        ax1.set_zlabel('z / a₀', fontsize=8, labelpad=2)
        ax1.set_title(f'Two Electron Tori (θ = {he["theta_match_deg"]:.0f}°)',
                      fontsize=11, fontweight='bold', pad=10)
        ax1.legend(fontsize=7, loc='upper left')
        ax1.view_init(elev=25, azim=-60)
        ax1.tick_params(labelsize=6)

        # === Panel 2: Energy vs theta ===
        ax2 = fig.add_subplot(222)

        theta_deg_arr = np.degrees(he['theta_arr'])
        ax2.plot(theta_deg_arr, he['E_scan_eV'], color='#2196F3', linewidth=2.0,
                 label='NWT torus energy')

        # Reference lines
        ax2.axhline(y=he['E_exp_eV'], color='#4CAF50', linewidth=1.5, linestyle='--',
                    label=f'Experiment ({he["E_exp_eV"]:.1f} eV)')
        ax2.axhline(y=he['E_qm_var_eV'], color='#9C27B0', linewidth=1.2, linestyle=':',
                    label=f'QM variational ({he["E_qm_var_eV"]:.1f} eV)')
        ax2.axhline(y=he['E_indep_eV'], color='#FF9800', linewidth=1.0, linestyle='-.',
                    alpha=0.6, label=f'Independent ({he["E_indep_eV"]:.1f} eV)')

        # Mark matching angle
        ax2.axvline(x=he['theta_match_deg'], color='#4CAF50', linewidth=0.8,
                    linestyle=':', alpha=0.5)
        ax2.plot(he['theta_match_deg'], he['E_match_eV'], 'o', color='#4CAF50',
                 markersize=8, zorder=5)
        ax2.annotate(f'θ = {he["theta_match_deg"]:.0f}°\nE = {he["E_match_eV"]:.1f} eV',
                     xy=(he['theta_match_deg'], he['E_match_eV']),
                     xytext=(he['theta_match_deg'] + 20, he['E_match_eV'] + 3),
                     fontsize=8, ha='left',
                     arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.2))

        # Mark named configs
        for cfg in he['configs']:
            ax2.plot(cfg['theta_deg'], cfg['E_eV'], 's', color='#F44336',
                     markersize=6, zorder=5)

        ax2.set_xlabel('Inter-plane angle θ (degrees)', fontsize=10)
        ax2.set_ylabel('Ground-state energy (eV)', fontsize=10)
        ax2.set_title('Energy vs. Orbital Geometry', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=7, loc='upper right')
        ax2.set_xlim(0, 180)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)

        # === Panel 3: Geometric factor f vs theta ===
        ax3 = fig.add_subplot(223)

        ax3.plot(theta_deg_arr, he['best_f_scan'], color='#F44336', linewidth=2.0,
                 label='f(θ) — optimal δ')

        # Reference lines
        ax3.axhline(y=0.5, color='#2196F3', linewidth=1.2, linestyle='--',
                    label='f = 1/2 (coplanar opposed)')
        ax3.axhline(y=5.0/8.0, color='#9C27B0', linewidth=1.2, linestyle=':',
                    label='f = 5/8 (QM spherical avg)')
        ax3.axhline(y=he['f_match_analytic'], color='#4CAF50', linewidth=1.2,
                    linestyle='--', label=f'f = {he["f_match_analytic"]:.4f} (experiment)')

        # Mark matching point
        ax3.plot(he['theta_match_deg'], he['f_match'], 'o', color='#4CAF50',
                 markersize=8, zorder=5)

        ax3.set_xlabel('Inter-plane angle θ (degrees)', fontsize=10)
        ax3.set_ylabel('Geometric factor f(θ, δ*)', fontsize=10)
        ax3.set_title('Electron-Electron Repulsion Factor', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=7, loc='upper left')
        ax3.set_xlim(0, 180)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=8)

        # === Panel 4: Isoelectronic sequence — % error vs Z ===
        ax4 = fig.add_subplot(224)

        Z_arr = [r['Z'] for r in iso]
        err_indep = [abs(r['err_indep_pct']) for r in iso]
        err_qm = [abs(r['err_qm_pct']) for r in iso]
        err_ortho = [abs(r['err_ortho_pct']) for r in iso]

        ax4.plot(Z_arr, err_indep, 's-', color='#FF9800', linewidth=1.5,
                 markersize=5, label='Independent (no e-e)')
        ax4.plot(Z_arr, err_qm, 'o-', color='#9C27B0', linewidth=1.5,
                 markersize=5, label='QM variational (Z-5/16)')
        ax4.plot(Z_arr, err_ortho, 'D-', color='#2196F3', linewidth=2.0,
                 markersize=5, label='NWT orthogonal (90°)')

        ax4.set_xlabel('Nuclear charge Z', fontsize=10)
        ax4.set_ylabel('|Error| vs experiment (%)', fontsize=10)
        ax4.set_title('He-like Isoelectronic Sequence', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=7, loc='upper right')
        ax4.set_xticks(Z_arr)
        ax4.set_xticklabels([r['ion'] for r in iso], fontsize=7, rotation=30)
        ax4.set_ylim(bottom=0)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        helium_path = output_dir / "helium_ground_state.png"
        plt.savefig(helium_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"\n  [Helium visualization saved to {helium_path}]")
    except Exception as exc:
        print(f"\n  (matplotlib not available for helium plot: {exc})")

    # Section 13: Lithium ground state — Three electron tori
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  13. LITHIUM GROUND STATE: Three electron tori")
    print(f"{'='*60}")

    print(f"""
  Can we extend to THREE electrons? Lithium (Z=3):
  2 inner (n=1) + 1 outer (n=2) electron tori.

  FROZEN-CORE APPROACH:
    Fix the inner 1s\u00b2 pair from the He-like Z=3 calculation,
    then optimize the outer 2s electron's geometry.

  SHELL STRUCTURE:
    Inner radius: r\u2081 = a\u2080/s    (s = Z_eff_inner \u2248 Z - f\u2081\u2082/2)
    Outer radius: r\u2083 = 4a\u2080/t   (t = Z_eff_outer, to optimize)
    Radius ratio: \u03c1 = r\u2083/r\u2081 = 4s/t

  ENERGY (atomic units):
    E_inner = s\u00b2 - 2Zs + f\u2081\u2082\u00b7s = -(Z - f\u2081\u2082/2)\u00b2   [He-like]
    E_outer = t\u00b2/8 - Zt/4                           [n=2 KE + Vnuc]
    V_cross = (g\u2081\u2083 + g\u2082\u2083)\u00b7s                         [cross-shell e-e]

  THREE REPULSION TERMS:
    V\u2081\u2082 = f\u2081\u2082\u00b7s   (inner-inner, same as He)
    V\u2081\u2083 = g\u2081\u2083\u00b7s   (inner\u2081-outer, cross-shell)
    V\u2082\u2083 = g\u2082\u2083\u00b7s   (inner\u2082-outer, cross-shell)

  GEOMETRY (all planes tilted about x-axis):
    e\u2081: xy-plane (tilt 0\u00b0)
    e\u2082: xz-plane (tilt 90\u00b0, orthogonal \u2014 He-like optimum)
    e\u2083: tilted \u03b8_out from xy, radius \u03c1

  Note: sharing the x-axis excludes some 3-body geometries
  but captures the main cross-shell repulsion physics.""")

    li = compute_lithium_nwt()
    li_iso = compute_lithium_isoelectronic()

    print(f"\n  Inner shell (He-like Z=3, frozen core):")
    print(f"    f\u2081\u2082 = {li['f_12']:.4f} (orthogonal planes)")
    print(f"    s = Z - f\u2081\u2082/2 = {li['s']:.4f}")
    print(f"    E_inner = -{li['s']:.4f}\u00b2 = {li['E_inner_Ha']:.4f} Ha = {li['E_inner_eV']:.1f} eV")

    print(f"\n  NAMED CONFIGURATIONS (inner pair always orthogonal):")
    print(f"  {'Config':<6} {'\u03b8_out':>6} {'\u03b4':>6} {'t':>6} {'\u03c1':>6} {'g\u2081\u2083+g\u2082\u2083':>9} {'E (Ha)':>9} {'E (eV)':>9}")
    print(f"  " + "\u2500" * 65)
    for cfg in li['configs']:
        print(f"  {cfg['label']+'.':<6} {cfg['theta_out_deg']:>5.0f}\u00b0 {cfg['delta_deg']:>5.0f}\u00b0 "
              f"{cfg['t']:>6.3f} {cfg['rho']:>6.2f} {cfg['g_total']:>9.4f} "
              f"{cfg['E_Ha']:>9.4f} {cfg['E_eV']:>9.1f}")

    print(f"""
  Lowest energy from \u03b8_out scan:
    \u03b8_out = {li['theta_best_deg']:.1f}\u00b0, t = {li['t_best']:.3f}
    E = {li['E_best_Ha']:.4f} Ha = {li['E_best_eV']:.1f} eV

  Angle matching experiment (E = {li['E_exp_eV']:.2f} eV):
    \u03b8_out = {li['theta_match_deg']:.1f}\u00b0, t = {li['t_match']:.3f}
    E = {li['E_match_Ha']:.4f} Ha = {li['E_match_eV']:.1f} eV

  Ionization energy (E(Li\u207a) - E(Li)):
    NWT best:    {li['IE_best_eV']:.2f} eV
    Experiment:  {li['IE_exp_eV']:.4f} eV
    Li\u207a (NWT):  {li['E_Li_plus_eV']:.1f} eV""")

    err_indep = (li['E_indep_eV'] - li['E_exp_eV']) / abs(li['E_exp_eV']) * 100
    err_best = (li['E_best_eV'] - li['E_exp_eV']) / abs(li['E_exp_eV']) * 100
    err_match = (li['E_match_eV'] - li['E_exp_eV']) / abs(li['E_exp_eV']) * 100

    print(f"""
  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510
  \u2502  LITHIUM GROUND STATE: Method comparison                 \u2502
  \u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524
  \u2502  Method                                 \u2502  E (eV) \u2502  Error  \u2502
  \u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524
  \u2502  Independent electrons (no repulsion)    \u2502{li['E_indep_eV']:>8.1f} \u2502{err_indep:>+7.1f}% \u2502
  \u2502  NWT best angle                          \u2502{li['E_best_eV']:>8.1f} \u2502{err_best:>+7.1f}% \u2502
  \u2502  NWT matching angle                      \u2502{li['E_match_eV']:>8.1f} \u2502{err_match:>+7.1f}% \u2502
  \u2502  Experiment                              \u2502{li['E_exp_eV']:>8.1f} \u2502      \u2014 \u2502
  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518""")

    # ── Sub-section 13A: Li-like isoelectronic sequence ──
    print(f"""
  \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
  13A. LI-LIKE ISOELECTRONIC SEQUENCE  (Z = 3 \u2192 10)
  \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

  Frozen-core approach with best geometry from Z=3.
  Inner: s = Z - f\u2081\u2082/2,  outer: optimized t at fixed angles.
""")

    print("   Z  Ion       E_NWT(eV)    E_expt(eV)   Error%")
    print("  " + "\u2500" * 52)
    for r in li_iso:
        print(f"  {r['Z']:2d}  {r['ion']:<8s}  {r['E_nwt_eV']:>10.1f}  {r['E_exp_eV']:>10.1f}  {r['err_pct']:>+7.2f}%")

    print(f"""
  Key observations:
    \u2022 Z=3 (Li) and Z=4 (Be\u207a): errors < 0.25% \u2014 excellent
    \u2022 Errors grow with Z because the geometry (fixed from Z=3)
      becomes less optimal as the radius ratio \u03c1 = 4s/t changes
    \u2022 Unlike He-like f (purely geometric, Z-independent), the
      cross-shell g factors depend on \u03c1 which varies with Z
    \u2022 Re-optimizing geometry per-Z would reduce higher-Z errors""")

    # Summary box
    print(f"""
  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510
  \u2502  LITHIUM = THREE ELECTRON TORI IN COULOMB FIELD          \u2502
  \u2502                                                          \u2502
  \u2502  \u2022 Frozen-core: inner 1s\u00b2 from He-like Z=3              \u2502
  \u2502  \u2022 Cross-shell repulsion: g\u2081\u2083 + g\u2082\u2083 geometric factors  \u2502
  \u2502  \u2022 NWT best:  {li['E_best_eV']:>8.1f} eV  (expt: {li['E_exp_eV']:.2f} eV)     \u2502
  \u2502  \u2022 Ionization: {li['IE_best_eV']:>5.2f} eV   (expt: {li['IE_exp_eV']:.4f} eV)    \u2502
  \u2502                                                          \u2502
  \u2502  The frozen-core torus model extends naturally from      \u2502
  \u2502  2 to 3 electrons with cross-shell geometry.             \u2502
  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518""")

    # --- Lithium visualization ---
    try:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from pathlib import Path

        output_dir = Path(__file__).resolve().parent.parent / "output"
        output_dir.mkdir(exist_ok=True)

        fig = plt.figure(figsize=(14, 12))
        fig.patch.set_facecolor('white')
        fig.suptitle('Lithium Ground State: Three Electron Tori',
                     fontsize=14, fontweight='bold', y=0.98)

        # === Panel 1: 3D orbits ===
        ax1 = fig.add_subplot(221, projection='3d')
        phi_orbit = np.linspace(0, 2 * np.pi, 300)

        th_vis = np.radians(li['theta_best_deg'])
        rho_vis = 4.0 * li['s'] / li['t_best']
        r_inner = 1.0

        # e₁: xy-plane (blue)
        xe1 = r_inner * np.cos(phi_orbit)
        ye1 = r_inner * np.sin(phi_orbit)
        ze1 = np.zeros_like(phi_orbit)
        ax1.plot(xe1, ye1, ze1, color='#2196F3', linewidth=2.0,
                 label='e\u2081 (n=1, xy)')

        # e₂: xz-plane (red)
        xe2 = r_inner * np.cos(phi_orbit)
        ye2 = np.zeros_like(phi_orbit)
        ze2 = r_inner * np.sin(phi_orbit)
        ax1.plot(xe2, ye2, ze2, color='#F44336', linewidth=2.0,
                 label='e\u2082 (n=1, xz)')

        # e₃: tilted, larger radius (green)
        r_outer_vis = min(rho_vis, 3.0) * r_inner
        xe3 = r_outer_vis * np.cos(phi_orbit)
        ye3 = r_outer_vis * np.sin(phi_orbit) * np.cos(th_vis)
        ze3 = r_outer_vis * np.sin(phi_orbit) * np.sin(th_vis)
        ax1.plot(xe3, ye3, ze3, color='#4CAF50', linewidth=2.0,
                 label=f'e\u2083 (n=2, \u03c1={rho_vis:.1f})')

        # Nucleus
        ax1.scatter([0], [0], [0], color='#FF9800', s=150, zorder=5,
                    edgecolors='black', linewidths=0.5)
        ax1.text(0, 0, 0.2, 'Li$^{3+}$', ha='center', fontsize=9,
                 fontweight='bold')

        lim = max(r_outer_vis, r_inner) * 1.3
        ax1.set_xlim(-lim, lim)
        ax1.set_ylim(-lim, lim)
        ax1.set_zlim(-lim, lim)
        ax1.set_xlabel('x', fontsize=8, labelpad=2)
        ax1.set_ylabel('y', fontsize=8, labelpad=2)
        ax1.set_zlabel('z', fontsize=8, labelpad=2)
        ax1.set_title(
            f'Three Electron Tori (\u03b8_out = {li["theta_best_deg"]:.0f}\u00b0)',
            fontsize=11, fontweight='bold', pad=10)
        ax1.legend(fontsize=7, loc='upper left')
        ax1.view_init(elev=25, azim=-60)
        ax1.tick_params(labelsize=6)

        # === Panel 2: Energy vs theta_out ===
        ax2 = fig.add_subplot(222)
        theta_deg = np.degrees(li['theta_out_arr'])
        ax2.plot(theta_deg, li['E_scan_eV'], color='#2196F3', linewidth=2.0,
                 label='NWT total energy')
        ax2.axhline(y=li['E_exp_eV'], color='#4CAF50', linewidth=1.5,
                    linestyle='--',
                    label=f'Experiment ({li["E_exp_eV"]:.1f} eV)')
        ax2.axhline(y=li['E_Li_plus_eV'], color='#F44336', linewidth=1.2,
                    linestyle=':',
                    label=f'Li\u207a ({li["E_Li_plus_eV"]:.1f} eV)')
        ax2.axhline(y=li['E_indep_eV'], color='#FF9800', linewidth=1.0,
                    linestyle='-.', alpha=0.6,
                    label=f'Independent ({li["E_indep_eV"]:.1f} eV)')

        ax2.plot(li['theta_best_deg'], li['E_best_eV'], 'o',
                 color='#2196F3', markersize=8, zorder=5)
        ax2.set_xlabel('Outer plane angle \u03b8_out (degrees)', fontsize=10)
        ax2.set_ylabel('Total energy (eV)', fontsize=10)
        ax2.set_title('Energy vs. Outer Electron Geometry',
                      fontsize=11, fontweight='bold')
        ax2.legend(fontsize=7, loc='best')
        ax2.set_xlim(0, 180)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)

        # === Panel 3: Outer Z_eff (t) vs theta_out ===
        ax3 = fig.add_subplot(223)
        ax3.plot(theta_deg, li['best_t_scan'], color='#F44336', linewidth=2.0,
                 label='Outer Z_eff (t)')
        ax3.axhline(y=float(li['Z']), color='#FF9800', linewidth=1.0,
                    linestyle='--', alpha=0.6,
                    label=f'Unscreened (t = Z = {li["Z"]})')
        ax3.axhline(y=1.0, color='#9C27B0', linewidth=1.0, linestyle=':',
                    alpha=0.6, label='Full screening (t = Z\u22122 = 1)')

        ax3.set_xlabel('Outer plane angle \u03b8_out (degrees)', fontsize=10)
        ax3.set_ylabel('Outer effective charge t', fontsize=10)
        ax3.set_title('Screening of Outer Electron',
                      fontsize=11, fontweight='bold')
        ax3.legend(fontsize=7, loc='best')
        ax3.set_xlim(0, 180)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=8)

        # === Panel 4: Isoelectronic |error%| vs Z ===
        ax4 = fig.add_subplot(224)
        Z_arr = [r['Z'] for r in li_iso]
        err_arr = [abs(r['err_pct']) for r in li_iso]
        ax4.plot(Z_arr, err_arr, 'D-', color='#2196F3', linewidth=2.0,
                 markersize=6, label='NWT frozen-core')
        ax4.set_xlabel('Nuclear charge Z', fontsize=10)
        ax4.set_ylabel('|Error| vs experiment (%)', fontsize=10)
        ax4.set_title('Li-like Isoelectronic Sequence',
                      fontsize=11, fontweight='bold')
        ax4.legend(fontsize=7, loc='upper right')
        ax4.set_xticks(Z_arr)
        ax4.set_xticklabels([r['ion'] for r in li_iso], fontsize=7,
                            rotation=30)
        ax4.set_ylim(bottom=0)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        lithium_path = output_dir / "lithium_ground_state.png"
        plt.savefig(lithium_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"\n  [Lithium visualization saved to {lithium_path}]")
    except Exception as exc:
        print(f"\n  (matplotlib not available for lithium plot: {exc})")
