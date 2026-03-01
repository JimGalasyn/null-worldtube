"""
Borromean resonance energy calculation.

Three (p,q) torus knots in Borromean configuration: axes along x, y, z.
Computes self-energy + pairwise mutual inductance to find whether the
Borromean topology introduces a mass scale beyond the single-torus result.

Physics: the proton as a collective resonance on a Borromean 3-torus
(three merged photons). If the linking energy depends on R differently
than the self-energy, the topology picks out a unique mass scale.
"""

import numpy as np
from .constants import (
    c, hbar, e_charge, mu0, alpha, MeV, TorusParams,
    PARTICLE_MASSES,
)
from .core import torus_knot_curve, compute_total_energy, find_self_consistent_radius


# ──────────────────────────────────────────────────────────────────────
# 1. Borromean geometry
# ──────────────────────────────────────────────────────────────────────

def borromean_torus_curves(R, p=2, q=1, N=500):
    """
    Generate three (p,q) torus knot curves in Borromean configuration.

    The three torii are related by cyclic permutation of coordinates
    (120° rotation about the (1,1,1) body diagonal), ensuring exact
    Z₃ symmetry:

        Torus A: axis along z  —  (x, y, z)          original
        Torus B: axis along x  —  (x, y, z) → (z, x, y)
        Torus C: axis along y  —  (x, y, z) → (y, z, x)

    Each torus has r = α × R (fine-structure tube radius).

    Returns list of 3 dicts with keys: xyz, dxyz, ds, params.
    """
    r = alpha * R
    params = TorusParams(R=R, r=r, p=p, q=q)
    lam, xyz_a, dxyz_a, ds_a = torus_knot_curve(params, N)

    # Curve A: axis along z — identity
    curve_a = {'xyz': xyz_a, 'dxyz': dxyz_a, 'ds': ds_a, 'params': params}

    # Curve B: axis along x — cyclic (x,y,z) → (z,x,y)
    xyz_b = np.stack([xyz_a[:, 2], xyz_a[:, 0], xyz_a[:, 1]], axis=-1)
    dxyz_b = np.stack([dxyz_a[:, 2], dxyz_a[:, 0], dxyz_a[:, 1]], axis=-1)
    curve_b = {'xyz': xyz_b, 'dxyz': dxyz_b, 'ds': ds_a.copy(), 'params': params}

    # Curve C: axis along y — cyclic (x,y,z) → (y,z,x)
    xyz_c = np.stack([xyz_a[:, 1], xyz_a[:, 2], xyz_a[:, 0]], axis=-1)
    dxyz_c = np.stack([dxyz_a[:, 1], dxyz_a[:, 2], dxyz_a[:, 0]], axis=-1)
    curve_c = {'xyz': xyz_c, 'dxyz': dxyz_c, 'ds': ds_a.copy(), 'params': params}

    return [curve_a, curve_b, curve_c]


# ──────────────────────────────────────────────────────────────────────
# 2. Mutual inductance (Neumann formula)
# ──────────────────────────────────────────────────────────────────────

def compute_mutual_inductance(curve1, curve2):
    """
    Mutual inductance via Neumann formula (vectorized).

        M = (μ₀/4π) ∮∮ (dl₁ · dl₂) / |r₁ − r₂|

    O(N²) per pair, fully vectorized with numpy broadcasting.

    Regularization: distances are clipped to the tube radius r,
    since the current is distributed over a finite cross-section,
    not a mathematical line. This handles the case where curves on
    orthogonal torii pass through the same point (e.g., A-C pair
    on the x-axis).

    Returns M in henries.
    """
    xyz1, dxyz1 = curve1['xyz'], curve1['dxyz']
    xyz2, dxyz2 = curve2['xyz'], curve2['dxyz']
    r_tube = curve1['params'].r  # tube radius for regularization
    N = len(xyz1)
    dlam = 2 * np.pi / N

    # Finite arc-length elements
    dl1 = dxyz1 * dlam  # (N, 3)
    dl2 = dxyz2 * dlam  # (N, 3)

    # Pairwise separation vectors: (N, 1, 3) − (1, N, 3) → (N, N, 3)
    sep = xyz1[:, np.newaxis, :] - xyz2[np.newaxis, :, :]
    dist = np.sqrt(np.sum(sep**2, axis=-1))  # (N, N)

    # Regularize: clip distance to tube radius (finite wire thickness)
    dist = np.maximum(dist, r_tube)

    # Dot products dl1[i] · dl2[j]: (N, 1, 3) * (1, N, 3) → sum → (N, N)
    dot = np.sum(dl1[:, np.newaxis, :] * dl2[np.newaxis, :, :], axis=-1)

    M = (mu0 / (4 * np.pi)) * np.sum(dot / dist)
    return M


# ──────────────────────────────────────────────────────────────────────
# 3. Mutual energy
# ──────────────────────────────────────────────────────────────────────

def compute_mutual_energy(M, I):
    """
    Energy from mutual inductance between one pair.

    Magnetic: U_mag = M × I²
    Electric: U_elec = U_mag  (v = c → equal field energies)
    Total per pair: 2 × M × I²

    Returns energy in joules.
    """
    return 2.0 * M * I**2


# ──────────────────────────────────────────────────────────────────────
# 4. Borromean total energy
# ──────────────────────────────────────────────────────────────────────

def compute_borromean_energy(R, p=2, q=1, N=500):
    """
    Total energy of three (p,q) torus knots in Borromean link.

        E = 3 × E_single  +  3 × 2 × M × I²
            ───────────       ─────────────────
            self-energies     mutual (3 pairs, mag+elec)

    Returns dict with full breakdown.
    """
    r = alpha * R
    params = TorusParams(R=R, r=r, p=p, q=q)

    # Single-torus energy (circulation + EM self-energy)
    E_single_J, E_single_MeV, breakdown = compute_total_energy(params, N)
    I = breakdown['I_amps']
    L_self = breakdown['L_ind_henry']

    # Generate Borromean curves and compute all three mutual inductances
    curves = borromean_torus_curves(R, p, q, N)
    M_AB = compute_mutual_inductance(curves[0], curves[1])
    M_AC = compute_mutual_inductance(curves[0], curves[2])
    M_BC = compute_mutual_inductance(curves[1], curves[2])

    # Use mean for total (should be identical by Z₃ symmetry)
    M_mean = (M_AB + M_AC + M_BC) / 3.0
    U_mutual_pair = compute_mutual_energy(M_mean, I)

    # Totals
    E_3singles = 3 * E_single_J
    U_mutual_total = 3 * U_mutual_pair
    E_borromean = E_3singles + U_mutual_total

    correction = E_borromean / E_3singles if E_3singles != 0 else float('inf')
    M_over_L = M_mean / L_self if L_self != 0 else float('inf')

    spread = max(abs(M_AB - M_AC), abs(M_AB - M_BC), abs(M_AC - M_BC))
    M_spread = spread / (abs(M_mean) + 1e-99)

    return {
        'R': R,
        'r': r,
        'R_fm': R * 1e15,
        'p': p, 'q': q,
        'E_single_J': E_single_J,
        'E_single_MeV': E_single_MeV,
        'E_3singles_J': E_3singles,
        'E_3singles_MeV': 3 * E_single_MeV,
        'M_AB': M_AB, 'M_AC': M_AC, 'M_BC': M_BC,
        'M_mean': M_mean,
        'M_symmetry_spread': M_spread,
        'U_mutual_pair_J': U_mutual_pair,
        'U_mutual_pair_MeV': U_mutual_pair / MeV,
        'U_mutual_total_J': U_mutual_total,
        'U_mutual_total_MeV': U_mutual_total / MeV,
        'E_borromean_J': E_borromean,
        'E_borromean_MeV': E_borromean / MeV,
        'mass_MeV': E_borromean / MeV,
        'correction_factor': correction,
        'M_over_L': M_over_L,
        'I_amps': I,
        'L_self': L_self,
        'breakdown': breakdown,
    }


# ──────────────────────────────────────────────────────────────────────
# 5. R-scan
# ──────────────────────────────────────────────────────────────────────

def scan_borromean_energy(R_min=1e-17, R_max=1e-14, n_points=50, p=2, q=1, N=500):
    """Sweep R logarithmically and compute Borromean energy at each point."""
    R_values = np.geomspace(R_min, R_max, n_points)
    results = []
    for R_val in R_values:
        results.append(compute_borromean_energy(R_val, p=p, q=q, N=N))
    return results


# ──────────────────────────────────────────────────────────────────────
# 6. Analysis output
# ──────────────────────────────────────────────────────────────────────

def _ascii_plot(results):
    """ASCII plot of correction factor vs R."""
    width = 55
    corrections = [r['correction_factor'] for r in results]
    R_fms = [r['R_fm'] for r in results]

    c_min, c_max = min(corrections), max(corrections)
    c_range = c_max - c_min

    if c_range < 1e-12:
        print(f"│  Correction factor ≈ {np.mean(corrections):.10f} (flat)")
        print(f"│")
        for i in range(0, len(results), max(1, len(results) // 15)):
            bar = " " * (width // 2) + "×"
            print(f"│  {R_fms[i]:8.4f} fm |{bar}")
        return

    for i in range(0, len(results), max(1, len(results) // 20)):
        frac = (corrections[i] - c_min) / c_range
        pos = int(frac * (width - 1))
        bar = " " * pos + "×"
        print(f"│  {R_fms[i]:8.4f} fm |{bar}")

    print(f"│  {'':>11} |{'─' * width}|")
    print(f"│  {'':>11}  {c_min:.8f}{' ' * max(0, width - 22)}{c_max:.8f}")


def print_borromean_analysis():
    """Full Borromean resonance energy analysis."""

    print("=" * 70)
    print("BORROMEAN RESONANCE ENERGY")
    print("  Three (p,q) torus knots in Borromean configuration")
    print("  Hypothesis: linking energy determines proton mass scale")
    print("=" * 70)

    p, q = 2, 1
    N = 500
    m_p_MeV = PARTICLE_MASSES['proton']

    # ─── 1. Geometry ──────────────────────────────────────────────
    print("\n┌─ 1. BORROMEAN GEOMETRY ─────────────────────────────────")
    print(f"│  Three ({p},{q}) torus knots on orthogonal torii")
    print(f"│  Each torus: r = αR = {alpha:.6f} × R")
    print(f"│  Torus A: axis ẑ   B: axis x̂   C: axis ŷ")
    print(f"│")
    print(f"│  Pairwise linking number: 0  (each pair unlinked)")
    print(f"│  Milnor μ̄(1,2,3) = ±1     (triple non-trivially linked)")
    print(f"└────────────────────────────────────────────────────────")

    # ─── 2. Convergence check ─────────────────────────────────────
    print("\n┌─ 2. CONVERGENCE CHECK ─────────────────────────────────")
    R_test = 0.5e-15  # 0.5 fm
    print(f"│  R = {R_test*1e15:.2f} fm")
    print(f"│")
    print(f"│  {'N':>6}  {'M (H)':>14}  {'E_borr (MeV)':>14}  {'correction':>12}")
    print(f"│  " + "─" * 52)

    for N_test in [200, 500, 1000]:
        res = compute_borromean_energy(R_test, p=p, q=q, N=N_test)
        print(f"│  {N_test:6d}  {res['M_mean']:14.6e}  "
              f"{res['E_borromean_MeV']:14.6f}  {res['correction_factor']:12.10f}")

    print(f"└────────────────────────────────────────────────────────")

    # ─── 3. Z₃ symmetry verification ─────────────────────────────
    print("\n┌─ 3. Z₃ SYMMETRY CHECK ─────────────────────────────────")
    res_sym = compute_borromean_energy(R_test, p=p, q=q, N=N)
    print(f"│  M_AB = {res_sym['M_AB']:+.6e} H")
    print(f"│  M_AC = {res_sym['M_AC']:+.6e} H")
    print(f"│  M_BC = {res_sym['M_BC']:+.6e} H")
    print(f"│  Spread: {res_sym['M_symmetry_spread']:.2e} (should be ≪ 1)")
    print(f"└────────────────────────────────────────────────────────")

    # ─── 4. Single vs Borromean at proton scale ───────────────────
    print("\n┌─ 4. SINGLE TORUS vs BORROMEAN AT PROTON SCALE ────────")

    sol = find_self_consistent_radius(m_p_MeV, p=p, q=q, r_ratio=alpha)
    R_p = None

    if sol:
        R_p = sol['R']
        print(f"│  Single-torus proton: R_p = {R_p*1e15:.6f} fm")
        print(f"│    E_single = {sol['E_total_MeV']:.4f} MeV  "
              f"(match: {sol['match_ppm']:.2f} ppm)")
        print(f"│")

        res_p = compute_borromean_energy(R_p, p=p, q=q, N=N)
        print(f"│  At R = R_p:")
        print(f"│    3 × E_single     = {res_p['E_3singles_MeV']:.4f} MeV")
        print(f"│    3 × U_mutual     = {res_p['U_mutual_total_MeV']:+.6f} MeV")
        print(f"│    E_borromean      = {res_p['E_borromean_MeV']:.4f} MeV")
        print(f"│    Correction factor = {res_p['correction_factor']:.10f}")
        print(f"│    M / L_self       = {res_p['M_over_L']:+.6e}")
        print(f"│")
        link_frac = res_p['U_mutual_total_MeV'] / res_p['E_borromean_MeV'] * 100
        print(f"│  Linking energy fraction: {link_frac:+.6f}%")
    else:
        print(f"│  Could not find self-consistent R for proton with ({p},{q})")

    print(f"└────────────────────────────────────────────────────────")

    # ─── 5. R-scan ────────────────────────────────────────────────
    print("\n┌─ 5. R-SCAN: BORROMEAN CORRECTION vs RADIUS ────────────")
    print(f"│  Scanning R from 0.01 fm to 10 fm")
    print(f"│")

    results = scan_borromean_energy(
        R_min=0.01e-15, R_max=10e-15, n_points=50, p=p, q=q, N=N
    )

    print(f"│  {'R (fm)':>10}  {'E_borr (MeV)':>14}  {'correction':>12}  "
          f"{'M/L_self':>12}  {'mass (MeV)':>12}")
    print(f"│  " + "─" * 66)

    step = max(1, len(results) // 20)
    for i in range(0, len(results), step):
        r = results[i]
        print(f"│  {r['R_fm']:10.4f}  {r['E_borromean_MeV']:14.4f}  "
              f"{r['correction_factor']:12.10f}  "
              f"{r['M_over_L']:+12.6e}  {r['mass_MeV']:12.4f}")

    corrections = [r['correction_factor'] for r in results]
    corr_min, corr_max = min(corrections), max(corrections)
    corr_range = corr_max - corr_min
    corr_mean = np.mean(corrections)
    relative_var = corr_range / (abs(corr_mean) + 1e-99)

    print(f"│")
    print(f"│  Correction factor range: [{corr_min:.10f}, {corr_max:.10f}]")
    print(f"│  Variation: {corr_range:.2e}  (mean: {corr_mean:.10f})")

    if relative_var < 1e-3:
        print(f"│  → CONSTANT (R-independent): Borromean energy is a pure")
        print(f"│    topological rescaling of 3 × E_single.")
        print(f"│    No new mass scale from linking.")
    else:
        print(f"│  → R-DEPENDENT: the Borromean topology introduces")
        print(f"│    scale-dependent structure. Look for resonance.")

    print(f"└────────────────────────────────────────────────────────")

    # ─── 6. ASCII plot ────────────────────────────────────────────
    print("\n┌─ 6. CORRECTION FACTOR vs R ────────────────────────────")
    _ascii_plot(results)
    print(f"└────────────────────────────────────────────────────────")

    # ─── 7. Borromean proton mass ─────────────────────────────────
    print("\n┌─ 7. BORROMEAN PROTON MASS ─────────────────────────────")
    print(f"│  Find R where E_borromean = m_p = {m_p_MeV:.4f} MeV")
    print(f"│")

    target_J = m_p_MeV * MeV

    # Bisection (geometric) over R
    R_lo, R_hi = 1e-18, 1e-12
    E_lo = compute_borromean_energy(R_lo, p=p, q=q, N=200)['E_borromean_J']
    E_hi = compute_borromean_energy(R_hi, p=p, q=q, N=200)['E_borromean_J']

    if E_lo > target_J > E_hi:
        for _ in range(60):
            R_mid = np.sqrt(R_lo * R_hi)
            E_mid = compute_borromean_energy(R_mid, p=p, q=q, N=200)['E_borromean_J']
            if E_mid > target_J:
                R_lo = R_mid
            else:
                R_hi = R_mid

        R_borr = np.sqrt(R_lo * R_hi)
        res_borr = compute_borromean_energy(R_borr, p=p, q=q, N=N)

        print(f"│  R_borromean = {R_borr*1e15:.6f} fm")
        print(f"│  E_borromean = {res_borr['E_borromean_MeV']:.4f} MeV")
        match_ppm = abs(res_borr['E_borromean_MeV'] - m_p_MeV) / m_p_MeV * 1e6
        print(f"│  Match: {match_ppm:.1f} ppm")
        print(f"│")

        if R_p:
            R_ratio = R_borr / R_p
            print(f"│  R_borromean / R_single_proton = {R_ratio:.8f}")
            if abs(R_ratio - 3.0) < 0.01:
                print(f"│  ≈ 3  (mass splits equally among 3 photons)")
            print(f"│")

        print(f"│  Breakdown at R_borromean:")
        print(f"│    Per-photon E_circ = {res_borr['breakdown']['E_circ_MeV']:.4f} MeV")
        print(f"│    Per-photon E_self = {res_borr['breakdown']['U_total_MeV']:.6f} MeV")
        print(f"│    Per-photon E_total= {res_borr['E_single_MeV']:.4f} MeV")
        print(f"│    3 × single        = {res_borr['E_3singles_MeV']:.4f} MeV")
        print(f"│    Mutual energy     = {res_borr['U_mutual_total_MeV']:+.6f} MeV")
        print(f"│    E_borromean       = {res_borr['E_borromean_MeV']:.4f} MeV")
    else:
        print(f"│  Cannot bracket m_p in R range [1e-18, 1e-12] m")
        print(f"│  E(R_lo) = {E_lo/MeV:.4e} MeV, E(R_hi) = {E_hi/MeV:.4e} MeV")

    print(f"└────────────────────────────────────────────────────────")

    # ─── 8. Physics summary ───────────────────────────────────────
    print("\n┌─ 8. PHYSICS SUMMARY ───────────────────────────────────")
    print(f"│")
    print(f"│  Q1: What is M/L_self?")
    print(f"│      {res_sym['M_over_L']:+.6e}  "
          f"(at R = {R_test*1e15:.2f} fm, N = {N})")
    print(f"│")

    print(f"│  Q2: Is the correction factor R-independent?")
    if relative_var < 1e-3:
        print(f"│      YES — correction ≈ {corr_mean:.10f}")
        print(f"│      Borromean energy = {corr_mean:.6f} × 3 × E_single")
        borr_corr = corr_mean - 1.0
        print(f"│      Linking adds {borr_corr*100:+.6f}% to 3 × E_single")
        print(f"│      → No new mass scale from topology alone.")
    else:
        print(f"│      NO — varies by {relative_var*100:.4f}% across R range")
        # Find R of extremum
        idx_min = np.argmin(corrections)
        idx_max = np.argmax(corrections)
        print(f"│      Min correction at R = {results[idx_min]['R_fm']:.4f} fm")
        print(f"│      Max correction at R = {results[idx_max]['R_fm']:.4f} fm")

    print(f"│")
    if R_p:
        res_at_Rp = compute_borromean_energy(R_p, p=p, q=q, N=N)
        borr_mass = res_at_Rp['E_borromean_MeV']
        print(f"│  Q3: Borromean mass at R_p = {borr_mass:.4f} MeV")
        print(f"│      3 × m_p = {3*m_p_MeV:.4f} MeV")
        print(f"│      Ratio to m_p: {borr_mass/m_p_MeV:.6f}")
        print(f"│")

    print(f"│  Q4: Does E_borromean(R) have a minimum or resonance?")
    masses = [r['mass_MeV'] for r in results]
    # Check for non-monotonic behavior in correction factor
    diffs = np.diff(corrections)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    if sign_changes > 0:
        print(f"│      {sign_changes} inflection(s) detected in correction factor")
    else:
        monotone = "decreasing" if corrections[-1] < corrections[0] else "increasing"
        if abs(corrections[-1] - corrections[0]) < 1e-8:
            monotone = "flat"
        print(f"│      Correction factor is {monotone} — no resonance")

    print(f"│")
    print(f"└────────────────────────────────────────────────────────")
