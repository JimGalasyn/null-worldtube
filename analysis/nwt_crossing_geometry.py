#!/usr/bin/env python3
"""
NWT Crossing-Point Geometry: Exact α from R1 matrix element at crossings

The flat formula α = 1/(√2 κ²) gives 1/137.76 (0.52% from PDG 1/137.04).
Session 3 identified that the curvature correction to ⟨cos²θ⟩ overshoots
by 22% — meaning the physical coupling uses crossing-point values, not
the angle average.

This script computes:
1. Exact (s, θ) coordinates of self-crossings for torus knots
2. The R1 matrix element evaluated at those specific points
3. Whether the crossing geometry closes the 0.52% gap

For a torus knot T(p,q) on torus (R, r) with κ = R/r:
  Centerline: γ(t) = ((R + r cos(qt))cos(pt), (R + r cos(qt))sin(pt), r sin(qt))
  Crossings: where γ(t₁) projected ≈ γ(t₂) projected, t₁ ≠ t₂
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad

KAPPA = np.pi**2
ALPHA_PDG = 1/137.036


def torus_knot_point(t, p, q, R, r):
    """3D point on torus knot T(p,q)."""
    phi = p * t
    theta = q * t
    rho = R + r * np.cos(theta)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    z = r * np.sin(theta)
    return np.array([x, y, z])


def find_crossings_projection(p, q, R, r, n_search=10000):
    """Find self-crossings of T(p,q) in the xy-projection.

    Crossings occur where two different parameter values t₁, t₂
    give the same (x,y) but different z.
    """
    # Sample the knot densely
    t_vals = np.linspace(0, 2*np.pi, n_search, endpoint=False)
    pts = np.array([torus_knot_point(t, p, q, R, r) for t in t_vals])

    crossings = []

    # For each pair of segments, check if they cross in xy-projection
    # Use the winding: crossings of T(p,q) are at specific parameter values
    # For T(2,3): n_crossings = min(2×2, 3×1) = min(4,3) = 3
    n_expected = min(p*(q-1), q*(p-1)) if p > 1 and q > 1 else 0

    if n_expected == 0:
        return []

    # Analytical approach for torus knots:
    # Crossings occur where φ(t₁) = φ(t₂) mod 2π AND
    # the xy positions match (same rho × cos/sin(phi))
    # This means pt₁ = pt₂ + 2πk for some integer k, AND
    # R + r cos(qt₁) ≈ R + r cos(qt₂) OR the angles conspire

    # For T(p,q), crossings in the standard projection are at:
    # t₁ = 2π(m + n/q)/p, t₂ = 2π(m - n/q)/p for integers m, n
    # with n ∈ {1, ..., q-1} (gives the q-1 distinct crossing types per period)

    # More precisely, solve: (R+r cos(qt₁))e^{ipt₁} = (R+r cos(qt₂))e^{ipt₂}
    # This requires both rho and phi to match.

    # Numerical approach: find close encounters in xy
    xy = pts[:, :2]
    for i in range(n_search):
        for j in range(i + n_search//4, min(i + 3*n_search//4, n_search)):
            ji = j % n_search
            dx = xy[i, 0] - xy[ji, 0]
            dy = xy[i, 1] - xy[ji, 1]
            dist_xy = np.sqrt(dx**2 + dy**2)
            if dist_xy < 0.5 * r:  # close in projection
                dz = abs(pts[i, 2] - pts[ji, 2])
                if dz > 0.1 * r:  # different z → actual crossing
                    # Refine with fsolve
                    t1_approx = t_vals[i]
                    t2_approx = t_vals[ji]

                    def residual(params):
                        t1, t2 = params
                        p1 = torus_knot_point(t1, p, q, R, r)
                        p2 = torus_knot_point(t2, p, q, R, r)
                        return [p1[0] - p2[0], p1[1] - p2[1]]

                    try:
                        sol = fsolve(residual, [t1_approx, t2_approx],
                                    full_output=True)
                        t1_sol, t2_sol = sol[0]
                        if abs(t1_sol - t2_sol) > 0.01:  # distinct
                            # Check if this is a new crossing
                            is_new = True
                            for tc in crossings:
                                if (abs(t1_sol - tc['t1']) < 0.05 or
                                    abs(t1_sol - tc['t2']) < 0.05):
                                    is_new = False
                                    break
                            if is_new:
                                pt1 = torus_knot_point(t1_sol, p, q, R, r)
                                pt2 = torus_knot_point(t2_sol, p, q, R, r)
                                crossings.append({
                                    't1': t1_sol % (2*np.pi),
                                    't2': t2_sol % (2*np.pi),
                                    'pos': pt1,
                                    'z_gap': pt1[2] - pt2[2],
                                    'theta1': (q * t1_sol) % (2*np.pi),
                                    'theta2': (q * t2_sol) % (2*np.pi),
                                })
                    except:
                        pass

        if len(crossings) >= n_expected:
            break

    return crossings[:n_expected]


def crossing_geometry_analysis(p, q, kappa, beta=np.sqrt(5/4)):
    """Full analysis of crossing geometry for T(p,q) on torus with κ."""
    R = beta  # in units of ξ
    r = R / kappa

    print(f"\n{'━'*90}")
    print(f"  T({p},{q}) on torus κ={kappa:.4f}, R={R:.6f}ξ, r={r:.6f}ξ")
    print(f"{'━'*90}")

    crossings = find_crossings_projection(p, q, R, r)
    n_cross = len(crossings)

    if n_cross == 0:
        print(f"  No crossings found (unknotted or T(p,1)/T(1,q))")
        # For unknotted carriers, R1 acts uniformly along the path
        # The effective coupling uses the PATH-AVERAGED cos(θ)

        # For mode (2,1) on unknot (1,1): the mode has phase Ψ = 2φ + θ
        # The poloidal angle θ varies along the mode path
        # cos(θ) averaged over the mode path:
        print(f"\n  Path-averaged cos(θ) for mode ({p},{q}):")

        def cos_theta_along_path(t):
            theta = q * t
            return np.cos(theta)

        def cos2_theta_along_path(t):
            theta = q * t
            return np.cos(theta)**2

        avg_cos, _ = quad(cos_theta_along_path, 0, 2*np.pi)
        avg_cos /= (2*np.pi)
        avg_cos2, _ = quad(cos2_theta_along_path, 0, 2*np.pi)
        avg_cos2 /= (2*np.pi)

        print(f"    ⟨cos(θ)⟩ = {avg_cos:.6f}")
        print(f"    ⟨cos²(θ)⟩ = {avg_cos2:.6f}")
        print(f"    For flat torus: ⟨cos²(θ)⟩ = 1/2 = 0.500000")

        # Curvature-corrected average
        # On a torus, the metric factor h_φ = R + r cos(θ) weights the inner/outer
        # differently. The proper average is ∫cos²θ × h_φ dθ / ∫h_φ dθ
        def weighted_cos2(t):
            theta = q * t
            h = kappa + np.cos(theta)  # = (R + r cos θ)/r
            return np.cos(theta)**2 * h

        def weight(t):
            theta = q * t
            return kappa + np.cos(theta)

        num, _ = quad(weighted_cos2, 0, 2*np.pi)
        den, _ = quad(weight, 0, 2*np.pi)
        avg_cos2_curved = num / den

        print(f"    ⟨cos²(θ)⟩_curved = {avg_cos2_curved:.6f} (metric-weighted)")

        # Now compute α from these
        alpha_flat = 1.0 / (2.0 * kappa**2)  # using ⟨cos²⟩=1/2
        alpha_curved = avg_cos2_curved / kappa**2
        alpha_sqrt2 = 1.0 / (np.sqrt(2) * kappa**2)  # our formula

        print(f"\n  α estimates:")
        print(f"    Flat:    α = 1/(2κ²)        = {alpha_flat:.8f}  1/α = {1/alpha_flat:.4f}")
        print(f"    √2:     α = 1/(√2 κ²)      = {alpha_sqrt2:.8f}  1/α = {1/alpha_sqrt2:.4f}")
        print(f"    Curved: α = ⟨cos²⟩_c / κ²  = {alpha_curved:.8f}  1/α = {1/alpha_curved:.4f}")
        print(f"    PDG:    α                   = {ALPHA_PDG:.8f}  1/α = {1/ALPHA_PDG:.4f}")
        print(f"\n    Flat error:   {abs(alpha_flat - ALPHA_PDG)/ALPHA_PDG * 100:.4f}%")
        print(f"    √2 error:    {abs(alpha_sqrt2 - ALPHA_PDG)/ALPHA_PDG * 100:.4f}%")
        print(f"    Curved error: {abs(alpha_curved - ALPHA_PDG)/ALPHA_PDG * 100:.4f}%")

        return avg_cos2_curved

    print(f"  Found {n_cross} crossings")

    for i, cr in enumerate(crossings):
        theta1 = cr['theta1']
        theta2 = cr['theta2']
        print(f"\n  Crossing {i+1}:")
        print(f"    t₁ = {cr['t1']:.6f}, t₂ = {cr['t2']:.6f}")
        print(f"    θ₁ = {theta1:.6f} ({np.degrees(theta1):.2f}°)")
        print(f"    θ₂ = {theta2:.6f} ({np.degrees(theta2):.2f}°)")
        print(f"    z-gap = {cr['z_gap']:.6f}ξ")
        print(f"    cos(θ₁) = {np.cos(theta1):.6f}")
        print(f"    cos(θ₂) = {np.cos(theta2):.6f}")
        print(f"    Position: ({cr['pos'][0]:.4f}, {cr['pos'][1]:.4f}, {cr['pos'][2]:.4f})")

    # Average cos²(θ) at crossings
    cos2_at_crossings = np.mean([
        0.5 * (np.cos(cr['theta1'])**2 + np.cos(cr['theta2'])**2)
        for cr in crossings
    ])
    print(f"\n  ⟨cos²(θ)⟩ at crossings = {cos2_at_crossings:.6f}")
    print(f"  Compare: flat = 0.500000, curved avg = (computed above)")

    # α from crossing values
    alpha_crossing = cos2_at_crossings / kappa**2
    print(f"\n  α from crossing geometry:")
    print(f"    α = ⟨cos²θ⟩_cross / κ² = {alpha_crossing:.8f}")
    print(f"    1/α = {1/alpha_crossing:.4f}")
    print(f"    Error vs PDG: {abs(alpha_crossing - ALPHA_PDG)/ALPHA_PDG * 100:.4f}%")

    return cos2_at_crossings


# ══════════════════════════════════════════════════════════════════════
# MAIN: Analyze crossing geometry for all relevant carriers
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 90)
    print("CROSSING-POINT GEOMETRY: Exact α from R1 matrix element at crossings")
    print("=" * 90)

    print(f"\n  PDG: α = {ALPHA_PDG:.8f}, 1/α = {1/ALPHA_PDG:.4f}")
    print(f"  Flat formula: α = 1/(√2 π⁴) = {1/(np.sqrt(2)*KAPPA**2):.8f}, "
          f"1/α = {np.sqrt(2)*KAPPA**2:.4f}")
    print(f"  Gap: {abs(1/(np.sqrt(2)*KAPPA**2) - ALPHA_PDG)/ALPHA_PDG * 100:.4f}%")

    # ── 1. Mode (2,1) on unknot — the electron ──────────────────────
    print("\n\n" + "=" * 90)
    print("1. ELECTRON: Mode (2,1) on unknot carrier (1,1)")
    print("   No carrier crossings. R1 is a self-interaction.")
    print("=" * 90)
    crossing_geometry_analysis(2, 1, KAPPA)

    # ── 2. What if we use the EFFECTIVE torus knot? ──────────────────
    print("\n\n" + "=" * 90)
    print("2. EFFECTIVE (2,1) torus knot in standard projection")
    print("   The (2,1) torus knot has crossing number 0 — it's unknotted!")
    print("   But the MODE wraps twice toroidally → it samples different θ")
    print("=" * 90)

    # For the (2,1) mode, the key is that the mode samples θ = t
    # as it goes around the torus (since q=1, θ = t).
    # The R1 interaction involves cos(θ) where θ is the POLOIDAL
    # angle at the point where the writhe change happens.

    # The writhe-change probability depends on WHERE on the tube
    # the perturbation occurs. The mode has maximal amplitude
    # at the outer equator (θ=0, where h_φ is largest).

    # Compute the mode-weighted cos²(θ):
    print("\n  Mode-amplitude-weighted ⟨cos²(θ)⟩:")
    print("  The mode amplitude |ψ(θ)|² ∝ 1/(R+r cos θ) on the tube")
    print("  (inverse of the metric factor — denser where tube is thinner)")

    kappa = KAPPA

    # Mode density: for a (p,q) mode on a torus, the energy density
    # goes as 1/h_φ² = 1/(R + r cos θ)² (concentrated at inner equator)
    # But the COUPLING to radiation goes as h_φ (outer equator radiates more)
    # The product: coupling × density ∝ 1/h_φ

    # Several physically motivated weightings:
    weightings = {
        'uniform': lambda th: 1.0,
        'mode_density (1/h²)': lambda th: 1.0 / (kappa + np.cos(th))**2,
        'radiation (h)': lambda th: kappa + np.cos(th),
        'coupling×density (1/h)': lambda th: 1.0 / (kappa + np.cos(th)),
        'metric (h)': lambda th: kappa + np.cos(th),
    }

    for name, weight_fn in weightings.items():
        def integrand_num(th):
            return np.cos(th)**2 * weight_fn(th)
        def integrand_den(th):
            return weight_fn(th)

        num, _ = quad(integrand_num, 0, 2*np.pi)
        den, _ = quad(integrand_den, 0, 2*np.pi)
        cos2_w = num / den

        alpha_w = cos2_w / kappa**2
        err = abs(alpha_w - ALPHA_PDG) / ALPHA_PDG * 100

        print(f"    {name:30s}: ⟨cos²θ⟩ = {cos2_w:.8f}, "
              f"α = {alpha_w:.8f}, 1/α = {1/alpha_w:.4f}, err = {err:.4f}%")

    # ── 3. The √2 interpretation ─────────────────────────────────────
    print("\n\n" + "=" * 90)
    print("3. THE 1/√2 QUESTION: What weighting gives ⟨cos²θ⟩ = 1/√2?")
    print("=" * 90)

    # If α = 1/(√2 κ²) then ⟨cos²θ⟩_effective = 1/√2 = 0.70711
    # This is LARGER than any physical weighting gives (all give 0.5-0.6)
    # So the √2 does NOT come from an angular average.

    # The √2 comes from the DEGENERACY factor (2 modes) and
    # the per-channel normalization, as we derived in the eigensolver.

    print(f"\n  Target: ⟨cos²θ⟩_eff = 1/√2 = {1/np.sqrt(2):.8f}")
    print(f"  All angular averages give ⟨cos²θ⟩ ≈ 0.5 (flat) to 0.55 (curved)")
    print(f"  The √2 factor is NOT an angular average.")
    print(f"\n  It comes from the EIGENSOLVER decomposition:")
    print(f"    g²(R1) = (1/κ)² × overlap = (1/κ)² × 2.00")
    print(f"    The factor 2.00 = sum over Δn_θ=+1 AND Δn_θ=-1")
    print(f"    Per channel: g² = (1/κ)² × 1.00")
    print(f"    α = g²/√2 (amplitude vs intensity) = 1/(√2 κ²)")

    # ── 4. The 0.52% gap: κ vs π² ────────────────────────────────────
    print("\n\n" + "=" * 90)
    print("4. THE 0.52% GAP: κ = π² is approximate")
    print("=" * 90)

    kappa_exact = 1.0 / np.sqrt(np.sqrt(2) * ALPHA_PDG)
    alpha_at_pi2 = 1.0 / (np.sqrt(2) * np.pi**4)

    print(f"  If α = 1/(√2 κ²) exactly, then:")
    print(f"    κ_exact = 1/√(√2 α) = {kappa_exact:.8f}")
    print(f"    π²      =             {np.pi**2:.8f}")
    print(f"    Difference: {abs(kappa_exact - np.pi**2)/np.pi**2 * 100:.4f}%")
    print(f"    This 0.26% in κ → 0.52% in α (doubled by κ² dependence)")

    print(f"\n  Conversely, α at κ = π² exactly:")
    print(f"    α(π²) = {alpha_at_pi2:.8f}")
    print(f"    α_PDG = {ALPHA_PDG:.8f}")
    print(f"    Ratio: {alpha_at_pi2/ALPHA_PDG:.8f}")

    # ── 5. Can crossing geometry give the exact correction? ──────────
    print("\n\n" + "=" * 90)
    print("5. CROSSING GEOMETRY CORRECTION TO κ")
    print("=" * 90)

    # The trefoil T(2,3) has 3 crossings. Let's find them.
    print("\n  Trefoil T(2,3) crossings:")
    beta = np.sqrt(5/4)
    crossing_geometry_analysis(2, 3, KAPPA, beta=beta)

    # Also try T(3,2)
    print("\n  Trefoil T(3,2) crossings:")
    crossing_geometry_analysis(3, 2, KAPPA, beta=beta)

    # ── 6. Physical κ from phase closure at crossings ─────────────────
    print("\n\n" + "=" * 90)
    print("6. DOES THE CROSSING GEOMETRY SHIFT κ FROM π² TO κ_exact?")
    print("=" * 90)

    print(f"""
  If the physical κ is determined by phase closure INCLUDING the
  curvature correction at crossing points, then:

    κ_phys = π² × (1 + δ_crossing)

  where δ_crossing accounts for the modified effective path length
  at the crossings (where two tube strands are close and the
  condensate density is depleted between them).

  Required shift: κ_exact - π² = {kappa_exact - np.pi**2:.6f}
  Fractional: δ = {(kappa_exact - np.pi**2)/np.pi**2:.6f}

  This is a {abs(kappa_exact - np.pi**2)/np.pi**2 * 100:.4f}% correction — tiny.
  But it would make α EXACT.

  The physical mechanism: at a crossing, the two tube strands
  approach within d ~ 2r. The condensate density between them
  is depleted (healing). This effectively REDUCES the tube
  radius at the crossing by a factor ~ (1 - r/d) = (1 - 1/2).
  A 50% reduction at the crossing, over the ~r/L fraction of
  the path that's affected, gives:

    δ ≈ -(r/L) × 0.5 × (number of crossings)

  For the (2,1) unknot: 0 crossings → δ = 0 → κ = π² (baseline)
  For trefoil T(2,3): 3 crossings → δ = -3 × 0.5 × r/L

  With r/L ≈ {beta/KAPPA / (2*np.pi*beta*np.sqrt(4+1/KAPPA**2)):.6f}:
""")

    L_21 = 2*np.pi*beta*np.sqrt(4 + 1/KAPPA**2)
    r_val = beta/KAPPA
    r_over_L = r_val / L_21
    delta_est = -3 * 0.5 * r_over_L
    print(f"    r/L = {r_over_L:.6f}")
    print(f"    δ_est = -3 × 0.5 × r/L = {delta_est:.6f}")
    print(f"    δ_needed = {(kappa_exact - np.pi**2)/np.pi**2:.6f}")
    print(f"    Ratio: {delta_est / ((kappa_exact - np.pi**2)/np.pi**2):.4f}")


if __name__ == '__main__':
    main()
