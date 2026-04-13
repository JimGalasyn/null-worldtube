#!/usr/bin/env python3
"""
Phase A: Gauge group from vortex crossing phases.

The strand-exchange operator at a torus knot crossing is physically the
Aharonov-Bohm phase induced when one vortex strand passes through the
gauge field of another.

For a BPS vortex with flux Φ = 2π/e, the Aharonov-Bohm phase when a
test charge passes at distance d from the core is:
    θ_AB = e ∮ A·dl = 2π(1 − Q(d/ξ))
where Q(ρ) is the BPS gauge profile function satisfying the Bogomolny
equation, and ξ = 1/(ev) is the core radius.

For a (p,q) torus knot on a torus of aspect ratio κ = R/r, the strand
separation at each crossing is determined by the knot geometry.

This script:
  1. Parametrizes torus knot crossings and computes strand separations
  2. Solves the BPS vortex profile for Q(ρ) numerically
  3. Computes the Aharonov-Bohm phase at each crossing
  4. Identifies the phase with the gauge coupling
"""

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Torus knot crossing geometry
# ══════════════════════════════════════════════════════════════════════

def torus_knot_crossings(p, q, kappa):
    """Find the strand separations at crossings of a T(p,q) torus knot.

    A T(p,q) torus knot on a torus with major radius R, minor radius r = R/κ
    is parametrized by:
        θ(t) = qt   (poloidal angle)
        φ(t) = pt   (toroidal angle)
    for t ∈ [0, 2π).

    Crossings in the toroidal projection occur when two points t₁, t₂
    have the same toroidal angle: pt₁ ≡ pt₂ (mod 2π), i.e., t₂ = t₁ + 2π/p.

    At such a crossing, the poloidal angle difference is:
        Δθ = q × 2π/p

    The 3D strand separation (chord distance across the torus tube) is:
        d = 2r × |sin(Δθ/2)| = 2R/κ × |sin(πq/p)|

    Returns list of (t1, t2, d/ξ) for each crossing,
    where d/ξ = d × v = 2β/κ × |sin(πq/p)| with β = R/ξ.
    """
    # Number of crossings for T(p,q): |p-1|×|q-1| for a standard knot
    # but for the toroidal projection of T(p,q), there are p(q-1) crossings
    # Actually, the standard crossing number for T(p,q) with p<q is:
    # c = min(p(q-1), q(p-1))
    # For T(2,3): min(2×2, 3×1) = min(4,3) = 3

    delta_theta = 2 * np.pi * q / p  # poloidal separation at crossing
    d_over_r = 2 * abs(np.sin(delta_theta / 2))  # strand separation / r

    # Number of distinct crossings
    n_crossings = min(p * (q - 1), q * (p - 1))

    return {
        'n_crossings': n_crossings,
        'delta_theta': delta_theta,
        'd_over_r': d_over_r,
        'd_over_xi': lambda beta: d_over_r * beta / kappa,
        'description': f'T({p},{q}) on torus κ={kappa:.4f}'
    }


# ══════════════════════════════════════════════════════════════════════
# STEP 2: BPS vortex profile
# ══════════════════════════════════════════════════════════════════════

def solve_bps_profile(rho_max=15.0, n_points=500):
    """Solve the BPS Bogomolny equations for a single n=1 vortex.

    Correct BPS equations (self-dual, positive flux):
        X'(ρ) = Q(ρ) X(ρ) / ρ          [Higgs profile]
        Q'(ρ) = -ρ (1 - X(ρ)²) / 2     [gauge profile]

    where X = |ψ|/v (Higgs), Q = unscreened flux fraction,
    ρ = r/ξ with ξ = 1/(ev) (BPS core radius).

    Boundary conditions:
        X(0) = 0,  Q(0) = 1   (vortex core: zero Higgs, full flux)
        X(∞) = 1,  Q(∞) = 0   (vacuum: full Higgs, screened flux)

    The AB phase at distance ρ from the core: θ_AB = 2π(1 - Q(ρ)).
    """
    rho = np.linspace(1e-6, rho_max, n_points)

    def odes(rho, y):
        X, Q = y
        X = np.maximum(X, 1e-30)
        rho_safe = np.maximum(rho, 1e-30)
        dXdr = Q * X / rho_safe
        dQdr = -rho_safe * (1 - X**2) / 2
        return np.vstack([dXdr, dQdr])

    def bcs(ya, yb):
        # 2 equations → 2 BCs: X(0) ≈ 0 and Q(∞) ≈ 0
        return np.array([ya[0] - 1e-6,   # X(0) ≈ 0
                         yb[1] - 0.0])     # Q(∞) = 0

    # Initial guess matching known BPS asymptotic behavior
    f_guess = np.tanh(rho / np.sqrt(2))
    q_guess = 1.0 / (1 + rho**2)  # monotone decrease from 1 to 0
    y_init = np.vstack([f_guess, q_guess])

    sol = solve_bvp(odes, bcs, rho, y_init, tol=1e-10, max_nodes=10000)

    if not sol.success:
        print(f"  BPS solver warning: {sol.message}")
    else:
        print(f"  BPS solver converged, {len(sol.x)} nodes")

    rho_sol = sol.x
    X_sol = sol.y[0]
    Q_sol = sol.y[1]

    # Verify boundary conditions
    print(f"  X(0) = {X_sol[0]:.6e} (target: 0)")
    print(f"  Q(0) = {Q_sol[0]:.6f} (target: 1)")
    print(f"  X(∞) = {X_sol[-1]:.6f} (target: 1)")
    print(f"  Q(∞) = {Q_sol[-1]:.6f} (target: 0)")

    # Build interpolator for Q(ρ)
    Q_interp = interp1d(rho_sol, Q_sol, kind='cubic',
                         bounds_error=False, fill_value=(1.0, 0.0))

    return rho_sol, X_sol, Q_sol, Q_interp


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Aharonov-Bohm phase at crossings
# ══════════════════════════════════════════════════════════════════════

def aharonov_bohm_phase(Q_func, d_over_xi):
    """Compute the AB phase when a strand passes at distance d from a vortex core.

    θ_AB = 2π(1 − Q(d/ξ))

    where Q(ρ) is the gauge profile.
    At d >> ξ: Q → 0, θ → 2π (full flux enclosed)
    At d = 0: Q → 1, θ → 0 (no flux enclosed)
    """
    Q_val = float(Q_func(d_over_xi))
    theta = 2 * np.pi * (1 - Q_val)
    return theta, Q_val


# ══════════════════════════════════════════════════════════════════════
# MAIN COMPUTATION
# ══════════════════════════════════════════════════════════════════════

KAPPA = np.pi**2
BETA_E = np.sqrt(5) / 2  # electron β
ALPHA_EXP = 1 / 137.036

print("=" * 72)
print("PHASE A: GAUGE GROUP FROM VORTEX CROSSING PHASES")
print("=" * 72)
print()

# ── Step 1: Crossing geometry ──
print("─── Step 1: Torus Knot Crossing Geometry ───")
print()

knots = [
    ('Trefoil T(2,3)', 2, 3, 'SU(3) — 3 crossings → 8 Gell-Mann generators'),
    ('Hopf link T(2,2)', 2, 2, 'SU(2) — 2 crossings → 3 Pauli generators'),
]

for name, p, q, gauge in knots:
    info = torus_knot_crossings(p, q, KAPPA)
    d_xi = info['d_over_xi'](BETA_E)
    print(f"  {name}")
    print(f"    Crossings: {info['n_crossings']}")
    print(f"    Poloidal separation: Δθ = 2πq/p = {np.degrees(info['delta_theta']):.1f}°")
    print(f"    Strand separation: d/r = {info['d_over_r']:.4f}")
    print(f"    At electron β = √5/2, κ = π²:")
    print(f"      d/ξ = 2β sin(πq/p)/κ = {d_xi:.6f}")
    print(f"    Gauge group: {gauge}")
    print()

# ── Step 2: BPS profile ──
print("─── Step 2: BPS Vortex Profile ───")
print()
rho, f_prof, a_prof, Q = solve_bps_profile()
print(f"  Solved BPS Bogomolny equations on ρ ∈ [0, 20]")
print(f"  Q(0) = {Q(0):.6f} (should be 1)")
print(f"  Q(∞) = {Q(20):.6f} (should be 0)")
print(f"  Q(1) = {Q(1):.6f} (at one core radius)")
print()

# ── Step 3: AB phases at crossings ──
print("─── Step 3: Aharonov-Bohm Phases at Crossings ───")
print()

for name, p, q, gauge in knots:
    info = torus_knot_crossings(p, q, KAPPA)
    d_xi = info['d_over_xi'](BETA_E)
    theta, Q_val = aharonov_bohm_phase(Q, d_xi)

    print(f"  {name}")
    print(f"    d/ξ = {d_xi:.6f}")
    print(f"    Q(d/ξ) = {Q_val:.6f}")
    print(f"    θ_AB = 2π(1−Q) = {theta:.6f} rad = {np.degrees(theta):.4f}°")
    print(f"    θ_AB / 2π = {theta/(2*np.pi):.6f}")
    print()

    # The gauge coupling is related to θ_AB
    # In the R-matrix: R = exp(i θ_AB × generator)
    # The coupling g² ~ θ_AB / (normalization)
    # For SU(N): α_N = g²/(4π) = θ_AB/(4π × normalization)

    # Compare to known couplings:
    coupling = theta / (2 * np.pi)
    print(f"    Coupling = θ/(2π) = {coupling:.6f}")
    print(f"    α_exp = 1/137 = {ALPHA_EXP:.6f}")
    print(f"    α_GUT ≈ 1/40 = {1/40:.6f}")
    print(f"    coupling / α = {coupling / ALPHA_EXP:.2f}")
    print()

# ── Step 4: Scan over β and κ ──
print("─── Step 4: How Crossing Phase Depends on β and κ ───")
print()
print(f"  For the trefoil T(2,3):")
print(f"  {'β':>8} {'κ':>8} {'d/ξ':>10} {'Q(d/ξ)':>10} {'θ_AB/(2π)':>12} {'≈ α?':>10}")

info_trefoil = torus_knot_crossings(2, 3, KAPPA)
for beta in [0.5, BETA_E, 1.5, 2.0, 3.0, 5.0, 8.944]:
    d_xi = info_trefoil['d_over_xi'](beta)
    theta, Q_val = aharonov_bohm_phase(Q, d_xi)
    coupling = theta / (2 * np.pi)
    print(f"  {beta:>8.4f} {KAPPA:>8.4f} {d_xi:>10.6f} {Q_val:>10.6f} "
          f"{coupling:>12.6f} {1/coupling if coupling > 0 else float('inf'):>10.1f}")

print()

# ── Scan over κ at fixed β = β_e ──
print(f"  Scan over κ at β = √5/2:")
print(f"  {'κ':>8} {'d/ξ':>10} {'Q(d/ξ)':>10} {'θ_AB/(2π)':>12} {'1/coupling':>10}")
for kappa in [1, 2, 3, np.pi, 5, np.pi**2, 15, 20, 50, 100]:
    info_k = torus_knot_crossings(2, 3, kappa)
    d_xi = info_k['d_over_xi'](BETA_E)
    theta, Q_val = aharonov_bohm_phase(Q, d_xi)
    coupling = theta / (2 * np.pi)
    inv = 1/coupling if coupling > 1e-10 else float('inf')
    print(f"  {kappa:>8.4f} {d_xi:>10.6f} {Q_val:>10.6f} "
          f"{coupling:>12.6f} {inv:>10.1f}")

print()

# ── KEY QUESTION: is there a κ where θ_AB/(2π) = α? ──
print("─── Step 5: Does θ_AB/(2π) = α at some κ? ───")
print()
from scipy.optimize import brentq

def coupling_minus_alpha(kappa_val):
    info_k = torus_knot_crossings(2, 3, kappa_val)
    d_xi = info_k['d_over_xi'](BETA_E)
    theta, _ = aharonov_bohm_phase(Q, d_xi)
    return theta / (2 * np.pi) - ALPHA_EXP

# Check if there's a crossing
c_low = coupling_minus_alpha(1.0)
c_high = coupling_minus_alpha(100.0)
print(f"  θ/(2π) − α at κ=1: {c_low:+.6f}")
print(f"  θ/(2π) − α at κ=100: {c_high:+.6f}")

if c_low * c_high < 0:
    kappa_alpha = brentq(coupling_minus_alpha, 1.0, 100.0)
    print(f"  θ/(2π) = α at κ* = {kappa_alpha:.6f}")
    print(f"  π² = {np.pi**2:.6f}")
    print(f"  κ*/π² = {kappa_alpha/np.pi**2:.6f} ({(kappa_alpha/np.pi**2-1)*100:+.4f}%)")
else:
    print(f"  No crossing in [1, 100] — coupling is {'always above' if c_low > 0 else 'always below'} α")
    # Find where coupling = α_GUT ≈ 1/40
    try:
        kappa_gut = brentq(lambda k: coupling_minus_alpha(k) + ALPHA_EXP - 1/40, 1.0, 100.0)
        print(f"  θ/(2π) = 1/40 (α_GUT) at κ = {kappa_gut:.4f}")
    except:
        pass

print()
print("=" * 72)
print("PHASE A COMPLETE")
print("=" * 72)
