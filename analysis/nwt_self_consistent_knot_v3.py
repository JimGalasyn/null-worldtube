"""
Self-consistent (2,1) electron knot — Version 3: The exact solution.

The phase around a (2,1) knot in the superfluid vacuum is:

  Φ/(2π) = 2√(β² + 1)      [exact in large-κ limit]

where β = R/ξ (orbit radius in units of healing length).

Self-consistency (Φ = 2πm, integer m) gives:

  β = √(m²/4 - 1)

  m=3: β = √(5/4) = √5/2 ≈ 1.1180
       R = (√5/2) × ξ = (√5/2) × ƛ_C
       n(R) = 3/√5 ≈ 1.3416

This is the FIRST RESONANCE (m=1,2 don't give physical β).
The electron torus sits at R = 1.118 × ƛ_C, slightly outside
the healing length, with 3 wavelengths fitting around the knot path.

The derivation chain:
  topology (2,1) + vacuum profile n(r) = √(1+ξ²/r²) → R/ξ = √5/2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patches as mpatches

# --- Constants ---
c = 2.998e8
hbar = 1.055e-34
m_e = 9.109e-31
e_charge = 1.602e-19
alpha = 1 / 137.036
eps0 = 8.854e-12

lam_C = hbar / (m_e * c)
r_e = alpha * lam_C
xi = lam_C
fm = 1e-15

kappa_santos = np.sqrt(1 / (4 * np.pi * alpha))
kappa_macken = np.pi**2

print("=" * 70)
print("SELF-CONSISTENT (2,1) ELECTRON: THE EXACT SOLUTION")
print("=" * 70)


# =====================================================================
# EXACT FORMULA DERIVATION
# =====================================================================

print("""
DERIVATION:
  The (p,q) knot on a torus with R >> r wraps p times toroidally.
  Path length per period: L = 2πR × p (leading order in 1/κ).
  Refractive index at R: n(R) = √(1 + ξ²/R²) = √(1 + 1/β²).

  Phase per period:
    Φ = k₀ × n(R) × L = (1/ξ) × √(1 + 1/β²) × 2πRp
      = 2πp × (R/ξ) × √(1 + ξ²/R²)
      = 2πp × √(β² + 1)

  Resonance: Φ = 2πm
    p√(β² + 1) = m
    β = √(m²/p² - 1)

  For (2,1) electron (p=2):
    β = √(m²/4 - 1)

    m=1: β² = -3/4     → no solution
    m=2: β² = 0        → R = 0 (degenerate)
    m=3: β² = 5/4      → β = √5/2 = 1.1180...   ★ FIRST PHYSICAL RESONANCE
    m=4: β² = 3        → β = √3 = 1.7321...
    m=5: β² = 21/4     → β = √21/2 = 2.2913...
""")

beta_exact = np.sqrt(5) / 2
R_exact = beta_exact * xi
n_exact = np.sqrt(1 + 1/beta_exact**2)
r_santos = R_exact / kappa_santos
r_macken = R_exact / kappa_macken

print(f"THE m = 3 SOLUTION:")
print(f"  β = R/ξ = √5/2 = {beta_exact:.10f}")
print(f"  R = {R_exact/fm:.2f} fm = {beta_exact:.4f} × ƛ_C")
print(f"  n(R) = √(1 + 4/5) = √(9/5) = 3/√5 = {n_exact:.10f}")
print(f"  Santos: r = {r_santos/fm:.2f} fm")
print(f"  Macken: r = {r_macken/fm:.2f} fm")

# Verify the formula
print(f"\n  Verification: 2√(β² + 1) = 2√(5/4 + 1) = 2 × 3/2 = {2*np.sqrt(beta_exact**2 + 1):.10f}")


# =====================================================================
# FULL INTEGRAL VERIFICATION (finite κ corrections)
# =====================================================================

print("\n" + "=" * 70)
print("FINITE-κ CORRECTIONS")
print("=" * 70)

def full_phase(beta, kappa, p, q):
    """Exact phase/(2π) including finite-κ corrections."""
    def integrand(t):
        cos_qt = np.cos(q * t)
        rho_sq = 1.0 + 1.0/kappa**2 + 2.0*cos_qt/kappa
        rho_sq = max(rho_sq, 1e-12)
        n = np.sqrt(1.0 + 1.0/(beta**2 * rho_sq))
        ds = np.sqrt(p**2 * (1.0 + cos_qt/kappa)**2 + q**2/kappa**2)
        return n * ds

    result, _ = quad(integrand, 0, 2*np.pi, limit=500, epsrel=1e-12)
    return beta * result / (2 * np.pi)


# Find exact β for m=3 at each κ
print(f"\nExact β(κ) for m=3 resonance:")
print(f"{'κ':>10} {'β_exact':>12} {'R (fm)':>10} {'n(R)':>10} {'β-√5/2':>12} {'dev%':>8}")
print("-" * 65)

eta_scan = np.linspace(0.5, 2.0, 2000)

for kappa in [2.0, 2.5, 3.0, kappa_santos, 5.0, 7.0, kappa_macken, 15, 20, 50, 100, 1000]:
    phase_scan = np.array([full_phase(e, kappa, 2, 1) for e in eta_scan])
    for i in range(len(eta_scan) - 1):
        if (phase_scan[i] - 3.0) * (phase_scan[i+1] - 3.0) < 0:
            beta_sol = brentq(
                lambda b: full_phase(b, kappa, 2, 1) - 3.0,
                eta_scan[i], eta_scan[i+1], xtol=1e-14)
            R_sol = beta_sol * xi
            n_sol = np.sqrt(1 + 1/beta_sol**2)
            delta = beta_sol - beta_exact
            dev_pct = delta / beta_exact * 100
            kstr = f"{kappa:.2f}"
            if abs(kappa - kappa_santos) < 0.01:
                kstr = "Santos"
            elif abs(kappa - kappa_macken) < 0.01:
                kstr = "π²"
            print(f"{kstr:>10} {beta_sol:>12.8f} {R_sol/fm:>10.2f} "
                  f"{n_sol:>10.6f} {delta:>+12.8f} {dev_pct:>+8.4f}")
            break


# =====================================================================
# ALL (p,q) KNOTS — FIRST RESONANCES
# =====================================================================

print("\n" + "=" * 70)
print("ALL (p,q) TOPOLOGIES: FIRST PHYSICAL RESONANCE")
print("=" * 70)

print(f"\nFormula: β = √(m²/p² - 1), first physical m = p+1 (for q=1)")
print(f"\n{'(p,q)':<8} {'m':>3} {'β = R/ξ':>12} {'n(R)':>10} {'R (fm)':>10} {'Identity':>20}")
print("-" * 68)

for p in range(1, 6):
    for q in [1]:
        m_first = p + 1  # first m that gives β² > 0
        # Check: β² = m²/p² - 1 > 0 requires m > p
        beta_val = np.sqrt(m_first**2 / p**2 - 1)
        n_val = np.sqrt(1 + 1/beta_val**2)
        R_val = beta_val * xi

        # Identify β
        ident = ""
        test_vals = [
            ("1", 1.0), ("√2", np.sqrt(2)), ("√3", np.sqrt(3)),
            ("2", 2.0), ("√5", np.sqrt(5)), ("√5/2", np.sqrt(5)/2),
            ("√(5/4)", np.sqrt(5/4)), ("√(8/9)", np.sqrt(8/9)),
            ("√(21)/2", np.sqrt(21)/2), ("√(3)", np.sqrt(3)),
            ("2√2/3", 2*np.sqrt(2)/3), ("√(5)/3", np.sqrt(5)/3),
            ("φ", (1+np.sqrt(5))/2),
        ]
        for name, val in test_vals:
            if abs(beta_val - val) / max(val, 0.01) < 0.001:
                ident = f"β = {name}"
                break

        if not ident:
            # Try to express as √(a/b)
            for a in range(1, 30):
                for b in range(1, 10):
                    if abs(beta_val - np.sqrt(a/b)) < 0.0001:
                        ident = f"β = √({a}/{b})"
                        break
                if ident:
                    break

        print(f"({p},{q})    {m_first:>3} {beta_val:>12.6f} {n_val:>10.6f} "
              f"{R_val/fm:>10.2f} {ident:>20}")

        # Also show m_first+1
        m_next = m_first + 1
        beta_next = np.sqrt(m_next**2 / p**2 - 1)
        n_next = np.sqrt(1 + 1/beta_next**2)
        R_next = beta_next * xi

        ident2 = ""
        for name, val in test_vals:
            if abs(beta_next - val) / max(val, 0.01) < 0.001:
                ident2 = f"β = {name}"
                break
        if not ident2:
            for a in range(1, 50):
                for b in range(1, 20):
                    if abs(beta_next - np.sqrt(a/b)) < 0.0001:
                        ident2 = f"β = √({a}/{b})"
                        break
                if ident2:
                    break

        print(f"         {m_next:>3} {beta_next:>12.6f} {n_next:>10.6f} "
              f"{R_next/fm:>10.2f} {ident2:>20}")


# =====================================================================
# THE PARTICLE FAMILY FROM TOPOLOGY
# =====================================================================

print("\n" + "=" * 70)
print("PARTICLE FAMILY FROM KNOT TOPOLOGY")
print("=" * 70)

print("""
Each (p,q) knot gives a tower of resonances at:
  R_m = ξ × √(m²/p² - 1)    for m = p+1, p+2, ...

The FIRST resonance (m = p+1) is the ground state = the particle.
Higher m = excited states = heavier versions (muon? tau?).

KEY: The particle mass is NOT determined by the resonance condition alone
(the mass cancels out). But the RATIO R/ξ is determined. This means:
  - The electron shape (R/ξ = √5/2) is fixed by topology
  - To fix the mass, we need a second condition (energy balance)
""")

# For the (2,1) electron, the resonance tower:
print("(2,1) ELECTRON RESONANCE TOWER:")
print(f"{'m':>3} {'β = R/ξ':>12} {'n(R)':>10} {'Interpretation':>30}")
print("-" * 58)

for m in range(3, 9):
    beta = np.sqrt(m**2/4 - 1)
    n_r = np.sqrt(1 + 1/beta**2)
    if m == 3:
        interp = "Electron (ground state)"
    elif m == 4:
        interp = "First excitation"
    elif m == 5:
        interp = "Second excitation"
    else:
        interp = f"m={m} excitation"
    print(f"{m:>3} {beta:>12.6f} {n_r:>10.6f} {interp:>30}")


# Could these be the three generations?
print(f"""
IF the three generations are the m=3,4,5 resonances:
  Electron: β = √(5/4) = {np.sqrt(5/4):.6f}
  Muon:     β = √3     = {np.sqrt(3):.6f}
  Tau:      β = √(21/4)= {np.sqrt(21/4):.6f}

  Mass ratios would depend on the second equation (energy balance).
  The mode volume V ∝ R × r² ∝ β/κ² × ξ³ scales with β.
  If mass ∝ 1/R (higher mode = larger orbit = LESS confined = lighter?):
    m_μ/m_e ≈ β_e/β_μ = √(5/4)/√3 = √(5/12) ≈ 0.645
  This gives muon LIGHTER than electron — wrong direction.

  If mass ∝ mode energy ∝ m (mode number):
    m_μ/m_e = 4/3 ≈ 1.33
  Actual ratio: m_μ/m_e = 206.8 — way too big.

  The three generations probably aren't simple overtones of the SAME resonance.
  More likely: they are (2,1) solitons with different NONLINEAR structure
  (higher EH orders, as found in Phase 15).
""")


# =====================================================================
# SECOND EQUATION: WHAT FIXES THE MASS?
# =====================================================================

print("=" * 70)
print("WHAT FIXES THE MASS?")
print("=" * 70)

print("""
Phase closure gives R/ξ = √5/2 but doesn't fix m_e.
The second equation must come from ENERGY BALANCE.

Candidate 1: EM energy = rest mass
  U = ε₀ ∫ E² dV = m_e c²
  For a mode with field E₀ in volume V_eff:
    ε₀ E₀² V_eff = m_e c²

  The field E₀ is set by charge quantization (total charge = e).
  This gives a relationship between m_e, R, r, and α.

Candidate 2: Macken impedance balance
  Already derived: r/R = 1/π² from Z_s = c³/G impedance matching.
  Combined with R/ξ = √5/2:
    r = R/π² = (√5/2) × ξ/π²
    r/ξ = √5/(2π²) = 0.1132

  This fixes both R and r in terms of ξ = ℏ/(m_e c).
  The mass is then determined by the charge condition.

Candidate 3: Classical electron radius
  The classical electron radius r_e = αƛ_C = α × ξ.
  If r_e = r_tube (tube radius = classical radius):
    α = r/R × R/ξ = (1/κ) × √5/2
    κ = √5/(2α) = √5/(2 × 1/137.036) = √5 × 137.036/2 = 153.2

  This gives κ ≈ 153, much larger than Santos (3.3) or Macken (9.87).
  Not obviously wrong — it would mean a very thin tube.

Let's compute the electromagnetic energy for each scenario:
""")

mu0 = 4 * np.pi * 1e-7

# For a (2,1) mode with charge e, the field near the tube surface is:
# E ~ e/(4πε₀r²) (Coulomb field at tube radius)
# B ~ μ₀I/(2πr) where I = e × c/(2πR) (current from charge traveling at c)

for kname, kappa in [("Santos", kappa_santos), ("Macken π²", kappa_macken),
                      ("κ=153", np.sqrt(5)/(2*alpha))]:
    R = beta_exact * xi
    r = R / kappa
    V_torus = 2 * np.pi**2 * R * r**2

    # Electric field at tube surface (Coulomb)
    E_field = e_charge / (4 * np.pi * eps0 * r**2)

    # Energy stored
    # U_E ≈ ε₀/2 × E² × V_eff
    # V_eff is not the whole torus — the field is concentrated near the surface
    # For a TM mode in a tube: V_eff ~ πr²L × (field filling factor)
    # Approximate: V_eff ~ r² × 2πR (toroidal slice × circumference)
    V_eff = r**2 * 2 * np.pi * R
    U_E = 0.5 * eps0 * E_field**2 * V_eff

    print(f"  {kname}: κ = {kappa:.2f}")
    print(f"    R = {R/fm:.1f} fm, r = {r/fm:.2f} fm")
    print(f"    E at surface = {E_field:.3e} V/m")
    print(f"    U_E ≈ {U_E:.3e} J = {U_E/(m_e*c**2):.4f} m_ec²")

    # Classical self-energy: U = e²/(8πε₀r) = αℏc/(2r)
    U_class = alpha * hbar * c / (2 * r)
    print(f"    Classical self-energy = {U_class/(m_e*c**2):.4f} m_ec²")
    print()


# The classical self-energy condition: U = αℏc/(2r) = m_e c²
# → r = αℏ/(2m_e c) = αξ/2 = r_e/2
# This is the classical electron radius (up to factor 2)!

print("CLASSICAL SELF-ENERGY CONDITION:")
print(f"  U = αℏc/(2r) = m_e c²")
print(f"  → r = αξ/2 = r_e/2 = {r_e/(2*fm):.2f} fm")
print(f"  → κ = R/r = (√5/2)ξ / (αξ/2) = √5/α = {np.sqrt(5)/alpha:.1f}")
print(f"  This is a HUGE κ — extremely thin tube.")
print()
print(f"  But with the Williamson/Macken factor (full surface integral):")
print(f"  U = αℏc/(2πr) = m_e c²  →  r = αξ/(2π) = {alpha*xi/(2*np.pi)/fm:.4f} fm")
print(f"  → κ = √5 × π/α = {np.sqrt(5)*np.pi/alpha:.1f}")


# =====================================================================
# THE ACTUAL CLOSURE: PHASE + ENERGY TOGETHER
# =====================================================================

print("\n" + "=" * 70)
print("THE CLOSED LOOP: PHASE + ENERGY = BOTH EQUATIONS")
print("=" * 70)

print("""
EQUATION 1 (Phase closure):
  R/ξ = √5/2    with ξ = ℏ/(m_e c)
  → R = (√5/2) × ℏ/(m_e c)

EQUATION 2 (Energy balance — classical self-energy):
  m_e c² = αℏc/(f × r)    where f is a geometric factor (≈ 2-4π)
  With r = R/κ:
  m_e c² = αℏcκ/(f × R) = αℏcκ/(f × (√5/2)ξ) = αm_e c² × 2κ/(f√5)
  1 = 2ακ/(f√5)
  κ = f√5/(2α)

  For f = 2: κ = √5/α = 306.5
  For f = 2π: κ = π√5/(α) = 963.1

These are much larger than Santos or Macken κ.

ALTERNATIVE: The energy is NOT classical self-energy but MODE energy.
For a torus eigenmode, U depends on the MODE PROFILE, not just Coulomb.
The mode energy scales differently from 1/r.
""")


# =====================================================================
# THE KEY INSIGHT: WHAT THE RESONANCE TELLS US
# =====================================================================

print("=" * 70)
print("SUMMARY: WHAT THE RESONANCE CONDITION ACTUALLY DETERMINES")
print("=" * 70)

print(f"""
THE RESULT:
  For a (2,1) torus knot in a superfluid vacuum with n(r) = √(1+ξ²/r²),
  the first phase-closure resonance (m=3) occurs at:

    R = (√5/2) × ξ    (exact in the thin-torus limit)

  where ξ = ƛ_C = ℏ/(m_e c) = {xi/fm:.1f} fm is the healing length.

  R = (√5/2) × ƛ_C = {R_exact/fm:.1f} fm

WHAT THIS MEANS:
  The electron torus does NOT sit at R = ξ (the healing length).
  It sits slightly OUTSIDE: R = 1.118 × ξ.

  At this radius, the refractive index is:
    n(R) = 3/√5 = {3/np.sqrt(5):.6f} ≈ 1.342

  And exactly 3 photon wavelengths fit around the (2,1) knot path.

THE NUMBERS:
  β = R/ξ = √5/2 = {np.sqrt(5)/2:.6f}
  n = 3/√5 = {3/np.sqrt(5):.6f}
  m = 3 (mode number)
  3/√5 × √5/2 = 3/2 = 1.5 (the "optical radius" in units of ξ)
  √(β² + 1) = 3/2 exactly (this IS the resonance condition)

BEAUTIFUL IDENTITY:
  β² + 1 = 5/4 + 1 = 9/4 = (3/2)²
  So: √(β² + 1) = 3/2 exactly — a Pythagorean triple!
  The (2,1) knot picks out the 3-4-5 Pythagorean structure:
    "3" wavelengths, "4" = p² = 2², hypotenuse "5" = β²+1...
    Wait: 3² = 4 + 5 → 9 = 4 + 5 ✓
    Actually: β² = 5/4, 1 = 4/4, β²+1 = 9/4
    This is the identity (3/2)² = (√5/2)² + 1²
    Or equivalently: 3² = (√5)² + 2²  →  9 = 5 + 4 ✓
""")


# =====================================================================
# FIGURE
# =====================================================================

print("\n=== Generating figures ===")

fig = plt.figure(figsize=(20, 20))

# --- Panel 1: Phase vs β showing m=3 crossing ---
ax1 = fig.add_subplot(3, 2, 1)
beta_plot = np.linspace(0.2, 3.5, 500)

# Analytical formula
phase_analytic = 2 * np.sqrt(beta_plot**2 + 1)
ax1.plot(beta_plot, phase_analytic, 'b-', linewidth=3, label='2√(β²+1) [large κ]')

# Numerical at finite κ
for kname, kappa, color, ls in [("Santos", kappa_santos, 'green', '--'),
                                  ("Macken", kappa_macken, 'purple', '--')]:
    phase_num = np.array([full_phase(b, kappa, 2, 1) for b in beta_plot])
    ax1.plot(beta_plot, phase_num, color=color, linewidth=2, linestyle=ls,
             label=f'Exact, κ={kname}')

# Mark resonances
for m in range(2, 8):
    ax1.axhline(m, color='orange', linestyle=':', alpha=0.5)
    ax1.text(3.55, m, f'm={m}', fontsize=9, color='orange', fontweight='bold')

# Mark the m=3 solution
ax1.plot(beta_exact, 3.0, 'r*', markersize=25, zorder=5,
         label=f'β = √5/2 = {beta_exact:.4f}')
ax1.axvline(beta_exact, color='red', linestyle='--', alpha=0.5)
ax1.axvline(1.0, color='gray', linestyle=':', alpha=0.3, label='β = 1 (R = ξ)')

ax1.set_xlabel('β = R/ξ  (orbit radius / healing length)', fontsize=13)
ax1.set_ylabel('Φ/(2π)  (number of wavelengths)', fontsize=13)
ax1.set_title('(2,1) Electron: Phase Closure Condition\nΦ = 2πm → m wavelengths fit',
              fontsize=13)
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.2, 3.5)
ax1.set_ylim(1.5, 7.5)


# --- Panel 2: The Pythagorean triangle ---
ax2 = fig.add_subplot(3, 2, 2)
ax2.set_aspect('equal')

# Draw the right triangle: sides √5/2, 1, hypotenuse 3/2
triangle_x = [0, np.sqrt(5)/2, np.sqrt(5)/2, 0]
triangle_y = [0, 0, 1, 0]
ax2.fill(triangle_x, triangle_y, alpha=0.15, color='blue')
ax2.plot(triangle_x, triangle_y, 'b-', linewidth=3)

# Right angle marker
sq_size = 0.08
ax2.plot([np.sqrt(5)/2 - sq_size, np.sqrt(5)/2 - sq_size, np.sqrt(5)/2],
         [0, sq_size, sq_size], 'b-', linewidth=1.5)

# Labels
ax2.text(np.sqrt(5)/4, -0.12, r'β = √5/2', fontsize=14, ha='center',
         fontweight='bold', color='red')
ax2.text(np.sqrt(5)/2 + 0.08, 0.5, '1', fontsize=14, ha='left',
         fontweight='bold', color='blue')
ax2.text(np.sqrt(5)/4 - 0.1, 0.6, r'$\frac{3}{2}$ = hyp.', fontsize=14, ha='center',
         fontweight='bold', color='purple')

# Equation
ax2.text(0.5, 1.4,
         r'$\left(\frac{3}{2}\right)^2 = \left(\frac{\sqrt{5}}{2}\right)^2 + 1^2$'
         '\n\n'
         r'$\frac{9}{4} = \frac{5}{4} + \frac{4}{4}$'
         '\n\n'
         'Or equivalently:  3² = (√5)² + 2²',
         fontsize=12, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax2.set_xlim(-0.3, 1.5)
ax2.set_ylim(-0.3, 1.8)
ax2.set_title('The Pythagorean Structure\nβ² + 1 = (m/p)² → 5/4 + 1 = 9/4',
              fontsize=13)
ax2.grid(True, alpha=0.2)


# --- Panel 3: Refractive index profile with orbit marked ---
ax3 = fig.add_subplot(3, 2, 3)
r_plot = np.linspace(0.1, 5.0, 1000)
n_plot = np.sqrt(1 + 1/r_plot**2)  # n vs r/ξ

ax3.plot(r_plot, n_plot, 'b-', linewidth=3, label=r'$n(r) = \sqrt{1 + \xi^2/r^2}$')

# Mark key points
n_at_xi = np.sqrt(2)
n_at_R = 3/np.sqrt(5)
n_at_re = np.sqrt(1 + 1/(alpha)**2)

ax3.plot(1.0, n_at_xi, 'ko', markersize=10, zorder=5)
ax3.annotate(f'R = ξ\nn = √2 = {n_at_xi:.3f}', xy=(1.0, n_at_xi),
             xytext=(1.5, 2.0), fontsize=10,
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax3.plot(beta_exact, n_at_R, 'r*', markersize=20, zorder=5)
ax3.annotate(f'R = √5/2 ξ\nn = 3/√5 = {n_at_R:.4f}\n(m=3 resonance)',
             xy=(beta_exact, n_at_R),
             xytext=(2.0, 1.6), fontsize=10, color='red', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='red', lw=2))

# Shade the resonance region
ax3.axhline(n_at_R, color='red', linestyle='--', alpha=0.5)
ax3.axvline(beta_exact, color='red', linestyle='--', alpha=0.5)

ax3.set_xlabel('r/ξ', fontsize=13)
ax3.set_ylabel('n(r)', fontsize=13)
ax3.set_title('Superfluid Refractive Index\nElectron orbit at n = 3/√5', fontsize=13)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0.1, 4)
ax3.set_ylim(0.9, 4)


# --- Panel 4: Convergence of β to √5/2 vs κ ---
ax4 = fig.add_subplot(3, 2, 4)

kappa_conv = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
beta_conv = []
eta_scan2 = np.linspace(0.5, 2.0, 2000)

for kap in kappa_conv:
    phase_s = np.array([full_phase(e, kap, 2, 1) for e in eta_scan2])
    found = False
    for i in range(len(eta_scan2) - 1):
        if (phase_s[i] - 3.0) * (phase_s[i+1] - 3.0) < 0:
            b = brentq(lambda x: full_phase(x, kap, 2, 1) - 3.0,
                       eta_scan2[i], eta_scan2[i+1], xtol=1e-14)
            beta_conv.append(b)
            found = True
            break
    if not found:
        beta_conv.append(np.nan)

ax4.semilogx(kappa_conv, beta_conv, 'bo-', markersize=8, linewidth=2,
             label='Numerical β(κ)')
ax4.axhline(beta_exact, color='red', linewidth=2, linestyle='--',
            label=f'√5/2 = {beta_exact:.6f} (κ→∞ limit)')
ax4.axvline(kappa_santos, color='green', linestyle=':', alpha=0.7, label='Santos κ')
ax4.axvline(kappa_macken, color='purple', linestyle=':', alpha=0.7, label='Macken κ')

ax4.set_xlabel('κ = R/r (aspect ratio)', fontsize=13)
ax4.set_ylabel('β = R/ξ', fontsize=13)
ax4.set_title('Convergence of Orbit Radius to √5/2\nas Torus Becomes Thinner',
              fontsize=13)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(1.05, 1.2)


# --- Panel 5: All (p,q=1) resonance towers ---
ax5 = fig.add_subplot(3, 2, 5)
colors_p = ['blue', 'red', 'green', 'orange', 'purple']

for p_val, color in zip(range(1, 6), colors_p):
    m_range = np.arange(p_val + 1, p_val + 8)
    betas = np.sqrt(m_range**2 / p_val**2 - 1)
    ax5.plot(m_range, betas, 'o-', color=color, markersize=8, linewidth=2,
             label=f'p={p_val}: β = √(m²/{p_val}² - 1)')

ax5.axhline(beta_exact, color='red', linestyle='--', alpha=0.3)
ax5.set_xlabel('Mode number m', fontsize=13)
ax5.set_ylabel('β = R/ξ', fontsize=13)
ax5.set_title('Resonance Towers for Each Winding Number p\nq=1 (single poloidal loop)',
              fontsize=13)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)


# --- Panel 6: Summary ---
ax6 = fig.add_subplot(3, 2, 6)
ax6.axis('off')

summary = f"""
╔══════════════════════════════════════════════════════╗
║  SELF-CONSISTENT (2,1) ELECTRON TORUS               ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Topology:  (p,q) = (2,1) torus knot                ║
║  Vacuum:    n(r) = √(1 + ξ²/r²)                     ║
║                                                      ║
║  PHASE CLOSURE:                                      ║
║    Φ = 2πp√(β²+1) = 2πm                             ║
║    √(β²+1) = m/p                                     ║
║                                                      ║
║  FIRST RESONANCE (m=3):                              ║
║    β = R/ξ = √5/2 = {np.sqrt(5)/2:.6f}                     ║
║    n(R) = 3/√5 = {3/np.sqrt(5):.6f}                       ║
║    R = {R_exact/fm:.1f} fm                                 ║
║                                                      ║
║  3 wavelengths of the photon fit around              ║
║  the (2,1) knot path. This is the MINIMUM            ║
║  mode that supports a (2,1) standing wave.           ║
║                                                      ║
║  Santos (κ=3.30): R = {xi*1.103/fm:.0f} fm, r = {xi*1.103/kappa_santos/fm:.0f} fm      ║
║  Macken (κ=π²):   R = {xi*1.116/fm:.0f} fm, r = {xi*1.116/kappa_macken/fm:.0f} fm       ║
║                                                      ║
║  IDENTITY: β² + 1 = (3/2)²                          ║
║  5/4 + 4/4 = 9/4  (Pythagorean)                     ║
║                                                      ║
║  The electron exists because 3² = (√5)² + 2².       ║
╚══════════════════════════════════════════════════════╝
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

fig.suptitle(r'Self-Consistent (2,1) Electron: $R/\xi = \sqrt{5}/2$, $m = 3$',
             fontsize=18, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('analysis/nwt_self_consistent_knot_v3.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved nwt_self_consistent_knot_v3.png")


# =====================================================================
# DERIVATION CHAIN FIGURE (for paper)
# =====================================================================

fig2, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.axis('off')

chain = r"""
                    THE DERIVATION CHAIN

    Superfluid vacuum with healing length $\xi$
                        │
                        ▼
        $n(r) = \sqrt{1 + \xi^2/r^2}$    (graded-index lens)
                        │
                        ▼
        Photon on (2,1) torus knot at radius $R$
                        │
                        ▼
     Optical phase:  $\Phi = 4\pi\sqrt{R^2 + \xi^2} / \xi$
                        │
                        ▼
     Phase closure:  $\Phi = 2\pi m$  (standing wave)
                        │
                        ▼
     $\sqrt{R^2 + \xi^2} = \frac{m\xi}{2}$  → first: $m = 3$
                        │
                        ▼
     $R = \frac{\sqrt{5}}{2}\,\xi$        $n(R) = \frac{3}{\sqrt{5}}$
                        │
                        ▼
     With $\xi = \bar{\lambda}_C = \frac{\hbar}{m_e c}$:
                        │
                        ▼
     ┌─────────────────────────────────────────┐
     │  $R = \frac{\sqrt{5}\,\hbar}{2\,m_e\,c} = 431.9$ fm       │
     │                                         │
     │  The electron orbit radius is fixed     │
     │  by topology and vacuum structure.      │
     └─────────────────────────────────────────┘
"""

ax.text(0.5, 0.5, chain, transform=ax.transAxes,
        fontsize=13, verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('analysis/nwt_derivation_chain.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved nwt_derivation_chain.png")
