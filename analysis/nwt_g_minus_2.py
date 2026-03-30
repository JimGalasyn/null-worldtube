#!/usr/bin/env python3
"""
NWT Anomalous Magnetic Moment: g-2 from Kelvin Wave Zero-Point Fluctuations

The electron is a (2,1) vortex knot on a torus with κ = R/r = π².
Its magnetic moment arises from the circulating vortex current.
The "bare" g-factor is g=2 from the p=2 double-winding topology.

The anomalous correction (g-2)/2 comes from quantum fluctuations
of the vortex core — Kelvin waves. These are helical oscillations
of the vortex line around its equilibrium position, analogous to
the virtual photon loops that give the Schwinger correction in QED.

KELVIN WAVES ON A VORTEX:
  A vortex ring of radius R and core size a supports Kelvin modes
  with azimuthal number m and frequency:
    ω_m = (m Γ)/(4π² R²) × [ln(8R/a) - C(m)]
  where Γ = circulation quantum, C(m) depends on the core model.

  Each mode has zero-point amplitude:
    ⟨δr²⟩_m = ℏ/(2 m_eff ω_m)

  The total ZPE correction to the enclosed area A = πR²:
    δA/A = Σ_m ⟨δr²⟩_m / R²

  This modifies the magnetic moment:
    μ = (g/2) × μ_0 × (1 + δA/A)
    → (g-2)/2 = δA/A

THE KEY QUESTION:
  Does Σ_m ⟨δr²⟩_m / R² = α/(2π) = 1/(2π√2 π⁴)?

  For the NWT electron:
    Γ = κ_circ = 2πℏ/m*
    R = β ξ = (√5/2) ξ
    a = r = R/κ = R/π²
    m_eff = m* = m_e/√2
    ln(8R/a) = ln(8κ) = ln(8π²) ≈ 4.37
"""

import numpy as np

# ── Physical constants ────────────────────────────────────────────────
hbar = 1.054571817e-34
c = 2.99792458e8
m_e = 9.1093837015e-31
alpha_pdg = 7.2973525693e-3
mu_B = 9.2740100783e-24  # Bohr magneton

# g-2 experimental values
ae_exp = 1.15965218128e-3  # (g-2)/2 for electron (most precise measurement)
ae_qed_1loop = alpha_pdg / (2*np.pi)  # Schwinger term

# NWT parameters
xi = hbar / (m_e * c)
m_star = m_e / np.sqrt(2)
kappa_circ = 2 * np.pi * hbar / m_star
c_s = c / np.sqrt(2)
KAPPA = np.pi**2
beta_e = np.sqrt(5.0/4.0)
R_e = beta_e * xi
r_e = R_e / KAPPA
alpha_nwt = 1.0 / (np.sqrt(2) * KAPPA**2)

print("=" * 90)
print("NWT ANOMALOUS MAGNETIC MOMENT: g-2 FROM KELVIN WAVES")
print("=" * 90)

print(f"""
  Physical parameters:
    ξ = {xi*1e15:.2f} fm (healing length)
    m* = m_e/√2 = {m_star:.4e} kg
    κ_circ = 2πℏ/m* = {kappa_circ:.4e} m²/s
    κ = π² = {KAPPA:.4f} (aspect ratio)
    β = √5/2 = {beta_e:.6f}
    R = βξ = {R_e*1e15:.2f} fm
    r = R/κ = {r_e*1e15:.2f} fm
    ln(8κ) = {np.log(8*KAPPA):.4f}

  Target:
    (g-2)/2 experimental = {ae_exp:.12e}
    α/(2π) Schwinger     = {ae_qed_1loop:.12e}
    Difference            = {ae_exp - ae_qed_1loop:.6e} (higher-order QED)
""")


# ══════════════════════════════════════════════════════════════════════
# APPROACH 1: Direct Kelvin wave mode sum
# ══════════════════════════════════════════════════════════════════════

print("=" * 90)
print("APPROACH 1: Kelvin wave zero-point fluctuations")
print("=" * 90)

print("""
  Kelvin wave dispersion on a vortex ring:
    ω(m) = (m κ_circ)/(4π² R²) × [ln(8R/a) - C(m)]

  For a thin core (Kelvin's formula, hollow core):
    C(m) = 1/2 + 1/(2m)  for m ≥ 1

  For a solid core (Thomson):
    C(m) = (2m-1)/(2m) for m ≥ 1

  The zero-point amplitude of mode m:
    ⟨δr²⟩_m = ℏ/(2 m_eff ω_m × R²/R²) ... need to be careful with units

  Actually, for a vortex ring, each Kelvin mode m has energy:
    E_m = (1/2) M_eff ω_m² A_m²
  where A_m is the mode amplitude and M_eff is the effective mass.

  The effective mass per mode for a vortex ring:
    M_m = ρ₀ × π r² × 2πR = ρ₀ π r² L  (mass of fluid in the tube)

  But more precisely, the kinetic energy of a Kelvin wave involves
  the INDUCED velocity field, which extends beyond the core:
    M_m ≈ ρ₀ × π R × ln(8R/r) × ... (depends on m)

  Let me use the standard result from Barenghi & Donnelly:
    The zero-point displacement of mode m on a vortex ring:
    ⟨δr²⟩_m = ℏ/(2 ω_m × M_eff_per_mode)
""")

# Kelvin wave frequencies on the (2,1) torus knot
# The torus knot modifies the Kelvin spectrum:
# On a straight vortex: ω(k) = (Γ k²)/(4π) × ln(1/(k a))
# On a ring: ω(m) = (m Γ)/(4π² R²) × [ln(8R/a) - C(m)]
# On a (2,1) knot: additional curvature/torsion corrections

# For now, use the ring dispersion (leading order)
Gamma = kappa_circ  # circulation quantum
R = R_e
a = r_e  # core radius = tube radius
log_factor = np.log(8*R/a)  # = ln(8κ) for R/a = κ

print(f"  Circulation: Γ = {Gamma:.4e} m²/s")
print(f"  Ring radius: R = {R:.4e} m")
print(f"  Core radius: a = {a:.4e} m")
print(f"  ln(8R/a) = ln(8κ) = {log_factor:.4f}")

# Kelvin mode frequencies
print(f"\n  Kelvin wave spectrum (ring approximation):")
print(f"  {'m':>4s} {'ω_m (rad/s)':>15s} {'f_m (Hz)':>15s} {'ℏω_m (eV)':>12s}")
for m in range(1, 21):
    C_m = 0.5 + 0.5/m  # hollow core
    omega_m = m * Gamma / (4*np.pi**2 * R**2) * (log_factor - C_m)
    if omega_m <= 0:
        continue
    f_m = omega_m / (2*np.pi)
    E_m_eV = hbar * omega_m / 1.602e-19
    if m <= 10 or m == 20:
        print(f"  {m:4d} {omega_m:15.4e} {f_m:15.4e} {E_m_eV:12.4e}")

# Zero-point area correction
# The magnetic moment of the electron = current × area
# μ = (e/T) × πR² where T = period of circulation
# Each Kelvin mode m shifts R → R + δr_m cos(mφ - ω_m t)
# The time-averaged area correction from mode m:
# δA_m = πR² × 2⟨δr²⟩_m / R² = 2π⟨δr²⟩_m (for circular mode shape)
# But for a cos(mφ) mode, the area correction averages to zero for m≥2
# Only m=1 (translation mode) and quadrupole effects contribute

# MORE CAREFULLY:
# The magnetic moment involves the VECTOR area enclosed by the current loop.
# For a Kelvin wave perturbation δr(φ) = A_m cos(mφ):
#   The current loop becomes r(φ) = R + A_m cos(mφ)
#   The enclosed area: A = (1/2)∮ r² dφ = πR² + πA_m²/2  (for any m≥1)
#   So δA/A = A_m²/(2R²) per mode

# The zero-point amplitude:
# For a quantum vortex ring, the effective mass for mode m is:
#   M_m = ρ₀ × π a² × 2πR × f(m) where f(m) accounts for the
#   induced flow. For m=1 (translation): f=1. For higher m: f ~ m.
#
# The standard result (Barenghi, Donnelly, Vinen):
#   ⟨A_m²⟩ = ℏ/(2 M_m ω_m)

print(f"\n\n  Zero-point area correction:")
print(f"  δA/A = Σ_m A_m²/(2R²) = Σ_m ℏ/(4 M_m ω_m R²)")
print(f"")

# Effective mass of the vortex ring
# M_ring = ρ₀ × π a² × L where L = 2πR is the ring circumference
# But ρ₀ for the NWT condensate: ρ₀ = m* n₀ where n₀ = 1/ξ³ (one particle per ξ³)
# Or from the vortex energy: E = (1/2)ρ₀ κ² R (ln(8R/a) - 1/2) × 2π
# Setting E = m_e c² gives ρ₀

# From our earlier work: the electron mass IS the vortex ring energy
# m_e c² = (1/2) ρ₀ Γ² × 2πR × (ln(8R/a) - 1/2)
# → ρ₀ = m_e c² / (π Γ² R (ln(8R/a) - 1/2))

rho_0 = m_e * c**2 / (np.pi * Gamma**2 * R * (log_factor - 0.5))
print(f"  ρ₀ = m_e c²/(πΓ²R(ln(8κ)-½)) = {rho_0:.4e} kg/m³")

# Effective mass per Kelvin mode
# M_m = ρ₀ × π a² × 2πR × (1 + corrections for m)
# For thin core: M_m ≈ ρ₀ π a² × 2πR for all m (leading order)
M_ring = rho_0 * np.pi * a**2 * 2*np.pi * R
print(f"  M_ring = ρ₀ πa² × 2πR = {M_ring:.4e} kg")
print(f"  M_ring / m_e = {M_ring/m_e:.6f}")

# Sum over Kelvin modes
delta_A_over_A = 0
mode_contributions = []

# UV cutoff: modes with wavelength λ = 2πR/m > 2πa → m < R/a = κ
m_max = int(R/a)  # = κ ≈ 10
# IR cutoff: m ≥ 1 (no m=0 mode for Kelvin waves)

print(f"\n  Cutoffs: m_min = 1, m_max = R/a = κ = {m_max}")
print(f"  (modes with m > κ don't fit in the tube cross-section)")

print(f"\n  {'m':>4s} {'ω_m':>12s} {'M_eff':>12s} {'⟨A²⟩/(2R²)':>14s} {'cumulative':>14s}")

for m in range(1, m_max + 1):
    C_m = 0.5 + 0.5/m
    omega_m = m * Gamma / (4*np.pi**2 * R**2) * (log_factor - C_m)
    if omega_m <= 0:
        continue

    # Effective mass per mode: for higher m, the induced flow is more localized
    # M_m ≈ M_ring / m (shorter wavelength → less fluid involved)
    # This is the standard result for Kelvin waves on thin cores
    M_m = M_ring / m

    # Zero-point amplitude squared
    A_m_sq = hbar / (2 * M_m * omega_m)

    # Area correction from this mode
    dA = A_m_sq / (2 * R**2)
    delta_A_over_A += dA
    mode_contributions.append((m, omega_m, M_m, dA))

    print(f"  {m:4d} {omega_m:12.4e} {M_m:12.4e} {dA:14.6e} {delta_A_over_A:14.6e}")

print(f"\n  TOTAL δA/A = {delta_A_over_A:.6e}")
print(f"  Compare: α/(2π) = {alpha_pdg/(2*np.pi):.6e}")
print(f"  Ratio: {delta_A_over_A / (alpha_pdg/(2*np.pi)):.4f}")


# ══════════════════════════════════════════════════════════════════════
# APPROACH 2: Analytical summation
# ══════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*90}")
print(f"APPROACH 2: Analytical mode sum")
print(f"{'='*90}")

print(f"""
  Each mode contributes:
    δA_m/A = ℏ/(4 M_m ω_m R²)
           = ℏ/(4 × (M_ring/m) × (mΓ/(4π²R²))(ln8κ-C_m) × R²)
           = ℏ × 4π²R² × m / (4 M_ring × m × Γ × (ln8κ-C_m) × R²)
           = π²ℏ / (M_ring Γ (ln8κ-C_m))

  For large m: C_m → 1/2, so (ln8κ-C_m) → ln8κ-1/2 = const
  The sum becomes:
    δA/A = Σ_{{m=1}}^κ π²ℏ / (M_ring Γ (ln8κ-1/2))
         = κ × π²ℏ / (M_ring Γ (ln8κ-1/2))

  Let's compute this:
""")

numerator = KAPPA * np.pi**2 * hbar
denominator = M_ring * Gamma * (log_factor - 0.5)
delta_A_analytical = numerator / denominator

print(f"  κ × π²ℏ / (M_ring × Γ × (ln8κ-½))")
print(f"  = {KAPPA:.4f} × {np.pi**2:.4f} × {hbar:.4e}")
print(f"    / ({M_ring:.4e} × {Gamma:.4e} × {log_factor-0.5:.4f})")
print(f"  = {delta_A_analytical:.6e}")
print(f"")
print(f"  Compare: α/(2π) = {alpha_pdg/(2*np.pi):.6e}")
print(f"  Ratio: {delta_A_analytical / (alpha_pdg/(2*np.pi)):.4f}")

# ══════════════════════════════════════════════════════════════════════
# APPROACH 3: Dimensionless analysis — does α/(2π) emerge?
# ══════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*90}")
print(f"APPROACH 3: Dimensionless — expressing δA/A in terms of α and κ")
print(f"{'='*90}")

print(f"""
  All physical quantities in terms of ξ, m*, c_s:
    Γ = 2πℏ/m* = 2π(ξ c_s)         [circulation quantum]
    R = βξ                            [major radius]
    a = R/κ = βξ/κ                    [tube radius]
    M_ring = ρ₀ πa² × 2πR            [tube fluid mass]
    ρ₀ = m_e c² / (πΓ²R(lnΛ-½))     [condensate density]
    lnΛ = ln(8κ)                      [logarithmic factor]

  Substituting M_ring:
    M_ring = [m_ec²/(πΓ²R(lnΛ-½))] × π(R/κ)² × 2πR
           = m_ec² × 2π R² / (Γ² κ² (lnΛ-½))

  The area correction:
    δA/A = κ π²ℏ / (M_ring Γ (lnΛ-½))
         = κ π²ℏ Γ² κ² (lnΛ-½) / (m_ec² 2πR² Γ (lnΛ-½))
         = κ³ π² ℏ Γ / (2π m_ec² R²)
         = κ³ π ℏ Γ / (m_ec² R²)

  With Γ = 2πℏ/m* and R = βξ:
    δA/A = κ³ π × ℏ × 2πℏ/m* / (m_ec² × β²ξ²)
         = 2π² κ³ ℏ² / (m* m_ec² β² ξ²)

  Using ξ = ℏ/(m_ec), m* = m_e/√2:
    δA/A = 2π² κ³ ℏ² √2 / (m_e × m_ec² × β² × ℏ²/(m_e²c²))
         = 2π² κ³ √2 m_e c² / (m_ec² β²)  ... wait, let me redo this carefully
""")

# Let me just compute numerically and express in terms of α, κ, β
# We have:
#   δA/A (numerical) = {delta_A_analytical}
#   α = 1/(√2 κ²)
#   α/(2π) = 1/(2π√2 κ²)

# What combination of α, κ, β gives our numerical result?
target = delta_A_analytical
schwinger = alpha_pdg / (2*np.pi)

print(f"\n  Numerical δA/A = {target:.6e}")
print(f"  α/(2π)         = {schwinger:.6e}")
print(f"  Ratio           = {target/schwinger:.6f}")
print(f"")

# Try various combinations
print(f"  Testing dimensionless expressions:")
combos = [
    ("α/(2π)", alpha_nwt/(2*np.pi)),
    ("α/(π)", alpha_nwt/np.pi),
    ("α/(4π)", alpha_nwt/(4*np.pi)),
    ("α²/(2π)", alpha_nwt**2/(2*np.pi)),
    ("1/(2π κ²)", 1/(2*np.pi*KAPPA**2)),
    ("1/(2π√2 κ²)", 1/(2*np.pi*np.sqrt(2)*KAPPA**2)),
    ("β²/(2π κ³)", beta_e**2/(2*np.pi*KAPPA**3)),
    ("κ/(2π β² ln(8κ)²)", KAPPA/(2*np.pi*beta_e**2*np.log(8*KAPPA)**2)),
    ("1/(4π² κ (ln8κ-½))", 1/(4*np.pi**2*KAPPA*(np.log(8*KAPPA)-0.5))),
    ("1/(2π² κ ln8κ)", 1/(2*np.pi**2*KAPPA*np.log(8*KAPPA))),
    ("α × β²/(π κ)", alpha_nwt*beta_e**2/(np.pi*KAPPA)),
]

for label, value in combos:
    ratio = value / schwinger
    err = abs(value - schwinger)/schwinger * 100
    mark = " ◄◄◄" if err < 5 else " ◄" if err < 20 else ""
    print(f"    {label:35s} = {value:.6e}  ratio={ratio:.4f}  err={err:.1f}%{mark}")

# Also check: what if we use (g-2)/2 directly from the mode sum?
# The ratio δA/A / [α/(2π)] tells us how many factors of α we're off

print(f"\n\n{'='*90}")
print(f"APPROACH 4: The (2,1) knot correction to mode counting")
print(f"{'='*90}")

print(f"""
  On a simple ring, the Kelvin wave sum gives δA/A ∝ κ (number of modes).
  On the (2,1) torus KNOT, the topology modifies the mode counting:

  The (2,1) knot wraps twice around the torus. Each Kelvin mode on
  the knot has a DOUBLED path length (2× the unknotted ring), so:
    - The mode frequencies are halved: ω_m → ω_m/2
    - The effective mass is doubled: M_m → 2M_m
    - The ZPE per mode: ⟨A²⟩ = ℏ/(2×2M×ω/2) = ℏ/(2Mω) (unchanged!)

  But the NUMBER of modes changes: on the (2,1) knot, modes must
  satisfy a PHASE CLOSURE condition. Only modes with:
    m = n × p  (n integer, p=2 for the (2,1) knot)
  are allowed — the mode number must be a multiple of p.

  This reduces the number of modes by a factor of p:
    Σ_m → Σ_{{n=1}}^{{κ/p}} = κ/(2p) modes instead of κ modes

  Wait — that would give δA/A ∝ κ/p, and we need it to be α/(2π).
  Let me check:
    κ/(2p) / (κ × something) → need to track the factors carefully.
""")

# Let me just compute with the p=2 knot correction
# On the (2,1) knot, allowed Kelvin modes have m = 2n (even only)
# because the knot wraps twice around
delta_A_knot = 0
for m in range(2, m_max+1, 2):  # even modes only
    C_m = 0.5 + 0.5/m
    omega_m = m * Gamma / (4*np.pi**2 * R**2) * (log_factor - C_m)
    if omega_m <= 0: continue
    M_m = M_ring / m
    A_m_sq = hbar / (2 * M_m * omega_m)
    delta_A_knot += A_m_sq / (2 * R**2)

print(f"  Even-mode sum (p=2 knot): δA/A = {delta_A_knot:.6e}")
print(f"  All-mode sum:             δA/A = {delta_A_analytical:.6e}")
print(f"  Ratio (knot/ring):              = {delta_A_knot/delta_A_analytical:.4f}")
print(f"  Compare α/(2π):                 = {schwinger:.6e}")
print(f"  Knot sum / [α/(2π)]:            = {delta_A_knot/schwinger:.4f}")

# Summary
print(f"\n\n{'='*90}")
print(f"SUMMARY")
print(f"{'='*90}")
print(f"""
  The Kelvin wave zero-point mode sum gives:

    (g-2)/2 = δA/A = {delta_A_analytical:.6e}  (all modes, ring)
    (g-2)/2 = δA/A = {delta_A_knot:.6e}  (even modes, (2,1) knot)

  Target: α/(2π) = {schwinger:.6e}

  The ratio of our result to the Schwinger term:
    Ring:  {delta_A_analytical/schwinger:.4f}
    Knot:  {delta_A_knot/schwinger:.4f}

  We are off by a factor of ~{delta_A_analytical/schwinger:.1f} (ring) or ~{delta_A_knot/schwinger:.1f} (knot).

  The factor {delta_A_analytical/schwinger:.2f} ≈ {delta_A_analytical/schwinger:.1f} needs to be understood.
  Possible sources:
    - The M_m = M_ring/m scaling may be wrong for a torus (not a ring)
    - The (2,1) knot has additional constraints from the poloidal winding
    - The area correction formula needs the VECTOR area, not scalar
    - The ρ₀ determination from m_ec² may double-count
""")


if __name__ == '__main__':
    pass
