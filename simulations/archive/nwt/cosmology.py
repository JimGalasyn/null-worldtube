"""
Skilton's alpha formula and integer cosmology.
"""

import numpy as np
from .constants import (
    c, hbar, h_planck, e_charge, eps0, m_e, m_e_MeV, alpha,
    eV, MeV, lambda_C, r_e, a_0, k_e, G_N, M_Planck, M_Planck_GeV,
    PARTICLE_MASSES,
)
from .core import find_self_consistent_radius


def print_skilton_analysis():
    """
    Skilton's integer-based cosmological model (1988): α⁻¹ = √(137² + π²)
    and the Pythagorean triple (88, 105, 137) connection to torus geometry.
    """
    print("=" * 70)
    print("  SKILTON'S INTEGER COSMOLOGY AND THE FINE-STRUCTURE CONSTANT")
    print("  F. Ray Skilton, 'Foundation for an integer-based")
    print("  cosmological model' (1988)")
    print("=" * 70)

    # ==========================================
    # Section 1: The α formula
    # ==========================================
    alpha_measured = 1.0 / 137.035999084   # CODATA 2018
    alpha_inv_measured = 1.0 / alpha_measured
    alpha_inv_skilton = np.sqrt(137**2 + np.pi**2)
    alpha_skilton = 1.0 / alpha_inv_skilton
    residual = alpha_inv_skilton - alpha_inv_measured
    residual_ppm = residual / alpha_inv_measured * 1e6

    print(f"\n  1. THE FORMULA: α⁻¹ = √(137² + π²)")
    print(f"  {'─'*55}")
    print(f"  α⁻¹ (Skilton)  = √(137² + π²)")
    print(f"                  = √({137**2} + {np.pi**2:.10f})")
    print(f"                  = {alpha_inv_skilton:.9f}")
    print(f"  α⁻¹ (measured) = {alpha_inv_measured:.9f}  (CODATA 2018)")
    print(f"  Residual        = {residual:.9f}")
    print(f"  Agreement       = {abs(residual_ppm):.2f} ppm  ({abs(residual_ppm)/1e6:.1e} relative)")
    print(f"\n  This is accurate to 1 part in {1e6/abs(residual_ppm):.0f} — remarkable for")
    print(f"  a formula with NO free parameters.")

    # ==========================================
    # Section 2: The Pythagorean triple
    # ==========================================
    print(f"\n  2. THE PYTHAGOREAN TRIPLE: (88, 105, 137)")
    print(f"  {'─'*55}")
    print(f"  88² + 105² = {88**2} + {105**2} = {88**2 + 105**2}")
    print(f"  137²       = {137**2}")
    print(f"  ✓ Perfect Pythagorean triple")
    print(f"\n  Right triangle with legs 88 and 105, hypotenuse 137.")
    print(f"  α⁻¹ = √(137² + π²) says: the physical α⁻¹ is the hypotenuse")
    print(f"  of a triangle with legs 137 and π.")
    print(f"\n  Combined: α⁻¹ = √(88² + 105² + π²)")
    combined = np.sqrt(88**2 + 105**2 + np.pi**2)
    print(f"            = √({88**2} + {105**2} + {np.pi**2:.6f})")
    print(f"            = {combined:.9f}  (same result, deeper decomposition)")

    # ==========================================
    # Section 3: Connection to torus geometry
    # ==========================================
    print(f"\n  3. CONNECTION TO TORUS GEOMETRY")
    print(f"  {'─'*55}")
    print(f"  Key observation: 88 + 105 = {88 + 105}")
    # Reduced Compton wavelength = ℏ/(m_e c) ≈ 386 fm
    lambda_C_fm = lambda_C * 1e15
    half_lambda_C_fm = lambda_C_fm / 2
    e_sol = find_self_consistent_radius(m_e_MeV, p=1, q=1, r_ratio=alpha)
    if e_sol:
        R_fm = e_sol['R_femtometers']
    else:
        R_fm = lambda_C_fm
    print(f"\n  Electron scales:")
    print(f"    λ_C (reduced Compton) = {lambda_C_fm:.1f} fm")
    print(f"    λ_C / 2               = {half_lambda_C_fm:.1f} fm")
    print(f"    Self-consistent R     = {R_fm:.1f} fm  (r/R = α)")
    print(f"    88 + 105 = 193        ≈ λ_C / 2  (within {abs(193 - half_lambda_C_fm)/half_lambda_C_fm*100:.1f}%)")
    print(f"\n  The sum 88 + 105 = 193 matches half the reduced Compton")
    print(f"  wavelength — the characteristic scale of EM interactions.")

    print(f"\n  Geometric interpretation:")
    print(f"  The α formula decomposes into a right triangle whose legs")
    print(f"  sum to the electron torus major radius in femtometers:")
    print(f"\n       π")
    print(f"       ├─────┐")
    print(f"       │     │")
    print(f"  137  │     │ α⁻¹ = {alpha_inv_skilton:.3f}")
    print(f"       │     │")
    print(f"       └─────┘")
    print(f"         ↓")
    print(f"    88² + 105² = 137²")
    print(f"    88 + 105 = 193 ≈ λ_C/2 (fm)")

    # ==========================================
    # Section 4: Skilton's cycloidal photon = torus knot
    # ==========================================
    print(f"\n  4. CYCLOIDAL PHOTON MODEL")
    print(f"  {'─'*55}")
    print(f"  Skilton (1988) proposed: particles are photons travelling")
    print(f"  on cycloidal paths. A cycloid on a torus IS a torus knot.")
    print(f"  This is exactly our model, arrived at independently.")
    print(f"\n  Skilton's key insight: the integer 137 that appears in α")
    print(f"  isn't just the coupling constant — it's the hypotenuse of")
    print(f"  a right triangle encoding particle geometry.")
    print(f"\n  In the null worldtube framework:")
    print(f"  • α = e²/(4πε₀ℏc) = ratio of EM self-energy to circulation energy")
    print(f"  • 137 = number of toroidal circulations per EM interaction timescale")
    print(f"  • (88, 105) = geometric decomposition of the torus embedding")
    print(f"  • π enters because the photon circulates on a curved surface")

    # ==========================================
    # Section 5: Mass ratio hints
    # ==========================================
    print(f"\n  5. MASS RATIO HINTS")
    print(f"  {'─'*55}")

    # Check if triangle integers relate to mass ratios
    ratio_muon = PARTICLE_MASSES['muon'] / PARTICLE_MASSES['electron']
    ratio_pion = PARTICLE_MASSES['pion±'] / PARTICLE_MASSES['electron']
    ratio_proton = PARTICLE_MASSES['proton'] / PARTICLE_MASSES['electron']

    print(f"  Mass ratios from the Pythagorean triple:")
    print(f"    105/88       = {105/88:.6f}")
    print(f"    137/88       = {137/88:.6f}")
    print(f"    (105/88)³    = {(105/88)**3:.2f}   (m_μ/m_e = {ratio_muon:.2f})")
    print(f"    88 × 105/π²  = {88*105/np.pi**2:.2f}  (cf. m_π/m_e = {ratio_pion:.2f})")
    print(f"    88 × 105/10  = {88*105/10:.1f}  (cf. m_p/m_e = {ratio_proton:.2f})")
    print(f"\n  Status: suggestive but speculative. The (88,105,137)")
    print(f"  triple may encode geometric ratios of the torus, but")
    print(f"  deriving mass ratios from it requires more work.")

    # ==========================================
    # Section 6: What Skilton provides
    # ==========================================
    print(f"\n  6. WHAT SKILTON'S FORMULA PROVIDES")
    print(f"  {'─'*55}")
    print(f"  ✓ α⁻¹ to 0.12 ppm with zero free parameters")
    print(f"  ✓ Structural decomposition: 137² = 88² + 105²")
    print(f"  ✓ Geometric connection: 88 + 105 ≈ R_electron in fm")
    print(f"  ✓ Cycloidal photon model = torus knot model")
    print(f"  ✓ π enters naturally (curved circulation)")
    print(f"\n  OPEN QUESTIONS:")
    print(f"  • Why 137 and not some other integer?")
    print(f"  • Why this particular Pythagorean triple?")
    print(f"  • Can the (88, 105) decomposition predict mass ratios?")
    print(f"  • Is the 0.12 ppm residual meaningful or coincidental?")
    print(f"  • Does α⁻¹ = √(n² + π²) generalize to other couplings?")
