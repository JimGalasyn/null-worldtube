#!/usr/bin/env python3
"""
NWT speed-of-light derivation: emergent vs built-in c
=======================================================

Question: can c be derived from NWT-internal parameters (v_EW, ρ_0)
rather than being an independent input?

Two readings:

  (1) Lorentz-invariant: in NWT's relativistic Lagrangian
      L = |∂_μ ψ|² - V(|ψ|²) + ...,  c is built in by the metric
      structure.  All wave perturbations propagate at c by
      construction.  We CANNOT derive c from condensate parameters
      in this view; c is the unit-conversion factor between length
      and time.

  (2) Volovik-style emergent: the underlying NWT condensate has
      non-relativistic dynamics with an emergent Lorentz invariance
      at long wavelengths.  c emerges as the speed of long-wavelength
      perturbations, c² = (∂P/∂ρ)|_{ρ_0}.  In this view, c IS derived
      from condensate parameters.

In NWT papers as currently written, view (1) is the explicit
framework.  This script explores what view (2) would require and
what dimensionless ratios NWT predicts in either case.

The script also tabulates the dimensionless predictions involving
{ℏ, c, k_B, e, G} that NWT *does* make:  Compton-Bekenstein,
Compton-Unruh, Margolus-Levitin saturation, m_e/m_Pl, α.
"""

from __future__ import annotations

import numpy as np


# Constants (CODATA 2018 / SI exact post-2019)
HBAR  = 1.054571817e-34       # J·s        (exact by SI)
C     = 299792458.0           # m/s        (exact by SI)
K_B   = 1.380649e-23          # J/K        (exact by SI)
E_CH  = 1.602176634e-19       # C          (exact by SI)
G_N   = 6.67430e-11           # m^3 kg^-1 s^-2 (measured, ±22 ppm)
ALPHA = 1.0 / 137.035999084   # fine-structure (CODATA)

M_E   = 9.1093837015e-31      # kg, electron mass
M_E_MEV = M_E * C**2 / 1.602176634e-13   # MeV


def section(s: str) -> None:
    print()
    print("=" * 78)
    print(" " + s)
    print("=" * 78)


# ====================================================================
# Section 1: Compton-scale natural units
# ====================================================================

def section1_natural_units() -> None:
    section("Section 1: NWT-natural units at the Compton scale")

    lambda_C = HBAR / (M_E * C)  # reduced Compton wavelength
    omega_C = M_E * C**2 / HBAR   # Compton frequency
    T_C = 2 * np.pi / omega_C     # Compton period

    print(f"""
  Choose the electron Compton scale as NWT-natural:
    Length:  ƛ_C = ℏ/(m_e c)         = {lambda_C:.6e} m
    Time:    1/ω_C = ℏ/(m_e c²)      = {1/omega_C:.6e} s
    Energy:  ℏω_C = m_e c²            = {M_E*C**2:.6e} J  =  {M_E_MEV:.4f} MeV
    Mass:    m_e                      = {M_E:.6e} kg

  In these units (m_e = ƛ_C = 1/ω_C = 1):
    ℏ = 1   (= m_e × c² × 1/ω_C)
    c = 1   (= ƛ_C × ω_C)
    k_B = ?  (depends on the temperature unit)

  Choose temperature unit s.t. T_eff = m_e c²/(2π k_B) = 1/(2π):
    Then k_B = 1.

  In this NWT-natural unit system, all four constants {{ℏ, c, k_B, e}}
  equal 1.  Dimensionless predictions are what NWT actually delivers.
""")


# ====================================================================
# Section 2: Wave-speed in the BPS Lagrangian (Lorentz-invariant view)
# ====================================================================

def section2_lorentz_invariant_c() -> None:
    section("Section 2: c from the relativistic BPS Lagrangian (view 1)")

    print(r"""
  NWT's BPS Lagrangian (Paper 16 §5, schematically):

      L = |D_μ ψ|² - V(|ψ|²) + (Skyrme/Hopf terms)

  For a homogeneous condensate ψ = ψ_0 + δψ at the BPS minimum
  |ψ_0|² = v²/2:

  Wave equation for δψ (Goldstone phase mode):
      (∂_t² - ∇²) δψ = 0

  This propagates at SPEED 1 in natural units — c is built in by
  the Lorentz-invariant kinetic term |D_μ ψ|².  The speed is
  independent of v_EW, ρ_0, or any condensate parameter.

  CONCLUSION (Lorentz-invariant view):
  c is NOT derivable from v_EW or ρ_0 in NWT's standard framework.
  c is the unit-conversion factor between length and time.

  However:  NWT predicts SPECIFIC RATIOS involving v_EW, ρ_0, ξ,
  ℏ, c, m_e (all dimensionful) that reduce the set of independent
  inputs.  We tabulate these in Section 3.
""")


# ====================================================================
# Section 3: Volovik-style emergent c (view 2)
# ====================================================================

def section3_emergent_c() -> None:
    section("Section 3: Volovik-style emergent c (view 2)")

    print(r"""
  In a non-relativistic superfluid (e.g., He-3 superfluid), the
  speed of long-wavelength perturbations c_s is given by

      c_s² = (∂P/∂ρ)|_{ρ_0}

  where P is pressure and ρ is density at the condensate point.
  For a BEC with chemical potential μ:  c_s² = μ/m  (Bogoliubov).

  Volovik's analog gravity programme posits that LORENTZ INVARIANCE
  ITSELF is emergent from underlying non-relativistic dynamics.  In
  that picture, c is the c_s of the underlying condensate at long
  wavelengths.

  For NWT to fit this template:
    - The BPS condensate has an underlying non-relativistic structure.
    - The Lorentz-invariant Lagrangian of Paper 16 §5 emerges only
      at long wavelengths.
    - c emerges from c_s² = (∂P/∂ρ)|_{ρ_0}.

  This is a deep RESEARCH HYPOTHESIS, not currently in NWT papers.
  It would mean:

      c = c_s(BPS condensate)
        = √( (∂P/∂ρ)|_{ρ_0} )
        = function of v_EW (or ρ_0)

  This ELIMINATES c as an independent dimensional input.

  Concrete computation for NWT (sketch):
    - Compute P(ρ) for the BPS condensate from the L1 Lagrangian.
    - At ρ_0 = v²/2:  evaluate ∂P/∂ρ.
    - Express in terms of v_EW (or ρ_0).
    - The resulting c_s² should equal c² = 1 in natural units, BUT
      with a specific dimensionless coefficient that NWT predicts.

  This is the cleanest formulation of Jim's intuition:
  c² = (NWT topological factor) × v_EW²/(ℏ²)

  For a Mexican-hat potential V = -μ²|ψ|² + λ|ψ|⁴:
    Minimum at v² = μ²/λ.
    P = -V(|ψ_0|²) - (gradient terms at higher k)
    ∂P/∂ρ = ?

  For the standard relativistic Higgs (Lorentz-invariant):
    Phonon dispersion ω² = c²k² + m_H²  (no 'c_s' distinct from c)
    The phonon speed equals c by Lorentz invariance.

  For a non-relativistic Bogoliubov BEC:
    ω² = c_s² k² + ℏ²k⁴/(4m²)
    c_s = √(μ/m) — DEPENDS on μ and m.

  NWT would need to be in the second category (non-relativistic
  underlying) for c to emerge from condensate properties.  This is a
  NEW research direction.
""")


# ====================================================================
# Section 4: Dimensionless predictions NWT actually makes
# ====================================================================

def section4_dimensionless_predictions() -> None:
    section("Section 4: Dimensionless predictions involving ℏ, c, k_B, e, G")

    # Compute the various predictions
    m_Pl = np.sqrt(HBAR * C / G_N)
    m_e_over_m_Pl = M_E / m_Pl

    # Bekenstein bound on Compton sphere
    # S ≤ 2π R E / ℏc  (Bekenstein)
    # R = ƛ_C, E = m_e c²
    # S_max = 2π × ƛ_C × m_e c² / ℏc = 2π × (ℏ/m_e c) × m_e c² / ℏc = 2π
    S_max_compton = 2 * np.pi
    S_max_compton_bits = S_max_compton / np.log(2)

    # Margolus-Levitin saturation at Compton scale
    # τ_min = πℏ/(2E), so for E = m_e c² and τ = T_C = 2πℏ/(m_e c²):
    # bits per period = E × T_C / (πℏ ln 2) × 2 = 4 / ln 2
    bits_per_compton_period = 4 / np.log(2)

    # Compton-Unruh temperature
    T_eff = M_E * C**2 / (2 * np.pi * K_B)  # in Kelvin
    kT_eff_over_mec2 = K_B * T_eff / (M_E * C**2)  # = 1/(2π) by construction

    print(f"""
  NWT's dimensionless predictions involving the unit-conversion
  constants:

  [α]  Fine-structure constant (Paper 8a):
       α = e²/(4πε_0 ℏc) = 1/(25π√3 + 1) = {1/(25*np.pi*np.sqrt(3) + 1):.6e}
       CODATA α                          = {ALPHA:.6e}
       Agreement                          = {(1/(25*np.pi*np.sqrt(3)+1) - ALPHA)/ALPHA*100:+.4f}%

  [m_e/m_Pl]  Electron-Planck mass ratio (Paper 17):
       m_e/m_Pl predicted = (8/7)(1+α/7+3α²) α^(21/2) = {(8/7)*(1+ALPHA/7+3*ALPHA**2)*ALPHA**(21/2):.6e}
       CODATA m_e/m_Pl   = {m_e_over_m_Pl:.6e}
       Agreement          = {((8/7)*(1+ALPHA/7+3*ALPHA**2)*ALPHA**(21/2) - m_e_over_m_Pl)/m_e_over_m_Pl*100:+.4f}%

  [Bekenstein]  Compton sphere maximum information content:
       S_max = 2π nats = {2*np.pi:.4f} nats  =  {2*np.pi/np.log(2):.4f} bits
       (involves ℏ and c implicitly via R = ƛ_C and E = m_e c²)

  [Margolus-Levitin]  Compton-period info saturation:
       4/ln(2) = {4/np.log(2):.4f} bits per Compton period
       (involves ℏ and c via τ_min = πℏ/(2 m_e c²))

  [Compton-Unruh]  Effective temperature ratio:
       k_B T_eff / (m_e c²) = 1/(2π) = {1/(2*np.pi):.6f}
       (involves ℏ, c, k_B implicitly)

  [G in bits]  Information content of the gravitational hierarchy:
       log_2(m_e/m_Pl) = {np.log2(m_e_over_m_Pl):.2f} bits
       log_2(G m_e²/(ℏc)) = 2 × log_2(m_e/m_Pl) = {2*np.log2(m_e_over_m_Pl):.1f} bits

  All of these are DIMENSIONLESS NUMBERS that NWT predicts (or
  would predict, given the QEC reframe of Paper 17).  The
  unit-conversion constants ℏ, c, k_B, e, G enter the SI numerical
  values of dimensional quantities, but NWT's predictions are
  dimensionless and unit-independent.
""")


# ====================================================================
# Section 5: NWT input reduction
# ====================================================================

def section5_input_reduction() -> None:
    section("Section 5: How many independent dimensional inputs does NWT need?")

    print(r"""
  Tracing through NWT's structure:

  1. NWT-internal scales (all dimensional):
       v_EW   (electroweak VEV, ~246 GeV)
       ρ_0    (BPS condensate density at minimum)
       ξ      (healing length, ~ℏ/(m_H c))
       m_e    (electron mass)
       m_W    (W boson mass)
       m_H    (Higgs mass)

  2. Relations among them (NWT-derived):
       ρ_0 = v²/2                      (Higgs minimum)
       ξ = ℏc / m_H                    (healing length)
       m_W = g v_EW / 2                (W mass)
       m_H = √(2λ) v_EW                (Higgs mass)
       m_e = y_e v_EW / √2             (electron Yukawa)

       Plus dimensionless NWT predictions:
         g = √(4πα) × (some factor)    (gauge coupling, NWT-derived)
         λ = (NWT-derived from BPS)
         y_e = (NWT-derived from topology, gives m_e/m_W)

  3. Unit-conversion constants:
       ℏ, c, k_B, e   (post-2019 SI: exact-by-definition)

  4. So the MINIMAL NWT INPUT SET is:
       - Dimensionless inputs:  topological data (trefoil, b2.13)
       - Dimensional input:     ONE absolute mass scale (v_EW or m_e)
       - Unit conventions:      kilogram, second, kelvin, ampere

  Everything else is derivable.  Specifically:
    - All particle masses (24-particle spectrum, Paper 6)
    - All gauge couplings (Paper 8a, Paper 13)
    - All mixing angles (CKM, PMNS — Paper 13)
    - α                (Paper 8a)
    - m_e/m_Pl → G     (Paper 17)
    - ℏ, c, k_B, e values in SI (unit conventions)

  The reduction relative to the SM is dramatic:  SM has ~30 free
  parameters; NWT (with the QEC reframe) has 1 dimensional input
  plus topological data.  The cost is the topological data is
  STRUCTURALLY rich (b2.13 bijection, Spin(7) representation theory,
  octonion algebra, K_7 graph state moments), but it's all derived
  from the trefoil.

  Jim's question reframed:
  Can c be eliminated from the dimensional input set?  In the
  Lorentz-invariant view (current NWT papers):  no, c is built into
  the metric structure as a unit-conversion factor.  In a Volovik-
  style emergent view:  yes, c² = (∂P/∂ρ)|_{ρ_0} would derive c from
  v_EW.  This is a NEW research direction.
""")


# ====================================================================
# Section 6: A concrete sketch of the Volovik-style derivation
# ====================================================================

def section6_volovik_sketch() -> None:
    section("Section 6: Sketch of a Volovik-style derivation of c in NWT")

    print(r"""
  If we take the emergent-c view seriously, here's the calculation
  outline.  Starting point:  NWT condensate as a non-relativistic
  superfluid with field ψ.

  Step 1.  Non-relativistic condensate Lagrangian:
      L_NR = iℏ ψ* ∂_t ψ - (ℏ²/2m) |∇ψ|² - V(|ψ|²)
             - (λ_skyrme) [non-relativistic Skyrme term]
             ...

  Step 2.  At BPS minimum, |ψ_0|² = ρ_0 = v²/2.
             Pressure  P = -∂E/∂V_volume |_S
             Density   ρ = |ψ|² × m   (mass density)

  Step 3.  Bogoliubov dispersion of small perturbations δψ:
             ω(k)² = (c_s² k²) + (ℏ²k⁴/(4m²))
             with  c_s² = μ/m
             where μ = ∂E/∂N is the chemical potential.

  Step 4.  At BPS minimum, μ = (∂V/∂|ψ|²)|_{|ψ_0|²}.
             For Mexican-hat V:  μ = -μ² + 2λ|ψ_0|² = μ² (with conventions)
             So  c_s² = μ²/m = (Higgs mass scale)²/m

  Step 5.  Identification:  c_s = c (in the relativistic limit).
             This requires  c² = μ²/m × (NWT factor).

  Step 6.  Express in NWT-native variables:
             For the Higgs sector,  m = m_H/c²  (relativistic conversion)
             so  c² = (μ²/m_H) × c²
             which gives  μ² = m_H/c² × c² = m_H ... circular?

  This is where the LORENTZ INVARIANCE of NWT's Lagrangian closes
  the loop:  the Bogoliubov dispersion in a Lorentz-invariant theory
  reduces to ω² = c²k² + m_H² at all k, with c built in.

  To EXIT this loop and derive c structurally, one would need to:
    (a) Postulate a non-relativistic underlying Lagrangian for the
        condensate, with c emerging only at long wavelengths.
    (b) Compute c_s² in this underlying theory in terms of v_EW.
    (c) Verify Lorentz invariance EMERGES at long wavelengths.

  This is the Volovik programme:  Lorentz invariance is emergent,
  not fundamental.  Implementing it in NWT would be a substantial
  research programme — likely a Paper 19 or later.

  HONEST CONCLUSION:
  In NWT as currently formulated (Papers 5-17), c is built in by
  the Lorentz-invariant Lagrangian and CANNOT be derived from
  condensate parameters alone.  The c value in SI is set by unit
  conventions (post-2019 SI: c = 299,792,458 m/s exact).

  Jim's intuition is correct that NWT's INPUT SET can be reduced
  to one absolute scale (v_EW or m_e) plus topological data, with
  ℏ and c entering only as unit-conversion factors.  The deeper
  emergent-c claim is a fascinating Volovik-style hypothesis but
  requires extending NWT below its current Lorentz-invariant
  foundation.

  PUBLISHABLE DIRECTION:
  A 'Paper 19: Emergent c from the NWT BPS Condensate' would:
    1. Posit an underlying non-relativistic Lagrangian for the
       condensate (e.g., Gross-Pitaevskii with NWT-specific corrections).
    2. Compute Bogoliubov dispersion and extract c_s.
    3. Show that Lorentz invariance emerges at long wavelengths.
    4. Express c_s² as a specific function of v_EW (or ρ_0) with a
       NWT-topological coefficient.
    5. Predict (testable in principle) that c has a specific value
       set by NWT's BPS structure.

  This would complete the chain:
    α       (Paper 8a)
    m_X/m_e (Paper 6)
    G       (Paper 17)
    c       (Paper 19, hypothetical)

  with ℏ remaining as the only conventional unit (and ℏ is exact
  by SI definition).

  Result:  ALL physics from one absolute scale + topological data.
""")


def main() -> None:
    print("=" * 78)
    print(" Can NWT derive c from internal parameters? — investigation")
    print("=" * 78)

    section1_natural_units()
    section2_lorentz_invariant_c()
    section3_emergent_c()
    section4_dimensionless_predictions()
    section5_input_reduction()
    section6_volovik_sketch()

    section("Summary")
    print("""
  Two views of NWT's c:

  (1) Lorentz-invariant (current NWT papers):  c is built into the
      relativistic Lagrangian as a unit-conversion factor.  NOT
      derivable from condensate parameters.

  (2) Volovik-style emergent (hypothetical extension):  c emerges
      from c_s² = (∂P/∂ρ)|_{ρ_0} at long wavelengths in an underlying
      non-relativistic condensate.  IS derivable from v_EW (or ρ_0).

  Within view (1), NWT's INPUT SET reduces to:
    - Topological data (trefoil, b2.13, etc.)
    - One absolute mass scale (v_EW or m_e)
    - Unit conventions (post-2019 SI definitions)

  Everything else — including SI numerical values of ℏ, c, k_B, e,
  G — follows.  This is the practical reduction.

  Within view (2), c becomes a NWT-derived quantity, reducing the
  dimensional input set further.  This is a substantive research
  programme (Paper 19) that would extend NWT below its current
  Lorentz-invariant foundation.

  Jim's intuition is structurally right:  the EW VEV (v_EW) and the
  condensate density (ρ_0) are not independent (ρ_0 = v_EW²/2 from
  Higgs potential).  In view (2), c would join the list of
  parameters set by v_EW.  In view (1), c is a unit, not a parameter.

  The current Paper 17 stays in view (1).  A future paper could
  explore view (2) and derive c structurally.
""")


if __name__ == "__main__":
    main()
