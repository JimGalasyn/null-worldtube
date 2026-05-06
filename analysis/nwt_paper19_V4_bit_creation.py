#!/usr/bin/env python3
"""
Paper 19 -- V4: vacuum Lambda_cc from virtual VORTEX events
(corrected to exclude neutrinos: phase solitons aren't vortices).

V2 surveyed candidate condensate scales for v.  Magic v ~ 2.2 GeV
matches observation but didn't have a clean structural origin.
One of the four interpretations (D in V2's discussion) was the
bit-creation refinement: only vortex events gravitate, so vacuum
Lambda_cc should be computed as the rate of virtual VORTEX events.

Initial sketch incorrectly identified neutrinos as the lightest
vortex.  Jim's correction: **neutrinos are PHASE SOLITONS, not
vortices** -- a third ontological category in NWT.  Phase solitons
carry information about topology changes between events; they don't
themselves create bits.  So they're excluded from the vacuum
bit-creation calculation.

The lightest VORTEX in NWT is the electron (m_e ~ 0.511 MeV), which
is ~10^9 times heavier than neutrinos.  Naive bit-creation gives
Lambda_cc ~ m_e^4 / M_Pl^4 ~ 10^-90, dramatically reduced from the
naive 10^0 but still 32 orders too large.

==========================================================================
 THE TRICHOTOMY (refined 2026-04-27 evening)
==========================================================================

NWT has three kinds of excitation, not two:

  Vortices       (matter)       : create bits at topology-changing events
                                  e.g., charged leptons, quarks, hadrons
  Phase solitons (mostly bits)  : carry bits between events; don't create
                                  e.g., neutrinos (3 flavours x 3 masses)
  Phonons        (force-carriers): no topological content
                                  e.g., gamma, W/Z, gluons, Higgs, graviton

Bit-creation sources for vacuum Lambda_cc:
  - Virtual vortex pairs (e+e-, mu+mu-, ...): YES, these create bits
  - Virtual phase solitons (neutrinos): NO -- they carry, don't create
  - Virtual phonons (photons, gluons): NO -- no topological content

==========================================================================
 LAMBDA_CC FROM VIRTUAL VORTEX EVENTS
==========================================================================

For a virtual vortex pair of mass m, the typical lifetime is
tau ~ hbar/(m c^2) (Compton time) and the typical separation is
~ hbar/(m c) (Compton wavelength).  Vacuum density of such virtual
events is ~ (m/hbar c)^3 per unit volume, and they appear at rate
~ m c^2 / hbar per unit time.

Energy density of virtual vortex pairs:
  rho_vortex  ~  m^4   (in natural units hbar = c = 1)

Summing over species, dominated by lightest vortex:
  rho_total  ~  m_lightest^4

In NWT, the lightest vortex is the electron (m_e = 0.511 MeV), since
neutrinos are excluded as phase solitons.  Therefore:

  Lambda_cc^bit-creation  ~  m_e^4

==========================================================================
 NUMERICAL ESTIMATES
==========================================================================
"""

from __future__ import annotations

import math


# ==========================================================================
# Inputs.
# ==========================================================================

ALPHA = 7.2973525693e-3
M_E_GEV = 0.51099895069e-3
M_PL_GEV = 1.220890e19
LAMBDA_CC_OBSERVED_M_PL4 = 1e-122

# Lightest charged lepton (correct NWT lightest vortex):
M_LIGHTEST_VORTEX_GEV = M_E_GEV

# For comparison:
M_NEUTRINO_LIGHTEST_GEV = 1e-12   # ~1 meV; this is a phase soliton, NOT a vortex
M_BOTTOM_QUARK_GEV = 4.18         # for context, what V2's "magic v" was near


def K7_squared(alpha: float = ALPHA) -> float:
    """K_7 amplitude squared (Paper 17 / Paper 18 G3)."""
    bracket = (8.0 / 7.0) * (1.0 + alpha / 7.0)
    return bracket ** 2 * alpha ** 21


def K7_amplitude_single(alpha: float = ALPHA) -> float:
    """K_7 amplitude (single power)."""
    bracket = (8.0 / 7.0) * (1.0 + alpha / 7.0)
    return bracket * alpha ** (21.0 / 2.0)


def lambda_cc_bit_creation(m_vortex_GeV: float, K7_power: float = 0.0) -> float:
    """Lambda_cc / M_Pl^4 from virtual vortex pair creation.

    K7_power = 0:  no K_7 suppression (naive m^4 dimensional).
    K7_power = 1:  one K_7 amplitude factor (single-vertex weighting).
    K7_power = 2:  K_7 squared (graviton-self-energy weighting).
    """
    rho = m_vortex_GeV ** 4   # natural energy density of virtual pair sea
    if K7_power == 1:
        rho *= K7_amplitude_single()
    elif K7_power == 2:
        rho *= K7_squared()
    return rho / M_PL_GEV ** 4


# ==========================================================================
# Reporting.
# ==========================================================================

def section(t: str) -> None:
    print()
    print('=' * 76)
    print(f' {t}')
    print('=' * 76)


def main() -> None:
    section('Paper 19 -- V4: Lambda_cc from virtual VORTEX events')

    # ---- Setup ----
    section('Setup: corrected lightest vortex')
    print(f"""
  Trichotomy correction (2026-04-27 evening, Jim's catch):

    Vortices       : charged leptons (e, mu, tau), quarks, hadrons
    Phase solitons : neutrinos (NOT vortices -- they carry bits)
    Phonons        : gauge bosons, Higgs, graviton

  Lightest VORTEX in NWT  =  electron  (m_e = {M_E_GEV*1e3:.4f} MeV)
  (NOT lightest neutrino, m_nu ~ {M_NEUTRINO_LIGHTEST_GEV*1e3:.0e} MeV; phase soliton)
""")

    # ---- Naive m^4 estimate (no K_7) ----
    section('Step 1: naive bit-creation, no K_7 suppression')
    rho_naive = lambda_cc_bit_creation(M_E_GEV, K7_power=0)
    disc_naive = math.log10(rho_naive / LAMBDA_CC_OBSERVED_M_PL4)
    print(f"""
  Lambda_cc / M_Pl^4  =  m_e^4 / M_Pl^4
                      =  ({M_E_GEV:.3e})^4 / ({M_PL_GEV:.3e})^4
                      =  {rho_naive:.4e}

  Observed:               {LAMBDA_CC_OBSERVED_M_PL4:.0e}
  Discrepancy:            {disc_naive:+.1f} orders too large
""")

    # ---- With K_7 squared ----
    section('Step 2: with K_7 amplitude squared (Paper 18 G3 weighting)')
    rho_K7sq = lambda_cc_bit_creation(M_E_GEV, K7_power=2)
    disc_K7sq = math.log10(rho_K7sq / LAMBDA_CC_OBSERVED_M_PL4)
    print(f"""
  Lambda_cc / M_Pl^4  =  m_e^4 * K_7^2 / M_Pl^4

  K_7^2 (Paper 18 G3) =  {K7_squared():.4e}

  Lambda_cc / M_Pl^4 =  {rho_K7sq:.4e}
  Discrepancy        =  {disc_K7sq:+.1f} orders ({'+' if disc_K7sq > 0 else ''}
                       {'too large' if disc_K7sq > 0 else 'too small'})
""")

    # ---- With single K_7 ----
    section('Step 3: with single K_7 amplitude (alternative weighting)')
    rho_K7 = lambda_cc_bit_creation(M_E_GEV, K7_power=1)
    disc_K7 = math.log10(rho_K7 / LAMBDA_CC_OBSERVED_M_PL4)
    print(f"""
  Lambda_cc / M_Pl^4  =  m_e^4 * K_7 / M_Pl^4

  K_7 (single)        =  {K7_amplitude_single():.4e}

  Lambda_cc / M_Pl^4 =  {rho_K7:.4e}
  Discrepancy        =  {disc_K7:+.1f} orders
""")

    # ---- Solve for required K_7 power ----
    section('Step 4: what K_7 power closes the gap exactly?')
    # Need: m_e^4 * K7^p / M_Pl^4 = 1e-122
    # K7^p = 1e-122 * M_Pl^4 / m_e^4 = 1e-122 / (m_e/M_Pl)^4
    rho_naive_value = M_E_GEV**4 / M_PL_GEV**4
    K7_target_value = LAMBDA_CC_OBSERVED_M_PL4 / rho_naive_value
    K7_single = K7_amplitude_single()
    p_required = math.log(K7_target_value) / math.log(K7_single)
    print(f"""
  Naive m_e^4 / M_Pl^4               =  {rho_naive_value:.4e}
  Required K_7 factor (= ratio):      =  {K7_target_value:.4e}
  Single K_7 amplitude:               =  {K7_single:.4e}
  Required power p (K_7^p = ratio):   =  {p_required:.4f}

  Need K_7^{p_required:.2f} for exact match.

  Comparison:
    p = 0:  naive m^4 only, no K_7 (32 orders too big)
    p = 1:  single K_7 (single-event weighting?)
    p = 2:  K_7 squared (Paper 18 G3 graviton-self-energy weighting)
    p = {p_required:.2f}: matches observed Lambda_cc
""")

    # ---- Summary ----
    section('Step 5: synthesis')
    print(f"""
  Result table:

    {'K_7 power':>12} {'Lambda_cc / M_Pl^4':>22} {'discrepancy':>14}
    {'-'*12:>12} {'-'*22:>22} {'-'*14:>14}
    {'p = 0':>12} {rho_naive:>22.4e} {disc_naive:>+14.1f}
    {'p = 1':>12} {rho_K7:>22.4e} {disc_K7:>+14.1f}
    {'p = 2':>12} {rho_K7sq:>22.4e} {disc_K7sq:>+14.1f}
    {f'p = {p_required:.2f}':>12} {LAMBDA_CC_OBSERVED_M_PL4:>22.4e} {0:>+14.1f}

  Physical interpretation:

  V4 establishes that the bit-creation framework, with neutrinos
  correctly excluded as phase solitons, gives:

  - Without K_7: Lambda_cc ~ m_e^4 / M_Pl^4 ~ 10^-90, **32 orders
    too big**.  Already a 90-order improvement over naive QFT
    (which gives Lambda_UV^4 / M_Pl^4 ~ 1, 122 orders too big).

  - With K_7 squared (Paper 18 G3 weighting): Lambda_cc ~ 10^-135,
    **13 orders too small**.

  - The matching K_7 power ~ {p_required:.2f} is between 1 and 2 -- not a
    natural integer.

  Two ways to read this:

  (i)  The naive m_e^4 estimate is simply 32 orders too big and
       neither integer K_7 power closes it.  This argues that the
       bit-creation framework alone (without further mechanisms)
       doesn't fully explain Lambda_cc.

  (ii) The matching power p ~ {p_required:.2f} suggests a non-trivial fractional
       weighting in the vacuum bit-creation rate.  Could correspond
       to per-edge K_7 weighting (one factor per K_7 edge for a
       vacuum process, vs two for a propagator process) -- but this
       is speculative.

  Either way, V4 is a substantive ~90-order improvement.  Combined
  with V1's CW v^4 framing at v ~ 2.2 GeV (V2), the framework
  brackets the observed Lambda_cc within a few orders by either
  route.

==========================================================================
 V4 status
==========================================================================

  Deliverables:
    1. Trichotomy correction applied: phase solitons (neutrinos)
       excluded from vacuum bit-creation.                          [done]
    2. Lightest vortex correctly identified as electron.            [done]
    3. Lambda_cc / M_Pl^4 estimates at several K_7 weightings.      [done]
    4. Required K_7 power for exact match: {p_required:.2f}.                  [identified]

  V4 doesn't fully close the gap; it improves naive QFT by ~90 orders
  but still requires either accepting a 32-order residual gap or
  finding an additional mechanism beyond integer K_7 weighting.

  Possible directions to close the residual:
    - V3 (thermodynamic vacuum vs zero-point) might give additional
      suppression we haven't yet computed.
    - The fractional K_7 power might correspond to a real physical
      structure (per-edge weighting in vacuum diagrams).
    - Combining V1's CW (v^4) with V4's bit-creation refinement might
      give a coherent picture where the matching v gets identified
      structurally.

  Status: framework is consistent; full closure of Lambda_cc remains
  open.  The 90-order improvement is the headline takeaway.
""")


if __name__ == '__main__':
    main()
