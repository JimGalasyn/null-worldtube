#!/usr/bin/env python3
"""
Paper 19 -- V2: which condensate scale dominates Lambda_cc?

V1 computed Coleman-Weinberg V_eff from the BPS condensate's matter
content and showed that, treating v as a free parameter:

  v = 50 eV (NWT vortex scale, Paper 6)        : 31 orders too SMALL
  v = 246 GeV (electroweak Higgs VEV)          : 8 orders too BIG
  v ~ 4 GeV (~m_b, bottom quark)               : matches observation

V2 asks: what physical principle picks v?  Why does the matching
scale come out at a few GeV rather than at the natural-looking
electron Compton or electroweak scales?

==========================================================================
 THE PUZZLE
==========================================================================

NWT has potentially multiple condensate scales:
  * NWT vortex scale (Paper 6 BPS):       v_BPS ~ 50-90 eV
  * Standard SM Higgs VEV:                v_EW = 246 GeV
  * QCD chiral condensate:                f_pi ~ 100 MeV
  * Hadronic typical scale:               ~ GeV
  * Heaviest SM mass (top quark):         m_t ~ 173 GeV
  * GUT scale (Paper 14, L1b):            E_GUT ~ 7e15 GeV

In standard QFT, the COSMOLOGICAL CONSTANT is set by the
vacuum-energy zero-point of all matter fields, which scales as
the FOURTH POWER of the heaviest mass (since V_eff ~ Sigma m_i^4).

So in standard SM + GR: V_eff is dominated by the heaviest mass,
giving Lambda_cc ~ m_max^4, which is the famous problem.

In NWT, K_7 amplitude squared (Paper 17, Paper 18 G3) provides a
~10^-45 universal suppression of the gravitating vacuum energy
relative to its naive QFT value.  This brings naive Sakharov from
Lambda_UV^4 down to Lambda_UV^4 * K_7^2 ~ 10^-45 * 1 = 10^-45 of
M_Pl^4 -- still 77 orders too big.

V1 added Coleman-Weinberg's m^4 dimensional analysis: V_eff ~ v^4,
not Lambda_UV^4.  This brings the gap further down, BUT depends
sensitively on v.

V2 investigates which v is physically picked.

==========================================================================
 SCALE-BY-SCALE EVALUATION
==========================================================================

For each candidate scale v, we compute:
  Lambda_cc / M_Pl^4  =  (loop factor) * (v / M_Pl)^4 * K_7^2

and report the discrepancy from observed Lambda_cc / M_Pl^4 ~ 10^-122.
"""

from __future__ import annotations

import math


# ==========================================================================
# Inputs.
# ==========================================================================

ALPHA = 7.2973525693e-3
M_PL_GEV = 1.220890e19
LAMBDA_CC_OBSERVED_M_PL4 = 1e-122

# Loop factor from V1: V_eff = -0.0112 * m_H^4 = -0.0056 * v^4  (at BPS lambda = e^2/2)
# We work with absolute value.
LOOP_FACTOR = 0.0056

# K_7 amplitude squared (Paper 17, Paper 18 G3).
def K7_squared(alpha: float = ALPHA) -> float:
    bracket = (8.0 / 7.0) * (1.0 + alpha / 7.0)
    return bracket ** 2 * alpha ** 21


def lambda_cc_M_Pl4(v_GeV: float) -> float:
    """Lambda_cc / M_Pl^4 from CW + K_7 with given v."""
    V_eff = LOOP_FACTOR * v_GeV ** 4   # GeV^4
    Lambda_cc = V_eff * K7_squared()
    return Lambda_cc / M_PL_GEV ** 4


# ==========================================================================
# Candidate scales (in GeV).
# ==========================================================================

CANDIDATE_SCALES_GEV = [
    # NWT-internal scales:
    ('Hubble H_0',              3e-43),
    ('CMB temperature',         2.35e-13),
    ('Cosmological neutrino',   1e-12),
    ('Lightest active neutrino', 1e-11),    # ~10 meV
    ('Heaviest active neutrino', 5e-11),    # ~50 meV
    ('Electron mass',           5.11e-4),
    ('Muon mass',               0.106),
    ('Pion mass',               0.140),
    ('Kaon mass',               0.494),
    ('GeV (proton)',            0.938),
    ('Tau mass',                1.777),
    ('Charm quark',             1.27),
    ('J/psi',                   3.10),
    ('Bottom quark',            4.18),
    ('Upsilon Y(1S)',           9.46),
    ('W boson',                 80.4),
    ('Z boson',                 91.2),
    ('Higgs boson',             125.1),
    ('Top quark',               173.0),
    ('Electroweak VEV',         246.0),
    ('GUT scale (Paper 14)',    7.4e15),
    ('M_Pl',                    M_PL_GEV),
    # NWT-specific:
    ('NWT vortex scale (Paper 6)', 50e-9),
]


# ==========================================================================
# Find the magic v that matches observed Lambda_cc.
# ==========================================================================

def find_matching_v_GeV() -> float:
    """Solve  LOOP_FACTOR * v^4 * K_7_squared = LAMBDA_CC_OBSERVED * M_Pl^4
    for v.
    """
    target_V_eff = LAMBDA_CC_OBSERVED_M_PL4 * M_PL_GEV ** 4 / K7_squared()
    v_required = (target_V_eff / LOOP_FACTOR) ** 0.25
    return v_required


# ==========================================================================
# Reporting.
# ==========================================================================

def section(t: str) -> None:
    print()
    print('=' * 76)
    print(f' {t}')
    print('=' * 76)


def main() -> None:
    section('Paper 19 -- V2: which condensate scale dominates Lambda_cc?')

    # ---- Setup ----
    section('Setup')
    K7_sq = K7_squared()
    print(f"\n  Loop factor (from V1):        {LOOP_FACTOR}")
    print(f"  K_7 amplitude squared:        {K7_sq:.4e}")
    print(f"  Observed Lambda_cc / M_Pl^4: {LAMBDA_CC_OBSERVED_M_PL4:.0e}")

    # ---- Survey ----
    section('Step 1: Lambda_cc for candidate v scales')
    print(f"\n  {'scale':>30} {'v (GeV)':>14} "
          f"{'Lambda_cc / M_Pl^4':>22} {'discrepancy (orders)':>22}")
    print('  ' + '-' * 92)
    for name, v_GeV in CANDIDATE_SCALES_GEV:
        Lambda_cc = lambda_cc_M_Pl4(v_GeV)
        if Lambda_cc > 0:
            disc = math.log10(Lambda_cc / LAMBDA_CC_OBSERVED_M_PL4)
        else:
            disc = float('-inf')
        sign = '+' if disc > 0 else ''
        marker = ''
        if abs(disc) < 1.0:
            marker = '  <<< MATCH'
        elif abs(disc) < 3.0:
            marker = '  <-- close'
        print(f"  {name:>30} {v_GeV:>14.4e} {Lambda_cc:>22.4e} "
              f"{sign}{disc:>+12.2f}{marker}")

    # ---- Find the matching v ----
    section('Step 2: solve for the magic v')
    v_match = find_matching_v_GeV()
    print(f"\n  v that exactly matches observed Lambda_cc:")
    print(f"    v_match = {v_match:.4f} GeV  =  {v_match*1000:.1f} MeV")
    print(f"\n  Compare to nearby SM scales:")
    candidates_near = [
        ('charm quark m_c',    1.27),
        ('tau mass',           1.777),
        ('J/psi mass',         3.10),
        ('bottom quark m_b',   4.18),
        ('chi_b1(1P)',         9.89),
        ('Y(1S) Upsilon',      9.46),
    ]
    for name, m in candidates_near:
        ratio = v_match / m
        print(f"    {name:>20}  =  {m:.3f} GeV    "
              f"v_match / m = {ratio:.3f}")

    # ---- Discussion ----
    section('Step 3: interpretation')
    print(f"""
  The matching scale v_match ~ {v_match:.2f} GeV sits in the hadronic
  range -- specifically near (in order of nearness):

    * J/psi (charmonium 1S):   3.10 GeV   (ratio {v_match/3.10:.3f})
    * tau mass:                1.78 GeV   (ratio {v_match/1.78:.3f})
    * charm quark:             1.27 GeV   (ratio {v_match/1.27:.3f})
    * bottom quark:            4.18 GeV   (ratio {v_match/4.18:.3f})

  None of these is a perfect match.  The closest is roughly the
  charm/charmonium scale.

  Possible interpretations:

  (A) **Numerical coincidence**.  The matching v is set by a
      collection of dimensional factors (LOOP_FACTOR, K_7 amplitude,
      observed Lambda_cc).  There's no a priori reason to expect it
      to land on an SM scale, and the proximity to charmonium is
      probably ~30 % accidental.

  (B) **The physical condensate scale IS hadronic**.  If NWT's BPS
      condensate has its physical scale set by QCD-like hadronic
      dynamics (rather than electroweak or GUT), then v_match being
      hadronic is structurally significant.  This would mean the
      ψ-condensate has its IR-relevant scale where chiral symmetry
      is broken in QCD (~GeV), not at electroweak v_EW.

  (C) **CW dominance from heaviest mass is wrong**.  In the standard
      SM picture, V_eff is dominated by m_top.  But if NWT's matter
      content is more like the BPS-condensate fluctuations (Higgs +
      Higgsed gauge + Goldstones at the BPS critical coupling), the
      relevant scale is the BPS scale, NOT the SM top quark.  The
      v that picks out THIS scale would be model-dependent on which
      condensate is "the" L1 condensate.

  (D) **The bit-creation refinement changes the answer**.  If only
      virtual VORTEX events gravitate (not phonon zero-point modes),
      then V_eff isn't the right thing to compute.  Instead, we need
      the rate of virtual vortex creation in vacuum, which is set by
      the LIGHTEST vortex (neutrino).  See V4 / future work.

  V2 doesn't pick between (A)-(D); it just lays them out.  A real
  closure requires either:

    * deriving v from NWT's structural BPS calibration (Paper 6),
    * or showing that bit-creation refinement (D) gives the
      neutrino-mass-dominated answer with no further adjustments,
    * or accepting (A) and treating the agreement as a numerical
      ballpark check.

  V2 status: candidate condensate scales surveyed; v ~ 4 GeV
  (charm/bottom range) gives observed Lambda_cc with K_7 + CW;
  the physical principle picking this v is not yet identified.

==========================================================================
 OPEN: V3 -- thermodynamic vs zero-point distinction
==========================================================================

  Volovik's argument: vacuum energy that GRAVITATES is the
  thermodynamic ground state of the actual condensate, NOT the
  zero-point sum.  Coleman-Weinberg (V1) computed the zero-point
  sum at one loop.  These are two different objects.

  V3 should clarify which is the right thing for Lambda_cc.

==========================================================================
 OPEN: V4 -- bit-creation refinement
==========================================================================

  Per transduction-vortex-only.md, only vortex events make bits;
  phonon zero-point doesn't.  If true, then Lambda_cc should be
  computed as the rate of virtual VORTEX events, not as
  Coleman-Weinberg vacuum energy.

  The lightest vortex (neutrino) would dominate, giving
  Lambda_cc ~ m_nu^4.  With m_nu ~ 1 meV (cosmological neutrino
  background), Lambda_cc ~ (1 meV)^4 ~ observed value.

  V4 should make this calculation explicit and check whether it
  closes the residual gap.
""")


if __name__ == '__main__':
    main()
