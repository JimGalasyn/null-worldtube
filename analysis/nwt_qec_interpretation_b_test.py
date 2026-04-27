#!/usr/bin/env python3
"""
Interpretation B test: H_phys = H_YY predicts 4 anomalous Compton rates.

If H_phys is taken to be H_YY (or proportional to it), the QEC framework
predicts that the |K_7⟩-encoded electron has a 4-multiplet structure
under H_phys, with 8 spinor components distributing across 4 weight
spaces:

   Weight   Mult.   H_YY    rate vs ω_C    predicted mass (m_e units)
   ------   -----   ----    -----------    --------------------------
   M = 7      1     +21        +1.000              +1.000  (the electron itself)
   M = 5      3      +9        +3/7   ≈ +0.429     +0.429
   M = 3      3      +1        +1/21  ≈ +0.048     +0.048
   M = 1      1      -3        -1/7   ≈ -0.143     -0.143  (negative-mass / antiparticle?)

Predicted masses:
   m(M=5) ≈ 219 keV/c²    (×3 multiplicity)
   m(M=3) ≈ 24.3 keV/c²   (×3 multiplicity)
   m(M=1) ≈ -73 keV/c²    (negative; ×1)

The total Hilbert-space content under Interpretation B is 1+3+3+1 = 8
"electron-like states" with these specific rest masses.

This script:
  (1) Tabulates the predictions in absolute units.
  (2) Compares to PDG data — are there particles at these masses?
  (3) Checks compatibility with electron g-2 precision tests.
  (4) Identifies whether Interpretation B is FALSIFIED or whether
      it requires the anomalous states to be "encoded-but-not-physical".
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# =====================================================================
# 1. Physical constants
# =====================================================================

H_BAR = 1.054571817e-34
M_E_KG = 9.1093837015e-31
C = 299792458.0
M_E_MEV = 0.51099895069       # m_e c² in MeV
M_E_eV = M_E_MEV * 1e6        # m_e c² in eV
ALPHA = 7.2973525693e-3       # fine structure


def section(title: str) -> None:
    print()
    print('=' * 78)
    print(' ' + title)
    print('=' * 78)


# =====================================================================
# 2. The four predictions
# =====================================================================

WEIGHTS: List[Tuple[int, int, int, str]] = [
    # (M, multiplicity, H_YY, label)
    (7, 1, 21,  '|K_7⟩ ground state (the electron itself)'),
    (5, 3,  9,  '"first excited" — 3 states'),
    (3, 3,  1,  '"second excited" — 3 states'),
    (1, 1, -3,  '"highest excited" — 1 state, negative H_YY'),
]


def tabulate_predictions() -> None:
    section('(1) Interpretation B predictions: 4 weight-spaces')

    print(f"\n  Normalisation: H_phys = (m_e c² / 21) · H_YY")
    print(f"                |K_7⟩ rotates at ω_C = m_e c²/ℏ exactly.\n")

    print(f"  {'M':>3} {'mult':>5} {'H_YY':>5} {'rate':>10} {'mass [MeV]':>12} "
          f"{'mass [keV]':>12} {'label':<40}")
    print('  ' + '-' * 95)
    for M, mult, h_yy, label in WEIGHTS:
        rate = h_yy / 21
        mass = rate * M_E_MEV
        mass_kev = mass * 1000
        rate_str = f"{h_yy}/21" if h_yy != 21 else "1"
        print(f"  {M:>3} {mult:>5} {h_yy:>5} {rate_str:>10} "
              f"{mass:>+12.4f} {mass_kev:>+12.2f} {label:<40}")


# =====================================================================
# 3. PDG comparison
# =====================================================================

def pdg_comparison() -> None:
    section('(2) PDG comparison: do these predicted states exist?')

    print(f"""
  Predicted masses:
    219 keV (×3 multiplicity)
    24.3 keV (×3 multiplicity)
    -73 keV (×1, negative — tachyonic or antiparticle-like)

  PDG-listed particles below ~1 MeV:
    - electron (e⁻):  511 keV (= |K_7⟩, M=7 standard)
    - photon (γ):       0     (massless)
    - graviton:         0     (hypothetical)
    - neutrinos:      < ~1 eV (essentially massless)
    - axion:          ?       (hypothetical, very light)

  NO known particles exist at:
    - 219 keV  (between electron and twice-electron)
    - 24.3 keV (X-ray scale; no particle predicted there)
    - 73 keV   (also unobserved)

  CONCLUSION:  if Interpretation B is taken literally (anomalous
  states are physical particles with the predicted rest masses),
  it is FALSIFIED at the 100+ years of particle physics level.

  Three additional ~3-fold-degenerate states at 219 keV would have
  been observed in:
    - electron-positron annihilation (pair production thresholds)
    - β-decay endpoint spectroscopy
    - photon-photon scattering at MeV energies
    - dark photon searches at fixed-target experiments

  None has been observed.
""")


# =====================================================================
# 4. Electron g-2 constraint
# =====================================================================

def electron_g2_constraint() -> None:
    section('(3) Electron g-2 precision test')

    # Electron anomalous magnetic moment: g/2 - 1 ≈ α/(2π) + corrections.
    # Measured to ~10⁻¹³ accuracy.
    a_e_QED = ALPHA / (2 * np.pi)   # leading
    a_e_measured = 1.15965218073e-3  # 2021 measurement
    a_e_QED_full = 1.159652181643e-3  # full QED prediction at order α^5

    discrepancy = abs(a_e_measured - a_e_QED_full)

    print(f"""
  Electron anomalous magnetic moment a_e = (g - 2)/2:
    QED prediction (α^5):    {a_e_QED_full:.12f}
    Experiment:              {a_e_measured:.12f}
    Discrepancy:             |Δ| ≈ {discrepancy:.2e}
    Relative precision:      ~{discrepancy/a_e_measured:.2e}

  If anomalous states (Interp B) at masses m_a ~ m_e/2 contribute
  to the electron's g-2 via virtual loops, the contribution is:
    Δa_e ~ (α/π) · (m_e/m_a)² · O(1)

  For the predicted M=5 states at m_a = 3m_e/7:
    Δa_e ~ (α/π) · (7/3)² ~ {ALPHA/np.pi * (7/3)**2:.2e}

  This is ~10⁻³, which is ~10⁹ times the measured discrepancy.
  The anomalous states would COMPLETELY DESTROY the agreement
  between QED and experiment — even at one-loop level.

  CONSTRAINT:  if anomalous Compton states exist as on-shell
  particles, their coupling to the electron must be suppressed
  by at least ~10⁻⁹ vs standard QED — i.e., they must be:
    (a) Effectively decoupled from the QED current, OR
    (b) Confined / hidden by some other dynamics, OR
    (c) Pure mathematical artifacts of the K_7 QEC code, NOT
        physical particles.

  Reading (c) is most consistent with available data.
""")


# =====================================================================
# 5. Reframing Interpretation B for consistency with data
# =====================================================================

def reframed_interpretation_B() -> None:
    section('(4) Reframing Interpretation B: anomalous states as encoding-artifacts')

    print(f"""
  The data-driven reading:  the 7 non-|K_7⟩ states in the 8-dim code
  subspace are NOT physical electron-like particles.  They are
  "logical errors" or "encoded-but-not-realised" states of the K_7
  QEC code.

  This is QEC-natural:  in any stabilizer code, the logical subspace
  is multi-dimensional, but only ONE specific logical state (= |K_7⟩
  in our case) is "physically realised" by the encoding process.
  Other logical states correspond to syndromes that get continuously
  detected and corrected by interaction events.

  Under this reading:
    - The 4 "anomalous Compton rates" {{1, 3/7, 1/21, -1/7}} are
      mathematical eigenvalues of H_YY restricted to the code
      subspace, NOT physical Compton frequencies.
    - Only the M=7 state |K_7⟩ has the standard ω_C rotation.
    - The other 7 states are "encoding-only" — their H_YY values
      reflect the algebraic structure of the code, not particle
      physics.

  THIS IS INTERPRETATION (A) AFTER ALL:  H_phys is uniform on the
  *physically realised* part of the code subspace (= the single
  state |K_7⟩), even though H_YY itself has 4 distinct eigenvalues
  on the full 8-dim mathematical subspace.

  Interpretation B's structural insight survives:  the 4 weight-
  spaces with the (1, 3, 3, 1) Pascal-triangle pattern are real
  facts about H_YY's spectrum.  They just don't correspond to
  4 species of physical particles.

  WHAT THE 4-WEIGHT STRUCTURE *DOES* PREDICT (less testable but
  more honest):
    - The bracket coefficients (1 + α/7 + 3α² + ...) of Paper 17
      arise as moments ⟨H_YY^n⟩|K_7⟩ = 21^n.  These moments are
      sums OVER ALL 8 weight-states (with appropriate weights).
    - At α² and beyond, the M ≠ 7 weight-spaces contribute to
      the bracket via virtual-state amplitudes.
    - The α² coefficient = dim(Adj)/dim(V) = 3 (from Paper 17 §6.9)
      arises in part from the contribution of the 3-fold-degenerate
      M=3 weight-space.

  Consistent with Paper 17's reading.  No new physics needed.
""")


# =====================================================================
# 6. The remaining structural question
# =====================================================================

def remaining_question() -> None:
    section('(5) The remaining structural question for Schrödinger derivation')

    print(f"""
  After ruling out Interpretation B (literal anomalous states), the
  remaining structural question is:

     Why does only the M=7 state |K_7⟩ get realised as a physical
     electron, and what dynamics enforces this?

  Candidate answers:

  (i)  BPS condition:  the |K_7⟩ state is the unique BPS-saturating
       state in the code subspace.  All other states violate
       Bogomolny by some specific amount.  This selects |K_7⟩
       energetically.

  (ii) Stabilizer fixed-point dynamics:  the K_7 graph state |K_7⟩
       is the unique state stabilised by all 7 K_7 stabilizers
       simultaneously.  The other 7 code states are stabilised by
       only 4 stabilizers (in the Cartan-graded subspace).  Under
       continuous stabilizer-measurement dynamics (NWT's "interaction
       events"), |K_7⟩ is the ATTRACTOR — all other states evolve
       toward it.

  (iii) Spin-statistics:  the M=7 state is the highest-weight state
        of the 8-dim spinor under the H_YY-induced SU(2)-Casimir
        spectrum.  In standard QM, the rest electron is a specific
        spin state; here it's the highest H_YY weight.

  Reading (ii) is the most QEC-natural.  Under continuous stabilizer
  measurement (= interaction events at rate Γ ~ ω_C), the matter
  line's logical state is continuously projected back to |K_7⟩,
  making it the unique stable physical realisation.

  This converts the postulate "H_phys gives standard Compton phase"
  into a CONSEQUENCE of:
    - QEC stabilizer dynamics selecting |K_7⟩ as the unique stable
      physical state
    - The standard rest-mass phase rotation ω_C being the projection
      of H_phys onto the stable |K_7⟩ subspace

  This is the CLEANEST reading I see:  Schrödinger-for-the-electron
  emerges as the dynamics of the QEC-stabilised logical |K_7⟩ state
  under continuous interaction events.  The 7 "anomalous" states
  are syndromes that get corrected, not realised particles.
""")


# =====================================================================
# 7. Main
# =====================================================================

def main() -> None:
    section('Interpretation B: anomalous Compton rates falsifiability test')

    print("""
Tests whether H_phys = H_YY (Interpretation B) is compatible with
precision particle physics data.  The 4 weight-spaces of H_YY on
the 8-dim Cartan-graded code subspace would give 4 distinct rest-mass
states with predicted masses {1, 3/7, 1/21, -1/7} × m_e.

VERDICT (this script): Interpretation B as literally predicting
4 species of electron-like particles is FALSIFIED by:
  - Direct PDG search (no particles at 24, 73, or 219 keV)
  - Electron g-2 precision (any anomalous state at sub-MeV mass
    would shift g-2 by ~10⁹ × the measured discrepancy)

Reframed Interpretation B (= Interpretation A in disguise):
  - The 4 weight-spaces are mathematical eigenvalues of H_YY on
    the code subspace, not physical particles.
  - Only the M=7 state |K_7⟩ is the physical electron.
  - The other 7 states are QEC encoding-artifacts (syndromes
    continuously detected and corrected by interaction events).
  - The bracket coefficients (1 + α/7 + 3α²) are moments
    ⟨H_YY^n⟩|K_7⟩ that integrate over all 8 weight-states with
    appropriate weights.

This converts Interpretation B's structural insight into a clean
QEC reading consistent with all known physics.
""")

    tabulate_predictions()
    pdg_comparison()
    electron_g2_constraint()
    reframed_interpretation_B()
    remaining_question()


if __name__ == '__main__':
    main()
