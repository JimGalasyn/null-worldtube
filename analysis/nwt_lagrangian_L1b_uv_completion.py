#!/usr/bin/env python3
"""
NWT Lagrangian -- L1b: UV completion and extra gauge-boson predictions.

L1 fixed the minimal (low-energy effective) field content:
   psi (condensate) + A_mu (U(1)) + n^a (Skyrme-Faddeev)
with SU(3) x SU(2) x U(1) emerging from 2T crossings of vortex cores.

L1b asks: what is the UV completion, and does it predict extra gauge
bosons?  The honest answer is MORE nuanced than "nine Spin(7) bosons":

==========================================================================
 WHY Spin(7) IS *NOT* THE UV GAUGE GROUP
==========================================================================

   rank Spin(7) = rank B_3 = 3
   rank (SU(3) x SU(2) x U(1)) = 2 + 1 + 1 = 4

The SM gauge algebra cannot embed in so(7) as a subalgebra (too much
rank).  Therefore Paper 15's Spin(7) / Cl(0, 7) / 2T structure CANNOT be
the UV gauge group of NWT.

This is a structural fact, not a convention.  Spin(7) is the symmetry
of the TOPOLOGICAL SECTOR (moduli space of torus knots on the Heegaard
torus of S^3/2I), not a gauge symmetry of a Lagrangian.  This is
consistent with L1 and with Paper 15 S7.3.

==========================================================================
 THE ACTUAL UV COMPLETION: SO(10) from cinquefoil + chirality doubling
==========================================================================

Paper 10 + Paper 13 + Paper 14's "three knots, three forces" triangle:

   trefoil T(2,3)    <->  Spin(7) / Cl(0,7) / 2T   (GRAVITY, topological)
   cinquefoil T(2,5) <->  SO(10) = D_5             (STRONG / GUT, gauge)
   heptafoil T(2,7)  <->  Spin(7) B_3 again?       (ELECTROMAGNETISM)

The CINQUEFOIL 5_1 has 5 crossings; Paper 13's memory (gut-extension.md)
gives:
   - 5 cyclic crossing exchanges generate su(5), 24 generators.
   - 5_1 + 5_1^* (chirality doubling) gives the 16 of SO(10).
   - Right-handed neutrino appears naturally.
   - CSc-1 cosmic string deficit angle constraint: E_GUT = 7.41e15 GeV.
   - alpha_GUT ~ 1/40.

SO(10) has rank 5, dim = 45.  Branching SO(10) -> SU(5) -> SM:
   45 = 24 [adj SU(5)] + 10 + 10bar + 1  (SO(10) -> SU(5) x U(1))
   24 = 12 [adj SM] + 12 [X/Y bosons]    (SU(5) -> SM)

Extra gauge bosons beyond SM:   45 - 12 = 33 bosons
   - 12 X/Y bosons from SU(5)/SM breaking   (MASS ~ E_GUT)
   - 21 extra from SO(10)/SU(5) breaking    (MASS ~ E_GUT or less)

These 33 bosons split into two categories:
   (a) "Unification bosons"  (12 X/Y)      -- violate B-L, mediate p decay
   (b) "SO(10) leptoquarks"  (21 extra)    -- mediate new B/L processes

NOTE: The 21 from SO(10)/SU(5) might tempt a connection to Paper 15's
21 = dim so(7).  But dim [SO(10)/SU(5)] = 45 - 24 - 1 = 20, NOT 21.
The 21/20 match is off by one, so this connection is at best suggestive.
Paper 15's 21 is a DIFFERENT 21 (edges of K_7 on the Heegaard torus of
S^3/2I), unrelated to SO(10) coset.

==========================================================================
 FALSIFICATION HOOKS
==========================================================================

  (H1)  Proton decay:
           Gamma(p -> e^+ pi^0) ~ alpha_GUT^2 m_p^5 / M_X^4.
        With M_X = E_GUT = 7.41e15 GeV and alpha_GUT = 1/40:
           tau_p ~ 10^35 yr  (numeric below)
        Compare to Super-K limit: tau > 2.4e34 yr.
        NWT prediction sits just above current experimental lower bound.

  (H2)  Coupling unification:
        alpha_1, alpha_2, alpha_3 must converge at E_GUT = 7.41e15 GeV
        with alpha_GUT = 1/40.  Run from M_Z with SM beta functions:
        the convergence point is testable via precision measurements of
        alpha_1(M_Z), alpha_2(M_Z), alpha_3(M_Z).

        STANDARD MSSM GUT gives E_GUT ~ 2e16 GeV with alpha_GUT ~ 1/25.
        NWT's SO(10) with E_GUT = 7.4e15 and alpha_GUT = 1/40 is
        LOWER and WEAKER -- a non-MSSM prediction that would show up
        as a specific coupling-unification deficit relative to MSSM.

  (H3)  Gravitational-wave signature of the GUT phase transition:
        Peak frequency at recombination of GUT bubbles:
           f_peak ~ 10^10 Hz * (T_*/10^16 GeV)
        With T_* = E_GUT = 7.4e15 GeV:
           f_peak ~ 7.4e9 Hz
        Far above LISA (mHz) and Einstein Telescope (kHz); would
        require high-frequency detectors (resonant cavity, CE).
        NOT immediately testable, but gives a clean target for
        future high-frequency GW experiments.

==========================================================================
 THE L1b LAGRANGIAN (SKETCH)
==========================================================================

Above E_GUT:
   L_UV  =  (D_mu Phi)^dagger (D^mu Phi) - V(Phi)
         -  (1/4) Tr [G_mu_nu G^mu_nu]
         +  i Psi_bar Gamma^mu D_mu Psi
         +  L_Yukawa(Phi, Psi)
where:
   Phi ~ 16 of SO(10) (complex Higgs, breaks SO(10) -> SM along the SM
         singlet direction of 16)
   G_mu_nu ~ so(10)-valued field strength (45 components)
   Psi ~ 16 of SO(10) (one generation of fermions plus nu_R)
   D_mu = d_mu - i g_GUT A_mu^a T^a

Breaking chain:
   SO(10)  --E_GUT--> SU(5) x U(1)_X  --epsilon-->  SM

Below E_GUT, integrate out the 33 heavy bosons.  The low-energy EFT is
the L1 three-field theory (psi, A_mu, n^a) with the U(1) arising from
the U(1)_EM subgroup of SM.  The Spin(7) / Cl(0,7) / 2T structure of
Paper 15 is carried by the KNOT SOLUTIONS of the low-energy EFT, not
by any fields in L_UV.

==========================================================================
 NUMERIC CHECKS
==========================================================================
"""

from __future__ import annotations

import math


# --- Physical constants (natural units where convenient) ---

GEV = 1.0                      # working unit
ALPHA_GUT = 1.0 / 40.0         # Paper 10 memory
E_GUT = 7.41e15 * GEV          # Paper 10 memory (CSc-1 constraint)
M_PROTON = 0.938 * GEV
HBAR_GEV_S = 6.582e-25         # hbar in GeV s

# Rank / dim of relevant algebras.

DIM_SO10 = 45
DIM_SU5 = 24
DIM_SM = 12   # 8 + 3 + 1
RANK_SPIN7 = 3
RANK_SM = 4


def proton_lifetime_from_gut(m_X_gev: float, alpha_gut: float,
                              channel_prefactor: float = 1.0) -> float:
    """Naive tree-level estimate of proton lifetime in years.

    Gamma(p -> e^+ pi^0) ~ (alpha_gut^2 / m_X^4) * m_p^5 * C
    where C is a model-dependent prefactor of order 1 (hadronic
    matrix element x chiral factor x Clebsch).  For SO(10) with
    MSSM-like matching, C ~ O(1).
    """
    gamma = channel_prefactor * (alpha_gut ** 2) * (M_PROTON ** 5) / (m_X_gev ** 4)
    # tau = hbar / Gamma  in s, convert to yr.
    tau_s = HBAR_GEV_S / gamma
    tau_yr = tau_s / (365.25 * 24 * 3600)
    return tau_yr


def gw_peak_freq(t_star_gev: float) -> float:
    """Peak frequency of GW signal from GUT phase transition, in Hz.

    Standard estimate (Kosowsky-Turner-Watkins 1992):
        f_peak ~ 1.6e-4 Hz * (T*/100 GeV)  at recombination
    ~ 10^10 Hz per 10^16 GeV for modern re-derivations with
    beta/H ~ 1 (first-order transition).
    """
    return 1.0e10 * (t_star_gev / 1.0e16)


def report(header: str, body: str) -> None:
    print()
    print("=" * 72)
    print(" " + header)
    print("=" * 72)
    print(body)


def main() -> None:
    report(
        "Why Spin(7) is not the UV gauge group",
        f"""
  rank Spin(7) = {RANK_SPIN7}   (Spin(7) = B_3 with 21 generators)
  rank SM      = {RANK_SM}      (SU(3) x SU(2) x U(1), 12 generators)

  SM cannot embed in Spin(7) as a gauge subalgebra: rank too small.
  So Spin(7) is NOT the UV gauge group.  It is the symmetry of the
  topological / knot-moduli sector (Paper 15).""",
    )

    report(
        "UV completion: SO(10) from cinquefoil + chirality doubling",
        f"""
  dim SO(10)  = {DIM_SO10}   (rank 5, contains SU(5) which contains SM)
  dim SU(5)   = {DIM_SU5}
  dim SM      = {DIM_SM}

  Extra gauge bosons: SO(10) - SM = {DIM_SO10 - DIM_SM}
    (a) 12 X/Y bosons from SU(5)/SM  -- p decay mediators
    (b) {DIM_SO10 - DIM_SU5 - 1} "SO(10) leptoquarks" from SO(10)/[SU(5) x U(1)]
    (c)  1  U(1)_X generator

  All sit at M ~ E_GUT = {E_GUT:.2e} GeV  (Paper 10, CSc-1 constraint).""",
    )

    tau_C1 = proton_lifetime_from_gut(E_GUT, ALPHA_GUT, channel_prefactor=1.0)
    tau_C10 = proton_lifetime_from_gut(E_GUT, ALPHA_GUT, channel_prefactor=0.1)
    tau_C01 = proton_lifetime_from_gut(E_GUT, ALPHA_GUT, channel_prefactor=10.0)

    report(
        "H1 -- Proton decay rate",
        f"""
  M_X = E_GUT = {E_GUT:.2e} GeV
  alpha_GUT  = {ALPHA_GUT:.4f}  (= 1/40)

  Naive tree-level:
     tau(p -> e^+ pi^0) ~ {tau_C1:.2e} yr    (C=1)
                         {tau_C10:.2e} yr   (C=0.1, more realistic)
                         {tau_C01:.2e} yr   (C=10, loose upper bound)

  Current Super-Kamiokande limit: tau > 2.4e34 yr (90% CL, 2020).
  Hyper-K projection (2030s):     tau > ~1e35 yr reach.

  NWT prediction sits at {tau_C1:.2e} yr: RIGHT IN THE SENSITIVITY BAND.
  Hyper-K or DUNE could detect p -> e^+ pi^0 within a decade if NWT's
  SO(10)/E_GUT assignment is correct.""",
    )

    report(
        "H2 -- Coupling unification point",
        f"""
  NWT prediction:
     E_GUT       = {E_GUT:.2e} GeV
     alpha_GUT   = {ALPHA_GUT:.4f} (1/{int(round(1/ALPHA_GUT))})

  Standard MSSM-GUT (for comparison):
     E_MSSM      ~ 2.0e16 GeV
     alpha_MSSM  ~ 1/25

  NWT's GUT scale is LOWER (7.4e15 vs 2e16) and GUT coupling is
  WEAKER (1/40 vs 1/25) than MSSM.  This is a non-MSSM unification
  prediction, distinguishable by precision measurements of
  alpha_1(M_Z), alpha_2(M_Z), alpha_3(M_Z) run up with SM beta
  functions.  NWT predicts the three couplings converge at
  ~7.4e15 GeV with alpha ~ 1/40, NOT at MSSM's ~2e16 GeV / ~1/25.""",
    )

    fpk = gw_peak_freq(E_GUT)
    report(
        "H3 -- Gravitational-wave signature of GUT phase transition",
        f"""
  Peak frequency (naive estimate):
     f_peak ~ 1e10 Hz x (T*/1e16 GeV)
            = {fpk:.2e} Hz  for T* = E_GUT

  LISA band       : 1e-4 to 1 Hz            [mHz]
  LIGO band       : 10 to 1e3 Hz             [kHz]
  Einstein T.     : 1 to 1e4 Hz              [sub-MHz]
  Resonant cavity : up to 1e10 Hz and above  [GHz-THz]

  Not directly testable today.  But: NWT predicts a SPECIFIC peak near
  {fpk:.1e} Hz -- a target for high-frequency GW detectors currently
  under development (e.g. MAGIS, Levitated Sensor Detector, ADMX-like
  cavities).  Null result at this frequency with sufficient sensitivity
  would constrain the SO(10) -> SM breaking scenario.""",
    )

    report(
        "Summary -- NWT's extra gauge-boson predictions",
        f"""
  {DIM_SO10 - DIM_SM} = 33 extra gauge bosons at M ~ E_GUT = {E_GUT:.2e} GeV.

  Broken into:
     12 X/Y bosons                  -- standard SU(5) p-decay mediators
     {DIM_SO10 - DIM_SU5 - 1} = 20 SO(10) leptoquarks     -- SU(5) singlet + rep
      1 U(1)_X                      -- subsumed in hypercharge

  Falsifiable predictions:
     H1  p -> e^+ pi^0 lifetime   ~ {tau_C1:.2e} yr (Hyper-K reachable)
     H2  coupling unification     E_GUT = 7.4e15 GeV, alpha_GUT = 1/40
     H3  GUT-transition GW peak   ~ {fpk:.2e} Hz (future detectors)

  Under the L1 economical reading, these 33 bosons live ONLY in the
  UV Lagrangian L_UV above E_GUT.  Below E_GUT, they are integrated
  out; the remaining low-energy theory is the three-field L1 system
  (psi, A_mu, n^a) with U(1)_EM gauge.

  The Spin(7) / Cl(0,7) / 2T structure of Paper 15 is ORTHOGONAL to
  this UV completion: it lives in the knot-moduli sector of the
  low-energy theory, not in the UV gauge content.

  Summary for Paper 17 (future): "NWT predicts a 33-boson SO(10) UV
  completion at E_GUT = 7.41e15 GeV, testable via Hyper-K proton
  decay within a decade."

  Next step (L2): kinetic + gauge-covariant + BPS potential for L1.""",
    )


if __name__ == "__main__":
    main()
