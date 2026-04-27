#!/usr/bin/env python3
"""
Proportionality constant for H_phys = κ · H_YY from a single
information-theoretic postulate.

Phase 8e established: |K_7⟩ has 21 bits of pairwise mutual information,
distributed exactly 1 bit per K_7 edge.  This script derives the
proportionality constant κ in H_phys = κ · H_YY from a single
postulate connecting bits of information to units of dimensionless
phase.

THE BIT-QUANTUM POSTULATE:

  Each K_7 edge contributes
    - 1 bit of pairwise mutual information to |K_7⟩       (Phase 8e)
    - 1 entangling CZ-gate to the graph-state encoding     (Phase 8c)
    - 1 so(7) generator via b2.13 bijection                (Paper 16)
    - 1 unit of dimensionless H_YY eigenvalue (= Y_u Y_v)  (Route A)

  POSTULATE: these four "1-per-edge" structures coincide in a single
  information-action quantum: each edge represents 1 bit of
  information processing AND 1 unit of dimensionless action / phase.

This is the QEC analog of "Schrödinger phase = action/ℏ", with the
action quantized in units of bits (= 1/ln 2 nats, via Landauer's
k_B T ln 2 energy per bit at thermal temperature T).

CONSEQUENCE:  Schrödinger emerges with no additional postulate beyond
this bit-quantum.  Specifically, traversing one K_7 Eulerian circuit
(= 21 edges) accumulates 21 dimensionless phase units.  Identifying
this with one full Compton-period rotation (2π) fixes:

  κ_natural × 21 = 2π   →   κ_natural = 2π / 21
  κ_physical = m_e c² / 21 = m_e c² / dim(Adj)

The Schrödinger equation for the rest electron then follows:

  i ℏ ∂_t |K_7⟩ = (m_e c² / 21) · H_YY · |K_7⟩
              = (m_e c² / 21) · 21 · |K_7⟩
              = m_e c² · |K_7⟩

with rotation rate ω_C = m_e c²/ℏ on |K_7⟩, the unique stable
attractor of continuous syndrome measurement (morning's Reading ii).

This is a CLEAN reduction of the Schrödinger derivation to a single
information-action quantum postulate, in place of the ad-hoc
"natural unit = Compton period" assumption of Phase 8a.
"""

from __future__ import annotations

from typing import List

import numpy as np


def section(title: str) -> None:
    print()
    print('=' * 78)
    print(' ' + title)
    print('=' * 78)


# =====================================================================
# 1. The bit-quantum hypothesis stated formally
# =====================================================================

def state_bit_quantum_postulate() -> None:
    section('(1) The Bit-Quantum Postulate')

    print(f"""
  POSTULATE (single):

     Each K_7 edge in the graph state |K_7⟩ corresponds to:
       (a) 1 entangling CZ gate (graph-state construction)
       (b) 1 so(7) gauge generator (b2.13 bijection)
       (c) 1 bit of pairwise mutual information (Phase 8e: I(u:v) = 1)
       (d) 1 unit of dimensionless H_YY eigenvalue (Y_u Y_v)
       (e) 1 unit of dimensionless action / phase

     All five "1-per-edge" structures are IDENTIFIED as a single
     information-action quantum.

  This single postulate replaces the Phase 8a postulate "natural
  unit = Compton period" with one that has a direct info-theoretic
  interpretation: action is quantized in units of bits, and the
  per-edge bit IS the per-edge phase quantum.

  CONNECTION TO LANDAUER'S PRINCIPLE:
     Landauer says erasing 1 bit at temperature T costs k_B T ln 2.
     For the rest electron at "self-temperature" T_eff = m_e c² / k_B,
     1 bit of erasure = m_e c² · ln 2 of energy.

     Per Compton period T_C = 2π ℏ / (m_e c²):
       Energy budget = m_e c² (the rest energy, persistent)
       Bits per T_C  = 21 (matter-line internal info content)
       Energy/bit/T_C = m_e c² / 21
       Compare Landauer per bit at T_eff = m_e c² · ln 2

       Ratio: (m_e c² / 21) / (m_e c² · ln 2) = 1 / (21 · ln 2) ≈ 0.0688

     The QEC framework is sub-Landauer by a factor of 1/(21 ln 2),
     which is the ratio of "K_7 graph efficiency" to "thermal Landauer
     baseline".  Equivalently, the K_7 code's parallelism (21 edges
     processed simultaneously per Compton tick) reduces the per-bit
     energy cost by a factor of 21 ln 2 ≈ 14.6 below thermal Landauer.

     This is consistent with QEC fault-tolerance: error correction
     codes can be more efficient than thermal erasure when the code
     has structural redundancy.
""")


# =====================================================================
# 2. Derive κ from the postulate
# =====================================================================

def derive_kappa() -> None:
    section('(2) Derivation of κ = m_e c² / dim(Adj)')

    print(f"""
  Given the bit-quantum postulate:

  Step 1: H_YY|K_7⟩ = 21 |K_7⟩  exactly (Phase 8e + Route A).
          Eigenvalue = dim(Adj) = number of K_7 edges = bit count.

  Step 2: Each edge contributes 1 dimensionless phase unit (postulate).
          Total phase per Eulerian circuit traversal = 21.

  Step 3: One Eulerian circuit IS one full coherent "revolution" of
          the matter-line state in the QEC code.  Identify this with
          one Compton period (= 2π phase rotation in standard QM).

  Step 4: Solve:
            κ_natural × 21 = 2π
            κ_natural = 2π / 21    (dimensionless)

  Step 5: Convert to physical units via Compton period:
            T_C = 2π / ω_C = 2π ℏ / (m_e c²)
            κ_physical = ω_C ℏ / 21 = m_e c² / dim(Adj)

  RESULT:  H_phys = (m_e c² / 21) · H_YY = (m_e c² / dim(Adj)) · H_YY.

  On |K_7⟩:  H_phys |K_7⟩ = (m_e c² / 21) · 21 · |K_7⟩ = m_e c² · |K_7⟩
            ⇒  i ℏ ∂_t |K_7⟩ = m_e c² · |K_7⟩
            ⇒  Standard rest-mass Schrödinger evolution.

  DERIVATION INPUTS:
    - m_e c² from BPS condition μ_BPS × L_trefoil (Paper 13)
    - dim(Adj) from Spin(7) Lie theory + b2.13
    - Bit-quantum postulate (this work, single new postulate)

  Result is sharper than Phase 8a's two-postulate setup.
""")


# =====================================================================
# 3. Compare to the action-quantization route (Phase 8b)
# =====================================================================

def compare_to_action_quantization() -> None:
    section('(3) Comparison with Phase 8b action-quantization route')

    print(f"""
  Phase 8b tried to derive the natural unit from BPS μ=π +
  Bohr-Sommerfeld action quantization S/ℏ = 2π·n.

  Heawood Eulerian winding gave n = |p|+|q| = 8 on T².  This
  predicted T_circuit = 8 T_C, conflicting with H_YY's eigenvalue
  21 unless one accepts a non-natural normalisation.  CLEAN NEGATIVE
  RESULT.

  Phase 8f (this work) replaces the action-quantization argument with
  the bit-quantum postulate:

    Action argument (8b):  S = 2πn ℏ where n = topological winding.
                            For Heegaard winding 8, gives 8 T_C per
                            circuit.  Doesn't match H_YY = 21.

    Bit-quantum (8f):      S = N · ℏ where N = bits processed per
                            circuit.  For 21 K_7 edges = 21 bits,
                            gives κ = m_e c² / 21 directly.

  These give DIFFERENT predictions:
    Action:    T_circuit = 8 T_C    (action-quantized winding)
    Bit-quant: T_circuit = T_C       (one circuit per Compton period)

  The bit-quantum reading is consistent with the morning's syndrome-
  attractor reading and Phase 8c's Cartan-graded structure.

  The action-quantization reading uses a different mathematical
  setup (Bohr-Sommerfeld phase-space loops) that doesn't apply
  cleanly to graph-walk dynamics.

  We endorse the bit-quantum reading as the cleaner foundation.
""")


# =====================================================================
# 4. Falsifiable consequences
# =====================================================================

def falsifiable_consequences() -> None:
    section('(4) Falsifiable consequences of the bit-quantum postulate')

    print(f"""
  If the bit-quantum postulate is correct, several specific
  consequences follow:

  (A) CROSS-GROUP RELATION (so(N) generalisation):
      For so(N) at level k with K_N as carrier:
        H_phys(so(N)) = m_e c² / dim(Adj_so(N)) · H_YY(K_N)
      where dim(Adj_so(N)) = N(N-1)/2 = |E(K_N)|.

      For so(9): κ = m_e c² / 36
      For so(11): κ = m_e c² / 55

      If the framework generalises to higher rank, the proportionality
      constant for the so(N)-Schrödinger equation should be
      m_e c² / dim(Adj_so(N)).

  (B) ENERGY LEVEL SPACING IN BOUND STATES:
      For |K_7⟩-encoded electron in an external potential, the
      energy spectrum should be:
         E_n = E_0 + n · (m_e c² / 21) · ΔH_YY_n
      where ΔH_YY_n is the H_YY shift between levels.

      Since H_YY has 4 distinct eigenvalues {{-3, +1, +9, +21}} on
      the code subspace, the level spacing in a confining potential
      should reflect this 4-tier structure with specific ratios.

  (C) DECOHERENCE TIMESCALES:
      Under continuous syndrome measurement at rate Γ ~ ω_C, the
      coherence time of the |K_7⟩-encoded electron should be:
        τ_coh ~ T_C × dim(Adj) / (max info leakage per edge per T_C)
              ~ 21 T_C  (using max 1 bit per edge per Compton period)
              ≈ 1.7 × 10^{{-19}} s

      This is the "intrinsic" decoherence time of the matter line
      at the QEC level, distinct from environmental decoherence.

  (D) RECIPROCITY WITH LANDAUER:
      Per-bit energy cost = m_e c² / dim(Adj) (this work)
      Compare Landauer at thermal-equiv temp T_eff = m_e c²/k_B:
         k_B T_eff ln 2 = m_e c² · ln 2 ≈ 0.347 · m_e c²
      Ratio = (1/dim(Adj)) / ln 2 = 1/(21 ln 2) ≈ 0.069

      The QEC framework processes each bit at ~7% of the thermal
      Landauer energy cost — a "QEC efficiency factor" of
      1/(dim(Adj) · ln 2).  Falsifiable claim about how matter-line
      QEC compares to thermal information processing.
""")


# =====================================================================
# 5. The status of the derivation
# =====================================================================

def derivation_status() -> None:
    section('(5) Status of the QEC Schrödinger derivation')

    print(f"""
  AT THIS POINT, the derivation chain looks like:

  EXTERNAL INPUTS (from prior NWT papers):
     - m_e c² from BPS (Paper 13)
     - K_7 graph as carrier (Paper 16 b2.13)
     - Spin(7) Lie data: dim(V) = 7, dim(Adj) = 21, dim(S) = 8

  STRUCTURAL POSTULATE (this work):
     - Each K_7 edge = 1 bit of info = 1 unit of dimensionless phase

  DERIVED:
     - H_YY closed form: (M² - dim(V))/2  (Phase 8c, Route A)
     - Bracket coefficients ⟨H_YY^n⟩ = dim(Adj)^n  (parallel session)
     - Schrödinger evolution rate κ = m_e c²/dim(Adj)  (this work)

  REMAINING OPEN:
     - Why is the bit-quantum the right action quantum?  Possibly
       follows from the Shannon-Margolis-Levitin information-energy
       bounds, but no rigorous derivation yet.
     - Continuum-spatial extension: standard Schrödinger acts on
       ℝ³-valued wavefunctions; |K_7⟩ is finite-dim.  Bridge via
       infinite K_7-like substructure or Heegaard-genus extension.
     - 8-dim spinor → 2-dim Pauli reduction: 6 of 8 components must
       gap out to recover standard electron spin.

  Compared to Phase 8a (postulate "natural unit = Compton period")
  this is sharper:  one info-theoretic postulate, three derived
  results, one cross-group prediction.

  Net status:  one structural postulate away from a full QEC
  derivation of Schrödinger for the rest electron in NWT framework.
""")


# =====================================================================
# 6. Numerical check: the predictions
# =====================================================================

def numerical_predictions() -> None:
    section('(6) Numerical predictions')

    H_BAR = 1.054571817e-34
    M_E = 9.1093837015e-31
    C = 299792458.0

    omega_C = M_E * C**2 / H_BAR
    T_C = 2 * np.pi / omega_C

    dim_Adj = 21
    kappa_phys = M_E * C**2 / dim_Adj

    print(f"""
  Numerical evaluation of the Phase 8f predictions:

    Compton frequency:   ω_C = {omega_C:.4e} rad/s
    Compton period:      T_C = {T_C:.4e} s
    m_e c²:              = {M_E * C**2 * 6.242e12:.6e} eV
                         = {M_E * C**2 / 1.602e-13:.4f} MeV

    Per-edge phase rate (= κ_phys):  m_e c² / 21 = {kappa_phys:.4e} J
                                                = {kappa_phys / 1.602e-13:.4f} MeV
                                                = {kappa_phys / 1.602e-19:.4e} eV
    Per-edge frequency:                          = {kappa_phys / H_BAR:.4e} rad/s

    Compton bits per Compton period: 21 (= dim(Adj))
    Compton bits per second:         {21 / T_C:.3e} bps

    Landauer energy per bit at T_eff = m_e c²/k_B:
       k_B T_eff ln 2 = m_e c² ln 2 = {M_E * C**2 * np.log(2) / 1.602e-13:.4f} MeV
       Ratio (κ_phys / Landauer): 1/(21 ln 2) = {1/(21*np.log(2)):.4f}

    The K_7 framework processes information at ~6.9% of the
    Landauer thermal cost per bit.

  Cross-group predictions (so(N) generalisation):
    so(7):  κ = m_e c² / 21 = {0.511/21*1000:.3f} keV
    so(9):  κ = m_e c² / 36 = {0.511/36*1000:.3f} keV
    so(11): κ = m_e c² / 55 = {0.511/55*1000:.3f} keV
""")


# =====================================================================
# 7. Main
# =====================================================================

def main() -> None:
    section('Phase 8f -- Proportionality constant κ from the bit-quantum postulate')

    print(f"""
The Phase 8e finding (|K_7⟩ has 21 bits of pairwise mutual info,
exactly 1 bit per edge) suggests a single information-theoretic
postulate that fixes the proportionality constant κ in
H_phys = κ · H_YY.

POSTULATE: each K_7 edge contributes 1 bit of mutual information
AND 1 unit of dimensionless phase / action.

CONSEQUENCE: κ = m_e c² / dim(Adj) = m_e c² / 21 (with m_e from BPS).

This is sharper than Phase 8a's "natural unit = Compton period"
postulate because it ties phase quantisation directly to bit
quantisation, producing falsifiable predictions about decoherence
times, level spacings, and cross-group prefactors.
""")

    state_bit_quantum_postulate()
    derive_kappa()
    compare_to_action_quantization()
    falsifiable_consequences()
    derivation_status()
    numerical_predictions()


if __name__ == '__main__':
    main()
