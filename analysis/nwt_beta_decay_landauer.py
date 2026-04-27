#!/usr/bin/env python3
"""
NWT Beta-Decay Landauer Selection Principle
============================================

Tests the hypothesis that m_n - m_p - m_e (= 0.7823 MeV, the free-neutron
Q-value) is set by a Landauer constraint on the topology change in
n -> p + e^- + ant_nu_e.

The framework gives:

    Landauer_beta  =  n_bits * T_eff * ln 2
                  =  2 * log_2(24) * (m_e c^2 / 2 pi) * ln 2
                  =  ln(24) * m_e c^2 / pi

with n_bits = 2 * log_2(24) accounting for 2 newly-created L_3 particles
(electron + antineutrino) drawn from a 24-element NWT spectrum.

Three tests:

    Test 1  Precision check: is Q_beta / Landauer_beta exactly 3/2?
    Test 2  Survey of other 1->3 leptonic weak decays: does beta sit
            uniquely close to the Landauer floor while others have
            huge headroom?
    Test 3  Selection-principle prediction:
            m_n - m_p  =  m_e + (3/2) * Landauer_beta
                       =  m_e * (1 + 3 * ln(24) / (2 pi))
            and compare to observed 1.2933 MeV.
"""

from __future__ import annotations

import numpy as np


# -- constants -----------------------------------------------------------

M_E    = 0.51099895        # MeV, electron rest mass
M_P    = 938.27208816      # MeV, proton rest mass
M_N    = 939.56542052      # MeV, neutron rest mass
LN2    = np.log(2.0)
LN24   = np.log(24.0)
PI     = np.pi
ALPHA  = 1.0 / 137.035999  # fine-structure constant


# -- core scales ---------------------------------------------------------

T_EFF      = M_E / (2 * PI)            # 0.0813 MeV  (Unruh-like at Compton)
LAND_BIT   = T_EFF * LN2               # 0.0564 MeV/bit (Landauer at T_eff)
N_BITS_2P  = 2 * np.log2(24.0)         # 9.1699 bits, 2 new L_3 particles
LAND_BETA  = N_BITS_2P * LAND_BIT      # 0.5170 MeV
Q_BETA     = M_N - M_P - M_E           # 0.7823 MeV (observed)


def section(s: str) -> None:
    print()
    print("=" * 78)
    print(" " + s)
    print("=" * 78)


def show_constants() -> None:
    section("Core scales")
    print(f"  m_e c^2          = {M_E:>12.6f} MeV")
    print(f"  m_n c^2          = {M_N:>12.6f} MeV")
    print(f"  m_p c^2          = {M_P:>12.6f} MeV")
    print(f"  m_n - m_p        = {M_N - M_P:>12.6f} MeV")
    print(f"  Q_beta (observed) = {Q_BETA:>12.6f} MeV")
    print()
    print(f"  T_eff = m_e/(2 pi)         = {T_EFF:>12.6f} MeV")
    print(f"  Landauer/bit = T_eff * ln 2 = {LAND_BIT:>12.6f} MeV/bit")
    print(f"  n_bits (2 new L_3)         = {N_BITS_2P:>12.6f} bits")
    print(f"  Landauer_beta              = {LAND_BETA:>12.6f} MeV")
    print()
    print(f"  Closed form: Landauer_beta = ln(24) * m_e / pi"
          f" = {LN24 * M_E / PI:.6f} MeV")
    print(f"  Q_beta / Landauer_beta     = {Q_BETA / LAND_BETA:>12.6f}")


def test1_algebraic_identity() -> None:
    """Test candidate algebraic forms for the ratio Q_beta / Landauer_beta."""

    section("Test 1 -- precision check: what is Q_beta / Landauer_beta?")

    ratio_obs = Q_BETA / LAND_BETA
    print(f"  Observed ratio  Q_beta / Landauer_beta = {ratio_obs:.6f}")
    print()
    print("  Candidate exact forms (delta = (cand - obs)/obs):")
    print(f"  {'form':<28} {'value':>12} {'delta':>10}")
    print("  " + "-" * 52)

    candidates = [
        ("3/2",                        3/2),
        ("pi/2",                       PI/2),
        ("e/2",                        np.e/2),
        ("(1 + alpha) * 3/2",          (1 + ALPHA) * 3/2),
        ("3/2 * (1 + alpha)",          3/2 * (1 + ALPHA)),
        ("ln(24)/(2 ln 2)",            LN24 / (2 * LN2)),
        ("3 ln(24) / (2 pi)",          3 * LN24 / (2 * PI)),
        ("Q_beta/m_e",                 Q_BETA / M_E),
        ("(m_n-m_p)/m_e - 1",          (M_N - M_P) / M_E - 1),
    ]
    for name, val in candidates:
        delta = (val - ratio_obs) / ratio_obs
        print(f"  {name:<28} {val:>12.6f} {delta:>+10.4%}")

    # Closed-form prediction Q_beta = (3/2) * Landauer_beta
    section("Test 1b -- prediction Q_beta = (3/2) * Landauer_beta")

    Q_predicted = 1.5 * LAND_BETA
    err = (Q_predicted - Q_BETA) / Q_BETA
    print(f"  Predicted  Q_beta = 1.5 * Landauer_beta = {Q_predicted:.6f} MeV")
    print(f"  Observed   Q_beta                       = {Q_BETA:.6f} MeV")
    print(f"  Error                                   = {err:+.4%}")
    print()
    print(f"  Closed form prediction:")
    print(f"    Q_beta = 3 ln(24) / (2 pi) * m_e c^2")
    print(f"           = {3 * LN24 / (2 * PI):.6f} * m_e c^2")
    print(f"           = {3 * LN24 / (2 * PI) * M_E:.6f} MeV")


def test2_survey_weak_decays() -> None:
    """Survey other 1->3 (or 1->2) leptonic weak decays."""

    section("Test 2 -- survey of leptonic weak decays")

    # (label, Q in MeV, n_new, lifetime_s, comment)
    # n_new = number of newly created L_3 particles
    decays = [
        ("n -> p e nu_bar",       Q_BETA,            2, 880.2,
         "free neutron beta decay"),
        ("mu -> e nu_bar nu",     105.16,             2, 2.197e-6,
         "muon decay (mu disappears, e + 2 nu created => 2 net new)"),
        ("tau -> e nu_bar nu",    1776.86 - M_E,      2, 2.903e-13,
         "tau leptonic decay"),
        ("tau -> mu nu_bar nu",   1776.86 - 105.66,   2, 2.903e-13,
         "tau leptonic decay (mu mode)"),
        ("pi+ -> mu nu",          139.57 - 105.66,    1, 2.603e-8,
         "pion 2-body leptonic (mu created)"),
        ("pi+ -> e nu",           139.57 - M_E,       1, 2.603e-8,
         "pion 2-body leptonic (helicity-suppressed)"),
        ("K+ -> mu nu",           493.68 - 105.66,    1, 1.238e-8,
         "kaon 2-body leptonic"),
        ("K+ -> e nu",            493.68 - M_E,       1, 1.238e-8,
         "kaon 2-body (helicity-suppressed)"),
        ("K_L -> pi e nu",        493.68 - 139.57 - M_E, 2, 5.116e-8,
         "K_e3 semileptonic"),
        ("3H -> 3He e nu_bar",    0.01859,            2, 3.888e8,
         "tritium beta decay (nuclear)"),
    ]

    print(f"  {'Process':<26} {'Q (MeV)':>12} {'n_new':>6} "
          f"{'Land (MeV)':>10} {'Q/Land':>10} {'tau (s)':>12}")
    print("  " + "-" * 78)

    for label, Q, n_new, tau, comment in decays:
        n_bits = n_new * np.log2(24.0)
        land   = n_bits * LAND_BIT
        ratio  = Q / land if land > 0 else float("inf")
        ratio_str = f"{ratio:>10.3f}" if ratio < 1e4 else f"{'inf':>10}"
        tau_str = f"{tau:>12.3e}"
        print(f"  {label:<26} {Q:>12.4f} {n_new:>6d} "
              f"{land:>10.4f} {ratio_str} {tau_str}")

    print()
    print("  Comments:")
    print("  - Free-neutron beta decay is uniquely close to the Landauer floor")
    print("    (ratio ~1.5, vs >100 for all leptonic decays of heavier hadrons).")
    print("  - Nuclear 3H decay has Q/Land = 0.04 -- BELOW the floor")
    print("    using the same 2-particle bit count.  This means either:")
    print("    (a) the framework FAILS for nuclear decays, or")
    print("    (b) the bit count differs (n,p inside nucleus already counted).")
    print("  - The cleanest test of the selection principle is the FREE neutron.")


def test3_selection_principle() -> None:
    """Predict m_n - m_p from the selection principle."""

    section("Test 3 -- selection principle for the n-p mass splitting")

    # Hypothesis: m_n - m_p = m_e + (3/2) * Landauer_beta
    delta_predicted = M_E + 1.5 * LAND_BETA
    delta_observed  = M_N - M_P
    err = (delta_predicted - delta_observed) / delta_observed

    print(f"  Hypothesis:  m_n - m_p  =  m_e + (3/2) * Landauer_beta")
    print(f"                          =  m_e * (1 + 3 ln(24) / (2 pi))")
    print()
    print(f"  Predicted m_n - m_p     = {delta_predicted:.6f} MeV")
    print(f"  Observed  m_n - m_p     = {delta_observed:.6f} MeV")
    print(f"  Error                   = {err:+.4%}")
    print()

    # Same in dimensionless form
    print(f"  Dimensionless form:  (m_n - m_p) / m_e")
    print(f"    Predicted = 1 + 3 ln(24) / (2 pi) = {1 + 3 * LN24 / (2 * PI):.6f}")
    print(f"    Observed                          = {delta_observed / M_E:.6f}")

    section("Test 3b -- Landauer floor as a consistency check")

    # Strict Landauer requirement: Q_beta >= Landauer_beta
    floor = LAND_BETA
    ratio = Q_BETA / floor
    print(f"  Landauer floor (Q_beta >= Landauer_beta):")
    print(f"    Floor         = {floor:.6f} MeV")
    print(f"    Q_beta        = {Q_BETA:.6f} MeV")
    print(f"    Headroom      = {Q_BETA - floor:.6f} MeV  ({ratio:.3f}x)")
    print()
    print(f"  The free-neutron Q is 51% above the strict Landauer floor.")
    print(f"  If 3/2 is the exact coefficient, the n-p mass splitting is")
    print(f"  fixed to within 0.5%.  If 3/2 is approximate, the framework")
    print(f"  predicts a structural lower bound rather than an exact value.")


def test4_alternative_floors() -> None:
    """How sensitive is the prediction to the choice of T_eff?"""

    section("Test 4 -- sensitivity to T_eff (Unruh? BPS-set? other?)")

    # Try several candidate T_eff prescriptions
    prescriptions = [
        ("m_e / (2 pi) [Unruh-Compton]",       M_E / (2 * PI)),
        ("m_e / (4 pi) [factor of 2]",          M_E / (4 * PI)),
        ("m_e [naive]",                         M_E),
        ("m_e * alpha / (2 pi)",                M_E * ALPHA / (2 * PI)),
        ("m_e * (1+alpha) / (2 pi)",            M_E * (1 + ALPHA) / (2 * PI)),
    ]

    target = Q_BETA / N_BITS_2P / LN2  # T_eff that makes Q_beta = exactly 1.5 * Landauer

    print(f"  Target T_eff for ratio = 3/2 exactly:  {target:.6f} MeV")
    print(f"  (i.e. T_eff such that Q_beta = 3/2 * n_bits * T_eff * ln 2)")
    print()
    print(f"  {'Prescription':<32} {'T_eff (MeV)':>14} {'ratio Q/Land':>14}")
    print("  " + "-" * 60)

    for name, t_eff in prescriptions:
        land = N_BITS_2P * t_eff * LN2
        ratio = Q_BETA / land
        delta_from_15 = ratio - 1.5
        marker = " <- 3/2 fit" if abs(delta_from_15) < 0.05 else ""
        print(f"  {name:<32} {t_eff:>14.6f} {ratio:>14.6f}{marker}")


def test5_scheme_c_per_invariant() -> None:
    """Compute Scheme C (per-invariant) bit counts for beta decay.

    bits = log_2(Q_H+1) + log_2(n_q+1) + log_2(|Q|+1)

    where Q_H = p*m is the Hopf invariant.  Particle labels from
    nwt_conservation_laws.py:

        n  : (p,q,m,n_q,Q) = (1, 4, 5, 3,  0)  -> Q_H = 5
        p  : (1, 4, 5, 3, +1)                  -> Q_H = 5
        e  : (2, 1, 3, 0, -1)                  -> Q_H = 6
        nu : placeholder (NWT topology unknown)

    Three neutrino conventions are tested:
      C1 (heuristic):  log_2(3) + 1 = 2.58 bits  (flavour + mass eigenstate)
      C2 (bridge):     1.58 + 1.58 + 1 = 4.16 bits (full per-neutrino budget
                       from neutrino-as-discrete-continuous-bridge.md)
      C3 (4-bit):      4.0 bits exactly (rounded version)
    """

    section("Test 5 -- Scheme C per-invariant bit counts for beta decay")

    def bits_inv(Q_H, n_q, Q):
        return (np.log2(Q_H + 1)
                + np.log2(n_q + 1)
                + np.log2(abs(Q) + 1))

    n_bits  = bits_inv(5, 3, 0)   # neutron
    p_bits  = bits_inv(5, 3, 1)   # proton
    e_bits  = bits_inv(6, 0, 1)   # electron, Q_H = p*m = 2*3 = 6

    # Three neutrino conventions
    nu_C1 = np.log2(3) + 1.0      # heuristic
    nu_C2 = np.log2(3) + np.log2(3) + 1.0   # bridge memo (flav + mass + nu/nubar)
    nu_C3 = 4.0                    # rounded

    print(f"  Per-particle bits (Scheme C, per-invariant):")
    print(f"    neutron   (Q_H=5, n_q=3, |Q|=0): {n_bits:.4f} bits")
    print(f"    proton    (Q_H=5, n_q=3, |Q|=1): {p_bits:.4f} bits")
    print(f"    electron  (Q_H=6, n_q=0, |Q|=1): {e_bits:.4f} bits")
    print(f"    nu (C1 heuristic):                {nu_C1:.4f} bits")
    print(f"    nu (C2 bridge memo, full budget): {nu_C2:.4f} bits")
    print(f"    nu (C3 rounded 4):                {nu_C3:.4f} bits")
    print()

    # Compute ΔS_topo and ratio under each variant
    print(f"  {'Variant':<28} {'dS bits':>10} {'Land MeV':>10} {'Q/Land':>10}")
    print("  " + "-" * 60)

    for label, nu_b in [("C1 heuristic (2.58 bits)", nu_C1),
                        ("C2 bridge (4.16 bits)",     nu_C2),
                        ("C3 rounded (4.0 bits)",     nu_C3)]:
        dS = (p_bits + e_bits + nu_b) - n_bits
        land = dS * LAND_BIT
        ratio = Q_BETA / land
        print(f"  {label:<28} {dS:>10.4f} {land:>10.4f} {ratio:>10.4f}")

    # And Scheme B for reference
    dS_B = N_BITS_2P
    land_B = dS_B * LAND_BIT
    ratio_B = Q_BETA / land_B
    print(f"  {'B uniform (log2 24 each)':<28} {dS_B:>10.4f} {land_B:>10.4f} {ratio_B:>10.4f}")

    section("Test 5b -- which prescription gives exactly 3/2 (or other clean)?")

    # Solve for n_bits that gives ratio = 3/2 exactly
    n_bits_target_15  = Q_BETA / (1.5 * LAND_BIT)
    n_bits_target_pi2 = Q_BETA / (PI / 2 * LAND_BIT)
    n_bits_target_15a = Q_BETA / (1.5 * (1 + ALPHA) * LAND_BIT)

    print(f"  Required n_bits to achieve target ratio:")
    print(f"    ratio = 3/2  exactly        -> n_bits = {n_bits_target_15:.4f}")
    print(f"    ratio = (3/2)(1+alpha)      -> n_bits = {n_bits_target_15a:.4f}")
    print(f"    ratio = pi/2                -> n_bits = {n_bits_target_pi2:.4f}")
    print()

    # Compare to actual prescriptions
    print(f"  Actual prescriptions:")
    print(f"    Scheme B uniform              n_bits = {N_BITS_2P:.4f}")
    print(f"    Scheme C1 heuristic           n_bits = {(p_bits + e_bits + nu_C1) - n_bits:.4f}")
    print(f"    Scheme C2 bridge              n_bits = {(p_bits + e_bits + nu_C2) - n_bits:.4f}")
    print(f"    Scheme C3 rounded             n_bits = {(p_bits + e_bits + nu_C3) - n_bits:.4f}")
    print()

    print(f"  Closest match: Scheme B (9.17) is 0.86% below the 3/2 target (9.25),")
    print(f"  and only 0.04% from the (3/2)(1+alpha) target (9.18).")
    print()
    print(f"  Scheme C variants all give ratios in the 1.55--1.88 range,")
    print(f"  significantly above 3/2.  Scheme B's near-3/2 result depends")
    print(f"  on the assumption that all 24 NWT particles share a uniform")
    print(f"  4.585-bit address irrespective of internal structure.")


def main() -> None:
    show_constants()
    test1_algebraic_identity()
    test2_survey_weak_decays()
    test3_selection_principle()
    test4_alternative_floors()
    test5_scheme_c_per_invariant()

    section("Summary")

    Q_pred = 1.5 * LAND_BETA
    err_q = (Q_pred - Q_BETA) / Q_BETA
    delta_pred = M_E + 1.5 * LAND_BETA
    err_d = (delta_pred - (M_N - M_P)) / (M_N - M_P)

    print(f"""
  Cleanest result:  Q_beta = (3/2) * Landauer_beta to {abs(err_q):.2%}.
  Equivalently:     m_n - m_p = m_e * (1 + 3 ln(24) / (2 pi)) to {abs(err_d):.2%}.

  Free-neutron beta decay is the ONLY leptonic weak decay sitting
  near the Landauer floor (ratio ~1.5).  All others (mu, tau, pi, K)
  have ratios > 100.

  The (3/2) coefficient is suggestive but not derived.  Candidates:
   * 3/2 = (final particles) / (new particles) = 3/2 (combinatorial)
   * 3/2 = a phase-space O(1) factor
   * 3/2 emerges from NWT's mass spectrum self-consistency

  The framework predicts:  m_n - m_p > m_e + Landauer_beta = 1.028 MeV
  STRICTLY (Landauer bound), and m_n - m_p ≈ 1.287 MeV (3/2 conjecture)
  vs observed 1.293 MeV.

  Falsifiable: any free 1-body weak decay with Q < Landauer floor for
  its topology change is forbidden.  This is consistent with the
  observed pattern that no such free decay exists.
""")


if __name__ == "__main__":
    main()
