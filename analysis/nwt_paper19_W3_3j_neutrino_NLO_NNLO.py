#!/usr/bin/env python3
"""
Paper 19 -- W3.3 step J: NLO/NNLO corrections for the Spin(8) phase
soliton mass formula.

Paper 17 established for the electron:

    m_e  =  (8/7)  *  alpha^(21/2)  *  (1 + alpha/7)  *  (1 + 3 alpha^2)  *  m_Pl

with:
    8/7         =  dim(spinor) / dim(vector) for Spin(7)
    21          =  dim Adj(so(7)) = K_7 edges
    (1+alpha/7) =  NLO, per-K_7-vertex correction (Paper 17 §6.5)
    (1+3 alpha^2)= NNLO with 3 = dim(Adj)/dim(V) = rank(so(7))

For Spin(8) (K_8 phase soliton, hypothesis from W3.3-I):

    m_1  =  (8/8)  *  alpha^(28/2)  *  NLO_8  *  NNLO_8  *  m_Pl
         =  alpha^14  *  NLO_8 * NNLO_8 * m_Pl

The structural identity rank(so(N)) = dim(Adj)/dim(V) holds for B_n
(= so(2n+1)) but BREAKS for D_n (= so(2n)).  For so(8) = D_4:
    dim(Adj) / dim(V)  =  28 / 8  =  7/2  =  3.5
    rank(so(8))        =  4
These DIFFER -- a real ambiguity at the NNLO level.

We compute three candidate Spin(8) NLO/NNLO recipes:

    OPTION I -- Direct dim(V) extension:
        NLO_8  = (1 + alpha/8)
        NNLO_8 = (1 + (28/8) alpha^2) = (1 + 3.5 alpha^2)

    OPTION II -- K_7 retained (Dehn-surgery interpretation):
        NLO_8  = (1 + alpha/7)
        NNLO_8 = (1 + 3 alpha^2)
        (= same as electron; m_1/m_e = clean (7/8) alpha^(7/2))

    OPTION III -- Rank-based extension:
        NLO_8  = (1 + alpha/8)
        NNLO_8 = (1 + 4 alpha^2)   [4 = rank(so(8)) = D_4 rank]

For each, predict m_1, m_2, m_3, and the sum, and check against:
    - Cosmological bound:  sum m_nu  <  0.12 eV
    - Oscillation lower bound:  m_3  >=  sqrt(Delta m_31^2)  =  51 meV
    - KATRIN direct measurement: m_eff < 0.45 eV

Output -> analysis/output/W3_3j_neutrino_NLO/
  options_table.txt
  summary.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[0] / "output" / "W3_3j_neutrino_NLO"
OUT.mkdir(parents=True, exist_ok=True)


# Constants (CODATA-like)
ALPHA = 7.2973525693e-3            # fine structure constant
M_E_OBS_EV = 0.51099895e6          # observed electron mass (eV)
M_PL_EV = 1.220890e28              # Planck mass (eV)

DELTA_M21_SQ = 7.42e-5             # eV^2
DELTA_M32_SQ = 2.515e-3            # eV^2
DELTA_M31_SQ = DELTA_M21_SQ + DELTA_M32_SQ

COSMO_SUM_BOUND = 0.12             # eV
OSC_M3_LOWER = np.sqrt(DELTA_M31_SQ)  # eV ~ 0.0510


def m_e_paper17(NLO=True, NNLO=True):
    """Paper 17 electron mass formula.

    m_e = (8/7) alpha^(21/2) (1+alpha/7) (1+3 alpha^2) m_Pl
    """
    LO = (8.0 / 7.0) * ALPHA ** (21.0 / 2.0) * M_PL_EV
    if NLO:
        LO *= (1 + ALPHA / 7.0)
    if NNLO:
        LO *= (1 + 3 * ALPHA ** 2)
    return LO


def m_nu_spin8(option: str) -> float:
    """Spin(8) phase-soliton predictions under different NLO/NNLO recipes.

    LO is always (8/8) * alpha^(28/2) * m_Pl  =  alpha^14 m_Pl.
    """
    LO = (8.0 / 8.0) * ALPHA ** (28.0 / 2.0) * M_PL_EV
    if option == "I":
        # Direct dim(V) extension
        NLO = 1 + ALPHA / 8.0
        NNLO = 1 + (28.0 / 8.0) * ALPHA ** 2
    elif option == "II":
        # K_7-retained (Dehn surgery on K_7 substrate)
        NLO = 1 + ALPHA / 7.0
        NNLO = 1 + 3 * ALPHA ** 2
    elif option == "III":
        # Rank-based for D_4
        NLO = 1 + ALPHA / 8.0
        NNLO = 1 + 4 * ALPHA ** 2
    else:
        raise ValueError(f"unknown option: {option}")
    return LO * NLO * NNLO


def neutrino_spectrum(m_1: float) -> tuple[float, float, float, float]:
    """Three-generation spectrum from m_1 (lightest) and oscillation deltas."""
    m_2 = np.sqrt(m_1 ** 2 + DELTA_M21_SQ)
    m_3 = np.sqrt(m_1 ** 2 + DELTA_M31_SQ)
    sum_ = m_1 + m_2 + m_3
    return m_1, m_2, m_3, sum_


def main():
    print("=" * 70)
    print("W3.3-J  Spin(8) phase-soliton NLO/NNLO corrections")
    print("=" * 70)

    # Verify Paper 17 m_e formula
    print("\n--- PAPER 17 ELECTRON FORMULA VERIFICATION ---")
    m_e_LO = m_e_paper17(NLO=False, NNLO=False)
    m_e_NLO = m_e_paper17(NLO=True, NNLO=False)
    m_e_NNLO = m_e_paper17(NLO=True, NNLO=True)
    print(f"  Observed m_e:                   {M_E_OBS_EV * 1e-6:.6f} MeV")
    print(f"  LO  (8/7) * alpha^(21/2) * m_Pl: "
          f"{m_e_LO * 1e-6:.6f} MeV  "
          f"(err {(m_e_LO / M_E_OBS_EV - 1) * 100:+.3f}%)")
    print(f"  + NLO (1+alpha/7):              "
          f"{m_e_NLO * 1e-6:.6f} MeV  "
          f"(err {(m_e_NLO / M_E_OBS_EV - 1) * 100:+.3f}%)")
    print(f"  + NNLO (1+3 alpha^2):           "
          f"{m_e_NNLO * 1e-6:.6f} MeV  "
          f"(err {(m_e_NNLO / M_E_OBS_EV - 1) * 100:+.3f}%)")

    # Spin(8) options
    print("\n--- SPIN(8) NEUTRINO PREDICTIONS (THREE OPTIONS) ---")

    options_data = []
    for opt, desc in [("I", "Direct dim(V): (1+alpha/8)(1+(28/8)alpha^2)"),
                      ("II", "K_7-retained: (1+alpha/7)(1+3 alpha^2)"),
                      ("III", "Rank D_4:    (1+alpha/8)(1+4 alpha^2)")]:
        m_1 = m_nu_spin8(opt)
        m_1_, m_2_, m_3_, sum_ = neutrino_spectrum(m_1)
        cosmo_pass = sum_ < COSMO_SUM_BOUND
        osc_pass = m_3_ >= OSC_M3_LOWER
        options_data.append({
            "option": opt, "desc": desc,
            "m_1": m_1_, "m_2": m_2_, "m_3": m_3_, "sum": sum_,
            "cosmo_pass": cosmo_pass, "osc_pass": osc_pass,
        })
        print(f"\n  Option {opt}: {desc}")
        print(f"    m_1 = {m_1_ * 1e3:>8.4f} meV")
        print(f"    m_2 = {m_2_ * 1e3:>8.4f} meV  "
              f"(= sqrt(m_1^2 + Delta_21^2))")
        print(f"    m_3 = {m_3_ * 1e3:>8.4f} meV  "
              f"(= sqrt(m_1^2 + Delta_31^2))")
        print(f"    sum = {sum_ * 1e3:>8.4f} meV   "
              f"(cosmo bound 120 meV: {'PASS' if cosmo_pass else 'FAIL'})")
        print(f"    m_3 >= 51 meV osc bound: "
              f"{'PASS' if osc_pass else 'FAIL'}")

    # Comparison
    print("\n--- COMPARISON ---")
    print(f"\n  {'option':>8}  {'m_1 (meV)':>12}  {'m_3 (meV)':>12}  "
          f"{'sum (meV)':>12}  {'cosmo':>8}  {'osc':>6}")
    print("  " + "-" * 76)
    for r in options_data:
        print(f"  {r['option']:>8}  {r['m_1']*1e3:>12.4f}  "
              f"{r['m_3']*1e3:>12.4f}  {r['sum']*1e3:>12.4f}  "
              f"{'PASS' if r['cosmo_pass'] else 'FAIL':>8}  "
              f"{'PASS' if r['osc_pass'] else 'FAIL':>6}")

    print()
    print("  Differences between options (relative to Option I):")
    base = options_data[0]
    for r in options_data[1:]:
        rel_m1 = r["m_1"] / base["m_1"] - 1
        rel_sum = r["sum"] / base["sum"] - 1
        print(f"    Option {r['option']} vs I:  "
              f"m_1 differs by {rel_m1 * 100:+.4f}%, "
              f"sum differs by {rel_sum * 100:+.4f}%")

    print()
    print("  All three options give very similar predictions because")
    print("  NLO/NNLO are O(0.1%) corrections.  Distinguishing them")
    print("  requires precision m_nu measurements at sub-percent level,")
    print("  which is far beyond current experimental capability.")

    # Theoretical preference
    print("\n--- THEORETICAL ANALYSIS ---")
    print()
    print("  WHICH OPTION IS MOST DEFENSIBLE?")
    print()
    print("  Option I (NLO=1+alpha/8, NNLO=1+(28/8) alpha^2):")
    print("    Most direct extension of Paper 17's logic.  Uses K_8 vertex")
    print("    count for NLO (analogous to '7 = K_7 vertex count' there)")
    print("    and dim(Adj)/dim(V) = 28/8 = 7/2 for NNLO.")
    print("    PRO: structurally clean, mirrors Paper 17.")
    print("    CON: 7/2 isn't an integer; uneasy for so(8).")
    print()
    print("  Option II (NLO=1+alpha/7, NNLO=1+3 alpha^2 - SAME AS ELECTRON):")
    print("    Phase soliton lives ON the K_7 substrate (Dehn surgery on")
    print("    K_7 boundary), so NLO/NNLO inherit K_7 structure.  Only")
    print("    the LO power changes due to surgery topology.")
    print("    PRO: Clean structural identity m_1/m_e = (7/8) alpha^(7/2)")
    print("         after corrections cancel.  Connects to existing NWT")
    print("         machinery (K_7, PSL(2,7), trefoil-knot framework).")
    print("    CON: Requires explicit derivation of Dehn-surgery soliton")
    print("         from NWT Lagrangian.")
    print()
    print("  Option III (NLO=1+alpha/8, NNLO=1+4 alpha^2 - rank-based):")
    print("    Uses rank(so(8)) = 4 for NNLO, which is the canonical")
    print("    Casimir number for D_4 series.")
    print("    PRO: Uses the 'right' rank for D_4 series.")
    print("    CON: Doesn't match dim(Adj)/dim(V) for D_4 (= 7/2),")
    print("         which is a structural identity for B_n in Paper 17.")
    print()
    print("  THE D_4 STRUCTURAL AMBIGUITY:")
    print()
    print("  Paper 17's NNLO derivation used '3 = dim(Adj)/dim(V) = rank'")
    print("  for so(7).  These are equal only for B_n series (so(2n+1)).")
    print("  For D_4 (so(8)), they DIFFER (3.5 vs 4).  This means the")
    print("  PaperE17 derivation needs to be redone for D_4 to determine")
    print("  whether 7/2 (Option I) or 4 (Option III) is the correct")
    print("  extension.  Until that derivation is done, Option II")
    print("  (K_7-retained, no D_4 ambiguity) is the structurally")
    print("  cleanest choice.")
    print()
    print("  RECOMMENDATION: Option II (K_7-retained) until D_4 derivation")
    print("                  resolves the ambiguity.")

    # Falsifiability
    print("\n--- FALSIFIABILITY ---")
    print()
    print("  All three options give m_1 in 14.83-14.87 meV range.")
    print("  All three options give m_3 in 53.00-53.03 meV range.")
    print("  All three options give sum in 84.97-85.05 meV range.")
    print()
    print("  Distinguishing the three options requires sub-0.1% precision")
    print("  on m_nu, which corresponds to <15 microeV resolution.")
    print()
    print("  Current experiments:")
    print("    KATRIN: m_eff < 0.45 eV (2024)")
    print("    Cosmology + DESI: sum < 0.12 eV at 95% CL")
    print("    Future Project 8 (tritium beta): O(40 meV) sensitivity")
    print("    Future cosmology (CMB-S4 + DESI-II): sum at ~10 meV level")
    print()
    print("  Falsification thresholds:")
    print("    If m_1 is measured to be < 5 meV: ALL Spin(8) options falsified.")
    print("    If sum is found to be < 60 meV:    ALL Spin(8) options falsified.")
    print("    If oscillation hierarchy is INVERTED (m_3 < m_1):")
    print("                                       Spin(8) NORMAL hierarchy")
    print("                                       interpretation falsified.")

    # Summary
    summary = [
        "Paper 19 -- W3.3-J  Spin(8) NLO/NNLO corrections for neutrinos",
        "=" * 70,
        "",
        "BASELINE (Paper 17 electron, for reference):",
        f"  Observed m_e:  {M_E_OBS_EV * 1e-6:.6f} MeV",
        f"  LO  formula:   {m_e_LO * 1e-6:.6f} MeV  "
        f"(err {(m_e_LO / M_E_OBS_EV - 1) * 100:+.3f}%)",
        f"  + NLO:         {m_e_NLO * 1e-6:.6f} MeV  "
        f"(err {(m_e_NLO / M_E_OBS_EV - 1) * 100:+.3f}%)",
        f"  + NNLO:        {m_e_NNLO * 1e-6:.6f} MeV  "
        f"(err {(m_e_NNLO / M_E_OBS_EV - 1) * 100:+.3f}%)",
        "",
        "SPIN(8) NEUTRINO PREDICTIONS (m_1, lightest):",
        f"  {'option':>8}  {'m_1 (meV)':>12}  {'m_3 (meV)':>12}  "
        f"{'sum (meV)':>12}  {'cosmo':>8}  {'osc':>6}",
        "  " + "-" * 76,
    ]
    for r in options_data:
        summary.append(
            f"  {r['option']:>8}  {r['m_1']*1e3:>12.4f}  "
            f"{r['m_3']*1e3:>12.4f}  {r['sum']*1e3:>12.4f}  "
            f"{'PASS' if r['cosmo_pass'] else 'FAIL':>8}  "
            f"{'PASS' if r['osc_pass'] else 'FAIL':>6}"
        )
    summary += [
        "",
        "OPTIONS DESCRIBED:",
        "  I:   NLO = (1 + alpha/8),  NNLO = (1 + 7/2 alpha^2)",
        "       (direct K_8/Spin(8) extension of Paper 17 logic)",
        "  II:  NLO = (1 + alpha/7),  NNLO = (1 + 3 alpha^2)",
        "       (K_7-retained: Dehn surgery on K_7 substrate)",
        "  III: NLO = (1 + alpha/8),  NNLO = (1 + 4 alpha^2)",
        "       (rank D_4 = 4 NNLO coefficient)",
        "",
        "STRUCTURAL FINDING:",
        "  The Paper 17 identity '3 = dim(Adj)/dim(V) = rank' is specific",
        "  to the B_n series (so(2n+1)).  For so(8) = D_4, dim(Adj)/dim(V)",
        "  = 7/2 differs from rank = 4.  This ambiguity at the NNLO level",
        "  is a real structural feature of D_4 vs B_n.",
        "",
        "  Option II (K_7-retained) sidesteps the ambiguity by inheriting",
        "  the same NLO/NNLO factors as the electron, on the grounds that",
        "  the phase soliton lives on the K_7 substrate (Dehn surgery on",
        "  the K_7 boundary).  This gives a CLEAN structural identity:",
        "",
        "    m_1 / m_e  =  (7/8) * alpha^(7/2)",
        "                =  (7/8) / 137^3.5  =  2.905e-8",
        "",
        "  After NLO/NNLO factors cancel.  Numerologically this matches",
        "  the K_7 -> K_8 'add 7 edges' step:",
        "    +7 edges  ->  alpha^(7/2) suppression",
        "    Spin(8) triality -> (8/8 = 1 vs 8/7 for Spin(7)) prefactor",
        "    inverted ratio of vertices: 7/8.",
        "",
        "FALSIFIABILITY:",
        "  All three options predict m_1 in 14.83-14.87 meV (resolution",
        "  ~0.04 meV).  Distinguishing them requires sub-microeV neutrino-",
        "  mass measurement, beyond foreseeable experimental capability.",
        "",
        "  At current experimental precision, the THREE OPTIONS ARE",
        "  INDISTINGUISHABLE.  Future experiments can falsify the entire",
        "  Spin(8) hypothesis class if they find:",
        "    - m_1 < 5 meV (would imply m_1 << m_3, normal hierarchy",
        "      with extreme suppression -- inconsistent with Spin(8))",
        "    - sum m_nu < 60 meV (cosmologically tight)",
        "    - inverted hierarchy m_1 > m_3",
        "",
        "RECOMMENDATION:",
        "  Use Option II (K_7-retained NLO/NNLO) until a D_4-specific",
        "  derivation resolves the (7/2) vs (4) NNLO ambiguity.  Option",
        "  II has the cleanest structural form:",
        "",
        "    m_e = (8/7) alpha^(21/2) m_Pl * (1+alpha/7) (1+3 alpha^2)",
        "    m_1 = (8/8) alpha^(28/2) m_Pl * (1+alpha/7) (1+3 alpha^2)",
        "",
        "  Ratio m_1/m_e = (7/8) alpha^(7/2) is structurally clean.",
        "",
        "NEXT STEPS:",
        "  1. Derive D_4 NNLO coefficient from first principles (analog",
        "     of Paper 17 §6.9-6.11 for so(8) instead of so(7)).",
        "  2. Cross-check with Paper 13 PMNS work (3 PMNS angles from",
        "     trefoil topology) to see if K_8 phase-soliton hypothesis",
        "     is consistent with observed PMNS structure.",
        "  3. Information-theoretic test: how many bits actually live",
        "     in a single neutrino's mass eigenstate?",
    ]

    summary_text = "\n".join(summary)
    (OUT / "summary.txt").write_text(summary_text)
    print()
    print(summary_text)
    print()
    print(f"Wrote {OUT/'summary.txt'}")


if __name__ == "__main__":
    main()
