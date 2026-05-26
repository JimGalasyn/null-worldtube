#!/usr/bin/env python3
"""
Paper 6 Mass-Spectrum Assignment Degeneracy Probe
==================================================

Tests whether the canonical (p,q,m_phase,n_q) assignments in
Paper 6 / Paper 11 are unique fits to the observed particle
masses, or whether alternative assignments with smaller error
exist that were missed by the original search.

The companion script analysis/nwt_tau_lepton_search.py found
that the tau lepton has a lepton-sector candidate
(2,19,45,0) at +0.017% error, vs the canonical baryon-sector
(3,4,17,3) at +0.73%.  This script extends the search to
the full Paper 6 list to determine whether the original
classification is broadly fragile.

For each particle the script reports:
  - the canonical (p,q,m_phase,n_q) and its error
  - the best alternative (lepton, meson, baryon sector each)
  - whether (p,q) is Pythagorean ('Skilton triangle')
  - whether the canonical assignment is dominated by a tighter fit
"""

from __future__ import annotations

import math
from math import gcd

# Constants
M_E_MEV = 0.5109989461

# Reference (electron)
P_E, Q_E, M_PHASE_E = 2, 1, 3
BETA_E = math.sqrt(M_PHASE_E ** 2 / P_E ** 2 - 1)
LN_8_BETA_E = math.log(8 * BETA_E)


def is_pythagorean(p: int, q: int) -> bool:
    """True if (p, q, h) is an integer Pythagorean triple for some h."""
    h2 = p * p + q * q
    h = int(round(math.sqrt(h2)))
    return h * h == h2 and h > 0


def mass_ratio(p: int, q: int, m_phase: int, n_q: int) -> float | None:
    """Predicted m / m_e from the Paper 6 / 11 mass formula."""
    if p <= 0:
        return None
    inner = m_phase ** 2 / p ** 2 - 1.0
    if inner <= 0:
        return None
    beta = math.sqrt(inner)
    if beta <= 0:
        return None
    if n_q == 0:
        nq_factor = 1.0
    else:
        nq_factor = float(n_q) ** q
    return ((p * p + q * q) / 5.0) * (beta / BETA_E) * (
        math.log(8 * beta) / LN_8_BETA_E
    ) * nq_factor


def search_best(
    target_ratio: float,
    p_max: int = 15,
    q_max: int = 30,
    m_max: int = 100,
    n_q_values: tuple[int, ...] = (0, 2, 3),
    tol_pct: float = 5.0,
) -> dict:
    """For each n_q sector, return the best (p,q,m_phase) fit."""
    best_per_sector: dict[int, tuple] = {}
    for n_q in n_q_values:
        best = None
        for p in range(1, p_max + 1):
            for q in range(1, q_max + 1):
                if gcd(p, q) != 1:
                    continue
                for m_phase in range(p + 1, m_max + 1):
                    r = mass_ratio(p, q, m_phase, n_q)
                    if r is None:
                        continue
                    err = (r - target_ratio) / target_ratio * 100.0
                    if abs(err) > tol_pct:
                        continue
                    if best is None or abs(err) < abs(best[3]):
                        best = (p, q, m_phase, err, r)
        if best is not None:
            best_per_sector[n_q] = best
    return best_per_sector


# Paper 6 canonical assignments to test.  (Subset focused on "is the
# baryon classification fragile" — leptons + the four most studied
# baryons + a few mesons for control.)
CANONICAL = [
    # (name, m_exp_MeV, p, q, m_phase, n_q, canonical_err_pct)
    ("e",     0.511,    2, 1,  3, 0,  0.00),
    ("mu",    105.66,   1, 8,  9, 0, -1.97),
    ("tau",   1776.86,  3, 4, 17, 3, +0.73),
    ("pi0",   135.00,   7, 3, 18, 2, +0.24),
    ("pi+",   139.57,   3, 5,  5, 2, -0.20),
    ("K+",    493.68,   2, 5,  8, 2, +0.36),
    ("p",     938.27,   1, 4,  5, 3, -0.13),
    ("n",     939.57,   1, 4,  5, 3, -0.26),
    ("Lambda",1115.70,  3, 4, 12, 3, -0.08),
    ("Sigma", 1189.40,  1, 4,  6, 3, +0.29),
    ("Delta", 1232.00,  5, 4, 15, 3, -0.80),
    ("Xi",    1314.90,  5, 4, 16, 3, +2.21),
    ("Omega", 1672.50,  7, 4, 19, 3, -0.21),
]


def banner(s: str) -> None:
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


def main() -> None:
    print("=" * 72)
    print("ASSIGNMENT DEGENERACY PROBE FOR PAPER 6 SPECTRUM")
    print("=" * 72)
    print(f"  Searching: p ∈ [1,15], q ∈ [1,30], m_phase ∈ [p+1, 100]")
    print(f"  n_q sectors: 0 (lepton), 2 (meson), 3 (baryon)")
    print(f"  Tolerance: ±5%")

    rows = []
    for name, m_exp, p_c, q_c, mph_c, nq_c, err_c in CANONICAL:
        target = m_exp / M_E_MEV
        best = search_best(target)
        canon_pyth = is_pythagorean(p_c, q_c)
        rows.append((name, m_exp, p_c, q_c, mph_c, nq_c, err_c,
                     canon_pyth, best))

    # Print result table.
    banner("PER-PARTICLE RESULTS")
    print(f"  {'Particle':>9}  {'sector':>6}  {'(p,q,m,n_q)':>15}  "
          f"{'err %':>8}  {'pyth':>5}")
    print(f"  {'-'*9}  {'-'*6}  {'-'*15}  {'-'*8}  {'-'*5}")

    for (name, m_exp, p_c, q_c, mph_c, nq_c, err_c, canon_pyth,
         best) in rows:
        # Canonical row.
        print(f"  {name:>9}  {'CANON':>6}  "
              f"({p_c},{q_c},{mph_c},{nq_c}){'':>4}  "
              f"{err_c:+8.3f}  {'YES' if canon_pyth else '   ':>5}")
        # Best alternative in each sector.
        for sector_label, n_q in [("LEP", 0), ("MES", 2), ("BAR", 3)]:
            if n_q in best:
                p, q, mph, err, r = best[n_q]
                same_as_canon = (p == p_c and q == q_c and
                                 mph == mph_c and n_q == nq_c)
                marker = " <- canon" if same_as_canon else ""
                pyth = is_pythagorean(p, q)
                # Highlight if alternative is much better than canonical.
                better = abs(err) < abs(err_c) - 0.1 and not same_as_canon
                tag = " *BETTER*" if better else ""
                print(f"  {'':9}  {sector_label:>6}  "
                      f"({p},{q},{mph},{n_q}){'':>4}  "
                      f"{err:+8.3f}  {'YES' if pyth else '   ':>5}"
                      f"{tag}{marker}")
        print()

    # Summary: how many particles have a better alternative than canonical?
    banner("SUMMARY")
    n_dominated = 0
    n_pyth_canon = 0
    n_pyth_winners = 0
    print(f"  {'Particle':>9}  {'canon err':>10}  {'best alt err':>13}  "
          f"{'sector':>7}  {'pyth?':>5}")
    print(f"  {'-'*9}  {'-'*10}  {'-'*13}  {'-'*7}  {'-'*5}")
    for (name, m_exp, p_c, q_c, mph_c, nq_c, err_c, canon_pyth,
         best) in rows:
        if canon_pyth:
            n_pyth_canon += 1
        # Find the best alternative across all sectors that isn't the
        # canonical one.
        best_alt = None
        for n_q, info in best.items():
            p, q, mph, err, r = info
            same_as_canon = (p == p_c and q == q_c and mph == mph_c
                             and n_q == nq_c)
            if same_as_canon:
                continue
            if best_alt is None or abs(err) < abs(best_alt[1]):
                best_alt = ((p, q, mph, n_q), err)
        if best_alt is None:
            print(f"  {name:>9}  {err_c:+10.3f}  {'(no alt)':>13}  "
                  f"{'-':>7}  {'-':>5}")
            continue
        (p, q, mph, n_q), alt_err = best_alt
        sector = {0: "LEP", 2: "MES", 3: "BAR"}[n_q]
        pyth = is_pythagorean(p, q)
        if pyth:
            n_pyth_winners += 1
        dominated = abs(alt_err) < abs(err_c) - 0.1
        if dominated:
            n_dominated += 1
        flag = " *DOM*" if dominated else ""
        print(f"  {name:>9}  {err_c:+10.3f}  {alt_err:+13.3f}  "
              f"{sector:>7}  {'YES' if pyth else '   ':>5}{flag}")

    print(f"\n  Particles where canonical assignment is DOMINATED by an")
    print(f"  alternative (|alt err| < |canon err| - 0.1%): {n_dominated} of {len(rows)}")
    print(f"  Canonical assignments that are Pythagorean: {n_pyth_canon}")
    print(f"  Best-alternative assignments that are Pythagorean: "
          f"{n_pyth_winners}")
    print()
    print("INTERPRETATION:")
    print("  - Many DOMINATED rows -> the Paper 6 classification is fragile;")
    print("    the (p,q,m,n_q) -> particle map needs additional selection")
    print("    rules beyond mass-fit.")
    print("  - Few DOMINATED rows -> Paper 6 was lucky or used implicit")
    print("    selection rules; tau may be a special case.")
    print("  - Pythagorean (Skilton) hits in the WINNERS column would")
    print("    suggest the Pythagorean structure IS the missing rule.")


if __name__ == "__main__":
    main()
