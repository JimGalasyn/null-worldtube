#!/usr/bin/env python3
"""
Paper 15 b2.14 -- The 8/7 prefactor as a Spin(7) spinor-to-vector
dimension ratio.

Target:  the  (8/7)  prefactor in  m_e / m_Pl = (8/7)(1 + alpha/7) alpha^(21/2).

Strategy.  The Cl(0, 7) construction (b2.12) realises 2T subset Spin(7)
acting on R^8 = octonions via the Clifford bivectors.  This splits
R^8 = (real axis, 1 dim) (+) (imaginary octonions, 7 dim).
The spinor and vector representations of Spin(7) have dimensions 8
and 7 respectively.

Candidate physical interpretations of the 8/7 prefactor:

  (A) dim(spinor) / dim(vector)  =  8 / 7.
      Matter (electron vortex) propagates in the spinor rep (8-dim
      octonion space via Cl(0, 7)); gauge fields exchange through
      the vector rep (7-dim Im(O)).  Amplitude normalization ratio.

  (B) dim(spinor)^2 / dim(adjoint) = 64/21.  Ruled out (not 8/7).

  (C) C_2(vector) / C_2(spinor).  For Spin(7):
         C_2(vector) = n(n-1) = 6  (n = 3)
         C_2(spinor) = 21/4
      Ratio: 6 / (21/4) = 24/21 = 8/7.
      So the ratio of quadratic Casimirs of vector to spinor is
      EXACTLY 8/7.  This is a structural invariant of Spin(7).

  (D) Dynkin index ratio T(vector)/T(spinor).  For Spin(7), both
      vector and spinor have Dynkin index 2, so ratio is 1.
      Does not give 8/7.

Interpretations (A) and (C) both give 8/7 from independent
structural sources, suggesting the prefactor is genuinely
determined by Spin(7) representation theory.

This script:
  (1) Verifies (A) explicitly.
  (2) Computes the Spin(7) Casimirs and verifies (C).
  (3) Computes several other Spin(7) invariants and confirms
      (A) and (C) are the structural sources.
  (4) Identifies the physical setting where (A) would appear as
      an amplitude normalization.
"""

from __future__ import annotations

import numpy as np

# =========================================================================
# Spin(7) representation theory.
# =========================================================================

def spin7_casimir(weights: tuple[int, int, int]) -> float:
    """Quadratic Casimir eigenvalue on an irrep of Spin(7) = B_3.

    Using the formula  C_2 = (lambda + rho) . (lambda + rho) - rho . rho
    where  rho = (5/2, 3/2, 1/2)  (half-sum of positive roots for B_3).

    For Spin(7) = B_3, positive roots are {e_i +- e_j : i<j} union {e_i},
    totaling 9 positive roots.  Half-sum: (5, 3, 1)/2.

    Weight argument: `weights` is the Dynkin label in e_i-basis.
      vector rep   (highest weight): (1, 0, 0)
      spinor rep   (highest weight): (1/2, 1/2, 1/2)
      adjoint rep  (highest weight): (1, 1, 0)
    """
    l = np.array([float(x) for x in weights])
    rho = np.array([5/2, 3/2, 1/2])
    v1 = l + rho
    v2 = rho
    return (v1 @ v1) - (v2 @ v2)


def dim_from_weights(weights: tuple[int, int, int]) -> int:
    """Weyl dimension formula for Spin(7) = B_3.

    dim(lambda) = prod_{alpha in positive roots} (lambda + rho) . alpha
                                                 / rho . alpha

    Positive roots for B_3: e_1 + e_2, e_1 - e_2, e_1 + e_3, e_1 - e_3,
    e_2 + e_3, e_2 - e_3, e_1, e_2, e_3.
    """
    l = np.array([float(x) for x in weights])
    rho = np.array([5/2, 3/2, 1/2])
    pos_roots = [
        np.array([1, 1, 0]),
        np.array([1, -1, 0]),
        np.array([1, 0, 1]),
        np.array([1, 0, -1]),
        np.array([0, 1, 1]),
        np.array([0, 1, -1]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]
    num = 1.0
    den = 1.0
    for a in pos_roots:
        num *= (l + rho) @ a
        den *= rho @ a
    return round(num / den)


# =========================================================================
# Main.
# =========================================================================

def main():
    print("=" * 72)
    print(" b2.14 -- The 8/7 prefactor from Spin(7) rep theory")
    print("=" * 72)

    # --- Key Spin(7) irreps ---
    reps = {
        "vector"  : (1, 0, 0),
        "spinor"  : (0.5, 0.5, 0.5),
        "adjoint" : (1, 1, 0),
    }

    print("\n[1] Spin(7) irrep dimensions and Casimirs:")
    print(f"    {'irrep':<10} {'dim':>6} {'Casimir C_2':>14}")
    data = {}
    for name, hw in reps.items():
        # For non-integer weights (spinor), dim_from_weights still
        # works if we use rationals; use float and round.
        dim = dim_from_weights(hw)
        c2 = spin7_casimir(hw)
        data[name] = (dim, c2)
        print(f"    {name:<10} {dim:>6} {c2:>14.4f}")

    # --- (A) dim ratio ---
    ratio_A = data["spinor"][0] / data["vector"][0]
    print(f"\n[2] Candidate (A): dim(spinor) / dim(vector)")
    print(f"    = {data['spinor'][0]} / {data['vector'][0]} "
          f"= {ratio_A:.6f}  (equals 8/7 = {8/7:.6f}?  "
          f"{'YES' if abs(ratio_A - 8/7) < 1e-10 else 'NO'})")

    # --- (B) dim^2 / adjoint ---
    ratio_B = data["spinor"][0] ** 2 / data["adjoint"][0]
    print(f"\n[3] Candidate (B): dim(spinor)^2 / dim(adjoint)")
    print(f"    = {data['spinor'][0]**2} / {data['adjoint'][0]} "
          f"= {ratio_B:.4f}  (equals 8/7?  "
          f"{'YES' if abs(ratio_B - 8/7) < 1e-10 else 'NO'})")

    # --- (C) Casimir ratio ---
    ratio_C = data["vector"][1] / data["spinor"][1]
    print(f"\n[4] Candidate (C): C_2(vector) / C_2(spinor)")
    print(f"    = {data['vector'][1]:.4f} / {data['spinor'][1]:.4f} "
          f"= {ratio_C:.6f}  (equals 8/7?  "
          f"{'YES' if abs(ratio_C - 8/7) < 1e-10 else 'NO'})")

    # --- (D) Dynkin index ---
    # Dynkin index: T(pi) = (dim(pi) / dim(adj)) * C_2(pi)
    print(f"\n[5] Candidate (D): T(vector) / T(spinor) (Dynkin indices)")
    T_vec = (data["vector"][0] / data["adjoint"][0]) * data["vector"][1]
    T_spin = (data["spinor"][0] / data["adjoint"][0]) * data["spinor"][1]
    print(f"    T(vector) = (7/21) * 6     = {T_vec:.4f}")
    print(f"    T(spinor) = (8/21) * 21/4  = {T_spin:.4f}")
    print(f"    Ratio: {T_vec / T_spin:.6f}  (equals 8/7?  "
          f"{'YES' if abs(T_vec / T_spin - 8/7) < 1e-10 else 'NO'})")
    print(f"    (In Spin(7) both vector and spinor have Dynkin index 2;")
    print(f"    this is a known special property of B_3.)")

    # --- Summary and physical interpretation ---
    print()
    print("=" * 72)
    print(" STRUCTURAL SOURCE OF 8/7")
    print("=" * 72)
    print(f"""
  Two independent structural routes give 8/7 exactly:

    (A)  8/7  =  dim(Spin(7) spinor) / dim(Spin(7) vector)
              =  dim(octonions) / dim(Im(octonions))

    (C)  8/7  =  C_2(vector) / C_2(spinor)  for Spin(7) = B_3
              =  6 / (21/4)
              =  24 / 21

  These are RELATED, not independent: in fact, for any simple Lie
  algebra, T(pi) = (dim pi / dim adj) C_2(pi), so the product
  dim(pi) * C_2(pi)  is proportional to the Dynkin index T(pi).  For
  B_3, T(vec) = T(spin) = 2, hence:

       dim(vec) * C_2(vec) = dim(spin) * C_2(spin),
  i.e.  dim(spin) / dim(vec) = C_2(vec) / C_2(spin).

  So (A) and (C) are the SAME equation, both equal to 8/7, reflecting
  a single underlying Spin(7) identity: the spinor and vector reps of
  B_3 have equal Dynkin index, so their dim and Casimir ratios are
  inverse to each other, and both give 8/7.

  Physical interpretation in the e -> sigma amplitude:

    * Matter (electron vortex) propagates in the 8-dim spinor of
      Spin(7) -- the full octonion space via the Cl(0, 7) Clifford
      action (b2.12).

    * Gauge exchanges (the 21 edges of K_7 in b2.13) happen via
      the 7-dim vector rep of Spin(7) = imaginary octonions.

    * The amplitude involves a "matter propagator" in the spinor
      channel and "gauge exchanges" in the vector channel.  At
      leading order, the cross-section ratio / normalization
      contains the factor  dim(spinor) / dim(vector) = 8/7.

  This gives a CONCRETE STRUCTURAL SOURCE for the 8/7 prefactor:
  the ratio of propagator dimensionalities in the matter (spinor)
  vs. gauge (vector) sectors of Spin(7).

  The result remains at the level of STRUCTURAL IDENTIFICATION -- a
  full dynamical derivation from a Lagrangian remains for future
  work -- but 8/7 is now traceable to a specific, well-defined
  Spin(7) invariant, consistent with the b2.12/b2.13 chain.
""")


if __name__ == "__main__":
    # Placeholder import guard (avoid importing Fraction unless needed)
    main()
