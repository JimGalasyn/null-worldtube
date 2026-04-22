#!/usr/bin/env python3
"""
Paper 15 b2.6 -- 2T's natural 8-dim real representation  triv + 3 + 2^R
and the decomposition of Lambda^2(7-dim imaginary) = 21-dim under 2T.

Motivation.  b2.5 showed that Paper 15's integers 7, 8, 21 match
Spin(7)'s vector, spinor, adjoint rep dimensions.  A rigorous
2T subset Spin(7) embedding via the octonion left-mult failed (the
action is not a group homomorphism because octonions are
non-associative).  This script takes a cleaner route:

We BUILD 2T's natural 8-dim real rep directly as a direct sum of
2T irreps:
    V_8  =  triv (1)  +  3  +  2^R (4)
         =    1       +   3  +   4             (real dims = 8)

Here "2^R" is the quaternionic 2-dim complex irrep of 2T realified
to a 4-dim real irrep (Schur indicator -1).  This matches the
expected split for Spin(7)'s 8-dim spinor restricted to its
quaternion sub-algebra.

Then we look at V_7 = 3 + 2^R (imaginary part) and compute
Lambda^2(V_7) = 21-dim under 2T.  The decomposition into 2T irreps
will tell us how the 21 "so(7) adjoint gauge bosons" organise into
natural channels.

Tests:
  (1) The direct sum V_8 is a genuine rep of 2T (multiplicativity
      holds because it's a direct sum of reps).
  (2) The 7-dim subspace V_7 is 2T-invariant (trivially, since it's
      a direct summand).
  (3) Lambda^2(V_7) has dimension 21 and decomposes into a specific
      sum of 2T irreps.
  (4) Check whether the decomposition has 21 distinct 1-dim pieces
      (which would naturally match '21 crossings of sqrt(alpha)
      each'), or whether it's some grouping of higher-dim irreps.

Character-theoretic computation: use the known 2T character table
and compute Lambda^2 character via
    chi_Lambda2(g) = (chi(g)^2 - chi(g^2)) / 2,
then decompose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nwt_2T_fusion_21channels_b2_4 import tT_character_table
from nwt_2T_character_7dim_b2_4 import decompose_character


def main():
    print("=" * 72)
    print(" b2.6 -- 2T 8-dim rep (triv+3+2^R), Lambda^2 of 7-dim part")
    print("=" * 72)

    chi, sizes, names, dims = tT_character_table()

    # --- (1) Character of the 8-dim rep V_8 = triv + 3 + 2^R ---------
    # Characters at each class (7 classes in standard order):
    #   class 0: {e},  size 1
    #   class 1: {-e}, size 1
    #   class 2: 6 order-4 (trace 0), size 6
    #   class 3: 4 order-6 A, size 4
    #   class 4: 4 order-6 B, size 4
    #   class 5: 4 order-3 A, size 4
    #   class 6: 4 order-3 B, size 4
    #
    # triv: (1, 1, 1, 1, 1, 1, 1)
    # 3:    (3, 3, -1, 0, 0, 0, 0)
    # 2^R = 2 realified.  The complex "2" irrep has chi = (2,-2,0,1,-1,1,-1).
    # Realified character = 2 * Re(chi_complex), so for real entries:
    #   2^R chi = (4, -4, 0, 2, -2, 2, -2)
    # (Schur indicator -1: remains irreducible as real, dim doubles from 2 to 4.)

    chi_triv = chi[0].real.astype(float)
    chi_3 = chi[6].real.astype(float)
    chi_2 = chi[3]  # complex "2"
    # Realification: character on V^R is chi(g) + chi(g)^* = 2 Re(chi).
    chi_2R = 2.0 * chi_2.real

    print("\n[1] Characters at the 7 conjugacy classes of 2T")
    print("    (class order: e, -e, 6(order-4), 6A, 6B, 3A, 3B):")
    print()
    print(f"    triv            : {chi_triv.astype(int)}")
    print(f"    3 (irrep)       : {chi_3.astype(int)}")
    print(f"    2^R (realified) : {chi_2R.astype(int)}")

    chi_V8 = chi_triv + chi_3 + chi_2R
    print(f"    V_8 = sum       : {chi_V8.astype(int)}  "
          f"(dim = {int(chi_V8[0])})")

    chi_V7 = chi_3 + chi_2R         # imaginary part
    print(f"    V_7 = V_8 - triv: {chi_V7.astype(int)}  "
          f"(dim = {int(chi_V7[0])})")

    # --- (2) Verify V_8 decomposes into 2T irreps consistently -------
    # Convert real character into complex (embedding R in C for decomposition).
    chi_V8_complex = chi_V8.astype(complex)
    mult = decompose_character(chi_V8_complex, chi, sizes)
    print("\n[2] V_8 decomposition into 2T irreps:")
    for name, m in zip(names, mult):
        if abs(m) > 0.01:
            print(f"    {name:10s}: {m.real:+.3f}")

    # For a REAL rep, complex decomposition should have:
    #  - trivial: integer multiplicity m_0
    #  - omega and omega^2 appearing in equal parts (conjugate pair)
    #  - quaternionic 2-dim irreps appearing with DOUBLED multiplicity
    #    (since 2^R = 2 + 2^* as complex, and 2 = 2^* since self-conj)
    #    actually for quaternionic irreps, chi is real on R-classes
    #    but still the multiplicity doubles when realified.

    # Check the expected decomposition: V_8 = triv + 3 + 2^R.
    # In complex form: V_8 = 1 * triv + 1 * 3 + 2 * "2".
    # (The "2" irrep is self-conjugate in 2T's character table because
    # its character is real: 2,-2,0,1,-1,1,-1.)
    expected = np.zeros(7, dtype=complex)
    expected[0] = 1   # triv
    expected[6] = 1   # 3
    expected[3] = 2   # 2 (doubled for realification)
    print(f"\n    Expected: triv + 3 + 2 * '2' = "
          f"(1,0,0,2,0,0,1) across irreps")
    expected_str = "(" + ", ".join(
        f"{int(expected[i].real)}" for i in range(7)) + ")"
    actual_str = "(" + ", ".join(
        f"{int(round(m.real))}" for m in mult) + ")"
    print(f"    Actual                           : {actual_str}")
    match = actual_str == expected_str
    print(f"    {'PASS' if match else 'FAIL'}")

    # --- (3) Lambda^2 of V_7 (should have dim 21) --------------------
    # chi_Lambda2(g) = (chi(g)^2 - chi(g^2)) / 2.
    # For diagonal-block V_7 = 3 + 2^R, chi(g) = chi_3(g) + chi_2R(g).
    # chi(g^2): need chi at the squared class.  For 2T classes:
    #   e^2 = e       (class 0 -> 0)
    #   (-e)^2 = e    (class 1 -> 0)
    #   order-4^2 = -e (class 2 -> 1)
    #   order-6^2 = order-3 (classes 3,4 -> 5,6 or 6,5 depending)
    #   order-3^2 = order-3 (inverse) -- classes 5,6 cycle within themselves
    #
    # For clean computation, use the ABSTRACT identity
    #   chi(g^2) = evaluation of chi at the class of g^2.
    # Compute this for each class via group-theoretic knowledge.
    # Class ordering (standard):
    #   0: {e},    g^2 = e,    class 0
    #   1: {-e},   g^2 = e,    class 0
    #   2: order-4, g^2 = -e,   class 1
    #   3: order-6 A, g^2 = ??
    #       order-6 element g has g^6 = e, so g^2 has order 3.
    #       Is it class 5 (3A) or class 6 (3B)?  Depends on pairing.
    #   4: order-6 B, g^2 -> class 6 or 5.
    #   5: order-3 A, g^2 = order-3 A (since order 3, g^2 = g^{-1}, same class).
    #       Actually g^3 = e in SU(2) cover means g has order 3 (if trace
    #       matches), so g^2 = g^{-1} which is in the inverse class.  For
    #       the order-3 pair (5, 6), g^2 might be in class 5 OR 6.
    #   6: order-3 B, similarly.
    #
    # Best test: compute chi(g^2) NUMERICALLY from a matrix representative
    # of each class -- avoids relying on the class structure.

    # Load 2T matrices and compute squared-class indices.
    from nwt_poincare_sphere_b2_0 import generate_2I
    from nwt_2T_in_2I_check_b2_3a import find_2T_in, matrix_key

    twoI = generate_2I()
    T24 = find_2T_in(twoI)
    assert T24 is not None

    # Group 2T elements by conjugacy class (computed in b2.4).
    def class_of(g: np.ndarray,
                  groups_by_class: list[list[np.ndarray]]) -> int:
        k = matrix_key(g)
        for i, grp in enumerate(groups_by_class):
            if any(matrix_key(h) == k for h in grp):
                return i
        return -1

    # Build conjugacy classes manually.
    from nwt_2T_character_7dim_b2_4 import conjugacy_classes as conj_cls
    classes = conj_cls(T24)
    # Order classes to match the standard character-table order:
    # (trace +2), (trace -2), (size-6, trace 0), (trace +1) x 2,
    # (trace -1) x 2.
    def sort_key(c):
        rep = T24[c[0]]
        tr = round(np.trace(rep).real, 3)
        size = len(c)
        # Primary: trace DESCENDING for +2 first, but ASCENDING for the rest.
        if size == 1:
            return (0 if tr > 0 else 1, -tr)
        # For size-6 (trace 0), place right after the size-1 classes.
        if size == 6:
            return (2, 0)
        # For size-4 classes, place trace +1 before trace -1.
        return (3 if tr > 0 else 4, -tr, c[0])
    classes_sorted = sorted(classes, key=sort_key)
    # Sanity print
    print("\n[3] Computing chi(g^2) at each class numerically")
    class_reps = [T24[c[0]] for c in classes_sorted]
    chi_V7_sq = np.zeros(7)
    for i, g in enumerate(class_reps):
        g2 = g @ g
        # Find class index of g2
        ci = class_of(g2, [[T24[k] for k in c] for c in classes_sorted])
        chi_V7_sq[i] = chi_V7[ci]
        print(f"    class {i}: g^2 lies in class {ci}, "
              f"chi_V7(g^2) = {int(chi_V7_sq[i]):+d}")

    chi_Lambda2 = (chi_V7 ** 2 - chi_V7_sq) / 2.0
    print(f"\n    chi(Lambda^2 V_7) at classes = {chi_Lambda2}")
    print(f"    Dimension = chi(Lambda^2 V_7)(e) = "
          f"{int(chi_Lambda2[0])} "
          f"(expected 21 = 7*6/2)")

    # --- (4) Decompose Lambda^2 V_7 into 2T irreps -----------------
    chi_Lambda2_complex = chi_Lambda2.astype(complex)
    mult_L2 = decompose_character(chi_Lambda2_complex, chi, sizes)
    print("\n[4] Lambda^2 V_7 decomposition into 2T irreps:")
    for name, m in zip(names, mult_L2):
        if abs(m) > 0.01:
            print(f"    {name:10s}: {m.real:+.3f}"
                  f"{' + ' + str(round(m.imag, 3)) + 'i' if abs(m.imag) > 0.01 else ''}")

    dim_check = sum(m.real * d for m, d in zip(mult_L2, dims))
    print(f"    Total dim (should be 21): {dim_check:.1f}")

    # --- (5) Physical interpretation ---------------------------------
    print()
    print("=" * 72)
    print(" INTERPRETATION")
    print("=" * 72)
    print()
    print("    The 21-dimensional so(7) adjoint rep, restricted to 2T")
    print("    through the V_8 = triv + 3 + 2^R natural embedding,")
    print("    decomposes into the 2T irreps reported above.  Each")
    print("    irrep block represents a 2T-covariant sector; the total")
    print("    count of 21 real dimensions corresponds to 21 gauge-")
    print("    holonomy channels in the physical ansatz.")
    print()
    print("    CANDIDATE PHYSICAL AMPLITUDE (still conjectural):")
    print("    For the m_e / m_Pl transition amplitude on S^3/2I, each")
    print("    of the 21 real Lambda^2 V_7 basis elements contributes")
    print("    one sqrt(alpha) factor via Paper 13's transition-amplitude")
    print("    rule, giving alpha^(21/2).  The 2T-equivariance forces")
    print("    the amplitude to couple identically to all 21 components")
    print("    in its irrep orbits.")
    print()
    print("    The NLO (1 + alpha/7) factor would come from a one-loop")
    print("    graph with 7 vector-rep channels (V_7 acting in one of its")
    print("    sub-rep slots).  The 8/7 prefactor is the ratio of the")
    print("    full V_8 (8-dim) to the 'active' V_7 (7-dim), matching")
    print("    the Spin(7) spinor/vector dimension ratio.")


if __name__ == "__main__":
    main()
