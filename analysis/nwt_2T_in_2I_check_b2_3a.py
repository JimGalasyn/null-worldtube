#!/usr/bin/env python3
"""
Paper 15 b2.3a -- Rigorous verification of the 2T/2I structural claim.

Claim:  The 24 trefoils in the 2I orbit of T(2,3) biject with elements
of the binary tetrahedral group 2T (order 24), via the set
decomposition

    2I = 2T . Z_5 ,     2T intersect Z_5 = {e} ,

where Z_5 is the trefoil's stabilizer in 2I.  Combined with the McKay
correspondence 2T <-> extended E_6 Dynkin (7 nodes, 7 irreps of 2T
with dim^2 summing to 24), this would reinterpret Paper 15's

    lambda_1 = 168  =  7  x  24  =  |Irr(2T)|  x  |2T|.

This script performs three independent checks:

  (1) Build 2T explicitly as the 24 Hurwitz unit quaternions in
      SU(2) and verify it closes under multiplication: 24x24 products
      stay in the 24-element set (i.e., 2T is a subgroup).

  (2) Verify 2T is a subgroup of 2I by matching each Hurwitz unit
      to an element of the 120-element list generated in b2.0.

  (3) Identify the trefoil stabilizer Z_5 explicitly in 2I and verify
      2T intersect Z_5 = {identity}, so the set decomposition
      2I = 2T . Z_5 holds (by index counting).

Any single failure below is fatal for the structural claim.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nwt_poincare_sphere_b2_0 import generate_2I


# =========================================================================
# (1) Build 2T explicitly as Hurwitz quaternions in SU(2).
# =========================================================================

def quaternion_to_SU2(a: float, b: float, c: float, d: float) -> np.ndarray:
    """Unit quaternion q = a + b i + c j + d k  ->  SU(2) matrix.

    Convention:  i = [[i,0],[0,-i]],  j = [[0,1],[-1,0]],  k = [[0,i],[i,0]].
    Then q -> a*I + b*i + c*j + d*k.
    """
    return np.array([
        [a + 1j * b,  c + 1j * d],
        [-c + 1j * d, a - 1j * b],
    ], dtype=complex)


def hurwitz_2T():
    """Return the 24 Hurwitz unit quaternions as SU(2) matrices.

    Note: the resulting 2T is a *canonical* realisation in SU(2)
    but is NOT necessarily a subgroup of any GIVEN 2I (since 2I's
    matrix embedding is axis-dependent).  Use find_2T_in(2I) to
    locate a 2T-isomorphic subgroup inside a specific 2I list.
    """
    units = []
    for axis in range(4):
        for sign in (+1, -1):
            q = [0.0, 0.0, 0.0, 0.0]
            q[axis] = float(sign)
            units.append(quaternion_to_SU2(*q))
    for signs in product([+1, -1], repeat=4):
        q = [s / 2.0 for s in signs]
        units.append(quaternion_to_SU2(*q))
    assert len(units) == 24
    return units


def close_subgroup(twoI: list[np.ndarray],
                   generators: list[np.ndarray],
                   max_size: int = 121) -> dict | None:
    """Generate a subgroup by closure.  Returns {key: matrix} dict
    of the resulting subgroup, or None if size exceeds max_size.
    """
    gen = {matrix_key(np.eye(2, dtype=complex)):
           np.eye(2, dtype=complex)}
    for g in generators:
        gen[matrix_key(g)] = g
    frontier = list(generators)
    while frontier and len(gen) < max_size:
        new_frontier = []
        for g in frontier:
            for h in generators:
                for p in (g @ h, h @ g):
                    k = matrix_key(p)
                    if k not in gen:
                        if len(gen) >= max_size:
                            return None
                        gen[k] = p
                        new_frontier.append(p)
        frontier = new_frontier
    return gen


def find_2T_in(twoI: list[np.ndarray]) -> list[np.ndarray] | None:
    """Find a 24-element subgroup of 2I isomorphic to 2T.

    Strategy.  2T has an abstract presentation
        2T = < r, s | r^3 = s^3 = (rs)^2 = -e >.
    In SU(2) terms: r and s are order-6 elements (trace +1) whose
    product rs is an order-4 element (trace 0).  Such a pair
    generates exactly 2T.

    Iterate over pairs (r, s) of trace-1 elements until the product
    has trace 0 AND the generated closure has exactly 24 elements
    with the correct trace histogram.
    """
    traces = np.array([np.trace(g).real for g in twoI])
    idx_tr1 = [i for i in range(120) if abs(traces[i] - 1) < 1e-6]
    assert len(idx_tr1) == 20

    expected_hist = {2.0: 1, -2.0: 1, 0.0: 6, 1.0: 8, -1.0: 8}

    from collections import Counter
    for i in idx_tr1:
        r = twoI[i]
        for j in idx_tr1:
            if j == i:
                continue
            s = twoI[j]
            rs = r @ s
            tr_rs = np.trace(rs).real
            if abs(tr_rs) > 1e-6:            # rs must be order-4 (trace 0)
                continue
            sub = close_subgroup(twoI, [r, s], max_size=25)
            if sub is None or len(sub) != 24:
                continue
            # Check trace histogram
            hist = Counter()
            for m in sub.values():
                hist[round(np.trace(m).real, 6)] += 1
            if all(hist.get(t, 0) == c for t, c in expected_hist.items()):
                return list(sub.values())
    return None


def matrix_key(M: np.ndarray, tol: int = 8) -> tuple:
    return tuple(np.round(M.flatten(), tol).tolist())


def matrix_list_contains(M: np.ndarray, lst: list[np.ndarray],
                          tol: int = 8) -> int | None:
    """Return index of matching matrix in lst, or None if not found."""
    key = matrix_key(M, tol)
    for idx, L in enumerate(lst):
        if matrix_key(L, tol) == key:
            return idx
    return None


def test_subgroup_closure(elements: list[np.ndarray]) -> bool:
    """Verify that the set is closed under multiplication."""
    keys = {matrix_key(e) for e in elements}
    for a in elements:
        for b in elements:
            if matrix_key(a @ b) not in keys:
                return False
    return True


# =========================================================================
# (2) Trefoil stabiliser Z_5 explicitly.
# =========================================================================

def trefoil_stabilizer_Z5() -> list[np.ndarray]:
    """Stabiliser of T(2,3) on the Clifford torus under SU(2) left action.

    By the analysis in b2.3:  alpha = 4 pi k / 5 for k = 0..4 gives
    maximal-torus elements  diag(e^{i alpha}, e^{-i alpha})  that fix
    the trefoil's point set.  These five elements form Z_5.
    """
    elts = []
    for k in range(5):
        a = 4.0 * np.pi * k / 5.0
        g = np.array([[np.exp(1j * a), 0], [0, np.exp(-1j * a)]],
                     dtype=complex)
        elts.append(g)
    return elts


# =========================================================================
# (3) McKay data for 2T.
# =========================================================================

def check_mckay_2T() -> dict:
    """Return the known McKay correspondence data for 2T (binary
    tetrahedral group, E_6 hat extended Dynkin)."""
    return dict(
        num_irreps=7,
        irrep_dims=(1, 1, 1, 2, 2, 2, 3),
        sum_of_squares=1 + 1 + 1 + 4 + 4 + 4 + 9,
        extended_Dynkin="E_6 hat (7 nodes)",
        num_conjugacy_classes=7,
    )


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 72)
    print(" b2.3a -- verify the structural claim 168 = 7 x 24")
    print("           as  |Irr(2T)| x |2T|  with 2T subset 2I,")
    print("           2T intersect Z_5 = {e}.")
    print("=" * 72)

    # --- (1) 2T closure --------------------------------------------------
    print("\n[1] Build 2T as 24 Hurwitz units; verify multiplicative closure")
    T24 = hurwitz_2T()
    # Check distinctness
    keys = {matrix_key(m) for m in T24}
    assert len(keys) == 24, f"Expected 24 distinct Hurwitz units, got {len(keys)}"
    print(f"    Constructed {len(T24)} distinct Hurwitz unit SU(2) matrices.")
    closed = test_subgroup_closure(T24)
    print(f"    Multiplicative closure:  "
          f"{'PASS' if closed else 'FAIL'}")
    assert closed

    # --- (2) Find 2T subgroup INSIDE my 2I -----------------------------
    print("\n[2] Find a 2T subgroup inside the generated 2I list")
    print("    (Hurwitz 2T uses canonical quaternion axes; our 2I uses")
    print("     different axis choice, so we must SEARCH for an")
    print("     isomorphic subgroup rather than importing one.)")
    twoI = generate_2I()
    assert len(twoI) == 120
    T24_in_2I = find_2T_in(twoI)
    if T24_in_2I is None:
        print("    FAIL: no 24-element subgroup of trace {-2,-1,0,1,2}")
        print("    elements closes inside my 2I.  Structural claim fails.")
        sys.exit(1)
    print(f"    Found a 24-element subgroup inside 2I.  Trace histogram:")
    from collections import Counter
    trace_hist = Counter()
    for m in T24_in_2I:
        tr = round(np.trace(m).real, 6)
        trace_hist[tr] += 1
    for tr, ct in sorted(trace_hist.items(), reverse=True):
        print(f"      trace = {tr:+6.3f}  count = {ct}")
    expected = {2.0: 1, -2.0: 1, 0.0: 6, 1.0: 8, -1.0: 8}
    match = all(trace_hist.get(t, 0) == c for t, c in expected.items())
    print(f"    Matches 2T trace histogram "
          f"(1,1,6,8,8 at trace 2,-2,0,1,-1)?  "
          f"{'YES' if match else 'NO'}")
    assert match, "Subgroup trace histogram does not match 2T."
    # Use this as our 2T for remaining checks.
    T24 = T24_in_2I

    # --- (3) Z_5 stabilizer subset of 2I -------------------------------
    print("\n[3] Verify the trefoil stabilizer Z_5 is in 2I")
    Z5 = trefoil_stabilizer_Z5()
    Z5_in_2I = [matrix_list_contains(g, twoI) for g in Z5]
    print(f"    Z_5 elements in 2I: {sum(x is not None for x in Z5_in_2I)} / 5")
    assert None not in Z5_in_2I, "Z_5 element not in 2I — "\
        "probably generator mismatch; 2I generated from different axis."
    print(f"    PASS: Z_5 subset 2I.")

    # --- (4) 2T intersect Z_5 = {e} -------------------------------------
    print("\n[4] Verify 2T intersect Z_5 = {identity}")
    T24_keys = {matrix_key(m) for m in T24}
    Z5_keys = {matrix_key(m) for m in Z5}
    common = T24_keys & Z5_keys
    print(f"    |2T intersect Z_5| = {len(common)}")
    if len(common) == 1:
        print(f"    PASS:  only the identity is shared.")
    else:
        print(f"    FAIL: expected exactly 1 shared element (the identity).")
    assert len(common) == 1

    # --- (5) 2I = 2T . Z_5 as set ---------------------------------------
    print("\n[5] Verify 2I = 2T . Z_5 (every element of 2I is uniquely t r)")
    products = set()
    for t in T24:
        for r in Z5:
            products.add(matrix_key(t @ r))
    print(f"    |{{ t r : t in 2T, r in Z_5 }}| = {len(products)}")
    twoI_keys = {matrix_key(g) for g in twoI}
    print(f"    |2I|                            = {len(twoI_keys)}")
    missing_from_product = twoI_keys - products
    extra_in_product = products - twoI_keys
    print(f"    elements of 2I not reached by t r: {len(missing_from_product)}")
    print(f"    elements t r not in 2I          : {len(extra_in_product)}")
    if len(products) == 120 and not missing_from_product and not extra_in_product:
        print(f"    PASS:  2I = 2T . Z_5 as a set (disjoint unique factorisation).")
    else:
        print(f"    FAIL:  the set decomposition is not exact.")
    assert len(products) == 120

    # --- (6) McKay data for 2T -----------------------------------------
    print("\n[6] McKay correspondence for 2T")
    data = check_mckay_2T()
    print(f"    Number of irreducible reps         : "
          f"{data['num_irreps']} "
          f"(== Paper 15's A-tilde = 7) ")
    print(f"    Irrep dimensions                   : "
          f"{data['irrep_dims']}")
    print(f"    Sum of squares = |2T|              : "
          f"{data['sum_of_squares']} == 24 ")
    print(f"    Corresponding extended Dynkin      : "
          f"{data['extended_Dynkin']}")
    print(f"    Conjugacy classes = num. of irreps : "
          f"{data['num_conjugacy_classes']} ")

    # --- (7) Structural sum 168 = 7 x 24 -------------------------------
    print("\n[7] Final structural identification")
    print(f"    |2T|        = 24        (size of binary tetrahedral group)")
    print(f"    |Irr(2T)|   = 7         (# irreps of 2T, == McKay E_6 hat nodes)")
    print(f"    7 x 24      = {7 * 24}       (= lambda_1 on S^3/2I, verified in b2.0)")
    print(f"    Paper 15: lambda_1 / (A-tilde + 1) = 168 / 8 = 21")
    print(f"    = dim(Lambda^2 C^7)")
    print(f"    = # pairs of 2T irreps")
    print(f"    = crossings in a candidate Fano-equivariant diagram (b2.1c)")

    # --- Conjugacy-class verification for 2T (extra, optional) ---------
    # Classes of 2T:  {e} (1), {-e} (1), 6 order-4 (trace 0), 4 order-6 (trace 1),
    #   4 order-6 (trace 1 too? actually different classes split by inverse),
    #   4 order-3 (trace -1), 4 order-3 (trace -1 second class).
    # We do a fast conjugacy check by grouping by trace (partial invariant).
    print("\n[8] Conjugacy-class substructure of 2T (trace-histogram):")
    from collections import Counter
    trace_hist = Counter()
    for m in T24:
        tr = round(np.trace(m).real, 6)
        trace_hist[tr] += 1
    for tr, ct in sorted(trace_hist.items(), reverse=True):
        phi = 2 * np.arccos(np.clip(tr / 2, -1, 1)) / np.pi
        print(f"    trace = {tr:+6.3f}  n = {ct:2d}  "
              f"(phi/pi = {phi:.4f})")
    # Note: some 'classes' share traces (e.g. two order-3 classes both have trace -1).
    # Character-theoretic conjugacy class counts require character tables.

    print()
    print("=" * 72)
    print(" ALL CHECKS PASSED")
    print("=" * 72)
    print("\nConclusions:")
    print("  (1) 2T is a genuine index-5 subgroup of 2I.")
    print("  (2) The trefoil stabilizer Z_5 in 2I intersects 2T trivially.")
    print("  (3) 2I factors as the SET 2T . Z_5 uniquely (orbit-stabiliser).")
    print("  (4) 2T has exactly 7 irreducible representations by McKay (E_6 hat).")
    print("  (5) 7 x 24 = 168 = lambda_1 of the scalar Laplacian on S^3/2I.")
    print()
    print("The structural identification 168 = |Irr(2T)| x |2T| is confirmed.")
    print("What remains conjectural is whether this identification reflects a")
    print("DYNAMICAL mechanism (e.g. a transition amplitude that counts 21 =")
    print("C(7, 2) irrep-pair interactions) producing alpha^(21/2).")


if __name__ == "__main__":
    main()
