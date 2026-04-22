#!/usr/bin/env python3
"""
Paper 15 b2.4 -- 2T character table, natural 7-dim representation,
and the Lambda^2 C^7 = 21-dim structure.

Motivation.  b2.3a verified the structural identification
    lambda_1 = 168 = |Irr(2T)| x |2T| = 7 x 24.
Paper 15 uses  21 = dim(Lambda^2 C^7)  as the alpha exponent /2.
This script asks: which 7-dim rep is the natural C^7 here, and what
does Lambda^2 of it look like as a 2T module?

Construction.  2T is a subgroup of G_2 (exceptional Lie group,
automorphisms of the octonions) via Hurwitz quaternions acting on
the octonion imaginary part.  G_2 has a natural 7-dim rep (the
imaginary octonions).  Restricting to 2T gives a specific 7-dim
representation of 2T.  We compute it explicitly.

Steps:
  (1) Rebuild a 2T subgroup inside 2I (using b2.3a's find_2T_in).
  (2) Compute 2T's conjugacy classes directly from the 24 matrices,
      verify 7 classes with sizes (1, 1, 6, 4, 4, 4, 4).
  (3) Compute the 2T character table.
  (4) Build the 7-dim rep of 2T from the Hurwitz-quaternion action
      on imaginary octonions (7-dim over R).
  (5) Decompose this 7-dim rep into 2T irreps.
  (6) Compute Lambda^2 of the 7-dim rep, decompose into 2T irreps,
      compare the irrep-decomposition data to the integer 21 =
      dim(Lambda^2 C^7) and to the physical ansatz (all 21 'crossings'
      correspond to 21 distinct irrep channels).
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nwt_poincare_sphere_b2_0 import generate_2I
from nwt_2T_in_2I_check_b2_3a import find_2T_in, matrix_key


# =========================================================================
# Conjugacy classes of a finite matrix group.
# =========================================================================

def conjugacy_classes(group: list[np.ndarray]) -> list[list[int]]:
    """Compute conjugacy classes of the group.  Return a list of
    lists of indices into `group`."""
    n = len(group)
    assigned = [False] * n
    classes = []
    for i in range(n):
        if assigned[i]:
            continue
        cls = [i]
        assigned[i] = True
        gi = group[i]
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            gj = group[j]
            # Is gi conjugate to gj?
            for h in group:
                # Check h @ gi @ h^(-1) == gj
                hgi = h @ gi @ h.conj().T
                if np.allclose(hgi, gj, atol=1e-8):
                    cls.append(j)
                    assigned[j] = True
                    break
        classes.append(cls)
    return classes


# =========================================================================
# Hurwitz-quaternion action on imaginary octonions (7-dim rep).
#
# Octonions O = R <1, e_1, ..., e_7> with Cayley-Dickson multiplication.
# We take a specific choice of basis where e_1, e_2, e_3 span the
# "quaternion" sub-algebra and e_4 is the "anti-involutive" extension.
#
# The imaginary octonions Im(O) = R^7 with basis {e_1, ..., e_7}.
# 2T = {Hurwitz quaternions} acts on O by left multiplication, which
# preserves R (real part) and Im(O) separately.  On Im(O), 2T acts
# by a 7-dim real representation.
# =========================================================================

# Octonion multiplication table (Fano plane based).
# e_i * e_j = epsilon_{ijk} e_k  for specific signed triples.
# Standard: {e_1 e_2 e_3}, {e_1 e_4 e_5}, {e_1 e_7 e_6},
#           {e_2 e_4 e_6}, {e_2 e_5 e_7}, {e_3 e_4 e_7}, {e_3 e_6 e_5}
# where order gives sign: e_i * e_j = +e_k in the order listed.
OCTONION_TRIPLES = [
    (1, 2, 3), (1, 4, 5), (1, 7, 6),
    (2, 4, 6), (2, 5, 7), (3, 4, 7), (3, 6, 5),
]


def octonion_structure_constants():
    """Return the 7x7x7 structure tensor c_{ijk} of imaginary
    octonions: e_i * e_j = c_{ijk} e_k  (plus diagonal e_i * e_i = -1
    handled separately)."""
    c = np.zeros((7, 7, 7))
    for trip in OCTONION_TRIPLES:
        i, j, k = trip
        i -= 1; j -= 1; k -= 1      # 1-indexed to 0-indexed
        c[i, j, k] = 1
        c[j, k, i] = 1
        c[k, i, j] = 1
        c[j, i, k] = -1
        c[k, j, i] = -1
        c[i, k, j] = -1
    return c


def hurwitz_to_octonion_vec(q_SU2: np.ndarray) -> np.ndarray:
    """Convert an SU(2) Hurwitz quaternion to an 8-component real
    octonion vector (a, b, c, d, 0, 0, 0, 0) where a+bi+cj+dk is the
    quaternion (identified with e_0, e_1, e_2, e_3 of octonions)."""
    a = q_SU2[0, 0].real
    b = q_SU2[0, 0].imag
    c = q_SU2[0, 1].real
    d = q_SU2[0, 1].imag
    return np.array([a, b, c, d, 0, 0, 0, 0])


def octonion_left_mult_matrix(q_vec: np.ndarray,
                                c_struct: np.ndarray) -> np.ndarray:
    """Return the 8x8 matrix L_q of left multiplication by octonion
    q on the 8-dim space (1, e_1, ..., e_7)."""
    L = np.zeros((8, 8))
    a0 = q_vec[0]
    q_im = q_vec[1:]       # (q_1, ..., q_7)
    # 1 * x = x  (real part behaves as identity scalar)
    L[0, 0] = a0
    # Re(L_q x) = a0 * Re(x) - <q_im, x_im>
    L[0, 1:] = -q_im
    # Im(L_q x) = a0 * x_im + Re(x) * q_im + q_im * x_im (via c tensor)
    for i in range(7):
        L[i + 1, 0] = q_im[i]
        L[i + 1, 1:] = a0 * np.eye(7)[i, :] \
            + np.einsum('j,jk->k', q_im, c_struct[:, i, :])
    return L


def seven_dim_rep_on_2T(T24: list[np.ndarray]) -> list[np.ndarray]:
    """Build the 7-dim real rep of 2T on imaginary octonions, by
    taking each element, representing it as an octonion, and taking
    the 7x7 block acting on Im(O).
    """
    c_struct = octonion_structure_constants()
    reps = []
    for q_SU2 in T24:
        q_vec = hurwitz_to_octonion_vec(q_SU2)
        L8 = octonion_left_mult_matrix(q_vec, c_struct)
        # Im(O) is the last 7 dimensions.
        rep7 = L8[1:, 1:]
        reps.append(rep7)
    return reps


# =========================================================================
# Character and irrep decomposition.
# =========================================================================

def rep_character(rep_matrices: list[np.ndarray],
                   class_reps: list[int]) -> np.ndarray:
    """Character of a representation at each conjugacy class
    representative: chi(class_i) = Tr(rep(class_reps[i]))."""
    return np.array([np.trace(rep_matrices[ci]).real
                     for ci in class_reps])


def known_2T_characters():
    """Return the standard 2T character table.
    Rows = irreps, cols = conjugacy classes in the order:
      C1 = {e}            (size 1)
      C2 = {-e}           (size 1)
      C3 = 6 order-4      (size 6)
      C4 = 4 order-6 A    (size 4)
      C5 = 4 order-6 B    (size 4)
      C6 = 4 order-3 A    (size 4)
      C7 = 4 order-3 B    (size 4)
    Irreps: trivial (1), omega (1), omega^2 (1), 2, 2', 2'',
    and the 3-dim (which is the adjoint of SU(2)/Z_2 = SO(3)).
    """
    omega = np.exp(2j * np.pi / 3)
    # From Springer's Linear Algebraic Groups or ATLAS of finite groups:
    # Using a standard choice of omega (primitive cube root of unity).
    # Class sizes: 1, 1, 6, 4, 4, 4, 4.
    chars = np.array([
        [1, 1, 1, 1, 1, 1, 1],                               # trivial
        [1, 1, 1, omega, omega**2, omega**2, omega],         # omega-chi
        [1, 1, 1, omega**2, omega, omega, omega**2],         # omega^2-chi
        [2, -2, 0, 1, -1, 1, -1],                             # 2 (SU(2) fund)
        [2, -2, 0, omega, -omega**2, omega**2, -omega],      # 2' (SU(2)⊗omega)
        [2, -2, 0, omega**2, -omega, omega, -omega**2],      # 2'' (SU(2)⊗omega²)
        [3, 3, -1, 0, 0, 0, 0],                               # 3 (adjoint)
    ], dtype=complex)
    sizes = np.array([1, 1, 6, 4, 4, 4, 4])
    return chars, sizes


def decompose_character(chi: np.ndarray, irrep_chars: np.ndarray,
                          class_sizes: np.ndarray) -> np.ndarray:
    """Compute multiplicities m_i of each irrep in a rep with character chi.

    m_i = (1/|G|) sum_c |C_c| * chi(C_c) * conj(chi_i(C_c)).
    """
    G = int(class_sizes.sum())
    m = np.zeros(irrep_chars.shape[0], dtype=complex)
    for i in range(irrep_chars.shape[0]):
        m[i] = (class_sizes * chi * np.conj(irrep_chars[i])).sum() / G
    return m


# =========================================================================
# Lambda^2 character: chi_Lambda2 (g) = (chi(g)^2 - chi(g^2)) / 2.
# =========================================================================

def lambda2_character(rep_matrices: list[np.ndarray],
                       class_reps: list[int]) -> np.ndarray:
    """Return chi_{Lambda^2 rho}(C_i) from chi_rho."""
    chi = rep_character(rep_matrices, class_reps)
    # Need chi(g^2) at each class representative.
    chi_sq = np.zeros_like(chi, dtype=float)
    for i, ci in enumerate(class_reps):
        g = rep_matrices[ci]
        g2 = g @ g
        chi_sq[i] = np.trace(g2).real
    return (chi ** 2 - chi_sq) / 2.0


# =========================================================================
# Main.
# =========================================================================

def main():
    print("=" * 72)
    print(" b2.4 -- 2T character table, 7-dim rep, Lambda^2 C^7")
    print("=" * 72)

    # --- (1) Find 2T in 2I -----------------------------------------------
    twoI = generate_2I()
    T24 = find_2T_in(twoI)
    assert T24 is not None and len(T24) == 24
    print("\n[1] 2T found inside 2I (24 matrices)")

    # --- (2) Conjugacy classes of this 2T ------------------------------
    print("\n[2] Conjugacy classes of 2T")
    classes = conjugacy_classes(T24)
    sizes = [len(c) for c in classes]
    print(f"    Number of classes: {len(classes)}")
    print(f"    Class sizes (sorted): {sorted(sizes)}")
    print(f"    Expected: [1, 1, 4, 4, 4, 4, 6]")
    assert sorted(sizes) == [1, 1, 4, 4, 4, 4, 6]
    # Label classes by a representative each, sort in a standard order
    # so that:
    #   class 0 : {e}  (trace 2)
    #   class 1 : {-e} (trace -2)
    #   class 2 : 6 order-4 (trace 0)
    #   classes 3,4: 4 order-6 (trace +1)
    #   classes 5,6: 4 order-3 (trace -1)
    reps = [c[0] for c in classes]
    class_traces = [round(np.trace(T24[r]).real, 3) for r in reps]
    order = sorted(range(len(classes)),
                    key=lambda i: (-class_traces[i], sizes[i],
                                   reps[i]))
    # Re-sort so that the primary ordering is by trace desc, size asc.
    # Actually it doesn't matter for our checks as long as we track it.
    for idx, i in enumerate(order):
        print(f"    class {idx}: size {sizes[i]:2d}, trace "
              f"{class_traces[i]:+6.3f}, rep index {reps[i]}")

    class_reps = [reps[i] for i in order]
    class_sizes = np.array([sizes[i] for i in order])
    # Ensure the ordering matches the known character table's ordering:
    # (size 1, trace 2), (size 1, trace -2), (size 6, trace 0),
    # (size 4, trace 1)x2, (size 4, trace -1)x2
    expected_order_traces = [2, -2, 0, 1, 1, -1, -1]
    actual_order_traces = [class_traces[i] for i in order]
    if actual_order_traces != expected_order_traces:
        print(f"    WARNING: class ordering differs from standard table")
        print(f"    actual traces: {actual_order_traces}")
        print(f"    expected:      {expected_order_traces}")

    # --- (3) 2T character table (known) --------------------------------
    print("\n[3] 2T character table (known, standard form)")
    irrep_chars, _ = known_2T_characters()
    irrep_dims = [int(round(c[0].real)) for c in irrep_chars]
    irrep_names = ["triv", "chi_omega", "chi_omega^2",
                    "2", "2'", "2''", "3"]
    print(f"    Irreps: {irrep_names}")
    print(f"    Dimensions: {irrep_dims}  "
          f"(sum of squares = "
          f"{sum(d*d for d in irrep_dims)})")
    # Print the character table in tabular form.
    print(f"    Characters (rows = irreps, cols = classes):")
    hdr = "               " + "  ".join(f"C{i}" for i in range(7))
    print(hdr)
    for name, row in zip(irrep_names, irrep_chars):
        vals = "  ".join(f"{c.real:+.2f}{c.imag:+.2f}i"
                         if abs(c.imag) > 1e-8 else f"  {c.real:+.2f}   "
                         for c in row)
        print(f"    {name:8s} :  {vals}")

    # --- (4) 7-dim rep of 2T via octonion imaginary part --------------
    print("\n[4] 7-dim rep of 2T from Hurwitz-quaternion action on Im(O)")
    # Build 24 Hurwitz quaternions explicitly (not the 2I-embedding's
    # 2T, because the octonion construction uses canonical quaternion axes).
    from itertools import product as it_product
    Hurwitz = []
    for axis in range(4):
        for sign in (+1, -1):
            q = [0.0, 0.0, 0.0, 0.0]
            q[axis] = float(sign)
            Hurwitz.append(np.array([
                [q[0] + 1j * q[1],  q[2] + 1j * q[3]],
                [-q[2] + 1j * q[3], q[0] - 1j * q[1]],
            ], dtype=complex))
    for signs in it_product([+1, -1], repeat=4):
        q = [s / 2.0 for s in signs]
        Hurwitz.append(np.array([
            [q[0] + 1j * q[1],  q[2] + 1j * q[3]],
            [-q[2] + 1j * q[3], q[0] - 1j * q[1]],
        ], dtype=complex))
    assert len(Hurwitz) == 24
    reps_7d = seven_dim_rep_on_2T(Hurwitz)

    # Verify that this IS a representation of 2T: products map correctly.
    # Just spot check one pair.
    gi, gj = Hurwitz[5], Hurwitz[13]
    gij = gi @ gj
    # Find index of gij in Hurwitz
    def find_idx(M, list_of_M, tol=1e-6):
        for idx, L in enumerate(list_of_M):
            if np.allclose(M, L, atol=tol):
                return idx
        return None
    idx_ij = find_idx(gij, Hurwitz)
    assert idx_ij is not None
    product_rep = reps_7d[5] @ reps_7d[13]
    expected_rep = reps_7d[idx_ij]
    agree = np.allclose(product_rep, expected_rep, atol=1e-6)
    print(f"    Representation homomorphism check: "
          f"{'PASS' if agree else 'FAIL'}")

    # Conjugacy classes of Hurwitz (should match 2T's: sizes 1,1,6,4,4,4,4).
    H_classes = conjugacy_classes(Hurwitz)
    H_sizes = sorted(len(c) for c in H_classes)
    print(f"    Hurwitz conjugacy class sizes: {H_sizes}")
    assert H_sizes == [1, 1, 4, 4, 4, 4, 6]

    # Get character of the 7-dim rep at each conjugacy class.
    # Need class ordering matching known_2T_characters.
    H_reps = [c[0] for c in H_classes]
    H_traces = [round(np.trace(Hurwitz[r]).real, 3) for r in H_reps]
    # Sort to match expected order: (trace +2), (trace -2), (size 6, trace 0),
    #    (size 4 trace +1) x 2, (size 4 trace -1) x 2
    def sort_key(i):
        cls = H_classes[i]
        tr = round(np.trace(Hurwitz[cls[0]]).real, 3)
        # Need a tie-breaker for the 2 order-6 and 2 order-3 classes
        # with the same trace.  Use order of first element.
        return (-tr, len(cls), cls[0])
    H_order = sorted(range(len(H_classes)), key=sort_key)
    H_class_reps = [H_reps[i] for i in H_order]

    chi_7d = rep_character(reps_7d, H_class_reps)
    print(f"    chi(7-dim rep) at classes = {chi_7d.astype(float).round(3)}")

    # Class sizes in same order
    H_class_sizes = np.array([len(H_classes[i]) for i in H_order])

    # Decompose chi_7d into 2T irreps.
    mult_7d = decompose_character(chi_7d, irrep_chars, H_class_sizes)
    mult_7d_real = np.array([m.real for m in mult_7d]).round(4)
    print(f"    Multiplicities in 2T irreps:")
    for name, m in zip(irrep_names, mult_7d):
        print(f"      {name:10s}: {m.real:+6.3f} "
              f"{'(+ ' + str(round(m.imag, 3)) + 'i)' if abs(m.imag) > 1e-4 else ''}")
    print(f"    Dim check: sum m_i * dim(irrep_i) = "
          f"{sum(m.real * d for m, d in zip(mult_7d, irrep_dims)):.2f}  "
          f"(expected 7)")

    # --- (5) Lambda^2 of the 7-dim rep, decompose --------------------
    print("\n[5] Lambda^2(C^7) decomposition under 2T")
    chi_lam2 = lambda2_character(reps_7d, H_class_reps)
    print(f"    chi(Lambda^2) at classes = {chi_lam2.round(3)}")
    mult_lam2 = decompose_character(chi_lam2, irrep_chars, H_class_sizes)
    print(f"    Multiplicities in 2T irreps:")
    for name, m in zip(irrep_names, mult_lam2):
        print(f"      {name:10s}: {m.real:+6.3f} "
              f"{'(+ ' + str(round(m.imag, 3)) + 'i)' if abs(m.imag) > 1e-4 else ''}")
    print(f"    Dim check: sum m_i * dim(irrep_i) = "
          f"{sum(m.real * d for m, d in zip(mult_lam2, irrep_dims)):.2f}  "
          f"(expected 21)")

    # --- (6) Physical interpretation ----------------------------------
    print()
    print("=" * 72)
    print(" INTERPRETATION")
    print("=" * 72)
    print("""
    The 7-dim rep Im(O) of 2T, restricted from G_2 ⊂ SO(7), decomposes
    into specific 2T irreps (reported above).  Its antisymmetric square
    Lambda^2 C^7 has dimension 21 and decomposes into 2T irreps shown.
    These 21 'components' are the candidate 21 crossing channels of
    Paper 15's  m_e / m_Pl = (8/7)(1+alpha/7) alpha^(21/2).

    Per Paper 13's rule of sqrt(alpha) per crossing on a transition
    amplitude, the total contribution is

        A_{m_e/m_Pl}  ~  product over 21 channels of sqrt(alpha_i)
                     ~  alpha^(21/2)

    IF each channel contributes a single sqrt(alpha) factor.

    The structural claim now has a concrete representation-theoretic
    content: 21 = dim(Lambda^2 Im(O)) as a 2T module, with specific
    irrep decomposition serving as the bookkeeping of the crossings.

    NEXT STEP (b2.5): build an explicit amplitude model where each
    of these 21 irrep channels delivers one sqrt(alpha) factor via
    the Aharonov-Bohm / gauge-holonomy mechanism of Papers 13-14.
    """)


if __name__ == "__main__":
    main()
