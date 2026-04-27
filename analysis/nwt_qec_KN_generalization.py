#!/usr/bin/env python3
"""
NWT QEC Generalization to K_N
==============================

The K_7 graph state numerically generates Paper 17 bracket coefficients
exactly (`nwt_qec_bracket_test.py`).  This script tests the cross-group
prediction:  does the same structural identity hold for K_N with so(N)
for N = 9, 11?

For B_n = so(2n+1):
    dim(V)   = 2n+1
    dim(Adj) = n(2n+1) = (2n+1)(2n+1 - 1)/2 = C(2n+1, 2)   ✓ matches |E(K_{2n+1})|
    rank     = n
    dim(S)   = 2^n
    h^∨      = 2n - 1

Predictions for the bracket:
    LO  coef = 1
    NLO coef = 1/dim(V) = 1/(2n+1)
    NNLO coef = dim(Adj)/dim(V) = n  (= rank)

For so(7) [n=3]:  bracket = 1 + α/7 + 3α²       ← Paper 17
For so(9) [n=4]:  bracket = 1 + α/9 + 4α²       ← prediction
For so(11)[n=5]:  bracket = 1 + α/11 + 5α²      ← prediction

Cartan-graded subspace prediction:
  Fix (2n+1 - n) = n+1 stabilizers → 2^(2n+1 - (n+1)) = 2^n = dim(S) subspace.

For the prefactor, two candidates that COINCIDE for n=3 but diverge for n>=4:

  QEC reading:  prefactor = dim(S) / (dim(S) - 1) = 2^n / (2^n - 1)
                so(7):  8/7        so(9):  16/15        so(11):  32/31

  Lie reading:  prefactor = dim(S) / dim(V) = 2^n / (2n+1)
                so(7):  8/7        so(9):  16/9         so(11):  32/11

The K_9 graph state probe is the decisive test.

Hilbert space dimensions: K_7 = 128, K_9 = 512, K_11 = 2048.  All numpy-tractable.
"""

from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import List


# Pauli matrices
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)


def pauli_op_KN(positions: List[int], ops: List[np.ndarray], N: int) -> np.ndarray:
    """Tensor product on N qubits."""
    op = np.array([[1.0]], dtype=complex)
    pos_to_op = dict(zip(positions, ops))
    for q in range(N):
        op = np.kron(op, pos_to_op.get(q, I2))
    return op


def expect(state: np.ndarray, op: np.ndarray) -> complex:
    return np.vdot(state, op @ state)


def KN_graph_state(N: int) -> np.ndarray:
    """|K_N> = prod_{(u,v) in E(K_N)} CZ_{u,v} |+>^N."""
    DIM = 2**N
    state = np.ones(DIM, dtype=complex) / np.sqrt(DIM)
    for u, v in combinations(range(N), 2):
        for i in range(DIM):
            bit_u = (i >> (N - 1 - u)) & 1
            bit_v = (i >> (N - 1 - v)) & 1
            if bit_u == 1 and bit_v == 1:
                state[i] *= -1
    return state


def stabilizer_S(v: int, N: int) -> np.ndarray:
    """S_v = X_v * prod_{u != v} Z_u."""
    ops, positions = [], []
    for q in range(N):
        positions.append(q)
        ops.append(X if q == v else Z)
    return pauli_op_KN(positions, ops, N)


def section(s: str) -> None:
    print()
    print("=" * 78)
    print(" " + s)
    print("=" * 78)


def test_KN(N: int) -> dict:
    """Run the full bracket-coefficient + prefactor test on K_N."""

    # so(N) data (assuming N = 2n+1)
    assert N % 2 == 1, "N must be odd for so(N)"
    n = (N - 1) // 2

    dim_V    = N
    dim_Adj  = N * (N - 1) // 2     # = C(N,2)
    rank_son = n
    dim_S    = 2**n

    DIM_HILBERT = 2**N

    section(f"Testing K_{N}  -- so({N}) gauge group, n = {n}")

    print(f"  Lie-algebra data:")
    print(f"    dim(V)    = {dim_V}     (vector)")
    print(f"    dim(Adj)  = {dim_Adj}     (adjoint = C(N,2) = |E(K_N)|)")
    print(f"    rank      = {rank_son}     (Cartan dim)")
    print(f"    dim(S)    = {dim_S}     (spinor)")
    print(f"    Hilbert dim = 2^{N} = {DIM_HILBERT}")

    # Build state
    state = KN_graph_state(N)
    norm = np.linalg.norm(state)
    print(f"\n  |K_{N}> graph state built, norm = {norm:.6f}")

    # Verify stabilizers
    print(f"\n  Stabilizer eigenvalue check (should all be +1):")
    all_ok = True
    for v in range(N):
        S_v = stabilizer_S(v, N)
        ev = expect(state, S_v)
        if abs(ev - 1.0) > 1e-9:
            all_ok = False
            print(f"    S_{v}: {ev}  FAIL")
    print(f"    All {N} stabilizers verified at +1 eigenvalue: {'OK' if all_ok else 'FAIL'}")

    # Build H_YY = sum over edges Y_u Y_v
    H_YY = sum(pauli_op_KN([u, v], [Y, Y], N)
               for u, v in combinations(range(N), 2))

    # Compute moments <H_YY^n>
    print(f"\n  Moments <H_YY^k> on |K_{N}>:")
    print(f"    {'k':<4} {'<H_YY^k>':>14} {'predicted dim(Adj)^k':>22} {'match':>8}")
    moments = []
    H_YY_pow = np.eye(DIM_HILBERT, dtype=complex)
    for k in range(5):
        if k > 0:
            H_YY_pow = H_YY_pow @ H_YY
        m = expect(state, H_YY_pow).real
        moments.append(m)
        predicted = dim_Adj ** k
        match = "OK" if abs(m - predicted) < 1e-3 * max(predicted, 1) else "FAIL"
        print(f"    {k:<4} {m:>14.4f} {predicted:>22d} {match:>8}")

    # Bracket coefficients prediction
    bracket_alpha    = 1.0 / dim_V         # NLO coefficient
    bracket_alpha_sq = dim_Adj / dim_V     # NNLO coefficient

    print(f"\n  Predicted bracket = 1 + α/dim(V) + dim(Adj)/dim(V) × α²")
    print(f"                  = 1 + α/{dim_V} + {bracket_alpha_sq:.0f}α²")

    # Cartan-graded subspace probe
    section(f"Cartan-graded subspace of |K_{N}>")

    n_fix = N - rank_son   # Fix (N - rank) stabilizers
    expected_dim = 2**(N - n_fix)
    print(f"\n  Fix {n_fix} = (N - rank({N})) = {N} - {rank_son} stabilizers at +1")
    print(f"  Expected joint +1 eigenspace dim = 2^(N - {n_fix}) = 2^{rank_son} = {expected_dim}")
    print(f"  Compare: dim(S_so({N})) = 2^{rank_son} = {dim_S}")

    # Build the projector onto joint +1 eigenspace
    P = np.eye(DIM_HILBERT, dtype=complex)
    for v in range(n_fix):
        S_v = stabilizer_S(v, N)
        P = P @ (np.eye(DIM_HILBERT, dtype=complex) + S_v) / 2

    actual_dim = int(round(np.real(np.trace(P))))
    print(f"\n  Computed dim(joint +1 eigenspace) = {actual_dim}")
    print(f"  Match dim(S) prediction? {'OK' if actual_dim == dim_S else 'FAIL'}")

    # Verify |K_N> is in this subspace
    overlap = np.linalg.norm(P @ state)
    print(f"\n  ||P_S |K_{N}>||  =  {overlap:.6f}   (should be 1.0 if |K_N> in S)")

    # The two candidate prefactors
    print(f"\n  Two candidate prefactors at K_{N}:")
    print(f"    QEC reading:  dim(S) / (dim(S)-1) = {dim_S}/{dim_S-1} = {dim_S/(dim_S-1):.6f}")
    print(f"    Lie reading:  dim(S) / dim(V) = {dim_S}/{dim_V} = {dim_S/dim_V:.6f}")

    if N == 7:
        print(f"\n  At N=7, both readings give 8/7 — they coincide here.")
    else:
        ratio = (dim_S / (dim_S - 1)) / (dim_S / dim_V)
        print(f"\n  At N={N}, the two readings differ by factor {ratio:.4f}")
        print(f"  (= dim(V) / (dim(S) - 1) = {dim_V} / {dim_S - 1})")

    return dict(
        N=N, n=n, dim_V=dim_V, dim_Adj=dim_Adj, dim_S=dim_S,
        rank=rank_son, moments=moments,
        bracket=(bracket_alpha, bracket_alpha_sq),
        cartan_subspace_dim=actual_dim,
        K_N_in_subspace=overlap,
    )


def main() -> None:
    print("\n" + "=" * 78)
    print(" NWT QEC K_N Generalization Test")
    print(" Cross-group falsification of the QEC bracket-derivation reading")
    print("=" * 78)

    print("""
  Predictions for so(2n+1):
      bracket coefs:  α/dim(V) + rank · α²
      Cartan-graded subspace dim:  2^rank = dim(S)

  K_7  (n=3):  expected 1 + α/7 + 3α²,    Cartan dim 8 = dim(S)
  K_9  (n=4):  predicted 1 + α/9 + 4α²,   Cartan dim 16 = dim(S)
  K_11 (n=5):  predicted 1 + α/11 + 5α²,  Cartan dim 32 = dim(S)
""")

    results = {}

    # K_7 (sanity check, should match earlier result)
    results[7] = test_KN(7)

    # K_9 (the falsifiable cross-group test)
    results[9] = test_KN(9)

    # K_11 (additional confirmation if we have time)
    results[11] = test_KN(11)

    section("Summary")

    print(f"\n  {'N':<5} {'n':<5} {'<H_YY>':<10} {'predicted':<12} "
          f"{'<H_YY²>':<14} {'predicted':<14} {'Cartan dim':<12} {'predicted':<10}")
    print("  " + "-" * 90)
    for N, r in results.items():
        N, n = r['N'], r['n']
        adj = r['dim_Adj']
        m1 = r['moments'][1]
        m2 = r['moments'][2]
        cdim = r['cartan_subspace_dim']
        ds = r['dim_S']
        print(f"  {N:<5} {n:<5} {m1:<10.2f} {adj:<12d} "
              f"{m2:<14.2f} {adj**2:<14d} {cdim:<12d} {ds:<10d}")

    print("""
  Interpretation:

  If <H_YY^k>_{|K_N>} = dim(Adj_so(N))^k holds for ALL N tested, the
  bracket-coefficient identity is structural across the so(2n+1) family
  and not a so(7) coincidence.  The QEC reading IS the right framework.

  If the Cartan-graded subspace dim equals 2^rank = dim(S) for all N,
  the spinor-rep identification is also structural.

  At N=7, the prefactor reading is degenerate:
        dim(S)/(dim(S)-1) = dim(S)/dim(V) = 8/7

  At N=9 and N=11, these readings differ.  The QEC subspace structure
  predicts one (16/15, 32/31), the Lie rep ratio predicts another
  (16/9, 32/11).  Need to test which (if either) corresponds to the
  physical "boundary projection" claim of Paper 17.
""")


if __name__ == "__main__":
    main()
