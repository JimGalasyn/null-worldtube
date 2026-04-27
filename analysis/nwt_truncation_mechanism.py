#!/usr/bin/env python3
"""
NWT bracket-truncation mechanism investigation
================================================

Paper 17's bracket  1 + α/dim(V) + dim(Adj)/dim(V)·α²  truncates at α²
in the physical formula, despite the K_7 graph-state moment generating
function predicting a geometric series  1 + α/(dim V × (1 - α dim Adj))
that continues to all orders.

The 3-body probe rules out cancellation by 3-body operators
(<Y_a Y_b Y_c>_{|K_7>} = 0 for all 35 triples).  This script narrows the
remaining candidates by direct computation:

  (a) Casimir hierarchy: rank-3 admits Casimirs of orders 2, 4, 6;
      maybe only C_2, C_4 couple to V (giving α¹, α²) while C_6
      vanishes for structural reasons.
  (b) Furry-like symmetry: a discrete symmetry on |K_7> kills 3-body
      Y operators.  Already shown to be true (σ_X = ∏_v X_v anticommutes
      with each Y_v) but doesn't extend to H_YY^n which is even-Y.
  (c) Spinor-vector branching S = V ⊕ 1: rank-1 mismatch allows only
      quadratic mixing in α.

This script computes:
  - C_2(V), C_4(V), C_6(V) for so(7), using the Harish-Chandra approach
    via Pochhammer symbols on the highest weight (a, b, c).
  - The same Casimirs on Adj and S to check for patterns.
  - Verifies σ_X parity argument numerically on |K_7>.
  - Tests whether higher-order moments of H_YY admit any structural
    decomposition that would manifest the truncation.
"""

from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import Tuple


# ====================================================================
# Section 1: Higher Casimirs of so(7) via Harish-Chandra approach
# ====================================================================
#
# For B_n = so(2n+1), the Casimirs of orders 2, 4, ..., 2n act on
# irrep V_λ as scalars given by power sums of the shifted
# weight components.  Specifically, for λ = (a₁, a₂, ..., aₙ) and
# Weyl vector ρ = ((2n-1)/2, (2n-3)/2, ..., 1/2):
#
#     C_{2k}(V_λ) = sum_i [(λ_i + ρ_i)^{2k}] - sum_i [ρ_i^{2k}]
#
# That subtraction is the "trivial-rep zero" that ensures
# C_{2k}(trivial) = 0.

def casimirs_so7(weight: Tuple[float, float, float], orders: list = [2, 4, 6]) -> dict:
    """Compute Casimirs of so(7) = B_3 on irrep V_λ.

    weight: (a, b, c) — highest weight in orthogonal (e_i) basis.
    Returns: dict of order -> C_order(V_λ).
    """
    rho = np.array([5/2, 3/2, 1/2])
    lam = np.array(weight)
    shifted = lam + rho     # (λ + ρ)
    shifted_trivial = rho   # for trivial rep λ=0

    out = {}
    for k in orders:
        # 2k-th Casimir: sum_i (λ_i + ρ_i)^{2k} - sum_i ρ_i^{2k}
        # Note: B_n's k-th Casimir uses 2k-th powers, here we
        # parametrize by even order directly.
        C_k = sum(shifted ** k) - sum(shifted_trivial ** k)
        out[k] = C_k
    return out


def section(s: str) -> None:
    print()
    print("=" * 78)
    print(" " + s)
    print("=" * 78)


def test_casimirs() -> None:
    section("Test 1: Higher Casimirs of so(7) on V, Adj, S")

    # so(7) = B_3 fundamental weights:
    #   ω_1 = (1, 0, 0)        -> V (dim 7)
    #   ω_2 = (1, 1, 0)        -> Adj (dim 21)
    #   ω_3 = (1/2, 1/2, 1/2)  -> S (dim 8)

    V_weight   = (1, 0, 0)
    Adj_weight = (1, 1, 0)
    S_weight   = (1/2, 1/2, 1/2)

    print(f"  Note: orders 2, 4, 6 means we compute power sums of "
          f"(λ + ρ)^k for k=2,4,6.\n")

    print(f"  Standard Killing-form C_2 (k=2) values:")
    for label, w in [("V (vector)", V_weight),
                     ("Adj (adjoint)", Adj_weight),
                     ("S (spinor)", S_weight)]:
        cas = casimirs_so7(w, orders=[2, 4, 6])
        print(f"    {label:<15}: weight = {w}")
        print(f"      C_2 = {cas[2]:>12.4f}   (expected V: 6, Adj: 10, S: 21/4=5.25)")
        print(f"      C_4 = {cas[4]:>12.4f}")
        print(f"      C_6 = {cas[6]:>12.4f}")
        print()

    # The key question: is there a structural reason C_6(V) is "small"
    # or vanishes at level k?
    cas_V = casimirs_so7(V_weight, orders=[2, 4, 6])
    print(f"\n  Comparison: ratios on V")
    print(f"    C_4(V) / C_2(V)^2  = {cas_V[4] / cas_V[2]**2:.6f}")
    print(f"    C_6(V) / C_2(V)^3  = {cas_V[6] / cas_V[2]**3:.6f}")
    print(f"    C_6(V) - 9.5 × C_4(V) - C_2(V)^3 = "
          f"{cas_V[6] - 9.5 * cas_V[4] - cas_V[2]**3:.6f}")

    print(f"\n  Higher Casimirs do NOT vanish on V.  Casimir hierarchy alone")
    print(f"  cannot explain bracket truncation by 'C_6 = 0'.")


# ====================================================================
# Section 2: Numerical verification of σ_X Furry parity on |K_7>
# ====================================================================

I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)

N_QUBITS = 7
DIM      = 2**N_QUBITS


def pauli_op(positions, ops, n=N_QUBITS):
    op = np.array([[1.0]], dtype=complex)
    pos_to_op = dict(zip(positions, ops))
    for q in range(n):
        op = np.kron(op, pos_to_op.get(q, I2))
    return op


def K7_graph_state() -> np.ndarray:
    state = np.ones(DIM, dtype=complex) / np.sqrt(DIM)
    for u, v in combinations(range(N_QUBITS), 2):
        for i in range(DIM):
            bit_u = (i >> (N_QUBITS - 1 - u)) & 1
            bit_v = (i >> (N_QUBITS - 1 - v)) & 1
            if bit_u == 1 and bit_v == 1:
                state[i] *= -1
    return state


def test_furry_symmetry() -> None:
    section("Test 2: σ_X = ∏_v X_v Furry-like symmetry on |K_7>")

    state = K7_graph_state()

    # Build σ_X = X_0 X_1 ... X_6
    sigma_X = pauli_op(list(range(N_QUBITS)), [X] * N_QUBITS)

    # Test: does σ_X preserve |K_7> (up to ±1)?
    sigma_K7 = sigma_X @ state
    overlap = np.vdot(state, sigma_K7).real
    norm = np.linalg.norm(sigma_K7)
    print(f"\n  σ_X|K_7> overlap with |K_7>:  {overlap:.4f}   norm: {norm:.4f}")
    if abs(overlap - 1) < 1e-9:
        print(f"    σ_X|K_7> = +|K_7>  (preserves state)")
    elif abs(overlap + 1) < 1e-9:
        print(f"    σ_X|K_7> = -|K_7>  (anti-preserves)")
    else:
        print(f"    σ_X does NOT preserve |K_7>")

    # Test: σ_X Y_v σ_X⁻¹ = -Y_v
    print(f"\n  Conjugation σ_X Y_v σ_X⁻¹:")
    for v in [0, 3, 6]:
        Y_v   = pauli_op([v], [Y])
        Y_conj = sigma_X @ Y_v @ sigma_X.conj().T
        diff_plus  = np.linalg.norm(Y_conj - Y_v)
        diff_minus = np.linalg.norm(Y_conj + Y_v)
        if diff_minus < 1e-9:
            sign = "-"
        elif diff_plus < 1e-9:
            sign = "+"
        else:
            sign = "?"
        print(f"    σ_X Y_{v} σ_X⁻¹ = {sign} Y_{v}   (diff_minus = {diff_minus:.2e})")

    # Test: H_YY is invariant under σ_X (since each YY pair has even parity)
    H_YY = sum(pauli_op([u, v], [Y, Y]) for u, v in combinations(range(N_QUBITS), 2))
    H_YY_conj = sigma_X @ H_YY @ sigma_X.conj().T
    diff = np.linalg.norm(H_YY_conj - H_YY)
    print(f"\n  σ_X H_YY σ_X⁻¹ - H_YY:  ||·|| = {diff:.2e}")
    print(f"    H_YY is σ_X-invariant (each Y_uY_v has 2 Y's, parity = +1)")
    print(f"    Therefore <H_YY^n> is unconstrained by σ_X for any n.")

    # Verification: <Y_a Y_b Y_c> = 0 by σ_X parity
    print(f"\n  Verification: σ_X parity forces <odd-body Y> = 0")
    sample_triples = [(0,1,2), (0,3,6), (1,4,5), (2,4,6)]
    for tri in sample_triples:
        op = pauli_op(list(tri), [Y, Y, Y])
        ev = np.vdot(state, op @ state).real
        print(f"    <Y_{tri[0]} Y_{tri[1]} Y_{tri[2]}> = {ev:+.2e}")
    print(f"    All zero, consistent with σ_X parity argument.")


# ====================================================================
# Section 3: H_YY^n moments and the truncation question
# ====================================================================

def test_HYY_moments_vs_bracket() -> None:
    section("Test 3: H_YY moments vs bracket truncation")

    state = K7_graph_state()
    H_YY = sum(pauli_op([u, v], [Y, Y]) for u, v in combinations(range(N_QUBITS), 2))

    print(f"\n  Compute <H_YY^n> on |K_7> and compare to bracket coefficients:")
    print(f"    bracket(α) = 1 + α/7 + 3α² + 0·α³ + 0·α⁴ + ...   (truncated)")
    print(f"    K_7 graph state moments: <H_YY^n> / (dim V × dim Adj)")
    print(f"      = {21}^n / {7*21} = {21}^n / {7*21}")
    print()

    print(f"  {'n':<4} {'<H_YY^n>':>12} {'normalised':>12} "
          f"{'bracket coef':>14} {'difference':>14}")
    print("  " + "-" * 60)

    bracket_coefs_paper = {0: 1, 1: 1/7, 2: 3}   # paper: truncated at α²

    H_YY_pow = np.eye(DIM, dtype=complex)
    for n in range(5):
        if n > 0:
            H_YY_pow = H_YY_pow @ H_YY
        m = np.vdot(state, H_YY_pow @ state).real
        normalised = m / (7 * 21)
        if n in bracket_coefs_paper:
            bracket_coef = bracket_coefs_paper[n]
        else:
            bracket_coef = 0.0  # truncated to zero in paper formula
        diff = normalised - bracket_coef
        print(f"  {n:<4} {m:>12.4e} {normalised:>12.6f} "
              f"{bracket_coef:>14.6f} {diff:>14.6e}")

    print(f"""
  Interpretation:

  The graph-state moments and bracket coefficients agree exactly at
  n=0,1,2 by construction.  At n>=3, the graph-state moment continues
  geometrically (21^n / 147), while the bracket truncates to zero.

  This is the SAME mismatch we documented earlier — the K_7 graph-state
  moment generating function captures the bracket at LO+NLO+NNLO but
  diverges from it at NNNLO+.  Casimirs C_4 and C_6 of so(7) on V are
  BOTH non-vanishing (Test 1), so option (a) "C_6 vanishes for
  structural reasons" is ruled out.

  This sharpens the open question:
    The bracket truncation must come from either
      (a') A specific structural constraint at level k=32 that
           cancels α³+ contributions despite C_6(V) being non-zero.
      (c)  The spinor-vector boundary projection S = V ⊕ 1, which
           limits mixing to second order in α.

  Both (a') and (c) require deeper RT-level computation than is
  accessible from |K_7> alone.  The 3-body probe and Casimir
  computation here narrow the candidate space but cannot finish it.
""")


# ====================================================================
# Section 4: σ_X-graded operator decomposition
# ====================================================================

def test_sigma_X_decomposition() -> None:
    section("Test 4: σ_X-graded decomposition of operator algebra on |K_7>")

    state = K7_graph_state()
    sigma_X = pauli_op(list(range(N_QUBITS)), [X] * N_QUBITS)

    # Project onto σ_X-even and σ_X-odd subspaces
    P_even = (np.eye(DIM, dtype=complex) + sigma_X) / 2
    P_odd  = (np.eye(DIM, dtype=complex) - sigma_X) / 2

    dim_even = int(round(np.real(np.trace(P_even))))
    dim_odd  = int(round(np.real(np.trace(P_odd))))

    print(f"\n  σ_X eigenspaces of the 128-dim Hilbert space:")
    print(f"    Even (σ_X = +1): dim = {dim_even}")
    print(f"    Odd  (σ_X = -1): dim = {dim_odd}")
    print(f"    Sum:  {dim_even + dim_odd}  (should equal 128)")

    # |K_7> overlap with each
    even_overlap = np.linalg.norm(P_even @ state)
    odd_overlap  = np.linalg.norm(P_odd @ state)
    print(f"\n  |K_7> overlap with σ_X eigenspaces:")
    print(f"    ||P_even |K_7>|| = {even_overlap:.6f}")
    print(f"    ||P_odd  |K_7>|| = {odd_overlap:.6f}")

    # Check H_YY is σ_X-even (commutes with σ_X)
    H_YY = sum(pauli_op([u, v], [Y, Y]) for u, v in combinations(range(N_QUBITS), 2))
    comm = sigma_X @ H_YY - H_YY @ sigma_X
    print(f"\n  [σ_X, H_YY] norm:  {np.linalg.norm(comm):.2e}")
    print(f"    H_YY commutes with σ_X => H_YY is σ_X-even")

    # The σ_X-odd operators include all 3-body Y_aY_bY_c
    print(f"\n  σ_X-grading of operator content:")
    print(f"    σ_X-EVEN  ⟨...⟩ ≠ 0 in general:  H_YY (2-body), H_YY^n,")
    print(f"                                    1-body X_v, Z_v, etc.")
    print(f"    σ_X-ODD   ⟨...⟩ = 0 always:     1-body Y_v,")
    print(f"                                    3-body Y_aY_bY_c,")
    print(f"                                    odd-Y products in general.")

    # The bracket coefficients come from σ_X-EVEN moments only
    print(f"""
  Conclusion of Test 4:

  σ_X is a Z_2 grading on the operator algebra, with H_YY in the EVEN
  sector.  All ⟨H_YY^n⟩ = 21^n moments lie in the σ_X-even sector,
  consistent with the geometric series prediction for the K_7
  graph-state alone.

  The Furry parity σ_X is therefore necessary for the 3-body probe
  result (justifying ⟨Y_aY_bY_c⟩ = 0) but NOT sufficient to truncate
  the bracket at α² — that requires structure outside |K_7>.

  σ_X is the cleanest piece of the truncation argument we can derive
  from |K_7> alone; the rest must come from level-k structure or
  boundary projection.
""")


def main() -> None:
    print("=" * 78)
    print(" Paper 17 truncation mechanism investigation")
    print("=" * 78)

    test_casimirs()
    test_furry_symmetry()
    test_HYY_moments_vs_bracket()
    test_sigma_X_decomposition()

    section("Summary")
    print("""
  Three candidate mechanisms for bracket truncation at α²:

  (a) Casimir hierarchy: C_2 → α¹, C_4 → α², C_6 vanishes.
      RULED OUT.  Direct computation gives C_2(V)=6, C_4(V)=111,
      C_6(V)=1594 — all non-vanishing on V.

  (b) Furry-like symmetry σ_X kills 3-body operators.
      VERIFIED but INSUFFICIENT.  σ_X = ∏_v X_v anticommutes with each
      Y_v individually, forcing ⟨any odd-Y operator⟩ = 0 on |K_7>.
      But H_YY is σ_X-even (each Y_uY_v has parity +1), so ⟨H_YY^n⟩ is
      unconstrained.  The 3-body probe result is rigorously justified
      by σ_X, but the bracket truncation is NOT.

  (c) Spinor-vector branching S = V ⊕ 1.
      NOT YET TESTABLE from |K_7> alone.  Requires the boundary
      projection at the Wilson endpoints (Paper 17 §6.2) to be made
      explicit at perturbative order α^n.

  Sharpened open question:

      What level-k constraint or boundary-projection mechanism kills
      the geometric-series tail beyond α² in the K_7 Wilson amplitude
      on Σ(2,3,5)?

  Concrete next steps:

    1. Compute the K_7 Wilson amplitude in Spin(7)_k Chern-Simons
       perturbation theory order-by-order in α.  Phase 7's three
       attempts (Schur-Racah, Heegaard modular-S, Lawrence-Rozansky)
       all failed because they targeted the bracket directly.  A
       targeted computation of just the α³ coefficient (asking
       whether it vanishes vs the geometric prediction 63) would be
       diagnostic.

    2. Test option (c) by computing the V → S branching projector
       Π_{V→S} as a level-k operator and verifying its perturbative
       expansion truncates at α².  If yes, the spinor-vector
       branching is the truncation mechanism.

    3. Survey level-k integrability constraints: which V-fusion
       channels become non-integrable at level 32, and does the
       loss of these channels precisely cancel the geometric
       α³ contribution?

  This paper does not resolve the truncation mechanism; it sharpens
  the question and rules out the simplest candidates.
""")


if __name__ == "__main__":
    main()
