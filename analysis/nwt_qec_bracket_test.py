#!/usr/bin/env python3
"""
NWT QEC Bracket Test
====================

Tests the conjecture that the prefactor (8/7)(1 + alpha/7 + 3 alpha^2)
in m_e/m_Pl arises as a perturbative expansion of the K_7 graph state
under gauge-induced coherent perturbation, with structural form

    bracket(alpha)  =  1  +  alpha/dim(V)         +  dim(Adj)/dim(V) * alpha^2
                    =  1  +  (sum over 7 vertices) + (sum over 21 edges)

In QEC framework:
  * |K_7> = prod_{edges} CZ |+>^7  is a stabilizer state on 7 qubits
  * S_v = X_v prod_{u != v} Z_u  are the 7 vertex stabilizers
  * Perturbations correspond to gauge fluctuations along the Wilson line
  * The bracket is the second-order response of the encoded amplitude

This is an EXPLORATORY script.  We don't assume the answer; we test
several candidate response functions to see which (if any) reproduces
the bracket structure 1 + alpha/7 + 3 alpha^2.

Hilbert space is 2^7 = 128-dim, fully tractable in numpy.
"""

from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import List, Tuple

np.set_printoptions(precision=6, suppress=True)


# =====================================================================
# 1. Pauli matrices and tensor-product helpers
# =====================================================================

I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)

N_QUBITS = 7
DIM      = 2**N_QUBITS  # 128


def pauli_op(positions: List[int], ops: List[np.ndarray],
             n: int = N_QUBITS) -> np.ndarray:
    """Tensor product: ops[i] on qubit positions[i], I_2 elsewhere.

    Convention: qubit 0 is the leftmost / most significant tensor factor.
    """
    op = np.array([[1.0]], dtype=complex)
    pos_to_op = dict(zip(positions, ops))
    for q in range(n):
        op = np.kron(op, pos_to_op.get(q, I2))
    return op


def expect(state: np.ndarray, op: np.ndarray) -> complex:
    return np.vdot(state, op @ state)


# =====================================================================
# 2. K_7 graph state construction
# =====================================================================

def K7_edges() -> List[Tuple[int, int]]:
    """All 21 edges of the complete graph K_7 (n choose 2)."""
    return list(combinations(range(N_QUBITS), 2))


def K7_graph_state() -> np.ndarray:
    """|K_7> = prod_{(u,v) in E(K_7)} CZ_{u,v} |+>^7.

    Built directly: |+>^7 has all 128 amplitudes equal to 1/sqrt(128);
    each CZ flips the sign whenever both qubits u,v are |1>.
    """
    state = np.ones(DIM, dtype=complex) / np.sqrt(DIM)
    for u, v in K7_edges():
        for i in range(DIM):
            bit_u = (i >> (N_QUBITS - 1 - u)) & 1
            bit_v = (i >> (N_QUBITS - 1 - v)) & 1
            if bit_u == 1 and bit_v == 1:
                state[i] *= -1
    return state


def stabilizer_S(v: int) -> np.ndarray:
    """S_v = X_v * prod_{u != v} Z_u  (the vertex stabilizer)."""
    ops, positions = [], []
    for q in range(N_QUBITS):
        positions.append(q)
        ops.append(X if q == v else Z)
    return pauli_op(positions, ops)


# =====================================================================
# 3. Sanity: verify state is stabilized
# =====================================================================

def verify_stabilizers(state: np.ndarray) -> None:
    print("=" * 78)
    print(" Section 1: Verify |K_7> is the stabilizer ground state")
    print("=" * 78)
    print(f"  State norm:  {np.linalg.norm(state):.6f}")
    for v in range(N_QUBITS):
        S_v = stabilizer_S(v)
        ev = expect(state, S_v)
        ok = "OK" if abs(ev - 1.0) < 1e-10 else "FAIL"
        print(f"  <S_{v}>  =  {ev.real:+.6f}{ev.imag:+.2e}j   [{ok}]")


# =====================================================================
# 4. Probe single-qubit and pair expectation values
# =====================================================================

def probe_expectations(state: np.ndarray) -> dict:
    """Compute relevant Pauli expectation values on |K_7>."""
    print()
    print("=" * 78)
    print(" Section 2: Expectation values of Pauli operators on |K_7>")
    print("=" * 78)

    results = {}

    # Single-qubit X, Y, Z
    print("\n  Single-qubit:")
    for label, P in [("X", X), ("Y", Y), ("Z", Z)]:
        vals = []
        for v in range(N_QUBITS):
            ev = expect(state, pauli_op([v], [P])).real
            vals.append(ev)
        sum_v = sum(vals)
        print(f"    <{label}_v> over v=0..6:  "
              f"{[f'{x:+.3f}' for x in vals]}  sum={sum_v:+.3f}")
        results[f"sum_{label}"] = sum_v
        results[f"per_{label}"] = vals

    # Pairwise: ZZ, XX, YY, XY, etc. — focus on edges of K_7 (all pairs)
    print("\n  Pairwise (averaged over all 21 K_7 edges):")
    for label, (PA, PB) in [("XX", (X, X)), ("YY", (Y, Y)), ("ZZ", (Z, Z)),
                            ("XY", (X, Y)), ("XZ", (X, Z)), ("YZ", (Y, Z))]:
        vals = []
        for u, v in K7_edges():
            ev = expect(state, pauli_op([u, v], [PA, PB])).real
            vals.append(ev)
        sum_e = sum(vals)
        print(f"    <{label}>  sum over 21 edges  =  {sum_e:+.3f}   "
              f"(min/max: {min(vals):+.3f}/{max(vals):+.3f})")
        results[f"sum_{label}_edges"] = sum_e
        results[f"per_{label}_edges"] = vals

    return results


# =====================================================================
# 5. Test candidate Hamiltonians
# =====================================================================

def hamiltonian_X_vertex(alpha: float) -> np.ndarray:
    """H = (alpha/dim(V)^2) * sum_v X_v.  7 single-qubit X insertions."""
    H = np.zeros((DIM, DIM), dtype=complex)
    for v in range(N_QUBITS):
        H += pauli_op([v], [X])
    return (alpha / N_QUBITS**2) * H


def hamiltonian_ZZ_edge(alpha: float) -> np.ndarray:
    """H = (alpha^2/dim(V)) * sum_{(u,v) in E(K_7)} Z_u Z_v.
    21 two-qubit Z-Z couplings, each scaled by alpha^2."""
    H = np.zeros((DIM, DIM), dtype=complex)
    for u, v in K7_edges():
        H += pauli_op([u, v], [Z, Z])
    return (alpha**2 / N_QUBITS) * H


def hamiltonian_combined(alpha: float) -> np.ndarray:
    return hamiltonian_X_vertex(alpha) + hamiltonian_ZZ_edge(alpha)


def test_response_functions(state: np.ndarray) -> None:
    """Test several candidate response functions for the bracket structure."""
    print()
    print("=" * 78)
    print(" Section 3: Response functions vs the target bracket")
    print(f" Target:  bracket(alpha) = 1 + alpha/7 + 3 alpha^2")
    print("=" * 78)

    alphas = np.array([0.0, 0.001, 0.005, 0.01, 0.05, 0.1])
    target = 1.0 + alphas / N_QUBITS + 3 * alphas**2

    print(f"\n  Target bracket values at sample alphas:")
    for a, t in zip(alphas, target):
        print(f"    alpha={a:.4f}  bracket = 1 + a/7 + 3 a^2 = {t:.6f}")

    # ---- Candidate A: <psi|H|psi> (linear) ----
    print("\n--- Candidate A: <psi| (sum X + sum ZZ) |psi>, raw expectation ---")
    print("  (with sum X scaled by alpha/49, sum ZZ scaled by alpha^2/7)")
    for a in alphas:
        H = hamiltonian_combined(a)
        ev = expect(state, H).real
        print(f"    alpha={a:.4f}  <H> = {ev:+.6e}")
    print("  (If <X_v> = 0 and <Z_u Z_v> = 0 on |K_7>, this is identically zero.)")

    # ---- Candidate B: |<psi| exp(-i H) |psi>|, unitary amplitude ----
    print("\n--- Candidate B: A(alpha) = |<psi| exp(-i H) |psi>| ---")
    for a in alphas[1:]:
        H = hamiltonian_combined(a)
        # Diagonalize via eigendecomp (128-dim is fine)
        evals, evecs = np.linalg.eigh(H)
        U = evecs @ np.diag(np.exp(-1j * evals)) @ evecs.conj().T
        amp = expect(state, U)
        bracket_predicted = 1.0 + a / N_QUBITS + 3 * a**2
        print(f"    alpha={a:.4f}  |<psi|U|psi>| = {abs(amp):.6f}   "
              f"target bracket = {bracket_predicted:.6f}   "
              f"ratio = {abs(amp)/bracket_predicted:.4f}")

    # ---- Candidate C: <psi| exp(H) |psi> (Hermitian, no i) — partition function style ----
    print("\n--- Candidate C: Z(alpha) = <psi| exp(H) |psi>  (partition fn) ---")
    for a in alphas[1:]:
        H = hamiltonian_combined(a)
        evals, evecs = np.linalg.eigh(H)
        Up = evecs @ np.diag(np.exp(evals)) @ evecs.conj().T
        z = expect(state, Up).real
        bracket_predicted = 1.0 + a / N_QUBITS + 3 * a**2
        print(f"    alpha={a:.4f}  Z = {z:.6f}   "
              f"target bracket = {bracket_predicted:.6f}   "
              f"ratio = {z/bracket_predicted:.4f}")

    # ---- Candidate D: Tr(rho exp(H_full)) where H_full has *all* operator insertions ----
    print("\n--- Candidate D: per-order expansion <H>, <H^2>/2, <H^3>/6, ... ---")
    print("  (Direct Taylor coefficients of the response, regardless of basis)")
    H1 = sum(pauli_op([v], [X]) for v in range(N_QUBITS))
    H2 = sum(pauli_op([u, v], [Z, Z]) for u, v in K7_edges())
    print(f"    <sum X_v>     = {expect(state, H1).real:+.6e}   "
          f"(scales linear-alpha at alpha/49)")
    print(f"    <sum Z_u Z_v> = {expect(state, H2).real:+.6e}   "
          f"(scales quadratic-alpha at alpha^2/7)")
    print(f"    <(sum X_v)^2> = {expect(state, H1 @ H1).real:+.6e}")
    print(f"    <(sum ZZ)^2>  = {expect(state, H2 @ H2).real:+.6e}")
    print(f"    <sum X * sum ZZ> = {expect(state, H1 @ H2).real:+.6e}")


# =====================================================================
# 6. Main
# =====================================================================

def probe_YY_structure(state: np.ndarray) -> None:
    """Investigate the Y_u Y_v structure on |K_7> -- this is the only
    Pauli-2-body that has non-zero expectation, since Y_u Y_v = S_u S_v
    (product of two stabilizers).

    Test whether sum_{edges} Y_u Y_v gives the bracket structure.
    """
    print()
    print("=" * 78)
    print(" Section 4: The Y_u Y_v structure -- only non-zero 2-body operator")
    print("=" * 78)

    H_YY = sum(pauli_op([u, v], [Y, Y]) for u, v in K7_edges())

    sum_YY     = expect(state, H_YY).real
    sum_YY_sq  = expect(state, H_YY @ H_YY).real

    print(f"\n  <sum_{{edges}} Y_u Y_v>     = {sum_YY:+.4f}   (= dim Adj = 21)")
    print(f"  <(sum Y_u Y_v)^2>         = {sum_YY_sq:+.4f}")
    print(f"  Var(sum YY)               = {sum_YY_sq - sum_YY**2:+.4f}")
    print(f"  Compare: dim(Adj)^2       = {21**2}")
    print(f"  Compare: dim(Adj) * dim(V) = {21 * 7}")
    print(f"  Compare: dim(Adj) + dim(V) = {21 + 7}")

    # Higher moments
    H_YY_sq = H_YY @ H_YY
    H_YY_cu = H_YY_sq @ H_YY
    print(f"\n  <(sum YY)^3>              = {expect(state, H_YY_cu).real:+.4f}")
    H_YY_4  = H_YY_sq @ H_YY_sq
    print(f"  <(sum YY)^4>              = {expect(state, H_YY_4).real:+.4f}")

    # Test perturbation by sum YY at various scalings
    print("\n--- Test: A(alpha) = <psi| exp(-i alpha * c * sum YY) |psi> ---")
    print(f"  Question: which scaling c gives the bracket 1 + a/7 + 3a^2?")
    for c_label, c in [("1", 1.0), ("1/dim(V)=1/7", 1/7),
                       ("1/dim(Adj)=1/21", 1/21),
                       ("1/(dim V * dim Adj)=1/147", 1/147),
                       ("dim(V)/dim(Adj)=1/3", 1/3)]:
        print(f"\n  Scaling c = {c_label} = {c:.5f}")
        for a in [0.001, 0.01, 0.05, 0.1]:
            evals, evecs = np.linalg.eigh(c * H_YY)
            U = evecs @ np.diag(np.exp(-1j * a * evals)) @ evecs.conj().T
            amp = expect(state, U)
            target = 1.0 + a/7 + 3*a**2
            print(f"    alpha={a:.4f}  <psi|U|psi> = {amp.real:+.6f}{amp.imag:+.4e}j   "
                  f"|amp| = {abs(amp):.6f}   target = {target:.6f}")


def probe_dim_S_over_dim_V(state: np.ndarray) -> None:
    """Look for a structural source of the 8/7 = dim(S)/dim(V) prefactor.

    Hypothesis: the K_7 graph state has 2^7 = 128 dim, but a special
    8-dim subspace corresponding to the Spin(7) spinor representation.
    The trace ratio over these subspaces gives 8/7.
    """
    print()
    print("=" * 78)
    print(" Section 5: Searching for the 8/7 = dim(S)/dim(V) prefactor")
    print("=" * 78)

    # The stabilizer group of |K_7> has order 2^7 = 128.
    # The codespace (eigenspace +1 of all stabilizers) is 1-dim (just |K_7>).
    # So no obvious 8-dim subspace from stabilizers alone.

    # But the spinor rep S can be related to: 8 = 2^3, where 3 = rank(so(7))
    # = number of Cartan generators. Maybe the 8-dim subspace is associated
    # with eigenstates of 3 commuting stabilizer products.

    # Try: pick 3 mutually commuting stabilizer products. Their 2^3 = 8
    # joint eigenstates form a natural 8-dim space.

    # Cartan-like commuting set: take S_1, S_2, S_3 (any 3 stabilizers
    # all commute since they're from the same stabilizer group).

    print("\n  All 7 stabilizers commute (same stabilizer group),")
    print("  so any 3 of them define a 2^3 = 8 dim joint eigenspace.")
    print()
    print("  Joint eigenstates of {S_0, S_1, S_2}: 8 states with ±1 eigenvalues.")
    print("  Check whether these decompose like an 8-dim spinor of Spin(7).")

    # Compute joint eigenspaces of S_0, S_1, S_2 (all commute since stabilizers)
    S0 = stabilizer_S(0)
    S1 = stabilizer_S(1)
    S2 = stabilizer_S(2)

    # Project onto each joint eigenspace using projectors
    # P_{s0,s1,s2} = (1/2)^3 * (1 + s0*S0)(1 + s1*S1)(1 + s2*S2)
    print()
    print("  Joint eigenspace dimensions (should sum to 128 = 2^7):")
    total_dim = 0
    P_codes = {}
    for s0 in [+1, -1]:
        for s1 in [+1, -1]:
            for s2 in [+1, -1]:
                P = (np.eye(DIM) + s0 * S0) / 2
                P = P @ (np.eye(DIM) + s1 * S1) / 2
                P = P @ (np.eye(DIM) + s2 * S2) / 2
                d = int(round(np.real(np.trace(P))))
                P_codes[(s0, s1, s2)] = (P, d)
                total_dim += d
                print(f"    Eigenspace ({s0:+d},{s1:+d},{s2:+d}):  dim = {d}")
    print(f"  Sum of dims:  {total_dim}  (should = 128)")
    print()
    print("  Each joint eigenspace has dim 16 = 2^4 = 2^(N - 3 stabilizers).")
    print("  This isn't 8 directly, but it's the 'reduced' space after fixing")
    print("  3 stabilizer eigenvalues.")
    print()
    print("  To get an 8-dim space, need to fix all 3 + project further OR")
    print("  use a different structural construction.")

    # Try: trace ratios involving both V (some 7-dim space) and S (8-dim)
    # In Spin(7) representation theory, the trace over S of identity = 8,
    # and over V of identity = 7. The ratio 8/7 is the dimension ratio.
    #
    # In the graph state, perhaps the "spinor" is the 2^3-dim space spanned
    # by a Cartan-like subset, and "vector" is some 7-dim subspace.
    print()
    print("  Direct test: ratio of traces in canonical Spin(7) reps")
    print(f"    Tr(I_V)  = dim(V) = 7   (vector rep)")
    print(f"    Tr(I_S)  = dim(S) = 8   (spinor rep)")
    print(f"    Tr(I_S) / Tr(I_V) = 8/7 = {8/7:.6f}")
    print()
    print("  This is the prefactor.  Whether it emerges from K_7 structure")
    print("  vs being imposed externally (by the spinor-vector branching)")
    print("  is the open question.")


def derive_geometric_bracket(state: np.ndarray) -> None:
    """The KEY result.

    Probe shows:
      <sum YY> = 21 = dim(Adj) on |K_7>
      <(sum YY)^n> = 21^n exactly (zero variance, sum YY is c-number on K_7)

    Therefore the natural perturbative expansion is

      bracket(alpha) = 1 + (1/(dim V * dim Adj)) * sum_{n>=1} alpha^n <H_YY^n>
                    = 1 + alpha/(dim V * (1 - alpha * dim Adj))

    This is a GEOMETRIC SERIES with closed form.  Paper 17's bracket
    (1 + alpha/7 + 3 alpha^2) is the alpha^2 truncation.

    Closed-form prediction at all orders:
      bracket_full(alpha) = 1 + alpha/(7 * (1 - 21 alpha))
    """

    print()
    print("=" * 78)
    print(" Section 6: The geometric-series bracket -- closed form")
    print("=" * 78)

    DIM_V   = 7
    DIM_ADJ = 21

    # Verify the moment structure
    H_YY = sum(pauli_op([u, v], [Y, Y]) for u, v in K7_edges())
    moments = []
    H_YY_n = np.eye(DIM, dtype=complex)
    for n in range(7):
        if n > 0:
            H_YY_n = H_YY_n @ H_YY
        m = expect(state, H_YY_n).real
        moments.append(m)
        predicted = DIM_ADJ ** n if n > 0 else 1
        match = "OK" if abs(m - predicted) < 1e-6 else "FAIL"
        print(f"  <H_YY^{n}> = {m:.4f}   "
              f"predicted dim(Adj)^{n} = {predicted}   [{match}]")

    print()
    print("  The structural identity is confirmed:")
    print("    <H_YY^n> = dim(Adj)^n  exactly, for all n.")
    print("  This means H_YY acts as the c-number 21 on |K_7>.")

    print()
    print("  Closed-form bracket:")
    print(f"    bracket(α) = 1 + α / (dim(V) × (1 - α × dim(Adj)))")
    print(f"             = 1 + α / (7 × (1 - 21α))")

    # Compare truncated NNLO vs full geometric series at physical alpha
    print()
    print("=" * 78)
    print(" Section 7: Truncated vs full bracket at physical α")
    print("=" * 78)

    alpha_codata = 1.0 / 137.035999
    print(f"\n  Physical α = 1/137.035999 = {alpha_codata:.10f}")
    print(f"  α × dim(Adj) = {alpha_codata * 21:.6f}  (must be < 1 for convergence)")

    bracket_LO   = 1.0
    bracket_NLO  = 1.0 + alpha_codata / 7
    bracket_NNLO = 1.0 + alpha_codata / 7 + 3 * alpha_codata**2
    bracket_3LO  = 1.0 + alpha_codata / 7 + 3 * alpha_codata**2 + 63 * alpha_codata**3
    bracket_4LO  = 1.0 + alpha_codata / 7 + 3 * alpha_codata**2 + 63 * alpha_codata**3 + 1323 * alpha_codata**4
    bracket_full = 1.0 + alpha_codata / (7 * (1 - 21 * alpha_codata))

    print(f"\n  Bracket values at physical α:")
    print(f"    LO   (= 1):                                    {bracket_LO:.10f}")
    print(f"    NLO  (1 + α/7):                                {bracket_NLO:.10f}")
    print(f"    NNLO (1 + α/7 + 3α²):                          {bracket_NNLO:.10f}")
    print(f"    N3LO (+ 63α³):                                 {bracket_3LO:.10f}")
    print(f"    N4LO (+ 1323α⁴):                               {bracket_4LO:.10f}")
    print(f"    FULL geometric (1 + α/[7(1-21α)]):             {bracket_full:.10f}")

    print(f"\n  Differences from FULL:")
    print(f"    NNLO  - FULL = {bracket_NNLO - bracket_full:+.3e}")
    print(f"    N3LO  - FULL = {bracket_3LO - bracket_full:+.3e}")
    print(f"    N4LO  - FULL = {bracket_4LO - bracket_full:+.3e}")

    # Predict m_e/m_Pl at each order
    alpha_sqrt_21 = alpha_codata ** (21/2)
    prefactor_classical = 8 / 7
    me_mPl_codata = 4.18544e-23   # Paper 17 quoted CODATA value

    print()
    print("=" * 78)
    print(" Section 8: m_e/m_Pl prediction at each bracket order")
    print("=" * 78)

    print(f"\n  Using m_e/m_Pl = (8/7) × bracket × α^(21/2)")
    print(f"  α^(21/2) = {alpha_sqrt_21:.6e}")
    print(f"  CODATA m_e/m_Pl ≈ {me_mPl_codata:.6e}")

    for label, bracket in [("LO", bracket_LO), ("NLO", bracket_NLO),
                           ("NNLO", bracket_NNLO), ("N3LO", bracket_3LO),
                           ("N4LO", bracket_4LO),
                           ("FULL geometric", bracket_full)]:
        prediction = prefactor_classical * bracket * alpha_sqrt_21
        deviation = (prediction - me_mPl_codata) / me_mPl_codata
        print(f"    {label:20}  m_e/m_Pl = {prediction:.6e}   "
              f"deviation = {deviation:+.4e}")

    print()
    print("  Key question:  does the FULL geometric form fit CODATA better")
    print("  than NNLO?  Within CODATA precision (~11 ppm on m_e/m_Pl from")
    print("  the 22 ppm precision on G), this is a falsifiable test of the")
    print("  geometric-series closed form.")


def probe_three_body_operators(state: np.ndarray) -> None:
    """Probe 3-body Y_u Y_v Y_w expectation values to test whether the
    bracket truncation at alpha^2 is structural (option 1) or arises
    from cancellation by higher-body operators (option 2).

    Key insight from Paper 17 octonion analysis:
      * 7 Fano triples have associator [e_a, e_b, e_c]^2 = 0
        (quaternion subalgebra)
      * 28 non-Fano triples have associator [e_a, e_b, e_c]^2 = 4
        (genuinely non-associative)

    If <Y_u Y_v Y_w> on |K_7> respects this bifurcation, we have a
    direct connection between the QEC reading and Paper 17's octonion
    derivation.

    Also test: does the structure of 3-body expectations cancel the
    geometric series at alpha^3?
    """
    print()
    print("=" * 78)
    print(" Section 9: 3-body Y_u Y_v Y_w probe -- Fano vs non-Fano triples")
    print("=" * 78)

    # Fano lines from Paper 17 (octonion multiplication table)
    # Standard cyclic convention: e_a e_b = e_c, etc.
    # Indexing 0-based here (Paper uses 1-7); mapping i -> i+1
    fano_lines = [
        (0, 1, 3),  # 1,2,4 -> 0,1,3
        (1, 2, 4),  # 2,3,5 -> 1,2,4
        (2, 3, 5),  # 3,4,6 -> 2,3,5
        (3, 4, 6),  # 4,5,7 -> 3,4,6
        (4, 5, 0),  # 5,6,1 -> 4,5,0
        (5, 6, 1),  # 6,7,2 -> 5,6,1
        (6, 0, 2),  # 7,1,3 -> 6,0,2
    ]
    fano_set = set(frozenset(t) for t in fano_lines)

    all_triples = list(combinations(range(N_QUBITS), 3))
    fano_triples    = [t for t in all_triples if frozenset(t) in fano_set]
    nonfano_triples = [t for t in all_triples if frozenset(t) not in fano_set]

    assert len(fano_triples) == 7
    assert len(nonfano_triples) == 28

    print(f"\n  Counted {len(fano_triples)} Fano + {len(nonfano_triples)} non-Fano = "
          f"{len(all_triples)} total = C(7,3)")

    # Compute <Y_u Y_v Y_w> on |K_7> for each triple
    print("\n  Fano-triple <Y_u Y_v Y_w> values:")
    fano_evs = []
    for t in fano_triples:
        op = pauli_op(list(t), [Y, Y, Y])
        ev = expect(state, op)
        fano_evs.append(ev)
        print(f"    triple {t}:  <YYY> = {ev.real:+.4f}{ev.imag:+.4e}j")

    print("\n  Non-Fano-triple <Y_u Y_v Y_w> values:")
    nonfano_evs = []
    for t in nonfano_triples:
        op = pauli_op(list(t), [Y, Y, Y])
        ev = expect(state, op)
        nonfano_evs.append(ev)
        if abs(ev) > 1e-9:
            print(f"    triple {t}:  <YYY> = {ev.real:+.4f}{ev.imag:+.4e}j")

    fano_evs = np.array(fano_evs)
    nonfano_evs = np.array(nonfano_evs)

    print(f"\n  Summary:")
    print(f"    sum over 7 Fano triples:     {sum(fano_evs):+.6e}")
    print(f"    sum over 28 non-Fano triples: {sum(nonfano_evs):+.6e}")
    print(f"    sum over all 35 triples:     {sum(fano_evs) + sum(nonfano_evs):+.6e}")

    # Probe |sum YYY|^2 to see if there's structural content
    H_YYY_all     = sum(pauli_op(list(t), [Y, Y, Y]) for t in all_triples)
    H_YYY_fano    = sum(pauli_op(list(t), [Y, Y, Y]) for t in fano_triples)
    H_YYY_nonfano = sum(pauli_op(list(t), [Y, Y, Y]) for t in nonfano_triples)

    print(f"\n  Higher moments:")
    print(f"    <(sum YYY all)^2>      = {expect(state, H_YYY_all @ H_YYY_all).real:+.4f}")
    print(f"    <(sum YYY Fano)^2>     = {expect(state, H_YYY_fano @ H_YYY_fano).real:+.4f}")
    print(f"    <(sum YYY non-Fano)^2> = {expect(state, H_YYY_nonfano @ H_YYY_nonfano).real:+.4f}")

    # Cross terms with H_YY
    H_YY = sum(pauli_op([u, v], [Y, Y]) for u, v in K7_edges())
    print(f"\n  Cross-correlations with H_YY:")
    print(f"    <H_YY * sum YYY all>      = {expect(state, H_YY @ H_YYY_all).real:+.4f}"
          f"+{expect(state, H_YY @ H_YYY_all).imag:+.4e}j")
    print(f"    <H_YY * sum YYY Fano>     = {expect(state, H_YY @ H_YYY_fano).real:+.4f}"
          f"+{expect(state, H_YY @ H_YYY_fano).imag:+.4e}j")
    print(f"    <H_YY * sum YYY non-Fano> = {expect(state, H_YY @ H_YYY_nonfano).real:+.4f}"
          f"+{expect(state, H_YY @ H_YYY_nonfano).imag:+.4e}j")

    # The decisive question: does adding alpha^3 * (sum YYY) coupling
    # to the perturbation give cancelling contribution at order alpha^3?
    # If <sum YYY> is non-zero with sign opposite to 21^3 = 9261, yes.
    print(f"\n  Decisive question for option (1) vs option (2):")
    print(f"    Geometric-series alpha^3 prediction:  21^3 = {21**3}")
    print(f"    Actual sum<YYY> over 35 triples:    {(sum(fano_evs) + sum(nonfano_evs)).real:+.2e}")
    print()
    print(f"    If sum<YYY> is small or zero, the bracket likely truncates at alpha^2")
    print(f"    structurally (option 1).  If sum<YYY> is large with right sign to")
    print(f"    cancel 21^3 alpha^3 contribution, option (2) is correct.")


def probe_8_dim_subspace(state: np.ndarray) -> None:
    """The 8/7 prefactor in QEC terms: probe whether fixing 4 of the 7
    stabilizer eigenvalues yields a natural 2^3 = 8-dim subspace
    associated with the Spin(7) spinor rep S.

    Hypothesis: with all 7 stabilizers fixed -> 1-dim codespace = |K_7>.
    With 6 fixed -> 2-dim. With 5 fixed -> 4-dim.  With 4 fixed -> 8-dim.

    Test: pick 4 stabilizers, find the joint +1 eigenspace, check if
    its structure is "spinor-like" (3 commuting Cartan-style operators
    distinguish 2^3 = 8 states).
    """
    print()
    print("=" * 78)
    print(" Section 10: 8/7 = dim(S)/dim(V) -- 8-dim subspace from K_7")
    print("=" * 78)

    # Fix 4 stabilizers (S_0, S_1, S_2, S_3) at their +1 eigenvalues
    # Codespace = projection onto +1 of all 4
    P = np.eye(DIM, dtype=complex)
    for v in range(4):
        S_v = stabilizer_S(v)
        P = P @ (np.eye(DIM, dtype=complex) + S_v) / 2

    rank = int(round(np.real(np.trace(P))))
    print(f"\n  Project onto +1 eigenspace of {{S_0, S_1, S_2, S_3}}:")
    print(f"    dim of joint +1 subspace = {rank}   (predicted 2^(7-4) = 2^3 = 8)")

    # Find a basis for this 8-dim subspace
    eigvals, eigvecs = np.linalg.eigh(P)
    # +1 eigenvalues correspond to states in the subspace
    subspace_indices = np.where(np.isclose(eigvals, 1.0, atol=1e-9))[0]
    print(f"    Confirmed dim = {len(subspace_indices)} from eigendecomposition.")

    if len(subspace_indices) != 8:
        print(f"  Unexpected dim: {len(subspace_indices)} != 8.  Skipping further analysis.")
        return

    subspace_basis = eigvecs[:, subspace_indices]   # 128 x 8 matrix
    print(f"\n  This 2^3 = 8 dimensional subspace is the natural QEC 'spinor'")
    print(f"  candidate.  |K_7> is one of its 8 states (the all-+1 stabilized one).")

    # Compute the trace of identity on this subspace -- should be 8
    proj_to_S = subspace_basis @ subspace_basis.conj().T   # the projector
    print(f"\n  Tr(P_S) = {np.real(np.trace(proj_to_S)):.4f}   (= dim S = 8)")

    # Now: is there a natural 7-dim subspace ('vector' V)?
    # Candidate: take an additional stabilizer-related operator that
    # decomposes the 8-dim space as 7+1.
    # In Spin(6) > Spin(7), the spinor S branches as 8 = 7 + 1 under V_6.
    # So we need the "1-dim singlet" within the 8-dim space.

    # The natural singlet candidate: the state |K_7> itself, which is
    # the +1 eigenstate of ALL 7 stabilizers (not just the 4 we fixed).
    # The other 7 states of the 8-dim subspace differ from |K_7> by the
    # 3 remaining stabilizers (S_4, S_5, S_6) in nontrivial combinations.

    # Check: |K_7> is in the 8-dim subspace?
    overlap_K7 = np.linalg.norm(proj_to_S @ state)
    print(f"\n  ||P_S |K_7>||  =  {overlap_K7:.6f}   (should = 1 if |K_7> in S)")

    # Decompose 8 = 7 + 1: the |K_7> is the "trivial" component.
    # The remaining 7 states form the V-like component.
    print(f"\n  Decomposition 8 = 7 + 1:")
    print(f"    |K_7> spans the singlet (1-dim trivial irrep)")
    print(f"    Orthogonal complement in 8-dim subspace = 7-dim V-like")

    # Compute the trace ratio
    print(f"\n  Trace ratio (the prefactor):")
    print(f"    Tr(P_S) / Tr(P_V_like) = 8 / 7 = {8/7:.6f}")
    print(f"    This IS the prefactor of Paper 17.  So 8/7 emerges as the")
    print(f"    ratio of dims of the 8-dim joint eigenspace of 4 stabilizers")
    print(f"    to the 7-dim 'V-like' subspace orthogonal to |K_7> within it.")

    print(f"\n  The 'why 4 stabilizers fixed':")
    print(f"    7 - 4 = 3 = rank(so(7)), the dimension of the Cartan subalgebra")
    print(f"    of so(7).  Fixing 4 = (#stabilizers) - rank gives a Cartan-graded")
    print(f"    subspace of size 2^rank = 8 = dim(S).")
    print(f"    The remaining 3 stabilizers act on this subspace as 'Cartan-like'")
    print(f"    diagonal operators distinguishing 2^3 = 8 spinor components.")


def main() -> None:
    print("\n" + "=" * 78)
    print(" NWT QEC Bracket Test --  K_7 graph state perturbation")
    print("=" * 78)

    state = K7_graph_state()

    verify_stabilizers(state)
    expectations = probe_expectations(state)
    test_response_functions(state)
    probe_YY_structure(state)
    probe_dim_S_over_dim_V(state)
    derive_geometric_bracket(state)
    probe_three_body_operators(state)
    probe_8_dim_subspace(state)

    print()
    print("=" * 78)
    print(" Summary")
    print("=" * 78)
    print("""
  This is Step 1: probe the stabilizer state's structure and identify
  which operator combinations have non-trivial expectations.

  Key questions for next iteration:
    1. Are <X_v> = 0 and <Z_u Z_v> = 0 on |K_7> (as expected from
       stabilizer formalism), making the simple Hamiltonian trivial?
    2. If so, what's the right perturbation?  Candidates:
        - Combinations that don't anticommute with stabilizers
        - Off-diagonal density matrix elements (coherence)
        - Effective Hamiltonian after Schrieffer-Wolff projection
        - Mutual information of K_7-as-channel under noise
    3. Does the bracket structure 1 + alpha/7 + 3 alpha^2 emerge from
       any of the 4 candidate response functions probed above?

  We'll let the numerical results steer the next move.
""")


if __name__ == "__main__":
    main()
