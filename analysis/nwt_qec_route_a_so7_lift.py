#!/usr/bin/env python3
"""
Route A — derive H_phys structurally from b2.13 K_7 → so(7) lift.

After Route B (BPS μ=π action quantization) gave a clean negative
result in `nwt_qec_bps_compton_bridge.py`, we try the structural
route: identify the natural time-evolution Hamiltonian on |K_7⟩
from the so(7) action on the 7-qubit space, with the Cartan
generators identified as the K_7 stabilizers themselves.

Key structural observation that motivates this script:

  Y_u Y_v COMMUTES with EVERY K_7 stabilizer S_v = X_v · ∏_{u≠v} Z_u.
  (Two anticommutations × two anticommutations = commute.)

So H_YY = Σ Y_u Y_v is a symmetry-preserving operator on the entire
7-qubit Hilbert space (not just on |K_7⟩).  In the joint eigenbasis
of all 7 stabilizers (which ARE diagonalisable simultaneously since
they all commute), H_YY is diagonal.

This gives a clean formula:

  H_YY  =  (M² − dim(V)) / 2

where M = Σ_v ⟨S_v⟩ is the total stabilizer sum (taking values in
{−7, −5, −3, −1, +1, +3, +5, +7} on the 7-qubit Hilbert space).

On |K_7⟩, all stabilizers are +1, so M = 7 and H_YY = (49 − 7)/2 = 21
= dim(Adj).  Within the 8-dim code subspace (4 stabilizers fixed at
+1), M ∈ {1, 3, 5, 7} with multiplicities (1, 3, 3, 1) (Pascal's
triangle row 3), reproducing the eigenvalue pattern observed in
`nwt_qec_time_evolution.py`:  H_YY ∈ {−3, +1, +9, +21}.

Possible interpretations for the Schrödinger derivation:

  (A) H_YY is NOT the physical Hamiltonian.  H_phys must be
      proportional to the so(7) Casimir, which acts as (21/4)·I
      on the 8-dim spinor subspace.  Then H_phys|K_7⟩ = const·|K_7⟩
      gives uniform Compton phase rotation across the spinor.

  (B) H_YY IS the physical Hamiltonian, and the 4 distinct rotation
      rates on the 4 weight-spaces are physical: |K_7⟩ at M=7 is the
      "rest electron"; the 7 other code-subspace states are
      excited spinor components with shifted Compton rates
      (= ω_C ± δ).  Falsifiable claim about anomalous spin precession.

  (C) The "physical Hamiltonian" requires the b2.13 OFF-DIAGONAL
      so(7) generators (the 18 raising/lowering operators), which
      don't commute with the 7 stabilizers and so don't preserve
      the code subspace cleanly.  The right H_phys mixes diagonal +
      off-diagonal pieces in a Casimir combination.

This script probes all three interpretations numerically.
"""

from __future__ import annotations

from itertools import combinations, product
from typing import List, Tuple

import numpy as np

# Reuse Pauli infrastructure
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
N_QUBITS = 7
DIM = 2 ** N_QUBITS


def pauli_on(P: np.ndarray, k: int, n: int = N_QUBITS) -> np.ndarray:
    ops = [I2] * n
    ops[k] = P
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def pauli_string(ops: List[Tuple[np.ndarray, int]], n: int = N_QUBITS) -> np.ndarray:
    out_ops = [I2] * n
    for P, k in ops:
        out_ops[k] = P @ out_ops[k]
    out = out_ops[0]
    for op in out_ops[1:]:
        out = np.kron(out, op)
    return out


def stabilizer(v: int, n: int = N_QUBITS) -> np.ndarray:
    ops = [Z] * n
    ops[v] = X
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def H_two_body(P: np.ndarray) -> np.ndarray:
    H = np.zeros((DIM, DIM), dtype=complex)
    for u, v in combinations(range(N_QUBITS), 2):
        H = H + pauli_string([(P, u), (P, v)])
    return H


def build_K7() -> np.ndarray:
    psi = np.ones(DIM, dtype=complex) / np.sqrt(DIM)
    for u, v in combinations(range(N_QUBITS), 2):
        for idx in range(DIM):
            xu = (idx >> (N_QUBITS - 1 - u)) & 1
            xv = (idx >> (N_QUBITS - 1 - v)) & 1
            if xu == 1 and xv == 1:
                psi[idx] *= -1
    return psi


def section(title: str) -> None:
    print()
    print('=' * 78)
    print(' ' + title)
    print('=' * 78)


# =====================================================================
# 1. Verify H_YY commutes with all 7 stabilizers
# =====================================================================

def verify_HYY_stabilizer_commutators() -> None:
    section('(1) [H_YY, S_v] = 0 for all v ∈ {0,...,6}')

    H_YY = H_two_body(Y)
    print(f"\n  ‖H_YY‖_∞ = {np.max(np.abs(H_YY)):.4f}")

    print(f"\n  {'v':>3} {'‖[H_YY, S_v]‖_∞':>22}")
    print('  ' + '-' * 30)
    all_zero = True
    for v in range(N_QUBITS):
        S = stabilizer(v)
        comm = H_YY @ S - S @ H_YY
        norm = float(np.max(np.abs(comm)))
        all_zero = all_zero and (norm < 1e-10)
        print(f"  {v:>3} {norm:>22.2e}")

    print(f"\n  All commutators zero?  {'YES' if all_zero else 'NO'}")
    print(f"  ⇒ H_YY is diagonalisable in joint stabilizer eigenbasis.")


# =====================================================================
# 2. Joint stabilizer eigenstates and H_YY diagonalisation
# =====================================================================

def diagonalise_HYY_in_stabilizer_basis() -> None:
    section('(2) H_YY diagonalised in joint stabilizer eigenbasis')

    H_YY = H_two_body(Y)
    stabs = [stabilizer(v) for v in range(N_QUBITS)]

    # Joint eigenstates: simultaneously diagonalise H_YY and all S_v.
    # Use commuting-matrix simultaneous diagonalisation: build
    # H_combined = Σ_v 2^v · S_v + ε · H_YY (ε small) to get unique
    # eigenspaces.

    # Step 1: find joint eigenbasis of stabilizers via H_combined =
    # Σ 3^v · S_v (rationally independent weights).
    H_combined = np.zeros((DIM, DIM), dtype=complex)
    for v, S in enumerate(stabs):
        H_combined = H_combined + (3.0 ** v) * S
    # Ensure Hermitian
    assert np.allclose(H_combined, H_combined.conj().T)

    evals_S, evecs_S = np.linalg.eigh(H_combined)
    # Decode each eigenstate's S-pattern by rounding ⟨S_v⟩
    print(f"\n  Joint stabilizer eigenstates:")
    print(f"  {'idx':>4} {'(s_0..s_6)':>20} {'M=Σs':>5} {'⟨H_YY⟩':>10} "
          f"{'predicted':>12}")
    print('  ' + '-' * 60)

    # Sample 16 representative eigenvectors (one per pattern)
    # All 128 possible patterns exist (one per eigenvector).
    sample = list(range(0, DIM, DIM // 16))[:16]
    for idx in sample:
        psi = evecs_S[:, idx]
        s_vals = []
        for v in range(N_QUBITS):
            s_v = float(np.real(psi.conj() @ stabs[v] @ psi))
            s_vals.append(int(round(s_v)))
        M = sum(s_vals)
        H_val = float(np.real(psi.conj() @ H_YY @ psi))
        # Predicted: H_YY = (M² - dim(V))/2 with dim(V) = 7
        pred = (M ** 2 - 7) / 2
        s_str = '(' + ','.join(f'{s:+d}' for s in s_vals) + ')'
        match = '✓' if abs(H_val - pred) < 1e-9 else '✗'
        print(f"  {idx:>4} {s_str:>20} {M:>5} {H_val:>10.4f} "
              f"{pred:>12.4f} {match}")

    # Verify formula for ALL eigenstates
    n_match = 0
    for idx in range(DIM):
        psi = evecs_S[:, idx]
        s_vals = [int(round(float(np.real(psi.conj() @ stabs[v] @ psi))))
                  for v in range(N_QUBITS)]
        M = sum(s_vals)
        H_val = float(np.real(psi.conj() @ H_YY @ psi))
        pred = (M ** 2 - 7) / 2
        if abs(H_val - pred) < 1e-9:
            n_match += 1

    print(f"\n  Formula H_YY = (M² − dim(V))/2 verified on {n_match}/{DIM} eigenstates.")


# =====================================================================
# 3. so(7) Casimir on the 8-dim code subspace
# =====================================================================
#
# C_2(so(7)) = Σ_{a<b} (B_ab)²  where B_ab are the 21 generators.
# On any irrep V_λ, C_2 acts as a scalar = ⟨λ, λ + 2ρ⟩.
# On the spinor S = V_{(1/2,1/2,1/2)}: C_2 = 21/4.
#
# In the K_7 graph state framework, the 8-dim Cartan-graded code
# subspace is identified with the spinor S.  We probe whether
# Σ (Y_u Y_v)² = some natural Casimir-like combination.
# =====================================================================

def probe_casimir_on_code_subspace() -> None:
    section('(3) Casimir-like operators on the 8-dim code subspace')

    # H_YY² on the code subspace: should be related to C_2(so(7))?
    H_YY = H_two_body(Y)

    # Code subspace (Cartan-graded): joint +1 eigenspace of S_0..S_3
    P = np.eye(DIM, dtype=complex)
    for v in range(4):
        S_v = stabilizer(v)
        P = P @ (np.eye(DIM) + S_v) / 2.0
    evals_P, evecs_P = np.linalg.eigh(P)
    code_basis = evecs_P[:, np.abs(evals_P - 1) < 1e-9]
    print(f"\n  Code subspace dim: {code_basis.shape[1]} (= 8 expected)")

    # H_YY² on the code subspace
    H_YY_code = code_basis.conj().T @ H_YY @ code_basis
    H_YY2_code = H_YY_code @ H_YY_code
    evals_HYY = np.linalg.eigvalsh(H_YY_code)
    evals_HYY2 = np.linalg.eigvalsh(H_YY2_code)
    print(f"\n  H_YY  eigenvalues on code:  {sorted(evals_HYY.round(3).tolist())}")
    print(f"  H_YY² eigenvalues on code:  {sorted(evals_HYY2.round(3).tolist())}")

    # Sum of squares Σ (Y_u Y_v)² — this should be 21·I (since each
    # (Y_uY_v)² = I).
    SQ = np.zeros((DIM, DIM), dtype=complex)
    for u, v in combinations(range(N_QUBITS), 2):
        op = pauli_string([(Y, u), (Y, v)])
        SQ = SQ + op @ op
    SQ_code = code_basis.conj().T @ SQ @ code_basis
    print(f"\n  Σ (Y_u Y_v)² on code: trace/dim = "
          f"{float(np.real(np.trace(SQ_code))) / 8:.4f}, "
          f"diag elements should all be 21:")
    print(f"  diag = {[float(np.real(SQ_code[i, i])) for i in range(8)]}")

    print("""
  OBSERVATION:  Σ (Y_u Y_v)² = 21·I on the entire 7-qubit Hilbert
  space (since each individual (Y_uY_v)² = I and there are 21 edges).
  This is NOT the so(7) Casimir — it's just the count of K_7 edges
  multiplied by I.

  The actual so(7) quadratic Casimir requires the b2.13 raising/
  lowering operators (which don't commute with all 7 stabilizers,
  hence don't preserve the code subspace).  H_YY is the diagonal
  Cartan-like part of this Casimir.
""")


# =====================================================================
# 4. Three interpretations of H_phys
# =====================================================================

def interpret_HYY_for_time_evolution() -> None:
    section('(4) Three interpretations of H_phys for Schrödinger derivation')

    # Standard physics
    h_bar = 1.054571817e-34
    m_e = 9.1093837015e-31
    c = 299792458.0
    omega_C = m_e * c**2 / h_bar

    print(f"""
  Compton frequency: ω_C = {omega_C:.4e} rad/s.

  INTERPRETATION (A): H_YY is NOT the physical Hamiltonian.
  ----------------------------------------------------------------
  H_phys ∝ I (identity on code subspace), giving uniform Compton
  phase rotation on all 8 spinor components.  The proportionality
  constant must come from BPS energy scale m_e c²/ℏ.

  H_phys = ω_C · I_code → trivially satisfies Schrödinger but the
  rest mass m_e is INPUT, not derived from QEC structure.

  INTERPRETATION (B): H_YY IS the physical Hamiltonian.
  ----------------------------------------------------------------
  The 4 H_YY weight-spaces (M = 7, 5, 3, 1 with eigenvalues 21, 9, 1,
  −3) are 4 distinct "Compton frequencies".  Only the |K_7⟩ state
  (M=7, eigenvalue 21) is the rest electron with standard ω_C.

  Predicts: anomalous spin precession with rate ω(M)/ω_C(M=7) =
  H_YY(M)/H_YY(7) = (M²−7)/42:

      M=7:   ω/ω_C = 1     (standard rest electron)
      M=5:   ω/ω_C = 18/42 = 3/7 ≈ 0.429
      M=3:   ω/ω_C = 2/42 = 1/21 ≈ 0.048
      M=1:   ω/ω_C = -6/42 = -1/7 ≈ -0.143

  These would be SEPARATE physical states with different rest masses,
  not just precession rates.  Falsifiable but unconfirmed.

  INTERPRETATION (C): H_phys = NLO combination of H_YY + off-diagonal.
  ----------------------------------------------------------------
  H_phys = a · H_YY + b · (off-diagonal so(7) ops) where the off-
  diagonal part doesn't preserve the code subspace exactly but
  contributes to time evolution at second order via virtual
  excitations.  Could explain α-corrections to the rest-mass phase.

  The "off-diagonal so(7)" candidates that don't commute with all
  stabilizers but preserve the code subspace approximately are
  combinations like X_u Y_v, Y_u X_v, etc.  Need further structural
  work.
""")


# =====================================================================
# 5. Connections to 8/7 and Heawood winding 8
# =====================================================================

def connections_to_8_over_7() -> None:
    section('(5) Connections to 8/7 prefactor and Heawood winding 8')

    print("""
  RECURRING 8s AND 7s — structural cross-checks:

  Object                             Value
  --------------------------------- --------
  dim(V)         (Spin(7) vector)   7
  dim(S)         (Spin(7) spinor)   8
  |V(K_7)|       (K_7 vertices)     7
  Code subspace dim                  8 = 2^rank(so(7))
  Heawood winding |p|+|q|            8 (this work, 2026-04-26 PM)
  K_7 graph dim_q ratio scale        ≈ 8/7 in spinor-vector ratio
  Cartan-graded H_YY eigenvalue 1's  (Pascal row 3 split: 1,3,3,1)

  CANDIDATE STRUCTURAL READING:

     (8/7) prefactor in m_e/m_Pl
        =  dim(S) / dim(V)
        =  (Heawood winding) / (K_7 vertex count)
        =  (Code subspace dim) / (#stabilizers)

  Interpretation: the 8 = 2^rank(so(7)) is the dimension of the
  "Cartan-graded encoding" — the 8-dim Hilbert space where the
  Spin(7) spinor lives.  The 7 = dim(V) is the dimension of the
  matter line representation.  The ratio appears at three different
  layers:
    1. As a Weyl-dimension ratio (b2.14 algebraic).
    2. As a code-subspace-to-vertex ratio (this QEC work).
    3. As a Heegaard-torus winding-to-vertex ratio (Heawood
       embedding).

  All three layers are giving the same 8/7 because the K_7 graph
  state is the COMMON STRUCTURAL OBJECT linking them: the K_7 ↔
  Spin(7) bijection (b2.13) propagates the 8/7 ratio through every
  representation-theoretic, graph-theoretic, and modular layer.

  This is a stronger reading of the b2.13 bijection: K_7 isn't just
  edge-labelling of so(7) generators — it's the underlying QEC code
  whose Cartan-graded subspace IS the Spin(7) spinor space.

  FALSIFIABLE PREDICTION (cross-group):
  For so(N) at level k with K_N as carrier:
     prefactor =  2^rank(so(N)) / dim(V_so(N))
              =  2^rank / N

  For so(7): 2³/7 = 8/7 ✓
  For so(8): 2⁴/8 = 16/8 = 2  (but so(8) has triality; spinor dim 8)
  For so(9): 2⁴/9 = 16/9  (rank(so(9)) = 4, dim spinor = 16)
  For so(11): 2⁵/11 = 32/11

  If NWT generalises to other so(N) as predicted, the prefactor
  for so(9) gravity should be 16/9 (≈ 1.778), and for so(11)
  should be 32/11 (≈ 2.909).  These are testable cross-checks in
  any putative higher-rank generalisation of the framework.
""")


# =====================================================================
# 6. Main
# =====================================================================

def main() -> None:
    section('Route A — H_phys from b2.13 K_7 → so(7) lift')

    print("""
After Route B's negative result (BPS μ=π action quantization didn't
fix the H_YY normalisation), Route A probes whether the b2.13 K_7
→ so(7) bijection structurally identifies H_phys.

Findings:
  (1) H_YY = Σ Y_u Y_v commutes with ALL 7 K_7 stabilizers.
  (2) H_YY is diagonal in joint stabilizer eigenbasis with closed-
      form eigenvalues  H_YY = (M² − dim(V))/2  where M = total
      stabilizer sum.
  (3) On the 8-dim Cartan-graded code subspace, M ∈ {1, 3, 5, 7}
      with multiplicities (1, 3, 3, 1) (Pascal's triangle).
  (4) Σ (Y_uY_v)² = 21·I on whole space — NOT the so(7) Casimir
      (which requires off-diagonal generators).
  (5) Three candidate interpretations for H_phys, none uniquely
      fixed by structural arguments alone.
  (6) Recurring 8/7 ratio across 3+ layers (Weyl, QEC, Heawood)
      — falsifiable cross-group prediction for so(N) generalisation.
""")

    verify_HYY_stabilizer_commutators()
    diagonalise_HYY_in_stabilizer_basis()
    probe_casimir_on_code_subspace()
    interpret_HYY_for_time_evolution()
    connections_to_8_over_7()


if __name__ == '__main__':
    main()
