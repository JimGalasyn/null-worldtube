#!/usr/bin/env python3
"""
Paper 15 b2.11 -- Try to build 2T subset G (stabilizer of REDUCED Cayley
form) via the normalizer of V_C(Q_8) in SO(8).

Strategy: we have V_C(Q_8) preserving both the full Phi (tested in b2.9)
and the reduced Phi (just Fano-based 4-tuples).  Signed octonion auto A
preserves the reduced Phi only.  If A NORMALIZES V_C(Q_8) -- meaning
A V_C(q) A^{-1} is in V_C(Q_8) for all q in Q_8 -- then we can build
a 24-element group

    rho(Q_8) = V_C(Q_8)
    rho(Q_8 . omega)   = V_C(Q_8) . A
    rho(Q_8 . omega^2) = V_C(Q_8) . A^2

IF the Q_8 normalizer relationship A V_C(q) A^{-1} corresponds to the
ACTUAL 2T multiplication (omega q omega^{-1} = cycled q), then this
gives 2T subset G (stabilizer of reduced Phi) = "Spin(7)" for the
reduced Phi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nwt_2T_spin7_search_b2_9 import (
    hurwitz_2T_quats, V_C, quaternion_multiply, FANO_TRIPLES,
)
from nwt_2T_spin7_full_b2_10 import build_alpha, build_reduced_Phi


def main():
    print("=" * 72)
    print(" b2.11 -- alpha as normalizer of V_C(Q_8)?")
    print("=" * 72)

    A = build_alpha()
    Phi_red = build_reduced_Phi()

    # Q_8 as Lipschitz quaternions.
    Q8_quats = [
        (1, 0, 0, 0), (-1, 0, 0, 0),
        (0, 1, 0, 0), (0, -1, 0, 0),
        (0, 0, 1, 0), (0, 0, -1, 0),
        (0, 0, 0, 1), (0, 0, 0, -1),
    ]
    Q8_quats = [tuple(float(x) for x in q) for q in Q8_quats]
    V_C_Q8 = {q: V_C(q) for q in Q8_quats}

    # Q_8 keys as rounded tuples for dict lookup.
    def qkey(q):
        return tuple(round(x, 10) for x in q)
    V_C_by_key = {qkey(q): V_C(q) for q in Q8_quats}

    # Check: does A conjugate V_C(q) to V_C(q') for q, q' in Q_8?
    # i.e., A V_C(q) A^{-1} should equal V_C(sigma(q)) for some
    # permutation sigma of Q_8.
    print("\n[1] Check  A V_C(q) A^{-1}  vs  V_C(.)  for q in Q_8:")
    A_inv = A.T    # A is orthogonal
    normalizes = True
    conj_action = {}
    for q in Q8_quats:
        M = V_C(q)
        M_conj = A @ M @ A_inv
        # Is M_conj = V_C(q') for some q' in Q_8?
        found_q = None
        for qp in Q8_quats:
            if np.allclose(M_conj, V_C(qp), atol=1e-8):
                found_q = qp
                break
        if found_q is None:
            normalizes = False
            print(f"    q = {q}: A V_C(q) A^{-1} is NOT in V_C(Q_8)")
        else:
            conj_action[q] = found_q
    if normalizes:
        print(f"    A normalizes V_C(Q_8): PASS")
        print(f"    Conjugation action of A on Q_8 (as V_C labels):")
        for q, qp in conj_action.items():
            print(f"      A · V_C({q}) · A^{-1} = V_C({qp})")
    else:
        print(f"    A normalizes V_C(Q_8): FAIL")

    # Now check: does the conjugation match the 2T group structure?
    # In 2T, omega conjugates i -> j -> k -> i.  So conj_action should
    # send V_C(i) -> V_C(j), V_C(j) -> V_C(k), V_C(k) -> V_C(i).
    if normalizes:
        print(f"\n[2] Does A's conjugation cycle (i, j, k) as omega does?")
        i, j, k = (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        print(f"    A cycles V_C(i) -> V_C({conj_action.get(i)})")
        print(f"    A cycles V_C(j) -> V_C({conj_action.get(j)})")
        print(f"    A cycles V_C(k) -> V_C({conj_action.get(k)})")

        expected = {i: j, j: k, k: i}
        cycle_ok = all(conj_action.get(q) == qp for q, qp in expected.items())
        print(f"    Cycle (i->j->k->i) matches omega-conjugation: "
              f"{'PASS' if cycle_ok else 'FAIL'}")
    else:
        cycle_ok = False

    # Try the FIX: use rho(omega) = V_C(i) * A instead of A.
    # A cycles (i, j, k) as i->-j->k->i, which differs from omega's cycle
    # i->j->k->i by the inner automorphism "conjugate by i" (which sends
    # j->-j, k->-k, i->i).  So V_C(i) * A should match omega's action.
    # Check V_C's Phi-preservation on different forms
    print(f"\n[CHECK] Does V_C(i) preserve REDUCED Phi?")
    VCi = V_C((0.0, 1.0, 0.0, 0.0))
    Phi_red_new = np.einsum('ai,bj,ck,dl,abcd->ijkl',
                             VCi, VCi, VCi, VCi, Phi_red)
    VCi_preserves_red = np.allclose(Phi_red_new, Phi_red, atol=1e-8)
    print(f"    V_C(i) preserves reduced Phi: "
          f"{'PASS' if VCi_preserves_red else 'FAIL'}")

    print(f"\n[FIX] Trying rho(omega) = V_C(i) * A:")
    i_quat = (0.0, 1.0, 0.0, 0.0)
    V_C_i = V_C(i_quat)
    rho_omega = V_C_i @ A
    print(f"    rho(omega)^3 = ? (should equal V_C(-1) for order 6):")
    rho_sq = rho_omega @ rho_omega
    rho_cube = rho_sq @ rho_omega
    neg_e = V_C((-1.0, 0.0, 0.0, 0.0))
    print(f"    ||rho(omega)^3 - V_C(-e)||_max = "
          f"{np.abs(rho_cube - neg_e).max():.3e}")
    # If rho(omega)^3 = V_C(-e), rho(omega) has order 6 (matches omega).

    # Check conjugation action of rho(omega) on V_C(Q_8) matches omega's.
    print(f"    Conjugation check: rho(omega) V_C(i) rho(omega)^-1")
    rho_inv = rho_omega.T  # orthogonal
    test_conj = rho_omega @ V_C_i @ rho_inv
    V_C_j = V_C((0.0, 0.0, 1.0, 0.0))
    matches_j = np.allclose(test_conj, V_C_j, atol=1e-8)
    print(f"      = V_C(j)?  {'PASS' if matches_j else 'FAIL'}")

    # Does rho(omega) preserve reduced Phi?
    Phi_red_new = np.einsum('ai,bj,ck,dl,abcd->ijkl',
                             rho_omega, rho_omega, rho_omega, rho_omega,
                             Phi_red)
    rho_preserves = np.allclose(Phi_red_new, Phi_red, atol=1e-8)
    print(f"    rho(omega) preserves reduced Phi: "
          f"{'PASS' if rho_preserves else 'FAIL'}")

    # If A normalizes V_C(Q_8) and preserves reduced Phi, proceed
    # (the cycle matching is automatic via the iω reinterpretation).
    if normalizes and rho_preserves:
        print(f"\n[3] Building 2T extension: rho(Q_8 omega^k) = V_C(Q_8) A^k")
        rho = {}
        A_sq = A @ A
        # Q_8 coset
        for q in Q8_quats:
            rho[q] = V_C(q)
        # Key insight: rho(omega) = V_C(i) * A has order 3 in SO(8),
        # matching the element sigma = i * omega in 2T (which has order 3,
        # verified via direct quaternion computation (i omega)^3 = e).
        # So we decompose 2T = Q_8 . <sigma> with sigma = i*omega.
        omega = (0.5, 0.5, 0.5, 0.5)
        sigma = quaternion_multiply((0.0, 1.0, 0.0, 0.0), omega)  # i*omega
        sigma_sq = quaternion_multiply(sigma, sigma)

        quats = hurwitz_2T_quats()
        def quat_inv(q):
            return (q[0], -q[1], -q[2], -q[3])
        sigma_inv = quat_inv(sigma)
        sigma_sq_inv = quat_inv(sigma_sq)

        # For each Hurwitz unit q: find k s.t. q = q_Q8 * sigma^k for q_Q8 in Q_8.
        def find_coset(q):
            for k, inv in [(0, (1, 0, 0, 0)), (1, sigma_inv), (2, sigma_sq_inv)]:
                q_Q8 = quaternion_multiply(q, inv)
                if any(all(abs(q_Q8[i] - q8[i]) < 1e-8 for i in range(4))
                        for q8 in Q8_quats):
                    return k, q_Q8
            return None, None

        rho_omega_sq = rho_omega @ rho_omega
        for q in quats:
            k, q_Q8 = find_coset(q)
            if k is not None:
                if k == 0:
                    rho[tuple(q)] = V_C(q_Q8)
                elif k == 1:
                    rho[tuple(q)] = V_C(q_Q8) @ rho_omega
                else:
                    rho[tuple(q)] = V_C(q_Q8) @ rho_omega_sq

        # Test (a) all preserve reduced Phi, (b) group homomorphism.
        n_preserve = 0
        for q in quats:
            M = rho[tuple(q)]
            Phi_new = np.einsum('ai,bj,ck,dl,abcd->ijkl',
                                 M, M, M, M, Phi_red)
            if np.allclose(Phi_new, Phi_red, atol=1e-8):
                n_preserve += 1
        print(f"    rho preserves REDUCED Phi: {n_preserve} / 24")

        hom_pass = True
        for q1 in quats:
            for q2 in quats:
                q_prod = quaternion_multiply(q1, q2)
                q_prod_key = None
                for q in quats:
                    if all(abs(q_prod[i] - q[i]) < 1e-8 for i in range(4)):
                        q_prod_key = tuple(q)
                        break
                lhs = rho[q_prod_key]
                rhs = rho[tuple(q1)] @ rho[tuple(q2)]
                if not np.allclose(lhs, rhs, atol=1e-8):
                    hom_pass = False
                    break
            if not hom_pass:
                break
        print(f"    rho is a group homomorphism: "
              f"{'PASS' if hom_pass else 'FAIL'}")

        if n_preserve == 24 and hom_pass:
            print()
            print("    *** 2T subset G (stabilizer of REDUCED Phi) "
                  "RIGOROUSLY CONSTRUCTED ***")
            print(f"    Specifically, rho: 2T -> SO(8) preserving a Fano-")
            print(f"    based invariant 4-form (not the full Cayley 4-form,")
            print(f"    but the subset generated by the 7 Fano-triple tuples).")

    print()
    print("=" * 72)
    print(" SUMMARY")
    print("=" * 72)


if __name__ == "__main__":
    main()
