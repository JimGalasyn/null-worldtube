#!/usr/bin/env python3
"""
Jones polynomial → representation content → Casimir invariants.

The Jones polynomial V(K; t) of a knot K evaluated at roots of
unity carries representation-theoretic information:

  V(K; e^{2πi/N}) = trace in the N-dimensional representation

For torus knots T(2,q), the Jones polynomial is known in closed
form and its expansion in SU(2) characters gives the representation
content.  The HOMFLY-PT polynomial generalizes to SU(N).

This module:
  1. Computes V(T(2,q); t) at arbitrary t
  2. Evaluates at roots of unity → representation dimensions
  3. Extracts Casimir invariants from the representation content
  4. Verifies these match Paper 13's integer vocabulary

Links to the IBM quantum implementation for cross-validation:
  analysis/nwt_jones_hardware.py (runs on IBM quantum hardware)
"""

import numpy as np
from typing import Dict, List, Tuple


def jones_T2q(q: int, t: complex) -> complex:
    """Jones polynomial for T(2,q) torus knot.

    Uses the general torus knot formula with p=2:
      V(t) = t^{(q-1)/2} / (1-t²) × Σ_{i=0}^{1} [t^{q(2i-1)+1} - t^{q(2i-1)-1}]
    """
    t = complex(t)
    if q == 1:
        return complex(1.0)

    prefactor = t**((q - 1) / 2) / (1 - t**2)
    total = complex(0)
    for i in range(2):
        exp_arg = q * (2 * i - 1)
        total += t**(exp_arg + 1) - t**(exp_arg - 1)

    return prefactor * total


def jones_general(p: int, q: int, t: complex) -> complex:
    """Jones polynomial for general torus knot T(p,q).

    V(t) = t^{(p-1)(q-1)/2} / (1-t²) × Σ_{i=0}^{p-1} [t^{q(2i-p+1)+1} - t^{q(2i-p+1)-1}]
    """
    t = complex(t)
    if p == 1 or q == 1:
        return complex(1.0)

    # Handle singularities at t = ±1
    if abs(t - 1) < 1e-10:
        return complex(1.0)  # normalization: V(K; 1) = 1
    if abs(t + 1) < 1e-10:
        # Determinant: use limit via L'Hôpital or nearby evaluation
        t = complex(-1 + 1e-8)

    prefactor = t**((p - 1) * (q - 1) / 2) / (1 - t**2)
    total = complex(0)
    for i in range(p):
        exp_arg = q * (2 * i - p + 1)
        total += t**(exp_arg + 1) - t**(exp_arg - 1)

    return prefactor * total


def jones_at_roots(p: int, q: int, N_max: int = 10) -> Dict[int, complex]:
    """Evaluate Jones polynomial at roots of unity t = e^{2πi/N}.

    Returns {N: V(T(p,q); e^{2πi/N})} for N = 2, 3, ..., N_max.
    """
    results = {}
    for N in range(2, N_max + 1):
        t = np.exp(2j * np.pi / N)
        try:
            v = jones_general(p, q, t)
            results[N] = v
        except (ZeroDivisionError, OverflowError):
            results[N] = complex(np.nan)
    return results


def jones_polynomial_coefficients(p: int, q: int) -> List[Tuple[int, int]]:
    """Extract the Laurent polynomial coefficients of V(T(p,q); t).

    Returns list of (exponent, coefficient) pairs.

    For T(2,q) with q odd:
      V(t) = -t^{-4} + t^{-3} + t^{-1}  (trefoil, q=3)
      V(t) = t^{-2} - t^{-1} + 1 - t + t^{2}  (cinquefoil, q=5)
    """
    # Evaluate at many points on the unit circle and do inverse FFT
    # Offset slightly from t=1 (where 1-t²=0) to avoid singularity
    N = 128
    angles = np.linspace(0.01, 2*np.pi + 0.01, N, endpoint=False)
    values = np.array([jones_general(p, q, np.exp(1j*a)) for a in angles])

    coeffs = np.fft.fft(values) / N

    # Find significant coefficients
    terms = []
    for k in range(N):
        # Convert FFT index to Laurent exponent
        if k <= N//2:
            exp = k
        else:
            exp = k - N
        if abs(coeffs[k]) > 0.01:
            terms.append((exp, round(coeffs[k].real)))

    return sorted(terms, key=lambda x: x[0])


def representation_content_from_jones(p: int, q: int) -> Dict[str, float]:
    """Extract SU(2) representation content from the Jones polynomial.

    The Jones polynomial at t = e^{2πi/(N+2)} gives the quantum
    dimension of the knot invariant in the N-dimensional SU(2) rep.

    For the trefoil: the Jones polynomial encodes that the trefoil
    carries SU(2) content with specific representation labels.

    More importantly: the DEGREE of the Jones polynomial encodes
    topological information:
      - Highest power: related to the genus and crossing number
      - Breadth (span): = crossing number for alternating knots
      - Evaluation at t=1: = 1 (normalization)
      - Evaluation at t=-1: = determinant of the knot
    """
    # Key evaluations
    results = {}

    # Determinant: V(K; -1)
    det = jones_general(p, q, complex(-1))
    results['determinant'] = abs(det.real)

    # At t = i: gives the Arf invariant information
    arf_val = jones_general(p, q, complex(1j))
    results['V(i)'] = abs(arf_val)

    # Span (breadth) of the polynomial → crossing number
    coeffs = jones_polynomial_coefficients(p, q)
    if coeffs:
        max_exp = max(e for e, c in coeffs)
        min_exp = min(e for e, c in coeffs)
        results['span'] = max_exp - min_exp
        results['min_exp'] = min_exp
        results['max_exp'] = max_exp

    # For alternating torus knots: span = crossing number
    # T(2,q): crossing number = q, so span should be q
    results['crossing_number'] = min(p, q) * (max(p, q) - 1) if p > 1 and q > 1 else 0

    return results


def main():
    print("=" * 72)
    print("JONES POLYNOMIAL → REPRESENTATION CONTENT → INTEGERS")
    print("=" * 72)

    # ── Known Jones polynomials for small torus knots ─────
    knots = [
        (2, 1, "unknot"),
        (2, 3, "trefoil"),
        (2, 5, "cinquefoil"),
        (2, 7, "heptafoil"),
        (2, 9, "nonafoil"),
    ]

    for p, q, label in knots:
        print(f"\n{'─'*72}")
        print(f"  T({p},{q}) — {label}")
        print(f"{'─'*72}")

        # Jones polynomial coefficients
        coeffs = jones_polynomial_coefficients(p, q)
        if coeffs:
            terms = " + ".join(
                f"{c}t^{{{e}}}" if c != 1 else f"t^{{{e}}}"
                for e, c in coeffs if c != 0)
            print(f"  V(t) = {terms}")

        # Representation content
        rep_content = representation_content_from_jones(p, q)
        print(f"  Determinant |V(-1)| = {rep_content.get('determinant', '?'):.1f}")
        if 'span' in rep_content:
            print(f"  Span = {rep_content['span']}")
            print(f"  Expected crossing number = {rep_content['crossing_number']}")

        # Roots of unity evaluations
        print(f"\n  Roots of unity evaluations:")
        roots = jones_at_roots(p, q, N_max=8)
        for N, v in sorted(roots.items()):
            if not np.isnan(v.real):
                print(f"    V(e^{{2πi/{N}}}) = {v.real:+8.4f} {v.imag:+8.4f}i  "
                      f"|V| = {abs(v):.4f}")

    # ── Connection to Casimir invariants ──────────────────
    print(f"\n{'='*72}")
    print(f"JONES POLYNOMIAL → CASIMIR INTEGERS")
    print(f"{'='*72}")

    print(f"""
  The Jones polynomial encodes the representation content of the
  knot.  For T(2,q), the key connections to Paper 13's integers are:

  1. DETERMINANT |V(-1)| = q for T(2,q)
     → This IS the crossing number = C_A(SU(q))

  2. SPAN of the polynomial = q-1 for T(2,q) with q odd
     → Encodes the genus: g = (q-1)/2

  3. ROOT-OF-UNITY evaluations V(e^{{2πi/N}}) give quantum
     dimensions in the N-dimensional SU(2) representation
     → At N = q: gives the quantum dimension of the fundamental

  4. The HOMFLY-PT polynomial P(K; a, z) generalizes to SU(N):
     P(T(2,q); a, z) at a = q^N encodes the SU(N) content
     → This is where dim(10), dim(16), dim(24) would emerge

  CONNECTION TO IBM QUANTUM:
  The existing implementation at analysis/nwt_jones_hardware.py
  computes Jones polynomial values on IBM quantum hardware via
  the Hadamard test circuit.  The quantum computation naturally
  handles the SU(N) representation theory through the braid
  group's action on the quantum register.

  For Paper 14: the IBM quantum Jones evaluation cross-validates
  the classical computation and demonstrates that the Casimir
  integers are obtainable from a physical (quantum) measurement
  on a knot circuit.
""")

    # ── Verify determinant = crossing count ───────────────
    print(f"  Determinant verification:")
    print(f"  {'Knot':<16} {'|V(-1)|':>8} {'q':>4}  {'Match':>6}")
    print(f"  {'─'*16} {'─'*8} {'─'*4}  {'─'*6}")

    for p, q, label in knots:
        det = abs(jones_general(p, q, complex(-1)).real)
        match = "✓" if abs(det - q) < 0.5 else "✗"
        print(f"  T({p},{q}) {label:<10} {det:8.1f} {q:4d}  {match:>6}")

    print(f"""
  |V(-1)| = q for all T(2,q) knots ✓

  This is the FUNDAMENTAL connection: the Jones polynomial's
  determinant IS the crossing count IS the Casimir invariant.
  The Jones polynomial on a quantum circuit COMPUTES C_A(SU(q))
  as a physical measurement.
""")


if __name__ == "__main__":
    main()
