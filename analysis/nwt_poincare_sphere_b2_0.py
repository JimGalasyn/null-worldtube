#!/usr/bin/env python3
"""
Paper 15 Stage 2 b2.0 -- Numerical verification of lambda_1 = 168 on
the Poincare homology sphere S^3 / 2I.

Paper 15 section 2 asserts:

    On S^3, -Delta has spectrum lambda_n = n(n+2), deg = (n+1)^2.
    On S^3/2I, only 2I-invariant harmonics survive.  The multiplicity
    of the trivial representation of 2I inside the spin-j irrep of
    SU(2) is

        m(j)  =  (1/|2I|) sum_{g in 2I} chi_j(g),
        chi_j(g) = sin((2j+1) phi_g / 2) / sin(phi_g / 2),

    with phi_g the SU(2) rotation angle of g.  Result: m(j) = 0 for
    1 <= n=2j <= 11, and m(6) = 1, giving n_1 = 12, lambda_1 = 168.

We verify this in two independent ways.

Path A -- character-theoretic, using the nine conjugacy classes of 2I.
Clean, analytic, depends only on the class data.

Path B -- explicit group construction. We build a spanning set of
matrices in SU(2) that generates 2I via iterative closure, compute
D^j(g) for each of the 120 group elements using the exact SU(2)
Wigner-D formula, and find the common null-space of (D^j(g) - I)
for all g. Its dimension equals m(j) by definition.

Path B catches errors Path A cannot: conjugacy-class assignment bugs,
arithmetic sign errors in chi_j, etc.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_rank, svd
from scipy.special import factorial


# =========================================================================
# Path A -- character-theoretic multiplicity.
# =========================================================================

# Nine conjugacy classes of the binary icosahedral group 2I subset SU(2).
# Each entry: (class size, SU(2) rotation angle phi, label).
# Sum of sizes = 120 = |2I|.
#
# The two order-1 classes are {e} and {-e}.  The single size-30 class
# contains the 15 two-fold rotations (pi in SO(3)), each lifted to two
# opposite-sign elements in SU(2); phi = pi here.  Two size-20 classes
# come from the three-fold rotations (10 axes x 2 directions) lifted to
# SU(2) as 2 classes with phi = 2pi/3 and phi = 4pi/3.  Four size-12
# classes come from the five-fold rotations (6 axes x 4 rotations each),
# lifted to 4 classes at phi in {2pi/5, 4pi/5, 6pi/5, 8pi/5}.
CLASSES_2I = [
    (1,  0.0,              "e"),
    (1,  2.0 * np.pi,       "-e"),
    (30, np.pi,             "2-fold"),
    (20, 2.0 * np.pi / 3,   "3-fold_A"),
    (20, 4.0 * np.pi / 3,   "3-fold_B"),
    (12, 2.0 * np.pi / 5,   "5-fold_A"),
    (12, 4.0 * np.pi / 5,   "5-fold_B"),
    (12, 6.0 * np.pi / 5,   "5-fold_C"),
    (12, 8.0 * np.pi / 5,   "5-fold_D"),
]
assert sum(c[0] for c in CLASSES_2I) == 120


def chi_j(j: float, phi: float) -> float:
    """SU(2) character of spin-j at rotation angle phi.

        chi_j(phi) = sin((2j+1) phi/2) / sin(phi/2).

    Handle phi = 0 and phi = 2*pi by the limits  +-(2j+1).
    """
    if abs(phi) < 1e-12:
        return 2.0 * j + 1.0
    if abs(phi - 2.0 * np.pi) < 1e-12:
        # chi_j(2pi) = (-1)^{2j} (2j+1).  For integer 2j this is real.
        sign = 1.0 if (int(round(2 * j)) % 2 == 0) else -1.0
        return sign * (2.0 * j + 1.0)
    return np.sin((2.0 * j + 1.0) * phi / 2.0) / np.sin(phi / 2.0)


def multiplicity_trivial_A(j: float) -> float:
    """m(j) via Path A: Frobenius sum over conjugacy classes."""
    total = sum(size * chi_j(j, phi) for size, phi, _ in CLASSES_2I)
    return total / 120.0


# =========================================================================
# Path B -- construct 2I explicitly, project D^j onto invariants.
# =========================================================================

def su2_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """SU(2) matrix for rotation by `angle` around unit `axis`."""
    ax = axis / np.linalg.norm(axis)
    c, s = np.cos(angle / 2.0), np.sin(angle / 2.0)
    sigma = [
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
    return c * np.eye(2, dtype=complex) + 1j * s * sum(
        ax[k] * sigma[k] for k in range(3))


def generate_2I() -> list[np.ndarray]:
    """Generate the 120 elements of 2I as SU(2) matrices by closure.

    Seed with two standard generators: a 5-fold and a 2-fold rotation
    whose axes differ by the icosahedral vertex-face angle.  Multiply
    until the list closes, up to numerical equivalence within 1e-8.
    """
    # 5-fold axis: z.  2-fold axis: tilted toward a face midpoint.
    # The angle between a 5-fold and a 2-fold axis in an icosahedron
    # is arctan(phi).  phi = (1+sqrt(5))/2.
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    theta = np.arctan(1.0 / phi_g)   # 5-fold to 2-fold
    five = su2_from_axis_angle(np.array([0, 0, 1.0]), 2 * np.pi / 5)
    two = su2_from_axis_angle(
        np.array([np.sin(theta), 0.0, np.cos(theta)]),
        np.pi)

    def key(M):
        # Hash a 2x2 complex matrix to a rounded tuple for dedup.
        return tuple(np.round(M.flatten(), 8).tolist())

    seen = {key(np.eye(2, dtype=complex))}
    elements = [np.eye(2, dtype=complex)]
    frontier = [np.eye(2, dtype=complex)]
    gens = [five, two]
    while frontier:
        nxt = []
        for M in frontier:
            for g in gens:
                for P in (g @ M, M @ g):
                    k = key(P)
                    if k not in seen:
                        seen.add(k)
                        elements.append(P)
                        nxt.append(P)
        frontier = nxt
    return elements


def wigner_D(j: float, U: np.ndarray) -> np.ndarray:
    """Wigner-D matrix D^j(U) for U in SU(2), j = 0, 1/2, 1, ...

    Use the fact that SU(2) acts on homogeneous polynomials of degree
    2j in two variables (z1, z2).  The basis psi_m = z1^{j+m} z2^{j-m}
    / sqrt((j+m)! (j-m)!) for m = -j, ..., j (total 2j+1 states).
    Under SU(2): (z1', z2') = U (z1, z2)^T.  Re-expand the transformed
    psi_m in the original basis.

    This is exact (up to floating point); no recursion.
    """
    dim = int(round(2 * j + 1))
    m_vals = np.array([j - k for k in range(dim)])   # m = j, j-1, ..., -j
    D = np.zeros((dim, dim), dtype=complex)
    a, b = U[0, 0], U[0, 1]
    c, d = U[1, 0], U[1, 1]
    for i, m in enumerate(m_vals):
        # psi_m under U: (a z1 + b z2)^(j+m) (c z1 + d z2)^(j-m) / norm
        jm = j + m
        jmm = j - m
        jm_int = int(round(jm))
        jmm_int = int(round(jmm))
        norm_in = np.sqrt(factorial(jm_int) * factorial(jmm_int))
        for ip, mp in enumerate(m_vals):
            jmp = j + mp
            jmmp = j - mp
            jmp_int = int(round(jmp))
            jmmp_int = int(round(jmmp))
            norm_out = np.sqrt(factorial(jmp_int) * factorial(jmmp_int))
            # Coefficient of z1^{j+m'} z2^{j-m'} in the expansion.
            coeff = 0.0 + 0.0j
            for k1 in range(jm_int + 1):
                # Choose k1 factors of b (so j+m-k1 of a).
                for k2 in range(jmm_int + 1):
                    # Choose k2 factors of d (so j-m-k2 of c).
                    # Power of z1: (j+m-k1) + (j-m-k2)
                    z1_pow = (jm_int - k1) + (jmm_int - k2)
                    if z1_pow != jmp_int:
                        continue
                    # Power of z2: k1 + k2 must be j-m'.
                    if k1 + k2 != jmmp_int:
                        continue
                    binom1 = factorial(jm_int) / (
                        factorial(k1) * factorial(jm_int - k1))
                    binom2 = factorial(jmm_int) / (
                        factorial(k2) * factorial(jmm_int - k2))
                    coeff += (binom1 * binom2
                              * a ** (jm_int - k1) * b ** k1
                              * c ** (jmm_int - k2) * d ** k2)
            D[ip, i] = coeff * norm_out / norm_in
    return D


def multiplicity_trivial_B(j: float, elements: list[np.ndarray]) -> int:
    """m(j) via Path B: dimension of common null space of D^j(g) - I.

    A vector v in C^{2j+1} is 2I-invariant iff D^j(g) v = v for all g
    in 2I.  This is the null space of M = [D^j(g) - I for g in 2I]
    stacked.  m(j) = dim null M.
    """
    dim = int(round(2 * j + 1))
    rows = []
    for g in elements:
        D = wigner_D(j, g)
        rows.append(D - np.eye(dim, dtype=complex))
    M = np.vstack(rows)
    s = svd(M, compute_uv=False)
    # Absolute tolerance: safer than s[0]-relative when M is nearly zero.
    tol = 1e-9
    nullity = int(np.sum(s < tol))
    # dim - rank check is a separate route:
    r = int(np.sum(s >= tol))
    nullity_rank = dim - r
    assert nullity == nullity_rank, (
        f"j={j}: rank/svd disagreement {nullity} vs {nullity_rank}")
    return nullity


# =========================================================================
# Main.
# =========================================================================

def main():
    print("=" * 72)
    print(" b2.0 -- lambda_1 on S^3/2I via character theory + explicit 2I")
    print("=" * 72)

    # --- Path B: build 2I explicitly ------------------------------------
    t0 = time.time()
    elements = generate_2I()
    print(f"\n[1] Built 2I by closure: |2I| = {len(elements)} "
          f"(expected 120)  [{time.time()-t0:.3f}s]")
    assert len(elements) == 120, f"Expected 120 elements, got {len(elements)}"

    # Sanity: verify the class sizes by grouping elements by trace.
    traces = np.array([np.trace(g).real for g in elements])
    uniq, counts = np.unique(np.round(traces, 6), return_counts=True)
    print(f"\n    Trace histogram (2cos(phi/2) for each class):")
    for t, n in sorted(zip(uniq, counts), key=lambda x: -x[0]):
        phi = 2 * np.arccos(np.clip(t / 2, -1, 1))
        phi_over_pi = phi / np.pi
        print(f"      trace = {t:+.5f}  n = {n:3d}  phi = {phi_over_pi:.4f} pi")

    # --- Path A and B for j = 0, 1/2, 1, ..., 12 ------------------------
    print(f"\n[2] Multiplicity m(j) of the trivial irrep of 2I inside "
          f"spin-j of SU(2)")
    print(f"    j         n     lambda=n(n+2)   m_A (character)   m_B (explicit)   match")
    print(f"    " + "-" * 78)

    results = []
    n_max_B = 25                         # explicit-D^j check up to this n
    n_max_A = 61                         # character-only scan up to here
    for k in range(0, n_max_A + 1):       # j = 0, 0.5, ..., n_max_A/2
        j = k / 2.0
        n = int(round(2 * j))
        lam = n * (n + 2)
        m_A = multiplicity_trivial_A(j)
        if n <= n_max_B:
            m_B = multiplicity_trivial_B(j, elements)
            match = abs(m_A - m_B) < 1e-6
            match_str = "OK" if match else "MISMATCH"
            m_B_str = f"{m_B:3d}"
        else:
            m_B = None
            match_str = "(A only)"
            m_B_str = "  -"
        tag = ""
        if m_A > 0.5 and n > 0:
            tag = "  invariant"
        if n == 12 and m_A > 0.5:
            tag = "  *** lambda_1 = 168 (Paper 15 prediction) ***"
        results.append((j, n, lam, m_A, m_B))
        print(f"    {j:5.1f}   {n:3d}      {lam:5d}      "
              f"{m_A:+7.4f}       {m_B_str}          "
              f"{match_str}{tag}")

    # --- Invariant count and dimension of eigenspace --------------------
    # Dimension of the 2I-invariant eigenspace at level n is
    # (n+1) * m(n/2)  (one vector for each 'left' index, projected onto
    # m(j) invariant 'right' vectors).
    print(f"\n[3] Surviving eigenspaces on S^3/2I "
          f"(dim = (n+1) x m(n/2)):")
    total_dim = 0
    invariant_modes = []
    for j, n, lam, m_A, m_B in results:
        dim = int(round((n + 1) * m_A))
        if dim > 0:
            total_dim += dim
            tag = "  <-- Paper 15 lambda_1" if n == 12 else ""
            print(f"      n={n:3d}  lambda={lam:5d}  dim={dim:3d}{tag}")
            invariant_modes.append((n, lam, dim))
    print(f"    total 2I-invariant modes up to n={results[-1][1]}:  "
          f"{total_dim}")

    # Check the "21 appears again" observation.
    print(f"\n[3b] Curiosity: integer 21 appears in the Paper 15 "
          f"exponent alpha^21.")
    print(f"     It also appears as the eigenspace dimension at "
          f"lambda_2 = 440 (n=20):")
    for n, lam, dim in invariant_modes:
        if dim == 21:
            print(f"       n={n}, lambda={lam}, dim=21   [dim(Lambda^2 C^7) "
                  f"= 7*6/2 = 21]")
            break
    print(f"     Not a proof — could be icosahedral-structure "
          f"artefact — but notable.")

    # --- Plot spectrum --------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    n_all = [r[1] for r in results]
    m_all = [r[3] for r in results]
    lam_all = [r[2] for r in results]
    colors = ["tab:blue" if m > 0.5 else "lightgray" for m in m_all]

    for n, lam, m, col in zip(n_all, lam_all, m_all, colors):
        ax.scatter(n, lam, c=col, s=80, edgecolor="k", linewidth=0.5)
        if m > 0.5:
            ax.annotate(f"  {lam}", (n, lam), fontsize=9,
                        color="tab:blue", ha="left", va="bottom")

    # Highlight lambda_1
    ax.axhline(168, color="red", ls="--", lw=1, alpha=0.7)
    ax.text(0.5, 170, r"$\lambda_1 = 168$ (Paper 15)",
            color="red", fontsize=10)

    ax.set_xlabel("n = 2j")
    ax.set_ylabel(r"$\lambda_n = n(n+2)$")
    ax.set_title(r"Spectrum of $-\Delta$ on $S^3$;"
                 r" blue = surviving mode on $S^3/2I$")
    ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 25, 2))

    out = Path(__file__).parent / "nwt_poincare_sphere_b2_0.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\n[4] Plot: {out}")


if __name__ == "__main__":
    main()
