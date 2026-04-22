#!/usr/bin/env python3
"""
Paper 15 b2.3 -- Trefoil T(2,3) in S^3 and its 2I orbit:
stabiliser, orbit size, and pairwise linking numbers.

Motivation.  The (+1)-Dehn surgery on T(2,3) produces S^3/2I.
The underlying trefoil, lifted to S^3 and sent through each of
the 120 elements of the binary icosahedral group, generates an
orbit of (120 / stabiliser) copies of the trefoil.  The pairwise
linking data of that orbit is a 2I-invariant topological observable
that might expose the '21' or '168' appearing in Paper 15's
alpha^(21/2).

Setup.
  - S^3 = SU(2) via the map
       (z_1, z_2)  <->  [[z_1, -conj(z_2)],
                         [z_2,  conj(z_1)]]
  - Clifford torus: |z_1|^2 = |z_2|^2 = 1/2.
  - T(2,3):  gamma(t) = ( e^{2it} / sqrt(2),  e^{3it} / sqrt(2) ),  t in [0, 2pi).
  - Left action:  g * gamma(t) = g (z_1, z_2)^T  (g in SU(2)).

Computations.
  (1) Stabilizer of gamma under the 2I left action: size of
      { g in 2I : g * gamma = gamma  as a point set }.
  (2) Orbit size = 120 / |stab|.
  (3) Pairwise linking numbers Lk(g * gamma, h * gamma) for all
      distinct orbit pairs, via stereographic projection to R^3
      and the discrete Gauss linking-number formula.

If Paper 15's '21' has a topological origin in this configuration,
it should appear in the linking-number histogram.

Method for linking numbers.
  Stereographic projection from the point (-1, 0, 0, 0) in R^4
  (antipode of a reasonable 'origin' on S^3) sends each point
  x = (a, b, c, d) in S^3 to the R^3 point

      phi(x) = (b, c, d) / (1 + a)

  (valid for a > -1).  Two generic orbit curves miss the
  antipode; for those rare cases where a point lies at a = -1,
  we could rotate the projection, but we skip the pathological
  configurations.

  Discrete Gauss formula:
      Lk(A, B)  =  (1 / 4 pi) sum_{i, j}  (r_i^A - r_j^B) .
                   ( dr_i^A  x  dr_j^B )  /  |r_i^A - r_j^B|^3
  Round to the nearest integer; a well-chosen N_t makes the
  residual numerical error << 0.5 so this is unambiguous.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nwt_poincare_sphere_b2_0 import generate_2I


# =========================================================================
# Trefoil parameterisation in S^3 = SU(2).
# =========================================================================

def trefoil_on_clifford(N_t: int = 240) -> np.ndarray:
    """T(2,3) on the Clifford torus as an array of points in R^4.

    Parameterise gamma(t) = (z_1, z_2) = (e^{2it}/sqrt(2), e^{3it}/sqrt(2)).
    Represent each point as (a, b, c, d) = (Re z_1, Im z_1, Re z_2, Im z_2)
    in R^4.

    Returns shape (N_t, 4).
    """
    t = np.linspace(0.0, 2.0 * np.pi, N_t, endpoint=False)
    z1 = np.exp(2j * t) / np.sqrt(2.0)
    z2 = np.exp(3j * t) / np.sqrt(2.0)
    return np.stack([z1.real, z1.imag, z2.real, z2.imag], axis=1)


def apply_SU2(g: np.ndarray, pts_R4: np.ndarray) -> np.ndarray:
    """Apply g in SU(2) acting on (z_1, z_2) by left multiplication.

    g @ (z_1, z_2)^T = (g_{00} z_1 + g_{01} z_2, g_{10} z_1 + g_{11} z_2).

    pts_R4 shape (N, 4).  Output shape (N, 4).
    """
    z1 = pts_R4[:, 0] + 1j * pts_R4[:, 1]
    z2 = pts_R4[:, 2] + 1j * pts_R4[:, 3]
    w1 = g[0, 0] * z1 + g[0, 1] * z2
    w2 = g[1, 0] * z1 + g[1, 1] * z2
    return np.stack([w1.real, w1.imag, w2.real, w2.imag], axis=1)


# =========================================================================
# Setwise equality of two orbit-curves up to reparameterisation.
# =========================================================================

def point_set_equal(A: np.ndarray, B: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if the point sets of A and B are the same (unordered).

    We compute a distance matrix A-to-B and require every row (and
    every column) has at least one entry within tol.  This avoids
    requiring a specific permutation ordering.
    """
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    return (d.min(axis=1).max() < tol) and (d.min(axis=0).max() < tol)


# =========================================================================
# Stereographic projection S^3 -> R^3.
# =========================================================================

def stereographic(pts_R4: np.ndarray, pole: np.ndarray) -> np.ndarray:
    """Stereographic projection of S^3 points onto R^3, projection from
    a pole in S^3.  If pole = (p0, p1, p2, p3), send x ~= pole to infinity.

    We compute a rotation that sends pole to (1, 0, 0, 0), then project.
    """
    # Orthonormal basis with pole as first vector.
    p = pole / np.linalg.norm(pole)
    # Simple rotation: pick an orthonormal basis extending p.  Use
    # Householder-like construction.
    e1 = np.array([1.0, 0.0, 0.0, 0.0])
    if np.allclose(p, e1):
        Q = np.eye(4)
    elif np.allclose(p, -e1):
        Q = -np.eye(4)
    else:
        # Householder reflection sending p to e1.
        v = p - e1
        v /= np.linalg.norm(v)
        Q = np.eye(4) - 2.0 * np.outer(v, v)
    # Rotate pts so pole -> e1.
    rotated = pts_R4 @ Q.T
    a = rotated[:, 0]
    rest = rotated[:, 1:]
    # Standard: phi(a, b, c, d) = (b, c, d) / (1 - a).
    denom = 1.0 - a
    out = rest / denom[:, None]
    return out


# =========================================================================
# Discrete Gauss linking number for two closed curves in R^3.
# =========================================================================

def linking_number(A: np.ndarray, B: np.ndarray) -> float:
    """Gauss linking integral, discretised.

    Input: A, B shape (N_A, 3), (N_B, 3), each a CLOSED curve
    (last segment connects back to first).

    Lk = (1/4pi) sum_{i, j}  (r_A[i] - r_B[j]) . (dr_A[i] x dr_B[j])
                             / |r_A[i] - r_B[j]|^3
    dr_A[i] = A[(i+1) % N_A] - A[i]
    midpoint evaluation is more accurate; use midpoints r_A[i] =
    (A[(i+1) % N_A] + A[i]) / 2.
    """
    N_A, N_B = A.shape[0], B.shape[0]
    Ap = np.roll(A, -1, axis=0)
    Bp = np.roll(B, -1, axis=0)
    dA = Ap - A
    dB = Bp - B
    rA = 0.5 * (A + Ap)      # midpoints
    rB = 0.5 * (B + Bp)
    # Pairwise r_A - r_B: shape (N_A, N_B, 3)
    R = rA[:, None, :] - rB[None, :, :]
    # Pairwise dr_A x dr_B: shape (N_A, N_B, 3)
    dAcross = np.cross(dA[:, None, :], dB[None, :, :])
    # numerator: R . dAcross
    num = (R * dAcross).sum(axis=-1)
    # denominator: |R|^3
    Rmag = np.linalg.norm(R, axis=-1)
    # Avoid division by zero (curves shouldn't intersect)
    denom = np.where(Rmag < 1e-10, 1.0, Rmag ** 3)
    contribution = num / denom
    Lk = contribution.sum() / (4.0 * np.pi)
    return Lk


# =========================================================================
# Main.
# =========================================================================

def main():
    print("=" * 72)
    print(" b2.3 -- Trefoil T(2,3) orbit under 2I, linking numbers")
    print("=" * 72)

    elements = generate_2I()
    print(f"\n[1] 2I loaded: {len(elements)} elements")

    N_t = 800
    gamma = trefoil_on_clifford(N_t)
    print(f"    gamma parameterised with N_t = {N_t} points on the")
    print(f"    Clifford torus:  (e^{{2it}}/sqrt(2), e^{{3it}}/sqrt(2)).")

    # ---- Stabiliser -----------------------------------------------------
    print("\n[2] Stabiliser of gamma under 2I left action")
    stab = []
    for idx, g in enumerate(elements):
        gamma_g = apply_SU2(g, gamma)
        if point_set_equal(gamma_g, gamma, tol=1e-4):
            stab.append(idx)
    print(f"    |stab| = {len(stab)}")
    print(f"    orbit size  = 120 / {len(stab)} = {120 // len(stab)}")

    # ---- Orbit (unique curves) -----------------------------------------
    # Pick one representative per coset g Stab.
    orbit_reps = []
    seen_sets = []
    for g in elements:
        gamma_g = apply_SU2(g, gamma)
        already = any(point_set_equal(gamma_g, s, tol=1e-4)
                      for s in seen_sets)
        if not already:
            seen_sets.append(gamma_g)
            orbit_reps.append(g)
    print(f"    orbit reps (unique curves) = {len(orbit_reps)}")

    # ---- Pairwise linking numbers --------------------------------------
    # Project stereographically from a generic pole.
    pole = np.array([0.1, 0.2, -0.7, 0.68])
    pole /= np.linalg.norm(pole)

    print(f"\n[3] Pairwise linking numbers among {len(orbit_reps)} orbit curves")
    print(f"    (stereographic projection pole = {pole.round(3)})")
    t0 = time.time()
    K = len(seen_sets)
    Lk = np.zeros((K, K))
    for i in range(K):
        Ai = stereographic(seen_sets[i], pole)
        for j in range(i + 1, K):
            Aj = stereographic(seen_sets[j], pole)
            lk = linking_number(Ai, Aj)
            Lk[i, j] = lk
            Lk[j, i] = lk
    print(f"    computed {K*(K-1)//2} pairs in {time.time()-t0:.1f}s")

    # Round to integers and tabulate.
    Lk_int = np.round(Lk).astype(int)
    residual = np.max(np.abs(Lk - Lk_int))
    print(f"    max |Lk - round(Lk)| = {residual:.3e}  "
          f"(integer if << 0.5)")

    # Linking-number histogram.
    upper_lk = []
    for i in range(K):
        for j in range(i + 1, K):
            upper_lk.append(Lk_int[i, j])
    cnt = Counter(upper_lk)
    print(f"\n    Pairwise linking-number distribution:")
    for v in sorted(cnt):
        print(f"      Lk = {v:+3d}  count = {cnt[v]:4d}")
    total_abs = sum(abs(v) * c for v, c in cnt.items())
    total_signed = sum(v * c for v, c in cnt.items())
    print(f"    sum of |Lk|  over pairs = {total_abs}")
    print(f"    sum of  Lk   over pairs = {total_signed}")
    print(f"    (pairs total = {K*(K-1)//2} = C({K},2))")

    # ---- Check for structural integers --------------------------------
    print(f"\n[4] Comparison to structural integers:")
    per_curve = [sum(abs(Lk_int[i, j]) for j in range(K) if j != i)
                 for i in range(K)]
    for target in [21, 60, 120, 168]:
        found = []
        if any(v == target for v in cnt):
            found.append(f"pair count")
        if total_abs == target:
            found.append("sum |Lk|")
        if total_signed == target:
            found.append("sum Lk")
        if K == target:
            found.append("orbit size")
        if any(p == target for p in per_curve):
            found.append(f"per-curve |Lk|-sum")
        if found:
            print(f"    {target}: FOUND in -- " + ", ".join(found))
        else:
            print(f"    {target}: not found")

    # Extra: factorisations of the observed integers.
    print(f"\n[4b] Structural factorisations of observed integers:")
    print(f"    orbit size K = {K}                ={K} = 120/|stab| = 120/{len(stab)}")
    print(f"    pairs C(K,2) = {K*(K-1)//2}")
    print(f"    sum|Lk|      = {total_abs}")
    print(f"    168 / K       = {168 / K:.3f}  (Paper 15 lambda_1 / orbit size)")
    if K > 0 and 168 % K == 0:
        print(f"    168 = {168 // K} x {K}   (168 is divisible by K!)")
    if total_abs > 0 and total_abs % 168 == 0:
        print(f"    sum|Lk| = {total_abs // 168} x 168")
    if total_abs > 0 and total_abs % 21 == 0:
        print(f"    sum|Lk| = {total_abs // 21} x 21")

    # ---- Plot: one representative stereographic projection -----------
    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    colors = plt.cm.viridis(np.linspace(0, 1, len(seen_sets)))
    for i, s in enumerate(seen_sets[:min(20, K)]):
        p = stereographic(s, pole)
        # Close the curve for plotting.
        p_closed = np.vstack([p, p[0:1]])
        ax1.plot(p_closed[:, 0], p_closed[:, 1], p_closed[:, 2],
                 color=colors[i], alpha=0.7, lw=0.7)
    ax1.set_title(f"First {min(20, K)} orbit trefoils, "
                  f"stereographic R^3 projection")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(Lk_int, cmap="RdBu_r",
                    vmin=-max(abs(Lk_int.min()), Lk_int.max()),
                    vmax=max(abs(Lk_int.min()), Lk_int.max()))
    ax2.set_title(f"Pairwise linking-number matrix ({K} curves)")
    ax2.set_xlabel("orbit curve j")
    ax2.set_ylabel("orbit curve i")
    plt.colorbar(im, ax=ax2, fraction=0.046)

    out = Path(__file__).parent / "nwt_trefoil_orbit_linking_b2_3.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\n[5] Plot: {out}")


if __name__ == "__main__":
    main()
