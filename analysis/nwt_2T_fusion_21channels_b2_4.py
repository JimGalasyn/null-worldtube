#!/usr/bin/env python3
"""
Paper 15 b2.4 -- The 21 channels as pairs of 2T irreps.

Motivation.  b2.3a verified  168 = |Irr(2T)| x |2T| = 7 x 24.
Paper 15 writes  21 = dim(Lambda^2 C^7),  but this script argues the
structurally right interpretation is

    21  =  C(7, 2)  =  # unordered pairs of distinct 2T irreps

which is the same integer but carries genuine 2T-representation
content: the 21 pairs (pi_i, pi_j) with i != j, each defining a
tensor-product channel  pi_i (x) pi_j.

If a transition amplitude m_e / m_Pl is the PRODUCT over these 21
channels of one sqrt(alpha) factor each (Paper 13 rule for OPEN paths),
the total is alpha^(21/2), matching Paper 15 exactly.

This script:
  (1) Uses the known 2T character table (7 classes, 7 irreps, dims
      (1,1,1,2,2,2,3)).
  (2) Computes the fusion coefficients  N_{ijk}  for every ordered
      triple  pi_i (x) pi_j = sum_k N_{ijk} pi_k
      via character inner products.
  (3) Enumerates the 21 unordered pairs {pi_i, pi_j} with i != j and
      reports their fusion decomposition.
  (4) Checks structural sum rules: total dim of all 21 pair-products,
      integer-valuedness, etc.
  (5) Discusses the physical-ansatz picture where 21 pairs x
      sqrt(alpha) per pair => alpha^(21/2).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================================================================
# 2T character table  (7 irreps, 7 conjugacy classes).
# Class order:  C1 = {e},  C2 = {-e},  C3 = 6 order-4,
#               C4 = 4 order-6 A,  C5 = 4 order-6 B,
#               C6 = 4 order-3 A,  C7 = 4 order-3 B.
# =========================================================================

def tT_character_table():
    """Character table of 2T (binary tetrahedral, order 24).

    Class order:  (e, -e, 2A, 6A, 6B, 3A, 3B),  sizes (1,1,6,4,4,4,4).
    Derivation of the row entries:
      * 2a := SU(2) fundamental rep.  chi(g) = trace of g in SU(2) = 2cos(theta/2).
        At 6A, 6B: trace +1 (angle 2pi/3).  At 3A, 3B: trace -1 (angle 4pi/3).
      * chi_omega is the order-3 abelianization character.  Convention:
        chi_omega at a chosen 3A element = omega.  By class-function and
        the relation 6A = -e * 3A, chi_omega(6A) = chi_omega(3A) = omega
        (since chi_omega(-e) = 1).
      * 2b = 2a (x) chi_omega, 2c = 2a (x) chi_omega^2.
      * "3" = 3-dim irrep (spin-1 of SU(2)/Z_2 = SO(3)).  chi(2A) = -1
        (axis rotation by pi gives rotation matrix with trace -1).
    """
    w = np.exp(2j * np.pi / 3)  # omega
    w2 = w ** 2
    chi = np.array([
        [1, 1, 1,  1,  1,  1,  1],                      # trivial
        [1, 1, 1,  w,  w2, w,  w2],                     # omega-chi
        [1, 1, 1,  w2, w,  w2, w],                      # omega^2-chi
        [2, -2, 0, 1,  1,  -1, -1],                     # 2a  (= SU(2) fund)
        [2, -2, 0, w,  w2, -w, -w2],                    # 2b  (= 2a x omega)
        [2, -2, 0, w2, w,  -w2, -w],                    # 2c  (= 2a x omega^2)
        [3, 3, -1, 0,  0,  0,  0],                      # 3   (adjoint)
    ], dtype=complex)
    sizes = np.array([1, 1, 6, 4, 4, 4, 4])
    names = ["triv", "chi_w", "chi_w^2", "2a", "2b", "2c", "3"]
    dims = [int(round(chi[i, 0].real)) for i in range(7)]
    return chi, sizes, names, dims


def fusion_coefficient(chi: np.ndarray, sizes: np.ndarray,
                        i: int, j: int, k: int) -> int:
    """N_{ijk} = (1/|G|) sum_c |C_c| chi_i(c) chi_j(c) conj(chi_k(c))."""
    G = int(sizes.sum())
    total = (sizes * chi[i] * chi[j] * np.conj(chi[k])).sum() / G
    # Must be a non-negative integer.
    total_re = total.real
    total_im = total.imag
    assert abs(total_im) < 1e-8, f"Non-real N_{i}{j}{k}: {total}"
    return int(round(total_re))


def fusion_rule(chi: np.ndarray, sizes: np.ndarray, names: list[str],
                 dims: list[int], i: int, j: int) -> dict[int, int]:
    """Return {k: N_{ijk}} for all k, representing
    pi_i (x) pi_j = sum_k N_{ijk} pi_k."""
    return {k: fusion_coefficient(chi, sizes, i, j, k) for k in range(7)}


def format_fusion(rule: dict[int, int], names: list[str]) -> str:
    parts = []
    for k, n in rule.items():
        if n > 0:
            if n == 1:
                parts.append(names[k])
            else:
                parts.append(f"{n}*{names[k]}")
    return " + ".join(parts) if parts else "0"


# =========================================================================
# Main.
# =========================================================================

def main():
    print("=" * 72)
    print(" b2.4 -- Fusion rules of 2T and the 21-channel interpretation")
    print("=" * 72)

    chi, sizes, names, dims = tT_character_table()
    print(f"\n[1] 2T character table (7 irreps):")
    print(f"    Irrep dims: {dims}  (sum of squares = "
          f"{sum(d*d for d in dims)})")

    # --- (2) Full fusion table -------------------------------------------
    print(f"\n[2] Fusion products pi_i (x) pi_j (21 unordered i<j pairs)")
    print(f"    Total should cover 21 = C(7,2) pairs.\n")
    total_dim = 0
    pair_data = []
    for i, j in combinations(range(7), 2):
        rule = fusion_rule(chi, sizes, names, dims, i, j)
        pair_dim = sum(rule[k] * dims[k] for k in rule)
        expected = dims[i] * dims[j]
        assert pair_dim == expected, (
            f"dim mismatch for pi_{i} (x) pi_{j}: "
            f"{pair_dim} vs {expected}")
        rhs = format_fusion(rule, names)
        print(f"    {names[i]:<9s} (x) {names[j]:<9s} = {rhs}")
        total_dim += pair_dim
        pair_data.append((i, j, rule, pair_dim))

    print(f"\n    Total dim sum over 21 pairs: {total_dim}")
    print(f"    Expected: sum_{{i<j}} dim(pi_i)*dim(pi_j) = "
          f"{sum(dims[i]*dims[j] for i,j in combinations(range(7),2))}")

    # --- (3) Channel-counting summary ------------------------------------
    print(f"\n[3] Channel structure")
    print(f"    Number of unordered pairs (i,j) with i != j: "
          f"{len(list(combinations(range(7), 2)))}")
    print(f"    (Equals 21 = C(7,2); matches Paper 15's exponent.)")

    # Count each pair's dim and per-channel irrep content.
    print(f"\n    Pair dims (sorted):")
    sorted_dims = sorted([(pd, names[i], names[j])
                           for i, j, _, pd in pair_data])
    dim_counter = {}
    for pd, _, _ in sorted_dims:
        dim_counter[pd] = dim_counter.get(pd, 0) + 1
    for d, c in sorted(dim_counter.items()):
        print(f"      dim {d}: {c} pair(s)")

    # --- (4) Global sum rules for the 21-channel hypothesis --------------
    print(f"\n[4] Sum rules")
    total_irrep_count = {k: 0 for k in range(7)}
    for i, j, rule, _ in pair_data:
        for k, n in rule.items():
            total_irrep_count[k] += n
    print(f"    Total multiplicity of each irrep across 21 pairs:")
    for k in range(7):
        print(f"      {names[k]:<10s}: {total_irrep_count[k]}")
    print(f"    Total: {sum(total_irrep_count.values())}")

    # --- (5) Ansatz: 21 channels -> alpha^(21/2) ------------------------
    print(f"\n[5] Ansatz connecting 21 channels to alpha^(21/2):")
    print()
    print("    Paper 13's rule on an OPEN (transition) amplitude:")
    print("      each crossing contributes sqrt(alpha).")
    print()
    print("    Ansatz: the transition amplitude  <e_SM | sigma | e_grav>")
    print("    is a sum/product of 21 sub-amplitudes, one for each")
    print("    irrep pair (pi_i, pi_j) of 2T,  i < j.  If each sub-")
    print("    amplitude carries exactly one gauge-holonomy crossing,")
    print("    then")
    print()
    print("        A ~ (sqrt(alpha))^21 = alpha^(21/2).")
    print()
    print("    Concrete realisation (conjectural): associate each 2T")
    print("    irrep pi_i with a conjugacy class of the trefoil's 2I")
    print("    orbit representatives; each pair of irreps produces a")
    print("    braid crossing where the two strands of conjugate")
    print("    trefoils meet.  Counting gives 21 = C(|Irr(2T)|, 2) = 21.")

    # --- Plot: fusion adjacency / K_7 with pair dims ---------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    # Place 7 irreps around a circle.
    pos = {}
    for k in range(7):
        angle = 2 * np.pi * k / 7 + np.pi / 2
        pos[k] = (np.cos(angle), np.sin(angle))
        ax.plot(pos[k][0], pos[k][1], "o", color="tab:blue",
                 markersize=20)
        ax.annotate(names[k], pos[k],
                     textcoords="offset points",
                     xytext=(15, 15), fontsize=11)
    # Draw 21 edges, color by dim.
    for i, j, rule, pd in pair_data:
        (x1, y1), (x2, y2) = pos[i], pos[j]
        color = plt.cm.viridis(pd / 6.0)  # dims range 1..6
        ax.plot([x1, x2], [y1, y2], "-", color=color, alpha=0.6,
                 lw=0.8 + 0.3 * pd)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"K_7 on 2T irreps — 21 pair channels"
                 r"  (=  dim$(\Lambda^2 C^7)$)",
                 fontsize=12)
    out = Path(__file__).parent / "nwt_2T_fusion_21channels_b2_4.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\n[6] Plot: {out}")


if __name__ == "__main__":
    main()
