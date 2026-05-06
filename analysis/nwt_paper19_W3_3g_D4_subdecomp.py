#!/usr/bin/env python3
"""
Paper 19 -- W3.3 step G: D_4 sub-decomposition of the K_7 matter sector.

W3.3-F showed that an O_h-equivariant E_g coupling (e.g., T_1, T_2) shifts
5 of 6 anti-sym K_7 modes -- only the 2 A_1g modes are Schur-protected.

This step asks: does a SMALLER subgroup like D_4 (tetragonal, fixing the
z-axis) reveal additional symmetry protection?  The cubic decomposition
6 = 1 + 2 + 3 (A_1g + E_g + T_1u under O_h) refines under D_4 as:

  A_1g of O_h  ->  A_1
  E_g of O_h   ->  A_1 + B_1     (cubic doublet splits)
  T_1u of O_h  ->  A_2 + E       (cubic triplet splits)

So the 5-dim eigenspace at energy 7 (= 1 mode A_1 from E_g + 1 mode B_1
from E_g + 1 mode A_2 from T_1u + 2 modes E from T_1u) has finer
D_4 structure than under O_h.

Under T_1 = B_1 of D_4 (the "x^2 - y^2" shear):
  - <A_1 | T_1 | A_1>   = 0       (Schur: A_1 x B_1 = B_1)
  - <B_1 | T_1 | B_1>   = 0       (Schur: B_1 x B_1 = A_1)
  - <A_2 | T_1 | A_2>   = 0       (Schur: A_2 x B_1 = B_2)
  - <E | T_1 | E>       != 0      (E x B_1 = E)
  - <A_1 | T_1 | B_1>   != 0      (off-diagonal A_1 <-> B_1 mixing)

So T_1 has:
  - Diagonal A_1, B_1, A_2 zero contributions
  - Off-diagonal A_1<->B_1 mixing (gives ±a eigenvalue split)
  - E diagonal (gives ±b eigenvalue split)
  - A_2 isolated at zero shift  <-- D_4 SYMMETRY-PROTECTED MODE

Eigenvalue spectrum of T_1 within the 5-dim eigenspace:
  +/- a  (A_1-B_1 mixing)
  0     (A_2 protected)
  +/- b  (E doublet split)

That's 4 nonzero shifts + 1 zero shift = 5 of 5 modes accounted for, with
the A_2 = T_1u_z mode SYMMETRY-PROTECTED under D_4.

Compared to W3.3-F (under O_h, 5 of 6 shift, only A_1g doesn't):  D_4 has
ONE EXTRA MODE PROTECTED -- the cubic-T_1u-z-axis mode -- via D_4-A_2
Schur protection.

CHECK: this matches the W3.3-F numerical observation that under T_1
(= phi_1 sweep), the T_1u_x-or-T_1u_z mode had shift exactly 0.

Under T_2 = A_1 of D_4 (the "3z^2 - r^2" shear):
  - All D_4 irreps contain A_1 in their tensor product with themselves
  - So all 5 modes shift -> consistent with W3.3-F numerical
    "phi_2 shifts 6 modes".

Output -> analysis/output/W3_3g_D4/
  d4_decomposition.txt   : irrep tally + Schur predictions
  shift_matrix.png       : T_1 matrix on D_4 basis
  summary.txt
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[0] / "output" / "W3_3g_D4"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Setup (same as W3.3-F)
# ---------------------------------------------------------------------------

POSITIONS = np.array([
    [0, 0, 0],   # v_0  origin
    [+1, 0, 0],  # v_1  +x
    [-1, 0, 0],  # v_2  -x
    [0, +1, 0],  # v_3  +y
    [0, -1, 0],  # v_4  -y
    [0, 0, +1],  # v_5  +z
    [0, 0, -1],  # v_6  -z
], dtype=np.float64)


def k7_laplacian() -> np.ndarray:
    return 7.0 * np.eye(7) - np.ones((7, 7))


T_1 = np.diag([0, +1.0, +1.0, -1.0, -1.0, 0.0, 0.0]) / np.sqrt(2.0)
T_2 = np.diag([0, +1.0, +1.0, +1.0, +1.0, -2.0, -2.0]) / np.sqrt(6.0)


def H0(delta: float) -> np.ndarray:
    H = k7_laplacian().copy()
    H[0, 0] += delta
    return H


# ---------------------------------------------------------------------------
# Irrep basis under D_4 (tetragonal, z-axis fixed)
# ---------------------------------------------------------------------------

# Within the 5-dim eigenspace at energy 7 (delta != 0 case), the
# decomposition under D_4 ⊂ O_h is:
#
#   1 mode A_1 (from cubic E_g_2 = 3z^2 - r^2)
#   1 mode B_1 (from cubic E_g_1 = x^2 - y^2)
#   1 mode A_2 (from cubic T_1u_z = +z - -z)            <- z-axis vector
#   2 modes E (from cubic T_1u_x, T_1u_y)               <- xy doublet
#
# Plus 2 modes outside the energy-7 eigenspace (the A_1g_v0, A_1g_others
# split by the Delta term).

D4_BASIS = {
    "A_1_from_E_g":  np.array([0, +1, +1, +1, +1, -2, -2], dtype=np.float64) / (2.0 * np.sqrt(3.0)),
    "B_1_from_E_g":  np.array([0, +1, +1, -1, -1,  0,  0], dtype=np.float64) / 2.0,
    "A_2_from_T_1u": np.array([0,  0,  0,  0,  0, +1, -1], dtype=np.float64) / np.sqrt(2.0),
    "E_x_from_T_1u": np.array([0, +1, -1,  0,  0,  0,  0], dtype=np.float64) / np.sqrt(2.0),
    "E_y_from_T_1u": np.array([0,  0,  0, +1, -1,  0,  0], dtype=np.float64) / np.sqrt(2.0),
}


# ---------------------------------------------------------------------------
# Verify D_4 irrep classification + compute T_1, T_2 in this basis
# ---------------------------------------------------------------------------

def compute_perturbation_matrix(V: np.ndarray, basis: dict) -> tuple:
    """Compute V in the D_4 basis: V_ij = <basis_i | V | basis_j>."""
    names = list(basis.keys())
    n = len(names)
    M = np.zeros((n, n), dtype=np.float64)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            M[i, j] = float(basis[ni] @ V @ basis[nj])
    return names, M


def main():
    print("=" * 70)
    print("W3.3-G   D_4 sub-decomposition of K_7 matter sector")
    print("=" * 70)

    # Verify our D_4 basis spans the 5-dim eigenspace at energy 7 (delta=1).
    delta = 1.0
    H = H0(delta)
    eigvals_h0, eigvecs_h0 = np.linalg.eigh(H)

    print(f"\nH_0 eigenvalues at delta = {delta}:")
    for k in range(7):
        print(f"  mode {k}: {eigvals_h0[k]:.4f}")

    # Project D_4 basis onto the 5-dim eigenspace at 7.
    print(f"\nProjection of D_4 basis vectors onto eigenspaces of H_0:")
    print(f"  {'D_4 basis':>20}  {'overlap with eigenvalue 7':>30}")
    print("  " + "-" * 55)
    for name, b in D4_BASIS.items():
        # For each eigenvector at energy 7, compute <eigvec | b>
        in_7_subspace = 0.0
        for k in range(7):
            if abs(eigvals_h0[k] - 7.0) < 1e-6:
                in_7_subspace += float(eigvecs_h0[:, k] @ b) ** 2
        print(f"  {name:>20}  {in_7_subspace:>30.6f}")

    # Compute T_1 and T_2 matrices in D_4 basis.
    print(f"\nT_1 matrix in D_4 basis:")
    names, M_T1 = compute_perturbation_matrix(T_1, D4_BASIS)
    print(f"  {'':>20}  " + "  ".join(f"{n:>12}" for n in names))
    for i, ni in enumerate(names):
        print(f"  {ni:>20}  " + "  ".join(f"{M_T1[i, j]:>+12.6f}" for j in range(len(names))))

    print(f"\nT_2 matrix in D_4 basis:")
    names, M_T2 = compute_perturbation_matrix(T_2, D4_BASIS)
    print(f"  {'':>20}  " + "  ".join(f"{n:>12}" for n in names))
    for i, ni in enumerate(names):
        print(f"  {ni:>20}  " + "  ".join(f"{M_T2[i, j]:>+12.6f}" for j in range(len(names))))

    # Eigenvalues of T_1 and T_2 within the 5-dim eigenspace.
    eig_T1 = np.linalg.eigvalsh(0.5 * (M_T1 + M_T1.T))
    eig_T2 = np.linalg.eigvalsh(0.5 * (M_T2 + M_T2.T))

    print(f"\nEigenvalues of T_1 in 5-dim eigenspace (raw):")
    for e in eig_T1:
        print(f"  {e:>+12.6f}")

    print(f"\nEigenvalues of T_2 in 5-dim eigenspace (raw):")
    for e in eig_T2:
        print(f"  {e:>+12.6f}")

    # With lambda = 0.5, shifts are 0.5 * eigenvalues:
    lam = 0.5
    print(f"\nShifts at lambda = {lam}:")
    print(f"  Under T_1 (B_1 of D_4): {[round(lam * e, 4) for e in eig_T1]}")
    print(f"  Under T_2 (A_1 of D_4): {[round(lam * e, 4) for e in eig_T2]}")

    # Schur prediction summary.
    print(f"\nSCHUR ANALYSIS:")
    print(f"  T_1 transforms as B_1 of D_4.  Diagonal entries:")
    print(f"    <A_1 | T_1 | A_1> = {M_T1[0,0]:>+.6f}  (predicted 0; A_1 x B_1 = B_1)")
    print(f"    <B_1 | T_1 | B_1> = {M_T1[1,1]:>+.6f}  (predicted 0; B_1 x B_1 = A_1)")
    print(f"    <A_2 | T_1 | A_2> = {M_T1[2,2]:>+.6f}  (predicted 0; A_2 x B_1 = B_2)")
    print(f"    <E_x | T_1 | E_x> = {M_T1[3,3]:>+.6f}  (E x B_1 = E, allowed)")
    print(f"    <E_y | T_1 | E_y> = {M_T1[4,4]:>+.6f}")
    print(f"    OFF-DIAG: <A_1 | T_1 | B_1> = {M_T1[0,1]:>+.6f}  "
          f"(A_1 x B_1 = B_1, allowed -> mixing)")
    print(f"    OFF-DIAG: <A_2 | T_1 | E_x> = {M_T1[2,3]:>+.6f}  "
          f"(A_2 x B_1 = B_2 != E, predicted 0)")

    # Comparison with W3.3-F numerical (W3.3-F gave shifts ±0.204, ±0.354, 0
    # under T_1).
    print(f"\nCOMPARISON WITH W3.3-F NUMERICAL (under T_1):")
    print(f"  W3.3-F:  shifts {{-0.354, -0.204, 0, +0.204, +0.354}}")
    print(f"  W3.3-G:  shifts {[round(lam * e, 4) for e in sorted(eig_T1)]}")

    # ---------- Summary ----------
    summary = ["Paper 19 -- W3.3-G   D_4 sub-decomposition of K_7 matter",
               "=" * 70,
               "",
               "Setup:",
               "  Same K_7 placement as W3.3-F (cubic O_h, vertices at",
               "  origin + ±x, ±y, ±z).  H_0 = L_K7 + Delta |v_0><v_0|",
               f"  with Delta = {delta} breaks the A_1g_v0 vs A_1g_others",
               "  degeneracy.",
               "",
               "  We classify the 5-dim eigenspace at energy 7 under D_4",
               "  (the tetragonal subgroup of O_h that fixes z-axis):",
               "    A_1 (1-dim)  : cubic E_g_2 = (3z^2 - r^2)/sqrt(6)",
               "    B_1 (1-dim)  : cubic E_g_1 = (x^2 - y^2)/sqrt(2)",
               "    A_2 (1-dim)  : cubic T_1u_z = (e_+z - e_-z)/sqrt(2)",
               "    E   (2-dim)  : cubic T_1u_x, T_1u_y",
               "  Under O_h the cubic 5 = E_g (2) + T_1u (3) is the only",
               "  decomposition; under D_4 it refines to 1+1+1+2.",
               "",
               "T_1 = (x^2 - y^2) shear is B_1 of D_4.  Schur predictions:",
               "  <A_1 | T_1 | A_1>   = 0     (A_1 x B_1 = B_1 != A_1)",
               "  <B_1 | T_1 | B_1>   = 0     (B_1 x B_1 = A_1 != B_1)",
               "  <A_2 | T_1 | A_2>   = 0     (A_2 x B_1 = B_2 != A_2)",
               "  <E | T_1 | E>      != 0     (E x B_1 = E, allowed)",
               "  <A_1 | T_1 | B_1>  != 0     (off-diagonal mixing)",
               "  <A_2 | T_1 | E>     = 0     (A_2 x B_1 = B_2 != E)",
               "",
               "Computed T_1 matrix in D_4 basis:",
               f"  {'':>20}  " + "  ".join(f"{n:>12}" for n in names),
              ]
    for i, ni in enumerate(names):
        summary.append(
            f"  {ni:>20}  " + "  ".join(f"{M_T1[i, j]:>+12.6f}" for j in range(len(names)))
        )

    summary += [
        "",
        f"Eigenvalues of T_1 within 5-dim eigenspace:",
        f"  {sorted([round(e, 4) for e in eig_T1])}",
        "",
        "INTERPRETATION:",
        "  - A_1, B_1 sub-blocks: off-diagonal mixing only (no diagonal).",
        "    The A_1-B_1 2x2 sub-block has eigenvalues +/- 1/sqrt(6) ~",
        "    +/- 0.408 (raw); times lambda = 0.5: +/- 0.204.",
        "  - A_2 sub-block: ZERO at first order (D_4 Schur-protected!).",
        "    This is the cubic T_1u_z = z-axis matter mode -- it is",
        "    ALWAYS unaffected by an x^2 - y^2 perturbation, by D_4",
        "    symmetry.  This is one mode MORE protected than the O_h",
        "    Schur analysis revealed.",
        "  - E sub-block: diagonal +1/sqrt(2) and -1/sqrt(2) (eigenvalues",
        "    of B_1 acting on E doublet); times lambda = 0.5: +/- 0.354.",
        "",
        "T_2 = (3z^2 - r^2) shear is A_1 of D_4 (z-symmetric).  Schur:",
        "  <X | T_2 | X> != 0 for ALL X (since A_1 x X = X always).",
        "  So all 5 modes shift under T_2; no D_4 protection.",
        "",
        "  Computed T_2 matrix in D_4 basis (verify diagonal):",
        f"  {'':>20}  " + "  ".join(f"{n:>12}" for n in names),
    ]
    for i, ni in enumerate(names):
        summary.append(
            f"  {ni:>20}  " + "  ".join(f"{M_T2[i, j]:>+12.6f}" for j in range(len(names)))
        )

    summary += [
        "",
        f"Eigenvalues of T_2 within 5-dim eigenspace:",
        f"  {sorted([round(e, 4) for e in eig_T2])}",
        "",
        "VERDICT:",
        "  D_4 sub-decomposition reveals an EXTRA SYMMETRY-PROTECTED",
        "  MODE under T_1 perturbation: the A_2 = z-axis-T_1u mode is",
        "  unaffected by (x^2 - y^2) shears, by D_4 Schur (A_2 x B_1",
        "  = B_2 != A_2).",
        "",
        "  This shows that going to a SUBGROUP of O_h gives FINER",
        "  selectivity in matter-gauge coupling.  Under O_h alone, all 5",
        "  E_g + T_1u modes shift under T_1; under D_4, only 4 of them",
        "  shift, with the A_2 mode protected.",
        "",
        "  Physical reading: voxel-space's e_g shear coupling can be",
        "  TUNED for selectivity by aligning with specific axes.  An",
        "  (x^2 - y^2) shear leaves the z-axis matter unaffected.  This",
        "  is closer to the SM Higgs mechanism (selective coupling) than",
        "  the O_h-equivariant case (couples to all charged matter).",
        "",
        "STILL NOT FULL SM HIGGS:",
        "  D_4 protects 1 mode out of 5 (rather than 4 of 5 in the SM",
        "  Higgs case where only 1-doublet shifts).  To match SM, we'd",
        "  need protection of 3-color-triplet (= cubic T_1u) AND singlet",
        "  (= cubic A_1g), with only the 2-doublet (= cubic E_g)",
        "  shifting.  This requires per-particle Yukawa input as in SM,",
        "  not just symmetry analysis.",
        "",
        "STRUCTURAL TAKEAWAY:",
        "  The hierarchy {O_h, D_4, ...} of subgroups gives a hierarchy",
        "  of selectivity.  Each subgroup provides additional Schur",
        "  protection.  Combined with the geometric flexibility of which",
        "  subgroup the 'Higgs VEV' picks out, voxel-space can produce",
        "  varied matter-mass structures -- but never the *ad hoc*",
        "  per-particle SM mass hierarchy without additional input.",
    ]

    summary_text = "\n".join(summary)
    (OUT / "summary.txt").write_text(summary_text)
    print()
    print(summary_text)
    print()
    print(f"Wrote {OUT/'summary.txt'}")


if __name__ == "__main__":
    main()
