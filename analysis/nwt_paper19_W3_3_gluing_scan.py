#!/usr/bin/env python3
"""
Paper 19 -- W3.3: Gluing-rule scan for SM-like K_7 mass splittings.

W3.2 step 1 found that with the simplest "symmetric" gluing rule
(qubit i in voxel A coupled to qubit i in voxel B across every shared
face), the 6 anti-symmetric internal branches of the K_7 graph state
are degenerate at gap m^2 = 7 kappa_int = lambda_max(L_K_7).

Different gluing rules can BREAK this 6-fold degeneracy.  In the
present family we generalise to PERMUTATION gluings: for each of
the 12 FCC face directions d in {0..11}, pick a permutation
sigma_d in S_7 specifying which qubit indices couple across each
face of that direction.  Energy:

    E_face = (kappa_glue/2) sum_{i=1..7}
             (psi_i^A - psi_{sigma_d(i)}^B)^2.

Symmetric gluing = (sigma_d = identity for all d).  Different
sigma_d patterns split the 6-fold degenerate K_7 anti-symmetric
optical level into multiplets, with multiplicities determined by
how the sigma_d generate a subgroup of S_7 and how the lattice
average of the 12 permutation matrices acts on the 6-dim standard
S_7 representation.

The hunt: does any natural gluing rule give an SM-like multiplet
structure?  Candidate target patterns:

  - 3 + 3 (two lepton generations × charge eigenstates, etc.)
  - 1 + 2 + 3 (Wigner-multiplet-like)
  - 1 + 5 (singlet + SU(5) fundamental)
  - 2 + 2 + 2 (three doublets)
  - 6 (no splitting, generic transverse rule)

We do NOT yet attempt to reproduce specific mass RATIOS -- those
depend on kappa_glue / kappa_int and on the gluing rule's specific
matrix structure.  Here we just classify SPLITTING PATTERNS.

Rules scanned:

  R0  identity            symmetric, baseline (6-fold degenerate)
  R1  cyclic_dir          sigma_d = (1 2 3 4 5 6 7)^d  (cyclic shift by direction)
  R2  random              i.i.d. random permutations per direction (3 samples)
  R3  cubic_axis_swap     for direction d in {0..11}, swap qubit-pair (a, b)
                           where (a, b) is the FCC direction's two non-zero
                           components mapped to qubit labels 1..7 modulo 7
  R4  pairwise_transposi  sigma_d = (1 d+2)  (transposition swapping 1 and d+2)
  R5  involution_set      a curated involution set respecting the 12-fold
                           directional structure (e.g. opposite directions
                           use inverse permutations)

For each rule, we:
  1. Build the K_7-internal Hessian with per-face permutation gluing.
  2. Diagonalise.
  3. Identify the 6 * n_cells anti-symmetric optical eigenvalues
     (those with omega^2 close to 7 kappa_int + 12 kappa_glue at k = 0).
  4. Histogram the eigenvalue distribution.
  5. Classify the multiplet structure at k = 0 (i.e. the eigenvalues
     of the 7x7 average permutation matrix, restricted to the 6-dim
     anti-symmetric subspace).

Output -> analysis/output/W3_3_gluing_scan/
  spectrum_RULE.png   : per-rule eigenvalue histogram
  multiplet_table.png : multiplet structure summary
  summary.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/home/jim/repos/Morphospace")

import matplotlib.pyplot as plt
import numpy as np

from morphospace.physics.rhombic_grid import (
    build_rhombic_topology,
    fcc_grid_positions,
    init_regular_3d,
)

OUT = Path(__file__).resolve().parents[0] / "output" / "W3_3_gluing_scan"
OUT.mkdir(parents=True, exist_ok=True)


# 12 canonical FCC face directions (matching Morphospace's grids.py order).
FCC_DIRS = np.array([
    [+1, +1,  0], [+1, -1,  0], [-1, +1,  0], [-1, -1,  0],
    [+1,  0, +1], [ 0, +1, +1], [-1,  0, +1], [ 0, -1, +1],
    [+1,  0, -1], [ 0, +1, -1], [-1,  0, -1], [ 0, -1, -1],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def build_tissue(n: int, scale: float = 1.0):
    pos = fcc_grid_positions(n, n, n, scale)
    topo = build_rhombic_topology(pos, scale)
    state = init_regular_3d(topo, scale)
    return topo, np.asarray(state.vertices, dtype=np.float64)


def interface_directions(topo) -> np.ndarray:
    """Return interface direction *index* (0..11) for each interface_pair.

    Direction index = which of the 12 FCC neighbours the j-th cell is from
    the i-th cell.
    """
    centers = np.asarray(topo.cell_centers, dtype=np.float64)
    pairs = np.asarray(topo.interface_pairs, dtype=np.int64)
    dir_idx = np.zeros(len(pairs), dtype=np.int64)
    for k, (ci, cj) in enumerate(pairs):
        diff = centers[cj] - centers[ci]
        # Find which FCC direction matches diff (up to sign).
        # In Morphospace's convention, interface_pairs has (min, max), so
        # diff might be reversed; find best-matching direction.
        best = -1
        best_score = -np.inf
        for d in range(12):
            score = float(np.dot(diff, FCC_DIRS[d]))
            if score > best_score:
                best_score = score
                best = d
        dir_idx[k] = best
    return dir_idx


# ---------------------------------------------------------------------------
# Permutation gluing -> Hessian
# ---------------------------------------------------------------------------

def perm_matrix(sigma: np.ndarray) -> np.ndarray:
    """Permutation matrix P with (P)_{i,j} = 1 if sigma(i) == j else 0.

    Convention: P @ v permutes a vector v according to sigma; the Hessian
    off-diagonal coupling involves P + P^T (symmetrised).
    """
    n = len(sigma)
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        P[i, int(sigma[i])] = 1.0
    return P


def k7_laplacian() -> np.ndarray:
    return 7.0 * np.eye(7) - np.ones((7, 7))


def internal_hessian_perm_gluing(topo, kappa_int: float,
                                 kappa_glue: float,
                                 sigma_per_dir: list[np.ndarray]
                                 ) -> np.ndarray:
    """Hessian with per-face-direction permutation gluing.

    For each interface (ci, cj) with direction index d:
      E_face = (kappa_glue/2) * sum_i (q_i^ci - q_{sigma_d(i)}^cj)^2.

    Hessian block structure:
      diag(ci): + kappa_glue * I_7      (per face incident to ci)
      diag(cj): + kappa_glue * I_7      (per face incident to cj)
      offdiag (ci, cj): - kappa_glue * P_{sigma_d}
      offdiag (cj, ci): - kappa_glue * P_{sigma_d}^T
    Plus K_7 internal: kappa_int * L_K7 in each diagonal block.
    """
    n_c = topo.n_cells
    n = 7 * n_c
    H = np.zeros((n, n), dtype=np.float64)

    L_K7 = k7_laplacian()
    pairs = np.asarray(topo.interface_pairs, dtype=np.int64)
    dir_idx = interface_directions(topo)

    # K_7 internal at each cell.
    for c in range(n_c):
        H[7 * c:7 * c + 7, 7 * c:7 * c + 7] += kappa_int * L_K7

    # Inter-voxel coupling per face.
    for k, (ci, cj) in enumerate(pairs):
        d = int(dir_idx[k])
        sigma = sigma_per_dir[d]
        P = perm_matrix(sigma)
        i0 = 7 * int(ci)
        j0 = 7 * int(cj)
        # Diagonal contribution to ci and cj.
        H[i0:i0 + 7, i0:i0 + 7] += kappa_glue * np.eye(7)
        H[j0:j0 + 7, j0:j0 + 7] += kappa_glue * np.eye(7)
        # Off-diagonal: -P (and -P^T for symmetry).
        H[i0:i0 + 7, j0:j0 + 7] -= kappa_glue * P
        H[j0:j0 + 7, i0:i0 + 7] -= kappa_glue * P.T

    return H


# ---------------------------------------------------------------------------
# Gluing rule library
# ---------------------------------------------------------------------------

def rule_identity() -> list[np.ndarray]:
    """R0 -- symmetric (identity) gluing on all 12 directions."""
    return [np.arange(7, dtype=np.int64) for _ in range(12)]


def rule_cyclic_dir() -> list[np.ndarray]:
    """R1 -- sigma_d = (cyclic shift by d): qubit i -> qubit (i + d) mod 7."""
    return [np.roll(np.arange(7), d) for d in range(12)]


def rule_random(seed: int) -> list[np.ndarray]:
    """R2 -- i.i.d. random permutations per direction."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(7) for _ in range(12)]


def rule_pairwise_transposition() -> list[np.ndarray]:
    """R4 -- sigma_d = (1, (d % 6) + 2): swap qubit 0 with qubit (d mod 6)+1.

    All 6 non-trivial qubit-pair swaps appear (one for each adjacent-pair
    direction class).
    """
    out = []
    for d in range(12):
        sig = np.arange(7, dtype=np.int64)
        i = (d % 6) + 1
        sig[0], sig[i] = sig[i], sig[0]
        out.append(sig)
    return out


def rule_cubic_axis_swap() -> list[np.ndarray]:
    """R3 -- the FCC direction (a_x, a_y, a_z) has 2 nonzero components.

    For direction d with non-zero indices p, q in {0,1,2}, swap qubits
    p+1 and q+1 (using qubits 1, 2, 3 for x, y, z and leaving 0, 4, 5, 6
    fixed).
    """
    out = []
    for d in range(12):
        v = FCC_DIRS[d]
        nz = [k for k in range(3) if v[k] != 0]
        sig = np.arange(7, dtype=np.int64)
        if len(nz) == 2:
            a, b = nz[0] + 1, nz[1] + 1
            sig[a], sig[b] = sig[b], sig[a]
        out.append(sig)
    return out


def rule_inverse_pair() -> list[np.ndarray]:
    """R5 -- opposite FCC directions get inverse permutations.

    Build an involution set: directions 0..5 get arbitrary permutations,
    directions 6..11 (opposites) get the inverses.  Use cyclic shifts.
    """
    out = []
    for d in range(12):
        if d < 6:
            out.append(np.roll(np.arange(7), d + 1))
        else:
            # Inverse of cyclic shift by k is shift by -k.
            inv_shift = -((d - 6) + 1)
            out.append(np.roll(np.arange(7), inv_shift))
    return out


# ---------------------------------------------------------------------------
# Diagnostic: at-k=0 multiplet decomposition
# ---------------------------------------------------------------------------

def k0_multiplet(sigma_per_dir: list[np.ndarray]) -> dict:
    """Compute eigenvalue spectrum of the average permutation matrix
    P_avg = (1/12) sum_d P_{sigma_d}, restricted to the 6-dim anti-
    symmetric subspace of K_7.

    The k = 0 anti-symmetric optical-mode eigenvalues at this k are
        omega^2(k=0)  =  7 kappa_int + 12 kappa_glue (1 - lambda(P_avg))
    so the multiplet structure of P_avg's eigenvalues directly gives
    the gluing-induced mass splitting.  Generally degenerate eigenvalues
    of P_avg correspond to degenerate phonon multiplets.
    """
    P_sum = sum(perm_matrix(s) for s in sigma_per_dir)
    P_avg = P_sum / 12.0
    P_sym = 0.5 * (P_avg + P_avg.T)

    # Restrict to anti-symmetric subspace (orthogonal to all-ones).
    e_sym = np.ones(7) / np.sqrt(7.0)
    Q = np.eye(7) - np.outer(e_sym, e_sym)
    P_anti = Q @ P_sym @ Q
    # Diagonalize within anti-symmetric subspace (will give 1 zero eigvec
    # corresponding to the projector kernel + 6 real ones).
    eigvals, eigvecs = np.linalg.eigh(P_anti)
    # Drop the projector-kernel eigenvalue: its eigenvector is e_sym
    # (largest overlap with the all-ones direction).  The remaining 6
    # are the genuine anti-symmetric eigenvalues.
    overlaps = np.abs(eigvecs.T @ e_sym)
    drop_idx = int(np.argmax(overlaps))  # highest overlap = e_sym direction
    real_eigs = np.delete(eigvals, drop_idx)
    real_eigs = np.sort(real_eigs)

    # Cluster nearly-equal eigenvalues
    eps = 1e-7
    clusters = []
    for v in real_eigs:
        placed = False
        for c in clusters:
            if abs(c["lambda"] - v) < eps:
                c["mult"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"lambda": float(v), "mult": 1})

    return {
        "P_avg_eigs": real_eigs.tolist(),
        "multiplets": clusters,
        "n_unique": len(clusters),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_rule(name: str, sigma_per_dir: list[np.ndarray],
             topo, kappa_int: float, kappa_glue: float
             ) -> dict:
    """Diagonalise internal Hessian for a given gluing rule, return analysis."""
    H = internal_hessian_perm_gluing(topo, kappa_int, kappa_glue, sigma_per_dir)
    eigvals = np.linalg.eigvalsh(0.5 * (H + H.T))
    eigvals = np.where(eigvals < 0, 0.0, eigvals)
    omega = np.sqrt(eigvals)

    # k=0 multiplet from the analytic average-permutation argument.
    mp = k0_multiplet(sigma_per_dir)

    return {
        "name": name,
        "omega": omega,
        "k0_multiplets": mp["multiplets"],
        "k0_n_unique": mp["n_unique"],
        "k0_eigs": mp["P_avg_eigs"],
        "sigma_per_dir": sigma_per_dir,
    }


def main():
    n_grid = 4
    kappa_int = 1.0
    kappa_glue = 0.5

    print(f"NWT W3.3   gluing-rule scan   "
          f"kappa_int = {kappa_int}, kappa_glue = {kappa_glue}\n")

    print("Building tissue ...", flush=True)
    topo, _ = build_tissue(n=n_grid)
    n_c = topo.n_cells
    print(f"  n_cells = {n_c},  n_internal_DOFs = {7 * n_c}")

    rules = [
        ("R0_identity",        rule_identity()),
        ("R1_cyclic_dir",      rule_cyclic_dir()),
        ("R2_random_seed1",    rule_random(seed=1)),
        ("R2_random_seed2",    rule_random(seed=2)),
        ("R2_random_seed3",    rule_random(seed=3)),
        ("R3_cubic_axis_swap", rule_cubic_axis_swap()),
        ("R4_pair_transp",     rule_pairwise_transposition()),
        ("R5_inverse_pair",    rule_inverse_pair()),
    ]

    results = []
    for name, sig in rules:
        print(f"\nRule {name} ...", flush=True)
        r = run_rule(name, sig, topo, kappa_int, kappa_glue)
        results.append(r)
        # Show k=0 multiplet structure.
        mults = r["k0_multiplets"]
        pattern = " + ".join(f"{c['mult']}" for c in mults)
        print(f"  k=0 multiplet pattern (anti-sym 6-dim): {pattern}")
        print(f"  P_avg eigenvalues: "
              f"{[round(e, 6) for e in r['k0_eigs']]}")
        # K_7 intra-cell gap is sqrt(7 kappa_int) regardless of gluing.
        # The optical band runs from sqrt(7 kappa_int) (at lambda(P_avg)=1)
        # downward as kappa_glue increases.
        # Expected eigenvalue range: 7 kappa_int + (some inter-vox shift).

    # ---------- Plot 1: spectra per rule ----------
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, r in zip(axes.flat, results):
        omega2 = r["omega"] ** 2
        # Drop the 1 sym zero mode for clarity
        omega2_nontriv = omega2[1:]
        ax.hist(omega2_nontriv, bins=60, color="steelblue",
                edgecolor="k", alpha=0.7)
        ax.axvline(7 * kappa_int, color="r", ls="--",
                   label=r"$7\kappa_{int}$ (K7 gap)")
        ax.set_xlabel(r"$\omega^2$")
        ax.set_ylabel("count")
        pattern = "+".join(str(c["mult"]) for c in r["k0_multiplets"])
        ax.set_title(f"{r['name']}\nk=0 multiplet: {pattern}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "spectra_per_rule.png", dpi=130)
    plt.close(fig)

    # ---------- Plot 2: multiplet structures ----------
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    rule_names = [r["name"] for r in results]
    for i, r in enumerate(results):
        eigs = r["k0_eigs"]
        # Plot each eigenvalue as a horizontal line at that rule index.
        for v in eigs:
            # Convert P_avg eigenvalue to mass^2 contribution:
            # m^2(eig) = 7 kappa_int + 12 kappa_glue (1 - lambda).
            m2 = 7 * kappa_int + 12 * kappa_glue * (1 - v)
            ax.scatter(i, m2, s=80, color="tab:blue", alpha=0.6,
                       edgecolor="k")
    ax.set_xticks(range(len(rule_names)))
    ax.set_xticklabels(rule_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(r"$m^2$ (k=0 anti-symmetric mode)")
    ax.set_title(r"K7 anti-symmetric optical-mode masses by gluing rule "
                 r"(kappa_int = 1, kappa_glue = 0.5)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "multiplet_structures.png", dpi=130)
    plt.close(fig)

    # ---------- Summary ----------
    summary = ["Paper 19 -- W3.3   Gluing-rule scan for K_7 mass splittings",
               "FCC rhombic-dodecahedral spacetime tissue",
               "=" * 70,
               f"n = {n_grid},  n_cells = {n_c},  "
               f"kappa_int = {kappa_int},  kappa_glue = {kappa_glue}",
               "",
               "K_7 anti-symmetric optical mode mass^2 at k = 0:",
               "  m^2 = 7 kappa_int + 12 kappa_glue (1 - lambda(P_avg))",
               "",
               f"{'rule':>22}  {'k=0 multiplet':>20}  {'P_avg eigenvalues':>40}",
               "-" * 90,
              ]
    for r in results:
        pattern = "+".join(str(c["mult"]) for c in r["k0_multiplets"])
        eigs_str = "[" + ", ".join(f"{e:+.4f}" for e in r["k0_eigs"]) + "]"
        summary.append(f"{r['name']:>22}  {pattern:>20}  {eigs_str:>40}")

    summary += [
        "",
        "INTERPRETATION:",
        "  R0 (identity = symmetric gluing) gives 6-fold degenerate optical",
        "    branches as expected (W3.2 step 1 baseline): all P_avg",
        "    eigenvalues at +1, single multiplet (6).",
        "  R1 (cyclic) generally gives full splitting into distinct modes",
        "    OR specific multiplets depending on whether the average",
        "    permutation has non-trivial degeneracies.",
        "  R2 (random) typically gives  6  fully split modes (almost surely),",
        "    no multiplet structure -- a generic perturbation.",
        "  R3 (cubic axis-swap) preserves a subgroup of S_7 fixing qubit 0",
        "    and the qubits 4,5,6 (the 'singlet + outer' qubits), so the",
        "    splitting respects the cubic point group; multiplet structure",
        "    encodes how the 6-rep of S_7 restricts to the cubic subgroup.",
        "  R4 (pairwise transposition) and R5 (inverse pair) are different",
        "    representative families.",
        "",
        "SM-LIKE PATTERNS TO LOOK FOR:",
        "  3+3   -- two doublet-like multiplets (charged lepton-quark mass",
        "          structure or two-generation splitting)",
        "  1+2+3 -- Wigner-style multiplet ladder",
        "  1+5   -- singlet + SU(5)-fundamental-like",
        "  2+2+2 -- three doublets (matches 3 SM generations, each a doublet)",
        "  1+1+1+1+1+1 -- 6 distinct masses (3 charged leptons + 3 neutrinos,",
        "          if neutrinos get mass).",
        "  6     -- no splitting (unbroken symmetry; not SM-like at the level",
        "          of mass spectrum but consistent with degenerate-flavour",
        "          multiplet symmetry like in pure-gauge limits).",
        "",
        "Without further structure (kappa_int / kappa_glue ratio, specific",
        "matching to physical particles), this scan only reveals which",
        "structural patterns are *possible* under permutation gluings.",
        "The next steps are:",
        "  - Identify which rule class is selected by W3.4 projection",
        "    (the BI -> 3+1 map should constrain the gluing).",
        "  - Tune kappa_glue / kappa_int to match observed mass ratios",
        "    once a multiplet structure is fixed.",
        "  - Add internal cross-coupling between gluing rules at different",
        "    edges of K_7 (richer than per-face permutation).",
    ]

    summary_text = "\n".join(summary)
    (OUT / "summary.txt").write_text(summary_text)
    print()
    print(summary_text)
    print()
    print(f"Wrote {OUT/'spectra_per_rule.png'}")
    print(f"Wrote {OUT/'multiplet_structures.png'}")
    print(f"Wrote {OUT/'summary.txt'}")


if __name__ == "__main__":
    main()
