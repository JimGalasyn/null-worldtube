#!/usr/bin/env python3
"""
Paper 19 -- W3.2 step 1: Phonon spectrum on FCC voxels with K_7 internal
qubit DOFs.

Building on step 0 (positional pair-spring acoustics on the rhombic-
dodecahedral tessellation), each voxel now carries 7 scalar amplitudes
psi_i(R) -- one per qubit of its internal K_7 graph state.

Two coupling channels:

  Internal (intra-voxel)  : K_7 graph Laplacian on the 7 qubits.
                            kappa_int * Sum_{(i,j) edge of K_7} (psi_i - psi_j)^2.
  Gluing (inter-voxel)    : "symmetric" rule -- at every shared face,
                            qubit i of voxel A is sprung to qubit i of
                            voxel B for all i = 1..7.
                            kappa_glue * Sum_{<A,B> faces} Sum_i (psi_i^A - psi_i^B)^2.

This is the simplest non-trivial gluing.  Richer gluing rules (per-face
qubit-pair selection, K_7-symmetry-equivariant rules, etc.) are deferred
to W3.3.

Decoupled sectors.  In this minimal model the positional (vertex
displacement) and internal (qubit amplitude) DOFs do not mix, so the
Hessian is block-diagonal.  We compute and analyse the two sectors
separately, then combine them in a final spectrum.

K_7 graph Laplacian
  L_K7 = 7 I - J          (J = all-ones 7x7)
  spectrum = {0 (mult 1), 7 (mult 6)}.

The single zero eigenvector is the all-ones vector v_sym = (1,...,1)/sqrt(7);
the 6 non-zero eigenvectors span its orthogonal complement (anti-symmetric
modes).  Under the symmetric gluing, these two sub-sectors decouple in
Fourier space:

  symmetric branch      :  omega^2_sym(k) = kappa_glue * f(k)  (+ kappa_mass)
  6 anti-symmetric br.  :  omega^2_anti(k) = 7 kappa_int + kappa_glue * f(k)
                                            (+ kappa_mass)

where f(k) = 4 * Sum_{d in 12 FCC dirs} sin^2((k . d)/2)
       --> |k|^2 at small k.

So at low |k|:

  symmetric branch: omega ~ c_glue |k|       (linear, gapless if kappa_mass=0)
  anti-symmetric  : omega^2 ~ 7 kappa_int + c_glue^2 |k|^2
                          --> particle-like  E^2 = m^2 + p^2  (Klein-Gordon!)
                          with mass^2 = 7 kappa_int.

The K_7 LAPLACIAN EIGENVALUE 7 (= dim K_7 - 1) is therefore literally
the squared mass of the optical/internal-anti-symmetric modes.  This
is the first concrete structural-identity bridge between graph-state
combinatorics and emergent particle-like dispersion in NWT's voxel
framework.

In the BI / voxel-space framing this is suggestive: the gapless
acoustic + internal-symmetric branches play the role of GAUGE/photon-
like excitations, while the gapped anti-symmetric branches play the
role of MATTER (with K_7-graph mass).  Not yet identified with SM
particles -- that's a Paper 19 W4 deliverable -- but the *structure*
of the dispersion (massless gauge + massive matter) drops out of the
voxel picture for free.

Output -> analysis/output/W3_2b_k7_internal/
  spectrum_external.png   : acoustic sector (from step 0)
  spectrum_internal.png   : K_7 internal sector with symmetric gluing
  combined_spectrum.png   : both sectors overlaid
  k7_eigenvalues.png      : isolated K_7 Laplacian eigenvalue check
  summary.txt             : counts, gaps, structural identifications
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

OUT = Path(__file__).resolve().parents[0] / "output" / "W3_2b_k7_internal"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Topology + external sector (re-used from step 0)
# ---------------------------------------------------------------------------

def build_tissue(n: int, scale: float = 1.0):
    pos = fcc_grid_positions(n, n, n, scale)
    topo = build_rhombic_topology(pos, scale)
    state = init_regular_3d(topo, scale)
    rest = np.asarray(state.vertices, dtype=np.float64)
    return topo, rest


def edge_pairs_from_topology(topo) -> np.ndarray:
    cfi = np.asarray(topo.cell_face_indices)
    pairs = set()
    for ci in range(topo.n_cells):
        for fi in range(12):
            f = cfi[ci, fi]
            for k in range(4):
                a, b = int(f[k]), int(f[(k + 1) % 4])
                if a == b:
                    continue
                pairs.add((min(a, b), max(a, b)))
    return np.array(sorted(pairs), dtype=np.int64)


def external_hessian(topo, K_ext: float = 1.0) -> np.ndarray:
    """Pair-spring Hessian on polyhedral edges:  K_ext * L_v (x) I_3."""
    pairs = edge_pairs_from_topology(topo)
    L_v = np.zeros((topo.n_vertices, topo.n_vertices), dtype=np.float64)
    for i, j in pairs:
        L_v[i, j] -= 1.0
        L_v[j, i] -= 1.0
        L_v[i, i] += 1.0
        L_v[j, j] += 1.0
    return K_ext * np.kron(L_v, np.eye(3, dtype=np.float64))


# ---------------------------------------------------------------------------
# K_7 internal sector with symmetric gluing
# ---------------------------------------------------------------------------

def k7_laplacian() -> np.ndarray:
    """Graph Laplacian of K_7 (complete graph on 7 vertices).

    L_K7 = (degree) I - A = 6 I - (J - I) = 7 I - J,
    eigenvalues {0 (x1), 7 (x6)}.
    """
    return 7.0 * np.eye(7) - np.ones((7, 7))


def internal_hessian(topo,
                     kappa_int: float = 1.0,
                     kappa_glue: float = 1.0,
                     kappa_mass: float = 0.0) -> np.ndarray:
    """Hessian for the qubit-amplitude sector.

    Sized (7 * n_cells, 7 * n_cells).  Block structure:
      diagonal blocks  : kappa_int * L_K7 + (degree(c) * kappa_glue) * I_7
                         (degree(c) = number of face-neighbours of cell c
                          under the current tissue's adjacency)
      off-diag blocks  : -kappa_glue * I_7  for each cell pair (c1, c2)
                         in topo.interface_pairs.
      mass             : + kappa_mass * I  globally (optional).

    The diagonal-block I_7 piece comes from the symmetric gluing: each
    of the 7 qubits picks up one unit of kappa_glue per shared face.
    """
    n_c = topo.n_cells
    n = 7 * n_c
    H = np.zeros((n, n), dtype=np.float64)

    L_K7 = k7_laplacian()
    interface_pairs = np.asarray(topo.interface_pairs, dtype=np.int64)

    # K_7 internal coupling at each cell.
    for c in range(n_c):
        H[7 * c:7 * c + 7, 7 * c:7 * c + 7] += kappa_int * L_K7

    # Symmetric gluing across each shared face.
    for ci, cj in interface_pairs:
        i0 = 7 * int(ci)
        j0 = 7 * int(cj)
        # Diagonal contribution to both cells.
        for k in range(7):
            H[i0 + k, i0 + k] += kappa_glue
            H[j0 + k, j0 + k] += kappa_glue
            H[i0 + k, j0 + k] -= kappa_glue
            H[j0 + k, i0 + k] -= kappa_glue

    if kappa_mass != 0.0:
        H += kappa_mass * np.eye(n)

    return H


# ---------------------------------------------------------------------------
# Diagonalise + classify
# ---------------------------------------------------------------------------

def spectrum(H: np.ndarray) -> np.ndarray:
    eigs = np.linalg.eigvalsh(0.5 * (H + H.T))
    eigs[eigs < 0] = 0.0
    return np.sqrt(eigs)


def classify_internal_modes(H_int: np.ndarray, n_cells: int,
                            tol: float = 1e-7) -> dict:
    """Identify symmetric vs anti-symmetric internal modes.

    The all-ones vector in qubit space (the K_7 zero-eigenvector) lifted
    to each cell tells us which Laplacian eigenvalue contributes.  We
    project each Hessian eigenvector onto v_sym = (1/sqrt(7)) * 1_7
    PER CELL and then ask whether the projection is mostly within the
    symmetric or anti-symmetric subspace.
    """
    eigvals, eigvecs = np.linalg.eigh(0.5 * (H_int + H_int.T))
    eigvals = np.where(eigvals < 0, 0.0, eigvals)

    # Build the (7 * n_cells) symmetric-subspace projector.  Symmetric
    # at cell c = (1/sqrt(7)) * 1_7 placed at indices [7c, 7c+7].
    n = 7 * n_cells
    P_sym = np.zeros((n, n_cells), dtype=np.float64)
    for c in range(n_cells):
        P_sym[7 * c:7 * c + 7, c] = 1.0 / np.sqrt(7.0)
    # Projector onto symmetric subspace = P_sym P_sym^T.
    Pi_sym = P_sym @ P_sym.T

    # For each eigenvector, compute its symmetric-subspace overlap.
    sym_weight = np.einsum("ij,jk,ik->i", eigvecs.T, Pi_sym, eigvecs.T)

    is_sym = sym_weight > 0.5
    omega = np.sqrt(eigvals)

    return {
        "omega": omega,
        "sym_weight": sym_weight,
        "is_sym": is_sym,
        "n_sym": int(is_sym.sum()),
        "n_anti": int((~is_sym).sum()),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_one(n: int, K_ext: float, kappa_int: float, kappa_glue: float,
            kappa_mass: float = 0.0) -> dict:
    topo, _ = build_tissue(n=n)

    H_ext = external_hessian(topo, K_ext=K_ext)
    H_int = internal_hessian(topo,
                             kappa_int=kappa_int,
                             kappa_glue=kappa_glue,
                             kappa_mass=kappa_mass)

    omega_ext = spectrum(H_ext)
    cls = classify_internal_modes(H_int, topo.n_cells)
    omega_int = cls["omega"]

    # Predicted optical (anti-symmetric) gap at k = 0:
    #   omega_anti(k=0) = sqrt(7 kappa_int + kappa_mass)
    gap_predicted = float(np.sqrt(7.0 * kappa_int + kappa_mass))

    # Empirical optical gap = lowest non-zero anti-symmetric omega.
    omega_anti = omega_int[~cls["is_sym"]]
    omega_anti.sort()
    gap_empirical = float(omega_anti[0]) if omega_anti.size else float("nan")

    n_zero_ext = int(np.sum(omega_ext < 1e-7))
    n_zero_int = int(np.sum(omega_int < 1e-7))

    return {
        "n": n,
        "n_cells": topo.n_cells,
        "n_verts": topo.n_vertices,
        "n_dof_ext": H_ext.shape[0],
        "n_dof_int": H_int.shape[0],
        "n_zero_ext": n_zero_ext,
        "n_zero_int": n_zero_int,
        "omega_ext": omega_ext,
        "omega_int": omega_int,
        "is_sym_int": cls["is_sym"],
        "n_sym_int": cls["n_sym"],
        "n_anti_int": cls["n_anti"],
        "gap_predicted": gap_predicted,
        "gap_empirical": gap_empirical,
        "gap_err_pct": float(100.0 * (gap_empirical - gap_predicted)
                             / gap_predicted),
    }


def main():
    K_ext = 1.0
    kappa_int = 1.0
    kappa_glue = 0.5
    kappa_mass = 0.0

    print(f"NWT W3.2 step 1   K_ext = {K_ext}, kappa_int = {kappa_int}, "
          f"kappa_glue = {kappa_glue}, kappa_mass = {kappa_mass}\n")

    grid_sizes = [3, 4, 5]
    results = []
    for n in grid_sizes:
        print(f"  Building n = {n} ...", flush=True)
        r = run_one(n, K_ext=K_ext, kappa_int=kappa_int,
                    kappa_glue=kappa_glue, kappa_mass=kappa_mass)
        results.append(r)
        print(f"    n_cells = {r['n_cells']}, n_verts = {r['n_verts']}")
        print(f"    external DOFs = {r['n_dof_ext']},  zero modes = "
              f"{r['n_zero_ext']}  (expect 3)")
        print(f"    internal DOFs = {r['n_dof_int']},  zero modes = "
              f"{r['n_zero_int']}  (expect 1: global qubit shift)")
        print(f"    internal symmetric branches    : "
              f"{r['n_sym_int']:4d}  (expect 1 * n_cells = "
              f"{r['n_cells']})")
        print(f"    internal anti-symmetric branches: "
              f"{r['n_anti_int']:4d}  (expect 6 * n_cells = "
              f"{6 * r['n_cells']})")
        print(f"    optical gap   predicted = sqrt(7 kappa_int + "
              f"kappa_mass) = {r['gap_predicted']:.6f}")
        print(f"    optical gap   empirical = {r['gap_empirical']:.6f}  "
              f"(err = {r['gap_err_pct']:+.4f} %)")
        print()

    # Use largest grid for plots
    rL = results[-1]

    # ---------- Plot 1: external sector ----------
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(np.arange(len(rL["omega_ext"])), np.sort(rL["omega_ext"]),
            "b-", ms=3, lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("mode rank")
    ax.set_ylabel(r"$\omega$")
    ax.set_title(f"External (acoustic) sector, n = {rL['n']}: "
                 f"{rL['n_zero_ext']} zero modes")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "spectrum_external.png", dpi=130)
    plt.close(fig)

    # ---------- Plot 2: internal sector classified ----------
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    omega_int_sorted = np.sort(rL["omega_int"])
    is_sym = rL["is_sym_int"]
    rank = np.arange(len(rL["omega_int"]))
    # Need to sort labels by frequency for plotting
    order = np.argsort(rL["omega_int"])
    is_sym_sorted = is_sym[order]
    ax.scatter(rank[is_sym_sorted], omega_int_sorted[is_sym_sorted],
               s=12, c="tab:blue", label="symmetric (gapless)")
    ax.scatter(rank[~is_sym_sorted], omega_int_sorted[~is_sym_sorted],
               s=12, c="tab:red", label="anti-symmetric (K_7-gapped)")
    ax.axhline(rL["gap_predicted"], color="k", ls="--", lw=1,
               label=f"predicted gap = sqrt(7 kappa_int) "
                     f"= {rL['gap_predicted']:.3f}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("mode rank")
    ax.set_ylabel(r"$\omega$")
    ax.set_title(f"Internal (K_7) sector, n = {rL['n']}: "
                 f"{rL['n_sym_int']} sym + {rL['n_anti_int']} anti")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "spectrum_internal.png", dpi=130)
    plt.close(fig)

    # ---------- Plot 3: combined spectrum ----------
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    omega_ext_sorted = np.sort(rL["omega_ext"])
    ax.plot(np.arange(len(omega_ext_sorted)), omega_ext_sorted,
            "g.-", ms=3, lw=1, alpha=0.6, label="external (acoustic)")
    omega_int_sym = omega_int_sorted[is_sym_sorted]
    omega_int_anti = omega_int_sorted[~is_sym_sorted]
    ax.plot(np.arange(len(omega_int_sym)), omega_int_sym,
            "b.-", ms=3, lw=1, alpha=0.7, label="internal symmetric")
    ax.plot(np.arange(len(omega_int_anti)), omega_int_anti,
            "r.-", ms=3, lw=1, alpha=0.7, label="internal anti-symmetric")
    ax.axhline(rL["gap_predicted"], color="k", ls="--", lw=1,
               label=f"K_7 gap = sqrt(7) "
                     f"$\\sqrt{{\\kappa_{{int}}}}$"
                     f" = {rL['gap_predicted']:.3f}")
    ax.set_xlabel("mode rank within sector")
    ax.set_ylabel(r"$\omega$")
    ax.set_title(f"NWT W3.2 step 1: full phonon spectrum, n = {rL['n']}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "combined_spectrum.png", dpi=130)
    plt.close(fig)

    # ---------- Plot 4: K_7 Laplacian eigenvalues (independent check) ----------
    L_K7 = k7_laplacian()
    eigs_K7 = np.linalg.eigvalsh(L_K7)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(np.arange(len(eigs_K7)), eigs_K7, "ko-", ms=8)
    ax.axhline(7, color="r", ls="--", alpha=0.6, label="$\\lambda = 7$ (mult 6)")
    ax.axhline(0, color="b", ls="--", alpha=0.6, label="$\\lambda = 0$ (mult 1)")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel(r"$\lambda(L_{K_7})$")
    ax.set_title(r"K_7 graph Laplacian spectrum")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "k7_eigenvalues.png", dpi=130)
    plt.close(fig)

    # ---------- Summary ----------
    summary = ["Paper 19 -- W3.2 step 1  K_7 internal qubit DOFs on FCC voxels",
               "=" * 70,
               f"K_ext = {K_ext},  kappa_int = {kappa_int},  "
               f"kappa_glue = {kappa_glue},  kappa_mass = {kappa_mass}",
               "",
               "K_7 graph Laplacian eigenvalues:",
               f"  numerical: {eigs_K7.tolist()}",
               f"  predicted: [0] + [7] * 6",
               "",
               f"{'n':>3}  {'cells':>6}  {'verts':>6}  "
               f"{'ext DOF':>8}  {'ext zero':>9}  "
               f"{'int DOF':>8}  {'int zero':>9}  "
               f"{'n_sym':>6}  {'n_anti':>7}  "
               f"{'gap_pred':>9}  {'gap_emp':>9}  {'err %':>9}",
               "-" * 130]
    for r in results:
        summary.append(
            f"{r['n']:>3}  {r['n_cells']:>6}  {r['n_verts']:>6}  "
            f"{r['n_dof_ext']:>8}  {r['n_zero_ext']:>9}  "
            f"{r['n_dof_int']:>8}  {r['n_zero_int']:>9}  "
            f"{r['n_sym_int']:>6}  {r['n_anti_int']:>7}  "
            f"{r['gap_predicted']:>9.4f}  {r['gap_empirical']:>9.4f}  "
            f"{r['gap_err_pct']:>+9.4f}")

    summary += [
        "",
        "STRUCTURAL IDENTIFICATION:",
        "  K_7 graph Laplacian spectrum -> internal phonon spectrum.",
        "  ",
        "  optical-branch gap m^2 = 7 kappa_int = lambda_max(L_K7)",
        "  ",
        "  i.e. the SQUARED MASS of the 6 anti-symmetric internal branches",
        "  is *exactly* the largest K_7 Laplacian eigenvalue, in units of",
        "  the internal coupling kappa_int.  For kappa_mass = 0, the empirical",
        "  gap matches sqrt(7) to numerical precision.",
        "",
        "INTERPRETATION (BI / voxel-space framing):",
        "  - 3 acoustic branches (gapless, linear at low k):",
        "      bulk translation modes; emergent c = sqrt(K_ext) * a.",
        "  - 1 internal-symmetric branch (gapless, linear at low k):",
        "      qubit-space 'global mode'; analogous to a Goldstone /",
        "      gauge degree of freedom that propagates between voxels.",
        "  - 6 internal-anti-symmetric branches (gapped, parabolic at",
        "    low k):",
        "      mass^2 = lambda_max(L_K7) = 7 kappa_int.  At low k these",
        "      have Klein-Gordon dispersion E^2 = m^2 + p^2 -- the",
        "      generic relativistic-particle dispersion drops out of",
        "      the voxel-space lattice for free.  Candidate analogues",
        "      of MATTER fields (Paper 19 W4 will test species",
        "      identification).",
        "",
        "PHYSICAL TAKEAWAY:",
        "  The K_7 graph spectrum literally appears in the phonon",
        "  spectrum.  The Laplacian eigenvalue 7 (= |V(K_7)|, the number",
        "  of qubits) sets the squared mass of the 6 internal optical",
        "  branches.  This is the first concrete bridge between graph-",
        "  state combinatorics and emergent particle-like dispersion in",
        "  NWT's BI framework.",
        "",
        "NEXT STEPS:",
        "  W3.2 step 2: extract Bloch dispersion omega(k) along high-",
        "    symmetry directions in the FCC Brillouin zone.  Fit c_L,",
        "    c_T separately for the acoustic sector; verify the K_7",
        "    gap and parabolic dispersion of the optical branches.",
        "  W3.3: scan gluing rules.  The symmetric rule used here is",
        "    the simplest of many.  Alternatives: per-face qubit-pair",
        "    selection (e.g. matching the rhombic face's 4 vertices to",
        "    a 4-element subset of K_7's vertices); K_7-symmetry-",
        "    equivariant rules using PSL(2,7) or S_7 actions.  Different",
        "    gluings give different optical-branch structure -- one of",
        "    these may match the SM particle content under coarse-graining.",
        "  W3.4: project to 3+1 spacetime.  The internal-symmetric and",
        "    anti-symmetric branches will shed (4 - 3) = 1 polarization",
        "    on coarse-graining; need to identify which.",
    ]

    summary_text = "\n".join(summary)
    (OUT / "summary.txt").write_text(summary_text)
    print(summary_text)
    print()
    print(f"Wrote {OUT/'spectrum_external.png'}")
    print(f"Wrote {OUT/'spectrum_internal.png'}")
    print(f"Wrote {OUT/'combined_spectrum.png'}")
    print(f"Wrote {OUT/'k7_eigenvalues.png'}")
    print(f"Wrote {OUT/'summary.txt'}")


if __name__ == "__main__":
    main()
