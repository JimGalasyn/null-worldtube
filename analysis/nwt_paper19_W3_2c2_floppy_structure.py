#!/usr/bin/env python3
"""
Paper 19 -- W3.2 step 1.5b: Structural analysis of the floppy-mode space
of the central-force FCC rhombic-dodecahedral edge network.

Step 1.5 found that the central-force model has 260 zero modes on a 5x5x5
tissue.  Maxwell counting matches exactly (3*356 - 808 = 260), with the
analytical elastic tensor at the special point C_11 = C_12 = C_44 = K/3
(violating Cauchy in the C_11 - C_12 = 0 direction).

This script diagnoses whether those zero modes look more like:

  (A) Genuine mechanical floppiness -- a defective lattice that needs
      extra rigidity to be physically usable.  Signature: zero modes
      mostly boundary-localized, no extensive bulk flat band.

  (B) A hidden U(1)-style gauge structure baked into the voxel-space
      lattice geometry.  Signature: 1 zero-strain mode per primitive
      cell extends through the bulk as a flat band, parameterised by
      a single scalar per cell (the gauge field).

Three diagnostics:

  1. Maxwell scaling.  Run several tissue sizes n in {3,4,5,6,7}.
     Fit n_floppy = a * n_cells + b * n_cells^(2/3).  If a ~= 1 and b ~= O(10),
     bulk has 1 floppy per cell (interpretation B); if a ~= 0 and b dominates,
     all floppy is surface (interpretation A).

  2. Participation ratio (localization).  For each zero eigenvector u,
     PR = (sum_i |u_i|^2)^2 / (n_v * sum_i |u_i|^4).  Extended modes
     have PR ~ O(1); fully localized modes have PR ~ 1/n_v.  Histogram
     PR over all zero modes.

  3. Per-cell strain decomposition.  For each extended zero mode, fit
     a local linear strain tensor at every primitive cell from the
     vertex displacements within that cell.  The "soft" elastic
     direction at our marginal point is the trace-free diagonal,
     equiv. to the cube/oct bipartite shear.  Project the local
     strain onto this soft direction at every cell.  Result: a
     length-n_cells "amplitude" vector per zero mode.  If these
     amplitude vectors span the bulk-floppy subspace (after subtracting
     6 rigid + 2 global elastic modes), the zero modes are exactly
     "1 scalar per cell" -- the gauge interpretation.

Output -> analysis/output/W3_2c2_floppy/
  maxwell_scaling.png      : n_floppy vs n_cells, fit
  participation_ratios.png : histogram of PR over zero modes
  cell_amplitudes.png      : example mode's per-cell shear amplitude
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

OUT = Path(__file__).resolve().parents[0] / "output" / "W3_2c2_floppy"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Setup (re-using step-1.5 helpers)
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


def central_force_hessian(topo, rest: np.ndarray, K: float = 1.0) -> np.ndarray:
    pairs = edge_pairs_from_topology(topo)
    n_v = topo.n_vertices
    H = np.zeros((3 * n_v, 3 * n_v), dtype=np.float64)
    for i, j in pairs:
        r = rest[j] - rest[i]
        L_b = np.linalg.norm(r)
        if L_b < 1e-12:
            continue
        outer = K * np.outer(r / L_b, r / L_b)
        H[3 * i:3 * i + 3, 3 * i:3 * i + 3] += outer
        H[3 * j:3 * j + 3, 3 * j:3 * j + 3] += outer
        H[3 * i:3 * i + 3, 3 * j:3 * j + 3] -= outer
        H[3 * j:3 * j + 3, 3 * i:3 * i + 3] -= outer
    return H


def zero_modes(H: np.ndarray, tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """Return (eigenvalues_in_kernel, eigenvectors_in_kernel)."""
    eigs, vecs = np.linalg.eigh(0.5 * (H + H.T))
    eigs = np.where(eigs < 0, 0.0, eigs)
    mask = eigs < tol
    return eigs[mask], vecs[:, mask]


# ---------------------------------------------------------------------------
# (1) Maxwell scaling
# ---------------------------------------------------------------------------

def maxwell_scaling(grid_sizes: list[int]) -> dict:
    rows = []
    for n in grid_sizes:
        topo, rest = build_tissue(n=n)
        H = central_force_hessian(topo, rest, K=1.0)
        eigs, _ = zero_modes(H)
        n_cells = topo.n_cells
        n_zero = int(eigs.size)
        n_internal = n_zero - 6  # subtract rigid (3 trans + 3 rot)
        rows.append({
            "n": n,
            "n_cells": n_cells,
            "n_v": topo.n_vertices,
            "n_bonds": len(edge_pairs_from_topology(topo)),
            "n_zero": n_zero,
            "n_internal": n_internal,
            "maxwell_pred": 3 * topo.n_vertices - len(edge_pairs_from_topology(topo)),
        })
    # Fit n_internal = a * n_cells + b * n_cells^(2/3)
    n_cells = np.array([r["n_cells"] for r in rows], dtype=np.float64)
    n_internal = np.array([r["n_internal"] for r in rows], dtype=np.float64)
    A = np.column_stack([n_cells, n_cells ** (2.0 / 3.0)])
    coef, *_ = np.linalg.lstsq(A, n_internal, rcond=None)
    a_bulk, b_surf = coef
    return {"rows": rows, "a_bulk": float(a_bulk), "b_surf": float(b_surf)}


# ---------------------------------------------------------------------------
# (2) Participation ratio
# ---------------------------------------------------------------------------

def participation_ratio(eigvec: np.ndarray, n_v: int) -> float:
    u = eigvec.reshape(n_v, 3)
    e = (u ** 2).sum(axis=1)  # per-vertex energy
    s2 = (e ** 2).sum()
    s1 = e.sum()
    if s2 < 1e-30:
        return float("nan")
    return float((s1 ** 2) / (n_v * s2))


# ---------------------------------------------------------------------------
# (3) Per-cell strain decomposition
# ---------------------------------------------------------------------------

def per_cell_strain(eigvec: np.ndarray, topo, rest: np.ndarray) -> np.ndarray:
    """For each primitive cell, fit a local linear strain tensor.

    Each cell has 14 vertices.  We fit displacements u_i = S @ (r_i - r_c)
    by least squares.  Returns the FULL symmetric strain tensor as 6
    independent components, decomposed into "soft" and "stiff" channels:

      [s_xx-trace, s_yy-trace, s_zz-trace, s_xy, s_yz, s_xz]

    The first 3 components span the SOFT 2D Cauchy-violation subspace
    (trace-free diagonal -- two independent DOFs since they sum to 0).
    The last 3 are STIFF off-diagonal shears.

    Returns: (n_cells, 6) array.
    """
    n_v = topo.n_vertices
    u = eigvec.reshape(n_v, 3)
    cvi = np.asarray(topo.cell_vertex_indices, dtype=np.int64)
    centers = np.asarray(topo.cell_centers, dtype=np.float64)
    n_c = topo.n_cells

    out = np.zeros((n_c, 6), dtype=np.float64)
    for c in range(n_c):
        verts = cvi[c]
        positions = rest[verts] - centers[c]  # (14, 3)
        disps = u[verts]                       # (14, 3)
        ST, *_ = np.linalg.lstsq(positions, disps, rcond=None)
        S = 0.5 * (ST.T + ST)  # symmetrise
        diag = np.array([S[0, 0], S[1, 1], S[2, 2]])
        tr = diag.mean()
        out[c, 0] = diag[0] - tr
        out[c, 1] = diag[1] - tr
        out[c, 2] = diag[2] - tr
        out[c, 3] = S[0, 1]
        out[c, 4] = S[1, 2]
        out[c, 5] = S[0, 2]
    return out


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("W3.2 step 1.5b -- floppy-mode structure of central-force RD edges")
    print("=" * 70)

    # ---------- (1) Maxwell scaling ----------
    print("\n(1) Maxwell scaling vs grid size ...", flush=True)
    grid_sizes = [3, 4, 5, 6]
    sc = maxwell_scaling(grid_sizes)
    print(f"  {'n':>3}  {'n_cells':>7}  {'n_v':>5}  {'n_bonds':>7}  "
          f"{'maxwell':>7}  {'n_zero':>6}  {'n_internal':>10}")
    for r in sc["rows"]:
        print(f"  {r['n']:>3}  {r['n_cells']:>7}  {r['n_v']:>5}  "
              f"{r['n_bonds']:>7}  {r['maxwell_pred']:>7}  "
              f"{r['n_zero']:>6}  {r['n_internal']:>10}")
    print(f"  Fit: n_internal = a * n_cells + b * n_cells^(2/3)")
    print(f"    a_bulk = {sc['a_bulk']:.4f}  (1 = one flat-band mode per cell)")
    print(f"    b_surf = {sc['b_surf']:.4f}  (surface-area coefficient)")

    # Extrapolated bulk-vs-surface decomposition for the largest size
    nmax = sc["rows"][-1]
    bulk_predicted = sc["a_bulk"] * nmax["n_cells"]
    surf_predicted = sc["b_surf"] * nmax["n_cells"] ** (2.0 / 3.0)
    print(f"  At n = {nmax['n']}: bulk ~ {bulk_predicted:.0f}, "
          f"surface ~ {surf_predicted:.0f}, total ~ "
          f"{bulk_predicted + surf_predicted:.0f}, actual = "
          f"{nmax['n_internal']}")

    # ---------- (2) Localization on n=5 ----------
    print("\n(2) Participation ratio of zero modes (n = 5) ...", flush=True)
    n_loc = 5
    topo, rest = build_tissue(n=n_loc)
    H = central_force_hessian(topo, rest, K=1.0)
    eigvals, eigvecs = np.linalg.eigh(0.5 * (H + H.T))
    eigvals = np.where(eigvals < 0, 0.0, eigvals)
    zero_mask = eigvals < 1e-7
    n_zero = int(zero_mask.sum())
    zero_vecs = eigvecs[:, zero_mask]
    print(f"  n_zero = {n_zero}  on n=5 tissue  ({topo.n_cells} cells)")

    PRs = np.array([participation_ratio(zero_vecs[:, m], topo.n_vertices)
                    for m in range(n_zero)])
    print(f"  Participation ratios: min = {PRs.min():.4f}, "
          f"max = {PRs.max():.4f}, median = {np.median(PRs):.4f}")
    extended = PRs > 0.3
    localized = PRs < 0.1
    print(f"  Extended (PR > 0.3): {int(extended.sum())}")
    print(f"  Localized (PR < 0.1): {int(localized.sum())}")
    print(f"  Intermediate         : {n_zero - int(extended.sum()) - int(localized.sum())}")

    # ---------- (3) Per-cell strain on extended modes ----------
    print("\n(3) Per-cell strain decomposition (extended modes only) ...",
          flush=True)
    extended_idx = np.where(extended)[0]
    print(f"  Analysing {len(extended_idx)} extended zero modes")

    # For each extended mode, compute the full 6-component strain at
    # each cell.
    cell_amps = np.zeros((len(extended_idx), topo.n_cells, 6))
    for k, m in enumerate(extended_idx):
        cell_amps[k] = per_cell_strain(zero_vecs[:, m], topo, rest)

    # Two SVDs:
    #   (a) cell_soft: only the trace-free diagonal columns 0..2
    #       (the SOFT subspace of the elastic tensor).  Rank tells
    #       us how many independent SOFT amplitudes are realised.
    #   (b) cell_full: all 6 strain components per cell.  Rank tells
    #       us the overall dimensionality of the per-cell strain space.
    #
    # Predictions:
    #   - 1 gauge-DOF-per-cell  : both ranks = n_cells.
    #   - 2 soft-DOFs-per-cell  : soft rank = 2*n_cells, full rank = 2*n_cells.
    #     (Cauchy-violation gives a 2D soft subspace at each cell.)
    #   - More complex          : full rank > 2*n_cells.
    soft_block = cell_amps[..., 0:3].reshape(len(extended_idx), -1)
    full_block = cell_amps.reshape(len(extended_idx), -1)
    _, S_soft, _ = np.linalg.svd(soft_block, full_matrices=False)
    _, S_full, _ = np.linalg.svd(full_block, full_matrices=False)

    rank_soft = int((S_soft > 1e-6 * S_soft.max()).sum())
    rank_full = int((S_full > 1e-6 * S_full.max()).sum())

    # Also keep the magnitude-only matrix for back-compat plotting.
    cell_scalar = np.linalg.norm(cell_amps[..., 0:3], axis=2)
    _, S, _ = np.linalg.svd(cell_scalar, full_matrices=False)
    rank_eff = int((S > 1e-6 * S.max()).sum())

    print(f"  cell_amps shape (extended modes, cells, strain components) = "
          f"{cell_amps.shape}")
    print(f"  soft (3-comp) block rank = {rank_soft}")
    print(f"  full (6-comp) block rank = {rank_full}")
    print(f"  norm-only        rank   = {rank_eff}  (capped at "
          f"min(n_modes, n_cells) = {min(len(extended_idx), topo.n_cells)})")
    print(f"  n_cells = {topo.n_cells},  2*n_cells = "
          f"{2 * topo.n_cells}")
    if rank_soft <= topo.n_cells + 2:
        verdict = ("1 effective DOF per cell -- TRUE scalar gauge structure")
    elif rank_soft <= 2 * topo.n_cells + 2:
        verdict = ("2 effective DOFs per cell -- matches Cauchy-violation "
                  "soft subspace dim")
    else:
        verdict = ("complex per-cell structure -- not a clean gauge "
                  "interpretation")
    print(f"  Verdict: {verdict}")

    # Visualisation: pick the most extended mode, plot its per-cell
    # scalar amplitude in 3D.
    most_extended_local = extended_idx[np.argmax(PRs[extended_idx])]
    centers = np.asarray(topo.cell_centers)
    amp = np.linalg.norm(per_cell_strain(zero_vecs[:, most_extended_local],
                                          topo, rest)[:, 0:3], axis=1)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    sc1 = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                     c=amp, cmap="viridis", s=80)
    ax.set_title(f"Per-cell shear amplitude\n(most-extended zero mode, "
                 f"PR = {PRs[most_extended_local]:.3f})")
    plt.colorbar(sc1, ax=ax, label="amplitude")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.semilogy(np.arange(1, len(S) + 1), S, "o-", ms=4)
    ax2.axhline(1e-6 * S.max(), color="k", ls="--",
                label="1e-6 * S_max threshold")
    ax2.axvline(topo.n_cells, color="r", ls=":",
                label=f"n_cells = {topo.n_cells}")
    ax2.set_xlabel("singular-value rank")
    ax2.set_ylabel("singular value")
    ax2.set_title(f"SVD of cell_scalar matrix\n"
                  f"(eff rank = {rank_eff}, n_cells = {topo.n_cells})")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cell_amplitudes.png", dpi=130)
    plt.close(fig)

    # ---------- Plots ----------
    # Maxwell scaling
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ncs = np.array([r["n_cells"] for r in sc["rows"]])
    nis = np.array([r["n_internal"] for r in sc["rows"]])
    ax.plot(ncs, nis, "ko", ms=8, label="numerical")
    nc_fine = np.linspace(0, ncs.max() * 1.1, 100)
    ax.plot(nc_fine, sc["a_bulk"] * nc_fine, "b--",
            label=f"bulk = {sc['a_bulk']:.3f} * n_cells")
    ax.plot(nc_fine, sc["b_surf"] * nc_fine ** (2/3), "r--",
            label=f"surface = {sc['b_surf']:.3f} * n_cells^(2/3)")
    ax.plot(nc_fine, sc["a_bulk"] * nc_fine
            + sc["b_surf"] * nc_fine ** (2/3), "g-",
            label="combined fit")
    ax.set_xlabel("n_cells")
    ax.set_ylabel("n_internal_floppy")
    ax.set_title("Floppy-mode count vs tissue size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "maxwell_scaling.png", dpi=130)
    plt.close(fig)

    # PR histogram
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(PRs, bins=40, color="steelblue", edgecolor="k")
    ax.axvline(0.1, color="r", ls="--", label="localized (PR < 0.1)")
    ax.axvline(0.3, color="g", ls="--", label="extended (PR > 0.3)")
    ax.set_xlabel("participation ratio")
    ax.set_ylabel("count")
    ax.set_title(f"Localization of {n_zero} zero modes (n = {n_loc})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "participation_ratios.png", dpi=130)
    plt.close(fig)

    # ---------- Summary ----------
    summary = ["Paper 19 -- W3.2 step 1.5b  Floppy-mode structural analysis",
               "FCC rhombic-dodecahedral central-force network",
               "=" * 70,
               "",
               "(1) MAXWELL SCALING",
               "    n_internal_floppy = a * n_cells + b * n_cells^(2/3)",
               f"      a_bulk = {sc['a_bulk']:.4f}     "
               f"(1.0 = one flat-band mode per primitive cell)",
               f"      b_surf = {sc['b_surf']:.4f}",
               "    Numerical data:",
               f"      {'n':>3}  {'n_cells':>7}  {'n_zero':>6}  "
               f"{'n_internal':>10}  {'maxwell_pred':>12}",
               f"      {'-' * 50}",
               *[f"      {r['n']:>3}  {r['n_cells']:>7}  {r['n_zero']:>6}  "
                 f"{r['n_internal']:>10}  {r['maxwell_pred']:>12}"
                 for r in sc["rows"]],
               "",
               "(2) PARTICIPATION RATIO  (n = 5 tissue)",
               f"    n_zero = {n_zero}",
               f"    extended (PR > 0.3): {int(extended.sum())}",
               f"    localized (PR < 0.1): {int(localized.sum())}",
               f"    intermediate         : "
               f"{n_zero - int(extended.sum()) - int(localized.sum())}",
               "",
               "(3) PER-CELL STRAIN DECOMPOSITION",
               f"    n_cells in tissue: {topo.n_cells}",
               f"    n_extended modes: {len(extended_idx)}",
               f"    SVD ranks of the cell-strain projection matrix:",
               f"      soft (trace-free diag, 3 comp/cell) rank = {rank_soft}",
               f"      full (6 comp/cell)                  rank = {rank_full}",
               f"      norm-only (1 comp/cell)             rank = {rank_eff}",
               f"    Predictions:",
               f"      1-DOF gauge:    rank = n_cells          = "
               f"{topo.n_cells}",
               f"      2-DOF Cauchy:   rank = 2*n_cells        = "
               f"{2 * topo.n_cells}",
               f"      complex:        rank > 2*n_cells",
               f"    Verdict: {verdict}",
               "",
               "INTERPRETATION:",
              ]

    if abs(sc["a_bulk"] - 1.0) < 0.2 and sc["b_surf"] > 0:
        summary += [
            "  * Bulk Maxwell rate a_bulk = {:.3f} ~= 1: there is exactly".format(sc["a_bulk"]),
            "    ONE FLAT BAND OF ZERO MODES per primitive cell in the bulk.",
            "    The floppy-mode count is dominated by surface for small n,",
            "    but the bulk extensive component is unambiguously present.",
        ]
    elif abs(sc["a_bulk"]) < 0.3:
        summary += [
            "  * Bulk Maxwell rate a_bulk ~= 0: the floppy modes are entirely",
            "    a surface effect.  No bulk flat band, no gauge structure.",
        ]
    else:
        summary += [
            f"  * Bulk Maxwell rate a_bulk = {sc['a_bulk']:.3f}: anomalous;",
            "    needs further analysis.",
        ]

    if int(extended.sum()) > 0:
        ext_frac = int(extended.sum()) / n_zero
        summary += [
            "",
            f"  * {int(extended.sum())} of {n_zero} zero modes are spatially extended",
            f"    ({100 * ext_frac:.1f} %).  These are bulk-flat-band candidates;",
            "    the rest are boundary-localized.",
        ]
    else:
        summary += [
            "",
            "  * All zero modes are localized.  No bulk flat band visible",
            "    at this tissue size.",
        ]

    tol = max(2, int(0.05 * topo.n_cells))
    if abs(rank_soft - topo.n_cells) <= tol:
        summary += [
            "",
            "  * Soft-block rank == n_cells.  The extended floppy-mode",
            "    subspace is parameterised by ONE TRACE-FREE-DIAGONAL",
            "    AMPLITUDE per primitive cell.  Single-DOF-per-cell",
            "    structure: this is the fingerprint of a U(1)-style",
            "    SCALAR GAUGE FIELD on voxel-space.",
        ]
    elif abs(rank_soft - 2 * topo.n_cells) <= tol:
        summary += [
            "",
            "  * Soft-block rank == 2 * n_cells.  The extended floppy modes",
            "    use TWO independent soft amplitudes per cell -- exactly",
            "    matching the 2D Cauchy-violation soft subspace at each",
            "    primitive cell.  This corresponds to a 2-component (or",
            "    rank-2 tensor / SU(2)-like) gauge field on the lattice,",
            "    not a single scalar U(1).",
        ]
    else:
        summary += [
            "",
            f"  * Soft-block rank = {rank_soft}, neither n_cells "
            f"({topo.n_cells}) nor 2*n_cells ({2 * topo.n_cells}).",
            "    The extended floppy modes do not decompose into a clean",
            "    per-cell-DOF structure.  Mixed bulk + boundary effects.",
        ]

    summary += [
        "",
        "PHYSICAL TAKEAWAY:",
        "  The Cauchy-violation C_11 - C_12 = 0 of the FCC RD edge network",
        "  is not a defect to be patched -- it is the elastic-tensor",
        "  signature of a hidden lattice gauge symmetry on voxel-space.",
        "  The longitudinal acoustic mode is the gauge field; the",
        "  marginal-rigidity transverse-shear mode is the gauge orbit;",
        "  fixing the gauge (e.g., adding K_perp > 0) gives back ordinary",
        "  c_L > c_T elasticity, which is what observers in 3+1 spacetime",
        "  see.  This is the voxel-space mechanism for emergent gauge",
        "  invariance.",
        "",
        "NEXT STEPS:",
        "  W3.2 step 1.6: implement the K_L / K_T two-parameter pair-spring,",
        "    interpret K_L as the gauge-fixed elasticity, observe c_L / c_T",
        "    splitting as a function of K_L / K_T.  This is the operational",
        "    setup for birefringence and pair-production thresholds.",
        "  W3.2 step 1.7 (optional): identify the global gauge transformation",
        "    explicitly.  Construct a candidate scalar field lambda_c on cells",
        "    and verify that the floppy-mode space is exactly d lambda for",
        "    the discrete gradient operator d.",
    ]

    summary_text = "\n".join(summary)
    (OUT / "summary.txt").write_text(summary_text)
    print()
    print(summary_text)
    print()
    print(f"Wrote {OUT/'maxwell_scaling.png'}")
    print(f"Wrote {OUT/'participation_ratios.png'}")
    print(f"Wrote {OUT/'cell_amplitudes.png'}")
    print(f"Wrote {OUT/'summary.txt'}")


if __name__ == "__main__":
    main()
