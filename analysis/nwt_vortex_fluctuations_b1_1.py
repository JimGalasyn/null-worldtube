#!/usr/bin/env python3
"""
Stage 2 b1.1 — Gauge+scalar coupled fluctuation spectrum around BPS n=1 vortex.

Extends b1.0 by including gauge-field fluctuations and their coupling to
the Higgs modulus fluctuations.  The operator is a 2D Cartesian
discretization of the 2×2 block

    H_block  =  [ -Δ + f²(r)                   2(1-a(r)) f(r)/r       ]
                [ 2(1-a(r)) f(r)/r    -Δ + a²(r)/r² + ½(3f²(r)-1) ]

acting on (a_1, φ_2)^T  (or equivalently (a_2, φ_1)^T up to sign of the
coupling, with identical spectrum).  This is a subblock of the full
Alonso-Izquierdo et al. 4×4 operator H+; we are dropping the scalar-
phase coupling H_{34} = −2 V_k ∂_k (to be added in b1.2).

Physics to look for
-------------------
(a) **Translational zero modes**: the (a_1, φ_2) and (a_2, φ_1) blocks
    should each contain ONE exact zero mode (x-translation and
    y-translation of the vortex respectively).  We expect to see an
    eigenvalue very close to zero, with eigenfunction looking like
    the x- (or y-) derivative of the background.
(b) **Gauge-field bound states**: the coupled system may bind gauge
    fluctuations below m_A² = 1 (BPS gauge mass = m_H).
(c) **Shift of the Higgs bound state** from the b1.0 value of 0.650
    due to coupling with gauge fluctuations.
"""

from __future__ import annotations

import sys
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import bmat, diags, eye as speye
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).parent))
from nwt_vortex_gravity_flat import solve_bps_profile


# =========================================================================
# BPS background on a 2D Cartesian grid.
# =========================================================================

def bps_background_2d(N: int, L: float):
    """Return (x, y, r, f_grid, a_grid, coupling, V2_term).

    Grid is [-L, L]² with N × N cells (cell-centered).
    f_grid, a_grid are from the BPS first-order solver, interpolated.
    coupling = 2(1-a)f/r    (the gauge-scalar off-diagonal block).
    V2_term  = a²/r²        (the scalar-diagonal gauge-squared term).
    """
    dx = 2.0 * L / N
    x = np.linspace(-L + dx / 2, L - dx / 2, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt(X * X + Y * Y)

    rho, f, a, _, _ = solve_bps_profile(
        rho_min=1e-3, rho_max=max(3.0 * L, 40.0),
        N=3000, dense=12000)
    f_r = np.interp(r, rho, f, left=0.0, right=1.0)
    a_r = np.interp(r, rho, a, left=0.0, right=1.0)

    # Regularize r=0 artefacts with r_safe = max(r, dx/2)
    r_safe = np.maximum(r, dx * 0.5)

    coupling = 2.0 * (1.0 - a_r) * f_r / r_safe        # 2(1-a)f/r
    V2_term = (a_r / r_safe) ** 2                      # a²/r²  = V_μV_μ
    return x, X, Y, r, dx, f_r, a_r, coupling, V2_term


# =========================================================================
# Sparse 2D Laplacian (5-point stencil, Dirichlet).
# =========================================================================

def laplacian_2d(N: int, dx: float):
    """Build -Δ on an N×N grid with Dirichlet BC, flattened index i*N+j."""
    # 1D -d²/dx² with Dirichlet
    e = np.ones(N)
    D1 = diags([-e[1:], 2 * e, -e[1:]], offsets=[-1, 0, 1],
               shape=(N, N), format="csr") / dx ** 2
    I = speye(N, format="csr")
    # -Δ = D1⊗I + I⊗D1
    from scipy.sparse import kron
    L2 = kron(D1, I, format="csr") + kron(I, D1, format="csr")
    return L2


# =========================================================================
# Assemble the 2×2 block operator.
# =========================================================================

def build_block_operator(N, dx, f_grid, a_grid, coupling, V2_term,
                         mode: str):
    """Assemble the block operator.  `mode`:
        "vortex"  — full BPS background.
        "vacuum"  — f=1, a=0 (pure Higgs broken vacuum, no gauge field).
    Returns a (2N², 2N²) sparse symmetric matrix.
    """
    L2 = laplacian_2d(N, dx)
    N2 = N * N
    if mode == "vortex":
        f2_flat = (f_grid ** 2).ravel()
        V2_flat = V2_term.ravel()
        coup_flat = coupling.ravel()
        scalar_mass_flat = 0.5 * (3.0 * f_grid ** 2 - 1.0).ravel()
    elif mode == "vacuum":
        f2_flat = np.ones(N2)
        V2_flat = np.zeros(N2)
        coup_flat = np.zeros(N2)
        scalar_mass_flat = np.ones(N2)   # ½(3·1 - 1) = 1
    else:
        raise ValueError(mode)

    H_aa = L2 + diags(f2_flat)                           # gauge block
    H_pp = L2 + diags(V2_flat + scalar_mass_flat)        # scalar block
    H_ap = diags(coup_flat)                              # coupling

    H = bmat([[H_aa, H_ap],
              [H_ap, H_pp]], format="csr")
    return H


# =========================================================================
# Main.
# =========================================================================

def main():
    print("=" * 72)
    print(" b1.1  —  Coupled (gauge, Higgs-modulus) fluctuations around BPS")
    print("=" * 72)

    N = 96
    L = 10.0
    x, X, Y, r, dx, f_grid, a_grid, coupling, V2_term = \
        bps_background_2d(N, L)
    print(f"\n[1] Grid: {N}² on [{-L:.1f},{L:.1f}]², dx={dx:.4f} "
          f"({1.0/dx:.1f} pts/ξ)")
    print(f"    BPS: f(core≈0)={f_grid[N//2, N//2]:.3e}  "
          f"a(core≈0)={a_grid[N//2, N//2]:.3e}  "
          f"f(corner)={f_grid[0,0]:.4f}  a(corner)={a_grid[0,0]:.4f}")

    t0 = time.time()
    H_vortex = build_block_operator(N, dx, f_grid, a_grid,
                                    coupling, V2_term, mode="vortex")
    H_vacuum = build_block_operator(N, dx, f_grid, a_grid,
                                    coupling, V2_term, mode="vacuum")
    print(f"\n[2] Block operators built "
          f"(shape {H_vortex.shape}, nnz={H_vortex.nnz}), "
          f"{time.time()-t0:.2f}s")

    # -- Diagonalize for lowest k_eigen eigenvalues
    k_eigen = 60
    t0 = time.time()
    ω2_vortex = eigsh(H_vortex, k=k_eigen, which="SA",
                      return_eigenvectors=False)
    ω2_vortex.sort()
    print(f"\n[3] Vortex spectrum (lowest {k_eigen}): "
          f"{time.time()-t0:.2f}s")
    t0 = time.time()
    ω2_vacuum = eigsh(H_vacuum, k=k_eigen, which="SA",
                      return_eigenvectors=False)
    ω2_vacuum.sort()
    print(f"    Vacuum spectrum (lowest {k_eigen}): "
          f"{time.time()-t0:.2f}s")

    # =====================================================================
    # [4] Look for zero modes (expect ONE x-translational zero mode
    #     in this block; the other translation lives in the (a_2,φ_1) block)
    # =====================================================================
    print(f"\n[4] Lowest 10 vortex eigenvalues (ω²):")
    for i, v in enumerate(ω2_vortex[:10]):
        tag = "  ← ZERO MODE?" if abs(v) < 0.05 else ""
        tag_bound = "  (bound, < m²=1)" if 0 < v < 1.0 else ""
        print(f"    n={i:>2d}:  ω² = {v:+12.6f}{tag}{tag_bound}")
    print(f"\n    Vacuum lowest 10 (for comparison):")
    for i, v in enumerate(ω2_vacuum[:10]):
        print(f"    n={i:>2d}:  ω² = {v:+12.6f}")

    # =====================================================================
    # [5] Casimir sum (truncated at k_eigen; raw, unregularized)
    # =====================================================================
    ω_v = np.sqrt(np.maximum(ω2_vortex, 0.0))
    ω_0 = np.sqrt(np.maximum(ω2_vacuum, 0.0))
    diff = ω_v - ω_0
    S_block = 0.5 * diff.sum()
    print(f"\n[5] Casimir sum for this block, k_cut={k_eigen}:")
    print(f"    ½ Σ (ω_vortex − ω_vacuum) = {S_block:+.6f}")
    print(f"    Doubling for (a_2, φ_1) block: 2 × {S_block:+.4f} "
          f"= {2*S_block:+.4f}")
    print(f"    Compared to b1.0 scalar-only partial sum ≈ −1.14:")
    print(f"       Full gauge+scalar (without V·∂ piece) ≈ {2*S_block:+.4f}")

    # =====================================================================
    # [6] Plots
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))

    ax = axes[0]
    im = ax.pcolormesh(X, Y, coupling, shading="auto", cmap="viridis")
    ax.set_aspect("equal"); ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.set_title(r"Gauge-scalar coupling $2(1-a)f/r$")
    ax.set_xlabel("x / ξ"); ax.set_ylabel("y / ξ")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    ax.plot(np.arange(len(ω2_vortex)), ω2_vortex, "o-",
            label="vortex", ms=4)
    ax.plot(np.arange(len(ω2_vacuum)), ω2_vacuum, "x-",
            label="vacuum", ms=5, alpha=0.6)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.axhline(1.0, color="r", ls="--", lw=1.0, label=r"$m^2=1$")
    ax.set_xlabel("n (eigenvalue index)")
    ax.set_ylabel(r"$\omega^2$")
    ax.set_title("Coupled spectrum (low-n)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 3.0)
    ax.set_xlim(0, 40)

    ax = axes[2]
    # Show the per-mode shift ω_vortex − ω_vacuum
    ax.plot(np.arange(len(diff)), diff, "o-", ms=4)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("n"); ax.set_ylabel(r"$\omega_{\rm vortex} - \omega_{\rm vacuum}$")
    ax.set_title("Per-mode Casimir shift (raw, UV-unregulated)")
    ax.grid(alpha=0.3)

    fig.suptitle("Paper 15 Stage 2 b1.1 — coupled gauge+scalar, "
                 "BPS n=1 (2×2 block, V·∂ piece deferred)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(__file__).parent / "nwt_vortex_fluctuations_b1_1.png"
    fig.savefig(out, dpi=140)
    print(f"\n[6] Plot: {out}")


if __name__ == "__main__":
    main()
