#!/usr/bin/env python3
"""
Stage 2 b1.2 — Full 4x4 coupled fluctuation operator H+ around a BPS
n=1 Nielsen-Olesen vortex, including the V_k*d_k scalar-phase coupling.

Extends b1.1 from a 2x2 sub-block ((a_1, psi_2) with 2(1-a)f/r coupling)
to the full 4-component H+ in the (a_1, a_2, psi_1, psi_2) Cartesian
basis of Alonso-Izquierdo, Garcia Fuertes, Mayado & Mateos Guilarte
(2016) PRD 94, 125003 (arXiv:1605.09175), Eq. (7).

Explicit form at BPS (D_1 psi_1 = D_2 psi_2 = (1-a)f/r,
D_1 psi_2 = D_2 psi_1 = 0, dk V_k = 0):

                 a_1         a_2          psi_1           psi_2
    a_1  [ -L + f2          0            0           2(1-a)f/r    ]
    a_2  [    0         -L + f2     -2(1-a)f/r           0        ]
    psi_1[    0       -2(1-a)f/r   -L + M2(r)       -2*V.grad     ]
    psi_2[ 2(1-a)f/r      0         +2*V.grad       -L + M2(r)    ]

where L = Delta (scalar Laplacian), M2(r) = (3 f^2 - 1)/2 + a^2/r^2,
and V.grad = V_k d_k with V_k the background U(1) gauge field.

New in b1.2 versus b1.1:
  * Full 4-component operator (not a 2x2 sub-block).
  * The 3-4 and 4-3 entries include the V.grad coupling; the matrix
    is no longer block-diagonal in (a_1, psi_2) / (a_2, psi_1).
  * Expect 2 translational zero modes (x- and y-translation) instead
    of the 1 in b1.1.
  * Discrete V.grad is symmetrized (V_k D_k + D_k V_k)/2 so the full
    H+ is exactly symmetric as a sparse matrix -- required for eigsh.

Vacuum comparison operator H0: -Delta * I_4 + diag(1,1,1,1), matching
the paper's H0 (page with Eq. (14) region). All four components are
decoupled at the topologically trivial vacuum (f=1, a=0, V=0).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import bmat, csr_matrix, diags, eye as speye, kron
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).parent))
from nwt_vortex_gravity_flat import solve_bps_profile


# =========================================================================
# 2D Cartesian background on [-L, L]^2 with N x N cell-centered grid.
# =========================================================================

def bps_background_2d(N: int, L: float):
    """Return (x, X, Y, r, dx, f, a, V1, V2, coupling, M2).

    f(r), a(r) from the BPS first-order solver, interpolated onto the grid.
    V1, V2 are the background gauge field components V_k = a(r) * d_k theta:
        V1 = -a * y/r^2,   V2 =  a * x/r^2.
    coupling = 2 (1-a) f / r  (off-diagonal gauge-scalar entry at BPS).
    M2 = (3 f^2 - 1)/2 + a^2/r^2  (scalar diagonal mass^2).
    """
    dx = 2.0 * L / N
    x = np.linspace(-L + dx / 2, L - dx / 2, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt(X * X + Y * Y)

    rho, f_p, a_p, _, _ = solve_bps_profile(
        rho_min=1e-3, rho_max=max(3.0 * L, 40.0),
        N=3000, dense=12000,
    )
    f = np.interp(r, rho, f_p, left=0.0, right=1.0)
    a = np.interp(r, rho, a_p, left=0.0, right=1.0)

    r_safe = np.maximum(r, dx * 0.5)
    V1 = -a * Y / (r_safe ** 2)
    V2 = +a * X / (r_safe ** 2)

    coupling = 2.0 * (1.0 - a) * f / r_safe
    M2 = 0.5 * (3.0 * f ** 2 - 1.0) + (a / r_safe) ** 2

    return x, X, Y, r, dx, f, a, V1, V2, coupling, M2


# =========================================================================
# Sparse differential operators on an N x N grid with Dirichlet BC.
# Flattened index: i * N + j, with i->x, j->y.
# =========================================================================

def laplacian_2d(N: int, dx: float) -> csr_matrix:
    """-Delta with Dirichlet BC, 5-point stencil."""
    e = np.ones(N)
    D1 = diags([-e[1:], 2 * e, -e[1:]], offsets=[-1, 0, 1],
               shape=(N, N), format="csr") / dx ** 2
    I = speye(N, format="csr")
    return kron(D1, I, format="csr") + kron(I, D1, format="csr")


def grad_2d(N: int, dx: float):
    """Centered-difference d_x and d_y with Dirichlet (zero) BC.

    Both are antisymmetric as sparse matrices, which is the continuum
    property d_x^dag = -d_x used in symmetrizing V.grad below.
    """
    e = np.ones(N - 1)
    d1 = diags([-e, e], offsets=[-1, 1], shape=(N, N),
               format="csr") / (2.0 * dx)
    I = speye(N, format="csr")
    Dx = kron(d1, I, format="csr")
    Dy = kron(I, d1, format="csr")
    return Dx, Dy


def v_dot_grad_symmetrized(V1: np.ndarray, V2: np.ndarray,
                           Dx: csr_matrix, Dy: csr_matrix) -> csr_matrix:
    """Symmetrized V.grad = (V_k D_k + D_k V_k) / 2.

    As a continuum operator V_k d_k is *anti*-self-adjoint when d_k V_k = 0
    (the BPS condition for the background V). The symmetrized discrete form

        A = ( diag(V) Dk + Dk diag(V) ) / 2

    is exactly antisymmetric on the grid (A^T = -A), independent of how well
    d_k V_k vanishes numerically, so the full H+ stays exactly symmetric
    when entered as -2*A at (3,4) and +2*A at (4,3).
    """
    V1d = diags(V1.ravel(), format="csr")
    V2d = diags(V2.ravel(), format="csr")
    return 0.5 * (V1d @ Dx + Dx @ V1d + V2d @ Dy + Dy @ V2d)


# =========================================================================
# Assemble the full 4x4 H+.
# =========================================================================

def build_Hplus(N: int, dx: float, f: np.ndarray, a: np.ndarray,
                V1: np.ndarray, V2: np.ndarray,
                coupling: np.ndarray, M2: np.ndarray,
                mode: str) -> csr_matrix:
    """Assemble H+ as a sparse 4N^2 x 4N^2 matrix.

    mode = 'vortex': Alonso-Izquierdo Eq. (7) at the BPS background.
    mode = 'vacuum': H0 = -Delta * I_4 + diag(1,1,1,1). All 4 blocks
                     decoupled, matches Eq. (14) vacuum operator.
    """
    L2 = laplacian_2d(N, dx)
    Dx, Dy = grad_2d(N, dx)
    N2 = N * N
    zero = csr_matrix((N2, N2))

    if mode == "vortex":
        f2 = (f ** 2).ravel()
        M2_flat = M2.ravel()
        coup = coupling.ravel()
        Vgrad = v_dot_grad_symmetrized(V1, V2, Dx, Dy)

        H11 = L2 + diags(f2)
        H22 = L2 + diags(f2)
        H33 = L2 + diags(M2_flat)
        H44 = L2 + diags(M2_flat)

        H14 = diags(coup)          # a_1 <-> psi_2 : +2(1-a)f/r
        H23 = -diags(coup)         # a_2 <-> psi_1 : -2(1-a)f/r
        H34 = -2.0 * Vgrad         # psi_1 <-> psi_2 : -2 V.grad
        H43 = +2.0 * Vgrad         # psi_2 <-> psi_1 : +2 V.grad

        H = bmat([
            [H11, zero, zero, H14],
            [zero, H22, H23, zero],
            [zero, H23, H33, H34],
            [H14, zero, H43, H44],
        ], format="csr")
    elif mode == "vacuum":
        I4 = speye(N2, format="csr")
        H11 = L2 + I4
        H = bmat([
            [H11, zero, zero, zero],
            [zero, H11, zero, zero],
            [zero, zero, H11, zero],
            [zero, zero, zero, H11],
        ], format="csr")
    else:
        raise ValueError(mode)

    return H


def check_symmetry(H: csr_matrix, label: str) -> None:
    diff = H - H.T
    norm = np.abs(diff).max() if diff.nnz > 0 else 0.0
    print(f"    |H - H^T|_max ({label}) = {norm:.3e}")


# =========================================================================
# Main.
# =========================================================================

def main():
    print("=" * 72)
    print(" b1.2  --  Full 4x4 H+ including V.grad coupling")
    print(" Alonso-Izquierdo et al. PRD 94, 125003 (2016), Eq. (7), at BPS")
    print("=" * 72)

    N = 96
    L = 10.0

    x, X, Y, r, dx, f, a, V1, V2, coupling, M2 = bps_background_2d(N, L)
    print(f"\n[1] Grid: {N}^2 on [{-L:.1f},{L:.1f}]^2,  dx={dx:.4f}"
          f"  ({1.0/dx:.1f} pts/xi)")
    print(f"    BPS profile check:")
    print(f"      f(core)={f[N//2,N//2]:.3e}  f(edge)={f[0,0]:.4f}"
          f"  (expect 0 / ->1)")
    print(f"      a(core)={a[N//2,N//2]:.3e}  a(edge)={a[0,0]:.4f}"
          f"  (expect 0 / ->1)")
    print(f"      max|V_k|={max(np.abs(V1).max(), np.abs(V2).max()):.3f}"
          f"  max|coupling|={np.abs(coupling).max():.3f}")

    t0 = time.time()
    Hv = build_Hplus(N, dx, f, a, V1, V2, coupling, M2, mode="vortex")
    t_vort = time.time() - t0
    t0 = time.time()
    H0 = build_Hplus(N, dx, f, a, V1, V2, coupling, M2, mode="vacuum")
    t_vac = time.time() - t0
    print(f"\n[2] Operators built:")
    print(f"    vortex H+: shape {Hv.shape}, nnz={Hv.nnz}, {t_vort:.2f}s")
    print(f"    vacuum H0: shape {H0.shape}, nnz={H0.nnz}, {t_vac:.2f}s")
    check_symmetry(Hv, "vortex")
    check_symmetry(H0, "vacuum")

    # --- Diagonalize -----------------------------------------------------
    k_eigen = 80
    print(f"\n[3] Diagonalizing lowest {k_eigen} eigenvalues...")
    t0 = time.time()
    w2_v = eigsh(Hv, k=k_eigen, which="SA", return_eigenvectors=False)
    w2_v.sort()
    print(f"    vortex: {time.time()-t0:.2f}s")
    t0 = time.time()
    w2_0 = eigsh(H0, k=k_eigen, which="SA", return_eigenvectors=False)
    w2_0.sort()
    print(f"    vacuum: {time.time()-t0:.2f}s")

    # --- Zero-mode inspection -------------------------------------------
    print(f"\n[4] Lowest 12 vortex eigenvalues (expect TWO near-zero for "
          f"x- and y-translation):")
    n_near_zero = 0
    for i, v in enumerate(w2_v[:12]):
        tag = ""
        if abs(v) < 0.02:
            tag = "  <-- ZERO MODE"
            n_near_zero += 1
        elif 0.0 < v < 1.0:
            tag = "  (bound, omega^2 < m^2=1)"
        print(f"    n={i:>2d}:  omega^2 = {v:+12.6e}{tag}")
    print(f"\n    Zero modes found: {n_near_zero}  (expected: 2 for n=1 BPS)")

    print(f"\n    Lowest 12 vacuum eigenvalues (expect threshold at 1):")
    for i, v in enumerate(w2_0[:12]):
        print(f"    n={i:>2d}:  omega^2 = {v:+12.6e}")

    # --- Casimir partial sum --------------------------------------------
    # Zero modes must be subtracted: they are collective coordinates and
    # contribute via the Jacobian, not via (1/2) omega. For a first cut we
    # set omega=0 for the (bogus negative) near-zero entries.
    w2_v_clip = np.where(np.abs(w2_v) < 0.02, 0.0, w2_v)
    w2_v_clip = np.maximum(w2_v_clip, 0.0)
    w2_0_clip = np.maximum(w2_0, 0.0)
    omega_v = np.sqrt(w2_v_clip)
    omega_0 = np.sqrt(w2_0_clip)
    diff = omega_v - omega_0
    S = 0.5 * diff.sum()
    # Compare to b1.1: 2 * (-0.95) = -1.90 (full two-block sum).
    print(f"\n[5] Casimir partial sum (raw, UV-unregulated, k_cut={k_eigen}):")
    print(f"    (1/2) sum_n (omega_vortex - omega_vacuum) = {S:+.4f}")
    print(f"    b1.1 reference (2x2 block doubled):         -1.9000")
    print(f"    Difference attributable to V.grad + full coupling:"
          f"  {S - (-1.90):+.4f}")

    # --- Plots -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))

    ax = axes[0]
    im = ax.pcolormesh(X, Y, coupling, shading="auto", cmap="viridis")
    ax.set_aspect("equal"); ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.set_title(r"Gauge-scalar coupling $2(1-a)f/r$")
    ax.set_xlabel(r"$x/\xi$"); ax.set_ylabel(r"$y/\xi$")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    # V_k quiver over the coupling magnitude
    speed = np.sqrt(V1 ** 2 + V2 ** 2)
    im = ax.pcolormesh(X, Y, speed, shading="auto", cmap="magma")
    step = max(1, N // 16)
    ax.quiver(X[::step, ::step], Y[::step, ::step],
              V1[::step, ::step], V2[::step, ::step],
              color="white", alpha=0.85, scale=8)
    ax.set_aspect("equal"); ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.set_title(r"Background gauge field $V_k = a\,\partial_k\theta$")
    ax.set_xlabel(r"$x/\xi$")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    ax.plot(np.arange(len(w2_v)), w2_v, "o-", ms=4, label="vortex")
    ax.plot(np.arange(len(w2_0)), w2_0, "x-", ms=5, alpha=0.6,
            label="vacuum")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.axhline(1.0, color="r", ls="--", lw=1.0, label=r"$m^2=1$")
    ax.set_xlabel("n (eigenvalue index)")
    ax.set_ylabel(r"$\omega^2$")
    ax.set_title(f"Full 4x4 H+ spectrum, k_cut={k_eigen}")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 3.0); ax.set_xlim(0, 50)

    fig.suptitle("Paper 15 Stage 2 b1.2 -- full 4x4 H+ with V.grad, "
                 "BPS n=1", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(__file__).parent / "nwt_vortex_fluctuations_b1_2.png"
    fig.savefig(out, dpi=140)
    print(f"\n[6] Plot: {out}")


if __name__ == "__main__":
    main()
