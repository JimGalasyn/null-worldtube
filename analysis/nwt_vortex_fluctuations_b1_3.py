#!/usr/bin/env python3
"""
Stage 2 b1.3 -- Faddeev-Popov ghost operator + zero-mode subtraction.

Extends b1.2 by adding the ghost contribution to the one-loop Casimir
energy of the BPS n=1 vortex. Alonso-Izquierdo et al. (2016) Eq. (37):

    Delta E_VC  =  (hbar m / 2) [ Tr sqrt(H+) - Tr sqrt(H0)
                                   - Tr sqrt(H_G) + Tr sqrt(H0_G) ]

The FP ghost operator in background gauge at xi=1 is a scalar Schrodinger
operator (Alonso-Izquierdo Eq. (27) context):

    H_G    = -Delta + |psi|^2  =  -Delta + f^2(r)     (at BPS)
    H0_G   = -Delta + 1                               (vacuum)

New in b1.3:
  * H_G, H0_G built as scalar sparse operators (N^2 x N^2, not 4N^2).
  * Translational zero modes of H+ explicitly identified and excluded
    from the Casimir sum (they are collective coordinates).
  * Ghost subtraction with the correct sign/factor per Eq. (37).
  * Reports bosonic, ghost, and total Casimir at matched k_cut in omega.

Not yet done (deferred to b1.4):
  * Heat-kernel / Mellin transform to continue ζ(s) to s = -1/2
    (i.e., true spectral zeta-function regularization).
  * The partial sum is still UV-truncated at k_cut modes; the
    ghost-subtracted difference is LESS divergent but still truncated.

Interpretation: with the ghost subtraction, the UV-divergent Seeley-
DeWitt c_1 coefficient partially cancels between bosonic and ghost
sectors, so the truncated partial sum converges faster with k_cut.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, diags, eye as speye
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).parent))
from nwt_vortex_fluctuations_b1_2 import (
    bps_background_2d, build_Hplus, laplacian_2d, check_symmetry,
)


# =========================================================================
# Scalar FP ghost operator.
# =========================================================================

def build_Hghost(N: int, dx: float, f: np.ndarray, mode: str) -> csr_matrix:
    """H_G = -Delta + f^2(x)   (vortex)  or  -Delta + 1  (vacuum)."""
    L2 = laplacian_2d(N, dx)
    if mode == "vortex":
        mass_sq = (f ** 2).ravel()
    elif mode == "vacuum":
        mass_sq = np.ones(N * N)
    else:
        raise ValueError(mode)
    return L2 + diags(mass_sq, format="csr")


# =========================================================================
# Matched-k_cut Casimir partial sum.
#
# Uses a common eigenvalue cap omega^2_cap across all 4 spectra so that
# we're summing equivalent portions of the continuum, not equivalent
# *counts* that sample different portions of the scalar vs 4-vector
# continuum.
# =========================================================================

def partial_casimir(omega2, zero_mode_cut: float = 0.02) -> np.ndarray:
    """Return positive omega values, zero modes replaced by 0."""
    w2 = np.where(np.abs(omega2) < zero_mode_cut, 0.0, omega2)
    w2 = np.maximum(w2, 0.0)
    return np.sqrt(w2)


def matched_sum(omega_arrays: dict, cap: float) -> dict:
    """Sum omega for each spectrum, restricted to omega <= cap."""
    out = {}
    for k, w in omega_arrays.items():
        out[k] = w[w <= cap].sum()
    return out


# =========================================================================
# Main.
# =========================================================================

def main():
    print("=" * 72)
    print(" b1.3  --  H+ + ghost + zero-mode subtraction")
    print(" Alonso-Izquierdo et al. (2016), Eq. (37) at BPS")
    print("=" * 72)

    N = 96
    L = 10.0

    x, X, Y, r, dx, f, a, V1, V2, coupling, M2 = bps_background_2d(N, L)
    print(f"\n[1] Grid: {N}^2 on [{-L:.1f},{L:.1f}]^2,  dx={dx:.4f}")

    # -- Build all four operators ----------------------------------------
    t0 = time.time()
    Hp_vort = build_Hplus(N, dx, f, a, V1, V2, coupling, M2, "vortex")
    Hp_vac  = build_Hplus(N, dx, f, a, V1, V2, coupling, M2, "vacuum")
    Hg_vort = build_Hghost(N, dx, f, "vortex")
    Hg_vac  = build_Hghost(N, dx, f, "vacuum")
    print(f"\n[2] Operators built ({time.time()-t0:.2f}s):")
    print(f"    H+   shape={Hp_vort.shape}, nnz={Hp_vort.nnz}")
    print(f"    H0   shape={Hp_vac.shape}, nnz={Hp_vac.nnz}")
    print(f"    H_G  shape={Hg_vort.shape}, nnz={Hg_vort.nnz}")
    print(f"    H0_G shape={Hg_vac.shape}, nnz={Hg_vac.nnz}")
    check_symmetry(Hp_vort, "H+")
    check_symmetry(Hg_vort, "H_G")

    # -- Diagonalize -----------------------------------------------------
    k_H = 120      # bosonic cap (4N^2 space)
    k_G = 30       # ghost cap (N^2 space, same density of states up to
                   #            factor 4, so k_G ~ k_H/4 is fair)
    print(f"\n[3] Diagonalizing: k_H={k_H} (H+/H0), k_G={k_G} (H_G/H0_G)...")
    t0 = time.time()
    w2_Hp_v = np.sort(eigsh(Hp_vort, k=k_H, which="SA",
                             return_eigenvectors=False))
    print(f"    H+  vortex : {time.time()-t0:.2f}s")
    t0 = time.time()
    w2_Hp_0 = np.sort(eigsh(Hp_vac, k=k_H, which="SA",
                             return_eigenvectors=False))
    print(f"    H0  vacuum : {time.time()-t0:.2f}s")
    t0 = time.time()
    w2_Hg_v = np.sort(eigsh(Hg_vort, k=k_G, which="SA",
                             return_eigenvectors=False))
    print(f"    H_G vortex : {time.time()-t0:.2f}s")
    t0 = time.time()
    w2_Hg_0 = np.sort(eigsh(Hg_vac, k=k_G, which="SA",
                             return_eigenvectors=False))
    print(f"    H0_G vacuum: {time.time()-t0:.2f}s")

    # -- Identify zero modes ---------------------------------------------
    n_zm = int(np.sum(np.abs(w2_Hp_v) < 0.02))
    print(f"\n[4] H+ zero modes detected: {n_zm} (expect 2 for n=1 BPS)")
    print(f"    Lowest 6 H+ vortex eigenvalues:")
    for i, w in enumerate(w2_Hp_v[:6]):
        tag = "  <-- zero mode" if abs(w) < 0.02 else ""
        print(f"      {i}: omega^2 = {w:+.6e}{tag}")

    # -- Convert to omega, zero modes -> 0 -------------------------------
    om_Hp_v = partial_casimir(w2_Hp_v)
    om_Hp_0 = partial_casimir(w2_Hp_0)
    om_Hg_v = partial_casimir(w2_Hg_v)
    om_Hg_0 = partial_casimir(w2_Hg_0)

    # Common spectral cap: use the smallest highest-omega across all four
    # spectra so we sum equivalent energy ranges. This is coarser than
    # zeta-reg but corrects the "same k count, different omega range" bias.
    omega_cap = min(om_Hp_v.max(), om_Hp_0.max(),
                    om_Hg_v.max(), om_Hg_0.max())
    print(f"\n[5] Common spectral cap omega <= {omega_cap:.4f}")

    sums_cap = matched_sum({
        "Hp_v": om_Hp_v, "Hp_0": om_Hp_0,
        "Hg_v": om_Hg_v, "Hg_0": om_Hg_0}, omega_cap)
    cnts_cap = {k: int((arr <= omega_cap).sum())
                for k, arr in {
                    "Hp_v": om_Hp_v, "Hp_0": om_Hp_0,
                    "Hg_v": om_Hg_v, "Hg_0": om_Hg_0}.items()}
    print(f"    Sum omega (capped):")
    print(f"      H+ vortex  [{cnts_cap['Hp_v']} modes]: "
          f"{sums_cap['Hp_v']:.4f}")
    print(f"      H0 vacuum  [{cnts_cap['Hp_0']} modes]: "
          f"{sums_cap['Hp_0']:.4f}")
    print(f"      H_G vortex [{cnts_cap['Hg_v']} modes]: "
          f"{sums_cap['Hg_v']:.4f}")
    print(f"      H0_G vac   [{cnts_cap['Hg_0']} modes]: "
          f"{sums_cap['Hg_0']:.4f}")

    bosonic = sums_cap["Hp_v"] - sums_cap["Hp_0"]
    ghost   = sums_cap["Hg_v"] - sums_cap["Hg_0"]
    Delta_E = 0.5 * (bosonic - ghost)

    print(f"\n[6] Casimir components (capped, units of m_H = 1, hbar = 1):")
    print(f"    Sum_omega (H+ vortex - H0) = {bosonic:+.4f}  (bosonic)")
    print(f"    Sum_omega (H_G - H0_G)     = {ghost:+.4f}  (ghost, to be SUBTRACTED)")
    print(f"    Delta E_VC = (1/2)(bosonic - ghost) = {Delta_E:+.4f}")

    # -- Naive all-modes sum for comparison ------------------------------
    bosonic_all = om_Hp_v.sum() - om_Hp_0.sum()
    ghost_all   = om_Hg_v.sum() - om_Hg_0.sum()
    Delta_E_all = 0.5 * (bosonic_all - ghost_all)
    print(f"\n[7] For comparison: un-capped (k_cut truncated) sums:")
    print(f"    bosonic all = {bosonic_all:+.4f}")
    print(f"    ghost all   = {ghost_all:+.4f}")
    print(f"    Delta E all = {Delta_E_all:+.4f}")

    print(f"\n    b1.2 bosonic-only estimate: -1.9394")
    print(f"    b1.3 bosonic-only (cap): {bosonic / 2:+.4f}")
    print(f"    b1.3 with ghost subtraction: {Delta_E:+.4f}")
    print(f"    Ghost subtraction changes the one-loop Delta_mu by "
          f"{Delta_E - bosonic/2:+.4f}")

    # -- Plots -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(np.arange(len(w2_Hp_v)), w2_Hp_v, "o-", ms=3,
            label="H+ vortex")
    ax.plot(np.arange(len(w2_Hp_0)), w2_Hp_0, "x-", ms=4, alpha=0.6,
            label="H0 vacuum")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.axhline(1.0, color="r", ls="--", lw=1.0, label=r"$m^2=1$")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\omega^2$")
    ax.set_title(r"Bosonic 4x4 $H_+$ spectrum")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 2.0); ax.set_xlim(0, min(60, k_H))

    ax = axes[1]
    ax.plot(np.arange(len(w2_Hg_v)), w2_Hg_v, "s-", ms=4,
            label=r"$H_G$ vortex")
    ax.plot(np.arange(len(w2_Hg_0)), w2_Hg_0, "^-", ms=4, alpha=0.6,
            label=r"$H_{0G}$ vacuum")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.axhline(1.0, color="r", ls="--", lw=1.0, label=r"$m^2=1$")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\omega^2$")
    ax.set_title(r"FP ghost $H_G$ spectrum")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 2.0); ax.set_xlim(0, min(30, k_G))

    fig.suptitle("Paper 15 Stage 2 b1.3 -- full bosonic + ghost one-loop",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(__file__).parent / "nwt_vortex_fluctuations_b1_3.png"
    fig.savefig(out, dpi=140)
    print(f"\n[8] Plot: {out}")


if __name__ == "__main__":
    main()
