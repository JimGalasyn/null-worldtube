#!/usr/bin/env python3
"""
Stage 2 b1.4 -- spectral zeta-function regularization of the one-loop
BPS vortex Casimir energy.

Extends b1.3 by analytically continuing the formal trace

    Delta E = (m/2) [Tr sqrt(H+) - Tr sqrt(H0) - Tr sqrt(H_G) + Tr sqrt(H0_G)]

via zeta-function regularization:

    Tr sqrt(H) = zeta_H(-1/2),
    zeta_H(s) = (1/Gamma(s)) int_0^infty t^(s-1) Tr[e^(-tH)] dt

and Jorgenson-Lang decomposition of the combined integrand

    I(t) = K_H+(t) - K_H0(t) - K_HG(t) + K_H0G(t)

into a short-time (Seeley) piece and a numerical tail:

    I(t)  =  c_0 / t  +  c_2  +  c_4 t + ...   as t -> 0
    c_0[I] = 0  (areas cancel, since H+ and H0 have equal dim, same for ghost)
    c_2[I] = (1/4pi) int [ 4(1-f^2) - 2 a^2/r^2 ] d^2x

With the c_2 subtracted analytically (integral from 0 to 1 gives c_2 / s,
pole at s=0) and the numerical tail from 1 to infinity computed from
eigenvalues, zeta_I(-1/2) is finite:

    Gamma(-1/2) zeta_I(-1/2) = int_0^1 t^(-3/2) [I(t) - c_2] dt
                             + (-2 c_2)
                             + int_1^infty t^(-3/2) I(t) dt

    Delta E = (m/2) zeta_I(-1/2),     Gamma(-1/2) = -2 sqrt(pi).

Comparison target: Alonso-Izquierdo et al. (2016) and Baacke-Lavrelashvili
PRD 78, 085008 (2008) quote Delta mu ~ -0.279 hbar m for N=1 BPS vortex
at critical coupling. Our finite-box answer will have an L-dependent
log-IR contribution from the a^2/r^2 tail (known mass-renormalization
issue); we will check ball-park agreement and IR scaling.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.integrate import quad, trapezoid

sys.path.insert(0, str(Path(__file__).parent))
from nwt_vortex_fluctuations_b1_2 import bps_background_2d, build_Hplus
from nwt_vortex_fluctuations_b1_3 import build_Hghost


# =========================================================================
# Analytic c_2 Seeley coefficient from the background profile.
# =========================================================================

def seeley_c2_integrand(f: np.ndarray, a: np.ndarray,
                         r: np.ndarray, dx: float,
                         L: float) -> tuple[float, float, float]:
    """Return (c2_bosonic, c2_ghost, c2_I) with the Jorgenson-Lang
    normalization

        c2[H] = - (1/(4 pi)) integral Tr U(x) d^2 x

    Normal trick: skip the constant vacuum-mass pieces that cancel
    in the vortex-minus-vacuum differences. For the combined
    I = K_H+ - K_H0 - K_HG + K_H0G:

        c2[I] = + (1/(4 pi)) int [ (TrU0 - TrU+) - (U_G0 - U_G) ] d^2 x

    With TrU+ = 5 f^2 - 1 + 2 a^2/r^2, TrU0 = 4, U_G = f^2, U_G0 = 1:

        c2[I] = (1/(4 pi)) int [ (4 - 5 f^2 + 1 - 2 a^2/r^2)
                                  - (1 - f^2) ] d^2 x
              = (1/(4 pi)) int [ 4 - 4 f^2 - 2 a^2/r^2 ] d^2 x
              = (1/(4 pi)) int [ 4 (1 - f^2) - 2 a^2/r^2 ] d^2 x.
    """
    r_safe = np.maximum(r, dx * 0.5)
    integrand_bos = -(5.0 * f ** 2 - 1.0 + 2.0 * (a / r_safe) ** 2) \
                    + 4.0                                  # -TrU+ + TrU0
    integrand_gho = -(f ** 2) + 1.0                        # -U_G + U_G0
    integrand_I = integrand_bos - integrand_gho
    # integrand_I should equal 4(1 - f^2) - 2 a^2/r^2
    check = 4.0 * (1.0 - f ** 2) - 2.0 * (a / r_safe) ** 2
    err = np.abs(integrand_I - check).max()
    assert err < 1e-12, f"c2 integrand algebra mismatch: {err}"

    area_element = dx * dx
    c2_bos = -(1.0 / (4.0 * np.pi)) * (integrand_bos * area_element).sum()
    c2_gho = -(1.0 / (4.0 * np.pi)) * (integrand_gho * area_element).sum()
    # Sign convention used below for c_2[I] as the coefficient OF the
    # short-time expansion of I(t), which has the OPPOSITE sign of the
    # standard c_2[H] definition when expressed as
    #     I(t) ~ c_0/t + c_2 + ...
    # Let's compute directly:
    #    I(t) ~ -[c2_HP - c2_H0] + [c2_HG - c2_H0G]
    # Hmm -- be careful. K_H(t) ~ A/(4 pi t) tr I + c_2[H] + ...
    # with c_2[H] = +(1/(4 pi)) int Tr(-U) d^2 x = -(1/(4pi)) int TrU.
    # So K_H(t) ~ A tr(I) /(4 pi t) + c_2[H] + ...
    # Therefore I(t) = K_H+ - K_H0 - K_HG + K_H0G
    #                ~ [c2_H+ - c2_H0] - [c2_HG - c2_H0G] + O(t)
    c2_I_integrand = integrand_I         # + c2 of I at leading
    c2_I = (1.0 / (4.0 * np.pi)) * (c2_I_integrand * area_element).sum()
    return c2_bos, c2_gho, c2_I


# =========================================================================
# Build K_H(t) from eigenvalues.
# =========================================================================

def heat_kernel(eigvals: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """K(t) = sum_n exp(-t lambda_n), positive eigenvalues only."""
    lam = np.maximum(eigvals, 0.0)
    # outer product: (len(t_grid), len(lam))
    return np.exp(-t_grid[:, None] * lam[None, :]).sum(axis=1)


# =========================================================================
# Zeta-regularized Casimir.
# =========================================================================

def zeta_casimir(t_grid, I_t, c2_I, t_cut: float = 1.0,
                 verbose: bool = True) -> float:
    """Jorgenson-Lang regularized zeta(-1/2) with Seeley-at-small-t.

    Numerical I(t) is unreliable for t < t_cut because k_cut-truncated
    eigsh misses high-frequency modes whose contribution to the heat
    kernel is non-negligible for t * lambda_max ~ O(1).

    Strategy: for t < t_cut use the Seeley asymptotic I(t) ~ c_2[I]
    (plus higher-order terms we neglect); for t >= t_cut use the
    numerical I(t). The Mellin integral then decomposes as

        Gamma(-1/2) zeta_I(-1/2)
            = int_0^{t_cut} t^(-3/2) I_Seeley(t) dt          (analytic)
              + int_{t_cut}^{infty} t^(-3/2) I_numeric(t) dt (numerical)

    With I_Seeley(t) = c_2[I] (leading, area-subtracted) the first
    integral analytically continues to

        int_0^{t_cut} t^(-3/2) c_2 dt  =  c_2 * t_cut^s / s  at s=-1/2
                                        =  -2 c_2 / sqrt(t_cut).

    Higher Seeley terms (c_4 t, c_6 t^2, ...) would add
    +2 c_4 sqrt(t_cut) + (2/3) c_6 t_cut^(3/2) + ... , with c_4
    proportional to integrals of U^2 and Delta U. These are omitted
    and represent our leading systematic error.
    """
    # Analytic Seeley piece from 0 to t_cut.
    seeley_piece = -2.0 * c2_I / np.sqrt(t_cut)

    # Numerical piece from t_cut to infinity.
    mask_long = t_grid >= t_cut
    t_l = t_grid[mask_long]
    I_l = I_t[mask_long]
    integrand_l = t_l ** -1.5 * I_l
    int_long = trapezoid(integrand_l, t_l)

    Gamma_m12 = -2.0 * np.sqrt(np.pi)
    bracket = seeley_piece + int_long
    zeta_neg_half = bracket / Gamma_m12

    if verbose:
        print(f"    Seeley analytic -2 c2 / sqrt(t_cut) = {seeley_piece:+.5f}")
        print(f"    int_{{t_cut}}^inf  t^-3/2 I dt        = {int_long:+.5f}")
        print(f"    bracket total                     = {bracket:+.5f}")
        print(f"    zeta_I(-1/2) = bracket / (-2 sqrt pi) = "
              f"{zeta_neg_half:+.5f}")

    return zeta_neg_half


# =========================================================================
# Main.
# =========================================================================

def main():
    print("=" * 72)
    print(" b1.4  --  Zeta regularization of the one-loop BPS Casimir")
    print("=" * 72)

    N = 96
    L = 10.0

    x, X, Y, r, dx, f, a, V1, V2, coupling, M2 = bps_background_2d(N, L)
    print(f"\n[1] Grid: {N}^2 on [{-L:.1f},{L:.1f}]^2,  dx={dx:.4f}, "
          f"area={4*L*L:.1f}")

    # --- Seeley c_2 coefficient from background --------------------------
    c2_bos, c2_gho, c2_I = seeley_c2_integrand(f, a, r, dx, L)
    print(f"\n[2] Seeley c_2 coefficients (from profile):")
    print(f"    c_2[H+] - c_2[H0]    = {c2_bos:+.5f}")
    print(f"    c_2[H_G] - c_2[H0_G] = {c2_gho:+.5f}")
    print(f"    c_2[I] (combined)    = {c2_I:+.5f}")
    print(f"    NB: a^2/r^2 IR log piece makes this L-dependent.")

    # --- Diagonalize (with on-disk cache) --------------------------------
    k_H = 600    # bosonic (36864-dim)
    k_G = 400    # ghost (9216-dim)
    cache_dir = Path(__file__).parent / "output" / "b1_4_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"N{N}_L{L:g}_kH{k_H}_kG{k_G}"

    def _cached_eigsh(op, k, name):
        fn = cache_dir / f"eigs_{name}_{tag}.npy"
        if fn.exists():
            w = np.load(fn)
            if len(w) == k:
                print(f"    {name:9s}: cached  lambda_max={w[-1]:.3f}  {fn.name}")
                return w
        t0_ = time.time()
        w = np.sort(eigsh(op, k=k, which="SA",
                          return_eigenvectors=False))
        np.save(fn, w)
        print(f"    {name:9s}: {time.time()-t0_:.1f}s  "
              f"lambda_max={w[-1]:.3f}  (cached -> {fn.name})")
        return w

    print(f"\n[3] Diagonalizing: k_H={k_H}, k_G={k_G}   (cache: {cache_dir})")
    Hp_v = build_Hplus(N, dx, f, a, V1, V2, coupling, M2, "vortex")
    Hp_0 = build_Hplus(N, dx, f, a, V1, V2, coupling, M2, "vacuum")
    Hg_v = build_Hghost(N, dx, f, "vortex")
    Hg_0 = build_Hghost(N, dx, f, "vacuum")
    w_Hp_v = _cached_eigsh(Hp_v, k_H, "Hp_vort")
    w_Hp_0 = _cached_eigsh(Hp_0, k_H, "Hp_vac")
    w_Hg_v = _cached_eigsh(Hg_v, k_G, "Hg_vort")
    w_Hg_0 = _cached_eigsh(Hg_0, k_G, "Hg_vac")

    # Clip small negative zero modes (coarse-grid artefact).
    for arr in (w_Hp_v, w_Hp_0, w_Hg_v, w_Hg_0):
        arr[np.abs(arr) < 0.02] = 0.0
        arr[arr < 0] = 0.0

    # --- Heat kernel K_H(t) on a log t-grid -----------------------------
    t_grid = np.logspace(-2.0, 1.0, 400)        # 0.01 .. 10
    K_Hp_v = heat_kernel(w_Hp_v, t_grid)
    K_Hp_0 = heat_kernel(w_Hp_0, t_grid)
    K_Hg_v = heat_kernel(w_Hg_v, t_grid)
    K_Hg_0 = heat_kernel(w_Hg_0, t_grid)
    I_t = K_Hp_v - K_Hp_0 - K_Hg_v + K_Hg_0

    print(f"\n[4] Heat-kernel values at selected t:")
    print(f"          t       K_H+-K_H0   K_HG-K_H0G    I(t)")
    for t_probe in [0.02, 0.1, 1.0, 3.0]:
        j = np.argmin(np.abs(t_grid - t_probe))
        print(f"    {t_grid[j]:8.3f}  {K_Hp_v[j] - K_Hp_0[j]:+9.4f}   "
              f"{K_Hg_v[j] - K_Hg_0[j]:+9.4f}   {I_t[j]:+9.4f}")
    print(f"    c_2[I] (expected t->0 limit of I):   {c2_I:+9.4f}")
    print(f"    I(t_min) =                            {I_t[0]:+9.4f}"
          f"  (need k_cut high enough for t<<1 accuracy)")

    # --- Zeta regularization (Seeley at small t, numerical large t) -----
    print(f"\n[5] Zeta regularization (Seeley for t<t_cut, numerical for t>t_cut):")
    zeta_half = zeta_casimir(t_grid, I_t, c2_I, t_cut=1.0, verbose=True)
    Delta_E = 0.5 * zeta_half
    print(f"\n    Delta E_VC (t_cut=1.0) = {Delta_E:+.5f}"
          f"   [hbar=m=1]")
    print(f"    Literature (N=1 BPS, critical coupling): ~ -0.279")

    # --- Probe t_cut sensitivity ----------------------------------------
    # If result is independent of t_cut, Seeley + numerical are consistent.
    # If it varies, we're missing c_4 (and higher) Seeley terms.
    print(f"\n[6] Dependence on t_cut (drift indicates missing c_4 and higher):")
    for tc in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        z = zeta_casimir(t_grid, I_t, c2_I, t_cut=tc, verbose=False)
        print(f"    t_cut={tc:.2f}:  Delta E = {0.5*z:+.5f}")

    # Fit c_4 empirically from t_cut scan.
    # If Seeley with c_2 and c_4: bracket = -2 c_2/sqrt(t_cut)
    #                              + 2 c_4 sqrt(t_cut) + int_{t_cut}^inf...
    # So Delta_E(t_cut) - (asymptotic answer) = (c_4 / (-2 sqrt pi)) sqrt(t_cut) + ...
    # Scan provides an empirical read on c_4.
    tcs = np.array([0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    Des = np.array([0.5 * zeta_casimir(t_grid, I_t, c2_I, t_cut=tc,
                                        verbose=False) for tc in tcs])
    # Linear fit in sqrt(t_cut): Delta_E ~ A + B * sqrt(t_cut).
    X = np.sqrt(tcs)
    A, B = np.polyfit(X, Des, 1)[1], np.polyfit(X, Des, 1)[0]
    Delta_E_extrap = A  # intercept at t_cut -> 0
    c4_empirical = -2.0 * np.sqrt(np.pi) * B  # B = c_4 / (-2 sqrt pi), loosely
    print(f"\n    Linear fit Delta_E(t_cut) = A + B sqrt(t_cut):")
    print(f"      A (t_cut -> 0 intercept) = {A:+.4f}  (best Delta E estimate)")
    print(f"      B                        = {B:+.4f}")
    print(f"      implied c_4[I]           = {c4_empirical:+.4f}")
    print(f"    Literature target: -0.279")
    print(f"    Extrapolated Delta E = {Delta_E_extrap:+.4f}   "
          f"[agreement with literature: "
          f"{abs(Delta_E_extrap - (-0.279))/0.279*100:.1f}%]")

    # --- Plot -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    ax.semilogx(t_grid, K_Hp_v - K_Hp_0, "-", label=r"$K_{H_+} - K_{H_0}$")
    ax.semilogx(t_grid, K_Hg_v - K_Hg_0, "-",
                label=r"$K_{H_G} - K_{H_{0G}}$")
    ax.semilogx(t_grid, I_t, "k-", lw=1.8, label=r"$I(t)$")
    ax.axhline(c2_I, color="red", ls="--", lw=1,
               label=rf"$c_2[I] = {c2_I:+.3f}$")
    ax.axhline(0, color="k", lw=0.3)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("heat-kernel diff")
    ax.set_title("Heat-kernel differences")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    integrand = t_grid ** -1.5 * I_t
    ax.semilogx(t_grid, integrand, "-",
                label=r"$t^{-3/2} I(t)$ (raw)")
    integrand_sub = t_grid ** -1.5 * (I_t - c2_I)
    ax.semilogx(t_grid, integrand_sub, "-",
                label=r"$t^{-3/2}(I(t)-c_2)$ (subtracted)")
    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(1.0, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("integrand")
    ax.set_title(r"Mellin integrand for $\zeta_I(-1/2)$")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("Paper 15 Stage 2 b1.4 -- zeta-regularized "
                 "one-loop Casimir", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(__file__).parent / "nwt_vortex_fluctuations_b1_4.png"
    fig.savefig(out, dpi=140)
    print(f"\n[7] Plot: {out}")


if __name__ == "__main__":
    main()
