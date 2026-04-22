#!/usr/bin/env python3
"""
Stage 2 b1.5 -- Resolve the b1.4 sign discrepancy.

b1.4 gave  Delta mu = +0.25  (positive), vs the literature value
Delta mu = -0.279  (negative).  Magnitudes agree within ~10%, but
signs are opposite.

This script tests the hypothesis that the sign discrepancy is a
convention issue on the ghost's zero-point-energy coefficient.

The paper Alonso-Izquierdo et al. 2016, Eq. 37, writes:

    Delta_mu = (hbar m / 2) [Tr sqrt(H+) - Tr sqrt(H0)
                             - Tr sqrt(HG) + Tr sqrt(H0G)]

This treats the ghost with coefficient -(hbar m /2), same magnitude
as bosons.  But a complex Grassmann ghost has TWO real anticommuting
DOFs, so its zero-point energy should be -hbar (two real DOFs, each
contributing -(hbar/2) with fermionic sign).  Under this convention:

    Delta_mu = (hbar m / 2) (Tr sqrt(H+) - Tr sqrt(H0))
             - hbar m  (Tr sqrt(HG) - Tr sqrt(H0G))
             = (hbar m / 2) [zeta_bos(-1/2) - 2 zeta_gho(-1/2)]

where  zeta_bos = zeta_H+ - zeta_H0  and  zeta_gho = zeta_HG - zeta_H0G.

To test this: compute zeta_bos(-1/2) and zeta_gho(-1/2) SEPARATELY
via Mellin, then combine under both conventions:

    (Eq 37)  Delta_mu = (1/2) [zeta_bos(-1/2) - zeta_gho(-1/2)]
    (2 DOF)  Delta_mu = (1/2) [zeta_bos(-1/2) - 2 zeta_gho(-1/2)]

Whichever gives -0.279 is the correct convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

sys.path.insert(0, str(Path(__file__).parent))
from nwt_vortex_fluctuations_b1_2 import bps_background_2d


def heat_kernel(eigvals: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    lam = np.maximum(eigvals, 0.0)
    return np.exp(-t_grid[:, None] * lam[None, :]).sum(axis=1)


def c2_seeley(what: str, f: np.ndarray, a: np.ndarray,
               r: np.ndarray, dx: float) -> float:
    """Seeley c_2 coefficient of the heat-kernel DIFFERENCE for various ops.

    For H = -Delta + U on R^2:
        c_2[H] = -(1/(4 pi)) integral Tr U d^2 x .

    Returns c_2[H_vort] - c_2[H_vac] for the specified `what`:

        what = 'bos'  --  bosonic 4x4 H+: TrU = 5 f^2 - 1 + 2 a^2/r^2
                          vs TrU0 = 4, so
               c_2[bos-diff] = -(1/(4pi)) int (TrU_H+ - TrU_H0) d^2 x
                             = -(1/(4pi)) int (5 f^2 - 5 + 2 a^2/r^2)
                             = +(1/(4pi)) int (5(1-f^2) - 2 a^2/r^2)

        what = 'gho'  --  ghost scalar H_G: U = f^2  vs  U_0G = 1, so
               c_2[gho-diff] = -(1/(4pi)) int (f^2 - 1)
                             = +(1/(4pi)) int (1 - f^2)
    """
    r_safe = np.maximum(r, dx * 0.5)
    dA = dx * dx
    if what == "bos":
        integrand = 5.0 * (1.0 - f ** 2) - 2.0 * (a / r_safe) ** 2
    elif what == "gho":
        integrand = 1.0 - f ** 2
    else:
        raise ValueError(what)
    return (1.0 / (4.0 * np.pi)) * (integrand * dA).sum()


def zeta_neg_half(eigvals_vort: np.ndarray, eigvals_vac: np.ndarray,
                   c2: float, t_cut: float = 1.0) -> float:
    """Mellin/Jorgenson-Lang zeta at s = -1/2 for a diff-of-ops.

    Delta K(t) = K_vort(t) - K_vac(t) ~ c_2 + O(t)  as t -> 0.

    Gamma(-1/2) * zeta(-1/2)
        = int_0^t_cut (Seeley c_2) ... analytically continued
          + int_t_cut^inf  t^(-3/2) Delta K(t) dt
        = -2 c_2 / sqrt(t_cut)  +  numerical tail.

    zeta(-1/2) = bracket / Gamma(-1/2) = bracket / (-2 sqrt pi).
    """
    # Build heat kernel on log t-grid.
    t_grid = np.logspace(-2.0, 1.0, 400)
    K_vort = heat_kernel(eigvals_vort, t_grid)
    K_vac = heat_kernel(eigvals_vac, t_grid)
    DK = K_vort - K_vac

    # Seeley analytic piece.
    seeley = -2.0 * c2 / np.sqrt(t_cut)

    # Numerical tail.
    mask = t_grid >= t_cut
    tail = trapezoid(t_grid[mask] ** -1.5 * DK[mask], t_grid[mask])

    Gamma_m12 = -2.0 * np.sqrt(np.pi)
    return (seeley + tail) / Gamma_m12


def main():
    print("=" * 72)
    print(" b1.5  --  Resolve b1.4 sign discrepancy (ghost convention)")
    print("=" * 72)

    N = 96
    L = 10.0
    k_H = 600
    k_G = 400
    tag = f"N{N}_L{L:g}_kH{k_H}_kG{k_G}"
    cache = Path(__file__).parent / "output" / "b1_4_cache"

    # Load cached eigenvalues.
    w_Hp_v = np.load(cache / f"eigs_Hp_vort_{tag}.npy")
    w_Hp_0 = np.load(cache / f"eigs_Hp_vac_{tag}.npy")
    w_Hg_v = np.load(cache / f"eigs_Hg_vort_{tag}.npy")
    w_Hg_0 = np.load(cache / f"eigs_Hg_vac_{tag}.npy")
    # Clip zero-mode artefacts.
    for arr in (w_Hp_v, w_Hp_0, w_Hg_v, w_Hg_0):
        arr[np.abs(arr) < 0.02] = 0.0
        arr[arr < 0] = 0.0
    print(f"\n[1] Loaded cached eigenvalues "
          f"(N={N}, L={L}, k_H={k_H}, k_G={k_G}).")

    # Seeley coefficients for each diff.
    _, _, _, r, dx, f, a, _, _, _, _ = bps_background_2d(N, L)
    c2_bos_diff = c2_seeley("bos", f, a, r, dx)
    c2_gho_diff = c2_seeley("gho", f, a, r, dx)
    print(f"\n[2] Seeley c_2 coefficients (L = {L}):")
    print(f"    c_2[H+ - H0]:   {c2_bos_diff:+.5f}")
    print(f"    c_2[HG - H0G]:  {c2_gho_diff:+.5f}")

    # Zeta(-1/2) for each diff via Mellin.
    zeta_bos = zeta_neg_half(w_Hp_v, w_Hp_0, c2_bos_diff, t_cut=1.0)
    zeta_gho = zeta_neg_half(w_Hg_v, w_Hg_0, c2_gho_diff, t_cut=1.0)
    print(f"\n[3] zeta(-1/2) for each diff (t_cut = 1.0):")
    print(f"    zeta_bos-diff(-1/2):  {zeta_bos:+.5f}")
    print(f"    zeta_gho-diff(-1/2):  {zeta_gho:+.5f}")

    # Combine under both conventions.
    print(f"\n[4] Combined Delta_mu under two conventions:")
    Delta_mu_Eq37 = 0.5 * (zeta_bos - zeta_gho)
    Delta_mu_2DOF = 0.5 * zeta_bos - zeta_gho
    print(f"    (Eq 37 literal, ghost coeff = -1/2):")
    print(f"      Delta_mu = (1/2)(zeta_bos - zeta_gho)")
    print(f"               = {Delta_mu_Eq37:+.5f}")
    print(f"    (2 real DOF for ghost, ghost coeff = -1):")
    print(f"      Delta_mu = (1/2) zeta_bos - zeta_gho")
    print(f"               = {Delta_mu_2DOF:+.5f}")
    print()
    print(f"    Literature target:  -0.279")
    print()
    for label, val in [("Eq 37 literal", Delta_mu_Eq37),
                        ("2-DOF ghost", Delta_mu_2DOF)]:
        err = (val - (-0.279))
        pct = abs(err) / 0.279 * 100
        sign_ok = "SAME" if val < 0 else "OPPOSITE"
        print(f"    {label:<20}: {val:+.4f}, "
              f"error {err:+.4f} ({pct:.1f}%), sign {sign_ok}")

    # Scan t_cut for robustness.
    print(f"\n[5] t_cut scan for each convention:")
    print(f"    {'t_cut':>6}  {'zeta_bos':>10}  {'zeta_gho':>10}  "
          f"{'Eq37':>10}  {'2DOF':>10}")
    # Scan including smaller t_cut for extrapolation.
    tcs = [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    e37_vals = []
    e2d_vals = []
    for tc in tcs:
        zb = zeta_neg_half(w_Hp_v, w_Hp_0, c2_bos_diff, t_cut=tc)
        zg = zeta_neg_half(w_Hg_v, w_Hg_0, c2_gho_diff, t_cut=tc)
        e37 = 0.5 * (zb - zg)
        e2d = 0.5 * zb - zg
        e37_vals.append(e37)
        e2d_vals.append(e2d)
        print(f"    {tc:>6.2f}  {zb:>+10.4f}  {zg:>+10.4f}  "
              f"{e37:>+10.4f}  {e2d:>+10.4f}")

    # Fit Delta_mu(t_cut) = A + B * sqrt(t_cut).
    # The B * sqrt(t_cut) term comes from the missing c_4 Seeley
    # coefficient in our asymptotic expansion.  The A intercept is
    # the regularised answer (t_cut -> 0 limit).
    print(f"\n[6] Linear fit  Delta_mu(t_cut) = A + B sqrt(t_cut):")
    tcs_arr = np.array(tcs)
    sqrt_tc = np.sqrt(tcs_arr)
    # Use t_cut <= 1 for the fit (most reliable Seeley regime)
    mask_fit = tcs_arr <= 1.0
    for label, vals in [("Eq 37 literal", e37_vals), ("2-DOF ghost", e2d_vals)]:
        vals_arr = np.array(vals)
        B_fit, A_fit = np.polyfit(sqrt_tc[mask_fit], vals_arr[mask_fit], 1)
        err_from_lit = A_fit - (-0.279)
        sign_ok = "MATCH" if A_fit < 0 else "WRONG SIGN"
        print(f"    {label:<20}: A = {A_fit:+.4f}, B = {B_fit:+.4f}")
        print(f"                        A vs literature -0.279: "
              f"error {err_from_lit:+.4f} ({abs(err_from_lit)/0.279*100:.1f}%), "
              f"sign {sign_ok}")

    print()
    print("=" * 72)
    print(" ASSESSMENT")
    print("=" * 72)
    print(f"""
  Two conventions for the ghost contribution to Delta_mu differ by
  a factor of 2 on the ghost-zeta term.  Whichever matches the
  literature value  Delta_mu = -0.279  (Alonso-Izquierdo et al. 2016,
  Baacke-Lavrelashvili 2008) identifies the correct physical
  convention.

  Our computation:
    - Cached eigenvalues: b1.4 output (k_H=600, k_G=400, N=96, L=10).
    - Seeley c_2 computed from the BPS profile.
    - Mellin / Jorgenson-Lang regularization at t_cut = 1.0.
""")


if __name__ == "__main__":
    main()
