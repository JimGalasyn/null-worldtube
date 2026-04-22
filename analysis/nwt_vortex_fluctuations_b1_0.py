#!/usr/bin/env python3
"""
Stage 2 b1.0 — Scalar-fluctuation spectrum around a BPS Nielsen-Olesen vortex.

This is the first, simplest piece of the Alonso-Izquierdo et al. (2016)
one-loop vortex mass calculation.  We ignore the gauge-field fluctuations
and their mixing with the Higgs phase — those enter in b1.1 through the
full 4×4 matrix operator.  Here we diagonalize the pure scalar modulus
fluctuation:

        [-∇² + V_eff(ρ)] δφ  =  ω² δφ,

where V_eff(ρ) = (3 f(ρ)² − 1)/2 is the Higgs effective mass² in the BPS
n=1 vortex background (λ = 1/2 convention, e=v=1).

Angular separation  δφ(ρ, θ) = R_l(ρ) e^{ilθ}  reduces this to 1D per
angular-momentum sector.  We work in Langer variables  u = √ρ · R  so
the radial equation is of standard Schrödinger form:

        [ -d²/dρ² + (l² − 1/4)/ρ²  +  V_eff(ρ) ] u_n,l  =  ω²_n,l  u_n,l ,

with u(0) = u(R_max) = 0.  Discretize on a uniform radial grid,
diagonalize sparse, extract eigenvalues per l.

The vacuum comparison operator has V_eff = m_H² = 1 everywhere (Higgs mass
at BPS).  Both operators admit the same centrifugal barrier and the same
Dirichlet box, so their spectra converge at high n — making the Casimir-
like sum Σ (ω − ω_vac) well-defined in the large-N limit (though formally
still requiring zeta regularization, deferred to b1.2).

What this script tells us
-------------------------
(a) The vortex spectrum has bound states below m_H = 1 (trapped by the
    attractive core V_eff(0) = −1/2), plus a continuum above.
(b) The translational zero mode lives in the gauge-scalar coupled sector,
    NOT this scalar-only sector — so we expect NO exact zero mode here.
(c) The first scalar-sector partial-wave sums give us a baseline for
    order-of-magnitude estimation of Δμ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).parent))
from nwt_vortex_gravity_flat import solve_bps_profile


# =========================================================================
# Radial Schrödinger operator (Langer form, Dirichlet at both ends).
# =========================================================================

def radial_schrodinger(rho, V, l):
    """Build H = -d²/dρ² + (l²-1/4)/ρ² + V(ρ)  as CSR sparse matrix.

    Grid is uniform.  Dirichlet BC u(rho[0]) = u(rho[-1]) = 0 imposed by
    restricting to interior points (N-2 × N-2 matrix).
    """
    dr = rho[1] - rho[0]
    N = len(rho) - 2                       # interior points
    rho_int = rho[1:-1]
    V_int = V[1:-1]
    centrifugal = (l ** 2 - 0.25) / rho_int ** 2
    diag = 2.0 / dr ** 2 + centrifugal + V_int
    off = -1.0 / dr ** 2 * np.ones(N - 1)
    return diags([off, diag, off], offsets=[-1, 0, 1], format="csr")


def lowest_eigenvalues(H, k):
    vals = eigsh(H, k=k, which="SA", return_eigenvectors=False)
    vals.sort()
    return vals


# =========================================================================
# Main.
# =========================================================================

def main():
    print("=" * 72)
    print(" b1.0  —  Scalar fluctuation spectrum around BPS n=1 vortex")
    print("=" * 72)

    # -- BPS profile on an extended radial grid.
    rho_max = 60.0
    N_rho = 4000
    rho, f, a, fp, ap = solve_bps_profile(
        rho_min=1e-3, rho_max=rho_max, N=2000, dense=N_rho)
    dr = rho[1] - rho[0]
    print(f"\n[1] BPS profile: ρ ∈ [{rho[0]:.3g}, {rho[-1]:.1f}],  "
          f"N={N_rho} pts,  dr={dr:.4f}")
    print(f"    f(0)={f[0]:.3e},  f(ρ_max)={f[-1]:.6f}")

    # -- Effective Higgs mass² in vortex vs vacuum (BPS λ=1/2, v=1)
    V_vortex = 0.5 * (3.0 * f ** 2 - 1.0)
    V_vacuum = np.ones_like(rho)           # m_H² = 2λv² = 1 at BPS, λ=1/2
    mH2 = 1.0

    print(f"\n[2] Effective potentials:")
    print(f"    V_vortex(ρ→0)   = {V_vortex[0]:+.4f}   (attractive core)")
    print(f"    V_vortex(ρ→∞)   = {V_vortex[-1]:+.4f}   (= m_H² = {mH2})")
    print(f"    V_vacuum         = {V_vacuum[0]:+.4f}   (constant)")

    # -- Diagonalize sector by sector.
    l_values = list(range(0, 11))
    k_per_l = 40

    vortex_spec = {}
    vacuum_spec = {}
    for l in l_values:
        Hv = radial_schrodinger(rho, V_vortex, l)
        H0 = radial_schrodinger(rho, V_vacuum, l)
        vortex_spec[l] = lowest_eigenvalues(Hv, k_per_l)
        vacuum_spec[l] = lowest_eigenvalues(H0, k_per_l)

    # =====================================================================
    # [3] Report: bound states, mass gap, sum Σ(ω - ω_vac)
    # =====================================================================
    print(f"\n[3] Per-l spectra  (mass gap m_H² = {mH2}):")
    print(f"    {'l':>3}  {'#bound':>7}  {'ω²_min':>10}  {'ω²_max':>10}  "
          f"{'Σ(ω-ω_v)':>12}")
    partial_sums = {}
    total_shift = 0.0
    for l in l_values:
        omega2_v = vortex_spec[l]
        omega2_0 = vacuum_spec[l]
        n_bound = int(np.sum(omega2_v < mH2))
        # Casimir-like sum, truncated at the k_per_l cutoff
        # ω_n values (some bound-state eigenvalues can be slightly negative
        # from discretization — clip at 0 for √)
        omega_v = np.sqrt(np.maximum(omega2_v, 0.0))
        omega_0 = np.sqrt(np.maximum(omega2_0, 0.0))
        k_common = min(len(omega_v), len(omega_0))
        diff = omega_v[:k_common] - omega_0[:k_common]
        S_l = diff.sum()
        # Angular multiplicity: l=0 once, |l|>0 twice (±l degenerate)
        mult = 1 if l == 0 else 2
        partial_sums[l] = mult * S_l
        total_shift += mult * S_l
        print(f"    {l:>3d}  {n_bound:>7d}  {omega2_v[0]:+10.4f}  "
              f"{omega2_v[-1]:>10.3f}  {S_l:>+12.6f}  (×{mult})")

    print(f"\n    Partial-sum total (l=0..{max(l_values)}):  "
          f"ΔE_scalar ≈ (1/2) Σ = {0.5 * total_shift:+.6f}")
    print(f"    (classical μ_cl = π ≈ {np.pi:.6f}, ratio "
          f"{0.5*total_shift/np.pi:+.4e})")

    print("\n    NOTE: This is partial-wave-truncated and UV-unregulated.")
    print("    The sum slowly diverges as (l_max, k_per_l) → ∞; extracting")
    print("    a finite renormalized Δμ requires zeta regularization (b1.2).")

    # =====================================================================
    # [4] Bound-state count: which l sectors trap modes?
    # =====================================================================
    print(f"\n[4] Bound-state census (ω² < m_H² = 1):")
    n_bound_total = 0
    for l in l_values:
        omega2_v = vortex_spec[l]
        n_bound = int(np.sum(omega2_v < mH2))
        if n_bound > 0:
            mult = 1 if l == 0 else 2
            print(f"    l = ±{l}: {n_bound} bound state(s)  "
                  f"(ω² = {omega2_v[omega2_v < mH2]})  — multiplicity {mult}")
            n_bound_total += mult * n_bound
    print(f"    Total bound scalar modes (with ±l multiplicity): {n_bound_total}")

    # =====================================================================
    # [5] Plot
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5))

    ax = axes[0]
    ax.plot(rho, f, label=r"$f(\rho)$  BPS scalar profile")
    ax.plot(rho, V_vortex, label=r"$V_{\rm eff}(\rho) = (3f^2-1)/2$")
    ax.axhline(mH2, color="k", ls=":", lw=0.8, label=r"$m_H^2 = 1$")
    ax.axhline(0, color="gray", ls="-", lw=0.5)
    ax.set_xlim(0, 12); ax.set_ylim(-0.7, 1.2)
    ax.set_xlabel(r"$\rho / \xi$")
    ax.set_title("BPS background and Higgs effective potential")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    for l in l_values[:6]:
        vals = vortex_spec[l][:15]
        ax.plot([l] * len(vals), vals, "o", ms=4, alpha=0.7)
        vals0 = vacuum_spec[l][:15]
        ax.plot([l + 0.15] * len(vals0), vals0, "x", ms=5, color="gray",
                alpha=0.5)
    ax.axhline(mH2, color="r", ls="--", lw=1.0, label=r"$m_H^2 = 1$")
    ax.set_xlabel("l"); ax.set_ylabel(r"$\omega^2$")
    ax.set_title("Scalar spectrum: vortex (o) vs vacuum (x)")
    ax.set_ylim(-0.6, 3.5)
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ls = np.array(l_values)
    S_l_vals = np.array([partial_sums[l] / (1 if l == 0 else 2) for l in ls])
    cumsum = np.array([sum(partial_sums[ll] for ll in ls[:i + 1])
                       for i in range(len(ls))])
    ax.bar(ls - 0.15, S_l_vals, width=0.3, color="tab:blue",
           label=r"per-l $\Sigma(\omega-\omega_v)$")
    ax.plot(ls, cumsum, "o-", color="crimson", ms=6,
            label=r"cumulative $\sum_{l'\leq l}$ (×mult)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("l"); ax.set_ylabel("sum")
    ax.set_title("Partial-wave Casimir contributions")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("Paper 15 Stage 2 b1.0 — Scalar fluctuations "
                 "around BPS n=1 vortex", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(__file__).parent / "nwt_vortex_fluctuations_b1_0.png"
    fig.savefig(out, dpi=140)
    print(f"\n[6] Plot: {out}")

    # Save for b1.1/b1.2 consumption
    out_npz = Path(__file__).parent / "nwt_vortex_fluctuations_b1_0.npz"
    np.savez_compressed(
        out_npz,
        rho=rho, f=f, V_vortex=V_vortex,
        l_values=np.array(l_values),
        vortex_spec=np.array([vortex_spec[l] for l in l_values]),
        vacuum_spec=np.array([vacuum_spec[l] for l in l_values]),
        k_per_l=k_per_l, rho_max=rho_max,
    )
    print(f"    Spectra saved: {out_npz}")


if __name__ == "__main__":
    main()
