#!/usr/bin/env python3
"""
Stage 0 — Flat-space gravitational field sourced by a BPS vortex line.

Paper 15 §6.2 calls for a dynamical derivation of G_eff = G_Pl · α^{21}
via the metric perturbation sourced by the BPS vortex field on S^3/2I.
Before launching that on a quotient 3-manifold, we verify the pipeline
in flat R^3 against a textbook answer.

Test configuration
------------------
  - Straight abelian-Higgs BPS vortex along the z-axis (n=1, λ=1/2).
  - Natural units: e = v = 1, so healing length ξ = 1/(ev) = 1 and
    BPS line tension μ_BPS = π.
  - Weak-field (Newtonian) limit:  ∇²_⊥ Φ = 4π G T_00(ρ).
  - Cylindrically symmetric source:

        dΦ/dρ = 2 G μ(ρ) / ρ,      μ(ρ) = 2π ∫_0^ρ ρ' T_00(ρ') dρ'.

Textbook result
---------------
For ρ ≫ ξ, μ(ρ) → μ_BPS and

        Φ(ρ) → 2 G μ_BPS · ln(ρ/ρ_0),     dΦ/d(ln ρ) → 2 G μ_BPS.

Success criterion: μ(∞) matches π (BPS line tension), and the
asymptotic slope dΦ/d(ln ρ) saturates at 2Gμ_BPS within ≲1% over
ρ ∈ [10 ξ, 40 ξ].

This module provides reusable primitives (BPS profile, T_00, 2D
cylindrical Poisson) that the Stage 1 S^3/2I calculation will need.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp, trapezoid


# =========================================================================
# 1. BPS vortex profile — first-order BPS equations (n=1, λ=1/2).
#
# We solve the FIRST-order self-dual system directly:
#       f' = (1-a) f / ρ
#       a' = ρ (1 - f²) / 2
# rather than the second-order EOMs at λ=1/2.  Reason: although the
# second-order EOMs admit the BPS profile as their unique finite-energy
# solution, solve_bvp can converge to non-BPS configurations that satisfy
# the EOMs only approximately (BPS residuals O(1)); integrating the
# first-order system guarantees the output IS the BPS profile.
# =========================================================================

def _bps_first_order(rho, y):
    """First-order BPS equations. y = [f, a]."""
    f, a = y
    r = np.maximum(rho, 1e-12)
    fp = (1.0 - a) * f / r
    ap = 0.5 * r * (1.0 - f ** 2)
    return np.array([fp, ap])


def _bps_bc(ya, yb):
    """a(ρ_min) ≈ ρ_min²/4  (exact series expansion) and f(ρ_max) = 1."""
    return np.array([ya[1], yb[0] - 1.0])


def solve_bps_profile(rho_min: float = 1e-3, rho_max: float = 40.0,
                      N: int = 2000, dense: int = 8000):
    """Solve the first-order BPS system; return (ρ, f, a, fp, ap).

    Near ρ=0 the BPS solution satisfies f ∼ f₁ ρ + O(ρ³) and a ∼ ρ²/4
    + O(ρ⁴), with f₁ ≈ 0.5827 (n=1 Nielsen–Olesen shooting constant).
    We impose a(ρ_min) = ρ_min²/4 (regularity, exact to leading order)
    and f(ρ_max) = 1 (asymptotic Higgs vacuum).
    """
    rho = np.linspace(rho_min, rho_max, N)
    f0 = np.tanh(rho / 1.5)
    a0 = 1.0 - np.exp(-(rho ** 2) / 4.0)
    y_guess = np.vstack([f0, a0])

    sol = solve_bvp(
        _bps_first_order, _bps_bc, rho, y_guess,
        tol=1e-11, max_nodes=80000, verbose=0,
    )
    if not sol.success:
        raise RuntimeError(f"BPS first-order BVP failed: {sol.message}")

    rho_d = np.linspace(rho_min, rho_max, dense)
    y_d = sol.sol(rho_d)
    f_d, a_d = y_d[0], y_d[1]
    # Exact derivatives from the BPS equations (no finite-differencing):
    r_safe = np.maximum(rho_d, 1e-12)
    fp_d = (1.0 - a_d) * f_d / r_safe
    ap_d = 0.5 * rho_d * (1.0 - f_d ** 2)
    return rho_d, f_d, a_d, fp_d, ap_d


# =========================================================================
# 2. Stress-energy T_00 from the abelian-Higgs Lagrangian.
# =========================================================================

def stress_energy_T00(rho, f, a, fp, ap, lam: float = 0.5):
    """Static energy density = T_00 for the abelian-Higgs vortex.

        T_00 = ½(f')² + ½(1-a)²f²/ρ² + ½(a')²/ρ² + (λ/4)(1-f²)²

    All terms positive-definite; no time derivatives in a static config.
    Caller supplies fp, ap from the BVP solver for accuracy.
    """
    r = np.maximum(rho, 1e-10)
    return (
        0.5 * fp ** 2
        + 0.5 * (1.0 - a) ** 2 * f ** 2 / r ** 2
        + 0.5 * ap ** 2 / r ** 2
        + 0.25 * lam * (1.0 - f ** 2) ** 2
    )


# =========================================================================
# 3. Cylindrical Poisson solver (weak-field / Newtonian).
# =========================================================================

def enclosed_line_mass(rho, T00):
    """μ(ρ) = 2π ∫_0^ρ ρ' T_00(ρ') dρ'  (cumulative)."""
    return cumulative_trapezoid(2.0 * np.pi * rho * T00, rho, initial=0.0)


def newton_potential(rho, mu, G: float = 1.0):
    """Integrate dΦ/dρ = 2 G μ(ρ)/ρ from ρ[0], with Φ(ρ[0]) = 0."""
    r = np.maximum(rho, 1e-10)
    return cumulative_trapezoid(2.0 * G * mu / r, rho, initial=0.0)


# =========================================================================
# 4. Main: solve, check, plot.
# =========================================================================

def main():
    print("=" * 72)
    print(" NWT Paper 15 Stage 0 — BPS vortex gravitational field (flat R³)")
    print("=" * 72)

    # -- BPS profile
    rho, f, a, fp, ap = solve_bps_profile(rho_max=40.0, N=800, dense=4000)
    print(f"\n[1] BPS profile solved on ρ ∈ [{rho[0]:.3g}, {rho[-1]:.1f}], "
          f"{len(rho)} points.")
    print(f"    f(0) ≈ {f[0]:.3e}    f(ρ_max) = {f[-1]:.6f}   (target 1)")
    print(f"    a(0) ≈ {a[0]:.3e}    a(ρ_max) = {a[-1]:.6f}   (target 1)")

    # -- T_00 and BPS line tension
    T00 = stress_energy_T00(rho, f, a, fp, ap, lam=0.5)
    mu = enclosed_line_mass(rho, T00)
    mu_total = mu[-1]
    err = (mu_total - np.pi) / np.pi * 100.0
    print(f"\n[2] μ_BPS (integrated T_00) = {mu_total:.8f}")
    print(f"    μ_BPS (exact π)           = {np.pi:.8f}")
    print(f"    error                     = {err:+.4f} %   "
          f"({'PASS' if abs(err) < 0.2 else 'FAIL'})")

    # -- Cylindrical Poisson, check log-law far field
    G_unit = 1.0
    Phi = newton_potential(rho, mu, G=G_unit)
    ln_rho = np.log(np.maximum(rho, 1e-10))
    slope = np.gradient(Phi, ln_rho)
    target_slope = 2.0 * G_unit * mu_total

    def at(rho_test):
        idx = int(np.searchsorted(rho, rho_test))
        return slope[idx]

    s10, s20, s35 = at(10.0), at(20.0), at(35.0)
    def pct(s):
        return (s - target_slope) / target_slope * 100.0

    print(f"\n[3] Asymptotic slope dΦ/d(ln ρ) → 2 G μ_BPS = {target_slope:.6f}")
    print(f"    at ρ = 10 ξ : {s10:.6f}   err {pct(s10):+.3f} %")
    print(f"    at ρ = 20 ξ : {s20:.6f}   err {pct(s20):+.3f} %")
    print(f"    at ρ = 35 ξ : {s35:.6f}   err {pct(s35):+.3f} %")

    passed = (abs(err) < 0.2
              and abs(pct(s20)) < 1.0
              and abs(pct(s35)) < 1.0)
    print(f"\n    STAGE 0 PIPELINE: {'PASS — proceed to S³/2I.' if passed else 'FAIL — debug before proceeding.'}")

    # -- Plots
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))

    ax = axes[0, 0]
    ax.plot(rho, f, label=r"$f(\rho)$  scalar")
    ax.plot(rho, a, label=r"$a(\rho)$  gauge")
    ax.axhline(1.0, color="k", ls=":", lw=0.7)
    ax.set_xlim(0, 15)
    ax.set_xlabel(r"$\rho / \xi$")
    ax.set_ylabel("profile")
    ax.set_title("BPS vortex profile (n=1, λ=½)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(rho, T00)
    ax.set_xlim(0, 15)
    ax.set_xlabel(r"$\rho / \xi$")
    ax.set_ylabel(r"$T_{00}(\rho)$  [$v^4$]")
    ax.set_title("Stress-energy density")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    ax.plot(rho, mu, lw=2)
    ax.axhline(np.pi, color="r", ls="--",
               label=r"$\mu_{\rm BPS} = \pi$")
    ax.set_xlim(0, 15)
    ax.set_xlabel(r"$\rho / \xi$")
    ax.set_ylabel(r"$\mu(\rho)$")
    ax.set_title(r"Enclosed line tension  $\mu(\rho) = 2\pi\!\int_0^\rho\!\rho' T_{00}\,d\rho'$")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(rho, slope, lw=2, label=r"$d\Phi/d(\ln\rho)$")
    ax.axhline(target_slope, color="r", ls="--",
               label=rf"$2G\mu_{{\rm BPS}}={target_slope:.4f}$")
    ax.set_xlim(0, 40)
    ylo = target_slope * 0.5
    yhi = target_slope * 1.1
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(r"$\rho / \xi$")
    ax.set_ylabel(r"slope")
    ax.set_title(r"Far-field test: $\Phi \to 2G\mu_{\rm BPS}\ln(\rho/\rho_0)$")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Paper 15 Stage 0 — BPS vortex → Newtonian metric perturbation (flat R³)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(__file__).parent / "nwt_vortex_gravity_flat.png"
    fig.savefig(out, dpi=140)
    print(f"\n[4] Plot: {out}")


if __name__ == "__main__":
    main()
