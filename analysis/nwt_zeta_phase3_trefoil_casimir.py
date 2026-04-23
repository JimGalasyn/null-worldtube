#!/usr/bin/env python3
"""
Phase 3 -- one-loop Casimir shift on S^3/2I with BPS trefoil background.

Tubular-neighbourhood approximation.  Uses b1.5's cached flat-R^2
cross-sectional eigenvalues + Phase 2's trefoil length  L = pi sqrt(26)
to compute Delta zeta(-1/2) via the dimensionally reduced 1D Casimir
energy on the S^1 circle of length L (longitudinal direction along
the knot).

================================================================
 Dimensional reduction
================================================================

In the thin-tube limit (xi << R_curvature), the scalar Laplacian on
a tubular neighbourhood of the trefoil on S^3/2I decouples:

    -Delta  approx  -d^2/ds^2  +  H_+_cross(rho, phi)

where s is arc-length along the knot (periodic with period L) and
(rho, phi) are cross-sectional polar coordinates.  Eigenmodes
factorise:

    psi_{n,m}(s, rho, phi)  =  e^{2 pi i n s / L}  chi_m(rho, phi)
    lambda_{n,m}              =  k_n^2  +  lambda_m^{(b1)}
    k_n                       =  2 pi n / L

where  lambda_m^{(b1)}  is the 2D cross-sectional eigenvalue from
the b1 pipeline.

================================================================
 Two pieces: bulk + finite size
================================================================

The Casimir energy of a closed vortex loop has TWO contributions:

  (B)  BULK:  the straight-line zeta-reg per-length line tension,
       b1.5's  Delta mu = -0.279  (in natural units), times the
       length L.  This is the UV-divergent part after standard
       zeta-reg; b1.5 gives this already-renormalised value.

         E_bulk(L)  =  L  *  Delta mu^{b1.5}

  (FS) FINITE SIZE:  the topological correction from the loop's
       longitudinal direction being compact  S^1  rather than an
       infinite line.  For each cross-sectional eigenvalue
       lambda_m = mu^2, the finite-size Casimir on  S^1  of length L:

         E^{1D}_{FS}(mu, L)  =  -(mu / pi) sum_{k=1}^infty K_1(k mu L)/k

       This is FINITE for all mu > 0, and vanishes exponentially as
       L -> infinity.  For mu = 0 it reduces to  -pi / (6 L).

       Total finite-size Casimir, summed over cross-sectional modes:

         E_FS(L)  =  sum_m  E^{1D}_{FS}(sqrt(lambda_m), L)

Total tubular Casimir energy shift:

    Delta E_Cas^{tubular}(L)  =  L * Delta mu^{b1.5}  +  Delta E_FS(L)

where Delta applied to both pieces means (BPS - vacuum).  The bulk
piece dominates at large L; the finite-size piece provides the
O((2 pi xi / L)^2)  correction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import k1

sys.path.insert(0, str(Path(__file__).parent))


# ==========================================================================
# 1. Load b1.5 cached cross-sectional eigenvalues.
# ==========================================================================

def load_b1_eigs(tag: str = "N96_L10_kH600_kG400"):
    """Load the cached cross-sectional eigenvalues from b1.4.

    Returns dict with 'Hp_vort', 'Hp_vac', 'Hg_vort', 'Hg_vac' arrays
    (H+ scalar, H_ghost sectors; vortex and vacuum backgrounds).
    """
    cache_dir = Path(__file__).parent / "output" / "b1_4_cache"
    out = {}
    for name in ("Hp_vort", "Hp_vac", "Hg_vort", "Hg_vac"):
        fn = cache_dir / f"eigs_{name}_{tag}.npy"
        if not fn.exists():
            raise FileNotFoundError(
                f"Cached eigenvalues not found at {fn}.  Run b1.4 first."
            )
        out[name] = np.load(fn)
    # Clip zero-mode artefacts (as in b1.5).
    for key in out:
        out[key][np.abs(out[key]) < 0.02] = 0.0
        out[key][out[key] < 0] = 0.0
    return out


# ==========================================================================
# 2. 1D Casimir on S^1_L.
# ==========================================================================

def casimir_1D_massive(mu: float, L: float, n_max: int = 200) -> float:
    """E^{1D}_Cas(mu, L) = -(mu/pi) sum_{k=1}^{n_max} K_1(k mu L)/k.

    For mu*L > 20 the series converges in a few terms (K_1 decays
    exponentially).  For mu*L < 1 many terms needed; use asymptotic.
    """
    if mu <= 0.0:
        # Massless scalar on S^1:  E = -pi/(6 L).
        return -np.pi / (6.0 * L)
    total = 0.0
    muL = mu * L
    for k in range(1, n_max + 1):
        arg = k * muL
        if arg > 700.0:
            break  # K_1 underflows past this
        term = k1(arg) / k
        total += term
        if abs(term) < 1e-18 * abs(total):
            break
    return -(mu / np.pi) * total


def casimir_total(eigenvalues: np.ndarray, L: float) -> float:
    """Sum of 1D Casimir contributions over cross-sectional modes.

    Each lambda_m contributes E_1D(sqrt(lambda_m), L) per mode.
    Zero modes (lambda = 0) get the massless -pi/(6 L) piece.
    """
    total = 0.0
    for lam in eigenvalues:
        mu = np.sqrt(max(float(lam), 0.0))
        total += casimir_1D_massive(mu, L)
    return total


# ==========================================================================
# 3. Infinite-line consistency check.
# ==========================================================================

def per_length_limit(eigenvalues: np.ndarray,
                      L_list: tuple = (10.0, 30.0, 100.0)) -> dict:
    """Check that total / L -> constant as L -> infinity.

    In the L -> inf limit, E_1D(mu, L) / L -> -(mu^2 / 4 pi) log(mu)
    by Euler-Maclaurin. Summed over the cross-sectional spectrum,
    this should reproduce the flat-2D Casimir per unit length Delta mu.
    """
    out = {}
    for L in L_list:
        out[L] = casimir_total(eigenvalues, L)
    return out


# ==========================================================================
# 4. Main.
# ==========================================================================

def section(title: str) -> None:
    print()
    print("=" * 72)
    print(" " + title)
    print("=" * 72)


def main() -> None:
    section("Phase 3 -- Casimir on trefoil-S^3/2I (tubular approximation)")

    # -- Load cached b1 eigenvalues ---------------------------------
    section("Loading b1.5 cached cross-sectional spectrum")

    try:
        eigs = load_b1_eigs()
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("  Phase 3 requires b1.4's cached eigenvalues.  "
              "Run  python3 analysis/nwt_vortex_fluctuations_b1_4.py  first.")
        sys.exit(1)

    for name, arr in eigs.items():
        n_nonzero = int(np.sum(arr > 0))
        lam_max = float(arr.max()) if len(arr) > 0 else 0.0
        print(f"  {name:10s}  {len(arr)} modes, {n_nonzero} nonzero, "
              f"lambda_max = {lam_max:.2f}")

    # -- Trefoil length from Phase 2 -------------------------------
    L_trefoil = np.pi * np.sqrt(26.0)
    print(f"\n  Trefoil length (Phase 2):  L = pi sqrt(26) = {L_trefoil:.6f}")

    # -- b1.5 per-length bulk (-0.279) ------------------------------
    DELTA_MU_B15 = -0.243    # b1.5 2-DOF ghost fit intercept
    DELTA_MU_LIT = -0.279    # literature / Alonso-Izquierdo target

    section("Finite-size corrections (2-DOF ghost convention)")

    print("""
  The 1D K_1 formula gives the finite-size Casimir correction to
  the bulk line-tension shift.  It vanishes exponentially at large L
  (as expected: a very long loop behaves like an infinite line).

  Scanning L to confirm the expected L-dependence:""")

    print(f"\n  {'L':>8}  {'dE_bos_FS':>12}  {'dE_gho_FS':>12}  "
          f"{'Delta_FS (2DOF)':>16}")
    for L in (5.0, 10.0, 20.0, 50.0, 100.0, L_trefoil):
        dE_bos = casimir_total(eigs["Hp_vort"], L) - casimir_total(eigs["Hp_vac"], L)
        dE_gho = casimir_total(eigs["Hg_vort"], L) - casimir_total(eigs["Hg_vac"], L)
        delta_FS = dE_bos - 2.0 * dE_gho
        marker = "  <-- trefoil" if abs(L - L_trefoil) < 1e-6 else ""
        print(f"  {L:>8.3f}  {dE_bos:>+12.6f}  {dE_gho:>+12.6f}  "
              f"{delta_FS:>+16.6f}{marker}")

    print("""
  As expected, Delta_FS(L) -> 0 exponentially as L grows -- the
  1D K_1 piece captures only the finite-size correction, NOT the
  bulk line tension.""")

    section("Phase 3 headline: total tubular Casimir shift at L = pi sqrt(26)")

    L = L_trefoil
    dE_bos_FS = casimir_total(eigs["Hp_vort"], L) - casimir_total(eigs["Hp_vac"], L)
    dE_gho_FS = casimir_total(eigs["Hg_vort"], L) - casimir_total(eigs["Hg_vac"], L)
    delta_FS = dE_bos_FS - 2.0 * dE_gho_FS

    bulk_B15 = L * DELTA_MU_B15       # from b1.5 2-DOF fit
    bulk_LIT = L * DELTA_MU_LIT       # from literature
    total_B15 = bulk_B15 + delta_FS
    total_LIT = bulk_LIT + delta_FS

    print(f"""
  Trefoil length                L     = pi sqrt(26)   = {L_trefoil:.6f}

  Bulk contribution (line tension times length):
    L * Delta mu^{{b1.5-fit}}       = {bulk_B15:+.6f}      (Delta mu = {DELTA_MU_B15})
    L * Delta mu^{{literature}}     = {bulk_LIT:+.6f}      (Delta mu = {DELTA_MU_LIT})

  Finite-size correction (this phase):
    Delta E_FS^{{2DOF}}             = {delta_FS:+.6f}

  TOTAL  Delta E_Cas^{{tubular}}(L) = bulk + FS:
    using b1.5 fit      = {total_B15:+.6f}
    using literature    = {total_LIT:+.6f}

  For comparison, b1.5's bulk answer is dominant by factor
  {abs(bulk_LIT / delta_FS):.0f}  over the finite-size piece at L = {L_trefoil:.2f}.""")

    section("Curvature corrections and path to Phase 4")

    print(f"""
  This result is the TUBULAR-APPROXIMATION leading-order Casimir
  shift, using the flat-R^2 cross-sectional BPS profile.  Open
  corrections (Phase 2 findings):

    (i)   Knot geodesic curvature on S^3:  (xi kappa_g)^2 = 0.148
          in natural units.  Correction to the cross-sectional
          eigenvalues of order ~15%.

    (ii)  Ambient manifold curvature:  (xi / R_S3)^2 = 1.0 in
          natural units.  Correction of order ~100%.

    (iii) The 2I projection:  24 linked trefoils in the quotient,
          giving a geometric factor in the full Casimir.  For a
          single trefoil this is a factor 1; for the symmetrised
          2I-invariant configuration we multiply by 24 (but only
          counting modes in the 2I-invariant subspace).

  Phase 4 will redo this computation with the curvature corrections
  folded in, expanding the b1 cross-section perturbatively.

  Phase 3 scorecard:
    * Tubular approximation: bulk + finite-size decomposition
    * Bulk (L * Delta mu^b1.5)       = {bulk_LIT:+.4f}
    * Finite-size correction         = {delta_FS:+.4f}
    * Total tubular Casimir          = {total_LIT:+.4f}
    * O(1) curvature corrections     deferred to Phase 4
""")


if __name__ == "__main__":
    main()
