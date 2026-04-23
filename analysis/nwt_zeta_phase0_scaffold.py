#!/usr/bin/env python3
"""
Phase 0 scaffold -- heat kernel and Seeley-DeWitt asymptotic on S^3/2I.

Starting the multi-session research project to derive the gravitational
coupling 1/G dynamically at one loop on the Poincare homology sphere
S^3/2I in the BPS trefoil background.  This file is the Phase 0
scaffold: extend the 2I-invariant eigenvalue enumerator from b2_0 to
compute the heat kernel, and validate it against the Seeley-DeWitt
asymptotic expansion in 3D curved space.

================================================================
 Scope of the full project (Phases 0 -- 6)
================================================================

   Phase 0  (THIS FILE)    literature + scaffold + Seeley-DeWitt check
   Phase 1                 free scalar Casimir: ζ(-1/2) on S^3/2I
   Phase 2                 BPS trefoil profile on Heegaard torus
   Phase 3                 H_+ fluctuation operator with BPS background
   Phase 4                 Jorgenson-Lang ζ-reg on 3D curved manifold
   Phase 5                 Ghost port + assemble S_eff + extract 1/G
   Phase 6                 Validate vs. (8/7)^2 (1+α/7)^2 α^21

================================================================
 Phase 0 target: the heat kernel on S^3/2I (free scalar)
================================================================

Spectrum of the Laplacian on the unit 3-sphere:

    -Delta_{S^3} psi = lambda_n psi,
    lambda_n = n(n+2),   n = 0, 1, 2, ...,
    degeneracy on S^3:  (n+1)^2  =  (2j+1)^2  with j = n/2.

The (n+1)^2 eigenspace carries the SU(2) x SU(2) representation
(j, j), realised as matrix elements D^j_{mm'}(g).  Passing to the
quotient S^3/2I = 2I \ SU(2) restricts to left-2I-invariant
combinations, picking out m(j) copies of the (2j+1)-dim right
regular representation.  Hence

    2I-invariant degeneracy at level n = 2j:  g_n = m(j) * (n + 1)

with m(j) the multiplicity of the trivial rep of 2I in the spin-j
irrep of SU(2), computed by Frobenius (path A in b2_0):

    m(j)  =  (1/|2I|) sum_{classes C}  |C|  chi_j(phi_C)
          =  (1/120) [chi_j(0) + chi_j(2pi) + 30 chi_j(pi) + ... ]

Odd n (half-integer j) have  m(j) = 0  because -1 in 2I acts as
(-1)^{2j} = -1 on half-integer spin.  Only even n contribute.

First nonzero levels (verified in b2_0):

    n =  0,  j = 0,  m = 1,  g_0  = 1,    lambda = 0
    n =  2, ..., n = 10:  m = 0  (gap)
    n = 12,  j = 6,  m = 1,  g_12 = 13,   lambda = 168    <-- lambda_1
    n = 20,  j = 10, m = 1,  g_20 = 21,   lambda = 440
    n = 24,  j = 12, m = 1,  g_24 = 25,   lambda = 624
    ...

The heat kernel is

    K(t)  =  Tr exp(-t (-Delta))  =  sum_n  g_n  exp(-t lambda_n).

Seeley-DeWitt asymptotic in 3D curved space:

    K(t)  ~  (4 pi t)^{-3/2}  [ a_0 + a_2 t + a_4 t^2 + ... ]

with

    a_0  =  integral over M of  1 dV      =  Vol(M)
    a_2  =  integral over M of  R/6 dV    =  Vol(M) * R_{const}/6
    a_4  =  integral over M of  (5 R^2 - 2 R_{mu nu}^2 + 2 R_{mu nu rho sigma}^2)/360 dV + ...

For S^3 of radius r, Ricci scalar R = 6/r^2 = 6 (in our unit-r units),
and

    Vol(S^3)       =  2 pi^2
    Vol(S^3/2I)    =  2 pi^2 / 120  =  pi^2 / 60

For S^3/2I with r=1:  a_0 = pi^2/60,  a_2 = pi^2/60.

Validation: K(t) extracted from the finite eigenvalue sum should
match this asymptotic to the expected precision as t -> 0, once we
include enough modes.  This is the Phase 0 deliverable.

================================================================
 Key literature
================================================================

- Camporesi, R.  "Harmonic analysis and propagators on homogeneous
  spaces," Phys. Rep. 196 (1990) 1-134.  Heat kernels + Green
  functions on compact symmetric spaces, including S^n and their
  quotients.  Gives the explicit spectrum and the heat-kernel
  asymptotic.

- Elizalde, E., Odintsov, S., Romeo, A., Bytsenko, A., Zerbini, S.
  "Zeta Regularization Techniques with Applications," World
  Scientific (1994).  Standard text on ζ-regularisation, includes
  Casimir energy on S^3 and quotients by discrete groups.

- Gilkey, P. B.  "Invariance Theory, the Heat Equation, and the
  Atiyah-Singer Index Theorem," 2nd ed., CRC Press (1995).
  Canonical reference for Seeley-DeWitt coefficients on curved
  manifolds, including the full 3D formulae.

- Cahn, R. S. and Wolf, J. A.  "Zeta functions and their
  asymptotic expansions for compact symmetric spaces of rank one,"
  Comment. Math. Helv. 51 (1976) 1-21.  Rigorous ζ-function
  identities for spherical space forms.

- Ikeda, A.  "On the spectrum of a Riemannian manifold of positive
  constant curvature," Osaka J. Math. 17 (1980) 75-93.  Spectrum
  on S^n/Gamma for finite Gamma subset O(n+1).

- Alonso-Izquierdo, A., García Fuertes, W., Mateos Guilarte, J.
  "One-loop mass shift formula for kinks and self-dual vortices,"
  Phys. Rev. D 94 (2016) 125003.  Already cited in Paper 15; the
  flat-2D benchmark the b1 pipeline was validated against.

================================================================
 Key references to consult in Phase 1 for validation anchors
================================================================

- Dowker's papers on Casimir energy for free scalars on S^3 and
  lens spaces (various JHEP and Phys. Rev. D, late 1990s - 2000s).
  The full-S^3 conformally-coupled-scalar Casimir has a closed form
  that our pipeline should reproduce when we sum over ALL j
  (not just 2I-invariant ones) and divide by 120.

- For the 2I case specifically: any published ζ(0) or ζ(-1/2) values
  on the Poincare homology sphere would be an invaluable anchor.
  (If none exist in the literature, Phase 1 produces a first one.)

================================================================
 Code
================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nwt_poincare_sphere_b2_0 import multiplicity_trivial_A  # path A


# ==========================================================================
# Enumerate 2I-invariant scalar Laplacian eigenvalues on S^3/2I.
# ==========================================================================

def enumerate_spectrum(n_max: int = 200) -> list[tuple[int, int, float, int]]:
    """Return list of (n, j, lambda_n, g_n) for n = 0, 2, 4, ..., n_max.

    Odd n contribute nothing (half-integer spin has m(j) = 0 on S^3/2I
    with scalar boundary conditions).  Even n contribute g_n = m(n/2)*(n+1)
    copies of lambda_n = n(n+2).
    """
    out = []
    for n in range(0, n_max + 1, 2):
        j = n / 2.0
        m = int(round(multiplicity_trivial_A(j)))
        g = m * (n + 1)
        if g > 0:
            out.append((n, int(j), float(n * (n + 2)), g))
    return out


def heat_kernel(spectrum: list[tuple[int, int, float, int]],
                 t: float) -> float:
    """K(t) = sum_n g_n exp(-t lambda_n)."""
    return sum(g * np.exp(-t * lam) for (_, _, lam, g) in spectrum)


# ==========================================================================
# Seeley-DeWitt asymptotic on S^3/2I (r = 1, Ricci R = 6).
# ==========================================================================

VOL_S3_OVER_2I = np.pi ** 2 / 60.0  # |2I| = 120; Vol(S^3)=2pi^2
RICCI_S3 = 6.0                      # unit-radius S^3

def seeley_asymptotic(t: float, order: int = 2) -> float:
    """Leading-order heat-kernel asymptotic: K(t) ~ (4 pi t)^{-3/2} [a_0 + a_2 t + ...].

    a_0 = Vol(M)
    a_2 = Vol(M) * R / 6        (R constant on S^3)
    a_4 = Vol(M) * (5 R^2 - 2 R_mu_nu^2 + 2 R_mu_nu_rho_sigma^2)/360

    On S^3 of radius 1: R = 6, R_{mu nu} = 2 g_{mu nu} so R_{mu nu}^2 = 12,
    R_{mu nu rho sigma}^2 = 12 (Einstein-space-of-constant-curvature).
    Hence  a_4 / Vol(M)  =  (5*36 - 2*12 + 2*12) / 360 = 180/360 = 0.5.
    """
    a0 = VOL_S3_OVER_2I
    a2 = VOL_S3_OVER_2I * RICCI_S3 / 6.0
    a4 = VOL_S3_OVER_2I * 0.5
    pref = 1.0 / (4.0 * np.pi * t) ** 1.5
    terms = [a0]
    if order >= 2:
        terms.append(a2 * t)
    if order >= 4:
        terms.append(a4 * t ** 2)
    return pref * sum(terms)


# ==========================================================================
# Main: enumerate, tabulate, compare.
# ==========================================================================

def section(title: str) -> None:
    print()
    print("=" * 72)
    print(" " + title)
    print("=" * 72)


def main() -> None:
    section("Phase 0 scaffold -- heat kernel on S^3/2I")

    n_max = 200
    spectrum = enumerate_spectrum(n_max=n_max)
    total_modes = sum(g for (_, _, _, g) in spectrum)
    print(f"\nEnumerated 2I-invariant scalar spectrum up to n = {n_max}.")
    print(f"Number of 2I-invariant levels:  {len(spectrum)}")
    print(f"Total mode count (sum of g_n):  {total_modes}")

    section("First 10 nontrivial 2I-invariant levels")

    print(f"\n   {'n':>4}  {'j':>4}  {'lambda':>10}  {'g_n':>4}")
    for (n, j, lam, g) in spectrum[:10]:
        print(f"   {n:>4d}  {j:>4d}  {lam:>10.1f}  {g:>4d}")

    section("Heat kernel K(t) vs. Seeley-DeWitt asymptotic")

    print(f"\n   {'t':>8}  {'K(t) sum':>14}  {'SD (a_0+a_2)':>14}  "
          f"{'SD (a_0+a_2+a_4)':>18}  {'sum / SD(LO)':>14}")
    for t in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        K_sum = heat_kernel(spectrum, t)
        K_SD_2 = seeley_asymptotic(t, order=2)
        K_SD_4 = seeley_asymptotic(t, order=4)
        ratio = K_sum / K_SD_2 if K_SD_2 > 0 else float("nan")
        print(f"   {t:>8.4f}  {K_sum:>14.6e}  {K_SD_2:>14.6e}  "
              f"{K_SD_4:>18.6e}  {ratio:>14.6f}")

    section("Interpretation")

    print(f"""
  The enumerated sum K(t) = sum_n g_n exp(-t lambda_n) should converge
  to the Seeley-DeWitt asymptotic  (4 pi t)^(-3/2) [a_0 + a_2 t + a_4 t^2]
  as t -> 0, with the ratio  K(t) / SD  approaching 1.

  For intermediate t (0.01 - 0.1 roughly), we expect the ratio to be
  slightly below 1 because:
    (i)  the truncation at n_max misses high-j modes that contribute
         substantially at small t (where all modes are unsuppressed);
    (ii) higher Seeley-DeWitt coefficients a_6, a_8, ...  contribute
         at order t^3, t^4, ... -- not included here.

  As n_max -> infinity at small t, the ratio -> 1.
  As t -> 0 at fixed n_max, we need more modes to saturate.

  The Vol(S^3/2I) = pi^2/60 = {VOL_S3_OVER_2I:.6f} is baked into a_0.

  This is the foundation for Phase 1:  enumerate a large block of
  eigenvalues, apply the Jorgenson-Lang decomposition

      Gamma(s) zeta(s)  =  int_0^{{t*}} t^{{s-1}} (SD expansion) dt
                        +  int_{{t*}}^infty t^{{s-1}} K(t) dt

  at s = -1/2, and extract  zeta(-1/2)  for the free scalar Casimir
  on S^3/2I.  That quantity is the  no-background  analog of the
  b1.x Delta_mu =  -0.279  benchmark on flat 2D, and will validate
  the 3D-curved pipeline before we add the BPS trefoil in Phase 2.
""")

    section("Next-step checklist (Phase 1)")

    print("""
  [ ] Push n_max -> ~500 or beyond; check K(t) at small t converges.
  [ ] Tabulate {n, j, lambda, g_n} to check against Camporesi /
      Ikeda's published spectra on lens spaces and S^3/Gamma.
  [ ] Implement Jorgenson-Lang ζ(s) integration in 3D, with
      Seeley-DeWitt analytic piece at small t and numerical tail
      at large t.
  [ ] Compute zeta(-1/2) for the free scalar Laplacian on S^3/2I.
  [ ] Cross-validation: compute zeta(-1/2) on full S^3 (no 2I
      projection, sum over ALL j) and compare to published values
      (Dowker, Elizalde).  Our 2I result should be approximately
      (1/120) of the full-S^3 result modulo discrete-group
      corrections from the a_2 Seeley piece (invariant under quotient).
  [ ] If the free-scalar case validates, move to Phase 2
      (BPS trefoil profile on the Heegaard torus of S^3/2I).
""")


if __name__ == "__main__":
    main()
