#!/usr/bin/env python3
"""
Phase 2 -- BPS trefoil on the Heegaard torus of S^3/2I.

Builds the classical BPS vortex background needed for Phase 3's
fluctuation operator.  Two distinct aspects:

  (A)  THE TREFOIL:  a specific closed curve T(2,3) in S^3, embedded
       on the Clifford (Heegaard) torus, with intrinsic length,
       curvature, and torsion.  On S^3/2I, its 2I-orbit has size 24
       (stabiliser Z_5), giving a LINK of 24 trefoils (Paper 15
       S2.3).

  (B)  THE BPS PROFILE:  in the TUBULAR NEIGHBOURHOOD of the trefoil
       (coordinates: arc-length along the knot, perpendicular plane),
       the abelian-Higgs BPS equations reduce at leading order to
       the flat R^2 Nielsen-Olesen equations.  The cross-sectional
       profile  f(rho), a(rho)  is exactly the b1 profile.
       Curvature corrections start at O((xi kappa_knot)^2) where
       kappa_knot is the knot's geodesic curvature on S^3.

Output:
  - Trefoil parameterisation + geometry (length, curvature, torsion)
  - BPS cross-section reused from nwt_vortex_gravity_flat
  - Quantification of kappa_knot and the resulting leading curvature
    correction  (xi kappa_knot)^2
  - Integration weight  L_trefoil  for the Phase 3 fluctuation
    operator in tubular coordinates
  - Note on the 24-copy 2I-orbit structure

This is a scaffolding phase: we use the flat-R^2 BPS profile as the
zeroth-order approximation and quantify its range of validity.  The
Phase 3 fluctuation operator will use this cross-section with a
correction from the knot curvature and the ambient S^3/2I curvature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nwt_vortex_gravity_flat import solve_bps_profile, stress_energy_T00


# ==========================================================================
# 1. Trefoil on the Clifford torus of S^3.
# ==========================================================================
#
# S^3 = {(z_1, z_2) in C^2 : |z_1|^2 + |z_2|^2 = 1}.
# Clifford torus:  |z_1| = |z_2| = 1/sqrt(2).
# T(2,3) parameterisation:
#
#     gamma(theta)  =  (1/sqrt(2)) * (e^{2 i theta}, e^{3 i theta}),
#     theta in [0, 2 pi).
#
# This is a closed embedded curve winding 2 times longitudinally and
# 3 times meridianally on the Heegaard torus.
# ==========================================================================

def trefoil_curve(n_pts: int = 4000):
    """Return (theta, gamma_R8, dgamma_R8, d2gamma_R8) for the trefoil
    embedded in R^4 (=C^2).  Converts complex to R^4 representation.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    sqrt2 = np.sqrt(2.0)
    # z_1 = e^{2i theta} / sqrt(2), z_2 = e^{3 i theta} / sqrt(2)
    gamma = np.stack([
        np.cos(2.0 * theta) / sqrt2,
        np.sin(2.0 * theta) / sqrt2,
        np.cos(3.0 * theta) / sqrt2,
        np.sin(3.0 * theta) / sqrt2,
    ], axis=1)
    dgamma = np.stack([
        -2.0 * np.sin(2.0 * theta) / sqrt2,
         2.0 * np.cos(2.0 * theta) / sqrt2,
        -3.0 * np.sin(3.0 * theta) / sqrt2,
         3.0 * np.cos(3.0 * theta) / sqrt2,
    ], axis=1)
    d2gamma = np.stack([
        -4.0 * np.cos(2.0 * theta) / sqrt2,
        -4.0 * np.sin(2.0 * theta) / sqrt2,
        -9.0 * np.cos(3.0 * theta) / sqrt2,
        -9.0 * np.sin(3.0 * theta) / sqrt2,
    ], axis=1)
    return theta, gamma, dgamma, d2gamma


def trefoil_geometry():
    """Compute length, total absolute curvature, and related R^4 invariants.

    On S^3 with unit radius, the INTRINSIC length of the trefoil is
    obtained by integrating |dgamma/dtheta| dtheta.  For the Clifford-torus
    parameterisation above,  |dgamma/dtheta|^2 = (4 + 9)/2 = 13/2  constant,
    so L = 2 pi sqrt(13/2) = pi sqrt(26).

    The knot's GEODESIC CURVATURE on S^3 is the component of the second
    derivative tangent to the sphere (perpendicular to gamma itself, since
    gamma . gamma = 1 => gamma . dgamma = 0).

    On S^3 the Frenet apparatus gives:
      |d gamma/ds|^2 = 1 (unit-speed reparam.)
      kappa_g = |D_s T - T|  where  T = dgamma/ds,
      D_s = covariant derivative on S^3.
    """
    theta, gamma, dgamma, d2gamma = trefoil_curve()
    speed2 = np.sum(dgamma ** 2, axis=1)            # constant = 13/2
    speed = np.sqrt(speed2)
    dtheta = 2.0 * np.pi / len(theta)
    L = np.sum(speed) * dtheta                       # length on S^3

    # Unit tangent.
    T = dgamma / speed[:, None]
    # Second derivative w.r.t. arc length:  d^2 gamma/ds^2 = d2gamma/speed^2 - (T . d2gamma/speed^3) T
    #   (chain rule)
    d2gamma_ds2 = d2gamma / speed2[:, None]
    # On S^3, the ambient acceleration a = d^2 gamma/ds^2  decomposes into
    # (i) a radial piece  -gamma  due to the sphere constraint,
    # (ii) a geodesic-curvature piece tangent to S^3.
    # kappa_g = |a + gamma|.
    geodesic_acc = d2gamma_ds2 + gamma
    kappa_g_sq = np.sum(geodesic_acc ** 2, axis=1)
    kappa_g = np.sqrt(kappa_g_sq)

    # Average and max geodesic curvature.
    kappa_g_avg = kappa_g.mean()
    kappa_g_max = kappa_g.max()

    return {
        "length": L,
        "length_analytic": np.pi * np.sqrt(26.0),
        "speed_constant": speed[0],
        "kappa_g_avg": kappa_g_avg,
        "kappa_g_max": kappa_g_max,
    }


# ==========================================================================
# 2. BPS cross-sectional profile (reuse b1's flat-R^2 solver).
# ==========================================================================

def bps_cross_section():
    """Solve the BPS first-order ODE for the abelian-Higgs vortex on R^2.

    In the tubular neighbourhood of the trefoil, this profile IS the
    leading-order BPS cross-section, up to corrections of order
    (xi kappa_g)^2 from knot curvature and of order (xi / R_{S^3})^2
    from ambient manifold curvature.

    Returns (rho, f, a, fp, ap) on a dense grid, same as Stage 0.
    """
    return solve_bps_profile(rho_max=40.0, N=1500, dense=6000)


def tube_tension_check(rho, f, a, fp, ap):
    """Integrate T_00 to verify the BPS line tension  mu = pi  (v=1 units)."""
    from scipy.integrate import cumulative_trapezoid
    T00 = stress_energy_T00(rho, f, a, fp, ap, lam=0.5)
    mu = cumulative_trapezoid(2.0 * np.pi * rho * T00, rho, initial=0.0)[-1]
    return mu


# ==========================================================================
# 3. Curvature-correction estimate.
# ==========================================================================

def curvature_correction_scale(kappa_g: float, xi: float = 1.0,
                                  R_S3: float = 1.0) -> dict:
    """Leading-order correction to the flat BPS profile from knot
    curvature and ambient sphere curvature.

    Knot-curvature:  a vortex line with geodesic curvature kappa_g
    on a 3-manifold has its cross-section deformed at order
    (xi kappa_g)^2 by the centrifugal term.  For NWT / Paper 15, xi
    is the condensate healing length and kappa_g = 1/R_curvature.

    Ambient curvature:  the Ricci scalar  R = 6/R_S3^2  contributes
    at order  xi^2 / R_S3^2.

    Both corrections are small  (< 1%)  when  xi << R_curvature.
    In the regime  xi ~ R  (Paper 15's natural-units case with
    R_{S^3/2I} ~ 1) the corrections are O(1) and the flat-space
    profile is only a rough approximation.
    """
    knot_correction = (kappa_g * xi) ** 2
    ambient_correction = (xi / R_S3) ** 2
    return {
        "knot_correction_scale": knot_correction,
        "ambient_correction_scale": ambient_correction,
    }


# ==========================================================================
# 4. Main.
# ==========================================================================

def section(title: str) -> None:
    print()
    print("=" * 72)
    print(" " + title)
    print("=" * 72)


def main() -> None:
    section("Phase 2 -- BPS trefoil background on S^3/2I")

    section("Trefoil geometry on the Clifford torus of S^3")

    geo = trefoil_geometry()
    print(f"""
  Parameterisation:  gamma(theta) = (e^(2 i theta), e^(3 i theta)) / sqrt(2)
  (Clifford torus at |z_1| = |z_2| = 1/sqrt(2) inside S^3.)

  Length on unit S^3:
    numerical           = {geo['length']:.6f}
    analytic  pi sqrt(26) = {geo['length_analytic']:.6f}
    difference          = {abs(geo['length'] - geo['length_analytic']):.2e}
  Speed |dgamma/dtheta| = {geo['speed_constant']:.6f} (constant)

  Geodesic curvature on S^3 (arc-length parameterisation):
    mean kappa_g        = {geo['kappa_g_avg']:.6f}
    max  kappa_g        = {geo['kappa_g_max']:.6f}

  The trefoil has CONSTANT geodesic curvature because it lies on the
  Clifford torus (a flat 2-torus inside S^3) with constant winding
  numbers (2, 3) -- hence uniform curvature in both S^3 and T^2 sense.""")

    section("2I-orbit structure on S^3/2I")

    print("""
  The standard trefoil  gamma(theta)  above is NOT 2I-invariant:
  under left multiplication by g in 2I in SU(2) (acting on S^3 = SU(2)
  by left translation), gamma is mapped to another knot in S^3.  The
  stabiliser of the trefoil in 2I is Z_5 (Paper 15, memory
  paper15-b2-session.md), so the orbit

        O = { g . gamma : g in 2I }  has size |2I| / |Z_5| = 120 / 5 = 24.

  Hence the quotient S^3/2I contains exactly  24  linked trefoils
  (one orbit point each).  This is the 24-element 2T-orbit that
  underlies Paper 15's  168 = 7 x 24  factorisation via McKay.

  Phase 2 scope:  we treat ONE representative trefoil and its
  tubular neighbourhood.  Phase 3 will either (a) symmetrise over
  the 24 copies for the full 2I-invariant fluctuation operator, or
  (b) work on the 2I-invariant subspace of perturbations where the
  24-copy structure factors out by construction.""")

    section("BPS cross-sectional profile (flat-R^2 leading order)")

    rho, f, a, fp, ap = bps_cross_section()
    mu_num = tube_tension_check(rho, f, a, fp, ap)
    print(f"""
  Solved the abelian-Higgs BPS first-order ODE on the normal-plane
  cross-section.  f(rho) and a(rho) are standard Nielsen-Olesen n=1.
  Domain:  rho in [{rho[0]:.3f}, {rho[-1]:.1f}],  {len(rho)} points.

  Asymptotic boundary conditions:
    f(rho -> inf) = {f[-1]:.6f}   (target 1)
    a(rho -> inf) = {a[-1]:.6f}   (target 1)
  Core values:
    f(rho -> 0)  = {f[0]:.4e}
    a(rho -> 0)  = {a[0]:.4e}

  BPS line tension numerical check:
    mu = integral 2 pi rho T_00 drho  =  {mu_num:.8f}
    analytic  (pi v^2 |n|, v=|n|=1)    =  {np.pi:.8f}
    rel error                          =  {abs(mu_num - np.pi)/np.pi * 100:.4f} %""")

    section("Curvature corrections to the flat BPS profile")

    # In natural units v = 1, the healing length is xi = 1/(ev) with e = 1
    # at BPS, so xi = 1.  On unit S^3, R_S3 = 1.  These are the "natural
    # units" Paper 15 uses.
    xi = 1.0
    R_S3 = 1.0
    corr_natural = curvature_correction_scale(geo['kappa_g_avg'], xi, R_S3)

    # If we instead scale to the physical condensate (v = m_e) and
    # trefoil radius = electron Kerr ring ~ xi_SM / 2, we get:
    xi_physical_over_R = 0.5           # xi_SM = ring radius / 2
    corr_physical = curvature_correction_scale(
        geo['kappa_g_avg'], xi_physical_over_R * R_S3, R_S3)

    print(f"""
  Two relevant regimes:

  (A)  Natural-units Paper 15 scenario:  xi = R_S3 = 1.

       (xi kappa_g)^2                 = {corr_natural['knot_correction_scale']:.4f}
       (xi / R_S3)^2                  = {corr_natural['ambient_correction_scale']:.4f}

       Both are O(1).  The flat-R^2 BPS profile is a ROUGH leading-order
       approximation in this regime; curvature corrections are
       non-negligible and Phase 3 will need to include them.

  (B)  Electron Kerr-ring scenario:  xi = 0.5 R_trefoil, R_S3 = R_trefoil.

       (xi kappa_g)^2                 = {corr_physical['knot_correction_scale']:.4f}
       (xi / R_S3)^2                  = {corr_physical['ambient_correction_scale']:.4f}

       Corrections ~ 5-25 %, still meaningful but tractable as a
       perturbation.

  For Phase 3, we use the flat-R^2 profile  f(rho), a(rho)  as the
  background and treat both corrections perturbatively.""")

    section("Phase 2 deliverable summary")

    print(f"""
  Geometric setup:
    * Trefoil on Clifford torus of S^3, length  L = pi sqrt(26) = {geo['length']:.4f}
    * Constant geodesic curvature  kappa_g = {geo['kappa_g_avg']:.4f}
    * 2I-orbit size 24 (stabiliser Z_5 from Paper 15 / memory)

  BPS profile:
    * Flat-R^2 leading order, reused from b1 solver
    * mu = pi to {abs(mu_num - np.pi)/np.pi * 100:.3f} % (same Bogomolny saturation as b1.x)

  Curvature corrections:
    * Natural units (xi = R_S3 = 1):  O(1), perturbation suspect
    * Electron units (xi ~ 0.5 R):    O(25 %), tractable perturbatively

  Ready for Phase 3:  build the scalar-sector fluctuation operator
  H_+ on a tubular neighbourhood of the trefoil using the flat-R^2
  BPS background as zeroth order, with curvature corrections from
  (kappa_g xi)^2  and  (xi / R_S3)^2  terms to be included.
""")


if __name__ == "__main__":
    main()
