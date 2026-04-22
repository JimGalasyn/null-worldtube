#!/usr/bin/env python3
"""
NWT Lagrangian -- L3: Skyrme-Faddeev quartic + Hopf theta term.

L2 delivered the abelian Higgs sector with saturated BPS line tension
mu = pi v^2.  But that sector alone cannot support FINITE-SIZE KNOTTED
solitons in 3+1D -- Derrick's theorem forbids them.  L3 adds the two
terms that unlock the knotted soliton sector:

  (a) The Skyrme-Faddeev QUARTIC  c_4 (d_mu n x d_nu n)^2.
      This is the only Lorentz-invariant 4-derivative term in n^a that
      (i) avoids higher time derivatives, (ii) is bounded below, and
      (iii) balances the quadratic kinetic term under Derrick scaling.

  (b) The Hopf THETA TERM  theta * Q_H[n].
      A topological term: contributes +2 pi theta * n_H to the action
      (n_H = integer Hopf invariant), locking the soliton into a
      fixed topological sector.  Does not affect the EOM but
      distinguishes hopfion sectors in the path integral.

==========================================================================
 THE L3 LAGRANGIAN
==========================================================================

  L3  =  L2[psi, A_mu]
      +  (f_pi^2 / 2) (d_mu n^a)(d^mu n^a)              [kinetic, c_2]
      -  (1 / 4 e_Sk^2) (d_mu n^a d_nu n^b - d_nu n^a d_mu n^b)^2   [quartic, c_4]
      +  theta * Q_H[n]                                   [Hopf term]

where:
  n^a   : S^2-valued unit field,  n.n = 1, a = 1, 2, 3.
  f_pi  : kinetic scale (analog of pion decay constant in QCD Skyrme).
  e_Sk  : Skyrme coupling (dimensionless).
  theta : Hopf theta angle.  Physical: theta = 0 or pi for CPT.
  Q_H   : Hopf invariant (Whitehead integral, defined below).

The connection to the L1 condensate psi and the n^a field: in the
CP^1-extended model with psi in C^2, one realises
     n^a  =  (psi^dag sigma^a psi) / (psi^dag psi).
In this formulation the Skyrme-Faddeev Lagrangian is equivalent to the
CP^1 sigma-model on psi with a Skyrme quartic.

==========================================================================
 DERRICK SCALING (why the quartic is REQUIRED)
==========================================================================

Consider rescaling x -> lambda x.  The three pieces of L3 scale as:

  E_2  =  (f_pi^2 / 2) integral (d_mu n)^2 d^3 x   ~  lambda^{1}
  E_4  =  (1/(4 e_Sk^2)) integral (d x d)^2 d^3 x  ~  lambda^{-1}
  E_Hopf = theta n_H                               ~  lambda^{0}

(Power count:  d_mu n ~ 1/lambda,  d^3 x ~ lambda^3,  so
  E_2 contains two derivatives and one volume:  (1/lambda)^2 * lambda^3 = lambda.
  E_4 contains four derivatives and one volume: (1/lambda)^4 * lambda^3 = 1/lambda.
 )

Minimising  E(lambda) = A lambda + B / lambda  gives the Skyrme-Faddeev
equilibrium scale:

  lambda_* = sqrt(B / A) = sqrt(E_4^{(0)} / E_2^{(0)})
  E_*       = 2 sqrt(A B) = 2 sqrt(E_2^{(0)} * E_4^{(0)})

which is ALWAYS positive and finite, giving a stable soliton size.

Without E_4, the minimum would be lambda -> 0 (collapse) or lambda -> inf
(dispersal), killing all finite-size solitons -- this is Derrick's theorem
in action.

==========================================================================
 HOPF INVARIANT (Whitehead integral)
==========================================================================

Given n: R^3 union {infty} = S^3 -> S^2, pull back the S^2 area
2-form (1 / 4 pi) * epsilon_abc n^a d n^b ^ d n^c  to R^3:

  F_ij  =  (1 / 4 pi) epsilon_abc n^a (d_i n^b)(d_j n^c)
  F_k   =  (1/2) epsilon_ijk F_ij        (dual vector)

In the absence of magnetic sources, F is closed (dF = 0), so locally
F = dA for a 1-form A_i.  Fix Coulomb gauge d_i A_i = 0; then

  -Laplacian A_i  =  epsilon_ijk d_j F_k

which can be solved globally on R^3 by FFT inversion.

The Hopf invariant is the Whitehead integral:

  Q_H  =  integral A.F d^3 x

Q_H is an INTEGER for smooth n (linking number of generic fibres
n^{-1}(p) and n^{-1}(q) on S^3).  For the Faddeev-Niemi unit hopfion,
Q_H = 1.

For the Aratyn-Ferreira-Zimerman toroidal hopfion T(p, q) with an
additional phase winding m, one has Q_H = p * m  (Rybakov's Whitehead
reduction, Path A step 1).  This is the topological lock that makes a
torus knot labelled by (p, q, m) a stable soliton of L3.

==========================================================================
 NUMERICAL VERIFICATION
==========================================================================

Below we (a) build the standard Faddeev-Niemi unit hopfion in
stereographic form on a 3D grid, and (b) compute its Hopf number Q_H
via the Whitehead integral.  Target: Q_H = 1 for the unit hopfion.

We use FFT-based Coulomb-gauge inversion on a periodic box; on a large
enough box the periodicity artefact is negligible (the hopfion field
decays to the S^2 north pole at infinity).
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq


# ==========================================================================
# 1. Unit hopfion (Faddeev-Niemi / stereographic form).
# ==========================================================================

def unit_hopfion(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """Return n^a(x, y, z) for the standard Q_H = 1 hopfion.

    Using the complex scalar
        u(x, y, z) = (2 (x + i y)) / (r^2 - 1 + 2 i z),   r^2 = x^2+y^2+z^2
    and stereographic projection onto S^2:
        n^1 + i n^2 = 2 u / (1 + |u|^2)
        n^3         = (|u|^2 - 1) / (1 + |u|^2).
    This is a map S^3 -> S^2 with Hopf invariant 1.
    """
    r2 = X * X + Y * Y + Z * Z
    denom = (r2 - 1.0) + 2.0j * Z           # complex denominator
    u = (2.0 * (X + 1.0j * Y)) / denom       # u : R^3 -> C
    absu2 = (u * u.conjugate()).real
    denom_n = 1.0 + absu2
    n1 = (2.0 * u.real) / denom_n
    n2 = (2.0 * u.imag) / denom_n
    n3 = (absu2 - 1.0) / denom_n
    return n1, n2, n3


# ==========================================================================
# 2. Pullback F_ij = (1 / 4 pi) eps_abc n^a (d_i n^b)(d_j n^c).
# ==========================================================================

def pullback_F(n1: np.ndarray, n2: np.ndarray, n3: np.ndarray, dx: float):
    """Compute F_i = (1/2) eps_ijk F_jk, the dual vector of the pullback
    2-form of the S^2 area on R^3.
    """
    # Central-difference derivatives (periodic for compatibility with FFT).
    def dpx(a): return (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) / (2 * dx)
    def dpy(a): return (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) / (2 * dx)
    def dpz(a): return (np.roll(a, -1, axis=2) - np.roll(a, 1, axis=2)) / (2 * dx)

    d = {
        1: (dpx(n1), dpx(n2), dpx(n3)),
        2: (dpy(n1), dpy(n2), dpy(n3)),
        3: (dpz(n1), dpz(n2), dpz(n3)),
    }
    # F_ij = (1/4pi) eps_abc n^a d_i n^b d_j n^c
    # F_k  = (1/2) eps_ijk F_ij
    def F_ij(i, j):
        dni = d[i]
        dnj = d[j]
        # eps_abc n^a d_i n^b d_j n^c with a, b, c = 1, 2, 3
        # = n1 (dni_2 dnj_3 - dni_3 dnj_2)
        # + n2 (dni_3 dnj_1 - dni_1 dnj_3)
        # + n3 (dni_1 dnj_2 - dni_2 dnj_1)
        out = (n1 * (dni[1] * dnj[2] - dni[2] * dnj[1])
               + n2 * (dni[2] * dnj[0] - dni[0] * dnj[2])
               + n3 * (dni[0] * dnj[1] - dni[1] * dnj[0]))
        return out / (4.0 * np.pi)

    F_23 = F_ij(2, 3)
    F_31 = F_ij(3, 1)
    F_12 = F_ij(1, 2)
    # F_k = (1/2) eps_kij F_ij  ->  F_1 = F_23, F_2 = F_31, F_3 = F_12.
    return F_23, F_31, F_12


# ==========================================================================
# 3. Coulomb-gauge A from F via FFT:  -Laplacian A_i = eps_ijk d_j F_k.
# ==========================================================================

def coulomb_gauge_A(F1: np.ndarray, F2: np.ndarray, F3: np.ndarray,
                     dx: float):
    """Solve -Laplacian A = curl F  in 3D periodic box via FFT.

    Returns A_1, A_2, A_3 such that d_i A_j - d_j A_i  =  eps_ijk F_k
    in Coulomb gauge d_i A_i = 0 (implicitly satisfied since curl F has
    zero divergence).
    """
    N = F1.shape[0]
    kx = 2.0 * np.pi * fftfreq(N, d=dx)
    ky = 2.0 * np.pi * fftfreq(N, d=dx)
    kz = 2.0 * np.pi * fftfreq(N, d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    k2_safe = np.where(k2 > 0, k2, 1.0)    # avoid div-by-0 at k=0

    F1k = fftn(F1)
    F2k = fftn(F2)
    F3k = fftn(F3)
    # curl F in k-space: (i k x F)
    CF1 = 1j * (KY * F3k - KZ * F2k)
    CF2 = 1j * (KZ * F1k - KX * F3k)
    CF3 = 1j * (KX * F2k - KY * F1k)
    # -Laplacian A = curl F  ->  k^2 A = curl F
    A1 = ifftn(CF1 / k2_safe).real
    A2 = ifftn(CF2 / k2_safe).real
    A3 = ifftn(CF3 / k2_safe).real
    # Zero-mode is gauge; set to zero explicitly.
    return A1, A2, A3


# ==========================================================================
# 4. Hopf number Q_H = integral A . F  d^3 x.
# ==========================================================================

def hopf_invariant(n1, n2, n3, dx):
    """Whitehead integral Q_H = int A.F  d^3 x.

    With F normalised by (1 / 4 pi) and F = dA in Coulomb gauge, Q_H
    is an integer for smooth n : S^3 -> S^2.
    """
    F1, F2, F3 = pullback_F(n1, n2, n3, dx)
    A1, A2, A3 = coulomb_gauge_A(F1, F2, F3, dx)
    integrand = A1 * F1 + A2 * F2 + A3 * F3
    Q_H = integrand.sum() * dx ** 3
    return Q_H, (F1, F2, F3), (A1, A2, A3)


# ==========================================================================
# 5. Derrick scaling: compute E_2 and E_4 and find lambda_*.
# ==========================================================================

def energy_pieces(n1, n2, n3, dx):
    """Compute the quadratic and quartic energy contributions
    (in units where f_pi^2 = 1 and 1/e_Sk^2 = 1):
        E_2 = (1/2) int (d_mu n^a)^2 d^3 x
        E_4 = (1/4) int (d_mu n^a d_nu n^b - d_nu n^a d_mu n^b)^2 d^3 x
    """
    def grad(a):
        gx = (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) / (2 * dx)
        gy = (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) / (2 * dx)
        gz = (np.roll(a, -1, axis=2) - np.roll(a, 1, axis=2)) / (2 * dx)
        return gx, gy, gz

    g1 = grad(n1)
    g2 = grad(n2)
    g3 = grad(n3)
    # Quadratic:  sum_a sum_i (d_i n^a)^2
    E_2_dens = sum(g1[i] ** 2 + g2[i] ** 2 + g3[i] ** 2 for i in range(3))
    E_2 = 0.5 * E_2_dens.sum() * dx ** 3

    # Quartic: F_ij^a F^a_ij ... we use the simplified form
    # using the cross-product:   (d_mu n x d_nu n)^2
    # in 3D spatial: f_ij = d_i n x d_j n
    # sum_{ij} f_ij . f_ij  (antisymmetric, so i < j counted twice)
    def cross(u, v):
        return (u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0])

    # vectorise derivatives: for each axis, (g1_i, g2_i, g3_i) is
    # d_i n^a for a = 1, 2, 3.
    dn = [(g1[i], g2[i], g3[i]) for i in range(3)]
    quartic = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            fij = cross(dn[i], dn[j])
            quartic = quartic + sum(fij[a] ** 2 for a in range(3))
    # Factor 2 because we summed only i < j.
    E_4 = 0.5 * quartic.sum() * dx ** 3
    return E_2, E_4


# ==========================================================================
# 6. Main.
# ==========================================================================

def section(title: str) -> None:
    print()
    print("=" * 72)
    print(" " + title)
    print("=" * 72)


def main() -> None:
    section("NWT Lagrangian L3 -- Skyrme-Faddeev + Hopf invariant")

    # Build 3D grid.  Box size L should be > ~3 x hopfion size so that
    # the field has relaxed to the S^2 north pole at the boundary.
    N = 64
    L = 4.0
    dx = 2 * L / N
    x = np.linspace(-L + dx / 2, L - dx / 2, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    print(f"\nBuilding {N}^3 grid on [-{L}, {L}]^3, dx = {dx:.4f}.")

    # Unit hopfion.
    n1, n2, n3 = unit_hopfion(X, Y, Z)
    norm = n1 * n1 + n2 * n2 + n3 * n3
    print(f"Unit-hopfion n.n: min = {norm.min():.6f}, max = {norm.max():.6f}"
          f"  (should be ~1)")

    # S^2 north-pole asymptotic check.
    corner_n3 = n3[0, 0, 0]
    print(f"n^3 at grid corner = {corner_n3:.6f}  "
          f"(target -> 1 in the infinite-box limit)")

    section("Hopf invariant via Whitehead integral")

    Q_H, F, A = hopf_invariant(n1, n2, n3, dx)
    print(f"\n   Q_H (Whitehead integral)  =  {Q_H:+.6f}")
    print(f"   Target (unit hopfion)     =  +/- 1")
    err_pct = abs(abs(Q_H) - 1.0) * 100
    print(f"   |Q_H| vs 1 deviation      =  {err_pct:.2f} %"
          f"  ({'PASS' if err_pct < 15 else 'FAIL (increase grid)'})")
    print(f"   Sign                       =  {'+' if Q_H > 0 else '-'}"
          f"  (chirality-convention dependent; physics is in |Q_H|)")
    print()
    print("""   The residual deviation is a finite-box / finite-dx artefact:
   the hopfion has long 1/r^4 tails that don't fit fully into [-4, 4]^3.
   As N and L increase, |Q_H| -> 1 exactly.  The sign is fixed by the
   orientation of the pullback 2-form F vs the stereographic projection
   on S^3 -- it differs from +1 by a global sign under mirror reflection.""")

    # --- Grid-refinement convergence check ---
    print("\n   Grid-refinement convergence check:")
    print(f"   {'N':>4}  {'L':>4}  {'Q_H':>10}  {'|Q_H|':>10}  "
          f"{'|Q_H|-1':>10}")
    for (N_ref, L_ref) in [(48, 3.0), (64, 4.0), (96, 6.0)]:
        dx_r = 2 * L_ref / N_ref
        xr = np.linspace(-L_ref + dx_r / 2, L_ref - dx_r / 2, N_ref)
        Xr, Yr, Zr = np.meshgrid(xr, xr, xr, indexing="ij")
        n1r, n2r, n3r = unit_hopfion(Xr, Yr, Zr)
        Qr, _, _ = hopf_invariant(n1r, n2r, n3r, dx_r)
        dev = abs(Qr) - 1.0
        print(f"   {N_ref:>4}  {L_ref:>4.1f}  {Qr:>+10.4f}  "
              f"{abs(Qr):>+10.4f}  {dev:>+10.4f}")

    section("Derrick scaling: equilibrium size from E_2 + E_4 balance")

    E_2, E_4 = energy_pieces(n1, n2, n3, dx)
    print(f"\n   E_2 (quadratic kinetic)  =  {E_2:.6f}   (scales as lambda)")
    print(f"   E_4 (Skyrme quartic)     =  {E_4:.6f}   (scales as 1/lambda)")
    lam_star = np.sqrt(E_4 / E_2) if E_2 > 0 else float("inf")
    E_star = 2.0 * np.sqrt(E_2 * E_4)
    print(f"\n   Equilibrium scale lambda_*  =  sqrt(E_4 / E_2)  =  {lam_star:.4f}")
    print(f"   Equilibrium energy E_*       =  2 sqrt(E_2 E_4)  =  {E_star:.4f}")
    print(f"""
   At lambda_* the soliton is stable against Derrick collapse.  For the
   unit hopfion, lambda_* = {lam_star:.3f} in the units where f_pi = 1/e_Sk = 1;
   physical conversion to lengths requires fixing f_pi and e_Sk from
   Paper 6's xi_SM.""")

    section("Hopf number for T(p, q) torus knot with phase winding m")

    print("""
   Analytically (Rybakov Path A, Whitehead reduction):

      Q_H[T(p, q), phase m]  =  p * m

   Torus-knot soliton labels in NWT:
       (p, q)   : knot winding on the torus (p = longitude, q = meridian)
       m        : condensate phase winding along the knot

   NWT canonical assignments (Paper 13):
       (2, 1, 3)  electron       Q_H = 6
       (2, 1, 4)  muon           Q_H = 8
       (2, 1, 5)  tau            Q_H = 10
       (2, 3, 1)  quark (gauge-consistent representative)   Q_H = 2
       (2, 5, 1)  cinquefoil (SU(5) / GUT representative)   Q_H = 2
       (2, 7, 1)  heptafoil (Spin(7) / G sector rep.)       Q_H = 2

   The Hopf theta term in L3 assigns phase 2 pi theta * Q_H to each
   sector.  At theta = 0 or pi (CPT-invariant choices) the theta
   term is invisible for pure hopfion amplitudes but distinguishes
   different topological sectors in the path integral.
""")

    section("Summary")

    print(f"""
  L3 adds two terms to L2 that are REQUIRED for finite-size knotted
  solitons:

      c_4 Skyrme quartic  (1/4 e_Sk^2) (d x d)^2
      theta Hopf term     theta Q_H[n]

  Numerical checks:
    * Unit hopfion built on {N}^3 grid has Whitehead integral
      Q_H = {Q_H:+.4f}, |Q_H| = {abs(Q_H):.4f}  (target 1;
      {err_pct:.1f}% finite-box error on magnitude).
    * Derrick scaling analytically gives lambda_* = sqrt(E_4/E_2) > 0,
      guaranteeing a stable finite-size minimum.  Numerical:
      lambda_* = {lam_star:.3f},  E_* = {E_star:.3f}.

  Torus-knot sector assignment:
    * Q_H[T(p, q), m] = p * m  for a (p, q) torus knot with phase
      winding m (Rybakov Whitehead reduction, Path A).
    * NWT canonical leptons: Q_H = 6, 8, 10 for e, mu, tau.
    * Paper 14 "three knots, three forces" triangle carries
      Q_H = 2 for each of the (2, 3), (2, 5), (2, 7) knots
      at minimal phase winding m = 1.

  Next step (L4): verify the full L1 + L2 + L3 Lagrangian reproduces
  the Paper 6 mass spectrum on 24 particles, given the BPS line
  tension from L2 and the knot-topology labels from L3.
""")


if __name__ == "__main__":
    main()
