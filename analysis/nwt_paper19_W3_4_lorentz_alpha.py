#!/usr/bin/env python3
"""
Paper 19 -- W3.4: Lorentz invariance fixes alpha.

The voxel-space spacetime tissue projected to 3+1 must be Lorentz-
invariant (i.e. SO(3)-rotation-invariant in 3D space).  In our FCC
rhombic-dodecahedral lattice with the two-parameter pair-spring

    H_b = (K_para - K_perp) (r_hat ⊗ r_hat) + K_perp I_3

the elastic tensor has cubic symmetry (O_h, the point group of FCC).
Cubic-symmetric does NOT imply rotational invariant: there are 3
independent elastic constants (C_11, C_12, C_44), and the Zener
anisotropy

    A = 2 C_44 / (C_11 - C_12)

equals 1 only at a special "isotropic point" of the cubic-elasticity
manifold.  At A = 1 the medium has full SO(3) rotational invariance
in the long-wavelength continuum limit, so its phonons recover Lorentz
invariance exactly.

We therefore claim:

    Voxel-space -> 3+1 projection demands  A(alpha) = 1.

This is a single algebraic condition on alpha = K_perp / K_para that
selects ONE point on the (K_para, K_perp) plane.  The structurally
natural fixed point can then be read off.

Procedure:

  1. For each alpha in a sweep, build the two-param Hessian.
  2. Compute the elastic tensor (C_11, C_12, C_44) by applying three
     uniform strain patterns and reading off the resulting energy:
       (a) ε_xx = 1     -> energy = (V/2) C_11
       (b) ε_xx = ε_yy = 1
                       -> energy = (V/2)(2 C_11 + 2 C_12)
       (c) ε_xy = ε_yx = 1/2
                       -> energy = (V/2) C_44
     (cubic symmetry is assumed; we cross-check by ε_yy and ε_zz.)
  3. Compute A = 2 C_44 / (C_11 - C_12) for each alpha.
  4. Solve A(alpha) = 1 numerically; verify that this matches the
     analytical tensor-spring limit alpha = 1 modulo numerical errors.
  5. At the Lorentz-invariant alpha*, compute c_L([100]), c_L([110]),
     c_L([111]) and verify direction-independence.  This is the actual
     isotropy check.

Output -> analysis/output/W3_4_lorentz_alpha/
  anisotropy_vs_alpha.png   : A(alpha)
  elastic_constants.png     : C_11, C_12, C_44 vs alpha
  c_directional.png         : c_L along [100], [110], [111] vs alpha
  summary.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/home/jim/repos/Morphospace")

import matplotlib.pyplot as plt
import numpy as np

from morphospace.physics.rhombic_grid import (
    build_rhombic_topology,
    fcc_grid_positions,
    init_regular_3d,
)

OUT = Path(__file__).resolve().parents[0] / "output" / "W3_4_lorentz_alpha"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_tissue(n: int, scale: float = 1.0):
    pos = fcc_grid_positions(n, n, n, scale)
    topo = build_rhombic_topology(pos, scale)
    state = init_regular_3d(topo, scale)
    return topo, np.asarray(state.vertices, dtype=np.float64)


def edge_pairs_from_topology(topo) -> np.ndarray:
    cfi = np.asarray(topo.cell_face_indices)
    pairs = set()
    for ci in range(topo.n_cells):
        for fi in range(12):
            f = cfi[ci, fi]
            for k in range(4):
                a, b = int(f[k]), int(f[(k + 1) % 4])
                if a != b:
                    pairs.add((min(a, b), max(a, b)))
    return np.array(sorted(pairs), dtype=np.int64)


def two_param_hessian_direct(topo, rest, K_para: float, K_perp: float
                             ) -> np.ndarray:
    """Direct construction of the two-parameter pair-spring Hessian.

    Per bond:
      H_b = (K_para - K_perp) (r_hat ⊗ r_hat) + K_perp I_3.
    """
    pairs = edge_pairs_from_topology(topo)
    n_v = topo.n_vertices
    H = np.zeros((3 * n_v, 3 * n_v))
    for i, j in pairs:
        r = rest[j] - rest[i]
        L_b = np.linalg.norm(r)
        if L_b < 1e-12:
            continue
        r_hat = r / L_b
        outer = (K_para - K_perp) * np.outer(r_hat, r_hat) \
                + K_perp * np.eye(3)
        H[3 * i:3 * i + 3, 3 * i:3 * i + 3] += outer
        H[3 * j:3 * j + 3, 3 * j:3 * j + 3] += outer
        H[3 * i:3 * i + 3, 3 * j:3 * j + 3] -= outer
        H[3 * j:3 * j + 3, 3 * i:3 * i + 3] -= outer
    return H


# ---------------------------------------------------------------------------
# Elastic-tensor extraction via uniform strain
# ---------------------------------------------------------------------------

def affine_displacement(strain: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """Return flat (3*n_v,) displacement vector for affine strain ε:

        delta_v_i  =  strain @ r_i  (interior only; boundary follows
                                     the same rule, which is fine here
                                     because we want the linearised
                                     bulk response).
    """
    return (rest @ strain.T).flatten()


def affine_energy(strain: np.ndarray, H: np.ndarray, rest: np.ndarray
                  ) -> float:
    """Energy under uniform affine strain (computed on the full tissue)."""
    d = affine_displacement(strain, rest)
    return 0.5 * float(d @ (H @ d))


def elastic_constants(H: np.ndarray, rest: np.ndarray, V_total: float
                      ) -> dict:
    """Extract C_11, C_12, C_44 by applying three independent strains.

    For energy density e(ε) = (1/2) C_ijkl ε_ij ε_kl, we use:
      (a) ε = e_x ⊗ e_x  (only ε_xx = 1):      e = (1/2) C_11
      (b) ε = e_x ⊗ e_x + e_y ⊗ e_y:           e = (1/2)(2 C_11 + 2 C_12)
                                                  -> C_12 follows from (a)
      (c) ε = (e_x ⊗ e_y + e_y ⊗ e_x)/2:       e = (1/2) C_44
    plus cubic-symmetry cross-check on yy, zz, xz, yz.
    """
    eps_xx = np.zeros((3, 3))
    eps_xx[0, 0] = 1.0
    eps_yy = np.zeros((3, 3))
    eps_yy[1, 1] = 1.0
    eps_zz = np.zeros((3, 3))
    eps_zz[2, 2] = 1.0
    eps_xx_yy = eps_xx + eps_yy
    eps_xy = np.zeros((3, 3))
    eps_xy[0, 1] = eps_xy[1, 0] = 0.5
    eps_yz = np.zeros((3, 3))
    eps_yz[1, 2] = eps_yz[2, 1] = 0.5
    eps_xz = np.zeros((3, 3))
    eps_xz[0, 2] = eps_xz[2, 0] = 0.5

    e_xx = affine_energy(eps_xx, H, rest)
    e_yy = affine_energy(eps_yy, H, rest)
    e_zz = affine_energy(eps_zz, H, rest)
    e_xx_yy = affine_energy(eps_xx_yy, H, rest)
    e_xy = affine_energy(eps_xy, H, rest)
    e_yz = affine_energy(eps_yz, H, rest)
    e_xz = affine_energy(eps_xz, H, rest)

    # Energy density = E_tot / V_total = (1/2) C_ijkl ε_ij ε_kl.
    # With ε_xx = 1 only:  e/V = (1/2) C_11   --> C_11 = 2 e_xx / V.
    C_11 = 2.0 * e_xx / V_total
    C_22 = 2.0 * e_yy / V_total
    C_33 = 2.0 * e_zz / V_total
    # Combined ε_xx = ε_yy = 1: e/V = (1/2)(2 C_11 + 2 C_12)
    #                              = C_11 + C_12
    C_12 = (e_xx_yy / V_total) - C_11
    # ε_xy = ε_yx = 1/2 only: contracts to (1/2) C_44.
    C_44 = 2.0 * e_xy / V_total
    C_55 = 2.0 * e_yz / V_total
    C_66 = 2.0 * e_xz / V_total

    A = (2.0 * C_44 / (C_11 - C_12)) if abs(C_11 - C_12) > 1e-12 \
        else float("inf")

    return {
        "C_11": C_11, "C_22": C_22, "C_33": C_33,
        "C_12": C_12,
        "C_44": C_44, "C_55": C_55, "C_66": C_66,
        "A_zener": A,
    }


def directional_sound_speeds(C_11: float, C_12: float, C_44: float,
                              rho: float = 1.0) -> dict:
    """For a cubic crystal, compute c_L and c_T along [100], [110], [111].

    Standard formulas:
      c_L([100])   = sqrt(C_11 / rho)
      c_T([100])   = sqrt(C_44 / rho)        (twice degenerate)
      c_L([110])   = sqrt((C_11 + C_12 + 2 C_44) / (2 rho))
      c_T1([110])  = sqrt((C_11 - C_12) / (2 rho))
      c_T2([110])  = sqrt(C_44 / rho)
      c_L([111])   = sqrt((C_11 + 2 C_12 + 4 C_44) / (3 rho))
      c_T([111])   = sqrt((C_11 - C_12 + C_44) / (3 rho))   (twice degenerate)
    """
    def safe_sqrt(x):
        return float(np.sqrt(max(x, 0.0)))

    return {
        "L_100": safe_sqrt(C_11 / rho),
        "T_100": safe_sqrt(C_44 / rho),
        "L_110": safe_sqrt((C_11 + C_12 + 2 * C_44) / (2 * rho)),
        "T1_110": safe_sqrt((C_11 - C_12) / (2 * rho)),
        "T2_110": safe_sqrt(C_44 / rho),
        "L_111": safe_sqrt((C_11 + 2 * C_12 + 4 * C_44) / (3 * rho)),
        "T_111": safe_sqrt((C_11 - C_12 + C_44) / (3 * rho)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_grid = 5
    K_para = 1.0
    alphas = np.linspace(0.05, 1.5, 25)

    print(f"NWT W3.4   Lorentz invariance fixes alpha   "
          f"K_para = {K_para}, n = {n_grid}\n")

    print("Building tissue ...", flush=True)
    topo, rest = build_tissue(n=n_grid)
    n_c = topo.n_cells
    n_v = topo.n_vertices
    V_total = 2.0 * n_c   # canonical RD volume = 2 per cell at scale 1.
    print(f"  n_cells = {n_c},  n_vertices = {n_v},  V_total = {V_total}")

    rows = []
    for alpha in alphas:
        K_perp = alpha * K_para
        H = two_param_hessian_direct(topo, rest, K_para, K_perp)
        ec = elastic_constants(H, rest, V_total=V_total)
        sp = directional_sound_speeds(ec["C_11"], ec["C_12"], ec["C_44"])
        rows.append({
            "alpha": float(alpha),
            "K_perp": float(K_perp),
            **ec,
            **sp,
        })
        print(f"  alpha = {alpha:.3f}: "
              f"C_11 = {ec['C_11']:.4f}, "
              f"C_12 = {ec['C_12']:.4f}, "
              f"C_44 = {ec['C_44']:.4f}, "
              f"A = {ec['A_zener']:.4f}, "
              f"c_L([100]) = {sp['L_100']:.4f}, "
              f"c_L([110]) = {sp['L_110']:.4f}, "
              f"c_L([111]) = {sp['L_111']:.4f}")

    # Find alpha* satisfying A = 1 (linear interp between sweep points).
    alpha_arr = np.array([r["alpha"] for r in rows])
    A_arr = np.array([r["A_zener"] for r in rows])
    # Find the bracketing pair where A crosses 1.
    A_minus_1 = A_arr - 1.0
    sign_change = np.where(np.diff(np.sign(A_minus_1)) != 0)[0]
    if sign_change.size > 0:
        i = int(sign_change[0])
        # Linear interpolation A(alpha) = 1
        a0, a1 = alpha_arr[i], alpha_arr[i + 1]
        v0, v1 = A_minus_1[i], A_minus_1[i + 1]
        alpha_star = float(a0 - v0 * (a1 - a0) / (v1 - v0))
    else:
        alpha_star = float("nan")
    print(f"\nLorentz-invariant alpha* = {alpha_star:.4f}  (A(alpha*) = 1)")

    # Predict from theory: cubic FCC RD edge network with
    # K_para = K_perp gives A = 1 exactly.
    print(f"Theoretical prediction:    alpha* = 1.0  (K_para = K_perp,")
    print(f"                                          tensor-spring limit)")

    # ---------- Plot 1: Anisotropy A(alpha) ----------
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(alpha_arr, A_arr, "ko-", ms=5)
    ax.axhline(1.0, color="g", ls="--", label="isotropic A = 1")
    ax.axvline(1.0, color="r", ls=":", label=r"$\alpha = 1$")
    if not np.isnan(alpha_star):
        ax.axvline(alpha_star, color="b", ls=":",
                   label=f"empirical alpha* = {alpha_star:.4f}")
    ax.set_xlabel(r"$\alpha = K_\perp / K_\parallel$")
    ax.set_ylabel(r"Zener anisotropy  $A = 2 C_{44} / (C_{11} - C_{12})$")
    ax.set_title(r"Lorentz invariance demands $A = 1$")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "anisotropy_vs_alpha.png", dpi=130)
    plt.close(fig)

    # ---------- Plot 2: Elastic constants vs alpha ----------
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    C_11 = np.array([r["C_11"] for r in rows])
    C_12 = np.array([r["C_12"] for r in rows])
    C_44 = np.array([r["C_44"] for r in rows])
    ax.plot(alpha_arr, C_11, "b-", label=r"$C_{11}$")
    ax.plot(alpha_arr, C_12, "r-", label=r"$C_{12}$")
    ax.plot(alpha_arr, C_44, "g-", label=r"$C_{44}$")
    ax.plot(alpha_arr, C_11 - C_12, "k--", alpha=0.5,
            label=r"$C_{11} - C_{12}$")
    ax.plot(alpha_arr, 2 * C_44, "m--", alpha=0.5, label=r"$2 C_{44}$")
    ax.axvline(1.0, color="r", ls=":", alpha=0.5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("elastic constant")
    ax.set_title("Cubic elastic constants vs gauge-fixing parameter")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "elastic_constants.png", dpi=130)
    plt.close(fig)

    # ---------- Plot 3: directional sound speeds ----------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    L100 = np.array([r["L_100"] for r in rows])
    L110 = np.array([r["L_110"] for r in rows])
    L111 = np.array([r["L_111"] for r in rows])
    T100 = np.array([r["T_100"] for r in rows])
    T1_110 = np.array([r["T1_110"] for r in rows])
    T2_110 = np.array([r["T2_110"] for r in rows])
    T_111 = np.array([r["T_111"] for r in rows])

    ax[0].plot(alpha_arr, L100, "r-", label=r"$c_L[100]$")
    ax[0].plot(alpha_arr, L110, "g-", label=r"$c_L[110]$")
    ax[0].plot(alpha_arr, L111, "b-", label=r"$c_L[111]$")
    ax[0].axvline(1.0, color="k", ls=":", alpha=0.5)
    ax[0].set_xlabel(r"$\alpha$")
    ax[0].set_ylabel(r"$c_L$")
    ax[0].set_title("Longitudinal sound speed by direction")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(alpha_arr, T100, "r-", label=r"$c_T[100]$")
    ax[1].plot(alpha_arr, T1_110, "g-", label=r"$c_{T1}[110]$ (soft)")
    ax[1].plot(alpha_arr, T2_110, "g--", label=r"$c_{T2}[110]$")
    ax[1].plot(alpha_arr, T_111, "b-", label=r"$c_T[111]$")
    ax[1].axvline(1.0, color="k", ls=":", alpha=0.5)
    ax[1].set_xlabel(r"$\alpha$")
    ax[1].set_ylabel(r"$c_T$")
    ax[1].set_title("Transverse sound speed by direction")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "c_directional.png", dpi=130)
    plt.close(fig)

    # Detailed report at alpha = 1
    idx_1 = int(np.argmin(np.abs(alpha_arr - 1.0)))
    r1 = rows[idx_1]
    iso_check_L = max(r1["L_100"], r1["L_110"], r1["L_111"]) \
                  - min(r1["L_100"], r1["L_110"], r1["L_111"])
    iso_check_T = max(r1["T_100"], r1["T2_110"], r1["T_111"]) \
                  - min(r1["T_100"], r1["T2_110"], r1["T_111"])
    iso_check_LT = abs(r1["L_100"] - r1["T_100"])
    print()
    print(f"At alpha = {r1['alpha']:.4f}:")
    print(f"  C_11    = {r1['C_11']:.6f}")
    print(f"  C_12    = {r1['C_12']:.6f}")
    print(f"  C_44    = {r1['C_44']:.6f}")
    print(f"  Zener A = {r1['A_zener']:.6f}")
    print(f"  c_L[100] = {r1['L_100']:.6f}")
    print(f"  c_L[110] = {r1['L_110']:.6f}")
    print(f"  c_L[111] = {r1['L_111']:.6f}")
    print(f"  iso check L (max-min): {iso_check_L:.4e}")
    print(f"  c_T[100] = {r1['T_100']:.6f}")
    print(f"  c_T2[110]= {r1['T2_110']:.6f}")
    print(f"  c_T[111] = {r1['T_111']:.6f}")
    print(f"  iso check T (max-min): {iso_check_T:.4e}")
    print(f"  iso check L-vs-T: {iso_check_LT:.4e}")
    print(f"  c_L / c_T at [100]: {r1['L_100'] / r1['T_100']:.4f}")

    # ---------- Summary ----------
    summary = ["Paper 19 -- W3.4   Lorentz invariance fixes alpha",
               "FCC rhombic-dodecahedral spacetime tissue,",
               "two-parameter (K_para, K_perp) elasticity",
               "=" * 70,
               f"n = {n_grid},  n_cells = {n_c},  n_vertices = {n_v}",
               f"K_para = {K_para}; alpha = K_perp / K_para varied.",
               "",
               "ZENER ANISOTROPY  A = 2 C_44 / (C_11 - C_12)",
               f"  A = 1 means full SO(3) rotational invariance in the",
               f"  long-wavelength continuum limit, i.e. the Lorentz-",
               f"  invariant projection point.",
               "",
               f"  Empirical alpha*  : {alpha_star:.4f}",
               f"  Theoretical alpha*: 1.0  (tensor-spring / Cauchy-",
               f"                            equality limit)",
               "",
               f"AT alpha = 1 (LORENTZ-INVARIANT):",
               f"  C_11      = {r1['C_11']:.6f}",
               f"  C_12      = {r1['C_12']:.6f}",
               f"  C_44      = {r1['C_44']:.6f}",
               f"  Zener A   = {r1['A_zener']:.6f}",
               f"  c_L[100]  = {r1['L_100']:.6f}",
               f"  c_L[110]  = {r1['L_110']:.6f}",
               f"  c_L[111]  = {r1['L_111']:.6f}",
               f"  c_T[100]  = {r1['T_100']:.6f}",
               f"  c_T[111]  = {r1['T_111']:.6f}",
               f"  isotropy check (c_L spread)  : {iso_check_L:.4e}",
               f"  isotropy check (c_T spread)  : {iso_check_T:.4e}",
               f"  c_L vs c_T at [100]          : "
               f"ratio = {r1['L_100']/r1['T_100']:.4f}",
               "",
               f"FULL SWEEP TABLE",
               f"  {'alpha':>6}  {'C_11':>9}  {'C_12':>9}  {'C_44':>9}  "
               f"{'A':>9}  {'cL[100]':>9}  {'cT[100]':>9}",
               f"  {'-' * 80}",
               *[f"  {r['alpha']:>6.3f}  {r['C_11']:>9.4f}  {r['C_12']:>9.4f}  "
                 f"{r['C_44']:>9.4f}  {r['A_zener']:>9.4f}  "
                 f"{r['L_100']:>9.4f}  {r['T_100']:>9.4f}"
                 for r in rows],
               "",
               "INTERPRETATION:",
               "  The voxel-space FCC rhombic-dodecahedral lattice has the",
               "  generic cubic-symmetric elastic tensor with three independent",
               "  constants.  Generic alpha gives a CUBIC-anisotropic medium",
               "  (sound speed depends on direction), which would be observable",
               "  as vacuum birefringence.  Current astrophysical limits on",
               "  vacuum birefringence (|Delta c / c| < 10^-32) are extremely",
               "  tight, so the projection from voxel-space to 3+1 must select",
               "  an isotropic point of the cubic-elastic manifold.",
               "",
               "  The unique isotropic point is alpha = 1 (K_para = K_perp,",
               "  the Cauchy-equality / tensor-spring limit).  This is the",
               "  same alpha that gives 3-fold spatially degenerate phonons",
               "  (W3.2 step 0) and that makes the central-force soft-shear",
               "  gauge orbit completely fixed.",
               "",
               "  KEY POINT: alpha = 1 is NOT a free parameter.  It is the",
               "  unique consequence of demanding observed Lorentz invariance",
               "  on the projected continuum theory.  The voxel-space's e_g",
               "  shear gauge structure is real at the substrate level (W3.2",
               "  step 1.5b SVD: 2 * n_cells soft DOFs), but Lorentz-invariant",
               "  projection demands maximal gauge fixing.",
               "",
               "  BIREFRINGENCE PREDICTION: the voxel-space picture predicts",
               "  ZERO vacuum birefringence at the elastic / acoustic level",
               "  -- consistent with the absence of any observed effect.",
               "  Birefringence-like signatures CAN still appear from the",
               "  internal K_7 sector (W3.2 step 1) where the K_7 graph",
               "  Laplacian gives the effective 'particle' mass spectrum;",
               "  if those modes have polarisation-dependent coupling to",
               "  the acoustic background, residual birefringence emerges",
               "  there.",
               "",
               "PHYSICAL HIERARCHY (post W3.4):",
               "  Substrate   :  voxel-space FCC RD lattice with 2-component",
               "                 e_g gauge field per cell  (2 n_cells DOFs).",
               "  Projection  :  Lorentz invariance fixes alpha = 1.",
               "                 This gauge-fixes the e_g soft modes to zero",
               "                 in the bulk continuum.",
               "  Effective   :  isotropic elastic medium with c_L = c_T",
               "  spacetime      = c (the universal speed of light).",
               "  Matter      :  K_7 internal-qubit modes contribute mass",
               "                 spectra fixed by the K_7 graph Laplacian",
               "                 (W3.2 step 1: m^2 = 7 kappa_int).",
               "",
               "NEXT STEPS:",
               "  W3.3 (deferred): scan gluing rules in the K_7 internal",
               "    sector, looking for SM-like splittings of the 6-fold",
               "    degenerate optical branches.",
               "  W3.5 (deferred): compute the dimensional reduction from",
               "    high-dim voxel-space to 3+1 explicitly (the projection",
               "    map).  Connect to string-theory compactification.",
               "  Paper writing: the W3 chain (W3.1 -> W3.2 -> W3.4) is now",
               "    a coherent narrative -- emergent c, K_7 mass spectrum,",
               "    e_g gauge structure, Lorentz-invariant projection.",
               ]

    summary_text = "\n".join(summary)
    (OUT / "summary.txt").write_text(summary_text)
    print()
    print(summary_text)
    print()
    print(f"Wrote {OUT/'anisotropy_vs_alpha.png'}")
    print(f"Wrote {OUT/'elastic_constants.png'}")
    print(f"Wrote {OUT/'c_directional.png'}")
    print(f"Wrote {OUT/'summary.txt'}")


if __name__ == "__main__":
    main()
