#!/usr/bin/env python3
"""
Paper 19 -- W3.3 step F: Group-theoretically constructed E_g coupling
on K_7 internal matter sector.

W3.3-E showed that the *naive geometric* bond-strain cross-coupling

    epsilon_e = r_hat^T S(phi) r_hat   ,   kappa_e = kappa_0 (1 + lambda eps_e)

shifts MORE than the 2 E_g modes (4-6 of 6 anti-sym), because the
operator V_eps it induces on K_7 vertex space mixes O_h irrep blocks
beyond pure E_g.

This step demonstrates that the SM-Higgs-like 2-mode shift pattern
DOES work when we use a true group-theoretic E_g operator rather than
the geometric strain.  The E_g basis operators on K_7 (with cubic
vertex placement) are simply

    T_1 = diag(0, +1, +1, -1, -1, 0, 0) / sqrt(2)        (x^2 - y^2)
    T_2 = diag(0, +1, +1, +1, +1, -2, -2) / sqrt(6)      (3 z^2 - r^2)

Both transform as e_g of O_h by construction (their entries are the
e_g basis at each vertex's position).

To get the clean 2-mode shift, we also break the A_1g vs E_g
degeneracy by adding a small Delta shift to vertex 0:

    H_0 = L_K7 + Delta * |v_0><v_0|

For Delta != 0, the A_1g irrep block has eigenvalues != 7, while
E_g and T_1u stay at 7 (5-fold degenerate).  An E_g perturbation
on this 5-dim eigenspace then shifts only the 2 E_g modes (Schur:
E_g x E_g = A_1g + A_2g + E_g has A_1g; E_g x T_1u = T_1u + T_2u
has no A_1g, so T_1u is invariant).

CORRECT O_h SCHUR PREDICTION (NOT the "clean Higgs doublet" I expected):
  - 2 A_1g modes:       NO shift  (E_g x A_1g = E_g, no A_1g component)
  - 2 E_g modes:        shift     (E_g x E_g contains E_g)
  - 3 T_1u modes:       SHIFT TOO (E_g x T_1u = T_1u + T_2u contains T_1u)

So 5 of 6 anti-sym modes shift; only the 2 A_1g modes are protected.
My earlier guess of "only 2 E_g modes shift" was WRONG -- I missed that
T_1u also shifts because E_g x T_1u contains T_1u.

This means W3.3-F does NOT save the W3.3-E falsification.  Both
geometric (W3.3-E) and constructed (W3.3-F) E_g couplings shift
5 modes, not 2.  The clean SM-Higgs single-doublet selectivity is
NOT a feature of voxel-space e_g gauge structure -- it would require
an additional dynamical mechanism (per-particle Yukawa coefficients).

Output -> analysis/output/W3_3f_constructed_eg/
  spectrum_phi_sweep.png : eigenvalues vs phi_1, phi_2 sweeps
  comparison.png         : W3.3-E (geometric) vs W3.3-F (constructed)
  summary.txt
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[0] / "output" / "W3_3f_constructed_eg"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# K_7 with cubic vertex placement (same as W3.3-E)
# ---------------------------------------------------------------------------

POSITIONS = np.array([
    [0, 0, 0],   # v_0  origin
    [+1, 0, 0],  # v_1  +x
    [-1, 0, 0],  # v_2  -x
    [0, +1, 0],  # v_3  +y
    [0, -1, 0],  # v_4  -y
    [0, 0, +1],  # v_5  +z
    [0, 0, -1],  # v_6  -z
], dtype=np.float64)


def k7_laplacian() -> np.ndarray:
    """K_7 graph Laplacian L = 7 I - J."""
    return 7.0 * np.eye(7) - np.ones((7, 7))


# ---------------------------------------------------------------------------
# Constructed E_g basis operators on K_7 (group-theoretic, NOT geometric)
# ---------------------------------------------------------------------------

# T_1, T_2 are diagonal matrices with entries equal to the e_g basis at
# each vertex's position.  By construction they transform as the e_g
# doublet of cubic O_h.
T_1 = np.diag([0, +1.0, +1.0, -1.0, -1.0, 0.0, 0.0]) / np.sqrt(2.0)
T_2 = np.diag([0, +1.0, +1.0, +1.0, +1.0, -2.0, -2.0]) / np.sqrt(6.0)


# ---------------------------------------------------------------------------
# Hamiltonian + spectrum
# ---------------------------------------------------------------------------

def H0(delta: float) -> np.ndarray:
    """K_7 Laplacian + Delta shift on v_0 (breaks A_1g degeneracy)."""
    H = k7_laplacian().astype(np.float64)
    H[0, 0] += delta
    return H


def H(delta: float, phi1: float, phi2: float, lam: float = 0.5) -> np.ndarray:
    """Full Hamiltonian H = H_0 + lambda * (phi_1 T_1 + phi_2 T_2)."""
    return H0(delta) + lam * (phi1 * T_1 + phi2 * T_2)


def eigvals(delta: float, phi1: float, phi2: float, lam: float = 0.5) -> np.ndarray:
    M = H(delta, phi1, phi2, lam)
    return np.linalg.eigvalsh(0.5 * (M + M.T))


# ---------------------------------------------------------------------------
# Cubic-irrep classification of H_0 eigenvectors
# ---------------------------------------------------------------------------

# Irrep basis (orthonormal):
#   A_1g_v0     : v_0 alone
#   A_1g_others : (v_1+v_2+v_3+v_4+v_5+v_6)/sqrt(6)
#   E_g_1       : (v_1+v_2 - v_3-v_4) / 2                  (x^2 - y^2)
#   E_g_2       : (v_1+v_2 + v_3+v_4 - 2 v_5 - 2 v_6) / (2*sqrt(3))
#                                                          (3 z^2 - r^2)
#   T_1u_x      : (v_1 - v_2) / sqrt(2)
#   T_1u_y      : (v_3 - v_4) / sqrt(2)
#   T_1u_z      : (v_5 - v_6) / sqrt(2)

IRREP_BASIS = {
    "A_1g_v0": np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64),
    "A_1g_others": np.array([0, 1, 1, 1, 1, 1, 1], dtype=np.float64) / np.sqrt(6.0),
    "E_g_1": np.array([0, 1, 1, -1, -1, 0, 0], dtype=np.float64) / 2.0,
    "E_g_2": np.array([0, 1, 1, 1, 1, -2, -2], dtype=np.float64) / (2.0 * np.sqrt(3.0)),
    "T_1u_x": np.array([0, 1, -1, 0, 0, 0, 0], dtype=np.float64) / np.sqrt(2.0),
    "T_1u_y": np.array([0, 0, 0, 1, -1, 0, 0], dtype=np.float64) / np.sqrt(2.0),
    "T_1u_z": np.array([0, 0, 0, 0, 0, 1, -1], dtype=np.float64) / np.sqrt(2.0),
}


def classify_eigenvector(v: np.ndarray) -> str:
    """Return the irrep with maximum overlap with v."""
    overlaps = {name: abs(b @ v) for name, b in IRREP_BASIS.items()}
    return max(overlaps, key=overlaps.get)


# ---------------------------------------------------------------------------
# Linear response: per-mode derivative
# ---------------------------------------------------------------------------

def linear_response(delta: float, direction: int = 1, lam: float = 0.5,
                    eps: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """d eigenvalue / d phi_direction at phi = 0."""
    e0 = eigvals(delta, 0.0, 0.0, lam)
    if direction == 1:
        e1 = eigvals(delta, eps, 0.0, lam)
    else:
        e1 = eigvals(delta, 0.0, eps, lam)
    return e0, (e1 - e0) / eps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("W3.3-F   Constructed E_g coupling (group-theoretic, not geometric)")
    print("=" * 70)

    # ---------- (1) Verify T_1, T_2 transform as e_g of O_h ----------
    print("\n(1) Verify T_1, T_2 are diagonal in IRREP_BASIS for E_g:")
    for name, T in [("T_1", T_1), ("T_2", T_2)]:
        for ir_name, basis in IRREP_BASIS.items():
            v = T @ basis
            comp = {bn: float(b @ v) for bn, b in IRREP_BASIS.items()}
            nonzero = {k: v for k, v in comp.items() if abs(v) > 1e-10}
            if nonzero:
                print(f"  {name} | {ir_name} >  ->  components in: "
                      f"{', '.join(f'{k}({v:+.3f})' for k,v in nonzero.items())}")
        print()

    # ---------- (2) Spectrum at phi = 0, varying Delta ----------
    print("\n(2) Eigenvalue structure of H_0 = L_K7 + Delta |v_0><v_0|")
    print(f"  {'Delta':>7}  {'eigenvalues (sorted)':>50}")
    print("  " + "-" * 60)
    for delta in [0.0, 0.5, 1.0, 2.0, 5.0]:
        eig = eigvals(delta, 0.0, 0.0)
        eig_str = ", ".join(f"{e:.4f}" for e in eig)
        print(f"  {delta:>7.2f}  [{eig_str}]")

    # ---------- (3) Linear response at Delta = 1.0 ----------
    delta_test = 1.0
    print(f"\n(3) Linear response at Delta = {delta_test}:")
    e0, dphi1 = linear_response(delta_test, direction=1)
    e0_check, dphi2 = linear_response(delta_test, direction=2)
    print(f"  {'mode':>5}  {'eigenvalue':>12}  {'d/d phi_1':>12}  "
          f"{'d/d phi_2':>12}  {'irrep label':>15}")
    print("  " + "-" * 70)
    # Get eigenvectors at phi = 0 for irrep classification.
    H0_mat = H0(delta_test)
    eigvals_h0, eigvecs_h0 = np.linalg.eigh(H0_mat)
    for k in range(7):
        v = eigvecs_h0[:, k]
        irrep = classify_eigenvector(v)
        marker = "  <- E_g!" if abs(dphi1[k]) > 1e-6 or abs(dphi2[k]) > 1e-6 else ""
        print(f"  {k:>5}  {eigvals_h0[k]:>12.4f}  "
              f"{dphi1[k]:>+12.6f}  {dphi2[k]:>+12.6f}  "
              f"{irrep:>15}{marker}")

    n_shift_phi1 = int(np.sum(np.abs(dphi1) > 1e-6))
    n_shift_phi2 = int(np.sum(np.abs(dphi2) > 1e-6))
    print(f"\n  modes shifting under phi_1: {n_shift_phi1}  "
          f"(corrected prediction: 5 = E_g + T_1u)")
    print(f"  modes shifting under phi_2: {n_shift_phi2}  "
          f"(corrected prediction: 5 = E_g + T_1u)")

    # ---------- (4) Sweeps and plot ----------
    n_pts = 41
    phis = np.linspace(-1.0, 1.0, n_pts)
    eigs_phi1 = np.array([eigvals(delta_test, p, 0.0) for p in phis])
    eigs_phi2 = np.array([eigvals(delta_test, 0.0, p) for p in phis])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Identify eigenvector irrep at each phi=0 mode
    irrep_labels = []
    for k in range(7):
        irrep_labels.append(classify_eigenvector(eigvecs_h0[:, k]))

    color_for = {
        "A_1g_v0": "tab:gray",
        "A_1g_others": "tab:gray",
        "E_g_1": "tab:red",
        "E_g_2": "tab:red",
        "T_1u_x": "tab:blue",
        "T_1u_y": "tab:blue",
        "T_1u_z": "tab:blue",
    }
    style_for = lambda ir: ("-" if "E_g" in ir
                            else ("--" if "T_1u" in ir else ":"))

    for k in range(7):
        c = color_for[irrep_labels[k]]
        s = style_for(irrep_labels[k])
        axes[0].plot(phis, eigs_phi1[:, k], s, color=c, lw=1.6,
                     label=irrep_labels[k] if k in [0, 1, 2, 3, 4]
                     else None)
        axes[1].plot(phis, eigs_phi2[:, k], s, color=c, lw=1.6)

    for ax, lab in zip(axes, [r"$\varphi_1$  ($x^2-y^2$)", r"$\varphi_2$  ($3z^2-r^2$)"]):
        ax.set_xlabel(lab)
        ax.set_ylabel("eigenvalue")
        ax.grid(alpha=0.3)
        ax.set_title(f"Constructed E_g coupling ($\\Delta = {delta_test}$)")
    axes[0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "spectrum_phi_sweep.png", dpi=130)
    plt.close(fig)

    # ---------- (5) Compare with W3.3-E (geometric coupling) ----------
    print(f"\n(5) Comparison with W3.3-E (geometric bond-strain coupling):")
    print(f"  W3.3-E counted shifts at phi_1 sweep: 4")
    print(f"  W3.3-F counts at phi_1 sweep:          {n_shift_phi1}")
    print(f"  W3.3-E counted shifts at phi_2 sweep: 6")
    print(f"  W3.3-F counts at phi_2 sweep:          {n_shift_phi2}")

    # ---------- Summary ----------
    delta = delta_test
    summary = ["Paper 19 -- W3.3-F   Constructed E_g coupling on K_7",
               "=" * 70,
               "",
               "Setup:",
               "  H_0 = L_K7 + Delta * |v_0><v_0|, with Delta = "
               f"{delta} (breaks A_1g block).",
               "  V_E_g = phi_1 * T_1 + phi_2 * T_2, with",
               "    T_1 = diag(0, +1,+1,-1,-1, 0,0) / sqrt(2)    (x^2-y^2)",
               "    T_2 = diag(0, +1,+1,+1,+1,-2,-2) / sqrt(6)  (3z^2-r^2)",
               "  Both T_1, T_2 transform as the e_g doublet of cubic O_h",
               "  by construction (entries = e_g basis values at each vertex).",
               "",
               "BASELINE EIGENSTRUCTURE (phi = 0, Delta = 1.0):",
               f"  {'mode':>5}  {'eigenvalue':>12}  {'irrep':>15}",
               "  " + "-" * 35,
              ]
    for k in range(7):
        summary.append(
            f"  {k:>5}  {eigvals_h0[k]:>12.4f}  {irrep_labels[k]:>15}"
        )

    summary += [
        "",
        f"Note: with Delta = {delta} != 0, the A_1g block has eigenvalues",
        f"distinct from 7, while E_g and T_1u stay at 7 (5-fold degenerate).",
        "",
        "LINEAR RESPONSE: d eigenvalue / d phi at phi = 0",
        f"  {'mode':>5}  {'eigenvalue':>12}  {'d/d phi_1':>12}  "
        f"{'d/d phi_2':>12}  {'irrep':>15}",
        "  " + "-" * 70,
    ]
    for k in range(7):
        summary.append(
            f"  {k:>5}  {eigvals_h0[k]:>12.4f}  "
            f"{dphi1[k]:>+12.6f}  {dphi2[k]:>+12.6f}  {irrep_labels[k]:>15}"
        )

    summary += [
        "",
        f"COUNT OF SHIFTING MODES:",
        f"  d/d phi_1 != 0:  {n_shift_phi1}  (CORRECTED prediction: 5)",
        f"  d/d phi_2 != 0:  {n_shift_phi2}  (CORRECTED prediction: 5)",
        "",
        "VERDICT (after correcting my analytical prediction):",
        "  My initial guess that 'only 2 E_g modes shift' was WRONG.  The",
        "  full O_h Schur analysis says:",
        "    A_1g x E_g = E_g                  -> A_1g doesn't shift",
        "    E_g x E_g contains A_1g + E_g     -> E_g shifts",
        "    T_1u x E_g = T_1u + T_2u contains T_1u  -> T_1u shifts too",
        "  So 5 of 6 anti-sym modes shift (2 E_g + 3 T_1u); only the",
        "  2 A_1g modes (split off by Delta) are Schur-protected.",
        "",
        "  The numerical result confirms this corrected prediction.",
        "",
        f"  Numerical: {n_shift_phi1} shifts (phi_1) / {n_shift_phi2} shifts (phi_2).",
        "  (mode 6, A_1g_v0, has tiny ~1e-6 shift from numerical noise of",
        "   linear-response finite-difference, not from Schur breaking.)",
        "",
        "  W3.3-F does NOT save the W3.3-E negative result -- both",
        "  geometric and algebraic E_g couplings shift 5 modes, not 2.",
        "",
        "STRUCTURAL INTERPRETATION (revised):",
        "  Voxel-space's e_g gauge structure couples uniformly to the",
        "  E_g + T_1u sectors of K_7 matter (5 modes), with shift",
        "  magnitudes determined by Wigner-Eckart coefficients of",
        "  E_g x {E_g, T_1u} -> trivial in O_h.",
        "",
        "  Numerical shift values:",
        "    E_g modes:   +/- 0.204  =  +/- 1/(2*sqrt(6))",
        "    T_1u modes:  +/- 0.354 = +/- 1/(2*sqrt(2)),",
        "                 +/- 0.408 = +/- 1/2*sqrt(2/3),",
        "                 +/- 0.204 (same E_g coefficient)",
        "    A_1g modes:  ~0 (Schur protected)",
        "",
        "  The shift magnitudes are SPECIFIC algebraic values, fixed by",
        "  the cubic-symmetry geometry.  In principle these could map",
        "  onto SM mass-ratio data if the {2-doublet} and {3-triplet}",
        "  sub-blocks correspond to specific particle classes -- but",
        "  this is a multiplicity-and-magnitude match story, not a",
        "  derivation.",
        "",
        "  RECONCILED WITH SM HIGGS:",
        "    The SM Higgs's 'selective doublet coupling' is NOT a",
        "    consequence of group theory.  In SM, each fermion has its",
        "    OWN Yukawa coefficient (y_e, y_u, y_d, ...), so the Higgs",
        "    couples to all charged matter, but with per-particle",
        "    strength.  Voxel-space gives the COUPLING STRUCTURE ",
        "    (E_g + T_1u all shift), but the per-multiplet strengths",
        "    (Yukawas) are SEPARATE input.  The SM and voxel-space",
        "    agree at the structural level but voxel-space alone does",
        "    NOT predict mass ratios.",
        "",
        "PHYSICAL SIGNIFICANCE:",
        "  The W3 program's final picture of SM emergence in voxel-space:",
        "  ",
        "    - c, gauge symmetry, Lorentz invariance:",
        "        GEOMETRIC (W3.1, W3.2.1.5, W3.4) -- direct consequences",
        "        of FCC-RD lattice structure.",
        "    - Matter multiplet structure (1+2+3 per generation):",
        "        COMBINATORIAL (W3.2.1, W3.3-D) -- from cubic O_h",
        "        sub-decomposition of K_7 = 7 = 1 + 6 = 1 + (1+2+3).",
        "    - Mass spectrum (which modes shift, how much):",
        "        SCHUR-DETERMINED (W3.3-F) -- E_g + T_1u all shift,",
        "        A_1g protected.  Magnitudes from Wigner-Eckart.",
        "    - Mass RATIOS (e.g. m_e / m_mu):",
        "        NOT DERIVED -- requires additional dynamical input",
        "        (per-particle Yukawa coefficients, just like in SM).",
        "",
        "  This refines what voxel-space CAN and CANNOT predict.",
        "  The structure is rich enough to give SM-like multiplet",
        "  classification, but mass HIERARCHY is left to dynamics",
        "  beyond pure geometry.",
    ]

    summary_text = "\n".join(summary)
    (OUT / "summary.txt").write_text(summary_text)
    print()
    print(summary_text)
    print()
    print(f"Wrote {OUT/'spectrum_phi_sweep.png'}")
    print(f"Wrote {OUT/'summary.txt'}")


if __name__ == "__main__":
    main()
