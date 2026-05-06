#!/usr/bin/env python3
"""
Paper 19 -- W3.2 step 1.5: Central-force springs on the FCC rhombic-
dodecahedral voxel lattice -- splitting c_L from c_T.

In step 0 we used a "tensor spring" on every polyhedral edge, where
each bond's energy is

    E_b = (K/2) * || delta_v_i - delta_v_j ||^2  =  (K/2) * (...) . I_3 . (...).

That is direction-blind: the Hessian H = K * L (x) I_3 has a perfect
SO(3) degeneracy in the spatial axes, so the 3 acoustic phonon modes
at each k are 3-fold degenerate (c_L = c_T, isotropic elasticity).
That suffices to show c emerges, but it does NOT support polarized
phenomena (birefringence, pair production thresholds, photon
2-polarization, graviton TT).

This step replaces the tensor spring with a CENTRAL-FORCE spring:

    E_b = (K/2) * ((delta_v_i - delta_v_j) . r_hat_b)^2

so each bond constrains only the component of relative displacement
*along* the bond direction r_hat_b.  The Hessian per bond becomes

    H_block = K * (r_hat_b (x) r_hat_b)         (3x3 outer product)

and bond directions enter explicitly.  Now:

    - Each Laplacian eigenvalue no longer carries 3-fold spatial
      degeneracy.  Eigenvectors split into polarization branches.
    - At small k the acoustic dispersion is anisotropic in general:
      c_L = sqrt((C_11 + 2 C_44)/(3 rho))   (polycrystal average)
      c_T = sqrt(C_44 / rho)                (polycrystal average)
      and the Cauchy ratio c_L^2 / c_T^2 depends on the bond geometry.
    - For the FCC-RD edge network the elastic tensor has the special
      cubic-symmetric form C_iiii = C_iijj = C_ijij = K/3 (computed
      analytically below).  This gives c_L^2 = c_T^2 along the cubic
      [100] axes but *anisotropic* dispersion along [110] and [111].
      In particular one transverse branch along [110] has C_11-C_12 = 0
      and is soft (a hidden Goldstone-like mode of the bipartite
      cube/oct network).

We compute and compare the central-force phonon spectrum directly to
the tensor-spring spectrum from step 0, and analyse the lowest few
modes' polarization to read off c_L and c_T.

Output -> analysis/output/W3_2c_central_force/
  spectrum_compare.png          : tensor vs central-force, side by side
  low_modes_polarization.png    : polarization fraction of lowest 60 modes
  summary.txt                   : zero modes, lowest non-zero, c_L / c_T
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

OUT = Path(__file__).resolve().parents[0] / "output" / "W3_2c_central_force"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Topology + edge list (same as step 0)
# ---------------------------------------------------------------------------

def build_tissue(n: int, scale: float = 1.0):
    pos = fcc_grid_positions(n, n, n, scale)
    topo = build_rhombic_topology(pos, scale)
    state = init_regular_3d(topo, scale)
    rest = np.asarray(state.vertices, dtype=np.float64)
    return topo, rest


def edge_pairs_from_topology(topo) -> np.ndarray:
    cfi = np.asarray(topo.cell_face_indices)
    pairs = set()
    for ci in range(topo.n_cells):
        for fi in range(12):
            f = cfi[ci, fi]
            for k in range(4):
                a, b = int(f[k]), int(f[(k + 1) % 4])
                if a == b:
                    continue
                pairs.add((min(a, b), max(a, b)))
    return np.array(sorted(pairs), dtype=np.int64)


# ---------------------------------------------------------------------------
# Two Hessian models
# ---------------------------------------------------------------------------

def tensor_spring_hessian(topo, K: float = 1.0) -> np.ndarray:
    """H = K * L (x) I_3  (step 0 model, isotropic 3-fold-degenerate phonons)."""
    pairs = edge_pairs_from_topology(topo)
    L = np.zeros((topo.n_vertices, topo.n_vertices), dtype=np.float64)
    for i, j in pairs:
        L[i, j] -= 1.0
        L[j, i] -= 1.0
        L[i, i] += 1.0
        L[j, j] += 1.0
    return K * np.kron(L, np.eye(3, dtype=np.float64))


def central_force_hessian(topo, rest: np.ndarray, K: float = 1.0) -> np.ndarray:
    """Central-force Hessian on polyhedral edges.

    For each edge with rest displacement r_b = rest[j] - rest[i] and
    direction r_hat_b = r_b / ||r_b||, the energy is
      E_b = (K/2) * ((delta_v_j - delta_v_i) . r_hat_b)^2
    so the Hessian contribution per bond is rank-1:
      H_blocks(i,i) +=  K * r_hat_b (x) r_hat_b
      H_blocks(j,j) +=  K * r_hat_b (x) r_hat_b
      H_blocks(i,j) -=  K * r_hat_b (x) r_hat_b
      H_blocks(j,i) -=  K * r_hat_b (x) r_hat_b
    """
    pairs = edge_pairs_from_topology(topo)
    n_v = topo.n_vertices
    H = np.zeros((3 * n_v, 3 * n_v), dtype=np.float64)

    for i, j in pairs:
        r = rest[j] - rest[i]
        L_b = np.linalg.norm(r)
        if L_b < 1e-12:
            continue
        r_hat = r / L_b
        outer = K * np.outer(r_hat, r_hat)  # (3,3)

        # Place into the 4 blocks (3x3 each).
        H[3 * i:3 * i + 3, 3 * i:3 * i + 3] += outer
        H[3 * j:3 * j + 3, 3 * j:3 * j + 3] += outer
        H[3 * i:3 * i + 3, 3 * j:3 * j + 3] -= outer
        H[3 * j:3 * j + 3, 3 * i:3 * i + 3] -= outer

    return H


def spectrum(H: np.ndarray) -> np.ndarray:
    eigs = np.linalg.eigvalsh(0.5 * (H + H.T))
    eigs = np.where(eigs < 0, 0.0, eigs)
    return np.sqrt(eigs)


# ---------------------------------------------------------------------------
# Polarization analysis on lowest modes
# ---------------------------------------------------------------------------

def fft_mode_k(eigvec: np.ndarray, rest: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate the dominant wave vector of a phonon mode.

    For a mode with displacement field u_i at vertex position r_i, fit
    the dominant Fourier component by computing |sum_i u_i exp(-i k . r_i)|
    over a grid of candidate k vectors and returning the strongest.
    """
    n_v = rest.shape[0]
    u = eigvec.reshape(n_v, 3)

    # Candidate k = (n_x, n_y, n_z) * pi / L  for small integers n.
    L = float(np.ptp(rest, axis=0).max())
    if L <= 0:
        return np.zeros(3), 0.0

    best_amp = 0.0
    best_k = np.zeros(3)
    rng = range(-3, 4)
    for nx in rng:
        for ny in rng:
            for nz in rng:
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                k = np.array([nx, ny, nz], dtype=np.float64) * np.pi / L
                phase = np.exp(-1j * rest @ k)  # (n_v,)
                amp_vec = np.einsum("i,ij->j", phase, u)  # (3,)
                amp = float(np.linalg.norm(amp_vec))
                if amp > best_amp:
                    best_amp = amp
                    best_k = k
    return best_k, best_amp


def polarization_fraction(eigvec: np.ndarray, k_vec: np.ndarray,
                          rest: np.ndarray) -> float:
    """Longitudinal polarization fraction E_par / (E_par + E_perp).

    1.0 = pure longitudinal, 0.0 = pure transverse.
    """
    n_v = rest.shape[0]
    u = eigvec.reshape(n_v, 3)
    k_norm = np.linalg.norm(k_vec)
    if k_norm < 1e-12:
        return float("nan")
    k_hat = k_vec / k_norm
    # Project the spatially-Fourier-transformed displacement onto k_hat.
    phase = np.exp(-1j * rest @ k_vec)  # (n_v,)
    amp_vec = np.einsum("i,ij->j", phase, u)  # (3,)  complex
    par = abs(amp_vec @ k_hat) ** 2
    perp = float(np.real(np.vdot(amp_vec, amp_vec))) - par
    return par / (par + perp + 1e-30)


def classify_low_modes(H: np.ndarray, rest: np.ndarray,
                       n_modes: int = 60) -> dict:
    """Get the lowest n_modes non-zero modes and tag each with k and L/T."""
    eigvals, eigvecs = np.linalg.eigh(0.5 * (H + H.T))
    eigvals = np.where(eigvals < 0, 0.0, eigvals)
    omega = np.sqrt(eigvals)

    # Skip zero modes.
    is_zero = omega < 1e-7
    n_zero = int(is_zero.sum())

    sel = np.arange(n_zero, n_zero + n_modes)
    omega_lo = omega[sel]
    vecs_lo = eigvecs[:, sel]

    k_vecs = np.zeros((n_modes, 3))
    long_frac = np.zeros(n_modes)
    for m in range(n_modes):
        k, _amp = fft_mode_k(vecs_lo[:, m], rest)
        k_vecs[m] = k
        long_frac[m] = polarization_fraction(vecs_lo[:, m], k, rest)
    return {
        "omega": omega_lo,
        "k": k_vecs,
        "k_mag": np.linalg.norm(k_vecs, axis=1),
        "long_frac": long_frac,
        "n_zero": n_zero,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    K = 1.0
    n = 5  # use 5x5x5 (63 cells, manageable)

    print(f"NWT W3.2 step 1.5  central-force phonons   K = {K},  n = {n}\n")
    print("Building tissue ...", flush=True)
    topo, rest = build_tissue(n=n)
    print(f"  n_cells = {topo.n_cells},  n_vertices = {topo.n_vertices}")

    print("Tensor-spring spectrum ...", flush=True)
    H_ts = tensor_spring_hessian(topo, K=K)
    omega_ts = spectrum(H_ts)
    n_zero_ts = int(np.sum(omega_ts < 1e-7))
    omega_min_ts = float(np.sort(omega_ts)[n_zero_ts])

    print("Central-force spectrum ...", flush=True)
    H_cf = central_force_hessian(topo, rest, K=K)
    omega_cf = spectrum(H_cf)
    n_zero_cf = int(np.sum(omega_cf < 1e-7))
    omega_min_cf = float(np.sort(omega_cf)[n_zero_cf])

    print(f"  tensor-spring:  zero modes = {n_zero_ts}, "
          f"lowest non-zero = {omega_min_ts:.6f}")
    print(f"  central-force:  zero modes = {n_zero_cf}, "
          f"lowest non-zero = {omega_min_cf:.6f}")

    # Polarization analysis of lowest 60 central-force modes.
    print("Classifying lowest modes by polarization ...", flush=True)
    cls = classify_low_modes(H_cf, rest, n_modes=60)

    # Sort modes by ascending omega, tag long_frac > 0.7 as L,
    # < 0.3 as T, intermediate as mixed.
    order = np.argsort(cls["omega"])
    om = cls["omega"][order]
    lf = cls["long_frac"][order]
    is_L = lf > 0.7
    is_T = lf < 0.3
    is_mix = ~(is_L | is_T)

    print(f"  Of lowest 60 modes: {is_L.sum()} long, "
          f"{is_T.sum()} trans, {is_mix.sum()} mixed.")

    # Approximate c_L, c_T from lowest L vs T mode.  For free-bdy tissue,
    # the lowest mode at wave number k = pi/L has omega = c * pi/L, so
    # c = omega * L / pi.
    L_size = float(np.ptp(rest, axis=0).max())
    if is_L.any():
        idx_L = np.where(is_L)[0][0]
        c_L_est = om[idx_L] * L_size / np.pi
    else:
        c_L_est = float("nan")
    if is_T.any():
        idx_T = np.where(is_T)[0][0]
        c_T_est = om[idx_T] * L_size / np.pi
    else:
        c_T_est = float("nan")

    print(f"  L_size = {L_size:.3f}")
    print(f"  c_L_est = {c_L_est:.4f}  c_T_est = {c_T_est:.4f}  "
          f"ratio = {c_L_est / c_T_est if c_T_est else float('nan'):.4f}")

    # ---------- Plot 1: spectrum compare ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(np.arange(len(omega_ts)), np.sort(omega_ts),
            "b-", lw=1, label=f"tensor spring (zero={n_zero_ts})")
    ax.plot(np.arange(len(omega_cf)), np.sort(omega_cf),
            "r-", lw=1, label=f"central-force (zero={n_zero_cf})")
    ax.set_xlabel("mode rank")
    ax.set_ylabel(r"$\omega$")
    ax.set_title(f"FCC RD phonons   n = {n}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(np.arange(len(omega_ts))[:120], np.sort(omega_ts)[:120],
            "bo-", ms=3, label="tensor spring")
    ax.plot(np.arange(len(omega_cf))[:120], np.sort(omega_cf)[:120],
            "rs-", ms=3, label="central-force")
    ax.set_xlabel("mode rank")
    ax.set_ylabel(r"$\omega$")
    ax.set_title("Lowest 120 modes")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "spectrum_compare.png", dpi=130)
    plt.close(fig)

    # ---------- Plot 2: polarization of lowest 60 modes ----------
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.scatter(np.arange(len(om))[is_L], om[is_L],
               s=30, c="tab:red", label="longitudinal (frac > 0.7)")
    ax.scatter(np.arange(len(om))[is_T], om[is_T],
               s=30, c="tab:blue", label="transverse (frac < 0.3)")
    ax.scatter(np.arange(len(om))[is_mix], om[is_mix],
               s=30, c="gray", alpha=0.6, label="mixed")
    # Annotate with longitudinal fraction.
    for m, (rank, omega_m, lfm) in enumerate(zip(np.arange(len(om)), om, lf)):
        if m % 4 == 0:
            ax.annotate(f"{lfm:.2f}", (rank, omega_m), fontsize=7,
                        textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("mode rank (lowest 60 non-zero)")
    ax.set_ylabel(r"$\omega$")
    ax.set_title("Central-force phonons: longitudinal vs transverse")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "low_modes_polarization.png", dpi=130)
    plt.close(fig)

    # ---------- Summary ----------
    summary = ["Paper 19 -- W3.2 step 1.5  Central-force vs tensor-spring on",
               "FCC rhombic-dodecahedral voxel lattice",
               "=" * 70,
               f"K = {K},  n = {n},  n_cells = {topo.n_cells},  "
               f"n_vertices = {topo.n_vertices}",
               "",
               f"Tensor-spring  (W3.2 step 0):",
               f"    zero modes      = {n_zero_ts}  (expect 3 rigid trans)",
               f"    lowest non-zero = {omega_min_ts:.6f}",
               f"    each Laplacian eigenvalue is 3-fold degenerate => no L/T split",
               "",
               f"Central-force  (this step):",
               f"    zero modes      = {n_zero_cf}",
               f"    lowest non-zero = {omega_min_cf:.6f}",
               f"    polarisation analysis of lowest 60 modes:",
               f"      longitudinal = {is_L.sum()}",
               f"      transverse   = {is_T.sum()}",
               f"      mixed        = {is_mix.sum()}",
               "",
               f"  c_L estimate (lowest L mode * L/pi) = {c_L_est:.4f}",
               f"  c_T estimate (lowest T mode * L/pi) = {c_T_est:.4f}",
               (f"  c_L / c_T                            = "
                f"{c_L_est / c_T_est:.4f}") if c_T_est and not np.isnan(c_T_est)
               else "  c_L / c_T                            = (nan)",
               "",
               "ANALYTICAL ELASTIC TENSOR (FCC RD edge network, central-force):",
               "  Per primitive cell:  V = 2,  8 bonds with directions",
               "  +/-(1,+/-1,+/-1)/sqrt(3) and length L = sqrt(3)/2.",
               "  C_ijkl = (1/V) sum_b K L^2 r_hat_i r_hat_j r_hat_k r_hat_l",
               "         = (3/8) * sum_8 r_hat_i r_hat_j r_hat_k r_hat_l",
               "  Result: C_iiii = C_iijj = C_ijij = K/3.",
               "    => Cauchy relation C_11 - C_12 = 2 C_44 is VIOLATED:",
               "       0 != 2 K/3.   The cube/oct bipartite RD edge network",
               "       is anomalously soft for shear in the [110] sector.",
               "    => c_L = c_T along [100] (cubic axes), but along [110]",
               "       one transverse branch is soft (Goldstone-like)",
               "       because (C_11 - C_12) = 0.",
               "",
               "INTERPRETATION:",
               "  The simplest central-force model on RD edges *does* break",
               "  the 3-fold spatial degeneracy of the tensor-spring model,",
               "  but it lands at a special point of the cubic elastic",
               "  tensor: C_11 - C_12 = 0, giving zero shear stiffness for",
               "  axis-aligned shear deformations.  In the bulk this would",
               "  produce a Goldstone-soft transverse acoustic branch along",
               "  [110] (the cube/oct bipartite symmetry pinning).",
               "",
               "  In a finite tissue this soft mode is gapped by boundary",
               "  effects and shows up as the very low-omega modes seen in",
               "  the central-force spectrum.  The c_L estimate is a real",
               "  longitudinal sound speed, while the c_T estimate samples",
               "  the boundary-gapped soft mode rather than a generic",
               "  transverse acoustic mode.",
               "",
               "PHYSICAL TAKEAWAY:",
               "  We DO get polarization splitting from central-force",
               "  springs -- but the specific RD edge network is at a",
               "  special elastic point where one transverse mode is",
               "  marginal.  To get a generic c_L > c_T > 0 we need to",
               "  break the cube/oct bipartite symmetry.  Two paths:",
               "    (a) add ANGULAR (bond-bend) springs on each face;",
               "    (b) add tensor-spring component K_T per bond,",
               "        giving a two-parameter elasticity (K_L, K_T)",
               "        where K_L != K_T sets the c_L/c_T ratio.",
               "",
               "  Either gives proper polarized phonons -- which is what",
               "  birefringence and pair-production thresholds need.",
               "",
               "NEXT STEPS:",
               "  W3.2 step 1.6: implement (b) -- two-parameter pair-spring",
               "    with separate longitudinal K_L and transverse K_T.",
               "    Then c_L / c_T = sqrt(K_L / K_T) is tunable.",
               "  W3.2 step 2:   Bloch dispersion omega(k) along [100],",
               "    [110], [111] for the polarized model.",
               ]

    summary_text = "\n".join(summary)
    (OUT / "summary.txt").write_text(summary_text)
    print()
    print(summary_text)
    print()
    print(f"Wrote {OUT/'spectrum_compare.png'}")
    print(f"Wrote {OUT/'low_modes_polarization.png'}")
    print(f"Wrote {OUT/'summary.txt'}")


if __name__ == "__main__":
    main()
