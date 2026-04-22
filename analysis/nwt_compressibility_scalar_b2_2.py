#!/usr/bin/env python3
"""
Paper 15 b2.2 -- Condensate compressibility scalar on S^3/2I as the
mode carrying the lambda_1 = 168 pole.

Setting:
  Abelian-Higgs Lagrangian at BPS (lambda = e^2/2, natural units e=v=1):
      L  =  -|D phi|^2 - (1/4) F_{mu nu}^2 - (1/8)(|phi|^2 - 1)^2.
  Linearise around the VEV  phi = 1 + sigma + i eta  (co-rotating frame).
  The modulus fluctuation sigma (the condensate compressibility /
  "dilaton") satisfies a massive Klein-Gordon equation:
      (Box + m_H^2) sigma = j_sigma(x),
      m_H^2  =  2 lambda v^2  =  v^2  (at BPS, in units where v=1).

On S^3/2I at radius R, the compact spatial Laplacian
  -Delta_{S^3/2I} sigma = (lambda_n / R^2) sigma
with  lambda_n  taking 2I-invariant values  {0, 168, 440, 624, 960, ...}
(from b2.0).  The lowest-eigenvalue nontrivial mode therefore has
dispersion
    omega_1^2  =  m_H^2 + lambda_1 / R^2.

Claim (Volovik-style emergent gravity):  the condensate compressibility
scalar IS the "gravitational response" mode that Paper 15 Sec 7.2 means
when it speaks of the 'graviton propagator'.  The TT tensor graviton
decouples on S^3/2I at this eigenvalue (b2.1a), so the Paper 15 pole
is the sigma pole on the compressibility channel.

What is rigorous so far:
  * Scalar Laplacian spectrum on S^3/2I (b2.0).
  * Identification of sigma as the dominant gravity-response mode in
    Volovik-induced-gravity analog systems (standard result).
  * The pole structure of the sigma propagator at omega^2 = m_H^2 + 168/R^2.

What is still MISSING for a full Paper 15 Sec 7.2 derivation:
  * The CROSSING MECHANISM that turns the geometric integer 168 into
    the small number alpha^21 in the effective coupling.  The scalar
    pole alone is O(1) in alpha -- there is no alpha suppression from
    the Klein-Gordon equation itself.  Paper 13's alpha-per-crossing
    rule must enter via a self-energy / transition-amplitude diagram
    with 21 crossings (b2.1c structural observation), but that diagram
    has not been constructed in the NWT literature.

This script computes what we can close rigorously now (dispersion,
mass gap, 13-dim multiplicity at lambda_1) and records what remains
open for b2.3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nwt_poincare_sphere_b2_0 import multiplicity_trivial_A as m_2I


def scalar_spectrum_on_S3_2I(n_max: int = 40):
    """Return list of (n, lambda_n, multiplicity) for 2I-invariant
    scalar modes up to n_max."""
    modes = []
    for n in range(n_max + 1):
        j = n / 2.0
        mult = int(round((n + 1) * m_2I(j)))
        if mult > 0:
            modes.append((n, n * (n + 2), mult))
    return modes


def dispersion(lam: int, R: float = 1.0, m_H_sq: float = 1.0) -> float:
    """omega^2 = m_H^2 + lambda / R^2  (natural units, e=v=1)."""
    return m_H_sq + lam / (R ** 2)


def main():
    print("=" * 72)
    print(" b2.2 -- Compressibility scalar on S^3/2I carries lambda_1 pole")
    print("=" * 72)

    modes = scalar_spectrum_on_S3_2I(40)

    print()
    print(" Scalar 2I-invariant modes and their BPS dispersion on S^3/2I")
    print(" (m_H = v = 1 units; radius R parameterizes the compact scale)")
    print()
    print("   n    lambda_n   mult   omega^2 (R=1)   omega^2 (R=xi_cond)")
    print("  " + "-" * 60)
    for n, lam, mult in modes[:8]:
        w2_R1 = dispersion(lam, R=1.0)
        # xi_condensate = 1 in natural units; take R = 1 as canonical.
        tag = "  <-- Paper 15 lambda_1 pole" if n == 12 else ""
        print(f"  {n:3d}  {lam:6d}   {mult:3d}       {w2_R1:9.3f}"
              f"{tag}")

    # --- Interpretation ------------------------------------------------
    print()
    print(" The sigma propagator on S^3/2I has the lowest nontrivial pole at")
    print(f"    omega^2_1 = m_H^2 + 168/R^2   (13-fold degenerate, spin-6 of L)")
    print()
    print(" Decomposition of the 13 lambda_1-modes under I = A_5:")
    print("     13 = 1 + 3 + 4 + 5   (trivial + vector + tetrahedral + pentagonal)")
    print()
    print(" Volovik emergent-gravity reading: sigma is the condensate")
    print(" compressibility scalar.  Its long-wavelength dynamics IS the")
    print(" emergent 'gravitational response'; no separate TT graviton is")
    print(" needed for the lambda_1 pole.  Paper 15 Sec 7.2 should read")
    print(" 'compressibility-scalar propagator pole at lambda_1' rather than")
    print(" 'graviton propagator pole'.")

    # --- What this buys us -- and what it doesn't ---------------------
    print()
    print(" What IS derived (now, rigorously):")
    print("   - lambda_1 = 168 on S^3/2I is the correct spectral gap of")
    print("     the sigma propagator.")
    print("   - The 13-fold degeneracy splits under A_5 as 1+3+4+5.")
    print("   - The TT-graviton sector is empty at this eigenvalue,")
    print("     so the identification of 'graviton mode' with sigma is")
    print("     forced.")
    print()
    print(" What is NOT derived (open for b2.3):")
    print("   - Why alpha^21, not some other alpha^N, enters the effective")
    print("     coupling.  The sigma-propagator pole is O(alpha^0); the")
    print("     alpha^21 must come from a MATTER LOOP diagram with 21")
    print("     crossings (Paper 13 rule, b2.1c structural hint).")
    print("   - The explicit Feynman-like diagram construction realising")
    print("     21 crossings as a PSL(2,7)- or Fano-equivariant structure")
    print("     on the Heegaard torus of S^3/2I remains conjectural.")
    print()
    print(" A minimum-viable next step (b2.3):")
    print("   Compute the one-loop sigma self-energy on S^3/2I with a")
    print("   matter-field loop (scalar or fermion) winding around the")
    print("   BPS vortex configuration.  Count AB phase crossings per")
    print("   winding; check whether 21 emerges as a total crossing count")
    print("   for the naturally 2I-equivariant diagram.")

    # --- Plot: the dispersion relation on S^3/2I ----------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ns = [m[0] for m in modes]
    w2s = [dispersion(m[1], R=1.0) for m in modes]
    mults = [m[2] for m in modes]
    ax.scatter(ns, w2s, s=[40 + 5 * mult for mult in mults],
               color="tab:blue", alpha=0.8)
    for n, w2, mult in zip(ns, w2s, mults):
        if n == 12:
            ax.annotate(rf"$\lambda_1$ pole  $\omega^2 = {w2:.0f}$",
                        (n, w2), xytext=(n + 1, w2 + 80),
                        arrowprops={"arrowstyle": "->", "color": "red"},
                        fontsize=11, color="red")
        elif mult > 0:
            ax.annotate(f" dim={mult}", (n, w2), fontsize=8)
    ax.axhline(1, color="gray", ls="--", alpha=0.6)
    ax.text(ns[-1] - 5, 5, r"$m_H^2 = 1$ (BPS Higgs mass)",
            fontsize=9, color="gray")
    ax.set_xlabel(r"level  $n$  (scalar 2I-invariant modes only)")
    ax.set_ylabel(r"$\omega^2 = m_H^2 + \lambda_n / R^2$  (R=1)")
    ax.set_title(r"Compressibility scalar $\sigma$ dispersion on $S^3/2I$")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    out = Path(__file__).parent / "nwt_compressibility_scalar_b2_2.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print()
    print(f" Plot: {out}")


if __name__ == "__main__":
    main()
