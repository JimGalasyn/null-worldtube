#!/usr/bin/env python3
"""
NWT Hopf Link Component Analysis

Key physical argument (Jim Galasyn, 2026-03-29):
  Coprimality gcd(p_eff, q_eff) = 1 is required for a SINGLE continuous
  vortex to form a proper torus knot.  But Hopf(n) carriers are n SEPARATE
  vortex rings, linked but individually unknotted.  The coprimality condition
  applies per-component, not to the composite winding.

  For Hopf(n) carrier (n,n):
    - n separate rings, each following a (1,1) path on the torus
    - Mode (p_m, q_m) on each ring → per-component eff = (p_m, q_m)
    - Per-component coprimality: gcd(p_m, q_m) = 1 (already enforced)
    - Composite eff = (n·p_m, n·q_m), gcd = n·gcd(p_m,q_m) = n > 1
      → but this is fine because the composite ISN'T a single vortex

  This analysis tests two mass models:
    Model A: Mass from composite winding (current Paper 8 approach)
    Model B: Mass from per-component winding × linking energy

  The n_q = 2 convention for Hopf links also needs revisiting:
    - Hopf(2): 2 rings, linking number 1 per pair → 2 crossings
    - Hopf(3): 3 rings, 3 pairwise linkings → 6 crossings? Or still 2?
    - Physically: n_q should count the number of COMPONENTS (quarks)
"""

import numpy as np
from math import gcd

KAPPA = np.pi**2
ME = 0.511
BETA_E = np.sqrt(5.0/4.0)

def beta_fast(p, m):
    val = m**2 / p**2 - 1
    return np.sqrt(val) if val > 0 else None

def mass_single(p, q, m_int, n_q=0):
    """Mass for a single vortex component."""
    beta = beta_fast(p, m_int)
    if beta is None or beta <= 1e-6: return None
    pq = p**2 + q**2
    lr = np.log(8*beta) / np.log(8*BETA_E)
    if lr <= 0: return None
    geom = (pq/5.0) * (beta/BETA_E) * lr
    try:
        enh = float(n_q**q) if n_q > 0 and q > 0 else 1.0
    except (OverflowError, ValueError):
        return None
    if not np.isfinite(enh) or enh > 1e30: return None
    mass = geom * enh * ME
    return mass if np.isfinite(mass) and mass > 0 else None


# ── Model comparison ──────────────────────────────────────────────────
print("=" * 110)
print("HOPF LINK COMPONENT ANALYSIS")
print("=" * 110)

print("""
PHYSICAL ARGUMENT:
  A Hopf(n) carrier consists of n separate vortex rings, topologically
  linked but individually unknotted.  Each ring is a (1,1) unknot on the
  embedding torus, positioned at equal angular intervals.

  When mode (p_m, q_m) propagates on this carrier:
    - Each ring independently carries mode (p_m, q_m)
    - Per-ring coprimality: gcd(p_m, q_m) = 1 ✓ (always, by mode def)
    - The linking provides confinement (n_q = 2 for quark-antiquark)

  The composite effective winding (n·p_m, n·q_m) describes the TOTAL
  pattern of all n rings, not a single continuous vortex.  Coprimality
  of the composite is irrelevant because no single vortex line traces it.

  QUESTION: How should we compute mass?
    Model A: Use composite (n·p_m, n·q_m) in mass formula (current)
    Model B: Use per-component (p_m, q_m) × n components × linking energy
""")

# ── Test existing mesons with both models ─────────────────────────────
print("=" * 110)
print("TEST: Existing meson assignments — composite vs per-component")
print("=" * 110)

# From Paper 8 Table 1
mesons = [
    # (name, mass_exp, mode, carrier, m, Paper8 mass)
    ('π⁰',    135.0,  (3,1), (6,3), 22, 134.9),    # wait, carrier (6,3) isn't Hopf
    ('π±',    139.57, (3,1), (4,2), 27, 138.4),     # (4,2) isn't Hopf either
    ('η',     547.86, (2,1), (5,5), 17, 550.1),     # Hopf(5)!
    ('ρ',     775.26, (3,1), (5,5), 22, 769.8),     # Hopf(5)
    ('ω',     782.66, (1,1), (7,7),  9, 789.3),     # Hopf(7)
    ('K*',    891.67, (1,1), (5,5), 20, 887.7),     # Hopf(5)
    ("η'",    957.78, (3,1), (5,5), 24, 959.6),     # Hopf(5)
    ('φ',    1019.46, (1,1), (5,5), 22, 1011.0),    # Hopf(5)
    ('D⁰',  1864.8,  (1,1), (5,5), 35, 1856.9),    # Hopf(5)
    ('J/ψ',  3096.9,  (1,1), (7,7), 16, 3011.7),   # Hopf(7)
    ('B±',   5279.3,  (1,1), (7,7), 23, 5275.0),    # Hopf(7)
    ('Υ',    9460.3,  (1,1), (7,7), 35, 9405.4),    # Hopf(7)
]

print(f"\n  {'Name':>6s} {'m_exp':>8s} {'mode':>7s} {'carrier':>9s} "
      f"{'m':>3s} {'Composite':>12s} {'PerComp':>12s} {'m_exp':>8s}")
print(f"  {'─'*80}")

for name, mexp, (pm,qm), (pc,qc), m_int, p8mass in mesons:
    if pc != qc:
        # Not a Hopf carrier — skip per-component
        m_comp = mass_single(pm*pc, qm*qc, m_int, n_q=2)
        print(f"  {name:>6s} {mexp:8.1f} ({pm},{qm}) ({pc},{qc}) "
              f"{m_int:3d} {m_comp or 0:12.1f} {'(not Hopf)':>12s} {mexp:8.1f}")
        continue

    n = pc  # Hopf(n) carrier
    # Model A: composite winding
    pe, qe = pm*n, qm*n
    m_composite = mass_single(pe, qe, m_int, n_q=2)

    # Model B: per-component
    # Each component sees mode (pm, qm) on a (1,1) unknot
    # But the component ALSO knows it's linked → n_q applies
    # Total mass = n × mass_per_component? Or just mass_per_component with n_q?
    m_percomp = mass_single(pm, qm, m_int, n_q=2)

    # Model C: per-component winding, but n as a multiplier
    m_npercomp = n * mass_single(pm, qm, m_int, n_q=0) if mass_single(pm, qm, m_int, n_q=0) else None

    c_str = f"{m_composite:.1f}" if m_composite else "None"
    p_str = f"{m_percomp:.1f}" if m_percomp else "None"

    print(f"  {name:>6s} {mexp:8.1f} ({pm},{qm}) ({pc},{qc}) "
          f"{m_int:3d} {c_str:>12s} {p_str:>12s} {mexp:8.1f}")


# ── Per-component model for EWK bosons ────────────────────────────────
print("\n\n" + "=" * 110)
print("EWK BOSONS: Per-component model on Hopf carriers")
print("=" * 110)

print("""
  If coprimality applies per-component, then we check:
    gcd(p_mode, q_mode) = 1 → always true by construction
    gcd(n_q_per_component, q_mode) → confinement per component

  For Hopf(n):
    n_q_per_component = ? (each ring sees the other n-1 rings linking through it)

  Three sub-models for n_q:
    (i)   n_q = 2 always (Hopf link convention, regardless of n)
    (ii)  n_q = n (number of components)
    (iii) n_q = n-1 (number of OTHER components linking through each ring)
""")

print("  Scanning Hopf(2), Hopf(3), Hopf(5) for W/Z/H matches...\n")

EWK = [('W±', 80379.0, 1.0), ('Z⁰', 91188.0, 1.0), ('H⁰', 125100.0, 0.0)]

for n in [2, 3, 5]:
    print(f"\n  {'━'*100}")
    print(f"  Hopf({n}) carrier ({n},{n})")
    print(f"  {'━'*100}")

    for nq_model, nq_label in [(2, "n_q=2 (standard)"), (n, f"n_q={n} (components)"),
                                (n-1, f"n_q={n-1} (others)")]:
        print(f"\n    {nq_label}:")

        for bname, bmass, bspin in EWK:
            best = None
            best_err = 1.0

            for pm in range(1, 8):
                for qm in range(1, 8):
                    if gcd(pm, qm) != 1: continue

                    # Confinement per-component
                    if nq_model > 0 and gcd(nq_model, qm) != 1: continue

                    # Spin check: J_mode from p_mode parity
                    j_mode = 0.5 if pm % 2 == 1 else 0.0  # odd p → half-integer
                    # Actually let's try both conventions
                    j_mode_a = 0.5  # Paper 8 assumption
                    j_carrier = (nq_model - 1) / 2.0 if nq_model > 0 else 0
                    j_min = abs(j_mode_a - j_carrier)
                    j_max = j_mode_a + j_carrier
                    spins = [j_min + i for i in range(int(j_max - j_min) + 1)]

                    # Mass from COMPOSITE winding (current formula)
                    pe, qe = pm*n, qm*n
                    for m in range(2, 60):
                        mass = mass_single(pe, qe, m, n_q=nq_model)
                        if mass is None: continue
                        err = abs(mass - bmass) / bmass
                        if err < best_err:
                            best_err = err
                            best = {
                                'mode': (pm, qm), 'eff': (pe, qe), 'm': m,
                                'nq': nq_model, 'mass': mass, 'err': err,
                                'spins': spins, 'spin_ok': bspin in spins,
                                'conf': gcd(nq_model, qm) if nq_model > 0 else 0,
                            }

            if best and best['err'] < 0.05:
                pm, qm = best['mode']
                pe, qe = best['eff']
                sp = ','.join(f'{s}' for s in best['spins'])
                sok = '✓' if best['spin_ok'] else '✗'
                print(f"      {bname}: mode({pm},{qm}) eff({pe},{qe}) m={best['m']:2d} "
                      f"n_q={best['nq']} mass={best['mass']:8.0f} err={best['err']*100:.2f}% "
                      f"J=[{sp}] {sok}")
            else:
                print(f"      {bname}: no match within 5%")


# ── The deeper question: what IS n_q for a multi-component link? ──────
print("\n\n" + "=" * 110)
print("THE LINK CROSSING NUMBER QUESTION")
print("=" * 110)

print("""
  Standard crossing numbers for links:
    Hopf link (2 rings):     2 crossings
    Borromean rings (3):     6 crossings (but 0 pairwise linking!)
    3-component Hopf:        6 crossings (3 pairs × 2 crossings each)
    n-component Hopf:        n(n-1) crossings

  But Paper 8 uses n_q = 2 for ALL Hopf carriers regardless of n.

  This is physically motivated if n_q counts QUARK-ANTIQUARK pairs
  (always 1 pair = 2 quarks for mesons), not raw crossings.

  For the EWK extension, the choice matters:
    Hopf(3) with n_q = 2:  n_q^q enhancement moderate (2^q)
    Hopf(3) with n_q = 3:  n_q^q enhancement larger (3^q)
    Hopf(3) with n_q = 6:  n_q^q enhancement huge (6^q)
""")

# Test n_q = n (components) for Hopf(3) hitting W/Z/H
print("  Hopf(3) with n_q = 3 (one per component):")
print("  Confinement requires gcd(3, q_mode) = 1, i.e., q_mode not divisible by 3\n")

for bname, bmass, bspin in EWK:
    results = []
    for pm in range(1, 8):
        for qm in range(1, 8):
            if gcd(pm, qm) != 1: continue
            if gcd(3, qm) != 1: continue  # confinement
            pe, qe = pm*3, qm*3
            for m in range(2, 60):
                mass = mass_single(pe, qe, m, n_q=3)
                if mass is None: continue
                err = abs(mass - bmass) / bmass
                if err < 0.03:
                    j_c = 1.0  # (3-1)/2
                    spins = [0.5, 1.5]
                    results.append((err, pm, qm, pe, qe, m, mass, spins, bspin in spins))
    results.sort()
    for err, pm, qm, pe, qe, m_int, mass, spins, sok in results[:5]:
        sp = ','.join(f'{s}' for s in spins)
        print(f"    {bname}: mode({pm},{qm}) eff({pe},{qe}) m={m_int:2d} "
              f"mass={mass:8.0f} err={err*100:.2f}% J=[{sp}] {'✓' if sok else '✗'}")
    if not results:
        print(f"    {bname}: no matches")

# ── What about n_q = n for spin? ─────────────────────────────────────
print("\n\n" + "=" * 110)
print("SPIN WITH n_q = n (components)")
print("=" * 110)

print("""
  If n_q = n (number of Hopf components), then J_carrier = (n-1)/2:

    Hopf(2): n_q=2, J_carrier=1/2, J=[0, 1]     ← mesons (correct!)
    Hopf(3): n_q=3, J_carrier=1,   J=[1/2, 3/2]  ← baryon-like spins
    Hopf(4): n_q=4, J_carrier=3/2, J=[1, 2]       ← tetraquark-like
    Hopf(5): n_q=5, J_carrier=2,   J=[3/2, 5/2]   ← pentaquark-like

  For W (J=1) and Z (J=1):
    Need 1 in the spin multiplet → Hopf(2) gives [0,1] ✓ or Hopf(4) gives [1,2] ✓
    Hopf(3) gives [1/2, 3/2] — NO integer spins!

  For H (J=0):
    Need 0 in the spin multiplet → only Hopf(2) gives [0,1] ✓

  Conclusion: Hopf(2) is the ONLY Hopf carrier that produces J=0 and J=1.
  Hopf(3) cannot give integer spins for bosons.
""")

# ── Final: Hopf(2) focused scan ──────────────────────────────────────
print("=" * 110)
print("FOCUSED: Hopf(2) carrier — the only option for integer-spin bosons")
print("=" * 110)
print("""
  Carrier (2,2), n_q=2, J_carrier=1/2
  J_mode = 1/2 → J = [0, 1] ← includes both W/Z (J=1) and H (J=0)!
  Per-component coprimality: gcd(p_m, q_m) = 1 ✓
  Per-component confinement: gcd(2, q_m) = 1 → q_m must be odd
""")

for bname, bmass, bspin in EWK:
    results = []
    for pm in range(1, 10):
        for qm in range(1, 10):
            if gcd(pm, qm) != 1: continue
            if qm % 2 == 0: continue  # per-component confinement
            pe, qe = pm*2, qm*2
            for m in range(2, 70):
                mass = mass_single(pe, qe, m, n_q=2)
                if mass is None: continue
                err = abs(mass - bmass) / bmass
                if err < 0.03:
                    try:
                        nqq = 2**qe
                    except:
                        nqq = float('inf')
                    results.append((err, pm, qm, pe, qe, m, mass, nqq))
    results.sort()
    print(f"\n  {bname} (mass={bmass:.0f}, J={bspin}):")
    if results:
        for err, pm, qm, pe, qe, m_int, mass, nqq in results[:8]:
            beta = beta_fast(pe, m_int)
            geom = mass / (nqq * ME) if nqq > 0 else 0
            print(f"    mode({pm},{qm}) eff({pe},{qe}) m={m_int:2d} "
                  f"2^{qe}={nqq:>8d} geom={geom:8.2f} "
                  f"mass={mass:8.0f} err={err*100:.2f}%")
    else:
        print(f"    No matches within 3%")


# ── The W/Z mass ratio test ──────────────────────────────────────────
print("\n\n" + "=" * 110)
print("W/Z MASS RATIO TEST")
print("=" * 110)

mW = 80379.0
mZ = 91188.0
ratio = mZ / mW
print(f"\n  m_Z / m_W = {ratio:.6f}")
print(f"  1/cos(θ_W) = {1/np.cos(np.arcsin(np.sqrt(0.2312))):.6f}  (Weinberg angle)")
print(f"  Difference: {abs(ratio - 1/np.cos(np.arcsin(np.sqrt(0.2312))))/ratio*100:.3f}%")

print("""
  If W and Z share the same mode and carrier but differ only in m:
    m_Z/m_W = f(m_Z) / f(m_W) where f(m) = β(m) · ln(8β(m)) / (β_e · ln(8β_e))

  This ratio depends only on the phase closure integers, NOT on the mode or carrier.
  It's a purely geometric ratio — can we match it?
""")

# Check if same (mode, carrier) can give both W and Z at correct ratio
print("  Same (mode, carrier) hitting both W and Z:\n")
for pm in range(1, 8):
    for qm in range(1, 8):
        if gcd(pm, qm) != 1: continue
        if qm % 2 == 0: continue
        pe, qe = pm*2, qm*2

        w_matches = []
        z_matches = []
        for m in range(2, 70):
            mass = mass_single(pe, qe, m, n_q=2)
            if mass is None: continue
            if abs(mass - mW)/mW < 0.02: w_matches.append((m, mass))
            if abs(mass - mZ)/mZ < 0.02: z_matches.append((m, mass))

        for mw, massw in w_matches:
            for mz, massz in z_matches:
                if mz <= mw: continue
                pred_ratio = massz / massw
                ratio_err = abs(pred_ratio - ratio) / ratio * 100
                if ratio_err < 2:
                    print(f"    mode({pm},{qm}) eff({pe},{qe}): "
                          f"W@m={mw}({massw:.0f}) Z@m={mz}({massz:.0f}) "
                          f"ratio={pred_ratio:.4f} (err={ratio_err:.2f}%)")


if __name__ == '__main__':
    pass  # already runs at import
