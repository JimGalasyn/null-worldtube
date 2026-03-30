#!/usr/bin/env python3
"""
NWT Multi-Mode × Carrier Scan: Beyond "One Mode, Many Knots"

Explores whether higher-genus EM modes (beyond the (2,1) electron) can
propagate on knotted carriers.  Key question: can mode (1,4) — the old
proton mode — live on a trefoil carrier, and does it map to known particles?

Uses the fast analytical beta formula for scanning, with the full GPE
integral for verification of hits.
"""

import numpy as np
from math import gcd
import sys

# ── Constants ─────────────────────────────────────────────────────────
KAPPA = np.pi**2
ME_MEV = 0.511

# ── Fast analytical mass formula ──────────────────────────────────────
# β = √(m²/p² - 1) from simple phase closure
# Mass = (p²+q²)/5 · (β/β_e) · ln(8β)/ln(8β_e) · n_q^q · m_e

BETA_E = np.sqrt(9.0/4.0 - 1)  # m=3, p=2: √(9/4-1) = √(5/4) = √5/2

def beta_fast(p, m_int):
    """Fast β from analytical phase closure."""
    val = m_int**2 / p**2 - 1
    if val <= 0:
        return None
    return np.sqrt(val)

def mass_fast(p_eff, q_eff, m_int, n_q):
    """Fast NWT mass formula (analytical β)."""
    beta = beta_fast(p_eff, m_int)
    if beta is None or beta <= 1e-6:
        return None
    pq = p_eff**2 + q_eff**2
    log_ratio = np.log(8*beta) / np.log(8*BETA_E)
    if log_ratio <= 0:
        return None
    geom = (pq / 5.0) * (beta / BETA_E) * log_ratio
    try:
        enhance = float(n_q**q_eff) if n_q > 0 and q_eff > 0 else 1.0
    except (OverflowError, ValueError):
        return None
    if not np.isfinite(enhance) or enhance > 1e30:
        return None
    mass = geom * enhance * ME_MEV
    if not np.isfinite(mass) or mass <= 0 or mass > 1e12:
        return None
    return mass

def torus_knot_crossings(p, q):
    """Crossing number of torus knot T(p,q) for coprime p,q > 1."""
    return min(p*(q-1), q*(p-1))

def carrier_info(pc, qc):
    """Determine carrier type and crossing number."""
    if pc == 1 or qc == 1:
        return 'unknot', 0
    elif pc == qc:
        return f'Hopf({pc})', 2
    elif gcd(pc, qc) > 1:
        g = gcd(pc, qc)
        return f'link-{g}', g
    else:
        n_cross = torus_knot_crossings(pc, qc)
        names = {3: 'trefoil', 5: '5₁', 7: '7₁', 8: 'T(3,4)', 10: 'T(3,5)'}
        name = names.get(n_cross, f'T({pc},{qc})')
        return name, n_cross

def spin_multiplet(n_q):
    """Allowed spins from J_mode=1/2 + J_carrier=(n_q-1)/2."""
    if n_q == 0:
        return [0.5]
    j_carrier = (n_q - 1) / 2.0
    j_mode = 0.5
    j_min = abs(j_mode - j_carrier)
    j_max = j_mode + j_carrier
    return [j_min + i for i in range(int(j_max - j_min) + 1)]

# ── Known particles ──────────────────────────────────────────────────
KNOWN_PARTICLES = [
    ('e⁻',        0.511,   0, [0.5]),
    ('μ⁻',      105.66,    0, [0.5]),
    ('τ⁻',     1776.86,    0, [0.5]),
    ('π⁰',      135.0,     2, [0, 1]),
    ('π±',       139.57,   2, [0, 1]),
    ('K±',       493.68,   2, [0, 1]),
    ('K⁰',      497.61,   2, [0, 1]),
    ('η',        547.86,   2, [0, 1]),
    ('ρ',        775.26,   2, [0, 1]),
    ('ω',        782.66,   2, [0, 1]),
    ('K*',       891.67,   2, [0, 1]),
    ("η'",       957.78,   2, [0, 1]),
    ('φ',       1019.46,   2, [0, 1]),
    ('p',        938.27,   3, [0.5, 1.5]),
    ('n',        939.57,   3, [0.5, 1.5]),
    ('Λ',       1115.7,    3, [0.5, 1.5]),
    ('Σ',       1189.4,    3, [0.5, 1.5]),
    ('Δ',       1232.0,    3, [0.5, 1.5]),
    ('Ξ',       1314.9,    3, [0.5, 1.5]),
    ('Σ*',      1385.0,    3, [0.5, 1.5]),
    ('Ξ*',      1530.0,    3, [0.5, 1.5]),
    ('Ω⁻',     1672.5,    3, [0.5, 1.5]),
    ('D⁰',     1864.8,    2, [0, 1]),
    ('D±',      1869.7,    2, [0, 1]),
    ('Λc',      2286.5,    3, [0.5, 1.5]),
    ('Ξc',      2469.4,    3, [0.5, 1.5]),
    ('J/ψ',     3096.9,    2, [0, 1]),
    ('Ξcc⁺⁺',  3621.2,    3, [0.5, 1.5]),
    ('Ωc',      2695.2,    3, [0.5, 1.5]),
    ('Σc',      2453.0,    3, [0.5, 1.5]),
    ('Ωcc',     3738.0,    3, [0.5, 1.5]),  # predicted
    ('ψ2S',     3686.1,    2, [0, 1]),
    ('B±',      5279.3,    2, [0, 1]),
    ('Bs',      5366.9,    2, [0, 1]),
    ('Λb',      5619.6,    3, [0.5, 1.5]),
    ('Ξb',      5797.0,    3, [0.5, 1.5]),
    ('Ωb',      6046.0,    3, [0.5, 1.5]),
    ('Υ',       9460.3,    2, [0, 1]),
    ('X3872',   3872.0,    4, [1, 2]),
    ('Tcc',     3875.0,    4, [1, 2]),
    ('Zc3900',  3887.0,    4, [1, 2]),
    ('Z4430',   4478.0,    4, [1, 2]),
    ('Pc43',    4312.0,    5, [1.5, 2.5]),
    ('Pc44',    4380.0,    5, [1.5, 2.5]),
    ('Pc44a',   4440.0,    5, [1.5, 2.5]),
    ('W±',     80379.0,    0, [1.0]),
    ('Z⁰',    91188.0,    0, [1.0]),
    ('H⁰',   125100.0,    0, [0.0]),
]

# ── Scan configuration ───────────────────────────────────────────────
MODES = [
    (1,1), (2,1), (1,2), (3,1), (1,3),
    (3,2), (2,3), (4,1), (1,4),
]

CARRIERS = []
# Unknots
for q in range(1, 13):
    CARRIERS.append((1, q))
for p in range(2, 7):
    CARRIERS.append((p, 1))
# Hopf links
for n in range(2, 10):
    CARRIERS.append((n, n))
# Torus knots (coprime pairs)
for pc in range(2, 8):
    for qc in range(pc+1, 8):
        if gcd(pc, qc) == 1:
            CARRIERS.append((pc, qc))
            CARRIERS.append((qc, pc))

# Deduplicate
CARRIERS = list(set(CARRIERS))
CARRIERS.sort()

# ── Main scan ─────────────────────────────────────────────────────────
def run_scan():
    print("=" * 110)
    print("MULTI-MODE × CARRIER SCAN: Higher-genus modes on knotted carriers")
    print("=" * 110)
    print(f"\nβ_e = {BETA_E:.6f}")
    print(f"Modes: {MODES}")
    print(f"Carriers: {len(CARRIERS)} types")
    print(f"m range: [2, 50]\n")

    # Generate all predictions
    predictions = []
    for pm, qm in MODES:
        for pc, qc in CARRIERS:
            pe = pm * pc
            qe = qm * qc

            cname, n_cross = carrier_info(pc, qc)
            n_q = n_cross
            B = n_q % 2
            spins = spin_multiplet(n_q)

            for m in range(2, 51):
                mass = mass_fast(pe, qe, m, n_q)
                if mass is None or mass > 200000:
                    continue
                predictions.append({
                    'mode': (pm, qm),
                    'carrier': (pc, qc),
                    'cname': cname,
                    'eff': (pe, qe),
                    'm': m,
                    'n_q': n_q,
                    'B': B,
                    'spins': spins,
                    'mass': mass,
                })

    print(f"Total predictions: {len(predictions)}")

    # ── Match against known particles ─────────────────────────────────
    print("\n" + "=" * 110)
    print("MATCHES: All mode × carrier assignments within 3% of known particles")
    print("=" * 110)

    all_new_matches = []

    for pname, pmass, p_nq, p_spins in KNOWN_PARTICLES:
        matches = []
        for pred in predictions:
            err = abs(pred['mass'] - pmass) / pmass
            if err > 0.03:
                continue
            if p_nq > 0 and pred['n_q'] != p_nq:
                continue
            spin_ok = any(s in pred['spins'] for s in p_spins) or p_nq == 0
            if not spin_ok:
                continue

            is_new = (pred['mode'] != (2, 1) and pred['n_q'] > 0)
            matches.append((err, pred, is_new))

        matches.sort(key=lambda x: x[0])
        if not matches:
            continue

        new_in = [m for m in matches if m[2]]
        std_in = [m for m in matches if not m[2]]

        print(f"\n{'─'*100}")
        print(f"  {pname:10s}  m_exp = {pmass:.1f} MeV  (n_q={p_nq}, J={p_spins})")

        shown = 0
        for err, pred, is_new in matches:
            if shown >= 10:
                break
            pm, qm = pred['mode']
            pc, qc = pred['carrier']
            pe, qe = pred['eff']
            tag = " ★ NEW" if is_new else ""
            conf = gcd(pred['n_q'], qe) if pred['n_q'] > 0 else 0
            conf_str = f"gcd({pred['n_q']},{qe})={conf}" if pred['n_q'] > 0 else ""
            print(f"    mode({pm},{qm}) × carrier({pc},{qc}) [{pred['cname']:>10s}] "
                  f"→ eff({pe},{qe}) m={pred['m']:2d} n_q={pred['n_q']} "
                  f"mass={pred['mass']:8.1f} err={err*100:5.2f}% "
                  f"{conf_str} J={pred['spins']}{tag}")
            shown += 1

        if new_in:
            for err, pred, is_new in new_in[:3]:
                all_new_matches.append((pname, pmass, err, pred))

    # ── Focus: (1,4) on trefoil ───────────────────────────────────────
    print("\n\n" + "=" * 110)
    print("FOCUS: Mode (1,4) on all carrier types")
    print("=" * 110)

    for pc, qc in [(1,1), (1,2), (1,3), (2,2), (2,3), (3,2), (2,5), (5,2), (3,4), (4,3)]:
        pm, qm = 1, 4
        pe = pm * pc
        qe = qm * qc
        cname, n_cross = carrier_info(pc, qc)
        n_q = n_cross
        B = n_q % 2
        spins = spin_multiplet(n_q)
        conf = gcd(n_q, qe) if n_q > 0 else 0

        print(f"\n  Mode (1,4) × Carrier ({pc},{qc}) [{cname}]")
        print(f"  eff({pe},{qe}), n_q={n_q}, B={B}, J={spins}, "
              f"p²+q²={pe**2+qe**2}, "
              f"{'CONFINED' if conf <= 1 else f'DECONFINED gcd={conf}'}")

        results = []
        for m in range(2, 51):
            mass = mass_fast(pe, qe, m, n_q)
            if mass is None or mass <= 0:
                continue
            closest = min(KNOWN_PARTICLES, key=lambda p: abs(p[1] - mass) / max(p[1], 1))
            cerr = abs(closest[1] - mass) / closest[1] * 100
            results.append((m, mass, closest[0], cerr))

        if results:
            print(f"  {'m':>4s} {'Mass(MeV)':>12s} {'Closest':>12s} {'Err%':>8s}")
            for m, mass, cname2, cerr in results[:25]:
                mark = " ◄" if cerr < 3.0 else ""
                print(f"  {m:4d} {mass:12.1f} {cname2:>12s} {cerr:7.2f}%{mark}")

    # ── Focus: other interesting modes on trefoil ─────────────────────
    print("\n\n" + "=" * 110)
    print("FOCUS: Various modes on trefoil T(2,3)")
    print("=" * 110)

    trefoil_carriers = [(2, 3), (3, 2)]
    interesting_modes = [(1,1), (2,1), (3,1), (1,2), (3,2), (2,3), (1,3), (4,1), (1,4)]

    for pc, qc in trefoil_carriers:
        cname_c, n_cross = carrier_info(pc, qc)
        print(f"\n{'━'*100}")
        print(f"  Carrier ({pc},{qc}) [{cname_c}], crossings={n_cross}")
        print(f"{'━'*100}")

        for pm, qm in interesting_modes:
            pe = pm * pc
            qe = qm * qc
            n_q = n_cross
            spins = spin_multiplet(n_q)
            conf = gcd(n_q, qe)

            hits = []
            for m in range(2, 51):
                mass = mass_fast(pe, qe, m, n_q)
                if mass is None:
                    continue
                for pname, pmass, p_nq, p_spins in KNOWN_PARTICLES:
                    if p_nq > 0 and p_nq != n_q:
                        continue
                    err = abs(mass - pmass) / pmass
                    if err < 0.03:
                        hits.append((m, mass, pname, pmass, err*100))

            conf_str = f"gcd({n_q},{qe})={conf}"
            status = "CONFINED" if conf == 1 else f"DECONFINED"
            print(f"\n  mode({pm},{qm}) × ({pc},{qc}) → eff({pe},{qe})  "
                  f"p²+q²={pe**2+qe**2}  J={spins}  {conf_str} {status}")
            if hits:
                for m, mass, pn, pm2, err in hits[:5]:
                    print(f"    m={m:2d}: {mass:.1f} MeV → {pn} ({pm2:.1f}) err={err:.2f}%")
            else:
                # Show a few representative masses
                sample = []
                for m in [3, 5, 10, 15, 20, 30]:
                    mass = mass_fast(pe, qe, m, n_q)
                    if mass is not None:
                        sample.append((m, mass))
                if sample:
                    masses_str = ', '.join(f"m={m}→{mass:.0f}" for m, mass in sample)
                    print(f"    No hits. Sample masses: {masses_str}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 110)
    print("SUMMARY: New higher-mode assignments for known particles")
    print("=" * 110)

    if all_new_matches:
        # Deduplicate
        seen = set()
        unique = []
        for pname, pmass, err, pred in sorted(all_new_matches, key=lambda x: x[2]):
            key = (pname, pred['mode'], pred['carrier'], pred['m'])
            if key not in seen:
                seen.add(key)
                unique.append((pname, pmass, err, pred))

        print(f"\n  {len(unique)} unique new assignments (mode ≠ (2,1) on knotted carrier):\n")
        for pname, pmass, err, pred in unique[:30]:
            pm, qm = pred['mode']
            pc, qc = pred['carrier']
            pe, qe = pred['eff']
            conf = gcd(pred['n_q'], qe) if pred['n_q'] > 0 else 0
            print(f"    {pname:10s} ({pmass:8.1f}): mode({pm},{qm})×carrier({pc},{qc}) "
                  f"[{pred['cname']}] → eff({pe},{qe}) m={pred['m']:2d} "
                  f"mass={pred['mass']:.1f} err={err*100:.2f}% "
                  f"{'CONF' if conf<=1 else 'DECONF'}")
    else:
        print("\n  No new higher-mode assignments found.")

    # ── Degeneracy analysis ───────────────────────────────────────────
    print("\n\n" + "=" * 110)
    print("DEGENERACY: How many distinct (mode,carrier) pairs give the same eff(p,q)?")
    print("=" * 110)

    from collections import defaultdict
    eff_map = defaultdict(list)
    for pm, qm in MODES:
        for pc, qc in CARRIERS:
            pe = pm * pc
            qe = qm * qc
            cname_c, n_cross = carrier_info(pc, qc)
            eff_map[(pe, qe, n_cross)].append(((pm, qm), (pc, qc), cname_c))

    # Show degeneracies > 1
    degens = [(k, v) for k, v in eff_map.items() if len(v) > 1]
    degens.sort(key=lambda x: x[0][0]**2 + x[0][1]**2)
    print(f"\n  {len(degens)} degenerate effective windings found:\n")
    for (pe, qe, nq), combos in degens[:30]:
        spins = spin_multiplet(nq)
        print(f"  eff({pe},{qe}) n_q={nq} J={spins}:")
        for (pm, qm), (pc, qc), cn in combos:
            print(f"    mode({pm},{qm}) × carrier({pc},{qc}) [{cn}]")


if __name__ == '__main__':
    run_scan()
