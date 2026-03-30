#!/usr/bin/env python3
"""
NWT Electroweak Boson Analysis: W, Z, H as higher modes on knotted carriers.

Focused deep-dive into the multi-mode carrier assignments that hit the
electroweak boson masses.  For each candidate, analyzes:
  - Confinement (gcd condition)
  - Spin compatibility
  - n_q^q enhancement factor (what lifts the mass to ~100 GeV)
  - Physical interpretation
  - Consistency with known quantum numbers
"""

import numpy as np
from math import gcd

KAPPA = np.pi**2
ME_MEV = 0.511
BETA_E = np.sqrt(9.0/4.0 - 1)  # √5/2

def beta_fast(p, m_int):
    val = m_int**2 / p**2 - 1
    if val <= 0: return None
    return np.sqrt(val)

def mass_fast(p_eff, q_eff, m_int, n_q):
    beta = beta_fast(p_eff, m_int)
    if beta is None or beta <= 1e-6: return None
    pq = p_eff**2 + q_eff**2
    log_ratio = np.log(8*beta) / np.log(8*BETA_E)
    if log_ratio <= 0: return None
    geom = (pq / 5.0) * (beta / BETA_E) * log_ratio
    try:
        enhance = float(n_q**q_eff) if n_q > 0 and q_eff > 0 else 1.0
    except (OverflowError, ValueError):
        return None
    if not np.isfinite(enhance) or enhance > 1e30: return None
    mass = geom * enhance * ME_MEV
    if not np.isfinite(mass) or mass <= 0: return None
    return mass

def torus_knot_crossings(p, q):
    if p <= 1 or q <= 1: return 0
    if p == q: return 2  # Hopf link convention
    if gcd(p, q) > 1: return gcd(p, q)  # multi-component link
    return min(p*(q-1), q*(p-1))

def carrier_name(pc, qc):
    if pc == 1 or qc == 1: return 'unknot'
    if pc == qc: return f'Hopf({pc})'
    if gcd(pc, qc) > 1: return f'link-{gcd(pc,qc)}'
    c = torus_knot_crossings(pc, qc)
    names = {3: 'trefoil T(2,3)', 5: '5₁ = T(2,5)', 7: '7₁ = T(2,7)',
             8: 'T(3,4)', 10: 'T(3,5)', 12: 'T(3,7)', 14: 'T(2,15)??'}
    return names.get(c, f'T({pc},{qc}) [{c} crossings]')

def spin_multiplet(n_q):
    if n_q == 0: return [0.5]
    j_c = (n_q - 1) / 2.0
    j_min = abs(0.5 - j_c)
    j_max = 0.5 + j_c
    return [j_min + i for i in range(int(j_max - j_min) + 1)]


# ── Electroweak bosons ───────────────────────────────────────────────
EWK_BOSONS = [
    ('W±',  80379.0,  1, 1.0, 0),  # name, mass, |Q|, spin, B
    ('Z⁰',  91188.0,  0, 1.0, 0),
    ('H⁰', 125100.0,  0, 0.0, 0),
]

# ── Comprehensive scan ───────────────────────────────────────────────
def scan_ewk():
    print("=" * 120)
    print("ELECTROWEAK BOSONS AS MULTI-MODE KNOTTED VORTICES")
    print("=" * 120)

    MODES = [(pm, qm) for pm in range(1, 6) for qm in range(1, 6)
             if gcd(pm, qm) == 1 and pm + qm <= 8]

    # Broader carrier scan
    CARRIERS = set()
    for p in range(1, 10):
        CARRIERS.add((1, p))
        CARRIERS.add((p, 1))
    for n in range(2, 10):
        CARRIERS.add((n, n))
    for p in range(2, 10):
        for q in range(p+1, 10):
            if gcd(p, q) == 1:
                CARRIERS.add((p, q))
                CARRIERS.add((q, p))
    CARRIERS = sorted(CARRIERS)

    for bname, bmass, bQ, bspin, bB in EWK_BOSONS:
        print(f"\n\n{'━' * 120}")
        print(f"  {bname}  mass = {bmass:.0f} MeV  Q = {bQ}  J = {bspin}  B = {bB}")
        print(f"{'━' * 120}")

        matches = []
        for pm, qm in MODES:
            for pc, qc in CARRIERS:
                pe = pm * pc
                qe = qm * qc

                n_cross = torus_knot_crossings(pc, qc)
                n_q = n_cross
                B = n_q % 2
                spins = spin_multiplet(n_q)

                for m in range(2, 60):
                    mass = mass_fast(pe, qe, m, n_q)
                    if mass is None: continue
                    err = abs(mass - bmass) / bmass
                    if err > 0.03: continue

                    conf_gcd = gcd(n_q, qe) if n_q > 0 else 0
                    spin_ok = bspin in spins
                    matches.append({
                        'mode': (pm, qm), 'carrier': (pc, qc),
                        'eff': (pe, qe), 'm': m, 'n_q': n_q, 'B': B,
                        'spins': spins, 'mass': mass, 'err': err,
                        'conf_gcd': conf_gcd, 'spin_ok': spin_ok,
                        'cname': carrier_name(pc, qc),
                    })

        matches.sort(key=lambda x: x['err'])

        # Categorize
        confined_spinok = [m for m in matches if m['conf_gcd'] <= 1 and m['spin_ok']]
        confined_any = [m for m in matches if m['conf_gcd'] <= 1]
        deconfined_spinok = [m for m in matches if m['conf_gcd'] > 1 and m['spin_ok']]
        all_matches = matches

        print(f"\n  Total matches within 3%: {len(all_matches)}")
        print(f"  Confined + spin OK: {len(confined_spinok)}")
        print(f"  Confined (any spin): {len(confined_any)}")
        print(f"  Deconfined + spin OK: {len(deconfined_spinok)}")

        def show_matches(label, mlist, limit=15):
            if not mlist: return
            print(f"\n  ── {label} {'─' * (100 - len(label))}")
            print(f"  {'Mode':>8s} {'Carrier':>10s} {'Type':>18s} {'Eff':>10s} "
                  f"{'m':>3s} {'n_q':>3s} {'n_q^q':>12s} {'geom':>8s} "
                  f"{'Mass':>10s} {'Err%':>6s} {'J':>12s} {'Conf':>6s}")
            for m in mlist[:limit]:
                pm, qm = m['mode']
                pc, qc = m['carrier']
                pe, qe = m['eff']

                # Decompose mass into geometric part and n_q^q part
                beta = beta_fast(pe, m['m'])
                pq = pe**2 + qe**2
                if beta and beta > 0:
                    geom = (pq/5.0) * (beta/BETA_E) * (np.log(8*beta)/np.log(8*BETA_E)) * ME_MEV
                else:
                    geom = 0
                try:
                    nq_factor = float(m['n_q']**qe) if m['n_q'] > 0 else 1.0
                except:
                    nq_factor = float('inf')

                spins_str = ','.join(f'{s}' for s in m['spins'])
                conf_str = f"gcd={m['conf_gcd']}" if m['n_q'] > 0 else "n/a"
                nq_str = f"{nq_factor:.0f}" if np.isfinite(nq_factor) and nq_factor < 1e12 else "huge"

                print(f"  ({pm},{qm}){'':<3s} ({pc},{qc}){'':<3s} {m['cname']:>18s} "
                      f"({pe},{qe}){'':<3s} {m['m']:3d} {m['n_q']:3d} {nq_str:>12s} "
                      f"{geom:8.2f} {m['mass']:10.1f} {m['err']*100:5.2f}% "
                      f"[{spins_str}]{'':<3s} {conf_str}")

        show_matches("BEST: Confined + correct spin", confined_spinok)
        show_matches("Confined (spin mismatch)", [m for m in confined_any if not m['spin_ok']])
        show_matches("Deconfined + correct spin", deconfined_spinok)

    # ── Cross-comparison ─────────────────────────────────────────────
    print("\n\n" + "=" * 120)
    print("CROSS-ANALYSIS: Can one carrier type explain all three EWK bosons?")
    print("=" * 120)

    # For each carrier type, check if it can hit W, Z, and H
    carrier_hits = {}
    for pm, qm in MODES:
        for pc, qc in CARRIERS:
            pe = pm * pc
            qe = qm * qc
            n_cross = torus_knot_crossings(pc, qc)
            n_q = n_cross

            for bname, bmass, bQ, bspin, bB in EWK_BOSONS:
                for m in range(2, 60):
                    mass = mass_fast(pe, qe, m, n_q)
                    if mass is None: continue
                    err = abs(mass - bmass) / bmass
                    if err > 0.03: continue

                    conf_gcd = gcd(n_q, qe) if n_q > 0 else 0
                    key = (pc, qc)
                    if key not in carrier_hits:
                        carrier_hits[key] = {}
                    if bname not in carrier_hits[key]:
                        carrier_hits[key][bname] = []
                    carrier_hits[key][bname].append({
                        'mode': (pm, qm), 'eff': (pe, qe), 'm': m,
                        'mass': mass, 'err': err, 'conf_gcd': conf_gcd,
                        'n_q': n_q,
                    })

    # Find carriers that hit 2+ or all 3
    print("\n  Carriers hitting multiple EWK bosons:\n")
    for (pc, qc), hits in sorted(carrier_hits.items()):
        if len(hits) >= 2:
            cname = carrier_name(pc, qc)
            n_cross = torus_knot_crossings(pc, qc)
            bosons_hit = list(hits.keys())
            print(f"  Carrier ({pc},{qc}) [{cname}] n_q={n_cross}: hits {bosons_hit}")
            for bname in bosons_hit:
                best = min(hits[bname], key=lambda x: x['err'])
                pm, qm = best['mode']
                pe, qe = best['eff']
                conf = best['conf_gcd']
                print(f"    {bname}: mode({pm},{qm}) → eff({pe},{qe}) m={best['m']} "
                      f"mass={best['mass']:.0f} err={best['err']*100:.2f}% "
                      f"{'CONF' if conf <= 1 else f'DECONF gcd={conf}'}")

    # ── Trefoil deep dive ────────────────────────────────────────────
    print("\n\n" + "=" * 120)
    print("TREFOIL DEEP DIVE: All modes on (2,3) and (3,2) that hit EWK masses")
    print("=" * 120)

    for pc, qc in [(2, 3), (3, 2)]:
        cname = carrier_name(pc, qc)
        n_cross = torus_knot_crossings(pc, qc)
        print(f"\n  Carrier ({pc},{qc}) [{cname}], n_q={n_cross}")
        print(f"  {'Mode':>8s} {'Eff':>10s} {'m':>3s} {'n_q^q':>12s} "
              f"{'Geom(MeV)':>12s} {'Mass(MeV)':>12s} {'Target':>8s} {'Err%':>7s} {'Conf':>8s}")
        print(f"  {'─'*95}")

        for pm, qm in sorted(MODES):
            pe = pm * pc
            qe = qm * qc
            n_q = n_cross
            conf = gcd(n_q, qe)

            for m in range(2, 60):
                mass = mass_fast(pe, qe, m, n_q)
                if mass is None: continue

                for bname, bmass, bQ, bspin, bB in EWK_BOSONS:
                    err = abs(mass - bmass) / bmass
                    if err > 0.05: continue  # 5% for deep dive

                    beta = beta_fast(pe, m)
                    pq = pe**2 + qe**2
                    geom = (pq/5.0) * (beta/BETA_E) * (np.log(8*beta)/np.log(8*BETA_E)) * ME_MEV if beta else 0
                    try:
                        nq_factor = float(n_q**qe) if n_q > 0 else 1.0
                    except:
                        nq_factor = float('inf')
                    nq_str = f"{nq_factor:.0f}" if np.isfinite(nq_factor) and nq_factor < 1e10 else "huge"
                    conf_str = f"gcd={conf}" if n_q > 0 else "n/a"

                    print(f"  ({pm},{qm}){'':<3s} ({pe},{qe}){'':<3s} {m:3d} "
                          f"{nq_str:>12s} {geom:12.2f} {mass:12.1f} "
                          f"{bname:>8s} {err*100:6.2f}% {conf_str:>8s}")

    # ── The n_q^q decomposition ──────────────────────────────────────
    print("\n\n" + "=" * 120)
    print("n_q^q DECOMPOSITION: What lifts masses to EWK scale?")
    print("=" * 120)

    print("""
  The mass formula is: m = geometric(p_eff, q_eff, m_int) × n_q^q_eff × m_e

  For the electron: geometric = 1, n_q^q = 1 → m = 0.511 MeV
  For a proton: geometric ~ 5.2, n_q^q = 3^3 = 27 → m ~ 72 MeV (needs mode help)

  To reach ~80-125 GeV, we need geometric × n_q^q ~ 157,000 - 245,000

  Key n_q^q values for various carrier/mode combos:
""")

    combos = [
        (3, 3, "trefoil, q_eff=3"),
        (3, 4, "trefoil, q_eff=4"),
        (3, 6, "trefoil, q_eff=6"),
        (3, 8, "trefoil, q_eff=8"),
        (3, 12, "trefoil, q_eff=12"),
        (2, 4, "Hopf, q_eff=4"),
        (2, 6, "Hopf, q_eff=6"),
        (2, 8, "Hopf, q_eff=8"),
        (2, 9, "Hopf, q_eff=9"),
        (2, 12, "Hopf, q_eff=12"),
        (5, 3, "5₁ knot, q_eff=3"),
        (5, 4, "5₁ knot, q_eff=4"),
    ]
    print(f"  {'n_q':>4s} {'q_eff':>5s} {'n_q^q_eff':>15s} {'Context':>30s}")
    print(f"  {'─'*60}")
    for nq, qe, ctx in combos:
        try:
            val = nq**qe
            print(f"  {nq:4d} {qe:5d} {val:15,d} {ctx:>30s}")
        except:
            print(f"  {nq:4d} {qe:5d} {'overflow':>15s} {ctx:>30s}")


if __name__ == '__main__':
    scan_ewk()
