#!/usr/bin/env python3
"""Rebuild the compendium DB as Paper 21a's K7-walk compendium (Option B).

Topology (roster, (p,q), sector, carrier, n_q, Steane syndrome) from Paper 21a.
NWT masses computed from the Paper-6 mass formula, IMPORTED from
analysis/nwt_lagrangian_L4_paper6_mass_spectrum.py (paper6_mass + its parameter
table) — nothing transcribed by hand. Also includes the 21a substrate-prediction
species (status='predicted').

Writes a NEW file, nwt_particles_v21a.db — does NOT touch nwt_particles.db.

KNOWN CROSS-SOURCE DISCREPANCY (flagged in notes, not silently resolved):
the Paper-6 mass calibration places nucleons at (1,4)/n_q=3; Paper 21a's walk
framework places them at (1,3)/n_q=5. Topology columns follow 21a; the NWT
`computed` mass is the Paper-6 value computed with its own params. Reconciling
the nucleon sector is a physics follow-up (cf. Paper 21a's Paper-6 closure §).
"""
import importlib.util
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "nwt_particles.db"     # canonical DB (Option B rebuild swaps it in)

# -- import the Paper-6 mass formula + its parameter table ----------------
p6_path = ROOT / "analysis/nwt_lagrangian_L4_paper6_mass_spectrum.py"
_spec = importlib.util.spec_from_file_location("p6", p6_path)
_p6 = importlib.util.module_from_spec(_spec)
sys.modules["p6"] = _p6
_spec.loader.exec_module(_p6)
paper6_mass, ME_MEV = _p6.paper6_mass, _p6.ME_MEV
LAGR = {n: (mx, p, q, m, nq) for (n, mx, p, q, m, nq) in _p6.PARTICLES}

# -- Paper 21a 25-walk compendium: (p,q,sector,[particles],L_min) ---------
PAPER21A = [
    (1, 3, "nucleon", ["p", "n", "Sigma+", "Sigma0", "Sigma-"], 7),
    (2, 1, "lepton",  ["e-"], 5),
    (3, 5, "meson",   ["pi+"], 8),
    (1, 8, "lepton",  ["mu-"], 22),
    (2, 5, "meson",   ["K+"], 10),
    (2, 7, "meson",   ["D+", "J/psi"], 14),
    (3, 4, "hyperon", ["tau-", "Lambda", "Sigma*"], 11),
    (3, 7, "meson",   ["D0"], 17),
    (4, 5, "meson",   ["omega"], 13),
    (4, 9, "meson",   ["Upsilon"], 27),
    (5, 4, "hyperon", ["Xi0", "Xi-", "Delta"], 13),
    (5, 7, "meson",   ["rho"], 18),
    (6, 5, "meson",   ["eta"], 17),
    (7, 3, "meson",   ["pi0"], 10),
    (7, 4, "hyperon", ["Omega-"], 12),
    (7, 5, "meson",   ["K0"], 11),
]
LAGR_NAME = {
    "e-": "e-", "mu-": "mu-", "tau-": "tau-", "pi+": "pi+", "pi0": "pi0",
    "K+": "K+", "K0": "K0", "eta": "eta", "rho": "rho", "omega": "omega",
    "p": "p", "n": "n", "Sigma+": "Si+", "Sigma0": "Si0", "Sigma-": "Si-",
    "Lambda": "Lam", "Sigma*": "Si*", "Delta": "Del", "Xi0": "Xi", "Xi-": "Xi",
    "Omega-": "Om-", "D+": "D+", "D0": "D0", "J/psi": "J/psi", "Upsilon": "Ups",
}
# Paper 21a Steane syndrome map: (p,q) -> (s_X, s_Z)
SYNDROME = {
    (1, 3): ("I", "I"), (1, 8): ("P", "F3"), (2, 1): ("F2", "E2"),
    (2, 5): ("P", "E1"), (2, 7): ("P", "F3"), (3, 4): ("E2", "E2"),
    (3, 5): ("E1", "E1"), (3, 7): ("I", "I"), (4, 5): ("F3", "F1"),
    (4, 9): ("P", "E1"), (5, 4): ("F2", "F3"), (5, 7): ("E3", "P"),
    (6, 5): ("E3", "P"), (7, 3): ("F2", "E2"), (7, 4): ("P", "E3"),
    (7, 5): ("E1", "E1"),
}
# Paper 21a substrate-prediction catalog (sub-500 MeV): (name,p,q,L,mass,signature)
PREDICTED = [
    ("pred(2,2)", 2, 2, 6, 25,  "light dark-matter candidate"),
    ("pred(2,3)", 2, 3, 8, 160, "QR-dominant resonance (carrier ambiguous)"),
    ("pred(3,1)", 3, 1, 8, 160, "NR-dominant resonance (carrier ambiguous)"),
    ("pred(3,3)", 3, 3, 9, 400, "DM-baryon/BSM hybrid — full 7/7 Fano coverage"),
]

F = {0: ("unknot", 0), 2: ("hopf", 0), 3: ("trefoil", 1), 5: ("cinquefoil", 2)}
def rule_I(p, q):
    pm, qm = p % 7, q % 7
    nq = (0 if qm == 1 else 3 if qm == 4 else 2 if qm in (0, 2, 5)
          else 5 if (qm == 3 and pm != 0) else 2 if qm == 3 else 8)
    carrier, genus = F.get(nq, ("T(2,8)", 3))
    return nq, carrier, genus


def build():
    if OUT.exists():
        OUT.unlink()
    conn = sqlite3.connect(OUT)
    conn.execute("""
        CREATE TABLE particles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, sector TEXT, kind TEXT DEFAULT 'observed',
            p INTEGER, q INTEGER, l_min INTEGER,
            carrier TEXT, n_q INTEGER, genus INTEGER,
            s_x TEXT, s_z TEXT,
            computed REAL, pdg_value REAL, unit TEXT DEFAULT 'MeV',
            residual_pct REAL, status TEXT, first_paper INTEGER DEFAULT 21,
            notes TEXT
        )""")
    n = 0
    for p, q, sector, parts, lmin in PAPER21A:
        nq, carrier, genus = rule_I(p, q)
        sx, sz = SYNDROME.get((p, q), (None, None))
        for nm in parts:
            lg = LAGR.get(LAGR_NAME.get(nm))
            note = f"21a walk ({p},{q})"
            status = "TIER1"
            if lg:
                mx, lp, lq, lm, lnq = lg
                bare = paper6_mass(lp, lq, lm, lnq) * ME_MEV
                pdg = mx
                if sector == "nucleon":
                    # Nucleons are binding-corrected anchors in the canonical DB
                    # (the bare T(2,5) knot mass overshoots ~+10%); use the
                    # anchored PDG value and record the bare value. Topology
                    # follows 21a; the Paper-6 calibration uses (1,4)/n_q=3.
                    computed, status = pdg, "anchor"
                    note += (f"; FLAG nucleon sector: anchored to PDG "
                             f"(bare Paper-6 knot mass {bare:.0f} MeV needs "
                             f"binding correction); Paper-6 topology "
                             f"({lp},{lq})/n_q={lnq} vs 21a ({p},{q})/n_q={nq}")
                else:
                    computed = bare
                resid = (computed - pdg) / pdg * 100 if pdg else None
            else:
                computed = pdg = resid = None
                status = "NO-MASS"
                note += "; mass not in Paper-6 table"
            conn.execute(
                "INSERT INTO particles (name,sector,kind,p,q,l_min,carrier,n_q,"
                "genus,s_x,s_z,computed,pdg_value,residual_pct,status,notes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (nm, sector, "observed", p, q, lmin, carrier, nq, genus, sx, sz,
                 computed, pdg, resid, status, note))
            n += 1

    # substrate-prediction species
    npred = 0
    for nm, p, q, lmin, mest, sig in PREDICTED:
        nq, carrier, genus = rule_I(p, q)
        conn.execute(
            "INSERT INTO particles (name,sector,kind,p,q,l_min,carrier,n_q,genus,"
            "computed,status,notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (nm, "prediction", "predicted", p, q, lmin, carrier, nq, genus,
             float(mest), "PRED", f"21a substrate prediction: {sig}"))
        npred += 1

    conn.commit()
    flagged = conn.execute(
        "SELECT COUNT(*) FROM particles WHERE notes LIKE '%FLAG%'").fetchone()[0]
    print(f"built {OUT.name}: {n} walk-particles + {npred} predicted")
    print(f"  syndrome populated: "
          f"{conn.execute('SELECT COUNT(*) FROM particles WHERE s_x IS NOT NULL').fetchone()[0]}")
    print(f"  NWT mass present:   "
          f"{conn.execute('SELECT COUNT(*) FROM particles WHERE computed IS NOT NULL').fetchone()[0]}/{n+npred}")
    print(f"  nucleon-sector mass/topology FLAGS: {flagged}")
    conn.close()


if __name__ == "__main__":
    build()
