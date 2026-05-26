#!/usr/bin/env python3
"""
Exp 15/16 — cross-sector syndrome-equivalence test (Paper 21a Table 4, §7.2).

PRE-REGISTRATION (committed before any hardware submission).

Paper 21a predicts that particles the Standard Model assigns to *different* families
share a Steane [[7,1,3]] stabilizer-syndrome bin. This experiment prepares the walks
for three such equivalence classes on quantum hardware and checks that the members of
each class land in the SAME substrate-canonical (s_X, s_Z) bin:

    Class A  (P, F3) :  mu-  (1,8)   ==  D+/J-psi (2,7)       lepton  + 2 mesons
    Class B  (E2,E2) :  tau- (3,4)   ==  Lambda   ==  Sigma*  lepton  + 2 hyperons
    Class C  (F2,E2) :  e-   (2,1)   ==  pi0      (7,3)       lepton  + meson

mu- (1,8) and e- (2,1) were already confirmed in Exp 12 and serve here as
cross-experiment stability anchors; D+/J-psi (2,7), tau-/Lambda/Sigma* (3,4), and
pi0 (7,3) are NEW walks not previously run on chip. Class B's three particles share a
single walk (3,4) so compile to one circuit; the substantive new content is that walks
with DIFFERENT (p,q) -- mu- (1,8) vs D+ (2,7), and e- (2,1) vs pi0 (7,3) -- reach the
SAME (s_X, s_Z) bin via the deterministic walk-to-Pauli map.

HONEST SCOPE (per the de-ontologization arc): the equivalence syndromes are computable
on paper from the (p,q) -> Pauli-word map; a chip run confirms the predicted regrouping
is REALIZABLE and noise-robust on hardware, it does NOT discriminate NWT from standard
stabilizer QM (both predict the same syndromes). The framework's discriminating tests
are off-chip (the non-arbitrariness of the (p,q) assignment -- fixed by mass + charge +
isospin + baryon number via Gell-Mann-Nishijima, Paper 7 -- and detector searches for
the sub-500 MeV substrate-prediction species). This is a realizability demonstration.

Reuses the ancilla-free destructive-readout machinery of exp12_phase2_destructive.py
(the variant that reproduced 8/8 cross-vendor incl. IQM Garnet): two circuits per walk
(Z-basis + X-basis), 7 data qubits, no syndrome ancillas.

Run:
  python3 analysis/exp15_cross_sector_equivalence.py --mode dry-run --shots 2000
  python3 analysis/exp15_cross_sector_equivalence.py --mode hardware-submit \
      --backend ibm_kingston --shots 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("/home/jim/repos/nwt-substrate")))
sys.path.insert(0, str(Path(__file__).parent))

import exp12_phase2_destructive as base
from exp12_phase1_simulator_pilot import predicted_for
from nwt_substrate.heron.steane_pair_synthesis import COMPENDIUM_LOOKUP

# ---------------------------------------------------------------------------
# Equivalence classes (Paper 21a Table 4). syndrome is the predicted (s_X, s_Z).
# ---------------------------------------------------------------------------
EQUIV_CLASSES = {
    "A_(P,F3)":  {"syndrome": ("P", "F3"),  "members": [(1, 8), (2, 7)],
                  "names": {(1, 8): "mu-", (2, 7): "D+ / J/psi"},
                  "sm": "lepton + 2 mesons"},
    "B_(E2,E2)": {"syndrome": ("E2", "E2"), "members": [(3, 4)],
                  "names": {(3, 4): "tau- / Lambda / Sigma*"},
                  "sm": "lepton + 2 hyperons (one walk)"},
    "C_(F2,E2)": {"syndrome": ("F2", "E2"), "members": [(2, 1), (7, 3)],
                  "names": {(2, 1): "e-", (7, 3): "pi0"},
                  "sm": "lepton + meson"},
}
# distinct walks across all classes, preserving order
TARGETS = []
for cls in EQUIV_CLASSES.values():
    for pq in cls["members"]:
        if pq not in TARGETS:
            TARGETS.append(pq)
NEW_WALKS = {(2, 7), (3, 4), (7, 3)}  # not in Exp 12


def print_prereg():
    """Print + verify the pre-registered predictions from the precompiled Pauli words."""
    pq = {(e["p"], e["q"]): e for e in COMPENDIUM_LOOKUP["particles"]}
    norm = lambda s: s.replace("_", "")
    print("  PRE-REGISTERED PREDICTIONS (from COMPENDIUM Pauli words, no fitting):")
    all_ok = True
    for cname, cls in EQUIV_CLASSES.items():
        eX, eZ = cls["syndrome"]
        print(f"    {cname:11} predicted (s_X,s_Z)=({eX},{eZ})   [{cls['sm']}]")
        for m in cls["members"]:
            syn, lz = predicted_for(pq[m])
            ok = norm(syn[0]) == eX and norm(syn[1]) == eZ
            all_ok &= ok
            tag = "NEW" if m in NEW_WALKS else "Exp12-anchor"
            print(f"        {str(m):7} {cls['names'][m]:24} -> ({syn[0]},{syn[1]},{lz:+d})"
                  f"  {'ok' if ok else 'MISMATCH':8} [{tag}]")
    print(f"  PRE-REG GATE: {'PASS (all match Table 4)' if all_ok else '*** FAIL ***'}\n")
    return all_ok


def equivalence_analysis(results):
    """Given run results, confirm each class's members land in the same modal bin."""
    by_pq = {(r["p"], r["q"]): r for r in results if r.get("p") is not None}
    print("\n  === CROSS-SECTOR EQUIVALENCE ANALYSIS ===")
    n_classes_ok = 0
    for cname, cls in EQUIV_CLASSES.items():
        eX, eZ = cls["syndrome"]
        modal_bins, lzs, ok_members = [], [], True
        for m in cls["members"]:
            r = by_pq.get(m)
            if r is None:
                ok_members = False
                continue
            mx = r["result"]["x_half"]["modal_fano"]
            mz = r["result"]["z_half"]["modal_fano"]
            mlz = r["result"]["z_half"]["modal_lz"]
            modal_bins.append((mx, mz))
            lzs.append(mlz)
            hit = (mx == eX and mz == eZ)
            ok_members &= hit
            print(f"    {cname:11} {cls['names'][m]:24} modal=({mx},{mz},{mlz:+d})"
                  f"  pred=({eX},{eZ})  {'HIT' if hit else 'miss'}")
        coincide = len(set(modal_bins)) == 1 and ok_members
        n_classes_ok += int(coincide)
        verdict = "EQUIVALENCE CONFIRMED" if coincide else "equivalence NOT confirmed"
        extra = ""
        if cname == "A_(P,F3)" and len(set(lzs)) > 1:
            extra = "  (members share (s_X,s_Z) but differ in logical-Z: +1 vs -1, as predicted)"
        print(f"    -> {cname}: {verdict}{extra}\n")
    print(f"  EQUIVALENCE CLASSES CONFIRMED: {n_classes_ok}/{len(EQUIV_CLASSES)}")
    return n_classes_ok


def main():
    ap = argparse.ArgumentParser(description="Exp 15/16 cross-sector syndrome-equivalence")
    ap.add_argument("--mode", choices=["dry-run", "hardware-stage", "hardware-submit"], default="dry-run")
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--no-controls", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("  EXP 15/16 — CROSS-SECTOR SYNDROME EQUIVALENCE (Paper 21a Table 4)")
    print(f"  Mode: {args.mode}  Backend: {args.backend}  Shots: {args.shots}")
    print("=" * 78 + "\n")
    print_prereg()

    # Override the destructive driver's target list with our equivalence walks
    base.PHASE2_TARGETS = TARGETS
    items = base.build_items(include_controls=not args.no_controls)
    print(f"  Built {len(items)} items -> {2*len(items)} circuits "
          f"({len(TARGETS)} walks + controls, Z+X readout each)\n")

    if args.mode == "dry-run":
        results, label = base.run_dry_run(items, args.shots)
    else:
        results, label = base.run_hardware(items, args.backend, args.shots,
                                            submit=(args.mode == "hardware-submit"))
    if results is None:
        print("\n  Staged, not submitted.")
        return

    n_eq = equivalence_analysis(results)

    # Save under exp15 dir
    out_dir = Path(__file__).parent / "output" / "heron_runs" / "exp15_cross_sector"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"exp15_cross_sector_{args.mode}_{label}_{stamp}.json"
    import json
    payload = {
        "experiment": "exp15_16_cross_sector_equivalence",
        "mode": args.mode, "backend": label, "shots_per_circuit": args.shots,
        "timestamp": stamp, "variant": "ancilla-free destructive CSS readout",
        "equiv_classes": {k: {"syndrome": v["syndrome"], "members": v["members"],
                              "names": {str(m): n for m, n in v["names"].items()}}
                          for k, v in EQUIV_CLASSES.items()},
        "n_equiv_classes_confirmed": n_eq, "n_equiv_classes": len(EQUIV_CLASSES),
        "n_walks_pass": sum(1 for r in results if r.get("p") is not None and r["result"]["pass"]),
        "circuits": results,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  Saved: {out}")
    print("=" * 78)


if __name__ == "__main__":
    main()
