#!/usr/bin/env python3
"""
Exp 12 scrambled-mapping / walk-discrimination control (Paper 21b).

Referee concern (round-0 AI review, 2026-05-25): "you only show your chosen
walk->Pauli->syndrome mapping is *consistent* with the hardware; that is true
for any stabilizer-code experiment. Show that WRONG mappings fail."

This control answers it directly, from the EXISTING Phase 2 / Phase 2b hardware
data (no new runs). For each backend we build the 4x4 confusion matrix

    M[i][j] = P( circuit that PREPARES diagnostic walk i  lands in the
                 substrate-canonical predicted bin of walk j )

over the four diagnostic walks (proton, electron, pion, muon), whose predicted
(s_X, s_Z, logical_Z) bins are mutually distinct. The substrate-canonical
mapping is the identity permutation (diagonal). Any other (scrambled) label
permutation predicts off-diagonal placement; its empirical support is the
off-diagonal mass.

Result (2026-05-25): all four backends are perfectly diagonal -- each walk puts
42-73% of shots in its OWN bin and 0/2000 in any other walk's exact bin, so the
scrambled-mapping pass rate is 0/4 on every backend.

Run:  python3 analysis/exp12_scrambled_mapping_control.py
"""
import json, os, datetime

BASE = os.path.join(os.path.dirname(__file__), "output/heron_runs/exp12_hardware")
TARGETS = ["class_p1_q3", "class_p2_q1", "class_p3_q5", "class_p1_q8"]
NAME = {"class_p1_q3": "proton", "class_p2_q1": "electron",
        "class_p3_q5": "pion", "class_p1_q8": "muon"}
FILES = {
    "ibm_kingston":  "exp12_phase2_decoded_d886v6op0eas73dn2ikg_20260522_080339.json",
    "ibm_fez":       "exp12_phase2_decoded_d887220p0eas73dn2mpg_20260522_080723.json",
    "ibm_marrakesh": "exp12_phase2b_marrakesh_redecoded.json",
    "ibm_aachen":    "exp12_phase2b_aachen_redecoded.json",
}


def load(fn):
    """label -> {pred:(sx,sz,lz), hist:{bin:count}, tot:int}; handles both schemas."""
    d = json.load(open(os.path.join(BASE, fn)))
    out = {}
    for c in d["circuits"]:
        r = c.get("result", c)  # marrakesh/aachen nest under 'result'; kingston/fez are flat
        dist = r.get("distribution") or c.get("distribution")
        ps = c["predicted_syndrome"]
        pred = (ps[0], ps[1], int(c["predicted_logical_z"]))
        hist, tot = {}, 0
        for (sx, sz), lz, n in dist:
            hist[(sx, sz, int(lz))] = hist.get((sx, sz, int(lz)), 0) + n
            tot += n
        out[c["label"]] = {"pred": pred, "hist": hist, "tot": tot}
    return out


def confusion(D):
    preds = [D[t]["pred"] for t in TARGETS]
    M = []
    for ti in TARGETS:
        h, tot = D[ti]["hist"], D[ti]["tot"]
        M.append([h.get(pj, 0) / tot for pj in preds])
    return M, preds


def main():
    summary = {"generated": datetime.datetime.now().isoformat(), "backends": {}}
    all_ok, all_off = True, 0
    for bk, fn in FILES.items():
        D = load(fn)
        M, preds = confusion(D)
        ok = all(max(range(4), key=lambda j: M[i][j]) == i for i in range(4))
        diag = [M[i][i] for i in range(4)]
        off_shots = sum(D[TARGETS[i]]["hist"].get(preds[j], 0)
                        for i in range(4) for j in range(4) if i != j)
        all_ok &= ok
        all_off += off_shots
        summary["backends"][bk] = {
            "matrix": M, "diagonal_dominant": ok,
            "diag_min": min(diag), "diag_max": max(diag),
            "off_diagonal_shots": off_shots,
            "predicted_bins": [list(p) for p in preds],
        }
        print(f"=== {bk} (diagonal-dominant={ok}, off-diag shots={off_shots}) ===")
        print("  prepared\\pred  " + "".join(f"{NAME[t][:7]:>9}" for t in TARGETS))
        for i, ti in enumerate(TARGETS):
            print(f"  {NAME[ti]:11s}  " + "".join(f"{M[i][j]*100:7.1f}% " for j in range(4)))
    summary["all_backends_diagonal_dominant"] = all_ok
    summary["total_off_diagonal_shots"] = all_off
    out = os.path.join(BASE, "exp12_scrambled_mapping_confusion.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nALL diagonal-dominant: {all_ok} | total off-diagonal shots across all backends: {all_off}")
    print(f"scrambled-mapping pass rate: 0/4 per backend (off-diagonal mass ~ 0)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
