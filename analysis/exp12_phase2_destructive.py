"""Heron Exp 12 — Phase 2 ANCILLA-FREE destructive-readout variant.

SUPERSEDED (2026-05-25) by `exp12_qpu.py --vendor ibm`, which drives the
vendor-neutral `nwt_substrate.qpu` interface. This script is retained as the
record of the original ibm_fez 8/8 destructive run; new work should use the
library path. See `qpu_migration_guide.md`.

Motivation (2026-05-25): the IQM Garnet cross-vendor run of the standard
13-qubit Phase 2 circuit (7 data + 6 syndrome ancillas) produced fully-flat
output — 58 CZ after routing onto Garnet's 20-qubit grid blew through the
coherence budget (see memory iqm-garnet-phase2-decoherence-negative). This
variant removes the 6 syndrome ancillas and their 24 syndrome-extraction
CNOTs entirely.

Key idea — destructive CSS readout. For a CSS code the full stabilizer
syndrome is recoverable from DESTRUCTIVE single-basis measurements of the
DATA qubits, split across two circuits:

  * Z-readout circuit:  prep |0_L> + walk Pauli word + measure all 7 in Z.
        -> Z-stabilizer eigenvalues (parity over each Z-stab support)
        -> logical-Z eigenvalue (parity over Z_L support {0,1,2})
  * X-readout circuit:  prep |0_L> + walk Pauli word + H^7 + measure all 7.
        -> X-stabilizer eigenvalues (parity over each X-stab support)

The predicted (s_X, s_Z, logical_Z) triple is fully recovered, just split
across the two halves. Each half is a 7-qubit, ~15-CZ, depth-27 circuit on
Garnet (vs 58 CZ / depth 63 / 13 qubits for the ancilla version).

Trade: 2 circuits per walk instead of 1. The syndrome decode convention
(stabilizer supports, Hamming->Fano map, logical-Z parity) is identical to
the ancilla pipeline, so predictions from `predicted_for` apply unchanged.

This is valid because Phase 2 is a STATE-PREP-VALIDATION test (does the
walk's Pauli word land |0_L> in the predicted syndrome bin?), not a
multi-round QEC protocol — we do not need a non-demolition syndrome.

Modes mirror exp12_phase2_hardware.py:
  --mode dry-run        AerSimulator (FakeMarrakesh noise), free smoke test
  --mode hardware-stage transpile + report depth/2q, do NOT submit
  --mode hardware-submit submit to a (free) IBM backend, decode, save

Usage:
  python3 analysis/exp12_phase2_destructive.py --mode dry-run --shots 2000
  python3 analysis/exp12_phase2_destructive.py --mode hardware-submit \\
      --backend ibm_marrakesh --shots 2000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

NWT_SUBSTRATE = Path("/home/jim/repos/nwt-substrate")
sys.path.insert(0, str(NWT_SUBSTRATE))
sys.path.insert(0, str(Path(__file__).parent))

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

from nwt_substrate.heron.steane_pair_synthesis import (
    _add_steane_zero_prep,
    _apply_pauli_word,
    COMPENDIUM_LOOKUP,
)
from exp12_phase1_simulator_pilot import predicted_for
# Reuse IBM plumbing from the ancilla pipeline
from exp12_phase2_hardware import (
    make_dry_run_backend,
    make_hardware_service,
    get_hardware_backend,
)

# ---------------------------------------------------------------------------
# Steane decode constants (identical convention to the ancilla pipeline)
# ---------------------------------------------------------------------------
STEANE_STAB_SUPPORTS = [[q for q in range(7) if ((q + 1) >> k) & 1] for k in range(3)]
STEANE_LOGICAL_Z_SUPPORT = [0, 1, 2]
QUBIT_TO_FANO = {0: "E3", 1: "E2", 2: "F1", 3: "E1", 4: "F2", 5: "F3", 6: "P"}

PHASE2_TARGETS = [(1, 3), (2, 1), (3, 5), (1, 8)]
CONTROL_PREDICTED = {
    "identity": (("I", "I"), +1),
    "x7":       (("I", "I"), -1),
    "z7":       (("I", "I"), +1),
    "h7":       (("I", "I"), 0),  # 50/50 logical-Z split at (I, I)
}


def _fano_from_sysbits(sys_bits) -> str:
    """Map 7 data-qubit parities over the 3 stabilizer supports to a Fano point."""
    s = [sum(sys_bits[i] for i in STEANE_STAB_SUPPORTS[k]) % 2 for k in range(3)]
    pos = s[0] + 2 * s[1] + 4 * s[2]
    return "I" if pos == 0 else QUBIT_TO_FANO[pos - 1]


def _logical_z_from_sysbits(sys_bits) -> int:
    parity = sum(sys_bits[i] for i in STEANE_LOGICAL_Z_SUPPORT) % 2
    return +1 if parity == 0 else -1


def _sysbits_from_qiskit(bs: str):
    """Qiskit bitstrings are little-endian (rightmost char = classical bit 0).
    We measured sys[q] -> c[q], so qubit q is character bs[-(q+1)]."""
    bs = bs.replace(" ", "")
    return [int(bs[-(q + 1)]) for q in range(7)]


# ---------------------------------------------------------------------------
# Circuit construction — two 7-qubit circuits per item (Z-readout, X-readout)
# ---------------------------------------------------------------------------

def _build_pair(label: str, apply_op):
    """Build (z_circuit, x_circuit). `apply_op(qc, sys)` injects the walk Pauli
    word or control operation after |0_L> prep."""
    pair = {}
    for basis in ("z", "x"):
        sysr = QuantumRegister(7, "sys")
        creg = ClassicalRegister(7, "c")
        qc = QuantumCircuit(sysr, creg, name=f"{label}_{basis}")
        _add_steane_zero_prep(qc, sysr)
        apply_op(qc, sysr)
        qc.barrier()
        if basis == "x":
            for q in range(7):
                qc.h(sysr[q])
        for q in range(7):
            qc.measure(sysr[q], creg[q])
        pair[basis] = qc
    return pair["z"], pair["x"]


def build_items(include_controls: bool = True):
    pq_to_entry = {(e["p"], e["q"]): e for e in COMPENDIUM_LOOKUP["particles"]}
    items = []
    for p, q in PHASE2_TARGETS:
        entry = pq_to_entry[(p, q)]
        xb = entry["forward_pauli"]["x_part_bits"]
        zb = entry["forward_pauli"]["z_part_bits"]
        pred_syn, pred_lz = predicted_for(entry)
        zc, xc = _build_pair(
            f"class_p{p}_q{q}",
            lambda qc, sysr, xb=xb, zb=zb: _apply_pauli_word(qc, sysr, xb, zb),
        )
        items.append({
            "label": f"class_p{p}_q{q}", "p": p, "q": q,
            "particles": entry["particles"],
            "predicted_syndrome": list(pred_syn),
            "predicted_logical_z": pred_lz,
            "z_circ": zc, "x_circ": xc,
        })
    if include_controls:
        def ctrl_op(kind):
            def op(qc, sysr):
                if kind == "x7":
                    for q in range(7): qc.x(sysr[q])
                elif kind == "z7":
                    for q in range(7): qc.z(sysr[q])
                elif kind == "h7":
                    for q in range(7): qc.h(sysr[q])
            return op
        for kind in ("identity", "x7", "z7", "h7"):
            pred_syn, pred_lz = CONTROL_PREDICTED[kind]
            zc, xc = _build_pair(f"control_{kind}", ctrl_op(kind))
            items.append({
                "label": f"control_{kind}", "p": None, "q": None,
                "particles": [f"control_{kind}"],
                "predicted_syndrome": list(pred_syn),
                "predicted_logical_z": pred_lz,
                "z_circ": zc, "x_circ": xc,
            })
    return items


# ---------------------------------------------------------------------------
# Decode + verdict
# ---------------------------------------------------------------------------

def _counts_to_dists(z_counts, x_counts):
    """Return (z_dist over (z_fano, lz), x_dist over x_fano)."""
    z_dist, x_dist = Counter(), Counter()
    for bs, n in z_counts.items():
        sb = _sysbits_from_qiskit(bs)
        z_dist[(_fano_from_sysbits(sb), _logical_z_from_sysbits(sb))] += n
    for bs, n in x_counts.items():
        sb = _sysbits_from_qiskit(bs)
        x_dist[_fano_from_sysbits(sb)] += n
    return z_dist, x_dist


def verdict(item, z_counts, x_counts):
    z_dist, x_dist = _counts_to_dists(z_counts, x_counts)
    n_z = sum(z_dist.values()) or 1
    n_x = sum(x_dist.values()) or 1
    pred_x_fano, pred_z_fano = item["predicted_syndrome"]
    pred_lz = item["predicted_logical_z"]

    modal_x_fano, modal_x_n = x_dist.most_common(1)[0]
    (modal_z_fano, modal_z_lz), modal_z_n = z_dist.most_common(1)[0]

    x_pass = (modal_x_fano == pred_x_fano)

    if pred_lz == 0:  # h7: require (I,I) modal syndrome + ~50/50 logical-Z split
        n_plus = sum(v for (zf, lz), v in z_dist.items() if zf == pred_z_fano and lz == +1)
        n_minus = sum(v for (zf, lz), v in z_dist.items() if zf == pred_z_fano and lz == -1)
        tot = n_plus + n_minus
        pct_plus = (n_plus / tot) if tot else None
        z_pass = (modal_z_fano == pred_z_fano) and (pct_plus is not None and 0.35 <= pct_plus <= 0.65)
        lz_note = {"n_plus": n_plus, "n_minus": n_minus, "pct_plus": pct_plus}
    else:
        z_syn_ok = (modal_z_fano == pred_z_fano)
        lz_ok = (pred_lz not in (+1, -1)) or (modal_z_lz == pred_lz)
        z_pass = z_syn_ok and lz_ok
        lz_note = None

    return {
        "x_half": {
            "modal_fano": modal_x_fano, "modal_prob": modal_x_n / n_x,
            "predicted_fano": pred_x_fano,
            "pred_bin_prob": x_dist.get(pred_x_fano, 0) / n_x,
            "pass": bool(x_pass),
            "distribution": [[k, v] for k, v in x_dist.most_common(10)],
        },
        "z_half": {
            "modal_fano": modal_z_fano, "modal_lz": modal_z_lz,
            "modal_prob": modal_z_n / n_z,
            "predicted_fano": pred_z_fano, "predicted_lz": pred_lz,
            "pred_bin_prob": z_dist.get((pred_z_fano, pred_lz), 0) / n_z if pred_lz in (+1, -1) else None,
            "pass": bool(z_pass),
            "lz_note": lz_note,
            "distribution": [[list(k), v] for k, v in z_dist.most_common(10)],
        },
        "pass": bool(x_pass and z_pass),
    }


def _print_item(item, v):
    xh, zh = v["x_half"], v["z_half"]
    mark = "PASS" if v["pass"] else "FAIL"
    print(f"  {item['label']:<18} "
          f"X[{xh['modal_fano']:>2} p={xh['modal_prob']:.2f} pred={xh['predicted_fano']:>2} "
          f"{'ok' if xh['pass'] else 'XX'}]  "
          f"Z[{zh['modal_fano']:>2},{zh['modal_lz']:+d} p={zh['modal_prob']:.2f} "
          f"pred={zh['predicted_fano']:>2},{zh['predicted_lz']} "
          f"{'ok' if zh['pass'] else 'XX'}]  -> {mark}")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_dry_run(items, shots):
    name, backend = make_dry_run_backend()
    print(f"\n  Dry-run on {name} ({shots} shots/circuit)")
    results = []
    for it in items:
        zt = transpile(it["z_circ"], backend=backend, optimization_level=1)
        xt = transpile(it["x_circ"], backend=backend, optimization_level=1)
        zc = backend.run(zt, shots=shots).result().get_counts()
        xc = backend.run(xt, shots=shots).result().get_counts()
        v = verdict(it, zc, xc)
        _print_item(it, v)
        results.append({**{k: it[k] for k in it if not k.endswith("_circ")}, "result": v})
    return results, name


def run_hardware(items, backend_name, shots, submit: bool):
    service, channel = make_hardware_service()
    backend = get_hardware_backend(service, backend_name)
    # Transpile both halves of every item
    flat = []  # (item_idx, basis, transpiled)
    print(f"\n  Transpiling {2*len(items)} circuits (opt_level=3)...")
    for i, it in enumerate(items):
        for basis, circ in (("z", it["z_circ"]), ("x", it["x_circ"])):
            t = transpile(circ, backend=backend, optimization_level=3)
            flat.append((i, basis, t))
            print(f"    {it['label']+'_'+basis:<20} depth={t.depth():<3} n_2q={t.num_nonlocal_gates()}")
    if not submit:
        print("\n  ★ Staged, NOT submitted. Re-run with --mode hardware-submit.")
        return None, backend_name

    from qiskit_ibm_runtime import SamplerV2 as Sampler
    print(f"\n  ★★★ SUBMITTING {len(flat)} circuits to {backend.name} ({shots} shots) ★★★")
    sampler = Sampler(mode=backend)
    job = sampler.run([t for _, _, t in flat], shots=shots)
    job_id = job.job_id()
    print(f"  Job id={job_id}; polling...")
    res = job.result()
    print("  Job complete. Decoding...")
    # Regroup counts by item
    by_item = {i: {} for i in range(len(items))}
    for k, (i, basis, _) in enumerate(flat):
        by_item[i][basis] = res[k].data.c.get_counts()
    results = []
    for i, it in enumerate(items):
        v = verdict(it, by_item[i]["z"], by_item[i]["x"])
        v["job_id"] = job_id
        _print_item(it, v)
        results.append({**{k: it[k] for k in it if not k.endswith("_circ")}, "result": v})
    return results, backend_name


def save(results, mode, backend_label, shots):
    out_dir = Path(__file__).parent / "output" / "heron_runs" / "exp12_destructive"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"exp12_phase2_destructive_{mode}_{backend_label}_{stamp}.json"
    payload = {"mode": mode, "backend": backend_label, "shots_per_circuit": shots,
               "timestamp": stamp, "variant": "ancilla-free destructive CSS readout",
               "phase2_targets": PHASE2_TARGETS, "circuits": results}
    if results is not None:
        payload["n_pass"] = sum(1 for r in results if r["result"]["pass"])
        payload["n_total"] = len(results)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  Saved: {out}")
    if results is not None:
        print(f"  Overall: {payload['n_pass']}/{payload['n_total']} PASS")
    return out


def main():
    ap = argparse.ArgumentParser(description="Exp 12 Phase 2 destructive (ancilla-free) variant")
    ap.add_argument("--mode", choices=["dry-run", "hardware-stage", "hardware-submit"], default="dry-run")
    ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--no-controls", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("  EXP 12 PHASE 2 — ANCILLA-FREE DESTRUCTIVE READOUT")
    print(f"  Mode: {args.mode}  Backend: {args.backend}  Shots: {args.shots}")
    print("=" * 72)

    items = build_items(include_controls=not args.no_controls)
    print(f"\n  Built {len(items)} items -> {2*len(items)} circuits (Z + X readout each)")

    if args.mode == "dry-run":
        results, label = run_dry_run(items, args.shots)
        save(results, args.mode, label, args.shots)
    else:
        results, label = run_hardware(items, args.backend, args.shots,
                                       submit=(args.mode == "hardware-submit"))
        if results is not None:
            save(results, args.mode, label, args.shots)

    print("\n" + "=" * 72)
    print(f"  Done ({args.mode}).")
    print("=" * 72)


if __name__ == "__main__":
    main()
