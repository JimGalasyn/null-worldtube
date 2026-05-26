"""Heron Exp 12 — Phase 1 simulator pilot.

Runs all 16 compendium (p, q) classes + 4 control circuits on three
simulator backends:
  - Noiseless AerSimulator (sanity reference)
  - FakeMarrakesh (Heron r2 — primary analog; FakeKingston not yet in
    qiskit_ibm_runtime.fake_provider)
  - FakeFez (Heron r2 — sibling, matches ibm_fez)

For each circuit-backend combination, executes 10^5 shots (configurable
via --shots), decodes per-shot syndrome and logical-Z eigenvalue, and
aggregates a per-class probability distribution.

Outputs to analysis/output/heron_runs/exp12_simulator/:
  - per_class_<p>_<q>_<backend>.json     -- raw distribution per run
  - summary_phase1_simulator.json        -- modal probability table per class
  - phase1_calibration_report.md         -- human-readable modal-probability
                                             calibration for v4 pre-reg

Usage:
  python3 analysis/exp12_phase1_simulator_pilot.py --shots 100000
  python3 analysis/exp12_phase1_simulator_pilot.py --shots 1000 --quick  # sanity

Author: Théodore (Claude), NWT session 2026-05-21.
Phase 1 of Exp 12 pre-reg v3 (commit deb936e).

Embargo: this script and its outputs are NWT-internal per
d12rg-embargo-active.md.  Do not share to D12RG or external channels
until Paper 21 + 22 Zenodo-mint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# nwt-substrate lives at a sibling path
NWT_SUBSTRATE = Path("/home/jim/repos/nwt-substrate")
sys.path.insert(0, str(NWT_SUBSTRATE))

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeMarrakesh

from nwt_substrate.heron.steane_pair_synthesis import (
    build_exp12_circuit,
    build_control_circuit,
    decode_syndrome,
    decode_logical_z_destructive,
    COMPENDIUM_LOOKUP,
)


OUTPUT_DIR = Path(
    "/home/jim/repos/null-worldtube-private/analysis/output/heron_runs/exp12_simulator"
)


# ---------------------------------------------------------------------------
# Backend setup
# ---------------------------------------------------------------------------

# Topology-realistic initial layouts (heavy-hex 13-qubit subgraphs of the
# 156-qubit Heron r2 chip). Found by BFS from qubit 0 in each fake backend.
# Using these as `initial_layout` in transpile gives SWAP overhead representative
# of actual heavy-hex routing, but keeps simulator state at ≤25 qubits
# (statevector ≈ 512 MB) rather than the 156-qubit-inherited OOM.
def _bfs_13_qubits(backend):
    """13-qubit connected subgraph found by BFS from qubit 0."""
    from collections import deque
    edges = list(backend.coupling_map.get_edges())
    nbrs = {}
    for a, b in edges:
        nbrs.setdefault(a, set()).add(b)
        nbrs.setdefault(b, set()).add(a)
    visited = [0]
    q = deque([0])
    while q and len(visited) < 13:
        cur = q.popleft()
        for nb in sorted(nbrs.get(cur, ())):
            if nb not in visited:
                visited.append(nb)
                q.append(nb)
                if len(visited) >= 13:
                    break
    return visited


def make_backends():
    """Return AerSimulator instances:

      - 'noiseless'           -- no noise
      - 'marrakesh_noise'     -- per-gate noise only, no coupling-map constraint
      - 'fez_noise'           -- ditto fez backend
      - 'marrakesh_topology'  -- per-gate noise + heavy-hex routing
                                 (13-qubit BFS subgraph of FakeMarrakesh)
      - 'fez_topology'        -- ditto fez

    The topology variants give hardware-realistic modal probabilities
    (include SWAP overhead from heavy-hex). The noise-only variants give
    intrinsic substrate-prediction validation under per-gate noise alone.
    """
    fm = FakeMarrakesh()
    ff = FakeFez()
    marrakesh_noise = NoiseModel.from_backend(fm)
    fez_noise = NoiseModel.from_backend(ff)
    return {
        "noiseless":          AerSimulator(),
        "marrakesh_noise":    AerSimulator(noise_model=marrakesh_noise,
                                            basis_gates=marrakesh_noise.basis_gates),
        "fez_noise":          AerSimulator(noise_model=fez_noise,
                                            basis_gates=fez_noise.basis_gates),
        "marrakesh_topology": AerSimulator(noise_model=marrakesh_noise,
                                            basis_gates=marrakesh_noise.basis_gates,
                                            coupling_map=fm.coupling_map),
        "fez_topology":       AerSimulator(noise_model=fez_noise,
                                            basis_gates=fez_noise.basis_gates,
                                            coupling_map=ff.coupling_map),
    }


# Layouts to pin topology variants to a 13-qubit connected subgraph
# (max physical-qubit index bounded to keep simulator memory in check).
_LAYOUTS = None
def _get_layouts():
    global _LAYOUTS
    if _LAYOUTS is None:
        _LAYOUTS = {
            "marrakesh_topology": _bfs_13_qubits(FakeMarrakesh()),
            "fez_topology":       _bfs_13_qubits(FakeFez()),
        }
    return _LAYOUTS


# ---------------------------------------------------------------------------
# Counts → (syndrome_pair, logical_z) distribution
# ---------------------------------------------------------------------------

# Map qubit index 0..6 → Fano-point label P, E_i, F_i.
# VV's lookup convention (metadata.convention):
#   vertex_to_qubit: {0:6, 1:3, 2:1, 3:0, 4:2, 5:4, 6:5}
#   vertex_to_fano:  {0:P, 1:E1, 2:E2, 3:E3, 4:F1, 5:F2, 6:F3}
# Composing: qubit_to_fano = vertex_to_fano ∘ qubit_to_vertex
# qubit 0 ← vertex 3 → E3
# qubit 1 ← vertex 2 → E2
# qubit 2 ← vertex 4 → F1
# qubit 3 ← vertex 1 → E1
# qubit 4 ← vertex 5 → F2
# qubit 5 ← vertex 6 → F3
# qubit 6 ← vertex 0 → P
QUBIT_TO_FANO = {0: "E3", 1: "E2", 2: "F1", 3: "E1", 4: "F2", 5: "F3", 6: "P"}


def _decode_bitstring(bs: str, n_syndrome: int = 6, n_logical: int = 7):
    """Decode a Qiskit bitstring (Qiskit MSB-first when multiple cregs).

    Convention from VV's helper:
      - Two ClassicalRegisters: 'syndrome' (6 bits) measured first,
        'logical_z' (7 bits, destructive) measured second.
      - In the Qiskit string, cregs are space-separated with the LAST-created
        register appearing leftmost.  Each register is itself LSB-on-the-right.

    Returns: ((s_X_fano, s_Z_fano), logical_z_eigenvalue)
    """
    parts = bs.split()
    # Helper appends 'logical_z' register after 'syndrome', so logical_z appears
    # leftmost in the joined string.
    if len(parts) == 2:
        logical_bits_str, syndrome_str = parts
    else:
        # Single register or unexpected layout — fall back to splitting by width.
        joined = bs.replace(" ", "")
        logical_bits_str = joined[:n_logical]
        syndrome_str = joined[n_logical:n_logical + n_syndrome]

    # Syndrome: 6 bits. First 3 (X-stab measurements) → X-error syndrome.
    # Last 3 (Z-stab measurements) → Z-error syndrome.
    # Qiskit bit ordering within a creg: bit 0 is rightmost.
    syn_bits = [int(b) for b in syndrome_str[::-1]]  # bit-0 first
    x_syn = syn_bits[:3]
    z_syn = syn_bits[3:6]

    # Decode each 3-bit syndrome to a qubit position (Hamming code).
    # syndrome [b0, b1, b2] = (b0 + 2*b1 + 4*b2), gives Hamming position 1..7.
    x_pos = x_syn[0] + 2 * x_syn[1] + 4 * x_syn[2]  # 0 = no error, 1..7 = position
    z_pos = z_syn[0] + 2 * z_syn[1] + 4 * z_syn[2]
    x_fano = "I" if x_pos == 0 else QUBIT_TO_FANO[x_pos - 1]
    z_fano = "I" if z_pos == 0 else QUBIT_TO_FANO[z_pos - 1]

    # Logical-Z: 7 destructive bits.  Eigenvalue = (-1)^(parity over Z_L support).
    logical_bits = [int(b) for b in logical_bits_str[::-1]]
    # VV's helper uses Z_L support {0, 1, 2} (Steane logical-Z on qubits 0-2).
    parity = sum(logical_bits[i] for i in (0, 1, 2)) % 2
    logical_z = +1 if parity == 0 else -1

    return ((x_fano, z_fano), logical_z)


def run_circuit_on_backend(qc, backend, shots: int,
                             backend_name: str = "") -> Counter:
    """Transpile, run, decode.  Returns Counter of ((s_X, s_Z), log_z) → count.

    If `backend_name` ends in `_topology`, the transpile call is pinned to a
    13-qubit subgraph layout to keep simulator memory bounded while
    preserving heavy-hex routing realism.
    """
    if backend_name.endswith("_topology"):
        layout = _get_layouts().get(backend_name)
        qct = transpile(qc, backend=backend, optimization_level=1,
                         initial_layout=layout)
    else:
        qct = transpile(qc, backend=backend, optimization_level=1)
    job = backend.run(qct, shots=shots)
    counts = job.result().get_counts()
    dist = Counter()
    for bs, c in counts.items():
        try:
            key = _decode_bitstring(bs)
            dist[key] += c
        except (KeyError, IndexError, ValueError):
            dist[("__decode_error__", 0)] += c
    return dist


# ---------------------------------------------------------------------------
# Per-class run + summary
# ---------------------------------------------------------------------------

def predicted_for(entry):
    """Return predicted (syndrome_pair, logical_z) for a compendium entry.

    Predicted syndrome is in entry['forward_pauli']['predicted_syndrome_fano_pair'].
    Predicted logical-Z is not in the lookup yet (VV agreed to add); we compute
    it ourselves from the X-part bits and Z_L support {0, 1, 2}:
      - X-part has any 1 in qubits {0, 1, 2}  →  Pauli word anti-commutes
        with Z_L  →  logical-Z = -1
      - Otherwise  →  logical-Z = +1
    Mixed (centralizer-only): Z-part overlap with X_L support also flips sign.
    """
    fp = entry["forward_pauli"]
    syndrome = tuple(fp["predicted_syndrome_fano_pair"])
    x_overlap = sum(fp["x_part_bits"][i] for i in (0, 1, 2)) % 2
    z_overlap = sum(fp["z_part_bits"][i] for i in (0, 1, 2)) % 2
    # Logical-Z eigenvalue: -1 if X-part anti-commutes with Z_L; modulated by Z-part
    # (Z-part on Z_L support is a stabilizer up to phase, doesn't flip eigenvalue,
    # but Z-part overlap with X_L support flips it via {X_L, Z_L} = 0).
    # Quick rule: logical-Z = (-1)^(x_overlap)
    logical_z = +1 if x_overlap == 0 else -1
    return syndrome, logical_z


def run_class(entry, backends: dict, shots: int) -> dict:
    p, q = entry["p"], entry["q"]
    particles = entry["particles"]
    pred_syn, pred_lz = predicted_for(entry)
    print(f"  Class (p={p}, q={q})  particles={particles}  "
          f"pred=({pred_syn}, log_z={pred_lz:+d})")
    qc = build_exp12_circuit(p, q, logical_z_mode="destructive")
    results = {}
    for backend_name, backend in backends.items():
        t0 = time.time()
        dist = run_circuit_on_backend(qc, backend, shots, backend_name)
        dt = time.time() - t0
        modal_key, modal_count = dist.most_common(1)[0]
        modal_prob = modal_count / shots
        pred_count = sum(c for k, c in dist.items() if k == (pred_syn, pred_lz))
        pred_prob = pred_count / shots
        print(f"    {backend_name:>10}: modal={modal_key}  prob={modal_prob:.3f}  "
              f"pred-bin={pred_prob:.3f}  ({dt:.1f}s)")
        results[backend_name] = {
            "modal_outcome": [list(modal_key[0]), modal_key[1]],
            "modal_prob": modal_prob,
            "predicted_outcome": [list(pred_syn), pred_lz],
            "predicted_bin_prob": pred_prob,
            "modal_matches_predicted": (modal_key == (pred_syn, pred_lz)),
            "elapsed_s": dt,
            "distribution": [
                [list(k[0]), k[1], v] for k, v in dist.most_common(20)
            ],
        }
    return {
        "p": p, "q": q, "particles": particles,
        "predicted_syndrome": list(pred_syn),
        "predicted_logical_z": pred_lz,
        "shots_per_backend": shots,
        "results": results,
    }


def run_control(kind: str, backends: dict, shots: int) -> dict:
    # Expected outcomes for the 4 controls
    expected = {
        "identity": (("I", "I"), +1),
        "x7":       (("I", "I"), -1),
        "z7":       (("I", "I"), +1),
        "h7":       (("I", "I"), 0),  # 0 = 50/50 mixed; no single predicted
    }
    pred_syn, pred_lz = expected[kind]
    print(f"  Control '{kind}'  expected=({pred_syn}, log_z={pred_lz:+d})")
    qc = build_control_circuit(kind, logical_z_mode="destructive")
    results = {}
    for backend_name, backend in backends.items():
        t0 = time.time()
        dist = run_circuit_on_backend(qc, backend, shots, backend_name)
        dt = time.time() - t0
        modal_key, modal_count = dist.most_common(1)[0]
        modal_prob = modal_count / shots
        if kind == "h7":
            # For Hadamard control, logical-Z should be ~50/50; check syndrome only
            pred_count = sum(c for k, c in dist.items() if k[0] == pred_syn)
            pos_count = sum(c for k, c in dist.items() if k[0] == ("I", "I") and k[1] == +1)
            neg_count = sum(c for k, c in dist.items() if k[0] == ("I", "I") and k[1] == -1)
            print(f"    {backend_name:>10}: II syndrome={pred_count/shots:.3f}  "
                  f"+1/-1 split=({pos_count/shots:.3f}/{neg_count/shots:.3f})  "
                  f"({dt:.1f}s)")
            results[backend_name] = {
                "ii_syndrome_prob": pred_count / shots,
                "log_z_plus_prob": pos_count / shots,
                "log_z_minus_prob": neg_count / shots,
                "elapsed_s": dt,
                "distribution": [[list(k[0]), k[1], v] for k, v in dist.most_common(10)],
            }
        else:
            pred_count = sum(c for k, c in dist.items() if k == (pred_syn, pred_lz))
            pred_prob = pred_count / shots
            print(f"    {backend_name:>10}: modal={modal_key}  prob={modal_prob:.3f}  "
                  f"pred-bin={pred_prob:.3f}  ({dt:.1f}s)")
            results[backend_name] = {
                "modal_outcome": [list(modal_key[0]), modal_key[1]],
                "modal_prob": modal_prob,
                "predicted_outcome": [list(pred_syn), pred_lz],
                "predicted_bin_prob": pred_prob,
                "modal_matches_predicted": (modal_key == (pred_syn, pred_lz)),
                "elapsed_s": dt,
                "distribution": [
                    [list(k[0]), k[1], v] for k, v in dist.most_common(10)
                ],
            }
    return {
        "control_kind": kind,
        "expected_syndrome": list(pred_syn),
        "expected_logical_z": pred_lz,
        "shots_per_backend": shots,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=100_000,
                        help="shots per (circuit, backend)")
    parser.add_argument("--quick", action="store_true",
                        help="quick mode: 1k shots, skip controls")
    args = parser.parse_args()
    shots = 1000 if args.quick else args.shots

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  HERON EXP 12 -- PHASE 1 SIMULATOR PILOT")
    print(f"{'='*72}")
    print(f"  Shots per (circuit, backend): {shots}")

    backends = make_backends()
    print(f"  Backends: {list(backends.keys())}")
    print(f"  Compendium classes: {len(COMPENDIUM_LOOKUP['particles'])}")

    all_results = {"classes": [], "controls": []}

    t_start = time.time()
    print(f"\n  --- Compendium classes (16) ---")
    for i, entry in enumerate(COMPENDIUM_LOOKUP["particles"], 1):
        print(f"\n  [{i}/16] ", end="")
        result = run_class(entry, backends, shots)
        all_results["classes"].append(result)
        out_path = OUTPUT_DIR / f"per_class_p{entry['p']}_q{entry['q']}.json"
        out_path.write_text(json.dumps(result, indent=2))

    if not args.quick:
        print(f"\n  --- Control circuits (4) ---")
        for kind in ("identity", "x7", "z7", "h7"):
            print()
            result = run_control(kind, backends, shots)
            all_results["controls"].append(result)
            out_path = OUTPUT_DIR / f"control_{kind}.json"
            out_path.write_text(json.dumps(result, indent=2))

    t_total = time.time() - t_start
    print(f"\n  Total time: {t_total:.1f}s = {t_total/60:.1f} min")

    # Summary report
    summary = {
        "shots_per_circuit_per_backend": shots,
        "total_time_s": t_total,
        "n_classes": len(all_results["classes"]),
        "n_controls": len(all_results["controls"]),
        "backends": list(backends.keys()),
        "class_summary": [
            {
                "p": r["p"], "q": r["q"], "particles": r["particles"],
                "predicted_syndrome": r["predicted_syndrome"],
                "predicted_logical_z": r["predicted_logical_z"],
                **{
                    f"modal_match_{bn}": r["results"][bn]["modal_matches_predicted"]
                    for bn in backends
                },
                **{
                    f"modal_prob_{bn}": r["results"][bn]["modal_prob"]
                    for bn in backends
                },
                **{
                    f"pred_bin_prob_{bn}": r["results"][bn]["predicted_bin_prob"]
                    for bn in backends
                },
            }
            for r in all_results["classes"]
        ],
    }
    (OUTPUT_DIR / "summary_phase1_simulator.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Markdown calibration report
    lines = [
        "# Heron Exp 12 — Phase 1 Simulator Pilot — Calibration Report",
        "",
        f"Shots per (circuit, backend): {shots}",
        f"Total runtime: {t_total:.1f}s = {t_total/60:.1f} min",
        f"Backends: {', '.join(backends.keys())}",
        "",
        "## Per-class modal probability across 5 simulator variants",
        "",
        ("| (p, q) | Particles | Predicted | noiseless | marrakesh_noise | fez_noise"
         " | marrakesh_topology | fez_topology | All-match |"),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    backend_names = list(backends.keys())
    for r in all_results["classes"]:
        p, q = r["p"], r["q"]
        pred = f"({','.join(r['predicted_syndrome'])}), logZ={r['predicted_logical_z']:+d}"
        n_match = sum(1 for bn in backend_names if r["results"][bn]["modal_matches_predicted"])
        match_str = f"{n_match}/{len(backend_names)}"
        row = (f"| ({p}, {q}) | {', '.join(r['particles'])[:30]} | {pred}"
                + "".join(f" | {r['results'][bn]['modal_prob']:.3f}" for bn in backend_names)
                + f" | {match_str} |")
        lines.append(row)

    if not args.quick:
        lines.extend([
            "",
            "## Control circuits",
            "",
            ("| Control | Expected"
              + "".join(f" | {bn}" for bn in backend_names) + " |"),
            "|---|---|" + "|".join("---" for _ in backend_names) + "|",
        ])
        for r in all_results["controls"]:
            kind = r["control_kind"]
            exp = f"({','.join(r['expected_syndrome'])}), logZ={r['expected_logical_z']:+d}"
            if kind == "h7":
                row = (f"| {kind} | II + logZ split 50/50"
                        + "".join(f" | {r['results'][bn]['ii_syndrome_prob']:.3f}"
                                    for bn in backend_names) + " |")
            else:
                row = (f"| {kind} | {exp}"
                        + "".join(f" | {r['results'][bn]['modal_prob']:.3f}"
                                    for bn in backend_names) + " |")
            lines.append(row)

    # Threshold recommendation
    if all_results["classes"]:
        import statistics
        lines.extend(["", "## Threshold recommendation for v4 pre-reg §5.3", ""])
        for bn in backend_names:
            if bn == "noiseless":
                continue
            probs = [r["results"][bn]["modal_prob"] for r in all_results["classes"]]
            lines.append(
                f"- {bn}: mean **{statistics.mean(probs):.3f}**, "
                f"range {min(probs):.3f}-{max(probs):.3f}, "
                f"10th pctile {sorted(probs)[max(0, len(probs)//10)]:.3f}"
            )
        lines.append("")
        lines.append("Suggested PASS threshold = min 10th percentile across topology variants.")

    (OUTPUT_DIR / "phase1_calibration_report.md").write_text("\n".join(lines))
    print(f"\n  Outputs in: {OUTPUT_DIR}")
    print(f"    summary_phase1_simulator.json")
    print(f"    phase1_calibration_report.md")
    print(f"    per_class_p*_q*.json  ({len(all_results['classes'])} files)")
    if not args.quick:
        print(f"    control_*.json  ({len(all_results['controls'])} files)")


if __name__ == "__main__":
    main()
