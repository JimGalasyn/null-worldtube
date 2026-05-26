"""Heron Exp 12 — Phase 2 hardware confirmation.

Submits the 4 highest-diagnostic compendium (p, q) classes to real IBM
Heron hardware for cross-backend coincidence test per pre-reg DRAFT v3
(commit deb936e).

Phase 2 target classes (4 classes × 2000 shots × 2 backends = 8k shots/backend):

  | Class | Particle | Predicted (s_X, s_Z) | Logical-Z | Diagnostic |
  |-------|----------|------------------------|-----------|------------|
  | (1, 3) | proton  | (I, I)                | -1        | codespace baseline |
  | (2, 1) | e-      | (F2, E2)              | n/a       | off-diagonal lepton |
  | (3, 5) | pi+     | (E1, E1)              | n/a       | diagonal meson |
  | (1, 8) | mu-     | (P, F3)               | n/a       | cross-sector equiv. half |

Phase 1 simulator pilot calibration (confirmed all 4 classes at >68% modal prob
across 5 simulator backends; minimum 0.685 at marrakesh_topology for proton).

Run modes:

  --mode dry-run        Use AerSimulator (FakeKingston noise) — smoke test
                        (no IBM credentials touched, no hardware queue impact)
  --mode hardware-stage Instantiate QiskitRuntimeService, identify backend,
                        transpile circuits, verify queue depth. STOP before
                        sampler.run() — Jim reviews staged jobs before
                        explicit hardware-submit.
  --mode hardware-submit Actually submit transpiled circuits to IBM hardware.
                         WARNING: consumes free-tier quantum time.

Usage:
  # Smoke test:
  python3 analysis/exp12_phase2_hardware.py --mode dry-run --shots 100

  # Pre-flight verification (no quantum time consumed):
  python3 analysis/exp12_phase2_hardware.py --mode hardware-stage \\
      --backend ibm_kingston --shots 2000

  # Actual submission (BURNS free-tier time):
  python3 analysis/exp12_phase2_hardware.py --mode hardware-submit \\
      --backend ibm_kingston --shots 2000

Token at ~/.ibm_quantum_api_key.

Author: Théodore (Claude), NWT session 2026-05-22.
Phase 2 of Exp 12 pre-reg v3.

Embargo: NWT-internal per d12rg-embargo-active.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# nwt-substrate lives at a sibling path
NWT_SUBSTRATE = Path("/home/jim/repos/nwt-substrate")
sys.path.insert(0, str(NWT_SUBSTRATE))

from qiskit import transpile

from nwt_substrate.heron.steane_pair_synthesis import (
    build_exp12_circuit,
    build_control_circuit,
    COMPENDIUM_LOOKUP,
)


# Reuse Phase 1 decoder
from exp12_phase1_simulator_pilot import (
    QUBIT_TO_FANO,
    _decode_bitstring,
    predicted_for,
)


OUTPUT_DIR = Path(
    "/home/jim/repos/null-worldtube-private/analysis/output/heron_runs/exp12_hardware"
)

TOKEN_PATH = Path.home() / ".ibm_quantum_api_key"


# Phase 2 target classes per pre-reg v3 §3.3
PHASE2_TARGETS = [(1, 3), (2, 1), (3, 5), (1, 8)]


# ---------------------------------------------------------------------------
# Backend setup per mode
# ---------------------------------------------------------------------------

def make_dry_run_backend():
    """Return a single AerSimulator with FakeKingston-like noise for smoke
    testing. Mirrors marrakesh_noise backend from Phase 1 since FakeKingston
    isn't directly available in qiskit_ibm_runtime.fake_provider."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    fm = FakeMarrakesh()
    nm = NoiseModel.from_backend(fm)
    return ("dry-run-fake-marrakesh",
             AerSimulator(noise_model=nm, basis_gates=nm.basis_gates))


def load_ibm_token() -> str:
    """Load IBM Quantum API token from ~/.ibm_quantum_api_key."""
    if not TOKEN_PATH.exists():
        raise FileNotFoundError(
            f"IBM Quantum API key not found at {TOKEN_PATH}. "
            f"Place token in this file (chmod 600).")
    return TOKEN_PATH.read_text().strip()


def make_hardware_service():
    """Instantiate QiskitRuntimeService using token from ~/.ibm_quantum_api_key."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = load_ibm_token()
    # IBM Cloud is the current free-tier channel (after IBM Quantum Platform
    # migration). Falls back to ibm_quantum if the cloud channel fails.
    for channel in ("ibm_cloud", "ibm_quantum"):
        try:
            service = QiskitRuntimeService(channel=channel, token=token)
            print(f"  Connected to QiskitRuntimeService via channel={channel}")
            return service, channel
        except Exception as e:
            print(f"  channel={channel} failed: {type(e).__name__}: {e}")
            continue
    raise RuntimeError("Could not connect to any IBM channel with provided token")


def get_hardware_backend(service, backend_name: str):
    """Get backend handle by name, verify it's operational, report queue."""
    try:
        backend = service.backend(backend_name)
    except Exception as e:
        raise RuntimeError(
            f"Backend {backend_name} not accessible via this token's account: {e}")
    status = backend.status()
    queue_depth = status.pending_jobs
    operational = status.operational
    n_qubits = backend.num_qubits
    print(f"  Backend: {backend_name}")
    print(f"    operational: {operational}")
    print(f"    queue depth: {queue_depth} jobs pending")
    print(f"    n_qubits: {n_qubits}")
    if not operational:
        raise RuntimeError(f"Backend {backend_name} is not operational")
    return backend


# ---------------------------------------------------------------------------
# Circuit build + transpile
# ---------------------------------------------------------------------------

def build_phase2_circuits(include_controls: bool = True):
    """Return list of (label, circuit, predicted_outcome) tuples for Phase 2.

    4 compendium classes + 4 controls (identity, x7, z7, h7).
    """
    # COMPENDIUM_LOOKUP['particles'] is a list of entries; build a (p,q)→entry map
    pq_to_entry = {(e["p"], e["q"]): e for e in COMPENDIUM_LOOKUP["particles"]}
    circuits = []
    for p, q in PHASE2_TARGETS:
        entry = pq_to_entry.get((p, q))
        if entry is None:
            raise KeyError(f"No compendium entry for (p, q) = ({p}, {q})")
        qc = build_exp12_circuit(p, q, logical_z_mode="destructive")
        pred_syn, pred_lz = predicted_for(entry)
        circuits.append({
            "label": f"class_p{p}_q{q}",
            "p": p, "q": q,
            "particles": entry["particles"],
            "circuit": qc,
            "predicted_syndrome": list(pred_syn),
            "predicted_logical_z": pred_lz,
        })
    if include_controls:
        control_predicted = {
            "identity": (("I", "I"), +1),
            "x7":       (("I", "I"), -1),
            "z7":       (("I", "I"), +1),
            "h7":       (("I", "I"), 0),
        }
        for kind in ["identity", "x7", "z7", "h7"]:
            qc = build_control_circuit(kind, logical_z_mode="destructive")
            pred_syn, pred_lz = control_predicted[kind]
            circuits.append({
                "label": f"control_{kind}",
                "p": None, "q": None,
                "particles": [f"control_{kind}"],
                "circuit": qc,
                "predicted_syndrome": list(pred_syn),
                "predicted_logical_z": pred_lz,
            })
    return circuits


def transpile_for_backend(circuits, backend, optimization_level: int = 1):
    """Transpile all circuits for the target backend. Returns list with
    'circuit_transpiled' added to each dict."""
    print(f"  Transpiling {len(circuits)} circuits "
          f"(optimization_level={optimization_level})...")
    for c in circuits:
        t0 = time.time()
        qct = transpile(c["circuit"], backend=backend,
                          optimization_level=optimization_level)
        dt = time.time() - t0
        c["circuit_transpiled"] = qct
        c["transpile_s"] = dt
        # Report depth, gate counts
        depth = qct.depth()
        n_2q = qct.num_nonlocal_gates()
        print(f"    {c['label']:<24} depth={depth:<4} n_2q={n_2q:<5} "
              f"({dt:.2f}s)")
    return circuits


# ---------------------------------------------------------------------------
# Execution per mode
# ---------------------------------------------------------------------------

def execute_dry_run(circuits, backend_name: str, backend, shots: int):
    """Execute via backend.run() (AerSimulator-compatible)."""
    print(f"\n  Executing {len(circuits)} circuits on {backend_name} "
          f"({shots} shots each)...")
    for c in circuits:
        t0 = time.time()
        qct = c["circuit_transpiled"]
        job = backend.run(qct, shots=shots)
        counts = job.result().get_counts()
        dt = time.time() - t0
        dist = Counter()
        for bs, cnt in counts.items():
            try:
                key = _decode_bitstring(bs)
                dist[key] += cnt
            except (KeyError, IndexError, ValueError):
                dist[("__decode_error__", 0)] += cnt
        modal_key, modal_count = dist.most_common(1)[0]
        modal_prob = modal_count / shots
        pred_syn = tuple(c["predicted_syndrome"])
        pred_lz = c["predicted_logical_z"]
        pred_count = sum(v for k, v in dist.items() if k == (pred_syn, pred_lz))
        pred_prob = pred_count / shots
        match = (modal_key == (pred_syn, pred_lz))
        print(f"    {c['label']:<24} modal={modal_key} "
              f"prob={modal_prob:.3f} pred-bin={pred_prob:.3f} "
              f"match={match} ({dt:.1f}s)")
        c["result"] = {
            "shots": shots,
            "modal_outcome": [list(modal_key[0]) if isinstance(modal_key[0], tuple)
                                 else modal_key[0], modal_key[1]],
            "modal_prob": modal_prob,
            "predicted_outcome": [c["predicted_syndrome"], pred_lz],
            "predicted_bin_prob": pred_prob,
            "modal_matches_predicted": bool(match),
            "elapsed_s": dt,
            "distribution": [
                [list(k[0]) if isinstance(k[0], tuple) else k[0], k[1], v]
                for k, v in dist.most_common(20)
            ],
        }
    return circuits


def stage_hardware_jobs(circuits, backend, shots: int):
    """Hardware-stage mode: prepare jobs but do NOT submit.

    Reports per-circuit transpiled depth, estimated quantum time,
    and prints a summary. Jim then reviews before opting in to
    hardware-submit.
    """
    print(f"\n  STAGING {len(circuits)} circuits for {backend.name} "
          f"({shots} shots each)...")
    total_2q = 0
    for c in circuits:
        qct = c["circuit_transpiled"]
        depth = qct.depth()
        n_2q = qct.num_nonlocal_gates()
        total_2q += n_2q * shots
        c["staged"] = {
            "shots": shots,
            "depth": depth,
            "n_2q_gates": n_2q,
            "expected_quantum_time_s_per_shot": n_2q * 1e-6,  # ~1 µs per 2q gate
        }
        print(f"    {c['label']:<24} depth={depth:<4} n_2q={n_2q:<5} "
              f"shots={shots}")

    # Rough quantum-time estimate: ~1 µs per 2-qubit gate, plus ~50 µs
    # per shot overhead (measurement, reset, transit).
    n_circuits = len(circuits)
    qtime_estimate_s = total_2q * 1e-6 + n_circuits * shots * 50e-6
    print(f"\n  Estimated total quantum time: {qtime_estimate_s:.1f} s = "
          f"{qtime_estimate_s/60:.1f} min")
    print(f"  Total 2-qubit gate-shots: {total_2q:,}")
    print(f"\n  ★ Staged but NOT submitted. Re-run with --mode hardware-submit")
    print(f"  ★ to actually consume quantum time.")
    return circuits


def submit_hardware_jobs(circuits, backend, shots: int):
    """Hardware-submit mode: actually call sampler.run() to consume quantum time."""
    print(f"\n  ★★★ SUBMITTING {len(circuits)} circuits to {backend.name} ★★★")
    print(f"      ({shots} shots each, this WILL consume free-tier quantum time)")

    from qiskit_ibm_runtime import SamplerV2 as Sampler
    sampler = Sampler(mode=backend)
    # Submit as a single batch (more efficient than per-circuit)
    pubs = [c["circuit_transpiled"] for c in circuits]
    print(f"\n  Submitting batch ({len(pubs)} circuits)...")
    t0 = time.time()
    job = sampler.run(pubs, shots=shots)
    job_id = job.job_id()
    print(f"  Job submitted: id={job_id}")
    print(f"  Polling for results (this can take minutes to hours on free tier)...")
    result = job.result()
    dt = time.time() - t0
    print(f"  Job completed in {dt:.1f}s (wall-clock incl. queue)")

    # Decode results per circuit
    for i, c in enumerate(circuits):
        pub_result = result[i]
        # SamplerV2 BitArray extraction. The circuit has TWO ClassicalRegisters
        # named 'c_syn' (6 bits, syndrome) and 'c_lz' (7 bits, logical-Z); each
        # appears as a separate BitArray on pub_result.data. The decoder
        # expects them paired per-shot in the format "<logical_bits> <syndrome_bits>"
        # (space-separated, matching the simulator's joined-register convention).
        dist = Counter()
        try:
            syn_strs = pub_result.data.c_syn.get_bitstrings()
            lz_strs = pub_result.data.c_lz.get_bitstrings()
            for syn_bs, lz_bs in zip(syn_strs, lz_strs):
                combined = f"{lz_bs} {syn_bs}"
                try:
                    key = _decode_bitstring(combined)
                    dist[key] += 1
                except (KeyError, IndexError, ValueError):
                    dist[("__decode_error__", 0)] += 1
        except AttributeError:
            # Fallback for older / different register layouts: try the first
            # register that exposes get_counts(), or join_data().
            counts = {}
            for reg_name in dir(pub_result.data):
                if reg_name.startswith('_'):
                    continue
                try:
                    reg_data = getattr(pub_result.data, reg_name)
                    if hasattr(reg_data, 'get_counts'):
                        counts = reg_data.get_counts()
                        break
                except Exception:
                    continue
            if not counts:
                try:
                    counts = pub_result.join_data().get_counts()
                except Exception:
                    pass
            for bs, cnt in counts.items():
                try:
                    key = _decode_bitstring(bs)
                    dist[key] += cnt
                except (KeyError, IndexError, ValueError):
                    dist[("__decode_error__", 0)] += cnt
        modal_key, modal_count = dist.most_common(1)[0] if dist else (None, 0)
        modal_prob = modal_count / shots if dist else 0.0
        pred_syn = tuple(c["predicted_syndrome"])
        pred_lz = c["predicted_logical_z"]
        pred_count = sum(v for k, v in dist.items() if k == (pred_syn, pred_lz))
        pred_prob = pred_count / shots if dist else 0.0
        match = (modal_key == (pred_syn, pred_lz)) if modal_key else False
        print(f"    {c['label']:<24} modal={modal_key} prob={modal_prob:.3f} "
              f"pred-bin={pred_prob:.3f} match={match}")
        c["result"] = {
            "job_id": job_id,
            "shots": shots,
            "modal_outcome": [list(modal_key[0]) if modal_key
                                 and isinstance(modal_key[0], tuple)
                                 else (modal_key[0] if modal_key else None),
                                modal_key[1] if modal_key else None],
            "modal_prob": modal_prob,
            "predicted_outcome": [c["predicted_syndrome"], pred_lz],
            "predicted_bin_prob": pred_prob,
            "modal_matches_predicted": bool(match),
            "distribution": [
                [list(k[0]) if isinstance(k[0], tuple) else k[0], k[1], v]
                for k, v in dist.most_common(20)
            ],
        }
    return circuits


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(circuits, mode: str, backend_label: str, shots: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = OUTPUT_DIR / f"exp12_phase2_{mode}_{backend_label}_{ts}.json"

    payload = {
        "mode": mode,
        "backend": backend_label,
        "shots_per_circuit": shots,
        "timestamp": ts,
        "phase2_targets": [list(t) for t in PHASE2_TARGETS],
        "circuits": [],
    }
    for c in circuits:
        entry = {k: v for k, v in c.items()
                  if k not in ("circuit", "circuit_transpiled")}
        payload["circuits"].append(entry)

    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  Results saved: {out_file}")
    return out_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Heron Exp 12 Phase 2")
    parser.add_argument("--mode",
                        choices=["dry-run", "hardware-stage", "hardware-submit"],
                        default="dry-run",
                        help="dry-run: AerSimulator smoke test (default); "
                             "hardware-stage: transpile + verify, do NOT submit; "
                             "hardware-submit: actually submit to IBM hardware.")
    parser.add_argument("--backend", default="ibm_kingston",
                        help="IBM backend name (only for hardware-* modes). "
                             "Try 'ibm_kingston' or 'ibm_fez'.")
    parser.add_argument("--shots", type=int, default=2000,
                        help="shots per circuit. Default 2000 per pre-reg v3.")
    parser.add_argument("--no-controls", action="store_true",
                        help="skip 4 control circuits (default: include them).")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  HERON EXP 12 -- PHASE 2 HARDWARE CONFIRMATION")
    print(f"{'='*72}")
    print(f"  Mode:    {args.mode}")
    print(f"  Backend: {args.backend if args.mode != 'dry-run' else '(simulator)'}")
    print(f"  Shots:   {args.shots} per circuit")
    print(f"  Controls included: {not args.no_controls}")

    # Build circuits
    print(f"\n  Building Phase 2 circuits...")
    circuits = build_phase2_circuits(include_controls=not args.no_controls)
    print(f"  Built {len(circuits)} circuits (4 targets + "
          f"{0 if args.no_controls else 4} controls)")

    if args.mode == "dry-run":
        backend_name, backend = make_dry_run_backend()
        circuits = transpile_for_backend(circuits, backend)
        circuits = execute_dry_run(circuits, backend_name, backend, args.shots)
        save_results(circuits, args.mode, backend_name, args.shots)
    elif args.mode in ("hardware-stage", "hardware-submit"):
        print(f"\n  Connecting to IBM Quantum...")
        service, channel = make_hardware_service()
        backend = get_hardware_backend(service, args.backend)
        circuits = transpile_for_backend(circuits, backend)
        if args.mode == "hardware-stage":
            circuits = stage_hardware_jobs(circuits, backend, args.shots)
            save_results(circuits, args.mode, args.backend, args.shots)
        else:
            circuits = submit_hardware_jobs(circuits, backend, args.shots)
            save_results(circuits, args.mode, args.backend, args.shots)

    print(f"\n{'='*72}")
    print(f"  Phase 2 {args.mode} complete.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
