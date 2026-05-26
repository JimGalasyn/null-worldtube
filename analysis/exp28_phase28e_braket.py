"""Exp 28 Phase 28e — AWS Braket port for cross-vendor / cross-architecture.

Vendor-neutral cross-platform submitter for the Phase 28e cross-vendor
substrate-monism syndrome-agreement test (see EXP28_PREREG.md v2.4 §4.6
and exp28_hsub_derivation.md). Reuses the same Phase 28a Qiskit circuit
builder via `--ancilla-budget` selection, then converts to OpenQASM 3
and wraps as a Braket Program. The 8-qubit single-ancilla layout is
required for AQT IBEX Q1 (12 qubits) and useful for any ≤12q backend.

Async submission pattern: submit ALL circuits first (so they queue in
parallel), then poll each task to completion. Task ARNs are written to
stdout so an orphaned shell can be resumed via --resume-task-arns.

Usage::

    # Local Braket simulator (free; validates circuit semantics)
    python3 exp28_phase28e_braket.py --mode dry-run

    # Cost preview for AQT IBEX Q1 (12q trapped-ion, eu-north-1)
    python3 exp28_phase28e_braket.py --mode hardware-stage \\
        --backend aqt/Ibex-Q1 --shots 100

    # Actually submit to AQT Ibex-Q1 (queues until Mon/Tue/Wed/Fri 08-15 UTC)
    python3 exp28_phase28e_braket.py --mode hardware-submit \\
        --backend aqt/Ibex-Q1 --shots 100

    # Resume an orphaned run (task ARNs printed by a prior submission)
    python3 exp28_phase28e_braket.py --mode hardware-submit \\
        --backend aqt/Ibex-Q1 --shots 100 \\
        --resume-task-arns arn1 arn2 ...

Writes results JSON to analysis/output/heron_runs/exp28_chip_hbar/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Reuse Qiskit-side circuit builder for the substrate Pauli + Steane prep
sys.path.insert(0, str(Path(__file__).parent))
from exp28_chip_hbar import (  # noqa: E402
    build_phase28a_circuits,
    _add_steane_zero_prep,
    _pauli_label_from_bits,
    predicted_for,
    COMPENDIUM_LOOKUP,
    PHASE_28A_TARGETS,
    STEANE_STAB_SUPPORTS,
)
from exp12_phase2_hardware_braket import QUBIT_TO_FANO  # noqa: E402

# Braket QPU pricing (from exp12 file, verified 2026-05-23)
BACKEND_INFO = {
    "aqt/Ibex-Q1":         ("eu-north-1", 0.0235,   0.30),
    "iqm/Garnet":          ("eu-north-1", 0.00145,  0.30),
    "iqm/Emerald":         ("eu-north-1", 0.0016,   0.30),
    "rigetti/Cepheus-1-108Q": ("us-west-1", 0.000425, 0.30),
}

OUTPUT_DIR = (Path(__file__).parent
              / "output" / "heron_runs" / "exp28_chip_hbar")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Qiskit-circuit → Braket-Program conversion
# ---------------------------------------------------------------------------

def qiskit_to_braket_program(qiskit_circuit):
    """Convert a Qiskit QuantumCircuit to a Braket OpenQASM 3 Program.

    Decomposes PauliEvolutionGate via Qiskit transpile to a Braket-friendly
    basis (cx, h, rz, s, sdg, x, reset, measure), then emits OpenQASM 3
    via qiskit.qasm3.dumps and wraps as braket.ir.openqasm.Program.
    """
    from braket.ir.openqasm import Program
    from qiskit import qasm3, transpile

    # Decompose to a basis Braket / AQT can route. Use cx (cnot equiv. in
    # OpenQASM stdgates.inc) + single-qubit primitives + reset + measure.
    decomp = transpile(
        qiskit_circuit,
        basis_gates=['cx', 'h', 'rz', 's', 'sdg', 'x', 'reset', 'measure'],
        optimization_level=0,
    )
    # includes=[] inlines `gate` definitions instead of emitting
    # `include "stdgates.inc";`, which the Braket OpenQASM interpreter
    # can't resolve (no filesystem `stdgates.inc` shipped with Braket SDK).
    qasm_text = qasm3.dumps(decomp, includes=[])
    return Program(source=qasm_text), decomp


def build_phase28e_braket_circuits(n_angles: int = 16,
                                    ancilla_budget: int = 1):
    """Build the 4-walk × n_angles Phase 28e circuit set as Braket Programs.

    Used for backends that support the standard Phase 28a layout (deferred
    measurement on 13q OR mid-circuit-measure-with-classical-bit-readout).

    NOTE: this path is unsuitable for AWS Braket QPU submission because
    Braket's result.measurement_counts returns only the final QUBIT state
    (8 bits for the 8q layout, 13 bits for the 13q layout). For the 8q
    layout the 6 syndrome bits are written to classical c_syn during
    mid-circuit measure but those bits are not exposed in Braket's result
    API. Use build_phase28e_aqt_split_circuits() for AQT IBEX Q1 and other
    ≤12q backends.

    Returns list of (Program, decomp_qiskit_circuit, meta_dict).
    """
    qiskit_items = build_phase28a_circuits(n_angles=n_angles,
                                           ancilla_budget=ancilla_budget)
    out = []
    for item in qiskit_items:
        prog, decomp = qiskit_to_braket_program(item["circuit"])
        meta = {
            "label": item["label"],
            "walk_name": item["walk_name"],
            "p": item["p"], "q": item["q"],
            "theta_idx": item["theta_idx"],
            "theta": item["theta"],
            "n_x": item["n_x"], "n_z": item["n_z"],
            "predicted_syndrome": list(item["predicted_syndrome"]),
            "predicted_logical_z": item["predicted_logical_z"],
            "n_qubits": decomp.num_qubits,
            "n_2q_gates": decomp.num_nonlocal_gates(),
            "depth": decomp.depth(),
            "subcircuit": None,
        }
        out.append((prog, decomp, meta))
    return out


def _add_steane_prep_braket(circ):
    """Add Steane [[7,1,3]] |0_L> preparation directly via Braket SDK gates.

    Mirrors exp28_chip_hbar._add_steane_zero_prep on qubits 0..6.
    """
    circ.h(0)
    circ.h(1)
    circ.h(3)
    for src, targets in [
        (0, (2, 4, 6)),
        (1, (2, 5, 6)),
        (3, (4, 5, 6)),
    ]:
        for tgt in targets:
            circ.cnot(src, tgt)


def _add_pauli_evolution_braket(circ, x_bits, z_bits, theta):
    """Add exp(-i P theta / 2) for substrate Pauli word P, via Braket SDK gates.

    Standard pre-rotation + CNOT-ladder-Rz-reverse decomposition.
    Pre-rotation: H on X-qubits; H·S† on Y-qubits (both X and Z bits set).
    Z-qubits need no pre-rotation. I-qubits skip entirely.

    P operates on qubits 0..6 only.
    """
    # Map per-qubit Pauli letter (Qiskit endianness: x_bits[q]/z_bits[q]
    # both index by qubit number directly).
    pauli_per_q = []
    for q in range(7):
        x, z = x_bits[q], z_bits[q]
        if x and z:
            pauli_per_q.append('Y')
        elif x:
            pauli_per_q.append('X')
        elif z:
            pauli_per_q.append('Z')
        else:
            pauli_per_q.append('I')

    # Pre-rotation
    for q in range(7):
        if pauli_per_q[q] == 'X':
            circ.h(q)
        elif pauli_per_q[q] == 'Y':
            # Y basis change: HS† maps Y eigenstates to Z eigenstates.
            # In Braket: H then Sdg, then later Sdg† H = S H to reverse.
            circ.si(q)
            circ.h(q)

    # Non-I qubits
    non_id = [q for q in range(7) if pauli_per_q[q] != 'I']
    if non_id:
        target = non_id[0]
        # Forward CNOT ladder: every other non-I qubit → target
        for q in non_id[1:]:
            circ.cnot(q, target)
        circ.rz(target, theta)
        # Reverse CNOT ladder
        for q in reversed(non_id[1:]):
            circ.cnot(q, target)

    # Reverse pre-rotation
    for q in range(7):
        if pauli_per_q[q] == 'X':
            circ.h(q)
        elif pauli_per_q[q] == 'Y':
            circ.h(q)
            circ.s(q)


def _walk_meta_for(walk_p, walk_q):
    entry = next(
        (e for e in COMPENDIUM_LOOKUP["particles"]
         if e["p"] == abs(walk_p) and e["q"] == abs(walk_q)),
        None,
    )
    if entry is None:
        raise ValueError(f"No compendium entry for ({walk_p}, {walk_q})")
    x_bits = entry["forward_pauli"]["x_part_bits"]
    z_bits = entry["forward_pauli"]["z_part_bits"]
    pred_syn, pred_lz = predicted_for(entry)
    return entry, x_bits, z_bits, pred_syn, pred_lz


def build_phase28e_aqt_split_circuits(theta_value: float):
    """Build the AQT-compatible split-circuit Phase 28e set using Braket SDK
    directly (NOT via Qiskit → OpenQASM → Braket round-trip, because
    Braket's OpenQASM 3 LocalSimulator interpreter has a bug that produces
    incorrect output for Steane-prep + non-{0,2,4,6} stabilizer extraction
    — verified 2026-05-24).

    For each of 4 walks (electron, pion, muon, proton), produces 7 circuits:
      - 6 stabilizer-projection circuits (one per Steane stabilizer; ancilla
        encodes that single stabilizer's eigenvalue)
      - 1 logical-Z destructive circuit (system qubits measured at end;
        ancilla unused but still part of the 8-qubit register)

    Returns list of (braket.circuits.Circuit, None, meta_dict) — 28 entries.
    Each circuit uses ≤ 8 qubits and has all measurements at END (no
    mid-circuit measure, no reset).
    """
    from braket.circuits import Circuit

    out = []
    for walk_name, p, q in PHASE_28A_TARGETS:
        # 6 stabilizer-projection circuits (3 X + 3 Z)
        for stab_idx in range(6):
            entry, x_bits, z_bits, pred_syn, pred_lz = _walk_meta_for(p, q)
            c = Circuit()
            _add_steane_prep_braket(c)
            _add_pauli_evolution_braket(c, x_bits, z_bits,
                                         theta=theta_value)
            anc = 7
            if stab_idx < 3:
                support = STEANE_STAB_SUPPORTS[stab_idx]
                c.h(anc)
                for q_idx in support:
                    c.cnot(anc, q_idx)
                c.h(anc)
            else:
                support = STEANE_STAB_SUPPORTS[stab_idx - 3]
                for q_idx in support:
                    c.cnot(q_idx, anc)
            meta = {
                "label": f"exp28_{walk_name}_stab{stab_idx}",
                "walk_name": walk_name,
                "p": p, "q": q,
                "theta": theta_value,
                "subcircuit": ("X-stab" if stab_idx < 3 else "Z-stab"),
                "stab_idx": stab_idx,
                "predicted_syndrome": list(pred_syn),
                "predicted_logical_z": pred_lz,
                "n_x": sum(x_bits),
                "n_z": sum(z_bits),
                "n_qubits": len(c.qubits),
                "n_2q_gates": sum(
                    1 for instr in c.instructions
                    if len(instr.target) == 2),
                "depth": len(c.instructions),
            }
            out.append((c, None, meta))
        # Logical-Z circuit
        entry, x_bits, z_bits, pred_syn, pred_lz = _walk_meta_for(p, q)
        c = Circuit()
        _add_steane_prep_braket(c)
        _add_pauli_evolution_braket(c, x_bits, z_bits, theta=theta_value)
        # Touch the ancilla qubit so the bitstring is 8 chars (consistency
        # with stab circuits). Identity gate on qubit 7.
        c.i(7)
        meta = {
            "label": f"exp28_{walk_name}_lz",
            "walk_name": walk_name,
            "p": p, "q": q,
            "theta": theta_value,
            "subcircuit": "logical-Z",
            "stab_idx": None,
            "predicted_syndrome": list(pred_syn),
            "predicted_logical_z": pred_lz,
            "n_x": sum(x_bits),
            "n_z": sum(z_bits),
            "n_qubits": len(c.qubits),
            "n_2q_gates": sum(1 for instr in c.instructions
                                if len(instr.target) == 2),
            "depth": len(c.instructions),
        }
        out.append((c, None, meta))
    return out


# ---------------------------------------------------------------------------
# Result decoding
# ---------------------------------------------------------------------------

def _decode_syndrome_to_fano(s_x_bits, s_z_bits):
    """Map (s_X, s_Z) 3-bit each → (fano_x, fano_z) Fano-point labels.

    Mirrors exp12_phase2_hardware_braket convention: syndrome value k ∈ 0..7
    indexes into QUBIT_TO_FANO[k-1] (k=0 → 'I').
    """
    x_pos = s_x_bits[0] + 2 * s_x_bits[1] + 4 * s_x_bits[2]
    z_pos = s_z_bits[0] + 2 * s_z_bits[1] + 4 * s_z_bits[2]
    x_fano = "I" if x_pos == 0 else QUBIT_TO_FANO[x_pos - 1]
    z_fano = "I" if z_pos == 0 else QUBIT_TO_FANO[z_pos - 1]
    return (x_fano, z_fano)


def decode_phase28e_bitstring(bs: str) -> tuple:
    """Decode a Braket measurement_counts bitstring → ((s_X, s_Z), logical_z).

    For the 8q single-ancilla layout with mid-circuit syndrome measurement:
      - c_syn (6 bits) written at syndrome-extraction step, mid-circuit
      - c_lz (7 bits) written at end-of-circuit destructive Z-measure

    Braket's `measurement_counts` keys are bitstrings whose character order
    follows the OpenQASM 3 classical-register declaration order. In our
    circuit (declared via build_vacuum_to_walk_rabi_circuit):

      c_syn declared first → bs[0..5]  = (s_X_0, s_X_1, s_X_2, s_Z_0, s_Z_1, s_Z_2)
      c_lz declared second → bs[6..12] = sys qubit 0..6 destructive Z

    LOGICAL-Z eigenvalue from parity over Z_L support qubits 0, 1, 2.
    """
    if len(bs) != 13:
        raise ValueError(f"expected 13-bit string, got {len(bs)}: {bs!r}")

    s_x = [int(b) for b in bs[:3]]
    s_z = [int(b) for b in bs[3:6]]
    sys_bits = [int(b) for b in bs[6:13]]

    fano_pair = _decode_syndrome_to_fano(s_x, s_z)
    parity = (sys_bits[0] + sys_bits[1] + sys_bits[2]) % 2
    logical_z = +1 if parity == 0 else -1
    return (fano_pair, logical_z)


def analyze_counts(counts: dict, meta: dict, shots: int) -> dict:
    """Reduce measurement_counts → modal outcome + predicted-bin probability.

    Routing by meta['subcircuit']:
      - None or non-AQT path: 13-bit decoding (DEFERRED-MEAS layout)
      - 'X-stab' / 'Z-stab': bit 7 (ancilla) = single stabilizer eigenvalue
      - 'logical-Z': bits 0,1,2 = system qubits 0-2 → parity = logical-Z
    """
    sub = meta.get("subcircuit")
    dist = Counter()
    decode_errors = 0

    if sub in ("X-stab", "Z-stab"):
        # 8-bit bitstring; bit 7 (rightmost in Braket = qubit 7 = ancilla)
        # encodes the single-stabilizer eigenvalue
        ancilla_bit_total = 0
        for bs, cnt in counts.items():
            if len(bs) != 8:
                decode_errors += cnt
                continue
            anc_bit = int(bs[7])
            dist[anc_bit] += cnt
            ancilla_bit_total += anc_bit * cnt
        n_decoded = sum(dist.values())
        # Stabilizer +1 eigenvalue ⇄ anc_bit=0; -1 eigenvalue ⇄ anc_bit=1
        # Predicted: the bit s_X[stab_idx] or s_Z[stab_idx-3] from the
        # walk's predicted syndrome.
        stab_idx = meta["stab_idx"]
        pred_syn = meta["predicted_syndrome"]  # ((fano_x, fano_z), parity)
        # Need to derive predicted single-stab bit from the X-syndrome /
        # Z-syndrome bit-vectors recorded in the lookup. Fall back to the
        # x_syndrome_bits / z_syndrome_bits arrays stored on the entry.
        entry = next(
            e for e in COMPENDIUM_LOOKUP["particles"]
            if e["p"] == abs(meta["p"]) and e["q"] == abs(meta["q"])
        )
        if stab_idx < 3:
            pred_stab_bit = entry["forward_pauli"]["x_syndrome_bits"][stab_idx]
        else:
            pred_stab_bit = entry["forward_pauli"]["z_syndrome_bits"][stab_idx - 3]
        modal_bit, modal_count = (dist.most_common(1)[0]
                                   if dist else (None, 0))
        modal_prob = modal_count / n_decoded if n_decoded else 0.0
        pred_prob = dist.get(pred_stab_bit, 0) / n_decoded if n_decoded else 0.0
        return {
            "subcircuit": sub,
            "stab_idx": stab_idx,
            "modal_bit": modal_bit,
            "modal_prob": modal_prob,
            "predicted_bit": pred_stab_bit,
            "predicted_prob": pred_prob,
            "n_decoded": n_decoded,
            "decode_errors": decode_errors,
            "distribution": {str(k): v for k, v in dist.items()},
        }

    if sub == "logical-Z":
        # 8-bit bitstring; bits 0..6 = system qubits 0..6 (ancilla unused)
        # Logical-Z eigenvalue = (-1)^(parity over qubits 0,1,2)
        for bs, cnt in counts.items():
            if len(bs) != 8:
                decode_errors += cnt
                continue
            sys_bits = [int(b) for b in bs[:7]]
            parity = (sys_bits[0] + sys_bits[1] + sys_bits[2]) % 2
            lz = +1 if parity == 0 else -1
            dist[lz] += cnt
        n_decoded = sum(dist.values())
        pred_lz = meta["predicted_logical_z"]
        modal_lz, modal_count = (dist.most_common(1)[0]
                                  if dist else (None, 0))
        modal_prob = modal_count / n_decoded if n_decoded else 0.0
        pred_prob = dist.get(pred_lz, 0) / n_decoded if n_decoded else 0.0
        return {
            "subcircuit": sub,
            "modal_lz": modal_lz,
            "modal_prob": modal_prob,
            "predicted_lz": pred_lz,
            "predicted_prob": pred_prob,
            "n_decoded": n_decoded,
            "decode_errors": decode_errors,
            "distribution": {str(k): v for k, v in dist.items()},
        }

    # Fallback: standard 13-bit decoding (deferred-measurement layout)
    for bs, cnt in counts.items():
        try:
            key = decode_phase28e_bitstring(bs)
            dist[key] += cnt
        except (KeyError, IndexError, ValueError):
            decode_errors += cnt
    modal_key, modal_count = (dist.most_common(1)[0]
                              if dist else (None, 0))
    n_shots = sum(dist.values())
    modal_prob = modal_count / n_shots if n_shots else 0.0
    pred_syn = tuple(meta["predicted_syndrome"])
    pred_lz = meta["predicted_logical_z"]
    pred_key = (pred_syn, pred_lz)
    pred_prob = dist.get(pred_key, 0) / n_shots if n_shots else 0.0
    return {
        "modal_bin": modal_key,
        "modal_prob": modal_prob,
        "predicted_bin": pred_key,
        "predicted_prob": pred_prob,
        "n_decoded": n_shots,
        "decode_errors": decode_errors,
        "distribution": {str(k): v for k, v in dist.items()},
    }


def print_one_result(label, analysis, prefix="    "):
    sub = analysis.get("subcircuit")
    if sub in ("X-stab", "Z-stab"):
        pred = analysis["predicted_bit"]
        modal = analysis["modal_bit"]
        match = "✓" if modal == pred else "✗"
        print(f"{prefix}{label:<26} P(pred={pred})={analysis['predicted_prob']:.3f}  "
              f"modal={modal}  {match}  "
              f"({analysis['n_decoded']} dec, {analysis['decode_errors']} errs)")
        return
    if sub == "logical-Z":
        pred = analysis["predicted_lz"]
        modal = analysis["modal_lz"]
        match = "✓" if modal == pred else "✗"
        print(f"{prefix}{label:<26} P(pred={pred:+d})={analysis['predicted_prob']:.3f}  "
              f"modal={modal:+d}  {match}  "
              f"({analysis['n_decoded']} dec, {analysis['decode_errors']} errs)")
        return
    pred = analysis["predicted_bin"]
    modal = analysis["modal_bin"]
    match = "✓" if modal == pred else "✗"
    print(f"{prefix}{label:<22} P(pred)={analysis['predicted_prob']:.3f}  "
          f"modal={modal}  pred={pred}  {match}  "
          f"({analysis['n_decoded']} dec, {analysis['decode_errors']} errs)")


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

def resolve_device(backend: str, region: str | None = None,
                    use_simulator: bool = False):
    """Return (AwsDevice or LocalSimulator, arn, region, per_shot, task_fee)."""
    if use_simulator or backend == "local-simulator":
        from braket.devices import LocalSimulator
        return LocalSimulator(), "local-simulator", None, 0.0, 0.0

    if backend not in BACKEND_INFO:
        if backend.startswith("arn:aws:braket:"):
            arn = backend
            inferred_region = arn.split(":")[3]
            from braket.aws import AwsDevice
            return AwsDevice(arn), arn, inferred_region, None, 0.30
        raise ValueError(f"Unknown backend {backend!r}. "
                          f"Known: {list(BACKEND_INFO.keys())}")

    expected_region, per_shot, task_fee = BACKEND_INFO[backend]
    if region is None:
        region = expected_region
    arn = f"arn:aws:braket:{region}::device/qpu/{backend}"
    from braket.aws import AwsDevice
    return AwsDevice(arn), arn, region, per_shot, task_fee


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def run_dry_run(circuits, shots: int):
    """Local Braket simulator validation (free)."""
    from braket.devices import LocalSimulator
    device = LocalSimulator()
    print(f"\n  Executing {len(circuits)} circuits on local simulator "
          f"({shots} shots each)...")
    results = []
    for prog, decomp, meta in circuits:
        t0 = time.time()
        task = device.run(prog, shots=shots)
        result = task.result()
        dt = time.time() - t0
        counts = result.measurement_counts
        analysis = analyze_counts(counts, meta, shots)
        analysis["elapsed_s"] = dt
        print_one_result(meta["label"], analysis)
        results.append({**meta, "result": analysis})
    return results


def run_hardware_stage(circuits, backend, shots: int, region: str | None):
    """Resolve device + estimate cost, NO submit."""
    device, arn, region, per_shot, task_fee = resolve_device(
        backend, region=region)
    print(f"\n  Backend: {arn}")
    print(f"  Region: {region}")
    try:
        status = device.status
        print(f"  Status: {status}")
    except Exception as e:
        print(f"  Status fetch failed: {e}")
    if per_shot is not None:
        est = len(circuits) * shots * per_shot + len(circuits) * task_fee
        print(f"  Per-shot: ${per_shot:.6f}")
        print(f"  Task fee: ${task_fee:.2f}")
        print(f"  Est cost ({len(circuits)} × {shots}): ${est:.2f}")
    # Circuit stats
    print(f"\n  Circuit stats (first per walk):")
    seen = set()
    for prog, decomp, meta in circuits:
        if meta["walk_name"] in seen:
            continue
        seen.add(meta["walk_name"])
        print(f"    {meta['walk_name']:<10} N_X={meta['n_x']}  "
              f"qubits={meta['n_qubits']}  depth={meta['depth']}  "
              f"2q-gates={meta['n_2q_gates']}")
    print(f"\n  ★ Staged but NOT submitted. Re-run with --mode hardware-submit")
    return None


def run_hardware_submit(circuits, backend, shots: int, region: str | None,
                         resume_arns: list | None = None):
    """Submit async: phase 1 = submit all tasks; phase 2 = poll for results."""
    from braket.aws import AwsQuantumTask

    device, arn, region, per_shot, task_fee = resolve_device(
        backend, region=region)
    resume_arns = resume_arns or []
    print(f"\n  ★★★ SUBMITTING to {arn} (async pattern) ★★★")
    print(f"  Region: {region}")
    if per_shot is not None:
        new_submits = len(circuits) - len(resume_arns)
        est = (len(circuits) * shots * per_shot
               + new_submits * task_fee)
        print(f"  Estimated cost: ${est:.2f}")

    # ---- Phase 1: submit (or resume) all tasks ----
    print(f"\n  Phase 1: submitting all {len(circuits)} tasks...")
    tasks = []
    for i, (prog, decomp, meta) in enumerate(circuits):
        if i < len(resume_arns):
            task = AwsQuantumTask(arn=resume_arns[i])
            task_id = resume_arns[i]
            print(f"  [{i+1}/{len(circuits)}] {meta['label']:<22} resumed: "
                  f"id={task_id}")
        else:
            task = device.run(prog, shots=shots)
            task_id = task.id
            print(f"  [{i+1}/{len(circuits)}] {meta['label']:<22} submitted: "
                  f"id={task_id}")
        tasks.append((task, meta, task_id))

    # ---- Phase 2: poll all tasks ----
    print(f"\n  Phase 2: polling {len(tasks)} tasks for completion...")
    results = []
    poll_start = time.time()
    for i, (task, meta, task_id) in enumerate(tasks):
        t0 = time.time()
        try:
            result = task.result()  # blocks
            dt = time.time() - t0
            wall = time.time() - poll_start
            print(f"  [{i+1}/{len(tasks)}] {meta['label']:<22} "
                  f"completed in {dt:.1f}s (wall {wall:.1f}s)")
            counts = result.measurement_counts
            analysis = analyze_counts(counts, meta, shots)
            analysis["task_id"] = task_id
            analysis["elapsed_s"] = dt
            print_one_result(meta["label"], analysis, prefix="      ")
            results.append({**meta, "result": analysis})
        except Exception as e:
            wall = time.time() - poll_start
            print(f"  [{i+1}/{len(tasks)}] {meta['label']:<22} "
                  f"FAIL: {type(e).__name__}: {str(e)[:120]} (wall {wall:.1f}s)")
            results.append({**meta, "result": None, "task_id": task_id,
                             "error": str(e)})
    return results


def summarise(results):
    """Print per-walk modal-bin agreement summary.

    For split-circuit (AQT) results: reconstruct (s_X, s_Z, L_Z) per walk from
    the 6 stab + 1 LZ subcircuit results, then compare to prediction.

    For deferred-measurement results: summarise per-walk modal-bin directly.
    """
    by_walk = {}
    for r in results:
        if r.get("result") is None:
            continue
        by_walk.setdefault(r["walk_name"], []).append(r)
    is_split = any(r.get("subcircuit") for r in results
                    if r.get("result") is not None)
    print()
    if is_split:
        print("Walk         N_X   reconstructed (s_X, s_Z, L_Z)   predicted               match?")
        print("-" * 92)
        n_pass = 0
        for walk_name in ["electron", "pion", "muon", "proton"]:
            items = by_walk.get(walk_name, [])
            if not items:
                print(f"{walk_name:11s} (no data)")
                continue
            stab_modal = {}  # stab_idx -> modal bit
            lz_modal = None
            for item in items:
                sub = item.get("subcircuit")
                a = item["result"]
                if sub in ("X-stab", "Z-stab"):
                    stab_modal[item["stab_idx"]] = a["modal_bit"]
                elif sub == "logical-Z":
                    lz_modal = a["modal_lz"]
            if len(stab_modal) != 6 or lz_modal is None:
                print(f"{walk_name:11s} INCOMPLETE: stabs={sorted(stab_modal)}, lz={lz_modal}")
                continue
            s_x = [stab_modal[k] for k in range(3)]
            s_z = [stab_modal[k] for k in range(3, 6)]
            fano_pair = _decode_syndrome_to_fano(s_x, s_z)
            reconstructed = (fano_pair, lz_modal)
            pred_syn = tuple(items[0]["predicted_syndrome"])
            pred_lz = items[0]["predicted_logical_z"]
            predicted = (pred_syn, pred_lz)
            match = (reconstructed == predicted)
            if match:
                n_pass += 1
            marker = "✓" if match else "✗"
            n_x = items[0]["n_x"]
            print(f"{walk_name:11s} {n_x:>3d}   {reconstructed}   "
                  f"{predicted}   {marker}")
        print("-" * 92)
        print(f"Phase 28e cross-vendor syndrome agreement (split): {n_pass}/4 walks")
        return n_pass

    # Standard deferred-measurement summary
    print("Walk        N_X   θ=π modal-bin  matches pred?")
    print("-" * 58)
    n_pass = 0
    for walk_name in ["electron", "pion", "muon", "proton"]:
        items = by_walk.get(walk_name, [])
        if not items:
            print(f"{walk_name:11s} (no data)")
            continue
        items.sort(key=lambda x: x.get("theta_idx", 0))
        item = items[-1]
        modal = item["result"]["modal_bin"]
        pred = item["result"]["predicted_bin"]
        n_x = item["n_x"]
        match = (modal == pred)
        marker = "✓" if match else "✗"
        if match:
            n_pass += 1
        print(f"{walk_name:11s} {n_x:>3d}   {modal}  {marker} pred={pred}")
    print("-" * 58)
    print(f"Phase 28e modal-bin agreement: {n_pass}/4 walks")
    return n_pass


def save_results(results, mode: str, backend_label: str, shots: int):
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_backend = backend_label.replace("/", "_")
    fname = (f"exp28_phase28e_braket_{mode}_{safe_backend}_{ts}.json")
    out_path = OUTPUT_DIR / fname
    payload = {
        "mode": mode,
        "backend": backend_label,
        "shots": shots,
        "timestamp": ts,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved results -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp 28 Phase 28e cross-vendor Braket submitter")
    parser.add_argument("--mode",
                        choices=["dry-run", "hardware-stage", "hardware-submit"],
                        default="dry-run")
    parser.add_argument("--backend", default="aqt/Ibex-Q1",
                        help="Braket backend (e.g. aqt/Ibex-Q1, iqm/Garnet, "
                             "iqm/Emerald, rigetti/Cepheus-1-108Q)")
    parser.add_argument("--region", default=None,
                        help="AWS region; auto-picked from backend if unset")
    parser.add_argument("--shots", type=int, default=100,
                        help="Shots per circuit (default 100)")
    parser.add_argument("--n-angles", type=int, default=16,
                        help="Theta sweep points (default 16)")
    parser.add_argument("--ancilla-budget", type=int, choices=[1, 6], default=6,
                        help="6 = standard 13q deferred-measurement layout "
                             "(IQM Garnet/Emerald, ≥13q backends). "
                             "1 = single-ancilla 8q layout (mid-circuit "
                             "measure path; NOTE Braket does NOT expose "
                             "classical bits from mid-circuit measures — "
                             "use --aqt-split for AQT instead).")
    parser.add_argument("--aqt-split", action="store_true",
                        help="Build the AQT-compatible split-circuit set: "
                             "28 circuits (4 walks × (6 stab + 1 LZ)) at "
                             "θ=π only, each ≤8q with all measurements at "
                             "end. Required for AQT IBEX Q1 substantive "
                             "Phase 28e submission. Overrides --n-angles "
                             "(always θ=π) and --ancilla-budget.")
    parser.add_argument("--resume-task-arns", nargs="+", default=None,
                        help="Resume orphaned tasks by ARN list")
    args = parser.parse_args()

    print(f"Exp 28 Phase 28e Braket — mode={args.mode}, "
          f"backend={args.backend}, shots={args.shots}, "
          f"n_angles={args.n_angles}, ancilla_budget={args.ancilla_budget}")
    print()

    import numpy as np
    if args.aqt_split:
        print("Building AQT split-circuit Phase 28e set "
              "(4 walks × 7 subcircuits = 28 circuits, θ=π only)...")
        circuits = build_phase28e_aqt_split_circuits(theta_value=float(np.pi))
    else:
        print("Building 4 × n_angles Phase 28e circuits...")
        circuits = build_phase28e_braket_circuits(
            n_angles=args.n_angles,
            ancilla_budget=args.ancilla_budget,
        )
    print(f"Built {len(circuits)} circuits "
          f"({circuits[0][2]['n_qubits']}q each, "
          f"{circuits[0][2]['n_2q_gates']}-2q for first)")

    if args.mode == "dry-run":
        results = run_dry_run(circuits, args.shots)
        n_pass = summarise(results)
        save_results(results, args.mode, "local-simulator", args.shots)
        return 0 if n_pass == 4 else 1

    if args.mode == "hardware-stage":
        run_hardware_stage(circuits, args.backend, args.shots, args.region)
        return 0

    # hardware-submit
    results = run_hardware_submit(
        circuits, args.backend, args.shots, args.region,
        resume_arns=args.resume_task_arns,
    )
    n_pass = summarise(results)
    save_results(results, args.mode, args.backend, args.shots)
    return 0 if n_pass == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
