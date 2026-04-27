#!/usr/bin/env python3
"""
Experiment 5 — Syndrome distribution on |K_7⟩.

Phase 8h finding: |K_7⟩ is the unique +1 eigenstate of all 7
stabilizers S_v.  The 128 simultaneous-eigenspaces of the abelian
stabilizer group {S_0,...,S_6} are all 1-dimensional and span the
full 7-qubit Hilbert space.

Trick: applying the K_7 uncompute circuit (CZ_E first, then H^7)
followed by Z-basis measurement directly samples the syndrome
distribution.  Bitstring b ↔ syndrome s_v = (-1)^{b_v}.  Bitstring
all-zeros is the +1 syndrome, equivalent to |K_7⟩.

Three tests using the same uncompute circuit:

  (i)   Input |0⟩^7 → after uncompute = H^7|0⟩^7 = |+⟩^7
        → Z-measure: UNIFORM over 128 bitstrings
        Tests "random initial state has 1/128 overlap with each
        syndrome class".

  (ii)  Input |K_7⟩ → after uncompute = |0⟩^7
        → Z-measure: DELTA at bitstring 0000000
        Tests "|K_7⟩ is the +1 eigenstate exactly".  Direct fidelity
        to |K_7⟩ on hardware.

  (iii) Input |+⟩^7 → after uncompute = H^7 |K_7⟩
        → Z-measure: structured non-uniform distribution
        Tests intermediate behavior between (i) and (ii).

Aggregate signature: in (i), entropy of syndrome distribution
should be ≈ 7 bits (uniform).  In (ii), entropy should be ≈ 0
(degenerate).  Hardware brings entropy below 7 in (i) (depolarization)
and above 0 in (ii) (depolarization toward uniform).

Usage:
  python3 nwt_qec_heron_exp5.py                       # numpy + Aer
  python3 nwt_qec_heron_exp5.py --backend ibm_marrakesh   # real HW
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nwt_qec_heron_experiment import (
    N_QUBITS, DIM, build_K7_state, build_K7_qiskit_circuit,
    section, QISKIT_AVAILABLE,
)

if QISKIT_AVAILABLE:
    from qiskit import QuantumCircuit, ClassicalRegister, transpile
    from qiskit_aer import AerSimulator


# =====================================================================
# Reference (numpy) syndrome distributions
# =====================================================================

def k7_uncompute_state(psi: np.ndarray) -> np.ndarray:
    """Apply K_7 uncompute (CZ_E first, then H^7) to a 7-qubit state."""
    out = psi.copy()
    # 1. Apply all CZ edges (CZ is diagonal in Z basis: flip sign on |11>)
    for u, v in combinations(range(N_QUBITS), 2):
        for idx in range(DIM):
            xu = (idx >> (N_QUBITS - 1 - u)) & 1
            xv = (idx >> (N_QUBITS - 1 - v)) & 1
            if xu == 1 and xv == 1:
                out[idx] *= -1
    # 2. Apply H on every qubit
    H = (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    H_full = H
    for _ in range(N_QUBITS - 1):
        H_full = np.kron(H_full, H)
    return H_full @ out


def state_zero() -> np.ndarray:
    psi = np.zeros(DIM, dtype=complex)
    psi[0] = 1.0
    return psi


def state_plus() -> np.ndarray:
    return np.ones(DIM, dtype=complex) / np.sqrt(DIM)


def syndrome_distribution_numerical(psi_input: np.ndarray) -> np.ndarray:
    """Apply K_7 uncompute and return |amplitude|² over all 128 bitstrings."""
    psi_out = k7_uncompute_state(psi_input)
    return np.abs(psi_out) ** 2


def shannon_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def reference_simulation() -> None:
    section('PART A: Numpy reference for syndrome distributions')

    print(f"\n  Three input states; each gets K_7-uncompute applied,")
    print(f"  then measured in Z basis.  Bitstring b ↔ syndrome (-1)^b.\n")

    inputs = {
        '|0⟩^7':  state_zero(),
        '|+⟩^7':  state_plus(),
        '|K_7⟩':  build_K7_state(),
    }

    print(f"  {'input':>12} {'P(0000000)':>14} {'P_max':>10} {'argmax':>10} "
          f"{'H_2':>8} {'H_2 / 7':>10}")
    print('  ' + '-' * 70)
    for label, psi in inputs.items():
        p = syndrome_distribution_numerical(psi)
        p_zero = float(p[0])
        p_max = float(p.max())
        argmax = int(np.argmax(p))
        entropy = shannon_entropy(p)
        print(f"  {label:>12} {p_zero:>14.6f} {p_max:>10.6f} "
              f"{format(argmax, '07b'):>10} {entropy:>8.3f} "
              f"{entropy/7:>10.3f}")

    print(f"""
  Predictions:
    |0⟩^7  → uniform over 128 syndromes → H_2 = log_2(128) = 7.000
    |+⟩^7  → structured (non-uniform but smooth)
    |K_7⟩  → δ at bitstring 0 → H_2 = 0.000

  The contrast between H_2 ≈ 7 (uniform) for |0⟩^7 and H_2 ≈ 0
  (degenerate) for |K_7⟩ is the headline signal.  Hardware noise
  pulls both toward intermediate values (≈ 7 for |K_7⟩ in fully
  depolarized limit).
""")


# =====================================================================
# Qiskit circuits
# =====================================================================

def k7_uncompute_circuit() -> 'QuantumCircuit':
    """K_7 uncompute: CZ_E first, then H^7."""
    qc = QuantumCircuit(7, name='K7_uncompute')
    for u, v in combinations(range(7), 2):
        qc.cz(u, v)
    for q in range(7):
        qc.h(q)
    return qc


def syndrome_circuit_from_zero() -> 'QuantumCircuit':
    """|0⟩^7 → uncompute → measure.  Predict uniform over 128."""
    qc = QuantumCircuit(7)
    qc.barrier()  # mark logical boundary; prevents optimizer from removing
    qc.compose(k7_uncompute_circuit(), inplace=True)
    qc.add_register(ClassicalRegister(7, name='c'))
    qc.measure(range(7), range(7))
    return qc


def syndrome_circuit_from_plus() -> 'QuantumCircuit':
    """|+⟩^7 → uncompute → measure.  Predict structured."""
    qc = QuantumCircuit(7)
    for q in range(7):
        qc.h(q)
    qc.barrier()
    qc.compose(k7_uncompute_circuit(), inplace=True)
    qc.add_register(ClassicalRegister(7, name='c'))
    qc.measure(range(7), range(7))
    return qc


def syndrome_circuit_from_K7() -> 'QuantumCircuit':
    """|K_7⟩ → uncompute → measure.  Predict δ at 0000000.

    Barrier between prep and uncompute prevents the transpiler from
    collapsing the round-trip (prep · prep⁻¹ = I) into a no-op — we
    WANT the hardware to actually execute both halves so the round-trip
    fidelity reflects real gate errors.
    """
    qc = build_K7_qiskit_circuit()
    qc.barrier()
    qc.compose(k7_uncompute_circuit(), inplace=True)
    qc.add_register(ClassicalRegister(7, name='c'))
    qc.measure(range(7), range(7))
    return qc


# =====================================================================
# Counts → distribution + summary stats
# =====================================================================

def counts_to_dist(counts: Dict[str, int], n_shots: int) -> np.ndarray:
    """Convert qiskit counts dict (right-to-left bitstrings) to a 128-vector
    indexed by integer with bit 0 = qubit 0.  Returns probabilities."""
    p = np.zeros(DIM)
    for bs, c in counts.items():
        # qiskit bitstring is right-to-left: bs[-1] = qubit 0
        idx = 0
        for q in range(7):
            idx |= (int(bs[-(q + 1)]) << q)
        p[idx] += c / n_shots
    return p


def summarize(label: str, p: np.ndarray, predicted_uniform: bool) -> dict:
    """Print + return summary of a syndrome distribution."""
    n_shots_inferred = round(1.0 / np.min(p[p > 0])) if (p > 0).any() else 0
    p_zero = float(p[0])
    p_max = float(p.max())
    argmax = int(np.argmax(p))
    entropy = shannon_entropy(p)
    n_nonzero = int(np.sum(p > 0))
    # Total variation distance from uniform
    p_unif = np.ones(DIM) / DIM
    tv = float(0.5 * np.sum(np.abs(p - p_unif)))

    if predicted_uniform:
        # χ² for uniformity: counts vs N/128 each
        # We don't know N exactly here; use n_shots inferred or skip
        pass

    print(f"  {label}")
    print(f"    P(0000000)         = {p_zero:.4f}  (predict "
          f"{'1.0000' if not predicted_uniform else '0.0078 = 1/128'})")
    print(f"    P_max              = {p_max:.4f}   at "
          f"bitstring {format(argmax, '07b')}")
    print(f"    H_2 (Shannon)      = {entropy:.3f} bits  (predict "
          f"{'7.000' if predicted_uniform else '0.000'})")
    print(f"    H_2 / 7            = {entropy/7:.3f}")
    print(f"    n_nonzero bitstrings = {n_nonzero} / 128")
    print(f"    TV distance from uniform = {tv:.4f}")

    return dict(p_zero=p_zero, p_max=p_max, entropy=entropy,
                n_nonzero=n_nonzero, tv=tv, argmax=argmax)


# =====================================================================
# Run on backend
# =====================================================================

def run_syndrome_tests(backend, shots: int = 4000,
                         use_sampler: bool = False) -> None:
    label = getattr(backend, 'name', backend.__class__.__name__)
    section(f'Experiment 5 on {label}  ({shots} shots × 3 prep circuits)')

    circuits = {
        '|0⟩^7':  syndrome_circuit_from_zero(),
        '|+⟩^7':  syndrome_circuit_from_plus(),
        '|K_7⟩':  syndrome_circuit_from_K7(),
    }
    labels = list(circuits.keys())

    print(f"\n  Transpiling 3 circuits...")
    transpiled = transpile(list(circuits.values()), backend=backend,
                            optimization_level=3, seed_transpiler=42)
    cz_counts = [tc.count_ops().get('cz', 0) + tc.count_ops().get('ecr', 0)
                  for tc in transpiled]
    for lbl, cz in zip(labels, cz_counts):
        print(f"    {lbl:>10}:  {cz:>3} CZ/ECR after transpile")

    print(f"\n  Submitting...")
    if use_sampler:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        sampler = Sampler(mode=backend)
        job = sampler.run(transpiled, shots=shots)
        print(f"    Job ID: {job.job_id()}")
        print(f"    (Look up later at https://quantum.cloud.ibm.com)")
        result = job.result()
        get_counts = lambda i: result[i].data.c.get_counts()
    else:
        job = backend.run(transpiled, shots=shots)
        result = job.result()
        get_counts = lambda i: result.get_counts(i)
    print(f"  Job complete.\n")

    summaries = {}
    for i, lbl in enumerate(labels):
        counts = get_counts(i)
        p = counts_to_dist(counts, shots)
        predicted_uniform = (lbl != '|K_7⟩')
        summaries[lbl] = summarize(lbl, p, predicted_uniform)
        print()

    # Aggregate comparison
    section_title = 'Aggregate signal: H_2(|0⟩^7) vs H_2(|K_7⟩)'
    print(f"  {section_title}")
    print(f"  {'-' * len(section_title)}")
    h_zero = summaries['|0⟩^7']['entropy']
    h_K7 = summaries['|K_7⟩']['entropy']
    print(f"    H_2(|0⟩^7)         = {h_zero:.3f}  (predict 7.000)")
    print(f"    H_2(|K_7⟩)         = {h_K7:.3f}  (predict 0.000)")
    print(f"    Δ entropy          = {h_zero - h_K7:.3f} bits")
    print(f"    P(0|K_7 input)     = {summaries['|K_7⟩']['p_zero']:.4f}  "
          f"(direct |⟨K_7|⟨K_7|⟩|² fidelity ≈ this)")
    print(f"    P_max(|0⟩ input)   = {summaries['|0⟩^7']['p_max']:.4f}  "
          f"(predict ≈ 1/128 = 0.0078; >> 0.0078 means noise-bias)")

    print(f"""
  Interpretation:
    • Large Δ entropy (between |0⟩^7 and |K_7⟩ inputs) confirms
      the syndrome eigenspaces are non-degenerate AND that |K_7⟩ is
      the unique +1-syndrome state.
    • H_2(|0⟩^7) close to 7 confirms uniform syndrome dist.
    • P(0|K_7 input) is the round-trip fidelity for |K_7⟩ prep + uncompute;
      directly bounds how much of |K_7⟩'s amplitude is preserved.
""")


# =====================================================================
# Heron submission
# =====================================================================

def heron_submission(backend_name: str = None, shots: int = 4000) -> None:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("ERROR: qiskit-ibm-runtime not installed.")
        return

    section('Experiment 5: IBM Heron submission')

    token_file = os.path.expanduser('~/.ibm_quantum_api_key')
    if not os.path.exists(token_file):
        print(f"\n  ERROR: {token_file} not found.")
        return
    token = open(token_file).read().strip()
    service = QiskitRuntimeService(channel='ibm_cloud', token=token)

    if backend_name is None:
        candidates = [b for b in service.backends(simulator=False,
                                                    operational=True)
                       if b.num_qubits >= 100]
        if not candidates:
            print("\n  ERROR: no Heron-class backend operational.")
            return
        backend = min(candidates, key=lambda b: b.status().pending_jobs)
    else:
        backend = service.backend(backend_name)

    status = backend.status()
    print(f"  Backend:       {backend.name}")
    print(f"  Qubits:        {backend.num_qubits}")
    print(f"  Status:        {status.status_msg}")
    print(f"  Pending jobs:  {status.pending_jobs}")
    print(f"  Shots:         {shots} per circuit (×3 circuits)")
    print(f"  Total shots:   {shots * 3}")

    run_syndrome_tests(backend, shots=shots, use_sampler=True)


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Experiment 5 — syndrome distribution on |K_7⟩')
    parser.add_argument('--shots', type=int, default=4000)
    parser.add_argument('--skip-numpy', action='store_true')
    parser.add_argument('--skip-aer', action='store_true')
    parser.add_argument('--backend', type=str, default=None)
    args = parser.parse_args()

    section('Experiment 5 — Syndrome distribution test on |K_7⟩')

    print(f"""
  Phase 8h test: the 128 stabilizer-syndrome eigenspaces of K_7 are
  all 1-dim and span the full 7-qubit Hilbert space.  Apply K_7
  uncompute + Z-measure to read out syndrome directly.  Compare
  three input states.
""")

    if not args.skip_numpy:
        reference_simulation()

    if QISKIT_AVAILABLE and not args.skip_aer:
        try:
            sim = AerSimulator()
            run_syndrome_tests(sim, shots=args.shots, use_sampler=False)
        except Exception as e:
            print(f"\nAerSimulator part failed: {e}")

    if args.backend is not None:
        heron_submission(args.backend, shots=args.shots)


if __name__ == '__main__':
    main()
