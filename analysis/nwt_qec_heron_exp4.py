#!/usr/bin/env python3
"""
Experiment 4 — 3-body ⟨Y_u Y_v Y_w⟩ = 0 null test on |K_7⟩.

Phase 8e finding: every 3-body Y-correlator vanishes EXACTLY on the
graph state |K_7⟩ because Y_u Y_v Y_w cannot be written as a product
of K_7 stabilizers (the X-part needs S_u S_v S_w, but that gives Z on
the COMPLEMENT of {u,v,w}, not on {u,v,w} itself).

Hence the cleanest statistical test of the framework:
  - Predicted: ⟨Y_u Y_v Y_w⟩ = 0 for all C(7,3) = 35 triples.
  - Hardware noise shows up as variance ONLY.
  - σ_per_triple ≈ 1/√N_shots in the noiseless limit.
  - Aggregate test: χ² = Σ_t (mean_t / σ_t)² ~ 35 dof under null.

Reuses circuit + Heron-submission infrastructure from
nwt_qec_heron_experiment.py.

Usage:
  python3 nwt_qec_heron_exp4.py                       # numpy + Aer
  python3 nwt_qec_heron_exp4.py --backend ibm_marrakesh   # real HW
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

# Reuse infrastructure from the Exp 1/2 script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nwt_qec_heron_experiment import (
    N_QUBITS, DIM, I2, X, Y, Z, H, S, S_dag, gate_on,
    build_K7_state, build_K7_qiskit_circuit,
    measure_pauli_string_expectation, section,
    QISKIT_AVAILABLE,
)

if QISKIT_AVAILABLE:
    from qiskit import QuantumCircuit, ClassicalRegister, transpile
    from qiskit_aer import AerSimulator


# =====================================================================
# Triple enumeration
# =====================================================================

TRIPLES: List[Tuple[int, int, int]] = list(combinations(range(N_QUBITS), 3))
assert len(TRIPLES) == 35, f"Expected 35 triples, got {len(TRIPLES)}"

# The 7 Fano lines of K_7 (each pair of vertices lies on exactly one line).
# This is one canonical labelling; PSL(2,7) acts edge-transitively on these.
FANO_LINES: List[Tuple[int, int, int]] = [
    (0, 1, 3), (0, 2, 4), (0, 5, 6), (1, 2, 5),
    (1, 4, 6), (2, 3, 6), (3, 4, 5),
]
assert len(FANO_LINES) == 7

FANO_SET = set(FANO_LINES)
NONFANO_SET = set(TRIPLES) - FANO_SET
assert len(NONFANO_SET) == 28


# =====================================================================
# Circuit construction
# =====================================================================

def YYY_measurement_circuit(u: int, v: int, w: int) -> 'QuantumCircuit':
    """Circuit measuring ⟨Y_u Y_v Y_w⟩ on |K_7⟩.

    Y → Z basis rotation on each of the 3 measured qubits: apply S†
    then H.  After rotation, Y_u Y_v Y_w becomes Z_u Z_v Z_w, measured
    as (-1)^(b_u + b_v + b_w).
    """
    qc = build_K7_qiskit_circuit()
    qc.add_register(ClassicalRegister(3, name='c'))
    for q in (u, v, w):
        qc.sdg(q)
        qc.h(q)
    qc.measure([u, v, w], [0, 1, 2])
    return qc


def parse_YYY_counts(counts: Dict[str, int],
                       n_shots: int) -> Tuple[float, float]:
    """⟨Y_u Y_v Y_w⟩ = average of (-1)^(b_u + b_v + b_w)."""
    total = 0.0
    for bitstring, count in counts.items():
        # Qiskit bitstrings are right-to-left.  cbit 0 is rightmost.
        b0 = int(bitstring[-1])
        b1 = int(bitstring[-2])
        b2 = int(bitstring[-3])
        ev = (-1) ** (b0 + b1 + b2)
        total += ev * count
    mean = total / n_shots
    var = 1 - mean ** 2  # for ±1 outcomes
    return float(mean), float(np.sqrt(var / n_shots))


# =====================================================================
# Numpy reference simulation
# =====================================================================

def reference_simulation(n_shots: int = 4000) -> None:
    section(f'PART A: Pure-numpy reference (n_shots = {n_shots})')

    psi = build_K7_state()

    print(f"\n  35 triples — predicted ⟨Y_u Y_v Y_w⟩ = 0 exactly\n")
    print(f"  {'triple':>10} {'exact':>10} {'sampled':>12} {'σ':>10}")
    print('  ' + '-' * 46)

    chisq = 0.0
    max_abs = 0.0
    rms_sample = 0.0
    nonzero_exact = 0
    for (u, v, w) in TRIPLES:
        paulis = [(Y, u), (Y, v), (Y, w)]
        e_exact, _ = measure_pauli_string_expectation(psi, paulis, n_shots=0)
        e_samp, sigma = measure_pauli_string_expectation(psi, paulis,
                                                           n_shots=n_shots)
        if abs(e_exact) > 1e-10:
            nonzero_exact += 1
        chisq += (e_samp / sigma) ** 2 if sigma > 0 else 0.0
        max_abs = max(max_abs, abs(e_samp))
        rms_sample += e_samp ** 2
        print(f"  ({u},{v},{w}){'':>3} {e_exact:>+10.4f} {e_samp:>+12.4f} "
              f"{sigma:>10.4f}")

    rms_sample = np.sqrt(rms_sample / 35)
    print(f"\n  Exact non-zero triples: {nonzero_exact} / 35")
    print(f"  RMS sampled ⟨YYY⟩:     {rms_sample:.4f}")
    print(f"  max |⟨YYY⟩|:           {max_abs:.4f}")
    print(f"  χ² (35 dof):           {chisq:.2f}  "
          f"(expected ≈ 35 under null)")
    print(f"\n  All 35 exact values are 0 — Y_uY_vY_w is NOT a K_7 "
          f"stabilizer.")
    print(f"  This is the cleanest experimental test: any nonzero hardware")
    print(f"  reading is pure noise, no theory parameter to fit.")


# =====================================================================
# Qiskit-on-backend (Aer or Heron)
# =====================================================================

def run_yyy_on_backend(backend, shots: int = 4000,
                        use_sampler: bool = False) -> None:
    """Run all 35 ⟨Y_u Y_v Y_w⟩ measurements on a Qiskit backend.

    use_sampler=True for IBM hardware (SamplerV2 primitive),
    use_sampler=False for AerSimulator (.run + .result()).
    """
    label = getattr(backend, 'name', backend.__class__.__name__)
    section(f'Experiment 4 on {label}  ({shots} shots × 35 triples)')

    print(f"\n  Building 35 YYY circuits...")
    circuits = [YYY_measurement_circuit(u, v, w) for (u, v, w) in TRIPLES]

    print(f"  Transpiling for backend...")
    transpiled = transpile(circuits, backend=backend, optimization_level=3,
                            seed_transpiler=42)

    cz_counts = [tc.count_ops().get('cz', 0) + tc.count_ops().get('ecr', 0)
                  for tc in transpiled]
    print(f"  CZ/ECR per circuit (after transpile): "
          f"min={min(cz_counts)}, max={max(cz_counts)}, "
          f"avg={sum(cz_counts)/len(cz_counts):.1f}")

    print(f"\n  Submitting {len(circuits)} circuits...")

    if use_sampler:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        sampler = Sampler(mode=backend)
        job = sampler.run(transpiled, shots=shots)
        print(f"    Job ID: {job.job_id()}")
        print(f"\n  Waiting for results...")
        print(f"  (Look up later at https://quantum.cloud.ibm.com)")
        result = job.result()
        get_counts = lambda i: result[i].data.c.get_counts()
    else:
        job = backend.run(transpiled, shots=shots)
        result = job.result()
        get_counts = lambda i: result.get_counts(i)

    print(f"  Job complete.\n")

    print(f"  {'triple':>10} {'⟨YYY⟩':>12} {'σ':>10} {'z':>8}")
    print('  ' + '-' * 44)
    means = np.zeros(35)
    sigmas = np.zeros(35)
    chisq = 0.0
    for i, (u, v, w) in enumerate(TRIPLES):
        counts = get_counts(i)
        mean, sigma = parse_YYY_counts(counts, shots)
        means[i] = mean
        sigmas[i] = sigma
        z = mean / sigma if sigma > 0 else 0.0
        chisq += z ** 2
        print(f"  ({u},{v},{w}){'':>3} {mean:>+12.4f} {sigma:>10.4f} "
              f"{z:>+8.2f}")

    rms = float(np.sqrt(np.mean(means ** 2)))
    max_abs = float(np.max(np.abs(means)))
    sum_mag = float(np.sum(np.abs(means)))
    aggr_var = float(np.sum(sigmas ** 2))
    print(f"\n  Aggregate test under null hypothesis ⟨YYY⟩ = 0 ∀ triples:")
    print(f"    Σ |⟨YYY⟩|:             {sum_mag:.4f}")
    print(f"    RMS ⟨YYY⟩:             {rms:.4f}")
    print(f"    max |⟨YYY⟩|:           {max_abs:.4f}")
    print(f"    χ² (35 dof):           {chisq:.2f}")
    print(f"    χ²_reduced:            {chisq / 35:.3f}  "
          f"(≈ 1.0 means consistent with shot-noise-only null)")

    # Bremermann/Compton: predicted noiseless RMS scales as 1/√shots
    expected_rms_noiseless = 1.0 / np.sqrt(shots)
    print(f"\n  Expected noiseless RMS at {shots} shots: "
          f"{expected_rms_noiseless:.4f}")
    if rms > 3 * expected_rms_noiseless:
        print(f"  Observed RMS is {rms / expected_rms_noiseless:.1f}× the "
              f"noiseless prediction — gate-error contribution dominates.")
    else:
        print(f"  Observed RMS is consistent with shot-noise-only null.")

    # Fano vs non-Fano breakdown — does PSL(2,7) edge structure leave a
    # signature in the residual hardware-noise pattern?
    print(f"\n  Fano-line vs non-Fano breakdown (PSL(2,7) edge structure):")
    chi_fano = 0.0
    chi_nonfano = 0.0
    z_fano = []
    z_nonfano = []
    for i, t in enumerate(TRIPLES):
        z = means[i] / sigmas[i] if sigmas[i] > 0 else 0.0
        if t in FANO_SET:
            chi_fano += z ** 2
            z_fano.append(z)
        else:
            chi_nonfano += z ** 2
            z_nonfano.append(z)

    z_fano = np.array(z_fano)
    z_nonfano = np.array(z_nonfano)
    chi_red_fano = chi_fano / 7
    chi_red_nonfano = chi_nonfano / 28

    print(f"    {'group':>10} {'n':>3} {'χ²':>8} {'χ²_red':>8} "
          f"{'mean z':>8} {'RMS |z|':>9}")
    print('    ' + '-' * 50)
    print(f"    {'Fano':>10} {7:>3} {chi_fano:>8.2f} "
          f"{chi_red_fano:>8.3f} {z_fano.mean():>+8.3f} "
          f"{np.sqrt(np.mean(z_fano**2)):>9.3f}")
    print(f"    {'non-Fano':>10} {28:>3} {chi_nonfano:>8.2f} "
          f"{chi_red_nonfano:>8.3f} {z_nonfano.mean():>+8.3f} "
          f"{np.sqrt(np.mean(z_nonfano**2)):>9.3f}")

    # Welch-style test for difference in mean z² between groups
    var_fano = np.var(z_fano ** 2, ddof=1) if len(z_fano) > 1 else 0.0
    var_nonfano = np.var(z_nonfano ** 2, ddof=1) if len(z_nonfano) > 1 else 0.0
    se_diff = np.sqrt(var_fano / len(z_fano)
                       + var_nonfano / len(z_nonfano))
    diff = float(np.mean(z_fano ** 2) - np.mean(z_nonfano ** 2))
    z_diff = diff / se_diff if se_diff > 0 else 0.0
    print(f"\n    Δ⟨z²⟩ = ⟨z²⟩_Fano - ⟨z²⟩_non-Fano = {diff:+.3f}")
    print(f"    SE(Δ) = {se_diff:.3f}")
    print(f"    z-score for Fano excess: {z_diff:+.2f}σ")
    if abs(z_diff) > 3:
        print(f"    → SIGNIFICANT Fano structure detected.")
    elif abs(z_diff) > 2:
        print(f"    → Suggestive but not conclusive.")
    else:
        print(f"    → Consistent with no Fano-structure preference.")


# =====================================================================
# Heron submission
# =====================================================================

def heron_submission(backend_name: str = None, shots: int = 4000) -> None:
    """Submit Experiment 4 to a Heron-class IBM backend."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("ERROR: qiskit-ibm-runtime not installed.")
        return

    section('Experiment 4: IBM Heron submission')

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
    print(f"  Shots:         {shots} per triple (×35 triples)")
    print(f"  Total shots:   {shots * 35}")

    run_yyy_on_backend(backend, shots=shots, use_sampler=True)


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Experiment 4 — ⟨Y_u Y_v Y_w⟩ = 0 null test on |K_7⟩')
    parser.add_argument('--shots', type=int, default=4000,
                        help='Shots per triple')
    parser.add_argument('--skip-numpy', action='store_true')
    parser.add_argument('--skip-aer', action='store_true')
    parser.add_argument('--backend', type=str, default=None,
                        help='IBM Heron backend name (e.g., ibm_marrakesh).  '
                             'If set, runs hardware submission.')
    args = parser.parse_args()

    section('Experiment 4 — ⟨Y_u Y_v Y_w⟩ = 0 on |K_7⟩')

    print(f"""
  Phase 8e null test on all 35 triples of K_7.
  Predicted: every ⟨Y_u Y_v Y_w⟩ = 0 EXACTLY.
  This is the cleanest hardware test — no theory parameter to fit,
  hardware noise shows up as variance only.
""")

    if not args.skip_numpy:
        reference_simulation(n_shots=args.shots)

    if QISKIT_AVAILABLE and not args.skip_aer:
        try:
            sim = AerSimulator()
            run_yyy_on_backend(sim, shots=args.shots, use_sampler=False)
        except Exception as e:
            print(f"\nAerSimulator part failed: {e}")

    if args.backend is not None:
        heron_submission(args.backend, shots=args.shots)


if __name__ == '__main__':
    main()
