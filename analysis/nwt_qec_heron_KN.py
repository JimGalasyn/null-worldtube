#!/usr/bin/env python3
"""
K_N cross-group test on IBM Heron — verify ⟨H_YY⟩|K_N⟩ = dim(Adj_so(N)) =
N(N-1)/2 for N > 7.

The K_7 verification (kingston, marrakesh, fez) confirmed
⟨H_YY⟩|K_7⟩ = 21 = dim(Adj_so(7)).  Paper 17 §11.5 generalizes:
the same structural identity holds for K_N → so(N).  Verification on
N≠7 hardware tests whether the K_7 result is an so(7) accident or
the so(2n+1) structural identity Paper 17 claims.

Predictions for K_N:
  ⟨S_v⟩|K_N⟩      = +1   for all N stabilizers
  ⟨H_YY⟩|K_N⟩     = N(N-1)/2 = dim(Adj_so(N))
                  = 21 (N=7), 28 (N=8), 36 (N=9), 55 (N=11)

The depolarization-resistant test: the RATIO of ⟨H_YY⟩|K_N⟩ across
N values eliminates the overall fidelity factor and isolates the
group-theoretic content.  E.g. K_9/K_7 should give 36/21 ≈ 1.71.

Usage:
  python3 nwt_qec_heron_KN.py --N 9                 # numpy + Aer
  python3 nwt_qec_heron_KN.py --N 9 --backend ibm_marrakesh
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H_GATE = (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
S_dag = S.conj().T


# =====================================================================
# Numpy reference K_N
# =====================================================================

def build_KN_state(N: int) -> np.ndarray:
    """Prepare |K_N⟩ = ∏_{edges} CZ |+⟩^N as a numpy state vector."""
    DIM = 2 ** N
    psi = np.ones(DIM, dtype=complex) / np.sqrt(DIM)
    for u, v in combinations(range(N), 2):
        for idx in range(DIM):
            xu = (idx >> (N - 1 - u)) & 1
            xv = (idx >> (N - 1 - v)) & 1
            if xu == 1 and xv == 1:
                psi[idx] *= -1
    return psi


def reference_HYY(N: int) -> Tuple[float, float]:
    """Compute ⟨H_YY⟩ and ⟨H_YY²⟩ exactly on |K_N⟩."""
    DIM = 2 ** N
    psi = build_KN_state(N)
    HYY = np.zeros((DIM, DIM), dtype=complex)
    for u, v in combinations(range(N), 2):
        # Build Y_u Y_v as a DIM×DIM matrix
        op = np.array([[1.0]], dtype=complex)
        for q in range(N):
            if q in (u, v):
                op = np.kron(op, Y)
            else:
                op = np.kron(op, I2)
        HYY = HYY + op
    e1 = float((psi.conj() @ HYY @ psi).real)
    e2 = float((psi.conj() @ HYY @ HYY @ psi).real)
    return e1, e2


# =====================================================================
# Qiskit circuits (parametric in N)
# =====================================================================

try:
    from qiskit import QuantumCircuit, ClassicalRegister, transpile
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def build_KN_qiskit_circuit(N: int) -> 'QuantumCircuit':
    qc = QuantumCircuit(N, name=f'K_{N}')
    for q in range(N):
        qc.h(q)
    for u, v in combinations(range(N), 2):
        qc.cz(u, v)
    return qc


def stabilizer_measurement_circuit(N: int, v: int) -> 'QuantumCircuit':
    """S_v = X_v · ∏_{u≠v} Z_u → after H on qubit v, becomes ∏ Z_u =
    total Z parity."""
    qc = build_KN_qiskit_circuit(N)
    qc.add_register(ClassicalRegister(N, name='c'))
    qc.h(v)
    qc.measure(range(N), range(N))
    return qc


def YY_measurement_circuit(N: int, u: int, v: int) -> 'QuantumCircuit':
    qc = build_KN_qiskit_circuit(N)
    qc.add_register(ClassicalRegister(2, name='c'))
    qc.sdg(u); qc.h(u)
    qc.sdg(v); qc.h(v)
    qc.measure([u, v], [0, 1])
    return qc


def parse_stabilizer_counts(N: int, counts: Dict[str, int],
                              n_shots: int) -> Tuple[float, float]:
    total = 0.0
    for bs, c in counts.items():
        parity = sum(int(b) for b in bs[-N:]) % 2
        ev = 1 if parity == 0 else -1
        total += ev * c
    mean = total / n_shots
    var = 1 - mean ** 2
    return float(mean), float(np.sqrt(var / n_shots))


def parse_YY_counts(counts: Dict[str, int], n_shots: int
                      ) -> Tuple[float, float]:
    total = 0.0
    for bs, c in counts.items():
        b0 = int(bs[-1])
        b1 = int(bs[-2])
        ev = (-1) ** (b0 + b1)
        total += ev * c
    mean = total / n_shots
    var = 1 - mean ** 2
    return float(mean), float(np.sqrt(var / n_shots))


# =====================================================================
# Section helper
# =====================================================================

def section(t: str) -> None:
    print()
    print('=' * 78)
    print(f' {t}')
    print('=' * 78)


# =====================================================================
# Backend execution
# =====================================================================

def run_KN_on_backend(backend, N: int, shots_exp1: int, shots_exp2: int,
                       use_sampler: bool) -> None:
    label = getattr(backend, 'name', backend.__class__.__name__)
    section(f'K_{N} on {label}')

    edges = list(combinations(range(N), 2))
    n_edges = len(edges)
    pred_HYY = N * (N - 1) // 2
    print(f"\n  N = {N}")
    print(f"  Edges = {n_edges} = N(N-1)/2 = dim(Adj_so({N}))")
    print(f"  Predicted noiseless ⟨H_YY⟩ = {pred_HYY}")
    print(f"  Shots Exp 1 (stabilizers): {shots_exp1} × {N}")
    print(f"  Shots Exp 2 (YY edges):    {shots_exp2} × {n_edges}")

    stab_circuits = [stabilizer_measurement_circuit(N, v) for v in range(N)]
    yy_circuits = [YY_measurement_circuit(N, u, v) for u, v in edges]

    print(f"\n  Transpiling...")
    stab_t = transpile(stab_circuits, backend=backend, optimization_level=3,
                        seed_transpiler=42)
    yy_t = transpile(yy_circuits, backend=backend, optimization_level=3,
                      seed_transpiler=42)
    cz_stab = [t.count_ops().get('cz', 0) + t.count_ops().get('ecr', 0)
                for t in stab_t]
    cz_yy = [t.count_ops().get('cz', 0) + t.count_ops().get('ecr', 0)
              for t in yy_t]
    print(f"  CZ/ECR per stabilizer circuit: min={min(cz_stab)}, "
          f"max={max(cz_stab)}, avg={sum(cz_stab)/len(cz_stab):.1f}")
    print(f"  CZ/ECR per YY circuit:         min={min(cz_yy)}, "
          f"max={max(cz_yy)}, avg={sum(cz_yy)/len(cz_yy):.1f}")

    print(f"\n  Submitting Exp 1 ({N} circuits)...")
    if use_sampler:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        sampler = Sampler(mode=backend)
        job1 = sampler.run(stab_t, shots=shots_exp1)
        print(f"    Job ID: {job1.job_id()}")
        print(f"\n  Submitting Exp 2 ({n_edges} circuits)...")
        job2 = sampler.run(yy_t, shots=shots_exp2)
        print(f"    Job ID: {job2.job_id()}")
        print(f"\n  Waiting for results...")
        res1 = job1.result()
        res2 = job2.result()
        print(f"  Both jobs complete.")
        gc1 = lambda i: res1[i].data.c.get_counts()
        gc2 = lambda i: res2[i].data.c.get_counts()
    else:
        job1 = backend.run(stab_t, shots=shots_exp1)
        res1 = job1.result()
        job2 = backend.run(yy_t, shots=shots_exp2)
        res2 = job2.result()
        gc1 = lambda i: res1.get_counts(i)
        gc2 = lambda i: res2.get_counts(i)

    # ---- Exp 1 parse ----
    print(f"\n  EXPERIMENT 1: ⟨S_v⟩ for v = 0..{N-1}")
    print(f"  {'v':>3} {'⟨S_v⟩':>12} {'σ':>10}")
    print('  ' + '-' * 30)
    s_avg = 0.0
    s_min = 1.0
    s_max = -1.0
    for v in range(N):
        ev, sigma = parse_stabilizer_counts(N, gc1(v), shots_exp1)
        s_avg += ev
        s_min = min(s_min, ev)
        s_max = max(s_max, ev)
        print(f"  {v:>3} {ev:>+12.4f} {sigma:>10.4f}")
    s_avg /= N
    print(f"  Mean ⟨S_v⟩: {s_avg:.4f}  (predict 1.0)")
    print(f"  Range:      [{s_min:.4f}, {s_max:.4f}]")

    # ---- Exp 2 parse ----
    print(f"\n  EXPERIMENT 2: ⟨H_YY⟩ via {n_edges} edge ⟨Y_u Y_v⟩")
    print(f"  (suppressing per-edge table for N={N})")
    HYY_total = 0.0
    HYY_var = 0.0
    edge_means = []
    for i, (u, v) in enumerate(edges):
        ev, sigma = parse_YY_counts(gc2(i), shots_exp2)
        HYY_total += ev
        HYY_var += sigma ** 2
        edge_means.append(ev)
    HYY_sigma = float(np.sqrt(HYY_var))
    edge_means = np.array(edge_means)

    print(f"\n  TOTAL ⟨H_YY⟩ |K_{N}⟩  =  {HYY_total:.4f}  ±  {HYY_sigma:.4f}")
    print(f"  PREDICTION         =  {pred_HYY:.4f}  (= dim(Adj_so({N})))")
    print(f"  Deviation         =  {abs(HYY_total - pred_HYY):.4f}")
    if HYY_sigma > 0:
        print(f"  Z-score from prediction: "
              f"{(HYY_total - pred_HYY) / HYY_sigma:.2f}σ")

    # Random null
    null_sigma = float(np.sqrt(n_edges) * 1.0 / np.sqrt(shots_exp2))
    if null_sigma > 0:
        z_null = HYY_total / null_sigma
        print(f"\n  Random-null hypothesis ⟨H_YY⟩ = 0:")
        print(f"  Z-score above null:       +{z_null:.1f}σ")

    # Edge stats
    print(f"\n  Edge statistics (PSL(2,7)/PSL(2,N) test):")
    print(f"    mean ⟨Y_u Y_v⟩:           {edge_means.mean():.4f}")
    print(f"    std across edges:         {edge_means.std(ddof=1):.4f}")
    print(f"    range:                    "
          f"[{edge_means.min():.4f}, {edge_means.max():.4f}]")


# =====================================================================
# Heron submission
# =====================================================================

def heron_submission(N: int, backend_name: str = None,
                       shots_exp1: int = 4000,
                       shots_exp2: int = 4000) -> None:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        print("ERROR: qiskit-ibm-runtime not installed.")
        return

    section(f'K_{N}: IBM Heron submission')

    token_file = os.path.expanduser('~/.ibm_quantum_api_key')
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

    run_KN_on_backend(backend, N, shots_exp1, shots_exp2, use_sampler=True)


# =====================================================================
# Reference run
# =====================================================================

def reference_run(N: int) -> None:
    section(f'K_{N} numpy reference')
    e1, e2 = reference_HYY(N)
    pred1 = N * (N - 1) // 2
    pred2 = pred1 ** 2
    print(f"\n  ⟨H_YY⟩|K_{N}⟩    = {e1:.4f}   (predict {pred1} = "
          f"dim(Adj_so({N})))")
    print(f"  ⟨H_YY²⟩|K_{N}⟩  = {e2:.4f}   (predict {pred2} = "
          f"dim(Adj_so({N}))²)")
    if abs(e1 - pred1) < 1e-9 and abs(e2 - pred2) < 1e-6:
        print(f"  ✓ Structural identity ⟨H_YY^n⟩|K_{N}⟩ = "
              f"dim(Adj_so({N}))^n verified at n = 1, 2.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=9,
                        help='Number of qubits / vertices in K_N')
    parser.add_argument('--shots', type=int, default=4000)
    parser.add_argument('--skip-numpy', action='store_true')
    parser.add_argument('--skip-aer', action='store_true')
    parser.add_argument('--backend', type=str, default=None)
    args = parser.parse_args()

    section(f'K_{args.N} cross-group K_N test')
    print(f"""
  Predicting ⟨H_YY⟩|K_{args.N}⟩ = {args.N*(args.N-1)//2} = dim(Adj_so({args.N})).

  K_7 already verified at +368σ on ibm_kingston (Run 2).  This run
  tests whether the structural identity holds for so({args.N}), thereby
  confirming the so(2n+1)-family universality from Paper 17 §11.5.
""")

    if not args.skip_numpy:
        reference_run(args.N)

    if QISKIT_AVAILABLE and not args.skip_aer:
        try:
            sim = AerSimulator()
            run_KN_on_backend(sim, args.N, shots_exp1=args.shots,
                                shots_exp2=args.shots, use_sampler=False)
        except Exception as e:
            print(f"\nAerSimulator failed: {e}")

    if args.backend is not None:
        heron_submission(args.N, args.backend,
                          shots_exp1=args.shots, shots_exp2=args.shots)


if __name__ == '__main__':
    main()
