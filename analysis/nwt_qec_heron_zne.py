#!/usr/bin/env python3
"""
Zero-noise extrapolation (ZNE) for ⟨H_YY⟩|K_N⟩ on Heron.

Standard error-mitigation technique: run the same circuit at multiple
"noise scaling factors" λ ∈ {1, 3, 5, ...}, then extrapolate the
observable to λ → 0.  Identity folding (each gate G replaced by G ·
G⁻¹ · G ... so the logical operation is preserved but gate count
multiplies by λ) is the standard implementation.

For our K_N |K_N⟩ measurement:
  ⟨Y_u Y_v⟩(λ) → fit a model in λ → extrapolate to λ = 0
  ⟨H_YY⟩|K_N⟩(λ=0) = sum over edges of extrapolated values

Linear extrapolation from λ ∈ {1, 3}:
  ⟨Y_u Y_v⟩(0) = (3·⟨Y_u Y_v⟩(1) - ⟨Y_u Y_v⟩(3)) / 2

Variance under this 2-point linear ZNE:
  σ²_ZNE = (9·σ²(1) + σ²(3)) / 4 ≈ 2.5·σ²(1) for similar shot counts

Each gate-folded circuit is 3× as deep as the original, so QPU time
scales with the SUM of folded depths.  For K_9 with original 88 CZs,
folded version has 264 CZs.

Usage:
  python3 nwt_qec_heron_zne.py --N 9 --backend ibm_fez --shots 8000
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nwt_qec_heron_KN import (
    YY_measurement_circuit, parse_YY_counts, section
)


# =====================================================================
# Identity-folding noise amplification
# =====================================================================

def fold_circuit(qc, scale: int, fold_gates=('cz',)):
    """Replace each gate G in `fold_gates` with the sequence
    G · G^(-1) · G · G^(-1) · ... so that the gate count for those
    gates becomes scale × original.  scale must be a positive odd
    integer ≥ 1.

    Only the named gates are folded.  Default is CZ-only ("two-qubit
    gate folding"), which is the standard recipe for ZNE on graph-
    state experiments where 2-qubit error dominates.  Folding all
    gates would pull in sxdg (SX-dagger) which is NOT in the Heron
    R2 native ISA.  CZ is self-inverse so its fold is just CZ·CZ·CZ.
    """
    from qiskit import QuantumCircuit
    if scale < 1 or scale % 2 == 0:
        raise ValueError(f"scale must be a positive odd integer, got {scale}")
    n_extra_pairs = (scale - 1) // 2

    folded = QuantumCircuit(*qc.qregs, *qc.cregs)
    for instr in qc.data:
        op = instr.operation
        qubits = instr.qubits
        clbits = instr.clbits
        folded.append(op, qubits, clbits)
        if op.name in fold_gates and n_extra_pairs > 0:
            for _ in range(n_extra_pairs):
                # For self-inverse gates (cz, x, h), inverse() == op.
                # For asymmetric gates, this would be op.inverse(),
                # but we restrict fold_gates to self-inverse only by
                # default (cz on Heron).
                folded.append(op.inverse(), qubits, clbits)
                folded.append(op, qubits, clbits)
    return folded


# =====================================================================
# ZNE on K_N
# =====================================================================

def submit_zne_K_N(N: int, backend, shots: int,
                     scales: List[int] = (1, 3)) -> dict:
    """Submit ZNE for K_N ⟨H_YY⟩.  Returns observed values per scale."""
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    section(f'K_{N} ZNE on {backend.name}')
    edges = list(combinations(range(N), 2))
    n_edges = len(edges)
    print(f"\n  Scales: {list(scales)}  shots/circuit: {shots}")
    print(f"  Total circuits: {len(scales)} × {n_edges} = "
          f"{len(scales) * n_edges}")
    print(f"  Approximate QPU sec ~ depth × shots × Nedges / 1e6 ...")

    # Build YY circuits at logical level
    yy_logical = [YY_measurement_circuit(N, u, v) for u, v in edges]

    # Transpile ONCE per scale, NOT after folding (folding is gate-level)
    print(f"\n  Transpiling base (λ=1)...")
    yy_base = transpile(yy_logical, backend=backend, optimization_level=3,
                         seed_transpiler=42)
    cz_base = [t.count_ops().get('cz', 0) for t in yy_base]
    print(f"  Base CZ count per circuit: avg {sum(cz_base)/len(cz_base):.1f}")

    sampler = Sampler(mode=backend)

    # Submit each scale.  λ=1 is the base.  λ=3, 5, ... fold the base
    # circuit at the gate level (so routing is preserved).
    job_ids = {}
    yy_folded = {1: yy_base}
    for s in scales:
        if s == 1:
            continue
        folded = [fold_circuit(c, s) for c in yy_base]
        cz_folded = [t.count_ops().get('cz', 0) for t in folded]
        print(f"  Folded λ={s}: CZ count per circuit avg "
              f"{sum(cz_folded)/len(cz_folded):.1f} (vs "
              f"{sum(cz_base)/len(cz_base):.1f} at λ=1)")
        yy_folded[s] = folded

    print(f"\n  Submitting {len(scales)} batches...")
    jobs = {}
    for s in scales:
        job = sampler.run(yy_folded[s], shots=shots)
        jobs[s] = job
        job_ids[s] = job.job_id()
        print(f"    λ={s}: Job ID {job.job_id()}")

    print(f"\n  Waiting for all results...")
    results = {}
    for s, job in jobs.items():
        results[s] = job.result()
        print(f"    λ={s}: result retrieved")
    print(f"  All jobs complete.\n")

    # Parse counts → expectation values per (λ, edge)
    edge_data: Dict[int, np.ndarray] = {}
    edge_sigma: Dict[int, np.ndarray] = {}
    for s in scales:
        means = np.zeros(n_edges)
        sigmas = np.zeros(n_edges)
        for i, (u, v) in enumerate(edges):
            counts = results[s][i].data.c.get_counts()
            m, sig = parse_YY_counts(counts, shots)
            means[i] = m
            sigmas[i] = sig
        edge_data[s] = means
        edge_sigma[s] = sigmas

    return dict(N=N, edges=edges, scales=list(scales),
                edge_data=edge_data, edge_sigma=edge_sigma,
                job_ids=job_ids, shots=shots,
                cz_base=sum(cz_base)/len(cz_base))


# =====================================================================
# Linear extrapolation in λ
# =====================================================================

def linear_extrapolate(scales: List[int],
                         values: List[np.ndarray],
                         sigmas: List[np.ndarray]) -> Tuple[np.ndarray,
                                                              np.ndarray]:
    """Linear extrapolation of <O>(λ) to λ=0 using least-squares fit.

    For 2 points (λ_1, λ_2) the linear extrapolation is:
      f(0) = (λ_2 · f(λ_1) - λ_1 · f(λ_2)) / (λ_2 - λ_1)
    For (1, 3): f(0) = (3 f(1) - f(3)) / 2.

    Returns (extrapolated_means, extrapolated_sigmas).
    """
    if len(scales) == 2:
        s1, s2 = scales
        f1, f2 = values
        sig1, sig2 = sigmas
        coef1 = s2 / (s2 - s1)
        coef2 = -s1 / (s2 - s1)
        means = coef1 * f1 + coef2 * f2
        sigsq = (coef1 ** 2) * (sig1 ** 2) + (coef2 ** 2) * (sig2 ** 2)
        return means, np.sqrt(sigsq)

    # Multi-point: weighted least-squares linear fit per edge
    n_edges = len(values[0])
    means = np.zeros(n_edges)
    sigsq = np.zeros(n_edges)
    for e in range(n_edges):
        x = np.array(scales, dtype=float)
        y = np.array([v[e] for v in values])
        s = np.array([sg[e] for sg in sigmas])
        w = 1.0 / s ** 2
        # Linear fit y = a + b·x, evaluate at x=0
        Sw = w.sum()
        Sx = (w * x).sum()
        Sxx = (w * x * x).sum()
        Sy = (w * y).sum()
        Sxy = (w * x * y).sum()
        det = Sw * Sxx - Sx ** 2
        a = (Sxx * Sy - Sx * Sxy) / det
        # Var(a) = Sxx / det
        means[e] = a
        sigsq[e] = Sxx / det
    return means, np.sqrt(sigsq)


# =====================================================================
# Reporting
# =====================================================================

def report(result: dict, N_target: int) -> None:
    section(f'ZNE result for ⟨H_YY⟩|K_{result["N"]}⟩')

    n_edges = len(result['edges'])
    pred = N_target * (N_target - 1) // 2

    # Per-scale ⟨H_YY⟩
    print(f"\n  Per-scale ⟨H_YY⟩:")
    print(f"  {'λ':>3} {'⟨H_YY⟩':>12} {'σ_total':>10} {'avg ⟨YY⟩':>12}")
    print('  ' + '-' * 42)
    HYY_per_scale = {}
    sigma_per_scale = {}
    for s in result['scales']:
        v = result['edge_data'][s]
        sig = result['edge_sigma'][s]
        HYY = float(v.sum())
        sigma_HYY = float(np.sqrt((sig ** 2).sum()))
        HYY_per_scale[s] = HYY
        sigma_per_scale[s] = sigma_HYY
        avg_yy = float(v.mean())
        print(f"  {s:>3} {HYY:>+12.4f} {sigma_HYY:>10.4f} {avg_yy:>12.4f}")

    # Linear extrapolation per edge → sum to get ⟨H_YY⟩(λ=0)
    scales = result['scales']
    means_list = [result['edge_data'][s] for s in scales]
    sigmas_list = [result['edge_sigma'][s] for s in scales]
    extrap_means, extrap_sigmas = linear_extrapolate(scales, means_list,
                                                        sigmas_list)
    HYY_zne = float(extrap_means.sum())
    sigma_zne = float(np.sqrt((extrap_sigmas ** 2).sum()))

    print(f"\n  ZNE-extrapolated ⟨H_YY⟩|K_{result['N']}⟩(λ=0):")
    print(f"    Linear extrapolation:  {HYY_zne:.4f}  ±  {sigma_zne:.4f}")
    print(f"    Structural prediction: {pred}  (= dim(Adj_so({N_target})))")
    print(f"    Deviation:             {HYY_zne - pred:+.4f}")
    print(f"    Z-score from pred:     {(HYY_zne - pred) / sigma_zne:+.2f}σ")
    print(f"    Relative deviation:    {(HYY_zne - pred)/pred:+.2%}")

    # Falsification
    print(f"\n  Falsification of alternative M values "
          f"(linear ZNE extrapolation):")
    for M_alt in [pred - 6, pred - 3, pred, pred + 3, pred + 6]:
        z = (HYY_zne - M_alt) / sigma_zne
        ok = "  ✓" if abs(z) < 3 else "  ✗"
        print(f"    M = {M_alt:>3}  z = {z:+6.2f}σ{ok}")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=9)
    parser.add_argument('--backend', type=str, default='ibm_fez')
    parser.add_argument('--shots', type=int, default=8000)
    parser.add_argument('--instance', type=str, default=None,
                        help='IBM Cloud instance CRN (for PAYG); '
                             'default uses open plan')
    parser.add_argument('--scales', type=int, nargs='+', default=[1, 3])
    args = parser.parse_args()

    section('Zero-noise extrapolation for ⟨H_YY⟩|K_N⟩')
    print(f"""
  N        = {args.N}
  Backend  = {args.backend}
  Shots    = {args.shots} per circuit
  Scales   = {args.scales}
  Instance = {args.instance or 'open plan (default)'}
""")

    from qiskit_ibm_runtime import QiskitRuntimeService
    token = open(os.path.expanduser('~/.ibm_quantum_api_key')).read().strip()
    kwargs = dict(channel='ibm_cloud', token=token)
    if args.instance:
        kwargs['instance'] = args.instance
    service = QiskitRuntimeService(**kwargs)
    backend = service.backend(args.backend)

    result = submit_zne_K_N(args.N, backend, args.shots, args.scales)
    report(result, args.N)


if __name__ == '__main__':
    main()
