#!/usr/bin/env python3
"""Resume a partial ZNE run: pick up an already-submitted λ=1 job,
submit λ=3, combine, and report.

This is a one-off helper — use only if a prior submission failed
mid-flight (e.g., because of an ISA-incompatible fold).  Going
forward, just use nwt_qec_heron_zne.py end-to-end.
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nwt_qec_heron_KN import (
    YY_measurement_circuit, parse_YY_counts, section
)
from nwt_qec_heron_zne import fold_circuit, linear_extrapolate, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--backend', type=str, required=True)
    parser.add_argument('--shots', type=int, required=True)
    parser.add_argument('--instance', type=str, required=True)
    parser.add_argument('--lambda1-job', type=str, required=True,
                        help='Existing λ=1 SamplerV2 job id')
    parser.add_argument('--scale-3', type=int, default=3)
    args = parser.parse_args()

    section(f'ZNE continuation: pick up λ=1 job, submit λ={args.scale_3}')
    print(f"\n  N        = {args.N}")
    print(f"  backend  = {args.backend}")
    print(f"  shots    = {args.shots}")
    print(f"  λ=1 job  = {args.lambda1_job}")

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit import transpile

    token = open(os.path.expanduser('~/.ibm_quantum_api_key')).read().strip()
    service = QiskitRuntimeService(channel='ibm_cloud', token=token,
                                     instance=args.instance)
    backend = service.backend(args.backend)

    # Re-build & re-transpile the base circuits with the same seed
    edges = list(combinations(range(args.N), 2))
    n_edges = len(edges)
    yy_logical = [YY_measurement_circuit(args.N, u, v) for u, v in edges]
    print(f"\n  Re-transpiling base (λ=1) circuits (seed=42)...")
    yy_base = transpile(yy_logical, backend=backend, optimization_level=3,
                         seed_transpiler=42)

    # Fold for λ=scale_3 with CZ-only fold (avoids sxdg ISA issue)
    print(f"  Folding for λ={args.scale_3} (CZ-only)...")
    yy_folded = [fold_circuit(c, args.scale_3, fold_gates=('cz',))
                  for c in yy_base]
    cz_base = sum(c.count_ops().get('cz', 0) for c in yy_base) / len(yy_base)
    cz_folded = sum(c.count_ops().get('cz', 0)
                     for c in yy_folded) / len(yy_folded)
    print(f"    CZ avg: λ=1 → {cz_base:.1f}, λ={args.scale_3} → "
          f"{cz_folded:.1f} (ratio {cz_folded/cz_base:.2f})")

    # Submit λ=3
    print(f"\n  Submitting λ={args.scale_3} batch...")
    sampler = Sampler(mode=backend)
    job3 = sampler.run(yy_folded, shots=args.shots)
    print(f"    λ={args.scale_3} Job ID: {job3.job_id()}")

    # Fetch λ=1
    print(f"\n  Fetching λ=1 result...")
    job1 = service.job(args.lambda1_job)
    print(f"    λ=1 status: {job1.status()}")
    res1 = job1.result()
    print(f"  λ=1 result retrieved.")

    print(f"\n  Waiting for λ={args.scale_3}...")
    res3 = job3.result()
    print(f"  λ={args.scale_3} result retrieved.\n")

    # Parse counts
    edge_data = {1: np.zeros(n_edges), args.scale_3: np.zeros(n_edges)}
    edge_sigma = {1: np.zeros(n_edges), args.scale_3: np.zeros(n_edges)}
    for i, (u, v) in enumerate(edges):
        m1, s1 = parse_YY_counts(res1[i].data.c.get_counts(), args.shots)
        m3, s3 = parse_YY_counts(res3[i].data.c.get_counts(), args.shots)
        edge_data[1][i] = m1
        edge_sigma[1][i] = s1
        edge_data[args.scale_3][i] = m3
        edge_sigma[args.scale_3][i] = s3

    result = dict(N=args.N, edges=edges,
                   scales=[1, args.scale_3],
                   edge_data=edge_data, edge_sigma=edge_sigma,
                   shots=args.shots)
    report(result, args.N)

    # Final usage check
    u = service.usage()
    print(f"\n  PAYG usage: {u['usage_consumed_seconds']}/{u['usage_limit_seconds']} sec")


if __name__ == '__main__':
    main()
