#!/usr/bin/env python3
"""Fetch + parse a Heron job result by job_id.

Usage:
  python3 nwt_qec_heron_fetch.py <job_id> --exp {2,4,5}
"""
import argparse
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_exp5(result, shots: int) -> None:
    from nwt_qec_heron_exp5 import counts_to_dist, summarize, shannon_entropy
    labels = ['|0⟩^7', '|+⟩^7', '|K_7⟩']
    summaries = {}
    for i, lbl in enumerate(labels):
        counts = result[i].data.c.get_counts()
        p = counts_to_dist(counts, shots)
        predicted_uniform = (lbl != '|K_7⟩')
        print()
        summaries[lbl] = summarize(lbl, p, predicted_uniform)

    print()
    print('  Aggregate signal:')
    h_zero = summaries['|0⟩^7']['entropy']
    h_K7 = summaries['|K_7⟩']['entropy']
    print(f'    H_2(|0⟩^7) = {h_zero:.3f}  (predict 7.000)')
    print(f'    H_2(|K_7⟩) = {h_K7:.3f}  (predict 0.000)')
    print(f'    Δ entropy  = {h_zero - h_K7:.3f} bits')
    print(f'    P(0|K_7 input) = {summaries["|K_7⟩"]["p_zero"]:.4f}  '
          f'(round-trip fidelity)')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('job_id')
    parser.add_argument('--exp', type=int, default=5,
                        help='Experiment number (5 supported)')
    parser.add_argument('--shots', type=int, default=4000)
    args = parser.parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService
    token = open(os.path.expanduser('~/.ibm_quantum_api_key')).read().strip()
    service = QiskitRuntimeService(channel='ibm_cloud', token=token)
    job = service.job(args.job_id)
    print(f'Job ID:   {args.job_id}')
    print(f'Backend:  {job.backend().name}')
    print(f'Status:   {job.status()}')

    print('Waiting for result...')
    result = job.result()
    print('Result retrieved.\n')

    if args.exp == 5:
        parse_exp5(result, args.shots)
    else:
        print(f'Unsupported --exp {args.exp}')
        sys.exit(1)


if __name__ == '__main__':
    main()
