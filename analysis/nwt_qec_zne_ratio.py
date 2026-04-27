#!/usr/bin/env python3
"""K_9_ZNE / K_7_ZNE ratio test on ibm_fez.

Both K_7 and K_9 have ZNE (lambda in {1, 3}, CZ-only fold).  Apply
linear and exponential extrapolation per-edge, sum to <H_YY>, then
take the ratio K_9 / K_7.  The non-foldable error floor (state prep,
readout, idle T1/T2) cancels in the ratio if it's universal across
N -- this gives a sharp test of the structural identity
<H_YY>|K_N> = N(N-1)/2 = dim(Adj_so(N)).
"""

from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nwt_qec_heron_KN import parse_YY_counts, section
from nwt_qec_zne_reanalysis import extrap_linear, extrap_exponential


CRN = ('crn:v1:bluemix:public:quantum-computing:us-east:'
        'a/c84951b63d86411c876c4fd868dee84a:'
        'e7173ddf-4db9-438c-ad3c-e0b49469f0ed::')

# Job IDs of the two ZNE runs
JOB_IDS = {
    7: {1: 'd7n9812k4prs73drtvpg', 3: 'd7n981ik4prs73drtvq0'},
    9: {1: 'd7n93us97osc73drnnog', 3: 'd7n956a94dqs73eunus0'},
}


def fetch_zne(service, N: int, shots: int = 4000):
    edges = list(combinations(range(N), 2))
    n_edges = len(edges)
    means_lambda = {}
    sigmas_lambda = {}
    for s, jid in JOB_IDS[N].items():
        res = service.job(jid).result()
        means = np.zeros(n_edges)
        sigmas = np.zeros(n_edges)
        for i, (u, v) in enumerate(edges):
            counts = res[i].data.c.get_counts()
            m, sig = parse_YY_counts(counts, shots)
            means[i] = m
            sigmas[i] = sig
        means_lambda[s] = means
        sigmas_lambda[s] = sigmas
    return edges, means_lambda, sigmas_lambda


def main() -> None:
    section('K_9 / K_7 ZNE ratio test on ibm_fez')

    from qiskit_ibm_runtime import QiskitRuntimeService
    token = open(os.path.expanduser('~/.ibm_quantum_api_key')).read().strip()
    service = QiskitRuntimeService(channel='ibm_cloud', token=token,
                                     instance=CRN)

    print(f"\n  Fetching K_7 + K_9 ZNE results (4 jobs)...")
    data = {}
    for N in (7, 9):
        edges, m_lam, s_lam = fetch_zne(service, N)
        data[N] = dict(edges=edges, m=m_lam, s=s_lam)
        print(f"    K_{N}: λ=1 ⟨H_YY⟩={m_lam[1].sum():.4f}, "
              f"λ=3 ⟨H_YY⟩={m_lam[3].sum():.4f}")

    # Per-edge extrapolation, sum to ⟨H_YY⟩(0)
    print(f"\n  Per-N ZNE extrapolation:")
    print(f"  {'N':>3} {'method':>14} {'⟨H_YY⟩(0)':>12} {'σ':>8} "
          f"{'pred':>5} {'/pred':>8}")
    print('  ' + '-' * 60)

    HYY = {}  # HYY[(N, method)] = (mean, sigma)
    for N in (7, 9):
        v1 = data[N]['m'][1]
        v3 = data[N]['m'][3]
        s1 = data[N]['s'][1]
        s3 = data[N]['s'][3]
        pred = N * (N - 1) // 2

        m_lin, sig_lin = extrap_linear(v1, v3, s1, s3)
        m_exp, sig_exp = extrap_exponential(v1, v3, s1, s3)

        H_lin = float(m_lin.sum())
        sigH_lin = float(np.sqrt((sig_lin ** 2).sum()))
        H_exp = float(m_exp.sum())
        sigH_exp = float(np.sqrt((sig_exp ** 2).sum()))

        HYY[(N, 'linear')] = (H_lin, sigH_lin)
        HYY[(N, 'exponential')] = (H_exp, sigH_exp)

        print(f"  {N:>3} {'linear':>14} {H_lin:>+12.4f} "
              f"{sigH_lin:>8.4f} {pred:>5} {H_lin/pred:>8.4f}")
        print(f"  {N:>3} {'exponential':>14} {H_exp:>+12.4f} "
              f"{sigH_exp:>8.4f} {pred:>5} {H_exp/pred:>8.4f}")

    # Ratio test
    section('Cross-group ratio test K_9_ZNE / K_7_ZNE')

    pred_ratio = 36 / 21
    print(f"\n  Structural prediction: {pred_ratio:.4f}\n")
    print(f"  {'method':>14} {'K_9_ZNE':>10} {'K_7_ZNE':>10} "
          f"{'ratio':>10} {'σ_ratio':>10} {'(ratio-pred)/pred':>20} "
          f"{'z':>8}")
    print('  ' + '-' * 80)
    for method in ('linear', 'exponential'):
        H7, s7 = HYY[(7, method)]
        H9, s9 = HYY[(9, method)]
        ratio = H9 / H7
        # Var(H9/H7) ≈ (1/H7)² Var(H9) + (H9/H7²)² Var(H7)
        var_ratio = (s9 / H7) ** 2 + (H9 * s7 / H7 ** 2) ** 2
        sig_ratio = np.sqrt(var_ratio)
        deviation = (ratio - pred_ratio) / pred_ratio
        z_score = (ratio - pred_ratio) / sig_ratio
        print(f"  {method:>14} {H9:>+10.4f} {H7:>+10.4f} "
              f"{ratio:>10.4f} {sig_ratio:>10.4f} "
              f"{deviation:>+19.2%}  {z_score:>+7.2f}")

    # Falsification of alternative M(K_9) values via ratio
    print(f"\n  Falsification of alternative M(K_9) via exponential ratio:")
    H7_exp, s7_exp = HYY[(7, 'exponential')]
    H9_exp, s9_exp = HYY[(9, 'exponential')]
    obs_ratio = H9_exp / H7_exp
    sig_obs_ratio = np.sqrt((s9_exp/H7_exp)**2 + (H9_exp*s7_exp/H7_exp**2)**2)

    print(f"  Observed K_9_ZNE / K_7_ZNE = {obs_ratio:.4f} ± {sig_obs_ratio:.4f}")
    print(f"  Predicted (M(K_9)=36, M(K_7)=21): {36/21:.4f}\n")

    print(f"  {'M(K_9)':>8} {'pred ratio':>12} {'(pred-obs)/obs':>18} "
          f"{'z':>8} {'verdict':>10}")
    print('  ' + '-' * 65)
    for M_K9 in [21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51]:
        pred_r = M_K9 / 21
        deviation = (pred_r - obs_ratio) / obs_ratio
        z = (pred_r - obs_ratio) / sig_obs_ratio
        verdict = '✓' if abs(z) < 3 else '✗'
        print(f"  {M_K9:>8} {pred_r:>12.4f} {deviation:>+17.2%}  "
              f"{z:>+7.2f} {verdict:>10}")

    # Final usage check
    u = service.usage()
    print(f"\n  Final PAYG usage: {u['usage_consumed_seconds']}/"
          f"{u['usage_limit_seconds']} sec")


if __name__ == '__main__':
    main()
