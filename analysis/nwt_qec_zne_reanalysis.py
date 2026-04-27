#!/usr/bin/env python3
"""Re-analyze the K_9 ZNE run with both linear and exponential
extrapolation models.  No new QPU.

For depolarizing noise on each fold-affected gate, ⟨P⟩(λ) decays
exponentially in λ (since folding multiplies gate count, and
fidelity is multiplicative across gates).  Linear extrapolation
under-corrects.  Exponential is the physically motivated model:

    ⟨P⟩(λ) = c · b^λ
    with c = ⟨P⟩(λ=0) the noiseless value
         b = per-noise-unit fidelity factor

For 2 points (λ=1, λ=3):
    b² = ⟨P⟩(3) / ⟨P⟩(1)
    c  = ⟨P⟩(1) / b = ⟨P⟩(1) × sqrt(⟨P⟩(1) / ⟨P⟩(3))
"""

from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nwt_qec_heron_KN import parse_YY_counts, section


def extrap_linear(v1, v3, s1, s3):
    """Linear in λ at points (1, 3): f(0) = (3 v1 - v3) / 2."""
    means = (3 * v1 - v3) / 2
    sigsq = (9 * s1 ** 2 + s3 ** 2) / 4
    return means, np.sqrt(sigsq)


def extrap_exponential(v1, v3, s1, s3):
    """Exponential model f(λ) = c · b^λ with 2 anchors:
        b = sqrt(v3 / v1),  c = v1 / b = v1 · sqrt(v1 / v3)

    Variance via delta method:
        ∂c/∂v1 = (3/2) sqrt(v1 / v3),
        ∂c/∂v3 = -(1/2) v1^(3/2) v3^(-3/2)
    """
    # Mask: only valid where v1, v3 > 0 and same sign
    same_sign = (v1 * v3) > 0
    means = np.zeros_like(v1)
    sigmas = np.zeros_like(v1)
    if not np.all(same_sign):
        # Fall back to linear for sign-mismatched edges
        m_l, s_l = extrap_linear(v1, v3, s1, s3)
        means[~same_sign] = m_l[~same_sign]
        sigmas[~same_sign] = s_l[~same_sign]

    valid = same_sign
    v1v = np.abs(v1[valid])
    v3v = np.abs(v3[valid])
    sign = np.sign(v1[valid])
    b = np.sqrt(v3v / v1v)
    c = v1v / b   # always positive
    means[valid] = sign * c
    # Variance
    dcdv1 = 1.5 * np.sqrt(v1v / v3v)
    dcdv3 = -0.5 * v1v ** 1.5 / v3v ** 1.5
    var = (dcdv1 ** 2) * s1[valid] ** 2 + (dcdv3 ** 2) * s3[valid] ** 2
    sigmas[valid] = np.sqrt(var)
    return means, sigmas


def main() -> None:
    section('K_9 ZNE re-analysis: linear vs exponential extrapolation')

    crn = 'crn:v1:bluemix:public:quantum-computing:us-east:a/c84951b63d86411c876c4fd868dee84a:e7173ddf-4db9-438c-ad3c-e0b49469f0ed::'
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = open(os.path.expanduser('~/.ibm_quantum_api_key')).read().strip()
    service = QiskitRuntimeService(channel='ibm_cloud', token=token,
                                     instance=crn)

    # Same job IDs as the continuation run
    job_ids = {1: 'd7n93us97osc73drnnog',
                3: 'd7n956a94dqs73eunus0'}

    edges = list(combinations(range(9), 2))
    n_edges = len(edges)
    shots = 4000

    print(f"\n  Fetching results...")
    edge_data = {}
    edge_sigma = {}
    for s, jid in job_ids.items():
        job = service.job(jid)
        res = job.result()
        means = np.zeros(n_edges)
        sigmas = np.zeros(n_edges)
        for i, (u, v) in enumerate(edges):
            counts = res[i].data.c.get_counts()
            m, sig = parse_YY_counts(counts, shots)
            means[i] = m
            sigmas[i] = sig
        edge_data[s] = means
        edge_sigma[s] = sigmas
        print(f"    λ={s}: ⟨H_YY⟩ = {means.sum():+.4f} ± "
              f"{np.sqrt((sigmas**2).sum()):.4f}")

    # Linear and exponential extrapolation per edge
    v1 = edge_data[1]
    v3 = edge_data[3]
    s1 = edge_sigma[1]
    s3 = edge_sigma[3]

    print(f"\n  Distribution of ⟨YY⟩ values:")
    print(f"    λ=1: min {v1.min():.4f}, max {v1.max():.4f}, "
          f"avg {v1.mean():.4f}")
    print(f"    λ=3: min {v3.min():.4f}, max {v3.max():.4f}, "
          f"avg {v3.mean():.4f}")
    print(f"    λ=3 negative edges: {(v3 < 0).sum()} / 36 "
          f"(noise pushes some below zero)")

    means_lin, sigmas_lin = extrap_linear(v1, v3, s1, s3)
    means_exp, sigmas_exp = extrap_exponential(v1, v3, s1, s3)

    HYY_lin = float(means_lin.sum())
    sig_lin = float(np.sqrt((sigmas_lin ** 2).sum()))
    HYY_exp = float(means_exp.sum())
    sig_exp = float(np.sqrt((sigmas_exp ** 2).sum()))

    pred = 36

    print(f"\n  ZNE-extrapolated ⟨H_YY⟩|K_9⟩(λ=0):")
    print(f"  {'method':>16} {'⟨H_YY⟩(0)':>12} {'σ':>8} "
          f"{'/ pred':>8} {'(o-p)/σ':>9}")
    print('  ' + '-' * 60)
    print(f"  {'Linear':>16} {HYY_lin:>+12.4f} {sig_lin:>8.4f} "
          f"{HYY_lin/pred:>8.4f} {(HYY_lin-pred)/sig_lin:>+9.2f}")
    print(f"  {'Exponential':>16} {HYY_exp:>+12.4f} {sig_exp:>8.4f} "
          f"{HYY_exp/pred:>8.4f} {(HYY_exp-pred)/sig_exp:>+9.2f}")
    print(f"  {'Structural':>16} {pred:>12} {'-':>8} "
          f"{1.0:>8.4f} {0:>+9.2f}")

    # Falsification under each model
    print(f"\n  Falsification of alternative M values:")
    print(f"  {'M':>4} {'lin Δ':>10} {'lin z':>8} {'exp Δ':>10} {'exp z':>8}")
    print('  ' + '-' * 50)
    for M_alt in [25, 28, 30, 33, 36, 39, 42, 45]:
        lin_z = (HYY_lin - M_alt) / sig_lin
        exp_z = (HYY_exp - M_alt) / sig_exp
        ok_lin = '✓' if abs(lin_z) < 3 else '✗'
        ok_exp = '✓' if abs(exp_z) < 3 else '✗'
        print(f"  {M_alt:>4} {HYY_lin-M_alt:>+10.4f} "
              f"{lin_z:>+7.2f}{ok_lin} {HYY_exp-M_alt:>+10.4f} "
              f"{exp_z:>+7.2f}{ok_exp}")

    # Discussion
    print(f"""
  Interpretation:

  ZNE corrects for noise that scales with fold count (gate-induced
  depolarization).  It does NOT correct for non-foldable error
  sources: state prep, measurement readout, idle decoherence (T1/T2
  during gates and between gates), and crosstalk on simultaneous
  gates.  These contribute an irreducible floor that ZNE leaves
  intact.

  For our K_9 measurement on ibm_fez (88 CZ depth):
    Linear ZNE recovers      {HYY_lin:.2f} / 36 = {HYY_lin/pred:.1%}
    Exponential ZNE recovers {HYY_exp:.2f} / 36 = {HYY_exp/pred:.1%}
    Raw (no mitigation)      19.57 / 36 = 54.4%

  Exponential ZNE recovers an additional {(HYY_exp - 19.57)/19.57:.0%} over
  raw, but still falls {(pred - HYY_exp)/pred:.0%} short of the structural
  prediction.  The remaining gap is the non-foldable-error floor —
  not direct evidence against M(K_9) = 36 since the SAME floor would
  apply to ANY value of M.

  To convert ZNE into a sharp falsification test, we'd need to apply
  ZNE to K_7 as well (so the non-foldable bias cancels in the ratio
  K_9_ZNE / K_7_ZNE).  That would cost another ~$60-80 PAYG.
""")


if __name__ == '__main__':
    main()
