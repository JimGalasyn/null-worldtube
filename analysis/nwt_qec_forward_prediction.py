#!/usr/bin/env python3
"""
Forward-predict ⟨H_YY⟩|K_N⟩ from device-published per-gate fidelities.

The K_9 cross-group test was qualitatively positive but
quantitatively underdetermined: with per-CZ fidelity as a free
parameter, M for K_9 fits anywhere in 30-45.  This script removes
the free parameter by using IBM-published per-gate fidelities for
the specific gates the transpiler actually used.

Method (no fitting):
  1. Re-transpile K_N circuits on each backend (same seed=42).
  2. For each circuit, walk the gate sequence; multiply per-gate
     fidelities to estimate process fidelity F.
  3. Predict ⟨Y_u Y_v⟩_obs ≈ F (since ideal value is 1 on |K_N⟩).
  4. Sum across edges → predicted ⟨H_YY⟩_obs.
  5. Compare to actual measured ⟨H_YY⟩.

If predicted ≈ observed across multiple K_N values without fitting
anything, that's the sharp confirmation of the structural identity
⟨H_YY⟩|K_N⟩ = N(N-1)/2.  If predicted disagrees consistently across
N values, that flags either a model deficiency or a structural
discrepancy.
"""

from __future__ import annotations

import os
import sys
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nwt_qec_heron_KN import (
    build_KN_qiskit_circuit, stabilizer_measurement_circuit,
    YY_measurement_circuit,
)


# =====================================================================
# Per-gate fidelity walk
# =====================================================================

def circuit_process_fidelity(circuit, target) -> Tuple[float, dict]:
    """Walk gates in a transpiled circuit; multiply (1 - err) for each.
    Returns (F_estimate, breakdown).

    We use the multiplicative model F = ∏ (1 - err_g), which is the
    leading-order approximation for stochastic per-gate error.
    Single-qubit Z (rz) gates are treated as error-free (virtual).
    Measurement readout error: separate factor (1 - 2*err_m) for
    each measured qubit (Pauli-±1 outcome attenuation).
    """
    F = 1.0
    cz_count = 0
    sx_count = 0
    rz_count = 0
    x_count = 0
    measure_qubits = []
    for instr in circuit.data:
        op = instr.operation
        qubits = [circuit.find_bit(q).index for q in instr.qubits]
        name = op.name
        if name == 'cz':
            qpair = tuple(qubits)
            props = target['cz'].get(qpair) or target['cz'].get(qpair[::-1])
            if props is not None and props.error is not None:
                F *= max(1.0 - props.error, 0.0)
            cz_count += 1
        elif name == 'sx':
            props = target['sx'].get(tuple(qubits))
            if props is not None and props.error is not None:
                F *= max(1.0 - props.error, 0.0)
            sx_count += 1
        elif name == 'x':
            props = target['x'].get(tuple(qubits))
            if props is not None and props.error is not None:
                F *= max(1.0 - props.error, 0.0)
            x_count += 1
        elif name == 'rz':
            rz_count += 1  # virtual, no error
        elif name == 'measure':
            measure_qubits.extend(qubits)

    # Readout error for measured qubits: each measured ±1 outcome is
    # flipped with prob ε, so observed expectation gets factor (1-2ε).
    # For two-qubit Pauli we get product of two such factors.
    F_meas = 1.0
    for q in measure_qubits:
        props = target['measure'].get((q,))
        if props is not None and props.error is not None:
            F_meas *= max(1.0 - 2.0 * props.error, 0.0)

    breakdown = dict(cz=cz_count, sx=sx_count, x=x_count, rz=rz_count,
                       measure=len(measure_qubits),
                       F_gates=F, F_readout=F_meas)
    return float(F * F_meas), breakdown


# =====================================================================
# Forward-predict <H_YY> for K_N on a backend
# =====================================================================

def forward_predict_HYY(N: int, backend) -> dict:
    from qiskit import transpile
    edges = list(combinations(range(N), 2))
    yy_circuits = [YY_measurement_circuit(N, u, v) for u, v in edges]
    # Same seed as in actual submissions
    yy_t = transpile(yy_circuits, backend=backend, optimization_level=3,
                      seed_transpiler=42)
    target = backend.target

    edge_F = []
    breakdowns = []
    for tc in yy_t:
        F, bk = circuit_process_fidelity(tc, target)
        edge_F.append(F)
        breakdowns.append(bk)

    edge_F = np.array(edge_F)
    HYY_pred = float(edge_F.sum())
    cz_total = sum(bk['cz'] for bk in breakdowns)
    cz_avg = cz_total / len(breakdowns)
    return dict(N=N, n_edges=len(edges),
                HYY_pred=HYY_pred, edge_F=edge_F,
                cz_per_circuit_avg=cz_avg,
                F_gates_avg=float(np.mean([bk['F_gates'] for bk in breakdowns])),
                F_readout_avg=float(np.mean([bk['F_readout']
                                                for bk in breakdowns])))


def forward_predict_stabilizers(N: int, backend) -> dict:
    from qiskit import transpile
    stab_circuits = [stabilizer_measurement_circuit(N, v) for v in range(N)]
    stab_t = transpile(stab_circuits, backend=backend, optimization_level=3,
                        seed_transpiler=42)
    target = backend.target
    Sv_F = []
    for tc in stab_t:
        F, bk = circuit_process_fidelity(tc, target)
        Sv_F.append(F)
    return dict(Sv_F=np.array(Sv_F), Sv_mean=float(np.mean(Sv_F)))


# =====================================================================
# Reporting
# =====================================================================

def section(t: str) -> None:
    print()
    print('=' * 78)
    print(f' {t}')
    print('=' * 78)


def main() -> None:
    section('Forward-prediction: ⟨H_YY⟩|K_N⟩ from device-published gate fidelities')

    print(f"""
  No fitting.  Read per-gate fidelities directly from each backend's
  target.  Walk transpiled circuits and accumulate the multiplicative
  fidelity factor.  Sum over edges.

  Predicted ⟨H_YY⟩|K_N⟩ should equal observed if the framework's
  structural identity ⟨H_YY⟩|K_N⟩ = N(N-1)/2 holds AND the leading-
  order multiplicative-fidelity model is adequate.
""")

    from qiskit_ibm_runtime import QiskitRuntimeService
    token = open(os.path.expanduser('~/.ibm_quantum_api_key')).read().strip()
    service = QiskitRuntimeService(channel='ibm_cloud', token=token)

    # All Heron datasets we have observed values for:
    runs = [
        ('ibm_kingston', 7, 16.4715, 0.0448),  # Run 2
        ('ibm_marrakesh', 7, 15.2815, 0.0493),
        ('ibm_fez',       5,  8.6095, 0.0180),  # K_5 anchor (8K shots)
        ('ibm_fez',       7, 12.2520, 0.0569),
        ('ibm_fez',       9, 19.5730, 0.0781),
    ]

    backends = {}
    for name in set(r[0] for r in runs):
        backends[name] = service.backend(name)
        print(f"  Loaded backend: {name}")

    section('Forward predictions vs observations')
    print(f"\n  {'device':>14} {'N':>3} {'pred':>9} {'obs':>9} "
          f"{'σ_obs':>8} {'(o-p)/σ':>9} {'avg CZ':>7}")
    print('  ' + '-' * 68)

    results = []
    for name, N, obs, sig in runs:
        backend = backends[name]
        pred_data = forward_predict_HYY(N, backend)
        pred = pred_data['HYY_pred']
        z = (obs - pred) / sig
        cz = pred_data['cz_per_circuit_avg']
        print(f"  {name:>14} {N:>3} {pred:>9.4f} {obs:>9.4f} "
              f"{sig:>8.4f} {z:>+9.2f} {cz:>7.1f}")
        results.append(dict(name=name, N=N, pred=pred, obs=obs,
                              sig=sig, z=z, pred_data=pred_data))

    print(f"""
  Interpretation:
    |z| < 3   → forward prediction agrees with observation; the
                structural identity ⟨H_YY⟩|K_N⟩ = N(N-1)/2 is sharply
                confirmed.
    |z| ~ 5-20 → minor model deficiency in the multiplicative-fidelity
                  approximation (correlated errors, calibration drift,
                  etc.) — predictions in the right ballpark.
    |z| >> 20  → either the structural identity disagrees with data,
                  or the device-published fidelities don't reflect what
                  the transpiled circuits actually saw.
""")

    # Cross-group sharpness test for ibm_fez (now with K_5 anchor)
    section('Cross-group sharpness (ibm_fez, K_5 + K_7 + K_9)')
    fez_runs = [r for r in results if r['name'] == 'ibm_fez']
    fez_runs.sort(key=lambda r: r['N'])

    if len(fez_runs) >= 3:
        print(f"\n  Three same-device datapoints anchor the multi-N fit:")
        print(f"  {'N':>3} {'pred':>9} {'obs':>9} {'σ':>8} "
              f"{'obs/pred':>10}")
        print('  ' + '-' * 45)
        bias_factors = []
        for r in fez_runs:
            bias = r['obs'] / r['pred']
            bias_factors.append(bias)
            print(f"  {r['N']:>3} {r['pred']:>9.4f} {r['obs']:>9.4f} "
                  f"{r['sig']:>8.4f} {bias:>10.4f}")

        bias = np.array(bias_factors)
        bias_mean = float(bias.mean())
        bias_std = float(bias.std(ddof=1))
        bias_rel_std = bias_std / bias_mean
        print(f"\n  Multi-N obs/pred bias factor:")
        print(f"    mean:                    {bias_mean:.4f}")
        print(f"    std across N:            {bias_std:.4f}")
        print(f"    relative std:            {bias_rel_std:.2%}")
        print(f"\n  If the structural identity ⟨H_YY⟩|K_N⟩ = N(N-1)/2 holds,")
        print(f"  obs/pred should be approximately constant across N (just")
        print(f"  the universal circuit-vs-gate-fidelity bias).  Variation")
        print(f"  across N flags either a bad noise model or a structural")
        print(f"  discrepancy.")
        if bias_rel_std < 0.05:
            print(f"  → obs/pred CONSTANT across N at <5%: structural")
            print(f"    identity ⟨H_YY⟩|K_N⟩ = N(N-1)/2 confirmed.")
        elif bias_rel_std < 0.15:
            print(f"  → obs/pred consistent across N at ~10%: structural")
            print(f"    identity supported, with residual systematic bias.")
        else:
            print(f"  → obs/pred VARIES significantly across N: structural")
            print(f"    identity challenged OR noise model inadequate.")

        # Pairwise ratios
        print(f"\n  Pairwise structural ratio tests:")
        print(f"  {'pair':>10} {'obs ratio':>12} {'pred ratio':>12} "
              f"{'noiseless':>12} {'pred-obs/obs':>14}")
        print('  ' + '-' * 65)
        for i in range(len(fez_runs)):
            for j in range(i + 1, len(fez_runs)):
                r1, r2 = fez_runs[i], fez_runs[j]
                obs_ratio = r2['obs'] / r1['obs']
                pred_ratio = r2['pred'] / r1['pred']
                struct_ratio = (r2['N'] * (r2['N'] - 1)) / (r1['N'] * (r1['N'] - 1))
                deviation = (pred_ratio - obs_ratio) / obs_ratio
                print(f"  K_{r2['N']}/K_{r1['N']}  {obs_ratio:>12.4f} "
                      f"{pred_ratio:>12.4f} {struct_ratio:>12.4f} "
                      f"{deviation:>+13.2%}")

        # Falsification test against alternative M(K_9)
        print(f"\n  Falsification of alternative M(K_9) values:")
        K7_data = next(r for r in fez_runs if r['N'] == 7)
        K9_data = next(r for r in fez_runs if r['N'] == 9)
        for M_alt in [25, 30, 36, 42, 50]:
            # Forward-predict K_9 assuming structural M_alt instead of 36
            scaled_pred = K9_data['pred'] * (M_alt / 36)
            pred_ratio_alt = scaled_pred / K7_data['pred']
            obs_ratio = K9_data['obs'] / K7_data['obs']
            deviation = (pred_ratio_alt - obs_ratio) / obs_ratio
            ok = "  ✓" if abs(deviation) < 0.05 else "  ✗"
            print(f"    M(K_9)={M_alt:>3}  pred ratio "
                  f"{pred_ratio_alt:.4f}  vs obs {obs_ratio:.4f}  "
                  f"({deviation:+.2%}){ok}")


if __name__ == '__main__':
    main()
