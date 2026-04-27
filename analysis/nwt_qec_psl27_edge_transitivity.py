#!/usr/bin/env python3
"""
Re-analysis of existing Exp 2 data (no new shots): PSL(2,7) edge-
transitivity test.

Prediction: PSL(2,7) acts transitively on K_7's edge set, so on
|K_7⟩ every ⟨Y_u Y_v⟩ = +1 EXACTLY.  Under depolarizing noise the
21 edge expectations should remain statistically indistinguishable
up to gate-fidelity uniformity.

Hardware reality: heavy-hex topology breaks K_7 symmetry — each edge
gets a different routing depth (some pairs need more SWAPs than
others), so we expect SOME edge-by-edge dispersion correlated with
routing distance.

Two sub-tests:
  (a) Per-dataset: is the spread across the 21 edges consistent with
      shot-noise-only?  (χ² "all edges equal" test.)
  (b) Cross-device: do the same edges appear noisy on different
      devices?  (Pearson correlation between edge-orderings of
      kingston Run 2 and marrakesh data.)

Reads from analysis/heron_results/2026-04-26_ibm_*_run*.txt files
that the existing experiments wrote out.  No new Heron shots needed.
"""

from __future__ import annotations

import os
import re
import sys
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# Parser
# =====================================================================

def parse_yy_block(path: str) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """Parse a Heron result text file and return {(u,v): (mean, sigma)}
    for the 21 K_7 edges from the EXPERIMENT 2 / pair section.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    text = open(path).read()
    # Lines like:  (0,1)             +0.8695     0.0078
    pat = re.compile(r'^\s*\((\d+),(\d+)\)\s+([+\-]?\d+\.\d+)\s+(\d+\.\d+)\s*$',
                      re.MULTILINE)
    out = {}
    for m in pat.finditer(text):
        u, v = int(m.group(1)), int(m.group(2))
        # Only K_7 edge-pair format (u<v, u in 0..6, v in 0..6)
        if u < v <= 6:
            mean = float(m.group(3))
            sigma = float(m.group(4))
            out[(u, v)] = (mean, sigma)
    return out


def parse_stabilizer_block(path: str) -> Dict[int, Tuple[float, float]]:
    """Parse the EXP 1 stabilizer ⟨S_v⟩ block."""
    text = open(path).read()
    pat = re.compile(r'^\s+(\d)\s+([+\-]?\d+\.\d+)\s+(\d+\.\d+)\s*$',
                      re.MULTILINE)
    out = {}
    for m in pat.finditer(text):
        v = int(m.group(1))
        if 0 <= v <= 6:
            out[v] = (float(m.group(2)), float(m.group(3)))
    return out


# =====================================================================
# PSL(2,7) edge-transitivity tests
# =====================================================================

EDGES = list(combinations(range(7), 2))
assert len(EDGES) == 21


def edge_chisq_all_equal(data: Dict[Tuple[int, int], Tuple[float, float]],
                          mean_estimator: str = 'weighted') \
        -> Tuple[float, float, float, float]:
    """χ² test of the null hypothesis '⟨Y_u Y_v⟩ is the same on every edge'.

    Under depolarizing noise + perfect PSL(2,7) symmetry, all 21 edges
    have a common true mean μ and per-edge variance σ_e².  Returns
    (μ_hat, χ², dof, p_value_approx_via_z).
    """
    means = np.array([data[e][0] for e in EDGES])
    sigmas = np.array([data[e][1] for e in EDGES])
    weights = 1.0 / sigmas ** 2
    if mean_estimator == 'weighted':
        mu_hat = float(np.sum(weights * means) / np.sum(weights))
    else:
        mu_hat = float(means.mean())
    chisq = float(np.sum(((means - mu_hat) / sigmas) ** 2))
    dof = 20  # 21 edges, 1 free parameter (mu_hat)
    chi_red = chisq / dof
    # Approx Gaussian z for excess: chisq ~ N(dof, sqrt(2*dof)) under null
    z = (chisq - dof) / np.sqrt(2 * dof)
    return mu_hat, chisq, dof, z


def cross_device_correlation(d1: Dict[Tuple[int, int], Tuple[float, float]],
                               d2: Dict[Tuple[int, int], Tuple[float, float]]
                               ) -> Tuple[float, float]:
    """Pearson correlation between edge-by-edge means.  If the SAME
    edges are noisy on both devices, correlation ≈ 1.  If routing-
    induced noise is fully device-specific, correlation ≈ 0."""
    means1 = np.array([d1[e][0] for e in EDGES])
    means2 = np.array([d2[e][0] for e in EDGES])
    r = float(np.corrcoef(means1, means2)[0, 1])
    # Fisher z transform for SE
    n = 21
    se = 1.0 / np.sqrt(n - 3)
    return r, se


# =====================================================================
# Reporting
# =====================================================================

def section(title: str) -> None:
    print()
    print('=' * 78)
    print(f' {title}')
    print('=' * 78)


def report_dataset(label: str,
                     data: Dict[Tuple[int, int], Tuple[float, float]]) -> dict:
    section(f'Dataset: {label}')
    means = np.array([data[e][0] for e in EDGES])
    sigmas = np.array([data[e][1] for e in EDGES])

    mu_hat, chisq, dof, z = edge_chisq_all_equal(data)

    print(f"\n  21 edge ⟨Y_u Y_v⟩ values:")
    print(f"  {'edge':>8} {'⟨YY⟩':>10} {'σ':>10} {'(YY-μ)/σ':>10}")
    print('  ' + '-' * 44)
    sorted_by_mean = sorted(EDGES, key=lambda e: data[e][0])
    for e in sorted_by_mean:
        m, s = data[e]
        z_e = (m - mu_hat) / s
        print(f"  ({e[0]},{e[1]}){'':>4} {m:>+10.4f} {s:>10.4f} {z_e:>+10.2f}")

    print(f"\n  Summary statistics:")
    print(f"    n edges:                 21")
    print(f"    mean (unweighted):       {means.mean():.4f}")
    print(f"    std across edges:        {means.std(ddof=1):.4f}")
    print(f"    coefficient of variation: {means.std(ddof=1)/abs(means.mean()):.4f}")
    print(f"    avg per-edge σ:          {sigmas.mean():.4f}")
    print(f"    range:                   "
          f"[{means.min():.4f}, {means.max():.4f}]  "
          f"= {means.max() - means.min():.4f}")
    print(f"    spread / per-edge σ:     "
          f"{(means.max() - means.min()) / sigmas.mean():.2f}")

    print(f"\n  PSL(2,7) edge-equality test (null: all 21 edges share a "
          f"common true mean):")
    print(f"    μ̂ (weighted):            {mu_hat:.4f}")
    print(f"    χ²:                      {chisq:.2f}  (dof = {dof})")
    print(f"    χ²_reduced:              {chisq/dof:.2f}  "
          f"(≈ 1 means edges statistically equal)")
    print(f"    z-score for excess:      {z:+.2f}σ")
    if z > 5:
        print(f"    → STRONG edge-asymmetry: PSL(2,7) symmetry is "
              f"observably broken at the hardware level.")
    elif z > 2:
        print(f"    → Suggestive edge-asymmetry: hardware breaks PSL(2,7) "
              f"by routing/calibration variance.")
    else:
        print(f"    → No significant edge-asymmetry: data is consistent "
              f"with PSL(2,7) edge-equivalence at hardware level.")

    return dict(mu_hat=mu_hat, chisq=chisq, dof=dof, z=z,
                means=means, sigmas=sigmas)


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    section('PSL(2,7) edge-transitivity re-analysis (no new Heron shots)')

    print(f"""
  Re-analysis of existing Exp 2 ⟨Y_u Y_v⟩ data on |K_7⟩.

  Prediction: PSL(2,7) acts edge-transitively on K_7, so all 21 edge
  expectation values should be the same.  Hardware noise breaks this
  symmetry to the extent that gate errors are routing-dependent.

  Test: under the null '⟨Y_u Y_v⟩ is the same on every edge', χ² =
  Σ ((⟨YY⟩_e - μ̂) / σ_e)² should follow χ²(20).
""")

    files = [
        ('ibm_kingston Run 2 (140K shots)',
         'analysis/heron_results/2026-04-26_ibm_kingston_run2.txt'),
        ('ibm_marrakesh   (140K shots)',
         'analysis/heron_results/2026-04-26_ibm_marrakesh_run1.txt'),
        ('ibm_kingston Run 1 (32K shots)',
         'analysis/heron_results/2026-04-26_ibm_kingston_run1.txt'),
    ]

    datasets = {}
    for label, path in files:
        full = os.path.join(REPO, path)
        try:
            d = parse_yy_block(full)
            if len(d) != 21:
                print(f"WARNING: {label} returned {len(d)} edges (need 21)")
                continue
            datasets[label] = d
        except Exception as e:
            print(f"Could not parse {label}: {e}")

    summaries = {}
    for label, data in datasets.items():
        summaries[label] = report_dataset(label, data)

    # Cross-device test
    if 'ibm_kingston Run 2 (140K shots)' in datasets and \
       'ibm_marrakesh   (140K shots)' in datasets:
        section('Cross-device edge-by-edge correlation')
        d_k = datasets['ibm_kingston Run 2 (140K shots)']
        d_m = datasets['ibm_marrakesh   (140K shots)']
        r, se = cross_device_correlation(d_k, d_m)
        print(f"""
  Pearson correlation between ⟨Y_u Y_v⟩ on kingston vs marrakesh
  across the 21 K_7 edges:

    r = {r:+.4f}  (SE ≈ {se:.4f})

  Interpretation:
    r ≈ +1  → SAME edges are noisy on both devices: routing-dependent
              and device-independent, fundamentally heavy-hex
              topology speaking.
    r ≈ 0   → edge-by-edge noise is device-specific (calibration,
              local two-qubit fidelities).
    r < 0   → anti-correlation, very unusual.
""")
        if r > 0.6:
            print(f"  → STRONG cross-device agreement: edge-noise pattern "
                  f"is heavy-hex-routing-driven, not calibration noise.")
        elif r > 0.3:
            print(f"  → MODERATE agreement: routing structure visible "
                  f"alongside per-device calibration variance.")
        else:
            print(f"  → WEAK agreement: edge-noise is mostly device-"
                  f"specific calibration variance.")

        # Show the ranked list of "noisiest" edges (lowest mean) on
        # each device side-by-side
        rank_k = sorted(EDGES, key=lambda e: d_k[e][0])
        rank_m = sorted(EDGES, key=lambda e: d_m[e][0])
        print(f"\n  Five 'noisiest' edges on each device (lowest ⟨YY⟩):")
        print(f"  {'rank':>5}  {'kingston':>14}  {'marrakesh':>14}")
        print('  ' + '-' * 40)
        for i in range(5):
            ek, em = rank_k[i], rank_m[i]
            print(f"  {i+1:>5}  ({ek[0]},{ek[1]}) {d_k[ek][0]:+.4f}  "
                  f"({em[0]},{em[1]}) {d_m[em][0]:+.4f}")

    # Bottom-line
    section('Bottom line')
    print("""
  PSL(2,7) symmetry is an exact statement about |K_7⟩ — the
  underlying state really has all 21 edges equivalent.  Hardware
  inevitably breaks this by routing-induced gate errors.  This
  re-analysis quantifies HOW MUCH hardware breaks the symmetry,
  not WHETHER (we know it does).

  The χ²_reduced number across edges is the precise hardware-
  signature of how much PSL(2,7) edge-transitivity is preserved
  on each Heron device.  Cross-device correlation tells us whether
  the residual asymmetry is fundamental (heavy-hex topology) or
  device-specific (calibration).
""")


if __name__ == '__main__':
    main()
