#!/usr/bin/env python3
"""
NWT Jones Polynomial Analysis

Compute Jones polynomial invariants for all NWT particle torus knots T(p,q)
and test whether they correlate with observed masses.

Key formulas:
  - Jones polynomial for T(p,q) at q_param = e^(2πi/N):
    Uses the Rosso-Jones formula via representation theory
  - For torus knots, the Jones polynomial has a closed-form expression

The Jones polynomial for the (p,q) torus knot is:
  J(T(p,q); t) = (1-t^2) / (1-t^(p+1)(1-t^(q+1))) * sum terms

Simpler: use the HOMFLY polynomial specialization.
For torus knot T(m,n), the colored Jones at q = e^(2πi/N):
  J_N(T(m,n); q) = q^((1-N)(mn)) * Σ q^(-kN) (q^(1-N))_k  [for T(2,2t+1)]

We'll use the direct computation via the writhe and Kauffman bracket.

References:
  - Rosso & Jones 1993
  - Hikami & Lovejoy 2016
  - Morton 1995 (Jones polynomial for torus knots)
"""

import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt

# ── NWT Particle Table ──────────────────────────────────────────────
# (name, mass_exp_MeV, p, q, n_q, category)
PARTICLES = [
    ('e',        0.511,    2, 1, 0, 'lepton'),
    ('μ',        105.66,   1, 8, 0, 'lepton'),
    ('π⁰',      135.0,    7, 3, 2, 'meson'),
    ('π±',       139.57,   3, 5, 2, 'meson'),
    ('K±',       493.68,   2, 5, 2, 'meson'),
    ('K⁰',       497.61,   7, 5, 2, 'meson'),
    ('η',        547.86,   6, 5, 2, 'meson'),
    ('ρ⁰',       775.26,   5, 7, 2, 'meson'),
    ('ω',        782.66,   4, 5, 2, 'meson'),
    ('K*',       891.67,   6, 5, 2, 'meson'),
    ('p',        938.27,   1, 4, 3, 'baryon'),
    ('n',        939.57,   1, 4, 3, 'baryon'),
    ('Λ',       1115.7,    3, 4, 3, 'baryon'),
    ('Σ',       1189.4,    1, 4, 3, 'baryon'),
    ('Δ',       1232.0,    5, 4, 3, 'baryon'),
    ('Ξ',       1314.9,    5, 4, 3, 'baryon'),
    ('Σ*',      1385.0,    3, 4, 3, 'baryon'),
    ('Ω⁻',      1672.5,    7, 4, 3, 'baryon'),
    ('τ',       1776.86,   3, 4, 3, 'stealth'),
    ('D⁰',      1864.8,    3, 7, 2, 'meson'),
    ('D±',       1869.7,    2, 7, 2, 'meson'),
    ('Λc',      2286.5,    1, 5, 3, 'baryon'),
    ('Ξc',      2469.4,    1, 4, 3, 'baryon'),
    ('J/ψ',     3096.9,    2, 7, 2, 'meson'),
    ('B±',       5279.3,   10, 7, 2, 'meson'),
    ('Λb',      5619.6,    3, 5, 3, 'baryon'),
    ('Υ',       9460.3,    4, 9, 2, 'meson'),
]


# ── Jones Polynomial for Torus Knots ────────────────────────────────

def jones_torus_knot(p, q, t):
    """
    Jones polynomial for torus knot T(p,q) evaluated at t.

    Using the formula (Morton 1995, Proposition 2):
    For coprime p, q:
      V_{T(p,q)}(t) = (t^((p-1)(q-1)/2)) / (1 - t^2) *
                       sum_{i=0}^{p-1} (t^(q*(2i-p+1)+1) - t^(q*(2i-p+1)-1))
                       * product terms

    Simpler closed form (Labastida-Perez, Jones 1987):
      V_{T(p,q)}(t) = t^((p-1)(q-1)/2) * (1-t^2)^(-1) *
                       [sum_{j} (-1)^j t^(a_j)]

    We use the direct recursive computation for small knots.
    For T(2,n) (the most common family), there's a very simple formula:

      V_{T(2,n)}(t) = (-1)^((n-1)/2) * t^((n-1)/2) *
                       (1 + sum_{k=1}^{(n-1)/2} (-t^(-2))^k * (1-t^(2k+1))/(1-t))

    Actually, let's use the most general explicit formula.
    """
    # Use numpy for complex arithmetic
    t = complex(t)

    if p == 1 or q == 1:
        # T(1,q) or T(p,1) is the unknot
        return complex(1.0)

    if p == 2:
        # T(2,q) torus knots — use explicit formula
        # V_{T(2,q)}(t) for q odd:
        #   = -t^((q+1)/2) * (1 - sum_{k=1}^{(q-1)/2} t^(-2k)(1-t))  / (1-t^2)
        # Easier: use skein relation recursion
        return _jones_T2n(q, t)

    # General case: use the Rosso-Jones-type formula
    # V_{T(p,q)}(t) = t^{(p-1)(q-1)/2} / (1-t^2) *
    #   sum_{i=0}^{p-1} [t^{q(2i-p+1)+1} - t^{q(2i-p+1)-1}]
    prefactor = t**((p-1)*(q-1)/2) / (1 - t**2)
    total = 0
    for i in range(p):
        exp_arg = q * (2*i - p + 1)
        total += t**(exp_arg + 1) - t**(exp_arg - 1)

    return prefactor * total


def _jones_T2n(n, t):
    """
    Jones polynomial for T(2,n) using the explicit formula.

    For T(2,n) with n odd (torus knot):
    V(t) = -t^{(n+1)/2} * [1 - t^(-1) + t^(-2) - ... + t^{-(n-1)}] / (1+t^(-1))

    Simplified:
    V_{T(2,n)}(t) = (-1)^{(n-1)/2} * t^{(n-3)/2} / (1 - t^(-2)) *
                     (1 - t^{-(n+1)})
    """
    t = complex(t)
    if n == 1:
        return complex(1.0)

    # Direct computation: sum of geometric series
    # V_{T(2,n)}(t) = -t^{(n-1)/2} / (1 + t) * sum_{k=0}^{n-1} (-t)^k
    # But this needs care with signs and normalization.

    # Use the Kauffman bracket approach for accuracy:
    # <T(2,n)> involves a recursive structure
    # For small n, just compute directly.

    # Actually, use the known result:
    # V_{T(2,n)}(t) = (t^{(n-1)/2} - t^{(n+1)/2}) / (t - t^{-1})
    #                 * (1 - t^{-n-1}) / (1 - t^{-2})
    # This simplifies to a sum.

    # Safest: direct evaluation via the general formula with p=2
    prefactor = t**((2-1)*(n-1)/2) / (1 - t**2)
    total = 0
    for i in range(2):
        exp_arg = n * (2*i - 2 + 1)
        total += t**(exp_arg + 1) - t**(exp_arg - 1)

    return prefactor * total


def jones_at_root_of_unity(p, q, N):
    """
    Evaluate Jones polynomial at t = e^(2πi/N).

    The N-th root of unity evaluation is physically significant:
    - N=5: connected to electron (p²+q²=5 for (2,1))
    - N=3: connected to phase closure m=3
    - N=4: connected to proton q=4
    """
    t = np.exp(2j * np.pi / N)
    return jones_torus_knot(p, q, t)


# ── Compute for all NWT particles ───────────────────────────────────

def analyze_jones_mass_correlation():
    """Compute Jones polynomial values and test correlation with mass."""

    print("=" * 70)
    print("NWT JONES POLYNOMIAL ANALYSIS")
    print("=" * 70)

    # Evaluate at several roots of unity
    roots = [3, 4, 5, 6, 7, 8]

    results = []
    for name, mass, p, q, n_q, cat in PARTICLES:
        row = {'name': name, 'mass': mass, 'p': p, 'q': q, 'n_q': n_q, 'cat': cat}
        row['pq_factor'] = p**2 + q**2

        for N in roots:
            try:
                J = jones_at_root_of_unity(p, q, N)
                row[f'J_abs_{N}'] = abs(J)
                row[f'J_phase_{N}'] = np.angle(J)
                row[f'J_real_{N}'] = J.real
            except (ZeroDivisionError, OverflowError, ValueError):
                row[f'J_abs_{N}'] = np.nan
                row[f'J_phase_{N}'] = np.nan
                row[f'J_real_{N}'] = np.nan

        results.append(row)

    # Print table
    print(f"\n{'Particle':>8s} {'mass':>8s} {'(p,q)':>6s} {'p²+q²':>5s} "
          f"{'|J₅|':>8s} {'|J₃|':>8s} {'|J₄|':>8s} {'|J₇|':>8s}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:>8s} {r['mass']:>8.1f} ({r['p']},{r['q']})  "
              f"{r['pq_factor']:>5d} "
              f"{r.get('J_abs_5', np.nan):>8.3f} "
              f"{r.get('J_abs_3', np.nan):>8.3f} "
              f"{r.get('J_abs_4', np.nan):>8.3f} "
              f"{r.get('J_abs_7', np.nan):>8.3f}")

    # Correlation analysis
    print(f"\n{'=' * 70}")
    print("CORRELATION: |J_N| vs ln(mass)")
    print(f"{'=' * 70}")

    masses = np.array([r['mass'] for r in results])
    log_masses = np.log(masses)

    for N in roots:
        j_vals = np.array([r.get(f'J_abs_{N}', np.nan) for r in results])
        mask = np.isfinite(j_vals) & (j_vals > 0)
        if np.sum(mask) > 3:
            log_j = np.log(j_vals[mask])
            corr = np.corrcoef(log_masses[mask], log_j)[0, 1]
            print(f"  N={N}: r = {corr:+.4f} ({np.sum(mask)} particles)")

    # Phase analysis
    print(f"\n{'=' * 70}")
    print("PHASE ANALYSIS: arg(J₅) — does phase encode quantum numbers?")
    print(f"{'=' * 70}")

    for r in results:
        phase = r.get('J_phase_5', np.nan)
        if np.isfinite(phase):
            # Express phase as fraction of π
            frac = phase / np.pi
            print(f"  {r['name']:>8s}  ({r['p']},{r['q']})  "
                  f"arg(J₅)/π = {frac:+.4f}  "
                  f"≈ {Fraction(frac).limit_denominator(20)}")

    # Key test: does |J_5| * n_q^q correlate with mass?
    print(f"\n{'=' * 70}")
    print("KEY TEST: |J₅| × n_q^q vs mass")
    print(f"{'=' * 70}")

    for r in results:
        j5 = r.get('J_abs_5', np.nan)
        n_q = r['n_q']
        q = r['q']
        enhance = n_q**q if n_q > 0 else 1
        if np.isfinite(j5) and j5 > 0:
            product = j5 * enhance
            ratio = r['mass'] / product if product > 0 else np.nan
            print(f"  {r['name']:>8s}  |J₅|={j5:>8.3f}  n_q^q={enhance:>6d}  "
                  f"product={product:>10.1f}  mass/prod={ratio:>8.3f}")

    return results


def plot_jones_vs_mass(results):
    """Plot Jones polynomial values against mass."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    cat_colors = {'lepton': '#1f77b4', 'meson': '#ff7f0e',
                  'baryon': '#2ca02c', 'stealth': '#d62728'}

    for idx, N in enumerate([3, 4, 5, 6, 7, 8]):
        ax = axes[idx // 3, idx % 3]
        for r in results:
            j_val = r.get(f'J_abs_{N}', np.nan)
            if np.isfinite(j_val) and j_val > 0:
                ax.scatter(r['mass'], j_val,
                           c=cat_colors.get(r['cat'], 'gray'),
                           s=40, alpha=0.7, edgecolors='black', linewidths=0.3)
                ax.annotate(r['name'], (r['mass'], j_val),
                            fontsize=5, xytext=(2, 2), textcoords='offset points')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Mass (MeV)')
        ax.set_ylabel(f'|J(T(p,q); e^{{2πi/{N}}})|')
        ax.set_title(f'N = {N} (root of unity)')
        ax.grid(True, alpha=0.2)

    plt.suptitle('Jones Polynomial |J| vs Mass for NWT Particle Torus Knots',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis/nwt_jones_vs_mass.png', dpi=200, bbox_inches='tight')
    print("\nSaved: analysis/nwt_jones_vs_mass.png")
    plt.close()


# ── Dehn Surgery Analysis ───────────────────────────────────────────

def analyze_surgery():
    """
    Explore Dehn surgery on torus knots and its connection to neutrinos.

    Key idea: Dehn surgery on T(p,q) with coefficient m gives a Seifert
    fibered space. Different m = different particles. The "surgery data"
    (what changes between old and new m) = neutrino.
    """
    print(f"\n{'=' * 70}")
    print("DEHN SURGERY ANALYSIS — NEUTRINOS AS TOPOLOGY CHANGE")
    print(f"{'=' * 70}")

    # Group particles by (p,q) — same knot, different m
    from collections import defaultdict
    knot_families = defaultdict(list)
    for name, mass, p, q, n_q, cat in PARTICLES:
        knot_families[(p, q)].append((name, mass, n_q, cat))

    print("\nKnot families (same topology, different excitations):")
    print("-" * 50)
    for (p, q), members in sorted(knot_families.items()):
        if len(members) > 1:
            print(f"\n  T({p},{q}):")
            for name, mass, n_q, cat in sorted(members, key=lambda x: x[1]):
                print(f"    {name:>8s}  {mass:>8.1f} MeV  n_q={n_q}  ({cat})")

    # Surgery transitions: which decays correspond to changing m?
    print(f"\n\nSurgery transitions (weak decays = topology change):")
    print("-" * 50)

    # Known weak decays and their NWT interpretation
    decays = [
        ('n → p + e⁻ + ν̄_e',    'T(1,4) m=5 → T(1,4) m=5 + T(2,1) m=3 + surgery'),
        ('μ → e + ν̄_e + ν_μ',   'T(1,8) m=9 → T(2,1) m=3 + surgery + surgery'),
        ('τ → μ + ν̄_μ + ν_τ',   'T(3,4) m=17 → T(1,8) m=9 + surgery + surgery'),
        ('π⁺ → μ⁺ + ν_μ',       'T(3,5) m=5 → T(1,8) m=9 + surgery'),
        ('Λ → p + π⁻',           'T(3,4) m=12 → T(1,4) m=5 + T(3,5) m=5 (strong, no surgery)'),
    ]

    for decay, nwt_interp in decays:
        print(f"\n  {decay}")
        print(f"    NWT: {nwt_interp}")

    # The surgery coefficient changes
    print(f"\n\nSurgery coefficient Δm for weak decays:")
    print("-" * 50)
    print("  n → p:       same (p,q)=(1,4), same m=5, but n_q changes? No...")
    print("  μ → e:       (1,8) m=9 → (2,1) m=3: Δ(p,q) = (1,-7), Δm = -6")
    print("  τ → μ:       (3,4) m=17 → (1,8) m=9: Δ(p,q) = (-2,4), Δm = -8")
    print("  π⁺ → μ⁺:     (3,5) m=5 → (1,8) m=9: Δ(p,q) = (-2,3), Δm = +4")
    print()
    print("  The neutrino carries: Δp, Δq, Δm — the surgery data!")
    print("  Different neutrino flavors = different (Δp, Δq) sectors")


if __name__ == "__main__":
    results = analyze_jones_mass_correlation()
    plot_jones_vs_mass(results)
    analyze_surgery()
