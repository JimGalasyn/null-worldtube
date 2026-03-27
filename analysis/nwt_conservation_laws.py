#!/usr/bin/env python3
"""
NWT Conservation Laws: What quantum number combinations are conserved?

In standard physics:
  - Electric charge Q is conserved in ALL interactions
  - Baryon number B is conserved
  - Lepton number L is conserved
  - Strangeness S conserved in strong, violated in weak

These must be expressible as functions of (p, q, m, n_q).
Search for linear combinations a*p + b*q + c*m + d*n_q that are
conserved across known decays.
"""

import numpy as np
from itertools import product as iproduct

# ── Particle data with standard quantum numbers ─────────────────────
# (name, mass, p, q, m, n_q, Q_charge, B_baryon, L_lepton, S_strangeness)
PARTICLES = {
    'e':    {'p': 2, 'q': 1, 'm': 3,  'nq': 0, 'Q': -1, 'B': 0, 'L': 1,  'S': 0},
    'ē':    {'p': 2, 'q': 1, 'm': 3,  'nq': 0, 'Q': +1, 'B': 0, 'L': -1, 'S': 0},  # positron
    'μ':    {'p': 1, 'q': 8, 'm': 9,  'nq': 0, 'Q': -1, 'B': 0, 'L': 1,  'S': 0},
    'τ':    {'p': 3, 'q': 4, 'm': 17, 'nq': 3, 'Q': -1, 'B': 0, 'L': 1,  'S': 0},  # stealth baryon but lepton number!
    'ν_e':  {'p': 0, 'q': 0, 'm': 0,  'nq': 0, 'Q': 0,  'B': 0, 'L': 1,  'S': 0},  # unknown (p,q,m)
    'ν̄_e':  {'p': 0, 'q': 0, 'm': 0,  'nq': 0, 'Q': 0,  'B': 0, 'L': -1, 'S': 0},
    'ν_μ':  {'p': 0, 'q': 0, 'm': 0,  'nq': 0, 'Q': 0,  'B': 0, 'L': 1,  'S': 0},
    'ν̄_μ':  {'p': 0, 'q': 0, 'm': 0,  'nq': 0, 'Q': 0,  'B': 0, 'L': -1, 'S': 0},
    'ν_τ':  {'p': 0, 'q': 0, 'm': 0,  'nq': 0, 'Q': 0,  'B': 0, 'L': 1,  'S': 0},
    'ν̄_τ':  {'p': 0, 'q': 0, 'm': 0,  'nq': 0, 'Q': 0,  'B': 0, 'L': -1, 'S': 0},
    'π⁺':   {'p': 3, 'q': 5, 'm': 5,  'nq': 2, 'Q': +1, 'B': 0, 'L': 0,  'S': 0},
    'π⁻':   {'p': 3, 'q': 5, 'm': 5,  'nq': 2, 'Q': -1, 'B': 0, 'L': 0,  'S': 0},
    'π⁰':   {'p': 7, 'q': 3, 'm': 18, 'nq': 2, 'Q': 0,  'B': 0, 'L': 0,  'S': 0},
    'K⁺':   {'p': 2, 'q': 5, 'm': 8,  'nq': 2, 'Q': +1, 'B': 0, 'L': 0,  'S': +1},
    'K⁻':   {'p': 2, 'q': 5, 'm': 8,  'nq': 2, 'Q': -1, 'B': 0, 'L': 0,  'S': -1},
    'p':    {'p': 1, 'q': 4, 'm': 5,  'nq': 3, 'Q': +1, 'B': 1, 'L': 0,  'S': 0},
    'p̄':    {'p': 1, 'q': 4, 'm': 5,  'nq': 3, 'Q': -1, 'B': -1,'L': 0,  'S': 0},
    'n':    {'p': 1, 'q': 4, 'm': 5,  'nq': 3, 'Q': 0,  'B': 1, 'L': 0,  'S': 0},
    'Λ':    {'p': 3, 'q': 4, 'm': 12, 'nq': 3, 'Q': 0,  'B': 1, 'L': 0,  'S': -1},
    'Σ⁺':   {'p': 1, 'q': 4, 'm': 6,  'nq': 3, 'Q': +1, 'B': 1, 'L': 0,  'S': -1},
    'Σ⁻':   {'p': 1, 'q': 4, 'm': 6,  'nq': 3, 'Q': -1, 'B': 1, 'L': 0,  'S': -1},
    'Δ⁺⁺':  {'p': 5, 'q': 4, 'm': 15, 'nq': 3, 'Q': +2, 'B': 1, 'L': 0,  'S': 0},
    'Ξ⁰':   {'p': 5, 'q': 4, 'm': 16, 'nq': 3, 'Q': 0,  'B': 1, 'L': 0,  'S': -2},
    'Ξ⁻':   {'p': 5, 'q': 4, 'm': 16, 'nq': 3, 'Q': -1, 'B': 1, 'L': 0,  'S': -2},
    'Ω⁻':   {'p': 7, 'q': 4, 'm': 19, 'nq': 3, 'Q': -1, 'B': 1, 'L': 0,  'S': -3},
    'Σ*⁺':  {'p': 3, 'q': 4, 'm': 14, 'nq': 3, 'Q': +1, 'B': 1, 'L': 0,  'S': -1},
    'D⁰':   {'p': 3, 'q': 7, 'm': 7,  'nq': 2, 'Q': 0,  'B': 0, 'L': 0,  'S': 0},
    'D⁺':   {'p': 2, 'q': 7, 'm': 5,  'nq': 2, 'Q': +1, 'B': 0, 'L': 0,  'S': 0},
    'J/ψ':  {'p': 2, 'q': 7, 'm': 7,  'nq': 2, 'Q': 0,  'B': 0, 'L': 0,  'S': 0},
    'γ':    {'p': 0, 'q': 0, 'm': 0,  'nq': 0, 'Q': 0,  'B': 0, 'L': 0,  'S': 0},
}

# ── Known decays ────────────────────────────────────────────────────
# (parent, [daughters])
DECAYS = [
    # Weak leptonic
    ('n',    ['p', 'e', 'ν̄_e']),
    ('μ',    ['e', 'ν̄_e', 'ν_μ']),
    ('τ',    ['μ', 'ν̄_μ', 'ν_τ']),
    ('π⁺',   ['μ', 'ν_μ']),        # note: μ⁺ but we use μ for simplicity
    ('K⁺',   ['μ', 'ν_μ']),

    # Strong
    ('Σ*⁺',  ['Λ', 'π⁺']),
    ('Δ⁺⁺',  ['p', 'π⁺']),

    # Weak hadronic
    ('Λ',    ['p', 'π⁻']),
    ('Ξ⁻',   ['Λ', 'π⁻']),
    ('Ω⁻',   ['Λ', 'K⁻']),

    # EM
    ('π⁰',   ['γ', 'γ']),
]


def check_conservation():
    """Check which standard quantum numbers are conserved."""
    print("=" * 80)
    print("STANDARD CONSERVATION LAW VERIFICATION")
    print("=" * 80)

    print(f"\n{'Decay':>25s}  {'ΔQ':>4s} {'ΔB':>4s} {'ΔL':>4s} {'ΔS':>4s}  Status")
    print("-" * 65)

    for parent, daughters in DECAYS:
        if parent not in PARTICLES:
            continue

        par = PARTICLES[parent]
        sums = {k: 0 for k in ['Q', 'B', 'L', 'S']}
        for d in daughters:
            if d in PARTICLES:
                for k in sums:
                    sums[k] += PARTICLES[d][k]

        deltas = {k: par[k] - sums[k] for k in ['Q', 'B', 'L', 'S']}
        status = "✓" if all(v == 0 for v in deltas.values()) else "✗ ΔS≠0" if deltas['S'] != 0 and deltas['Q'] == 0 and deltas['B'] == 0 else "✗"

        decay_str = f"{parent} → {' + '.join(daughters)}"
        print(f"{decay_str:>25s}  {deltas['Q']:>+4d} {deltas['B']:>+4d} "
              f"{deltas['L']:>+4d} {deltas['S']:>+4d}  {status}")


def search_linear_combinations():
    """
    Search for linear combinations of (p, q, m, n_q) that equal
    standard conserved quantum numbers Q, B, L.

    Try: Q = a*p + b*q + c*m + d*n_q (mod something?)
    """
    print(f"\n{'=' * 80}")
    print("SEARCH: Linear combinations of (p,q,m,n_q) → standard quantum numbers")
    print("=" * 80)

    # Use particles with known charge to set up the system
    # We need: a*p + b*q + c*m + d*n_q = Q for each particle

    # Collect data matrix
    test_particles = ['e', 'p', 'n', 'π⁺', 'π⁻', 'K⁺', 'Λ', 'Σ⁺', 'Ξ⁻', 'Ω⁻',
                       'μ', 'D⁺', 'J/ψ', 'Δ⁺⁺']

    for target in ['Q', 'B', 'L']:
        print(f"\n  --- Searching for {target} = a·p + b·q + c·m + d·n_q ---")

        A = []
        y = []
        names = []
        for pname in test_particles:
            if pname in PARTICLES:
                par = PARTICLES[pname]
                A.append([par['p'], par['q'], par['m'], par['nq']])
                y.append(par[target])
                names.append(pname)

        A = np.array(A, dtype=float)
        y = np.array(y, dtype=float)

        # Least squares fit
        result, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
        a, b, c, d = result

        print(f"  Best fit: {target} ≈ {a:.4f}·p + {b:.4f}·q + {c:.4f}·m + {d:.4f}·n_q")

        # Check fit quality
        y_pred = A @ result
        for i, pname in enumerate(names):
            err = y[i] - y_pred[i]
            if abs(err) > 0.01:
                print(f"    {pname:>8s}: {target}={y[i]:+.0f}, pred={y_pred[i]:+.3f}, err={err:+.3f}")

        residual = np.sum((y - y_pred)**2)
        print(f"  Total residual: {residual:.6f}")

        if residual > 0.1:
            # Try with mod arithmetic
            print(f"\n  Linear fit failed. Trying modular arithmetic...")
            for mod in [2, 3, 4, 5, 6]:
                for a, b, c, d in iproduct(range(mod), repeat=4):
                    match = True
                    for i, pname in enumerate(names):
                        par = PARTICLES[pname]
                        pred = (a*par['p'] + b*par['q'] + c*par['m'] + d*par['nq']) % mod
                        actual = int(y[i]) % mod
                        if pred != actual:
                            match = False
                            break
                    if match and not (a == 0 and b == 0 and c == 0 and d == 0):
                        print(f"    mod {mod}: {target} ≡ {a}·p + {b}·q + {c}·m + {d}·n_q (mod {mod})")


def search_nonlinear():
    """Try nonlinear combinations including p²+q², gcd, etc."""
    print(f"\n{'=' * 80}")
    print("SEARCH: Nonlinear combinations → quantum numbers")
    print("=" * 80)

    from math import gcd

    test_particles = ['e', 'p', 'n', 'π⁺', 'π⁻', 'K⁺', 'Λ', 'Σ⁺', 'Ξ⁻', 'Ω⁻',
                       'μ', 'D⁺', 'J/ψ', 'Δ⁺⁺', 'τ']

    print(f"\n{'Particle':>8s} {'(p,q)':>6s} {'m':>3s} {'nq':>3s} "
          f"{'Q':>3s} {'B':>3s} {'L':>3s} {'S':>3s}  "
          f"{'p²+q²':>6s} {'gcd':>4s} {'p*q':>4s} {'p-q':>4s} "
          f"{'m mod 2':>7s} {'m mod 3':>7s} {'p mod 2':>7s} {'q mod 2':>7s}")
    print("-" * 110)

    for pname in test_particles:
        par = PARTICLES[pname]
        p, q, m, nq = par['p'], par['q'], par['m'], par['nq']
        g = gcd(p, q) if p > 0 and q > 0 else 0
        print(f"{pname:>8s} ({p},{q})  {m:>3d} {nq:>3d} "
              f"{par['Q']:>+3d} {par['B']:>+3d} {par['L']:>+3d} {par['S']:>+3d}  "
              f"{p**2+q**2:>6d} {g:>4d} {p*q:>4d} {p-q:>+4d} "
              f"{m%2:>7d} {m%3:>7d} {p%2:>7d} {q%2:>7d}")

    # Look for patterns
    print(f"\n  Observations:")
    print(f"  - Baryon number B: all baryons have n_q=3, all mesons n_q=2, leptons n_q=0")
    print(f"    → B = 1 if n_q=3, B = 0 if n_q≤2")
    print(f"    → More precisely: B = floor(n_q/3) for conventional particles")

    print(f"\n  - Lepton number L: leptons have n_q=0 (e, μ) or n_q=3 (τ stealth)")
    print(f"    → L = 1 if {'{'}category = lepton{'}'} — this is the τ puzzle!")
    print(f"    → τ has B=0 but n_q=3: it's a baryon by topology, lepton by number")

    print(f"\n  - Charge Q: this is the hard one. Let's look at charge within families...")

    # Within each knot family, what determines charge?
    print(f"\n  Charge within T(1,4) family:")
    for pname in ['p', 'n', 'Σ⁺', 'Σ⁻']:
        if pname in PARTICLES:
            par = PARTICLES[pname]
            print(f"    {pname:>4s}: m={par['m']}, Q={par['Q']:+d}")

    print(f"\n  Charge within T(5,4) family:")
    for pname in ['Δ⁺⁺', 'Ξ⁰', 'Ξ⁻']:
        if pname in PARTICLES:
            par = PARTICLES[pname]
            print(f"    {pname:>4s}: m={par['m']}, Q={par['Q']:+d}")

    print(f"\n  Note: particles with SAME (p,q,m,n_q) can have DIFFERENT charges!")
    print(f"  → Charge is NOT determined by (p,q,m,n_q) alone")
    print(f"  → Need additional quantum number: isospin projection I_3")
    print(f"  → In NWT, I_3 might be related to the ORIENTATION of the knot")
    print(f"     (left-handed vs right-handed embedding)")


def analyze_strangeness():
    """Strangeness in NWT: is it encoded in the quantum numbers?"""
    print(f"\n{'=' * 80}")
    print("STRANGENESS: Is S encoded in (p,q,m,n_q)?")
    print("=" * 80)

    strange_particles = [
        ('K⁺',  +1, 2, 5, 8,  2),
        ('K⁻',  -1, 2, 5, 8,  2),
        ('Λ',   -1, 3, 4, 12, 3),
        ('Σ⁺',  -1, 1, 4, 6,  3),
        ('Ξ⁻',  -2, 5, 4, 16, 3),
        ('Ω⁻',  -3, 7, 4, 19, 3),
    ]

    non_strange = [
        ('π⁺',   0, 3, 5, 5,  2),
        ('p',     0, 1, 4, 5,  3),
        ('n',     0, 1, 4, 5,  3),
        ('Δ⁺⁺',  0, 5, 4, 15, 3),
    ]

    print(f"\n  Strange particles:")
    print(f"  {'Name':>6s} {'S':>3s} {'(p,q)':>6s} {'m':>3s} {'nq':>3s} {'q_flavor':>8s}")
    for name, S, p, q, m, nq in strange_particles:
        print(f"  {name:>6s} {S:>+3d} ({p},{q})  {m:>3d} {nq:>3d}   q={q}")

    print(f"\n  Non-strange particles:")
    for name, S, p, q, m, nq in non_strange:
        print(f"  {name:>6s} {S:>+3d} ({p},{q})  {m:>3d} {nq:>3d}   q={q}")

    print(f"\n  Pattern: In Paper 6, flavor = poloidal winding q:")
    print(f"    q=3: pion sector (π⁰)")
    print(f"    q=4: baryon sector (p, n, Λ, Σ, Ξ, Ω)")
    print(f"    q=5: light meson sector (π±, K, η, ω, ...)")
    print(f"    q=7: charm sector (D, J/ψ)")
    print(f"    q=9: bottom sector (Υ)")
    print(f"")
    print(f"  But strangeness cuts ACROSS q sectors:")
    print(f"    K± has q=5 (light meson) but S=±1")
    print(f"    Λ, Σ have q=4 (baryon) but S=-1")
    print(f"    Ξ has q=4 (baryon) but S=-2")
    print(f"    Ω has q=4 (baryon) but S=-3")
    print(f"")
    print(f"  Hypothesis: strangeness = number of 'strange' quarks")
    print(f"    In NWT: each strange quark corresponds to a specific")
    print(f"    sub-harmonic mode on the torus. The number of such")
    print(f"    modes = |S|. For baryons (q=4), the p quantum number")
    print(f"    encodes the flavor content:")
    print(f"    p=1: 0 strange quarks (p, n, Σ) → S=0 or -1")
    print(f"    p=3: 1 strange quark (Λ, Σ*) → S=-1")
    print(f"    p=5: 2 strange quarks (Ξ, Δ) → S=-2 or 0")
    print(f"    p=7: 3 strange quarks (Ω) → S=-3")
    print(f"")
    print(f"  For q=4 baryons: |S| correlates with (p-1)/2:")
    for name, S, p, q, m, nq in strange_particles + non_strange:
        if q == 4 and nq == 3:
            s_pred = (p - 1) // 2
            print(f"    {name:>6s}: p={p}, |S|={abs(S)}, (p-1)/2={s_pred}, "
                  f"{'✓' if s_pred == abs(S) else '✗'}")


if __name__ == "__main__":
    check_conservation()
    search_linear_combinations()
    search_nonlinear()
    analyze_strangeness()
