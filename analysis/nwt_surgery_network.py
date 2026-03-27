#!/usr/bin/env python3
"""
NWT Surgery Network: Mapping Particle Transitions as Dehn Surgery

Every particle decay/creation event is a topology change on the
torus knot. The "surgery data" — the change in (p,q,m,n_q) — is
carried by the decay products.

Key physics:
  - Strong decays: same (p,q) family, Δn_q possible, no neutrino
  - Weak decays: different (p,q), requires neutrino to carry Δ(p,q,m)
  - Electromagnetic: same (p,q,m,n_q), photon carries energy only

The surgery selection rules should determine which decays are allowed
and predict the branching ratios.
"""

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Particle table ──────────────────────────────────────────────────
# (name, mass, p, q, m, n_q, category, dominant_decay)
PARTICLES = [
    ('e',       0.511,   2, 1, 3,  0, 'lepton',  'stable'),
    ('μ',      105.66,   1, 8, 9,  0, 'lepton',  'e + ν̄_e + ν_μ'),
    ('π⁰',    135.0,    7, 3, 18, 2, 'meson',   'γγ'),
    ('π±',     139.57,   3, 5, 5,  2, 'meson',   'μ + ν_μ'),
    ('K±',     493.68,   2, 5, 8,  2, 'meson',   'μ + ν_μ'),
    ('K⁰',    497.61,   7, 5, 15, 2, 'meson',   'π⁺π⁻ / πlν'),
    ('η',      547.86,   6, 5, 15, 2, 'meson',   'γγ / 3π'),
    ('ρ⁰',    775.26,   5, 7, 7,  2, 'meson',   'π⁺π⁻'),
    ('ω',      782.66,   4, 5, 17, 2, 'meson',   'π⁺π⁻π⁰'),
    ('K*',     891.67,   6, 5, 21, 2, 'meson',   'Kπ'),
    ('p',      938.27,   1, 4, 5,  3, 'baryon',  'stable'),
    ('n',      939.57,   1, 4, 5,  3, 'baryon',  'p + e⁻ + ν̄_e'),
    ('Λ',     1115.7,    3, 4, 12, 3, 'baryon',  'p + π⁻'),
    ('Σ',     1189.4,    1, 4, 6,  3, 'baryon',  'Nπ'),
    ('Δ',     1232.0,    5, 4, 15, 3, 'baryon',  'Nπ'),
    ('Ξ',     1314.9,    5, 4, 16, 3, 'baryon',  'Λπ'),
    ('Σ*',    1385.0,    3, 4, 14, 3, 'baryon',  'Σπ / Λπ'),
    ('Ω⁻',   1672.5,    7, 4, 19, 3, 'baryon',  'ΛK⁻ / Ξπ'),
    ('τ',     1776.86,   3, 4, 17, 3, 'stealth', 'μν̄_μν_τ / eν̄_eν_τ'),
    ('D⁰',    1864.8,    3, 7, 7,  2, 'meson',   'Kπ / Kππ'),
    ('D±',     1869.7,    2, 7, 5,  2, 'meson',   'K⁰π±'),
    ('Λc',    2286.5,    1, 5, 3,  3, 'baryon',  'pK⁻π⁺'),
    ('Ξc',    2469.4,    1, 4, 10, 3, 'baryon',  'Ξπ'),
    ('J/ψ',   3096.9,    2, 7, 7,  2, 'meson',   'hadrons / l⁺l⁻'),
    ('B±',     5279.3,   10, 7, 25, 2, 'meson',   'D̄⁰π± / J/ψK±'),
    ('Λb',    5619.6,    3, 5, 14, 3, 'baryon',  'Λcπ⁻ / J/ψΛ'),
    ('Υ',     9460.3,    4, 9, 8,  2, 'meson',   'l⁺l⁻ / ggg'),
]


def classify_transition(p1, q1, m1, nq1, p2, q2, m2, nq2):
    """Classify a transition by what changes."""
    dp = p2 - p1
    dq = q2 - q1
    dm = m2 - m1
    dnq = nq2 - nq1

    if dp == 0 and dq == 0 and dnq == 0:
        if dm == 0:
            return 'identical'
        else:
            return f'excitation (Δm={dm:+d})'
    elif dp == 0 and dq == 0 and dnq != 0:
        return f'n_q change (Δn_q={dnq:+d}, Δm={dm:+d})'
    else:
        return f'topology change (Δp={dp:+d}, Δq={dq:+d}, Δm={dm:+d}, Δn_q={dnq:+d})'


def analyze_known_decays():
    """Map known particle decays to surgery operations."""
    print("=" * 80)
    print("KNOWN PARTICLE DECAYS AS SURGERY OPERATIONS")
    print("=" * 80)

    # Build lookup
    lookup = {}
    for name, mass, p, q, m, nq, cat, decay in PARTICLES:
        lookup[name] = (p, q, m, nq, mass)

    # Known decays: (parent, [daughters], type)
    decays = [
        # Weak decays (topology change — neutrinos involved)
        ('n',   ['p', 'e', 'ν̄_e'],          'weak'),
        ('μ',   ['e', 'ν̄_e', 'ν_μ'],        'weak'),
        ('τ',   ['μ', 'ν̄_μ', 'ν_τ'],        'weak'),
        ('π±',  ['μ', 'ν_μ'],                 'weak'),
        ('K±',  ['μ', 'ν_μ'],                 'weak'),
        ('D±',  ['K⁰', 'π±'],                 'weak'),  # c → s transition

        # Strong decays (same topology family — no neutrinos)
        ('Δ',   ['p', 'π±'],                  'strong'),
        ('Σ',   ['p', 'π⁰'],                  'strong'),  # approx
        ('Σ*',  ['Λ', 'π±'],                  'strong'),
        ('Ξ',   ['Λ', 'π±'],                  'weak'),   # actually weak (s→u)
        ('Ω⁻',  ['Λ', 'K±'],                  'weak'),
        ('ρ⁰',  ['π±', 'π±'],                 'strong'),  # simplified
        ('K*',  ['K±', 'π⁰'],                 'strong'),

        # Electromagnetic (photon emission — same topology)
        ('π⁰',  ['γ', 'γ'],                   'EM'),
        ('η',   ['γ', 'γ'],                   'EM'),
    ]

    print(f"\n{'Decay':>40s}  {'Type':>6s}  Surgery Data")
    print("-" * 90)

    for parent, daughters, dtype in decays:
        if parent not in lookup:
            continue

        p1, q1, m1, nq1, mass1 = lookup[parent]

        # Find daughter with known NWT assignment
        daughter_info = []
        neutrino_count = 0
        total_surgery = {'dp': 0, 'dq': 0, 'dm': 0, 'dnq': 0}

        for d in daughters:
            if d.startswith('ν') or d.startswith('ν̄'):
                neutrino_count += 1
                daughter_info.append(f'{d}')
            elif d == 'γ':
                daughter_info.append('γ')
            elif d in lookup:
                p2, q2, m2, nq2, mass2 = lookup[d]
                dp = p2 - p1
                dq = q2 - q1
                dm = m2 - m1
                dnq = nq2 - nq1
                daughter_info.append(f'{d}({p2},{q2})m={m2}')
                total_surgery['dp'] += dp
                total_surgery['dq'] += dq
                total_surgery['dm'] += dm
                total_surgery['dnq'] += dnq

        decay_str = f"{parent}({p1},{q1})m={m1} → {' + '.join(daughter_info)}"

        if dtype == 'weak':
            surgery_str = (f"Δp={total_surgery['dp']:+d} Δq={total_surgery['dq']:+d} "
                          f"Δm={total_surgery['dm']:+d} Δn_q={total_surgery['dnq']:+d} "
                          f"+ {neutrino_count}ν")
        elif dtype == 'strong':
            surgery_str = (f"Δp={total_surgery['dp']:+d} Δq={total_surgery['dq']:+d} "
                          f"Δm={total_surgery['dm']:+d} Δn_q={total_surgery['dnq']:+d}")
        else:
            surgery_str = "annihilation → γ"

        print(f"{decay_str:>50s}  {dtype:>6s}  {surgery_str}")

    return decays


def analyze_surgery_conservation():
    """
    What is conserved in surgery?

    Hypothesis: weak decays conserve some combination of (p,q,m,n_q) mod N.
    Strong decays conserve (p,q) exactly.
    """
    print(f"\n{'=' * 80}")
    print("CONSERVATION LAWS IN SURGERY")
    print("=" * 80)

    lookup = {}
    for name, mass, p, q, m, nq, cat, decay in PARTICLES:
        lookup[name] = {'p': p, 'q': q, 'm': m, 'nq': nq, 'mass': mass}

    # For each known decay, compute the total quantum numbers on each side
    # and see what's conserved
    test_decays = [
        # (parent, [daughters_with_NWT_assignment])
        ('n',  ['p', 'e']),      # + ν̄_e (weak)
        ('μ',  ['e']),            # + ν̄_e + ν_μ (weak)
        ('π±', ['μ']),            # + ν_μ (weak)
        ('K±', ['μ']),            # + ν_μ (weak)
        ('Δ',  ['p', 'π±']),     # strong
        ('Σ*', ['Λ', 'π±']),     # strong
        ('Ξ',  ['Λ', 'π±']),     # weak (strangeness change)
        ('ρ⁰', ['π±', 'π±']),    # strong (both pions)
        ('Λ',  ['p', 'π±']),     # weak
        ('Ω⁻', ['Λ', 'K±']),    # weak
        ('τ',  ['μ']),            # + ν̄_μ + ν_τ (weak)
    ]

    print(f"\n{'Decay':>20s}  {'Σp':>4s} {'Σq':>4s} {'Σm':>4s} {'Σnq':>4s}  "
          f"{'Σp':>4s} {'Σq':>4s} {'Σm':>4s} {'Σnq':>4s}  "
          f"{'Δp':>4s} {'Δq':>4s} {'Δm':>4s} {'Δnq':>4s}  Type")
    print(f"{'':>20s}  {'─parent─':>17s}  {'─daughters─':>17s}  {'─deficit─':>17s}")
    print("-" * 90)

    for parent, daughters in test_decays:
        if parent not in lookup:
            continue
        par = lookup[parent]

        # Sum daughter quantum numbers
        sum_d = {'p': 0, 'q': 0, 'm': 0, 'nq': 0}
        for d in daughters:
            if d in lookup:
                for key in sum_d:
                    sum_d[key] += lookup[d][key]

        deficit = {key: par[key] - sum_d[key] for key in ['p', 'q', 'm', 'nq']}

        # Determine type
        if deficit['p'] == 0 and deficit['q'] == 0:
            dtype = 'strong/EM'
        else:
            dtype = 'WEAK (ν carries deficit)'

        print(f"{parent+'→'+'+'.join(daughters):>20s}  "
              f"{par['p']:>4d} {par['q']:>4d} {par['m']:>4d} {par['nq']:>4d}  "
              f"{sum_d['p']:>4d} {sum_d['q']:>4d} {sum_d['m']:>4d} {sum_d['nq']:>4d}  "
              f"{deficit['p']:>+4d} {deficit['q']:>+4d} {deficit['m']:>+4d} {deficit['nq']:>+4d}  "
              f"{dtype}")

    print(f"\n  The DEFICIT in (p, q, m, n_q) = what the neutrino(s) carry away!")
    print(f"  Strong decays: deficit in (p,q) should be zero (topology preserved)")


def analyze_excitation_towers():
    """
    Within each knot family, map the excitation spectrum.
    Mass spacing = surgery coefficient × string tension.
    """
    print(f"\n{'=' * 80}")
    print("EXCITATION TOWERS: Mass vs Phase Closure Integer m")
    print("=" * 80)

    from collections import defaultdict
    from scipy.optimize import brentq
    from scipy.integrate import quad

    KAPPA = np.pi**2
    ME = 0.511

    families = defaultdict(list)
    for name, mass, p, q, m_int, nq, cat, decay in PARTICLES:
        families[(p, q, nq)].append((name, mass, m_int, cat))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    plot_idx = 0

    for (p, q, nq), members in sorted(families.items()):
        if len(members) < 2:
            continue

        members_sorted = sorted(members, key=lambda x: x[2])
        print(f"\n  T({p},{q}) n_q={nq}  [confinement: {nq}^{q} = {nq**q if nq > 0 else 1}]")
        print(f"  {'Name':>8s} {'mass':>8s} {'m':>4s} {'Δm':>4s} {'Δmass':>8s} {'Δmass/Δm':>10s}")
        print(f"  {'-'*50}")

        m_vals = []
        mass_vals = []
        for i, (name, mass, m_int, cat) in enumerate(members_sorted):
            dm = m_int - members_sorted[i-1][2] if i > 0 else 0
            d_mass = mass - members_sorted[i-1][1] if i > 0 else 0
            ratio = d_mass / dm if dm > 0 else 0
            print(f"  {name:>8s} {mass:>8.1f} {m_int:>4d} {dm:>+4d} {d_mass:>+8.1f} {ratio:>10.1f}")
            m_vals.append(m_int)
            mass_vals.append(mass)

        # Linear fit: mass ≈ a × m + b
        if len(m_vals) >= 2:
            coeffs = np.polyfit(m_vals, mass_vals, 1)
            print(f"  Linear fit: mass ≈ {coeffs[0]:.1f} × m + {coeffs[1]:.1f}")
            print(f"  String tension: {coeffs[0]:.1f} MeV per unit m")

            # Plot
            if plot_idx < 6:
                ax = axes[plot_idx // 3, plot_idx % 3]
                ax.scatter(m_vals, mass_vals, s=60, zorder=5,
                          edgecolors='black', linewidths=0.5)
                for name, mass, m_int, cat in members_sorted:
                    ax.annotate(name, (m_int, mass), fontsize=8,
                               xytext=(4, 4), textcoords='offset points')
                m_fit = np.linspace(min(m_vals)-1, max(m_vals)+1, 50)
                ax.plot(m_fit, np.polyval(coeffs, m_fit), 'r--', lw=1, alpha=0.5)
                ax.set_xlabel('Phase closure integer m')
                ax.set_ylabel('Mass (MeV)')
                ax.set_title(f'T({p},{q}) n_q={nq}')
                ax.grid(True, alpha=0.2)
                plot_idx += 1

    plt.suptitle('Excitation Towers: Mass vs Surgery Coefficient m',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis/nwt_surgery_towers.png', dpi=200, bbox_inches='tight')
    print(f"\nSaved: analysis/nwt_surgery_towers.png")
    plt.close()


if __name__ == "__main__":
    analyze_known_decays()
    analyze_surgery_conservation()
    analyze_excitation_towers()
