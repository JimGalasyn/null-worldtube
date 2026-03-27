#!/usr/bin/env python3
"""
Publication-quality figures for Paper 7:
"Charge, Leptons, and the Genus of Torus Knots"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'figure.dpi': 300,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})

# ── Particle data ───────────────────────────────────────────────────
# (name, p, q, n_q, category, mass_MeV)
PARTICLES = [
    ('e',       2, 1, 0, 'lepton',  0.511),
    ('μ',       1, 8, 0, 'lepton',  105.66),
    ('τ',       3, 4, 3, 'stealth', 1776.86),
    ('π⁰',     7, 3, 2, 'meson',   135.0),
    ('π±',      3, 5, 2, 'meson',   139.57),
    ('K±',      2, 5, 2, 'meson',   493.68),
    ('K⁰',     7, 5, 2, 'meson',   497.61),
    ('η',       6, 5, 2, 'meson',   547.86),
    ('ρ',       5, 7, 2, 'meson',   775.26),
    ('ω',       4, 5, 2, 'meson',   782.66),
    ('K*',      6, 5, 2, 'meson',   891.67),
    ('D±',      2, 7, 2, 'meson',   1869.7),
    ('D⁰',     3, 7, 2, 'meson',   1864.8),
    ('J/ψ',    2, 7, 2, 'meson',   3096.9),
    ('B±',     10, 7, 2, 'meson',   5279.3),
    ('Υ',       4, 9, 2, 'meson',   9460.3),
    ('p, n',    1, 4, 3, 'baryon',  938.9),
    ('Σ',       1, 4, 3, 'baryon',  1189.4),
    ('Λ',       3, 4, 3, 'baryon',  1115.7),
    ('Δ',       5, 4, 3, 'baryon',  1232.0),
    ('Ξ',       5, 4, 3, 'baryon',  1314.9),
    ('Σ*',      3, 4, 3, 'baryon',  1385.0),
    ('Ω⁻',     7, 4, 3, 'baryon',  1672.5),
    ('Λc',      1, 5, 3, 'baryon',  2286.5),
    ('Ξc',      1, 4, 3, 'baryon',  2469.4),
    ('Λb',      3, 5, 3, 'baryon',  5619.6),
]

CAT_COLORS = {
    'lepton':  '#1f77b4',
    'meson':   '#ff7f0e',
    'baryon':  '#2ca02c',
    'stealth': '#d62728',
}
CAT_MARKERS = {
    'lepton':  'o',
    'meson':   's',
    'baryon':  '^',
    'stealth': 'D',
}

# Per-particle label offsets: name -> (dx, dy) in offset points
LABEL_OFFSETS = {
    'τ':    (8, -10),
    'Λ':    (6, -10),
    'Σ*':   (6, -10),
    'p, n': (-35, 4),
}
DEFAULT_OFFSET = (6, 4)


def figure1_genus():
    """Figure 1: Genus vs constituent count — the lepton criterion."""
    from collections import defaultdict

    fig, ax = plt.subplots(figsize=(9, 6))

    # Lepton region box
    rect = FancyBboxPatch((-0.8, -0.4), 2.3, 0.8,
        boxstyle='round,pad=0.1', lw=1.2, edgecolor='#1f77b4',
        facecolor='#1f77b4', alpha=0.08, ls='--')
    ax.add_patch(rect)
    ax.text(2.0, 0.0, 'LEPTONS', fontsize=11, color='#1f77b4',
            fontweight='bold', ha='left', va='center', alpha=0.6)
    ax.text(2.0, -0.25, '($g = 0, n_q = 0$)', fontsize=8, color='#1f77b4',
            ha='left', va='center', alpha=0.6)

    # Group particles by (genus, n_q) and category
    groups = defaultdict(lambda: defaultdict(list))
    for name, p, q, nq, cat, mass in PARTICLES:
        g = (p - 1) * (q - 1) // 2
        groups[(g, nq)][cat].append(name)

    # Manual label offsets for specific groups: (g, nq, cat) -> (dx, dy, ha)
    GROUP_OFFSETS = {
        (0, 0, 'lepton'):   (6, 4, 'left'),      # e, μ
        (0, 3, 'baryon'):   None,                   # p, n, Σ / Λc, Ξc — split manually
        (3, 3, 'stealth'):  (6, -12, 'left'),     # τ
        (3, 3, 'baryon'):   (0, 8, 'center'),      # Λ, Σ*
        (6, 3, 'baryon'):   (6, 4, 'left'),       # Δ, Ξ
        (2, 2, 'meson'):    (0, 8, 'center'),     # K±
        (3, 2, 'meson'):    None,                   # D±, J/ψ — handled manually
        (6, 2, 'meson'):    (6, 4, 'left'),       # π⁰, ω, D⁰
        (12, 2, 'meson'):   (6, 4, 'left'),       # K⁰, ρ, Υ
        (27, 2, 'meson'):   (-6, 4, 'right'),     # B± (far right)
    }

    for (g, nq), cat_dict in sorted(groups.items()):
        for cat, names in cat_dict.items():
            # Plot one dot per category at this point
            ax.scatter(g, nq,
                       c=CAT_COLORS[cat], marker=CAT_MARKERS[cat],
                       s=80, zorder=5, edgecolors='black',
                       linewidths=0.5, alpha=0.8)

            # Combined label
            label = ', '.join(names)
            offset_info = GROUP_OFFSETS.get((g, nq, cat), (6, 4, 'left'))
            if offset_info is None:
                # Manual split labels
                if (g, nq, cat) == (0, 3, 'baryon'):
                    # p, n, Σ above; Λc, Ξc below
                    above = ', '.join(names[:2])  # 'p, n' and 'Σ'
                    below = ', '.join(names[2:])  # 'Λc' and 'Ξc'
                    ax.annotate(above, (g, nq), fontsize=6.5,
                                xytext=(0, 8), textcoords='offset points',
                                ha='center',
                                bbox=dict(boxstyle='round,pad=0.12', fc='white',
                                          ec='none', alpha=0.8))
                    ax.annotate(below, (g, nq), fontsize=6.5,
                                xytext=(0, -12), textcoords='offset points',
                                ha='center',
                                bbox=dict(boxstyle='round,pad=0.12', fc='white',
                                          ec='none', alpha=0.8))
                else:
                    # Default split: first above, rest below (e.g., D± / J/ψ)
                    for i, n in enumerate(names):
                        dy = 8 if i == 0 else -12
                        ax.annotate(n, (g, nq), fontsize=6.5,
                                    xytext=(0, dy), textcoords='offset points',
                                    ha='center',
                                    bbox=dict(boxstyle='round,pad=0.12', fc='white',
                                              ec='none', alpha=0.8))
            else:
                dx, dy, ha = offset_info
                ax.annotate(label, (g, nq), fontsize=6.5,
                            xytext=(dx, dy), textcoords='offset points',
                            ha=ha,
                            bbox=dict(boxstyle='round,pad=0.12', fc='white',
                                      ec='none', alpha=0.8))

    # Genus dividing line
    ax.axvline(0.0, color='gray', ls=':', lw=1.5, alpha=0.5)
    ax.annotate('genus = 0\n(unknots)', xy=(0.3, 3.8), fontsize=8,
                ha='left', color='gray', fontstyle='italic')
    ax.annotate('genus > 0\n(true knots)', xy=(2.5, 3.42), fontsize=8,
                ha='center', color='gray', fontstyle='italic')

    # τ callout — below the dot, clearing the "genus = 0" label above
    ax.annotate('τ: genus 3\n→ true knot,\nnot a lepton!',
                xy=(3.0, 2.95), xytext=(3.0, 2.35),
                fontsize=8, color='#d62728', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='#d62728', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5))

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
               ms=9, markeredgecolor='black', markeredgewidth=0.5,
               label='Leptons ($g=0, n_q=0$)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e',
               ms=9, markeredgecolor='black', markeredgewidth=0.5,
               label='Mesons ($n_q=2$)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#2ca02c',
               ms=9, markeredgecolor='black', markeredgewidth=0.5,
               label='Baryons ($n_q=3$)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#d62728',
               ms=9, markeredgecolor='black', markeredgewidth=0.5,
               label='τ (stealth baryon)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              framealpha=0.95, fontsize=9)

    ax.set_xlabel('Genus  g = (p−1)(q−1)/2')
    ax.set_ylabel('Constituent count  $n_q$')
    ax.set_yticks([0, 2, 3])
    ax.set_yticklabels(['0 (lepton)', '2 (meson)', '3 (baryon)'])
    ax.set_xlim(-1, 29)
    ax.set_ylim(-0.8, 4.2)
    ax.grid(True, alpha=0.15)
    ax.set_title('The Genus Criterion: Leptons are Genus-Zero Unknots')

    plt.tight_layout()
    plt.savefig('papers/figures/paper7_fig1_genus.png', dpi=300,
                bbox_inches='tight')
    print("Saved: papers/figures/paper7_fig1_genus.png")
    plt.close()


if __name__ == "__main__":
    figure1_genus()
