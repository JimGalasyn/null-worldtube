#!/usr/bin/env python3
"""
Publication-quality figures for Paper 6:
"The Particle Mass Spectrum from Torus Knot Mode-Locking"

Generates 6 figures + the full 45-row LaTeX table.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.optimize import brentq
from scipy.integrate import quad
from math import gcd

# ── Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'figure.dpi': 300,
    'font.family': 'serif', 'mathtext.fontset': 'cm',
})

ME_MEV = 0.51100
KAPPA = np.pi**2

# ── Core calculations ───────────────────────────────────────────────

def knot_ds_dt(t, p, q, kappa):
    theta = q * t
    return np.sqrt(p**2 * (kappa + np.cos(theta))**2 + q**2)

def n_gpe(t, q, kappa, xi_over_R):
    theta = q * t
    xi_rho = xi_over_R * kappa / (kappa + np.cos(theta))
    return np.sqrt(1.0 + xi_rho**2)

def phase_integral_hat(p, q, kappa, xi_over_R):
    def integrand(t):
        return n_gpe(t, q, kappa, xi_over_R) * knot_ds_dt(t, p, q, kappa)
    result, _ = quad(integrand, 0, 2*np.pi, limit=200)
    return result / (2*np.pi)

def find_beta_corrected(p, q, m_int, kappa=KAPPA):
    def residual(beta):
        return (beta/kappa) * phase_integral_hat(p, q, kappa, 1.0/beta) - m_int
    betas = np.linspace(0.1, 50.0, 500)
    vals = [residual(b) for b in betas]
    for i in range(len(vals)-1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i]*vals[i+1] < 0:
            try: return brentq(residual, betas[i], betas[i+1], xtol=1e-10)
            except: pass
    return None

def beta_simple(p, m_int):
    val = m_int**2/p**2 - 1
    return np.sqrt(val) if val > 0 else None

BETA_E_C = find_beta_corrected(2, 1, 3)
BETA_E_S = np.sqrt(5)/2
PQ_E = 5

def mass_formula(p, q, m_int, n_q, corrected=True):
    if corrected:
        beta = find_beta_corrected(p, q, m_int)
        bref = BETA_E_C
    else:
        beta = beta_simple(p, m_int)
        bref = BETA_E_S
    if beta is None or beta <= 0: return None, None
    pq = p**2 + q**2
    geom = (pq/PQ_E) * (beta/bref) * (np.log(8*beta)/np.log(8*bref))
    enhance = n_q**q if n_q > 0 else 1
    return geom * enhance * ME_MEV, beta

# ── Particle table ──────────────────────────────────────────────────
# (name, mass_exp, n_q, p, q, m, sector)
TABLE = [
    ('$e$',           0.511,  0, 2, 1, 3,  'L'),
    ('$\\mu$',       105.66,  0, 1, 8, 9,  'H'),
    ('$\\pi^0$',     135.0,   2, 7, 3, 18, 'L'),
    ('$\\pi^\\pm$',  139.57,  2, 3, 5, 5,  'L'),
    ('$K^\\pm$',     493.68,  2, 2, 5, 8,  'L'),
    ('$K^0$',        497.61,  2, 7, 5, 15, 'L'),
    ('$\\eta$',      547.86,  2, 6, 5, 15, 'L'),
    ('$\\rho^0$',    775.26,  2, 5, 7, 7,  'L'),
    ('$\\omega$',    782.66,  2, 4, 5, 17, 'L'),
    ('$K^*$',        891.67,  2, 6, 5, 21, 'L'),
    ('$p$',          938.27,  3, 1, 4, 5,  'L'),
    ('$n$',          939.57,  3, 1, 4, 5,  'L'),
    ("$\\eta'$",     957.78,  2, 6, 5, 22, 'L'),
    ('$a_0(980)$',   980.0,   2, 9, 5, 23, 'L'),
    ('$f_0(980)$',   990.0,   2, 1, 5, 8,  'L'),
    ('$\\phi$',     1019.46,  2, 6, 5, 23, 'L'),
    ('$\\Lambda$',  1115.7,   3, 3, 4, 12, 'L'),
    ('$h_1(1170)$', 1166.0,   2, 3, 5, 20, 'L'),
    ('$\\Sigma$',   1189.4,   3, 1, 4, 6,  'L'),
    ('$b_1(1235)$', 1229.5,   2, 4, 5, 24, 'L'),
    ('$a_1(1260)$', 1230.0,   2, 2, 5, 16, 'L'),
    ('$\\Delta$',   1232.0,   3, 5, 4, 15, 'L'),
    ('$K_1(1270)$', 1272.0,   2, 3, 5, 21, 'L'),
    ('$f_2(1270)$', 1275.5,   2, 6, 5, 27, 'L'),
    ('$f_1(1285)$', 1281.9,   2, 6, 5, 27, 'L'),
    ('$\\Xi$',      1314.9,   3, 5, 4, 16, 'L'),
    ('$a_2(1320)$', 1318.3,   2, 1, 5, 10, 'L'),
    ('$\\Sigma^*$', 1385.0,   3, 3, 4, 14, 'L'),
    ('$K_1(1400)$', 1403.0,   2, 3, 5, 23, 'L'),
    ('$N^*(1440)$', 1440.0,   3, 4, 5, 7,  'L'),
    ('$N^*(1520)$', 1520.0,   3, 3, 4, 15, 'L'),
    ('$\\Xi^*$',    1530.0,   3, 7, 4, 18, 'L'),
    ('$\\Omega^-$', 1672.5,   3, 7, 4, 19, 'L'),
    ('$N^*(1680)$', 1680.0,   3, 5, 4, 19, 'L'),
    ('$\\tau$',     1776.86,  3, 3, 4, 17, 'H'),
    ('$D^0$',       1864.8,   2, 3, 7, 7,  'H'),
    ('$D^\\pm$',    1869.7,   2, 2, 7, 5,  'H'),
    ('$\\Lambda_c$',2286.5,   3, 1, 5, 3,  'H'),
    ('$\\Xi_c$',    2469.4,   3, 1, 4, 10, 'H'),
    ('$J/\\psi$',   3096.9,   2, 2, 7, 7,  'H'),
    ('$\\psi(2S)$', 3686.1,   2, 3, 7, 11, 'H'),
    ('$\\psi(3770)$',3773.7,  2, 2, 7, 8,  'H'),
    ('$B^\\pm$',    5279.3,   2, 10,7, 25, 'H'),
    ('$\\Lambda_b$',5619.6,   3, 3, 5, 14, 'H'),
    ('$\\Upsilon(1S)$',9460.3,2, 4, 9, 8,  'H'),
]

# Plain-text names for figures
TABLE_PLAIN = [
    ('e',0.511,0,2,1,3,'L','lepton'),('μ',105.66,0,1,8,9,'H','lepton'),
    ('π⁰',135.0,2,7,3,18,'L','meson'),('π±',139.57,2,3,5,5,'L','meson'),
    ('K±',493.68,2,2,5,8,'L','meson'),('K⁰',497.61,2,7,5,15,'L','meson'),
    ('η',547.86,2,6,5,15,'L','meson'),('ρ⁰',775.26,2,5,7,7,'L','meson'),
    ('ω',782.66,2,4,5,17,'L','meson'),('K*',891.67,2,6,5,21,'L','meson'),
    ('p',938.27,3,1,4,5,'L','baryon'),('n',939.57,3,1,4,5,'L','baryon'),
    ("η'",957.78,2,6,5,22,'L','meson'),('a₀',980.0,2,9,5,23,'L','meson'),
    ('f₀',990.0,2,1,5,8,'L','meson'),('φ',1019.46,2,6,5,23,'L','meson'),
    ('Λ',1115.7,3,3,4,12,'L','baryon'),('h₁',1166.0,2,3,5,20,'L','meson'),
    ('Σ',1189.4,3,1,4,6,'L','baryon'),('b₁',1229.5,2,4,5,24,'L','meson'),
    ('a₁',1230.0,2,2,5,16,'L','meson'),('Δ',1232.0,3,5,4,15,'L','baryon'),
    ('K₁',1272.0,2,3,5,21,'L','meson'),('f₂',1275.5,2,6,5,27,'L','meson'),
    ('f₁',1281.9,2,6,5,27,'L','meson'),('Ξ',1314.9,3,5,4,16,'L','baryon'),
    ('a₂',1318.3,2,1,5,10,'L','meson'),('Σ*',1385.0,3,3,4,14,'L','baryon'),
    ("K₁'",1403.0,2,3,5,23,'L','meson'),('N*14',1440.0,3,4,5,7,'L','baryon'),
    ('N*15',1520.0,3,3,4,15,'L','baryon'),('Ξ*',1530.0,3,7,4,18,'L','baryon'),
    ('Ω⁻',1672.5,3,7,4,19,'L','baryon'),('N*16',1680.0,3,5,4,19,'L','baryon'),
    ('τ',1776.86,3,3,4,17,'H','stealth'),
    ('D⁰',1864.8,2,3,7,7,'H','meson'),('D±',1869.7,2,2,7,5,'H','meson'),
    ('Λc',2286.5,3,1,5,3,'H','baryon'),('Ξc',2469.4,3,1,4,10,'H','baryon'),
    ('J/ψ',3096.9,2,2,7,7,'H','meson'),('ψ2S',3686.1,2,3,7,11,'H','meson'),
    ('ψ37',3773.7,2,2,7,8,'H','meson'),('B±',5279.3,2,10,7,25,'H','meson'),
    ('Λb',5619.6,3,3,5,14,'H','baryon'),('Υ',9460.3,2,4,9,8,'H','meson'),
    # Tetraquarks (n_q=4)
    ('X(3872)',3872.0,4,1,3,27,'H','tetra'),
    ('Tcc',3875.0,4,1,3,27,'H','tetra'),
    ('Zc',3887.0,4,1,3,27,'H','tetra'),
    ('X(4274)',4274.0,4,4,5,6,'H','tetra'),
    ('Z(4430)',4478.0,4,6,5,8,'H','tetra'),
    ('X(4500)',4506.0,4,6,5,8,'H','tetra'),
    # Pentaquarks (n_q=5)
    ('Pc43',4312.0,5,1,3,17,'H','penta'),
    ('Pc44',4380.0,5,11,3,27,'H','penta'),
    ('Pc44a',4440.0,5,2,3,28,'H','penta'),
    ('Pc44b',4457.0,5,2,3,28,'H','penta'),
    ('Pcs',4459.0,5,2,3,28,'H','penta'),
]

CAT_COLORS = {'lepton':'#1f77b4','meson':'#ff7f0e','baryon':'#2ca02c','stealth':'#d62728',
              'tetra':'#9467bd','penta':'#8c564b'}
CAT_MARKERS = {'lepton':'o','meson':'s','baryon':'^','stealth':'D',
               'tetra':'d','penta':'p'}

def compute_all():
    """Compute predictions for all particles."""
    results = []
    for name, mass_exp, n_q, p, q, m_int, sector, cat in TABLE_PLAIN:
        corrected = (sector == 'L')
        m_pred, beta = mass_formula(p, q, m_int, n_q, corrected)
        if m_pred is None: continue
        err = (m_pred/mass_exp - 1)*100
        enhance = n_q**q if n_q > 0 else 1
        results.append({
            'name': name, 'mass_exp': mass_exp, 'n_q': n_q,
            'p': p, 'q': q, 'm': m_int, 'sector': sector, 'cat': cat,
            'beta': beta, 'enhance': enhance, 'mass_pred': m_pred, 'err': err,
        })
    return results

print("Computing all predictions...", flush=True)
RESULTS = compute_all()
print(f"  Done: {len(RESULTS)} particles")

legend_elements = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#1f77b4',ms=8,
           markeredgecolor='black',markeredgewidth=0.8,label='Leptons ($n_q=0$)'),
    Line2D([0],[0],marker='s',color='w',markerfacecolor='#ff7f0e',ms=8,
           markeredgecolor='black',markeredgewidth=0.8,label='Mesons ($n_q=2$)'),
    Line2D([0],[0],marker='^',color='w',markerfacecolor='#2ca02c',ms=8,
           markeredgecolor='black',markeredgewidth=0.8,label='Baryons ($n_q=3$)'),
    Line2D([0],[0],marker='D',color='w',markerfacecolor='#d62728',ms=8,
           markeredgecolor='black',markeredgewidth=0.8,label='$\\tau$ (stealth baryon)'),
    Line2D([0],[0],marker='d',color='w',markerfacecolor='#9467bd',ms=8,
           markeredgecolor='black',markeredgewidth=0.8,label='Tetraquarks ($n_q=4$)'),
    Line2D([0],[0],marker='p',color='w',markerfacecolor='#8c564b',ms=8,
           markeredgecolor='black',markeredgewidth=0.8,label='Pentaquarks ($n_q=5$)'),
]


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 2: Predicted vs Measured + Error Histogram
# ═══════════════════════════════════════════════════════════════════

def figure2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # (a) Predicted vs measured
    for r in RESULTS:
        ax1.scatter(r['mass_exp'], r['mass_pred'], c=CAT_COLORS[r['cat']],
                    marker=CAT_MARKERS[r['cat']], s=40, alpha=0.6,
                    edgecolors='black', linewidths=0.5, zorder=5)
    lim = [0.3, 12000]
    ax1.plot(lim, lim, 'k--', lw=0.8, alpha=0.5)
    # ±3% bands
    ax1.fill_between(lim, [x*0.97 for x in lim], [x*1.03 for x in lim],
                     alpha=0.08, color='green')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('Measured mass (MeV)')
    ax1.set_ylabel('Predicted mass (MeV)')
    ax1.set_title('(a)')
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.legend(handles=legend_elements, fontsize=7, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.2)

    # (b) Error histogram
    errs = [r['err'] for r in RESULTS]
    ax2.hist(errs, bins=20, range=(-4, 4), color='#4c72b0', alpha=0.7,
             edgecolor='black', linewidth=0.5)
    ax2.axvline(0, color='red', ls='--', lw=1.5)
    ax2.axvline(np.median(np.abs(errs)), color='green', ls=':', lw=1.5,
                label=f'Median |err| = {np.median(np.abs(errs)):.2f}%')
    ax2.set_xlabel('Prediction error (%)')
    ax2.set_ylabel('Count')
    ax2.set_title('(b)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('papers/figures/paper6_fig4_scatter.png', dpi=300, bbox_inches='tight')
    print("  Saved: paper6_fig4_scatter.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 3: Baryon Spectrum
# ═══════════════════════════════════════════════════════════════════

def figure3():
    baryons = [r for r in RESULTS if r['cat'] == 'baryon' and r['mass_exp'] < 2000]
    baryons.sort(key=lambda x: x['mass_exp'])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = [r['name'] for r in baryons]
    exp = [r['mass_exp'] for r in baryons]
    pred = [r['mass_pred'] for r in baryons]
    x = np.arange(len(names))
    width = 0.35

    bars_exp = ax.bar(x - width/2, exp, width, label='Experiment', color='#4c72b0', alpha=0.8)
    bars_pred = ax.bar(x + width/2, pred, width, label='$n_q^q$ formula', color='#55a868', alpha=0.8)

    # Mark q=5 Roper differently
    for i, r in enumerate(baryons):
        if r['q'] == 5:
            ax.bar(x[i] + width/2, pred[i], width, color='#c44e52', alpha=0.8)
            ax.annotate('$q\\!=\\!5$', (x[i] + width/2, pred[i] + 40),
                        fontsize=8, ha='center', color='#c44e52',
                        fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('Mass (MeV)')
    ax.set_title('Light baryons: all $q=4$ ($3^4=81$) except Roper $N^*(1440)$ at $q=5$ ($3^5=243$)')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2, axis='y')
    fig.subplots_adjust(bottom=0.18)
    plt.savefig('papers/figures/paper6_fig2_baryons.png', dpi=300, bbox_inches='tight')
    print("  Saved: paper6_fig2_baryons.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 4: Flavor Towers
# ═══════════════════════════════════════════════════════════════════

def figure4():
    fig, ax = plt.subplots(figsize=(8, 5))

    # Group mesons by q
    q_groups = {3: [], 5: [], 7: [], 9: []}
    for r in RESULTS:
        if r['cat'] == 'meson' and r['q'] in q_groups:
            q_groups[r['q']].append(r)

    colors = {3: '#c44e52', 5: '#4c72b0', 7: '#55a868', 9: '#8172b2'}
    labels = {3: '$q=3$: $2^3=8$\n(π⁰)', 5: '$q=5$: $2^5=32$\n(light)',
              7: '$q=7$: $2^7=128$\n(charm)', 9: '$q=9$: $2^9=512$\n(bottom)'}

    for q_val in [3, 5, 7, 9]:
        particles = sorted(q_groups[q_val], key=lambda x: x['mass_exp'])
        for i, r in enumerate(particles):
            ax.scatter(q_val, r['mass_exp'],
                       c=colors[q_val], s=50, zorder=5, edgecolors='black',
                       linewidths=0.5, alpha=0.7)
            # Combine ψ2S and ψ37 into a single label
            if r['name'] in ('ψ2S', 'ψ37'):
                if r['name'] == 'ψ2S':
                    ax.annotate('ψ2S, ψ37', (q_val, r['mass_exp']),
                                fontsize=5.5, xytext=(4, 2), textcoords='offset points')
            # Combine D⁰ and D± into a single label
            elif r['name'] in ('D⁰', 'D±'):
                if r['name'] == 'D⁰':
                    ax.annotate('D⁰, D±', (q_val, r['mass_exp']),
                                fontsize=5.5, xytext=(4, 2), textcoords='offset points')
            # Nudge J/ψ label slightly right
            elif r['name'] == 'J/ψ':
                ax.annotate(r['name'], (q_val, r['mass_exp']),
                            fontsize=5.5, xytext=(6, 2), textcoords='offset points')
            elif len(particles) <= 8 or i % 3 == 0:
                ax.annotate(r['name'], (q_val, r['mass_exp']),
                            fontsize=5.5, xytext=(4, 2), textcoords='offset points')

    # Enclose dense q=5 cluster with a light box and label
    from matplotlib.patches import FancyBboxPatch
    cluster_box = FancyBboxPatch((4.82, 430), 0.36, 1140, boxstyle='round,pad=0.05',
                                  lw=0.8, edgecolor='#4c72b0', facecolor='none',
                                  alpha=0.5, ls='--')
    ax.add_patch(cluster_box)
    ax.text(5.28, 1050, '(14 mesons)', fontsize=7, color='#4c72b0',
            fontstyle='italic', va='center')

    # Confinement factor labels
    for q_val in [3, 5, 7, 9]:
        ax.annotate(labels[q_val], (q_val, 50), fontsize=8, ha='center',
                    color=colors[q_val], fontweight='bold')

    # Legend
    from matplotlib.lines import Line2D as L2D
    leg = [L2D([0],[0], marker='o', color='w', markerfacecolor=colors[q], ms=8,
               markeredgecolor='black', markeredgewidth=0.5, label=labels[q].replace('\n',' '))
           for q in [3, 5, 7, 9]]
    ax.legend(handles=leg, fontsize=7, loc='upper left', framealpha=0.9)

    ax.set_xticks([3, 5, 7, 9])
    ax.set_xlabel('Poloidal winding $q$ (= flavor)', fontsize=11)
    ax.set_ylabel('Mass (MeV)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Meson flavor towers: $q=3$ (π⁰), $q=5$ (light), $q=7$ (charm), $q=9$ (bottom)')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('papers/figures/paper6_fig3_flavor.png', dpi=300, bbox_inches='tight')
    print("  Saved: paper6_fig3_flavor.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 5: Cascade Argument
# ═══════════════════════════════════════════════════════════════════

def figure5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Compound growth: n_q^w for w = 0..q
    for nq, color, label in [(2, '#ff7f0e', 'Mesons: $n_q=2$'),
                               (3, '#2ca02c', 'Baryons: $n_q=3$')]:
        windings = range(0, 10)
        vals = [nq**w for w in windings]
        ax1.semilogy(list(windings), vals, 'o-', color=color, lw=2, ms=6, label=label)

    ax1.axhline(81, color='#2ca02c', ls=':', alpha=0.4)
    ax1.annotate('$3^4\\!=\\!81$ (proton)', xy=(4, 81), xytext=(1.0, 40),
                 fontsize=8, color='#2ca02c',
                 arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=0.8),
                 bbox=dict(boxstyle='round,pad=0.15', fc='white',
                           ec='none', alpha=0.9))
    ax1.axhline(128, color='#ff7f0e', ls=':', alpha=0.4)
    ax1.annotate('$2^7\\!=\\!128$ (charm)', xy=(7, 128), xytext=(6.0, 350),
                 fontsize=8, color='#ff7f0e',
                 arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=0.8),
                 bbox=dict(boxstyle='round,pad=0.15', fc='white',
                           ec='none', alpha=0.9))
    ax1.set_xlabel('Poloidal winding $q$')
    ax1.set_ylabel('Confinement factor $n_q^q$')
    ax1.set_title('(a) Cascade: compound growth')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # (b) Confinement condition gcd(n_q, q) = 1
    q_range = range(1, 11)
    nq_range = range(1, 7)
    confined = np.zeros((len(list(nq_range)), len(list(q_range))))
    for i, nq in enumerate(nq_range):
        for j, q in enumerate(q_range):
            if gcd(nq, q) == 1:
                confined[i, j] = 1

    ax2.imshow(confined, cmap='RdYlGn', aspect='auto', origin='lower',
               extent=[0.5, 10.5, 0.5, 6.5], alpha=0.8)
    ax2.plot(4, 3, 'r*', ms=18, markeredgecolor='black', markeredgewidth=0.8)
    ax2.annotate('proton\n$(n_q\\!=\\!3, q\\!=\\!4)$', (4, 3), fontsize=7,
                 xytext=(15, -15), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='red'))
    ax2.set_xlabel('Poloidal winding $q$')
    ax2.set_ylabel('Constituents $n_q$')
    ax2.set_title('(b) Confinement: gcd$(n_q, q) = 1$ (green)')
    ax2.set_xticks(range(1, 11))
    ax2.set_yticks(range(1, 7))

    plt.tight_layout()
    plt.savefig('papers/figures/paper6_fig1_cascade.png', dpi=300, bbox_inches='tight')
    print("  Saved: paper6_fig1_cascade.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 6: Mass vs Rotation Number (THE mass staircase)
# ═══════════════════════════════════════════════════════════════════

def figure6():
    from adjustText import adjust_text
    from matplotlib.patches import ConnectionPatch
    fig, ax = plt.subplots(figsize=(10, 6))

    # Zoom box bounds (in data coords)
    ZOOM_RHO = (0.45, 1.80)
    ZOOM_MASS = (400, 2150)

    def in_cluster(rho, mass):
        return ZOOM_RHO[0] <= rho <= ZOOM_RHO[1] and ZOOM_MASS[0] <= mass <= ZOOM_MASS[1]

    # Manual label overrides for particles OUTSIDE the cluster
    MANUAL = {
        'e':   {'offset': (4, -8), 'arrow': False},
        'π⁰':  {'offset': (4, 3),  'arrow': False},
        'π±':  {'offset': (4, -8), 'arrow': False},
        'K±':  {'offset': (8, 0), 'arrow': False},
        'p':   {'offset': (8, 0), 'arrow': False, 'label': 'p, n'},
        'n':   {'skip': True},
        'f₀':  {'offset': (8, 0), 'arrow': False},
        'Σ':   {'offset': (8, 0), 'arrow': False},
        'a₁':  {'offset': (8, -6), 'arrow': False},
        'a₂':  {'offset': (8, 0), 'arrow': False},
        'D⁰':  {'offset': (8, 3), 'arrow': False},
        'D±':  {'offset': (8, -6), 'arrow': False},
        'Λc':  {'offset': (8, 0), 'arrow': False},
        'Ξc':  {'offset': (8, 0), 'arrow': False},
        'J/ψ': {'offset': (8, -4), 'arrow': False},
        'ψ2S': {'offset': (8, 3), 'arrow': False},
        'ψ37': {'offset': (8, 4), 'arrow': False},
        'Υ':   {'offset': (8, 0), 'arrow': False},
        'B±':  {'offset': (0, 10), 'arrow': False, 'ha': 'center'},
        'Λb':  {'offset': (8, 0), 'arrow': False},
        'μ':   {'offset': (6, 3), 'arrow': False},
        # Exotics: combine near-degenerate states
        # Exotics at rho=3: combine X/Tcc/Zc
        'X(3872)': {'offset': (0, -10), 'arrow': False, 'label': 'X, Tcc, Zc', 'ha': 'center'},
        'Tcc':     {'skip': True},
        'Zc':      {'skip': True},
        # Exotics at rho=3: Pc43 just above
        'Pc43':    {'offset': (8, 0), 'arrow': False},
        # Exotics at low rho, high mass: spread vertically
        'X(4274)': {'offset': (0, 10), 'arrow': False, 'ha': 'center'},
        'Z(4430)': {'offset': (0, -18), 'arrow': False, 'label': 'Z(4430),\nX(4500)', 'ha': 'center'},
        'X(4500)': {'skip': True},
        'Pc44':    {'offset': (0, 10), 'arrow': False, 'ha': 'center', 'label': 'Pc(4380)'},
        'Pc44a':   {'offset': (0, -18), 'arrow': False, 'label': 'Pcs,\nPc(44,44b)', 'ha': 'center'},
        'Pc44b':   {'skip': True},
        'Pcs':     {'skip': True},
    }
    bbox_style = dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.75)

    # --- Main plot: scatter all, label only non-cluster ---
    for r in sorted(RESULTS, key=lambda x: x['q']/x['p']):
        rho = r['q'] / r['p']
        if rho > 9: continue
        ax.scatter(rho, r['mass_exp'], c=CAT_COLORS[r['cat']],
                   marker=CAT_MARKERS[r['cat']], s=55, zorder=5,
                   edgecolors='black', linewidths=0.4, alpha=0.6)
        if in_cluster(rho, r['mass_exp']):
            continue  # labels go in the inset
        if r['name'] in MANUAL:
            m = MANUAL[r['name']]
            if m.get('skip'):
                pass
            else:
                label = m.get('label', r['name'])
                ax.annotate(label, (rho, r['mass_exp']), fontsize=6.5,
                            xytext=m['offset'], textcoords='offset points',
                            ha=m.get('ha', 'left'), bbox=bbox_style,
                            arrowprops=dict(arrowstyle='-', color='gray',
                                            lw=0.5, alpha=0.5) if m.get('arrow', True) else None)
        else:
            ax.annotate(r['name'], (rho, r['mass_exp']), fontsize=6.5,
                        xytext=(4, 3), textcoords='offset points', bbox=bbox_style)

    ax.set_xlabel('Rotation number $\\rho = q/p$', fontsize=11)
    ax.set_ylabel('Mass (MeV)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('The particle mass staircase')
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right', framealpha=0.9,
              labelspacing=0.8)
    ax.grid(True, alpha=0.2)

    # --- Zoom box on main axes ---
    from matplotlib.patches import Rectangle
    rect = Rectangle((ZOOM_RHO[0], ZOOM_MASS[0]), ZOOM_RHO[1]-ZOOM_RHO[0],
                      ZOOM_MASS[1]-ZOOM_MASS[0], lw=0.6, edgecolor='gray',
                      facecolor='none', ls='--', zorder=10)
    ax.add_patch(rect)

    # --- Inset axes ---
    axins = ax.inset_axes([0.30, 0.10, 0.48, 0.45])  # [left, bottom, width, height] in axes fraction
    LEFT_NUDGE = {'Ξ', 'Δ', 'N*14'}  # place label to the left
    COMBINE = {'f₂': 'f₁, f₂', 'f₁': None}  # combine onto one label
    inset_texts = []
    for r in sorted(RESULTS, key=lambda x: x['q']/x['p']):
        rho = r['q'] / r['p']
        if not in_cluster(rho, r['mass_exp']):
            continue
        axins.scatter(rho, r['mass_exp'], c=CAT_COLORS[r['cat']],
                      marker=CAT_MARKERS[r['cat']], s=40, zorder=5,
                      edgecolors='black', linewidths=0.4, alpha=0.6)
        # Skip labels that are combined into another
        if r['name'] in COMBINE and COMBINE[r['name']] is None:
            continue
        label = COMBINE.get(r['name'], r['name'])
        if r['name'] in LEFT_NUDGE:
            offset = (-6, 0)
            ha = 'right'
        else:
            offset = (6, 0)
            ha = 'left'
        axins.annotate(label, (rho, r['mass_exp']), fontsize=5.5,
                       xytext=offset, textcoords='offset points', ha=ha,
                       bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                 ec='none', alpha=0.8))

    axins.set_xlim(ZOOM_RHO)
    axins.set_ylim(ZOOM_MASS)
    axins.grid(True, alpha=0.15)
    axins.set_xlabel('$\\rho$', fontsize=8)
    axins.set_ylabel('Mass (MeV)', fontsize=8)
    axins.tick_params(labelsize=7)

    # Connect zoom box to inset with lines
    # Lower-left of zoom box to lower-left of inset
    con1 = ConnectionPatch(xyA=(ZOOM_RHO[0], ZOOM_MASS[0]), coordsA=ax.transData,
                           xyB=(0, 0), coordsB=axins.transAxes,
                           color='gray', ls='--', lw=0.6)
    # Upper-right of zoom box to upper-right of inset
    con2 = ConnectionPatch(xyA=(ZOOM_RHO[1], ZOOM_MASS[1]), coordsA=ax.transData,
                           xyB=(1, 1), coordsB=axins.transAxes,
                           color='gray', ls='--', lw=0.6)
    fig.add_artist(con1)
    fig.add_artist(con2)

    plt.tight_layout()
    plt.savefig('papers/figures/paper6_fig5_staircase.png', dpi=300, bbox_inches='tight')
    print("  Saved: paper6_fig5_staircase.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  LATEX TABLE
# ═══════════════════════════════════════════════════════════════════

def generate_latex_table():
    """Generate the full 45-row LaTeX table."""
    lines = []
    lines.append('% Auto-generated by paper6_figures.py')
    lines.append('\\begin{longtable}{rlrrrrrrrr}')
    lines.append('\\caption{Complete mass predictions from the $n_q^q$ formula.}')
    lines.append('\\label{tab:spectrum} \\\\')
    lines.append('\\toprule')
    lines.append(' & Particle & $m_{\\text{exp}}$ & $(p,q)$ & $m$ & $n_q$ & $n_q^q$ & $\\beta$ & $m_{\\text{pred}}$ & Error \\\\')
    lines.append(' &          & (MeV) &         &     &       &        &         & (MeV) & (\\%) \\\\')
    lines.append('\\midrule')
    lines.append('\\endfirsthead')
    lines.append('\\multicolumn{10}{c}{\\tablename\\ \\thetable\\ -- continued} \\\\')
    lines.append('\\toprule')
    lines.append(' & Particle & $m_{\\text{exp}}$ & $(p,q)$ & $m$ & $n_q$ & $n_q^q$ & $\\beta$ & $m_{\\text{pred}}$ & Error \\\\')
    lines.append('\\midrule')
    lines.append('\\endhead')

    sorted_table = sorted(zip(TABLE, RESULTS), key=lambda x: x[0][1])

    for i, ((name_tex, mass_exp, n_q, p, q, m_int, sector), r) in enumerate(sorted_table):
        enhance = n_q**q if n_q > 0 else 1
        err_str = f'${r["err"]:+.2f}$'
        lines.append(
            f'{i+1} & {name_tex} & {mass_exp:.2f} & $({p},{q})$ & {m_int} & '
            f'{n_q} & {enhance} & {r["beta"]:.3f} & {r["mass_pred"]:.1f} & {err_str} \\\\'
        )

    lines.append('\\bottomrule')
    lines.append('\\end{longtable}')

    table_tex = '\n'.join(lines)
    with open('papers/paper6_table.tex', 'w') as f:
        f.write(table_tex)
    print("  Saved: paper6_table.tex")
    return table_tex


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs('papers/figures', exist_ok=True)

    print("\nGenerating Figure 2: Predicted vs Measured...")
    figure2()
    print("Generating Figure 3: Baryon Spectrum...")
    figure3()
    print("Generating Figure 4: Flavor Towers...")
    figure4()
    print("Generating Figure 5: Cascade Argument...")
    figure5()
    print("Generating Figure 6: Mass Staircase...")
    figure6()
    print("\nGenerating LaTeX table...")
    generate_latex_table()

    # Summary
    errors = [abs(r['err']) for r in RESULTS]
    print(f"\n{'='*60}")
    print(f"Paper 6 figures complete.")
    print(f"  Particles: {len(RESULTS)}")
    print(f"  Within 1%: {sum(1 for e in errors if e < 1)}")
    print(f"  Within 3%: {sum(1 for e in errors if e < 3)}")
    print(f"  Median:    {np.median(errors):.2f}%")
    print(f"  RMS:       {np.sqrt(np.mean(np.array(errors)**2)):.2f}%")
    print(f"  Max:       {max(errors):.2f}%")
    print(f"{'='*60}")
