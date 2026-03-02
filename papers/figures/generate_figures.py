#!/usr/bin/env python3
"""
Generate all 5 publication figures for Paper 1 (Lepton Spectrum).

Usage:
    python3 generate_figures.py          # all figures
    python3 generate_figures.py 1        # just figure 1
    python3 generate_figures.py 2 4      # figures 2 and 4

Requires: matplotlib, numpy
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ── Publication style ──────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.edgecolor': '#333333',
    'grid.color': '#dddddd',
    'grid.linewidth': 0.4,
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
})

# Sector colors (consistent across all figures)
C_FERMION = '#c62828'   # red — fermion masses
C_GAUGE   = '#1565c0'   # blue — gauge/Higgs
C_CKM     = '#2e7d32'   # green — CKM
C_PMNS    = '#e65100'   # orange — PMNS
C_KNOT    = '#c62828'   # knot curve color
C_TORUS   = '#1565c0'   # torus wireframe


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: The (2,1) Torus Knot
# ═══════════════════════════════════════════════════════════════════

def figure1_torus_knot():
    """3D rendering of the (2,1) torus knot on a transparent torus."""
    fig = plt.figure(figsize=(3.4, 3.0))
    ax = fig.add_subplot(111, projection='3d')

    R, r = 1.0, 0.25  # slightly exaggerated r/R for visibility
    p, q = 2, 1
    N = 2000

    # Knot curve
    lam = np.linspace(0, 2 * np.pi, N)
    theta = p * lam
    phi = q * lam
    x_k = (R + r * np.cos(phi)) * np.cos(theta)
    y_k = (R + r * np.cos(phi)) * np.sin(theta)
    z_k = r * np.sin(phi)

    # Draw knot with color gradient showing circulation
    cmap = plt.cm.Reds
    colors = cmap(np.linspace(0.35, 0.95, N))
    for i in range(N - 1):
        ax.plot([x_k[i], x_k[i+1]], [y_k[i], y_k[i+1]], [z_k[i], z_k[i+1]],
                color=colors[i], linewidth=2.0, alpha=0.9)

    # Torus wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 30)
    U, V = np.meshgrid(u, v)
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    ax.plot_surface(X, Y, Z, alpha=0.04, color=C_TORUS, linewidth=0)
    ax.plot_wireframe(X, Y, Z, alpha=0.12, color=C_TORUS, linewidth=0.25,
                      rstride=3, cstride=3)

    # Radius annotations — R arrow along the equatorial plane
    arrow_theta = np.pi * 0.15
    ax.plot([0, R * np.cos(arrow_theta)],
            [0, R * np.sin(arrow_theta)],
            [0, 0], 'k-', linewidth=0.8)
    ax.text(0.45 * np.cos(arrow_theta) - 0.05,
            0.45 * np.sin(arrow_theta) + 0.1,
            0.05, r'$R$', fontsize=10, ha='center', color='black')

    # r arrow — from torus center circle outward at the front
    r_theta = 0.0
    cx, cy = R * np.cos(r_theta), R * np.sin(r_theta)
    ax.plot([cx, cx + r], [cy, cy], [0, 0], 'k-', linewidth=0.8)
    ax.text(cx + r * 0.5, cy + 0.08, 0.07, r'$r$', fontsize=10,
            ha='center', color='black')

    # α = r/R label
    ax.text2D(0.02, 0.02, r'$\alpha = r/R$', transform=ax.transAxes,
              fontsize=9, color='#444444')

    # Current direction arrows along knot (triangular markers)
    arrow_indices = np.linspace(100, N - 100, 6, dtype=int)
    for idx in arrow_indices:
        dx = x_k[idx+5] - x_k[idx-5]
        dy = y_k[idx+5] - y_k[idx-5]
        dz = z_k[idx+5] - z_k[idx-5]
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        ax.quiver(x_k[idx], y_k[idx], z_k[idx],
                  dx/norm, dy/norm, dz/norm,
                  length=0.08, color='black', arrow_length_ratio=0.6,
                  linewidth=0.6)

    ax.view_init(elev=25, azim=30)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-0.6, 0.6)
    ax.set_axis_off()
    ax.set_title('(2,1) torus knot', fontsize=10, pad=-5)

    path = os.path.join(OUTDIR, 'figure1_torus_knot.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Complete Scorecard — Predicted vs Measured
# ═══════════════════════════════════════════════════════════════════

# Shared data from Table II (paper1_lepton_spectrum.tex lines 286-321)
# Format: (label, predicted, measured, sector, is_dimensionless)
SCORECARD_DATA = [
    # Fermion masses (MeV) — entries 1-7
    (r'$m_\tau$',    1776.9,    1776.86,   'fermion', False),
    (r'$m_t$',       170400.0,  172760.0,  'fermion', False),
    (r'$m_b$',       4172.0,    4180.0,    'fermion', False),
    (r'$m_c$',       1265.0,    1270.0,    'fermion', False),
    (r'$m_s$',       94.1,      93.4,      'fermion', False),
    (r'$m_d$',       4.54,      4.67,      'fermion', False),
    (r'$m_u$',       2.36,      2.16,      'fermion', False),
    # Gauge/Higgs masses (MeV) — entries 10-13
    (r'$v$',         246340.0,  246220.0,  'gauge',   False),
    (r'$m_H$',       125970.0,  125100.0,  'gauge',   False),
    (r'$m_W$',       80400.0,   80370.0,   'gauge',   False),
    (r'$m_Z$',       91600.0,   91190.0,   'gauge',   False),
    # Dimensionless gauge/Higgs — entries 8-9
    (r'$\sin^2\!\theta_W$', 0.23077, 0.23122, 'gauge', True),
    (r'$\alpha_s$',  0.11676,   0.1179,    'gauge',   True),
    # CKM — entries 14-17
    (r'$V_{us}$',    0.2214,    0.2243,    'ckm',     True),
    (r'$V_{cb}$',    0.0429,    0.0408,    'ckm',     True),
    (r'$V_{ub}$',    0.00376,   0.00382,   'ckm',     True),
    (r'$\delta_{CP}$', 1.1416,  1.144,     'ckm',     True),
    # PMNS — entries 18-21
    (r'$\sin^2\!\theta_{12}$', 0.3077, 0.307, 'pmns', True),
    (r'$\sin^2\!\theta_{23}$', 0.5556, 0.561, 'pmns', True),
    (r'$\sin^2\!\theta_{13}$', 0.0218, 0.0220, 'pmns', True),
    (r'$\delta_{CP}^\nu$', np.pi, np.radians(177), 'pmns', True),
]

SECTOR_COLORS = {
    'fermion': C_FERMION, 'gauge': C_GAUGE,
    'ckm': C_CKM, 'pmns': C_PMNS,
}
SECTOR_LABELS = {
    'fermion': 'Fermion masses', 'gauge': 'Gauge / Higgs',
    'ckm': 'CKM', 'pmns': 'PMNS',
}


def _add_error_histogram(ax, subset_data):
    """Add error histogram to an axes from a list of (label, pred, meas, sector) tuples."""
    errors = [abs(p - m) / m * 100 for _, p, m, s in subset_data if m != 0]
    present_sectors = sorted(set(s for _, _, _, s in subset_data),
                             key=list(SECTOR_COLORS.keys()).index)
    bins = np.linspace(0, 10, 21)
    for sec in present_sectors:
        sec_errors = [abs(p - m) / m * 100
                      for _, p, m, s in subset_data if s == sec and m != 0]
        if sec_errors:
            ax.hist(sec_errors, bins=bins, color=SECTOR_COLORS[sec], alpha=0.6,
                    label=SECTOR_LABELS[sec], edgecolor='white', linewidth=0.3)
    if errors:
        median_err = np.median(errors)
        ax.axvline(median_err, color='black', linewidth=1.0, linestyle='--',
                   label=f'Median: {median_err:.1f}%')
    ax.set_xlabel('|Error| (%)')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 10)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8, edgecolor='none')


def figure2a_masses():
    """Scorecard: masses and energy scales (log-log) with error histogram."""
    from matplotlib.ticker import FuncFormatter

    data = SCORECARD_DATA
    masses = [(l, p, m, s) for l, p, m, s, dim in data if not dim]

    fig = plt.figure(figsize=(3.4, 4.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.30)

    # ── Scatter panel ──
    ax = fig.add_subplot(gs[0])
    for label, pred, meas, sector in masses:
        ax.scatter(meas, pred, c=SECTOR_COLORS[sector], s=35,
                   zorder=5, edgecolors='none')
    lo, hi = 0.3, 3.5e5
    ax.plot([lo, hi], [lo, hi], 'k-', linewidth=0.5, alpha=0.4, zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Measured (MeV)')
    ax.set_ylabel('Predicted (MeV)')
    ax.set_title('Masses and energy scales', fontsize=10, pad=6)
    ax.set_aspect('equal')

    # Label well-separated points directly
    ew_labels = {r'$m_t$', r'$v$', r'$m_H$', r'$m_W$', r'$m_Z$'}
    offsets_mass = {
        r'$m_u$':    (-0.30, 0.0),
        r'$m_d$':    (0.10, -0.18),
        r'$m_s$':    (0.10, 0.08),
        r'$m_c$':    (-0.26, 0.06),
        r'$m_\tau$': (0.10, 0.08),
        r'$m_b$':    (0.10, -0.10),
    }
    for label, pred, meas, sector in masses:
        if label in ew_labels:
            continue
        dx, dy = offsets_mass.get(label, (0.10, 0.0))
        ax.annotate(label,
                    xy=(meas, pred),
                    xytext=(10**(np.log10(meas) + dx),
                            10**(np.log10(pred) + dy)),
                    fontsize=7.5, color=SECTOR_COLORS[sector],
                    ha='left' if dx > 0 else 'right', va='center')

    # Zoomed inset for electroweak cluster — positioned in the empty center
    ax_ins = ax.inset_axes([0.30, 0.12, 0.48, 0.38])
    ew_masses = [(l, p, m, s) for l, p, m, s in masses if l in ew_labels]
    for label, pred, meas, sector in ew_masses:
        ax_ins.scatter(meas, pred, c=SECTOR_COLORS[sector], s=28,
                       zorder=5, edgecolors='none')
    ins_lo, ins_hi = 6e4, 3.2e5
    ax_ins.plot([ins_lo, ins_hi], [ins_lo, ins_hi], 'k-',
                linewidth=0.4, alpha=0.3, zorder=1)
    ax_ins.set_xscale('log')
    ax_ins.set_yscale('log')
    ax_ins.set_xlim(ins_lo, ins_hi)
    ax_ins.set_ylim(ins_lo, ins_hi)
    ax_ins.tick_params(labelsize=0)  # hide ticks — labels provide context
    for spine in ax_ins.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#888888')

    ew_offsets = {
        r'$m_t$': (0.05, 0.05),
        r'$v$':   (0.05, -0.05),
        r'$m_H$': (-0.10, -0.04),
        r'$m_W$': (-0.10, 0.04),
        r'$m_Z$': (0.05, -0.04),
    }
    for label, pred, meas, sector in ew_masses:
        dx, dy = ew_offsets.get(label, (0.05, 0.0))
        ax_ins.annotate(label,
                        xy=(meas, pred),
                        xytext=(10**(np.log10(meas) + dx),
                                10**(np.log10(pred) + dy)),
                        fontsize=7.5, color=SECTOR_COLORS[sector],
                        ha='left' if dx > 0 else 'right', va='center',
                        arrowprops=dict(arrowstyle='-', color='#cccccc',
                                        lw=0.3, shrinkA=1, shrinkB=1))

    ax.indicate_inset_zoom(ax_ins, edgecolor='#999999', linewidth=0.5, alpha=0.5)

    # Legend
    for sec in ['fermion', 'gauge']:
        ax.scatter([], [], c=SECTOR_COLORS[sec], s=20, label=SECTOR_LABELS[sec])
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9,
              edgecolor='none', handletextpad=0.3)

    # ── Histogram panel ──
    ax_hist = fig.add_subplot(gs[1])
    _add_error_histogram(ax_hist, masses)

    path = os.path.join(OUTDIR, 'figure2a_masses.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


def figure2b_dimensionless():
    """Scorecard: dimensionless parameters (linear) with error histogram."""
    data = SCORECARD_DATA
    dimless = [(l, p, m, s) for l, p, m, s, dim in data if dim]

    fig = plt.figure(figsize=(3.4, 4.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.30)

    # ── Scatter panel ──
    ax = fig.add_subplot(gs[0])
    for label, pred, meas, sector in dimless:
        ax.scatter(meas, pred, c=SECTOR_COLORS[sector], s=35,
                   zorder=5, edgecolors='none')
    lo_d, hi_d = -0.08, 3.5
    ax.plot([lo_d, hi_d], [lo_d, hi_d], 'k-', linewidth=0.5, alpha=0.4, zorder=1)
    ax.set_xlim(lo_d, hi_d)
    ax.set_ylim(lo_d, hi_d)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title('Dimensionless parameters', fontsize=10, pad=6)
    ax.set_aspect('equal')

    # Label well-separated points directly
    direct_labels = {
        r'$\delta_{CP}$':         (0.12, -0.12),
        r'$\sin^2\!\theta_{23}$': (0.12, -0.15),
        r'$\delta_{CP}^\nu$':     (-0.35, -0.12),
    }
    for label, pred, meas, sector in dimless:
        if label in direct_labels:
            dx, dy = direct_labels[label]
            ax.annotate(label,
                        xy=(meas, pred),
                        xytext=(meas + dx, pred + dy),
                        fontsize=7.5, color=SECTOR_COLORS[sector],
                        ha='left' if dx > 0 else 'right', va='center',
                        arrowprops=dict(arrowstyle='-', color='#cccccc',
                                        lw=0.3) if abs(dx) > 0.2 else None)

    # Zoomed inset for the small-value cluster
    ax_ins = ax.inset_axes([0.30, 0.08, 0.62, 0.44])
    small_dimless = [(l, p, m, s) for l, p, m, s in dimless if m < 0.35]
    for label, pred, meas, sector in small_dimless:
        ax_ins.scatter(meas, pred, c=SECTOR_COLORS[sector], s=28,
                       zorder=5, edgecolors='none')
    ins_lo, ins_hi = -0.015, 0.34
    ax_ins.plot([ins_lo, ins_hi], [ins_lo, ins_hi], 'k-', linewidth=0.4,
                alpha=0.3, zorder=1)
    ax_ins.set_xlim(ins_lo, ins_hi)
    ax_ins.set_ylim(ins_lo, ins_hi)
    ax_ins.set_aspect('equal')
    ax_ins.tick_params(labelsize=5.5)
    ax_ins.set_xticks([0, 0.1, 0.2, 0.3])
    ax_ins.set_yticks([0, 0.1, 0.2, 0.3])
    for spine in ax_ins.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#888888')

    inset_offsets = {
        r'$\sin^2\!\theta_W$':    (0.06, -0.035),
        r'$\alpha_s$':            (0.06,  0.025),
        r'$V_{us}$':              (-0.10,  0.035),
        r'$V_{cb}$':              (0.06, -0.02),
        r'$V_{ub}$':              (0.05,  0.02),
        r'$\sin^2\!\theta_{13}$': (0.06, -0.012),
        r'$\sin^2\!\theta_{12}$': (-0.12, -0.035),
    }
    for label, pred, meas, sector in small_dimless:
        if label in inset_offsets:
            dx, dy = inset_offsets[label]
            ax_ins.annotate(label,
                            xy=(meas, pred),
                            xytext=(meas + dx, pred + dy),
                            fontsize=6.5, color=SECTOR_COLORS[sector],
                            ha='left' if dx > 0 else 'right', va='center',
                            arrowprops=dict(arrowstyle='-', color='#cccccc',
                                            lw=0.3, shrinkA=1, shrinkB=1))

    ax.indicate_inset_zoom(ax_ins, edgecolor='#999999', linewidth=0.5, alpha=0.5)

    # Legend
    for sec in ['gauge', 'ckm', 'pmns']:
        ax.scatter([], [], c=SECTOR_COLORS[sec], s=20, label=SECTOR_LABELS[sec])
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9,
              edgecolor='none', handletextpad=0.3)

    # ── Histogram panel ──
    ax_hist = fig.add_subplot(gs[1])
    _add_error_histogram(ax_hist, dimless)

    path = os.path.join(OUTDIR, 'figure2b_dimensionless.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Derivation Chain Flowchart
# ═══════════════════════════════════════════════════════════════════

def figure3_derivation_chain():
    """Horizontal flowchart: knot → α → m_e → masses → gauge → mixing."""
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.set_xlim(-1.0, 10.5)
    ax.set_ylim(-0.8, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Box definitions: (x_center, y_center, width, height, text, color)
    boxes = [
        (0.0,  0.2, 1.4, 0.8, '(2,1) knot\n$k = 3$',            '#ffcdd2'),
        (2.0,  0.2, 1.2, 0.8, r'$\alpha = r/R$',                 '#ffcdd2'),
        (3.8,  0.2, 1.2, 0.8, r'$m_e$',                           '#ffcdd2'),
        (5.6,  0.2, 1.4, 0.8, 'Koide\n9 masses',                  '#ffcdd2'),
        (7.5,  0.2, 1.3, 0.8, 'Gauge\nsector',                    '#bbdefb'),
        (9.5,  0.2, 1.4, 0.8, 'Mixing\nmatrices',                 '#c8e6c9'),
    ]

    # Draw boxes
    for cx, cy, w, h, text, color in boxes:
        box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                             boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor='#333333',
                             linewidth=0.7)
        ax.add_patch(box)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=8,
                linespacing=1.3)

    # Arrows between boxes
    arrow_pairs = [
        (0.0 + 0.7,  2.0 - 0.6),
        (2.0 + 0.6,  3.8 - 0.6),
        (3.8 + 0.6,  5.6 - 0.7),
        (5.6 + 0.7,  7.5 - 0.65),
        (7.5 + 0.65, 9.5 - 0.7),
    ]
    for x_start, x_end in arrow_pairs:
        ax.annotate('', xy=(x_end, 0.2), xytext=(x_start, 0.2),
                    arrowprops=dict(arrowstyle='->', color='#333333',
                                   lw=1.2, shrinkA=2, shrinkB=2))

    # Mechanism labels above arrows
    labels = [
        (1.0,  0.78, 'topology'),
        (2.9,  0.78, r'$m_e = \frac{\alpha^2 m_P}{2\pi k}$'),
        (4.7,  0.78, 'Koide'),
        (6.55, 0.78, 'modes'),
        (8.5,  0.78, r'Koide $S^2$'),
    ]
    for x, y, text in labels:
        ax.text(x, y, text, ha='center', va='bottom', fontsize=7,
                color='#555555', style='italic')

    # Sector color legend below
    legend_items = [
        (2.5, -0.55, C_FERMION, 'Fermion masses'),
        (5.0, -0.55, C_GAUGE, 'Gauge / Higgs'),
        (7.5, -0.55, C_CKM, 'CKM'),
        (9.5, -0.55, C_PMNS, 'PMNS'),
    ]
    for x, y, col, label in legend_items:
        ax.plot(x - 0.15, y, 's', color=col, markersize=5)
        ax.text(x, y, label, fontsize=6.5, va='center', color='#333333')

    path = os.path.join(OUTDIR, 'figure3_derivation_chain.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: Koide Generation Sphere
# ═══════════════════════════════════════════════════════════════════

def figure4_koide_sphere():
    """3D unit sphere with Koide mass vectors and Z3 symmetry."""
    fig = plt.figure(figsize=(3.4, 3.4))
    ax = fig.add_subplot(111, projection='3d')

    # Koide formula: sqrt(m_i) proportional to 1 + sqrt(2)*cos(theta_K + 2*pi*i/3)
    theta_K = (6 * np.pi + 2) / 9  # the Koide angle

    # Three mass vectors on the unit sphere
    # Polar angle from Koide: each vector's z-component
    # Represent as unit vectors with azimuthal separation of 120°
    vectors = []
    colors_v = [C_FERMION, C_GAUGE, C_CKM]
    gen_labels = [r'$\sqrt{m_1}$', r'$\sqrt{m_2}$', r'$\sqrt{m_3}$']

    for i in range(3):
        # Koide amplitude (normalized to unit sphere)
        amp = 1 + np.sqrt(2) * np.cos(theta_K + 2 * np.pi * i / 3)
        # Map to polar angle on sphere
        polar = np.arccos(amp / (1 + np.sqrt(2)))  # normalize so max is at pole
        azimuth = 2 * np.pi * i / 3  # Z3 symmetry: 120° apart

        vx = np.sin(polar) * np.cos(azimuth)
        vy = np.sin(polar) * np.sin(azimuth)
        vz = np.cos(polar)
        vectors.append((vx, vy, vz))

    # Draw transparent unit sphere
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.05, color='#888888', linewidth=0)
    ax.plot_wireframe(xs, ys, zs, alpha=0.08, color='#aaaaaa', linewidth=0.2,
                      rstride=3, cstride=3)

    # Draw equatorial circle
    eq_t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(eq_t), np.sin(eq_t), np.zeros_like(eq_t),
            color='#999999', linewidth=0.5, linestyle='--', alpha=0.5)

    # Draw the three Koide vectors
    for i, (vx, vy, vz) in enumerate(vectors):
        ax.quiver(0, 0, 0, vx, vy, vz, color=colors_v[i],
                  arrow_length_ratio=0.08, linewidth=1.5)
        ax.text(vx * 1.15, vy * 1.15, vz * 1.15, gen_labels[i],
                fontsize=9, color=colors_v[i], ha='center', va='center')
        # Project onto equatorial plane
        ax.plot([vx, vx], [vy, vy], [0, vz],
                color=colors_v[i], linewidth=0.4, linestyle=':', alpha=0.5)
        ax.scatter([vx], [vy], [0], color=colors_v[i], s=10, alpha=0.5,
                   zorder=5)

    # Z3 symmetry arc on equatorial plane
    arc_t = np.linspace(0, 2 * np.pi / 3, 30)
    r_arc = 0.35
    ax.plot(r_arc * np.cos(arc_t), r_arc * np.sin(arc_t),
            np.zeros_like(arc_t), color='#666666', linewidth=0.6)
    ax.text(0.28, 0.22, 0.0, r'$120°$', fontsize=7, color='#666666')

    # theta_K annotation
    ax.text2D(0.02, 0.95, r'$\theta_K = \frac{6\pi + 2}{9}$',
              transform=ax.transAxes, fontsize=9, color='black')

    ax.view_init(elev=20, azim=35)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_axis_off()
    ax.set_title('Koide generation sphere', fontsize=10, pad=-5)

    path = os.path.join(OUTDIR, 'figure4_koide_sphere.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: Torus Mode Structure
# ═══════════════════════════════════════════════════════════════════

def figure5_torus_modes():
    """Schematic grid of the 13 EM modes on the torus."""
    fig, ax = plt.subplots(figsize=(3.4, 4.0))
    ax.set_xlim(-0.8, 5.8)
    ax.set_ylim(-2.5, 5.8)
    ax.axis('off')

    ax.set_title('Electromagnetic modes on the torus', fontsize=10, pad=8)

    # Mode categories
    # EM modes (10 total): 4 toroidal + 1 poloidal + 4 mixed + 1 metric-like
    # Weak-coupled (3): specific modes → sin²θ_W = 3/13
    # Plus 5 metric modes - 1 zero mode = 4 additional

    C_EM = '#bbdefb'      # blue — EM modes
    C_WEAK = '#c8e6c9'    # green — weak modes
    C_METRIC = '#fff9c4'  # yellow — metric modes
    C_ZERO = '#f5f5f5'    # gray — zero mode

    cell_w, cell_h = 0.85, 0.70
    text_size = 7

    def draw_cell(x, y, label, color, sublabel=None):
        h = cell_h + (0.15 if sublabel else 0)
        box = FancyBboxPatch((x - cell_w/2, y - h/2), cell_w, h,
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='#555555',
                             linewidth=0.5)
        ax.add_patch(box)
        if sublabel:
            ax.text(x, y + 0.08, label,
                    ha='center', va='center', fontsize=text_size)
            ax.text(x, y - 0.25, sublabel, ha='center', va='center',
                    fontsize=5.5, color='#555555', style='italic')
        else:
            ax.text(x, y, label,
                    ha='center', va='center', fontsize=text_size)

    # Row 1: Toroidal modes (n=1..4, m=0) — EM
    y = 4.5
    ax.text(-0.3, y, 'Toroidal', fontsize=8, ha='right', va='center',
            fontweight='bold', color='#333333')
    for i, n in enumerate([1, 2, 3, 4]):
        draw_cell(0.8 + i * 1.1, y, f'$n={n}$\n$m=0$', C_EM)

    # Row 2: Poloidal mode (n=0, m=1) — EM
    y = 3.5
    ax.text(-0.3, y, 'Poloidal', fontsize=8, ha='right', va='center',
            fontweight='bold', color='#333333')
    draw_cell(0.8, y, '$n=0$\n$m=1$', C_EM)

    # Row 3: Mixed modes (n·m ≠ 0) — 1 EM + 3 weak
    y = 2.5
    ax.text(-0.3, y, 'Mixed', fontsize=8, ha='right', va='center',
            fontweight='bold', color='#333333')
    draw_cell(0.8, y, '$n=1$\n$m=1$', C_EM)
    # Three weak-coupled mixed modes
    for i, (n, m) in enumerate([(1, -1), (2, 1), (2, -1)]):
        draw_cell(1.9 + i * 1.1, y, f'$n={n}$\n$m={m}$', C_WEAK, 'weak')

    # Row 4: Metric modes — 5 modes
    y = 1.5
    ax.text(-0.3, y, 'Metric', fontsize=8, ha='right', va='center',
            fontweight='bold', color='#333333')
    metric_labels = [r'$g_{tt}$', r'$g_{\theta\theta}$',
                     r'$g_{\phi\phi}$', r'$g_{t\theta}$', r'$g_{t\phi}$']
    for i, ml in enumerate(metric_labels):
        draw_cell(0.8 + i * 1.1, y, ml, C_METRIC)

    # Row 5: Zero mode (subtracted)
    y = 0.5
    ax.text(-0.3, y, 'Zero', fontsize=8, ha='right', va='center',
            fontweight='bold', color='#333333')
    draw_cell(0.8, y, '$n=0$\n$m=0$', C_ZERO, r'$-1$')

    # Summary equation at bottom (two lines — matplotlib mathtext lacks \underbrace)
    ax.text(2.7, -0.9,
            r'$10\;\mathrm{(EM)} + 3\;\mathrm{(weak)} = 13\;\mathrm{modes}$',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5',
                      edgecolor='#999999', linewidth=0.5))
    ax.text(2.7, -1.6,
            r'$\sin^2\!\theta_W = 3\,/\,13 \approx 0.231$',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#e8f5e9',
                      edgecolor='#999999', linewidth=0.5))

    # Legend — horizontal row above the grid
    legend_items = [
        (C_EM, 'EM (10)'), (C_WEAK, 'Weak (3)'),
        (C_METRIC, 'Metric (5)'), (C_ZERO, 'Zero (−1)')
    ]
    legend_y = 5.4
    legend_start = 0.0
    legend_spacing = 1.45
    for i, (color, label) in enumerate(legend_items):
        bx = legend_start + i * legend_spacing
        ax.add_patch(FancyBboxPatch((bx - 0.12, legend_y - 0.1), 0.24, 0.2,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='#888888',
                                    linewidth=0.3))
        ax.text(bx + 0.22, legend_y, label, fontsize=6, va='center',
                color='#333333')

    path = os.path.join(OUTDIR, 'figure5_torus_modes.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 6: Pythagorean Resonance on the Unwrapped Torus
# ═══════════════════════════════════════════════════════════════════

def figure6_pythagorean_resonance():
    """Pythagorean resonance condition: (kp)² + q² = N² on the unwrapped torus.

    Two panels compare the meson (k=2) and baryon (k=3) sectors.
    Quarter-circles at integer radii N show where exact resonances live.
    A lattice point ON a circle means the standing wave closes perfectly;
    a point between circles means a phase defect and finite lifetime.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8))

    theta_arc = np.linspace(0, np.pi / 2, 200)

    # ── Panel (a): k=2, mode (1,1) → legs (2,1), N = √5 ≈ 2.24 ──
    ax = axes[0]
    kp, qm = 2, 1
    N_exact = np.sqrt(kp**2 + qm**2)   # √5
    N_lo = int(np.floor(N_exact))       # 2
    N_hi = int(np.ceil(N_exact))        # 3
    gmax = 4

    # Shaded annular "gap" between N=2 and N=3
    from matplotlib.patches import Polygon as MplPolygon
    theta_fill = np.linspace(0, np.pi / 2, 200)
    inner = np.column_stack([N_lo * np.cos(theta_fill), N_lo * np.sin(theta_fill)])
    outer = np.column_stack([N_hi * np.cos(theta_fill), N_hi * np.sin(theta_fill)])
    gap_verts = np.vstack([inner, outer[::-1]])
    gap_patch = MplPolygon(gap_verts, closed=True, facecolor='#ffebee',
                           edgecolor='none', alpha=0.6, zorder=0)
    ax.add_patch(gap_patch)

    # Light grid lines
    for i in range(gmax + 1):
        ax.axhline(i, color='#f0f0f0', linewidth=0.3, zorder=0)
        ax.axvline(i, color='#f0f0f0', linewidth=0.3, zorder=0)

    # Lattice points
    for i in range(gmax + 1):
        for j in range(gmax + 1):
            ax.plot(i, j, '.', color='#cccccc', markersize=2, zorder=1)

    # Quarter-circles at integer radii
    for N in range(1, gmax + 1):
        if N == N_lo or N == N_hi:
            ax.plot(N * np.cos(theta_arc), N * np.sin(theta_arc),
                    color='#ef9a9a', linewidth=0.8, linestyle='--', zorder=2)
        else:
            ax.plot(N * np.cos(theta_arc), N * np.sin(theta_arc),
                    color='#e0e0e0', linewidth=0.4, zorder=1)

    # Right triangle
    ax.plot([0, kp], [0, 0], color=C_FERMION, linewidth=1.5, zorder=3)
    ax.plot([kp, kp], [0, qm], color=C_GAUGE, linewidth=1.5, zorder=3)
    ax.plot([0, kp], [0, qm], color='#666666', linewidth=1.5,
            linestyle='--', zorder=3)

    # Right-angle marker
    sq = 0.18
    ax.plot([kp - sq, kp - sq, kp], [0, sq, sq],
            color='#999999', linewidth=0.5, zorder=3)

    # Endpoint: × (misses all circles)
    ax.plot(kp, qm, 'x', color=C_FERMION, markersize=10,
            markeredgewidth=2.0, zorder=5)

    # Leg labels
    ax.text(kp / 2, -0.18, r'$kp = 2$', ha='center', va='top',
            fontsize=9, color=C_FERMION)
    ax.text(kp + 0.18, qm / 2, r'$q = 1$', ha='left', va='center',
            fontsize=9, color=C_GAUGE)

    # Hypotenuse label
    angle_deg = np.degrees(np.arctan2(qm, kp))
    ax.text(kp / 2 - 0.15, qm / 2 + 0.18,
            r'$\sqrt{5} \approx 2.24$',
            ha='center', va='bottom', fontsize=8, color='#666666',
            rotation=angle_deg)

    # Circle radius labels
    ax.text(N_lo + 0.08, 0.15, r'$N\!=\!2$', fontsize=6.5,
            color='#ef9a9a', ha='left')
    ax.text(N_hi + 0.08, 0.15, r'$N\!=\!3$', fontsize=6.5,
            color='#ef9a9a', ha='left')

    # Defect and particle labels
    ax.text(kp + 0.22, qm + 0.22, r'$\delta = 1$', fontsize=9,
            color=C_FERMION, fontweight='bold', ha='left')
    ax.text(gmax - 0.1, gmax - 0.1, r'K$^\pm$  mode',
            fontsize=8, color='#888888', ha='right', va='top',
            style='italic')

    ax.set_xlim(-0.3, gmax + 0.3)
    ax.set_ylim(-0.5, gmax + 0.3)
    ax.set_aspect('equal')
    ax.set_xlabel(r'Toroidal  $kp$', fontsize=9)
    ax.set_ylabel(r'Poloidal  $q$', fontsize=9)
    ax.set_title(r'(a)  Meson ($k\!=\!2$): phase defect',
                 fontsize=10, pad=8)
    ax.set_xticks(range(gmax + 1))
    ax.set_yticks(range(gmax + 1))

    # ── Panel (b): k=3, mode (1,4) → legs (3,4), N = 5 exactly ──
    ax = axes[1]
    kp, qm = 3, 4
    N_exact = 5
    gmax = 6

    # Light grid lines
    for i in range(gmax + 1):
        ax.axhline(i, color='#f0f0f0', linewidth=0.3, zorder=0)
        ax.axvline(i, color='#f0f0f0', linewidth=0.3, zorder=0)

    # Lattice points
    for i in range(gmax + 1):
        for j in range(gmax + 1):
            ax.plot(i, j, '.', color='#cccccc', markersize=2, zorder=1)

    # Quarter-circles at integer radii
    for N in range(1, gmax + 1):
        if N == N_exact:
            ax.plot(N * np.cos(theta_arc), N * np.sin(theta_arc),
                    color='#a5d6a7', linewidth=1.8, zorder=2)
        else:
            ax.plot(N * np.cos(theta_arc), N * np.sin(theta_arc),
                    color='#e0e0e0', linewidth=0.4, zorder=1)

    # Right triangle
    ax.plot([0, kp], [0, 0], color=C_FERMION, linewidth=1.5, zorder=3)
    ax.plot([kp, kp], [0, qm], color=C_GAUGE, linewidth=1.5, zorder=3)
    ax.plot([0, kp], [0, qm], color=C_CKM, linewidth=2.0, zorder=3)

    # Right-angle marker
    sq = 0.22
    ax.plot([kp - sq, kp - sq, kp], [0, sq, sq],
            color='#999999', linewidth=0.5, zorder=3)

    # Endpoint: filled circle ON the N=5 circle
    ax.plot(kp, qm, 'o', color=C_CKM, markersize=8,
            markeredgecolor='white', markeredgewidth=0.5, zorder=5)

    # Leg labels
    ax.text(kp / 2, -0.18, r'$kp = 3$', ha='center', va='top',
            fontsize=9, color=C_FERMION)
    ax.text(kp + 0.22, qm / 2, r'$q = 4$', ha='left', va='center',
            fontsize=9, color=C_GAUGE)

    # Hypotenuse label
    angle_deg = np.degrees(np.arctan2(qm, kp))
    ax.text(kp / 2 - 0.35, qm / 2 + 0.4, r'$N = 5$',
            ha='center', va='bottom', fontsize=10, color=C_CKM,
            fontweight='bold', rotation=angle_deg)

    # Circle label
    ax.text(N_exact + 0.12, 0.25, r'$N\!=\!5$', fontsize=7,
            color='#2e7d32', ha='left')

    # Defect and particle labels
    ax.text(kp + 0.22, qm + 0.3, r'$\delta = 0$', fontsize=9,
            color=C_CKM, fontweight='bold', ha='left')
    ax.text(gmax - 0.1, gmax - 0.1,
            r'proton: $(3,4,5)$',
            fontsize=8, color='#888888', ha='right', va='top',
            style='italic')

    ax.set_xlim(-0.3, gmax + 0.3)
    ax.set_ylim(-0.5, gmax + 0.3)
    ax.set_aspect('equal')
    ax.set_xlabel(r'Toroidal  $kp$', fontsize=9)
    ax.set_ylabel(r'Poloidal  $q$', fontsize=9)
    ax.set_title(r'(b)  Baryon ($k\!=\!3$): exact resonance',
                 fontsize=10, pad=8)
    ax.set_xticks(range(gmax + 1))
    ax.set_yticks(range(gmax + 1))

    # ── Bottom equation ──
    fig.text(0.5, -0.04,
             r'Resonance condition:  $(kp)^2 + q^2 = N^2$'
             r'$\quad\longrightarrow\quad$'
             r'integer $N$ $=$ exact closure (stable)',
             ha='center', fontsize=9)

    fig.tight_layout(w_pad=2.5)

    path = os.path.join(OUTDIR, 'figure6_pythagorean_resonance.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

FIGURES = {
    '1':  ('Figure 1: Torus knot',          figure1_torus_knot),
    '2a': ('Figure 2a: Masses',             figure2a_masses),
    '2b': ('Figure 2b: Dimensionless',      figure2b_dimensionless),
    '3':  ('Figure 3: Derivation chain',    figure3_derivation_chain),
    '4':  ('Figure 4: Koide sphere',        figure4_koide_sphere),
    '5':  ('Figure 5: Torus modes',         figure5_torus_modes),
    '6':  ('Figure 6: Pythagorean resonance', figure6_pythagorean_resonance),
}

if __name__ == '__main__':
    which = sys.argv[1:] if len(sys.argv) > 1 else FIGURES.keys()
    for key in which:
        if key in FIGURES:
            name, func = FIGURES[key]
            print(f"Generating {name}...")
            func()
        else:
            print(f"Unknown figure: {key}")
    print("Done.")
