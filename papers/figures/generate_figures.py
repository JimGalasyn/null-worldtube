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

def _draw_torus_knot_panel(ax, R, r, p, q, cmap, color_lo, color_hi,
                           gap_frac=0.0, N=2000, lw=2.0,
                           show_annotations=False, show_arrows=False):
    """Draw a (p,q) torus knot on a 3D axes with optional gap and annotations.

    gap_frac: fraction of 2π to leave as a gap (0 = closed, 0.04 = 4% gap).
    """
    torus_color = C_TORUS

    # Knot curve (with optional gap)
    if gap_frac > 0:
        lam = np.linspace(gap_frac * 2 * np.pi, (1 - gap_frac) * 2 * np.pi, N)
    else:
        lam = np.linspace(0, 2 * np.pi, N)
    theta = p * lam
    phi = q * lam
    x_k = (R + r * np.cos(phi)) * np.cos(theta)
    y_k = (R + r * np.cos(phi)) * np.sin(theta)
    z_k = r * np.sin(phi)

    # Color gradient
    colors = cmap(np.linspace(color_lo, color_hi, N))
    for i in range(N - 1):
        ax.plot([x_k[i], x_k[i+1]], [y_k[i], y_k[i+1]], [z_k[i], z_k[i+1]],
                color=colors[i], linewidth=lw, alpha=0.9)

    # Gap endpoint markers
    if gap_frac > 0:
        ax.plot([x_k[0]], [y_k[0]], [z_k[0]], 'o', color=colors[0],
                markersize=4, zorder=5)
        ax.plot([x_k[-1]], [y_k[-1]], [z_k[-1]], 'o', color=colors[-1],
                markersize=4, zorder=5)

    # Torus wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 30)
    U, V = np.meshgrid(u, v)
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    ax.plot_surface(X, Y, Z, alpha=0.04, color=torus_color, linewidth=0)
    ax.plot_wireframe(X, Y, Z, alpha=0.12, color=torus_color, linewidth=0.25,
                      rstride=3, cstride=3)

    # (R, r annotations now handled by 2D cross-section inset)

    if show_arrows:
        arrow_indices = np.linspace(150, N - 150, 5, dtype=int)
        for idx in arrow_indices:
            step = 20
            dx = x_k[idx+step] - x_k[idx-step]
            dy = y_k[idx+step] - y_k[idx-step]
            dz = z_k[idx+step] - z_k[idx-step]
            norm = np.sqrt(dx**2 + dy**2 + dz**2)
            ax.quiver(x_k[idx], y_k[idx], z_k[idx],
                      dx/norm, dy/norm, dz/norm,
                      length=0.22, color='black', arrow_length_ratio=0.45,
                      linewidth=1.2)

    return x_k, y_k, z_k


def figure1_torus_knot():
    """Multi-panel figure: electron, cross-section, K± phase defect, proton."""
    fig = plt.figure(figsize=(7.0, 5.8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[4, 2], hspace=0.06, wspace=0.02)

    # ── (a) Electron (2,1) — top left ──
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')
    _draw_torus_knot_panel(ax_a, R=1.0, r=0.25, p=2, q=1,
                           cmap=plt.cm.Reds, color_lo=0.35, color_hi=0.95,
                           show_arrows=True)
    ax_a.view_init(elev=25, azim=30)
    ax_a.set_xlim(-1.15, 1.15)
    ax_a.set_ylim(-1.15, 1.15)
    ax_a.set_zlim(-0.32, 0.32)
    ax_a.set_box_aspect([1, 1, 0.28])
    ax_a.set_axis_off()
    ax_a.set_title(r'(a)  Electron $(2,1)$', fontsize=9, pad=-8)

    # ── (b) K± meson — phase defect — top right ──
    ax_b = fig.add_subplot(gs[0, 1], projection='3d')
    _draw_torus_knot_panel(ax_b, R=1.0, r=0.5, p=2, q=1,
                           cmap=plt.cm.Oranges, color_lo=0.25, color_hi=0.90,
                           gap_frac=0.06, lw=2.2)
    ax_b.view_init(elev=20, azim=220)
    bnd = 1.55
    ax_b.set_xlim(-bnd, bnd)
    ax_b.set_ylim(-bnd, bnd)
    ax_b.set_zlim(-0.65, 0.65)
    ax_b.set_box_aspect([1, 1, 0.42])
    ax_b.set_axis_off()
    ax_b.set_title(r'(b)  K$^\pm$ — phase defect', fontsize=9, pad=-8)

    # ── Cross-section diagram — bottom left ──
    ax_xs = fig.add_subplot(gs[1, 0])
    R_d, r_d = 1.0, 0.35  # exaggerated r for clarity

    # Symmetry axis (dashed vertical line)
    ax_xs.plot([0, 0], [-0.6, 0.6], color='#aaaaaa', linewidth=0.5,
               linestyle='--', zorder=0)

    # Two tube cross-sections
    for sign in [-1, 1]:
        circle = plt.Circle((sign * R_d, 0), r_d, fill=True,
                             facecolor=C_TORUS, alpha=0.10,
                             edgecolor=C_TORUS, linewidth=1.2, zorder=2)
        ax_xs.add_patch(circle)

    # R arrow: origin to right circle center
    ax_xs.annotate('', xy=(R_d - 0.02, 0), xytext=(0.02, 0),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=0.8))
    ax_xs.text(R_d / 2, 0.10, r'$R$', fontsize=9, ha='center', va='bottom',
               color='black')

    # r arrow: right circle center upward to edge
    ax_xs.annotate('', xy=(R_d, r_d - 0.02), xytext=(R_d, 0.02),
                   arrowprops=dict(arrowstyle='<->', color=C_FERMION, lw=0.8))
    ax_xs.text(R_d + 0.10, r_d / 2, r'$r$', fontsize=9, ha='left',
               va='center', color=C_FERMION)

    # Center marker
    ax_xs.plot(0, 0, '+', color='#999999', markersize=6, markeredgewidth=0.6)

    # α = r/R label
    ax_xs.text(0.0, -0.58, r'$\alpha = r/R$', fontsize=9, ha='center',
               va='top', color='#444444')

    ax_xs.set_xlim(-1.65, 1.65)
    ax_xs.set_ylim(-0.70, 0.60)
    ax_xs.set_aspect('equal')
    ax_xs.axis('off')
    ax_xs.set_title('Meridional cross-section', fontsize=8, color='#666666',
                     pad=4)

    # ── (c) Proton — (3,4,5) resonance — bottom right ──
    ax_c = fig.add_subplot(gs[1, 1], projection='3d')
    _draw_torus_knot_panel(ax_c, R=1.0, r=1/3, p=3, q=4,
                           cmap=plt.cm.Greens, color_lo=0.30, color_hi=0.90,
                           gap_frac=0.0, lw=1.8)
    ax_c.view_init(elev=25, azim=30)
    bnd_c = 1.45
    ax_c.set_xlim(-bnd_c, bnd_c)
    ax_c.set_ylim(-bnd_c, bnd_c)
    ax_c.set_zlim(-0.5, 0.5)
    ax_c.set_box_aspect([1, 1, 0.35])
    ax_c.set_axis_off()
    ax_c.set_title(r'(c)  Proton — $(3,4,5)$ resonance', fontsize=9, pad=-8)
    ax_c.text2D(0.50, 0.02, r'$3^2+4^2=5^2$', transform=ax_c.transAxes,
                fontsize=8, ha='center', color='#2e7d32')

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
    stacked_errors = []
    stacked_colors = []
    stacked_labels = []
    for sec in present_sectors:
        sec_errors = [abs(p - m) / m * 100
                      for _, p, m, s in subset_data if s == sec and m != 0]
        if sec_errors:
            stacked_errors.append(sec_errors)
            stacked_colors.append(SECTOR_COLORS[sec])
            stacked_labels.append(SECTOR_LABELS[sec])
    if stacked_errors:
        ax.hist(stacked_errors, bins=bins, color=stacked_colors,
                label=stacked_labels, stacked=True,
                edgecolor='white', linewidth=0.3)
    if errors:
        median_err = np.median(errors)
        ax.axvline(median_err, color='black', linewidth=1.0, linestyle='--',
                   label=f'Median: {median_err:.1f}%')
    ax.set_xlabel('|Error| (%)')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 10)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
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
    # Force matching ticks on both axes
    from matplotlib.ticker import LogLocator, NullFormatter
    loc = LogLocator(base=10, numticks=10)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Label well-separated points directly
    ew_labels = {r'$m_t$', r'$v$', r'$m_H$', r'$m_W$', r'$m_Z$'}
    offsets_mass = {
        r'$m_u$':    (-0.15, 0.0),
        r'$m_d$':    (0.10, -0.18),
        r'$m_s$':    (0.10, 0.08),
        r'$m_c$':    (-0.15, 0.06),
        r'$m_\tau$': (0.15, 0.0),
        r'$m_b$':    (0.15, 0.0),
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
    ax_ins = ax.inset_axes([0.35, 0.12, 0.48, 0.38])
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
    ax_ins.tick_params(which='both', labelsize=5.5)
    ax_ins.set_xticks([1e5, 2e5, 3e5])
    ax_ins.set_yticks([1e5, 2e5, 3e5])
    ax_ins.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_ins.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for spine in ax_ins.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#888888')

    ew_offsets = {
        r'$m_t$': (0.04, 0.0),
        r'$v$':   (0.05, -0.05),
        r'$m_H$': (-0.04, 0.0),
        r'$m_W$': (-0.01, 0.06),
        r'$m_Z$': (0.04, 0.0),
    }
    no_callout_inset = {r'$m_H$', r'$m_W$'}
    for label, pred, meas, sector in ew_masses:
        dx, dy = ew_offsets.get(label, (0.05, 0.0))
        kwargs = dict(fontsize=7.5, color=SECTOR_COLORS[sector],
                      ha='left' if dx > 0 else 'right', va='center')
        if label not in no_callout_inset:
            kwargs['arrowprops'] = dict(arrowstyle='-', color='#cccccc',
                                        lw=0.3, shrinkA=1, shrinkB=1)
        ax_ins.annotate(label,
                        xy=(meas, pred),
                        xytext=(10**(np.log10(meas) + dx),
                                10**(np.log10(pred) + dy)),
                        **kwargs)

    rect, connectors = ax.indicate_inset_zoom(ax_ins, edgecolor='#999999', linewidth=0.5, alpha=0.5)
    for conn in connectors:
        if conn is not None:
            conn.set_linewidth(0.5)

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
    # Force matching ticks on both axes
    ticks_d = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    ax.set_xticks(ticks_d)
    ax.set_yticks(ticks_d)

    # Label well-separated points directly
    direct_labels = {
        r'$\delta_{CP}$':         (-0.12, 0.12),
        r'$\sin^2\!\theta_{23}$': (-0.15, 0.30),
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
                                        lw=0.3, shrinkA=1, shrinkB=1)
                        if abs(dx) + abs(dy) > 0.2 else None)

    # Zoomed inset for the small-value cluster
    ax_ins = ax.inset_axes([0.38, 0.08, 0.60, 0.44])
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
        r'$\sin^2\!\theta_W$':    (0.04, -0.03),
        r'$\alpha_s$':            (0.06,  0.0),
        r'$V_{us}$':              (-0.10,  0.05),
        r'$V_{cb}$':              (0.06,  0.0),
        r'$V_{ub}$':              (-0.06,  0.03),
        r'$\sin^2\!\theta_{13}$': (0.06, -0.02),
        r'$\sin^2\!\theta_{12}$': (-0.06, -0.02),
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

    rect, connectors = ax.indicate_inset_zoom(ax_ins, edgecolor='#999999', linewidth=0.5, alpha=0.5)
    for conn in connectors:
        if conn is not None:
            conn.set_linewidth(0.5)

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

    # Draw the three Koide vectors as line segments
    for i, (vx, vy, vz) in enumerate(vectors):
        ax.plot([0, vx], [0, vy], [0, vz],
                color=colors_v[i], linewidth=2, solid_capstyle='round')
        label_r = 1.30 if i < 2 else 1.15  # push m1, m2 labels further out
        lx, ly, lz = vx * label_r, vy * label_r, vz * label_r
        if i == 0:  # m1: nudge down and away from arrow
            lz -= 0.15
        if i == 1:  # m2: nudge upper-right
            lx += 0.08
            lz += 0.08
        ax.text(lx, ly, lz, gen_labels[i],
                fontsize=9, color=colors_v[i], ha='center', va='center')
        # Project onto equatorial plane
        ax.plot([vx, vx], [vy, vy], [0, vz],
                color=colors_v[i], linewidth=0.4, linestyle=':', alpha=0.5)
        ax.scatter([vx], [vy], [0], color=colors_v[i], s=10, alpha=0.5,
                   zorder=5)

    # Radial lines from center to equatorial projections (for m2, m3)
    for i in [1, 2]:  # m2 (blue), m3 (green)
        vx, vy, vz = vectors[i]
        ax.plot([0, vx], [0, vy], [0, 0],
                color=colors_v[i], linewidth=0.5, linestyle=':', alpha=0.5)

    # Z3 symmetry arc on equatorial plane
    arc_t = np.linspace(0, 2 * np.pi / 3, 30)
    r_arc = 0.35
    ax.plot(r_arc * np.cos(arc_t), r_arc * np.sin(arc_t),
            np.zeros_like(arc_t), color='#666666', linewidth=0.6)
    # Place 120° label at midpoint of arc, pushed outward
    arc_mid = len(arc_t) // 2
    ax.text(r_arc * np.cos(arc_t[arc_mid]) * 1.3 - 0.18,
            r_arc * np.sin(arc_t[arc_mid]) * 1.3,
            -0.04, r'$120°$', fontsize=7, color='#666666')

    # theta_K: dashed z-axis and arc from z-axis to m3 vector
    ax.plot([0, 0], [0, 0], [0, 1.1], color='#888888', linewidth=0.5,
            linestyle='--', alpha=0.6)
    # Arc from z-axis to m3 in the plane containing both
    vx3, vy3, vz3 = vectors[2]
    polar3 = np.arccos(vz3)  # polar angle of m3
    azimuth3 = np.arctan2(vy3, vx3)
    r_tk = 0.45  # arc radius
    arc_polar = np.linspace(0, polar3, 30)
    arc_x = r_tk * np.sin(arc_polar) * np.cos(azimuth3)
    arc_y = r_tk * np.sin(arc_polar) * np.sin(azimuth3)
    arc_z = r_tk * np.cos(arc_polar)
    ax.plot(arc_x, arc_y, arc_z, color=C_CKM, linewidth=0.6)
    # Label at arc midpoint
    mid = len(arc_polar) // 2
    ax.text(arc_x[mid] + 0.06, arc_y[mid] + 0.06, arc_z[mid],
            r'$\theta_K$', fontsize=8, color=C_CKM)
    # Keep the equation in the corner
    ax.text2D(0.02, 0.95, r'$\theta_K = \frac{6\pi + 2}{9}$',
              transform=ax.transAxes, fontsize=9, color='black')

    ax.view_init(elev=20, azim=330)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_box_aspect([1, 1, 1])
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
    """Schematic grid of the 13 EM modes on the torus.

    Mode counting (from supplementary S8.1):
      p²=4 toroidal + q²=1 poloidal + p²q²=4 mixed + (p²+q²)=5 metric − 1 zero = 13
    EM (10) = 4 toroidal + 1 poloidal + 4 mixed + 1 metric
    Weak (3) = 3 metric modes that couple to poloidal (weak isospin) sector
    Zero (−1) = gauge d.o.f. subtracted from metric
    """
    fig, ax = plt.subplots(figsize=(3.4, 5.2))
    ax.set_xlim(-1.5, 6.0)
    ax.set_ylim(-1.8, 6.8)
    ax.axis('off')

    ax.set_title('Electromagnetic modes on the torus', fontsize=10, pad=8)

    C_EM = '#bbdefb'      # blue — EM modes
    C_WEAK = '#c8e6c9'    # green — weak modes
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

    # Row 1: Toroidal modes (p²=4, m=0) — all EM
    y = 5.5
    ax.text(-0.3, y, 'Toroidal', fontsize=8, ha='right', va='center',
            fontweight='bold', color='#333333')
    ax.text(-0.3, y - 0.22, r'$p^2\!=\!4$', fontsize=6, ha='right',
            va='center', color='#888888')
    for i, n in enumerate([1, 2, 3, 4]):
        draw_cell(0.8 + i * 1.1, y, f'$n={n}$\n$m=0$', C_EM)

    # Row 2: Poloidal mode (q²=1) — EM
    y = 4.5
    ax.text(-0.3, y, 'Poloidal', fontsize=8, ha='right', va='center',
            fontweight='bold', color='#333333')
    ax.text(-0.3, y - 0.22, r'$q^2\!=\!1$', fontsize=6, ha='right',
            va='center', color='#888888')
    draw_cell(0.8, y, '$n=0$\n$m=1$', C_EM)

    # Row 3: Mixed modes (p²q²=4) — all EM
    y = 3.5
    ax.text(-0.3, y, 'Mixed', fontsize=8, ha='right', va='center',
            fontweight='bold', color='#333333')
    ax.text(-0.3, y - 0.22, r'$p^2 q^2\!=\!4$', fontsize=6, ha='right',
            va='center', color='#888888')
    for i, (n, m) in enumerate([(1, 1), (1, -1), (2, 1), (2, -1)]):
        draw_cell(0.8 + i * 1.1, y, f'$n={n}$\n$m={m}$', C_EM)

    # Row 4: Knot-metric modes (p²+q²=5) — 1 EM + 3 weak + 1 zero
    y = 2.5
    ax.text(-0.3, y, 'Metric', fontsize=8, ha='right', va='center',
            fontweight='bold', color='#333333')
    ax.text(-0.3, y - 0.22, r'$p^2\!+\!q^2\!=\!5$', fontsize=6, ha='right',
            va='center', color='#888888')
    for i, (label, color, sub) in enumerate([
            ('$g_1$', C_EM, r'$\gamma$'), ('$g_2$', C_WEAK, r'$W^+$'),
            ('$g_3$', C_WEAK, r'$W^-$'), ('$g_4$', C_WEAK, r'$Z$'),
            ('$g_0$', C_ZERO, r'$-1$')]):
        draw_cell(0.8 + i * 1.0, y, label, color, sub)

    # Summary equation at bottom
    ax.text(2.1, 1.0,
            r'$10\;\mathrm{(EM)} + 3\;\mathrm{(weak)} = 13\;\mathrm{modes}$',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5',
                      edgecolor='#999999', linewidth=0.5))
    ax.text(2.1, 0.1,
            r'$\sin^2\!\theta_W = 3\,/\,13 \approx 0.231$',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#e8f5e9',
                      edgecolor='#999999', linewidth=0.5))

    # Tally annotations on right side (count + particle)
    ax.text(5.6, 5.5, r'4 $\gamma$', fontsize=8, ha='center', va='center',
            color=C_GAUGE, fontweight='bold')
    ax.text(5.6, 4.5, r'1 $\gamma$', fontsize=8, ha='center', va='center',
            color=C_GAUGE, fontweight='bold')
    ax.text(5.6, 3.5, r'4 $\gamma$', fontsize=8, ha='center', va='center',
            color=C_GAUGE, fontweight='bold')
    ax.text(5.6, 2.5, '5', fontsize=8, ha='center', va='center',
            color='#666666', fontweight='bold')

    # Legend — horizontal row above the grid
    legend_items = [
        (C_EM, 'EM (10)'), (C_WEAK, 'Weak (3)'),
        (C_ZERO, 'Zero (−1)')
    ]
    legend_y = 6.4
    legend_spacing = 1.45
    legend_start = 2.1 - (len(legend_items) - 1) * legend_spacing / 2
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
