#!/usr/bin/env python3
"""
Animation: hydrogen photoionization in the null worldtube model.

Narrative in 5 phases:
  1. BOUND (60 fr)      — electron torus orbits proton, circulating photon visible
  2. PHOTON APPROACH (75 fr) — gold EM wave packet enters from left
  3. ABSORPTION (45 fr)  — flash/glow, torus pulses, wave fades, energy transfers
  4. EJECTION (90 fr)    — torus exits on escape trajectory with dotted trail
  5. HOLD (30 fr)        — bare proton remains, summary text

300 frames at 25 fps = 12 seconds.  Outputs GIF via pillow.

Usage:
    python -m simulations.nwt.anim_ionization
    python -m simulations.nwt --anim-ionization
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

from .atoms import compute_hydrogen_orbital, compute_photoionization_hydrogen
from .constants import alpha, a_0, c, hbar, m_e, eV, lambda_C, TorusParams
from .core import find_self_consistent_radius

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Phase durations (frames) ────────────────────────────────────────
N_BOUND    = 60
N_APPROACH = 80
N_ABSORB   = 15     # quick flash — absorption is near-instantaneous
N_EJECT    = 115
N_HOLD     = 30
N_FRAMES   = N_BOUND + N_APPROACH + N_ABSORB + N_EJECT + N_HOLD  # 300
PHASE_ENDS = np.cumsum([N_BOUND, N_APPROACH, N_ABSORB, N_EJECT, N_HOLD])
PHASE_NAMES = ['BOUND', 'PHOTON APPROACH', 'ABSORPTION', 'EJECTION', 'HOLD']
FPS = 25

# ── Visual parameters ───────────────────────────────────────────────
R_ORBIT  = 3.0       # visual Bohr orbit radius (units)
R_TORUS  = 0.3       # visual torus major radius (~10x exaggerated)
r_TORUS  = 0.10      # visual torus minor radius

CLR_BG       = '#060612'
CLR_PROTON   = '#e74c3c'
CLR_ELECTRON = '#3498db'
CLR_E_GLOW   = '#85c1e9'
CLR_PHOTON_C = 'white'        # circulating photon
CLR_PHOTON_W = '#f1c40f'      # incoming wave
CLR_FLASH    = '#ffffaa'
CLR_ORBIT    = '#2ecc71'
CLR_TRAIL    = '#3498db'
CLR_INFO     = '#a0a0a0'
CLR_SIGMA    = '#e67e22'
CLR_MARKER   = '#e74c3c'
PHASE_CLRS   = ['#3498db', '#f1c40f', '#e74c3c', '#2ecc71', '#95a5a6']


# ── Geometry helpers ─────────────────────────────────────────────────

def torus_knot_2d(cx, cy, R, r, phase_offset=0.0, N=200):
    """(2,1) torus knot projection to 2D (figure-8 curve)."""
    t = np.linspace(0, 2 * np.pi, N)
    x = cx + (R + r * np.cos(t)) * np.cos(2 * t + phase_offset)
    y = cy + (R + r * np.cos(t)) * np.sin(2 * t + phase_offset)
    return x, y


def photon_on_knot_2d(cx, cy, R, r, phase, phase_offset=0.0):
    """Position of circulating photon on the (2,1) knot."""
    t = phase % (2 * np.pi)
    x = cx + (R + r * np.cos(t)) * np.cos(2 * t + phase_offset)
    y = cy + (R + r * np.cos(t)) * np.sin(2 * t + phase_offset)
    return x, y


def em_wavefronts(x_offset, y_span=4.5, wavelength=3.0, n_lines=8,
                  envelope_width=6.0):
    """
    Broad EM plane-wave: vertical wavefronts sweeping left-to-right.

    The photon wavelength (91 nm) is ~1700x the Bohr radius, so the wave
    is enormous compared to the atom — like ocean swells passing a cork.
    Returns list of (x_array, y_array, alpha) for each wavefront line.
    """
    fronts = []
    for i in range(n_lines):
        x_pos = x_offset + i * wavelength
        # Gaussian envelope centred on the group
        centre = x_offset + n_lines * wavelength / 2
        dist = abs(x_pos - centre)
        env = np.exp(-dist**2 / (2 * (envelope_width / 3)**2))
        y = np.linspace(-y_span, y_span, 80)
        x = np.full_like(y, x_pos)
        # Slight sinusoidal ripple so it reads as a wave, not bars
        x = x + 0.15 * np.sin(2 * np.pi * y / 3.0)
        fronts.append((x, y, env))
    return fronts


# ── Build animation ─────────────────────────────────────────────────

def run_animation():
    """Compute physics, build figure, render GIF."""

    # ── Physics data ────────────────────────────────────────────────
    print("Computing hydrogen orbital + photoionization data...")
    orb   = compute_hydrogen_orbital(Z=1, n=1)
    photo = compute_photoionization_hydrogen(Z=1)

    E_ion_eV      = photo['E_ion_eV']
    sigma_thresh  = photo['sigma_threshold_cm2']
    r_orbit_pm    = orb['r_n_pm']
    v_over_c      = orb['v_over_c']
    sigma_cm2     = photo['sigma_S_cm2']

    # ── Figure layout ───────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), facecolor=CLR_BG)
    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[3, 1], height_ratios=[9, 1],
                  hspace=0.08, wspace=0.08,
                  left=0.04, right=0.96, bottom=0.04, top=0.92)

    ax_main = fig.add_subplot(gs[0, 0])

    gs_side = gs[0, 1].subgridspec(2, 1, hspace=0.35)
    ax_sigma  = fig.add_subplot(gs_side[0])
    ax_energy = fig.add_subplot(gs_side[1])

    ax_timeline = fig.add_subplot(gs[1, :])

    fig.suptitle('Hydrogen Photoionization  \u2014  Null Worldtube Model',
                 color='white', fontsize=15, fontweight='bold', y=0.97)

    # ── Cross-section panel (static background) ─────────────────────
    ax_sigma.loglog(photo['E_photon_eV'], sigma_cm2,
                    color=CLR_SIGMA, linewidth=1.5)
    ax_sigma.set_facecolor(CLR_BG)
    ax_sigma.set_xlabel('E (eV)', color=CLR_INFO, fontsize=8)
    ax_sigma.set_ylabel('\u03c3 (cm\u00b2)', color=CLR_INFO, fontsize=8)
    ax_sigma.set_title('Cross-section \u03c3(E)', color='white',
                       fontsize=9, fontweight='bold')
    ax_sigma.tick_params(colors=CLR_INFO, labelsize=7)
    for sp in ax_sigma.spines.values():
        sp.set_color('#1a1a3a')
    ax_sigma.axhline(sigma_thresh, color=CLR_INFO, linewidth=0.5,
                     linestyle=':', alpha=0.5)
    ax_sigma.text(photo['E_photon_eV'][-1] * 0.5, sigma_thresh * 2.0,
                  f'\u03c3\u2080 = {sigma_thresh:.1e} cm\u00b2',
                  color=CLR_INFO, fontsize=7, ha='right')

    sigma_marker, = ax_sigma.plot([], [], 'o', color=CLR_MARKER, markersize=8,
                                  markeredgecolor='white', markeredgewidth=1,
                                  zorder=10)

    # ── Energy-level panel (static background) ──────────────────────
    ax_energy.set_facecolor(CLR_BG)
    ax_energy.set_xlim(0, 1)
    ax_energy.set_ylim(-16, 5)
    ax_energy.set_axis_off()
    ax_energy.set_title('Energy levels', color='white',
                        fontsize=9, fontweight='bold')
    # n=1 level
    ax_energy.plot([0.15, 0.85], [-13.6, -13.6], color=CLR_ELECTRON,
                   linewidth=2)
    ax_energy.text(0.90, -13.6, 'n=1\n\u221213.6 eV', color=CLR_ELECTRON,
                   fontsize=7, va='center')
    # Continuum edge
    ax_energy.plot([0.15, 0.85], [0, 0], color=CLR_ORBIT, linewidth=2)
    ax_energy.text(0.90, 0, 'E=0\n(free)', color=CLR_ORBIT,
                   fontsize=7, va='center')
    # Continuum hatching
    for yh in np.arange(0.5, 5, 0.5):
        ax_energy.plot([0.15, 0.85], [yh, yh], color=CLR_ORBIT,
                       linewidth=0.3, alpha=0.3)

    energy_marker, = ax_energy.plot([], [], 's', color=CLR_PHOTON_W,
                                    markersize=10, markeredgecolor='white',
                                    markeredgewidth=1, zorder=10)
    energy_arrow, = ax_energy.plot([], [], '-', color=CLR_PHOTON_W,
                                   linewidth=2, alpha=0.6)

    # ── Timeline bar ────────────────────────────────────────────────
    ax_timeline.set_facecolor(CLR_BG)
    ax_timeline.set_xlim(0, N_FRAMES)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_axis_off()

    prev = 0
    for end, name, clr in zip(PHASE_ENDS, PHASE_NAMES, PHASE_CLRS):
        ax_timeline.axvspan(prev, end, alpha=0.3, color=clr)
        ax_timeline.text((prev + end) / 2, 0.5, name, color='white',
                         fontsize=8, ha='center', va='center',
                         fontweight='bold')
        prev = end

    tl_cursor, = ax_timeline.plot([], [], '|', color='white',
                                  markersize=20, markeredgewidth=2)

    # ── Phase helper ────────────────────────────────────────────────
    def get_phase(frame):
        """Return (phase_index, local t in [0,1])."""
        if frame < PHASE_ENDS[0]:
            return 0, frame / N_BOUND
        elif frame < PHASE_ENDS[1]:
            return 1, (frame - PHASE_ENDS[0]) / N_APPROACH
        elif frame < PHASE_ENDS[2]:
            return 2, (frame - PHASE_ENDS[1]) / N_ABSORB
        elif frame < PHASE_ENDS[3]:
            return 3, (frame - PHASE_ENDS[2]) / N_EJECT
        else:
            return 4, (frame - PHASE_ENDS[3]) / N_HOLD

    # ── Frozen electron position at moment of absorption ──────────
    # orbit_angle at the frame where phase 2 starts
    _abs_frame = PHASE_ENDS[1]
    _abs_circ = _abs_frame * 0.4
    _abs_orbit = _abs_circ * 0.15
    _abs_ex = R_ORBIT * np.cos(_abs_orbit)
    _abs_ey = R_ORBIT * np.sin(_abs_orbit)
    _eject_angle = np.arctan2(_abs_ey, _abs_ex)  # radially outward

    # ── Per-frame draw ──────────────────────────────────────────────
    def draw_frame(frame):
        ax_main.cla()
        ax_main.set_facecolor(CLR_BG)
        ax_main.set_xlim(-8, 8)
        ax_main.set_ylim(-5, 5)
        ax_main.set_aspect('equal')
        ax_main.set_axis_off()

        phase, t = get_phase(frame)
        circ = frame * 0.4            # circulating-photon speed
        orbit_angle = circ * 0.15     # slow orbital revolution

        # Phase label
        ax_main.text(0, 4.5, PHASE_NAMES[phase], color=PHASE_CLRS[phase],
                     fontsize=14, ha='center', fontweight='bold',
                     fontfamily='monospace')

        # ── PHASE 0: BOUND ─────────────────────────────────────────
        if phase == 0:
            _draw_orbit(ax_main)
            ex = R_ORBIT * np.cos(orbit_angle)
            ey = R_ORBIT * np.sin(orbit_angle)
            _draw_torus(ax_main, ex, ey, orbit_angle, circ)

            ax_main.text(0, -4.3,
                         f'n=1  r = {r_orbit_pm:.1f} pm  '
                         f'E = \u2212{E_ion_eV:.1f} eV  '
                         f'v/c = {v_over_c:.4f}',
                         color=CLR_INFO, fontsize=9, ha='center',
                         fontfamily='monospace')
            ax_main.text(6.5, -4.3, '(not to scale)', color=CLR_INFO,
                         fontsize=7, ha='right', style='italic')

            energy_marker.set_data([0.5], [-13.6])
            energy_arrow.set_data([], [])
            sigma_marker.set_data([], [])

        # ── PHASE 1: PHOTON APPROACH ───────────────────────────────
        elif phase == 1:
            _draw_orbit(ax_main)
            ex = R_ORBIT * np.cos(orbit_angle)
            ey = R_ORBIT * np.sin(orbit_angle)
            _draw_torus(ax_main, ex, ey, orbit_angle, circ)

            # Broad EM wavefronts sweeping left to right.
            # Time the envelope center to reach the electron at t=1.
            _wl, _nf = 2.8, 10
            _env_half = _nf * _wl / 2          # 14 units
            _wave_end = _abs_ex - _env_half     # center hits electron
            _wave_start = -30.0                 # well off-screen left
            wave_x0 = _wave_start + (_wave_end - _wave_start) * t
            fronts = em_wavefronts(wave_x0, y_span=4.8,
                                   wavelength=_wl, n_lines=_nf,
                                   envelope_width=8.0)
            for fx, fy, env in fronts:
                if env > 0.05:
                    ax_main.plot(fx, fy, color=CLR_PHOTON_W,
                                 linewidth=1.5, alpha=0.7 * env)
            ax_main.text(-6.5, 4.2,
                         '\u03b3 (13.6 eV)   \u03bb \u226b r_atom',
                         color=CLR_PHOTON_W, fontsize=9,
                         fontweight='bold')

            # Glow begins as wavefronts reach the electron (last ~20%)
            if t > 0.8:
                glow_t = (t - 0.8) / 0.2
                glow_a = 0.35 * glow_t
                circle = plt.Circle((_abs_ex, _abs_ey),
                                    0.5 + glow_t * 1.0,
                                    color=CLR_FLASH, alpha=glow_a,
                                    fill=True, zorder=5)
                ax_main.add_patch(circle)

            ax_main.text(0, -4.3,
                         f'\u03c9_photon/\u03c9_orbit = 1/2  (2:1 resonance)   '
                         f'\u03c3 = {sigma_thresh:.1e} cm\u00b2',
                         color=CLR_INFO, fontsize=9, ha='center',
                         fontfamily='monospace')

            e_top = -13.6 + 13.6 * t
            energy_marker.set_data([0.5], [-13.6])
            energy_arrow.set_data([0.5, 0.5], [-13.6, e_top])
            sigma_marker.set_data([E_ion_eV], [sigma_thresh])

        # ── PHASE 2: ABSORPTION (brief flash) ─────────────────────
        elif phase == 2:
            # Flash at frozen electron position
            flash_r = 0.5 + t * 4.0
            flash_a = max(0, 0.7 * (1 - t))
            circle = plt.Circle((_abs_ex, _abs_ey), flash_r,
                                color=CLR_FLASH, alpha=flash_a,
                                fill=True, zorder=5)
            ax_main.add_patch(circle)

            # Orbit fading out
            _draw_orbit(ax_main, alpha_scale=max(0, 1 - t))

            # Torus pulsing at frozen position
            pulse = 1 + 0.4 * np.sin(t * 10 * np.pi)
            kx, ky = torus_knot_2d(_abs_ex, _abs_ey,
                                   R_TORUS * pulse, r_TORUS * pulse)
            ax_main.plot(kx, ky, color=CLR_ELECTRON, linewidth=2,
                         alpha=max(0.2, 1 - t * 0.5))

            # Wavefronts continue through from where Phase 1 ended
            _wl, _nf = 2.8, 10
            _env_half = _nf * _wl / 2
            _wave_ph1_end = _abs_ex - _env_half
            wave_x0 = _wave_ph1_end + 18.0 * t  # continue rightward
            fronts = em_wavefronts(wave_x0, y_span=4.8,
                                   wavelength=_wl, n_lines=_nf,
                                   envelope_width=8.0)
            wave_a = max(0, 1 - t * 1.5)
            for fx, fy, env in fronts:
                if env > 0.05 and wave_a > 0.05:
                    ax_main.plot(fx, fy, color=CLR_PHOTON_W,
                                 linewidth=1.5,
                                 alpha=wave_a * 0.7 * env)

            ax_main.text(0, -4.3,
                         f'E_photon = {E_ion_eV:.1f} eV \u2192 '
                         f'E_kinetic \u2248 0 eV  (threshold)',
                         color=CLR_FLASH, fontsize=9, ha='center',
                         fontfamily='monospace')

            e_level = -13.6 + 13.6 * min(1.0, t * 2.0)
            energy_marker.set_data([0.5], [e_level])
            energy_arrow.set_data([0.5, 0.5], [-13.6, e_level])
            sigma_marker.set_data([E_ion_eV], [sigma_thresh])

        # ── PHASE 3: EJECTION ──────────────────────────────────────
        elif phase == 3:
            # Straight-line ejection from frozen absorption position
            speed = 0.3 + t * 7.0
            ex = _abs_ex + speed * np.cos(_eject_angle)
            ey = _abs_ey + speed * np.sin(_eject_angle)

            # Dotted trail — straight line from absorption point
            n_dots = max(2, int(t * 25))
            trail_x = np.linspace(_abs_ex, ex, n_dots)
            trail_y = np.linspace(_abs_ey, ey, n_dots)
            ax_main.plot(trail_x, trail_y, ':', color=CLR_TRAIL,
                         linewidth=1.5, alpha=0.5)

            # Fading flash remnant (early ejection only)
            if t < 0.15:
                flash_a = 0.3 * (1 - t / 0.15)
                circle = plt.Circle((_abs_ex, _abs_ey),
                                    3.0 + t * 4, color=CLR_FLASH,
                                    alpha=flash_a, fill=True, zorder=3)
                ax_main.add_patch(circle)

            # Torus if still in view
            if -8 < ex < 8 and -5 < ey < 5:
                _draw_torus(ax_main, ex, ey, _eject_angle, circ)

            # cos^2 theta angular pattern (fades early)
            if t < 0.35:
                a_pat = 0.5 * (1 - t / 0.35)
                ang = np.linspace(0, 2 * np.pi, 100)
                rp = 1.5 * np.cos(ang)**2
                ax_main.plot(rp * np.cos(ang) - 4.5,
                             rp * np.sin(ang) + 3.0,
                             color=CLR_PHOTON_W, linewidth=1,
                             alpha=a_pat)
                ax_main.text(-4.5, 4.2, 'cos\u00b2\u03b8',
                             color=CLR_PHOTON_W, fontsize=8,
                             ha='center', alpha=a_pat)

            ax_main.text(0, -4.3,
                         'Ejected e\u207b torus \u2014 free particle   '
                         'cos\u00b2\u03b8 angular distribution',
                         color=CLR_INFO, fontsize=9, ha='center',
                         fontfamily='monospace')

            energy_marker.set_data([0.5], [0.5 * t])
            energy_arrow.set_data([], [])
            E_marker = E_ion_eV * (1 + t * 0.5)
            sigma_val = np.interp(E_marker, photo['E_photon_eV'],
                                  sigma_cm2)
            sigma_marker.set_data([E_marker], [sigma_val])

        # ── PHASE 4: HOLD ──────────────────────────────────────────
        else:
            ax_main.text(0, 2.0, 'Bare proton remains',
                         color=CLR_PROTON, fontsize=13, ha='center',
                         fontweight='bold')
            ax_main.text(0, 0.5,
                         f'\u03b3 ({E_ion_eV:.1f} eV) + H(1s)  '
                         f'\u2192  p\u207a + e\u207b\n'
                         f'\u03c3_threshold = {sigma_thresh:.1e} cm\u00b2'
                         f'  = 6.3 Mb\n'
                         f'\u03c9_photon/\u03c9_orbit = 1/2 '
                         f'at threshold\n'
                         f'Angular distribution: '
                         f'd\u03c3/d\u03a9 \u221d cos\u00b2\u03b8',
                         color=CLR_INFO, fontsize=10, ha='center',
                         fontfamily='monospace', va='top',
                         linespacing=1.6)
            ax_main.text(0, -0.6, 'p\u207a', color=CLR_PROTON,
                         fontsize=12, ha='center', fontweight='bold')

            energy_marker.set_data([0.5], [1.0])
            energy_arrow.set_data([], [])
            sigma_marker.set_data([], [])

        # ── Proton (always) ─────────────────────────────────────────
        ax_main.plot(0, 0, 'o', color=CLR_PROTON, markersize=10,
                     markeredgecolor='white', markeredgewidth=1.5,
                     zorder=15)

        # ── Timeline cursor ─────────────────────────────────────────
        tl_cursor.set_data([frame], [0.5])
        return []

    # ── Small drawing helpers ───────────────────────────────────────
    def _draw_orbit(ax, alpha_scale=1.0):
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(R_ORBIT * np.cos(th), R_ORBIT * np.sin(th), '--',
                color=CLR_ORBIT, linewidth=1, alpha=0.5 * alpha_scale)

    def _draw_torus(ax, ex, ey, orient, circ):
        kx, ky = torus_knot_2d(ex, ey, R_TORUS, r_TORUS,
                                phase_offset=orient)
        ax.plot(kx, ky, color=CLR_ELECTRON, linewidth=1.5, alpha=0.8)
        ax.fill(kx, ky, color=CLR_ELECTRON, alpha=0.1)
        px, py = photon_on_knot_2d(ex, ey, R_TORUS, r_TORUS,
                                    circ * 3, phase_offset=orient)
        ax.plot(px, py, 'o', color=CLR_PHOTON_C, markersize=6,
                markeredgecolor=CLR_E_GLOW, markeredgewidth=2, zorder=10)

    # ── Render ──────────────────────────────────────────────────────
    print(f"Rendering {N_FRAMES} frames at {FPS} fps...")
    anim = FuncAnimation(fig, draw_frame, frames=N_FRAMES,
                         blit=False, interval=1000 // FPS)

    outpath = OUTPUT_DIR / 'anim_ionization.gif'
    anim.save(outpath, writer=PillowWriter(fps=FPS), dpi=100)
    print(f"Saved: {outpath}")
    plt.close()


if __name__ == '__main__':
    run_animation()
