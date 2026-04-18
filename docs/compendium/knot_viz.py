#!/usr/bin/env python3
"""
Torus knot particle visualizer — crawl phase.

Renders each NWT particle as its torus knot T(p,q) in 3D:
  - Translucent torus surface
  - Bright knot path on the surface
  - Phase coloring along the path (hue = phase angle)

Generates static PNG images suitable for the periodic table
and per-particle data cards.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import hsv_to_rgb
from pathlib import Path


def torus_surface(R=1.0, r=0.3, N_major=80, N_minor=40):
    """Generate a torus mesh for plotting."""
    theta = np.linspace(0, 2*np.pi, N_major)
    phi = np.linspace(0, 2*np.pi, N_minor)
    THETA, PHI = np.meshgrid(theta, phi)

    X = (R + r * np.cos(PHI)) * np.cos(THETA)
    Y = (R + r * np.cos(PHI)) * np.sin(THETA)
    Z = r * np.sin(PHI)

    return X, Y, Z


def torus_knot_path(p, q, R=1.0, r=0.3, N=500):
    """Generate the T(p,q) torus knot path on a torus (R, r).

    The knot winds p times around the major circle and q times
    around the minor circle.
    """
    t = np.linspace(0, 2*np.pi, N)

    # Slightly lift the knot off the surface for visibility
    r_knot = r * 1.02

    X = (R + r_knot * np.cos(q * t)) * np.cos(p * t)
    Y = (R + r_knot * np.cos(q * t)) * np.sin(p * t)
    Z = r_knot * np.sin(q * t)

    return X, Y, Z, t


def phase_colors(t, p, q, N):
    """Color the knot path by phase angle.

    The phase winds by 2πq around the minor circle.
    Map to hue for rainbow coloring.
    """
    phase = (q * t) % (2 * np.pi)
    hue = phase / (2 * np.pi)
    sat = np.ones_like(hue)
    val = np.ones_like(hue)
    hsv = np.stack([hue, sat, val], axis=-1)
    return hsv_to_rgb(hsv)


def render_particle(p, q, name, symbol, mass_str=None,
                     R=1.0, r=0.3, save_path=None,
                     elev=25, azim=-60, figsize=(5, 5)):
    """Render a single particle as its torus knot."""
    fig = plt.figure(figsize=figsize, facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Torus surface (translucent)
    X_t, Y_t, Z_t = torus_surface(R, r)
    ax.plot_surface(X_t, Y_t, Z_t, alpha=0.18, color='#4488cc',
                    shade=True, edgecolor='none')

    # Knot path
    X_k, Y_k, Z_k, t = torus_knot_path(p, q, R, r, N=1000)
    colors = phase_colors(t, p, q, len(t))

    # Plot as colored line segments
    for i in range(len(t) - 1):
        ax.plot(X_k[i:i+2], Y_k[i:i+2], Z_k[i:i+2],
                color=colors[i], linewidth=2.5, solid_capstyle='round')

    # Styling
    ax.set_xlim(-1.5*R, 1.5*R)
    ax.set_ylim(-1.5*R, 1.5*R)
    ax.set_zlim(-0.8*R, 0.8*R)
    ax.set_box_aspect([1.5, 1.5, 0.8])
    ax.axis('off')
    ax.view_init(elev=elev, azim=azim)

    # Title
    title = f"{symbol}"
    if mass_str:
        title += f"\n{mass_str}"
    title += f"\nT({p},{q})"
    ax.set_title(title, color='white', fontsize=14, fontweight='bold',
                 pad=-10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        print(f"  Saved: {save_path}")

    plt.close(fig)


def render_gallery():
    """Render a gallery of key NWT particles."""
    out_dir = Path(__file__).parent / "images"
    out_dir.mkdir(exist_ok=True)

    particles = [
        (2, 1, "electron", "e⁻", "0.511 MeV"),
        (2, 3, "trefoil", "T(2,3)", "SU(3) carrier"),
        (2, 5, "cinquefoil", "T(2,5)", "SU(5) carrier"),
        (2, 7, "heptafoil", "T(2,7)", "pre-GUT"),
        (1, 1, "hopf", "Hopf", "SU(2) carrier"),
        (3, 2, "trefoil_alt", "T(3,2)", "alt. trefoil"),
        (3, 4, "T34", "T(3,4)", "genus 3"),
    ]

    print("Rendering particle gallery...")
    for p, q, filename, symbol, mass_str in particles:
        path = out_dir / f"{filename}.png"
        render_particle(p, q, filename, symbol, mass_str,
                        save_path=path)

    # Also render a combined figure
    print("\nRendering combined figure...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8),
                              subplot_kw={'projection': '3d'},
                              facecolor='black')

    all_particles = [
        (2, 1, "e⁻", "0.511 MeV"),
        (1, 1, "Hopf", "SU(2)"),
        (2, 3, "Trefoil", "SU(3)"),
        (2, 5, "Cinquefoil", "SU(5)"),
        (2, 7, "Heptafoil", "pre-GUT"),
        (3, 2, "T(3,2)", "alt trefoil"),
        (3, 4, "T(3,4)", "genus 3"),
        (2, 9, "Nonafoil", "T(2,9)"),
    ]

    for idx, (p, q, symbol, label) in enumerate(all_particles):
        row, col = divmod(idx, 4)
        ax = axes[row, col]
        ax.set_facecolor('black')

        R, r = 1.0, 0.3
        X_t, Y_t, Z_t = torus_surface(R, r, N_major=50, N_minor=25)
        ax.plot_surface(X_t, Y_t, Z_t, alpha=0.15, color='#4488cc',
                        shade=True, edgecolor='none')

        X_k, Y_k, Z_k, t = torus_knot_path(p, q, R, r, N=600)
        colors = phase_colors(t, p, q, len(t))
        for i in range(0, len(t) - 1, 2):
            ax.plot(X_k[i:i+2], Y_k[i:i+2], Z_k[i:i+2],
                    color=colors[i], linewidth=2.0)

        ax.set_xlim(-1.4*R, 1.4*R)
        ax.set_ylim(-1.4*R, 1.4*R)
        ax.set_zlim(-0.7*R, 0.7*R)
        ax.set_box_aspect([1.4, 1.4, 0.7])
        ax.axis('off')
        ax.view_init(elev=25, azim=-60)
        ax.set_title(f"{symbol}\n{label}", color='white',
                     fontsize=11, fontweight='bold', pad=-5)

    plt.tight_layout()
    combined_path = out_dir / "knot_gallery.png"
    fig.savefig(combined_path, dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    print(f"  Saved: {combined_path}")
    plt.close(fig)


if __name__ == "__main__":
    render_gallery()
