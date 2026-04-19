#!/usr/bin/env python3
"""
NWT Particle Visualizer — wireframe carrier + bold resonance path.

Style matches figure1_torus_knot.png: wireframe torus with gridlines
showing the coordinate frame, bold colored path for the resonance.
White background, publication-quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import hsv_to_rgb
from pathlib import Path


# ── Carrier geometries ────────────────────────────────────

def torus_wireframe(R=1.0, r=0.3, N_major=40, N_minor=20):
    """Torus mesh for wireframe rendering."""
    theta = np.linspace(0, 2*np.pi, N_major)
    phi = np.linspace(0, 2*np.pi, N_minor)
    THETA, PHI = np.meshgrid(theta, phi)
    X = (R + r * np.cos(PHI)) * np.cos(THETA)
    Y = (R + r * np.cos(PHI)) * np.sin(THETA)
    Z = r * np.sin(PHI)
    return X, Y, Z, THETA, PHI


def knot_tube_wireframe(p_knot, q_knot, R=1.0, a=0.35, r_tube=0.15,
                         N_arc=200, N_circ=12):
    """Wireframe tube around a T(p,q) knot centerline."""
    t = np.linspace(0, 2*np.pi, N_arc, endpoint=False)
    # Centerline
    cx = (R + a*R*np.cos(q_knot*t)) * np.cos(p_knot*t)
    cy = (R + a*R*np.cos(q_knot*t)) * np.sin(p_knot*t)
    cz = a * R * np.sin(q_knot*t)
    center = np.array([cx, cy, cz])

    # Build frame via parallel transport
    tangent = np.gradient(center, axis=1)
    norms = np.sqrt(np.sum(tangent**2, axis=0))
    tangent /= norms[np.newaxis, :]

    if abs(tangent[2, 0]) < 0.9:
        n0 = np.cross(tangent[:, 0], [0, 0, 1])
    else:
        n0 = np.cross(tangent[:, 0], [1, 0, 0])
    n0 /= np.linalg.norm(n0)

    normals = np.zeros_like(center)
    normals[:, 0] = n0
    for i in range(1, N_arc):
        n = normals[:, i-1] - np.dot(normals[:, i-1], tangent[:, i]) * tangent[:, i]
        norm_val = np.linalg.norm(n)
        if norm_val > 1e-10:
            normals[:, i] = n / norm_val
        else:
            normals[:, i] = normals[:, i-1]
    binormals = np.cross(tangent.T, normals.T).T

    phi = np.linspace(0, 2*np.pi, N_circ)
    X = np.zeros((N_circ, N_arc))
    Y = np.zeros((N_circ, N_arc))
    Z = np.zeros((N_circ, N_arc))
    for j in range(N_circ):
        off = r_tube * (np.cos(phi[j]) * normals + np.sin(phi[j]) * binormals)
        X[j] = center[0] + off[0]
        Y[j] = center[1] + off[1]
        Z[j] = center[2] + off[2]

    return X, Y, Z, center


def torus_knot_path(p, q, R=1.0, r=0.3, N=1000):
    """T(p,q) winding on a simple torus — the resonance path."""
    t = np.linspace(0, 2*np.pi, N)
    r_path = r * 1.01  # just above the surface
    X = (R + r_path * np.cos(q * t)) * np.cos(p * t)
    Y = (R + r_path * np.cos(q * t)) * np.sin(p * t)
    Z = r_path * np.sin(q * t)
    return X, Y, Z, t


def resonance_on_knot_tube(center, p_res, q_res, r_tube=0.15, N=1500,
                            p_knot=2, q_knot=3, R=1.0, a=0.35):
    """Resonance winding on a knotted tube carrier.

    The resonance T(p_res, q_res) on a carrier T(p_knot, q_knot)
    is a CABLE KNOT: a torus knot on a slightly larger torus with
    combined winding numbers:

      p_total = p_res × p_knot
      q_total = q_res + p_res × q_knot

    E.g., proton (1,4) on trefoil (2,3): cable = T(2, 7)
          J/ψ (2,3) on trefoil (2,3): cable = T(4, 9)
    """
    p_total = p_res * p_knot
    q_total = q_res + p_res * q_knot

    t = np.linspace(0, 2*np.pi, N, endpoint=False)

    # Cable knot on a torus of minor radius (a*R + r_tube)
    a_cable = a + r_tube / R

    rx = (R + a_cable * R * np.cos(q_total * t)) * np.cos(p_total * t)
    ry = (R + a_cable * R * np.cos(q_total * t)) * np.sin(p_total * t)
    rz = a_cable * R * np.sin(q_total * t)

    return rx, ry, rz


# ── Rendering ─────────────────────────────────────────────

def render_particle(carrier_type, p_res, q_res, symbol, label,
                     save_path=None, figsize=(5, 5),
                     elev=25, azim=-60, path_color='#cc2222'):
    """Render: wireframe carrier + bold resonance path."""
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    R = 1.0
    r = 0.30  # torus minor radius (carrier tube thickness)

    if carrier_type == 'unknot':
        # Wireframe torus
        X, Y, Z, _, _ = torus_wireframe(R, r, N_major=40, N_minor=20)
        ax.plot_wireframe(X, Y, Z, color='#cccccc', linewidth=0.3,
                          alpha=0.5)
        # Resonance as torus knot path
        rx, ry, rz, _ = torus_knot_path(p_res, q_res, R, r, N=1000)
        ax.plot(rx, ry, rz, color=path_color, linewidth=3.0,
                solid_capstyle='round')

    elif carrier_type in ('trefoil', 'cinquefoil'):
        p_k = 2
        q_k = 3 if carrier_type == 'trefoil' else 5
        a = 0.35 if carrier_type == 'trefoil' else 0.30
        r_tube = 0.35  # fat tube so poloidal spiral is visible

        X, Y, Z, center = knot_tube_wireframe(
            p_k, q_k, R, a, r_tube, N_arc=200, N_circ=12)
        ax.plot_wireframe(X, Y, Z, color='#cccccc', linewidth=0.3,
                          alpha=0.4)
        # Carrier centerline
        ax.plot(center[0], center[1], center[2], color='#999999',
                linewidth=1.0, alpha=0.6)

        rx, ry, rz = resonance_on_knot_tube(
            center, p_res, q_res, r_tube, N=2000,
            p_knot=p_k, q_knot=q_k, R=R, a=a)
        ax.plot(rx, ry, rz, color=path_color, linewidth=2.5,
                solid_capstyle='round')

    elif carrier_type == 'hopf':
        # Two linked wireframe tori
        # Torus 1: in xy-plane
        X1, Y1, Z1, _, _ = torus_wireframe(R*0.7, r*0.5,
                                             N_major=30, N_minor=12)
        ax.plot_wireframe(X1, Y1, Z1, color='#cccccc', linewidth=0.3,
                          alpha=0.4)
        # Torus 2: in xz-plane, offset
        X2, Y2, Z2, _, _ = torus_wireframe(R*0.7, r*0.5,
                                             N_major=30, N_minor=12)
        # Rotate 90° around x-axis and offset
        X2_rot = X2 + 0.5
        Y2_rot = -Z2
        Z2_rot = Y2
        ax.plot_wireframe(X2_rot, Y2_rot, Z2_rot, color='#cccccc',
                          linewidth=0.3, alpha=0.4)

        # Resonance on torus 1
        rx1, ry1, rz1, _ = torus_knot_path(p_res, q_res, R*0.7,
                                             r*0.5, N=800)
        ax.plot(rx1, ry1, rz1, color=path_color, linewidth=2.5)

        # Resonance on torus 2 (rotated)
        rx2, ry2, rz2, _ = torus_knot_path(p_res, q_res, R*0.7,
                                             r*0.5, N=800)
        ax.plot(rx2 + 0.5, -rz2, ry2, color='#2266cc', linewidth=2.5)

    # Styling
    lim = 1.5 * R
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-0.8*lim, 0.8*lim)
    ax.set_box_aspect([1.5, 1.5, 0.8])
    ax.axis('off')
    ax.view_init(elev=elev, azim=azim)

    title = f"{symbol}\n{label}"
    ax.set_title(title, color='black', fontsize=14, fontweight='bold',
                 pad=-10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def render_gallery():
    """Render gallery in figure1 style."""
    out_dir = Path(__file__).parent / "images"
    out_dir.mkdir(exist_ok=True)

    particles = [
        ("unknot", 2, 1, "e⁻", "Electron (2,1)\nunknot carrier", "#cc2222"),
        ("hopf", 2, 1, "π", "Pion (2,1)\nHopf carrier", "#cc2222"),
        ("trefoil", 1, 4, "p", "Proton (1,4)\ntrefoil carrier", "#cc2222"),
        ("trefoil", 2, 3, "J/ψ", "J/ψ (2,3)\ntrefoil carrier", "#22aa44"),
        ("cinquefoil", 2, 5, "Pꜱ", "Pentaquark (2,5)\ncinquefoil carrier", "#cc8822"),
        ("hopf", 1, 2, "W±", "W boson (1,2)\nHopf carrier", "#2266cc"),
    ]

    print("Rendering particle gallery (figure1 style)...")
    for carrier, p, q, symbol, label, color in particles:
        filename = symbol.replace("±", "").replace("/", "").replace("ꜱ", "c")
        path = out_dir / f"{filename}.png"
        render_particle(carrier, p, q, symbol, label,
                        save_path=path, path_color=color)

    # Combined gallery
    print("\nRendering combined gallery...")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9),
                              subplot_kw={'projection': '3d'},
                              facecolor='white')

    for idx, (carrier, p, q, symbol, label, color) in enumerate(particles):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        ax.set_facecolor('white')

        R, r = 1.0, 0.30

        if carrier == 'unknot':
            X, Y, Z, _, _ = torus_wireframe(R, r, 30, 15)
            ax.plot_wireframe(X, Y, Z, color='#cccccc', linewidth=0.2,
                              alpha=0.4)
            rx, ry, rz, _ = torus_knot_path(p, q, R, r, 600)
            ax.plot(rx, ry, rz, color=color, linewidth=2.5)

        elif carrier in ('trefoil', 'cinquefoil'):
            q_k = 3 if carrier == 'trefoil' else 5
            a = 0.35 if carrier == 'trefoil' else 0.30
            X, Y, Z, ctr = knot_tube_wireframe(2, q_k, R, a, 0.35, 150, 14)
            ax.plot_wireframe(X, Y, Z, color='#cccccc', linewidth=0.2,
                              alpha=0.35)
            ax.plot(ctr[0], ctr[1], ctr[2], color='#999999', linewidth=0.8,
                    alpha=0.5)
            rx, ry, rz = resonance_on_knot_tube(
                ctr, p, q, 0.35, 1200,
                p_knot=2, q_knot=q_k, R=R, a=a)
            ax.plot(rx, ry, rz, color=color, linewidth=2.0)

        elif carrier == 'hopf':
            X1, Y1, Z1, _, _ = torus_wireframe(R*0.7, r*0.5, 25, 10)
            ax.plot_wireframe(X1, Y1, Z1, color='#cccccc', linewidth=0.2,
                              alpha=0.35)
            X2, Y2, Z2, _, _ = torus_wireframe(R*0.7, r*0.5, 25, 10)
            ax.plot_wireframe(X2+0.5, -Z2, Y2, color='#cccccc',
                              linewidth=0.2, alpha=0.35)
            rx1, ry1, rz1, _ = torus_knot_path(p, q, R*0.7, r*0.5, 500)
            ax.plot(rx1, ry1, rz1, color=color, linewidth=2.0)
            rx2, ry2, rz2, _ = torus_knot_path(p, q, R*0.7, r*0.5, 500)
            ax.plot(rx2+0.5, -rz2, ry2, color='#2266cc', linewidth=2.0)

        lim = 1.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-0.8*lim, 0.8*lim)
        ax.set_box_aspect([1.5, 1.5, 0.8])
        ax.axis('off')
        ax.view_init(elev=25, azim=-60)
        ax.set_title(f"{symbol}\n{label}", color='black',
                     fontsize=9, fontweight='bold', pad=-5)

    plt.tight_layout()
    gallery_path = out_dir / "knot_gallery.png"
    fig.savefig(gallery_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved: {gallery_path}")
    plt.close(fig)


if __name__ == "__main__":
    render_gallery()
