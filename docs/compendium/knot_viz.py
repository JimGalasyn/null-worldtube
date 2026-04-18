#!/usr/bin/env python3
"""
NWT Particle Visualizer — carrier knot + resonance winding.

Each particle is rendered as:
  1. CARRIER (gray tube): the knot/link topology that defines the
     particle family — unknot for leptons, Hopf link for mesons,
     trefoil for baryons, cinquefoil for pentaquarks.
  2. RESONANCE (colored path): the (p,q) winding mode on the
     carrier surface, phase-colored by poloidal angle.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import hsv_to_rgb
from pathlib import Path


# ── Carrier geometries ────────────────────────────────────

def circle_centerline(R=1.0, N=200):
    """Simple circle (unknot carrier for leptons)."""
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.array([R*np.cos(t), R*np.sin(t), np.zeros(N)]), t


def trefoil_centerline(R=1.0, a=0.35, N=400):
    """Trefoil knot T(2,3) centerline (baryon carrier)."""
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = (R + a*R*np.cos(3*t)) * np.cos(2*t)
    y = (R + a*R*np.cos(3*t)) * np.sin(2*t)
    z = a * R * np.sin(3*t)
    return np.array([x, y, z]), t


def cinquefoil_centerline(R=1.0, a=0.35, N=600):
    """Cinquefoil knot T(2,5) centerline (pentaquark carrier)."""
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = (R + a*R*np.cos(5*t)) * np.cos(2*t)
    y = (R + a*R*np.cos(5*t)) * np.sin(2*t)
    z = a * R * np.sin(5*t)
    return np.array([x, y, z]), t


def hopf_link_centerlines(R=1.0, r_sep=0.45, N=200):
    """Two linked circles (Hopf link carrier for mesons)."""
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    # Circle 1: in the xy-plane
    c1 = np.array([R*np.cos(t), R*np.sin(t), np.zeros(N)])
    # Circle 2: in the xz-plane, offset and linked through circle 1
    c2 = np.array([r_sep + R*np.cos(t), np.zeros(N), R*np.sin(t)])
    return c1, c2, t


# ── Tube rendering ────────────────────────────────────────

def tube_around_curve(centerline, r_tube=0.08, N_circ=16):
    """Generate a tube surface around a 3D curve.

    Returns (X, Y, Z) mesh arrays for plot_surface.
    """
    pts = centerline  # shape (3, N)
    N = pts.shape[1]

    # Compute tangent vectors
    tangent = np.zeros_like(pts)
    tangent[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    tangent[:, 0] = pts[:, 1] - pts[:, -1]
    tangent[:, -1] = pts[:, 0] - pts[:, -2]
    norms = np.sqrt(np.sum(tangent**2, axis=0))
    tangent /= norms[np.newaxis, :]

    # Build a local frame (normal, binormal) via parallel transport
    # Start with an arbitrary normal
    if abs(tangent[2, 0]) < 0.9:
        n0 = np.cross(tangent[:, 0], [0, 0, 1])
    else:
        n0 = np.cross(tangent[:, 0], [1, 0, 0])
    n0 /= np.linalg.norm(n0)

    normals = np.zeros_like(pts)
    normals[:, 0] = n0

    for i in range(1, N):
        # Project previous normal onto plane perpendicular to new tangent
        n = normals[:, i-1]
        n = n - np.dot(n, tangent[:, i]) * tangent[:, i]
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n /= norm
        else:
            n = normals[:, i-1]
        normals[:, i] = n

    binormals = np.cross(tangent.T, normals.T).T

    # Generate tube surface
    phi = np.linspace(0, 2*np.pi, N_circ, endpoint=False)

    X = np.zeros((N_circ, N))
    Y = np.zeros((N_circ, N))
    Z = np.zeros((N_circ, N))

    for j in range(N_circ):
        offset = r_tube * (np.cos(phi[j]) * normals + np.sin(phi[j]) * binormals)
        X[j] = pts[0] + offset[0]
        Y[j] = pts[1] + offset[1]
        Z[j] = pts[2] + offset[2]

    return X, Y, Z


def resonance_on_carrier(centerline, p_res, q_res, r_tube=0.08, N=1000):
    """Trace a (p_res, q_res) resonance winding on a carrier tube.

    The resonance winds p_res times along the carrier and q_res
    times around the tube cross-section.
    """
    pts = centerline  # shape (3, N_carrier)
    N_carrier = pts.shape[1]

    # Compute frame (same as tube_around_curve)
    tangent = np.zeros_like(pts)
    tangent[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    tangent[:, 0] = pts[:, 1] - pts[:, -1]
    tangent[:, -1] = pts[:, 0] - pts[:, -2]
    norms = np.sqrt(np.sum(tangent**2, axis=0))
    tangent /= norms[np.newaxis, :]

    if abs(tangent[2, 0]) < 0.9:
        n0 = np.cross(tangent[:, 0], [0, 0, 1])
    else:
        n0 = np.cross(tangent[:, 0], [1, 0, 0])
    n0 /= np.linalg.norm(n0)

    normals = np.zeros_like(pts)
    normals[:, 0] = n0
    for i in range(1, N_carrier):
        n = normals[:, i-1]
        n = n - np.dot(n, tangent[:, i]) * tangent[:, i]
        norm_val = np.linalg.norm(n)
        if norm_val > 1e-10:
            n /= norm_val
        normals[:, i] = n
    binormals = np.cross(tangent.T, normals.T).T

    # Parametrize the resonance path
    t = np.linspace(0, 2*np.pi * p_res, N, endpoint=False)
    s_idx = (t / (2*np.pi * p_res) * N_carrier).astype(int) % N_carrier
    phi_res = q_res * t / p_res  # poloidal angle

    r_res = r_tube * 1.08  # on the tube surface

    rx = pts[0, s_idx] + r_res * (np.cos(phi_res) * normals[0, s_idx] +
                                    np.sin(phi_res) * binormals[0, s_idx])
    ry = pts[1, s_idx] + r_res * (np.cos(phi_res) * normals[1, s_idx] +
                                    np.sin(phi_res) * binormals[1, s_idx])
    rz = pts[2, s_idx] + r_res * (np.cos(phi_res) * normals[2, s_idx] +
                                    np.sin(phi_res) * binormals[2, s_idx])

    # Phase colors
    hue = (phi_res % (2*np.pi)) / (2*np.pi)
    colors = hsv_to_rgb(np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=-1))

    return rx, ry, rz, colors


# ── Rendering ─────────────────────────────────────────────

def render_particle(carrier_type, p_res, q_res, symbol, label,
                     save_path=None, figsize=(5, 5),
                     elev=25, azim=-60):
    """Render a particle: gray carrier tube + colored resonance."""
    fig = plt.figure(figsize=figsize, facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    R = 1.0
    r_tube = 0.35  # fat tube so the poloidal spiral is dramatic

    if carrier_type == 'unknot':
        cl, _ = circle_centerline(R)
        carriers = [cl]
    elif carrier_type == 'trefoil':
        cl, _ = trefoil_centerline(R, a=0.35)
        carriers = [cl]
    elif carrier_type == 'cinquefoil':
        cl, _ = cinquefoil_centerline(R, a=0.30)
        carriers = [cl]
    elif carrier_type == 'hopf':
        c1, c2, _ = hopf_link_centerlines(R, r_sep=0.55)
        carriers = [c1, c2]
    else:
        cl, _ = circle_centerline(R)
        carriers = [cl]

    # Draw carrier tube(s) in gray
    for cl in carriers:
        X, Y, Z = tube_around_curve(cl, r_tube=r_tube, N_circ=16)
        ax.plot_surface(X, Y, Z, alpha=0.20, color='#888888',
                        shade=True, edgecolor='none')

    # Draw carrier centerline (thin white reference line)
    for cl in carriers:
        ax.plot(cl[0], cl[1], cl[2], color='white', linewidth=0.8,
                alpha=0.5)

    # Draw resonance winding on each carrier
    for cl in carriers:
        rx, ry, rz, colors = resonance_on_carrier(
            cl, p_res, q_res, r_tube=r_tube, N=2000)

        for i in range(0, len(rx) - 1, 2):
            ax.plot(rx[i:i+2], ry[i:i+2], rz[i:i+2],
                    color=colors[i], linewidth=2.5, solid_capstyle='round')

    # Styling
    lim = 1.6 * R
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-0.8*lim, 0.8*lim)
    ax.set_box_aspect([1.6, 1.6, 0.8])
    ax.axis('off')
    ax.view_init(elev=elev, azim=azim)

    title = f"{symbol}\n{label}"
    ax.set_title(title, color='white', fontsize=14, fontweight='bold',
                 pad=-10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def render_gallery():
    """Render gallery with correct carrier geometries."""
    out_dir = Path(__file__).parent / "images"
    out_dir.mkdir(exist_ok=True)

    # (carrier_type, p_res, q_res, filename, symbol, label)
    particles = [
        # Leptons on unknot carrier
        ("unknot", 2, 1, "electron", "e⁻", "lepton\nunknot carrier"),
        # Mesons on Hopf link carrier
        ("hopf", 2, 1, "pion", "π", "meson\nHopf carrier"),
        # Baryons on trefoil carrier
        ("trefoil", 1, 4, "proton", "p", "baryon\ntrefoil carrier"),
        # Quarkonium on trefoil-based carrier
        ("trefoil", 2, 3, "jpsi", "J/ψ", "charmonium\ntrefoil carrier"),
        # Pentaquark on cinquefoil carrier
        ("cinquefoil", 2, 5, "pentaquark", "Pꜱ", "pentaquark\ncinquefoil carrier"),
        # Gauge on Hopf
        ("hopf", 1, 2, "W_boson", "W±", "gauge boson\nHopf carrier"),
    ]

    print("Rendering particle gallery with carrier knots...")
    for carrier, p, q, filename, symbol, label in particles:
        path = out_dir / f"{filename}.png"
        render_particle(carrier, p, q, symbol, label, save_path=path)

    # Combined gallery
    print("\nRendering combined gallery...")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9),
                              subplot_kw={'projection': '3d'},
                              facecolor='black')

    for idx, (carrier, p, q, filename, symbol, label) in enumerate(particles):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        ax.set_facecolor('black')

        R = 1.0
        r_tube = 0.35

        if carrier == 'unknot':
            cl, _ = circle_centerline(R)
            cls = [cl]
        elif carrier == 'trefoil':
            cl, _ = trefoil_centerline(R, a=0.35)
            cls = [cl]
        elif carrier == 'cinquefoil':
            cl, _ = cinquefoil_centerline(R, a=0.30)
            cls = [cl]
        elif carrier == 'hopf':
            c1, c2, _ = hopf_link_centerlines(R, r_sep=0.55)
            cls = [c1, c2]

        for cl in cls:
            X, Y, Z = tube_around_curve(cl, r_tube=r_tube, N_circ=12)
            ax.plot_surface(X, Y, Z, alpha=0.20, color='#888888',
                            shade=True, edgecolor='none')

        for cl in cls:
            ax.plot(cl[0], cl[1], cl[2], color='white', linewidth=0.6,
                    alpha=0.4)

        for cl in cls:
            rx, ry, rz, colors = resonance_on_carrier(
                cl, p, q, r_tube=r_tube, N=1000)
            for i in range(0, len(rx) - 1, 2):
                ax.plot(rx[i:i+2], ry[i:i+2], rz[i:i+2],
                        color=colors[i], linewidth=2.0)

        lim = 1.6
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-0.8*lim, 0.8*lim)
        ax.set_box_aspect([1.6, 1.6, 0.8])
        ax.axis('off')
        ax.view_init(elev=25, azim=-60)
        ax.set_title(f"{symbol}\n{label}", color='white',
                     fontsize=10, fontweight='bold', pad=-5)

    plt.tight_layout()
    gallery_path = out_dir / "knot_gallery.png"
    fig.savefig(gallery_path, dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    print(f"  Saved: {gallery_path}")
    plt.close(fig)


if __name__ == "__main__":
    render_gallery()
