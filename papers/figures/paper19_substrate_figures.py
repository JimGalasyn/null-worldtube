#!/usr/bin/env python3
"""
Substrate-primitive figures for Papers 18 / 19.

Generates three publication-quality vector PDFs that illustrate the
core substrate objects:

  paper19_fig_trefoil.pdf
      The (2,3) torus knot (trefoil), the canonical NWT carrier knot
      that supports the BPS background of the Sakharov derivation
      and whose Dehn surgery yields the Brieskorn-Poincare sphere.

  paper19_fig_poincare_sphere.pdf
      The Brieskorn-Poincare homology 3-sphere Sigma(2,3,5) = S^3 / 2I,
      drawn as a regular dodecahedron with opposite faces identified
      via a 36-degree twist.

  paper19_fig_K7_heffter.pdf
      The complete graph K_7 on its unique Heffter triangular embedding
      on the torus, drawn in the square fundamental domain.

Style: light publication theme, vector PDF output, axes off where possible.

Usage:
    python3 diagrams/paper19_substrate_figures.py            # all three
"""

from __future__ import annotations
import argparse
import os
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


HERE = os.path.dirname(os.path.abspath(__file__))


PUB_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a",
    "axes.edgecolor": "#1a1a1a",
    "xtick.color": "#1a1a1a",
    "ytick.color": "#1a1a1a",
    "grid.color": "#cccccc",
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.10,
}


# ---------------------------------------------------------------------------
# Figure 1: trefoil knot drawn on (and slightly outside) its supporting torus
# ---------------------------------------------------------------------------

def fig_trefoil(out_path: str) -> None:
    plt.rcParams.update(PUB_STYLE)
    fig = plt.figure(figsize=(6.0, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    # Torus wireframe (sparse) -- 3D matplotlib's plot_surface occludes
    # lines regardless of zorder, so we use plot_wireframe with light
    # styling instead.
    R, r = 2.0, 0.7
    theta = np.linspace(0, 2 * np.pi, 60)
    phi = np.linspace(0, 2 * np.pi, 24)
    THETA, PHI = np.meshgrid(theta, phi)
    Xt = (R + r * np.cos(PHI)) * np.cos(THETA)
    Yt = (R + r * np.cos(PHI)) * np.sin(THETA)
    Zt = r * np.sin(PHI)
    ax.plot_wireframe(Xt, Yt, Zt,
                       color="#a3c4dc", linewidth=0.3, alpha=0.6,
                       rstride=3, cstride=3)

    # Trefoil curve.  Plot a single 3D polyline; matplotlib's 3D
    # depth-sorting is fragile when many short segments mix with surfaces,
    # so a single plot call with one strong colour reads more cleanly.
    t = np.linspace(0, 2 * np.pi, 600)
    rr = r + 0.18
    Xk = (R + rr * np.cos(3 * t)) * np.cos(2 * t)
    Yk = (R + rr * np.cos(3 * t)) * np.sin(2 * t)
    Zk = rr * np.sin(3 * t)
    ax.plot(Xk, Yk, Zk, color="#c0392b", linewidth=2.8, zorder=10)
    # Add a short shadow under the curve at z = -r (projected to torus
    # equator) for additional depth cue.
    ax.plot(Xk, Yk, np.full_like(Zk, -r - 0.01),
             color="#888", linewidth=0.9, alpha=0.35, zorder=1)

    # Title and caption
    ax.text2D(0.50, 0.97,
              r"$(p,q) = (2,3)$ torus knot $\equiv\;3_1$ (trefoil)",
              transform=ax.transAxes, ha="center", fontsize=12)
    ax.text2D(0.50, 0.04,
              r"3 crossings, $\gcd(p,q)=1$; Dehn surgery yields "
              r"$\Sigma(2,3,5)$",
              transform=ax.transAxes, ha="center", fontsize=10,
              color="#444444")

    ax.set_axis_off()
    ax.view_init(elev=30, azim=35)
    ax.set_box_aspect((1, 1, 0.45))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: regular dodecahedron with opposite-face identification arrows
# ---------------------------------------------------------------------------

def _regular_dodecahedron_verts():
    """Return 20 vertices of a unit regular dodecahedron."""
    phi = (1 + np.sqrt(5)) / 2
    iphi = 1 / phi
    cube = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    extra = []
    for sa in (-iphi, iphi):
        for sb in (-phi, phi):
            extra.append((0, sa, sb))
            extra.append((sb, 0, sa))
            extra.append((sa, sb, 0))
    verts = np.array(cube + extra, dtype=float)
    return verts / np.linalg.norm(verts[0])


def fig_poincare_sphere(out_path: str) -> None:
    """
    Three-panel figure for the Brieskorn-Poincare sphere face
    identification and the McKay correspondence to affine E8.

    Panel (a): 3D dodecahedron with one antipodal pair of pentagonal
               faces highlighted (red front, blue back).  All other
               faces are drawn in neutral cream so the eye locks onto
               the pair.  Vertex labels 1..5 mark the front pentagon.

    Panel (b): view down the front-back axis with the front pentagon
               (red) drawn at +36 deg and the back pentagon (blue) at
               0 deg.  An explicit arc shows the pi/5 = 36 deg twist
               that makes vertex k of the front land on vertex k of
               the back after the antipodal flip.

    Panel (c): the affine Dynkin diagram of E8 (E8-hat).  Each node is
               labeled by the dimension of the corresponding 2I irrep
               under the McKay correspondence; the six distinct
               dimensions {1, 2, 3, 4, 5, 6} are exactly the
               carrier-knot crossing numbers n_q in Paper 6.
    """
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon as MplPolygon, Arc, Circle
    from matplotlib import gridspec

    plt.rcParams.update(PUB_STYLE)
    fig = plt.figure(figsize=(10.0, 7.0))
    gs = gridspec.GridSpec(
        2, 2, height_ratios=[2.4, 1.0], width_ratios=[1.0, 1.0],
        hspace=0.18, wspace=0.05,
        left=0.03, right=0.97, top=0.94, bottom=0.04,
    )

    # =========================================================
    # Panel (a): 3D dodecahedron with one face-pair highlighted
    # =========================================================
    ax = fig.add_subplot(gs[0, 0], projection="3d")

    verts = _regular_dodecahedron_verts()
    hull = ConvexHull(verts)
    tris = [verts[s] for s in hull.simplices]

    # Group hull simplices into the 12 pentagonal faces using the
    # exact plane equations stored in hull.equations (each face's three
    # fan-triangles share an equation, modulo floating-point noise).
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, eq in enumerate(hull.equations):
        key = tuple(np.round(eq, 5))
        buckets[key].append(i)
    face_groups = list(buckets.values())
    face_normals = np.array([
        hull.equations[g[0]][:3] for g in face_groups
    ])
    face_centres = np.array([
        np.mean([verts[s].mean(axis=0)
                 for s in hull.simplices[g]], axis=0)
        for g in face_groups
    ])

    # Pick a face pair whose axis is more or less horizontal in the
    # camera frame (so both the front and back face are visible at
    # oblique angles).  We choose the face whose normal projects most
    # strongly onto the +x direction in world coords, then take its
    # antipode.  The camera (elev, azim) below is then tuned so this
    # axis is roughly horizontal on screen.
    f_top = int(np.argmax(face_normals[:, 0]))
    target_norm = -face_normals[f_top]
    f_bot = int(np.argmax(face_normals @ target_norm))

    # Render the 10 neutral faces first, then overlay the two
    # highlighted faces with strong saturated colors and zorder.
    # (Per-simplex facecolors on a single Poly3DCollection don't
    # always read crisply because matplotlib blends them with the
    # alpha channel during depth-sort, so we use two collections.)
    NEUTRAL = "#f4ecd9"
    FRONT_COLOR = "#d54545"   # warm red
    BACK_COLOR = "#3a78c2"    # cool blue
    highlighted = set(face_groups[f_top]) | set(face_groups[f_bot])
    neutral_tris = [tris[i] for i in range(len(tris))
                    if i not in highlighted]
    front_tris = [tris[i] for i in face_groups[f_top]]
    back_tris = [tris[i] for i in face_groups[f_bot]]

    # Draw the back face FIRST with full opacity so it sits behind
    # the dodecahedron's interior; then draw the 10 neutral faces
    # at low alpha so the back face ghosts through; then the front
    # face on top at full opacity.  This is the only way to make
    # both opposite faces simultaneously visible without
    # ray-tracing the dodecahedron interior.
    ax.add_collection3d(Poly3DCollection(
        back_tris, facecolor=BACK_COLOR, edgecolor="#1f3f70",
        linewidths=1.2, alpha=1.0, zorder=1,
    ))
    ax.add_collection3d(Poly3DCollection(
        neutral_tris, facecolor=NEUTRAL, edgecolor="#222222",
        linewidths=0.7, alpha=0.45, zorder=2,
    ))
    ax.add_collection3d(Poly3DCollection(
        front_tris, facecolor=FRONT_COLOR, edgecolor="#7a1f1f",
        linewidths=1.2, alpha=1.0, zorder=3,
    ))

    # True dodecahedron edges (filter out triangulation diagonals).
    edge_set = set()
    for s in hull.simplices:
        for i in range(3):
            a, b = sorted((s[i], s[(i + 1) % 3]))
            edge_set.add((a, b))
    if edge_set:
        lengths = np.array([np.linalg.norm(verts[a] - verts[b])
                            for a, b in edge_set])
        edge_len = np.min(lengths)
        true_edges = [(a, b) for (a, b), L in zip(edge_set, lengths)
                      if abs(L - edge_len) < 1e-6]
        for a, b in true_edges:
            ax.plot(*zip(verts[a], verts[b]), color="#222222",
                     linewidth=1.2, zorder=5)

    # Recover the 5 vertices of the top pentagon (the 5 distinct hull
    # vertices in the f_top simplex group) and label them 1..5 in
    # cyclic order around the face centre.
    top_vert_idxs = sorted({int(v) for s_idx in face_groups[f_top]
                            for v in hull.simplices[s_idx]})
    assert len(top_vert_idxs) == 5, (
        f"top face has {len(top_vert_idxs)} vertices, expected 5"
    )
    top_verts_xyz = verts[top_vert_idxs]
    top_centre = face_centres[f_top]
    top_normal = face_normals[f_top]
    # Angular ordering around the centre
    u = np.array([1.0, 0.0, 0.0])
    u = u - (u @ top_normal) * top_normal
    u /= np.linalg.norm(u)
    v = np.cross(top_normal, u)
    angs = np.array([
        np.arctan2((p - top_centre) @ v, (p - top_centre) @ u)
        for p in top_verts_xyz
    ])
    cyclic_order = np.argsort(angs)
    for k, oi in enumerate(cyclic_order):
        p = top_verts_xyz[oi]
        # nudge outward
        out = top_centre + 1.18 * (p - top_centre) + 0.10 * top_normal
        ax.text(out[0], out[1], out[2], f"{k + 1}",
                color="#7a1f1f", fontsize=11, ha="center", va="center",
                weight="bold", zorder=30)

    # Twist label arrow on the top face
    c = top_centre
    n = top_normal
    rho = 0.40
    offset = 0.20 * n
    ang_arc = np.linspace(0, 0.65, 30)
    pts = (c[:, None] + offset[:, None] +
           rho * (np.cos(ang_arc)[None, :] * u[:, None]
                  + np.sin(ang_arc)[None, :] * v[:, None]))
    ax.plot(pts[0], pts[1], pts[2], color="#7a1f1f",
             linewidth=2.0, zorder=22)
    ax.scatter([pts[0, -1]], [pts[1, -1]], [pts[2, -1]],
                s=55, c="#7a1f1f", marker=">", zorder=23)

    ax.text2D(0.50, 0.96, "(a) regular dodecahedron",
              transform=ax.transAxes, ha="center", fontsize=11)
    ax.text2D(0.50, 0.04,
              "front face (red) and opposite back face (blue)",
              transform=ax.transAxes, ha="center", fontsize=9,
              color="#444444")

    ax.set_axis_off()
    # Camera angle: look slightly above the equator with a small azim
    # offset so the +x axis (our front-back face axis) is close to
    # left-right on screen, and both faces are visible at oblique
    # angles.
    ax.view_init(elev=18, azim=12)
    L = 1.5
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L)
    ax.set_box_aspect((1, 1, 1))

    # =========================================================
    # Panel (b): axis view, explicit pi/5 twist
    # =========================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_aspect("equal")
    ax2.set_axis_off()

    # Two regular pentagons centred at origin, radius 1.
    # Back pentagon (drawn first, behind):  vertices at angles
    #     90 deg, 162 deg, 234 deg, 306 deg, 18 deg  (i.e., 90 + 72*k).
    # Front pentagon: same vertices rotated by +36 deg = pi/5.
    R = 1.0
    twist_deg = 36.0
    back_angs = np.deg2rad(90 + 72 * np.arange(5))
    front_angs = back_angs + np.deg2rad(twist_deg)
    back_pts = np.column_stack([R * np.cos(back_angs),
                                R * np.sin(back_angs)])
    front_pts = np.column_stack([R * np.cos(front_angs),
                                 R * np.sin(front_angs)])

    # Draw back pentagon (blue, behind) with full opacity so its outline
    # is clear; draw front pentagon (red) on top with partial opacity so
    # the back outline still reads.
    back_patch = MplPolygon(back_pts, closed=True, facecolor=BACK_COLOR,
                            edgecolor="#1f3f70", linewidth=1.5, alpha=0.55)
    ax2.add_patch(back_patch)
    front_patch = MplPolygon(front_pts, closed=True, facecolor=FRONT_COLOR,
                             edgecolor="#7a1f1f", linewidth=1.5, alpha=0.55)
    ax2.add_patch(front_patch)

    # Vertex labels: back face vertices = 1'..5' (italic prime), front
    # face vertices = 1..5.  Place labels just outside each vertex.
    pad = 0.18
    for k, (bp, fp) in enumerate(zip(back_pts, front_pts)):
        bdir = bp / np.linalg.norm(bp)
        fdir = fp / np.linalg.norm(fp)
        ax2.text(bp[0] + pad * bdir[0], bp[1] + pad * bdir[1],
                 f"${k + 1}'$",
                 color="#1f3f70", fontsize=12, ha="center", va="center",
                 weight="bold")
        ax2.text(fp[0] + pad * fdir[0], fp[1] + pad * fdir[1],
                 f"${k + 1}$",
                 color="#7a1f1f", fontsize=12, ha="center", va="center",
                 weight="bold")

    # Mark vertices
    ax2.scatter(back_pts[:, 0], back_pts[:, 1],
                s=28, c="#1f3f70", zorder=4)
    ax2.scatter(front_pts[:, 0], front_pts[:, 1],
                s=28, c="#7a1f1f", zorder=5)

    # Explicit pi/5 = 36 deg arc, between back-vertex 1 (at 90 deg) and
    # front-vertex 1 (at 90 + 36 deg).  Drawn as a circular arc at
    # radius r_arc, with a degree label.
    r_arc = 0.42
    arc = Arc((0, 0), 2 * r_arc, 2 * r_arc,
              theta1=90.0, theta2=90.0 + twist_deg,
              edgecolor="#222", linewidth=1.6, zorder=6)
    ax2.add_patch(arc)
    # Tick marks at the two endpoints
    for theta_deg in (90.0, 90.0 + twist_deg):
        th = np.deg2rad(theta_deg)
        x0, y0 = (r_arc - 0.05) * np.cos(th), (r_arc - 0.05) * np.sin(th)
        x1, y1 = (r_arc + 0.05) * np.cos(th), (r_arc + 0.05) * np.sin(th)
        ax2.plot([x0, x1], [y0, y1], color="#222", linewidth=1.3)
    # Radial reference rays from origin to the two vertex-1 markers,
    # so the angle is unambiguous.
    ax2.plot([0, 0.85 * back_pts[0, 0]],
             [0, 0.85 * back_pts[0, 1]],
             color="#1f3f70", linewidth=0.9, linestyle="--", zorder=2)
    ax2.plot([0, 0.85 * front_pts[0, 0]],
             [0, 0.85 * front_pts[0, 1]],
             color="#7a1f1f", linewidth=0.9, linestyle="--", zorder=2)

    # Arc label, placed at the arc midpoint, slightly outside.
    mid_th = np.deg2rad(90.0 + twist_deg / 2)
    ax2.text((r_arc + 0.13) * np.cos(mid_th),
             (r_arc + 0.13) * np.sin(mid_th),
             r"$\pi/5 = 36^\circ$",
             color="#222", fontsize=11, ha="left", va="center",
             weight="bold")

    # Title and footer.  Extra vertical headroom below the pentagons
    # (ylim down to -1.85 instead of -1.5) gives the caption ~1.5
    # lines of clearance below the bottom vertex label "3".
    ax2.set_xlim(-1.7, 1.7)
    ax2.set_ylim(-1.85, 1.6)
    ax2.text(0.50, 0.96, "(b) view down the front--back axis",
             transform=ax2.transAxes, ha="center", fontsize=11)
    ax2.text(0.50, 0.04,
             r"front pentagon (red, vertices $1\!\ldots\!5$) is twisted by"
             "\n"
             r"$+\pi/5$ from back pentagon (blue, $1'\!\ldots\!5'$);"
             "\n"
             r"identification $k \leftrightarrow k'$ defines $S^3/2I$",
             transform=ax2.transAxes, ha="center", fontsize=8.5,
             color="#444444")

    # =========================================================
    # Panel (c): affine E8 Dynkin diagram with 2I irrep dims
    # =========================================================
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_aspect("equal")
    ax3.set_axis_off()

    # Affine E8-hat has 9 nodes with Coxeter labels (= 2I irrep dims)
    #     1 - 2 - 3 - 4 - 5 - 6 - 4 - 2
    #                         |
    #                         3
    # The branch attaches to the node with mark 6 (sixth from the
    # affine node).  Sum of squares = 1+4+9+16+25+36+16+4+9 = 120
    # = |2I|, confirming the McKay correspondence.
    #
    # Highlight rule: the 6 distinct dimensions {1,2,3,4,5,6} are the
    # carrier-knot crossing numbers n_q in Paper 6, all populated by
    # SM particles.  We mark the FIRST occurrence of each distinct
    # dim with a thicker outline; repeats in the chain are drawn
    # plain to avoid implying they are different particle slots.
    chain_marks = [1, 2, 3, 4, 5, 6, 4, 2]
    chain_x = list(range(len(chain_marks)))
    chain_y = [0] * len(chain_marks)
    branch_idx_in_chain = 5      # node with mark 6
    branch_mark = 3
    branch_x, branch_y = float(chain_x[branch_idx_in_chain]), 1.0

    # Edges in the chain
    for i in range(len(chain_marks) - 1):
        ax3.plot([chain_x[i], chain_x[i + 1]],
                 [chain_y[i], chain_y[i + 1]],
                 color="#333", linewidth=1.5, zorder=1)
    # Branch edge
    ax3.plot([chain_x[branch_idx_in_chain], branch_x],
             [chain_y[branch_idx_in_chain], branch_y],
             color="#333", linewidth=1.5, zorder=1)

    # Node radius and which marks have been "seen" (to highlight first
    # occurrence of each distinct dim)
    seen = set()
    NODE_R = 0.22

    def draw_node(x, y, mark, label_above=False, is_affine=False):
        first = mark not in seen
        seen.add(mark)
        # First-occurrence nodes use a saturated fill;
        # repeat-mark nodes use a paler fill but keep the dim label.
        if first:
            face = "#fff5d6"
            edge = "#a25b00"
            ew = 1.6
        else:
            face = "#ffffff"
            edge = "#666"
            ew = 1.0
        circ = Circle((x, y), NODE_R, facecolor=face, edgecolor=edge,
                      linewidth=ew, zorder=3)
        ax3.add_patch(circ)
        ax3.text(x, y, str(mark), ha="center", va="center",
                 fontsize=11, color="#1a1a1a", weight="bold", zorder=4)
        if is_affine:
            # Mark the affine node distinctively
            ax3.text(x, y - 0.55, "affine", ha="center", va="top",
                     fontsize=8, color="#a25b00")
        if label_above:
            ax3.text(x + 0.32, y, "branch", ha="left", va="center",
                     fontsize=8, color="#a25b00")

    for i, m in enumerate(chain_marks):
        draw_node(chain_x[i], chain_y[i], m, is_affine=(i == 0))
    draw_node(branch_x, branch_y, branch_mark, label_above=True)

    # Title and footer
    ax3.set_xlim(-0.7, len(chain_marks) - 0.3)
    # Extra headroom below "affine" so the caption clears it by ~1 line.
    ax3.set_ylim(-1.15, 1.7)
    ax3.text(0.5, 0.96,
             r"(c) affine Dynkin diagram of $\hat E_8$;"
             r" node label = dim of corresponding $2I$ irrep"
             r" (McKay)",
             transform=ax3.transAxes, ha="center", fontsize=10)
    ax3.text(0.5, 0.04,
             r"the 6 distinct dims $\{1,2,3,4,5,6\}$ are exactly the"
             r" carrier-knot crossing numbers $n_q$ in Paper 6"
             r" (gold-rimmed = first occurrence);"
             r"  $\sum (\dim_i)^2 = 120 = |2I|$",
             transform=ax3.transAxes, ha="center", fontsize=8.5,
             color="#444444")

    # Suptitle
    fig.suptitle(
        "Brieskorn--Poincaré sphere "
        r"$\Sigma(2,3,5) = S^3 / 2I$:  "
        r"$|\pi_1|=|2I|=120$,  $\lambda_1=168=7{\cdot}24$",
        fontsize=12, y=0.99,
    )

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: K_7 on Heffter triangular embedding (square fundamental domain)
# ---------------------------------------------------------------------------

def fig_K7_heffter(out_path: str) -> None:
    plt.rcParams.update(PUB_STYLE)
    fig, ax = plt.subplots(figsize=(6.0, 6.4))

    # Vertex placement: skew sublattice (k/7, 3k mod 7 / 7), gives the
    # Heffter triangular embedding when edges are drawn as shortest
    # toroidal images.
    coords = np.array([[(k % 7) / 7.0,
                         (3 * k % 7) / 7.0] for k in range(7)])

    # Square fundamental domain with identification arrows
    boundary_color = "#666666"
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0],
             color=boundary_color, linewidth=1.6)
    for x_arrow, y_arrow in [
        ((0.50, 0.55), (0, 0)),     # bottom edge double-arrow group
        ((0.50, 0.55), (1, 1)),     # top
    ]:
        ax.annotate("", xy=(0.55, y_arrow[0]), xytext=(0.45, y_arrow[0]),
                     arrowprops=dict(arrowstyle="->", color=boundary_color,
                                       lw=1.5))
    ax.annotate("", xy=(0, 0.55), xytext=(0, 0.45),
                 arrowprops=dict(arrowstyle="->", color=boundary_color, lw=1.5))
    ax.annotate("", xy=(1, 0.55), xytext=(1, 0.45),
                 arrowprops=dict(arrowstyle="->", color=boundary_color, lw=1.5))

    # Edge drawing helper:\ for each pair (i, j), find the shortest toroidal
    # image of j relative to i, draw the segment.  If the segment crosses
    # the boundary, also draw the periodic-image segment so connectivity
    # is visually clear.
    def shortest(i, j):
        p1 = coords[i]
        best = (None, None)
        best_d = np.inf
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                p2 = coords[j] + np.array([dx, dy])
                d = np.linalg.norm(p2 - p1)
                if d < best_d:
                    best = (p1.copy(), p2.copy())
                    best_d = d
        return best

    edge_color = "#1a1a1a"
    for i, j in combinations(range(7), 2):
        p1, p2 = shortest(i, j)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                 color=edge_color, linewidth=0.9, alpha=0.85)
        # If the wrapped p2 is outside the unit square, also draw the
        # mirror segment in the unit square (other half of the wrap).
        out_x = int(p2[0] < 0) - int(p2[0] > 1)
        out_y = int(p2[1] < 0) - int(p2[1] > 1)
        if out_x != 0 or out_y != 0:
            shift = np.array([out_x, out_y])
            ax.plot([p1[0] + shift[0], p2[0] + shift[0]],
                     [p1[1] + shift[1], p2[1] + shift[1]],
                     color=edge_color, linewidth=0.9, alpha=0.85)

    # Vertices on top
    for k, (x, y) in enumerate(coords):
        ax.plot(x, y, "o", markersize=12,
                 markerfacecolor="#f4ecd9",
                 markeredgecolor="#1a1a1a",
                 markeredgewidth=1.5, zorder=5)
        ax.text(x, y, str(k), color="#1a1a1a",
                 ha="center", va="center", fontsize=9,
                 fontweight="bold", zorder=6)

    ax.text(0.5, 1.10,
             r"$K_7$ Heffter triangular embedding on $T^2$",
             ha="center", fontsize=12, transform=ax.transData)
    ax.text(0.5, -0.10,
             r"$V=7$,  $E=21$,  $F=14$,  $\chi=0$,  genus $=1$",
             ha="center", fontsize=10, color="#444444",
             transform=ax.transData)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.18, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trefoil", action="store_true")
    p.add_argument("--poincare", action="store_true")
    p.add_argument("--k7", action="store_true")
    p.add_argument("--outdir", default=HERE)
    args = p.parse_args()

    do_all = not (args.trefoil or args.poincare or args.k7)

    if args.trefoil or do_all:
        fig_trefoil(os.path.join(args.outdir, "paper19_fig_trefoil.pdf"))
    if args.poincare or do_all:
        fig_poincare_sphere(os.path.join(args.outdir, "paper19_fig_poincare_sphere.pdf"))
    if args.k7 or do_all:
        fig_K7_heffter(os.path.join(args.outdir, "paper19_fig_K7_heffter.pdf"))


if __name__ == "__main__":
    main()
