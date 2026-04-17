#!/usr/bin/env python3
"""
Gauss linking integral on the CINQUEFOIL T(2,5) knot.

The trefoil T(2,3) has D₃ symmetry → 3 sectors, character √3/2.
The cinquefoil T(2,5) has D₅ symmetry → 5 sectors, character sin(2π/5).

If the AB phase construction is universal across the knot hierarchy,
the cinquefoil should show:
  • Lk = ±5 for z-plane loops (q = 5 poloidal passages)
  • D₅ Fourier selection rule (m = 0 mod 5 only)
  • Character amplitude sin(2π/5) ≈ 0.9511

This tests whether the seven-step derivation generalizes from
trefoil (→ α_SM) to cinquefoil (→ α_GUT).

BONUS: also compute the HEPTAFOIL T(2,7) for the full hierarchy.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import time


def torus_knot_curve(p, q, N, R=None, a_torus=0.3):
    """T(p,q) centerline + derivatives."""
    if R is None:
        R = np.pi**2
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    dt = t[1] - t[0]
    a = a_torus

    x = (R + a*R*np.cos(q*t)) * np.cos(p*t)
    y = (R + a*R*np.cos(q*t)) * np.sin(p*t)
    z = a * R * np.sin(q*t)

    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    dz = np.gradient(z, dt)

    speed = np.sqrt(dx**2 + dy**2 + dz**2)

    from scipy.integrate import cumulative_trapezoid
    s = np.zeros(N)
    s[1:] = cumulative_trapezoid(speed, t)
    L = s[-1] + speed[-1] * dt

    Tx = dx / speed
    Ty = dy / speed
    Tz = dz / speed

    return (x, y, z, dx, dy, dz, Tx, Ty, Tz, speed, s, L, dt)


def circle_loop(center, radius, normal, N_pts):
    """Circular Wilson loop."""
    normal = np.array(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    if abs(normal[2]) < 0.9:
        v1 = np.cross(normal, [0, 0, 1])
    else:
        v1 = np.cross(normal, [1, 0, 0])
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)

    theta = np.linspace(0, 2 * np.pi, N_pts, endpoint=False)
    dt = theta[1] - theta[0]
    center = np.array(center, dtype=float)

    cx = center[0] + radius * (np.cos(theta)*v1[0] + np.sin(theta)*v2[0])
    cy = center[1] + radius * (np.cos(theta)*v1[1] + np.sin(theta)*v2[1])
    cz = center[2] + radius * (np.cos(theta)*v1[2] + np.sin(theta)*v2[2])

    dcx = np.gradient(cx, dt)
    dcy = np.gradient(cy, dt)
    dcz = np.gradient(cz, dt)

    return cx, cy, cz, dcx, dcy, dcz, dt


def gauss_linking(cx, cy, cz, dcx, dcy, dcz, dt_c,
                   kx, ky, kz, dkx, dky, dkz, dt_k):
    """Gauss linking integral (vectorized over K)."""
    N_c = len(cx)
    Lk = 0.0

    for i in range(N_c):
        rx = cx[i] - kx
        ry = cy[i] - ky
        rz = cz[i] - kz

        dist3 = (rx**2 + ry**2 + rz**2)**1.5
        dist3 = np.maximum(dist3, 1e-30)

        cross_x = dcy[i] * dkz - dcz[i] * dky
        cross_y = dcz[i] * dkx - dcx[i] * dkz
        cross_z = dcx[i] * dky - dcy[i] * dkx

        dot = rx * cross_x + ry * cross_y + rz * cross_z
        Lk += np.sum(dot / dist3) * dt_c * dt_k

    return Lk / (4 * np.pi)


def analyze_knot(p, q, label, R=None, a_torus=0.3, N_knot=2048):
    """Full holonomy analysis for T(p,q)."""
    if R is None:
        R = np.pi**2

    kx, ky, kz, dkx, dky, dkz, Tx, Ty, Tz, speed, s, L, dt_k = \
        torus_knot_curve(p, q, N_knot, R, a_torus)

    q_sym = q  # the D_q symmetry order
    dihedral = f"D_{q_sym}"
    char_amp = np.sin(2 * np.pi / q_sym)
    sector_angle = 2 * np.pi / q_sym

    print(f"\n{'='*72}")
    print(f"  T({p},{q}) — {label}")
    print(f"  Symmetry: {dihedral}, {q_sym} sectors of {360/q_sym:.1f}°")
    print(f"  Character amplitude: sin(2π/{q_sym}) = {char_amp:.6f}")
    print(f"  Arc length L = {L:.4f}")
    print(f"{'='*72}")

    # ── Test 1: Single-strand linking ─────────────────────
    print(f"\n  Test 1: Single-strand Wilson loop (ρ = 2ξ)")

    i_ref = N_knot // 4
    r0 = np.array([kx[i_ref], ky[i_ref], kz[i_ref]])
    T = np.array([Tx[i_ref], Ty[i_ref], Tz[i_ref]])

    cx, cy, cz, dcx, dcy, dcz, dt_c = circle_loop(r0, 2.0, T, 256)
    Lk = gauss_linking(cx, cy, cz, dcx, dcy, dcz, dt_c,
                        kx, ky, kz, dkx, dky, dkz, dt_k)
    print(f"    Lk = {Lk:.5f}  (expect 1.000)")

    # ── Test 2: z-plane loop (total linking) ──────────────
    print(f"\n  Test 2: z-plane loop (total linking number)")
    print(f"    T({p},{q}) winds q={q} times around minor circle")
    print(f"    → z-plane linking = ±{q}\n")

    for loop_r in [8.0, 10.0, 12.0, 15.0]:
        cx, cy, cz, dcx, dcy, dcz, dt_c = circle_loop(
            [0, 0, 0], loop_r, [0, 0, 1], 512)
        Lk = gauss_linking(cx, cy, cz, dcx, dcy, dcz, dt_c,
                           kx, ky, kz, dkx, dky, dkz, dt_k)
        print(f"    r = {loop_r:5.1f}: Lk = {Lk:8.4f}  "
              f"(expect ±{q})")

    # ── Test 3: Fourier analysis (D_q structure) ──────────
    print(f"\n  Test 3: Fourier analysis ({dihedral} structure)")

    N_sample = q_sym * 12  # multiple of q for clean Fourier
    phase_samples = np.zeros(N_sample)

    for k in range(N_sample):
        i_s = int(k * N_knot / N_sample) % N_knot
        r_pt = np.array([kx[i_s], ky[i_s], kz[i_s]])
        T_pt = np.array([Tx[i_s], Ty[i_s], Tz[i_s]])

        cx, cy, cz, dcx, dcy, dcz, dt_c = circle_loop(
            r_pt, 2.0, T_pt, 64)
        Lk = gauss_linking(cx, cy, cz, dcx, dcy, dcz, dt_c,
                           kx, ky, kz, dkx, dky, dkz, dt_k)
        phase_samples[k] = 2 * np.pi * Lk

    fft = np.fft.fft(phase_samples) / N_sample
    c0 = abs(fft[0])

    print(f"\n    {'mode':>6} {'|c_m|':>10} {'|c_m|/c_0':>10}  notes")
    print(f"    {'─'*6} {'─'*10} {'─'*10}  {'─'*30}")

    for m in range(min(3 * q_sym + 1, N_sample // 2)):
        cm = abs(fft[m])
        ratio = cm / c0 if c0 > 0 else 0
        tag = ""
        if m == 0:
            tag = f"DC (expect 2π = {2*np.pi:.4f})"
        elif m == q_sym:
            tag = f"★ {dihedral} fundamental"
        elif m == 2 * q_sym:
            tag = f"{dihedral} 2nd harmonic"
        elif m == 3 * q_sym:
            tag = f"{dihedral} 3rd harmonic"
        elif ratio > 0.001:
            tag = "nonzero"

        if m <= 2 * q_sym + 1 or ratio > 0.001:
            print(f"    {m:6d} {cm:10.6f} {ratio:10.6f}  {tag}")

    # ── AB phase construction ─────────────────────────────
    print(f"\n  AB PHASE CONSTRUCTION for T({p},{q}):")
    ab_per_sector = 2 * np.pi / q_sym
    phase_per_sector = ab_per_sector * char_amp
    total_phase = q_sym * phase_per_sector

    print(f"    Sectors: {q_sym}")
    print(f"    AB phase per sector: 2π/{q_sym} = {ab_per_sector:.6f}")
    print(f"    {dihedral} character: sin(2π/{q_sym}) = {char_amp:.6f}")
    print(f"    Effective phase per sector: {phase_per_sector:.6f}")
    print(f"    Total geometric phase: {q_sym} × {phase_per_sector:.4f}"
          f" = {total_phase:.6f}")
    print(f"    = 2π × sin(2π/{q_sym}) = {total_phase:.6f}")
    print(f"    (trefoil had: π√3 = {np.pi*np.sqrt(3):.6f})")

    return total_phase


def main():
    R = np.pi**2
    a_torus = 0.3

    print("=" * 72)
    print("KNOT HIERARCHY: GAUSS LINKING ON TREFOIL → CINQUEFOIL → HEPTAFOIL")
    print("=" * 72)

    results = {}

    for p, q, label in [(2, 3, "trefoil"), (2, 5, "cinquefoil"),
                         (2, 7, "heptafoil"), (2, 9, "nonafoil")]:
        phase = analyze_knot(p, q, label, R, a_torus)
        results[(p, q)] = phase

    # ── Hierarchy comparison ──────────────────────────────
    print(f"\n{'='*72}")
    print(f"KNOT HIERARCHY: AB PHASE CONSTRUCTION COMPARISON")
    print(f"{'='*72}")

    print(f"\n  {'Knot':<14} {'q':>3} {'D_q':>5} {'sin(2π/q)':>10} "
          f"{'Φ_total':>10} {'q²×Φ+1':>10} {'1/α_layer':>10}")
    print(f"  {'─'*14} {'─'*3} {'─'*5} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for (p, q), phase in sorted(results.items()):
        dq = f"D_{q}"
        char = np.sin(2 * np.pi / q)
        # Following the seven-step pattern:
        # 1/α = q²_next × Φ_total + N_wind
        # For trefoil: q²_next = q²_cinq = 25, Φ = π√3, 1/α = 25π√3+1 = 137.035
        # For cinquefoil: q²_next = q²_hept = 49?, Φ = 2π sin(2π/5)
        q_next = q + 2  # next odd number
        q2_next = q_next**2
        alpha_inv = q2_next * phase + 1

        print(f"  T(2,{q}) {['trefoil','cinquefoil','heptafoil','nonafoil'][(q-3)//2]:<10} "
              f"{q:3d} {dq:>5} {char:10.6f} {phase:10.6f} "
              f"{alpha_inv:10.4f} {alpha_inv:10.4f}")

    # ── The key prediction ────────────────────────────────
    print(f"\n{'='*72}")
    print(f"THE KNOT HIERARCHY COUPLING LADDER")
    print(f"{'='*72}")

    phase_tref = results[(2, 3)]
    phase_cinq = results[(2, 5)]
    phase_hept = results.get((2, 7), 0)

    # Trefoil: 1/α_SM = 25 × π√3 + 1 = 137.035
    alpha_inv_SM = 25 * phase_tref + 1
    print(f"\n  TREFOIL (SM layer):")
    print(f"    Φ = {phase_tref:.6f} = π√3 = {np.pi*np.sqrt(3):.6f}")
    print(f"    1/α_SM = q²_cinq × Φ + 1 = 25 × {phase_tref:.4f} + 1"
          f" = {alpha_inv_SM:.4f}")
    print(f"    CODATA: 137.036  →  {abs(alpha_inv_SM-137.036)/137.036*1e6:.1f} ppm")

    # Cinquefoil: 1/α_GUT = 49 × 2π sin(2π/5) + 1 ?
    alpha_inv_GUT = 49 * phase_cinq + 1
    print(f"\n  CINQUEFOIL (GUT layer):")
    print(f"    Φ = {phase_cinq:.6f} = 2π sin(2π/5)"
          f" = {2*np.pi*np.sin(2*np.pi/5):.6f}")
    print(f"    1/α_GUT = q²_hept × Φ + 1 = 49 × {phase_cinq:.4f} + 1"
          f" = {alpha_inv_GUT:.4f}")
    print(f"    Paper 10 prediction: 1/α_GUT ≈ 40.2")
    print(f"    NWT AB prediction: α_GUT = 3/(25π) → 1/α = {25*np.pi/3:.4f}")

    # Heptafoil
    if (2, 7) in results:
        alpha_inv_pre = 81 * phase_hept + 1
        print(f"\n  HEPTAFOIL (pre-GUT layer):")
        print(f"    Φ = {phase_hept:.6f} = 2π sin(2π/7)"
              f" = {2*np.pi*np.sin(2*np.pi/7):.6f}")
        print(f"    1/α_preGUT = q²_nona × Φ + 1 = 81 × {phase_hept:.4f} + 1"
              f" = {alpha_inv_pre:.4f}")

    print(f"\n  PATTERN: 1/α_layer = q²_next_layer × 2π sin(2π/q) + 1")
    print(f"  where q is the poloidal winding of the current layer's knot")
    print(f"  and q_next = q + 2 is the next torus knot in the hierarchy.")
    print(f"\n  This is a UNIVERSAL AB-phase construction that applies to")
    print(f"  every layer of the NWT knot hierarchy, not just the trefoil.")


if __name__ == "__main__":
    main()
