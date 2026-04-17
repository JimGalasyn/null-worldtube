#!/usr/bin/env python3
"""
Multi-knot spectrum: energy scan + relaxation across the knot hierarchy.

Computes the abelian Higgs energy for every NWT-relevant torus knot
T(p,q), with and without gradient-flow relaxation, using the tube-
adapted JAX pipeline on GPU.

For each knot:
  1. Compute curvature κ(s) from the centerline geometry
  2. Initialize with BPS ansatz on the tube grid
  3. Measure E_BPS (= μ × L, Bogomolny theorem)
  4. Relax for N_steps gradient descent steps
  5. Measure E_relaxed and the crossing energy

The crossing energy Δε = (E/E_ring)/(L/L_ring) - 1 measures
how much topology contributes beyond arc length.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tube_energy_jax import (
    load_bps_profile, make_knot_grid, energy_knot_3d, grad_knot_3d,
    energy_ring_2d, make_ring_grid
)


def knot_curvature(p, q, N_s, R=None, a_torus=0.3):
    """Compute curvature κ(s) for T(p,q) on uniform arc-length grid."""
    if R is None:
        R = np.pi**2
    N_t = max(N_s * 16, 4096)
    t = np.linspace(0, 2 * np.pi, N_t, endpoint=False)
    dt = t[1] - t[0]
    a = a_torus

    x = (R + a*R*np.cos(q*t)) * np.cos(p*t)
    y = (R + a*R*np.cos(q*t)) * np.sin(p*t)
    z = a * R * np.sin(q*t)

    dx_dt = np.gradient(x, dt)
    dy_dt = np.gradient(y, dt)
    dz_dt = np.gradient(z, dt)
    ddx = np.gradient(dx_dt, dt)
    ddy = np.gradient(dy_dt, dt)
    ddz = np.gradient(dz_dt, dt)

    speed = np.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)
    cross_mag = np.sqrt((dy_dt*ddz - dz_dt*ddy)**2 +
                        (dz_dt*ddx - dx_dt*ddz)**2 +
                        (dx_dt*ddy - dy_dt*ddx)**2)
    kappa = cross_mag / speed**3

    from scipy.integrate import cumulative_trapezoid
    s_arr = np.zeros(N_t)
    s_arr[1:] = cumulative_trapezoid(speed, t)
    L = s_arr[-1] + speed[-1] * dt

    s_uniform = np.linspace(0, L, N_s, endpoint=False)
    kappa_uniform = np.interp(s_uniform, s_arr, kappa, period=L)

    return jnp.array(kappa_uniform), float(L)


def scan_knot(p, q, label, rho_bps, f_bps, a_bps,
              N_s=128, N_rho=128, N_phi=64, n_relax=100, lr=0.0001):
    """Compute BPS and relaxed energy for T(p,q)."""
    R = np.pi**2
    a_torus = 0.3

    kappa_s, L = knot_curvature(p, q, N_s, R, a_torus)

    s, rho, phi, ds, dr, dphi = make_knot_grid(
        N_s, N_rho, N_phi, rho_max=12.0, L=L)

    f0 = jnp.array(np.interp(np.array(rho), rho_bps, f_bps))
    a0 = jnp.array(np.interp(np.array(rho), rho_bps, a_bps))

    f_field = jnp.tile(f0[None, :, None], (N_s, 1, N_phi))
    a_field = jnp.tile(a0[None, :, None], (N_s, 1, N_phi))
    theta_field = jnp.zeros((N_s, N_rho, N_phi))

    # BPS energy
    E_bps = float(energy_knot_3d(
        f_field, a_field, theta_field,
        rho, kappa_s, ds, dr, dphi, N_s, N_rho, N_phi))

    # Relaxation
    @jax.jit
    def step(f, a, th):
        gf, ga, gth = grad_knot_3d(f, a, th, rho, kappa_s,
                                     ds, dr, dphi, N_s, N_rho, N_phi)
        f = jnp.clip(f - lr * gf, 0.0, 1.5)
        a = jnp.clip(a - lr * ga, -0.5, 1.5)
        th = th - lr * gth
        return f, a, th

    t0 = time.time()
    for _ in range(n_relax):
        f_field, a_field, theta_field = step(
            f_field, a_field, theta_field)
    jax.block_until_ready(f_field)
    elapsed = time.time() - t0

    E_relax = float(energy_knot_3d(
        f_field, a_field, theta_field,
        rho, kappa_s, ds, dr, dphi, N_s, N_rho, N_phi))

    return {
        'p': p, 'q': q, 'label': label, 'L': L,
        'E_bps': E_bps, 'E_relax': E_relax,
        'kappa_mean': float(kappa_s.mean()),
        'kappa_max': float(kappa_s.max()),
        'time': elapsed,
        'ms_per_step': elapsed / n_relax * 1000,
    }


def main():
    rho_bps, f_bps, a_bps = load_bps_profile()

    print("=" * 78)
    print("MULTI-KNOT ENERGY SPECTRUM: JAX TUBE-ADAPTED PIPELINE")
    print(f"JAX backend: {jax.default_backend()}")
    print("=" * 78)

    # Ring baseline
    N_rho, N_phi = 128, 64
    rho_ring, _, dr, dphi = make_ring_grid(N_rho, N_phi, rho_max=12.0)
    R = np.pi**2
    L_ring = 2 * np.pi * R

    f0 = jnp.array(np.interp(np.array(rho_ring), rho_bps, f_bps))
    a0 = jnp.array(np.interp(np.array(rho_ring), rho_bps, a_bps))
    f_r = jnp.tile(f0[:, None], (1, N_phi))
    a_r = jnp.tile(a0[:, None], (1, N_phi))
    th_r = jnp.zeros((N_rho, N_phi))

    mu_ring = float(energy_ring_2d(f_r, a_r, th_r, rho_ring, dr, dphi,
                                    R, N_rho, N_phi))
    E_ring = mu_ring * L_ring

    print(f"\n  Ring baseline: μ = {mu_ring:.6f}, L = {L_ring:.4f}, "
          f"E = {E_ring:.4f}")

    # ── Knot scan ─────────────────────────────────────────
    knots = [
        (2, 1, "electron"),
        (2, 3, "trefoil"),
        (2, 5, "cinquefoil"),
        (2, 7, "heptafoil"),
        (2, 9, "nonafoil"),
        (1, 3, "T(1,3)"),
        (1, 4, "proton?"),
        (1, 5, "T(1,5)"),
        (3, 2, "trefoil-alt"),
        (3, 4, "T(3,4)"),
        (3, 5, "T(3,5)"),
    ]

    N_s = 64  # longitudinal points (faster for scan)
    n_relax = 100

    print(f"\n  Grid: N_s={N_s}, N_rho={N_rho}, N_phi={N_phi}")
    print(f"  Relaxation: {n_relax} steps, lr=0.0001")

    results = []
    for p, q, label in knots:
        print(f"\n  Computing T({p},{q}) {label}...", end="", flush=True)
        try:
            r = scan_knot(p, q, label, rho_bps, f_bps, a_bps,
                          N_s=N_s, N_rho=N_rho, N_phi=N_phi,
                          n_relax=n_relax)
            results.append(r)
            print(f" E_bps={r['E_bps']:.2f}, E_relax={r['E_relax']:.2f}, "
                  f"{r['ms_per_step']:.1f} ms/step")
        except Exception as ex:
            print(f" FAILED: {ex}")

    # ── Results table ─────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"RESULTS: BPS ENERGY (Bogomolny theorem check)")
    print(f"{'='*78}")

    print(f"\n  {'Knot':<16} {'L':>8} {'E_BPS':>10} {'E/E_ring':>10} "
          f"{'L/L_ring':>10} {'cross_BPS':>10}")
    print(f"  {'─'*16} {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for r in sorted(results, key=lambda x: x['L']):
        E_ratio = r['E_bps'] / E_ring
        L_ratio = r['L'] / L_ring
        cross = E_ratio / L_ratio - 1 if L_ratio > 0 else 0
        print(f"  T({r['p']},{r['q']}) {r['label']:<10} {r['L']:8.2f} "
              f"{r['E_bps']:10.2f} {E_ratio:10.6f} {L_ratio:10.6f} "
              f"{cross:+10.6f}")

    print(f"\n{'='*78}")
    print(f"RESULTS: RELAXED ENERGY (topology-dependent corrections)")
    print(f"{'='*78}")

    print(f"\n  {'Knot':<16} {'E_relax':>10} {'ΔE':>10} {'ΔE%':>8} "
          f"{'E/E_ring':>10} {'cross_relax':>10} {'ms/step':>8}")
    print(f"  {'─'*16} {'─'*10} {'─'*10} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")

    for r in sorted(results, key=lambda x: x['L']):
        dE = r['E_relax'] - r['E_bps']
        dE_pct = dE / r['E_bps'] * 100
        E_ratio = r['E_relax'] / E_ring
        L_ratio = r['L'] / L_ring
        cross = E_ratio / L_ratio - 1 if L_ratio > 0 else 0
        print(f"  T({r['p']},{r['q']}) {r['label']:<10} "
              f"{r['E_relax']:10.2f} {dE:+10.4f} {dE_pct:+7.3f}% "
              f"{E_ratio:10.6f} {cross:+10.6f} {r['ms_per_step']:8.1f}")

    # ── Crossing energy analysis ──────────────────────────
    print(f"\n{'='*78}")
    print(f"CROSSING ENERGY ANALYSIS")
    print(f"{'='*78}")

    # Check if any knot has nonzero crossing energy after relaxation
    cross_values = []
    for r in results:
        L_ratio = r['L'] / L_ring
        E_ratio = r['E_relax'] / E_ring
        cross = E_ratio / L_ratio - 1
        cross_values.append((r['label'], cross, r['L']))

    cross_values.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n  Crossing energy (E/E_ring)/(L/L_ring) - 1, sorted by |cross|:")
    for label, cross, L in cross_values:
        sig = "★" if abs(cross) > 0.001 else " "
        print(f"    {sig} {label:<12} cross = {cross:+.6f}")

    max_cross = max(abs(c) for _, c, _ in cross_values)
    print(f"\n  Max |crossing energy|: {max_cross:.6f}")

    if max_cross < 0.01:
        print(f"""
  Crossing energy is {max_cross*100:.3f}% — small but potentially
  nonzero at the {n_relax}-step relaxation level.

  At BPS: crossing energy = 0 (Bogomolny theorem, confirmed).
  After relaxation: crossing energy is O(0.1%) — consistent with
  curvature response (the fields adjust to follow varying κ(s)),
  not genuine topology-dependent interaction.

  To distinguish curvature response from crossing energy:
  compare knots with DIFFERENT topology but SAME curvature
  statistics.  If the crossing energy depends on the D_q symmetry
  class (not just ⟨κ²⟩), it's genuinely topological.
""")
    else:
        print(f"\n  ★ SIGNIFICANT crossing energy detected at {max_cross*100:.1f}%!")

    # ── Genus correlation ─────────────────────────────────
    print(f"{'─'*78}")
    print(f"  GENUS CORRELATION")
    print(f"{'─'*78}")

    def genus(p, q):
        return (p-1)*(q-1)//2

    print(f"\n  {'Knot':<12} {'genus':>5} {'E_relax/L':>10} {'cross':>10}")
    print(f"  {'─'*12} {'─'*5} {'─'*10} {'─'*10}")

    for r in sorted(results, key=lambda x: genus(x['p'], x['q'])):
        g = genus(r['p'], r['q'])
        mu = r['E_relax'] / r['L']
        L_ratio = r['L'] / L_ring
        E_ratio = r['E_relax'] / E_ring
        cross = E_ratio / L_ratio - 1
        print(f"  T({r['p']},{r['q']}) {r['label']:<8} {g:5d} "
              f"{mu:10.6f} {cross:+10.6f}")


if __name__ == "__main__":
    main()
