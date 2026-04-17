#!/usr/bin/env python3
"""
Phase B: Trefoil T(2,3) vortex on the tube-adapted JAX grid.

Uses the 3D energy_knot_3d function with curvature κ(s) computed
from the trefoil centerline.  Compares to the circular ring to
measure crossing energy (the topology-dependent part).

Grid: N_s × N_rho × N_phi in tube coordinates.
The trefoil has D₃ symmetry → curvature varies with period L/3.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
from pathlib import Path
from functools import partial

sys.path.insert(0, str(Path(__file__).parent))

from tube_energy_jax import (
    load_bps_profile, make_knot_grid, energy_knot_3d, grad_knot_3d,
    energy_ring_2d, make_ring_grid
)


def trefoil_curvature(N_s, R=None, a_torus=0.3):
    """Compute curvature κ(s) for T(2,3) on uniform arc-length grid."""
    if R is None:
        R = np.pi**2
    N_t = max(N_s * 16, 4096)
    t = np.linspace(0, 2 * np.pi, N_t, endpoint=False)
    dt = t[1] - t[0]
    a = a_torus

    x = (R + a*R*np.cos(3*t)) * np.cos(2*t)
    y = (R + a*R*np.cos(3*t)) * np.sin(2*t)
    z = a * R * np.sin(3*t)

    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    dz = np.gradient(z, dt)
    ddx = np.gradient(dx, dt)
    ddy = np.gradient(dy, dt)
    ddz = np.gradient(dz, dt)

    speed = np.sqrt(dx**2 + dy**2 + dz**2)
    cross_mag = np.sqrt((dy*ddz - dz*ddy)**2 +
                        (dz*ddx - dx*ddz)**2 +
                        (dx*ddy - dy*ddx)**2)
    kappa = cross_mag / speed**3

    from scipy.integrate import cumulative_trapezoid
    s_arr = np.zeros(N_t)
    s_arr[1:] = cumulative_trapezoid(speed, t)
    L = s_arr[-1] + speed[-1] * dt

    s_uniform = np.linspace(0, L, N_s, endpoint=False)
    kappa_uniform = np.interp(s_uniform, s_arr, kappa, period=L)

    return jnp.array(kappa_uniform), float(L)


def main():
    rho_bps, f_bps, a_bps = load_bps_profile()

    R = np.pi**2
    a_torus = 0.3

    print("=" * 72)
    print("PHASE B: TREFOIL T(2,3) ON TUBE-ADAPTED JAX GRID")
    print(f"JAX backend: {jax.default_backend()}")
    print("=" * 72)

    # ── Circular ring baseline ────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  BASELINE: Circular ring (R = π² = {R:.4f})")
    print(f"{'─'*72}")

    N_rho, N_phi = 128, 64
    rho_ring, phi_ring, dr, dphi = make_ring_grid(N_rho, N_phi, rho_max=12.0)
    L_ring = 2 * np.pi * R

    f0 = jnp.array(np.interp(np.array(rho_ring), rho_bps, f_bps))
    a0 = jnp.array(np.interp(np.array(rho_ring), rho_bps, a_bps))

    f_ring = jnp.tile(f0[:, None], (1, N_phi))
    a_ring = jnp.tile(a0[:, None], (1, N_phi))
    th_ring = jnp.zeros((N_rho, N_phi))

    mu_ring = float(energy_ring_2d(f_ring, a_ring, th_ring,
                                    rho_ring, dr, dphi, R, N_rho, N_phi))
    E_ring = mu_ring * L_ring

    print(f"  μ_ring = {mu_ring:.6f}  (π = {np.pi:.6f})")
    print(f"  L_ring = {L_ring:.4f}")
    print(f"  E_ring = μ × L = {E_ring:.4f}")

    # ── Trefoil ───────────────────────────────────────────
    for N_s in [64, 128, 256]:
        print(f"\n{'─'*72}")
        print(f"  TREFOIL T(2,3): N_s={N_s}, N_rho={N_rho}, N_phi={N_phi}")
        print(f"  DOF = {N_s * N_rho * N_phi:,} × 3 fields")
        print(f"{'─'*72}")

        kappa_s, L_tref = trefoil_curvature(N_s, R, a_torus)

        print(f"  L_trefoil = {L_tref:.4f}")
        print(f"  κ: min={float(kappa_s.min()):.5f}, "
              f"max={float(kappa_s.max()):.5f}, "
              f"mean={float(kappa_s.mean()):.5f}")

        s, rho, phi, ds_val, dr_val, dphi_val = make_knot_grid(
            N_s, N_rho, N_phi, rho_max=12.0, L=L_tref)

        # BPS initialization: f = f₀(ρ), a = a₀(ρ), θ = 0
        f0_rho = jnp.array(np.interp(np.array(rho), rho_bps, f_bps))
        a0_rho = jnp.array(np.interp(np.array(rho), rho_bps, a_bps))

        f_field = jnp.tile(f0_rho[None, :, None], (N_s, 1, N_phi))
        a_field = jnp.tile(a0_rho[None, :, None], (N_s, 1, N_phi))
        theta_field = jnp.zeros((N_s, N_rho, N_phi))

        # Energy
        t0 = time.time()
        E_tref = float(energy_knot_3d(
            f_field, a_field, theta_field,
            rho, kappa_s, ds_val, dr_val, dphi_val,
            N_s, N_rho, N_phi))
        dt_energy = time.time() - t0

        mu_tref = E_tref / L_tref

        print(f"  E_trefoil = {E_tref:.4f}  [{dt_energy:.3f}s]")
        print(f"  μ_trefoil = E/L = {mu_tref:.6f}")
        print(f"  μ_tref/μ_ring = {mu_tref/mu_ring:.8f}")
        print(f"  E_tref/E_ring = {E_tref/E_ring:.8f}")
        print(f"  L_tref/L_ring = {L_tref/L_ring:.8f}")
        cross = E_tref/E_ring / (L_tref/L_ring) - 1
        print(f"  Crossing energy: {cross:+.8f}")

        # Gradient
        t0 = time.time()
        grads = grad_knot_3d(
            f_field, a_field, theta_field,
            rho, kappa_s, ds_val, dr_val, dphi_val,
            N_s, N_rho, N_phi)
        dt_grad = time.time() - t0
        max_grad = max(float(jnp.max(jnp.abs(g))) for g in grads)
        print(f"  max|grad| = {max_grad:.6f}  [{dt_grad:.3f}s]")

    # ── Relaxation at N_s=128 ─────────────────────────────
    N_s = 128
    print(f"\n{'='*72}")
    print(f"  GRADIENT DESCENT: trefoil at N_s={N_s}")
    print(f"{'='*72}")

    kappa_s, L_tref = trefoil_curvature(N_s, R, a_torus)
    s, rho, phi, ds_val, dr_val, dphi_val = make_knot_grid(
        N_s, N_rho, N_phi, rho_max=12.0, L=L_tref)

    f0_rho = jnp.array(np.interp(np.array(rho), rho_bps, f_bps))
    a0_rho = jnp.array(np.interp(np.array(rho), rho_bps, a_bps))

    f_field = jnp.tile(f0_rho[None, :, None], (N_s, 1, N_phi))
    a_field = jnp.tile(a0_rho[None, :, None], (N_s, 1, N_phi))
    theta_field = jnp.zeros((N_s, N_rho, N_phi))

    E_init = float(energy_knot_3d(
        f_field, a_field, theta_field,
        rho, kappa_s, ds_val, dr_val, dphi_val,
        N_s, N_rho, N_phi))

    lr = 0.0001

    @jax.jit
    def step(f, a, th):
        gf, ga, gth = grad_knot_3d(f, a, th, rho, kappa_s,
                                     ds_val, dr_val, dphi_val,
                                     N_s, N_rho, N_phi)
        f = jnp.clip(f - lr * gf, 0.0, 1.5)
        a = jnp.clip(a - lr * ga, -0.5, 1.5)
        th = th - lr * gth
        return f, a, th

    print(f"\n  DOF = {N_s * N_rho * N_phi * 3:,}")
    print(f"  lr = {lr}")
    print(f"  {'Step':>6} {'E':>12} {'ΔE':>12} {'E/E_ring':>10} "
          f"{'cross':>10}")
    print(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")

    E_prev = E_init
    t0 = time.time()

    for i in range(100):
        f_field, a_field, theta_field = step(
            f_field, a_field, theta_field)

        if i % 20 == 0 or i == 99:
            E = float(energy_knot_3d(
                f_field, a_field, theta_field,
                rho, kappa_s, ds_val, dr_val, dphi_val,
                N_s, N_rho, N_phi))
            cross = (E / E_ring) / (L_tref / L_ring) - 1
            print(f"  {i:6d} {E:12.4f} {E - E_prev:+12.4f} "
                  f"{E/E_ring:10.6f} {cross:+10.6f}")
            E_prev = E

    elapsed = time.time() - t0
    E_final = float(energy_knot_3d(
        f_field, a_field, theta_field,
        rho, kappa_s, ds_val, dr_val, dphi_val,
        N_s, N_rho, N_phi))

    print(f"\n  100 steps in {elapsed:.2f}s ({elapsed/100*1000:.1f} ms/step)")
    print(f"  E_init  = {E_init:.4f}")
    print(f"  E_final = {E_final:.4f}")
    print(f"  ΔE = {E_final - E_init:+.4f} ({(E_final-E_init)/E_init*100:+.4f}%)")

    cross_init = (E_init / E_ring) / (L_tref / L_ring) - 1
    cross_final = (E_final / E_ring) / (L_tref / L_ring) - 1
    print(f"\n  Crossing energy (init):  {cross_init:+.8f}")
    print(f"  Crossing energy (final): {cross_final:+.8f}")

    print(f"\n{'='*72}")
    print(f"INTERPRETATION")
    print(f"{'='*72}")
    print(f"""
  The BPS ansatz on the trefoil gives E ∝ L (crossing energy ≈ 0),
  consistent with Phases A-D.  Gradient descent allows the fields
  to respond to the trefoil's varying curvature κ(s).

  If crossing energy remains ≈ 0 after relaxation: Bogomolny theorem
  confirmed in the full 3D JAX pipeline.

  If crossing energy becomes nonzero: the tube-adapted relaxation
  finds topology-dependent corrections that the BPS ansatz misses.

  Either way, this is the first FULL 3D RELAXATION on a knotted
  vortex with GPU-accelerated autodiff — the infrastructure for
  the complete particle mass spectrum computation.
""")


if __name__ == "__main__":
    main()
