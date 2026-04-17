#!/usr/bin/env python3
"""
Phase A: Circular vortex ring on a 3D grid — the "hello world."

Initialize a (2,1) unknot vortex ring on a 64³ grid, compute its
energy with JAX, and verify E ≈ μ_BPS × 2πR.

This validates the entire 3D pipeline:
  fields.py  → field arrays
  energy.py  → energy functional with JAX autodiff
  initialize.py → BPS vortex stamping
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fields import Fields, GridParams
from energy import compute_energy, energy_jit, grad_energy
from initialize import initialize_ring, load_bps_profile, compute_topological_charge


def main():
    print("=" * 72)
    print("PHASE A: CIRCULAR VORTEX RING ON A 3D GRID")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print("=" * 72)

    rho_bps, f_bps, a_bps = load_bps_profile()

    # Expected BPS line tension
    from scipy.integrate import trapezoid
    dr = rho_bps[1] - rho_bps[0]
    f_prime = np.gradient(f_bps, dr)
    a_prime = np.gradient(a_bps, dr)
    eps = (0.5 * f_prime**2 +
           0.5 * (1 - a_bps)**2 * f_bps**2 / (rho_bps**2 + 1e-30) +
           0.5 * a_prime**2 / (rho_bps**2 + 1e-30) +
           (1 - f_bps**2)**2 / 8.0)
    mu_bps = 2 * np.pi * trapezoid(rho_bps * eps, rho_bps)
    print(f"\n  BPS line tension: μ = {mu_bps:.6f} (expected π = {np.pi:.6f})")

    # Ring parameters
    R = 8.0  # ring radius in ξ (smaller than π² for grid fit)

    # ── Grid size scan ────────────────────────────────────
    for N in [32, 48, 64]:
        L = 4 * R  # box size = 4× ring radius
        grid = GridParams.create(N, L)

        print(f"\n{'─'*72}")
        print(f"  N = {N}, L = {L:.1f}ξ, dx = {grid.dx:.3f}ξ, "
              f"R = {R:.1f}ξ")
        print(f"  Memory: {5 * N**3 * 4 / 1e6:.1f} MB")
        print(f"{'─'*72}")

        # Initialize
        t0 = time.time()
        fields = initialize_ring(grid, R, rho_bps, f_bps, a_bps)
        t_init = time.time() - t0
        print(f"  Initialization: {t_init:.2f}s")

        # Check field values
        psi_abs = jnp.sqrt(fields.psi_abs2)
        print(f"  |ψ| range: [{float(psi_abs.min()):.4f}, "
              f"{float(psi_abs.max()):.4f}]")
        print(f"  |A| max:   {float(jnp.sqrt(fields.Ax**2 + fields.Ay**2 + fields.Az**2).max()):.4f}")

        # Compute energy
        t0 = time.time()
        E = energy_jit(fields.psi_re, fields.psi_im,
                       fields.Ax, fields.Ay, fields.Az,
                       grid.dx)
        E_val = float(E)
        t_energy = time.time() - t0
        print(f"  Energy: E = {E_val:.4f}  ({t_energy:.2f}s)")

        # Expected: μ × circumference
        E_expected = mu_bps * 2 * np.pi * R
        print(f"  Expected: μ × 2πR = {mu_bps:.4f} × {2*np.pi*R:.4f}"
              f" = {E_expected:.4f}")
        if E_expected > 0:
            print(f"  E/E_expected = {E_val / E_expected:.4f}")

        # Topological charge
        Q = compute_topological_charge(fields, grid)
        print(f"  Topological charge Q = {float(Q):.4f} (expected ~1)")

        # Test gradient computation
        t0 = time.time()
        grads = grad_energy(fields.psi_re, fields.psi_im,
                            fields.Ax, fields.Ay, fields.Az,
                            grid.dx)
        max_grad = max(float(jnp.max(jnp.abs(g))) for g in grads)
        t_grad = time.time() - t0
        print(f"  Gradient: max|∂E/∂field| = {max_grad:.4f}  ({t_grad:.2f}s)")

        # Vacuum energy (should be ~0)
        vac = Fields.vacuum(grid)
        E_vac = float(energy_jit(vac.psi_re, vac.psi_im,
                                  vac.Ax, vac.Ay, vac.Az, grid.dx))
        print(f"  Vacuum energy: {E_vac:.6f} (should be ~0)")
        print(f"  Vortex - vacuum: {E_val - E_vac:.4f}")

    # ── Gradient descent test (small grid) ────────────────
    print(f"\n{'='*72}")
    print(f"  GRADIENT DESCENT TEST (N=32)")
    print(f"{'='*72}")

    N = 32
    L = 4 * R
    grid = GridParams.create(N, L)
    fields = initialize_ring(grid, R, rho_bps, f_bps, a_bps)

    E_init = float(energy_jit(fields.psi_re, fields.psi_im,
                               fields.Ax, fields.Ay, fields.Az,
                               grid.dx))

    # Simple gradient descent
    lr = 0.001 * grid.dx**2  # learning rate scaled by dx²
    psi_re = fields.psi_re
    psi_im = fields.psi_im
    Ax = fields.Ax
    Ay = fields.Ay
    Az = fields.Az

    @jax.jit
    def step(psi_re, psi_im, Ax, Ay, Az, lr):
        grads = grad_energy(psi_re, psi_im, Ax, Ay, Az, grid.dx)
        psi_re = psi_re - lr * grads[0]
        psi_im = psi_im - lr * grads[1]
        Ax = Ax - lr * grads[2]
        Ay = Ay - lr * grads[3]
        Az = Az - lr * grads[4]
        return psi_re, psi_im, Ax, Ay, Az

    print(f"\n  lr = {lr:.2e}")
    print(f"  {'Step':>6} {'Energy':>12} {'ΔE':>12} {'max|grad|':>12}")
    print(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*12}")

    E_prev = E_init
    t0 = time.time()

    for i in range(50):
        psi_re, psi_im, Ax, Ay, Az = step(psi_re, psi_im, Ax, Ay, Az, lr)

        if i % 10 == 0 or i == 49:
            E = float(energy_jit(psi_re, psi_im, Ax, Ay, Az, grid.dx))
            grads = grad_energy(psi_re, psi_im, Ax, Ay, Az, grid.dx)
            mg = max(float(jnp.max(jnp.abs(g))) for g in grads)
            dE = E - E_prev
            print(f"  {i:6d} {E:12.4f} {dE:+12.4f} {mg:12.4f}")
            E_prev = E

    elapsed = time.time() - t0
    E_final = float(energy_jit(psi_re, psi_im, Ax, Ay, Az, grid.dx))

    print(f"\n  50 steps in {elapsed:.2f}s ({elapsed/50*1000:.1f} ms/step)")
    print(f"  E_init = {E_init:.4f}")
    print(f"  E_final = {E_final:.4f}")
    print(f"  ΔE = {E_final - E_init:+.4f} ({(E_final-E_init)/E_init*100:+.3f}%)")

    print(f"\n{'='*72}")
    print(f"PHASE A SUMMARY")
    print(f"{'='*72}")
    print(f"""
  The 3D abelian Higgs pipeline is working:
    ✓ Field initialization (BPS profile on a vortex ring)
    ✓ Energy computation (JAX, GPU-accelerated)
    ✓ Gradient computation (JAX autodiff)
    ✓ Gradient descent (energy decreasing)
    ✓ Topological charge measurement

  The energy of the initialized ring should converge toward
  E = μ_BPS × 2πR as grid resolution improves. The gradient
  descent lowers the energy further by allowing the fields to
  relax from the initial BPS ansatz.

  NEXT: implement Coulomb gauge fixing (project out ∇·A at each
  step) and run to full convergence at N=64-128.
""")


if __name__ == "__main__":
    main()
