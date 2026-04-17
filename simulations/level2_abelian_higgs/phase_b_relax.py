#!/usr/bin/env python3
"""
Level 2 Phase B: Gradient-flow relaxation beyond the BPS ansatz.

The BPS ansatz gives E ∝ arc length for ANY knot (Phase A result).
Relaxation allows the fields to respond to curvature and self-
interaction, lowering the energy below the BPS value.

The energy drop ΔE = E_BPS - E_relaxed is the "crossing energy"
from topology.  If nonzero and topology-dependent, it's where the
Casimir integers live.

Method: parameterize δf(s, ρ, φ) and δa(s, ρ, φ) as perturbations
on top of the BPS profile, then minimize E[f₀+δf, a₀+δa] using
L-BFGS-B with vectorized energy evaluation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import trapezoid
from pathlib import Path
import time


def load_bps_profile():
    npz_path = Path(__file__).parent.parent / "helmholtz_eigenvalue" / "bps_profile.npz"
    data = np.load(npz_path)
    return data["rho"], data["f"], data["a"]


def torus_knot_curvature(p, q, R, a_torus, N_s):
    """Compute curvature κ(s) on a uniform arc-length grid."""
    N_t = max(N_s * 16, 4096)
    t = np.linspace(0, 2 * np.pi, N_t, endpoint=False)
    dt = t[1] - t[0]

    x = (R + a_torus * R * np.cos(q * t)) * np.cos(p * t)
    y = (R + a_torus * R * np.cos(q * t)) * np.sin(p * t)
    z = a_torus * R * np.sin(q * t)

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

    return s_uniform, kappa_uniform, L


def compute_energy_vectorized(f_field, a_field, rho, phi, s_grid,
                               kappa_s, L, ds, dr, dphi):
    """
    Vectorized energy for fields f(s,ρ,φ) and a(s,ρ,φ).

    f_field: shape (N_s, N_rho, N_phi) — scalar amplitude |ψ|
    a_field: shape (N_s, N_rho, N_phi) — gauge field amplitude

    Energy density for ψ = f e^{iφ}:
      ε = ½(∂f/∂ρ)² + ½(1-a)²f²/ρ²
        + ½(∂f/∂s)²/h² + ½(∂a/∂ρ)²/ρ²
        + (1-f²)²/8
        + curvature metric corrections
    """
    N_s, N_rho, N_phi = f_field.shape

    # 3D coordinate arrays
    # rho[i_rho], phi[i_phi], kappa_s[i_s]
    RHO = rho[np.newaxis, :, np.newaxis]             # (1, N_rho, 1)
    PHI = phi[np.newaxis, np.newaxis, :]              # (1, 1, N_phi)
    KAPPA = kappa_s[:, np.newaxis, np.newaxis]         # (N_s, 1, 1)

    # Metric factor
    H = 1.0 - RHO * KAPPA * np.cos(PHI)               # (N_s, N_rho, N_phi)
    H = np.maximum(H, 0.01)  # avoid negative metric

    # ── Derivatives ──────────────────────────────────────

    # ∂f/∂ρ (central difference, axis=1)
    df_drho = np.zeros_like(f_field)
    df_drho[:, 1:-1, :] = (f_field[:, 2:, :] - f_field[:, :-2, :]) / (2*dr)
    df_drho[:, 0, :] = (f_field[:, 1, :] - f_field[:, 0, :]) / dr
    df_drho[:, -1, :] = (f_field[:, -1, :] - f_field[:, -2, :]) / dr

    # ∂a/∂ρ
    da_drho = np.zeros_like(a_field)
    da_drho[:, 1:-1, :] = (a_field[:, 2:, :] - a_field[:, :-2, :]) / (2*dr)
    da_drho[:, 0, :] = (a_field[:, 1, :] - a_field[:, 0, :]) / dr
    da_drho[:, -1, :] = (a_field[:, -1, :] - a_field[:, -2, :]) / dr

    # ∂f/∂s (central difference, periodic, axis=0)
    df_ds = np.zeros_like(f_field)
    df_ds[1:-1, :, :] = (f_field[2:, :, :] - f_field[:-2, :, :]) / (2*ds)
    df_ds[0, :, :] = (f_field[1, :, :] - f_field[-1, :, :]) / (2*ds)
    df_ds[-1, :, :] = (f_field[0, :, :] - f_field[-2, :, :]) / (2*ds)

    # ∂f/∂φ (central difference, periodic, axis=2)
    df_dphi = np.zeros_like(f_field)
    df_dphi[:, :, 1:-1] = (f_field[:, :, 2:] - f_field[:, :, :-2]) / (2*dphi)
    df_dphi[:, :, 0] = (f_field[:, :, 1] - f_field[:, :, -1]) / (2*dphi)
    df_dphi[:, :, -1] = (f_field[:, :, 0] - f_field[:, :, -2]) / (2*dphi)

    # ── Energy density ───────────────────────────────────

    RHO_safe = np.maximum(RHO, 0.01)

    # Scalar kinetic: ½(∂f/∂ρ)²
    eps_rho = 0.5 * df_drho**2

    # Scalar winding: ½(n-a)²f²/ρ²  (n=1)
    eps_wind = 0.5 * (1.0 - a_field)**2 * f_field**2 / RHO_safe**2

    # Scalar s-kinetic: ½(∂f/∂s)²/h²
    eps_s = 0.5 * df_ds**2 / H**2

    # Scalar φ-kinetic: ½(∂f/∂φ)²/ρ²
    eps_phi = 0.5 * df_dphi**2 / RHO_safe**2

    # Gauge: ½(∂a/∂ρ)²/ρ²
    eps_gauge = 0.5 * da_drho**2 / RHO_safe**2

    # Potential: (1-f²)²/8
    eps_pot = (1.0 - f_field**2)**2 / 8.0

    eps_total = eps_rho + eps_wind + eps_s + eps_phi + eps_gauge + eps_pot

    # Volume element: ρ h_s dρ dφ ds
    dV = RHO * H * dr * dphi * ds

    E = np.sum(eps_total * dV)

    return E


def relax_on_knot(p, q, R, a_torus, rho_raw, f_raw, a_raw,
                   N_s=32, N_rho=64, N_phi=16, rho_max=8.0,
                   max_iter=200):
    """
    Relax the abelian Higgs fields on a T(p,q) knot tube.

    Returns (E_bps, E_relaxed, delta_E, info).
    """
    # Geometry
    s_grid, kappa_s, L = torus_knot_curvature(p, q, R, a_torus, N_s)
    ds = s_grid[1] - s_grid[0] if N_s > 1 else L

    rho_min = 0.05
    rho = np.linspace(rho_min, rho_max, N_rho)
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    dr = rho[1] - rho[0]
    dphi = phi[1] - phi[0]

    # BPS initialization: f₀(ρ), a₀(ρ) replicated across (s, φ)
    f0 = np.interp(rho, rho_raw, f_raw)
    a0 = np.interp(rho, rho_raw, a_raw)

    f_init = np.tile(f0[np.newaxis, :, np.newaxis], (N_s, 1, N_phi))
    a_init = np.tile(a0[np.newaxis, :, np.newaxis], (N_s, 1, N_phi))

    # BPS energy
    E_bps = compute_energy_vectorized(f_init, a_init, rho, phi,
                                       s_grid, kappa_s, L, ds, dr, dphi)

    # Pack fields into a single vector for optimizer
    x0 = np.concatenate([f_init.ravel(), a_init.ravel()])

    n_f = f_init.size
    call_count = [0]

    def objective(x):
        call_count[0] += 1
        f_field = x[:n_f].reshape(N_s, N_rho, N_phi)
        a_field = x[n_f:].reshape(N_s, N_rho, N_phi)

        # Enforce physical constraints
        f_field = np.clip(f_field, 0.0, 1.5)
        a_field = np.clip(a_field, -0.5, 1.5)

        E = compute_energy_vectorized(f_field, a_field, rho, phi,
                                       s_grid, kappa_s, L, ds, dr, dphi)
        return E

    # Run L-BFGS-B
    t0 = time.time()
    result = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-12,
                               'gtol': 1e-8, 'disp': False,
                               'maxfun': max_iter * 10})
    elapsed = time.time() - t0

    # Extract relaxed fields
    f_relaxed = result.x[:n_f].reshape(N_s, N_rho, N_phi)
    a_relaxed = result.x[n_f:].reshape(N_s, N_rho, N_phi)

    E_relaxed = result.fun
    delta_E = E_bps - E_relaxed

    # How much did the fields change?
    df_max = np.max(np.abs(f_relaxed - f_init))
    da_max = np.max(np.abs(a_relaxed - a_init))

    info = {
        'E_bps': E_bps,
        'E_relaxed': E_relaxed,
        'delta_E': delta_E,
        'delta_E_pct': delta_E / E_bps * 100,
        'df_max': df_max,
        'da_max': da_max,
        'converged': result.success,
        'n_iter': result.nit,
        'n_feval': call_count[0],
        'time': elapsed,
        'L': L,
        'kappa_mean': np.mean(kappa_s),
        'kappa_max': np.max(kappa_s),
    }

    return info


def main():
    rho_raw, f_raw, a_raw = load_bps_profile()
    R = np.pi**2
    a_torus = 0.3

    print("=" * 72)
    print("LEVEL 2 PHASE B: GRADIENT-FLOW RELAXATION BEYOND BPS")
    print("=" * 72)

    knots = [
        (2, 1, "electron (unknot)"),
        (2, 3, "trefoil"),
        (2, 5, "cinquefoil"),
        (1, 4, "proton candidate"),
        (2, 7, "heptafoil"),
    ]

    results = []

    for p, q, label in knots:
        print(f"\n{'─'*72}")
        print(f"  T({p},{q}) — {label}")
        print(f"{'─'*72}")

        info = relax_on_knot(p, q, R, a_torus, rho_raw, f_raw, a_raw,
                              N_s=32, N_rho=48, N_phi=16, rho_max=8.0,
                              max_iter=100)

        print(f"    Arc length L = {info['L']:.2f}")
        print(f"    Curvature: mean = {info['kappa_mean']:.5f}, "
              f"max = {info['kappa_max']:.5f}")
        print(f"    E_BPS     = {info['E_bps']:.6f}")
        print(f"    E_relaxed = {info['E_relaxed']:.6f}")
        print(f"    ΔE = E_BPS - E_relaxed = {info['delta_E']:.6f} "
              f"({info['delta_E_pct']:+.4f}%)")
        print(f"    max|δf| = {info['df_max']:.6f}, "
              f"max|δa| = {info['da_max']:.6f}")
        print(f"    Converged: {info['converged']}, "
              f"iterations: {info['n_iter']}, "
              f"time: {info['time']:.1f}s")

        results.append((p, q, label, info))

    # ── Comparison table ──────────────────────────────────
    print(f"\n{'='*72}")
    print(f"RELAXATION ENERGY COMPARISON")
    print(f"{'='*72}")

    E_ref = [r[3]['E_relaxed'] for r in results if r[0] == 2 and r[1] == 1]
    E_ref = E_ref[0] if E_ref else results[0][3]['E_relaxed']
    L_ref = [r[3]['L'] for r in results if r[0] == 2 and r[1] == 1]
    L_ref = L_ref[0] if L_ref else results[0][3]['L']

    print(f"\n  {'T(p,q)':<20} {'E_BPS':>10} {'E_relax':>10} "
          f"{'ΔE':>10} {'ΔE%':>8} {'E/E_e':>8} {'L/L_e':>8} {'cross':>8}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for p, q, label, info in results:
        E_ratio = info['E_relaxed'] / E_ref
        L_ratio = info['L'] / L_ref
        cross = E_ratio / L_ratio - 1 if L_ratio > 0 else 0

        print(f"  T({p},{q}) {label:<14} "
              f"{info['E_bps']:10.3f} {info['E_relaxed']:10.3f} "
              f"{info['delta_E']:10.4f} {info['delta_E_pct']:+7.4f}% "
              f"{E_ratio:8.5f} {L_ratio:8.5f} {cross:+8.5f}")

    # ── Interpretation ────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"INTERPRETATION")
    print(f"{'='*72}")

    any_cross = any(abs(r[3]['E_relaxed']/E_ref / (r[3]['L']/L_ref) - 1) > 0.001
                    for r in results)

    if any_cross:
        print("""
  CROSSING ENERGY DETECTED.  The relaxed energy ratios DIFFER from
  arc-length ratios — topology contributes beyond mere length.
  This is where the Casimir integers begin to emerge from the
  Lagrangian.
""")
    else:
        print("""
  Crossing energy is below detection threshold at this resolution.
  Possible reasons:

  1. The torus knot strands are well-separated at R/r_tube = π² ≈ 10,
     so self-interaction is exponentially suppressed (the BPS field
     decays as e^{-√2 ρ} at large ρ, and inter-strand distance is
     ~6ξ → suppression ~e^{-6√2} ≈ 2×10⁻⁴).

  2. The curvature response (fields adjusting to follow the bend)
     is a smooth, topology-independent correction that doesn't
     distinguish trefoil from unknot.

  3. The topology-dependent physics may live in the GAUGE SECTOR
     (AB phases, crossing amplitudes) rather than in the energy
     minimum — consistent with the seven-step α derivation where
     topology enters through the D₃ character theory, not the
     energy.

  IMPLICATION: The mass hierarchy does NOT come from crossing
  energy in the abelian Higgs model at these aspect ratios.  It
  comes from:
    (a) The radial excitation spectrum (the m index)
    (b) The gauge-field topology (AB phases)
    (c) Possibly: a different Lagrangian sector (Faddeev-Skyrme,
        gravitational coupling, or Euler-Heisenberg corrections)

  This is an HONEST NEGATIVE that sharpens the theory.
""")


if __name__ == "__main__":
    main()
