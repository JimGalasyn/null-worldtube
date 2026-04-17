#!/usr/bin/env python3
"""
Gauge-field holonomy v2: Biot-Savart integral over the full knot.

Fixes the multi-strand issue from v1: instead of finding the nearest
centerline point and using its local BPS profile, compute A(x) as
the Biot-Savart integral over the ENTIRE vortex curve, with a
core-size regularization.

A(x) = (Φ₀/4π) ∮ T(s) ds / √(|x - r₀(s)|² + ξ²)

where ξ is the BPS core size (regularization), T(s) is the unit
tangent, and Φ₀ = 2π is the flux quantum.

The Wilson loop ∮ A·dl then gives the FULL AB phase including
contributions from all strands of the knot, not just the nearest.

For a loop linking the vortex once: Φ → 2π (exactly, in the
thin-tube limit ξ → 0).

TESTS:
  1. Single-strand Wilson loops at various ρ
  2. Large loops verifying total flux quantization
  3. D₃ sector phase measurement
  4. D₃ character amplitude extraction
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


def load_bps_profile():
    npz_path = Path(__file__).parent.parent / "helmholtz_eigenvalue" / "bps_profile.npz"
    data = np.load(npz_path)
    return data["rho"], data["f"], data["a"]


class TrefoilVortexBS:
    """3D gauge field for a BPS vortex via Biot-Savart."""

    def __init__(self, R=None, a_torus=0.3, xi_core=1.0, N_center=2048):
        if R is None:
            R = np.pi**2
        self.R = R
        self.a_torus = a_torus
        self.xi = xi_core
        self.N_center = N_center

        t = np.linspace(0, 2 * np.pi, N_center, endpoint=False)
        dt = t[1] - t[0]
        self.dt = dt

        # Centerline
        a = a_torus
        self.center_x = (R + a*R*np.cos(3*t)) * np.cos(2*t)
        self.center_y = (R + a*R*np.cos(3*t)) * np.sin(2*t)
        self.center_z = a * R * np.sin(3*t)

        # Tangent (dr/dt, NOT unit tangent — includes speed)
        self.dx = np.gradient(self.center_x, dt)
        self.dy = np.gradient(self.center_y, dt)
        self.dz = np.gradient(self.center_z, dt)

        speed = np.sqrt(self.dx**2 + self.dy**2 + self.dz**2)
        self.speed = speed

        # Arc length
        from scipy.integrate import cumulative_trapezoid
        s = np.zeros(N_center)
        s[1:] = cumulative_trapezoid(speed, t)
        self.L = s[-1] + speed[-1] * dt

    def gauge_field(self, point):
        """A(x) via Biot-Savart over the full knot.

        A(x) = C ∮ dr₀/dt × dt / √(|x-r₀|² + ξ²)

        where C is the normalization constant = 1/(2 × 2π) = 1/4π
        for Φ₀ = 2π.

        Actually: A_i(x) = (1/2) ∮ (dr₀/dt)_i dt / √(|x-r₀|² + ξ²)
        """
        px, py, pz = point

        # Distances from point to all centerline points
        ddx = px - self.center_x
        ddy = py - self.center_y
        ddz = pz - self.center_z

        dist2 = ddx**2 + ddy**2 + ddz**2 + self.xi**2
        inv_dist = 1.0 / np.sqrt(dist2)

        # A = (1/2) ∮ T(s) ds / |x-r₀| ≈ (1/2) Σ (dr/dt) dt / |x-r₀|
        # The factor 1/2 comes from Φ₀/(4π) × 2π-integral = Φ₀/2 = π
        # But actually we want ∮ A·dl = 2π × Lk, so let's calibrate.

        # Use: A = C × Σ (dr/dt)_i × inv_dist × dt
        # We'll calibrate C from a test Wilson loop.

        Ax = np.sum(self.dx * inv_dist) * self.dt
        Ay = np.sum(self.dy * inv_dist) * self.dt
        Az = np.sum(self.dz * inv_dist) * self.dt

        return np.array([Ax, Ay, Az])

    def wilson_loop(self, center, radius, normal, N_points=512):
        """∮ A·dl around a circular loop."""
        normal = np.array(normal, dtype=float)
        normal /= np.linalg.norm(normal)

        if abs(normal[2]) < 0.9:
            v1 = np.cross(normal, [0, 0, 1])
        else:
            v1 = np.cross(normal, [1, 0, 0])
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1)

        theta = np.linspace(0, 2 * np.pi, N_points, endpoint=False)
        dtheta = theta[1] - theta[0]

        phase = 0.0
        center = np.array(center, dtype=float)

        for i in range(N_points):
            pos = (center +
                   radius * np.cos(theta[i]) * v1 +
                   radius * np.sin(theta[i]) * v2)

            dl = (-np.sin(theta[i]) * v1 +
                   np.cos(theta[i]) * v2) * radius * dtheta

            A = self.gauge_field(pos)
            phase += np.dot(A, dl)

        return phase

    def calibrate(self):
        """Find the normalization constant by measuring a known
        Wilson loop (small loop around one strand, should give 2π)."""
        # Pick a point on the trefoil
        i_ref = 0
        r0 = np.array([self.center_x[i_ref],
                        self.center_y[i_ref],
                        self.center_z[i_ref]])

        # Tangent at this point
        T = np.array([self.dx[i_ref], self.dy[i_ref],
                       self.dz[i_ref]])
        T /= np.linalg.norm(T)

        # Wilson loop at radius 2ξ (well inside, single strand)
        raw_phase = self.wilson_loop(r0, 2.0 * self.xi, T,
                                      N_points=256)

        # The expected phase at ρ=2ξ for BPS is 2π·a(2)
        rho_bps, _, a_bps = load_bps_profile()
        a_at_2 = np.interp(2.0, rho_bps, a_bps)
        expected = 2 * np.pi * a_at_2

        # Calibration factor
        C = expected / raw_phase if abs(raw_phase) > 1e-10 else 1.0
        return C, raw_phase, expected


def main():
    rho_bps, f_bps, a_bps = load_bps_profile()

    R = np.pi**2
    a_torus = 0.3

    print("=" * 72)
    print("HOLONOMY v2: BIOT-SAVART OVER FULL KNOT")
    print("=" * 72)

    vortex = TrefoilVortexBS(R=R, a_torus=a_torus, xi_core=1.0,
                              N_center=2048)
    print(f"\n  Trefoil T(2,3): R = π² = {R:.4f}, a = {a_torus}")
    print(f"  Arc length L = {vortex.L:.4f}")
    print(f"  Core regularization ξ = {vortex.xi}")

    # ── Calibration ───────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  CALIBRATION")
    print(f"{'─'*72}")

    C, raw, expected = vortex.calibrate()
    print(f"    Raw Wilson loop at ρ=2ξ: {raw:.6f}")
    print(f"    Expected (2π·a(2)):      {expected:.6f}")
    print(f"    Calibration factor C = {C:.6f}")

    # ── Test 1: AB phase vs distance ──────────────────────
    print(f"\n{'─'*72}")
    print(f"  TEST 1: AB PHASE vs DISTANCE FROM CORE")
    print(f"{'─'*72}")

    # Pick a reference point on the trefoil
    i_ref = len(vortex.center_x) // 4
    r0 = np.array([vortex.center_x[i_ref],
                    vortex.center_y[i_ref],
                    vortex.center_z[i_ref]])
    T = np.array([vortex.dx[i_ref], vortex.dy[i_ref],
                   vortex.dz[i_ref]])
    T /= np.linalg.norm(T)

    print(f"\n    {'ρ/ξ':>6} {'Φ_raw':>10} {'Φ_cal':>10} "
          f"{'2π·a(ρ)':>10} {'Φ/2π':>8} {'match':>8}")
    print(f"    {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

    for rho_test in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]:
        phase_raw = vortex.wilson_loop(r0, rho_test, T, N_points=256)
        phase_cal = phase_raw * C
        a_expected = np.interp(rho_test, rho_bps, a_bps)
        phase_expected = 2 * np.pi * a_expected

        if abs(phase_expected) > 0.01:
            match = abs(phase_cal - phase_expected) / phase_expected * 100
        else:
            match = 0.0

        print(f"    {rho_test:6.1f} {phase_raw:10.5f} {phase_cal:10.5f} "
              f"{phase_expected:10.5f} {phase_cal/(2*np.pi):8.4f} "
              f"{match:7.2f}%")

    # ── Test 2: Single-strand loops along the knot ────────
    print(f"\n{'─'*72}")
    print(f"  TEST 2: WILSON LOOPS ALONG THE KNOT (ρ = 2ξ)")
    print(f"{'─'*72}")
    print(f"\n  D₃ symmetry → pattern should repeat every L/3.\n")

    n_test = 12
    s_step = vortex.L / n_test
    phases = []

    for k in range(n_test):
        i_s = int(k * vortex.N_center / n_test) % vortex.N_center
        r_pt = np.array([vortex.center_x[i_s],
                         vortex.center_y[i_s],
                         vortex.center_z[i_s]])
        T_pt = np.array([vortex.dx[i_s], vortex.dy[i_s],
                          vortex.dz[i_s]])
        T_pt /= np.linalg.norm(T_pt)

        phase = vortex.wilson_loop(r_pt, 2.0, T_pt, N_points=256) * C
        phases.append(phase)
        sector = k % 4  # D₃ has period 4 in 12 samples (= L/3)
        print(f"    s = {k*s_step:7.2f} (sector {k % 3}): "
              f"Φ = {phase:8.5f}  Φ/(2π) = {phase/(2*np.pi):.5f}")

    phases = np.array(phases)
    print(f"\n    Mean: {phases.mean():.5f} ± {phases.std():.5f}")
    print(f"    D₃ check: values repeat every 4 samples? "
          f"{np.allclose(phases[:4], phases[4:8], rtol=0.01)}")

    # ── Test 3: Large Wilson loops (linking number) ───────
    print(f"\n{'─'*72}")
    print(f"  TEST 3: LARGE WILSON LOOPS (LINKING NUMBER)")
    print(f"{'─'*72}")
    print(f"\n  Large loop in z=0 plane encircling the trefoil.")
    print(f"  T(2,3) winds 2× around z-axis → linking number = 2.")
    print(f"  Expected: Φ = 2 × 2π = {4*np.pi:.4f}\n")

    for loop_r in [5.0, 8.0, 12.0, 15.0, 20.0, 30.0]:
        phase = vortex.wilson_loop([0, 0, 0], loop_r,
                                    [0, 0, 1], N_points=512) * C
        print(f"    r = {loop_r:5.1f}: Φ = {phase:8.4f}  "
              f"Φ/(2π) = {phase/(2*np.pi):6.3f}  "
              f"(expect 2.000)")

    # ── Test 4: D₃ character amplitude ────────────────────
    print(f"\n{'─'*72}")
    print(f"  TEST 4: D₃ CHARACTER AMPLITUDE EXTRACTION")
    print(f"{'─'*72}")

    # The D₃ character amplitude at a crossing is √3/2.
    # To extract it: compare Wilson loops that go THROUGH
    # a crossing vs loops that avoid crossings.
    #
    # A crossing is where the trefoil crosses OVER itself in
    # a 2D projection.  In 3D, the strands don't touch but
    # pass close (at the inter-strand distance).
    #
    # The AB phase difference between a loop passing through
    # a crossing region and one that doesn't measures the
    # crossing amplitude.
    #
    # For the D₃ representation: the 2D rep matrix at 2π/3
    # rotation has eigenvalues e^{±2πi/3}.  The character
    # (trace) is 2cos(2π/3) = -1.  The off-diagonal elements
    # have magnitude sin(2π/3) = √3/2.
    #
    # In the AB context: the phase MODULATION around the knot
    # has Fourier components.  The fundamental modulation at
    # frequency 3 (matching the D₃ symmetry) encodes the
    # character amplitude.

    print(f"\n  Fourier analysis of the Wilson loop phase around the knot.")
    print(f"  The m=3 Fourier component encodes the D₃ character.\n")

    # Compute Wilson loop phase at many points along the knot
    N_sample = 96  # multiple of 3 for D₃
    s_samples = np.linspace(0, vortex.L, N_sample, endpoint=False)
    phase_samples = np.zeros(N_sample)

    for k in range(N_sample):
        i_s = int(k * vortex.N_center / N_sample) % vortex.N_center
        r_pt = np.array([vortex.center_x[i_s],
                         vortex.center_y[i_s],
                         vortex.center_z[i_s]])
        T_pt = np.array([vortex.dx[i_s], vortex.dy[i_s],
                          vortex.dz[i_s]])
        T_pt /= np.linalg.norm(T_pt)
        phase_samples[k] = vortex.wilson_loop(
            r_pt, 2.0, T_pt, N_points=128) * C

    # Fourier transform
    fft_coeffs = np.fft.fft(phase_samples) / N_sample
    freqs = np.fft.fftfreq(N_sample, d=vortex.L / N_sample)

    print(f"    {'mode':>6} {'|c_m|':>10} {'|c_m|/|c_0|':>12} {'phase':>8}")
    print(f"    {'─'*6} {'─'*10} {'─'*12} {'─'*8}")

    c0 = abs(fft_coeffs[0])
    for m in range(8):
        cm = abs(fft_coeffs[m])
        phase_m = np.angle(fft_coeffs[m])
        ratio = cm / c0 if c0 > 0 else 0
        tag = ""
        if m == 3:
            tag = f"  ← D₃ fundamental (expect √3/2 = {np.sqrt(3)/2:.4f}?)"
        elif m == 6:
            tag = "  ← D₃ harmonic"
        print(f"    {m:6d} {cm:10.6f} {ratio:12.6f} "
              f"{phase_m:8.4f}{tag}")

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"SUMMARY")
    print(f"{'='*72}")
    print(f"""
  The Biot-Savart construction gives the gauge field as an integral
  over the full knot curve, correctly handling multi-strand geometry.

  KEY RESULTS:
  • Near-core (ρ ≤ 2ξ): phase matches BPS profile by calibration
  • Large loops: should give integer linking number × 2π
  • The Fourier spectrum of the Wilson loop phase around the knot
    reveals the D₃ character structure

  The m=3 Fourier component corresponds to the fundamental D₃
  modulation.  If its amplitude relative to m=0 matches √3/2
  (or a related group-theoretic factor), it confirms that the
  D₃ character amplitude of the seven-step derivation is encoded
  in the 3D gauge field geometry.
""")


if __name__ == "__main__":
    main()
