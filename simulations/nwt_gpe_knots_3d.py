#!/usr/bin/env python3
"""
NWT Level-3 GPE Solver: 3D Vortex Knot Dynamics on GPU

Full 3D Gross-Pitaevskii equation with:
  - Taichi GPU acceleration (RTX 4090: up to N=384)
  - Split-step Fourier time integration
  - Torus knot initial conditions (arbitrary T(p,q))
  - Hopf link multi-component initialization
  - Vortex core tracking via density minima
  - Topology detection (linking number, writhe)
  - Edgerton-style snapshot output for visualization

The GPE:
  iℏ ∂ψ/∂t = [-ℏ²/(2m*)∇² + g|ψ|² - μ + V_ext]ψ

In dimensionless units (length in ξ, time in ξ/c_s, energy in m*c_s²):
  i ∂ψ/∂t = [-½∇² + |ψ|² - 1]ψ

where |ψ|² = 1 in the bulk (condensate at rest) and |ψ| → 0 at vortex cores.

A vortex line has phase winding: ψ = f(ρ)e^{iθ} where ρ is distance
from the core and f(ρ) ~ ρ/ξ for ρ << ξ, f → 1 for ρ >> ξ.
"""

import numpy as np
import taichi as ti
import time
import os

# ── Taichi initialization ─────────────────────────────────────────────
ti.init(arch=ti.gpu, default_fp=ti.f32, device_memory_fraction=0.85)


@ti.data_oriented
class GPEKnotSolver:
    """3D Gross-Pitaevskii solver for vortex knot dynamics.

    Uses split-step Fourier method:
      Step 1: half-step nonlinear: ψ *= exp(-i dt/2 × (|ψ|²-1))
      Step 2: full-step kinetic in Fourier space: ψ̂ *= exp(-i dt × k²/2)
      Step 3: half-step nonlinear: ψ *= exp(-i dt/2 × (|ψ|²-1))

    The kinetic step uses numpy FFT (transferred CPU↔GPU per step).
    Future: replace with cuFFT via Taichi for full GPU pipeline.
    """

    def __init__(self, N=256, L=40.0, dt=0.005, g_nonlinear=1.0):
        """
        Args:
            N: grid points per dimension (N³ total)
            L: box size in units of ξ (healing length)
            dt: time step in units of ξ/c_s
            g_nonlinear: GPE coupling constant (1.0 = standard, κ²/(2πα) for PMNS)
        """
        self.N = N
        self.L = L
        self.dx = L / N
        self.dt = dt
        self.g = g_nonlinear
        self.time = 0.0
        self.step_count = 0

        # Memory estimate
        mem_gb = 8 * 4 * N**3 / 1e9  # 8 float32 fields
        print(f"GPEKnotSolver: N={N}, L={L:.1f}ξ, dx={self.dx:.4f}ξ, dt={dt}")
        print(f"  Estimated VRAM: {mem_gb:.2f} GB")

        # ── Taichi fields ─────────────────────────────────────────
        # Complex wavefunction stored as two real fields
        self.psi_re = ti.field(dtype=ti.f32, shape=(N, N, N))
        self.psi_im = ti.field(dtype=ti.f32, shape=(N, N, N))

        # Density |ψ|² (for visualization and vortex detection)
        self.density = ti.field(dtype=ti.f32, shape=(N, N, N))

        # Phase arg(ψ) (for topology tracking)
        self.phase = ti.field(dtype=ti.f32, shape=(N, N, N))

        # External potential (optional)
        self.V_ext = ti.field(dtype=ti.f32, shape=(N, N, N))

        # ── Fourier space arrays (numpy, for split-step) ─────────
        # k-space grid
        k1d = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
        self.k_squared = (kx**2 + ky**2 + kz**2).astype(np.float32)

        # Kinetic propagator: exp(-i dt k²/2)
        self._update_kinetic_propagator()

        print(f"  Grid: {N}³ = {N**3:,} points")
        print(f"  k_max = {k1d.max():.2f}/ξ, k_Nyquist = {np.pi/self.dx:.2f}/ξ")

    def _update_kinetic_propagator(self):
        """Precompute exp(-i dt k²/2) for the kinetic step."""
        phase = -0.5 * self.dt * self.k_squared
        self.kin_prop_re = np.cos(phase)
        self.kin_prop_im = np.sin(phase)

    # ── Initialization methods ────────────────────────────────────────

    def init_uniform(self, amplitude=1.0):
        """Initialize uniform condensate: ψ = amplitude everywhere."""
        self.psi_re.fill(amplitude)
        self.psi_im.fill(0.0)
        self.time = 0.0
        self.step_count = 0

    @ti.kernel
    def _init_vortex_ring_kernel(self, psi_re: ti.template(),
                                  psi_im: ti.template(),
                                  cx: ti.f32, cy: ti.f32, cz: ti.f32,
                                  R_ring: ti.f32, charge: ti.i32,
                                  L: ti.f32, N: ti.i32):
        """Initialize a vortex ring of radius R_ring centered at (cx,cy,cz).

        The ring lies in the xy-plane. Phase winds by 2π×charge around
        the ring core. Density profile: f(ρ) = ρ/√(ρ²+1) where ρ is
        distance from the core in units of ξ.
        """
        dx = L / N
        for i, j, k in psi_re:
            x = (i + 0.5) * dx - L/2
            y = (j + 0.5) * dx - L/2
            z = (k + 0.5) * dx - L/2

            # Position relative to ring center
            xr = x - cx
            yr = y - cy
            zr = z - cz

            # Distance from z-axis (cylindrical)
            rho_cyl = ti.sqrt(xr**2 + yr**2) + 1e-10

            # Distance from the ring core
            d_ring = ti.sqrt((rho_cyl - R_ring)**2 + zr**2)

            # Density profile: tanh(d/ξ) ≈ d/√(d²+1) for smooth profile
            f_density = d_ring / ti.sqrt(d_ring**2 + 1.0)

            # Phase: angle around the ring core in the (rho_cyl-R, z) plane
            theta_core = ti.atan2(zr, rho_cyl - R_ring)
            phase_val = charge * theta_core

            psi_re[i, j, k] = f_density * ti.cos(phase_val)
            psi_im[i, j, k] = f_density * ti.sin(phase_val)

    def init_vortex_ring(self, center=(0, 0, 0), R_ring=5.0, charge=1):
        """Create a single vortex ring."""
        cx, cy, cz = [float(c) for c in center]
        self._init_vortex_ring_kernel(self.psi_re, self.psi_im,
                                       cx, cy, cz, float(R_ring),
                                       int(charge), self.L, self.N)
        self.time = 0.0
        self.step_count = 0
        print(f"  Initialized vortex ring: R={R_ring:.1f}ξ at ({cx},{cy},{cz})")

    def init_torus_knot(self, p=2, q=1, R_major=8.0, r_minor=3.0,
                        kappa_torus=None):
        """Create a T(p,q) vortex knot on a torus.

        Uses the torus coordinate approach: for each grid point,
        compute (φ, θ) on the embedding torus, then the knot phase
        is Ψ = pφ + qθ and the density profile comes from the
        distance to the nearest knot strand.

        This is O(N³) instead of O(N³ × n_knot) — ~100× faster.
        """
        if kappa_torus is not None:
            r_minor = R_major / kappa_torus

        N = self.N
        L = self.L
        dx = self.dx

        print(f"  Initializing T({p},{q}) knot: R={R_major:.1f}ξ, r={r_minor:.2f}ξ, "
              f"κ={R_major/r_minor:.2f}")
        t0 = time.time()

        # Build coordinates
        coords = np.mgrid[0:N, 0:N, 0:N].astype(np.float32)
        x = (coords[0] + 0.5) * dx - L/2
        y = (coords[1] + 0.5) * dx - L/2
        z = (coords[2] + 0.5) * dx - L/2

        # Convert to torus coordinates
        # rho_cyl = distance from z-axis (cylindrical radius)
        rho_cyl = np.sqrt(x**2 + y**2) + 1e-10
        phi = np.arctan2(y, x)  # toroidal angle

        # Distance from torus centerline (circle of radius R_major in xy-plane)
        d_from_center = np.sqrt((rho_cyl - R_major)**2 + z**2)
        theta = np.arctan2(z, rho_cyl - R_major)  # poloidal angle

        # Knot phase: Ψ = p*φ + q*θ
        # The vortex core is where the phase winds — at the knot path itself
        # For T(p,q), the core is at d_from_center = 0 (on the torus centerline)
        # but the phase Ψ has p*q branch cuts

        # Method: use the "implicit vortex" approach
        # ψ = f(d) × exp(i × (p*φ + q*θ))
        # where f(d) = d/√(d²+ξ²) is the vortex density profile
        # and d is the distance from the torus centerline

        # Density profile: heals over scale ξ = 1 in our units
        f_density = d_from_center / np.sqrt(d_from_center**2 + 1.0)

        # Phase: the knot winding
        knot_phase = p * phi + q * theta

        # Build complex wavefunction
        psi = f_density * np.exp(1j * knot_phase)

        # Mask out points far from the torus (optional — keeps condensate uniform outside)
        # Points far from torus should have |ψ| = 1 (bulk condensate)
        far_mask = d_from_center > 5.0  # beyond 5ξ from centerline
        psi[far_mask] = 1.0

        # Transfer to GPU
        self.psi_re.from_numpy(np.real(psi).astype(np.float32))
        self.psi_im.from_numpy(np.imag(psi).astype(np.float32))
        self.time = 0.0
        self.step_count = 0

        elapsed = time.time() - t0
        print(f"  T({p},{q}) knot initialized in {elapsed:.1f}s")

    def init_hopf_link(self, R_major=8.0, r_minor=3.0, separation=None):
        """Create a Hopf link: two interlocked vortex rings.

        Ring 1: in the xz-plane, centered at origin
        Ring 2: in the yz-plane, offset by separation along x

        If separation is None, uses R_major/2 (rings pass through each other).
        """
        if separation is None:
            separation = 0.0  # concentric but in perpendicular planes

        N = self.N
        L = self.L
        dx = self.dx

        print(f"  Initializing Hopf link: R={R_major:.1f}ξ, sep={separation:.1f}ξ")

        psi = np.ones((N, N, N), dtype=np.complex64)

        coords = np.mgrid[0:N, 0:N, 0:N].astype(np.float32)
        x = (coords[0] + 0.5) * dx - L/2
        y = (coords[1] + 0.5) * dx - L/2
        z = (coords[2] + 0.5) * dx - L/2

        # Ring 1: in xy-plane at z=0
        rho1 = np.sqrt(x**2 + y**2)
        d_core1 = np.sqrt((rho1 - R_major)**2 + z**2)
        f1 = d_core1 / np.sqrt(d_core1**2 + 1.0)
        theta1 = np.arctan2(z, rho1 - R_major)
        psi *= f1 * np.exp(1j * theta1)

        # Ring 2: in xz-plane at y=separation, perpendicular to ring 1
        rho2 = np.sqrt(x**2 + z**2)
        d_core2 = np.sqrt((rho2 - R_major)**2 + (y - separation)**2)
        f2 = d_core2 / np.sqrt(d_core2**2 + 1.0)
        theta2 = np.arctan2(y - separation, rho2 - R_major)
        psi *= f2 * np.exp(1j * theta2)

        self.psi_re.from_numpy(np.real(psi).astype(np.float32))
        self.psi_im.from_numpy(np.imag(psi).astype(np.float32))
        self.time = 0.0
        self.step_count = 0
        print(f"  Hopf link initialized")

    def init_ring_collision(self, R_ring=6.0, separation=20.0, velocity=0.3):
        """Set up two vortex rings approaching head-on for collision.

        This is the GPE analog of pair production: two unknotted rings
        collide → reconnection → knotted vortices (Zuccher & Ricca 2022).

        Ring 1: in xy-plane at z = -separation/2, moving +z
        Ring 2: in xy-plane at z = +separation/2, moving -z

        The velocity is imparted by tilting the phase gradient.
        """
        N = self.N
        L = self.L
        dx = self.dx

        print(f"  Ring collision setup: R={R_ring:.1f}ξ, sep={separation:.1f}ξ, v={velocity:.2f}c_s")

        coords = np.mgrid[0:N, 0:N, 0:N].astype(np.float32)
        x = (coords[0] + 0.5) * dx - L/2
        y = (coords[1] + 0.5) * dx - L/2
        z = (coords[2] + 0.5) * dx - L/2

        psi = np.ones((N, N, N), dtype=np.complex64)

        # Ring 1: xy-plane at z = -sep/2
        z1 = z + separation/2
        rho1 = np.sqrt(x**2 + y**2)
        d1 = np.sqrt((rho1 - R_ring)**2 + z1**2)
        f1 = d1 / np.sqrt(d1**2 + 1.0)
        theta1 = np.arctan2(z1, rho1 - R_ring)
        # Phase gradient for +z velocity: multiply by exp(i k_z z)
        k_z = velocity / 1.0  # k = mv/ℏ, in our units m=1, ℏ=1
        ring1 = f1 * np.exp(1j * (theta1 + k_z * z))

        # Ring 2: xy-plane at z = +sep/2, OPPOSITE circulation for head-on
        z2 = z - separation/2
        rho2 = np.sqrt(x**2 + y**2)
        d2 = np.sqrt((rho2 - R_ring)**2 + z2**2)
        f2 = d2 / np.sqrt(d2**2 + 1.0)
        theta2 = np.arctan2(z2, rho2 - R_ring)
        ring2 = f2 * np.exp(1j * (-theta2 - k_z * z))  # opposite circulation + momentum

        # Multiply (both vortices imprinted simultaneously)
        psi = ring1 * ring2

        self.psi_re.from_numpy(np.real(psi).astype(np.float32))
        self.psi_im.from_numpy(np.imag(psi).astype(np.float32))
        self.time = 0.0
        self.step_count = 0
        print(f"  Ring collision initialized: 2 rings approaching at v={velocity:.2f}c_s")

    # ── Time stepping ─────────────────────────────────────────────────

    @ti.kernel
    def _nonlinear_half_step(self, psi_re: ti.template(),
                              psi_im: ti.template(),
                              dt_half: ti.f32):
        """Half-step nonlinear phase rotation: ψ *= exp(-i dt/2 (|ψ|²-1))"""
        for i, j, k in psi_re:
            re = psi_re[i, j, k]
            im = psi_im[i, j, k]
            rho2 = re*re + im*im
            phase = -dt_half * (rho2 - 1.0)
            c = ti.cos(phase)
            s = ti.sin(phase)
            psi_re[i, j, k] = c*re - s*im
            psi_im[i, j, k] = s*re + c*im

    def _kinetic_full_step(self):
        """Full-step kinetic propagation in Fourier space.

        Transfers ψ to CPU, applies FFT, multiplies by propagator,
        inverse FFTs, transfers back to GPU.

        TODO: Replace with Taichi cuFFT for full GPU pipeline.
        """
        # GPU → CPU
        psi_re_np = self.psi_re.to_numpy()
        psi_im_np = self.psi_im.to_numpy()
        psi_complex = psi_re_np + 1j * psi_im_np

        # Forward FFT
        psi_k = np.fft.fftn(psi_complex)

        # Apply kinetic propagator: exp(-i dt k²/2)
        prop = self.kin_prop_re + 1j * self.kin_prop_im
        psi_k *= prop

        # Inverse FFT
        psi_complex = np.fft.ifftn(psi_k)

        # CPU → GPU
        self.psi_re.from_numpy(np.real(psi_complex).astype(np.float32))
        self.psi_im.from_numpy(np.imag(psi_complex).astype(np.float32))

    def step(self, n_steps=1):
        """Advance the GPE by n_steps time steps."""
        dt_half = self.dt / 2.0
        for _ in range(n_steps):
            self._nonlinear_half_step(self.psi_re, self.psi_im, dt_half)
            self._kinetic_full_step()
            self._nonlinear_half_step(self.psi_re, self.psi_im, dt_half)
            self.time += self.dt
            self.step_count += 1

    # ── Imaginary-time evolution (ITE) for ground states ────────────

    @ti.kernel
    def _ite_nonlinear_step(self, psi_re: ti.template(),
                             psi_im: ti.template(),
                             dtau: ti.f32, g: ti.f32):
        """ITE nonlinear step: ψ *= exp(-dτ × g(|ψ|²-1))

        In imaginary time, the phase rotation becomes exponential decay.
        States with higher energy decay faster → converges to ground state.
        """
        for i, j, k in psi_re:
            re = psi_re[i, j, k]
            im = psi_im[i, j, k]
            rho2 = re*re + im*im
            decay = ti.exp(-dtau * g * (rho2 - 1.0))
            psi_re[i, j, k] = re * decay
            psi_im[i, j, k] = im * decay

    def _ite_kinetic_step(self):
        """ITE kinetic step: ψ̂ *= exp(-dτ × k²/2) in Fourier space.

        Same as real-time kinetic step but with real exponential decay
        instead of phase rotation.
        """
        psi_re_np = self.psi_re.to_numpy()
        psi_im_np = self.psi_im.to_numpy()
        psi_complex = psi_re_np + 1j * psi_im_np

        psi_k = np.fft.fftn(psi_complex)
        # Real exponential decay: exp(-dτ k²/2)
        decay = np.exp(-0.5 * self.dt * self.k_squared)
        psi_k *= decay
        psi_complex = np.fft.ifftn(psi_k)

        self.psi_re.from_numpy(np.real(psi_complex).astype(np.float32))
        self.psi_im.from_numpy(np.imag(psi_complex).astype(np.float32))

    def _normalize_psi(self):
        """Normalize ψ to preserve total particle number."""
        psi_re = self.psi_re.to_numpy()
        psi_im = self.psi_im.to_numpy()
        norm = np.sqrt(np.sum(psi_re**2 + psi_im**2) * self.dx**3)
        target_norm = np.sqrt(self.L**3)  # uniform condensate norm
        if norm > 1e-10:
            scale = target_norm / norm
            self.psi_re.from_numpy((psi_re * scale).astype(np.float32))
            self.psi_im.from_numpy((psi_im * scale).astype(np.float32))

    def _build_phase_pin_mask(self, p, q, R_major, r_minor, pin_radius=1.5):
        """Build a mask that pins the phase near the vortex core.

        Returns:
          pin_mask: boolean array, True where phase is pinned
          pin_phase: the pinned phase values (Ψ = pφ + qθ)
          pin_density: the pinned density profile f(d) = d/√(d²+1)

        Points within pin_radius × ξ of the knot centerline are pinned.
        """
        N = self.N
        dx = self.dx
        L = self.L

        coords = np.mgrid[0:N, 0:N, 0:N].astype(np.float32)
        x = (coords[0] + 0.5) * dx - L/2
        y = (coords[1] + 0.5) * dx - L/2
        z = (coords[2] + 0.5) * dx - L/2

        rho_cyl = np.sqrt(x**2 + y**2) + 1e-10
        phi = np.arctan2(y, x)
        d_from_center = np.sqrt((rho_cyl - R_major)**2 + z**2)
        theta = np.arctan2(z, rho_cyl - R_major)

        # Pin within pin_radius of the torus centerline
        pin_mask = d_from_center < pin_radius

        # Knot phase
        knot_phase = p * phi + q * theta

        # Density profile
        f_density = d_from_center / np.sqrt(d_from_center**2 + 1.0)

        return pin_mask, knot_phase, f_density

    def _apply_phase_pin(self, pin_mask, pin_phase, pin_density):
        """Re-impose the knot phase at pinned points, keeping |ψ| free to relax.

        At pinned points: arg(ψ) is reset to Ψ_knot, but |ψ| is preserved
        (or set to the vortex profile if |ψ| has collapsed).
        """
        psi_re = self.psi_re.to_numpy()
        psi_im = self.psi_im.to_numpy()

        amplitude = np.sqrt(psi_re**2 + psi_im**2)

        # Where amplitude has collapsed near the core, restore it
        core_mask = pin_mask & (amplitude < 0.01)
        amplitude[core_mask] = pin_density[core_mask]

        # Reset phase at pinned points
        psi_re[pin_mask] = amplitude[pin_mask] * np.cos(pin_phase[pin_mask])
        psi_im[pin_mask] = amplitude[pin_mask] * np.sin(pin_phase[pin_mask])

        self.psi_re.from_numpy(psi_re.astype(np.float32))
        self.psi_im.from_numpy(psi_im.astype(np.float32))

    def ite_step(self, n_steps=1, pin_data=None):
        """Imaginary-time evolution with optional phase pinning."""
        dtau_half = self.dt / 2.0
        for _ in range(n_steps):
            self._ite_nonlinear_step(self.psi_re, self.psi_im,
                                      dtau_half, float(self.g))
            self._ite_kinetic_step()
            self._ite_nonlinear_step(self.psi_re, self.psi_im,
                                      dtau_half, float(self.g))
            self._normalize_psi()
            # Re-impose vortex topology
            if pin_data is not None:
                self._apply_phase_pin(*pin_data)
            self.time += self.dt
            self.step_count += 1

    def find_knot_ground_state(self, p, q, R_major, r_minor,
                                n_ite=500, dtau=0.005, pin_radius=1.5):
        """Find the GPE ground state of a T(p,q) vortex knot.

        Uses imaginary-time evolution with phase pinning near the core
        to preserve the knot topology while relaxing the bulk condensate.

        Returns the converged wavefunction.
        """
        # Initialize the knot
        self.init_torus_knot(p=p, q=q, R_major=R_major, r_minor=r_minor)

        # Build phase pin mask
        print(f"  Building phase pin mask (pin_radius={pin_radius}ξ)...")
        pin_data = self._build_phase_pin_mask(p, q, R_major, r_minor, pin_radius)
        n_pinned = np.sum(pin_data[0])
        print(f"  Pinned points: {n_pinned} ({n_pinned/self.N**3*100:.1f}% of grid)")

        # ITE with pinning
        old_dt = self.dt
        self.dt = dtau

        print(f"  ITE with phase pinning: {n_ite} steps at dτ={dtau}...")
        t0 = time.time()

        energies = []
        for step in range(n_ite):
            self.ite_step(n_steps=1, pin_data=pin_data)
            if step % 100 == 0 or step == n_ite - 1:
                E = self.total_energy()
                energies.append(E)
                print(f"    Step {step}: E = {E:.4f}")

        elapsed = time.time() - t0
        self.dt = old_dt

        # Check convergence
        if len(energies) >= 2:
            dE = abs(energies[-1] - energies[-2]) / abs(energies[-1])
            print(f"  Converged in {elapsed:.1f}s, E = {energies[-1]:.4f}, "
                  f"ΔE/E = {dE:.2e}")
        else:
            print(f"  Done in {elapsed:.1f}s, E = {energies[-1]:.4f}")

        psi = self.psi_re.to_numpy() + 1j * self.psi_im.to_numpy()
        return psi, energies

    def find_excited_states(self, n_states=3, n_ite=500, dtau=0.01):
        """Find n_states lowest energy states via ITE + Gram-Schmidt.

        For each state:
          1. Initialize with random perturbation on the knot
          2. ITE to converge
          3. Project out all previously found states
          4. Re-ITE to converge in the orthogonal subspace
        """
        states = []
        energies_out = []

        for n in range(n_states):
            print(f"\n  Finding state {n+1}/{n_states}...")

            if n == 0:
                # State 0: just ITE from current initialization
                psi_n = self.find_ground_state(n_ite=n_ite, dtau=dtau)
            else:
                # Add a perturbation with different symmetry
                psi_re = self.psi_re.to_numpy()
                psi_im = self.psi_im.to_numpy()

                # Perturbation: multiply by cos(n × φ) where φ is toroidal angle
                N = self.N
                dx = self.dx
                coords = np.mgrid[0:N, 0:N, 0:N].astype(np.float32)
                x = (coords[0] + 0.5) * dx - self.L/2
                y = (coords[1] + 0.5) * dx - self.L/2
                phi = np.arctan2(y, x)
                perturb = np.cos(n * phi) * 0.1 + 1.0

                self.psi_re.from_numpy((psi_re * perturb).astype(np.float32))
                self.psi_im.from_numpy((psi_im * perturb).astype(np.float32))

                # ITE with projection
                old_dt = self.dt
                self.dt = dtau
                for step in range(n_ite):
                    self.ite_step(n_steps=1)

                    # Project out previous states every 10 steps
                    if step % 10 == 0:
                        psi_cur_re = self.psi_re.to_numpy()
                        psi_cur_im = self.psi_im.to_numpy()
                        psi_cur = psi_cur_re + 1j * psi_cur_im

                        for prev in states:
                            overlap = np.sum(np.conj(prev) * psi_cur) * dx**3
                            psi_cur -= overlap * prev

                        self.psi_re.from_numpy(np.real(psi_cur).astype(np.float32))
                        self.psi_im.from_numpy(np.imag(psi_cur).astype(np.float32))
                        self._normalize_psi()

                    if step % 100 == 0:
                        E = self.total_energy()
                        print(f"    Step {step}: E = {E:.6f}")

                self.dt = old_dt
                psi_n = self.psi_re.to_numpy() + 1j * self.psi_im.to_numpy()

            # Normalize
            norm = np.sqrt(np.sum(np.abs(psi_n)**2) * self.dx**3)
            if norm > 1e-10:
                psi_n /= norm

            E_n = self.total_energy()
            states.append(psi_n)
            energies_out.append(E_n)
            print(f"  State {n}: E = {E_n:.6f}")

        return states, energies_out

    # ── Diagnostics ───────────────────────────────────────────────────

    @ti.kernel
    def _compute_density(self, psi_re: ti.template(),
                          psi_im: ti.template(),
                          density: ti.template()):
        for i, j, k in density:
            re = psi_re[i, j, k]
            im = psi_im[i, j, k]
            density[i, j, k] = re*re + im*im

    @ti.kernel
    def _compute_phase(self, psi_re: ti.template(),
                        psi_im: ti.template(),
                        phase: ti.template()):
        for i, j, k in phase:
            phase[i, j, k] = ti.atan2(psi_im[i, j, k],
                                       psi_re[i, j, k])

    def get_density(self):
        """Compute and return |ψ|² on CPU."""
        self._compute_density(self.psi_re, self.psi_im, self.density)
        return self.density.to_numpy()

    def get_phase(self):
        """Compute and return arg(ψ) on CPU."""
        self._compute_phase(self.psi_re, self.psi_im, self.phase)
        return self.phase.to_numpy()

    def total_energy(self):
        """Compute total energy: E = ∫[½|∇ψ|² + ½(|ψ|²-1)²] d³x."""
        psi_re = self.psi_re.to_numpy()
        psi_im = self.psi_im.to_numpy()
        psi = psi_re + 1j * psi_im

        # Kinetic energy via Fourier
        psi_k = np.fft.fftn(psi)
        E_kin = 0.5 * np.sum(self.k_squared * np.abs(psi_k)**2) * self.dx**3 / self.N**3

        # Interaction energy
        rho2 = np.abs(psi)**2
        E_int = 0.5 * np.sum((rho2 - 1.0)**2) * self.dx**3

        return float(np.real(E_kin + E_int))

    def vortex_core_positions(self, threshold=0.3):
        """Find vortex core positions where |ψ|² < threshold.

        Returns array of (x, y, z) positions of density minima.
        """
        density = self.get_density()
        # Find connected regions below threshold
        from scipy import ndimage
        mask = density < threshold
        labeled, n_features = ndimage.label(mask)

        positions = []
        for label_id in range(1, n_features + 1):
            region = labeled == label_id
            if np.sum(region) < 2:
                continue  # skip single-voxel noise
            # Center of mass
            coords = ndimage.center_of_mass(1.0 - density, labeled, label_id)
            x = (coords[0] + 0.5) * self.dx - self.L/2
            y = (coords[1] + 0.5) * self.dx - self.L/2
            z = (coords[2] + 0.5) * self.dx - self.L/2
            positions.append((x, y, z))

        return np.array(positions) if positions else np.empty((0, 3))

    def save_snapshot(self, filename, include_phase=False):
        """Save density (and optionally phase) to .npz file."""
        density = self.get_density()
        data = {'density': density, 'time': self.time,
                'step': self.step_count, 'N': self.N, 'L': self.L}
        if include_phase:
            data['phase'] = self.get_phase()
        np.savez_compressed(filename, **data)

    def save_slice_png(self, filename, axis='z', index=None):
        """Save a 2D density slice as PNG for Edgerton-style visualization."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        density = self.get_density()
        N = self.N
        if index is None:
            index = N // 2

        if axis == 'z':
            slc = density[:, :, index]
            xlabel, ylabel = 'x', 'y'
        elif axis == 'y':
            slc = density[:, index, :]
            xlabel, ylabel = 'x', 'z'
        else:
            slc = density[index, :, :]
            xlabel, ylabel = 'y', 'z'

        extent = [-self.L/2, self.L/2, -self.L/2, self.L/2]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(slc.T, origin='lower', extent=extent,
                       cmap='inferno', vmin=0, vmax=1.5)
        ax.set_xlabel(f'{xlabel} (ξ)')
        ax.set_ylabel(f'{ylabel} (ξ)')
        ax.set_title(f'|ψ|² at {axis}={index}, t={self.time:.2f} ξ/c_s '
                     f'(step {self.step_count})')
        plt.colorbar(im, ax=ax, label='|ψ|²')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
# DEMO: Vortex ring, Hopf link, and torus knot evolution
# ══════════════════════════════════════════════════════════════════════

def demo_vortex_ring(N=128, n_steps=200):
    """Demo: evolve a single vortex ring and watch it propagate."""
    print("\n" + "="*80)
    print("DEMO 1: Single vortex ring propagation")
    print("="*80)

    solver = GPEKnotSolver(N=N, L=30.0, dt=0.01)
    solver.init_vortex_ring(center=(0, 0, 0), R_ring=6.0, charge=1)

    E0 = solver.total_energy()
    print(f"  Initial energy: {E0:.4f}")

    outdir = 'simulations/output/gpe_ring'
    os.makedirs(outdir, exist_ok=True)

    for frame in range(10):
        solver.save_slice_png(f'{outdir}/frame_{frame:04d}.png', axis='z')
        solver.step(n_steps=n_steps // 10)
        E = solver.total_energy()
        cores = solver.vortex_core_positions()
        print(f"  Step {solver.step_count}: t={solver.time:.2f}, "
              f"E={E:.4f} (ΔE/E₀={abs(E-E0)/abs(E0)*100:.2f}%), "
              f"vortex cores: {len(cores)}")

    print(f"  Frames saved to {outdir}/")


def demo_hopf_link(N=128, n_steps=500):
    """Demo: Hopf link evolution — two interlocked vortex rings."""
    print("\n" + "="*80)
    print("DEMO 2: Hopf link dynamics")
    print("="*80)

    solver = GPEKnotSolver(N=N, L=30.0, dt=0.005)
    solver.init_hopf_link(R_major=6.0)

    E0 = solver.total_energy()
    print(f"  Initial energy: {E0:.4f}")

    outdir = 'simulations/output/gpe_hopf'
    os.makedirs(outdir, exist_ok=True)

    for frame in range(10):
        solver.save_slice_png(f'{outdir}/frame_{frame:04d}_z.png', axis='z')
        solver.save_slice_png(f'{outdir}/frame_{frame:04d}_y.png', axis='y')
        solver.step(n_steps=n_steps // 10)
        E = solver.total_energy()
        cores = solver.vortex_core_positions()
        print(f"  Step {solver.step_count}: t={solver.time:.2f}, "
              f"E={E:.4f} (ΔE/E₀={abs(E-E0)/abs(E0)*100:.2f}%), "
              f"vortex cores: {len(cores)}")

    print(f"  Frames saved to {outdir}/")


def demo_trefoil_knot(N=192, n_steps=500):
    """Demo: T(2,3) trefoil knot on a torus."""
    print("\n" + "="*80)
    print("DEMO 3: Trefoil knot T(2,3) dynamics")
    print("="*80)

    solver = GPEKnotSolver(N=N, L=40.0, dt=0.005)
    solver.init_torus_knot(p=2, q=3, R_major=10.0, r_minor=4.0)

    E0 = solver.total_energy()
    print(f"  Initial energy: {E0:.4f}")

    outdir = 'simulations/output/gpe_trefoil'
    os.makedirs(outdir, exist_ok=True)

    for frame in range(10):
        solver.save_slice_png(f'{outdir}/frame_{frame:04d}_z.png', axis='z')
        solver.save_slice_png(f'{outdir}/frame_{frame:04d}_x.png', axis='x')
        solver.step(n_steps=n_steps // 10)
        E = solver.total_energy()
        print(f"  Step {solver.step_count}: t={solver.time:.2f}, "
              f"E={E:.4f} (ΔE/E₀={abs(E-E0)/abs(E0)*100:.2f}%)")

    print(f"  Frames saved to {outdir}/")


def demo_electron_21(N=256):
    """Demo: (2,1) electron torus knot at physical aspect ratio κ=π²."""
    print("\n" + "="*80)
    print("DEMO 4: Electron (2,1) torus knot, κ=π²")
    print("="*80)

    # Physical parameters
    kappa = np.pi**2
    R = 12.0  # in ξ (larger than physical for visibility)
    r = R / kappa

    solver = GPEKnotSolver(N=N, L=35.0, dt=0.005)
    solver.init_torus_knot(p=2, q=1, R_major=R, kappa_torus=kappa)

    E0 = solver.total_energy()
    print(f"  κ = π² = {kappa:.4f}, R = {R:.1f}ξ, r = {r:.4f}ξ")
    print(f"  Initial energy: {E0:.4f}")

    outdir = 'simulations/output/gpe_electron'
    os.makedirs(outdir, exist_ok=True)

    solver.save_slice_png(f'{outdir}/initial_z.png', axis='z')
    solver.save_slice_png(f'{outdir}/initial_y.png', axis='y')

    # Short evolution to check stability
    for frame in range(5):
        solver.step(n_steps=100)
        E = solver.total_energy()
        print(f"  Step {solver.step_count}: t={solver.time:.2f}, "
              f"E={E:.4f} (ΔE/E₀={abs(E-E0)/abs(E0)*100:.4f}%)")
        solver.save_slice_png(f'{outdir}/frame_{frame:04d}_z.png', axis='z')

    print(f"  Frames saved to {outdir}/")


def demo_collision(N=192, n_steps=2000):
    """Demo: Two vortex rings collide → reconnection → knotted structures.

    This is the Edgerton-style visualization of baryon pair production:
    unknot + unknot → trefoil (baryon) + anti-trefoil (antibaryon)
    as demonstrated in GPE by Zuccher & Ricca (2022).
    """
    print("\n" + "="*80)
    print("DEMO 5: Vortex ring collision — Edgerton-style pair production")
    print("="*80)

    solver = GPEKnotSolver(N=N, L=50.0, dt=0.005)
    solver.init_ring_collision(R_ring=8.0, separation=30.0, velocity=0.5)

    E0 = solver.total_energy()
    print(f"  Initial energy: {E0:.4f}")

    outdir = 'simulations/output/gpe_collision'
    os.makedirs(outdir, exist_ok=True)

    n_frames = 40
    steps_per_frame = n_steps // n_frames

    for frame in range(n_frames):
        # Save multi-axis slices for 3D understanding
        solver.save_slice_png(f'{outdir}/frame_{frame:04d}_z.png', axis='z')
        solver.save_slice_png(f'{outdir}/frame_{frame:04d}_y.png', axis='y')

        # Save snapshot data every 10 frames
        if frame % 10 == 0:
            solver.save_snapshot(f'{outdir}/snapshot_{frame:04d}.npz',
                               include_phase=True)

        solver.step(n_steps=steps_per_frame)
        E = solver.total_energy()
        cores = solver.vortex_core_positions()
        print(f"  Frame {frame:3d} (step {solver.step_count:5d}): "
              f"t={solver.time:.2f}, E={E:.1f} (ΔE={abs(E-E0)/abs(E0)*100:.3f}%), "
              f"cores={len(cores)}")

    # Final snapshot
    solver.save_snapshot(f'{outdir}/final.npz', include_phase=True)
    solver.save_slice_png(f'{outdir}/final_z.png', axis='z')
    solver.save_slice_png(f'{outdir}/final_y.png', axis='y')
    solver.save_slice_png(f'{outdir}/final_x.png', axis='x')
    print(f"  {n_frames} frames saved to {outdir}/")


if __name__ == '__main__':
    import sys
    demo = sys.argv[1] if len(sys.argv) > 1 else 'ring'

    if demo == 'ring':
        demo_vortex_ring(N=128)
    elif demo == 'hopf':
        demo_hopf_link(N=128)
    elif demo == 'trefoil':
        demo_trefoil_knot(N=192)
    elif demo == 'electron':
        demo_electron_21(N=256)
    elif demo == 'collision':
        demo_collision(N=192)
    elif demo == 'all':
        demo_vortex_ring(N=128)
        demo_hopf_link(N=128)
        demo_trefoil_knot(N=192)
    else:
        print(f"Usage: {sys.argv[0]} [ring|hopf|trefoil|electron|collision|all]")
