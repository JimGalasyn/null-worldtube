"""
Abelian Higgs energy functional on a 3D grid.

E[ψ, A] = ∫ d³x [ |Dψ|² + ½ B² + (λ/4)(|ψ|²-v²)² ]

All operations use JAX for GPU acceleration and autodiff.
The energy function is designed to be differentiated via jax.grad.
"""

import jax
import jax.numpy as jnp


def _roll_diff(f: jnp.ndarray, axis: int, dx: float) -> jnp.ndarray:
    """Central difference ∂f/∂x_axis using periodic BCs."""
    return (jnp.roll(f, -1, axis=axis) - jnp.roll(f, 1, axis=axis)) / (2 * dx)


def _forward_diff(f: jnp.ndarray, axis: int, dx: float) -> jnp.ndarray:
    """Forward difference for curl (staggered-grid friendly)."""
    return (jnp.roll(f, -1, axis=axis) - f) / dx


def compute_energy(psi_re, psi_im, Ax, Ay, Az, dx, e=1.0, lam=1.0, v=1.0):
    """Compute total abelian Higgs energy.

    Args:
        psi_re, psi_im: real and imaginary parts of ψ, shape (N,N,N)
        Ax, Ay, Az: gauge field components, shape (N,N,N)
        dx: grid spacing
        e: gauge coupling (default 1)
        lam: scalar self-coupling (default 1 = BPS point)
        v: vacuum expectation value (default 1)

    Returns:
        E: total energy (scalar)
    """
    # Covariant derivatives: Dψ = ∇ψ - ieAψ
    # D_x ψ = ∂ψ/∂x - ieA_x ψ
    # For ψ = ψ_re + iψ_im:
    #   Re(D_x ψ) = ∂ψ_re/∂x + eA_x ψ_im
    #   Im(D_x ψ) = ∂ψ_im/∂x - eA_x ψ_re

    dpsi_re_dx = _roll_diff(psi_re, 0, dx)
    dpsi_re_dy = _roll_diff(psi_re, 1, dx)
    dpsi_re_dz = _roll_diff(psi_re, 2, dx)

    dpsi_im_dx = _roll_diff(psi_im, 0, dx)
    dpsi_im_dy = _roll_diff(psi_im, 1, dx)
    dpsi_im_dz = _roll_diff(psi_im, 2, dx)

    # |Dψ|² = |D_x ψ|² + |D_y ψ|² + |D_z ψ|²
    Dpsi_sq = (
        (dpsi_re_dx + e * Ax * psi_im)**2 +
        (dpsi_im_dx - e * Ax * psi_re)**2 +
        (dpsi_re_dy + e * Ay * psi_im)**2 +
        (dpsi_im_dy - e * Ay * psi_re)**2 +
        (dpsi_re_dz + e * Az * psi_im)**2 +
        (dpsi_im_dz - e * Az * psi_re)**2
    )

    # Magnetic field: B = ∇ × A
    # B_x = ∂A_z/∂y - ∂A_y/∂z
    # B_y = ∂A_x/∂z - ∂A_z/∂x
    # B_z = ∂A_y/∂x - ∂A_x/∂y
    Bx = _roll_diff(Az, 1, dx) - _roll_diff(Ay, 2, dx)
    By = _roll_diff(Ax, 2, dx) - _roll_diff(Az, 0, dx)
    Bz = _roll_diff(Ay, 0, dx) - _roll_diff(Ax, 1, dx)

    B_sq = Bx**2 + By**2 + Bz**2

    # Potential: V = (λ/4)(|ψ|² - v²)²
    psi_sq = psi_re**2 + psi_im**2
    V = (lam / 4.0) * (psi_sq - v**2)**2

    # Total energy density
    eps = Dpsi_sq + 0.5 * B_sq + V

    # Integrate: E = Σ ε × dx³
    E = jnp.sum(eps) * dx**3

    return E


def compute_energy_from_flat(x, N, dx, e=1.0, lam=1.0, v=1.0):
    """Energy from a flat parameter vector (for scipy.optimize)."""
    n3 = N**3
    psi_re = x[0*n3:1*n3].reshape(N, N, N)
    psi_im = x[1*n3:2*n3].reshape(N, N, N)
    Ax = x[2*n3:3*n3].reshape(N, N, N)
    Ay = x[3*n3:4*n3].reshape(N, N, N)
    Az = x[4*n3:5*n3].reshape(N, N, N)
    return compute_energy(psi_re, psi_im, Ax, Ay, Az, dx, e, lam, v)


# JIT-compiled energy and gradient
energy_jit = jax.jit(compute_energy)
grad_energy = jax.jit(jax.grad(compute_energy, argnums=(0, 1, 2, 3, 4)))
