"""
Initialize a vortex ring on a 3D grid.

For a circular vortex ring of radius R in the z=0 plane centered
at the origin:
  - |ψ(x)| = f₀(ρ) where ρ is the distance to the ring centerline
  - arg(ψ) winds by 2π around the vortex core (unit winding n=1)
  - A is in the azimuthal direction around the core: A_φ = a(ρ)/ρ

The BPS profile f₀(ρ), a(ρ) is loaded from Step 1.
"""

import jax.numpy as jnp
import numpy as np
from pathlib import Path
from fields import Fields, GridParams


def load_bps_profile():
    """Load BPS profile from Step 1."""
    npz_path = (Path(__file__).parent.parent /
                "helmholtz_eigenvalue" / "bps_profile.npz")
    data = np.load(npz_path)
    return data["rho"], data["f"], data["a"]


def initialize_ring(grid: GridParams, R: float,
                    rho_bps=None, f_bps=None, a_bps=None):
    """Initialize a circular vortex ring of radius R in the z=0 plane.

    Args:
        grid: GridParams for the 3D grid
        R: ring radius (in units of ξ)
        rho_bps, f_bps, a_bps: BPS profile arrays (loaded if None)

    Returns:
        Fields object with the vortex ring stamped
    """
    if rho_bps is None:
        rho_bps, f_bps, a_bps = load_bps_profile()

    N = grid.N
    X, Y, Z = grid.meshgrid

    # Distance from each grid point to the ring centerline (circle
    # of radius R in the z=0 plane):
    # The closest point on the ring to (x,y,z) is at angle
    # θ = atan2(y, x), giving ring point (R cos θ, R sin θ, 0).
    # Distance ρ = √((r_cyl - R)² + z²) where r_cyl = √(x²+y²).
    r_cyl = jnp.sqrt(X**2 + Y**2)
    rho = jnp.sqrt((r_cyl - R)**2 + Z**2)

    # Interpolate BPS profile
    # (Use numpy for interp, then convert to jax)
    rho_np = np.array(rho)
    f_vals = np.interp(rho_np, rho_bps, f_bps, right=1.0)
    a_vals = np.interp(rho_np, rho_bps, a_bps, right=1.0)
    f_field = jnp.array(f_vals)
    a_field = jnp.array(a_vals)

    # Phase winding: arg(ψ) = angle around the vortex core.
    # For a ring in the z=0 plane, the "angle around the core" is
    # the angle in the (r_cyl - R, z) plane:
    #   φ_core = atan2(z, r_cyl - R)
    # This winds by 2π around the core → unit winding number.
    phi_core = jnp.arctan2(Z, r_cyl - R + 1e-30)

    # ψ = f(ρ) × e^{iφ_core}
    psi_re = f_field * jnp.cos(phi_core)
    psi_im = f_field * jnp.sin(phi_core)

    # Gauge field: A is in the "azimuthal" direction around the core.
    # The azimuthal unit vector ê_φ around the core is:
    #   ê_φ = (-sin(φ_core) ê_ρ_cyl + cos(φ_core) ê_z) ... wait
    # Actually: the azimuthal direction around the vortex CORE is
    # in the plane perpendicular to the ring tangent.
    # For a ring in z=0, the tangent is ê_θ (toroidal direction).
    # The "core azimuthal" direction ê_φ_core is:
    #   ê_φ_core = (-sin φ_core) ê_r_cyl + cos(φ_core) ê_z
    #            = (-sin φ_core)(cos θ ê_x + sin θ ê_y) + cos(φ_core) ê_z
    # where θ = atan2(y, x) is the toroidal angle.

    theta = jnp.arctan2(Y, X)
    sin_phi = jnp.sin(phi_core)
    cos_phi = jnp.cos(phi_core)

    # Core azimuthal unit vector in Cartesian:
    ephi_x = -sin_phi * jnp.cos(theta)
    ephi_y = -sin_phi * jnp.sin(theta)
    ephi_z = cos_phi

    # A = a(ρ)/ρ × ê_φ_core
    # Safe division (avoid ρ=0)
    rho_safe = jnp.maximum(rho, 0.01)
    A_mag = a_field / rho_safe

    Ax = A_mag * ephi_x
    Ay = A_mag * ephi_y
    Az = A_mag * ephi_z

    return Fields(psi_re=psi_re, psi_im=psi_im, Ax=Ax, Ay=Ay, Az=Az)


def compute_topological_charge(fields: Fields, grid: GridParams):
    """Compute the topological winding number.

    Q = (1/2π) ∫ B · dS through a surface bounded by the ring.
    For a vortex ring in z=0, integrate B_z over the z=0 plane.
    """
    dx = grid.dx
    from energy import _roll_diff

    # B_z = ∂A_y/∂x - ∂A_x/∂y
    Bz = _roll_diff(fields.Ay, 0, dx) - _roll_diff(fields.Ax, 1, dx)

    # Integrate B_z over the z=0 plane (middle slice)
    mid = grid.N // 2
    flux = jnp.sum(Bz[:, :, mid]) * dx**2

    return flux / (2 * jnp.pi)
