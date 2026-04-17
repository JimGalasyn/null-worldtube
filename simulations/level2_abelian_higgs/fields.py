"""
3D abelian Higgs field arrays and operations.

Fields on a uniform N³ Cartesian grid:
  ψ = ψ_re + i ψ_im   (complex scalar, 2 real DOF)
  A = (A_x, A_y, A_z)  (gauge vector, 3 real DOF)

Total: 5 real fields × N³ points.

Uses JAX for GPU acceleration and autodiff.
"""

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GridParams:
    """3D uniform Cartesian grid parameters."""
    N: int          # grid points per side
    L: float        # box size (physical units, in ξ)
    dx: float       # grid spacing

    @classmethod
    def create(cls, N: int, L: float):
        return cls(N=N, L=L, dx=L / N)

    @property
    def coords(self):
        """Return (x, y, z) 1D coordinate arrays, centered at origin."""
        c = jnp.linspace(-self.L/2, self.L/2, self.N, endpoint=False)
        return c, c, c

    @property
    def meshgrid(self):
        """Return (X, Y, Z) 3D meshgrid arrays."""
        c = jnp.linspace(-self.L/2, self.L/2, self.N, endpoint=False)
        return jnp.meshgrid(c, c, c, indexing='ij')


@dataclass
class Fields:
    """Abelian Higgs field configuration on a 3D grid."""
    psi_re: jnp.ndarray   # shape (N, N, N)
    psi_im: jnp.ndarray   # shape (N, N, N)
    Ax: jnp.ndarray        # shape (N, N, N)
    Ay: jnp.ndarray        # shape (N, N, N)
    Az: jnp.ndarray        # shape (N, N, N)

    @classmethod
    def vacuum(cls, grid: GridParams):
        """Uniform vacuum: |ψ| = 1, A = 0."""
        N = grid.N
        return cls(
            psi_re=jnp.ones((N, N, N)),
            psi_im=jnp.zeros((N, N, N)),
            Ax=jnp.zeros((N, N, N)),
            Ay=jnp.zeros((N, N, N)),
            Az=jnp.zeros((N, N, N)),
        )

    def to_flat(self) -> jnp.ndarray:
        """Pack all fields into a single flat vector."""
        return jnp.concatenate([
            self.psi_re.ravel(), self.psi_im.ravel(),
            self.Ax.ravel(), self.Ay.ravel(), self.Az.ravel()
        ])

    @classmethod
    def from_flat(cls, x: jnp.ndarray, N: int):
        """Unpack a flat vector into Fields."""
        n3 = N**3
        return cls(
            psi_re=x[0*n3:1*n3].reshape(N, N, N),
            psi_im=x[1*n3:2*n3].reshape(N, N, N),
            Ax=x[2*n3:3*n3].reshape(N, N, N),
            Ay=x[3*n3:4*n3].reshape(N, N, N),
            Az=x[4*n3:5*n3].reshape(N, N, N),
        )

    @property
    def psi_abs2(self) -> jnp.ndarray:
        """|ψ|²"""
        return self.psi_re**2 + self.psi_im**2

    @property
    def memory_mb(self) -> float:
        """Total memory in MB."""
        return 5 * self.psi_re.nbytes / 1e6
