"""
Physical constants and base dataclass for the Null Worldtube model.
"""

import numpy as np
from dataclasses import dataclass

# Physical constants (SI)
c = 2.99792458e8          # speed of light (m/s)
hbar = 1.054571817e-34    # reduced Planck constant (J·s)
h_planck = 2 * np.pi * hbar  # Planck constant (J·s)
e_charge = 1.602176634e-19  # elementary charge (C)
eps0 = 8.8541878128e-12   # permittivity of free space (F/m)
mu0 = 1.2566370621e-6     # permeability of free space (H/m)
m_e = 9.1093837015e-31    # electron mass (kg)
m_e_MeV = 0.51099895      # electron mass (MeV/c²)
m_mu = 1.883531627e-28    # muon mass (kg)
m_p = 1.67262192369e-27   # proton mass (kg)
alpha = 7.2973525693e-3   # fine-structure constant
G_N = 6.67430e-11             # Newton's gravitational constant (m³/kg/s²)
eV = 1.602176634e-19      # electronvolt (J)
MeV = 1e6 * eV
M_Planck = np.sqrt(hbar * c / G_N)  # Planck mass (kg) ≈ 2.176e-8 kg
M_Planck_GeV = M_Planck * c**2 / (1e9 * eV)  # Planck mass (GeV) ≈ 1.221e19
l_Planck = np.sqrt(hbar * G_N / c**3)  # Planck length (m) ≈ 1.616e-35

# Derived
lambda_C = hbar / (m_e * c)    # reduced Compton wavelength (m) ≈ 3.86e-13 m
r_e = alpha * lambda_C          # classical electron radius ≈ 2.82e-15 m
a_0 = lambda_C / alpha          # Bohr radius ≈ 5.29e-11 m
k_e = 1.0 / (4 * np.pi * eps0)  # Coulomb constant

# Known particle masses (MeV/c²) for reference
PARTICLE_MASSES = {
    'electron': 0.51099895,
    'muon': 105.6583755,
    'pion±': 139.57039,
    'proton': 938.27208816,
    'tau': 1776.86,
}


@dataclass
class TorusParams:
    """Parameters defining a torus and winding numbers."""
    R: float          # major radius (m)
    r: float          # minor radius (m)
    p: int = 1        # toroidal winding number
    q: int = 1        # poloidal winding number
