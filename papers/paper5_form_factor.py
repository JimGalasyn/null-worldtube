"""
Generate form factor plot for Paper 5.

The (2,1) torus has a charge distribution that produces
oscillations in the form factor at Q ~ ℏ/R ≈ 0.5 MeV/c.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1  # Bessel functions

# Constants
hbar = 1.055e-34  # J·s
c = 2.998e8       # m/s
m_e = 9.109e-31   # kg
alpha = 1/137.036
MeV = 1.602e-13   # J
fm = 1e-15        # m

# Torus parameters
beta = np.sqrt(5)/2
kappa = 12/np.sqrt(7)
xi = hbar/(m_e*c)
R = beta * xi
r_tube = R / kappa

print(f"R = {R/fm:.1f} fm, r = {r_tube/fm:.1f} fm")
print(f"ℏ/R = {hbar*c/R/MeV:.3f} MeV")
print(f"ℏ/r = {hbar*c/r_tube/MeV:.3f} MeV")

# Form factor for a uniform torus
# F(Q) = ∫ ρ(r) e^{iQ·r} d³r / ∫ ρ(r) d³r
#
# For a thin torus ring of radius R in the xy-plane:
# F(Q) = J₀(QR sinθ) for scattering angle θ from the ring axis
#
# For a torus with tube radius r, averaging over orientations:
# F(Q) = <J₀(QR sinθ)> × [2J₁(Qr sinφ)/(Qr sinφ)]
#
# The orientation-averaged form factor for a RING of radius R:
# F_ring(Q) = (1/2) ∫₀^π J₀(QR sinθ) sinθ dθ = sin(QR)/(QR)
#
# Wait, that's the form factor for a SPHERE, not a ring!
# For a ring: F(Q) = J₀(QR) when Q is perpendicular to ring axis.
# Orientation-averaged: F_avg(Q) = (2/π) ∫₀^{π/2} J₀(QR sinθ) dθ
#
# Actually, the orientation-averaged form factor for a circular ring
# of radius R is:
# F(Q) = j₀(QR) = sin(QR)/(QR)    [spherical Bessel function]
#
# This is because averaging J₀(QR sinθ) over the sphere gives
# the sinc function (this is a standard result).

# For a torus with tube radius r << R:
# F(Q) ≈ j₀(QR) × Ω(Qr)
# where Ω is the tube form factor ≈ 2J₁(Qr)/(Qr) for a cylinder

# Let's compute numerically for proper comparison

# Q range
Q = np.linspace(0.001, 30, 5000)  # in units of 1/R
Q_fm = Q / R  # in 1/fm
Q_MeV = Q * hbar * c / (R * MeV)  # in MeV/c

# 1. Point particle (SM electron)
F_point = np.ones_like(Q)

# 2. Ring of radius R (orientation-averaged)
F_ring = np.sinc(Q / np.pi)  # sinc(x) = sin(πx)/(πx), so sinc(Q/π) = sin(Q)/Q

# 3. Torus with major R, minor r
# F_torus = F_ring × tube_factor
# Tube form factor: 2J₁(Qr/R × Q_norm)/(Qr/R × Q_norm)
Q_r = Q * r_tube / R  # Q in units of 1/r
tube_factor = np.where(Q_r > 0.001, 2*j1(Q_r)/Q_r, 1.0)
F_torus = F_ring * tube_factor

# 4. (2,1) knot modulation
# The (2,1) winding adds a cos(2φ) modulation to the charge density
# This creates a quadrupole pattern: F includes Y₂₀ term
# For orientation-averaged: adds oscillation at period π/R
F_knot = F_ring * tube_factor * np.cos(Q / 2)**2  # simplified knot modulation

# Cross-section: dσ/dQ² ∝ |F(Q)|²
sigma_point = F_point**2
sigma_ring = F_ring**2
sigma_torus = F_torus**2
sigma_knot = F_knot**2

# === FIGURE ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Form factor |F(Q)|² vs Q in MeV/c
ax1 = axes[0, 0]
ax1.semilogy(Q_MeV, sigma_point, 'k-', linewidth=1.5, label='Point particle (SM)', alpha=0.5)
ax1.semilogy(Q_MeV, sigma_ring, 'b-', linewidth=2, label=f'Ring R = {R/fm:.0f} fm')
ax1.semilogy(Q_MeV, sigma_torus, 'r-', linewidth=2, label=f'Torus R={R/fm:.0f}, r={r_tube/fm:.0f} fm')

ax1.set_xlabel('Momentum transfer Q (MeV/c)', fontsize=12)
ax1.set_ylabel(r'$|F(Q)|^2$', fontsize=12)
ax1.set_title('Electron Form Factor: Point vs Torus', fontsize=13)
ax1.legend(fontsize=10)
ax1.set_xlim(0, 8)
ax1.set_ylim(1e-4, 1.5)
ax1.grid(True, alpha=0.3)

# Mark key scales
ax1.axvline(hbar*c/(R*MeV), color='blue', linestyle=':', alpha=0.5)
ax1.text(hbar*c/(R*MeV)*1.05, 0.5, r'$\hbar/R$', fontsize=10, color='blue')
ax1.axvline(hbar*c/(r_tube*MeV), color='red', linestyle=':', alpha=0.5)
ax1.text(hbar*c/(r_tube*MeV)*1.05, 0.5, r'$\hbar/r$', fontsize=10, color='red')

# Panel 2: Low-Q region (the testable window)
ax2 = axes[0, 1]
Q_low = Q_MeV < 3
ax2.plot(Q_MeV[Q_low], sigma_point[Q_low], 'k-', linewidth=1.5,
         label='Point (SM)', alpha=0.5)
ax2.plot(Q_MeV[Q_low], sigma_torus[Q_low], 'r-', linewidth=2.5,
         label='NWT torus')

# Show the deviation
deviation = sigma_torus - sigma_point
ax2_twin = ax2.twinx()
ax2_twin.plot(Q_MeV[Q_low], (deviation/sigma_point)[Q_low] * 100, 'g--',
              linewidth=1.5, alpha=0.7)
ax2_twin.set_ylabel('Deviation from point (%)', fontsize=11, color='green')
ax2_twin.tick_params(axis='y', labelcolor='green')

ax2.set_xlabel('Q (MeV/c)', fontsize=12)
ax2.set_ylabel(r'$|F(Q)|^2$', fontsize=12)
ax2.set_title('Low-Q Region: Testable Window\n(Q < 3 MeV/c)', fontsize=13)
ax2.legend(fontsize=10, loc='lower left')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.8, 1.02)

# Panel 3: Form factor vs Q in fm⁻¹ (natural units)
ax3 = axes[1, 0]
ax3.plot(Q / R * fm, sigma_ring, 'b-', linewidth=2, label='Ring', alpha=0.7)
ax3.plot(Q / R * fm, sigma_torus, 'r-', linewidth=2.5, label='Torus')

ax3.set_xlabel(r'$Q$ (fm$^{-1}$)', fontsize=12)
ax3.set_ylabel(r'$|F(Q)|^2$', fontsize=12)
ax3.set_title('Form Factor in Natural Units', fontsize=13)
ax3.legend(fontsize=10)
ax3.set_xlim(0, 0.05)
ax3.grid(True, alpha=0.3)

# Mark first minimum
# First zero of sin(QR)/(QR) is at QR = π, i.e., Q = π/R
Q_min1 = np.pi / R
ax3.axvline(Q_min1 * fm, color='purple', linestyle='--', alpha=0.5)
ax3.text(Q_min1*fm*1.05, 0.5, f'First minimum\nQ = π/R = {Q_min1*fm:.4f} fm⁻¹\n'
         f'= {Q_min1*hbar*c/MeV:.2f} MeV/c', fontsize=9, color='purple')

# Panel 4: Physical summary
ax4 = axes[1, 1]
ax4.axis('off')

summary = f"""
ELECTRON FORM FACTOR PREDICTIONS

Torus geometry:
  R = (√5/2) ƛ_C = {R/fm:.1f} fm  (major radius)
  r = (√35/24) ƛ_C = {r_tube/fm:.1f} fm  (tube radius)

Key momentum scales:
  ℏ/R = {hbar*c/R/MeV:.3f} MeV/c  (torus onset)
  ℏ/r = {hbar*c/r_tube/MeV:.3f} MeV/c  (tube onset)
  π/R = {np.pi*hbar*c/R/MeV:.3f} MeV/c  (first minimum)

Oscillation period:
  ΔQ = 2πℏ/R = {2*np.pi*hbar*c/R/MeV:.3f} MeV/c

Deviation from point particle:
  At Q = 0.5 MeV: {((np.sinc(0.5*MeV*R/(np.pi*hbar*c)))**2 - 1)*100:.2f}%
  At Q = 1.0 MeV: {((np.sinc(1.0*MeV*R/(np.pi*hbar*c)))**2 - 1)*100:.2f}%
  At Q = 2.0 MeV: {((np.sinc(2.0*MeV*R/(np.pi*hbar*c)))**2 - 1)*100:.1f}%

Current experiments: Q > 100 MeV
  → torus unresolved (|F|² ≈ 0 at these Q)
  → need Q < 3 MeV to test
"""
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

fig.suptitle(r'Electron Form Factor: NWT Torus ($R = 432$ fm, $r = 95$ fm)',
             fontsize=15, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('papers/figures/paper5_form_factor.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved papers/figures/paper5_form_factor.png")
